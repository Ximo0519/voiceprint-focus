"""流式推理引擎：ONNX 单步 + STFT/OLA + 注册编码 + VAD 质量校验。

设计要点
- 网络只在 ONNX 里（enroll/step），STFT/iSTFT 在 numpy 侧完成，与训练口径一致；
- 4 帧（32ms）一组调用 step，等价于训练/评测的 chunk=4 交付口径；
- LSTM (h,c) 跨 chunk 缓存；OTA 累积与整段 iSTFT 数学等价，只有末尾 16ms 因实时
  拿不到未来样本（参考实现用 reflect，实时用零补齐）而有差异，交付文档中如实说明。
"""
from __future__ import annotations

import collections
import json
import os
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from dsp import HOP, PAD, SR, WIN, OnlineAnalyzer, OnlineISTFT, analyze_frames, analyze_full

NBAND = 32
HID = 256
LAYERS = 6
CHUNK_FRAMES = 4

# 每个 session 独立 ORT intra-op 线程池。实测：灌入场景下两路容量有提升（报告 8.6 节），
# 但长驻进程里开启后单路会退化到 39-41ms/块（同进程单 session A/B 无差异，退化来自
# 反复建/换池），因此默认关闭；多路灌入压测时再用 VF_PER_SESSION=1 开启。
_PER_SESSION = os.environ.get("VF_PER_SESSION", "0").strip().lower() not in (
    "0", "", "false", "no", "off")

# 输出端轻量后置降噪（默认开启，PD_FLOOR=-18dB；VF_POST_DENOISE=0 可关闭）。
# 复用输入频谱做稳态噪底估计 + 对输出谱减，保留相位，不改变 32ms 时间轴与 OLA 连续性。
# 2026-11-22 限幅扫描：-18dB 是「帧 p20 压到 -95dBFS 且门控抖动仍为 0」的最后一档；
# -21/-24 出现抖动与呼吸感。音乐/播客场景建议关闭或退到 -12dB。
POST_DENOISE = os.environ.get("VF_POST_DENOISE", "1").strip().lower() not in (
    "0", "", "false", "no", "off")

# RTF 计量插桩：仅在 VF_PROFILE=1 时记录分段时间；默认关闭，不改变推理与输出。
PROFILE = os.environ.get("VF_PROFILE", "0").strip().lower() not in (
    "0", "", "false", "no", "off")
PD_BETA = 1.0                # 过减因子
PD_FLOOR_DB = -18.0          # 增益下限（dB）；2026-11-22 扫描选定 -18
PD_FLOOR = 10 ** (PD_FLOOR_DB / 20)  # 线性增益下限
PD_SMOOTH = 0.6              # 帧间增益平滑，抑制 musical noise
PD_ALPHA = 0.08              # 噪底 EMA 系数（噪声段）
PD_NOISE_RATIO = 2.0         # 输入块能量低于最低能量 ×2（+3dB）才更新噪底
# 运行中开关降噪的过渡块数：增益在 4 块（128ms）内线性淡入/淡出，避免单块电平跳变。
PD_RAMP_BLOCKS = 4

# 高频搁架补偿（默认 0 = 关闭，VF_HF_SHELF_DB 控制）：3.2kHz 以上按 4 阶高通形状
# 渐进抬升到 +HF_SHELF_DB，用来抵消模型分离 + 后置降噪造成的高频发闷。
# 实现为**逐块频域幅度增益**（257 bins，零相位、零延迟），不做逐样本 IIR：
# 同一块内各频点乘固定实数，OLA 后等价于整段过一个固定零相位滤波器，跨块天然连续；
# model 与 skip/bypass 两条路径都经过，保证直通段与模型段音色一致。
HF_SHELF_DB = float(os.environ.get("VF_HF_SHELF_DB", "0") or 0)
HF_SHELF_FC = 3200.0


def _env_ms(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# 诊断录制（默认关闭，不影响推理与输出）：把「送进模型的原样输入」与「最终输出」各存一条
# 16k WAV，并把每块的模型/旁路标记写成同名 txt，用来区分噪声来自输入侧还是模型/OLA 侧。
#   VF_CAPTURE=1        开启
#   VF_CAPTURE_DIR      输出目录（默认 app/backend/captures）
#   VF_CAPTURE_MAX_S    每条流最多录多少秒（默认 30，超限后只停录、不影响推理）
CAPTURE = os.environ.get("VF_CAPTURE", "0").strip().lower() not in (
    "0", "", "false", "no", "off")
CAPTURE_DIR = os.environ.get(
    "VF_CAPTURE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures"))
CAPTURE_MAX_S = _env_ms("VF_CAPTURE_MAX_S", 30.0)
_CAP_SEQ = 0


# 旁路降级：滞回（进出双门限 + 最短驻留）+ 块间电平对齐 + 交界增益交叉淡化。
# 目的：消除「模型块 / 旁路块逐块混流」造成的块间电平跳变与 15.62Hz(=1/64ms) 包络调制。
# 只改应用层的输出增益，不改时间轴、不改 OLA、不改模型权重。
BYPASS_ENTER_MS = _env_ms("VF_LATE_MS", 150)        # 等待超过此值 → 进入旁路
BYPASS_EXIT_MS = _env_ms("VF_LATE_EXIT_MS", 60)     # 等待低于此值才允许退出（必须 < 进入门限）
BYPASS_DWELL_MS = _env_ms("VF_BYPASS_DWELL_MS", 300)  # 进入后最短驻留，避免阈值附近横跳
BYPASS_LEVEL_MATCH = os.environ.get("VF_BYPASS_MATCH", "1").strip().lower() not in (
    "0", "", "false", "no", "off")                  # 旁路块电平对齐到模型块
XFADE_MS = _env_ms("VF_XFADE_MS", 8.0)              # 交界处增益交叉淡化时长（5–10ms）
XFADE_SAMPLES = int(round(XFADE_MS * SR / 1000.0))
# 「旁路只在停顿段生效」：有声块即使队列等待超阈值也留在模型路径（用缓冲换延迟），
# 只有判为停顿的块才允许旁路。安全阀：单块等待超过 BYPASS_FORCE_MS 时仍强制旁路，
# 否则 RTF>1 时队列会无限累积。VF_BYPASS_PAUSE_ONLY=0 回到旧的「只看等待」逻辑。
BYPASS_PAUSE_ONLY = os.environ.get("VF_BYPASS_PAUSE_ONLY", "1").strip().lower() not in (
    "0", "", "false", "no", "off")
BYPASS_FORCE_MS = _env_ms("VF_BYPASS_FORCE_MS", 700.0)  # 安全阀硬上限（600–800 可用）
# 「跳空帧」：判为纯停顿的块不送模型，输出与旁路停顿块一致（−6dB 电平对齐 + pause −12dB），
# 省掉整块推理，时间轴/OLA 照常推进。判据复用 pause 前馈 VAD，并额外要求
# 「整块无过阈帧 + 块内 100–2000Hz 电平 ≤ SKIP_ABS_DB + 距上一个过阈帧 ≥ SKIP_HOLD_MS」，
# 避免门限被响段抬高后把弱语音切掉。默认开启（VF_SKIP_SILENT=0 可单独关闭）。
SKIP_SILENT = os.environ.get("VF_SKIP_SILENT", "1").strip().lower() not in (
    "0", "", "false", "no", "off")
SKIP_ABS_DB = float(os.environ.get("VF_SKIP_ABS_DB", "-52"))
SKIP_HOLD_MS = _env_ms("VF_SKIP_HOLD_MS", 60.0)
# 「跳空后预热」：连续 ≥SKIP_WARM_AFTER 块未进模型（跳空/迟到旁路）后，恢复时把最近
# SKIP_WARM_BLOCKS 块补喂模型，只推进 h/c、不产生输出、不碰 OLA/时间轴。
# VF_SKIP_WARM_BLOCKS=0 关闭；VF_SKIP_WARM_AFTER 调触发阈值。
SKIP_WARM_BLOCKS = max(0, int(float(os.environ.get("VF_SKIP_WARM_BLOCKS", "2"))))
SKIP_WARM_AFTER = max(1, int(float(os.environ.get("VF_SKIP_WARM_AFTER", "8"))))

# 停顿前馈（默认开启，VF_PAUSE_STEP=0 可关闭）：从**输入**算 5ms 步长的语音判决
# （100–2000Hz 能量，阈值=max(底噪+9dB, -48dBFS)），判停顿时对输出追加 -12dB，
# attack 8ms / release 30ms。与离线 pause_step_fix.py 同一套参数：前馈不增加延迟，
# 只让模型还没压下去的停顿残留在起始处提前消失。与 VF_POST_DENOISE 独立，可分别关。
PAUSE_STEP = os.environ.get("VF_PAUSE_STEP", "1").strip().lower() not in (
    "0", "", "false", "no", "off")
PAUSE_DB = _env_ms("VF_PAUSE_DB", 12.0)     # 停顿段额外衰减（dB）
PAUSE_ATT_MS = _env_ms("VF_PAUSE_ATT_MS", 8.0)
PAUSE_REL_MS = _env_ms("VF_PAUSE_REL_MS", 30.0)
PAUSE_HOP = 80                              # 5ms @16k
PAUSE_THR_OFF_DB = 9.0                      # 阈值 = 底噪 + 9dB
PAUSE_THR_ABS_DB = -48.0
PAUSE_HIST = 600                            # 噪声底滑窗 3s（5ms 帧）
LEVEL_EMA_DB = 0.05                                # 模式电平统计 EMA（约 0.6s 时间常数）
MATCH_GAIN_DB_LIMIT = 6.0                          # 电平对齐增益限幅 ±6dB


def make_session(path: str, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    # 默认多个 session 共享同一个全局 intra-op 线程池（大小由首个 session 决定），
    # 单路 1 线程时两路并发会争抢同一线程；开启 VF_PER_SESSION=1 可改为每 session
    # 独立线程池。注意该 ORT 构建中显式赋 False 会导致建 session 失败，故只在开启时设置。
    if _PER_SESSION:
        so.use_per_session_threads = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---------------- 注册侧：VAD 质量校验 + 编码 ----------------

def _frames_db(x: np.ndarray, win_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    """逐帧能量（dB）：窗 win_ms、帧移 hop_ms，供 VAD / 切分 / 音高共用。"""
    win, hop = int(win_ms * SR / 1000), int(hop_ms * SR / 1000)
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = np.asarray(x, np.float32)[idx]
    return 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)


def vad_report(x: np.ndarray) -> dict:
    """轻量能量 VAD：20ms 窗/10ms 帧移，噪声底用 20 分位近似。
    返回时长、语音占比、峰值、削波率与估计 SNR，供前端展示与注册门禁。"""
    x = np.asarray(x, np.float32)
    dur = len(x) / SR
    if len(x) < int(0.5 * SR):
        return {"ok": False, "reason": "太短", "duration_s": dur, "speech_ratio": 0.0,
                "peak": float(np.abs(x).max()) if len(x) else 0.0,
                "clip_ratio": 0.0, "snr_db": 0.0}
    edb = _frames_db(x)
    floor = float(np.percentile(edb, 20))
    thr = max(floor + 8.0, -55.0)
    speech = edb > thr
    ratio = float(speech.mean())
    peak = float(np.abs(x).max())
    clip = float(np.mean(np.abs(x) > 0.985))
    snr = float(np.percentile(edb[speech], 80) - floor) if speech.any() else 0.0
    ok = (2.5 <= dur <= 6.5) and ratio >= 0.08 and peak > 0.02 and clip < 0.02
    reason = ""
    if not ok:
        if not (2.5 <= dur <= 6.5):
            if dur < 2.5:
                reason = f"时长 {dur:.1f}s 不足 2.5s（还差 {2.5 - dur:.1f}s）"
            else:
                reason = f"时长 {dur:.1f}s 超过 6.5s（多 {dur - 6.5:.1f}s）"
        elif peak <= 0.02:
            reason = "几乎无声"
        elif ratio < 0.08:
            reason = "语音占比过低"
        else:
            reason = "削波过多"
    return {"ok": bool(ok), "reason": reason, "duration_s": round(dur, 3),
            "speech_ratio": round(ratio, 3), "peak": round(peak, 4),
            "clip_ratio": round(clip, 5), "snr_db": round(snr, 1)}


def enroll_mag(x: np.ndarray):
    mag = np.abs(analyze_full(x))[None].astype(np.float32)
    ne = mag.shape[2]
    valid = min(len(x) // HOP + 1, ne)
    mask = np.zeros((1, ne), np.bool_)
    mask[0, :valid] = True
    return mag, mask


class Enroller:
    def __init__(self, onnx_path: str):
        self.sess = make_session(onnx_path)

    def encode(self, x: np.ndarray):
        mag, mask = enroll_mag(x)
        t0 = time.perf_counter()
        E, g = self.sess.run(None, {"mag": mag, "mask": mask})
        return E.astype(np.float32), g.astype(np.float32), (time.perf_counter() - t0) * 1000

    def embed(self, x: np.ndarray) -> np.ndarray:
        """整段注册音 → 单条 L2 归一化说话人向量（帧级 E 在有效帧上平均）。"""
        mag, mask = enroll_mag(x)
        E, _g = self.sess.run(None, {"mag": mag, "mask": mask})
        v = E[0].astype(np.float64)
        m = mask[0]
        v = v[m].mean(axis=0) if m.any() else v.mean(axis=0)
        n = float(np.linalg.norm(v))
        return ((v / n) if n > 1e-9 else v).astype(np.float32)


# ---------------- 注册校验：同人一致性 + 音高分散 ----------------
# 两个阈值均由 2026-11-22/enroll-qc/calibrate_enroll_qc.py 实测分布标定（本地 TTS 8 音色），
# 真实麦克风路径未在标定集内；标定结论与失败模式见同目录 report.md。
ENROLL_ID_COS_MIN = float(os.environ.get("VF_ENROLL_ID_COS", "0.89"))
ENROLL_PITCH_SHIFT_MAX_ST = float(os.environ.get("VF_ENROLL_PITCH_ST", "6.5"))
ENROLL_PITCH_RANGE = (70.0, 400.0)   # F0 搜索范围（Hz）
ENROLL_HALF_MIN_S = 0.8              # 切半后每半最短时长


def split_enroll_halves(x: np.ndarray):
    """按 VAD 语音帧累计到 50% 处切分（切点吸附到该帧起点，避免切在音节中间）。
    返回 (前半, 后半, 切点秒数)。语音帧不足时退回时长等分。"""
    x = np.asarray(x, np.float32)
    edb = _frames_db(x)
    floor = float(np.percentile(edb, 20)) if len(edb) else -120.0
    thr = max(floor + 8.0, -55.0)
    sp = np.flatnonzero(edb > thr)
    hop = int(0.01 * SR)
    cut = int(sp[len(sp) // 2] * hop) if len(sp) >= 4 else len(x) // 2
    lo = int(ENROLL_HALF_MIN_S * SR)
    cut = int(min(max(cut, lo), max(lo, len(x) - lo)))
    return x[:cut], x[cut:], cut / SR


def _yin_f0(fr: np.ndarray, thr: float = 0.18) -> float:
    """YIN 基频：取首个低于阈值的谷（避免自相关取到倍周期）。未浊化返回 0。"""
    fr = fr.astype(np.float64)
    fr -= fr.mean()
    n = len(fr)
    if n < 64 or float(np.dot(fr, fr)) <= 1e-9:
        return 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(fr, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    sq = np.concatenate([[0.0], np.cumsum(fr * fr)])
    tau = np.arange(1, n)
    d = (sq[n - tau] - sq[0]) + (sq[n] - sq[tau]) - 2 * ac[tau]
    cm = np.ones(n)
    cm[1:] = np.maximum(d, 0.0) * tau / (np.cumsum(np.maximum(d, 0.0)) + 1e-12)
    lo = max(2, int(SR / ENROLL_PITCH_RANGE[1]))
    hi = min(n - 2, int(SR / ENROLL_PITCH_RANGE[0]))
    seg = cm[lo:hi]
    idx = np.flatnonzero(seg < thr)
    if not len(idx):
        return 0.0
    k = int(idx[0])
    if 1 <= k < len(seg) - 1:            # 抛物线细化到亚样本
        a, b, c = seg[k - 1], seg[k], seg[k + 1]
        den = a - 2 * b + c
        k = k + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    return float(SR / (lo + k))


def _f0_track(x: np.ndarray):
    """逐帧 F0（40ms/10ms），返回 (f0, 秒时间, 总帧数)。能量门限以下与非浊化帧跳过。"""
    win, hop = int(0.04 * SR), int(0.01 * SR)
    edb = _frames_db(x, 40.0, 10.0)
    gate = max(float(np.percentile(edb, 20)) + 8.0, -55.0)
    n = 1 + max(0, (len(x) - win) // hop)
    f0, ts = [], []
    for i in range(n):
        if edb[i] <= gate:
            continue
        f = _yin_f0(x[i * hop:i * hop + win])
        if f > 0:
            f0.append(f)
            ts.append(i * hop / SR)
    return np.asarray(f0, np.float64), np.asarray(ts, np.float64), n


def _win_medians(f0: np.ndarray, ts: np.ndarray, win_s: float = 0.5,
                 hop_s: float = 0.25) -> np.ndarray:
    """把帧级 F0 聚成 0.5s 窗中位数，抑制倍/半频跳变与瞬时估计噪声。"""
    if len(f0) < 6:
        return np.zeros(0)
    out, t0, end = [], ts[0], ts[-1]
    while t0 + win_s <= end + 1e-9:
        m = (ts >= t0) & (ts < t0 + win_s)
        if int(m.sum()) >= 6:
            out.append(float(np.median(f0[m])))
        t0 += hop_s
    return np.asarray(out, np.float64)


def pitch_stats(x: np.ndarray) -> dict:
    """基频统计：前后半段的窗级中位 F0 差 shift_st（半音）为主判据，
    span_st（全段 p10–p90）与 std_st 仅作诊断。F0 无法可靠提取时 checked=False（不拒绝）。"""
    a, b, cut = split_enroll_halves(x)
    fa, ta, na = _f0_track(a)
    fb, tb, nb = _f0_track(b)
    base = {"checked": False, "shift_st": 0.0, "span_st": 0.0, "std_st": 0.0,
            "f0_p10": 0.0, "f0_p50": 0.0, "f0_p90": 0.0,
            "voiced_ratio": round((len(fa) + len(fb)) / max(1, na + nb), 3),
            "n_voiced": int(len(fa) + len(fb)), "split_s": round(cut, 2)}
    wa, wb = _win_medians(fa, ta), _win_medians(fb, tb)
    if len(wa) < 2 or len(wb) < 2:
        return base
    allw = np.concatenate([wa, wb])
    p10, p50, p90 = (float(np.percentile(allw, q)) for q in (10, 50, 90))
    return {"checked": True,
            "shift_st": round(float(12 * np.log2(np.median(wb) / np.median(wa))), 2),
            "span_st": round(float(12 * np.log2(p90 / p10)), 2),
            "std_st": round(float(np.std(12 * np.log2(allw / p50))), 2),
            "f0_p10": round(p10, 1), "f0_p50": round(p50, 1), "f0_p90": round(p90, 1),
            "voiced_ratio": base["voiced_ratio"], "n_voiced": base["n_voiced"],
            "split_s": base["split_s"]}


ENROLL_HINTS = {
    "multi_speaker": "这段里似乎不止一个人的声音：请一个人连贯说完，不要中途换人或多人轮流说。",
    "pitch_unstable": "这段音高波动偏大：请用平时说话的音调，一段连贯讲完，不要刻意模仿或忽高忽低。",
}


def enroll_qc(wav: np.ndarray, enroller: "Enroller",
              cos_min: float | None = None, shift_max: float | None = None) -> dict:
    """注册侧两项校验：① 前后半段声纹一致性（余弦）；② 前后半段基频差（半音）。
    ok=False 时 code ∈ {multi_speaker, pitch_unstable}，hint 可直接展示给用户；
    基频提取不可靠时跳过音高判据（checked=False），避免把噪声注册误杀。"""
    cos_min = ENROLL_ID_COS_MIN if cos_min is None else cos_min
    shift_max = ENROLL_PITCH_SHIFT_MAX_ST if shift_max is None else shift_max
    a, b, cut_s = split_enroll_halves(wav)
    cos = float(np.dot(enroller.embed(a), enroller.embed(b)))
    ps = pitch_stats(wav)
    ok_id = bool(cos >= cos_min)
    ok_pd = (not ps["checked"]) or bool(abs(ps["shift_st"]) <= shift_max)
    # 半段音高跳变超过 8 半音（约一个性别档）更像「换了个人说」，优先按多人混录提示
    if (not ok_id) or (not ok_pd and abs(ps["shift_st"]) >= max(8.0, shift_max + 1.0)):
        code = "multi_speaker"
    elif not ok_pd:
        code = "pitch_unstable"
    else:
        code = ""
    return {"ok": bool(ok_id and ok_pd), "code": code,
            "hint": ENROLL_HINTS.get(code, ""),
            "identity": {"ok": ok_id, "cos": round(cos, 4), "cos_min": cos_min,
                         "split_s": round(cut_s, 2),
                         "half_a_s": round(len(a) / SR, 2), "half_b_s": round(len(b) / SR, 2)},
            "pitch": {**ps, "ok": ok_pd, "shift_max_st": shift_max}}


# ---------------- 流式会话 ----------------

@dataclass
class StepStat:
    chunk: int
    frames: int
    compute_ms: float
    samples_in: int
    needed_samples: int
    out_end: int
    out_len: int
    algo_latency_ms: float
    skipped: bool = False
    mode: str = "model"
    gain_db: float = 0.0


class PauseFeedForward:
    """输入侧停顿判决 → 对输出追加短时衰减（前馈，不增加延迟）。

    判决只依赖当前及之前的输入；输出块比输入滞后约 8ms（算法延迟），因此判决一定
    先于对应输出样本可用。参数与离线 `pause_step_fix.py` 一致：
      100–2000Hz 能量、阈值 = max(底噪+9dB, -48dBFS)（底噪取最近 3s 的 p10，因果）；
      连续 3 帧（15ms）低于阈值判停顿、连续 2 帧（10ms）高于阈值判语音；
      停顿段 -12dB，attack 8ms / release 30ms，5ms 帧插值到样本。
    与离线版唯一的差别：5ms 平滑用「当前帧与上一帧取平均」的因果 2 抽头，
    不做需要未来样本的对称卷积。
    """

    def __init__(self, extra_db: float = PAUSE_DB):
        from scipy.signal import butter, sosfilt
        self._sosfilt = sosfilt
        self.sos = butter(4, [100.0 / (SR / 2), 2000.0 / (SR / 2)],
                          btype="band", output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2), np.float64)
        self.rest = np.zeros(0, np.float32)
        self.hist: collections.deque = collections.deque(maxlen=PAUSE_HIST)
        self.state = True                      # 起始按语音处理，避免上来就压
        self.low_run = 0
        self.high_run = 0
        self.cur_db = 0.0
        self.prev_db: float | None = None
        self.n_frames = 0
        self.g_frames: collections.deque = collections.deque(maxlen=800)  # (起始样本, dB)
        self.marks: collections.deque = collections.deque(maxlen=800)     # (起始样本, 是否语音)
        self.extra_db = float(extra_db)
        self.a_att = 1.0 - np.exp(-(PAUSE_HOP / SR) / (PAUSE_ATT_MS / 1000.0))
        self.a_rel = 1.0 - np.exp(-(PAUSE_HOP / SR) / (PAUSE_REL_MS / 1000.0))
        self.pause_frames = 0

    def feed(self, x: np.ndarray):
        x = np.asarray(x, np.float32).reshape(-1)
        if len(x) == 0:
            return
        buf = np.concatenate([self.rest, x]) if len(self.rest) else x
        n = len(buf) // PAUSE_HOP
        if n == 0:
            self.rest = buf.astype(np.float32, copy=False)
            return
        use = buf[: n * PAUSE_HOP].astype(np.float64)
        self.rest = np.asarray(buf[n * PAUSE_HOP:], np.float32)
        yb, self.zi = self._sosfilt(self.sos, use, zi=self.zi)
        e = np.sqrt(np.mean(yb.reshape(n, PAUSE_HOP) ** 2, axis=1) + 1e-20)
        edb = 20.0 * np.log10(e + 1e-12)
        self.hist.extend(edb.tolist())
        noise = float(np.percentile(np.fromiter(self.hist, np.float64, len(self.hist)), 10))
        thr = max(noise + PAUSE_THR_OFF_DB, PAUSE_THR_ABS_DB)
        for v in edb:
            if v > thr:
                self.high_run += 1
                self.low_run = 0
            else:
                self.low_run += 1
                self.high_run = 0
            if self.state and self.low_run >= 3:
                self.state = False
            elif (not self.state) and self.high_run >= 2:
                self.state = True
            tgt = 0.0 if self.state else -self.extra_db
            coef = self.a_att if tgt < self.cur_db else self.a_rel
            self.cur_db += coef * (tgt - self.cur_db)
            g = self.cur_db if self.prev_db is None else 0.5 * (self.cur_db + self.prev_db)
            self.prev_db = self.cur_db
            self.g_frames.append((self.n_frames * PAUSE_HOP, g))
            # marks: (起始样本, 滞回后状态, 本帧是否高过阈值, 本帧 100–2000Hz 电平 dB)
            # 块判决用原始过阈帧，这样块尾刚起声、滞回还没翻的 5ms 也能被识别为有声。
            self.marks.append((self.n_frames * PAUSE_HOP, bool(self.state),
                               bool(v > thr), float(v)))
            self.n_frames += 1
            if not self.state:
                self.pause_frames += 1

    def gain_for(self, s0: int, n: int) -> np.ndarray:
        """输出样本区间 [s0, s0+n) 的线性增益。"""
        if n <= 0:
            return np.ones(max(0, n), np.float32)
        if not self.g_frames:
            return np.ones(n, np.float32)
        idx = np.fromiter((i for i, _ in self.g_frames), np.int64, len(self.g_frames))
        val = np.fromiter((v for _, v in self.g_frames), np.float64, len(self.g_frames))
        pos = np.arange(s0, s0 + n, dtype=np.float64) + 0.5
        gdb = np.interp(pos, idx + PAUSE_HOP / 2.0, val, left=val[0], right=val[-1])
        return (10.0 ** (gdb / 20.0)).astype(np.float32)


@dataclass
class StreamSession:
    step_path: str
    E: np.ndarray
    g: np.ndarray
    chunk_frames: int = CHUNK_FRAMES
    threads: int = 1
    post_denoise: bool | None = None  # None → 取环境变量 VF_POST_DENOISE
    capture: bool | None = None       # None → 取环境变量 VF_CAPTURE（诊断录制）

    def __post_init__(self):
        self.sess = make_session(self.step_path, self.threads)
        self.post_denoise = POST_DENOISE if self.post_denoise is None else bool(self.post_denoise)
        # 运行中可切换：pd_target 是目标（0/1），pd_mix 是当前混合系数（4 块内线性过渡）。
        self.pd_target = 1.0 if self.post_denoise else 0.0
        self.pd_mix = self.pd_target
        self.pd_emin: float | None = None
        self.pd_noise: np.ndarray | None = None
        self.pd_gain: np.ndarray | None = None
        self.pause_ff = PauseFeedForward() if PAUSE_STEP else None
        self.hf_gain = None
        if HF_SHELF_DB > 0.0:
            f = np.arange(WIN // 2 + 1, dtype=np.float64) * SR / WIN
            shape = f ** 4 / (f ** 4 + HF_SHELF_FC ** 4)
            self.hf_gain = (10.0 ** (HF_SHELF_DB * shape / 20.0)).astype(np.float32)[:, None]
        self.emask = np.ones((1, self.E.shape[1]), np.bool_)
        self.az = OnlineAnalyzer()
        self.istft = OnlineISTFT()
        self.h = np.zeros((LAYERS, NBAND, HID), np.float32)
        self.c = np.zeros((LAYERS, NBAND, HID), np.float32)
        self.pending = np.zeros((0, WIN), np.float32)
        self.samples_in = 0
        self.chunk_i = 0
        self.stats: list[StepStat] = []
        self.total_compute_ms = 0.0
        self.total_audio_s = 0.0
        self.closed = False
        # 旁路降级状态：滞回门限在 server 侧调用 should_bypass() 演进
        self.mode: str | None = None          # 上一块实际模式 "model" / "bypass"
        self.in_bypass = False
        self.bypass_since = 0.0
        self.bypass_chunks = 0
        self.mode_skips = 0                   # 跳空块数（不计入迟到旁路）
        self.mode_switches = 0
        self.atten_db: float | None = None         # 模型块「输出-输入」长期衰减
        self.last_out_db: float | None = None      # 上一块已发射输出的电平
        self.applied_gain_db = 0.0
        self.bypass_gain_db = 0.0                  # 最近一次旁路块的电平对齐增益
        # 「旁路只在停顿段生效」状态与统计（BYPASS_PAUSE_ONLY=1 时有效）
        self.cur_block_pause = False    # 最近一次 feed_pause() 的判决：True=整块按停顿
        self.cur_block_skip = False     # 本块是否「纯静音」可跳空（不送模型）
        self._last_loud_sample = -10 ** 9
        self._fed_chunk_s = 0.0         # 最近一次 feed_pause() 的输入块时长（秒）
        self.speech_chunks = 0
        self.speech_s = 0.0
        self.skip_chunks = 0
        self.skip_s = 0.0
        self.speech_keep_chunks = 0     # 迟到的有声块：旧逻辑会旁路，新逻辑留在模型
        self.speech_kept_s = 0.0
        self.pause_bypass_chunks = 0
        self.pause_bypass_s = 0.0
        self.force_bypass_chunks = 0    # 安全阀触发次数
        self.force_bypass_s = 0.0
        # 跳空后预热：连续未进模型的块计数 + 最近 K 块的帧（只保留补跑要用的）
        self._stale_run = 0
        self._warm_buf: collections.deque = collections.deque(
            maxlen=max(1, SKIP_WARM_BLOCKS))
        self.warm_chunks = 0
        self.warm_ms = 0.0
        self.prof: list[dict] | None = [] if PROFILE else None
        self.prof_push: list[float] | None = [] if PROFILE else None
        self.prof_push_ms = 0.0
        # 诊断录制（CAPTURE 关闭时全为 None，不产生任何额外开销）
        self._cap_on = CAPTURE if self.capture is None else bool(self.capture)
        if self._cap_on:
            global _CAP_SEQ
            _CAP_SEQ += 1
            self.cap_seq = _CAP_SEQ
            self.cap_name = time.strftime("cap_%Y%m%d-%H%M%S") + f"_s{self.cap_seq}"
        else:
            self.cap_seq = 0
            self.cap_name = None
        self.cap_in: list[np.ndarray] | None = [] if self._cap_on else None
        self.cap_out: list[np.ndarray] | None = [] if self._cap_on else None
        self.cap_in_n = 0
        self.cap_out_n = 0
        self.cap_files: dict | None = None

    # ---- 诊断录制：输入 / 输出 / 逐块标记 ----
    def _cap_in(self, x: np.ndarray):
        if self.cap_in is None:
            return
        cap = int(CAPTURE_MAX_S * SR)
        if self.cap_in_n >= cap:
            return
        x = np.asarray(x, np.float32).reshape(-1)[:cap - self.cap_in_n]
        if len(x):
            self.cap_in.append(x.copy())
            self.cap_in_n += len(x)

    def _cap_out(self, outs: list[np.ndarray]):
        if self.cap_out is None:
            return
        cap = int(CAPTURE_MAX_S * SR)
        for out in outs:
            if self.cap_out_n >= cap:
                return
            y = np.asarray(out, np.float32).reshape(-1)[:cap - self.cap_out_n]
            if len(y):
                self.cap_out.append(y.copy())
                self.cap_out_n += len(y)

    @staticmethod
    def _write_wav(path: str, data: np.ndarray):
        pcm = np.clip(np.asarray(data, np.float32), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())

    def _cap_write(self):
        """一条流结束时落盘：<name>_in.wav / _out.wav / _blocks.txt。只写一次。"""
        if self.cap_in is None or not self.cap_in:
            return
        if self.cap_files is not None:
            return
        try:
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            data_in = np.concatenate(self.cap_in)
            data_out = np.concatenate(self.cap_out) if self.cap_out else np.zeros(0, np.float32)
            f_in = os.path.join(CAPTURE_DIR, f"{self.cap_name}_in.wav")
            f_out = os.path.join(CAPTURE_DIR, f"{self.cap_name}_out.wav")
            f_blk = os.path.join(CAPTURE_DIR, f"{self.cap_name}_blocks.txt")
            self._write_wav(f_in, data_in)
            self._write_wav(f_out, data_out)
            with open(f_blk, "w", encoding="utf-8") as f:
                f.write("# vf_capture/1\n")
                f.write(f"# session={self.cap_name} sr={SR} post_denoise={int(bool(self.post_denoise))}"
                        f" max_s={CAPTURE_MAX_S:g}\n")
                f.write(f"# in_samples={len(data_in)} out_samples={len(data_out)}"
                        f" in_s={len(data_in)/SR:.3f} out_s={len(data_out)/SR:.3f}\n")
                f.write("# 输入=送进模型的原样样本（VAD 之后、TSE 之前）；输出=实际下发音频（后置降噪状态见上）\n")
                f.write("# columns: chunk,mode,compute_ms,out_len,skipped\n")
                for s in self.stats:
                    f.write(f"{s.chunk},{s.mode},{s.compute_ms},{s.out_len},{int(s.skipped)}\n")
            self.cap_files = {"dir": CAPTURE_DIR, "in": f_in, "out": f_out, "blocks": f_blk,
                              "in_s": round(len(data_in) / SR, 3),
                              "out_s": round(len(data_out) / SR, 3)}
        except OSError as e:
            self.cap_files = {"dir": CAPTURE_DIR, "error": str(e)}

    # ---- 内部：跑一个 chunk ----
    def _warm_if_needed(self):
        """跳空/旁路连续太久后，恢复时补喂最近 K 块：只推进 h/c，不产生输出。

        这些块当初已经按 −18dB 直通播放过，这里只是把模型状态补到 resume 前一帧，
        避免恢复首块用陈旧状态（离线实测电平偏高约 +8.6dB）。不碰 istft/OLA、
        不写 stats、不改时间轴，因此不会重复播放。
        """
        if (SKIP_WARM_BLOCKS <= 0 or self._stale_run < SKIP_WARM_AFTER
                or not self._warm_buf):
            self._stale_run = 0
            self._warm_buf.clear()
            return
        t0 = time.perf_counter()
        n = 0
        while self._warm_buf:
            frames = self._warm_buf.popleft()
            spec = analyze_frames(frames)[None]
            spec_ri = np.stack([spec.real, spec.imag], axis=1).astype(np.float32)
            _, self.h, self.c = self.sess.run(None, {
                "spec_ri": spec_ri, "E": self.E, "emask": self.emask, "g": self.g,
                "h": self.h, "c": self.c})
            n += 1
        dt = (time.perf_counter() - t0) * 1e3
        self.warm_chunks += n
        self.warm_ms += dt
        self.total_compute_ms += dt      # 算力口径计入，RTF 不虚低
        self._stale_run = 0

    def _run(self, frames: np.ndarray) -> np.ndarray:
        self._warm_if_needed()
        t_enter = time.perf_counter() if PROFILE else 0.0
        spec = analyze_frames(frames)[None]                 # (1,257,n)
        t_feat = time.perf_counter() if PROFILE else 0.0
        spec_ri = np.stack([spec.real, spec.imag], axis=1).astype(np.float32)
        t0 = time.perf_counter()
        pd_on = self.pd_mix > 0.0 or self.pd_target > 0.0
        pin = None
        if pd_on:
            pin = (spec[0].real ** 2 + spec[0].imag ** 2).astype(np.float32)
        t_prep = time.perf_counter() if PROFILE else 0.0
        est, self.h, self.c = self.sess.run(None, {
            "spec_ri": spec_ri, "E": self.E, "emask": self.emask, "g": self.g,
            "h": self.h, "c": self.c})
        t_run = time.perf_counter() if PROFILE else 0.0
        spec = est[0, 0] + 1j * est[0, 1]
        if pd_on:
            spec = self._post_denoise(spec, pin, self.pd_mix)
            self._pd_advance()
        spec = self._hf_shelf_spec(spec)
        t_post = time.perf_counter() if PROFILE else 0.0
        compute_ms = (time.perf_counter() - t0) * 1000
        self.istft.add_batch(spec)
        t_istft = time.perf_counter() if PROFILE else 0.0
        in_db = 20.0 * np.log10(float(np.sqrt(np.mean(np.square(
            frames, dtype=np.float64)))) + 1e-12)
        out = self._apply_out(self.istft.emit(), "model", in_db)
        if self.pause_ff is not None and len(out):
            # 前馈：用输入侧判决（样本级）给本块输出追加停顿衰减，不改长度与时间轴
            s0 = int(self.istft.next_q - PAD - len(out))
            out = (out * self.pause_ff.gain_for(s0, len(out))).astype(np.float32, copy=False)
        if PROFILE:
            t_end = time.perf_counter()
            self.prof.append({
                "chunk": self.chunk_i,
                "feat": (t_feat - t_enter) * 1e3,
                "prep": (t_prep - t_feat) * 1e3,
                "onnx": (t_run - t_prep) * 1e3,
                "post": (t_post - t_run) * 1e3,
                "istft": (t_istft - t_post) * 1e3,
                "emit": (t_end - t_istft) * 1e3,
                "total": (t_end - t_enter) * 1e3,
            })
        self.total_compute_ms += compute_ms
        n = frames.shape[0]
        self.total_audio_s += n * HOP / SR
        needed = (self.istft.next_frame - 1) * HOP + WIN - PAD
        out_end = self.istft.next_q - PAD
        self.stats.append(StepStat(
            chunk=self.chunk_i, frames=n, compute_ms=round(compute_ms, 3),
            samples_in=self.samples_in, needed_samples=int(needed),
            out_end=int(out_end), out_len=int(len(out)),
            algo_latency_ms=round((self.samples_in - out_end) * 1000.0 / SR, 2),
            mode="model", gain_db=round(self.applied_gain_db, 2)))
        self.chunk_i += 1
        return out

    def set_post_denoise(self, on: bool) -> None:
        """运行中切换后置降噪：立即生效，增益在 PD_RAMP_BLOCKS 块内线性淡入/淡出。

        不重建会话、不改变时间轴与延迟；关→开时沿用已有噪底估计（若为空则重新学习），
        开→关时保留 pd_noise/pd_gain，便于快速再次打开。
        """
        self.post_denoise = bool(on)
        self.pd_target = 1.0 if self.post_denoise else 0.0

    def _pd_advance(self) -> None:
        step = 1.0 / max(1, PD_RAMP_BLOCKS)
        if self.pd_mix < self.pd_target:
            self.pd_mix = min(self.pd_target, self.pd_mix + step)
        elif self.pd_mix > self.pd_target:
            self.pd_mix = max(self.pd_target, self.pd_mix - step)

    def _post_denoise(self, spec: np.ndarray, pin: np.ndarray,
                      mix: float = 1.0) -> np.ndarray:
        """用输入频谱的稳态噪底对输出做谱减；只改幅度、保留相位。

        - 噪底只在「输入块能量接近历史最低」时更新（噪声段），语音段不更新，
          因此不会把目标人声当成噪声学进去；
        - 增益下限 -12dB，帧间平滑，放在 OLA 之前，不改变时间轴。
        - mix<1 时把增益向 1 线性插值（1 + mix*(g-1)），用于开关降噪的过渡；
          平滑状态 pd_gain 仍按完整增益更新，避免过渡期污染后续平滑。
        """
        pout = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        pin_m = pin.mean(axis=1, keepdims=True)          # (257,1) 本块输入功率
        e_in = float(pin.mean())
        if self.pd_emin is None:
            self.pd_emin = e_in
            self.pd_noise = np.zeros_like(pin_m)
        if e_in < self.pd_emin:
            self.pd_emin = e_in
        else:
            self.pd_emin *= 1.0005                        # 约 +0.002dB/块缓慢回升
        if e_in < self.pd_emin * PD_NOISE_RATIO:          # 仅噪声段更新
            self.pd_noise = (1.0 - PD_ALPHA) * self.pd_noise + PD_ALPHA * pin_m
        gain = np.clip(1.0 - PD_BETA * self.pd_noise / (pout + 1e-12), PD_FLOOR, 1.0)
        if self.pd_gain is not None:
            gain = PD_SMOOTH * self.pd_gain + (1.0 - PD_SMOOTH) * gain
        self.pd_gain = gain[:, -1:].copy()
        if mix < 1.0:
            gain = 1.0 + mix * (gain - 1.0)
        return (spec * gain).astype(np.complex64)

    def _hf_shelf_spec(self, spec: np.ndarray) -> np.ndarray:
        """高频搁架：整块频谱乘一条固定的 257-bin 实数增益（逐块、零相位）。"""
        return spec if self.hf_gain is None else spec * self.hf_gain

    def feed_pause(self, x: np.ndarray) -> bool:
        """先喂输入侧停顿判决，返回本块是否「整块按停顿处理」。

        跨块边界规则：只要本块内任一 5ms 帧判为语音，整块就按有声处理——块内
        前段停顿、后段起声时，宁可多缓冲一个块（约 32ms），也不把起声裁进旁路。
        """
        self._fed_chunk_s = len(x) / float(SR)
        if self.pause_ff is None:
            self.cur_block_pause = False
            return False
        n0 = self.pause_ff.n_frames
        self.pause_ff.feed(x)
        d = self.pause_ff.n_frames - n0
        if d <= 0:                       # 不足一个 5ms 帧的残片：沿用上一块判决
            return self.cur_block_pause
        marks = self.pause_ff.marks
        k = min(d, len(marks))
        rows = [marks[-i] for i in range(k, 0, -1)]     # 时间正序
        has_speech = any(r[1] or r[2] for r in rows)
        self.cur_block_pause = not has_speech
        if has_speech:
            self.speech_chunks += 1
            self.speech_s += self._fed_chunk_s
            self._last_loud_sample = rows[-1][0]
        # 跳空判据：整块无过阈帧 + 块内 100–2000Hz 最高电平 ≤ 绝对上限
        # + 距上一个过阈帧 ≥ SKIP_HOLD_MS（静音挂起），三者同时成立才省掉推理。
        blk_start = rows[0][0]
        quiet = max(r[3] for r in rows) <= SKIP_ABS_DB
        hang = (blk_start - self._last_loud_sample) >= SKIP_HOLD_MS * SR / 1000.0
        self.cur_block_skip = bool(SKIP_SILENT and not has_speech and quiet and hang)
        if self.cur_block_skip:
            self.skip_chunks += 1
            self.skip_s += self._fed_chunk_s
        return self.cur_block_pause

    def should_bypass(self, waited_ms: float, pause: bool | None = None,
                      now: float | None = None) -> bool:
        """旁路降级判定。

        旧逻辑（VF_BYPASS_PAUSE_ONLY=0）：只看队列等待，进入用高门限、退出用低
        门限并强制最短驻留（避免 15.62Hz 的块率包络调制）。

        新逻辑（默认）：旁路只对判为停顿的块生效——有声块即使等待超阈值也留在
        模型路径上，用缓冲换延迟，保住分离质量；有声块一旦出现就从旁路立刻回
        模型（不守 dwell）。安全阀：任何块等待超过 BYPASS_FORCE_MS（默认 700ms）
        时仍强制旁路，避免 RTF>1 时队列无限累积。
        """
        now = time.perf_counter() if now is None else now
        # 没有输入侧停顿时钟（VF_PAUSE_STEP=0）时无法区分停顿块，退回旧的「只看等待」逻辑
        if BYPASS_PAUSE_ONLY and self.pause_ff is not None:
            is_pause = self.cur_block_pause if pause is None else bool(pause)
            if waited_ms > BYPASS_FORCE_MS:            # 安全阀优先于质量
                self.force_bypass_chunks += 1
                self.force_bypass_s += self._fed_chunk_s
                if not self.in_bypass:
                    self.in_bypass = True
                    self.bypass_since = now
                    self.mode_switches += 1
                return True
            if not is_pause:                           # 有声块：永远走模型
                if waited_ms > BYPASS_ENTER_MS:
                    self.speech_keep_chunks += 1
                    self.speech_kept_s += self._fed_chunk_s
                if self.in_bypass:
                    self.in_bypass = False
                    self.mode_switches += 1
                return False
            if not self.in_bypass:
                if waited_ms > BYPASS_ENTER_MS:
                    self.in_bypass = True
                    self.bypass_since = now
                    self.mode_switches += 1
                    self.pause_bypass_chunks += 1
                    self.pause_bypass_s += self._fed_chunk_s
                    return True
                return False
            dwelt_ms = (now - self.bypass_since) * 1000.0
            if waited_ms < BYPASS_EXIT_MS and dwelt_ms >= BYPASS_DWELL_MS:
                self.in_bypass = False
                self.mode_switches += 1
                return False
            self.pause_bypass_chunks += 1
            self.pause_bypass_s += self._fed_chunk_s
            return True
        if not self.in_bypass:
            if waited_ms > BYPASS_ENTER_MS:
                self.in_bypass = True
                self.bypass_since = now
                self.mode_switches += 1
                return True
            return False
        dwelt_ms = (now - self.bypass_since) * 1000.0
        if waited_ms < BYPASS_EXIT_MS and dwelt_ms >= BYPASS_DWELL_MS:
            self.in_bypass = False
            self.mode_switches += 1
            return False
        return True

    def _apply_out(self, out: np.ndarray, mode: str,
                   in_db: float | None = None) -> np.ndarray:
        """块级电平对齐 + 交界增益交叉淡化；不改变样本数、时间轴与相位。

        - 旁路块按「模型块的长期衰减」整体对齐到模型块响度（限幅 -6..0dB）：
          衰减量在同一块内容上由模型块自己测得（out-in），比拿两种不同内容的
          绝对电平相减稳健，不会因「旁路恰好落在响段」而出现整体忽大忽小；
        - 每次模式切换，在块首 XFADE_MS（默认 8ms = 128 样本）内把增益从「与上一块
          末电平连续的值」线性过渡到本模式稳态增益；
        - 交叉淡化只作用于增益包络，不做波形重叠，因此 OLA 归一与输出长度不变。
        """
        n = len(out)
        if n == 0:
            return out
        rms = float(np.sqrt(np.mean(np.square(out, dtype=np.float64))) + 1e-12)
        rdb = 20.0 * np.log10(rms)
        # 只用有声块统计：静音块（-240dB）会把 EMA 拖到地板
        if in_db is not None and in_db > -80.0 and rdb > -80.0:
            att = rdb - in_db
            self.atten_db = att if self.atten_db is None else \
                (1.0 - LEVEL_EMA_DB) * self.atten_db + LEVEL_EMA_DB * att

        gain = 1.0
        if mode == "bypass" and BYPASS_LEVEL_MATCH and self.atten_db is not None:
            gain = 10.0 ** (float(np.clip(self.atten_db,
                                          -MATCH_GAIN_DB_LIMIT, 0.0)) / 20.0)
            self.bypass_gain_db = 20.0 * np.log10(gain)
        self.applied_gain_db = 20.0 * np.log10(gain)

        if self.mode is not None and mode != self.mode and XFADE_SAMPLES > 0:
            # 起点增益：让本块首样本的响度接上上一块末响度（而不是直接跳到本模式增益）
            if self.last_out_db is not None and self.last_out_db > -80.0:
                g0 = float(np.clip(10.0 ** ((self.last_out_db - rdb) / 20.0), 0.25, 4.0))
            else:
                g0 = gain
            k = min(XFADE_SAMPLES, n)
            ramp = np.linspace(g0, gain, k, dtype=np.float32)
            out = out.copy()
            out[:k] *= ramp
            if k < n:
                out[k:] *= np.float32(gain)
        elif gain != 1.0:
            out = out * np.float32(gain)

        self.last_out_db = 20.0 * np.log10(
            float(np.sqrt(np.mean(np.square(out, dtype=np.float64)))) + 1e-12)
        self.mode = mode
        return out.astype(np.float32, copy=False)

    def _drain(self) -> list[np.ndarray]:
        outs = []
        while len(self.pending) >= self.chunk_frames:
            grp, self.pending = self.pending[:self.chunk_frames], self.pending[self.chunk_frames:]
            outs.append(self._run(grp))
        return outs

    def _bypass_run(self, frames: np.ndarray, skip: bool = False) -> np.ndarray:
        """降级旁路：不做模型推理，STFT→iSTFT 直通，只推进时间轴。

        LSTM 状态 (h,c) 不更新，模型上下文会缺这 32ms；STFT/OLA 照常推进，
        因此输出时间轴连续、不会漂移，也不会出现静音或爆音。
        skip=True 表示这是「跳空帧」（纯静音主动省算力），输出仍是 −6dB 电平
        对齐 + pause 前馈的直通，只是统计上不计入迟到旁路。
        """
        spec = analyze_frames(frames)[None]
        if self.hf_gain is not None:
            spec = spec * self.hf_gain
        self.istft.add_batch(spec[0])
        # 状态补跑用：记住最近 K 个未进模型的块（只存帧，不重复出音频）
        self._stale_run += 1
        if SKIP_WARM_BLOCKS > 0:
            self._warm_buf.append(np.array(frames, copy=True))
        out = self._apply_out(self.istft.emit(), "bypass")
        if self.pause_ff is not None and len(out):
            # 与 model 路径一致：旁路块同样叠加停顿前馈衰减，避免停顿段底噪被直通放出
            s0 = int(self.istft.next_q - PAD - len(out))
            out = (out * self.pause_ff.gain_for(s0, len(out))).astype(np.float32, copy=False)
        if skip:
            self.mode_skips += 1
        else:
            self.bypass_chunks += 1
        n = frames.shape[0]
        self.total_audio_s += n * HOP / SR
        needed = (self.istft.next_frame - 1) * HOP + WIN - PAD
        out_end = self.istft.next_q - PAD
        self.stats.append(StepStat(
            chunk=self.chunk_i, frames=n, compute_ms=0.0,
            samples_in=self.samples_in, needed_samples=int(needed),
            out_end=int(out_end), out_len=int(len(out)),
            algo_latency_ms=round((self.samples_in - out_end) * 1000.0 / SR, 2),
            skipped=True, mode="skip" if skip else "bypass",
            gain_db=round(self.applied_gain_db, 2)))
        self.chunk_i += 1
        return out

    def bypass(self, samples: np.ndarray, pause_fed: bool = False,
               skip: bool = False) -> list[np.ndarray]:
        """降级旁路 / 跳空路径：与 push() 完全相同的分帧/出帧节奏，只跳过模型。

        pause_fed=True 表示调用方已用 feed_pause() 喂过本块，避免重复推进停顿时钟。
        skip=True 表示纯静音跳空（输出仍是 −18dB 直通，只是不计入迟到旁路统计）。
        """
        if self.closed:
            return []
        x = np.asarray(samples, np.float32).reshape(-1)
        if len(x) == 0:
            return []
        self.samples_in += len(x)
        self._cap_in(x)
        if self.pause_ff is not None and not pause_fed:
            self.pause_ff.feed(x)
        frames = self.az.push(x)
        if len(frames):
            self.pending = frames if len(self.pending) == 0 else \
                np.concatenate([self.pending, frames])
        outs = []
        while len(self.pending) >= self.chunk_frames:
            grp, self.pending = self.pending[:self.chunk_frames], self.pending[self.chunk_frames:]
            outs.append(self._bypass_run(grp, skip))
        self._cap_out(outs)
        return outs

    def push(self, samples: np.ndarray, pause_fed: bool = False) -> list[np.ndarray]:
        if self.closed:
            return []
        t_push = time.perf_counter() if PROFILE else 0.0
        x = np.asarray(samples, np.float32).reshape(-1)
        if len(x) == 0:
            return []
        self.samples_in += len(x)
        self._cap_in(x)
        if self.pause_ff is not None and not pause_fed:
            self.pause_ff.feed(x)
        frames = self.az.push(x)
        if len(frames):
            self.pending = frames if len(self.pending) == 0 else \
                np.concatenate([self.pending, frames])
        outs = self._drain()
        self._cap_out(outs)
        if PROFILE:
            dt = (time.perf_counter() - t_push) * 1e3
            self.prof_push_ms += dt
            self.prof_push.append(dt)
        return outs

    def flush(self) -> list[np.ndarray]:
        """停止：补零补齐最后一帧，跑完余下帧并发射剩余样本。"""
        if self.closed:
            return []
        self.closed = True
        self.az.pad_tail(self.samples_in)
        frames = self.az.push(np.zeros(0, np.float32))
        if len(frames):
            self.pending = frames if len(self.pending) == 0 else \
                np.concatenate([self.pending, frames])
        outs = []
        while len(self.pending) > 0:
            n = min(self.chunk_frames, len(self.pending))
            grp, self.pending = self.pending[:n], self.pending[n:]
            outs.append(self._run(grp))
        # 尾帧补齐后仍可能剩余不足一帧覆盖的输出
        tail = self.istft.emit()
        if len(tail):
            outs.append(tail)
        self._cap_out(outs)
        return outs

    def summary(self) -> dict:
        if self._cap_on:
            self._cap_write()
        audio_s = max(1e-9, self.total_audio_s)
        comp = [s.compute_ms for s in self.stats if not s.skipped]
        skipped = [s for s in self.stats if s.skipped]
        out = {
            "chunks": len(self.stats),
            "chunk_frames": self.chunk_frames,
            "samples_in": self.samples_in,
            "audio_s": round(self.samples_in / SR, 3),
            "compute_total_ms": round(self.total_compute_ms, 1),
            "compute_mean_ms": round(float(np.mean(comp)) if comp else 0.0, 2),
            "compute_p95_ms": round(float(np.percentile(comp, 95)) if comp else 0.0, 2),
            "rtf": round(self.total_compute_ms / 1000.0 / audio_s, 3),
            "algo_latency_ms": self.stats[-1].algo_latency_ms if self.stats else None,
            "skipped_chunks": len(skipped),
            "skipped_ms": round(sum(s.out_len for s in skipped) * 1000.0 / SR, 1),
            "bypass_chunks": int(self.bypass_chunks),
            "bypass_ratio": round(self.bypass_chunks / max(1, len(self.stats)), 4),
            "mode_switches": int(self.mode_switches),
            "bypass_match_db": round(self.bypass_gain_db, 2),
            "xfade_ms": round(XFADE_SAMPLES * 1000.0 / SR, 1),
            "post_denoise": bool(self.post_denoise),
            "hf_shelf_db": HF_SHELF_DB,
            "pause_step": bool(self.pause_ff is not None),
            "bypass_pause_only": bool(BYPASS_PAUSE_ONLY),
            "bypass_force_ms": BYPASS_FORCE_MS,
            "skip_silent": bool(SKIP_SILENT),
            "skip_abs_db": SKIP_ABS_DB,
            "skip_hold_ms": SKIP_HOLD_MS,
            "skip_chunks": int(self.skip_chunks),
            "skip_s": round(self.skip_s, 3),
            "mode_skips": int(self.mode_skips),
            "warm_blocks": SKIP_WARM_BLOCKS,
            "warm_after": SKIP_WARM_AFTER,
            "warm_chunks": int(self.warm_chunks),
            "warm_ms": round(self.warm_ms, 1),
            "speech_chunks": int(self.speech_chunks),
            "speech_s": round(self.speech_s, 3),
            "speech_keep_chunks": int(self.speech_keep_chunks),
            "speech_kept_s": round(self.speech_kept_s, 3),
            "pause_bypass_chunks": int(self.pause_bypass_chunks),
            "pause_bypass_s": round(self.pause_bypass_s, 3),
            "force_bypass_chunks": int(self.force_bypass_chunks),
            "force_bypass_s": round(self.force_bypass_s, 3),
        }
        if self._cap_on and self.cap_files:
            out["capture"] = self.cap_files
        if PROFILE and self.prof:
            stages = {}
            for key in ("feat", "prep", "onnx", "post", "istft", "emit", "total"):
                v = np.asarray([r[key] for r in self.prof], np.float64)
                stages[key] = {
                    "mean": round(float(v.mean()), 3),
                    "p50": round(float(np.percentile(v, 50)), 3),
                    "p95": round(float(np.percentile(v, 95)), 3),
                    "max": round(float(v.max()), 3),
                }
            out["profile"] = {
                "blocks": len(self.prof),
                "stage_ms": stages,
                "push_wall_ms": round(self.prof_push_ms, 1),
                "rtf_push": round(self.prof_push_ms / 1000.0 / audio_s, 3),
                "rtf_model": round(stages["onnx"]["mean"] * len(self.prof)
                                   / 1000.0 / audio_s, 3),
                "rtf_legacy_compute": out["rtf"],
            }
        return out


def stats_msg(stat: StepStat) -> dict:
    return {"type": "stats", "chunk": stat.chunk, "frames": stat.frames,
            "compute_ms": stat.compute_ms, "samples_in": stat.samples_in,
            "needed_samples": stat.needed_samples, "out_end": stat.out_end,
            "out_len": stat.out_len, "algo_latency_ms": stat.algo_latency_ms,
            "skipped": stat.skipped, "mode": stat.mode, "gain_db": stat.gain_db}


if __name__ == "__main__":
    print(json.dumps({"chunk_frames": CHUNK_FRAMES, "sr": SR}, ensure_ascii=False))
