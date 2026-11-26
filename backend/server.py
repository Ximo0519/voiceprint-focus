"""声纹聚焦 · 场景1 实时服务：FastAPI + WebSocket + ONNX Runtime CPU。

上行契约
  {"type":"enroll"}            开始注册（随后二进制为 float32 单声道 16k 音频）
  {"type":"start"}             结束注册并开流；服务端做 VAD 质检 + 编码 → {"type":"ready"}
  二进制 frame                 流式音频（任意长度，建议 8~32ms）
  {"type":"stop"}              结束：补零冲洗尾帧 → 剩余 out → {"type":"done"}
下行契约
  {"type":"ready", ...}        注册质检结果 / 模型信息
  {"type":"stats", ...}        每个 32ms chunk 一项（紧接其二进制 out）
  二进制 out                   float32 LE 处理后的样本
  {"type":"done","summary":{...}}
  {"type":"error","message":...}

HTTP 文件接口（离线/批量，不动 WS 契约）
  POST /api/enroll/file      multipart: file(音频) + name(可选)
                             → {"ok":true,"enroll_id":"reg_...","vad":...,"enroll_qc":...}
                             质检与录制注册同链路（vad_report + enroll_qc），不过则 422
  GET  /api/enroll/list      已注册声纹列表（registrations/*.json）
  POST /api/process/file     multipart: file(音频) + enroll_id(可选，缺省用 demo)
                             + out_format(wav|f32)
                             → StreamingResponse：边处理边输出（WAV 头先行，PCM 分块下发）
  WS {"type":"start","enroll_id":"reg_..."}
                             跳过录制注册，直接用文件注册的声纹开流
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import struct
import time
import wave
import weakref
from collections import deque
from math import gcd

import numpy as np
import psutil
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    import soundfile as _sf
except ImportError:                                    # noqa: SIM108
    _sf = None

from dsp import HOP, PAD, WIN
from campp import DIAG_DIR as CAMPP_DIAG_DIR, MODEL_NAME as CAMPP_MODEL_NAME
from campp import THRESHOLD_REF as CAMPP_THRESHOLD_REF, CamppObserver
from campp import observe as campp_observe
from engine import (BYPASS_DWELL_MS, BYPASS_ENTER_MS, BYPASS_EXIT_MS,
                    BYPASS_FORCE_MS, BYPASS_LEVEL_MATCH, BYPASS_PAUSE_ONLY,
                    CAPTURE, CAPTURE_DIR, CAPTURE_MAX_S, HF_SHELF_DB,
                    ENROLL_DUR_MAX, ENROLL_DUR_MIN,
                    ENROLL_ID_COS_MIN, ENROLL_PITCH_SHIFT_MAX_ST,
                    PAUSE_ATT_MS, PAUSE_DB, PAUSE_REL_MS, PAUSE_STEP,
                    PD_FLOOR_DB, POST_DENOISE, SKIP_ABS_DB, SKIP_HOLD_MS, SKIP_SILENT,
                    SKIP_WARM_AFTER, SKIP_WARM_BLOCKS,
                    XFADE_MS, Enroller, StreamSession,
                    enroll_qc, split_enroll_halves, stats_msg, vad_report)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(APP_DIR, "models", "x16_stream")
DEMO_DIR = os.path.join(APP_DIR, "backend", "assets", "demo")
FRONT_DIR = os.path.join(APP_DIR, "frontend", "dist")
SR = 16000
# 录制注册上限：与 vad_report 的时长闸门共用同一口径（engine.ENROLL_DUR_MAX）
MAX_ENROLL_S = ENROLL_DUR_MAX
# 注册音频留档（默认开启）：16k 单声道 PCM_16 WAV + 同名 .json 元数据。
# 目录不进仓库（app/.gitignore 已忽略）；VF_ENROLL_SAVE=0 可关闭，VF_ENROLL_DIR 可改目录。
ENROLL_SAVE = os.environ.get("VF_ENROLL_SAVE", "1").strip().lower() not in (
    "0", "", "false", "no", "off")
ENROLL_DIR = os.environ.get(
    "VF_ENROLL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrollments"))
# 文件式注册：注册音频上限（秒）与批量处理文件上限（秒）；声纹嵌入持久化目录
MAX_ENROLL_FILE_S = float(os.environ.get("VF_MAX_ENROLL_FILE_S", ENROLL_DUR_MAX))
MAX_PROCESS_S = float(os.environ.get("VF_MAX_PROCESS_S", 1800.0))
# 上传体积上限：读之前先限长，避免恶意/误传的大文件把内存打满
MAX_UPLOAD_BYTES = int(float(os.environ.get("VF_MAX_UPLOAD_MB", 64)) * 1024 * 1024)
# CORS：默认只放行本机来源；外部来源用 VF_CORS_ORIGINS（逗号分隔）显式追加
CORS_BASE_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"]


def _cors_origins() -> list[str]:
    extra = [o.strip() for o in os.environ.get("VF_CORS_ORIGINS", "").split(",")
             if o.strip()]
    return CORS_BASE_ORIGINS + extra
REG_DIR = os.path.join(ENROLL_DIR, "registrations")
# 绑核开关（默认关，行为与之前完全一致）：
#   off/none/0 = 不绑（默认）；auto = 自动挑一颗最闲的 P 核；e = 一颗 E 核（对比用）；
#   p / p0 / p1 ... = 按自检出的 P 核列表下标选（核号不写死，换机器自动重排）。
CPU_AFFINITY = os.environ.get("VF_CPU_AFFINITY", "off").strip().lower()

log = logging.getLogger("voiceprint")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def _env_ms(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _fire_campp(wav: np.ndarray, ctx: dict, **extra) -> None:
    """把 CAM++ 旁路观察丢进后台线程：不阻塞注册、异常不外溢、不改变任何回包。"""
    ob = STATE.get("campp")
    if ob is None:
        return
    try:
        t = asyncio.create_task(asyncio.to_thread(
            campp_observe, ob, wav, {**ctx, **extra}))
        _BG.add(t)
        t.add_done_callback(_BG.discard)
    except Exception as e:                          # noqa: BLE001
        log.warning("campp schedule failed: %r", e)


def _save_enroll(wav: np.ndarray, meta: dict) -> dict | None:
    """注册音频留档：16k 单声道 PCM_16 WAV + 同名 .json 元数据。

    只做旁路留档：任何异常只告警，不改变注册判定与回包。
    """
    if not ENROLL_SAVE:
        return None
    try:
        os.makedirs(ENROLL_DIR, exist_ok=True)
        stem = "enroll_%s_%s_%02d" % (meta["ts_compact"], meta["session_id"], meta["seq"])
        p = os.path.join(ENROLL_DIR, stem + ".wav")
        pcm = np.clip(np.asarray(wav, np.float32), -1.0, 1.0)
        with wave.open(p, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((pcm * 32767.0).astype(np.int16).tobytes())
        full = {**meta, "wav": os.path.basename(p), "sr": SR,
                "samples": int(len(wav)), "duration_s": round(len(wav) / SR, 3),
                "outcome": "pending"}
        jp = os.path.join(ENROLL_DIR, stem + ".json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
        log.info("enroll saved: %s", p)
        return {"wav": p, "json": jp}
    except Exception as e:                          # noqa: BLE001
        log.warning("enroll save failed: %r", e)
        return None


def _tag_enroll(saved: dict | None, **extra) -> None:
    """把质检/注册结果回填到留档 .json（失败不影响主流程）。"""
    if not saved:
        return
    try:
        with open(saved["json"], encoding="utf-8") as f:
            meta = json.load(f)
        meta.update(extra)
        with open(saved["json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:                          # noqa: BLE001
        log.warning("enroll meta update failed: %r", e)


def _decode_audio(raw: bytes, filename: str) -> np.ndarray:
    """上传音频 → float32 单声道 16k。优先 soundfile（wav/flac/ogg）。

    只在显式 `.f32/.pcm/.raw` 扩展名时才退回裸 float32 PCM。前端 accept 里还有
    mp3/m4a/aac，这些格式解码失败时若把压缩字节当 PCM 读，会输出高幅刺耳噪声，
    必须明确报错而不是「猜」。多声道取均值，非 16k 用 resample_poly 重采样。
    """
    if _sf is not None:
        try:
            x, sr = _sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            x = np.ascontiguousarray(x.mean(axis=1), np.float32)
            if sr != SR:
                from scipy.signal import resample_poly
                g = gcd(int(sr), SR)
                x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            if x.size:
                return x
        except Exception:                           # noqa: BLE001
            pass
    if (filename or "").lower().endswith((".f32", ".pcm", ".raw")):
        x = np.frombuffer(raw, dtype=np.float32)
        if x.size >= 8000:
            return x.copy()
    raise ValueError("无法解码音频（支持 wav/flac/ogg；裸 PCM 请用 .f32 扩展名，%s）"
                     % (filename or "未知文件"))


def _save_registration(E: np.ndarray, g: np.ndarray, meta: dict) -> str:
    """声纹嵌入持久化：<REG_DIR>/<id>.npz + 同名 .json 元数据。"""
    os.makedirs(REG_DIR, exist_ok=True)
    rid = "reg_%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), os.urandom(3).hex())
    np.savez(os.path.join(REG_DIR, rid + ".npz"), E=E, g=g)
    with open(os.path.join(REG_DIR, rid + ".json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "id": rid}, f, ensure_ascii=False, indent=2)
    log.info("registration saved: %s", rid)
    return rid


def _load_registration(rid: str):
    """按 id 读回 (E, g, meta)；id 严格白名单校验，防目录穿越。"""
    if not rid or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid):
        return None
    p = os.path.join(REG_DIR, rid + ".npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    meta = {}
    jp = os.path.join(REG_DIR, rid + ".json")
    if os.path.exists(jp):
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return d["E"].astype(np.float32), d["g"].astype(np.float32), meta


def _list_registrations(limit: int = 50) -> list[dict]:
    if not os.path.isdir(REG_DIR):
        return []
    out = []
    for fn in os.listdir(REG_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(REG_DIR, fn), encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda m: m.get("ts", ""), reverse=True)
    return out[:limit]


class DemoAssetMissing(RuntimeError):
    """演示素材缺失（不进仓库），由调用方转成 400。"""


def _demo_reg():
    """demo 注册音频的嵌入（进程内缓存），供未带 enroll_id 的处理请求兜底。

    演示素材不进仓库（.gitignore），新克隆里可能不存在；缺素材时抛
    DemoAssetMissing 由调用方转 400，不能变成 500。
    """
    if STATE.get("demo_reg") is None:
        p = os.path.join(DEMO_DIR, "enroll", "demo_01.wav")
        try:
            with wave.open(p, "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()),
                                  np.int16).astype(np.float32) / 32767.0
        except (FileNotFoundError, wave.Error) as e:
            raise DemoAssetMissing(
                "演示素材缺失；请指定 enroll_id，或先运行 "
                "backend/tools/make_demo_assets.py 生成 demo 素材") from e
        E, g, _ = STATE["enroller"].encode(x)
        STATE["demo_reg"] = (E, g)
    return STATE["demo_reg"]


def _wav_header(n_samples: int) -> bytes:
    data = n_samples * 2
    return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16) +
            b"data" + struct.pack("<I", data))


def _new_session(E, g, capture=None):
    """建会话并登记到活跃表：降噪开关走运行期 STATE，而不是启动时的环境变量。"""
    s = StreamSession(os.path.join(MODEL_DIR, "step.onnx"), E, g,
                      threads=INFER_THREADS, post_denoise=STATE["post_denoise"],
                      capture=capture)
    STATE["active"][id(s)] = s
    return s


def _set_post_denoise(on: bool) -> dict:
    """运行中切换降噪：改全局值 + 应用到所有活跃会话（各自 4 块淡入/淡出）。"""
    STATE["post_denoise"] = bool(on)
    n = 0
    for s in list(STATE["active"].values()):
        s.set_post_denoise(bool(on))
        n += 1
    log.info("post_denoise -> %s (active sessions=%d)", bool(on), n)
    return {"post_denoise": bool(on), "active_sessions": n,
            "floor_db": PD_FLOOR_DB, "ramp_ms": 4 * 32}


# 流控参数（可用环境变量覆盖，便于压测对照）
IN_CAP_MS = _env_ms("VF_IN_MS", 320)        # 输入队列上限，满则丢最旧
OUT_CAP_MS = _env_ms("VF_OUT_MS", 1000)     # 发送队列上限，满则丢最旧
LATE_MS = _env_ms("VF_LATE_MS", 150)        # 块在队列里等待超过此值 → 旁路降级
CHUNK_BUDGET_MS = _env_ms("VF_BUDGET_MS", 32)  # 单块预算，超出记 overrun
CHUNK_SAMPLES = 4 * HOP                      # 一个推理块 = 32ms = 512 样本
CHUNK_FRAMES = CHUNK_SAMPLES // HOP          # 4 帧
INFER_THREADS = max(1, int(os.environ.get("VF_THREADS", 1)))  # ONNX intra_op 线程数

app = FastAPI(title="Voiceprint Focus · Streaming")
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=False)

STATE: dict = {"enroller": None, "load_ms": None, "ort": None,
               "campp": None, "campp_err": "",
               # 运行中可切的降噪开关：启动值仍取 VF_POST_DENOISE，之后由 /api/config 或 WS config 改
               "post_denoise": POST_DENOISE,
               # 活跃会话（弱引用，请求结束自动移除）：切换时逐个应用，不需要重建
               "active": weakref.WeakValueDictionary(),
               # 启动自检：拓扑与绑核结果（_load 时填充，health 直接读）
               "topo": {}, "affinity": {}}
_BG: set[asyncio.Task] = set()      # 旁路后台任务（防止被 GC 提前回收）


def _detect_topo() -> dict:
    """用 GetLogicalProcessorInformationEx(RelationProcessorCore) 读物理核/逻辑核/
    SMT/EfficiencyClass，判定大小核并给出 P/E 逻辑 CPU 列表——不写死核号。

    失败时退回 os.cpu_count() 且视为全 P 核（不绑核场景仍然可用）。
    """
    topo = {"logical": os.cpu_count() or 1, "physical": None, "smt": None,
            "hybrid": False, "p": [], "e": [], "source": "fallback"}
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.GetLogicalProcessorInformationEx
        fn.argtypes = [wintypes.DWORD, ctypes.c_void_p,
                       ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL
        length = wintypes.DWORD(0)
        fn(0, None, ctypes.byref(length))
        if not length.value:
            return topo
        buf = ctypes.create_string_buffer(length.value)
        if not fn(0, buf, ctypes.byref(length)):
            return topo
        raw = buf.raw[:length.value]
        cores, off = [], 0
        while off + 8 <= len(raw):
            rel, sz = struct.unpack_from("<II", raw, off)
            if sz <= 0 or off + sz > len(raw):
                break
            if rel == 0:                               # RelationProcessorCore
                d = off + 8
                eff = raw[d + 1]
                gc = struct.unpack_from("<H", raw, d + 22)[0]
                ids = []
                for i in range(gc):
                    base = d + 24 + i * 16
                    mask = struct.unpack_from("<Q", raw, base)[0]
                    grp = struct.unpack_from("<H", raw, base + 8)[0]
                    ids += [grp * 64 + b for b in range(64) if (mask >> b) & 1]
                cores.append((eff, ids))
            off += sz
        if cores:
            logical = sum(len(c[1]) for c in cores)
            maxeff = max(c[0] for c in cores)
            p = [i for eff, ids in cores if eff == maxeff for i in ids]
            e = [i for eff, ids in cores if eff != maxeff for i in ids]
            topo.update(logical=logical, physical=len(cores),
                        smt=logical > len(cores), hybrid=bool(e), p=p, e=e,
                        source="GetLogicalProcessorInformationEx")
    except Exception as ex:                            # noqa: BLE001
        log.warning("topo detect failed: %r", ex)
    if not topo["p"]:
        topo["p"] = list(range(topo["logical"]))
    return topo


def _init_affinity(sess) -> dict:
    """启动自检 + 可选绑核。

    - 总是探测拓扑；大小核机器上顺带测 P/E 各一颗的单块耗时（各 2 块，取较快值）；
    - 默认 off：测完立即恢复全核，行为与之前一致；
    - 绑核只在显式设置 VF_CPU_AFFINITY 时生效，且目标核由自检结果决定。
    """
    topo = STATE["topo"]
    info = {"mode": CPU_AFFINITY or "off", "applied": False, "cpu": None,
            "cpu_label": "", "bench_ms": {}, "note": ""}
    proc = psutil.Process()
    full = list(range(topo["logical"]))
    if topo["hybrid"] and topo["e"]:
        rng = np.random.default_rng(0)
        blk = (rng.standard_normal(CHUNK_SAMPLES) * 0.01).astype(np.float32)
        try:
            for name, ids in (("p", topo["p"]), ("e", topo["e"])):
                if not ids:
                    continue
                proc.cpu_affinity([ids[0]])
                sess.push(blk)                         # 丢弃首块（含分配）
                ts = []
                for _ in range(2):
                    t = time.perf_counter()
                    sess.push(blk)
                    ts.append((time.perf_counter() - t) * 1000.0)
                info["bench_ms"][name] = round(min(ts), 1)
        finally:
            proc.cpu_affinity(full)
    mode = (CPU_AFFINITY or "off").lower()
    if mode in ("", "off", "none", "0"):
        info["note"] = "未绑核（默认）：系统自由调度；多路并发、共享机器上吞吐更好"
        log.info("topo: logical=%s physical=%s smt=%s hybrid=%s P=%d E=%d bench=%s; affinity=off",
                 topo["logical"], topo["physical"], topo["smt"], topo["hybrid"],
                 len(topo["p"]), len(topo["e"]), info["bench_ms"])
        return info
    cand = topo["p"]
    loads = None
    if mode in ("auto", "p"):
        try:
            psutil.cpu_percent(percpu=True)
            time.sleep(0.3)
            loads = psutil.cpu_percent(percpu=True)
            cand = sorted(topo["p"], key=lambda c: loads[c] if c < len(loads) else 0)
        except Exception:                              # noqa: BLE001
            cand = topo["p"]
        cpu = cand[0]
    elif mode == "e":
        cpu = (topo["e"] or topo["p"])[0]
    elif mode.startswith("p") and mode[1:].isdigit():
        cpu = topo["p"][int(mode[1:]) % max(1, len(topo["p"]))]
    else:
        info["note"] = "无法识别的 VF_CPU_AFFINITY，按 off 处理"
        log.warning("unknown VF_CPU_AFFINITY=%r, treat as off", CPU_AFFINITY)
        return info
    proc.cpu_affinity([cpu])
    info.update(applied=True, cpu=cpu,
                cpu_label=("P" if cpu in topo["p"] else "E"),
                load_pct=(loads[cpu] if loads and cpu < len(loads) else None),
                note="已绑单核：降低迁移/争核抖动；单路实时更稳，但多路并发吞吐下降")
    log.info("topo: logical=%s physical=%s smt=%s hybrid=%s P=%d E=%d bench=%s; "
             "affinity=%s cpu=%s(%s) load=%s",
             topo["logical"], topo["physical"], topo["smt"], topo["hybrid"],
             len(topo["p"]), len(topo["e"]), info["bench_ms"], mode, cpu,
             info["cpu_label"], info["load_pct"])
    return info


@app.on_event("startup")
def _load():
    import onnxruntime as ort
    t0 = time.perf_counter()
    STATE["enroller"] = Enroller(os.path.join(MODEL_DIR, "enroll.onnx"))
    with open(os.path.join(MODEL_DIR, "export_report.json"), encoding="utf-8") as f:
        STATE["export"] = json.load(f)
    # 预热：跑一次 step，避免首块包含 JIT/内存分配开销
    warm = StreamSession(os.path.join(MODEL_DIR, "step.onnx"),
                         np.zeros((1, 64, 192), np.float32), np.ones((1, 32), np.float32),
                         capture=False)
    # 拓扑自检 + 可选绑核（必须在 flush 之前做：flush 会关闭会话，push 将不再推理）
    STATE["topo"] = _detect_topo()
    STATE["affinity"] = _init_affinity(warm)
    warm.push(np.zeros(16000, np.float32))
    warm.flush()
    STATE["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    STATE["ort"] = ort.__version__
    # CAM++ 旁路观察：只加载、只打日志；失败则禁用，不影响注册与开流
    try:
        ob = CamppObserver()
        ob.embed(np.zeros(16000, np.float32))       # 预热，避免首条注册含初始化开销
        STATE["campp"] = ob
    except Exception as e:                          # noqa: BLE001
        STATE["campp"] = None
        STATE["campp_err"] = repr(e)
        log.warning("campp bypass disabled: %r", e)


@app.get("/api/health")
def health():
    return {"ok": True, "ort": STATE.get("ort"), "load_ms": STATE.get("load_ms"),
            "model": "x16_tfmap_causal_bsrnn", "chunk_frames": 4, "chunk_ms": 32,
            "sr": SR, "infer_threads": INFER_THREADS,
            "onnx_intra_threads": INFER_THREADS, "onnx_inter_threads": 1,
            "post_denoise": STATE["post_denoise"],
            "post_denoise_env_default": POST_DENOISE,
            "post_denoise_floor_db": PD_FLOOR_DB,
            "post_denoise_ramp_blocks": 4,
            "post_denoise_switchable": True,
            "cpu_affinity": {**STATE.get("affinity", {}),
                             "topo": STATE.get("topo", {})},
            "hf_shelf_db": HF_SHELF_DB,
            "pause_step": {"enabled": PAUSE_STEP, "extra_db": PAUSE_DB,
                           "attack_ms": PAUSE_ATT_MS, "release_ms": PAUSE_REL_MS,
                           "vad": "100-2000Hz, hop 5ms, thr=max(floor+9dB,-48dBFS), floor=p10 of 3s"},

            "enroll_qc": {"cos_min": ENROLL_ID_COS_MIN,
                          "pitch_shift_max_st": ENROLL_PITCH_SHIFT_MAX_ST},
            "campp_bypass": {"enabled": STATE.get("campp") is not None,
                             "mode": "observe_only", "enforced": False,
                             "threshold_ref": CAMPP_THRESHOLD_REF,
                             "model": CAMPP_MODEL_NAME,
                             "diag_dir": CAMPP_DIAG_DIR,
                             "error": STATE.get("campp_err", "")},
            "capture": {"enabled": CAPTURE, "dir": CAPTURE_DIR, "max_s": CAPTURE_MAX_S},
            "enroll_save": {"enabled": ENROLL_SAVE, "dir": ENROLL_DIR,
                            "format": "wav pcm_s16le mono 16k",
                            "naming": "enroll_<YYYYmmdd-HHMMSS>_<session>_<seq>.wav (+ .json)"},
            "file_api": {"enroll": "POST /api/enroll/file",
                         "list": "GET /api/enroll/list",
                         "process": "POST /api/process/file (wav|f32, streaming)",
                         "ws_start_with": '{"type":"start","enroll_id":"reg_..."}',
                         "registrations_dir": REG_DIR,
                         "max_enroll_s": MAX_ENROLL_FILE_S,
                         "max_process_s": MAX_PROCESS_S},
            "bypass": {"enter_ms": BYPASS_ENTER_MS, "exit_ms": BYPASS_EXIT_MS,
                       "dwell_ms": BYPASS_DWELL_MS, "xfade_ms": XFADE_MS,
                       "level_match": BYPASS_LEVEL_MATCH,
                       "pause_only": BYPASS_PAUSE_ONLY, "force_ms": BYPASS_FORCE_MS},
            "skip_silent": {"enabled": SKIP_SILENT, "abs_db": SKIP_ABS_DB,
                            "hold_ms": SKIP_HOLD_MS,
                            "warm_blocks": SKIP_WARM_BLOCKS, "warm_after": SKIP_WARM_AFTER},
            "export": STATE.get("export")}


@app.get("/api/config")
def api_get_config():
    return {"post_denoise": STATE["post_denoise"], "floor_db": PD_FLOOR_DB,
            "ramp_blocks": 4, "active_sessions": len(STATE["active"])}


@app.post("/api/config")
def api_set_config(req: dict):
    """运行中切换降噪：{"post_denoise": true|false}。立即生效并作用于所有活跃会话，
    不需要重建会话；当前块起 4 块（128ms）线性过渡。"""
    if "post_denoise" not in req:
        return JSONResponse({"error": "缺少 post_denoise"}, status_code=400)
    v = req["post_denoise"]
    on = v if isinstance(v, bool) else str(v).strip().lower() in (
        "1", "true", "on", "yes")
    return {"ok": True, **_set_post_denoise(on)}


@app.get("/api/demo/list")
def demo_list():
    p = os.path.join(DEMO_DIR, "manifest.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    return [dict(it, mix=f"/api/demo/{it['id']}/mix", target=f"/api/demo/{it['id']}/target",
                 enroll=f"/api/demo/{it['id']}/enroll") for it in items]


@app.get("/api/demo/{item_id}/{kind}")
def demo_wav(item_id: str, kind: str):
    if kind not in ("mix", "target", "enroll") or not re.fullmatch(r"[A-Za-z0-9_]+", item_id):
        return JSONResponse({"error": "bad request"}, status_code=400)
    p = os.path.join(DEMO_DIR, kind, f"{item_id}.wav")
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/enroll/file")
async def enroll_file(file: UploadFile = File(...), name: str = Form("")):
    """文件式注册：上传音频 → 与录制注册完全相同的质检链
    （vad_report → enroll_qc → encode）→ 持久化声纹，返回 enroll_id。"""
    enroller: Enroller = STATE["enroller"]
    if int(file.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
        return JSONResponse({"ok": False, "error": "文件过大"}, status_code=413)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"ok": False,
                             "error": f"文件过大（上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB）"},
                            status_code=413)
    if not raw:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    try:
        wav = _decode_audio(raw, file.filename or "")
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    dur = len(wav) / SR
    if dur > MAX_ENROLL_FILE_S:
        return JSONResponse({"ok": False,
                             "error": f"注册音频过长（{dur:.1f}s，上限 {MAX_ENROLL_FILE_S:.1f}s）"},
                            status_code=422)
    saved = _save_enroll(wav, {
        "session_id": "file-" + os.urandom(3).hex(), "seq": 1, "source": "file",
        "name": name[:64], "filename": (file.filename or "")[:128],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ts_compact": time.strftime("%Y%m%d-%H%M%S"),
    })
    saved_name = os.path.basename(saved["wav"]) if saved else ""
    campp_ctx = {"run_id": "enrollfile-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"),
                                                 os.urandom(3).hex()),
                 "split": split_enroll_halves(wav)}
    q = vad_report(wav)
    if not q["ok"]:
        _tag_enroll(saved, outcome="vad_reject", vad=q)
        _fire_campp(wav, campp_ctx, outcome="vad_reject",
                    vad_ok=False, vad_reason=q["reason"])
        return JSONResponse({"ok": False, "code": "vad",
                             "message": f"注册音频质检未通过：{q['reason']}",
                             "vad": q, "enroll_saved": saved_name}, status_code=422)
    qc = await asyncio.to_thread(enroll_qc, wav, enroller)
    if not qc["ok"]:
        _tag_enroll(saved, outcome="qc_reject", vad=q, enroll_qc=qc)
        _fire_campp(wav, campp_ctx, outcome="qc_reject",
                    vad_ok=True, enroll_qc_ok=False, enroll_qc_code=qc["code"])
        return JSONResponse({"ok": False, "code": qc["code"], "message": qc["hint"],
                             "hint": qc["hint"], "vad": q, "enroll_qc": qc,
                             "enroll_saved": saved_name}, status_code=422)
    E, g, enc_ms = await asyncio.to_thread(enroller.encode, wav)
    rid = _save_registration(E, g, {
        "name": name[:64], "source": "file", "filename": (file.filename or "")[:128],
        "duration_s": round(dur, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "vad": q, "enroll_qc": qc, "archive": saved_name})
    _tag_enroll(saved, outcome="registered", reg_id=rid, vad=q, enroll_qc=qc)
    _fire_campp(wav, campp_ctx, outcome="registered",
                vad_ok=True, enroll_qc_ok=True, enroll_qc_code="")
    return {"ok": True, "enroll_id": rid, "duration_s": round(dur, 2),
            "encode_ms": round(enc_ms, 1), "vad": q, "enroll_qc": qc,
            "enroll_saved": saved_name}


@app.get("/api/enroll/list")
def enroll_list():
    return _list_registrations()


def _process_stream(x: np.ndarray, E: np.ndarray, g: np.ndarray, fmt: str):
    """把整段音频按 32ms 块喂 StreamSession，边算边产出（同步生成器，
    StreamingResponse 会放到线程池迭代，不阻塞事件循环）。

    与 WS worker 相同的三态路由（skip/bypass/model），但 waited_ms 恒为 0：
    文件处理没有实时排队，旁路只在停顿段生效（pause_only），不会因排队超时强切。
    输出总长严格等于输入长度（尾部混响余音截断，缺样补零）。"""
    sess = _new_session(E, g, capture=False)
    total = int(len(x))
    state = {"produced": 0}

    def enc(arr):
        if state["produced"] >= total:
            return None
        arr = np.asarray(arr, np.float32)
        if state["produced"] + len(arr) > total:
            arr = arr[:total - state["produced"]]
        state["produced"] += len(arr)
        if fmt == "wav":
            return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return arr.astype(np.float32).tobytes()

    def run_block(block):
        sess.feed_pause(block)
        if sess.cur_block_skip:
            return sess.bypass(block, True, True)
        if sess.should_bypass(0.0, pause=sess.cur_block_pause):
            return sess.bypass(block, True)
        return sess.push(block, True)

    if fmt == "wav":
        yield _wav_header(total)
    t0 = time.perf_counter()
    n = 0
    while n + CHUNK_SAMPLES <= total:
        for arr in run_block(x[n:n + CHUNK_SAMPLES]):
            b = enc(arr)
            if b:
                yield b
        n += CHUNK_SAMPLES
    if n < total:                       # 末尾不足一块也要过模型，避免丢音
        for arr in run_block(x[n:total]):
            b = enc(arr)
            if b:
                yield b
    for arr in sess.flush():
        b = enc(arr)
        if b:
            yield b
    if state["produced"] < total:       # 理论上不会发生，兜底补零对齐头部
        pad = total - state["produced"]
        state["produced"] = total
        yield (np.zeros(pad, np.int16).tobytes() if fmt == "wav"
               else np.zeros(pad, np.float32).tobytes())
    sm = sess.summary()
    log.info("process file done: audio_s=%.1f compute_s=%.1f rtf=%.3f elapsed_s=%.1f",
             sm.get("audio_s", 0.0), sm.get("compute_total_ms", 0.0) / 1000.0,
             sm.get("rtf", 0.0), time.perf_counter() - t0)


@app.post("/api/process/file")
async def process_file(file: UploadFile = File(...), enroll_id: str = Form(""),
                       out_format: str = Form("wav")):
    """较长音频文件流式处理：分块推理、分块下发。
    out_format=wav（默认，带 44 字节头可直接播放）或 f32（裸 float32 LE）。"""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": f"文件过大（上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB）"},
                            status_code=413)
    if not raw:
        return JSONResponse({"error": "空文件"}, status_code=400)
    try:
        x = _decode_audio(raw, file.filename or "")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    dur = len(x) / SR
    if dur > MAX_PROCESS_S:
        return JSONResponse({"error": f"文件过长（{dur:.1f}s，上限 {MAX_PROCESS_S:.0f}s）"},
                            status_code=422)
    if x.size < CHUNK_SAMPLES:
        return JSONResponse({"error": "音频太短（不足一个 32ms 块）"}, status_code=422)
    fmt = "f32" if out_format.strip().lower() == "f32" else "wav"
    if enroll_id:
        reg = _load_registration(enroll_id)
        if reg is None:
            return JSONResponse({"error": f"注册不存在: {enroll_id}"}, status_code=404)
        E, g, meta = reg
    else:
        try:
            E, g = _demo_reg()
        except DemoAssetMissing as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        meta = {"enroll_id": "", "source": "demo"}
    x = np.ascontiguousarray(x, np.float32)
    fname = re.sub(r"[^\w.\-]+", "_", os.path.basename(file.filename or "audio")) + ""
    stem = os.path.splitext(fname)[0] or "audio"
    return StreamingResponse(
        _process_stream(x, E, g, fmt),
        media_type="audio/wav" if fmt == "wav" else "application/octet-stream",
        headers={"Content-Disposition":
                 'attachment; filename="processed_%s.%s"' % (stem, fmt)},
        background=None)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    enroller: Enroller = STATE["enroller"]
    enroll_buf: list[np.ndarray] = []
    # 注册留档：每次连接一个会话标识；source 区分「录制」与「上传文件」；seq 防同名覆盖
    sess_id = "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), os.urandom(3).hex())
    enroll_source = "mic"
    enroll_seq = 0
    session: StreamSession | None = None
    mode = "idle"
    inbound: asyncio.Queue = asyncio.Queue()
    queued = {"samples": 0}
    carry = np.zeros(0, np.float32)   # 收包侧聚合缓冲：按「凑满整块帧」再跨线程
    carry_t = None
    received = 0                       # 已进入 carry 的累计样本（丢包会同步扣减）
    dispatched_frames = 0              # 已派发给推理线程的帧数
    in_cap_samples = max(CHUNK_SAMPLES, int(round(IN_CAP_MS * SR / 1000.0)))
    outq: deque = deque()
    out_cap = max(1, int(round(OUT_CAP_MS / 32)))
    drops = {"in_ms": 0.0, "out_ms": 0.0, "skipped_ms": 0.0, "overruns": 0}
    waits: list[float] = []           # RTF 插桩：每块排队等待（ms），仅 VF_PROFILE_OUT 时落盘
    worker_task: asyncio.Task | None = None
    wake = asyncio.Event()
    model_info = {"name": "实验16 · B2 帧级 TF-Map 条件化因果 BSRNN",
                  "chunk_frames": 4, "chunk_ms": 32, "sr": SR,
                  "post_denoise_floor_db": PD_FLOOR_DB,
                  "export_pass": bool(STATE.get("export", {}).get("pass"))}
    flow_info = {"in_cap_ms": IN_CAP_MS, "out_cap_ms": OUT_CAP_MS,
                 "late_ms": LATE_MS, "late_exit_ms": BYPASS_EXIT_MS,
                 "bypass_dwell_ms": BYPASS_DWELL_MS, "xfade_ms": XFADE_MS,
                 "bypass_level_match": BYPASS_LEVEL_MATCH,
                 "budget_ms": CHUNK_BUDGET_MS}

    def drop_oldest_in():
        try:
            item = inbound.get_nowait()
        except asyncio.QueueEmpty:
            return
        if item is not None:
            queued["samples"] -= len(item[0])
            drops["in_ms"] += len(item[0]) * 1000.0 / SR

    def frames_ready(samples: int) -> int:
        """与 OnlineAnalyzer 一致：左端补 PAD 后，每 HOP 个新样本多一帧。"""
        if samples < PAD + 1:
            return 0
        return (samples + PAD - WIN) // HOP + 1

    def enqueue_in(x, t_arr):
        # 背压：输入缓冲超过上限时丢最旧（时间上更早）的音频，而不是丢最新
        nonlocal carry, carry_t, received, dispatched_frames
        if len(x) > in_cap_samples:
            x = x[-in_cap_samples:]
        if carry.size == 0:
            carry_t = t_arr
        # 超限时优先丢 carry 里最旧的样本，再丢 inbound 队首
        while carry.size + queued["samples"] + len(x) > in_cap_samples:
            if carry.size:
                drop = int(min(carry.size, carry.size + queued["samples"] + len(x) - in_cap_samples))
                drops["in_ms"] += drop * 1000.0 / SR
                carry = carry[drop:]
                received -= drop
                if carry.size == 0:
                    carry_t = t_arr
            elif not inbound.empty():
                drop_oldest_in()
            else:
                break
        carry = np.concatenate([carry, x])
        received += len(x)
        # 凑满整块（4 帧）再跨线程：跨线程任务从 125/s 降到 31/s，避免 GIL 争用拖慢推理。
        # 用帧数而非固定 512 样本切块，保证首块也在「第 4 帧可用」的同一时刻派发，
        # 算法延迟维持 24ms（若按 512 对齐会在首块多等 384 样本 → 48ms）。
        ready = frames_ready(received)
        if ready - dispatched_frames >= CHUNK_FRAMES:
            block, carry = carry, np.zeros(0, np.float32)
            dispatched_frames = ready - (ready - dispatched_frames) % CHUNK_FRAMES
            queued["samples"] += len(block)
            inbound.put_nowait((block, carry_t))
            carry_t = time.perf_counter()

    def enqueue_out(item):
        if len(outq) >= out_cap:
            # 发送侧拥塞：丢最旧的音频块，控制消息（ready/done/error）不丢
            for i, it in enumerate(outq):
                if it.get("__audio"):
                    drops["out_ms"] += it["out_len"] * 1000.0 / SR
                    del outq[i]
                    break
        outq.append(item)
        wake.set()

    def build_stat(st):
        return {**stats_msg(st),
                "queue_in_ms": round((queued["samples"] + carry.size) * 1000.0 / SR, 1),
                "queue_out_ms": round(len(outq) * 32.0, 1),
                "dropped_in_ms": round(drops["in_ms"], 1),
                "dropped_out_ms": round(drops["out_ms"], 1),
                "skipped_ms": round(drops["skipped_ms"], 1),
                "overrun": st.compute_ms > CHUNK_BUDGET_MS}

    def emit(outs, sess):
        for out, st in zip(outs, sess.stats[-len(outs):] if outs else []):
            if st.skipped:
                drops["skipped_ms"] += st.out_len * 1000.0 / SR
            elif st.compute_ms > CHUNK_BUDGET_MS:
                drops["overruns"] += 1
            enqueue_out({"__audio": True, "msg": build_stat(st),
                         "bytes": out.astype(np.float32).tobytes(), "out_len": len(out)})

    # 所有下行文本共用一把锁：sender 的「stat 文本 + 音频二进制」必须成对原子发送，
    # 否则 pong/config/error 插进两者中间会让客户端把文本与音频错配；ASGI 也不保证
    # 同一连接上的并发 send 安全。
    send_lock = asyncio.Lock()

    async def send_json(obj):
        async with send_lock:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def sender():
        """唯一的发送者：与推理完全解耦，客户端读得慢只会导致丢帧，不会反压计算。"""
        while True:
            if not outq:
                wake.clear()
                if outq:
                    continue
                await wake.wait()
                continue
            item = outq.popleft()
            try:
                if item.get("__audio"):
                    async with send_lock:
                        await ws.send_text(json.dumps(item["msg"], ensure_ascii=False))
                        await ws.send_bytes(item["bytes"])
                else:
                    async with send_lock:
                        await ws.send_text(json.dumps(item, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                return

    async def cancel_worker():
        """重复 start / 重新注册前先停掉旧消费者，避免两个 worker 争抢同一 inbound。"""
        nonlocal worker_task
        if worker_task is not None and not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
        worker_task = None

    async def worker(sess: StreamSession):
        """推理消费者：只在独立线程里跑模型，事件循环始终能继续收包。"""
        while True:
            item = await inbound.get()
            if item is None:  # stop：冲洗尾帧后结束
                outs = await asyncio.to_thread(sess.flush)
                emit(outs, sess)
                summary = sess.summary()
                summary["drops"] = {k: (round(v, 1) if isinstance(v, float) else v)
                                    for k, v in drops.items()}
                enqueue_out({"type": "done", "summary": summary})
                prof_out = os.environ.get("VF_PROFILE_OUT")
                if prof_out:
                    try:
                        w = np.asarray(waits, np.float64)
                        rec = {
                            "ts": round(time.time(), 3),
                            "summary": summary,
                            "blocks": sess.prof or [],
                            "push_ms": sess.prof_push or [],
                            "drops": summary.get("drops"),
                            "waits_ms": {
                                "mean": round(float(w.mean()), 2) if w.size else 0.0,
                                "p50": round(float(np.percentile(w, 50)), 2) if w.size else 0.0,
                                "p95": round(float(np.percentile(w, 95)), 2) if w.size else 0.0,
                                "max": round(float(w.max()), 2) if w.size else 0.0,
                            },
                        }
                        with open(prof_out, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except OSError:
                        pass
                return
            x, t_arr = item
            queued["samples"] -= len(x)
            waited_ms = (time.perf_counter() - t_arr) * 1000.0
            waits.append(waited_ms)
            # 旁路只在停顿段生效（+ 安全阀）：先喂输入侧停顿时钟拿到本块判决，
            # 再决定走模型还是旁路；push/bypass 传 pause_fed=True 避免重复喂。
            sess.feed_pause(x)
            if sess.cur_block_skip:                  # 纯静音跳空：不送模型，−18dB 直通
                outs = await asyncio.to_thread(sess.bypass, x, True, True)
            elif sess.should_bypass(waited_ms, pause=sess.cur_block_pause):
                outs = await asyncio.to_thread(sess.bypass, x, True)
            else:
                outs = await asyncio.to_thread(sess.push, x, True)
            emit(outs, sess)

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                x = np.frombuffer(msg["bytes"], dtype=np.float32).copy()
                if mode == "enroll":
                    enroll_buf.append(x)
                    total = sum(len(a) for a in enroll_buf) / SR
                    if total > MAX_ENROLL_S:
                        mode = "idle"
                        await send_json({"type": "error",
                                         "message": "注册音频超长，已重置"})
                elif mode == "stream" and session is not None:
                    enqueue_in(x, time.perf_counter())
                continue

            raw = msg.get("text") or "{}"
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "非法 JSON"})
                continue
            typ = req.get("type")

            if typ == "enroll":
                enroll_buf = []
                enroll_source = str(req.get("source") or "mic")[:16]
                await cancel_worker()
                session = None
                mode = "enroll"
                await send_json({"type": "enroll_started",
                                 "max_s": MAX_ENROLL_S,
                                 "source": enroll_source})
            elif typ == "start":
                eid = str(req.get("enroll_id") or "").strip()
                if eid:
                    # 文件式注册开流：跳过录制缓冲，直接用已持久化的声纹
                    reg = _load_registration(eid)
                    if reg is None:
                        await send_json(
                            {"type": "error", "message": f"注册不存在: {eid}"})
                        continue
                    Er, gr, meta = reg
                    await cancel_worker()
                    session = _new_session(Er, gr)
                    while not inbound.empty():
                        drop_oldest_in()
                    queued["samples"] = 0
                    carry = np.zeros(0, np.float32)
                    carry_t = None
                    received = 0
                    dispatched_frames = 0
                    mode = "stream"
                    model_info["post_denoise"] = bool(session.post_denoise)
                    enqueue_out({"type": "ready", "model": model_info,
                                 "enroll_source": "file", "enroll_id": eid,
                                 "enroll_s": meta.get("duration_s"),
                                 "flow": flow_info})
                    worker_task = asyncio.create_task(worker(session))
                    continue
                if mode != "enroll" or not enroll_buf:
                    await send_json({"type": "error",
                                     "message": "请先完成声纹注册"})
                    continue
                wav = np.concatenate(enroll_buf)
                enroll_seq += 1
                saved = _save_enroll(wav, {
                    "session_id": sess_id, "seq": enroll_seq, "source": enroll_source,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "ts_compact": time.strftime("%Y%m%d-%H%M%S"),
                    "model": model_info["name"],
                })
                saved_name = os.path.basename(saved["wav"]) if saved else ""
                campp_ctx = {"run_id": "enroll-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"),
                                                         os.urandom(3).hex()),
                             "split": split_enroll_halves(wav)}
                q = vad_report(wav)
                if not q["ok"]:
                    _tag_enroll(saved, outcome="vad_reject", vad=q)
                    _fire_campp(wav, campp_ctx, outcome="vad_reject",
                                vad_ok=False, vad_reason=q["reason"])
                    mode = "idle"
                    await send_json({"type": "error",
                                     "message": f"注册音频质检未通过：{q['reason']}",
                                     "code": "vad", "vad": q,
                                     "enroll_saved": saved_name})
                    continue
                qc = await asyncio.to_thread(enroll_qc, wav, enroller)
                if not qc["ok"]:
                    _tag_enroll(saved, outcome="qc_reject", vad=q, enroll_qc=qc)
                    _fire_campp(wav, campp_ctx, outcome="qc_reject",
                                vad_ok=True, enroll_qc_ok=False,
                                enroll_qc_code=qc["code"])
                    mode = "idle"
                    await send_json({"type": "error",
                                     "message": qc["hint"], "code": qc["code"],
                                     "hint": qc["hint"], "vad": q,
                                     "enroll_qc": qc,
                                     "enroll_saved": saved_name})
                    continue
                _tag_enroll(saved, outcome="ready", vad=q, enroll_qc=qc)
                _fire_campp(wav, campp_ctx, outcome="ready",
                            vad_ok=True, enroll_qc_ok=True, enroll_qc_code="")
                E, g, enc_ms = await asyncio.to_thread(enroller.encode, wav)
                await cancel_worker()
                session = _new_session(E, g)
                while not inbound.empty():
                    drop_oldest_in()
                queued["samples"] = 0
                carry = np.zeros(0, np.float32)
                carry_t = None
                received = 0
                dispatched_frames = 0
                mode = "stream"
                model_info["post_denoise"] = bool(session.post_denoise)
                enqueue_out({"type": "ready", "vad": q, "enroll_qc": qc, "model": model_info,
                             "enroll_ms": round(enc_ms, 1),
                             "enroll_s": round(len(wav) / SR, 2),
                             "enroll_source": enroll_source,
                             "enroll_saved": saved_name,
                             "flow": flow_info})
                worker_task = asyncio.create_task(worker(session))
            elif typ == "stop":
                if session is None or mode != "stream":
                    await send_json({"type": "error",
                                     "message": "当前没有进行中的流"})
                    continue
                mode = "stopping"
                if carry.size:  # 末尾不足一块的样本也要交给推理，避免丢音
                    inbound.put_nowait((carry.copy(), carry_t or time.perf_counter()))
                    carry = np.zeros(0, np.float32)
                    carry_t = None
                inbound.put_nowait(None)
            elif typ == "ping":
                await send_json({"type": "pong", "t": req.get("t")})
            elif typ == "config":
                # 允许在开流前后随时切换：改全局值并应用到所有活跃会话
                if "post_denoise" not in req:
                    await send_json({"type": "error",
                                     "message": "缺少 post_denoise"})
                    continue
                v = req["post_denoise"]
                on = v if isinstance(v, bool) else str(v).strip().lower() in (
                    "1", "true", "on", "yes")
                r = _set_post_denoise(on)
                await send_json({"type": "config", **r})
            else:
                await send_json({"type": "error",
                                 "message": f"未知指令 {typ}"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await send_json({"type": "error", "message": f"服务异常: {e}"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        sender_task.cancel()
        if worker_task is not None and not worker_task.done():
            worker_task.cancel()


if os.path.isdir(FRONT_DIR):
    app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")


def main():
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
