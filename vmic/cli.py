"""vmic 命令行：devices / profiles / enroll / run。

约定
- 只读 backend 的 engine/dsp，不 import server，不碰 :8000 进程。
- 声纹档案与 Web 端共用 <app>/backend/enrollments/registrations。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import gcd

import numpy as np
import sounddevice as sd

import engine
from vmic import profiles
from vmic.audio import (AudioIn, AudioOut, DeviceError, describe, devices,
                        looks_virtual, resolve, SR_MODEL)
from vmic.pipeline import FocusPipeline

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_APP, "models", "x16_stream")


def _model_paths(model_dir: str) -> tuple[str, str]:
    step = os.path.join(model_dir, "step.onnx")
    enroll = os.path.join(model_dir, "enroll.onnx")
    for p in (step, enroll):
        if not os.path.exists(p):
            raise DeviceError(f"缺少模型文件：{p}\n"
                              "开源版不随仓库分发权重，请把 step.onnx / enroll.onnx "
                              "放到该目录，或用 --model-dir 指定。")
    return step, enroll


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(x, np.float32)
    g = gcd(int(sr_from), int(sr_to))
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(x, np.float32), sr_to // g, sr_from // g).astype(np.float32)


def _db(x: np.ndarray) -> float:
    if not len(x):
        return -120.0
    r = float(np.sqrt(np.mean(np.square(np.asarray(x, np.float32)), dtype=np.float64)))
    return round(20.0 * np.log10(r + 1e-12), 1)


def _fmt_ms(v) -> str:
    return "—" if v is None else f"{v:.0f}ms"


# ----------------------------- devices -----------------------------
def cmd_devices(a) -> int:
    print("输入端点（麦克风）：")
    for d in devices("in"):
        mark = " 默认" if d["index"] == sd.default.device[0] else ""
        print(f"  {describe(d)}{mark}")
    print("\n输出端点（虚拟声卡选这里）：")
    for d in devices("out"):
        mark = " 默认" if d["index"] == sd.default.device[1] else ""
        print(f"  {describe(d)}{mark}")
    print("\n虚拟麦克风用法：把本程序输出写到虚拟声卡的 Input（如 VB-CABLE 的 "
          "「CABLE Input」），\n然后在 Discord / Zoom / OBS 里把麦克风选成它的 Output"
          "（「CABLE Output」）。")
    return 0


# ----------------------------- profiles -----------------------------
def cmd_profiles(a) -> int:
    items = profiles.list_all()
    if not items:
        print(f"还没有声纹档案。目录：{profiles.reg_dir()}\n"
              "先运行：python -m vmic enroll")
        return 0
    print(f"声纹档案（{profiles.reg_dir()}）：")
    for m in items:
        print(f"  {m.get('id')}  {m.get('duration_s', '?')}s  "
              f"source={m.get('source', '?')}  name={m.get('name', '')}")
    return 0


# ----------------------------- enroll -----------------------------
def _enroll_from_mic(dev: dict, seconds: float) -> tuple[np.ndarray, int]:
    sr = int(dev["sr"])
    block = max(1, int(sr * 0.1))
    ch = max(1, min(2, int(dev["in"])))
    buf = []
    with sd.InputStream(device=dev["index"], samplerate=sr, channels=ch,
                        dtype="float32", blocksize=block, latency="low") as st:
        t0 = time.perf_counter()
        print(f"请开始说话（目标 {seconds:.0f}s，正常音量、同一人、不要有明显噪声）")
        while True:
            data, _ = st.read(block)
            x = data[:, 0] if ch == 1 else data.mean(axis=1)
            buf.append(np.array(x, np.float32, copy=True))
            el = time.perf_counter() - t0
            if int(el * 10) % 5 == 0:
                print(f"\r  已录 {el:4.1f}s / {seconds:.0f}s   电平 {_db(x):6.1f} dBFS",
                      end="", flush=True)
            if el >= seconds:
                break
    print()
    return np.concatenate(buf) if buf else np.zeros(0, np.float32), sr


def cmd_enroll(a) -> int:
    step, enroll_path = _model_paths(a.model_dir)
    enroller = engine.Enroller(enroll_path)
    if a.from_file:
        import soundfile as sf
        x, sr = sf.read(a.from_file, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        x = _resample(x, sr, SR_MODEL)
        source = "vmic-file"
        print(f"读取 {a.from_file}: {sr}Hz {len(x) / SR_MODEL:.2f}s")
    else:
        dev = resolve(a.device, "in")
        if not 3.0 <= a.seconds <= 6.0:
            print("注册时长需在 3–6 秒（质检要求 2.5–6.5 秒）", file=sys.stderr)
            return 2
        print(f"采集端点：{describe(dev)}")
        raw, sr = _enroll_from_mic(dev, a.seconds)
        x = _resample(raw, sr, SR_MODEL)
        source = "vmic-mic"

    if len(x) < int(0.5 * SR_MODEL):
        print("录音太短，未采集到有效音频", file=sys.stderr)
        return 2
    vad = engine.vad_report(x)
    if not vad["ok"]:
        print(f"注册质检未通过：{vad['reason']}", file=sys.stderr)
        return 2
    qc = engine.enroll_qc(x, enroller)
    if not qc["ok"]:
        print(f"注册质检未通过：{qc['hint']}", file=sys.stderr)
        print(f"  声纹一致性 cos={qc['identity']['cos']} "
              f"(下限 {qc['identity']['cos_min']})，"
              f"音高漂移 {qc['pitch']['shift_st']} 半音", file=sys.stderr)
        return 2
    E, g, enc_ms = enroller.encode(x)
    rid = profiles.save(E, g, {
        "name": (a.name or "")[:64], "source": source,
        "duration_s": round(len(x) / SR_MODEL, 2),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "vad": vad, "enroll_qc": qc,
    })
    print(f"注册成功：{rid}  （{len(x) / SR_MODEL:.2f}s，编码 {enc_ms:.0f}ms）")
    print(f"开始使用：python -m vmic run --profile {rid} --output \"CABLE Input\"")
    return 0


# ----------------------------- run -----------------------------
def _pick_profile(spec: str | None):
    if spec:
        got = profiles.load(spec)
        if got is None:
            raise DeviceError(f"找不到声纹档案：{spec}（python -m vmic profiles 查看）")
        return got
    items = profiles.list_all()
    if len(items) == 1:
        got = profiles.load(items[0]["id"])
        if got is not None:
            print(f"未指定 --profile，使用唯一档案 {items[0]['id']}")
            return got
    if not items:
        raise DeviceError("还没有声纹档案，先运行 python -m vmic enroll")
    ids = "、".join(m.get("id", "?") for m in items[:8])
    raise DeviceError(f"有多个档案，请用 --profile 指定：{ids}")


def cmd_run(a) -> int:
    step, _ = _model_paths(a.model_dir)
    E, g, meta = _pick_profile(a.profile)
    dev_in = resolve(a.input, "in")
    dev_out = resolve(a.output, "out")
    sr_in, sr_out = int(dev_in["sr"]), int(dev_out["sr"])

    pipe = FocusPipeline(step, E, g, sr_in=sr_in, sr_out=sr_out, threads=a.threads,
                         post_denoise=(a.post_denoise == "on"), buffer_ms=a.buffer)
    print(f"声纹档案：{meta.get('id', a.profile or '(唯一档案)')}  "
          f"{meta.get('duration_s', '?')}s")
    print(f"采集：{describe(dev_in)}")
    print(f"输出：{describe(dev_out)}")
    if not looks_virtual(dev_out["name"]):
        print("  注意：输出端点看起来不是虚拟声卡，声音会从本机扬声器放出来。")

    t0 = time.perf_counter()
    print("加载模型并预热 …", end="", flush=True)
    pipe.warmup()
    print(f" 完成（{time.perf_counter() - t0:.1f}s）")

    def toggle():
        new = not pipe.denoise
        pipe.set_denoise(new)
        print(f"[{time.strftime('%H:%M:%S')}] 后置降噪 → {'开' if new else '关'}"
              f"（{'-18dB 谱减' if new else '直通，适合音乐/已降噪场景'}）", flush=True)

    ctl, hint = (None, "未启用热键")
    if not a.no_hotkey:
        from vmic.control import start_hotkey
        ctl, hint = start_hotkey(toggle, lambda: None, a.hotkey)
    print(hint)
    print(f"降噪当前：{'开' if pipe.denoise else '关'}  "
          f"（启动参数 --post-denoise {a.post_denoise}）")
    print("开始处理，Ctrl+C 结束" + (f"，最长 {a.seconds:.0f}s" if a.seconds else ""))

    out = AudioOut(dev_out, sr_out, pipe.pull)
    inp = AudioIn(dev_in, sr_in, pipe.feed)
    pipe.start()
    out.start()
    inp.start()
    t_start = time.perf_counter()
    last_report = 0.0
    try:
        while True:
            time.sleep(0.2)
            el = time.perf_counter() - t_start
            if a.seconds and el >= a.seconds:
                break
            if el - last_report >= 5.0:
                last_report = el
                s = pipe.status()
                print(f"  {el:6.0f}s  块{s['chunks']} 模型{s['model']}/旁路{s['bypass']}/"
                      f"跳空{s['skip']}  计算p50 {_fmt_ms(s['compute_p50_ms'])} "
                      f"p95 {_fmt_ms(s['compute_p95_ms'])}  出队 {s['out_queue_ms']}ms  "
                      f"欠载 {s['underruns']}  丢{s['in_drops'] + s['out_drops']}",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        inp.stop()
        out.stop()
        pipe.stop()
        if ctl is not None:
            ctl.stop()

    s = pipe.summary()
    print("\n结束摘要：")
    print(f"  处理 {s.get('audio_s', 0)}s 音频，块 {s['chunks']}，"
          f"模型 {s['model']} / 旁路 {s['bypass']} / 跳空 {s['skip']}")
    print(f"  单块计算 均值 {s.get('compute_mean_ms')}ms  p95 {s.get('compute_p95_ms')}ms  "
          f"RTF {s.get('rtf')}  算法延迟 {_fmt_ms(s.get('algo_latency_ms'))}")
    print(f"  欠载 {s['underruns']}（启动 {s['startup_underruns']}），"
          f"输入丢块 {s['in_drops']}，输出丢块 {s['out_drops']}")
    if s.get("underruns", 0) > max(5, s["chunks"] * 0.01):
        print("  欠载偏多：把 --buffer 调大（如 40）或关闭其他占 CPU 的程序。")
    return 0


# ----------------------------- main -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vmic",
        description="声纹聚焦虚拟麦克风（Windows，本地离线）")
    p.add_argument("--model-dir", default=MODEL_DIR, help="模型目录")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出音频端点").set_defaults(func=cmd_devices)
    sub.add_parser("profiles", help="列出声纹档案").set_defaults(func=cmd_profiles)

    pe = sub.add_parser("enroll", help="注册声纹（麦克风录制或音频文件）")
    pe.add_argument("--device", default=None, help="输入端点（索引或名称子串）")
    pe.add_argument("--seconds", type=float, default=5.0, help="录制时长，3–6 秒")
    pe.add_argument("--from-file", default=None, help="改用音频文件注册（wav/flac）")
    pe.add_argument("--name", default="", help="备注名")
    pe.set_defaults(func=cmd_enroll)

    pr = sub.add_parser("run", help="开始虚拟麦克风")
    pr.add_argument("--profile", default=None, help="声纹档案 id（reg_...）或 .npz 路径")
    pr.add_argument("--input", default=None, help="真实麦克风（索引或名称子串）")
    pr.add_argument("--output", default=None, help="虚拟声卡 Input（如 \"CABLE Input\"）")
    pr.add_argument("--threads", type=int, default=1, help="ONNX intra-op 线程数")
    pr.add_argument("--post-denoise", choices=["on", "off"], default="on",
                    help="后置降噪初始状态（运行中可热键切换）")
    pr.add_argument("--buffer", type=int, default=40,
                    help="输出抖动缓冲 ms（40 为定档默认值；不要再往下调，调大只用于排查欠载）")
    pr.add_argument("--hotkey", default="ctrl+alt+d", help="降噪切换热键")
    pr.add_argument("--no-hotkey", action="store_true", help="不注册热键")
    pr.add_argument("--seconds", type=float, default=0.0, help="跑多少秒后自动结束（0=一直跑）")
    pr.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.func(a) or 0)
    except DeviceError as e:
        print(str(e), file=sys.stderr)
        return 2
    except sd.PortAudioError as e:
        print(f"音频设备错误：{e}", file=sys.stderr)
        return 2
