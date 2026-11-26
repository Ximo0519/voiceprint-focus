"""Windows 音频端点：枚举、按名字解析、采集/播放流、16k 双向重采样。"""
from __future__ import annotations

import re
import threading
from math import gcd

import numpy as np
from scipy.signal import resample_poly

try:  # PortAudio 不可用（CI / 无声卡机器）时不影响 Resampler 等纯 DSP 逻辑
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

SR_MODEL = 16000


def _require_sd():
    if sd is None:
        raise RuntimeError(
            "sounddevice/PortAudio 不可用：本机麦克风/播放仅支持 Windows/macOS 且有音频设备")


class DeviceError(RuntimeError):
    pass


def hostapi_name(dev: dict) -> str:
    return sd.query_hostapis(dev["hostapi"])["name"]


def devices(kind: str = "all") -> list[dict]:
    """列出端点：kind ∈ all / in / out。优先 WASAPI，其余 hostapi 排在后面。"""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if kind == "in" and d["max_input_channels"] < 1:
            continue
        if kind == "out" and d["max_output_channels"] < 1:
            continue
        out.append({"index": i, "name": d["name"].strip(),
                    "hostapi": hostapi_name(d),
                    "in": d["max_input_channels"], "out": d["max_output_channels"],
                    "sr": int(d["default_samplerate"])})
    out.sort(key=lambda x: (0 if "WASAPI" in x["hostapi"] else 1, x["index"]))
    return out


def resolve(spec: str | None, kind: str) -> dict:
    """把 --input/--output 的取值解析成端点。

    支持：纯数字索引；名字子串（大小写不敏感，唯一匹配）。空值取系统默认端点。
    """
    cands = devices(kind)
    if not cands:
        raise DeviceError("没有可用的音频端点")
    if not spec:
        # 空值：取系统默认端点的同名 WASAPI 版本（MME 默认会多出上百毫秒延迟），
        # 同名找不到就退回第一个 WASAPI 端点。
        try:
            idx = int(sd.default.device[0 if kind == "in" else 1])
            dev = sd.query_devices(idx)
            name = str(dev["name"]).strip()
        except Exception:  # noqa: BLE001
            name = ""
        if name:
            same = [c for c in cands if "WASAPI" in c["hostapi"] and c["name"] == name]
            if same:
                return same[0]
        wasapi = [c for c in cands if "WASAPI" in c["hostapi"]]
        if wasapi:
            return wasapi[0]
        return cands[0]
    s = str(spec).strip()
    if s.isdigit():
        for c in cands:
            if c["index"] == int(s):
                return c
        raise DeviceError(f"索引 {s} 不是可用的{'输入' if kind == 'in' else '输出'}端点")
    hits = [c for c in cands if s.lower() in c["name"].lower()]
    if not hits:
        raise DeviceError(
            f"找不到名称含「{s}」的{'输入' if kind == 'in' else '输出'}端点；"
            "用 `python -m vmic devices` 查看完整列表")
    if len(hits) > 1:
        # 同名跨 hostapi 时优先 WASAPI
        wasapi = [h for h in hits if "WASAPI" in h["hostapi"]]
        if len(wasapi) == 1:
            return wasapi[0]
        names = "；".join(f"[{h['index']}] {h['name']} ({h['hostapi']})" for h in hits)
        raise DeviceError(f"「{s}」匹配到多个端点，请改用索引：{names}")
    return hits[0]


def channel_count(dev: dict, kind: str) -> int:
    n = dev["in"] if kind == "in" else dev["out"]
    return max(1, int(n))


def block_samples(sr: int, ms: int = 10) -> int:
    return max(1, int(round(sr * ms / 1000.0)))


class Resampler:
    """固定倍率重采样（16k ↔ 设备采样率）。

    逐块调用无状态 `resample_poly` 时，每块按零边界做 FIR，块边缘会产生周期性
    台阶：10ms 输入块的块率是 100Hz，实测 1kHz 音两侧 ±100Hz 边带约 -67dB、
    全带外杂散约 -46dB。

    这里做有状态的 overlap-save：保留 32 个输入样本的「过去历史 + 未来前瞻」，
    把窗口一起重采样后只取当前块对应的输出。FIR 两侧都有真实上下文，等价于
    整段处理，且只引入约 0.7–2ms 的固定前瞻延迟（而不是 soxr LQ 的 31ms）。

    说明：soxr 的 LQ 档同样能压掉边带，但 48k→16k→48k 全链约 31ms 群延迟，
    会吃掉 118ms 端到端预算的 1/4；QQ 档抗混叠仅 -12dB，不可用。
    """

    _HIST = 64   # 输入侧历史样本数（≥ 多相 FIR 半长，scipy 默认 61 taps）
    _FUT = 32    # 输入侧前瞻样本数（同上，用于补足 FIR 的未来上下文）

    def __init__(self, from_sr: int, to_sr: int):
        g = gcd(int(from_sr), int(to_sr))
        self.up = int(to_sr) // g
        self.down = int(from_sr) // g
        self._hist = np.zeros(0, np.float32)     # 已处理样本的尾部
        self._pending = np.zeros(0, np.float32)  # 已收到但还缺前瞻、暂不处理的样本
        self._processed = 0                       # 已处理的输入样本总数
        self._emitted = 0                         # 已输出的样本总数

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.up == self.down == 1 or len(x) == 0:
            return x
        self._pending = (np.concatenate([self._pending, x])
                         if len(self._pending) else x)
        k = len(self._pending) - self._FUT
        if k <= 0:
            return np.zeros(0, np.float32)

        chunk, future = self._pending[:k], self._pending[k:]
        buf = np.concatenate([self._hist, chunk, future])
        y = resample_poly(buf, self.up, self.down).astype(np.float32)

        # 输出区间用全局累计坐标换算，避免逐块取整造成长度漂移
        r = self.up / self.down
        a = int(round(len(self._hist) * r))
        want = int(round((self._processed + k) * r)) - self._emitted
        out = y[a:a + want]
        if len(out) < want:                       # 浮点/取整边界兜底
            out = np.pad(out, (0, want - len(out)))

        self._processed += k
        self._emitted += want
        self._pending = future.copy()
        past = np.concatenate([self._hist, chunk])
        self._hist = past[-self._HIST:].copy()
        return out


class AudioIn:
    """采集流：固定 10ms 块回调，块与到达时刻交给 on_block。

    on_block 必须极短（只做入队）；重采样与推理都在管线线程里做。
    回调里的异常不能上抛到 PortAudio，统一吞掉并计数。
    """

    def __init__(self, dev: dict, sr: int, on_block, block_ms: int = 10,
                 channels: int = 1):
        self.dev = dev
        self.sr = int(sr)
        self.block = block_samples(sr, block_ms)
        self.channels = max(1, min(channels, channel_count(dev, "in")))
        self.on_block = on_block
        self.errors = 0
        self._stream: sd.InputStream | None = None

    def _cb(self, indata, frames, time_info, status):  # noqa: ANN001
        try:
            x = indata[:, 0] if self.channels == 1 else indata.mean(axis=1)
            self.on_block(np.array(x, np.float32, copy=True))
        except Exception:  # noqa: BLE001
            self.errors += 1

    def start(self):
        _require_sd()
        self._stream = sd.InputStream(
            device=self.dev["index"], samplerate=self.sr, channels=self.channels,
            dtype="float32", blocksize=self.block, latency="low", callback=self._cb)
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    @property
    def reported_latency_ms(self) -> float | None:
        try:
            lat = self._stream.latency
            v = lat[0] if isinstance(lat, tuple) else lat
            return round(float(v) * 1000.0, 1)
        except Exception:  # noqa: BLE001
            return None


class AudioOut:
    """播放流：回调里向 pull(n) 要 n 个样本；不足则补零并计欠载。"""

    def __init__(self, dev: dict, sr: int, pull, block_ms: int = 10,
                 channels: int = 2):
        self.dev = dev
        self.sr = int(sr)
        self.block = block_samples(sr, block_ms)
        self.channels = max(1, channel_count(dev, "out"))
        self.pull = pull
        self.underruns = 0
        self.errors = 0
        self._stream: sd.OutputStream | None = None

    def _cb(self, outdata, frames, time_info, status):  # noqa: ANN001
        try:
            y = self.pull(frames)
            if len(y) < frames:
                self.underruns += 1
                y = np.concatenate([y, np.zeros(frames - len(y), np.float32)])
            outdata[:] = y[:frames, None] if self.channels > 1 else y[:frames]
        except Exception:  # noqa: BLE001
            self.errors += 1
            outdata.fill(0)

    def start(self):
        _require_sd()
        self._stream = sd.OutputStream(
            device=self.dev["index"], samplerate=self.sr, channels=self.channels,
            dtype="float32", blocksize=self.block, latency="low", callback=self._cb)
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    @property
    def reported_latency_ms(self) -> float | None:
        try:
            lat = self._stream.latency
            v = lat[1] if isinstance(lat, tuple) else lat
            return round(float(v) * 1000.0, 1)
        except Exception:  # noqa: BLE001
            return None


_CABLE_RE = re.compile(r"cable|vb-audio|voicemeeter|virtual|streaming|loopback", re.I)


def looks_virtual(name: str) -> bool:
    return bool(_CABLE_RE.search(name))


def describe(dev: dict) -> str:
    flag = " ← 虚拟端点" if looks_virtual(dev["name"]) else ""
    return f"[{dev['index']}] {dev['name']} ({dev['hostapi']}, {dev['sr']}Hz){flag}"
