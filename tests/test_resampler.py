"""P1-5 回归：逐块重采样不得在块率上留下边带，且输出长度不能漂移。

旧实现每块独立 resample_poly（零边界）：10ms 块率 100Hz → 1kHz 音两侧
±100Hz 边带约 -67dB，全带外杂散约 -46dB。
"""
import numpy as np
import pytest

from vmic.audio import Resampler


def _tone(sr: int, freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sidebands(y: np.ndarray, sr: int, freq: float) -> tuple[float, float]:
    n = len(y)
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    peak = spec[np.argmin(np.abs(fr - freq))]
    db = lambda a: 20 * np.log10(a / peak + 1e-30)  # noqa: E731

    b100 = max(spec[(fr >= freq - 120) & (fr <= freq - 45)].max(),
               spec[(fr >= freq + 45) & (fr <= freq + 120)].max())
    mask = ((fr > 40) & (fr < 8000)) & ~((fr > freq - 10) & (fr < freq + 10))
    total = np.sqrt(np.sum(spec[mask] ** 2))
    return db(b100), db(total)


def _run_chain(seconds: float = 2.0):
    x = _tone(48000, 1000.0, seconds)
    down = Resampler(48000, 16000)
    up = Resampler(16000, 48000)
    y16 = np.concatenate([down(x[i:i + 480]) for i in range(0, len(x), 480)])
    y48 = np.concatenate([up(y16[i:i + 512]) for i in range(0, len(y16), 512)])
    return x, y16, y48


def test_length_is_conserved():
    x, y16, y48 = _run_chain(1.0)
    fut = Resampler._FUT          # 前瞻窗造成固定延迟，长度差应恰为 FUT 的折算值
    assert abs(len(y16) - (len(x) - fut) / 3) <= 2
    assert abs(len(y48) - (len(y16) - fut) * 3) <= 4


def test_block_boundary_sidebands_suppressed():
    _, _, y48 = _run_chain(2.0)
    b100, total = _sidebands(y48, 48000, 1000.0)
    assert b100 < -85.0, f"±100Hz 边带仍偏高: {b100:.1f}dB"
    assert total < -75.0, f"全带外杂散仍偏高: {total:.1f}dB"


def test_identity_ratio_passthrough():
    r = Resampler(16000, 16000)
    x = _tone(16000, 440.0, 0.1)
    out = r(x)
    assert np.array_equal(out, x)


def test_empty_block_is_noop():
    r = Resampler(48000, 16000)
    assert r(np.zeros(0, np.float32)).size == 0
