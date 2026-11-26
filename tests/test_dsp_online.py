"""dsp 在线链路等价性基线：OnlineAnalyzer / OnlineISTFT 必须与整段实现一致。

这两个类在 P0-2 会被改成滑动/环形缓冲，本文件的用例就是改造的 golden 护栏：
改造后数值必须不变（<1e-6），且缓冲长度必须有上界。
"""
import numpy as np

from dsp import (HOP, PAD, SR, WIN, WINDOW, OnlineAnalyzer, OnlineISTFT,
                 analyze_full)


def _offline_ola(spec: np.ndarray) -> np.ndarray:
    """整段 overlap-add + 窗平方归一，返回与 OnlineISTFT.emit 同坐标的样本。"""
    t = spec.shape[1]
    n = (t - 1) * HOP + WIN
    ola = np.zeros(n, np.float64)
    env = np.zeros(n, np.float64)
    w = WINDOW.astype(np.float64)
    for j in range(t):
        frame = np.fft.irfft(spec[:, j], n=WIN)
        off = j * HOP
        ola[off:off + WIN] += frame * w
        env[off:off + WIN] += w * w
    y = np.divide(ola, np.maximum(env, 1e-11),
                  out=np.zeros_like(ola), where=env > 1e-11)
    # emit 的输出坐标：padded 坐标 [PAD, T*HOP)
    return y[PAD:t * HOP]


def _signal(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    speechish = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))
    noise = 0.05 * rng.standard_normal(n)
    return (speechish + noise).astype(np.float32)


def test_analyzer_frames_match_full_analysis():
    """逐块分帧结果与 analyze_full 一致（末帧因右端补法不同单独排除）。"""
    x = _signal()
    az = OnlineAnalyzer()
    chunks = []
    for i in range(0, len(x), 160):          # 10ms 块，与 vmic 采集口径一致
        frames = az.push(x[i:i + 160])
        if len(frames):
            chunks.append(frames)
    az.pad_tail(len(x))
    tail = az.push(np.zeros(0, np.float32))
    if len(tail):
        chunks.append(tail)

    frames_time = np.concatenate(chunks, axis=0)
    got = np.fft.rfft(frames_time * WINDOW, n=WIN, axis=1).T.astype(np.complex64)
    ref = analyze_full(x)
    # 只比较「完全落在真实样本内」的帧：整段实现右端用 reflect，实时用零补齐
    n = (PAD + len(x) - WIN) // HOP + 1
    assert got.shape[1] >= n
    assert np.max(np.abs(got[:, :n] - ref[:, :n])) < 1e-5


def test_istft_stream_matches_offline_ola():
    """分块 add_batch/emit 与整段 OLA 数值一致。"""
    x = _signal(2.0, seed=1)
    spec = analyze_full(x)
    ref = _offline_ola(spec)

    istft = OnlineISTFT()
    outs = []
    for j in range(0, spec.shape[1], 4):     # 32ms 一组，与推理块一致
        istft.add_batch(spec[:, j:j + 4])
        out = istft.emit()
        if len(out):
            outs.append(out)
    got = np.concatenate(outs) if outs else np.zeros(0, np.float32)

    n = min(len(got), len(ref))
    assert n > SR                                   # 至少 1 秒有效输出
    assert np.max(np.abs(got[:n] - ref[:n])) < 1e-6


def test_online_pipeline_end_to_end_equivalence():
    """分帧 → 单位掩码 → iSTFT 全链路，与整段实现等价。"""
    x = _signal(2.0, seed=2)
    spec = analyze_full(x)
    ref = _offline_ola(spec)

    az = OnlineAnalyzer()
    istft = OnlineISTFT()
    outs = []
    pending = []
    for i in range(0, len(x), 160):
        frames = az.push(x[i:i + 160])
        if len(frames):
            pending.append(frames)
        while pending and sum(len(p) for p in pending) >= 4:
            buf = np.concatenate(pending, axis=0)
            take, rest = buf[:4], buf[4:]
            pending = [rest] if len(rest) else []
            istft.add_batch(np.fft.rfft(take * WINDOW, n=WIN, axis=1).T.astype(np.complex64))
            out = istft.emit()
            if len(out):
                outs.append(out)
    got = np.concatenate(outs) if outs else np.zeros(0, np.float32)

    n = min(len(got), len(ref))
    assert n > SR
    assert np.max(np.abs(got[:n] - ref[:n])) < 1e-5
