"""STFT / iSTFT 与重叠相加（与 torch.stft/istft 的 center=True 语义对齐）。

分块流式之所以能逐块 iSTFT：窗 512 / hop 128、窗平方归一，overlap-add 是线性运算，
逐块累加与整段一次合成数学等价（模块内 `OnlineISTFT` 负责发射已完整的样本）。
"""
from __future__ import annotations

import numpy as np

SR = 16000
WIN = 512
HOP = 128
PAD = WIN // 2  # 256


def hann_window(n: int = WIN) -> np.ndarray:
    """周期性 Hann（torch.hann_window 默认 periodic=True）。"""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


WINDOW = hann_window(WIN)


def analyze_full(wav: np.ndarray) -> np.ndarray:
    """整段分析：reflect 两端补齐 → 分帧 → rfft。返回 (257, T) complex64。"""
    x = np.asarray(wav, dtype=np.float32)
    if len(x) < PAD + 1:
        x = np.pad(x, (0, PAD + 1 - len(x)))
    padded = np.pad(x, (PAD, PAD), mode="reflect")
    T = 1 + len(x) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(T)[:, None]
    frames = padded[idx] * WINDOW
    return np.fft.rfft(frames, n=WIN, axis=1).astype(np.complex64).T


def analyze_frames(frames: np.ndarray) -> np.ndarray:
    """(n, 512) 时域帧 → (257, n) complex64。"""
    return np.fft.rfft(frames * WINDOW, n=WIN, axis=1).astype(np.complex64).T


class OnlineISTFT:
    """逐帧 overlap-add + 窗平方归一。emit() 只输出已收到完整窗覆盖的样本。"""

    def __init__(self):
        self.ola = np.zeros(0, np.float32)
        self.env = np.zeros(0, np.float32)
        self.next_frame = 0      # 下一帧写入位置（帧索引）
        self.next_q = PAD        # 下一个可发射的 padded 样本下标（跳过左补齐）

    def _ensure(self, n: int):
        if len(self.ola) < n:
            grow = n - len(self.ola)
            self.ola = np.concatenate([self.ola, np.zeros(grow, np.float32)])
            self.env = np.concatenate([self.env, np.zeros(grow, np.float32)])

    def add_batch(self, spec_batch: np.ndarray):
        """spec_batch: (257, n_frames)，从 self.next_frame 起顺序写入。"""
        n_frames = spec_batch.shape[1]
        self._ensure((self.next_frame + n_frames) * HOP + WIN)
        w2 = WINDOW * WINDOW
        for j in range(n_frames):
            frame = np.fft.irfft(spec_batch[:, j], n=WIN).astype(np.float32)
            off = (self.next_frame + j) * HOP
            self.ola[off:off + WIN] += frame * WINDOW
            self.env[off:off + WIN] += w2
        self.next_frame += n_frames

    def emit(self) -> np.ndarray:
        """输出所有已被完整覆盖的样本（padded 坐标 q ≤ 128*(next_frame-1)+127）。"""
        if self.next_frame == 0:
            return np.zeros(0, np.float32)
        q_end = (self.next_frame - 1) * HOP + HOP  # 独占上界
        q_end = max(q_end, self.next_q)
        out = self.ola[self.next_q:q_end].copy()
        env = self.env[self.next_q:q_end]
        out = np.where(env > 1e-11, out / np.maximum(env, 1e-11), 0.0)
        self.next_q = q_end
        return out.astype(np.float32)


class OnlineAnalyzer:
    """累积输入样本，按需产出完整帧（左端 reflect，右端由调用方补零）。"""

    def __init__(self):
        self.buf = np.zeros(0, np.float32)
        self.next_frame = 0
        self.primed = False

    def push(self, samples: np.ndarray) -> np.ndarray:
        """返回自上次调用以来新可用的帧 (n, 512)。"""
        x = np.asarray(samples, dtype=np.float32)
        self.buf = np.concatenate([self.buf, x])
        if not self.primed:
            if len(self.buf) < PAD + 1:
                return np.zeros((0, WIN), np.float32)
            left = self.buf[1:PAD + 1][::-1]
            self.buf = np.concatenate([left.astype(np.float32), self.buf])
            self.primed = True
        avail = (len(self.buf) - WIN) // HOP + 1
        if avail <= self.next_frame:
            return np.zeros((0, WIN), np.float32)
        idx = np.arange(WIN)[None, :] + HOP * np.arange(self.next_frame, avail)[:, None]
        frames = self.buf[idx].copy()
        self.next_frame = avail
        return frames

    def pad_tail(self, real_len: int):
        """停止时按 torch.stft 的帧数补零，使最后一帧完整。"""
        if not self.primed:
            return
        total_frames = 1 + real_len // HOP
        need = (total_frames - 1) * HOP + WIN
        if len(self.buf) < need:
            self.buf = np.concatenate(
                [self.buf, np.zeros(need - len(self.buf), np.float32)])
