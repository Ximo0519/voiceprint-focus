"""P0-1 回归：全零预热不得让后置降噪的噪底估计永久锁死。

两个入口都会喂纯数字静音：
- vmic FocusPipeline.warmup() 灌 50 块全零；
- Web 首块可能就是数字静音。
修复前：首块把 pd_emin 初始化为 0，之后 `pd_emin *= 1.0005` 恒为 0，
噪底永不更新，谱减增益恒 1.0（降噪完全失效且本会话不可恢复）。
"""
import collections

import numpy as np

import engine
from engine import StreamSession
from vmic.pipeline import FocusPipeline

NBINS = engine.WIN // 2 + 1


def _bare_session() -> StreamSession:
    """只构造 _post_denoise 需要的字段，不加载 ONNX。"""
    s = object.__new__(StreamSession)
    s.pd_emin = None
    s.pd_noise = None
    s.pd_gain = None
    return s


def _block(amp: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    spec = (rng.standard_normal((NBINS, 4)) * amp).astype(np.complex64)
    pin = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return spec, pin


def test_silent_block_does_not_poison_noise_floor():
    s = _bare_session()
    silence = np.zeros((NBINS, 4), np.complex64)
    for _ in range(50):                       # 模拟 warmup 的 50 块全零
        s._post_denoise(silence, np.zeros((NBINS, 4), np.float32))
    assert s.pd_emin is None, "数字静音不应初始化噪底"
    assert s.pd_noise is None

    # 第一块真实音频必须重新初始化，并且能学出噪底、产生 <1 的增益
    spec, pin = _block(0.1, seed=1)
    s._post_denoise(spec, pin)
    assert s.pd_emin is not None and s.pd_emin > 0.0
    for _ in range(3):
        out = s._post_denoise(spec, pin)
    assert np.any(s.pd_noise > 0.0)
    gain = np.abs(out) / (np.abs(spec) + 1e-12)
    assert float(gain.min()) < 0.99


class FakeSess:
    def __init__(self):
        self.pd_emin = 0.0                   # 预热后残留的坏状态
        self.pd_noise = np.zeros((NBINS, 1), np.float32)
        self.pd_gain = np.zeros((NBINS, 1), np.float32)
        self.pending = []
        self.stats = []
        self.total_compute_ms = 0.0
        self.total_audio_s = 0.0
        self.post_denoise = True
        self.pushes = 0

    def push(self, x, pause_fed=False):
        self.pushes += 1
        return []


def test_warmup_resets_only_pd_state():
    """预热后 PD 状态清空；OLA/停顿时钟相关字段不受影响（不重建会话）。"""
    pipe = object.__new__(FocusPipeline)
    pipe.sess = FakeSess()
    pipe._q_out = collections.deque()
    pipe._out_samples = 1234
    pipe._acc = np.zeros(0, np.float32)
    pipe._acc = np.ones(7, np.float32)

    pipe.warmup(blocks=3)

    assert pipe.sess.pd_emin is None
    assert pipe.sess.pd_noise is None
    assert pipe.sess.pd_gain is None
    assert pipe.sess.pushes >= 3           # 仍用真实会话预热 ORT
    assert pipe._out_samples == 0
    assert pipe._acc.size == 0
