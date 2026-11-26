"""P0-3 回归：vmic 模型分支必须把 pause_fed=True 传给 push()。

背景：worker 已在 L188 调过 sess.feed_pause(x16)。若 push() 不传 pause_fed，
push() 内部会二次喂停顿时钟，使 pause_ff.n_frames 以 2 倍速前进，
g_frames 的绝对坐标相对输出样本 s0 持续超前，约 4s 后 np.interp 退化为常量钳位。
"""
import collections
import threading
import time

import numpy as np

from vmic.audio import Resampler
from vmic.pipeline import FocusPipeline


class FakeSession:
    def __init__(self, on_push=None):
        self.calls = []
        self.cur_block_skip = False
        self.cur_block_pause = False
        self.pending = []
        self.stats = []
        self.total_compute_ms = 0.0
        self.total_audio_s = 0.0
        self.post_denoise = True
        self._on_push = on_push

    def feed_pause(self, x):
        self.calls.append(("feed_pause",))
        return False

    def should_bypass(self, waited_ms, pause=None):
        return False

    def push(self, x, pause_fed=False):
        self.calls.append(("push", pause_fed))
        if self._on_push is not None:
            self._on_push()
        return []

    def bypass(self, x, pause_fed=False, skip=False):
        self.calls.append(("bypass", pause_fed, skip))
        if self._on_push is not None:
            self._on_push()
        return []


def _make_pipe(sess):
    pipe = object.__new__(FocusPipeline)
    pipe.sess = sess
    pipe._stop = threading.Event()
    pipe._want_denoise = None
    pipe._q_in = collections.deque()
    pipe._q_out = collections.deque()
    pipe._out_samples = 0
    pipe._acc = np.zeros(0, np.float32)
    pipe._out_cap = 10
    pipe.block_out = 480
    pipe._lock = threading.Lock()
    pipe._down = Resampler(48000, 16000)
    pipe.counts = {"chunks": 0, "model": 0, "bypass": 0, "skip": 0,
                   "in_drops": 0, "out_drops": 0, "underruns": 0,
                   "startup_underruns": 0}
    pipe._compute_ms = collections.deque(maxlen=10)
    pipe._last_stat = None
    pipe.p_core = None
    pipe.affinity_bound = False
    return pipe


def _run_one_block(sess):
    pipe = _make_pipe(sess)
    pipe._q_in.append((np.zeros(480, np.float32), time.perf_counter()))
    # push 里停掉 worker，模拟处理完一块后收到 stop
    sess._on_push = pipe._stop.set
    pipe._run()
    return sess.calls


def test_model_path_passes_pause_fed():
    calls = _run_one_block(FakeSession())
    assert ("feed_pause",) in calls
    assert calls[-1] == ("push", True)
