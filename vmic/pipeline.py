"""流式管线：16k 推理 + 16k↔设备采样率转换 + 输入/输出抖动队列。

线程模型（与 server.py 的 reader/worker/sender 三段式同构）：
  PortAudio 采集回调 → q_in（10ms 块 + 到达时刻）
  worker 线程        → feed_pause / skip / bypass / push → 转回设备采样率 → q_out
  PortAudio 播放回调 → pull(n) 取样本，不足补零并计欠载

推理线程与两条音频回调解耦，回调里只做入队/出队，避免 GIL 抖动直接变成卡音。
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np

import engine
from vmic.audio import Resampler, SR_MODEL
from vmic.cpu import bind_current_thread, p_core_affinity

# 单块预算：4 帧 × 8ms = 32ms，与 Web 端一致的交付口径
CHUNK_MS = 32


class FocusPipeline:
    def __init__(self, step_path: str, E: np.ndarray, g: np.ndarray,
                 sr_in: int, sr_out: int, threads: int = 1,
                 post_denoise: bool = True, in_ms: int = 320, out_ms: int = 400,
                 buffer_ms: int = 40):
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.sess = engine.StreamSession(step_path, E, g, threads=threads,
                                         post_denoise=post_denoise, capture=False)
        self._down = Resampler(self.sr_in, SR_MODEL)
        self._up = Resampler(SR_MODEL, self.sr_out)
        self.block_out = max(1, int(round(self.sr_out * 0.01)))

        self._q_in: collections.deque = collections.deque()
        self.in_cap = max(4, int(round(in_ms / 10.0)))
        self._q_out: collections.deque = collections.deque()
        self._out_cap = max(4, int(round(out_ms / 10.0)))
        self._out_samples = 0
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._want_denoise: bool | None = None
        self._acc = np.zeros(0, np.float32)

        self.counts = {"chunks": 0, "model": 0, "bypass": 0, "skip": 0,
                       "in_drops": 0, "out_drops": 0, "underruns": 0,
                       "startup_underruns": 0}
        self._compute_ms: collections.deque = collections.deque(maxlen=400)
        # 输出抖动缓冲：模型按 32ms 成块产出，播放按 10ms 连续消耗。
        # 不预填的话队列深度会在 0 附近随机游走，每块都可能欠载；预填 buffer_ms 后
        # 深度稳定在 [buffer, buffer+32)，代价是同等延迟。
        self.buffer_ms = max(0, int(buffer_ms))
        self._prefill_samples = int(round(self.sr_out * self.buffer_ms / 1000.0))
        self._out_started = False
        self._out_wait_since: float | None = None
        self._last_stat: dict | None = None
        self._trim_at = 8192
        self.p_core: dict | None = None
        self.affinity_bound = False

    # ---------- 生命周期 ----------
    def warmup(self, blocks: int = 50):
        """预热：跑掉 ORT 首次调用开销，并推进 OLA/停顿时钟到稳态。产物丢弃。"""
        z = np.zeros(160, np.float32)
        for _ in range(blocks):
            self.sess.push(z)
        while len(self.sess.pending):
            self.sess.push(z)
        self._q_out.clear()
        self._out_samples = 0
        self._acc = np.zeros(0, np.float32)
        # 记下预热占用的统计量，summary 里扣掉，避免把预热算进 RTF
        self._warm_chunks = len(self.sess.stats)
        self._warm_compute_ms = self.sess.total_compute_ms
        self._warm_audio_s = self.sess.total_audio_s

    def start(self):
        self._thread = threading.Thread(target=self._run, name="vmic-worker",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # ---------- 数据入口 / 出口 ----------
    def feed(self, block: np.ndarray):
        """采集回调调用：入队，超上限丢最旧（与 Web 端「丢旧不丢新」同策略）。"""
        t = time.perf_counter()
        if len(self._q_in) >= self.in_cap:
            try:
                self._q_in.popleft()
                self.counts["in_drops"] += 1
            except IndexError:
                pass
        self._q_in.append((block, t))

    def pull(self, n: int) -> np.ndarray:
        """播放回调调用：取 n 个设备采样率样本；不足时返回现有部分（由调用方补零）。"""
        with self._lock:
            if not self._out_started:
                # 预填阶段：攒够 buffer_ms 再出声，之后队列深度稳定在 [buffer, buffer+32)
                if self._out_samples < self._prefill_samples:
                    now = time.perf_counter()
                    if self._out_wait_since is None:
                        self._out_wait_since = now
                    self.counts["startup_underruns"] += 1
                    # 兜底：跑不动时不能一直哑着，1s 后带病开播
                    if now - self._out_wait_since < 1.0:
                        return np.zeros(0, np.float32)
                self._out_started = True
                self._out_wait_since = None
            if not self._q_out:
                self.counts["underruns"] += 1
                return np.zeros(0, np.float32)
            parts, got = [], 0
            while got < n and self._q_out:
                a = self._q_out[0]
                need = n - got
                if len(a) <= need:
                    parts.append(a)
                    self._q_out.popleft()
                    got += len(a)
                else:
                    parts.append(a[:need])
                    self._q_out[0] = a[need:]
                    got += need
            self._out_samples = max(0, self._out_samples - got)
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

    # ---------- 降噪开关 ----------
    def set_denoise(self, on: bool):
        """运行中切换后置降噪：由 worker 在下一块生效，避免跨线程改推理状态。"""
        self._want_denoise = bool(on)

    @property
    def denoise(self) -> bool:
        return bool(self.sess.post_denoise)

    # ---------- worker ----------
    def _apply_pending_options(self):
        if self._want_denoise is None:
            return
        on = self._want_denoise
        self._want_denoise = None
        # 走引擎自带接口：4 块（128ms）线性淡入/淡出，不重建会话、不改时间轴与延迟，
        # 与 Web 端 /api/config 的行为一致。
        self.sess.set_post_denoise(on)

    def _push_out(self, y16: np.ndarray):
        y = self._up(y16)
        self._acc = y if self._acc.size == 0 else np.concatenate([self._acc, y])
        with self._lock:
            while len(self._acc) >= self.block_out:
                if len(self._q_out) >= self._out_cap:
                    self._q_out.popleft()
                    self._out_samples = max(0, self._out_samples - self.block_out)
                    self.counts["out_drops"] += 1
                self._q_out.append(self._acc[:self.block_out])
                self._out_samples += self.block_out
                self._acc = self._acc[self.block_out:]

    def _run(self):
        # 混合架构 CPU：把推理线程钉在 P 核，否则单块耗时会在 21–38ms 之间跳，
        # 32ms 的块预算一退到 E 核就欠载。非混合 CPU / 非 Windows 上为空操作。
        self.p_core = p_core_affinity()
        self.affinity_bound = bind_current_thread(self.p_core)
        while not self._stop.is_set():
            # 降噪开关在拿块之前先落地：即使当前没有新块（静音/麦被静音）也能立即生效
            self._apply_pending_options()
            try:
                block, t_arr = self._q_in.popleft()
            except IndexError:
                time.sleep(0.002)
                continue
            waited_ms = (time.perf_counter() - t_arr) * 1000.0
            x16 = self._down(block)
            sess = self.sess
            sess.feed_pause(x16)
            try:
                if sess.cur_block_skip:
                    outs = sess.bypass(x16, True, True)
                elif sess.should_bypass(waited_ms, pause=sess.cur_block_pause):
                    outs = sess.bypass(x16, True)
                else:
                    # pause_fed=True：L188 已喂过停顿时钟，避免 push 内部二次推进
                    # （否则 g_frames 时间轴以 2 倍速超前于输出样本 s0）
                    outs = sess.push(x16, True)
            except Exception as e:  # noqa: BLE001
                # 单块异常不能让整条流断掉：清空待处理块，继续跑
                self.counts["chunks"] += 1
                self._last_stat = {"error": repr(e)}
                continue
            if not outs:
                continue
            for st in sess.stats[-len(outs):]:
                self.counts["chunks"] += 1
                self.counts[st.mode if st.mode in ("model", "bypass", "skip")
                            else "model"] += 1
                if not st.skipped:
                    self._compute_ms.append(st.compute_ms)
                self._last_stat = {
                    "compute_ms": st.compute_ms, "mode": st.mode,
                    "algo_latency_ms": st.algo_latency_ms,
                    "out_len": st.out_len}
            for o in outs:
                self._push_out(o)
            if len(sess.stats) > self._trim_at:
                del sess.stats[:self._trim_at // 2]

    # ---------- 状态 ----------
    def status(self) -> dict:
        comp = np.asarray(self._compute_ms, np.float64)
        with self._lock:
            q_out_ms = self._out_samples * 1000.0 / self.sr_out
        return {
            **self.counts,
            "buffer_ms": self.buffer_ms,
            "in_queue_ms": round(len(self._q_in) * 10.0, 1),
            "out_queue_ms": round(q_out_ms, 1),
            "compute_p50_ms": round(float(np.percentile(comp, 50)), 2) if comp.size else None,
            "compute_p95_ms": round(float(np.percentile(comp, 95)), 2) if comp.size else None,
            "algo_latency_ms": (self._last_stat or {}).get("algo_latency_ms"),
            "denoise": self.denoise,
            "p_core_cpus": (self.p_core or {}).get("n"),
            "affinity_bound": self.affinity_bound,
        }

    def summary(self) -> dict:
        s = self.sess.summary()
        audio_s = max(1e-9, s.get("audio_s", 0.0) - getattr(self, "_warm_audio_s", 0.0))
        comp_ms = max(0.0, s.get("compute_total_ms", 0.0)
                      - getattr(self, "_warm_compute_ms", 0.0))
        chunks = self.counts["chunks"]
        return {**s, **self.status(),
                "audio_s": round(audio_s, 2), "chunks": max(0, chunks),
                "compute_total_ms": round(comp_ms, 1),
                "rtf": round(comp_ms / 1000.0 / audio_s, 3)}
