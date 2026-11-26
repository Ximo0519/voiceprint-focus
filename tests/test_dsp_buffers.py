"""P0-2 回归：在线 STFT/iSTFT 的缓冲必须有界，且不能随运行时长线性增长。

修复前 OnlineAnalyzer.buf 与 OnlineISTFT.ola/env 只增不减：
- 常驻进程（vmic）每小时约 0.7GB；
- 每次 push/add_batch 都整段 concatenate，总拷贝量 O(N²)，长时后单块拷贝即可击穿 32ms。
本用例只断言「长度有界」——它同时等价于「不存在 O(N²) 整段拷贝」。
"""
import numpy as np

from dsp import HOP, PAD, SR, WIN, OnlineAnalyzer, OnlineISTFT, analyze_frames


def _stream(seconds: float, chunk: int = 160):
    az = OnlineAnalyzer()
    ist = OnlineISTFT()
    rng = np.random.default_rng(0)
    n_total = int(seconds * SR)
    buf: list[np.ndarray] = []
    max_ola = 0
    for i in range(0, n_total, chunk):
        x = (0.1 * rng.standard_normal(min(chunk, n_total - i))).astype(np.float32)
        frames = az.push(x)
        if len(frames):
            buf.append(frames)
        while buf and sum(len(b) for b in buf) >= 4:
            allf = np.concatenate(buf, axis=0)
            take, rest = allf[:4], allf[4:]
            buf = [rest] if len(rest) else []
            ist.add_batch(analyze_frames(take))
            ist.emit()
            max_ola = max(max_ola, len(ist.ola))
    return az, ist, max_ola


def test_analyzer_buffer_bounded():
    az, _, _ = _stream(20)
    # 只需保留最后一帧的窗口（~WIN）加少量余量
    assert len(az.buf) < 4 * WIN, f"analyzer buf 无界增长: {len(az.buf)}"


def test_istft_buffers_bounded():
    _, ist, max_ola = _stream(20)
    assert len(ist.ola) < 4 * WIN, f"istft ola 无界增长: {len(ist.ola)}"
    assert len(ist.env) < 4 * WIN, f"istft env 无界增长: {len(ist.env)}"
    assert max_ola < 4 * WIN, f"istft ola 历史峰值过大: {max_ola}"


def test_absolute_coordinates_preserved():
    """裁剪不能改变绝对坐标：next_q 必须约等于已输出样本 + PAD 且单调递增。"""
    az, ist, _ = _stream(5)
    total_out = ist.next_q - PAD
    assert total_out > 0
    # 5s 实时流：输出样本数应与输入样本数同量级（尾部有 1 个窗的余量）
    assert abs(total_out - 5 * SR) <= 2 * WIN
    # 全局帧号仍按 HOP 累计，不受裁剪影响
    assert ist.next_frame == az.next_frame
