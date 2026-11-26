"""注册时长口径契约：前端、服务端常量与 vad_report 闸门必须是同一个 2.5–6.5s。

修复前：文件注册宣称 30s、录制宣称 8s、vad_report 硬编码 6.5s，
6.5s 以上的素材全部在 VAD 阶段被 422 拒绝，上限配置形同虚设。
"""
import numpy as np

import engine
import server


def _gated_tone(seconds: float) -> np.ndarray:
    """有声/停顿交替的信号：保证 20 分位噪底足够低，语音占比能过 8%。"""
    n = int(seconds * engine.SR)
    t = np.arange(n) / engine.SR
    gate = (np.mod(t, 0.3) < 0.2).astype(np.float32)
    x = 0.15 * gate * np.sin(2 * np.pi * 200 * t)
    x += 0.0005 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
    return x.astype(np.float32)


def test_accepts_duration_within_contract():
    for sec in (2.6, 5.0, 6.4):
        q = engine.vad_report(_gated_tone(sec))
        assert q["ok"], (sec, q)


def test_rejects_duration_outside_contract():
    q = engine.vad_report(_gated_tone(6.6))
    assert not q["ok"] and "超过" in q["reason"], q
    q = engine.vad_report(_gated_tone(2.4))
    assert not q["ok"] and "不足" in q["reason"], q


def test_server_limits_share_single_source_of_truth():
    assert engine.ENROLL_DUR_MIN == 2.5
    assert engine.ENROLL_DUR_MAX == 6.5
    assert server.MAX_ENROLL_S == engine.ENROLL_DUR_MAX
    assert server.MAX_ENROLL_FILE_S == engine.ENROLL_DUR_MAX
