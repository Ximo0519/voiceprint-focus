"""CAM++ 说话人验证 · 注册环节旁路观察（advisory only）。

定位
    只在注册音频落盘后额外算一次 CAM++ 嵌入与前后半段余弦，结果写进
    诊断导出（JSONL / 单条 JSON）与服务端日志。**不参与任何判定**：
    不改变 VAD、enroll_qc、注册成败与错误文案；阈值 0.523 仅作日志标注。

模型
    3D-Speaker CAM++（zh-cn, 16k-common），Apache-2.0，192 维，fbank 输入。
    前处理交给 sherpa-onnx（与 2026-11-22/speaker-verify 标定同一版本 2.0.2），
    避免自实现 Kaldi fbank 造成口径偏差。缺失依赖时整体禁用（fail-open）。

开关
    VF_CAMPP=0              关闭旁路（默认开）
    VF_CAMPP_DIAG_DIR=...   诊断输出目录（默认 backend/campp_diag）
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(
    APP_DIR, "models", "campplus",
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
DIAG_DIR = os.environ.get(
    "VF_CAMPP_DIAG_DIR", os.path.join(APP_DIR, "backend", "campp_diag"))
ENABLED = os.environ.get("VF_CAMPP", "1") != "0"
THRESHOLD_REF = 0.523          # 仅日志标注，不触发接受/拒绝
MODEL_NAME = "3dspeaker_speech_campplus_sv_zh-cn_16k-common"
SR = 16000

log = logging.getLogger("voiceprint")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class CamppObserver:
    """CAM++ 嵌入提取器。构造失败即禁用（调用方 catch）。"""

    def __init__(self, model_path: str | None = None, threads: int = 2):
        import sherpa_onnx
        self.model_path = model_path or DEFAULT_MODEL
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path, num_threads=int(threads),
            debug=False, provider="cpu")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.dim = int(self.ex.dim)
        self.model_bytes = os.path.getsize(self.model_path)
        try:
            import importlib.metadata as md
            self.sherpa_version = md.version("sherpa-onnx")
        except Exception:                       # noqa: BLE001
            self.sherpa_version = ""
        self.backend = "sherpa-onnx"

    def embed(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        st = self.ex.create_stream()
        st.accept_waveform(sample_rate=SR, waveform=x)
        st.input_finished()
        if not self.ex.is_ready(st):
            raise RuntimeError("sherpa stream 未就绪")
        v = np.asarray(self.ex.compute(st), np.float64).reshape(-1)
        n = float(np.linalg.norm(v))
        return (v / n if n > 1e-9 else v).astype(np.float32)


def _model_info(ob: CamppObserver) -> dict:
    return {"name": MODEL_NAME, "dim": ob.dim, "backend": ob.backend,
            "sherpa_onnx": ob.sherpa_version, "license": "Apache-2.0",
            "bytes": ob.model_bytes, "file": os.path.basename(ob.model_path)}


def _write(rec: dict) -> None:
    os.makedirs(DIAG_DIR, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    log.info("campp_bypass %s", line)                       # → server.out.log
    with open(os.path.join(DIAG_DIR, "campp_bypass.jsonl"), "a",
              encoding="utf-8") as f:                       # 累计 JSONL
        f.write(line + "\n")
    with open(os.path.join(DIAG_DIR, f"{rec['run_id']}.json"), "w",
              encoding="utf-8") as f:                       # 单次注册诊断导出
        json.dump(rec, f, ensure_ascii=False, indent=2)


def observe(ob: CamppObserver | None, wav: np.ndarray, ctx: dict) -> dict | None:
    """计算一条注册音频的 CAM++ 分数并落盘。永不抛异常、永不返回判定结果。

    ctx 可含：run_id、split(a,b,cut)、事件上下文（vad_ok/...）。仅用于记录。
    """
    if ob is None:
        return None
    try:
        a, b, cut_s = ctx.get("split") or (None, None, None)
        if a is None:
            cut = int(len(wav) // 2)
            a, b, cut_s = wav[:cut], wav[cut:], cut / SR
        t0 = time.perf_counter()
        ea = ob.embed(a)
        t1 = time.perf_counter()
        eb = ob.embed(b)
        t2 = time.perf_counter()
        cos = float(np.dot(ea, eb))
        rec = {
            "schema": "campp_bypass_v1",
            "ts": _now_iso(),
            "run_id": ctx.get("run_id", ""),
            "mode": "bypass_observe",
            "enforced": False,
            "applied_to_decision": False,
            "model": _model_info(ob),
            "audio": {"sr": SR, "dur_s": round(len(wav) / SR, 3),
                      "split_s": round(float(cut_s), 3),
                      "a_s": round(len(a) / SR, 3), "b_s": round(len(b) / SR, 3)},
            "score": {"cos": round(cos, 4),
                      "threshold_ref": THRESHOLD_REF,
                      "verdict_ref": ("above_ref" if cos >= THRESHOLD_REF
                                      else "below_ref"),
                      "note": "参考阈值仅作日志标注，不触发接受/拒绝"},
            "latency_ms": {"embed_a": round((t1 - t0) * 1e3, 1),
                           "embed_b": round((t2 - t1) * 1e3, 1),
                           "total": round((t2 - t0) * 1e3, 1)},
            "context": {k: v for k, v in ctx.items()
                        if k not in ("split", "run_id")},
        }
        _write(rec)
        return rec
    except Exception as e:                                  # noqa: BLE001
        log.warning("campp_bypass error run_id=%s: %r", ctx.get("run_id"), e)
        return None
