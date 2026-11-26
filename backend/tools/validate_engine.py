"""链路自检 + 实测：STFT 对齐、ONNX 在线输出 vs PyTorch 参考流式、SI-SNRi、延迟探针、RTF。

用法:
  python backend/tools/validate_engine.py --ckpt <avg_model.pt> \
      --onnx-dir models/x16_stream --data data/zh_ft/dev.csv --limit 300 \
      --out <work>/metrics/validate.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
from dsp import HOP, SR, WIN, analyze_full  # noqa: E402
from engine import Enroller, StreamSession, enroll_mag, vad_report  # noqa: E402
from model_def import load_model  # noqa: E402


def load_wav(path: str) -> np.ndarray:
    w, r = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    assert r == SR, f"{path} sr={r}"
    return w


def si_snr(target: np.ndarray, est: np.ndarray) -> float:
    target = target - target.mean()
    est = est - est.mean()
    s = np.dot(target, est) / (np.dot(target, target) + 1e-8)
    t = s * target
    n = est - t
    return 10 * np.log10((np.dot(t, t) + 1e-8) / (np.dot(n, n) + 1e-8))


def check_stft():
    wav = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    ref = torch.stft(torch.from_numpy(wav).unsqueeze(0), n_fft=WIN, hop_length=HOP,
                     window=torch.hann_window(WIN), return_complex=True)[0].numpy()
    mine = analyze_full(wav)
    return float(np.abs(ref - mine).max())


def stream_online(sess: StreamSession, x: np.ndarray, block: int = 128):
    outs = []
    for i in range(0, len(x), block):
        outs.extend(sess.push(x[i:i + block]))
    outs.extend(sess.flush())
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def pick_probe_T(mix):
    n = len(mix)
    w = 800
    lo, hi = n // 4, n - w
    if hi <= lo + 1:
        return n // 2
    c = np.concatenate([[0.0], np.cumsum(mix[:hi + w].astype(np.float64) ** 2)])
    e = c[w:hi + w] - c[:hi]
    return lo + int(np.argmax(e[lo:hi]))


def _win_sum(a, win):
    c = np.concatenate([[0.0], np.cumsum(a.astype(np.float64))])
    return c[win:] - c[:-win]


def detect_eff_latency_ms(y, yp, t_idx):
    """实验15/16 同一规则：8ms 窗、连续 2 窗超相对阈 1e-4 才算响应。"""
    probe_win = 128
    limit = t_idx - 2 * probe_win
    if limit <= 1:
        return 0.0
    ed = _win_sum((y - yp) ** 2, probe_win)
    ey = _win_sum(y * y, probe_win)
    thr = np.maximum(1e-10, 1e-4 * ey)
    hit = ed[:limit] > thr[:limit]
    idx = np.nonzero(hit[:-1] & hit[1:])[0]
    if len(idx) == 0:
        return 0.0
    return float((t_idx - int(idx[0])) * 1000.0 / SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--data", default="data/zh_ft/dev.csv")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t_all = time.time()
    rep = {"ckpt": os.path.abspath(args.ckpt), "data": os.path.abspath(args.data)}

    # 1) STFT 对齐
    rep["stft_max_abs_diff"] = check_stft()
    print(f"[1] numpy STFT vs torch.stft max|diff|={rep['stft_max_abs_diff']:.2e}")

    # 2) 载入权重与 ONNX
    model = load_model(args.ckpt)
    enr = Enroller(os.path.join(args.onnx_dir, "enroll.onnx"))
    step_path = os.path.join(args.onnx_dir, "step.onnx")

    rows = list(csv.DictReader(open(args.data, newline="", encoding="utf-8")))[:args.limit]
    print(f"[2] demo 条目 n={len(rows)}（本地 TTS dev 集，非 V1 评测集）")

    imps, ref_imps, diffs, tail_diffs = [], [], [], []
    vad_list = []
    t0 = time.time()
    probe = []
    for i, r in enumerate(rows):
        mix = load_wav(r["mixture_path"])
        tgt = load_wav(r["target_path"])
        en = load_wav(r["enrollment_path"])
        vad_list.append(vad_report(en))
        E, g, _ = enr.encode(en)
        sess = StreamSession(step_path, E, g)
        est = stream_online(sess, mix)

        # PyTorch 同结构参考（整段流式）
        with torch.no_grad():
            ref = model.forward_stream(torch.from_numpy(mix)[None],
                                       torch.from_numpy(en)[None],
                                       chunk_frames=4)[0].numpy()[:len(mix)]
        n = min(len(est), len(ref), len(tgt))
        d = np.abs(est[:n] - ref[:n])
        cut = max(0, n - 2 * WIN)
        diffs.append(float(d[:cut].max()) if cut > 0 else 0.0)
        tail_diffs.append(float(d[cut:].max()) if n > cut else 0.0)
        imps.append(si_snr(tgt[:n], est[:n]) - si_snr(tgt[:n], mix[:n]))
        ref_imps.append(si_snr(tgt[:n], ref[:n]) - si_snr(tgt[:n], mix[:n]))

        # 延迟探针：同一条音频，尾部从 T 起置零，走同样的实时分块路径
        if i < args.probe:
            t_idx = pick_probe_T(mix)
            xp = mix.copy()
            xp[t_idx:] = 0.0
            sess2 = StreamSession(step_path, E, g)
            yp = stream_online(sess2, xp)
            m2 = min(len(est), len(yp))
            probe.append({"idx": i, "T_ms": round(t_idx * 1000 / SR, 1),
                          "eff_latency_ms": round(detect_eff_latency_ms(
                              est[:m2], yp[:m2], t_idx), 1)})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(rows)}  mean_imp={np.mean(imps):+.3f}dB "
                  f"({(time.time()-t0):.1f}s)")

    rep["n"] = len(rows)
    rep["mean_si_snri_db"] = round(float(np.mean(imps)), 3)
    rep["std_si_snri_db"] = round(float(np.std(imps, ddof=1)), 3)
    rep["ci95_db"] = round(float(1.96 * np.std(imps, ddof=1) / np.sqrt(len(imps))), 3)
    rep["mean_si_snri_torch_ref_db"] = round(float(np.mean(ref_imps)), 3)
    rep["max_abs_diff_vs_torch_interior"] = float(np.max(diffs))
    rep["max_abs_diff_vs_torch_tail"] = float(np.max(tail_diffs))
    rep["vad_all_ok"] = bool(all(v["ok"] for v in vad_list))
    rep["vad_sample"] = vad_list[:3]
    rep["probe"] = probe
    rep["probe_max_ms"] = max((p["eff_latency_ms"] for p in probe), default=0.0)
    rep["probe_mean_ms"] = round(float(np.mean([p["eff_latency_ms"] for p in probe])), 1) \
        if probe else 0.0
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["elapsed_s"] = round(time.time() - t_all, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[3] SI-SNRi(ONNX在线)={rep['mean_si_snri_db']:+.3f}dB "
          f"CI95±{rep['ci95_db']:.2f} | torch参考={rep['mean_si_snri_torch_ref_db']:+.3f}dB")
    print(f"[4] max|ONNX-torch| 内部={rep['max_abs_diff_vs_torch_interior']:.2e} "
          f"尾部2窗={rep['max_abs_diff_vs_torch_tail']:.2e}")
    print(f"[5] 延迟探针 n={len(probe)} max={rep['probe_max_ms']}ms "
          f"mean={rep['probe_mean_ms']}ms")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
