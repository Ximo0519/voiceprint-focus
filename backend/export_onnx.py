"""导出实验16 流式学生为两个 ONNX（enroll / step），并做 PyTorch↔ONNX 等价校验。

- enroll.onnx:  mag(1,257,Ne) + mask(1,Ne)                 -> E(1,Ne,192), g(1,32)
- step.onnx:    spec_ri(1,2,257,T) + E + emask + g + h(6,32,256) + c(6,32,256)
                                                           -> est_ri(1,2,257,T), h', c'
STFT/iSTFT 在 ONNX 外（dsp.py）完成，图里只有网络本体，保证与训练代码逐算子同构。

用法: python app/backend/export_onnx.py --ckpt <avg_model.pt> --out app/models/x16_stream
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import ENC_DIM, NBAND, SPK_EMB_DIM, TFMapCausalBSRNN, load_model  # noqa: E402


class EnrollNet(nn.Module):
    """enrollment 编码：线性幅度谱 → E / 频带门控 g（整段离线，不占混合路径延迟）。"""

    def __init__(self, model: TFMapCausalBSRNN):
        super().__init__()
        self.m = model

    def forward(self, mag, mask):
        vf = mask.sum(-1, keepdim=True).clamp_min(1.0)
        tmag = (mag * mask.unsqueeze(1)).sum(-1) / vf
        band = torch.log1p(tmag @ self.m.tfmap.band_matrix)
        mu = band.mean(-1, keepdim=True)
        var = ((band - mu) ** 2).mean(-1, keepdim=True)
        band = (band - mu) / (torch.sqrt(var) + 1e-5)
        g = torch.sigmoid(self.m.tfmap.template_conv(band.unsqueeze(-1))).squeeze(-1)
        logm = torch.log(mag.clamp_min(1e-6))
        logm = logm - (logm * mask.unsqueeze(1)).sum(-1, keepdim=True) / vf.unsqueeze(-1)
        E = self.m.tfmap.enc(logm).transpose(1, 2).contiguous()
        return E, g


class StepNet(nn.Module):
    """32ms(4帧) 单步：编码 → 门控/注意力调制 → 6 层因果 BSNet → mask 合成。"""

    def __init__(self, model: TFMapCausalBSRNN):
        super().__init__()
        self.m = model

    def forward(self, spec_ri, E, emask, g, h, c):
        m = self.m
        B, _, _, T = spec_ri.shape
        feats, mixes = [], []
        idx = 0
        for i, bw in enumerate(m.band_width):
            sb = spec_ri[:, :, idx:idx + bw, :]
            mixes.append(sb)
            feats.append(m.BN[i](sb.reshape(B, bw * 2, T)))
            idx += bw
        x = torch.stack(feats, 1) * g.view(1, -1, 1, 1)  # (1,32,128,T)
        nb, C = m.nband, m.feature_dim
        M = m.tfmap.mproj(x.reshape(B, nb * C, T).transpose(1, 2))
        scores = (M @ E.transpose(1, 2)) / math.sqrt(float(SPK_EMB_DIM))
        scores = scores.masked_fill(~emask.unsqueeze(1), float("-inf"))
        ctx = torch.softmax(scores, dim=-1) @ E
        coeff = m.tfmap.lowrank(ctx).view(B, T, 2, nb)
        inj = torch.einsum("btnc,nce->btce", coeff, m.tfmap.basis)
        x = x + inj.permute(0, 2, 3, 1)

        xf = x.reshape(B, nb * C, T)
        hs, cs = [], []
        seps = m.separator.separation
        for li in range(1, len(seps)):
            st = {"band_rnn": (h[li - 1:li], c[li - 1:li])}
            xf = seps[li](xf, st)
            hh, cc = st["band_rnn"]
            hs.append(hh)
            cs.append(cc)
        h_out, c_out = torch.cat(hs, 0), torch.cat(cs, 0)

        sep = xf.view(B, nb, C, T)
        est = []
        for i, mask_func in enumerate(m.mask):
            out = mask_func(sep[:, i]).view(B, 2, 2, m.band_width[i], T)
            mk = out[:, 0] * torch.sigmoid(out[:, 1])
            mr, mi = mk[:, 0], mk[:, 1]
            mix_r, mix_i = mixes[i][:, 0], mixes[i][:, 1]
            est.append(torch.stack([mix_r * mr - mix_i * mi,
                                    mix_r * mi + mix_i * mr], 1))
        return torch.cat(est, 2), h_out, c_out


def export(ckpt: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt)
    print(f"[export] loaded {ckpt}  nband={model.nband} enc_dim={model.enc_dim}")

    enroll_net = EnrollNet(model).eval()
    step_net = StepNet(model).eval()

    # ---- enroll ----
    Ne = 125
    mag = torch.rand(1, ENC_DIM, Ne) + 0.1
    mask = torch.ones(1, Ne, dtype=torch.bool)
    p_enroll = os.path.join(out_dir, "enroll.onnx")
    t0 = time.time()
    torch.onnx.export(
        enroll_net, (mag, mask), p_enroll, opset_version=17,
        input_names=["mag", "mask"], output_names=["E", "g"],
        dynamic_axes={"mag": {2: "Ne"}, "mask": {1: "Ne"},
                      "E": {1: "Ne"}},
        do_constant_folding=True, dynamo=False)
    print(f"[export] enroll.onnx {os.path.getsize(p_enroll)/1e6:.1f}MB "
          f"{time.time()-t0:.1f}s")

    # ---- step ----
    T = 4
    spec = torch.randn(1, 2, ENC_DIM, T)
    E = torch.randn(1, Ne, SPK_EMB_DIM)
    emask = torch.ones(1, Ne, dtype=torch.bool)
    g = torch.rand(1, NBAND)
    h = torch.zeros(6, NBAND, 256)
    c = torch.zeros(6, NBAND, 256)
    p_step = os.path.join(out_dir, "step.onnx")
    t0 = time.time()
    torch.onnx.export(
        step_net, (spec, E, emask, g, h, c), p_step, opset_version=17,
        input_names=["spec_ri", "E", "emask", "g", "h", "c"],
        output_names=["est_ri", "h_out", "c_out"],
        dynamic_axes={"spec_ri": {3: "T"}, "E": {1: "Ne"}, "emask": {1: "Ne"},
                      "est_ri": {3: "T"}},
        do_constant_folding=True, dynamo=False)
    print(f"[export] step.onnx {os.path.getsize(p_step)/1e6:.1f}MB "
          f"{time.time()-t0:.1f}s")

    # ---- 等价校验（PyTorch wrapper vs ORT） ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    se = ort.InferenceSession(p_enroll, so, providers=["CPUExecutionProvider"])
    ss = ort.InferenceSession(p_step, so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    mag_n = (rng.random((1, ENC_DIM, 200)) + 0.05).astype(np.float32)
    mask_n = np.ones((1, 200), np.bool_)
    with torch.no_grad():
        E_t, g_t = enroll_net(torch.from_numpy(mag_n), torch.from_numpy(mask_n))
    E_o, g_o = se.run(None, {"mag": mag_n, "mask": mask_n})
    d_E = float(np.abs(E_t.numpy() - E_o).max())
    d_g = float(np.abs(g_t.numpy() - g_o).max())
    print(f"[check] enroll  max|dE|={d_E:.2e} max|dg|={d_g:.2e}")

    spec_n = rng.standard_normal((1, 2, ENC_DIM, 9)).astype(np.float32)
    trng = np.random.default_rng(7)
    E_n = trng.standard_normal((1, 173, SPK_EMB_DIM)).astype(np.float32)
    emask_n = np.ones((1, 173), np.bool_)
    g_n = trng.random((1, NBAND)).astype(np.float32)
    h_n = np.zeros((6, NBAND, 256), np.float32)
    c_n = np.zeros((6, NBAND, 256), np.float32)
    with torch.no_grad():
        est_t, h_t, c_t = step_net(
            torch.from_numpy(spec_n), torch.from_numpy(E_n),
            torch.from_numpy(emask_n), torch.from_numpy(g_n),
            torch.from_numpy(h_n), torch.from_numpy(c_n))
    est_o, h_o, c_o = ss.run(None, {"spec_ri": spec_n, "E": E_n, "emask": emask_n,
                                    "g": g_n, "h": h_n, "c": c_n})
    d_est = float(np.abs(est_t.numpy() - est_o).max())
    d_h = float(np.abs(h_t.numpy() - h_o).max())
    print(f"[check] step    max|dest|={d_est:.2e} max|dh|={d_h:.2e}")

    # 时序稳定性：同一输入连跑两次，h/c 是否确定性推进
    est_o2, h_o2, _ = ss.run(None, {"spec_ri": spec_n, "E": E_n, "emask": emask_n,
                                    "g": g_n, "h": h_o, "c": c_o})
    print(f"[check] step 二次调用确定性 max|d2|={float(np.abs(est_o2-est_o).max()):.2e}")

    report = {
        "ckpt": os.path.abspath(ckpt),
        "enroll_onnx": p_enroll, "step_onnx": p_step,
        "enroll_max_abs_diff_E": d_E, "enroll_max_abs_diff_g": d_g,
        "step_max_abs_diff_est": d_est, "step_max_abs_diff_h": d_h,
        "pass": bool(max(d_E, d_g, d_est, d_h) < 1e-3),
        "opset": 17, "chunk_frames": 4, "win": 512, "hop": 128, "sr": 16000,
    }
    with open(os.path.join(out_dir, "export_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[export] pass={report['pass']} → {out_dir}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = export(args.ckpt, args.out)
    sys.exit(0 if rep["pass"] else 2)


if __name__ == "__main__":
    main()
