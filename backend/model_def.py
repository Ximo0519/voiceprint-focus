# Copyright (c) 2026 Ke Zhang and the WeSep contributors (upstream WeSep)
# Copyright (c) 2026 voiceprint-focus contributors
# SPDX-License-Identifier: Apache-2.0
#
# 本文件是 wenet-e2e/WeSep（Apache-2.0）的衍生作品：
#   Upstream project: https://github.com/wenet-e2e/WeSep
#   Operative upstream layout: wesep_2024 分支（wesep/models/bsrnn.py）
# 已作显著修改（见 Apache-2.0 §4(b)）：
#   - bidirectional ResRNN -> 单向因果 LSTM，并增加 (h, c) 跨块状态缓存
#   - GroupNorm(1, C) -> 逐帧频率维归一（PerFrameNorm）
#   - band_comm 由双向改为单向
#   - 移除 utterance 级说话人条件通路（separation[0] / spk_model）
#   - 新增 TF-Map-lite 分支（谱模板门控 + 低秩加性注入）
#   - 将单一 forward 拆为 window/analyze/encode/mask_synth/istft 等辅助方法
#   - 新增 ONNX 友好加载器 load_model 与流式入口 forward_stream
# 上游署名与第三方条款见仓库根目录 NOTICE。

"""场景1 流式声纹聚焦 — WeSep BSRNN 的因果化推理裁剪版模型定义。

来源：scripts/w14_causal_train.py（CausalBSRNN）+ scripts/w16_tfmap_train.py（TFMapCausalBSRNN）。
只保留前向推理必需的部分：
  - 实验16 学生 = 实验14 因果 BSRNN + TF-Map-lite 帧级条件化；
  - 原 utterance 条件通路（separation[0] / spk_model）已在本实验前向中整段移除，
    因此这里不重建 ECAPA，也不参与计算（仅 state_dict 里有键，加载时 strict=False 跳过）。
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

SR = 16000
WIN = 512
HOP = 128
ENC_DIM = WIN // 2 + 1  # 257
FEATURE_DIM = 128
NBAND = 32
SPK_EMB_DIM = 192
NUM_REPEAT = 6
CHUNK_FRAMES = 4  # 32ms 交付口径


def band_widths(enc_dim: int = ENC_DIM) -> list[int]:
    """与 w14 一致：100/200/500/2000Hz 分段，最后一段吸收余量。"""
    bw = [int(np.floor(100 / (SR / 2.0) * enc_dim))] * 15
    bw += [int(np.floor(200 / (SR / 2.0) * enc_dim))] * 10
    bw += [int(np.floor(500 / (SR / 2.0) * enc_dim))] * 5
    bw += [int(np.floor(2000 / (SR / 2.0) * enc_dim))] * 1
    bw.append(enc_dim - int(np.sum(bw)))
    return bw


class PerFrameNorm(nn.Module):
    """逐帧归一（频率维统计），替代实验14 前的 GroupNorm(1,C)。"""

    def __init__(self, num_channels: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):  # (B, C, T)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        w = self.weight.view(1, -1, 1)
        b = self.bias.view(1, -1, 1)
        return (x - mu) / torch.sqrt(var + self.eps) * w + b


class CausalResRNN(nn.Module):
    """单向 ResRNN：时间轴 LSTM，跨 chunk 缓存 (h, c)。"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps
        self.norm = PerFrameNorm(input_size, self.eps)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True,
                           bidirectional=False)
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input, state=None):
        rnn_output, state = self.rnn(
            self.norm(input).transpose(1, 2).contiguous(), state)
        rnn_output = self.proj(rnn_output.reshape(-1, rnn_output.shape[2])).view(
            input.shape[0], input.shape[2], input.shape[1])
        return input + rnn_output.transpose(1, 2).contiguous(), state


class CausalBSNet(nn.Module):
    """band_rnn 沿时间(有状态)；band_comm 沿频带(无状态)。"""

    def __init__(self, in_channel: int, nband: int = NBAND):
        super().__init__()
        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = CausalResRNN(self.feature_dim, self.feature_dim * 2)
        self.band_comm = CausalResRNN(self.feature_dim, self.feature_dim * 2)

    def forward(self, input, states):
        B, N, T = input.shape
        band_output, states["band_rnn"] = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1),
            states.get("band_rnn"))
        band_output = band_output.view(B, self.nband, -1, T)
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(
            B * T, -1, self.nband)
        output, _ = self.band_comm(band_output)
        output = output.view(B, T, -1, self.nband).permute(
            0, 3, 2, 1).contiguous()
        return output.view(B, N, T)


class ConvBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class TFMapBranch(nn.Module):
    """TF-Map-lite：谱模板门控 g + 帧级交叉注意力（E 由注册音频离线算好）。"""

    def __init__(self, nband: int, band_width: list[int], feat_dim: int,
                 enc_dim: int, emb_dim: int = SPK_EMB_DIM):
        super().__init__()
        self.nband = nband
        self.template_conv = nn.Conv1d(nband, nband, 1)
        self.enc = nn.Sequential(ConvBlock1d(enc_dim, emb_dim),
                                 ConvBlock1d(emb_dim, emb_dim),
                                 ConvBlock1d(emb_dim, emb_dim))
        self.mproj = nn.Linear(nband * feat_dim, emb_dim)
        self.lowrank = nn.Linear(emb_dim, 2 * nband)
        self.basis = nn.Parameter(torch.randn(2, nband, feat_dim) * 0.02)
        bm = torch.zeros(enc_dim, nband)
        idx = 0
        for b, bw in enumerate(band_width):
            bm[idx:idx + bw, b] = 1.0 / bw
            idx += bw
        assert idx == enc_dim
        self.register_buffer("band_matrix", bm, persistent=False)


class TFMapCausalBSRNN(nn.Module):
    """实验16 学生（推理用）：整段与分块两条路径，接口与原训练脚本一致。"""

    def __init__(self, margs: dict | None = None):
        super().__init__()
        margs = margs or {}
        self.sr = margs.get("sr", SR)
        self.win = margs.get("win", WIN)
        self.stride = margs.get("stride", HOP)
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = margs.get("feature_dim", FEATURE_DIM)
        self.eps = torch.finfo(torch.float32).eps
        self.spk_emb_dim = margs.get("spk_emb_dim", SPK_EMB_DIM)

        self.band_width = band_widths(self.enc_dim)
        self.nband = len(self.band_width)

        self.BN = nn.ModuleList([
            nn.Sequential(PerFrameNorm(bw * 2, self.eps),
                          nn.Conv1d(bw * 2, self.feature_dim, 1))
            for bw in self.band_width
        ])

        sep_list = nn.ModuleList([nn.Module()])  # 占位：separation[0] 不再前向调用
        for _ in range(margs.get("num_repeat", NUM_REPEAT)):
            sep_list.append(CausalBSNet(self.nband * self.feature_dim, self.nband))
        self.separator = nn.Module()
        self.separator.separation = sep_list

        self.mask = nn.ModuleList([
            nn.Sequential(
                PerFrameNorm(self.feature_dim, self.eps),
                nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                nn.Tanh(),
                nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                nn.Tanh(),
                nn.Conv1d(self.feature_dim * 4, bw * 4, 1))
            for bw in self.band_width
        ])

        self.tfmap = TFMapBranch(self.nband, self.band_width,
                                 self.feature_dim, self.enc_dim)

    # ---------------- 基础算子 ----------------

    def _window(self, device, dtype):
        return torch.hann_window(self.win).to(device).type(dtype)

    def _analyze(self, wav):
        return torch.stft(wav, n_fft=self.win, hop_length=self.stride,
                          window=self._window(wav.device, wav.dtype),
                          return_complex=True)

    def _encode(self, spec, B):
        spec_RI = torch.stack([spec.real, spec.imag], 1)
        feats, mixes = [], []
        idx = 0
        for i, bw in enumerate(self.band_width):
            sb = spec_RI[:, :, idx:idx + bw].contiguous()
            mixes.append(spec[:, idx:idx + bw])
            feats.append(self.BN[i](sb.view(B, bw * 2, -1)))
            idx += bw
        return torch.stack(feats, 1), mixes

    def _mask_synth(self, sep_output, subband_mix_spec, B):
        est = []
        for i, mask_func in enumerate(self.mask):
            out = mask_func(sep_output[:, i]).view(B, 2, 2, self.band_width[i], -1)
            m = out[:, 0] * torch.sigmoid(out[:, 1])
            mr, mi = m[:, 0], m[:, 1]
            mix = subband_mix_spec[i]
            est.append(torch.complex(mix.real * mr - mix.imag * mi,
                                     mix.real * mi + mix.imag * mr))
        return torch.cat(est, 1)

    def _istft(self, est_spec, nsample, device, dtype):
        return torch.istft(est_spec, n_fft=self.win, hop_length=self.stride,
                           window=self._window(device, dtype), length=nsample)

    # ---------------- TF-Map 分支 ----------------

    def _enroll_tfmap(self, enroll, enroll_len=None):
        spec = self._analyze(enroll)
        mag = spec.abs()
        B, _, Ne = mag.shape
        dev = mag.device
        if enroll_len is None:
            valid = torch.full((B,), Ne, dtype=torch.long, device=dev)
        else:
            valid = torch.clamp(enroll_len.to(dev) // self.stride + 1, max=Ne)
        mask = torch.arange(Ne, device=dev).unsqueeze(0) < valid.unsqueeze(1)
        vf = valid.clamp_min(1).to(mag.dtype).view(-1, 1)
        tmag = (mag * mask.unsqueeze(1)).sum(-1) / vf
        band = torch.log1p(tmag @ self.tfmap.band_matrix)
        band = (band - band.mean(-1, keepdim=True)) / \
            (band.std(-1, unbiased=False, keepdim=True) + 1e-5)
        g = torch.sigmoid(self.tfmap.template_conv(band.unsqueeze(-1))).squeeze(-1)
        logm = torch.log(mag.clamp_min(1e-6))
        logm = logm - (logm * mask.unsqueeze(1)).sum(-1, keepdim=True) / vf.unsqueeze(-1)
        E = self.tfmap.enc(logm).transpose(1, 2).contiguous()
        return E, mask, g

    def _modulate(self, subband_feature, E, Emask, g):
        x = subband_feature * g.unsqueeze(-1).unsqueeze(-1)
        B, nb, C, T = x.shape
        Mf = x.reshape(B, nb * C, T).transpose(1, 2)
        M = self.tfmap.mproj(Mf)
        scores = (M @ E.transpose(1, 2)) / math.sqrt(E.shape[-1])
        scores = scores.masked_fill(~Emask.unsqueeze(1), float("-inf"))
        A = torch.softmax(scores, dim=-1)
        c = A @ E
        coeff = self.tfmap.lowrank(c).view(B, T, 2, nb)
        inj = torch.einsum("btnc,nce->btce", coeff, self.tfmap.basis)
        return x + inj.permute(0, 2, 3, 1)

    def _sep(self, subband_feature, states):
        B = subband_feature.shape[0]
        x = subband_feature.reshape(B, self.nband * self.feature_dim, -1)
        seps = self.separator.separation
        for idx in range(1, len(seps)):
            st = states.setdefault(idx, {})
            x = seps[idx](x, st)
        return x.view(B, self.nband, self.feature_dim, -1)

    # ---------------- 前向 ----------------

    def forward(self, input, enroll, enroll_len=None):
        B, nsample = input.shape
        spec = self._analyze(input)
        subband_feature, subband_mix = self._encode(spec, B)
        E, Emask, g = self._enroll_tfmap(enroll, enroll_len)
        feat = self._modulate(subband_feature, E, Emask, g)
        sep_output = self._sep(feat, {})
        est_spec = self._mask_synth(sep_output, subband_mix, B)
        return self._istft(est_spec, nsample, input.device, input.dtype)

    def forward_stream(self, input, enroll, chunk_frames=CHUNK_FRAMES,
                       enroll_len=None):
        B, nsample = input.shape
        spec = self._analyze(input)
        T = spec.shape[-1]
        E, Emask, g = self._enroll_tfmap(enroll, enroll_len)
        states = {}
        est_chunks = []
        for i in range(0, T, chunk_frames):
            spec_c = spec[..., i:i + chunk_frames]
            feat_c, mix_c = self._encode(spec_c, B)
            feat_c = self._modulate(feat_c, E, Emask, g)
            sep_c = self._sep(feat_c, states)
            est_chunks.append(self._mask_synth(sep_c, mix_c, B))
        est_spec = torch.cat(est_chunks, -1)
        return self._istft(est_spec, nsample, input.device, input.dtype)


# ---------------- 权重加载 ----------------

SKIP_PREFIXES = ("spk_model.", "separator.separation.0.")


def load_model(ckpt_path: str, device: str = "cpu") -> TFMapCausalBSRNN:
    model = TFMapCausalBSRNN()
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["models"][0] if isinstance(ck.get("models"), list) else ck["models"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith(SKIP_PREFIXES)]
    bad_unexpected = [k for k in unexpected if not k.startswith(SKIP_PREFIXES)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"权重与复刻结构不匹配: missing={bad_missing[:8]} "
            f"unexpected={bad_unexpected[:8]}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
