"""生成应用内置演示素材（16k 单声道 wav + manifest）。

说明：本地只有 data/zh_ft 的 TTS 语料（3 音色 × 3 音高）。dev.csv 里的官方混合
把同一音色的不同音高当不同说话人，且多干扰叠加，对只见过真人语音的模型过难；
这里按“不同音色 + 单干扰 + 0dB”构造固定演示集，enrollment 取同一说话人的**另一条**
语句（不使用混合里的目标段），避免自证。
"""
from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

SR = 16000
SEG = 3.0
# 语料目录不入库：默认取仓库根 data/zh_ft/utt，可用 VF_TTS_UTT_DIR 覆盖
UTT = os.environ.get(
    "VF_TTS_UTT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "zh_ft", "utt"))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "demo")

# (id, 目标音色, 干扰音色, 目标语句, 干扰语句, 注册来源, SIR_dB)
# 组合依据：demo_sweep.py 对 8 音色 × 2 份官方注册样本 × 56 干扰对做全扫，
# 取实测 SI-SNRi 为正且音色/性别有差异的组合；同音色/相近音色失败样本记录在扫描结果中。
ITEMS = [
    ("demo_01", "train_zh-TW-YunJheNeural__pp0", "train_zh-HK-HiuMaanNeural__pp0", "u00", "u01", "enroll_1", 0),
    ("demo_02", "train_zh-TW-YunJheNeural__pp0", "train_zh-HK-HiuGaaiNeural__pp0", "u02", "u03", "enroll_0", 0),
    ("demo_03", "train_zh-TW-YunJheNeural__pp0", "train_zh-TW-HsiaoChenNeural__pp0", "u03", "u02", "enroll_0", 0),
    ("demo_04", "train_zh-HK-HiuMaanNeural__pp0", "train_zh-TW-YunJheNeural__pp0", "u00", "u01", "enroll_1", 0),
    ("demo_05", "train_zh-HK-HiuMaanNeural__pp0", "train_zh-HK-WanLungNeural__pp0", "u04", "u03", "enroll_1", 0),
    ("demo_06", "dev_zh-CN-liaoning-XiaobeiNeural__pp0", "train_zh-HK-WanLungNeural__pp0", "u05", "u04", "enroll_0", 0),
]


def load(name: str) -> np.ndarray:
    p = os.path.join(UTT, name + ".wav")
    x, r = sf.read(p, dtype="float32")
    assert r == SR, (p, r)
    return x[:, 0] if x.ndim > 1 else x


def fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n].copy()
    return np.pad(x, (0, n - len(x)))


def main():
    for sub in ("mix", "enroll", "target"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    n = int(SEG * SR)
    manifest = []
    for mid, tgt_spk, itf_spk, tu, iu, eu, sir in ITEMS:
        tgt = fit(load(f"{tgt_spk}_{tu}"), n)
        itf = fit(load(f"{itf_spk}_{iu}"), n)
        if eu.startswith("enroll_"):
            enr = load(f"enroll_{tgt_spk}_{eu.split('_')[1]}")
        else:
            enr = load(f"{tgt_spk}_{eu}")
        if len(enr) > 5 * SR:
            enr = enr[:5 * SR]
        itf = itf * (np.sqrt(np.mean(tgt ** 2)) / (np.sqrt(np.mean(itf ** 2)) + 1e-9)) * 10 ** (-sir / 20)
        mix = tgt + itf
        peak = float(np.max(np.abs(mix)))
        scale = 0.95 / peak if peak > 0.95 else 1.0
        mix, tgt = mix * scale, tgt * scale
        for name, sig in (("mix", mix), ("target", tgt), ("enroll", enr)):
            sf.write(os.path.join(OUT, name, f"{mid}.wav"), sig.astype(np.float32), SR,
                     subtype="PCM_16")
        manifest.append({"id": mid, "name": f"{tgt_spk.split('__')[0].split('_')[-1]} / 干扰 {itf_spk.split('__')[0].split('_')[-1]}",
                         "target_voice": tgt_spk, "interf_voice": itf_spk, "enroll_src": eu,
                         "sir_db": sir, "seg_s": round(n / SR, 2),
                         "enroll_s": round(len(enr) / SR, 2)})
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(manifest)} demo items → {OUT}")


if __name__ == "__main__":
    main()
