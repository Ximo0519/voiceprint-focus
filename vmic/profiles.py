"""声纹档案：与 Web 服务同一套磁盘格式，两边可共用。

<dir>/registrations/<id>.npz  (E, g)
<dir>/registrations/<id>.json (元数据)
默认目录 <app>/backend/enrollments/registrations，可用 VF_ENROLL_DIR 覆盖。
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

import numpy as np

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(_APP, "backend", "enrollments")


def base_dir() -> str:
    return os.environ.get("VF_ENROLL_DIR") or DEFAULT_DIR


def reg_dir() -> str:
    return os.path.join(base_dir(), "registrations")


def save(E: np.ndarray, g: np.ndarray, meta: dict) -> str:
    d = reg_dir()
    os.makedirs(d, exist_ok=True)
    rid = "reg_%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), os.urandom(3).hex())
    np.savez(os.path.join(d, rid + ".npz"), E=E, g=g)
    with open(os.path.join(d, rid + ".json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "id": rid}, f, ensure_ascii=False, indent=2)
    return rid


def load(spec: str):
    """按 id（reg_xxx）或 .npz 路径读取 (E, g, meta)。"""
    if not spec:
        return None
    if os.path.isfile(spec):
        p = spec
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec):
            return None
        p = os.path.join(reg_dir(), spec + ".npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    meta = {}
    jp = os.path.splitext(p)[0] + ".json"
    if os.path.exists(jp):
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return d["E"].astype(np.float32), d["g"].astype(np.float32), meta


def list_all(limit: int = 100) -> list[dict]:
    out = []
    for jp in glob.glob(os.path.join(reg_dir(), "*.json")):
        try:
            with open(jp, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        m.setdefault("id", os.path.splitext(os.path.basename(jp))[0])
        out.append(m)
    out.sort(key=lambda m: m.get("ts", ""), reverse=True)
    return out[:limit]
