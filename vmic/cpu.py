"""混合架构 CPU（Intel P/E 核）的线程绑定：把推理线程钉在性能核上。

为什么需要：i7-14650HX 这类混合 CPU 上，普通 Python 线程会被调度到 E 核，
同一份推理单块耗时在 21ms 与 38ms 之间跳变（32ms 的块预算经不起这种退化）。
优先级只能影响调度倾向，实测不稳定；直接按 Windows 的 EfficiencyClass 取
「性能最高一档」的逻辑核做亲和性绑定是确定性的。

非 Windows、单档 CPU、或任何调用失败时返回 None，调用方保持原样。
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

RELATION_PROCESSOR_CORE = 0
ERROR_INSUFFICIENT_BUFFER = 122


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t), ("Group", wintypes.WORD),
                ("Reserved", wintypes.WORD * 3)]


def _records():
    """解析 GetLogicalProcessorInformationEx(RelationProcessorCore) 的原始缓冲。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetLogicalProcessorInformationEx.argtypes = [
        wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    k32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    length = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None,
                                         ctypes.byref(length))
    if length.value == 0:
        return None, k32
    buf = ctypes.create_string_buffer(length.value)
    if not k32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf,
                                                ctypes.byref(length)):
        return None, k32
    raw = bytes(buf[:length.value])
    out, off = [], 0
    while off + 32 <= len(raw):
        rel = int.from_bytes(raw[off:off + 4], "little")
        size = int.from_bytes(raw[off + 4:off + 8], "little")
        if size <= 0:
            break
        if rel == RELATION_PROCESSOR_CORE:
            eff = raw[off + 9]
            ngroups = int.from_bytes(raw[off + 30:off + 32], "little")
            for gi in range(max(1, ngroups)):
                ga = GROUP_AFFINITY.from_buffer_copy(raw, off + 32 + gi * 16)
                out.append({"eff": eff, "mask": int(ga.Mask), "group": int(ga.Group)})
        off += size
    return out, k32


def p_core_affinity() -> dict | None:
    """返回 {"mask": int, "group": int, "eff": int, "n": int}；不可用时 None。"""
    if os.name != "nt":
        return None
    try:
        recs, _ = _records()
        if not recs:
            return None
        effs = sorted({r["eff"] for r in recs})
        if len(effs) < 2:
            return None                     # 单档 CPU，没有 P/E 之分
        top = effs[-1]
        best = [r for r in recs if r["eff"] == top]
        mask = 0
        for r in best:
            mask |= r["mask"]
        if not mask:
            return None
        n = bin(mask).count("1")
        return {"mask": mask, "group": best[0]["group"], "eff": top, "n": n}
    except Exception:  # noqa: BLE001
        return None


def bind_current_thread(info: dict | None) -> bool:
    """把当前线程绑到 info 给出的逻辑核集合；失败返回 False。"""
    if os.name != "nt" or not info:
        return False
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentThread.restype = wintypes.HANDLE
        k32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetThreadAffinityMask.restype = ctypes.c_size_t
        return bool(k32.SetThreadAffinityMask(k32.GetCurrentThread(), info["mask"]))
    except Exception:  # noqa: BLE001
        return False
