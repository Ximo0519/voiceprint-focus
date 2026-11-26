"""运行中控制：Windows 全局热键（默认 Ctrl+Alt+D），失败时退回控制台按键。

只做一件事：回调一个 toggle。推理状态由 pipeline 在 worker 线程里安全切换。
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

VK = {"D": 0x44, "M": 0x4D, "F8": 0x77, "F9": 0x78}


class Hotkey(threading.Thread):
    """在独立线程注册并等待 WM_HOTKEY；stop() 用 WM_QUIT 唤醒消息循环。"""

    def __init__(self, mods: int, vk: int, callback, name: str):
        super().__init__(daemon=True, name="vmic-hotkey")
        self.mods = mods
        self.vk = vk
        self.callback = callback
        self.hotkey_name = name
        self.ok = False
        self.error = ""
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._tid = 0
        self._ready = threading.Event()
        self._stop = False

    def run(self):
        self._tid = int(self._kernel32.GetCurrentThreadId())
        try:
            if not self._user32.RegisterHotKey(None, 1, self.mods | MOD_NOREPEAT, self.vk):
                self.error = "RegisterHotKey 失败（可能已被其他程序占用）"
                self._ready.set()
                return
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self._ready.set()
            return
        self.ok = True
        self._ready.set()
        msg = wintypes.MSG()
        while self._user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY:
                try:
                    self.callback()
                except Exception:  # noqa: BLE001
                    pass
            if self._stop:
                break
        self._user32.UnregisterHotKey(None, 1)

    def wait_ready(self, timeout: float = 2.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self):
        self._stop = True
        if self._tid:
            try:
                self._user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
            except Exception:  # noqa: BLE001
                pass


class ConsoleKey(threading.Thread):
    """热键不可用时退回控制台按键：按 d 切换，q 退出。"""

    def __init__(self, on_toggle, on_quit):
        super().__init__(daemon=True, name="vmic-keys")
        self.on_toggle = on_toggle
        self.on_quit = on_quit
        self._stop = False

    def run(self):
        try:
            import msvcrt
        except ImportError:
            return
        while not self._stop:
            if msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if ch == "d":
                    self.on_toggle()
                elif ch == "q":
                    self.on_quit()
                    return
            time.sleep(0.05)

    def stop(self):
        self._stop = True


def start_hotkey(on_toggle, on_quit, hotkey: str = "ctrl+alt+d") -> tuple:
    """返回 (控制器, 提示文案)。控制器有 stop()。"""
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    mods, vk_name = 0, None
    for p in parts:
        if p in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif p == "alt":
            mods |= MOD_ALT
        elif p == "shift":
            mods |= MOD_SHIFT
        else:
            vk_name = p.upper()
    vk = VK.get(vk_name or "D")
    if vk is None or mods == 0:
        return None, f"热键「{hotkey}」无法解析，改用控制台按键（d 切换 / q 退出）"
    hk = Hotkey(mods, vk, on_toggle, hotkey)
    hk.start()
    hk.wait_ready()
    if hk.ok:
        return hk, f"降噪开关热键：{hotkey.upper()}（全局，切出窗口也生效）"
    key = ConsoleKey(on_toggle, on_quit) if sys.stdin is not None else None
    if key is not None:
        key.start()
    return key, f"{hk.error}，改用控制台按键（d 切换 / q 退出，需保持窗口焦点）"
