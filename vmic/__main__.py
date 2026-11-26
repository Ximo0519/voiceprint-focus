"""python -m vmic 入口：把 backend 加入 sys.path 后交给 cli。"""
import os
import sys

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_APP, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from vmic.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
