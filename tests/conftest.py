import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# backend 内部使用平铺 import（import dsp / import engine），与 server.py 的启动方式一致
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
