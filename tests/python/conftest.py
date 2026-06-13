"""测试配置文件 —— 将 python/ 目录加入 sys.path。

Test configuration for local Python prototypes.
确保 ``gnss_imu`` 包在测试时可直接导入，无需设置 PYTHONPATH。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 项目根目录 (Project1_GNSS_IMU/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Python 源码目录
PYTHON_DIR = PROJECT_ROOT / "python"

# 将 python/ 注入路径，使 ``from gnss_imu import ...`` 可用
sys.path.insert(0, str(PYTHON_DIR))
