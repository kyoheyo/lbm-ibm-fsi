"""lbm Python 前/后处理测试的 pytest 配置。"""
import sys
from pathlib import Path

# 将 lbm_pre / lbm_post 包加入搜索路径，无需安装即可直接导入
sys.path.insert(0, str(Path(__file__).parent.parent))
