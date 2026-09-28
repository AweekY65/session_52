import os
import sys

# 保证从仓库根目录以外运行 pytest 时也能 import robot_kinematics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
