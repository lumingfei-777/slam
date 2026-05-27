"""
程序入口 —— 从这里开始运行

运行方式：
  cd stereo_slam
  py -3 main.py

工作流程：
  1. 创建输出目录
  2. 计算左右图目录路径（相对于 main.py 的位置）
  3. 创建 SLAM 实例
  4. 运行主循环

目录结构：
  videos/
  ├── IR_LEFT/      ← 左目红外图像
  ├── IR_RIGHT/     ← 右目红外图像
  └── stereo_slam/  ← 源代码
      └── main.py   ← 从这里开始
"""

import os, sys
from config import OUT_DIR
from slam import SLAM


def main():
    # 确保输出目录存在
    os.makedirs(OUT_DIR, exist_ok=True)

    # 左图目录：相对于输出目录的上级目录的 IR_LEFT
    left_dir = os.path.join(os.path.dirname(OUT_DIR), '..', 'IR_LEFT')
    right_dir = os.path.join(os.path.dirname(OUT_DIR), '..', 'IR_RIGHT')

    print("=" * 60)
    print("  Stereo SLAM - Full Pipeline")
    print("  Algorithm: ORB + SGBM + PnP + BA + Loop Closure")
    print("=" * 60)

    slam = SLAM(left_dir, right_dir)
    slam.run()


if __name__ == '__main__':
    main()
