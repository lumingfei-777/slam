"""
数据集加载模块

功能：
  从 IR_LEFT/ 和 IR_RIGHT/ 两个目录中读取成对的左右图像。
  摄像头以固定频率拍摄，左右图像通过时间戳匹配成对。

文件名格式说明：
  011710_0000_d1776928462281910_s1776928462291708_IR_LEFT_Y8_848x480.png
  ├──序列号
  │   └──帧内序号
  │       └──设备时间戳（我们用它做匹配）
  │               └──系统时间戳
  │                       └──左右标记
  │                               └──像素格式和分辨率

为什么不用 OpenCV 直接读取？
  OpenCV 的 cv2.imread() 不支持中文路径。
  在中国 Windows 系统上，用户名可能含中文字符，
  所以用文件二进制读取 → numpy数组 → cv2.imdecode() 的方式绕过。
"""

import os
import re
import cv2
import numpy as np
from config import WIDTH, HEIGHT, OUT_DIR

# ── 文件名解析正则 ──
# 捕获三部分：序列号、设备时间戳、左/右标记
PATTERN = r'(\d+)_\d+_d(\d+)_s\d+_IR_(LEFT|RIGHT)_Y8_\d+x\d+\.png'


def _parse_timestamp(filename):
    """
    从文件名中提取设备时间戳和左右标记。

    返回:
        (设备时间戳, "LEFT" 或 "RIGHT")
        如果文件名不匹配返回 (None, None)
    """
    match = re.search(PATTERN, filename)
    if match:
        return int(match.group(2)), match.group(3)
    return None, None


def _imread_unicode(path):
    """
    支持 Unicode 路径的图片读取函数。

    为什么不用 cv2.imread():
        cv2.imread() 内部调用 C++ 标准库的 fopen()，
        在 Windows 上无法正确处理包含中文的路径。

    替代方案:
        1. 用 Python 的 open() 以二进制模式读取文件
        2. 将字节数据解码为 numpy uint8 数组
        3. 用 cv2.imdecode() 解析图像格式
    """
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _imsave_unicode(path, img):
    """同上，支持 Unicode 路径的图片保存。"""
    success, buf = cv2.imencode('.png', img)
    if not success:
        return False
    with open(path, 'wb') as f:
        f.write(buf.tobytes())
    return True


def load_stereo_pairs(left_dir, right_dir, verbose=True):
    """
    加载双目图像对 —— SLAM 管线的第一步。

    流程:
        1. 扫描左、右两个目录中的所有 PNG 文件
        2. 从文件名中提取设备时间戳
        3. 以时间戳为 key，找到左右对应的配对
        4. 加载图像并统一尺寸

    参数:
        left_dir:  左目图像目录路径
        right_dir: 右目图像目录路径
        verbose:   是否打印统计信息

    返回:
        list of (时间戳, 左图, 右图)

    数据结构说明:
        left_map = {时间戳: 文件路径}   # 左目索引
        right_map = {时间戳: 文件路径}  # 右目索引
        common = left_map.keys() & right_map.keys()  # 交集 = 成功配对的帧
    """
    left_dir = os.path.abspath(left_dir)
    right_dir = os.path.abspath(right_dir)

    # 用字典建立 时间戳 → 文件路径 的映射
    left_map: dict[int, str] = {}
    right_map: dict[int, str] = {}

    # 遍历左右两个目录
    for d, side_map in [(left_dir, left_map), (right_dir, right_map)]:
        if not os.path.isdir(d):
            print(f"Warning: directory not found: {d}")
            continue
        for fname in os.listdir(d):
            if not fname.endswith('.png'):
                continue
            ts, side = _parse_timestamp(fname)
            if ts is None:
                continue
            if side == 'LEFT':
                left_map[ts] = os.path.join(d, fname)
            elif side == 'RIGHT':
                right_map[ts] = os.path.join(d, fname)

    # 取左右时间戳的交集，升序排列
    common = sorted(set(left_map.keys()) & set(right_map.keys()))
    if verbose:
        print(f"LEFT: {len(left_map)} files, RIGHT: {len(right_map)} files, "
              f"matched: {len(common)} pairs")

    pairs = []
    for ts in common:
        left_img = _imread_unicode(left_map[ts])
        right_img = _imread_unicode(right_map[ts])
        if left_img is None:
            print(f"Warning: failed to load left image: {left_map[ts]}")
            continue
        if right_img is None:
            print(f"Warning: failed to load right image: {right_map[ts]}")
            continue
        # 统一尺寸（某些帧分辨率可能不一致）
        if left_img.shape[:2] != (HEIGHT, WIDTH):
            left_img = cv2.resize(left_img, (WIDTH, HEIGHT))
        if right_img.shape[:2] != (HEIGHT, WIDTH):
            right_img = cv2.resize(right_img, (WIDTH, HEIGHT))
        pairs.append((ts, left_img, right_img))

    if verbose:
        print(f"Successfully loaded {len(pairs)} stereo pairs")
    return pairs
