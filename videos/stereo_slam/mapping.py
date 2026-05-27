"""
地图建模块 —— SLAM 的核心数据结构

世界的两种表示：
  1. 稀疏地图：由 MapPoint（地图点）构成，仅保存特征点位置
  2. 关键帧：精选的"有信息量"的帧，保存位姿和观测

关键概念 —— 什么是"地图"？
  地图 = 所有 MapPoint 的集合 + 所有 KeyFrame 的集合
  MapPoint 位置是 3D 坐标 (x, y, z)
  KeyFrame 的 pose 是从相机坐标系到世界坐标系的变换矩阵 T_wc

  公式：X_world = T_wc * X_camera
  其中 T_wc = [R  t]
              [0  1]
"""

import cv2
import numpy as np
from config import (
    KF_MIN_TRANSLATION, KF_MIN_ROTATION, KF_MIN_TRACKED_RATIO,
)


class MapPoint:
    """
    地图点 —— SLAM 地图中的最小单元。

    一个 MapPoint 对应世界中的一个 3D 点，
    它由双目三角化（triangulation）恢复出来，
    被多个关键帧观测到。

    __slots__ 的作用：
    限制实例只能有这些属性，减少内存占用。
    对于可能有数万个点的地图，这个优化很关键。

    属性:
        id:          全局唯一 ID
        position:    世界坐标系下的 3D 位置 (3,) 向量
        descriptor:  首次观察到这个点时的 ORB 描述子
                     用于后续特征匹配
        first_kf_id: 第一次观测到该点的关键帧 ID
    """
    __slots__ = ('id', 'position', 'descriptor', 'first_kf_id')

    _next_id = 0  # 类变量，用于生成自增 ID

    def __init__(self, position, descriptor=None, first_kf_id=-1):
        self.id = MapPoint._next_id
        MapPoint._next_id += 1
        self.position = np.asarray(position, dtype=np.float64).ravel()
        self.descriptor = descriptor
        self.first_kf_id = first_kf_id


class KeyFrame:
    """
    关键帧 —— 被选中加入地图的帧。

    为什么不是每一帧都加入地图？
      相邻帧之间变化很小，信息冗余。
      只保留"关键"的帧可以降低计算量、减少地图冗余。

    关键帧的判定条件（见 is_keyframe）：
      1. 相机移动足够大
      2. 相机旋转足够大
      3. 追踪到的特征太少

    属性:
        id:         全局唯一 ID
        pose:       4x4 变换矩阵 T_wc（世界 ← 相机）
        pts_2d:     该帧中所有 MapPoint 的 2D 像素坐标
        point_ids:  与 pts_2d 一一对应的 MapPoint ID 列表
        timestamp:  设备时间戳
    """
    __slots__ = ('id', 'pose', 'pts_2d', 'point_ids', 'timestamp')

    _next_id = 0

    def __init__(self, pose, pts_2d, point_ids, timestamp=0):
        self.id = KeyFrame._next_id
        KeyFrame._next_id += 1
        self.pose = pose.copy()
        self.pts_2d = pts_2d          # list of (x, y) tuples
        self.point_ids = point_ids    # list of MapPoint IDs
        self.timestamp = timestamp


class Map:
    """
    地图容器 —— 管理所有 MapPoint 和 KeyFrame。

    结构很简单：
      - points:    dict {id → MapPoint}
      - keyframes: list [KeyFrame]（按顺序排列）
    """

    def __init__(self):
        self.points = {}       # id -> MapPoint
        self.keyframes = []    # list of KeyFrame
        # 重置 ID 计数器（方便多次运行测试）
        MapPoint._next_id = 0
        KeyFrame._next_id = 0

    def add_point(self, position, descriptor=None, first_kf_id=-1):
        pt = MapPoint(position, descriptor, first_kf_id)
        self.points[pt.id] = pt
        return pt.id

    def add_keyframe(self, pose, pts_2d, point_ids, timestamp=0):
        kf = KeyFrame(pose, pts_2d, point_ids, timestamp)
        self.keyframes.append(kf)
        return kf.id

    def get_trajectory(self):
        """提取所有关键帧的平移向量，构成轨迹。"""
        if not self.keyframes:
            return np.zeros((0, 3))
        traj = np.array([kf.pose[:3, 3] for kf in self.keyframes])
        return traj


def triangulate_stereo(kp_left, kp_right, matches, K_l, K_r, R, t):
    """
    双目三角化 —— 从左右匹配点恢复 3D 坐标。

    数学原理：
      左相机投影矩阵 P_l = K_l * [I | 0]
      右相机投影矩阵 P_r = K_r * [R | t]

      对于一对匹配点 (u_left, v_left) 和 (u_right, v_right)：
      u = P[0] * X / P[2] * X
      v = P[1] * X / P[2] * X

      每个点提供两个方程，4 个方程求解 3 个未知数 (X, Y, Z)
      用 SVD 解超定方程组 → 这就是 cv2.triangulatePoints 做的事

    参数:
        kp_left:  左图关键点列表
        kp_right: 右图关键点列表
        matches:  左右图之间的匹配（DMatch 对象）
        K_l, K_r: 左右相机内参矩阵
        R, t:     右目到左目的旋转和平移

    返回:
        pts_3d:   N×3 的 3D 点坐标（在左相机坐标系下）
        pts_l:    N×2 的左图匹配点像素坐标

    注意：
      返回的 3D 点在"左相机坐标系"下（以左目为原点），
      后面在 slam.py 中会通过 pose 变换到世界坐标系。
    """
    if len(matches) == 0:
        return np.zeros((0, 3)), []

    # 提取左右关键点的像素坐标
    pts_l = np.float64([kp_left[m.queryIdx].pt for m in matches])
    pts_r = np.float64([kp_right[m.trainIdx].pt for m in matches])

    # cv2.triangulatePoints 返回齐次坐标 (4×N)
    pts_4d = cv2.triangulatePoints(
        K_l @ np.hstack([np.eye(3), np.zeros((3, 1))]),   # 左目投影矩阵 P1
        K_r @ np.hstack([R, t.reshape(3, 1)]),            # 右目投影矩阵 P2
        pts_l.T, pts_r.T,
    )
    # 齐次坐标转欧氏坐标：除以第 4 维
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d, pts_l


def is_keyframe(prev_pose, curr_pose, tracked_ratio):
    """
    判断当前帧是否应该成为关键帧。

    逻辑：
      1. 前一帧为空（第一帧）→ 一定是关键帧
      2. 计算两帧之间的相对运动：
         - 平移量 delta_trans
         - 旋转角度 delta_angle
      3. 满足以下任一条件就选为关键帧：
         - 平移 > 阈值（比如 20cm）
         - 旋转 > 阈值（比如 8.6°）
         - 追踪到的特征太少（需要新的特征点）

    参数:
        prev_pose:    前一帧（或上一关键帧）的位姿 4×4
        curr_pose:    当前帧位姿 4×4
        tracked_ratio: 当前帧中能匹配到上一帧的比例

    返回:
        bool: 是否为关键帧
    """
    if prev_pose is None:
        return True

    # 相对变换 = 上一帧的逆 × 当前帧
    # 这给出了"从上一帧到当前帧"的变换
    delta = np.linalg.inv(prev_pose) @ curr_pose

    # 平移量（毫米）
    trans = np.linalg.norm(delta[:3, 3])

    # 旋转角（弧度）
    # 从旋转矩阵提取角度的公式：theta = arccos((trace(R) - 1) / 2)
    angle = np.arccos(
        np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)
    )

    return (trans > KF_MIN_TRANSLATION or
            angle > KF_MIN_ROTATION or
            tracked_ratio < KF_MIN_TRACKED_RATIO)
