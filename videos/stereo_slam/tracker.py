"""
追踪器模块 —— 帧间特征追踪与位姿估计

这个模块负责三件事：
  1. 提取图像特征（ORB 关键点 + 描述子）
  2. 在帧之间匹配特征（暴力匹配 + 比率测试）
  3. 通过 PnP 求解相机位姿

SLAM 追踪的核心流程：
  前一帧的 3D 地图点 → 投影到当前帧 → 与当前帧 ORB 匹配
  → PnP RANSAC 求解位姿 → 用内点做三角化扩展地图
"""

import cv2
import numpy as np
from config import (
    K_LEFT, D_IR,
    ORB_N_FEATURES, ORB_SCALE_FACTOR, ORB_N_LEVELS,
    MATCH_RATIO_TEST,
    PNP_MIN_MATCHES, PNP_ITERATIONS, PNP_REPROJ_ERROR, PNP_CONFIDENCE,
)


class Tracker:
    """
    追踪器 —— 每帧都会用到的功能都封装在这里。

    设计思路：
      每个 Tracker 对象持有自己的 ORB 提取器，
      这样提取参数在运行时不可变，保证一致性。
    """

    def __init__(self):
        self.orb = cv2.ORB_create(
            nfeatures=ORB_N_FEATURES,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_N_LEVELS,
        )

    def extract(self, image):
        """
        在单张图像上提取 ORB 特征。

        ORB 特征 = 关键点(keypoint) + 描述子(descriptor)
        关键点：位置、方向、尺度
        描述子：256 位二进制向量（用汉明距离衡量相似度）

        返回:
            kp:   关键点列表（每个包含 .pt, .angle, .size 等）
            desc: N×32 的 uint8 矩阵（每行是一个点的 256 位描述子）
        """
        kp, desc = self.orb.detectAndCompute(image, None)
        return kp, desc

    def match_ratio_test(self, desc1, desc2):
        """
        用 Lowe's ratio test 做特征匹配。

        为什么要用比率测试？
          只用最近邻匹配会引入大量误匹配。
          比率测试的思路：对每个特征点，
          找距离最近和次近的两个匹配。
          如果最近的距离 << 次近的距离，
          说明匹配是"独一份"的，可靠。

        数学上：
          good = (best_distance / second_best_distance) < ratio

        参数:
            desc1, desc2: 两帧的 ORB 描述子矩阵

        返回:
            list of cv2.DMatch: 通过测试的匹配
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # k=2 表示对每个点找 2 个最近邻
        raw = matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair  # m = 最近邻, n = 次近邻
                if m.distance < MATCH_RATIO_TEST * n.distance:
                    good.append(m)
        return good

    def estimate_pose_pnp(self, pts_3d, pts_2d):
        """
        用 PnP + RANSAC 估计相机位姿。

        PnP (Perspective-n-Point) 问题：
          给定 N 对 3D-2D 匹配点（3D 是世界坐标，2D 是像素坐标），
          求解相机的旋转 R 和平移 t。

        RANSAC 的作用：
          匹配中有误匹配（外点），RANSAC 反复随机采样，
          找到能支持最多匹配的位姿解，自动剔除外点。

        参数:
            pts_3d: N×3 的世界坐标系 3D 点
            pts_2d: N×2 的像素坐标点

        返回:
            rvec:        旋转向量（Rodrigues 表示，3×1）
            tvec:        平移向量（3×1）
            inlier_mask: 布尔数组，标记哪些点是内点
        """
        if len(pts_3d) < PNP_MIN_MATCHES:
            return None, None, []

        pts_3d = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 3)
        pts_2d = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 2)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K_LEFT, D_IR,
            iterationsCount=PNP_ITERATIONS,
            reprojectionError=PNP_REPROJ_ERROR,
            confidence=PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < PNP_MIN_MATCHES:
            return None, None, []

        inlier_mask = np.zeros(len(pts_3d), dtype=bool)
        inlier_mask[inliers.ravel()] = True
        return rvec, tvec, inlier_mask

    @staticmethod
    def rvec_tvec_to_matrix(rvec, tvec):
        """
        将旋转向量 + 平移向量 转换成 4×4 变换矩阵。

        Rodrigues 公式：
          旋转向量 θ*u（θ=角度, u=单位轴）
          转成旋转矩阵 R 用 cv2.Rodrigues()

        结果 T = [R  t]
                 [0  1]
        """
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        return T

    @staticmethod
    def matrix_to_rvec_tvec(T):
        """4×4 变换矩阵 → 旋转向量 + 平移向量（逆操作）。"""
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape(3, 1)
        return rvec, tvec
