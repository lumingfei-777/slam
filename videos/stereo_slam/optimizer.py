"""
优化器模块 —— 减少累积误差

SLAM 的"一致性"维护者，负责两个关键任务：

  1. 局部 BA（Bundle Adjustment）
     最小化重投影误差：让地图点的 3D 位置和相机的位姿
     能"最好地解释"观测到的 2D 像素坐标。

  2. 回环检测与修正
     发现相机回到之前的位置 → 计算漂移 → 纠正所有中间帧的位姿。

为什么需要优化？
  视觉 SLAM 的每一步都有噪声：
  - ORB 特征提取有像素误差
  - PnP 求解有数值误差
  - 三角化有深度不确定性
  这些误差会随帧数累积，不优化的话轨迹会"飘走"。
"""

import numpy as np
import cv2
from scipy.optimize import least_squares

from config import (
    K_LEFT, BA_WINDOW, BA_MAX_ITER,
    KF_MIN_TRANSLATION, LC_MIN_MATCHES, LC_SEARCH_DIST,
)


def local_bundle_adjustment(map_instance, kf_window=None):
    """
    局部 BA —— 优化最近的几个关键帧和它们看到的地图点。

    什么叫"重投影误差"？
      1. 有一个 3D 地图点 X，世界坐标已知
      2. 根据当前估计的相机位姿 T，把 X 投影到图像上
      3. 得到重投影坐标 (u', v')
      4. 和实际观测到的坐标 (u, v) 比较
      5. 误差 = (u' - u)^2 + (v' - v)^2

    BA 就是最小化所有帧所有点的重投影误差之和。

    参数:
        map_instance: Map 对象
        kf_window:    优化最近的 N 个关键帧（默认用配置值）

    返回:
        bool: 是否优化成功
    """
    if kf_window is None:
        kf_window = min(BA_WINDOW, 3)
    kfs = map_instance.keyframes
    if len(kfs) < 2:
        return False

    window = kfs[-kf_window:]  # 取最近的 N 帧

    # 收集这些关键帧能看到的所有地图点 ID
    visible_pt_ids = set()#创建空集合
    for kf in window:
        visible_pt_ids.update(kf.point_ids)#确保每个点只出现一次

    # 筛选有效点：排除太近(<10cm)或太远(>50m)的
    valid_pts = []
    for pid in visible_pt_ids:
        if pid not in map_instance.points:
            continue
        pos = map_instance.points[pid].position
        dist = np.linalg.norm(pos - map_instance.keyframes[-1].pose[:3, 3])
        #计算到相机的距离，还是有问题，计算的是到最后一帧的距离，应该是到每一帧的距离，后续优化再改吧
        if 100 < dist < 50000:
            valid_pts.append(pid)

    if len(valid_pts) < 20 or len(window) < 2:
        return False

    # 构建观测列表
    # obs[i] = (相机索引, 点索引, u, v)
    obs = []
    for ci, kf in enumerate(window):
        for pi, pid in enumerate(kf.point_ids):
            if pid in valid_pts:
                obs.append((ci, valid_pts.index(pid),
                           kf.pts_2d[pi][0], kf.pts_2d[pi][1]))

    if len(obs) < 30:
        return False

    # 固定第一个相机的位姿（固定参考系，避免退化）
    fixed = [True] + [False] * (len(window) - 1)
    n_cams = len(window)
    n_pts = len(valid_pts)

    # 构建优化变量向量 x
    # x = [cam0_r, cam0_t, cam1_r, cam1_t, ..., pt0_xyz, pt1_xyz, ...]
    # 每个相机 6 个参数（3 旋转 + 3 平移）
    # 每个点 3 个参数（x, y, z）
    cam_params = []
    for ci, kf in enumerate(window):
        r, t = _mat_to_rvec_tvec(kf.pose)
        cam_params.extend(r.ravel().tolist() + t.ravel().tolist())
    pt_params = []
    for pid in valid_pts:
        pt_params.extend(map_instance.points[pid].position.tolist())
    x0 = np.array(cam_params + pt_params, dtype=np.float64)

    cx, cy, fx, fy = K_LEFT[0, 2], K_LEFT[1, 2], K_LEFT[0, 0], K_LEFT[1, 1]

    def fun(x):
        """
        残差函数 —— scipy.least_squares 会优化这个函数的输出。

        对每个观测 (ci, pi, u_m, v_m)：
          1. 提取当前估计的相机位姿 (r, t)
          2. 提取当前估计的 3D 点位置 X
          3. 用投影方程计算重投影坐标 (u', v')
          4. 残差 = (u' - u, v' - v)

        投影方程：
          X_cam = R @ X + t        (世界 → 相机坐标系)
          u' = fx * X_cam_x / X_cam_z + cx
          v' = fy * X_cam_y / X_cam_z + cy
        """
        resid = np.zeros(len(obs) * 2)
        for i, (ci, pi, u_m, v_m) in enumerate(obs):
            r = x[ci*6:ci*6+3]
            t = x[ci*6+3:ci*6+6]
            X = x[n_cams*6 + pi*3: n_cams*6 + pi*3 + 3]
            R, _ = cv2.Rodrigues(r)
            Xc = R @ X + t
            if Xc[2] < 10:
                # 点在相机后面，给一个很大惩罚
                resid[i*2] = 1e4
                resid[i*2+1] = 1e4
            else:
                resid[i*2]   = fx * Xc[0] / Xc[2] + cx - u_m
                resid[i*2+1] = fy * Xc[1] / Xc[2] + cy - v_m
        return resid

    try:
        result = least_squares(fun, x0, method='trf',
                               max_nfev=BA_MAX_ITER,
                               ftol=1e-2, xtol=1e-2)
    except Exception:
        return False

    if not result.success:
        return False

    # 将优化结果写回
    x_opt = result.x
    for ci, kf in enumerate(window):
        if fixed[ci]:
            continue
        r = x_opt[ci*6:ci*6+3]
        t = x_opt[ci*6+3:ci*6+6].reshape(3, 1)
        kf.pose = _rvec_tvec_to_mat(r, t)

    for li, pid in enumerate(valid_pts):
        pos = x_opt[n_cams*6 + li*3: n_cams*6 + li*3 + 3]
        map_instance.points[pid].position = pos

    return True


def detect_loop_closure(map_instance, current_kf, all_kfs):
    """
    检测回环 —— 当前关键帧是否到了之前去过的地方？

    策略（简化版）：
      遍历所有非最近的关键帧，找与当前位姿距离最近的。
      如果距离 < 阈值，认为可能是回环。

    真实 SLAM 系统的回环检测更复杂：
      - DBoW2 词袋模型做图像检索
      - 几何验证（检查匹配是否一致）
      - 时序一致性（连续多帧确认）

    参数:
        map_instance: Map 对象（这里只用到 keyframes）
        current_kf:  当前关键帧
        all_kfs:     所有关键帧列表

    返回:
        loop_kf: 候选的回环关键帧，或 None
    """
    if len(all_kfs) < 10:
        return None
    curr_pos = current_kf.pose[:3, 3]
    best_dist = float('inf')
    best_kf = None
    # all_kfs[:-LC_SEARCH_DIST] 跳过最近的几帧，避免把自己检测成回环
    for kf in all_kfs[:-LC_SEARCH_DIST]:
        dist = np.linalg.norm(kf.pose[:3, 3] - curr_pos)
        if dist < KF_MIN_TRANSLATION * 2 and dist < best_dist:
            best_dist = dist
            best_kf = kf
    return best_kf


def correct_loop(map_instance, loop_kf, current_kf):
    """
    回环修正 —— 发现回环后纠正累积漂移。

    核心思路：
      1. 找到两个关键帧共同观测到的地图点
      2. 在 loop_kf 的相机坐标系下，用 Umeyama 方法
         对齐这些点的 3D 坐标，计算"应该有的"相对位姿
      3. 和"当前估计的"相对位姿比较 → 算出漂移
      4. 把漂移均匀分摊到中间的所有关键帧

    Umeyama 对齐（3D-3D 配准）：
      给定两组 3D 点 {P_i} 和 {Q_i}，求 R 和 t 使得
      Q_i ≈ R * P_i + t
      解：对协方差矩阵 H 做 SVD 分解

    为什么用 Rodrigues 做插值？
      旋转矩阵不能直接线性插值（结果不是有效旋转矩阵）。
      用旋转向量（轴角表示）做线性插值相当于对角度做 SLERP。
    """
    # ── 1. 找共同地图点 ──
    common_ids = [pid for pid in current_kf.point_ids
                  if pid in loop_kf.point_ids and pid in map_instance.points]

    if len(common_ids) < LC_MIN_MATCHES:
        return False

    world_pts = np.array([map_instance.points[pid].position for pid in common_ids])

    # ── 2. 将共同点转换到两个 KF 各自的相机坐标系下 ──
    T_li = np.linalg.inv(loop_kf.pose)
    pts_loop = (T_li[:3, :3] @ world_pts.T + T_li[:3, 3:4]).T

    T_ci = np.linalg.inv(current_kf.pose)
    pts_curr = (T_ci[:3, :3] @ world_pts.T + T_ci[:3, 3:4]).T

    # ── 3. Umeyama 3D-3D 对齐 ──
    #   求 R_rel, t_rel 使得 pts_curr ≈ R_rel * pts_loop + t_rel
    #   这个变换就是"从 loop KF 到 current KF 应有的相对位姿"
    c_loop = pts_loop.mean(axis=0)
    c_curr = pts_curr.mean(axis=0)
    H = (pts_loop - c_loop).T @ (pts_curr - c_curr)
    U, _, Vt = np.linalg.svd(H)
    R_rel = Vt.T @ U.T
    # 保证是旋转矩阵（行列式 = +1）
    if np.linalg.det(R_rel) < 0:
        Vt[-1] *= -1
        R_rel = Vt.T @ U.T
    t_rel = c_curr - R_rel @ c_loop

    T_rel = np.eye(4, dtype=np.float64)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = t_rel

    # ── 4. 计算漂移 ──
    #   T_expected: 如果回环一致，current KF 应有的位姿
    T_expected = loop_kf.pose @ T_rel
    #   T_drift: 从"当前估计的位姿"到"应有的位姿"的变换
    T_drift = np.linalg.inv(current_kf.pose) @ T_expected

    drift_trans = np.linalg.norm(T_drift[:3, 3])
    drift_rot = np.arccos(
        np.clip((np.trace(T_drift[:3, :3]) - 1) / 2, -1, 1)
    )

    if drift_trans < 10 and drift_rot < 0.02:
        return False  # 漂移太小，没必要修正

    # ── 5. 将漂移分摊到中间关键帧 ──
    loop_idx = map_instance.keyframes.index(loop_kf)
    curr_idx = map_instance.keyframes.index(current_kf)
    n = curr_idx - loop_idx

    if n < 2:
        return False

    # 将 T_drift 拆成轴角 + 平移，方便做线性插值
    r_drift, _ = cv2.Rodrigues(T_drift[:3, :3])
    t_drift = T_drift[:3, 3]

    for i in range(1, n + 1):
        alpha = i / n  # 0 → 1 线性增加
        kf = map_instance.keyframes[loop_idx + i]
        # 线性插值：旋转用 Rodrigues 轴角，平移直接线性
        r_alpha = (r_drift * alpha).ravel()
        t_alpha = t_drift * alpha
        R_alpha, _ = cv2.Rodrigues(r_alpha)
        T_corr = np.eye(4, dtype=np.float64)
        T_corr[:3, :3] = R_alpha
        T_corr[:3, 3] = t_alpha
        # 左乘修正量
        kf.pose = T_corr @ kf.pose

    # 同时更新共同地图点的世界坐标
    for pid in common_ids:
        pt = map_instance.points[pid]
        pt.position = T_drift[:3, :3] @ pt.position + T_drift[:3, 3]

    print(f"      Drift corrected: {drift_trans:.0f}mm, "
          f"{np.degrees(drift_rot):.1f}deg, "
          f"{len(common_ids)} common points, {n} KFs adjusted")
    return True


# ═══════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════


def _mat_to_rvec_tvec(T):
    """4×4 变换矩阵 → 旋转向量 + 平移向量"""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    return rvec, tvec


def _rvec_tvec_to_mat(rvec, tvec):
    """旋转向量 + 平移向量 → 4×4 变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T
