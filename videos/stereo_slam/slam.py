"""
SLAM 主模块 —— 整个管线的组织者

这里是所有模块的"胶水"代码：
  dataset.py   → 加载左右图像对
  tracker.py   → 提取 + 匹配 + PnP
  mapping.py   → 三角化 + 关键帧判定
  optimizer.py → BA + 回环修正

每一帧的处理流程（_process_frame）：
  1. CLAHE 增强（红外图像对比度低，先做直方图均衡）
  2. ORB 提取左右图特征
  3. 左右图特征匹配 → 双目三角化 → 恢复 3D 点
  4. (第一帧特殊处理：初始化地图)
  5. 当前帧 vs 上一帧特征匹配 → PnP 求解位姿
  6. 新三角化的 3D 点加入地图
  7. 判断是否需要关键帧
  8. 如果需要：加入关键帧 → 局部 BA → 回环检测

坐标系：
  世界坐标系 = 第一帧的相机坐标系
  后面每帧的位姿都是相对于世界坐标系的
"""

import cv2
import numpy as np
import os
from config import OUT_DIR, K_LEFT, K_RIGHT, R_STEREO, T_STEREO
from dataset import load_stereo_pairs
from tracker import Tracker
from mapping import Map, triangulate_stereo, is_keyframe
from optimizer import local_bundle_adjustment, detect_loop_closure, correct_loop


class SLAM:
    """
    SLAM 系统主类。

    职责：
      串联整个管线，管理帧间状态。

    帧间状态（为什么需要保存？）：
      prev_kp, prev_desc:        上一帧的 ORB 特征，用于当前帧的追踪匹配
      prev_stereo_pt_ids:        上一帧左图每个关键点对应的 MapPoint ID
                                 用于将上一帧的 3D 点传到当前帧做 PnP
      prev_pose:                 上一帧的位姿，用于关键帧判定
      all_poses:                 所有帧的平移向量，用于最后的轨迹输出
    """

    def __init__(self, left_dir, right_dir):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.map = Map()
        self.tracker = Tracker()
        # CLAHE = Contrast Limited Adaptive Histogram Equalization
        # 红外图像通常低对比度、低纹理，CLAHE 能显著增强特征
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.prev_kp = None
        self.prev_desc = None
        self.prev_stereo_pt_ids = None
        self.prev_pose = None
        self.all_poses = []  # 每一帧的平移 (x, y, z)

    def run(self):
        """主循环：加载数据 → 逐帧处理 → 输出结果。"""
        pairs = load_stereo_pairs(self.left_dir, self.right_dir, verbose=True)
        if len(pairs) == 0:
            print("No stereo pairs. Exiting.")
            return
        print(f"Processing {len(pairs)} frames...")
        for i, (ts, left_img, right_img) in enumerate(pairs):
            self._process_frame(i, ts, left_img, right_img)
        self._finish()

    def _process_frame(self, idx, ts, left_img, right_img):
        """
        处理一帧图像 —— SLAM 的核心循环。

        参数:
            idx:      帧序号（从 0 开始）
            ts:       设备时间戳
            left_img: 左目灰度图
            right_img: 右目灰度图
        """
        # ── Step 1: CLAHE 增强（红外图像对比度增强） ──
        left_enh = self.clahe.apply(left_img)
        right_enh = self.clahe.apply(right_img)

        # ── Step 2: 左右图分别提取 ORB 特征 ──
        kp_left, desc_left = self.tracker.extract(left_enh)
        kp_right, desc_right = self.tracker.extract(right_enh)

        # ── Step 3: 左右图特征匹配 + 双目三角化 ──
        stereo_matches = self.tracker.match_ratio_test(desc_left, desc_right)
        if len(stereo_matches) < 6:
            print(f"[{idx+1:3d}]  stereo_matches={len(stereo_matches)} -> skip")
            return

        # 三角化得到 3D 点（在左相机坐标系下）
        pts_3d_cam, _ = triangulate_stereo(
            kp_left, kp_right, stereo_matches,
            K_LEFT, K_RIGHT, R_STEREO, T_STEREO,
        )
        # 筛选有效点：
        #   1. 有限数值（非 NaN/Inf）
        #   2. Z > 100mm（过近的点误差大，不可靠）
        valid = np.isfinite(pts_3d_cam).all(axis=1) & (pts_3d_cam[:, 2] > 100)
        pts_3d_cam = pts_3d_cam[valid]
        stereo_matches = [m for j, m in enumerate(stereo_matches) if valid[j]]

        # ── Step 4: 第一帧特殊处理（初始化地图） ──
        # 第一帧没有"上一帧"可以做 PnP，直接设位姿为单位矩阵
        # 世界坐标系 = 第一帧的左相机坐标系
        if idx == 0:
            pose = np.eye(4, dtype=np.float64)
            pts_3d_world = pts_3d_cam.copy()
            # 将所有三角化点加入地图
            pt_ids = [
                self.map.add_point(pts_3d_world[j],
                                   desc_left[stereo_matches[j].queryIdx])
                for j in range(len(pts_3d_world))
            ]
            # 建立左图关键点索引 → MapPoint ID 的映射
            # 这样下一帧通过匹配就能找到对应的 3D 点
            self.prev_stereo_pt_ids = {
                m.queryIdx: pt_ids[j]
                for j, m in enumerate(stereo_matches)
            }
            self.prev_kp = kp_left
            self.prev_desc = desc_left
            self.prev_pose = pose.copy()

            # 第一帧也是第一个关键帧
            kf_pts, kf_pids = [], []
            for j in range(len(kp_left)):
                if j in self.prev_stereo_pt_ids:
                    kf_pids.append(self.prev_stereo_pt_ids[j])
                    kf_pts.append(kp_left[j].pt)
            self.map.add_keyframe(pose, kf_pts, kf_pids, ts)
            self.all_poses.append(pose[:3, 3].copy())
            print(f"[{idx+1:3d}] INIT   ...")
            return

        # ── Step 5: 帧间追踪 + PnP 位姿估计 ──
        # 当前帧左图 vs 上一帧左图 → 特征匹配
        track_matches = self.tracker.match_ratio_test(desc_left, self.prev_desc)

        # 收集能关联到地图点的 3D-2D 匹配对：
        #   - 3D: 地图点的世界坐标
        #   - 2D: 当前帧的左图像素坐标
        pts_3d_pnp, pts_2d_pnp = [], []
        for m in track_matches:
            tidx = m.trainIdx  # 上一帧（train）的关键点索引
            if tidx in self.prev_stereo_pt_ids:
                pid = self.prev_stereo_pt_ids[tidx]
                if pid in self.map.points:
                    pts_3d_pnp.append(self.map.points[pid].position)
                    pts_2d_pnp.append(kp_left[m.queryIdx].pt)

        # PnP RANSAC 求解位姿
        rvec, tvec, inlier_mask = self.tracker.estimate_pose_pnp(
            pts_3d_pnp, pts_2d_pnp
        )
        if rvec is None:
            print(f"[{idx+1:3d}] PnP FAILED  tracked={len(track_matches)}")
            # PnP 失败时只更新上一帧信息，不更新位姿
            # 这样下一帧还能尝试追踪
            self.prev_kp = kp_left
            self.prev_desc = desc_left
            return

        pose = self.tracker.rvec_tvec_to_matrix(rvec, tvec)

        # ── Step 6: 新三角化点加入地图 ──
        # 将相机坐标系的 3D 点转换到世界坐标系
        pose_inv = np.linalg.inv(pose)
        pts_3d_world = (pose_inv[:3, :3] @ pts_3d_cam.T +
                        pose_inv[:3, 3:4]).T

        # 最多新增 200 个点（避免地图膨胀过快）
        new_pt_ids = {}
        for j, m in enumerate(stereo_matches):
            if len(new_pt_ids) >= 200:
                break
            lidx = m.queryIdx
            if lidx in new_pt_ids:
                continue
            pid = self.map.add_point(pts_3d_world[j], desc_left[lidx],
                                     first_kf_id=idx + 1)
            new_pt_ids[lidx] = pid

        # 合并追踪到的旧点和新点
        # merged_pt_ids: 当前帧左图每个关键点 → 地图点 ID
        merged_pt_ids = {}
        for m in track_matches:
            qidx, tidx = m.queryIdx, m.trainIdx
            if tidx in self.prev_stereo_pt_ids:
                merged_pt_ids[qidx] = self.prev_stereo_pt_ids[tidx]
        merged_pt_ids.update(new_pt_ids)

        # 统计追踪到的旧点数量（用于关键帧判定）
        tracked_count = sum(
            1 for m in track_matches
            if m.trainIdx in self.prev_stereo_pt_ids
        )

        # ── Step 7: 关键帧判定 ──
        kf_needed = is_keyframe(
            self.prev_pose, pose,
            tracked_count / max(len(track_matches), 1)
        )

        # ── Step 8: 如果是关键帧 → 加入地图 + BA + 回环检测 ──
        if kf_needed:
            kf_pts, kf_pids = [], []
            for j in range(len(kp_left)):
                if j in merged_pt_ids:
                    kf_pids.append(merged_pt_ids[j])
                    kf_pts.append(kp_left[j].pt)
            self.map.add_keyframe(pose, kf_pts, kf_pids, ts)

            # 每隔 10 帧或前 3 帧做一次局部 BA
            run_ba = (len(self.map.keyframes) % 10 == 0) or \
                     len(self.map.keyframes) < 3
            if run_ba:
                local_bundle_adjustment(self.map)

            # 回环检测 + 修正
            loop_kf = detect_loop_closure(
                self.map, self.map.keyframes[-1], self.map.keyframes
            )
            if loop_kf is not None:
                if correct_loop(self.map, loop_kf, self.map.keyframes[-1]):
                    print(f"      -> Loop closed with KF {loop_kf.id}")
                else:
                    print(f"      -> Loop candidate "
                          f"(insufficient overlap) KF {loop_kf.id}")

        # ── 更新帧间状态 ──
        self.prev_kp = kp_left
        self.prev_desc = desc_left
        self.prev_pose = pose.copy()
        self.prev_stereo_pt_ids = merged_pt_ids
        self.all_poses.append(pose[:3, 3].copy())

        # ── 打印当前帧信息 ──
        pos = pose[:3, 3]
        print(
            f"[{idx+1:3d}] TRACK ts={ts}  "
            f"features={len(kp_left)}  "
            f"track={len(track_matches)}  "
            f"inliers={sum(inlier_mask) if inlier_mask is not None else 0}  "
            f"pts={len(self.map.points)}  "
            f"kfs={len(self.map.keyframes)}  "
            f"pose=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})  "
            f"dist={np.linalg.norm(pos):.0f}mm"
        )

    def _finish(self):
        """所有帧处理完成后的收尾工作：保存轨迹 CSV 和图片。"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        traj = np.array(self.all_poses) if self.all_poses else np.zeros((0, 3))

        # ── CSV 文件 ──
        csv_file = os.path.join(OUT_DIR, 'trajectory.csv')
        with open(csv_file, 'w', newline='') as f:
            import csv
            w = csv.writer(f)
            w.writerow(['frame', 'x_mm', 'y_mm', 'z_mm'])
            for i, p in enumerate(traj):
                w.writerow([i, f'{p[0]:.1f}', f'{p[1]:.1f}', f'{p[2]:.1f}'])
        print(f"Trajectory saved to: {csv_file}")

        # ── 轨迹图 ──
        if len(traj) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('Camera Trajectory (continuous)', fontsize=14)

            # 左图：俯视图 XZ
            ax1.plot(traj[:, 0], traj[:, 2], 'b-', linewidth=1.5,
                     label='trajectory')
            ax1.scatter(traj[0, 0], traj[0, 2], c='g', s=80, marker='s',
                        label='start')
            ax1.scatter(traj[-1, 0], traj[-1, 2], c='r', s=80, marker='o',
                        label='end')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Z (mm)')
            ax1.set_title('Top-down view (XZ)')
            ax1.axis('equal')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 右图：3D 视图
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-',
                     linewidth=1.5)
            ax2.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                        c='g', s=80, marker='s')
            ax2.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                        c='r', s=80, marker='o')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')
            ax2.set_zlabel('Z (mm)')
            ax2.set_title('3D trajectory')

            plot_path = os.path.join(OUT_DIR, 'trajectory.png')
            fig.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"Trajectory plot saved to: {plot_path}")

        # ── 统计数据 ──
        print(f"\n{'='*60}")
        print(f"SLAM Complete")
        print(f"  Total frames: {len(traj)}")
        print(f"  Keyframes: {len(self.map.keyframes)}")
        print(f"  Map points: {len(self.map.points)}")
        if len(traj) > 1:
            step = np.diff(traj, axis=0)
            print(f"  Trajectory length: "
                  f"{np.linalg.norm(step, axis=1).sum():.0f} mm")
            print(f"  Start: ({traj[0,0]:.0f}, {traj[0,1]:.0f}, {traj[0,2]:.0f})")
            print(f"  End:   ({traj[-1,0]:.0f}, {traj[-1,1]:.0f}, {traj[-1,2]:.0f})")
        print(f"{'='*60}")
