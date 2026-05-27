# 09 - 动手实现：从零搭建视觉 SLAM

> 目标：理解了我前面 8 个教程后，自己动手一步步写一个 SLAM
> 方式：每步 50-80 行代码，跑通再下一步

---

## 概述

我们会从最简单的功能开始，逐步叠加，最终搭出一个完整的双目 VO（视觉里程计）。

```
step1: 读取 + 显示图像 (20 行)
step2: ORB 特征 + 匹配 (30 行)
step3: 单帧立体匹配 + 深度图 (40 行)
step4: 两帧跟踪 + PnP (60 行)
step5: 多帧 VO 流水线 (80 行)
step6: 关键帧 + 局部 BA (100 行)
step7: 完整 SLAM (150 行)
```

**每个 step 都独立可运行。**

---

## Step 1：读取图像 🔴

```python
# step1_read_images.py
# 目标：读取左右图，显示基本信息

import cv2
import numpy as np

LEFT_DIR = '../IR_LEFT'
RIGHT_DIR = '../IR_RIGHT'

def read_image(path):
    """支持中文路径的图像读取"""
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

# 读取第一对
left = read_image(f'{LEFT_DIR}/011710_0000_d1776928462281910_*.png')
right = read_image(f'{RIGHT_DIR}/011700_0000_d1776928462415402_*.png')

print(f"左图尺寸: {left.shape}")
print(f"右图尺寸: {right.shape}")
print(f"数据类型: {left.dtype}")
print(f"像素值范围: {left.min()} ~ {left.max()}")

# 保存拼合图
combined = np.hstack([left, right])
cv2.imwrite('output/step1_stereo.png', combined)
print("已保存 output/step1_stereo.png")
```

✅ **验证：** 看到两张并排的灰度图，尺寸 848×480。

---

## Step 2：ORB 特征 🔴

```python
# step2_features.py
# 目标：提取特征 → 匹配 → 可视化

import cv2
import numpy as np

LEFT_DIR = '../IR_LEFT'
RIGHT_DIR = '../IR_RIGHT'

def read_image(path):
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def match_ratio_test(desc1, desc2, ratio=0.75):
    """比率测试匹配"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in raw:
        if len(m) == 2:  # 兼容不同 OpenCV 版本
            m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

left = read_image(f'{LEFT_DIR}/011710_*.png')
right = read_image(f'{RIGHT_DIR}/011700_*.png')

# ORB 特征提取
orb = cv2.ORB_create(nfeatures=1000)
kp_l, desc_l = orb.detectAndCompute(left, None)
kp_r, desc_r = orb.detectAndCompute(right, None)

# 匹配
matches = match_ratio_test(desc_l, desc_r)

print(f"左图特征: {len(kp_l)}")
print(f"右图特征: {len(kp_r)}")
print(f"匹配对: {len(matches)}")

# 可视化
match_img = cv2.drawMatches(
    left, kp_l, right, kp_r, matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite('output/step2_matches.png', match_img)
print("已保存 output/step2_matches.png")
```

✅ **验证：** 匹配线整齐平行，而不是杂乱交叉。

---

## Step 3：立体匹配 + 深度图 🔴

```python
# step3_stereo_depth.py
# 目标：用 StereoSGBM 算视差 → 转深度

import cv2
import numpy as np

LEFT_DIR = '../IR_LEFT'
RIGHT_DIR = '../IR_RIGHT'

FX = 596.25    # 焦距 x（像素）
BASELINE = 95.0  # 基线（mm）

def read_image(path):
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

# 读取
left = read_image(f'{LEFT_DIR}/011710_*.png')
right = read_image(f'{RIGHT_DIR}/011700_*.png')

# CLAHE 增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
left_enh = clahe.apply(left)
right_enh = clahe.apply(right)

# StereoSGBM 视差计算
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8*3*25,   # 经验公式
    P2=32*3*25,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_HH,
)
disparity = sgbm.compute(left_enh, right_enh).astype(np.float32) / 16.0

# 视差 → 深度（mm）
depth = np.zeros_like(disparity)
mask = disparity > 0
depth[mask] = FX * BASELINE / disparity[mask]

print(f"有效深度像素: {mask.sum()}/{depth.size} ({mask.sum()/depth.size*100:.1f}%)")
print(f"深度范围: {depth[mask].min():.0f} ~ {depth[mask].max():.0f} mm")
print(f"平均深度: {depth[mask].mean():.0f} mm")

# 保存深度图
depth_norm = np.clip(depth / 5000, 0, 1)  # 5m 以内
depth_color = cv2.applyColorMap((1-depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
depth_color[~mask] = 0
cv2.imwrite('output/step3_depth.png', depth_color)
print("已保存 output/step3_depth.png")
```

✅ **验证：** 近处物体更亮（深度值小），远处更暗。

---

## Step 4：两帧跟踪 + PnP 🔴

```python
# step4_tracking.py
# 目标：给定第一帧的 3D 点，跟踪第二帧

import cv2
import numpy as np

K = np.array([[596.25, 0, 424], [0, 540, 240], [0, 0, 1]])
FX = K[0, 0]
BASELINE = 95.0

def read_image(path):
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def match_ratio_test(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(desc1, desc2, k=2)
    return [m for m, n in raw if m.distance < ratio * n.distance]

def triangulate_stereo(kp_l, kp_r, matches, K):
    """双目三角化，返回左侧图中的 3D 点"""
    pts_l = np.float32([kp_l[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches])
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.eye(3), [[-BASELINE], [0], [0]]])
    pts_4d = cv2.triangulatePoints(P1, P2, pts_l.T, pts_r.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    # 只保留在相机前方的点
    valid = pts_3d[:, 2] > 100
    return pts_3d[valid], [m for j, m in enumerate(matches) if valid[j]]

# 读取第一帧
left0 = read_image('../IR_LEFT/011710_*.png')
right0 = read_image('../IR_RIGHT/011700_*.png')
# 第二帧
left1 = read_image('../IR_LEFT/011712_*.png')
right1 = read_image('../IR_RIGHT/011701_*.png')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(nfeatures=1000)

# 第一帧：初始化 3D 点
kp_l0, desc_l0 = orb.detectAndCompute(clahe.apply(left0), None)
kp_r0, desc_r0 = orb.detectAndCompute(clahe.apply(right0), None)
matches0 = match_ratio_test(desc_l0, desc_r0)
pts_3d, matches0 = triangulate_stereo(kp_l0, kp_r0, matches0, K)
print(f"第一帧 3D 点: {len(pts_3d)}")

# 存储关键点→3D 点的映射
kp_to_3d = {matches0[j].queryIdx: pts_3d[j] for j in range(len(pts_3d))}

# 第二帧：跟踪
kp_l1, desc_l1 = orb.detectAndCompute(clahe.apply(left1), None)
track_matches = match_ratio_test(desc_l1, desc_l0)

# 构建 3D-2D 对应
pts_3d_pnp = []
pts_2d_pnp = []
for m in track_matches:
    if m.trainIdx in kp_to_3d:
        pts_3d_pnp.append(kp_to_3d[m.trainIdx])
        pts_2d_pnp.append(kp_l1[m.queryIdx].pt)

print(f"PnP 匹配对: {len(pts_3d_pnp)}")

if len(pts_3d_pnp) >= 6:
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.float32(pts_3d_pnp), np.float32(pts_2d_pnp),
        K, None, iterationsCount=200, reprojectionError=8.0
    )
    R, _ = cv2.Rodrigues(rvec)
    print(f"旋转: {R}")
    print(f"平移 (mm): {tvec.ravel()}")
    print(f"内点: {len(inliers)}/{len(pts_3d_pnp)}")
else:
    print("PnP 匹配不足！")
```

✅ **验证：** 平移向量的 Z 分量约 20-30mm（帧间微小运动）。

---

## Step 5：多帧 VO 流水线 🟡

```python
# step5_vo.py
# 目标：遍历所有帧，估计每帧位姿

import cv2, os, sys
import numpy as np

# 引入前面的函数
from step4_tracking import read_image, match_ratio_test
from step4_tracking import triangulate_stereo

K = np.array([[596.25, 0, 424], [0, 540, 240], [0, 0, 1]])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(nfeatures=800)

# 从 dataset 模块获取图片对
sys.path.insert(0, '..')
from dataset import load_stereo_pairs

pairs = load_stereo_pairs('../IR_LEFT', '../IR_RIGHT', verbose=True)

prev_kp = None
prev_desc = None
prev_pt_map = {}  # kp_idx → 3D point
trajectory = []    # 存储每一帧的位置

for idx, (ts, left_img, right_img) in enumerate(pairs):
    left_enh = clahe.apply(left_img)
    right_enh = clahe.apply(right_img)

    kp, desc = orb.detectAndCompute(left_enh, None)
    kp_r, desc_r = orb.detectAndCompute(right_enh, None)

    if idx == 0:
        # 第一帧：初始化
        matches = match_ratio_test(desc, desc_r)
        pts_3d, matches = triangulate_stereo(kp, kp_r, matches, K)
        prev_pt_map = {matches[j].queryIdx: pts_3d[j]
                       for j in range(len(pts_3d))}
        prev_kp, prev_desc = kp, desc
        trajectory.append(np.zeros(3))
        print(f"[{idx+1:3d}] INIT   ts={ts}")
        continue

    # 跟踪
    track_matches = match_ratio_test(desc, prev_desc)
    pts_3d_pnp, pts_2d_pnp = [], []
    for m in track_matches:
        if m.trainIdx in prev_pt_map:
            pts_3d_pnp.append(prev_pt_map[m.trainIdx])
            pts_2d_pnp.append(kp[m.queryIdx].pt)

    if len(pts_3d_pnp) < 6:
        print(f"[{idx+1:3d}] PnP FAILED")
        trajectory.append(trajectory[-1])  # 复用上一帧位置
        continue

    _, rvec, tvec, _ = cv2.solvePnPRansac(
        np.float32(pts_3d_pnp), np.float32(pts_2d_pnp),
        K, None, iterationsCount=200, reprojectionError=8.0
    )
    R, _ = cv2.Rodrigues(rvec)
    pos = tvec.ravel()
    trajectory.append(pos.copy())
    print(f"[{idx+1:3d}] TRACK pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")

    # 下一帧用
    prev_kp, prev_desc = kp, desc

# 保存轨迹
traj = np.array(trajectory)
np.savetxt('output/step5_trajectory.txt', traj, header='x y z')

# 画轨迹
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.plot(traj[:, 0], traj[:, 2], 'b-')
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.title('Camera Trajectory (Top-down view)')
plt.axis('equal')
plt.grid(True)
plt.savefig('output/step5_trajectory.png')
print("已保存 output/step5_trajectory.png")
```

✅ **验证：** 轨迹平滑，没有剧烈跳变。

---

## Step 6：添加关键帧 + BA 🟡

在 Step 5 的基础上增加关键帧逻辑：

```python
# step6_keyframe_ba.py（节选）

def is_keyframe(prev_pose, curr_pose, tracked_ratio):
    """判断是否插入关键帧"""
    if prev_pose is None:
        return True
    delta = np.linalg.inv(prev_pose) @ curr_pose
    trans = np.linalg.norm(delta[:3, 3])
    # 条件：移动 > 200mm 或跟踪比 < 40%
    return trans > 200 or tracked_ratio < 0.4

# 在 VO 循环中添加：
keyframes = []  # 关键帧列表

if is_keyframe(prev_pose, curr_pose, tracked_ratio):
    keyframes.append(curr_pose)
    print(f"  插入关键帧 #{len(keyframes)}")
```

完整 BA 实现参考 `optimizer.py`。

---

## Step 7：完整 SLAM 系统 🔴

这是最终版本。你应该已经理解了每一步的原理。

```bash
python main.py
```

运行 `stereo_slam` 目录下的系统，阅读 `slam.py` 的每一行，理解它是怎么把前面的所有步骤串起来的。

---

## 总结：从 0 到 SLAM 的路线

```
Step 1: 读写图像
   │
Step 2: 特征提取 + 匹配
   │
Step 3: 立体匹配 → 深度图
   │
Step 4: PnP 帧间跟踪
   │
Step 5: 连续帧 VO
   │
Step 6: 关键帧 + BA
   │
Step 7: 完整 SLAM 系统

每一步都比上一步多 10-20 行代码
理解了每一步，就理解了整个 SLAM
```

---

## 当你卡住时

```
1. 打印所有中间变量的形状和数值
2. 可视化每一步的结果
3. 改参数（RANSAC阈值、ORB特征数、匹配比例）
4. 回到上一个能跑通的 step
5. 逐行对比和教程代码的差异
```
