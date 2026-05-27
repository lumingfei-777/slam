# 07 - SLAM 前端：帧间跟踪与位姿估计

> 目标：理解 PnP、三角化、帧间跟踪的完整流程

---

## 7.1 SLAM 前端在做什么 🔴

### 7.1.1 前端的任务

```
输入：左右图像对
输出：每一帧的相机位姿 + 新的 3D 地图点

核心问题：
  给定当前帧的特征点，以及它们与上一帧/地图的对应关系
  求当前相机在哪里？
```

### 7.1.2 一句话总结

```
前端 = 特征匹配 + PnP + 三角化
```

---

## 7.2 PnP 位姿估计 🔴

### 7.2.1 问题定义

**已知：**
- 一组 3D 世界点 $P_1, P_2, ..., P_n$
- 它们在当前图像上的投影 $p_1, p_2, ..., p_n$
- 相机内参 $K$

**求：**
- 当前相机的位姿 $[R \mid t]$

### 7.2.2 数学形式

$$s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = K \begin{bmatrix} R & t \end{bmatrix}
\begin{bmatrix} X_i \\ Y_i \\ Z_i \\ 1 \end{bmatrix}$$

6 个未知数（3 旋转 + 3 平移），每个 3D-2D 对给 2 个方程。

**至少需要 3 对点**，实际需要 6 对以上（加 RANSAC）。

### 7.2.3 OpenCV 实现

```python
# solvePnPRansac = EPnP 求解器 + RANSAC 剔除外点
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    pts_3d,              # (N, 3) 世界坐标
    pts_2d,              # (N, 2) 像素坐标
    K,                   # 3×3 内参
    None,                # 畸变（已去畸变则 None）
    iterationsCount=200, # RANSAC 迭代次数
    reprojectionError=8.0,  # 内点阈值（像素）
    confidence=0.99,     # 置信度
    flags=cv2.SOLVEPNP_ITERATIVE
)

# 旋转向量 → 旋转矩阵
R, _ = cv2.Rodrigues(rvec)

# 构建 4×4 变换矩阵
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = tvec.ravel()
```

### 7.2.4 💡 RANSAC 干了什么

```
1. 随机选 4 个 3D-2D 对
2. 用这 4 对算一个候选位姿
3. 用这个位姿测试所有匹配：
   重投影误差 < 阈值 → 内点
   否则 → 外点
4. 重复 1-3 共 200 次
5. 取内点最多的一次
6. 用所有内点重新优化
```

**效果：** 即使 50% 匹配是错误的，RANSAC 仍然能正确估计位姿。

### 7.2.5 验证 PnP 结果

```python
# 计算重投影误差
def compute_reprojection_error(R, t, K, pts_3d, pts_2d):
    """返回每个点的重投影误差（像素）"""
    pts_cam = (R @ pts_3d.T + t.reshape(3, 1)).T
    u_proj = K[0,0] * pts_cam[:,0] / pts_cam[:,2] + K[0,2]
    v_proj = K[1,1] * pts_cam[:,1] / pts_cam[:,2] + K[1,2]
    errors = np.sqrt((u_proj - pts_2d[:,0])**2 + 
                     (v_proj - pts_2d[:,1])**2)
    return errors

# 平均重投影误差应 < 2 像素
errors = compute_reprojection_error(R, t, K, inlier_3d, inlier_2d)
print(f"平均重投影误差: {errors.mean():.2f} 像素")
```

---

## 7.3 三角化 🔴

### 7.3.1 问题定义

**已知：**
- 左图点 $p_l$ 和右图点 $p_r$（或两帧间匹配点）
- 两帧的位姿 $[R_l \mid t_l]$ 和 $[R_r \mid t_r]$
- 相机内参 $K$

**求：** 对应的 3D 点 $P$

### 7.3.2 在双目 SLAM 中的三角化

因为双目相机的 $R$ 和 $t$ 已知（标定得到的）：

```python
# 双目三角化（基线已知）
def triangulate_stereo(kp_left, kp_right, matches, K_l, K_r, R, t):
    """从左右图匹配的 ORB 特征三角化 3D 点"""
    pts_l = np.float32([kp_left[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp_right[m.trainIdx].pt for m in matches])
    
    # 投影矩阵
    P1 = K_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_r @ np.hstack([R, t.reshape(3, 1)])
    
    # 三角化
    pts_4d = cv2.triangulatePoints(P1, P2, pts_l.T, pts_r.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    
    return pts_3d
```

### 7.3.3 💡 点应该在相机前方

```python
# 过滤：点在相机后方（Z < 0）的剔除
valid = pts_3d[:, 2] > 100  # 至少 100mm 远
pts_3d = pts_3d[valid]
```

---

## 7.4 帧间跟踪流程 🔴

### 7.4.1 数据流

```
帧 N-1:
  特征 kp_prev, desc_prev
  3D 点 pts_3d_prev
  关联: kp_idx → 3D_point_id
        │
        │ 当前帧到来
        ▼
帧 N:
  1. 提取 ORB 特征
  2. 与帧 N-1 匹配 → track_matches
  3. 从匹配中找 3D-2D 对：
     match.queryIdx → kp_current
     match.trainIdx → kp_prev → 3D_point
  4. PnP + RANSAC → 当前帧位姿
  5. 当前帧立体匹配 → 新 3D 点
  6. 新点加入地图
  7. 判断是否关键帧
```

### 7.4.2 关键代码

```python
# 1. 特征提取
kp_curr, desc_curr = orb.detectAndCompute(left_enhanced, None)

# 2. 与上一帧匹配
matches = match_ratio_test(desc_curr, desc_prev)

# 3. 构建 3D-2D 对应
pts_3d_pnp = []
pts_2d_pnp = []
for m in matches:
    # m.queryIdx: 当前帧的关键点索引
    # m.trainIdx: 上一帧的关键点索引
    prev_kp_idx = m.trainIdx
    if prev_kp_idx in prev_pt_map:  # 这个关键点有对应的 3D 点
        point_id = prev_pt_map[prev_kp_idx]
        pts_3d_pnp.append(map.points[point_id].position)
        pts_2d_pnp.append(kp_curr[m.queryIdx].pt)

# 4. PnP
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    np.array(pts_3d_pnp), np.array(pts_2d_pnp), K, None
)

# 5. 三角化新点
pts_3d_new = triangulate_stereo(kp_curr, kp_right, stereo_matches, ...)

# 6. 加入地图
for pt in pts_3d_new:
    map.add_point(pt)

# 7. 更新状态
prev_kp, prev_desc = kp_curr, desc_curr
prev_pose = current_pose
```

---

## 7.5 初始化 🔴

### 7.5.1 第一帧没有 3D 点

第一帧无法做 PnP，因为还没有 3D 地图。

**解法：** 用第一帧的立体匹配初始化地图。

```python
if idx == 0:
    # 第一帧：位姿设为单位矩阵（世界原点）
    pose = np.eye(4)
    
    # 立体匹配 → 三角化 → 加入地图
    pts_3d = triangulate_stereo(...)
    for pt in pts_3d:
        map.add_point(pt)
    
    # 记录关键点 → 3D 点的映射
    for j, match in enumerate(stereo_matches):
        prev_pt_map[match.queryIdx] = point_ids[j]
```

### 7.5.2 单目初始化更难

单目没有深度信息，需要**两帧之间有足够的平移**才能三角化。还需要恢复**尺度**（单目 SLAM 的尺度不确定性）。

---

## 7.6 常见问题 🟡

### 7.6.1 PnP 失败

```
症状：rvec/tvec 为 None
原因1：3D-2D 匹配对数 < 6
原因2：RANSAC 内点太少
原因3：3D 点分布不理想（全在一条线上）

处理：跳过这一帧，保留上一帧位姿
```

### 7.6.2 跟踪丢失

```
症状：连续多帧 PnP 失败
原因：快速运动 / 遮挡 / 低纹理

处理：尝试与最近的关键帧重定位匹配
      宽口径匹配 + 大 RANSAC 阈值
```

### 7.6.3 累积漂移

```
症状：轨迹漂浮不定，越来越偏离真实路径
原因：每帧 PnP 都有微小误差，不断累积

处理：这是正常现象！
      局部 BA 减小短时间漂移
      闭环检测修正长时间漂移
```

---

## ✅ 自测

```
1. PnP 的输入是什么？输出是什么？
2. RANSAC 为什么不直接用所有匹配算 PnP？
3. 第一帧的 3D 点从哪里来？
4. 如何判断 PnP 结果是否可靠？
5. 重投影误差的正常范围是多少？
```

**代码练习：**

```python
# 用数据集的两帧图像，手动实现：
# 1. 第一帧提取特征 + 立体匹配 → 初始化 3D 点
# 2. 第二帧提取特征 + 与第一帧匹配
# 3. PnP 求第二帧位姿
# 4. 可视化轨迹 + 3D 点云
```
