# Milestone 3：三角化建图与关键帧管理

## 1. 概述

本阶段实现了 SLAM 的**地图构建**模块，将每帧立体匹配产生的 3D 点组织成全局一致的**稀疏点云地图**，并通过**关键帧**机制控制计算成本。

```
每一帧
  │
  ├─ 立体匹配 (左-右) ─→ 视差 → 三角化 → 相机坐标系 3D 点
  │                                                    │
  ├─ PnP 跟踪 → 当前帧位姿 ───── 转换到世界坐标 ───────┤
  │                                                    │
  └─ 关键帧判断 ──→ 是 → 插入关键帧 → 局部 BA
       │
       否 → 继续跟踪
```

---

## 2. 三角化：从 2D 匹配到 3D 点

### 2.1 立体三角化原理

给定左右相机中匹配的一对特征点，以及相机的内外参，可以通过**三角测量**恢复 3D 坐标。

**投影关系：**

$$
s_l \begin{bmatrix}u_l\\v_l\\1\end{bmatrix} = K_l \cdot [I_{3\times3} \mid 0_{3\times1}] \cdot P
$$
$$
s_r \begin{bmatrix}u_r\\v_r\\1\end{bmatrix} = K_r \cdot [R_{stereo} \mid t_{stereo}] \cdot P
$$

其中 $P = (X, Y, Z)$ 是空间点，$K_l, K_r$ 是左右相机内参，$R_{stereo}, t_{stereo}$ 是双目外参。

### 2.2 OpenCV 实现

```python
cv2.triangulatePoints(P1, P2, pts_left, pts_right)
```

$P_1$ 和 $P_2$ 是 3×4 投影矩阵。返回齐次坐标 $(X, Y, Z, W)$，归一化后得到 3D 点。

### 2.3 三角化退化

**三角化在以下情况下不可靠：**
- **视差太小**（物体太远或基线太小）→ 深度噪声大
- **点在极线上滑动**（匹配精度差）→ 深度不确定性大
- **点在相机后方**（$Z < 0$）→ 直接丢弃

我们过滤条件：
```python
valid = np.isfinite(pts) & (pts[:, 2] > 100)  # Z > 100mm
```

---

## 3. 全局地图管理

### 3.1 `Map` 类

```python
class Map:
    points: dict[int, MapPoint]     # 所有地图点 {id: point}
    keyframes: list[KeyFrame]       # 关键帧列表
```

### 3.2 `MapPoint`

```python
class MapPoint:
    id: int                         # 全局唯一 ID（自动递增）
    position: np.ndarray(3,)        # 世界坐标 (mm)
    descriptor: np.ndarray(32,)     # ORB 描述子
    first_kf_id: int                # 首次被观测到的关键帧 ID
```

### 3.3 `KeyFrame`

```python
class KeyFrame:
    id: int                         # 全局唯一 ID
    pose: np.ndarray(4,4)           # 世界→相机 变换矩阵
    pts_2d: list[(float, float)]    # 观测到的 2D 关键点
    point_ids: list[int]            # 对应的地图点 ID 列表
    timestamp: int                  # 时间戳
```

### 3.4 地图构建流程

```
帧 N 处理完成后：
  1. 立体匹配 → 3D 点（相机坐标系）
  2. PnP → 位姿 T_w_c
  3. 3D 点转换到世界坐标：P_w = T_w_c^{-1} * P_c
  4. 添加到全局地图
  5. 建立 2D关键点 ←→ 3D地图点 的关联
```

---

## 4. 关键帧机制

### 4.1 为什么需要关键帧？

| 问题 | 说明 |
|------|------|
| 存储爆炸 | 每帧都存 → 90 帧 × 1000 点 = 90K 点，冗余 |
| 计算爆炸 | 每帧优化 → 实时性无法保障 |
| 冗余信息 | 相邻帧几乎相同，不提供新信息 |

### 4.2 关键帧选择条件

当以下任一条件满足时，插入新关键帧：

```python
def is_keyframe(prev_pose, curr_pose, tracked_ratio):
    delta = prev_pose^{-1} @ curr_pose
    
    # 条件 1：平移超过阈值
    trans = ||delta[:3, 3]|| > 200mm
    
    # 条件 2：旋转超过阈值  
    angle = arccos((trace(R) - 1) / 2) > 10°
    
    # 条件 3：跟踪质量下降
    tracked_ratio < 40%
```

**平移条件**：相机移动了足够的距离，场景发生明显变化

**旋转条件**：相机旋转角度较大，视野变化显著

**跟踪质量**：当前帧中能跟踪到的特征太少，需要更多地图点支持

### 4.3 关键帧的作用

1. **减少冗余**：仅保留信息量大的帧
2. **局部地图**：只维护关键帧附近的 3D 点
3. **后端优化**：BA 只在关键帧上执行
4. **闭环检测**：关键帧作为闭环候选

---

## 5. 坐标变换

### 5.1 坐标系统

| 坐标系 | 原点 | 说明 |
|--------|------|------|
| **世界坐标** | 第一帧左相机光心 | 全局统一 |
| **相机坐标** | 当前帧左相机光心 | Z 向前，X 向右，Y 向下 |

### 5.2 变换流程

```
相机坐标系下的点 P_c:
  Z_c = fx * B / disparity
  X_c = (u - cx) * Z_c / fx
  Y_c = (v - cy) * Z_c / fy

世界坐标系下的点 P_w:
  P_w = R_inv @ P_c + t_inv
  其中 [R_inv | t_inv] = T_w_c^{-1}
```

---

## 6. 代码结构

| 文件 | 关键类/函数 | 职责 |
|------|------------|------|
| `mapping.py` | `Map` | 全局地图容器 |
| `mapping.py` | `MapPoint` | 3D 点数据 |
| `mapping.py` | `KeyFrame` | 关键帧数据 |
| `mapping.py` | `triangulate_stereo()` | 立体三角化 |
| `mapping.py` | `is_keyframe()` | 关键帧决策 |
| `slam.py` | `SLAM._process_frame()` | 主流程编排 |

---

## 7. 当前限制

1. **无点云滤波**：重复/冗余点未合并
2. **无外点剔除**：错误的三角化点未被移除
3. **无共视管理**：不知道哪些关键帧看到哪些点
4. **地图线性增长**：每帧增加新点，地图无限膨胀

这些限制将在 Milestone 4 中通过全局优化和地图管理来改善。
