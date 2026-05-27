# 08 - SLAM 后端：优化、闭环与系统架构

> 目标：理解 BA 怎么工作、闭环怎么检测、完整系统怎么组织

---

## 8.1 为什么要后端优化 🔴

### 8.1.1 前端的缺陷

前端（视觉里程计）的问题：**误差会累积**。

```
帧1 → 帧2 → 帧3 → ... → 帧100
误差0  误差1  误差2        误差99

每帧漂移 1cm，100 帧后漂移 1m！
```

### 8.1.2 后端做什么

后端 = **全局优化**

```
把"所有"相机位姿和 3D 点放到一起优化
让所有重投影误差的总和最小
从而减少累积漂移
```

---

## 8.2 束调整 BA 🔴

### 8.2.1 BA 是什么

Bundle Adjustment（束调整）—— 名字的由来：

```
Bundle = "光束"（从 3D 点发出的光线）
Adjustment = "调整"

把所有 3D 点发出的光线（经过相机光心到像素） 
同时调整位姿和点，让所有光线最吻合。
```

### 8.2.2 BA 优化什么

```
需要优化的变量：
  {R₁, t₁, R₂, t₂, ..., Rₘ, tₘ}  ← m 个相机位姿
  {X₁, X₂, ..., Xₙ}                 ← n 个 3D 点

总共 6m + 3n 个变量

目标函数：
  min Σ ||重投影误差||²
```

### 8.2.3 重投影误差

```python
# 一个观测的误差
def compute_error(R, t, K, X_world, u_obs, v_obs):
    """
    R, t: 当前相机位姿
    K: 内参
    X_world: 3D 点世界坐标
    u_obs, v_obs: 观测到的像素坐标
    """
    # 世界 → 相机
    X_cam = R @ X_world + t
    
    # 投影到像素
    u_proj = K[0,0] * X_cam[0] / X_cam[2] + K[0,2]
    v_proj = K[1,1] * X_cam[1] / X_cam[2] + K[1,2]
    
    # 重投影误差
    error_u = u_proj - u_obs
    error_v = v_proj - v_obs
    
    return np.array([error_u, error_v])
```

### 8.2.4 💡 BA 的直觉理解

```
把 BA 想象成一个弹簧系统：

相机位姿 ← 弹簧（重投影误差） → 3D 点

目标 = 调整所有相机和点
      让所有弹簧的"拉力"总和最小
       
一个观测被多个相机共享 → "拉力"互相制约 → 整体一致
```

### 8.2.5 用 scipy 实现简单 BA

```python
from scipy.optimize import least_squares

def bundle_adjustment(cameras, points, observations):
    """
    cameras: [(R1,t1), (R2,t2), ...]
    points: [P1, P2, ...]
    observations: [(cam_idx, pt_idx, u, v), ...]
    """
    # 把所有参数展开成一维向量
    x0 = pack_params(cameras, points)
    
    def cost_func(x):
        cameras, points = unpack_params(x)
        residuals = []
        for cam_idx, pt_idx, u, v in observations:
            error = compute_error(
                cameras[cam_idx], K, points[pt_idx], u, v)
            residuals.extend(error)
        return np.array(residuals)
    
    result = least_squares(cost_func, x0, method='trf')
    return unpack_params(result.x)
```

### 8.2.6 局部 BA vs 全局 BA

```python
# 局部 BA：只优化最近的 N 个关键帧
# 速度快，适合实时
local_window = keyframes[-5:]  # 最近 5 帧
local_ba(local_window)

# 全局 BA：优化所有关键帧和点
# 速度慢，适合后台线程
global_ba(all_keyframes)
```

---

## 8.3 信息矩阵（权重）🟡

### 8.3.1 有些观测更可信

```
特征点在纹理丰富的区域 → 匹配准 → 权重高
特征点在纹理贫乏的区域 → 匹配差 → 权重低
近处的点 → 三角化准 → 权重高
远处的点 → 三角化差 → 权重低
```

### 8.3.2 加权 BA

$$\min \sum \mathbf{e}_{ij}^T \Omega_{ij} \mathbf{e}_{ij}$$

其中 $\Omega_{ij} = \Sigma_{ij}^{-1}$ 是信息矩阵。

```python
# 加权误差
error_weighted = error.T @ Omega @ error
```

**直觉：** 方差大（不确定）的观测 → 信息小 → 对优化影响小。

---

## 8.4 闭环检测 🟡

### 8.4.1 为什么需要闭环

```
没有闭环：误差一直累积，轨迹永远回不到正确位置
有闭环：检测到回到原点 → 把轨迹"拉"回来
```

### 8.4.2 简单实现

```python
def detect_loop_closure(current_kf, all_kfs):
    """
    检测当前关键帧是否与之前的关键帧形成闭环
    """
    # 只检查时间上较远的关键帧
    candidates = all_kfs[:-20]  # 跳过最近 20 帧
    
    for kf in candidates:
        # 位置够近？
        dist = np.linalg.norm(current_kf.position - kf.position)
        if dist > LOOP_CLOSURE_DIST:
            continue
        
        # 特征匹配够多？
        matches = match_features(current_kf.descriptors, kf.descriptors)
        if len(matches) < LOOP_MIN_MATCHES:
            continue
        
        # 几何验证通过？
        if geometry_verification(matches):
            return kf  # 找到闭环！
    
    return None  # 没找到
```

### 8.4.3 闭环修正

检测到闭环后，需要优化整个位姿图：

```python
# 位姿图优化（只优化位姿，不优化 3D 点）
# 比 BA 快很多

from scipy.optimize import least_squares

def pose_graph_optimization(keyframes, loop_pairs):
    """
    keyframes: 所有关键帧
    loop_pairs: [(kf_i, kf_j, relative_pose), ...]
    """
    # 构建位姿图
    # 节点 = 关键帧位姿
    # 边 = 帧间相对位姿 + 闭环相对位姿
    # 优化使得所有边的误差最小
    pass
```

---

## 8.5 关键帧 🔴

### 8.5.1 为什么需要关键帧

```
不选关键帧：每帧都参与 BA → 计算量爆炸
选关键帧：只选信息量大、有代表性的帧
```

### 8.5.2 何时插入关键帧

```python
def is_keyframe(prev_pose, curr_pose, tracked_points):
    # 条件1：移动了足够距离
    delta = prev_pose⁻¹ @ curr_pose
    translation = ||delta[:3, 3]||
    
    if translation > 200:  # mm
        return True
    
    # 条件2：旋转了足够角度
    angle = degrees(arccos((trace(R) - 1) / 2))
    if angle > 10:  # 度
        return True
    
    # 条件3：跟踪到的点太少
    if tracked_points < 40%:
        return True
    
    return False
```

### 8.5.3 关键帧也需剔除

```
如果新关键帧和前一帧几乎一样？→ 冗余，剔除
检查标准：新 KF 的 90% 的 3D 点都被其他 KF 看到了
```

---

## 8.6 系统架构 🔴

### 8.6.1 经典三线程结构（ORB-SLAM）

```
主线程（摄像头）
  │
  跟踪线程（Tracking）
  │ 实时处理每帧
  │ 特征提取 → 帧间匹配 → PnP → 关键帧决策
  │
  输出关键帧 → 局部建图线程（Local Mapping）
                 三角化新点 → 局部 BA → 剔除冗余 KF
                │
                输出关键帧 → 闭环线程（Loop Closing）
                             词袋查询 → 几何验证 → 全局优化
```

### 8.6.2 为什么需要多线程

```
跟踪线程：必须实时（30fps），每次<33ms
局部建图：可以慢一点（几帧处理一次）
闭环线程：可以更慢（几秒处理一次）

如果不分开：
  BA 跑一次 500ms → 跟踪被阻塞 → 丢帧
```

### 8.6.3 数据共享

```
地图（Map）是所有线程共享的：
  - 跟踪：读关键帧和 3D 点
  - 建图：写新的 3D 点
  - 闭环：修改位姿和点

需要加锁保护
```

---

## 8.7 完整 SLAM 流程图

```
        开始
          │
          ▼
    读取左右图像
          │
          ▼
    CLAHE 增强对比度
          │
          ▼
    ORB 特征提取
          │
          ▼
   ┌── 第一帧？────┐
   │  是            │  否
   ▼                ▼
初始化地图    与上一帧匹配
(位姿=I)        │
   │            ▼
   │      构建 3D-2D 对应
   │            │
   │            ▼
   │       PnP + RANSAC
   │            │
   │            ▼
   └──→ 立体匹配三角化
            │
            ▼
        添加新地图点
            │
            ▼
       关键帧判断
            │
       ┌────┴────┐
       │ 是       │ 否
       ▼          │
    插入关键帧    │
       │          │
       ▼          │
    局部 BA       │
       │          │
       ▼          │
    闭环检测      │
       │          │
       ▼          │
    更新状态      │
       │          │
       └──────────┘
            │
            ▼
        下一帧...
```

---

## ✅ 自测

```
1. BA 同时优化哪些变量？
2. 重投影误差怎么计算？
3. 信息矩阵在 BA 中的作用是什么？
4. 为什么要用关键帧？不用会怎样？
5. 闭环检测的基本思路是什么？
```

**代码练习：**

```python
# 用 scipy 实现一个微缩 BA：
# 3 个相机，10 个 3D 点
# 生成模拟观测数据
# BA 优化前后对比重投影误差
```
