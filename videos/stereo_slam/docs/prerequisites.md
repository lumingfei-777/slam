# 视觉 SLAM 前置知识大全

> 起点：大一新生水平（高三数学 + 刚学编程）
> 终点：能理解 SLAM 论文和源码
> 方式：不跳步，每个概念都告诉你"这为什么重要"

---

## 导言：学 SLAM 需要什么？

SLAM = Simultaneous Localization and Mapping（同时定位与建图）

要理解它，你需要从**三个方向**打好基础：

```
视觉 SLAM
    │
    ├── 数学（看懂公式、推导算法）
    │
    ├── 编程（把数学变成代码）
    │
    └── 计算机视觉/机器人（领域的常识）
```

本文按**必须掌握 → 建议了解 → 以后再说**三档标注：
- 🔴 **必须**：不懂就没法学
- 🟡 **建议**：懂了更顺畅
- 🟢 **了解**：用到再查

---

# 第一块：数学基础

## 1. 高中数学（🔴 必须，但只需要一部分）

### 1.1 向量

**要会什么：**

```
向量加法：    (1,2) + (3,4) = (4,6)
数乘：        2 × (1,2) = (2,4)
点积：       (1,2)·(3,4) = 1×3 + 2×4 = 11
模长：       ||(3,4)|| = √(3²+4²) = 5
夹角：       cosθ = (a·b) / (||a||·||b||)
叉积（3D）：  (1,0,0) × (0,1,0) = (0,0,1)
```

**为什么 SLAM 需要：**

| 用途 | 例子 |
|------|------|
| 表示 3D 点 | 地图点 = (X, Y, Z) |
| 表示平移 | 相机位置 = (x, y, z) |
| 计算距离 | 两点距离 = 向量差的模 |
| 判断方向 | 点积判断前后/左右 |

**怎么练：**
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)          # 加法
print(np.dot(a, b))   # 点积
print(np.linalg.norm(a))  # 模长
```

### 1.2 三角函数

**要会什么：**

```
sin, cos, tan 的定义和图像
sin²θ + cos²θ = 1
弧度制：π rad = 180°
atan2(y, x)：比 arctan 更好，能处理四个象限
```

**为什么 SLAM 需要：**

| 用途 | 说明 |
|------|------|
| 旋转 | 绕轴旋转用 sin/cos |
| 角度表示 | 欧拉角 (roll, pitch, yaw) |
| 极坐标 | 从像素坐标反算方向向量 |
| 三角化 | 从角度差算距离 |

### 1.3 指数、对数、幂运算

**要会什么：**

```
e^x, ln(x), x^a
指数函数的增长特性
对数能把乘法变加法：ln(ab) = ln(a) + ln(b)
```

**为什么 SLAM 需要：**

指数和对数在**李群李代数**中大量出现——这是 SLAM 优化中表示旋转的核心工具（以后会学到）。

---

## 2. 线性代数（🔴 必须，最重要）

线性代数是 SLAM 的**第一语言**。没有它，寸步难行。

### 2.1 矩阵基础

**定义：** 一个 $m \times n$ 的矩阵是一个有 $m$ 行 $n$ 列的数表。

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{bmatrix}
\quad (2 \times 3 \text{矩阵})
$$

**要会什么：**

```
矩阵加法：形状必须相同，对应位置相加
数乘：每个元素乘同一个数
转置 Aᵀ：行变列，列变行
乘法 A×B：A的列数必须等于B的行数
  (m×n) × (n×p) = (m×p)
单位矩阵 I：对角线为1，其余为0
矩阵的逆 A⁻¹：A × A⁻¹ = I
```

**怎么练：**

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A + B)
print(A @ B)      # 矩阵乘法
print(A.T)        # 转置
print(np.linalg.inv(A))  # 逆矩阵
```

**为什么 SLAM 需要：**

| 用途 | 矩阵形状 | 含义 |
|------|---------|------|
| 相机内参 K | 3×3 | 焦距、主点 |
| 旋转矩阵 R | 3×3 | 旋转 |
| 变换矩阵 T | 4×4 | 旋转 + 平移 |
| 本质矩阵 E | 3×3 | 两视图关系 |
| 协方差矩阵 Σ | d×d | 不确定性 |
| Hessian 矩阵 H | n×n | 优化 |

### 2.2 行列式

**定义：** 方阵 $A$ 的行列式 $\det(A)$ 是一个标量，表示矩阵的"缩放因子"。

**要会什么：**

```python
A = np.array([[1, 2], [3, 4]])
print(np.linalg.det(A))  # -2.0
```

**为什么 SLAM 需要：**
- 行列式 = 0 → 矩阵不可逆 → 退化情况（要避免）
- 旋转矩阵的行列式 = 1（这是约束条件）
- 特征值和行列式有关系

### 2.3 特征值与特征向量

**定义：** 对矩阵 $A$，如果 $Av = \lambda v$（$v \neq 0$），则 $\lambda$ 是特征值，$v$ 是特征向量。

**几何意义：** 特征向量是矩阵变换的"主轴方向"，特征值是那个方向的"缩放倍数"。

```python
A = np.array([[2, 1], [1, 2]])
eigvals, eigvecs = np.linalg.eig(A)
print("特征值:", eigvals)       # [3, 1]
print("特征向量:\n", eigvecs)   # [[0.707, -0.707], [0.707, 0.707]]
```

**为什么 SLAM 需要：**
- 协方差矩阵的特征分解 → 数据的主轴方向（PCA）
- 卡尔曼滤波中的协方差更新
- 矩阵的谱分解

### 2.4 奇异值分解 SVD

**定义：** 任意矩阵 $A$ 可以分解为 $A = U\Sigma V^T$

```
A (m×n) = U (m×m) × Σ (m×n) × Vᵀ (n×n)
          正交矩阵  对角阵(奇异值)  正交矩阵
```

```python
U, S, Vt = np.linalg.svd(A)
```

**为什么 SLAM 需要：** SVD 是 SLAM 中使用频率最高的矩阵分解，没有之一。

| 用途 | 原理 |
|------|------|
| 求解最小二乘 | $Ax=0$ 的解是 $V$ 的最后一列 |
| 本质矩阵分解 | 从 E 恢复 R,t |
| 三角化 | 解 $Ax=0$ 求 3D 点 |
| 基础矩阵估计 | 八点法 |
| PCA/降维 | 数据压缩 |
| 伪逆 | $A^+ = V\Sigma^+U^T$ |

**你必须记住：** SVD 是 MVP（最有价值算法）。

### 2.5 最小二乘

**问题：** 方程 $Ax = b$ 可能没有精确解（方程太多，未知数太少）。

**解法：** 找一个 $x$ 让误差 $\|Ax - b\|^2$ 最小。

**正规方程解：** $x = (A^TA)^{-1}A^Tb$

```python
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

**为什么 SLAM 需要：** SLAM 几乎所有问题都是最小二乘问题。

### 2.6 推荐学习顺序

```
1. 矩阵基本操作（加、乘、转置）
2. 解线性方程组 Ax = b
3. 逆矩阵、行列式
4. 特征值和特征向量
5. SVD（先会用，再理解推导）
6. 最小二乘
```

**推荐资源：**
- 3Blue1Brown 线性代数系列（B站，直观几何理解）
- MIT 18.06 线性代数（Gilbert Strang）
- 练习：用 numpy 实现所有操作

---

## 3. 概率与统计（🔴 必须）

### 3.1 基础概念

| 概念 | 公式 | SLAM 对应 |
|------|------|-----------|
| 概率 $P(A)$ | 事件发生的可能性 | 传感器读数正确的概率 |
| 条件概率 $P(A\|B)$ | B 发生下 A 的概率 | 已知地图，看到特征的概率 |
| 贝叶斯公式 | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ | SLAM 的核心思想 |
| 期望 $E[X]$ | 平均值 | 估计的位置 |
| 方差 $Var[X]$ | 分散程度 | 位置的不确定性 |
| 协方差 $Cov[X,Y]$ | 两个变量的关系 | 误差的相关性 |
| 正态分布 $\mathcal{N}(\mu, \sigma^2)$ | 最常见的分布 | SLAM 默认假设 |

**贝叶斯公式——SLAM 最核心的数学思想：**

$$
P(\text{位姿} \mid \text{观测}) = \frac{P(\text{观测} \mid \text{位姿}) P(\text{位姿})}{P(\text{观测})}
$$

翻译成人话：**有了观测数据后，更新对位姿的信念。**

### 3.2 多元正态分布

SLAM 中处理的不再是单个变量，而是**多维向量**。

$$
\mathcal{N}(\mu, \Sigma)
$$

其中 $\mu$ 是 $d$ 维均值向量，$\Sigma$ 是 $d \times d$ 协方差矩阵。

```python
from scipy.stats import multivariate_normal

mean = [0, 0]
cov = [[1, 0.5], [0.5, 2]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
```

**为什么 SLAM 需要：**
- 所有 SLAM 系统假设噪声服从多元正态分布
- 卡尔曼滤波假设状态和观测都是正态分布
- BA 的误差函数可以理解为负对数正态似然

### 3.3 最大似然估计（MLE）

**思想：** 找到参数 $\theta$，使得观测数据出现的**可能性最大**。

$$
\hat{\theta}_{MLE} = \arg\max_\theta P(\text{数据} \mid \theta)
$$

**在 SLAM 中：**

最小化重投影误差 = 最大化观测的似然（假设高斯噪声）。

$$
\min \|误差\|^2 \iff \max P(\text{观测} \mid \text{位姿}, \text{点})
$$

---

## 4. 微积分（🟡 建议掌握）

### 4.1 需要到什么程度

不需要精通 ε-δ 语言和复杂的积分技巧，但必须理解：

```
导数：函数在某点的变化率
  几何意义：切线斜率
  物理意义：速度

偏导数：多元函数对某个变量的导数
  把其他变量当常数求导

梯度∇f：所有偏导数组成的向量
  几何意义：函数增长最快的方向
```

```python
# 数值求导（理解原理就行）
def derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)
```

### 4.2 为什么要会这些

SLAM 的**核心是优化**，优化的**核心是求导**：

| 技术 | 需要的微积分 |
|------|------------|
| PnP 位姿估计 | 投影函数对位姿的导数 |
| 束调整 (BA) | 重投影误差对位姿和 3D 点的导数 |
| 高斯-牛顿法 | 一阶泰勒展开 |
| LM 算法 | 一阶泰勒展开 + 阻尼因子 |
| 卡尔曼滤波 | 状态转移的雅可比 |

### 4.3 矩阵求导（重要！）

SLAM 中求导的对象不是标量，是**向量对向量**：

$$
\frac{\partial f}{\partial x} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

这叫**雅可比矩阵** $J$，是 SLAM 源码中出现频率最高的数学对象。

```python
# 雅可比矩阵的数值验证
def jacobian_numerical(f, x, h=1e-6):
    """f: Rⁿ→Rᵐ, x: Rⁿ, 返回 m×n 雅可比"""
    f0 = f(x)
    m, n = len(f0), len(x)
    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        J[:, i] = (f(x_plus) - f0) / h
    return J
```

### 4.4 泰勒展开

**一阶近似：**

$$
f(x + \Delta x) \approx f(x) + J \cdot \Delta x
$$

**二阶近似：**

$$
f(x + \Delta x) \approx f(x) + J \cdot \Delta x + \frac{1}{2} \Delta x^T H \Delta x
$$

其中 $H$ 是 Hessian 矩阵（二阶导矩阵）。

**高斯-牛顿法**就是用一阶近似 LM 算法求解最小二乘的核心工具。

---

## 5. 优化理论（🟡 建议掌握）

### 5.1 无约束优化

**问题：** 找一个 $x$ 让 $f(x)$ 最小。

**梯度下降法：**
$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

**高斯-牛顿法（专门用于最小二乘）：**
$$
(J^T J) \Delta x = -J^T r
$$

**LM 算法（高斯-牛顿 + 梯度下降的混合体）：**
$$
(J^T J + \lambda I) \Delta x = -J^T r
$$

### 5.2 为什么 LM 是最常用的？

```
误差很大时 → λ 大 → 接近梯度下降（慢但可靠）
误差很小时 → λ 小 → 接近高斯-牛顿（快但易发散）
自动调节 λ → 又快又稳
```

SLAM 中的**束调整**就是用 LM 算法的。

---

# 第二块：编程基础

## 6. Python 基础（🔴 必须）

### 6.1 你需要会什么

```
变量与类型：int, float, list, dict, tuple
控制流：if, for, while
函数：def, return, 参数
类：class, __init__, self
文件读写：open, read, write
异常处理：try, except
```

### 6.2 不太需要（但以后会用到）

```
装饰器、元类、生成器、协程
这些在 SLAM 原型里几乎不用
```

### 6.3 练习

```python
# 你应该能轻松写出这样的代码
def compute_mean(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx**2 + dy**2 + dz**2)**0.5
```

---

## 7. NumPy（🔴 必须，SLAM 的"母语"）

NumPy 是 Python 的科学计算库，SLAM 代码里 90% 的操作都是 NumPy 操作。

### 7.1 基础

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])              # 1D
b = np.array([[1, 2], [3, 4]])       # 2D
c = np.zeros((3, 3))                 # 全零
d = np.eye(3)                        # 单位矩阵
e = np.random.randn(100)             # 正态分布随机数

# 形状操作
a.shape          # 形状
a.reshape(3, 1)  # 重塑
a.T              # 转置

# 数学运算
a + b            # 加法
a @ b            # 矩阵乘法
np.dot(a, b)     # 点积
np.linalg.norm(a)  # 模长
np.linalg.inv(a)   # 逆矩阵
np.linalg.eig(a)   # 特征分解
np.linalg.svd(a)   # SVD
```

### 7.2 广播（broadcasting）

这是 NumPy 最强大但也最容易混淆的特性：

```python
a = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
b = np.array([10, 20])                   # (2,)
print(a + b)  # b 自动变成 (3,2)，每行加 [10,20]
```

### 7.3 索引和切片

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[0, 1])       # 第0行第1列 = 2
print(a[:, 0])       # 所有行的第0列 = [1, 4, 7]
print(a[0:2, :])     # 前两行
print(a[a > 5])      # 布尔索引 = [6, 7, 8, 9]
```

### 7.4 你会天天用的操作

```python
# SLAM 中最常见的 NumPy 操作
K = np.eye(3)            # 相机内参
R = np.eye(3)            # 旋转矩阵
t = np.zeros(3)          # 平移向量
T = np.eye(4)            # 变换矩阵
T[:3, :3] = R
T[:3, 3] = t

# 3D 点变换
P_w = np.array([1, 2, 3])
P_c = R @ P_w + t        # 世界→相机

# 投影到像素
p = K @ P_c
u = p[0] / p[2]          # 归一化
v = p[1] / p[2]
```

### 7.5 推荐练习

```
1. 生成 1000 个随机 3D 点
2. 对它们施加旋转 + 平移
3. 投影到 2D 图像平面
4. 加入高斯噪声
5. 可视化原始和投影后的点

这就是 SLAM 前端的核心操作！
```

---

## 8. OpenCV 基础（🔴 必须）

### 8.1 图像操作

```python
import cv2

# 读写
img = cv2.imread('image.jpg')
cv2.imwrite('output.jpg', img)

# 基本属性
h, w = img.shape[:2]  # 高度，宽度
dtype = img.dtype      # uint8, float32

# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 缩放
small = cv2.resize(img, (640, 480))

# 画图
cv2.circle(img, (100, 100), 5, (0, 255, 0), -1)
cv2.line(img, (0,0), (100,100), (255,0,0), 2)
cv2.putText(img, 'hello', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
```

### 8.2 特征提取

```python
# ORB 特征（SLAM 的主力）
orb = cv2.ORB_create(nfeatures=1000)
kp, desc = orb.detectAndCompute(gray, None)

# 可视化关键点
kp_img = cv2.drawKeypoints(gray, kp, None)
```

### 8.3 特征匹配

```python
# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(desc1, desc2, k=2)

# 比率测试（剔除误匹配）
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# 可视化
match_img = cv2.drawMatches(img1, kp1, img2, kp2, good[:50], None)
```

### 8.4 几何计算

```python
# PnP 位姿估计
_, rvec, tvec, inliers = cv2.solvePnPRansac(
    pts_3d, pts_2d, K, dist_coeffs
)

# 三角化
pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
pts_3d = pts_4d[:3] / pts_4d[3]

# 本质矩阵分解
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

---

## 9. Matplotlib 可视化（🟡 建议）

在 SLAM 开发中，**看不到结果就没法 debug**。

```python
import matplotlib.pyplot as plt

# 2D 图
plt.plot(x_data, y_data, 'b-', label='trajectory')
plt.scatter(x, y, c='r', s=10)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.grid(True)
plt.show()

# 3D 图
ax = plt.figure().add_subplot(111, projection='3d')
ax.plot(traj[:,0], traj[:,1], traj[:,2])
ax.scatter(points[:,0], points[:,1], points[:,2], s=1)

# 多图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(image)
axes[1].scatter(u, v)
```

**可视化对 SLAM 的意义：**

```
你能看到轨迹 → 知道有没有漂移
你能看到匹配 → 知道匹配质量
你能看到误差 → 知道优化效果
你能看到点云 → 知道地图质量

看不见 → 只能猜 → 永远 debug 不出来
```

---

# 第三块：计算机视觉基础

## 10. 图像基础（🔴 必须）

### 10.1 图像是什么

图像就是**数字矩阵**：

- **灰度图**：$H \times W$ 的矩阵，每个值 0（黑）~ 255（白）
- **彩色图**：$H \times W \times 3$，三个通道 R, G, B

```python
# 一张 640×480 的灰度图 = 480 行 × 640 列
img = np.zeros((480, 640), dtype=np.uint8)  # 全黑
img[100:200, 200:400] = 255  # 白色矩形
```

### 10.2 图像特征

**什么是特征？** 图像中"与众不同"的点——角点、边缘、斑点。

**为什么 SLAM 需要特征？**

```
特征 = 图像中的"地标"
   ┌ 可重复检测（在不同视角下都能找到）
   ├ 有唯一描述（可以匹配）
   └ 位置精确（可以算几何）
```

**常见的特征：**

| 特征 | 速度 | 精度 | 专利 | SLAM 使用 |
|------|------|------|------|-----------|
| ORB | ⭐⭐⭐ | ⭐⭐ | 免费 | ORB-SLAM |
| SIFT | ⭐ | ⭐⭐⭐ | 收费 | 离线重建 |
| SURF | ⭐⭐ | ⭐⭐ | 收费 | 历史 |
| SuperPoint | ⭐⭐ | ⭐⭐⭐ | 免费（非商业）| 新研究 |

### 10.3 图像坐标系

```
  u (列/水平方向) →
v  ┌──────────────┐
(  │              │
行 │   图像      │
/  │              │
竖 │              │
直 │              │
)  └──────────────┘
  ↓
```

**关键：** 图像坐标 $(u, v)$ 的 $u$ 是列，$v$ 是行，**不是**数学上的 $(x, y)$。

---

## 11. 相机模型（🔴 必须）

### 11.1 针孔相机模型

这是 SLAM 的基本假设：

```
          3D 点 P = (X, Y, Z)
                │
                │
    ┌───────────┴───────────┐
    │    针孔 (光心 O)      │
    │                       │
    └───────────┬───────────┘
                │
                ▼
         图像平面 p = (u, v)
```

**投影公式（最重要的公式，没有之一）：**

$$
u = f_x \cdot \frac{X}{Z} + c_x
$$
$$
v = f_y \cdot \frac{Y}{Z} + c_y
$$

**矩阵形式（用内参矩阵 $K$）：**

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 
\underbrace{\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}}_{K}
\underbrace{\begin{bmatrix}
R & t
\end{bmatrix}}_{\text{外参}}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

### 11.2 每个参数的意义

| 参数 | 名称 | 影响 |
|------|------|------|
| $f_x, f_y$ | 焦距（像素） | 越大→图像越"放大"，视场角越小 |
| $c_x, c_y$ | 主点 | 光轴与图像的交点，通常在图像中心附近 |
| $R$ | 旋转矩阵 | 相机朝向 |
| $t$ | 平移向量 | 相机位置 |

### 11.3 畸变

真实相机有畸变，图像会变形：

```
无畸变                    径向畸变（桶形）
┌──────────────┐         ┌──────────────┐
│              │         │   ╭──────╮   │
│  直线是直的  │   →    │  ╱        ╲  │
│              │         │ ╲        ╱  │
└──────────────┘         │   ╰──────╯   │
                          └──────────────┘
```

用 OpenCV 去畸变：

```python
undistorted = cv2.undistort(img, K, dist_coeffs)
```

### 11.4 你必须能徒手写出的代码

```python
def project_points(points_3d, K, R, t):
    """
    points_3d: (N, 3) 世界坐标
    K: (3, 3) 内参
    R: (3, 3) 旋转
    t: (3,) 平移
    返回: (N, 2) 像素坐标
    """
    # 世界 → 相机
    pts_cam = (R @ points_3d.T + t.reshape(3, 1)).T
    # 透视除法
    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)
```

---

## 12. 坐标系与变换（🔴 必须）

### 12.1 坐标系

SLAM 中有三个主要坐标系：

```
世界坐标系 W         相机坐标系 C          图像坐标系 I
(地图的基准)        (相机看到的世界)       (像素位置)
    │                    │                    │
    │  R_wc, t_wc        │  透视投影 K        │
    └─────────→ 3D ──────┘─────────→ 2D ─────┘
```

**数据流动方向：**

```
世界点 P_w
    │
    │ 旋转 + 平移：P_c = R × P_w + t
    ▼
相机点 P_c
    │
    │ 透视投影：u = fx × Xc/Zc + cx
    ▼
像素点 (u, v)
```

### 12.2 变换矩阵

**4×4 齐次变换矩阵**是 SLAM 的"世界语"：

$$
T = \begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0_{1\times3} & 1
\end{bmatrix}
$$

**用这个矩阵变换一个 3D 点：**

```python
def transform_point(T, p):
    """T: 4×4, p: (3,), 返回变换后的 (3,)"""
    p_h = np.append(p, 1.0)      # 齐次坐标
    p_t_h = T @ p_h              # 4×4 @ 4×1 = 4×1
    return p_t_h[:3] / p_t_h[3]  # 转回3D
```

### 12.3 链式变换

$$
T_{w \to c_2} = T_{c_1 \to c_2} \times T_{w \to c_1}
$$

```python
# 从帧1到帧2的相对运动
T_1_to_2 = np.linalg.inv(T_w_to_2) @ T_w_to_1
```

---

# 第四块：SLAM 概念准备

## 13. SLAM 是做什么的？（🟡 建议先理解全局）

### 13.1 一个简单的故事

想象你被蒙上眼睛走进一个房间：

```
你看不到房间的全貌（没有地图）
你只能靠触摸感知周围（传感器观测）
你边走边摸，要回答两个问题：
  1. 我现在在哪？（定位）
  2. 这个房间长什么样？（建图）
```

这就是 SLAM。

### 13.2 SLAM 的经典流程图

```
传感器数据
    │
    ▼
前端（视觉里程计 VO）
    │ 帧间跟踪，估计短时间内的运动
    │ 输出：相邻帧的位姿变换
    ▼
后端优化
    │ 全局优化，减小累积误差
    │ 输出：所有帧的一致位姿
    ▼
建图
    │ 将 3D 点组织成地图
    │ 输出：点云/栅格地图
    ▼
闭环检测
    │ 判断是否回到之前的位置
    │ 输出：闭环修正
```

### 13.3 SLAM 的核心难题

```
1. 数据关联：两帧中的特征是不是同一个 3D 点？
   错一个，后面的全错

2. 累积漂移：帧间误差一点点积累
   走 100m 可能漂移 1m，走 1000m 漂移 10m

3. 闭环：怎么知道回到之前的位置？
   误报（假阳性）= 灾难
   漏报（假阴性）= 继续漂移

4. 一致性：地图中同一个地方不能有两个表示
   需要融合重复的 3D 点
```

---

## 14. 对 SLAM 的正确心态

### 14.1 需要多少数学？

**不是数学家的水平也能学会 SLAM。** 关键在于：

```
认识公式 ≠ 理解公式
理解公式 ≠ 能推导
能推导 ≠ 能实现
能实现 ≠ 能调通
能调通 ≠ 能做准
```

你不需要一次性理解所有公式，但每遇到一个公式，要能回答：

- **它在算什么？**
- **输入输出是什么？**
- **如果没有它会怎样？**

### 14.2 正确的学习方式

```
第一步：运行别人的代码（理解输入输出）
     │
第二步：修改参数（看变化）
     │
第三步：逐行理解（每行代码都问：为什么这么写？）
     │
第四步：自己重写（不抄，凭理解写）
     │
第五步：改进（换特征、换数据集、加模块）
```

### 14.3 不要陷入的陷阱

```
❌ "我要把所有数学学完再动手"
   → 永远动不了手。先用起来，再补数学。

❌ "我要自己实现完整的 SLAM"
   → 循序渐进：从单帧 PnP → 帧间跟踪 → 局部BA → 完整系统

❌ "别人的代码太复杂，我要从头写"
   → 先改别人的代码跑通，再逐步替换模块

✅ 每一次迭代只替换一个模块，其他用现成的
```

---

# 附录：前置知识自查表

## 数学

```
[ ] 我能在 numpy 中做矩阵乘法
[ ] 我理解 SVD 能用来解 Ax=0
[ ] 我会计算向量的点积和模长
[ ] 我知道什么是协方差矩阵
[ ] 我知道雅可比矩阵是干什么的
[ ] 我听说过 LM 算法
```

## 编程

```
[ ] 我能用 Python 写一个类
[ ] 我会用 numpy 操作矩阵
[ ] 我能用 OpenCV 读/写/显示图像
[ ] 我会用 ORB 提取特征并匹配
[ ] 我会用 matplotlib 画 3D 图
```

## 计算机视觉

```
[ ] 我知道投影公式 u = fx*X/Z + cx
[ ] 我知道 K, R, t 分别是什么
[ ] 我能将 3D 点投影到 2D
[ ] 我知道什么是特征点和描述子
[ ] 我听说过对极约束
```

## SLAM 入门

```
[ ] 我知道 SLAM 是 Simultaneous Localization And Mapping
[ ] 我知道前端和后端的大致分工
[ ] 我知道关键帧是干什么的
[ ] 我听说过束调整（Bundle Adjustment）
[ ] 我听说过闭环检测（Loop Closure）
```

**如果上述有超过 5 项答不上来，就从对应该项的前置知识开始补。**

---

# 资源推荐

## 数学

| 资源 | 适合 | 链接 |
|------|------|------|
| 3Blue1Brown 线性代数 | 几何直觉 | B站搜索 |
| 3Blue1Brown 微积分 | 几何直觉 | B站搜索 |
| MIT 18.06 线性代数 | 系统学习 | YouTube |
| The Matrix Cookbook | 查阅矩阵公式 | 免费 PDF |

## SLAM 入门

| 资源 | 类型 | 说明 |
|------|------|------|
| 《视觉SLAM十四讲》 | 书 | 最适合中文读者，从零开始 |
| ORB-SLAM2 源码 | 代码 | 最经典的 SLAM 实现 |
| TUM RGB-D 数据集 | 数据 | 最常用的 SLAM 测试集 |
| SLAM题库（知乎/CSDN） | 社区 | 遇到问题先搜索 |

> **最后一句：SLAM 不难，只是东西多。把每个"零件"独立学会，再组装起来，你就能理解整个系统。**
