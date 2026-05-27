# 03 - Python 与 NumPy：把数学变成代码

> 目标：能熟练用 numpy 写 SLAM 中的矩阵运算

---

## 3.1 Python 基础（快速回顾）🟡

### 你只需要会这些

```python
# 变量和类型
x = 3.14                        # float
name = "hello"                   # str
points = [[1,2], [3,4]]         # list
data = {"key": "value"}         # dict

# 控制流
for i in range(10):
    if i > 5:
        print(i)

# 函数
def add(a, b):
    return a + b

# 类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 列表推导
squares = [x**2 for x in range(10)]

# 文件读写
with open("file.txt", "r") as f:
    content = f.read()
```

### 不太需要的

```
装饰器、元类、生成器、协程、多线程
SLAM 原型里几乎用不到，用到再学
```

---

## 3.2 NumPy 核心 🔴

### 3.2.1 创建数组

```python
import numpy as np

# 从列表创建
a = np.array([1, 2, 3])              # 1D 向量
b = np.array([[1, 2], [3, 4]])       # 2D 矩阵

# 常用创建方式
zeros = np.zeros((3, 3))             # 全零矩阵
ones = np.ones((2, 4))               # 全一矩阵
I = np.eye(3)                        # 单位矩阵
random = np.random.randn(100)        # 正态分布随机数
lin = np.linspace(0, 1, 10)          # 等差数列
```

### 3.2.2 形状操作

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape          # (2, 3) - 2行3列
a.reshape(3, 2)  # 变成 3×2
a.T              # 转置 → (3, 2)
a.flatten()      # 展平 → (6,)

len(a)           # 2（第一维长度）
a.ndim           # 2（维度数）
a.size           # 6（元素总数）
```

### 3.2.3 索引和切片

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a[0, 1]           # 第0行第1列 → 2
a[0]              # 第0行 → [1, 2, 3]
a[:, 0]           # 所有行的第0列 → [1, 4, 7]
a[0:2, :]         # 前两行
a[a > 5]          # 布尔索引 → [6, 7, 8, 9]
```

**⚠️ 重点：** SLAM 中大量使用切片来操作矩阵的部分元素。

```python
T = np.eye(4)          # 4×4 变换矩阵
T[:3, :3] = R          # 设置旋转部分
T[:3, 3] = t           # 设置平移部分
```

### 3.2.4 矩阵运算 🔴

```python
# 加减乘除（对应元素）
a + b
a - b
a * b          # 逐元素乘法（不是矩阵乘法！）
a / b

# 矩阵乘法 ⚠️ 用 @ 不是 *
A @ B          # 矩阵乘法
np.dot(A, B)   # 同上，老式写法

# 逐元素乘法用 *
A * B          # Hadamard 乘积

# 其他运算
np.linalg.inv(A)      # 逆矩阵
np.linalg.det(A)      # 行列式
np.linalg.norm(v)     # 向量模长
np.linalg.solve(A, b) # 解 Ax=b
```

**💡 矩阵乘法用 @，逐元素乘法用 \*，千万别搞混！**

### 3.2.5 广播（boadcasting）

NumPy 自动扩展小数组的维度以匹配大数组：

```python
a = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
b = np.array([10, 20])                   # (2,)
print(a + b)  # b 自动扩展为 [[10,20], [10,20], [10,20]]
```

**在 SLAM 中：**

```python
# 将所有 3D 点平移
points = np.random.randn(100, 3)    # 100 个 3D 点
translation = np.array([1, 2, 3])   # 平移向量
points_shifted = points + translation  # 广播！一行代码平移所有点

# 旋转所有点
R = np.eye(3)                        # 旋转矩阵
points_rotated = (R @ points.T).T   # 转置再转置回来
```

---

## 3.3 SLAM 中最常见的 NumPy 操作

### 3.3.1 变换 3D 点

```python
def transform_points(T, points):
    """
    T: 4×4 变换矩阵
    points: (N, 3) N 个 3D 点
    返回: (N, 3) 变换后的点
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T + t.reshape(3, 1)).T
```

### 3.3.2 投影到像素

```python
def project_points(K, points_cam):
    """
    K: 3×3 内参
    points_cam: (N, 3) 相机坐标系下的点
    返回: (N, 2) 像素坐标
    """
    u = K[0, 0] * points_cam[:, 0] / points_cam[:, 2] + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / points_cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)
```

### 3.3.3 计算重投影误差

```python
def reprojection_error(T, K, pts_3d, pts_2d):
    """
    计算所有点的重投影误差
    """
    pts_cam = transform_points(T, pts_3d)
    pts_proj = project_points(K, pts_cam)
    errors = pts_proj - pts_2d
    return np.sqrt(np.sum(errors**2, axis=1))
```

---

## 3.4 调试技巧

### 3.4.1 检查 NaN 和 Inf

```python
# SLAM 中矩阵运算可能出现 NaN
if np.any(np.isnan(T)):
    print("变换矩阵有 NaN！")

# 安全的逆矩阵
try:
    T_inv = np.linalg.inv(T)
except np.linalg.LinAlgError:
    print("矩阵不可逆！")
```

### 3.4.2 检查矩阵性质

```python
# 检查旋转矩阵是否有效
def is_valid_rotation(R):
    """检查 R 是不是有效的旋转矩阵"""
    I = R.T @ R                      # 应该是单位矩阵
    return np.allclose(I, np.eye(3), atol=1e-6) and \
           abs(np.linalg.det(R) - 1.0) < 1e-6
```

---

## 3.5 性能提示

```python
# ❌ 慢：循环
for i in range(N):
    result[i] = a[i] * b[i]

# ✅ 快：向量化
result = a * b

# ❌ 慢：在循环中 append
pts = []
for p in points_3d:
    pts.append(transform_point(T, p))

# ✅ 快：一次性矩阵运算
pts = (T[:3,:3] @ points_3d.T + T[:3,3:4]).T
```

**规则：绝对不要用 Python 循环遍历大量数据，用向量化操作。**

---

## ✅ 自测

```
1. 矩阵乘法用哪个运算符？逐元素乘用哪个？
2. 广播的规则是什么？
3. 如何获取矩阵的形状？
4. 如何安全地求一个矩阵的逆？
5. 如何一次变换 N 个 3D 点？
```

**代码练习：**

```python
# 生成 1000 个随机 3D 点
# 创建随机旋转矩阵 R 和平移 t
# 变换所有点
# 验证：逆变换后是否回到原点
```
