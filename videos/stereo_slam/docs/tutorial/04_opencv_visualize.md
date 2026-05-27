# 04 - OpenCV 与可视化：看到你的算法在做什么

> 目标：会用 OpenCV 处理图像、提取特征，用 matplotlib 看结果

---

## 4.1 图像的本质 🔴

### 4.1.1 图像 = 矩阵

在计算机里，图像就是一个数字矩阵。

```
灰度图：   H × W 矩阵         每个像素 0（黑）~ 255（白）
彩色图：   H × W × 3 矩阵     每个像素 3 个值 (B, G, R)
```

```python
import cv2
import numpy as np

# 创建一个 100×200 的黑色灰度图
img = np.zeros((100, 200), dtype=np.uint8)
# 画一个白色矩形
img[20:80, 40:160] = 255

# 彩色图
color = np.zeros((100, 200, 3), dtype=np.uint8)
color[:, :, 0] = 255  # B 通道全蓝
```

### 4.1.2 坐标注意 ⚠️

```python
img.shape          # (行=高, 列=宽) = (H, W)
h, w = img.shape[:2]

# 访问像素：img[行, 列] 不是 img[x, y]！
pixel = img[50, 30]      # 第50行，第30列

# OpenCV 颜色通道是 BGR，不是 RGB！
blue = img[50, 30, 0]    # B
green = img[50, 30, 1]   # G
red = img[50, 30, 2]     # R
```

---

## 4.2 OpenCV 基础操作 🔴

### 4.2.1 读写显示

```python
# 读
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度
img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)       # 彩色

# 写
cv2.imwrite('output.png', img)

# 显示（在 Jupyter 中用 matplotlib）
cv2.imshow('window', img)
cv2.waitKey(0)         # 等按键
cv2.destroyAllWindows()
```

**⚠️ OpenCV 的 `imread` 不支持中文路径**，需要用：

```python
def imread_unicode(path):
    """支持中文路径的图片读取"""
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
```

### 4.2.2 预处理

```python
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 缩放
small = cv2.resize(img, (640, 480))

# CLAHE 增强对比度（对红外图特别有用）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# 高斯模糊去噪
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
```

### 4.2.3 画图

```python
# 画点
cv2.circle(img, (u, v), 3, (0, 255, 0), -1)   # 绿色实心圆

# 画线
cv2.line(img, (u1, v1), (u2, v2), (0, 0, 255), 2)

# 画文字
cv2.putText(img, 'hello', (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
```

---

## 4.3 ORB 特征提取和匹配（SLAM 核心！）🔴

### 4.3.1 特征提取

```python
# 创建 ORB 检测器
orb = cv2.ORB_create(
    nfeatures=1000,       # 提取多少个特征
    scaleFactor=1.2,      # 金字塔缩放比例
    nlevels=8,            # 金字塔层数
)

# 检测关键点 + 计算描述子
keypoints, descriptors = orb.detectAndCompute(gray, None)

# 关键点信息
for kp in keypoints[:5]:
    print(f"位置: ({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), "
          f"大小: {kp.size:.1f}, 角度: {kp.angle:.1f}°")
```

### 4.3.2 特征匹配

```python
# 暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 用 k=2 的 knnMatch 做比率测试
matches = bf.knnMatch(desc1, desc2, k=2)

# 比率测试：剔除模糊匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"总匹配: {len(matches)}, 好匹配: {len(good_matches)}")
```

### 4.3.3 💡 比率测试为什么有效

```
对于左图的特征点：
  m：右图中最像的点（距离最近）
  n：右图中第二像的点

如果 m.distance << n.distance：匹配很明确，保留
如果 m.distance ≈ n.distance：匹配模糊，不要
阈值 0.75 是经验值，可以调
```

### 4.3.4 可视化匹配

```python
match_img = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

---

## 4.4 PnP 与三角化 🔴

```python
# PnP：已知 3D-2D，求位姿
_, rvec, tvec, inliers = cv2.solvePnPRansac(
    pts_3d,           # (N, 3) 世界坐标
    pts_2d,           # (N, 2) 像素坐标
    K,                # 3×3 内参
    dist_coeffs,      # 畸变参数
    iterationsCount=200,
    reprojectionError=8.0,
    confidence=0.99,
)
R, _ = cv2.Rodrigues(rvec)  # 旋转向量 → 旋转矩阵

# 三角化：已知两帧位姿和匹配点，求 3D
pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
pts_3d = pts_4d[:3] / pts_4d[3]  # 齐次→非齐次
```

---

## 4.5 用 matplotlib 看结果 🔴

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D 图
plt.plot(x_data, y_data, 'b-', label='trajectory')
plt.scatter(x, y, c='r', s=10)
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.grid(True)
plt.axis('equal')

# 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           c='g', s=1, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 保存（不需要显示器）
plt.savefig('output.png', dpi=100, bbox_inches='tight')
plt.close()
```

### 💡 调试时的最佳做法

```
1. 特征匹配 → 画出连线，看误匹配多不多
2. PnP 结果 → 画重投影误差分布
3. 轨迹 → 画在 3D 图上，看平滑不平滑
4. 地图点 → 用不同颜色表示不同深度
```

---

## ✅ 自测

```
1. OpenCV 中彩色图像的通道顺序是什么？
2. ORB_create 的 nfeatures 参数控制什么？
3. 比率测试的 0.75 是什么意思？
4. cv2.Rodrigues 是做什么的？
5. 如何用 matplotlib 画 3D 轨迹？
```

**代码练习：**

```python
# 读取两张图片
# 提取 ORB 特征并匹配
# 可视化匹配结果
# 统计好匹配和坏匹配的数量
```
