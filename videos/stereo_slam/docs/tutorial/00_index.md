# SLAM 从零入门教程

> 起点：大一新生水平
> 终点：能理解 SLAM 代码、自己动手改参数、调模块

---

## 教程结构

```
tutorial/
│
├── 第一部分：数学基础
│   ├── 01_linear_algebra.md       ← 先看！最重要的基础
│   └── 02_probability_optimization.md
│
├── 第二部分：编程工具
│   ├── 03_python_numpy.md         ← 先过一遍
│   └── 04_opencv_visualize.md     ← 边学边试
│
├── 第三部分：计算机视觉
│   ├── 05_camera_geometry.md      ← 核心！
│   └── 06_features_matching.md    ← 核心！
│
├── 第四部分：SLAM 核心算法
│   ├── 07_slam_frontend.md        ← 前端：PnP + 三角化 + 跟踪
│   └── 08_slam_backend.md         ← 后端：BA + 闭环 + 系统
│
└── 第五部分：动手实践
    └── 09_implementation.md       ← 动手：一步步搭建 SLAM
```

---

## 学习顺序

### 路线 A：快速上手（推荐）
```
01_linear_algebra → 03_python_numpy → 05_camera_geometry → 09_implementation
```
然后缺什么补什么。

### 路线 B：系统学习
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

### 路线 C：以代码为中心
```
03 → 04 → 打开 main.py 跑通 → 遇到不懂的公式去 01/02/05 查
```

---

## 教程约定

```
🔴 必须掌握
🟡 建议了解
🟢 以后再说
💡 核心直觉
⚠️ 常见的坑
✅ 验证方法
```

每个小节的末尾都有一个 ✅ 自测题，答不上来说明需要复习。

---

## 如何查资料

遇到不懂的概念：

1. 先在这个教程里搜（按文件名找对应的模块）
2. 如果没找到，去《视觉SLAM十四讲》对应章节
3. 再搜不到，把概念名 + SLAM 作为关键词 Google
4. 最后：看 ORB-SLAM2 源码里怎么用这个概念的
