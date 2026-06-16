# 🖼️ 图像算法速成课 — 从经典到深度学习

> **80/20 原则**：用 20% 的核心知识覆盖 80% 的实际应用场景。

本课程涵盖传统图像处理（OpenCV/scikit-image）和深度学习计算机视觉（PyTorch/YOLO），共 **9 个模块**，每个模块包含理论讲解、代码实战、交互式 Demo 和真实应用案例。

---

## 📐 知识图谱

```
                        图像算法 80/20 核心知识
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   📷 经典图像处理         🧠 深度学习基础        🚀 实战应用
   (Modules 00-05)        (Modules 06-07)        (Module 08)
        │                     │                     │
   ┌────┼────┐           ┌────┼────┐           ┌────┼────┐
   │    │    │           │    │    │           │    │    │
 基础  滤波  特征       CNN  迁移   检测       YOLO 分割 跟踪
```

### 核心知识覆盖

| 层次 | 模块 | 核心技能 | 实际应用 |
|------|------|---------|---------|
| 基础 | 00 图像基础 | NumPy数组、颜色空间、像素操作 | 理解图像数据结构 |
| 经典 | 01 滤波增强 | 卷积、高斯/中值/双边滤波、直方图均衡、CLAHE | 去噪、美颜、老照片修复 |
| 经典 | 02 边缘特征 | Sobel/Canny/Laplacian、SIFT/ORB、FLANN+RANSAC | 全景拼接、图像配准 |
| 经典 | 03 几何变换 | 仿射/透视变换、单应矩阵、文档矫正 | 扫描全能王核心算法 |
| 经典 | 04 分割形态 | Otsu阈值、分水岭、GrabCut、腐蚀膨胀 | 物体计数、背景替换 |
| 经典 | 05 视频跟踪 | 光流法、CSRT/KCF跟踪器、背景减除 | 交通流量统计、运动分析 |
| DL | 06 CNN基础 | 卷积层/池化层/全连接层、PyTorch基础 | 手写数字识别 |
| DL | 07 迁移学习 | ResNet/VGG预训练、特征提取vs微调、数据增强 | 自定义图像分类器 |
| DL | 08 目标检测 | YOLO、Faster R-CNN、语义分割 | 行人检测、街景理解 |

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- pip
- （可选）NVIDIA GPU + CUDA（用于深度学习模块加速，CPU也可运行）

### 三步启动

```bash
# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # macOS/Linux

# 2. 安装依赖
cd image-algorithms
pip install -r requirements.txt

# 3. 启动 Jupyter
jupyter notebook
# 在浏览器中打开 notebooks/ 目录，从 00 开始学习
```

### 运行交互式 Demo

```bash
# 经典图像处理 Demo
python demos/filter_playground.py        # 滤波器画廊
python demos/edge_detector.py            # 边缘检测对比
python demos/panorama_stitcher.py        # 全景拼接
python demos/document_scanner.py         # 文档扫描仪
python demos/segmenter.py                # 图像分割
python demos/object_tracker.py           # 目标跟踪

# 深度学习 Demo
python demos/cnn_classifier.py           # CNN分类 + 热力图
python demos/object_detector.py          # YOLO目标检测
```

### 浏览器可视化 Demo

直接用浏览器打开：
- `html_demos/color-spaces.html` — 颜色空间交互式可视化
- `html_demos/kernel-visualizer.html` — 卷积核动态可视化

或用课程导航页：`course-index.html`

---

## 📚 模块目录

### 🔧 阶段一：经典图像处理

#### [00 — 环境准备与图像基础](notebooks/00_环境准备与图像基础.ipynb)
- 图像在计算机中的表示：NumPy数组
- RGB/HSV/Grayscale颜色空间
- 通道分解与像素操作
- 📝 **应用**：CT医学影像窗宽窗位调节原理

#### [01 — 滤波与图像增强](notebooks/01_滤波与图像增强.ipynb)
- 卷积的物理意义
- 高斯/中值/双边滤波
- 直方图均衡化 & CLAHE
- USM锐化
- 🎮 **Demo**：`filter_playground.py`
- 📝 **应用**：老照片修复流程

#### [02 — 边缘检测与特征匹配](notebooks/02_边缘检测与特征匹配.ipynb)
- Sobel/Canny/Laplacian边缘检测
- SIFT & ORB特征点提取
- FLANN特征匹配 + RANSAC
- 🎮 **Demo**：`edge_detector.py` + `panorama_stitcher.py`
- 📝 **应用**：全景图拼接

#### [03 — 几何变换实战](notebooks/03_几何变换实战.ipynb)
- 仿射变换（旋转/缩放/平移/剪切）
- 透视变换 & 单应矩阵
- 文档扫描完整流程
- 🎮 **Demo**：`document_scanner.py`
- 📝 **应用**：扫描全能王核心算法复现

#### [04 — 图像分割与形态学](notebooks/04_图像分割与形态学.ipynb)
- Otsu大津法 / 自适应阈值
- 分水岭算法 / GrabCut
- 腐蚀/膨胀/开闭运算
- 🎮 **Demo**：`segmenter.py`
- 📝 **应用**：工业零件自动计数与测量

#### [05 — 视频分析与目标跟踪](notebooks/05_视频分析与目标跟踪.ipynb)
- 稀疏光流 (Lucas-Kanade)
- 稠密光流 (Farneback)
- CSRT/KCF目标跟踪
- 背景减除
- 🎮 **Demo**：`object_tracker.py`
- 📝 **应用**：交通路口车流量统计

### 🧠 阶段二：深度学习基础

#### [06 — CNN基础与图像分类](notebooks/06_CNN基础与图像分类.ipynb)
- 卷积层/池化层/全连接层原理
- PyTorch基础 (Tensor, nn.Module, DataLoader)
- LeNet-5风格CNN构建
- MNIST & CIFAR-10训练
- 🎮 **Demo**：`kernel-visualizer.html`
- 📝 **应用**：手写数字识别

#### [07 — 迁移学习实战](notebooks/07_迁移学习实战.ipynb)
- ImageNet预训练模型 (ResNet/VGG/EfficientNet)
- 特征提取 vs 微调策略
- Albumentations数据增强
- Grad-CAM热力图可视化
- 🎮 **Demo**：`cnn_classifier.py`
- 📝 **应用**：宠物/植物品种识别

### 🚀 阶段三：实战应用

#### [08 — 目标检测与语义分割](notebooks/08_目标检测与语义分割.ipynb)
- YOLO实时目标检测
- Faster R-CNN原理
- DeepLabV3语义分割
- NMS非极大值抑制
- IoU/mAP评估指标
- 🎮 **Demo**：`object_detector.py`
- 📝 **应用**：街景全景理解

---

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 经典CV | OpenCV, scikit-image, Pillow | 滤波/特征/变换/分割/跟踪 |
| 数值计算 | NumPy, SciPy | 数组运算、科学计算 |
| 深度学习 | PyTorch, torchvision | CNN/迁移学习/目标检测 |
| 数据增强 | Albumentations | 训练数据扩充 |
| 目标检测 | Ultralytics YOLOv8 | 预训练检测模型推理 |
| 可视化 | Matplotlib, ipywidgets | 图表、交互式参数调节 |
| Web Demo | Gradio | 交互式Web应用 |
| 教学环境 | Jupyter Notebook | 代码+讲解一体化 |

---

## 📖 学习建议

### 按背景选择路径

| 你的背景 | 推荐路径 |
|---------|---------|
| 🆕 **零基础** | 00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08（按顺序） |
| 📷 **有CV基础** | 快速浏览 00-01 → 重点 02-05 → 深入 06-08 |
| 🧠 **有DL基础** | 快速浏览 06 → 重点 01-05（补经典CV）→ 07-08 |
| 🏃 **只想快速上手** | 每个模块只看"实际应用案例"部分 + 运行对应 Demo |

### 学习节奏建议

- **每天 1 个模块**（约 1-2 小时）：2 周完成
- **每周 2 个模块**（深度学习）：3 周完成
- 最重要的是：**动手运行代码 + 调参实验**

---

## 📦 项目结构

```
image-algorithms/
├── README.md                      # 本文件
├── requirements.txt               # 依赖清单
├── course-index.html              # 课程导航主页
│
├── images/samples/                # 示例图片
├── notebooks/                     # 📓 Jupyter Notebooks（9个）
│   ├── 00_环境准备与图像基础.ipynb
│   ├── 01_滤波与图像增强.ipynb
│   ├── 02_边缘检测与特征匹配.ipynb
│   ├── 03_几何变换实战.ipynb
│   ├── 04_图像分割与形态学.ipynb
│   ├── 05_视频分析与目标跟踪.ipynb
│   ├── 06_CNN基础与图像分类.ipynb
│   ├── 07_迁移学习实战.ipynb
│   └── 08_目标检测与语义分割.ipynb
│
├── demos/                         # 🖥️ Gradio 交互式应用（8个）
│   ├── filter_playground.py
│   ├── edge_detector.py
│   ├── panorama_stitcher.py
│   ├── document_scanner.py
│   ├── segmenter.py
│   ├── object_tracker.py
│   ├── cnn_classifier.py
│   └── object_detector.py
│
├── utils/                         # 🔧 共享工具模块
│   ├── image_utils.py
│   ├── visualization.py
│   └── sample_data.py
│
└── html_demos/                    # 🌐 浏览器可视化
    ├── color-spaces.html
    └── kernel-visualizer.html
```

---

## ❓ FAQ

**Q: 需要 GPU 吗？**
A: 不需要。所有代码都可以在 CPU 上运行。深度学习模块在 CPU 上会慢一些，但教学用的模型和数据量都很小，完全可接受。

**Q: 为什么同时学 Pillow、scikit-image 和 OpenCV？**
A: 三者在 NumPy 数组层面互通。Pillow 适合简单的 I/O 操作，scikit-image 的 API 更适合教学（清晰的函数名和参数），OpenCV 是工业界标准。掌握三者之间的关系是实际工作中的必备技能。

**Q: 为什么先教传统CV再教深度学习？**
A: 传统CV帮助你理解图像的底层表示和处理逻辑（为什么需要滤波、边缘检测的意义等），这些知识在调试深度学习模型时至关重要。此外，很多实际项目中传统CV和DL是混合使用的。

**Q: 可以在线运行 Notebook 吗？**
A: 可以。将 notebooks/ 目录上传到 Google Colab 或 Binder 即可在线运行。注意：Gradio Demo 需要本地运行。

**Q: 模型下载需要多久？**
A: 首次运行深度学习模块时：
- torchvision 预训练模型 (ResNet18 等)：~50MB
- YOLOv8n 模型：~6MB，首次运行自动下载
建议在网络良好的环境下首次运行。

---

## 📄 License

本教学项目仅供学习使用。使用的第三方库遵循各自的许可证。
