"""
高级可视化工具 — 直方图、特征匹配、分割叠加、检测框绘制
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

# —— 中文字体配置 ——
# 自动检测系统可用中文字体
_CN_FONT = None
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Source Han Sans SC",
]

for _f in _CN_FONT_CANDIDATES:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        _CN_FONT = _f
        break
    except Exception:
        continue

if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_histogram(image, title="直方图", figsize=(10, 4)):
    """
    绘制RGB三通道直方图。
    自动检测灰度图并切换为单通道显示。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if len(image.shape) == 2 or image.shape[2] == 1:
        # 灰度图
        if image.dtype == np.uint8:
            ax.hist(image.ravel(), bins=256, range=(0, 255), color="gray", alpha=0.7)
        else:
            ax.hist(image.ravel(), bins=256, range=(0, 1), color="gray", alpha=0.7)
        ax.set_xlabel("像素值")
        ax.set_ylabel("频数")
    else:
        colors = ("red", "green", "blue")
        for i, color in enumerate(colors):
            channel = image[:, :, i]
            if image.dtype == np.uint8:
                ax.hist(
                    channel.ravel(),
                    bins=256,
                    range=(0, 255),
                    color=color,
                    alpha=0.5,
                    label=f"{color.upper()} 通道",
                )
            else:
                ax.hist(
                    channel.ravel(),
                    bins=256,
                    range=(0, 1),
                    color=color,
                    alpha=0.5,
                    label=f"{color.upper()} 通道",
                )
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("像素值")
    ax.set_ylabel("频数")
    return fig, ax


def plot_kernel_3d(kernel, title="卷积核3D可视化"):
    """将卷积核渲染为3D柱状图"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    h, w = kernel.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    ax.bar3d(
        X.ravel(),
        Y.ravel(),
        np.zeros_like(kernel).ravel(),
        1, 1,
        kernel.ravel(),
        shade=True,
        alpha=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("列")
    ax.set_ylabel("行")
    return fig, ax


def draw_matches(img1, img2, kp1, kp2, matches, max_draw=50):
    """
    可视化特征点匹配（用于SIFT/ORB匹配结果）。
    使用OpenCV的drawMatches。
    """
    import cv2

    # 确保是uint8格式
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (img2 * 255).astype(np.uint8)

    # BGR转换（如果是RGB的OpenCV图片）
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    drawn = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 转回RGB用于matplotlib显示
    drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
    return drawn


def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)):
    """
    在图片上叠加半透明分割mask。

    参数:
        image: RGB图片 (H, W, 3)
        mask: 二值mask (H, W)，True/1表示前景
        alpha: 透明度
        color: 叠加颜色 (R, G, B) 范围 [0,1]
    返回:
        叠加后的RGB图片 (H, W, 3)
    """
    image = image.astype(float)
    if image.max() > 1:
        image = image / 255.0

    mask = mask.astype(float)
    if mask.max() > 1:
        mask = mask / mask.max()

    overlay = np.zeros_like(image)
    for i in range(3):
        overlay[:, :, i] = mask * color[i]

    result = image * (1 - alpha * mask[:, :, np.newaxis]) + overlay * (alpha * mask[:, :, np.newaxis])
    return np.clip(result, 0, 1)


def draw_bounding_boxes(image, boxes, labels=None, scores=None, figsize=(10, 8)):
    """
    在图片上绘制目标检测框。

    参数:
        image: RGB图片
        boxes: 检测框列表，每个格式为 [x1, y1, x2, y2]（像素坐标）
        labels: 类别标签列表
        scores: 置信度列表
    返回:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # 预定义颜色
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(boxes), 1)))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]

        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # 标签和置信度
        if labels is not None and scores is not None:
            text = f"{labels[i]}: {scores[i]:.2f}"
        elif labels is not None:
            text = labels[i]
        elif scores is not None:
            text = f"{scores[i]:.2f}"
        else:
            text = ""

        if text:
            ax.text(
                x1, y1 - 5, text,
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

    ax.axis("off")
    plt.tight_layout()
    return fig, ax
