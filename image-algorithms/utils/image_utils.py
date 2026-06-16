"""
图像I/O工具 — 统一各库之间的图片加载、显示、保存接口
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 项目图片目录
IMAGES_DIR = Path(__file__).parent.parent / "images" / "samples"


def load_image(path):
    """
    统一图片加载：支持路径字符串或Path对象，返回RGB格式的NumPy数组。
    自动处理 BGR→RGB 转换（OpenCV）、RGBA→RGB 转换（Pillow）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")

    img = Image.open(path)
    # 统一转为RGB
    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    return np.array(img)


def load_sample(filename):
    """加载 images/samples/ 目录下的示例图片"""
    return load_image(IMAGES_DIR / filename)


def display_image(img, title="", ax=None, figsize=(6, 4)):
    """
    显示单张图片（RGB格式NumPy数组）。
    自动处理 float [0,1] 和 uint8 [0,255] 两种范围。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    return ax


def display_images(images, titles=None, cols=2, figsize=(6, 3)):
    """
    网格显示多张图片。

    示例:
        display_images([img1, img2, img3], titles=["原图", "高斯", "中值"], cols=2)
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))

    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    if titles is None:
        titles = [""] * n

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")

    # 隐藏多余的子图
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def compare_images(original, processed, title_orig="原图", title_processed="处理后"):
    """
    并排对比原图和处理后的图片。
    """
    return display_images(
        [original, processed],
        titles=[title_orig, title_processed],
        cols=2,
        figsize=(5, 4),
    )


def img_to_bytes(img):
    """将NumPy图片数组转为PNG字节流（用于Gradio返回）"""
    import io
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def ensure_uint8(img):
    """确保图片为uint8格式 [0,255]"""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img.astype(np.uint8)


def ensure_float(img):
    """确保图片为float格式 [0,1]"""
    if img.dtype == np.float64 or img.dtype == np.float32:
        return img.astype(np.float64)
    return img.astype(np.float64) / 255.0
