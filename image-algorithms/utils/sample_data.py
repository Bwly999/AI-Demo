"""
合成示例数据生成 — 用于不需要真实照片的教学演示场景
"""

import numpy as np
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.parent / "images"


def checkerboard(size=256, square_size=32):
    """生成棋盘格图案 — 用于几何变换演示"""
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, square_size * 2):
        for j in range(0, size, square_size * 2):
            board[i : i + square_size, j : j + square_size] = 255
            board[i + square_size : i + square_size * 2, j + square_size : j + square_size * 2] = 255
    # 转成3通道RGB
    return np.stack([board, board, board], axis=2)


def shapes_image(size=400):
    """
    生成包含圆形、矩形、三角形的合成图像 — 用于分割和形态学演示。

    返回:
        (rgb_image, ground_truth_mask)
    """
    from skimage.draw import disk, polygon, rectangle

    img = np.ones((size, size, 3), dtype=np.float32) * 0.95  # 浅灰背景
    mask = np.zeros((size, size), dtype=np.uint8)

    # 白色圆形
    rr, cc = disk((size // 4, size // 4), size // 8)
    img[rr, cc] = [1, 1, 1]
    mask[rr, cc] = 1

    # 灰色矩形
    r_start, c_start = size // 2, size // 2
    rr, cc = rectangle((r_start, c_start), extent=(size // 6, size // 5))
    img[rr, cc] = [0.7, 0.7, 0.7]
    mask[rr, cc] = 2

    # 深灰色三角形
    r_center, c_center = size * 3 // 4, size // 4
    tri_r = np.array([
        r_center - size // 10,
        r_center + size // 10,
        r_center - size // 10,
    ])
    tri_c = np.array([
        c_center - size // 12,
        c_center,
        c_center + size // 12,
    ])
    rr, cc = polygon(tri_r, tri_c)
    img[rr, cc] = [0.3, 0.3, 0.3]
    mask[rr, cc] = 3

    # 添加一些噪声
    noise = np.random.normal(0, 0.02, img.shape)
    img = np.clip(img + noise, 0, 1)

    return (img * 255).astype(np.uint8), mask


def sine_grating(size=256, frequency=5, angle=0):
    """
    生成正弦光栅 — 用于频域/滤波概念演示。

    参数:
        size: 图片大小
        frequency: 频率（条纹数）
        angle: 旋转角度（度）
    """
    x = np.arange(size) - size / 2
    y = np.arange(size) - size / 2
    X, Y = np.meshgrid(x, y)

    theta = np.radians(angle)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)

    grating = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * X_rot / size)
    return (grating * 255).astype(np.uint8)


def random_dots(size=256, n_dots=50):
    """生成随机点阵 — 用于目标跟踪演示"""
    img = np.zeros((size, size), dtype=np.uint8)
    positions = np.random.randint(10, size - 10, (n_dots, 2))
    for r, c in positions:
        rr, cc = (r, c)  # simplified single-pixel dots
        if 0 <= r < size and 0 <= c < size:
            img[0 <= r < size, 0 <= c < size] = 255
    # Use disk for better visibility
    img = np.zeros((size, size), dtype=np.uint8)
    for r, c in positions:
        from skimage.draw import disk
        rr, cc = disk((r, c), 3, shape=img.shape)
        img[rr, cc] = 255
    return img


def save_sample_images():
    """生成并保存所有合成示例图片到 images/synthetic/ 目录"""
    synthetic_dir = IMAGES_DIR / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    # 棋盘格
    cb = checkerboard()
    Image.fromarray(cb).save(synthetic_dir / "checkerboard.png")

    # 几何形状
    shapes_img, shapes_mask = shapes_image()
    Image.fromarray(shapes_img).save(synthetic_dir / "shapes.png")

    # 正弦光栅
    for freq in [3, 8, 16]:
        sg = sine_grating(frequency=freq)
        Image.fromarray(sg).save(synthetic_dir / f"grating_f{freq}.png")

    print(f"合成示例图片已保存到: {synthetic_dir}")


if __name__ == "__main__":
    save_sample_images()
