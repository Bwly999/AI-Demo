"""
图像分割 — 多种分割方法交互式对比

支持 Otsu 阈值、自适应阈值、分水岭算法、GrabCut，
可查看分割叠加、掩膜和物体计数。
"""

import gradio as gr
import numpy as np
import cv2
from skimage import data, filters, segmentation, measure, color
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ensure_uint8, img_to_bytes
from utils.visualization import overlay_mask, plot_histogram


# ── 默认示例图片 ────────────────────────────────────────────
def load_default_image():
    img = data.coins()
    return np.stack([img, img, img], axis=-1)


# ── 分割方法实现 ────────────────────────────────────────────

def otsu_threshold(image):
    """Otsu 全局阈值分割"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, binary


def adaptive_threshold(image, block_size=11, C=2):
    """自适应阈值分割"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    return binary


def watershed_segmentation(image, min_distance=20, compactness=0):
    """分水岭分割"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 梯度图
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 距离变换
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 找标记点
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 未知区域
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 执行分水岭
    img_color = image.copy()
    markers = cv2.watershed(img_color, markers)

    # 生成掩膜（标记区域 > 1）
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255

    # 标记可视化
    markers_viz = np.zeros_like(image)
    unique_labels = np.unique(markers)
    for label in unique_labels:
        if label <= 1:
            continue
        markers_viz[markers == label] = np.random.randint(0, 256, 3)

    return markers_viz, mask, markers


def grabcut_segmentation(
    image, rect_x1, rect_y1, rect_x2, rect_y2, iterations=5
):
    """GrabCut 分割"""
    img = ensure_uint8(image)
    h, w = img.shape[:2]

    # 确保矩形在图像范围内
    x1 = max(0, min(rect_x1, w - 1))
    y1 = max(0, min(rect_y1, h - 1))
    x2 = max(x1 + 1, min(rect_x2, w))
    y2 = max(y1 + 1, min(rect_y2, h))

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)

    result_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    return result_mask


# ── 主处理函数 ──────────────────────────────────────────────
def process_segmentation(
    image, method, block_size, C_value, min_distance,
    rect_x1, rect_y1, rect_x2, rect_y2, grabcut_iterations,
):
    """执行分割并返回所有可视化结果"""
    if image is None:
        return [None] * 6 + ["请先上传图片"]

    img = ensure_uint8(image)
    mask = None
    param_info = ""
    markers_viz = None
    otsu_value = 0

    if method == "Otsu阈值":
        otsu_value, mask = otsu_threshold(img)
        param_info = f"Otsu 自动计算阈值：{otsu_value}\n基于类间方差最大化自动选取最优阈值"

    elif method == "自适应阈值":
        mask = adaptive_threshold(img, block_size, C_value)
        param_info = (
            f"Block Size = {block_size}，C = {C_value}\n"
            "对每个像素取其邻域的加权均值作为阈值，适合光照不均匀的图像"
        )

    elif method == "分水岭":
        markers_viz, mask, markers = watershed_segmentation(img, min_distance)
        n_labels = len(np.unique(markers)) - 2  # 减去背景(0)和边界(1)
        param_info = (
            f"最小距离 = {min_distance}\n"
            f"检测到 {n_labels} 个独立区域\n"
            "分水岭算法将梯度图看作地形，从局部极小值开始'注水'直到不同盆地相遇"
        )

    elif method == "GrabCut":
        mask = grabcut_segmentation(
            img, rect_x1, rect_y1, rect_x2, rect_y2, grabcut_iterations,
        )
        param_info = (
            f"矩形 ROI：({rect_x1}, {rect_y1}) 到 ({rect_x2}, {rect_y2})\n"
            f"迭代次数：{grabcut_iterations}\n"
            "GrabCut 使用图割方法迭代优化前景/背景分割"
        )

    if mask is None:
        return [None] * 6 + ["分割失败"]

    # 物体计数
    labeled = measure.label(mask > 0)
    n_objects = labeled.max()

    # 叠加图
    overlay_img = overlay_mask(img.astype(float) / 255.0, mask > 0, alpha=0.4, color=(0, 1, 0))
    overlay_img = (overlay_img * 255).astype(np.uint8)

    # 纯掩膜
    mask_3ch = np.stack([mask, mask, mask], axis=-1) if len(mask.shape) == 2 else mask

    # 直方图
    hist_fig, _ = plot_histogram(img, title="原图 RGB 直方图")
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    hist_fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    buf.seek(0)
    plt.close(hist_fig)
    import PIL.Image
    hist_img = np.array(PIL.Image.open(buf))

    # 标记可视化（分水岭专用）
    if markers_viz is None:
        markers_viz = np.zeros((10, 10, 3), dtype=np.uint8)

    info_text = (
        f"分割方法：{method}\n"
        f"{param_info}\n"
        f"检测到物体数量：{n_objects}"
    )

    return (
        overlay_img,                    # 分割叠加
        mask_3ch if isinstance(mask_3ch, np.ndarray) else mask_3ch,  # 掩膜
        hist_img,                       # 直方图
        markers_viz,                    # 标记可视化
        info_text,                      # 参数信息
        str(n_objects),                 # 物体数
        img_to_bytes(mask_3ch),         # 下载掩膜
    )


# ── UI 布局 ─────────────────────────────────────────────────
with gr.Blocks(title="图像分割", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 图像分割
        使用不同的分割方法将图像分离为前景和背景。
        支持 Otsu 阈值、自适应阈值、分水岭和 GrabCut 四种方法。
        """
    )

    with gr.Row():
        # ── 左侧：控制面板 ──
        with gr.Column(scale=1):
            input_image = gr.Image(label="输入图片", type="numpy", height=250)
            with gr.Row():
                load_btn = gr.Button("加载默认图片", size="sm")
                clear_btn = gr.Button("清除图片", size="sm")

            method = gr.Dropdown(
                choices=["Otsu阈值", "自适应阈值", "分水岭", "GrabCut"],
                value="Otsu阈值",
                label="分割方法",
            )

            # 自适应阈值参数
            with gr.Accordion("> 自适应阈值参数", open=False):
                block_size = gr.Slider(3, 51, value=11, step=2, label="Block Size（邻域大小）")
                C_value = gr.Slider(-10, 20, value=2, step=1, label="C（常数偏移）")

            # 分水岭参数
            with gr.Accordion("> 分水岭参数", open=False):
                min_distance = gr.Slider(5, 100, value=20, step=1, label="最小距离")

            # GrabCut 参数
            with gr.Accordion("> GrabCut 参数", open=True):
                gr.Markdown("在图像上设置前景矩形 ROI 的坐标：")
                with gr.Row():
                    rect_x1 = gr.Slider(0, 600, value=50, step=1, label="X1（左）")
                    rect_y1 = gr.Slider(0, 600, value=50, step=1, label="Y1（上）")
                with gr.Row():
                    rect_x2 = gr.Slider(1, 800, value=350, step=1, label="X2（右）")
                    rect_y2 = gr.Slider(1, 800, value=350, step=1, label="Y2（下）")
                grabcut_iterations = gr.Slider(1, 10, value=5, step=1, label="迭代次数")

            process_btn = gr.Button("执行分割", variant="primary")

        # ── 右侧：结果显示 ──
        with gr.Column(scale=2):
            with gr.Row():
                overlay_output = gr.Image(label="分割叠加（绿色=前景）", type="numpy", height=280)
                mask_output = gr.Image(label="分割掩膜", type="numpy", height=280)
            with gr.Row():
                histogram_output = gr.Image(label="原图直方图", type="numpy", height=200)
                markers_output = gr.Image(label="分水岭标记可视化", type="numpy", height=200)

            with gr.Row():
                info_text = gr.Textbox(label="分割信息", lines=5)
                object_count = gr.Textbox(label="物体数量", value="—")

            download_btn = gr.Button("下载分割掩膜")
            download_file = gr.File(label="下载文件", visible=True)

    # ── 事件绑定 ──
    load_btn.click(fn=load_default_image, outputs=[input_image])
    clear_btn.click(fn=lambda: None, outputs=[input_image])

    process_btn.click(
        fn=process_segmentation,
        inputs=[
            input_image, method,
            block_size, C_value, min_distance,
            rect_x1, rect_y1, rect_x2, rect_y2, grabcut_iterations,
        ],
        outputs=[
            overlay_output, mask_output, histogram_output,
            markers_output, info_text, object_count, download_file,
        ],
    )


if __name__ == "__main__":
    demo.launch()
