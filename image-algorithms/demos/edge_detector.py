"""
边缘检测对比 — 对比多种边缘检测方法的效果

支持 Canny、Sobel、Laplacian 三种边缘检测方法，
可并排对比两种方法，查看边缘叠加和纯边缘图像。
"""

import gradio as gr
import numpy as np
import cv2
from skimage import data
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ensure_uint8


# ── 默认示例图片 ────────────────────────────────────────────
def load_default_image():
    img = data.camera()
    return np.stack([img, img, img], axis=-1)


# ── Canny 边缘检测 ──────────────────────────────────────────
def canny_edge(image, low_thresh, high_thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return edges


# ── Sobel 边缘检测 ──────────────────────────────────────────
def sobel_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # 归一化到 [0, 255]
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


# ── Laplacian 边缘检测 ──────────────────────────────────────
def laplacian_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.abs(lap)
    lap = np.clip(lap, 0, 255).astype(np.uint8)
    return lap


# ── 生成边缘叠加图（红色边缘） ──────────────────────────────
def create_edge_overlay(image, edges):
    """在原始图像上用红色绘制边缘"""
    overlay = image.copy()
    edges_bool = edges > 30
    overlay[edges_bool] = [255, 0, 0]
    return overlay


# ── 主处理函数 ──────────────────────────────────────────────
def process_image(
    image,
    method_a,
    method_b,
    canny_low,
    canny_high,
    apply_thresh_a,
    apply_thresh_b,
    sobel_thresh_a,
    sobel_thresh_b,
    laplacian_thresh_a,
    laplacian_thresh_b,
):
    """执行边缘检测并返回所有输出"""
    if image is None:
        return [None] * 8

    img = ensure_uint8(image)

    # ── 方法 A ──
    edges_a_raw, edges_a, overlay_a = _detect_edges(
        img, method_a, canny_low, canny_high, apply_thresh_a,
        sobel_thresh_a, laplacian_thresh_a,
    )

    # ── 方法 B ──
    edges_b_raw, edges_b, overlay_b = _detect_edges(
        img, method_b, canny_low, canny_high, apply_thresh_b,
        sobel_thresh_b, laplacian_thresh_b,
    )

    # ── 元数据文本 ──
    meta_a = _method_meta(method_a)
    meta_b = _method_meta(method_b)

    return (
        overlay_a, edges_a, overlay_b, edges_b,
        f"【方法 A: {method_a}】\n{meta_a}",
        f"【方法 B: {method_b}】\n{meta_b}",
        _compare_side_by_side(edges_a, edges_b, method_a, method_b),
        _create_combined_overlay(img, edges_a, edges_b, method_a, method_b),
    )


def _detect_edges(img, method, canny_low, canny_high, apply_thresh, sobel_thresh, lap_thresh):
    """内部：执行单种边缘检测"""
    if method == "Canny":
        edges_raw = canny_edge(img, canny_low, canny_high)
    elif method == "Sobel":
        edges_raw = sobel_edge(img)
        if apply_thresh:
            edges_raw = (edges_raw > sobel_thresh).astype(np.uint8) * 255
    elif method == "Laplacian":
        edges_raw = laplacian_edge(img)
        if apply_thresh:
            edges_raw = (edges_raw > lap_thresh).astype(np.uint8) * 255
    else:
        edges_raw = np.zeros(img.shape[:2], dtype=np.uint8)

    # 纯边缘图
    edges_image = np.stack([edges_raw, edges_raw, edges_raw], axis=-1)
    # 边缘叠加图
    overlay = create_edge_overlay(img, edges_raw)

    return edges_raw, edges_image, overlay


def _method_meta(method):
    if method == "Canny":
        return "多级边缘检测，基于梯度幅值和非极大值抑制，使用双阈值确定强/弱边缘"
    elif method == "Sobel":
        return "一阶差分算子，分别计算 x 和 y 方向梯度，合并得到边缘强度"
    elif method == "Laplacian":
        return "二阶微分算子，检测像素值变化率的变化，对噪声敏感但能检测更细的边缘"
    return ""


def _compare_side_by_side(edges_a, edges_b, name_a, name_b):
    """并排对比两种方法的边缘图"""
    h = max(edges_a.shape[0], edges_b.shape[0])
    h_a, w_a = edges_a.shape[:2]
    h_b, w_b = edges_b.shape[:2]

    canvas = np.ones((h, w_a + w_b + 20, 3), dtype=np.uint8) * 255
    canvas[:h_a, :w_a] = edges_a
    canvas[:h_b, w_a + 20 : w_a + 20 + w_b] = edges_b

    # 添加标签
    cv2.putText(canvas, name_a, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, name_b, (w_a + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return canvas


def _create_combined_overlay(img, edges_a, edges_b, name_a, name_b):
    """在同一张图上用红色（方法A）和绿色（方法B）显示边缘"""
    combined = img.copy()
    edges_a_bool = edges_a > 30
    edges_b_bool = edges_b > 30
    combined[edges_a_bool] = [255, 0, 0]       # 红 — 方法 A
    combined[edges_b_bool] = [0, 255, 0]       # 绿 — 方法 B
    # 重叠区域显示为黄色
    overlap = edges_a_bool & edges_b_bool
    combined[overlap] = [255, 255, 0]

    cv2.putText(combined, f"红={name_a}  绿={name_b}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return combined


# ── UI 布局 ─────────────────────────────────────────────────
with gr.Blocks(title="边缘检测对比", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 边缘检测对比
        对比不同边缘检测方法的效果。上传图片或使用默认图片，选择两种方法进行并排对比。
        红色边缘属于方法 A，绿色边缘属于方法 B，重叠区域显示为黄色。
        """
    )

    with gr.Row():
        # ── 左侧：控制面板 ──
        with gr.Column(scale=1):
            input_image = gr.Image(label="输入图片", type="numpy", height=280)
            with gr.Row():
                load_btn = gr.Button("加载默认图片", size="sm")
                clear_btn = gr.Button("清除图片", size="sm")

            method_a = gr.Dropdown(
                choices=["Canny", "Sobel", "Laplacian"],
                value="Canny",
                label="方法 A（红色）",
            )
            method_b = gr.Dropdown(
                choices=["Canny", "Sobel", "Laplacian"],
                value="Sobel",
                label="方法 B（绿色）",
            )

            with gr.Accordion("> Canny 参数", open=True):
                canny_low = gr.Slider(10, 200, value=50, step=1, label="低阈值")
                canny_high = gr.Slider(50, 400, value=150, step=1, label="高阈值")

            with gr.Accordion("> 方法 A 阈值", open=False):
                apply_thresh_a = gr.Checkbox(label="启用二值化阈值", value=False)
                sobel_thresh_a = gr.Slider(10, 200, value=60, step=1, label="Sobel 阈值")
                laplacian_thresh_a = gr.Slider(5, 100, value=30, step=1, label="Laplacian 阈值")

            with gr.Accordion("> 方法 B 阈值", open=False):
                apply_thresh_b = gr.Checkbox(label="启用二值化阈值", value=False)
                sobel_thresh_b = gr.Slider(10, 200, value=60, step=1, label="Sobel 阈值")
                laplacian_thresh_b = gr.Slider(5, 100, value=30, step=1, label="Laplacian 阈值")

            process_btn = gr.Button("执行边缘检测", variant="primary")

        # ── 右侧：结果显示 ──
        with gr.Column(scale=2):
            with gr.Row():
                overlay_a = gr.Image(label="方法 A 边缘叠加（红色）", type="numpy", height=250)
                edges_a = gr.Image(label="方法 A 纯边缘", type="numpy", height=250)
            with gr.Row():
                overlay_b = gr.Image(label="方法 B 边缘叠加（绿色）", type="numpy", height=250)
                edges_b = gr.Image(label="方法 B 纯边缘", type="numpy", height=250)

            with gr.Row():
                with gr.Column():
                    info_a = gr.Textbox(label="方法 A 说明", lines=2)
                with gr.Column():
                    info_b = gr.Textbox(label="方法 B 说明", lines=2)

            gr.Markdown("### 并排对比与综合叠加")
            with gr.Row():
                side_by_side = gr.Image(label="并排对比", type="numpy")
                combined_overlay = gr.Image(label="综合叠加（红=A, 绿=B, 黄=重叠）", type="numpy")

    # ── 事件绑定 ──
    load_btn.click(fn=load_default_image, outputs=[input_image])
    clear_btn.click(fn=lambda: None, outputs=[input_image])

    process_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            method_a, method_b,
            canny_low, canny_high,
            apply_thresh_a, apply_thresh_b,
            sobel_thresh_a, sobel_thresh_b,
            laplacian_thresh_a, laplacian_thresh_b,
        ],
        outputs=[
            overlay_a, edges_a, overlay_b, edges_b,
            info_a, info_b,
            side_by_side, combined_overlay,
        ],
    )


if __name__ == "__main__":
    demo.launch()
