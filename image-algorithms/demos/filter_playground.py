"""
滤波器画廊 — 交互式探索各种图像滤波效果

支持高斯模糊、中值滤波、双边滤波、CLAHE增强、USM锐化，
可调整参数并查看处理前后的对比效果。
"""

import gradio as gr
import numpy as np
import cv2
from skimage import data
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ensure_uint8, ensure_float, img_to_bytes


# ── 默认示例图片 ────────────────────────────────────────────
def load_default_image():
    """加载默认示例图片（灰度转三通道RGB）"""
    img = data.camera()
    return np.stack([img, img, img], axis=-1)


# ── 滤波器应用 ──────────────────────────────────────────────
def apply_filter(
    image,
    filter_type,
    sigma,
    kernel_size,
    clip_limit,
    usm_amount,
    usm_radius,
    bilateral_d,
    bilateral_sigma_color,
    bilateral_sigma_spatial,
):
    """根据所选滤波器类型处理图像，返回 (结果图, 参数说明)"""
    if image is None:
        return None, '请先上传图片或点击「加载默认图片」'

    img = ensure_uint8(image)
    # 确保核大小为奇数
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if filter_type == "高斯模糊":
        result = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        info = (
            f"参数：σ = {sigma:.1f}，核大小 = {ksize}×{ksize}\n\n"
            "高斯模糊使用高斯核对图像进行加权平均平滑。\n"
            "sigma 控制模糊程度，核大小决定邻域范围。\n"
            "sigma 越大图像越模糊，常用于去除高斯噪声。"
        )

    elif filter_type == "中值滤波":
        result = cv2.medianBlur(img, ksize)
        info = (
            f"参数：核大小 = {ksize}×{ksize}\n\n"
            "中值滤波用邻域像素的中值替代中心像素。\n"
            "对椒盐噪声有极好的去除效果，同时能较好地保留边缘。\n"
            "核大小越大，去噪越强但细节损失也越多。"
        )

    elif filter_type == "双边滤波":
        result = cv2.bilateralFilter(
            img, bilateral_d, bilateral_sigma_color, bilateral_sigma_spatial
        )
        info = (
            f"参数：直径 d = {bilateral_d}，σ_color = {bilateral_sigma_color}，σ_spatial = {bilateral_sigma_spatial}\n\n"
            "双边滤波是一种保边去噪滤波器。\n"
            "它同时考虑像素的空间距离和颜色差异，\n"
            "在平滑区域的同时保留强边缘，适合人像美颜等场景。"
        )

    elif filter_type == "CLAHE增强":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            result = clahe.apply(img)
            result = np.stack([result, result, result], axis=-1)
        info = (
            f"参数：Clip Limit = {clip_limit:.1f}\n\n"
            "CLAHE（对比度受限自适应直方图均衡化）\n"
            "将图像分成小区域分别做直方图均衡化，\n"
            "同时限制对比度放大倍数，避免噪声过度放大。\n"
            "适合改善局部对比度不佳的图像。"
        )

    elif filter_type == "USM锐化":
        usm_radius_odd = usm_radius if usm_radius % 2 == 1 else usm_radius + 1
        blurred = cv2.GaussianBlur(img, (usm_radius_odd, usm_radius_odd), 0)
        result = cv2.addWeighted(img, 1.0 + usm_amount, blurred, -usm_amount, 0)
        info = (
            f"参数：Amount = {usm_amount:.1f}，半径 = {usm_radius}\n\n"
            "USM（Unsharp Masking，非锐化掩膜）锐化：\n"
            "先对原图做高斯模糊，再从原图中减去模糊图，\n"
            "将差值加权叠加回原图以增强边缘。\n"
            "Amount 控制锐化强度，半径控制影响范围。"
        )

    else:
        result = img
        info = "未知滤波器类型"

    return result, info


# ── 下载处理结果 ────────────────────────────────────────────
def download_result(image):
    if image is None:
        return None
    return img_to_bytes(image)


# ── UI 布局 ─────────────────────────────────────────────────
with gr.Blocks(title="滤波器画廊", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 滤波器画廊
        交互式探索各种图像滤波效果。上传自己的图片或使用默认图片，选择滤波器并调整参数，观察处理前后的对比。
        """
    )

    with gr.Row():
        # ── 左侧：控制面板 ──
        with gr.Column(scale=1):
            input_image = gr.Image(label="输入图片", type="numpy", height=300)
            with gr.Row():
                load_btn = gr.Button("加载默认图片", size="sm")
                clear_btn = gr.Button("清除图片", size="sm")

            filter_type = gr.Dropdown(
                choices=["高斯模糊", "中值滤波", "双边滤波", "CLAHE增强", "USM锐化"],
                value="高斯模糊",
                label="滤波器类型",
            )

            # 通用参数
            sigma = gr.Slider(0.1, 15.0, value=2.0, step=0.1, label="Sigma（高斯模糊）")
            kernel_size = gr.Slider(3, 31, value=5, step=2, label="核大小（高斯模糊/中值滤波）")

            # CLAHE 参数
            clip_limit = gr.Slider(0.5, 10.0, value=2.0, step=0.1, label="Clip Limit（CLAHE增强）")

            # USM 参数
            usm_amount = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="Amount（USM锐化）")
            usm_radius = gr.Slider(3, 31, value=5, step=2, label="半径（USM锐化）")

            # 双边滤波参数
            bilateral_d = gr.Slider(1, 30, value=9, step=2, label="直径 d（双边滤波）")
            bilateral_sigma_color = gr.Slider(1, 150, value=75, step=1, label="颜色空间 Sigma（双边滤波）")
            bilateral_sigma_spatial = gr.Slider(1, 150, value=75, step=1, label="坐标空间 Sigma（双边滤波）")

            process_btn = gr.Button("应用滤镜", variant="primary")

        # ── 右侧：结果显示 ──
        with gr.Column(scale=1):
            output_image = gr.Image(label="处理结果", type="numpy", height=300)
            param_text = gr.Textbox(label="参数说明与算法原理", lines=6)
            download_btn = gr.Button("下载处理结果")
            download_file = gr.File(label="下载文件", visible=True)

    # ── 事件绑定 ──
    load_btn.click(fn=load_default_image, outputs=[input_image])
    clear_btn.click(fn=lambda: None, outputs=[input_image])

    process_btn.click(
        fn=apply_filter,
        inputs=[
            input_image,
            filter_type,
            sigma,
            kernel_size,
            clip_limit,
            usm_amount,
            usm_radius,
            bilateral_d,
            bilateral_sigma_color,
            bilateral_sigma_spatial,
        ],
        outputs=[output_image, param_text],
    )

    download_btn.click(
        fn=download_result,
        inputs=[output_image],
        outputs=[download_file],
    )


if __name__ == "__main__":
    demo.launch()
