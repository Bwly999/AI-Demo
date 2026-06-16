"""
YOLO目标检测 — 使用YOLOv8进行通用物体检测

功能：
- 支持 YOLOv8n / YOLOv8s / YOLOv8m 三种规模模型
- 可调节置信度阈值和 IoU 阈值（NMS）
- 绘制颜色编码的检测框、类别标签和置信度
- 显示检测汇总（每类物体数量）
- 支持下载标注后的图片
- 记录推理耗时
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# 全局配置
# ============================================================================
DEVICE_DESC = "cuda" if __import__("torch").cuda.is_available() else "cpu"

MODEL_OPTIONS = {
    "YOLOv8n （Nano，速度最快）": "yolov8n.pt",
    "YOLOv8s （Small，速度与精度均衡）": "yolov8s.pt",
    "YOLOv8m （Medium，精度更高）": "yolov8m.pt",
}

# 模型缓存: {model_pt_name: YOLO instance}
_model_cache: dict = {}

# 用于颜色编码各类别的色表（按类名和类ID分别维护）
_CLASS_COLORS: dict = {}       # key=class_id (int) -> color tuple
_NAME_COLORS: dict = {}        # key=class_name (str) -> color tuple


# ============================================================================
# 模型加载（带缓存）
# ============================================================================
def load_model(model_key: str):
    """
    加载并缓存 YOLO 模型。首次运行自动下载权重文件。
    """
    from ultralytics import YOLO

    pt_name = MODEL_OPTIONS[model_key]

    if pt_name in _model_cache:
        return _model_cache[pt_name]

    model = YOLO(pt_name)
    # 预热：对空白图片跑一次推理
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        _ = model(dummy, verbose=False)
    except Exception:
        pass

    _model_cache[pt_name] = model
    return model


# ============================================================================
# 工具函数
# ============================================================================
def _get_class_color(class_id: int, class_name: str) -> tuple:
    """为每个类别分配固定颜色（同时按类名建立映射，供汇总使用）。"""
    if class_id not in _CLASS_COLORS:
        # 使用 HSV 色彩空间均匀分布色调
        hue = (class_id * 37) % 180  # 素数偏移避免相邻类颜色相近
        rgb = cv2.cvtColor(
            np.array([[[hue, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2RGB
        )
        color = tuple(int(c) for c in rgb[0, 0])
        _CLASS_COLORS[class_id] = color
        _NAME_COLORS[class_name] = color
    return _CLASS_COLORS[class_id]


def _draw_detections(image: np.ndarray, results) -> np.ndarray:
    """
    使用 OpenCV 在图片上手工绘制检测框、标签和置信度。
    返回 (annotated_image, detection_summary_dict)。
    """
    img = image.copy()
    summary = {}

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return img, summary

    boxes_data = results[0].boxes
    names = results[0].names  # {class_id: class_name}

    for box in boxes_data:
        # 获取检测信息
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = names.get(cls_id, f"class_{cls_id}")

        color = _get_class_color(cls_id, cls_name)

        # 绘制检测框
        thickness = max(2, int(min(img.shape[0], img.shape[1]) / 300))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 标签文本
        label = f"{cls_name} {conf:.2f}"

        # 计算文本尺寸以绘制背景
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness
        )
        # 标签背景放在框上方
        label_y = y1 - 4
        if label_y - th < 0:
            label_y = y1 + th + 4  # 如果超出顶部，放在框内顶部

        cv2.rectangle(
            img,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + 2),
            color,
            -1,  # 填充
        )
        cv2.putText(
            img, label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

        # 统计
        summary[cls_name] = summary.get(cls_name, 0) + 1

    return img, summary


def _build_summary_text(summary: dict) -> str:
    """将检测汇总字典格式化为中文文本。"""
    if not summary:
        return "未检测到任何物体"

    lines = [f"共检测到 **{sum(summary.values())}** 个物体，涉及 **{len(summary)}** 个类别：\n"]
    for cls_name, count in sorted(summary.items(), key=lambda x: -x[1]):
        color = _NAME_COLORS.get(cls_name, (128, 128, 128))
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        lines.append(
            f"- <span style='color:{hex_color};font-weight:bold'>■</span> "
            f"**{cls_name}**: {count} 个"
        )
    return "\n".join(lines)


# ============================================================================
# 检测推理
# ============================================================================
def detect_objects(
    image: np.ndarray,
    model_key: str,
    conf_threshold: float,
    iou_threshold: float,
):
    """
    对输入图片执行 YOLO 目标检测。

    返回:
        (annotated_image, summary_markdown, status_text, download_path)
    """
    if image is None:
        return None, "", "⚠️ 请先上传一张图片", None

    try:
        t0 = time.time()

        model = load_model(model_key)

        # YOLOv8 推理
        results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

        inference_time = time.time() - t0

        # 绘制检测结果
        annotated, summary = _draw_detections(image, results)
        summary_md = _build_summary_text(summary)

        # 保存标注图片供下载
        output_dir = Path(__file__).parent / ".detection_outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"detected_{int(time.time())}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        model_display = model_key.split("（")[0] if "（" in model_key else model_key
        status = (
            f"✅ 检测完成 | 模型: {model_display} | "
            f"耗时: {inference_time*1000:.1f} ms | "
            f"检测到 {sum(summary.values()) if summary else 0} 个物体 | "
            f"设备: {DEVICE_DESC}"
        )

        return annotated, summary_md, status, str(output_path)

    except Exception as e:
        error_msg = f"❌ 检测失败: {str(e)}"
        return None, error_msg, error_msg, None


# ============================================================================
# 构建 Gradio 界面
# ============================================================================
def build_ui():
    css = """
    .gradio-container { max-width: 1000px !important; margin: 0 auto !important; }
    .status-text { font-size: 14px; padding: 8px 12px; border-radius: 6px; }
    .summary-box { font-size: 15px; line-height: 1.7; }
    """

    with gr.Blocks(title="YOLO目标检测演示", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎯 YOLO 目标检测演示
            上传一张图片，使用 YOLOv8 进行通用物体检测。支持调节置信度与 IoU 阈值，检测结果可下载。

            **模型选项**: YOLOv8n（Nano，速度最快）· YOLOv8s（Small，均衡）· YOLOv8m（Medium，精度更高）
            """
        )

        # ---- 输入区 ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                input_image = gr.Image(
                    label="📷 上传图片",
                    type="numpy",
                    image_mode="RGB",
                )
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value=list(MODEL_OPTIONS.keys())[1],  # 默认 YOLOv8s
                    label="🔧 选择模型",
                    interactive=True,
                )
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="🎚 置信度阈值",
                    info="低于此值的检测结果将被过滤",
                )
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.45,
                    step=0.05,
                    label="📐 IoU 阈值 (NMS)",
                    info="NMS 的 IoU 阈值，值越大重复框越多",
                )
                run_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

        # ---- 输出区 ----
        with gr.Row():
            with gr.Column(scale=3):
                output_image = gr.Image(
                    label="📸 检测结果",
                    type="numpy",
                    image_mode="RGB",
                )
            with gr.Column(scale=2):
                summary_output = gr.Markdown(
                    value="*等待检测…*",
                    elem_classes=["summary-box"],
                )
                download_btn = gr.DownloadButton(
                    label="💾 下载标注图片",
                    variant="secondary",
                    visible=False,
                )

        status_text = gr.Textbox(
            label="状态",
            value="🟢 就绪 — 请上传图片并点击「开始检测」",
            interactive=False,
            elem_classes=["status-text"],
        )

        # 存储下载路径的隐藏状态
        download_path_state = gr.State(value=None)

        # ---- 事件绑定 ----
        def on_detect(image, model_key, conf, iou):
            annotated, summary_md, status, dp = detect_objects(image, model_key, conf, iou)
            if dp is not None:
                download_update = gr.update(value=dp, visible=True)
            else:
                download_update = gr.update(visible=False)
            return annotated, summary_md, status, download_update, dp

        run_btn.click(
            fn=on_detect,
            inputs=[input_image, model_selector, conf_slider, iou_slider],
            outputs=[
                output_image,
                summary_output,
                status_text,
                download_btn,
                download_path_state,
            ],
        )

        # ---- 使用说明 ----
        gr.Markdown(
            """
            ---
            ### 📖 使用说明
            1. **上传图片** — 支持常见格式（JPG/PNG/WebP等）
            2. **选择模型** — 根据需求在速度和精度之间权衡：
               - **YOLOv8n**: 模型最小，推理最快，适合实时场景
               - **YOLOv8s**: 速度与精度均衡，推荐日常使用
               - **YOLOv8m**: 精度更高，适合对准确率要求较高的场景
            3. **调节置信度阈值** — 值越高，检测结果越少但更可靠；值越低，漏检更少但可能有误检
            4. **调节 IoU 阈值** — 控制 NMS（非极大值抑制）的严格程度；值越大，重复框越多
            5. **点击开始检测** — 查看检测结果，可下载标注后的图片

            ### 💡 小贴士
            - 模型首次使用会自动下载权重文件（约 6-12 MB），请确保网络连接正常
            - YOLOv8 支持 80 类 COCO 数据集物体（人、车、动物、家具等常见类别）
            - 使用 GPU（CUDA）可大幅加速推理
            - 检测结果图片保存在 `demos/.detection_outputs/` 目录下
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
