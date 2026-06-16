"""
文档扫描仪 — 仿 CamScanner 的文档自动检测与校正

自动检测文档边缘（Canny + 轮廓检测），
进行透视变换校正，输出平整的扫描结果。
支持手动调整四个角点。
"""

import gradio as gr
import numpy as np
import cv2
from skimage import data
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ensure_uint8, img_to_bytes


# ── 默认示例图片 ────────────────────────────────────────────
def load_default_image():
    """生成一个带透视效果的"文档"图片作为示例"""
    img = data.coffee()
    img = ensure_uint8(img)
    h, w = img.shape[:2]

    # 模拟透视：定义源四边形的四个顶点和目标矩形
    margin = 30
    src_pts = np.float32([
        [margin + 40, margin],
        [w - margin + 20, margin + 30],
        [w - margin - 20, h - margin + 10],
        [margin - 10, h - margin - 20],
    ])

    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


# ── 文档角点自动检测 ────────────────────────────────────────
def detect_document_corners(image, canny_low, canny_high):
    """
    使用 Canny + 轮廓检测 + 多边形近似定位文档四个角点。
    返回：(角点坐标 [(x,y), ...], 可视化图片, 状态信息)
    """
    if image is None:
        return [], None, "请先上传图片"

    img = ensure_uint8(image)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 边缘检测
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # 膨胀边缘使其更连续
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 未找到，返回默认角点
        default_corners = [[10, 10], [w - 10, 10], [w - 10, h - 10], [10, h - 10]]
        viz = img.copy()
        _draw_corners(viz, default_corners)
        return default_corners, viz, "未检测到有效轮廓，使用默认角点，请手动调整"

    # 从大到小排序，取最大轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_corners = None
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            best_corners = approx.reshape(4, 2)
            break

    # 如果没有找到四边形，尝试更宽松的逼近
    if best_corners is None:
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) == 4:
                best_corners = approx.reshape(4, 2)
                break

    if best_corners is None:
        default_corners = [[10, 10], [w - 10, 10], [w - 10, h - 10], [10, h - 10]]
        viz = img.copy()
        _draw_corners(viz, default_corners)
        return default_corners, viz, "未找到四边形轮廓，使用默认角点，请手动调整"

    # 排序角点：左上、右上、右下、左下
    corners = _order_corners(best_corners)

    # 生成可视化
    viz = img.copy()
    _draw_corners(viz, corners)

    return corners.tolist(), viz, f"成功检测到文档角点"


def _order_corners(pts):
    """将四个角点排序为：左上、右上、右下、左下"""
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def _draw_corners(image, corners):
    """在图像上绘制角点和连线"""
    pts = np.array(corners, dtype=np.int32)
    for i, pt in enumerate(pts):
        cv2.circle(image, tuple(pt), 8, (0, 255, 0), -1)
        cv2.putText(image, str(i + 1), (pt[0] + 12, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 连线
    order = [0, 1, 2, 3, 0]
    for i in range(4):
        cv2.line(image, tuple(pts[order[i]]), tuple(pts[order[i + 1]]), (0, 255, 0), 2)


# ── 透视校正 ────────────────────────────────────────────────
def apply_perspective_correction(image, x1, y1, x2, y2, x3, y3, x4, y4, output_width, output_height):
    """
    根据用户指定的四个角点或自动检测的角点进行透视校正。
    """
    if image is None:
        return None

    img = ensure_uint8(image)
    h, w = img.shape[:2]

    src_pts = np.float32([
        [x1, y1],  # 左上
        [x2, y2],  # 右上
        [x3, y3],  # 右下
        [x4, y4],  # 左下
    ])
    dst_pts = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, M, (output_width, output_height))
    return result


# ── UI 布局 ─────────────────────────────────────────────────
with gr.Blocks(title="文档扫描仪", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 文档扫描仪
        模拟 CamScanner 功能：自动检测文档边缘并进行透视校正。
        支持自动检测和手动调整角点两种模式。
        """
    )

    with gr.Row():
        # ── 左侧：控制面板 ──
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传文档照片", type="numpy", height=250)
            with gr.Row():
                load_btn = gr.Button("加载示例图片", size="sm")
                clear_btn = gr.Button("清除图片", size="sm")

            gr.Markdown("### 自动检测参数")
            canny_low = gr.Slider(10, 200, value=50, step=1, label="Canny 低阈值")
            canny_high = gr.Slider(50, 400, value=150, step=1, label="Canny 高阈值")

            auto_detect_btn = gr.Button("自动检测角点", variant="secondary")
            detect_info = gr.Textbox(label="检测状态", lines=1, show_label=True)

            gr.Markdown("### 手动调整角点坐标")
            with gr.Row():
                with gr.Column():
                    x1 = gr.Number(label="X1（左上）", value=30, precision=0)
                    y1 = gr.Number(label="Y1（左上）", value=30, precision=0)
                with gr.Column():
                    x2 = gr.Number(label="X2（右上）", value=370, precision=0)
                    y2 = gr.Number(label="Y2（右上）", value=30, precision=0)
            with gr.Row():
                with gr.Column():
                    x3 = gr.Number(label="X3（右下）", value=370, precision=0)
                    y3 = gr.Number(label="Y3（右下）", value=370, precision=0)
                with gr.Column():
                    x4 = gr.Number(label="X4（左下）", value=30, precision=0)
                    y4 = gr.Number(label="Y4（左下）", value=370, precision=0)

            gr.Markdown("### 输出尺寸")
            with gr.Row():
                output_width = gr.Slider(200, 1200, value=400, step=10, label="输出宽度")
                output_height = gr.Slider(200, 1200, value=400, step=10, label="输出高度")

            scan_btn = gr.Button("执行扫描校正", variant="primary")

        # ── 右侧：结果显示 ──
        with gr.Column(scale=1):
            corner_viz = gr.Image(label="角点检测可视化", type="numpy", height=280)
            rectified_result = gr.Image(label="校正结果", type="numpy", height=280)

            download_btn = gr.Button("下载校正结果")
            download_file = gr.File(label="下载文件", visible=True)

    # ── 事件绑定 ──
    load_btn.click(fn=load_default_image, outputs=[input_image])
    clear_btn.click(fn=lambda: None, outputs=[input_image])

    def on_auto_detect(image, canny_low, canny_high):
        corners, viz, info = detect_document_corners(image, canny_low, canny_high)
        if len(corners) == 4:
            return (
                viz, info,
                int(corners[0][0]), int(corners[0][1]),
                int(corners[1][0]), int(corners[1][1]),
                int(corners[2][0]), int(corners[2][1]),
                int(corners[3][0]), int(corners[3][1]),
            )
        return viz, info, 30, 30, 370, 30, 370, 370, 30, 370

    auto_detect_btn.click(
        fn=on_auto_detect,
        inputs=[input_image, canny_low, canny_high],
        outputs=[corner_viz, detect_info, x1, y1, x2, y2, x3, y3, x4, y4],
    )

    scan_btn.click(
        fn=apply_perspective_correction,
        inputs=[input_image, x1, y1, x2, y2, x3, y3, x4, y4, output_width, output_height],
        outputs=[rectified_result],
    )

    def download_rectified(img):
        if img is None:
            return None
        return img_to_bytes(img)

    download_btn.click(
        fn=download_rectified,
        inputs=[rectified_result],
        outputs=[download_file],
    )


if __name__ == "__main__":
    demo.launch()
