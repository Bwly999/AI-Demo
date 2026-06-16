"""
目标跟踪 — 视频中的目标跟踪与光流可视化

支持 CSRT、KCF、MIL 跟踪器，以及稠密光流 HSV 可视化。
用户在第一帧绘制边界框，系统自动跟踪目标并生成结果视频。
"""

import gradio as gr
import numpy as np
import cv2
from skimage import data
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ensure_uint8


# ── 默认示例视频生成 ────────────────────────────────────────
def generate_default_video():
    """生成一个带移动圆形的示例视频"""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()

    w, h = 400, 300
    n_frames = 90
    fps = 15

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    cx, cy = 100, 150
    dx, dy = 3, 2

    for _ in range(n_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 240
        cv2.circle(frame, (cx, cy), 30, (50, 100, 200), -1)
        # 添加一些静态背景纹理
        cv2.rectangle(frame, (50, 50), (350, 250), (200, 200, 200), 2)
        cv2.putText(frame, "Tracking Demo", (120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        out.write(frame)

        cx += dx
        cy += dy
        if cx > w - 40 or cx < 40:
            dx *= -1
        if cy > h - 40 or cy < 40:
            dy *= -1

    out.release()
    return path


# ── 视频帧提取 ──────────────────────────────────────────────
def get_first_frame(video_path):
    """从视频中提取第一帧，用于绘制 ROI"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# ── OpenCV 目标跟踪 ─────────────────────────────────────────
OPENCV_TRACKERS = {
    "CSRT": cv2.TrackerCSRT_create,
    "KCF": cv2.TrackerKCF_create,
    "MIL": cv2.TrackerMIL_create,
}


def run_opencv_tracking(video_path, tracker_name, x1, y1, x2, y2):
    """使用 OpenCV tracker 进行目标跟踪"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "无法打开视频文件"

    tracker_fn = OPENCV_TRACKERS.get(tracker_name, cv2.TrackerCSRT_create)
    tracker = tracker_fn()

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None, "无法读取视频第一帧"

    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    tracker.init(first_frame, bbox)

    h, w = first_frame.shape[:2]
    trajectory = [(int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))]

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 15, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            bx, by, bw, bh = [int(v) for v in bbox]
            # 绘制跟踪框
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            # 记录轨迹
            cx, cy = bx + bw // 2, by + bh // 2
            trajectory.append((cx, cy))
            # 绘制轨迹
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        else:
            # 跟踪失败，仍绘制最后的轨迹
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

        cv2.putText(frame, f"Frame: {frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return out_path, f"跟踪完成：共 {frame_count} 帧，轨迹点 {len(trajectory)} 个"


# ── 稠密光流可视化 ──────────────────────────────────────────
def run_optical_flow(video_path):
    """使用稠密光流（Farneback）进行运动可视化"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "无法打开视频文件"

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None, "无法读取视频第一帧"

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # 创建 HSV 彩色编码图
    hsv_mask = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_mask[..., 1] = 255

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 15, (w * 2, h))  # 双倍宽度并排显示

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 计算光流幅值和角度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        flow_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        # 并排显示原图和光流
        side_by_side = np.hstack([frame, flow_bgr])
        cv2.putText(side_by_side, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Optical Flow", (w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(side_by_side, f"Frame: {frame_count}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(side_by_side)
        prev_gray = curr_gray
        frame_count += 1

    cap.release()
    out.release()

    return out_path, f"光流可视化完成：共 {frame_count} 帧"


# ── 主处理函数 ──────────────────────────────────────────────
def process_tracking(video_path, tracker_name, mode, x1, y1, x2, y2):
    """
    根据模式执行跟踪或光流。
    返回：(结果视频路径, 状态信息)
    """
    if video_path is None:
        return None, "请先上传视频文件"

    if mode == "光流可视化（稠密光流）":
        return run_optical_flow(video_path)
    else:
        # OpenCV 跟踪模式
        if x1 >= x2 or y1 >= y2:
            return None, "边界框无效：x1 < x2 且 y1 < y2 必须成立"
        return run_opencv_tracking(video_path, tracker_name, x1, y1, x2, y2)


# ── UI 布局 ─────────────────────────────────────────────────
with gr.Blocks(title="目标跟踪", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 目标跟踪
        上传视频文件，选择跟踪方法，绘制边界框后进行目标跟踪。
        支持 CSRT、KCF、MIL 三种跟踪器以及稠密光流可视化。
        """
    )

    with gr.Row():
        # ── 左侧：控制面板 ──
        with gr.Column(scale=1):
            video_input = gr.Video(label="上传视频", height=200)

            with gr.Row():
                load_default_btn = gr.Button("生成示例视频", size="sm")
                extract_frame_btn = gr.Button("提取第一帧", size="sm")

            mode = gr.Dropdown(
                choices=["CSRT 跟踪", "KCF 跟踪", "MIL 跟踪", "光流可视化（稠密光流）"],
                value="CSRT 跟踪",
                label="跟踪模式",
            )

            tracker_name = gr.Dropdown(
                choices=["CSRT", "KCF", "MIL"],
                value="CSRT",
                label="跟踪器类型（仅跟踪模式）",
            )

            first_frame = gr.Image(label="第一帧（在此绘制 ROI）", type="numpy", height=250)

            gr.Markdown("### ROI 边界框坐标")
            with gr.Row():
                x1 = gr.Slider(0, 600, value=80, step=1, label="X1（左）")
                y1 = gr.Slider(0, 600, value=120, step=1, label="Y1（上）")
            with gr.Row():
                x2 = gr.Slider(10, 800, value=180, step=1, label="X2（右）")
                y2 = gr.Slider(10, 800, value=200, step=1, label="Y2（下）")

            gr.Markdown(
                """**使用步骤：**
                1. 上传视频或点击"生成示例视频"
                2. 点击"提取第一帧"
                3. 参考第一帧，调整 ROI 坐标
                4. 选择跟踪模式，点击"开始跟踪"
                """
            )

            process_btn = gr.Button("开始跟踪", variant="primary")

        # ── 右侧：结果显示 ──
        with gr.Column(scale=1):
            result_video = gr.Video(label="跟踪结果", height=320)
            status_text = gr.Textbox(label="处理状态", lines=2)

    # ── 事件绑定 ──
    load_default_btn.click(
        fn=lambda: generate_default_video(),
        outputs=[video_input],
    )

    extract_frame_btn.click(
        fn=get_first_frame,
        inputs=[video_input],
        outputs=[first_frame],
    )

    process_btn.click(
        fn=process_tracking,
        inputs=[video_input, tracker_name, mode, x1, y1, x2, y2],
        outputs=[result_video, status_text],
    )


if __name__ == "__main__":
    demo.launch()
