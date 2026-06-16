"""
全景拼接 — 基于特征匹配的图像拼接

使用 SIFT/ORB 特征检测 + FLANN 匹配 + RANSAC 单应性矩阵
将两张图片拼接为全景图。
"""

import gradio as gr
import numpy as np
import cv2
from skimage import data, transform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ensure_uint8, img_to_bytes
from utils.visualization import draw_matches


# ── 默认示例图片 ────────────────────────────────────────────
def load_default_left():
    """加载默认左图"""
    img = data.coffee()
    return ensure_uint8(img)


def load_default_right():
    """加载默认右图（左图的裁剪+偏移模拟）"""
    img = data.coffee()
    img = ensure_uint8(img)
    h, w = img.shape[:2]
    # 裁剪右半部分作为右图
    right = img[:, w // 3 :].copy()
    return right


# ── 特征检测器 ──────────────────────────────────────────────
def get_detector(method="SIFT"):
    if method == "SIFT":
        return cv2.SIFT_create()
    else:
        return cv2.ORB_create(nfeatures=2000)


# ── FLANN 匹配器 ────────────────────────────────────────────
def get_flann_matcher(method="SIFT"):
    if method == "SIFT":
        index_params = dict(algorithm=1, trees=5)  # KD-Tree
    else:
        index_params = dict(
            algorithm=6,  # LSH
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


# ── 主拼接函数 ──────────────────────────────────────────────
def stitch_images(image_left, image_right, method, ratio_thresh, ransac_thresh):
    """
    拼接两张图片。
    返回：(特征匹配可视化, 全景拼接结果, 匹配数量, 质量信息)
    """
    if image_left is None or image_right is None:
        return None, None, "—", "请先上传左右两张图片"

    img1 = ensure_uint8(image_left)
    img2 = ensure_uint8(image_right)

    try:
        detector = get_detector(method)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, None, "0", "检测到的特征点不足（需要至少4对匹配点），请更换图片或方法"

        # FLANN 匹配
        matcher = get_flann_matcher(method)
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)

        if match_count < 4:
            return None, None, str(match_count), (
                f"经 ratio test 过滤后匹配点不足（仅 {match_count} 对，需要 >= 4）。\n"
                "请尝试提高 ratio 阈值或更换图片。"
            )

        # 计算单应性矩阵
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)

        if H is None:
            return None, None, str(match_count), "无法计算单应性矩阵，请尝试更换图片"

        inliers = mask.ravel().tolist().count(1)
        quality = (
            f"总匹配数：{match_count}\n"
            f"RANSAC 内点数：{inliers}\n"
            f"内点率：{inliers / match_count * 100:.1f}%\n"
            f"单应性矩阵 3×3：\n"
            + "\n".join(["  " + " ".join(f"{v:8.3f}" for v in row) for row in H])
        )

        # 绘制匹配图
        inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
        match_viz = draw_matches(img1, img2, kp1, kp2, inlier_matches, max_draw=80)

        # 图像融合拼接
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 计算右图变换后的四个角
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners2_transformed = cv2.perspectiveTransform(corners2, H)
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

        all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # 平移矩阵
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        # 变换右图并拼接
        panorama_w = x_max - x_min
        panorama_h = y_max - y_min

        warped_right = cv2.warpPerspective(img2, translation @ H, (panorama_w, panorama_h))

        panorama = warped_right.copy()
        panorama[-int(y_min) : -int(y_min) + h1, -int(x_min) : -int(x_min) + w1] = img1

        # 处理重叠区域的混合
        mask1 = np.zeros((panorama_h, panorama_w), dtype=np.float32)
        mask1[-int(y_min) : -int(y_min) + h1, -int(x_min) : -int(x_min) + w1] = 1.0
        mask2 = (warped_right.sum(axis=2) > 0).astype(np.float32)

        overlap = mask1 * mask2
        blend = np.zeros_like(panorama, dtype=np.float64)

        # 简单线性混合
        for c in range(3):
            blend[:, :, c] = (
                panorama[:, :, c].astype(np.float64) * (1 - overlap * 0.5)
                + warped_right[:, :, c].astype(np.float64) * overlap * 0.5
            )

        result = np.clip(blend, 0, 255).astype(np.uint8)

        return match_viz, result, str(match_count), quality

    except Exception as e:
        import traceback
        return None, None, "—", f"处理出错：{traceback.format_exc()}"


# ── UI 布局 ─────────────────────────────────────────────────
with gr.Blocks(title="全景拼接", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 全景拼接
        上传两张有重叠区域的图片，系统将自动检测特征点并进行匹配和拼接。
        支持 SIFT 和 ORB 两种特征检测器。左右两张图片应有 30%-50% 的重叠区域。
        """
    )

    with gr.Row():
        # ── 左侧：控制面板 ──
        with gr.Column(scale=1):
            with gr.Row():
                image_left = gr.Image(label="左图", type="numpy", height=200)
                image_right = gr.Image(label="右图", type="numpy", height=200)

            with gr.Row():
                load_left_btn = gr.Button("加载示例左图", size="sm")
                load_right_btn = gr.Button("加载示例右图", size="sm")

            method = gr.Dropdown(
                choices=["SIFT", "ORB"],
                value="SIFT",
                label="特征检测方法",
            )
            ratio_thresh = gr.Slider(
                0.5, 0.95, value=0.75, step=0.01,
                label="Lowe's Ratio Test 阈值",
            )
            ransac_thresh = gr.Slider(
                1.0, 10.0, value=5.0, step=0.5,
                label="RANSAC 重投影误差阈值",
            )

            process_btn = gr.Button("开始拼接", variant="primary")

            gr.Markdown(
                """**使用提示：**
                - SIFT 精度更高但需要 opencv-contrib
                - ORB 是免费替代，速度更快
                - Ratio 阈值越低匹配越严格
                - 若匹配失败可尝试提高阈值
                """
            )

        # ── 右侧：结果显示 ──
        with gr.Column(scale=2):
            match_viz = gr.Image(label="特征点匹配可视化", type="numpy", height=300)
            panorama_result = gr.Image(label="全景拼接结果", type="numpy")

            with gr.Row():
                match_count = gr.Textbox(label="匹配点数", value="—")
                quality_info = gr.Textbox(label="拼接质量信息", lines=8)

            download_btn = gr.Button("下载全景图")
            download_file = gr.File(label="下载文件", visible=True)

    # ── 事件绑定 ──
    load_left_btn.click(fn=load_default_left, outputs=[image_left])
    load_right_btn.click(fn=load_default_right, outputs=[image_right])

    process_btn.click(
        fn=stitch_images,
        inputs=[image_left, image_right, method, ratio_thresh, ransac_thresh],
        outputs=[match_viz, panorama_result, match_count, quality_info],
    )

    def download_pano(img):
        if img is None:
            return None
        return img_to_bytes(img)

    download_btn.click(
        fn=download_pano,
        inputs=[panorama_result],
        outputs=[download_file],
    )


if __name__ == "__main__":
    demo.launch()
