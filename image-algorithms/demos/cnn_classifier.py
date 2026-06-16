"""
CNN图像分类演示 — 使用预训练模型进行图像分类和可视化

功能：
- 支持 ResNet18 / ResNet50 / VGG16 / EfficientNet-B0 预训练模型
- Top-5 预测类别及置信度条形图
- Grad-CAM 热力图可视化，展示模型关注区域
- 记录推理耗时
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import json
import time
import urllib.request
import os
from pathlib import Path
from collections import OrderedDict
sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

# ============================================================================
# 全局配置
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
LABELS_CACHE = Path(__file__).parent / ".imagenet_labels.json"

MODEL_REGISTRY = {
    "ResNet18": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
    "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
    "VGG16": lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
    "EfficientNet-B0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}

# ImageNet 标准预处理
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 模型缓存
_model_cache: dict = {}
_gradcam_targets: dict = {}  # 缓存每个模型的最后一个卷积层名称


# ============================================================================
# ImageNet 标签加载
# ============================================================================
def load_imagenet_labels() -> list:
    """加载 ImageNet 1000 类标签，优先使用本地缓存。"""
    if LABELS_CACHE.exists():
        try:
            with open(LABELS_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, LABELS_CACHE)
        with open(LABELS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # 网络不可用时生成占位标签
        return [f"类别 {i}" for i in range(1000)]


IMAGENET_LABELS = load_imagenet_labels()


# ============================================================================
# Grad-CAM 实现
# ============================================================================
class _GradCAMHook:
    """用于捕获目标层的前向激活值和反向梯度的钩子管理器。"""

    def __init__(self, module: nn.Module):
        self.activations = None
        self.gradients = None
        self._forward_handle = module.register_forward_hook(self._forward_hook)
        self._backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, mod, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, mod, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def find_last_conv_layer(model: nn.Module) -> str:
    """自动查找模型的最后一个卷积层名称。"""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = name
    if last_conv is None:
        raise ValueError("未在模型中找到卷积层")
    return last_conv


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """根据点分隔的名称获取子模块。"""
    parts = name.split(".")
    module = model
    for p in parts:
        module = getattr(module, p)
    return module


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer_name: str,
) -> np.ndarray:
    """
    计算 Grad-CAM 热力图。

    参数:
        model: 推理模式下的模型（已在 CPU/GPU 上）
        input_tensor: 预处理后的输入张量 [1, C, H, W]
        target_class: 目标类别索引
        target_layer_name: 目标卷积层名称

    返回:
        热力图 NumPy 数组 (H, W)，值归一化到 [0, 1]
    """
    target_layer = get_module_by_name(model, target_layer_name)
    hook = _GradCAMHook(target_layer)

    model.zero_grad()
    output = model(input_tensor)
    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    activations = hook.activations  # [1, C, H', W']
    gradients = hook.gradients      # [1, C, H', W']

    hook.remove()

    # 全局平均池化梯度 -> 权重
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

    # 加权组合激活图
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
    cam = F.relu(cam)  # 只保留正值
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    将热力图叠加到原图上。

    参数:
        image: RGB 原图 (H, W, 3), uint8
        heatmap: 热力图 (H', W'), 归一化到 [0, 1]
        alpha: 叠加透明度

    返回:
        叠加后的 RGB 图片 (H, W, 3), uint8
    """
    # 将热力图缩放到原图尺寸
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # 应用 JET 颜色映射
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 叠加
    overlaid = (heatmap_colored * alpha + image * (1 - alpha)).astype(np.uint8)
    return overlaid


# ============================================================================
# 模型加载（带缓存）
# ============================================================================
def load_model(model_name: str) -> nn.Module:
    """加载并缓存预训练模型。"""
    if model_name in _model_cache:
        return _model_cache[model_name]

    factory = MODEL_REGISTRY[model_name]
    model = factory()
    model.to(DEVICE)
    model.eval()

    _model_cache[model_name] = model
    return model


def get_gradcam_target(model: nn.Module, model_name: str) -> str:
    """获取/缓存每个模型的 Grad-CAM 目标层名称。"""
    if model_name in _gradcam_targets:
        return _gradcam_targets[model_name]
    target = find_last_conv_layer(model)
    _gradcam_targets[model_name] = target
    return target


# ============================================================================
# 推理函数
# ============================================================================
def classify_image(image: np.ndarray, model_name: str, enable_gradcam: bool):
    """
    对输入图片执行分类推理，可选 Grad-CAM 可视化。

    返回:
        (top5_plot, gradcam_image, status_text)
    """
    if image is None:
        return None, None, "⚠️ 请先上传一张图片"

    try:
        # 预处理
        t0 = time.time()
        pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        input_tensor = IMAGENET_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

        # 加载模型
        model = load_model(model_name)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        inference_time = time.time() - t0

        top5_labels = [IMAGENET_LABELS[idx] for idx in top5_idx[0].cpu().numpy()]
        top5_scores = top5_prob[0].cpu().numpy()

        # 生成 Top-5 条形图
        top5_plot = _make_top5_plot(top5_labels, top5_scores)

        # Grad-CAM
        gradcam_image = None
        if enable_gradcam:
            try:
                target_layer = get_gradcam_target(model, model_name)
                # 创建需要梯度的输入张量副本
                input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)

                heatmap = compute_gradcam(
                    model, input_tensor_grad,
                    target_class=int(top5_idx[0, 0].item()),
                    target_layer_name=target_layer,
                )
                gradcam_image = overlay_heatmap(image, heatmap, alpha=0.45)
            except Exception as e:
                # Grad-CAM 失败不影响主流程
                gradcam_image = None
                print(f"[Grad-CAM 失败] {e}")

        status = f"✅ 推理完成 | 模型: {model_name} | 耗时: {inference_time*1000:.1f} ms | 设备: {DEVICE}"
        return top5_plot, gradcam_image, status

    except Exception as e:
        error_msg = f"❌ 推理失败: {str(e)}"
        return None, None, error_msg


def _make_top5_plot(labels: list, scores: np.ndarray):
    """使用 matplotlib 生成 Top-5 预测水平条形图（返回 numpy 图片供 Gradio 显示）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 尝试使用中文字体
    cn_font = None
    for fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        try:
            matplotlib.font_manager.findfont(fname, fallback_to_default=False)
            cn_font = fname
            break
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = range(len(labels))
    colors = plt.cm.Blues(0.4 + 0.6 * np.array(scores))

    bars = ax.barh(y_pos, scores, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10,
                       fontfamily=cn_font if cn_font else "sans-serif")
    ax.invert_yaxis()
    ax.set_xlabel("置信度", fontsize=11,
                  fontfamily=cn_font if cn_font else "sans-serif")
    ax.set_title("Top-5 预测结果", fontsize=13, fontweight="bold",
                 fontfamily=cn_font if cn_font else "sans-serif")
    ax.set_xlim(0, 1.05)

    # 在条形右侧标注百分比
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score*100:.1f}%", va="center", fontsize=9)

    fig.tight_layout()

    # 转为 numpy 数组
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return plot_img[:, :, :3]  # 丢弃 alpha 通道


# ============================================================================
# 构建 Gradio 界面
# ============================================================================
def build_ui():
    css = """
    .gradio-container { max-width: 960px !important; margin: 0 auto !important; }
    .status-text { font-size: 14px; padding: 8px 12px; border-radius: 6px; }
    """

    with gr.Blocks(title="CNN图像分类演示", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 CNN 图像分类演示
            上传一张图片，选择预训练模型，观察 Top-5 分类结果。可选 Grad-CAM 热力图查看模型关注的图像区域。

            **支持模型**: ResNet18 · ResNet50 · VGG16 · EfficientNet-B0（均在 ImageNet 上预训练）
            """
        )

        # ---- 输入区 ----
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📷 上传图片",
                    type="numpy",
                    image_mode="RGB",
                )
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_REGISTRY.keys()),
                        value="ResNet18",
                        label="🔧 选择模型",
                        interactive=True,
                    )
                enable_gradcam = gr.Checkbox(
                    value=True,
                    label="🔥 生成 Grad-CAM 热力图",
                    info="显示模型关注的图像区域（红色=高关注）",
                )
                run_btn = gr.Button("🚀 开始分类", variant="primary", size="lg")

        # ---- 输出区 ----
        with gr.Row():
            with gr.Column(scale=1):
                top5_output = gr.Image(
                    label="📊 Top-5 预测结果",
                    type="numpy",
                    image_mode="RGB",
                )
            with gr.Column(scale=1):
                gradcam_output = gr.Image(
                    label="🔥 Grad-CAM 热力图叠加",
                    type="numpy",
                    image_mode="RGB",
                )

        status_text = gr.Textbox(
            label="状态",
            value="🟢 就绪 — 请上传图片并点击「开始分类」",
            interactive=False,
            elem_classes=["status-text"],
        )

        # ---- 事件绑定 ----
        def on_classify(image, model_name, gradcam_enabled):
            top5_plot, gradcam_img, status = classify_image(image, model_name, gradcam_enabled)
            return top5_plot, gradcam_img, status

        run_btn.click(
            fn=on_classify,
            inputs=[input_image, model_selector, enable_gradcam],
            outputs=[top5_output, gradcam_output, status_text],
        )

        # ---- 使用说明 ----
        gr.Markdown(
            """
            ---
            ### 📖 使用说明
            1. **上传图片** — 点击上方区域或拖拽图片文件到虚线框内
            2. **选择模型** — 不同模型在准确率与速度上有差异：
               - ResNet18: 轻量快速，适合实时场景
               - ResNet50: 更深层，准确率更高
               - VGG16: 经典架构，特征提取能力强
               - EfficientNet-B0: 高效缩放，精度/速度均衡
            3. **开启/关闭 Grad-CAM** — 勾选后会额外生成热力图叠加（计算量较大）
            4. **点击开始分类** — 查看 Top-5 预测和模型关注的图像区域

            ### 💡 小贴士
            - 模型首次加载需要下载预训练权重，请确保网络连接正常
            - 使用 GPU (CUDA) 可大幅加速推理
            - Grad-CAM 仅在支持反向传播的模型上可用
            - 标签使用 ImageNet 英文标准类别名
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
