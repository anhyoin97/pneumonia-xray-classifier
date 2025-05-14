import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ============================
# 1. 모델 정의 및 로딩
# ============================

class GradCAMResNet:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        weighted_activations = self.activations[0] * pooled_gradients[:, None, None]
        heatmap = weighted_activations.sum(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        return heatmap

# ============================
# 2. 모델 불러오기
# ============================

from torchvision.models import resnet18
import torch.nn as nn

def get_resnet18():
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18().to(device)
model.load_state_dict(torch.load('resnet18_pneumonia.pth', map_location=device))

# Grad-CAM 타겟 레이어: 마지막 conv layer
target_layer = model.layer4[-1]

grad_cam = GradCAMResNet(model, target_layer)

# ============================
# 3. 전처리 함수
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# ============================
# 4. 예측 및 Grad-CAM 시각화
# ============================

def show_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    prob = probs[0][1].item()

    heatmap = grad_cam.generate(input_tensor, class_idx=pred_class)

    # 시각화
    img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[2] == 1:
        img_np = np.concatenate([img_np]*3, axis=-1)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    cam_result = heatmap_colored * 0.5 + img_np * 0.5
    cam_result = np.clip(cam_result, 0, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Overlay ({'PNEUMONIA' if pred_class else 'NORMAL'}: {prob:.2f})")
    plt.imshow(cam_result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================
# 5. 실행
# ============================
show_gradcam("./test_images/NORMAL2-IM-0196-0001.jpeg")
