import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image as PILImage  
from torchvision import transforms


# ============================
# 1. 모델 정의
# ============================

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Grad-CAM 대상
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        self.feature_maps = F.relu(self.conv2(x))  # <-- Hook 대상
        x = self.pool2(self.feature_maps)
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# 2. Grad-CAM 클래스
# ============================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        grad_cam = F.relu(grad_cam)
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()
        return grad_cam.cpu().numpy()

# ============================
# 3. 모델 불러오기
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN()
model.load_state_dict(torch.load('pneumonia_cnn.pth', map_location=device))
model.to(device).eval()

# ============================
# 4. 이미지 전처리
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# ============================
# 5. 예측 + Grad-CAM 실행
# ============================

# 예측할 파일명
test_image_path = './test_images/NORMAL2-IM-0196-0001.jpeg'

image = Image.open(test_image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Grad-CAM 생성
gradcam = GradCAM(model, model.conv2)
cam = gradcam.generate(input_tensor)

# ============================
# 6. 시각화
# ============================

# heatmap → RGB 변환
cam_resized = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0
heatmap = plt.cm.jet(cam_resized)[..., :3]
#heatmap = plt.cm.jet(cam)[..., :3]

# 원본 이미지와 heatmap 중첩
img_np = np.array(image.resize((224, 224))) / 255.0
if img_np.ndim == 2:
    img_np = np.stack([img_np]*3, axis=-1)  # 흑백 대비

superimposed = heatmap * 0.5 + img_np * 0.5

# 시각화 출력
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(superimposed)
plt.title("Grad-CAM Heatmap")
plt.axis('off')

plt.tight_layout()
plt.show()
