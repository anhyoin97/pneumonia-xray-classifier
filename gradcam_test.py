import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 모델 정의 및 마지막 conv layer 얻기
class ResNetWithCAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.feature_maps = None
        self.gradients = None

        # hook: 마지막 conv layer
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = self.model.layer4[1].conv2  # 마지막 conv 레이어
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, x):
        return self.model(x)

# 2. 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
raw_model.fc = nn.Linear(raw_model.fc.in_features, 2)
raw_model.load_state_dict(torch.load("resnet18_augmented.pth", map_location=device))

model = ResNetWithCAM().to(device)
#model.load_state_dict(torch.load("resnet18_augmented.pth", map_location=device))
model.model.load_state_dict(raw_model.state_dict())
model.eval()

# 3. 이미지 불러오기
image_path = "./test_images/NORMAL2-IM-0199-0001.jpeg" 
raw_image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(raw_image).unsqueeze(0).to(device)

# 4. forward + backward
output = model(input_tensor)
pred_class = torch.argmax(output)
model.zero_grad()
output[0][pred_class].backward()

# 5. Grad-CAM 계산
grads = model.gradients  # [1, C, H, W]
fmap = model.feature_maps  # [1, C, H, W]
weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP

cam = (weights * fmap).sum(dim=1).squeeze()  # [H, W]
cam = torch.relu(cam)
cam = cam.cpu().numpy()
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize 0~1

# 6. 시각화
img_np = np.array(raw_image.resize((224, 224))) / 255.0
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255.0
overlay = 0.4 * heatmap + 0.6 * img_np

# 7. 결과 출력
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("Input")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Overlay (Class: {pred_class.item()})")
plt.axis('off')

plt.tight_layout()
plt.show()
