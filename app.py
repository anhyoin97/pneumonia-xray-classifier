import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

# ============================
# 1. 모델 정의
# ============================

def get_resnet18_pretrained():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ============================
# 2. 전처리 정의
# ============================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ============================
# 3. 모델 불러오기
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18_pretrained().to(device)
model.load_state_dict(torch.load('resnet18_augmented.pth', map_location=device))
print(model.fc)
model.eval()

# ============================
# 4. 예측 함수
# ============================

def predict_image(image_path, threshold=0.6):
    # image = Image.open(image_path).convert("RGB")  
    image = Image.open(image_path).convert("L")  
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pneumonia_prob = probs[0][1].item()
        label = "PNEUMONIA" if pneumonia_prob > threshold else "NORMAL"
        return label, pneumonia_prob

# ============================
# 5. 실행 예시
# ============================

test_folder = './test_images'

for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(test_folder, filename)
        label, prob = predict_image(path)
        print(f"{filename} → {label} (폐렴 확률: {prob:.2f})")
