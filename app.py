import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ============================
# 모델 정의 (학습할 때와 동일)
# ============================

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# 모델 불러오기
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN()
model.load_state_dict(torch.load('pneumonia_cnn.pth', map_location=device))
model.to(device)
model.eval()

# ============================
# 이미지 전처리
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# ============================
# 예측 함수 (Softmax + Threshold 포함)
# ============================

def predict_image(image_path, threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pneumonia_prob = probs[0][1].item()

        # 임계값 적용
        if pneumonia_prob > threshold:
            pred_label = 'PNEUMONIA'
        else:
            pred_label = 'NORMAL'

    return pred_label, pneumonia_prob

# ============================
# 테스트 이미지 예측
# ============================

test_folder = './test_images'
threshold = 0.6  # 임계값 조정 가능

print("예측 결과 (Threshold = {:.2f}):".format(threshold))
for file in os.listdir(test_folder):
    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
        path = os.path.join(test_folder, file)
        label, confidence = predict_image(path, threshold)
        print(f"{file} → {label} (폐렴 확률: {confidence:.2f})")
