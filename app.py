import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

# 모델 정의
def get_resnet18_pretrained():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18_pretrained().to(device)
model.load_state_dict(torch.load("resnet18_augmented.pth", map_location=device))
model.eval()

# 테스트 폴더
test_folder = "./test_images"
threshold = 0.8

# 추론 및 확률 출력
for fname in sorted(os.listdir(test_folder)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_folder, fname)
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            pneumonia_prob = probs[0][1].item()
            label = "PNEUMONIA" if pneumonia_prob > threshold else "NORMAL"

        print(f"{fname} → {label} (폐렴 확률: {pneumonia_prob:.2f})")