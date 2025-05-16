import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 모델 정의
def get_resnet18_pretrained():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18_pretrained().to(device)
model.load_state_dict(torch.load("resnet18_augmented.pth", map_location=device))
model.eval()

# 추론 및 수집
true_labels = []
pred_labels = []
threshold = 0.8
test_folder = "./test_images"  # 폴더 경로로 맞추기
class_names = ["NORMAL", "PNEUMONIA"]

for fname in os.listdir(test_folder):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_folder, fname)
        label = 0 if fname.startswith("NORMAL") else 1

        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prob = torch.softmax(output, dim=1)[0][1].item()
            pred = 1 if prob > threshold else 0

        true_labels.append(label)
        pred_labels.append(pred)

# confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# classification report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("\nClassification Report:")
print(report)