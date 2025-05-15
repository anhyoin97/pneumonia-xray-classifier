import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.optim as optim
from focal_loss import FocalLoss  

# ============================
# 1. Dataset 정의 (NORMAL transform)
# ============================

class ChestXrayAugmentedDataset(Dataset):
    def __init__(self, folder_path, transform_normal=None, transform_pneumonia=None, max_per_class=1000):
        self.images = []
        self.transform_normal = transform_normal
        self.transform_pneumonia = transform_pneumonia
        normal_images = []
        pneumonia_images = []

        for label in ['NORMAL', 'PNEUMONIA']:
            label_path = os.path.join(folder_path, label)
            for img_name in os.listdir(label_path):
                full_path = os.path.join(label_path, img_name)
                if label == 'NORMAL':
                    normal_images.append((full_path, 0))
                else:
                    pneumonia_images.append((full_path, 1))

        normal_sampled = normal_images if len(normal_images) <= max_per_class else random.sample(normal_images, max_per_class)
        pneumonia_sampled = pneumonia_images if len(pneumonia_images) <= max_per_class else random.sample(pneumonia_images, max_per_class)

        self.images = normal_sampled + pneumonia_sampled
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert("RGB")

        if label == 0 and self.transform_normal:
            image = self.transform_normal(image)
        elif label == 1 and self.transform_pneumonia:
            image = self.transform_pneumonia(image)

        return image, label

# ============================
# 2. 모델 정의 (pretrained ResNet18)
# ============================

def get_resnet18_pretrained():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ============================
# 3. 전처리 정의
# ============================

transform_normal = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor()
])

transform_pneumonia = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================
# 4. 학습 준비
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ChestXrayAugmentedDataset(
    folder_path='./chest_xray/train',
    transform_normal=transform_normal,
    transform_pneumonia=transform_pneumonia,
    max_per_class=1000
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = get_resnet18_pretrained().to(device)
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ============================
# 5. 학습 루프
# ============================

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# ============================
# 6. 모델 저장
# ============================

torch.save(model.state_dict(), 'resnet18_augmented.pth')
print("모델 저장 완료: resnet18_augmented.pth")
