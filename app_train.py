import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

# ============================
# 1. Custom Dataset
# ============================

class ChestXrayDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.images = []
        self.transform = transform

        for label in ['NORMAL', 'PNEUMONIA']:
            label_path = os.path.join(folder_path, label)
            for img_name in os.listdir(label_path):
                self.images.append((os.path.join(label_path, img_name), 0 if label == 'NORMAL' else 1))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================
# 2. CNN 모델 정의
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
# 3. 학습 준비
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_path = './chest_xray/train'
train_dataset = ChestXrayDataset(folder_path=train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================
# 4. 학습 루프
# ============================

for epoch in range(3):
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
# 5. 모델 저장
# ============================

torch.save(model.state_dict(), 'pneumonia_cnn.pth')
print("모델 저장 완료: pneumonia_cnn.pth")
