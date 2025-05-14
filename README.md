
---

## ✅ 프로젝트 개요
- **X-ray 이미지를 기반으로 폐렴 여부를 분류하는 딥러닝 모델**  
- Chest X-ray 데이터를 분류하여 **폐렴 여부(PNEUMONIA vs NORMAL)** 를 판별


---

## 🗂 사용 기술

- Python 3.x
- PyTorch
- torchvision
- PIL (이미지 처리)
- matplotlib (시각화)

---

## 🧪 개선 과정 기록

### 1. 최초 모델 학습 (baseline)
#### 2024.05.13

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
```

- `chest_xray` 데이터로 학습
- 폐렴이미지 잘 분류 / 정상이미지 애매하게 분류
- Grad-CAM 시각화 결과: **폐 중심보다는 외곽에 집중**

![image](https://github.com/user-attachments/assets/c5a40933-9453-4b9f-bd1a-dd62bc07e289)

---

### 2. 중앙 집중 학습 적용
#### 2024.05.13
```python
class CentralFocus:
    def __call__(self, img, is_normal=True):
        if not is_normal:
            return img
        # 중심 영역만 유지하고 외곽 흐림
```

- `NORMAL` 이미지에 중앙만 선명하게 보여주도록 Blur 처리
- **결과: 정상 이미지를 폐렴으로 잘못 판단하는 경향 증가**
- 폐렴 확률 1.00으로 과확신 → False Positive 급증

![image](https://github.com/user-attachments/assets/de7ba00b-8241-4ac3-8736-f26d2452c330)

---

### 3. PNEUMONIA에만 CentralFocus(중앙 집중 학습) 적용
#### 2024.05.13
```python
if self.central_focus:
    image = self.central_focus(image, is_normal=(label == 1))
```

- 폐렴 이미지에만 중앙 강조 적용
- Grad-CAM 시선은 조금 개선되었으나, 여전히 NORMAL 정확도 낮음
- **결과: 중앙 집중 기법은 현재 모델의 성능을 저하시킴**
- 최종 모델은 **Augmentation만 적용한 baseline 버전이 가장 안정적**

![image](https://github.com/user-attachments/assets/5a1ac055-87d3-41b4-8c3c-5c139cf78654)

---
### 4. ResNet18 모델에 Grayscale 변환 적용
#### 2024.05.14
```python
image = Image.open(image_path).convert("L")
...
transforms.Grayscale(num_output_channels=3)
```
- 학습은 RGB 이미지 기반이었으나, 추론 시 Grayscale 강제 적용
- Grad-CAM 시선 이상, 예측 결과 전부 PNEUMONIA (확률 1.00)
- 결과: 학습/추론 채널 불일치로 인한 오작동 발생
---
### 5. pretrained=False → pretrained=True 적용
#### 2024.05.14
```python
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
```
- ImageNet 사전학습 가중치를 활용한 ResNet18 구조
- 예측 시 softmax 확률이 전반적으로 상승하고, 판단이 더 명확해짐
- 결과: 이전 모델보다 정상 이미지도 더 잘 구분하기 시작

---
### 6. PNEUMONIA에만 CentralFocus(중앙 집중 학습) 적용
#### 2024.05.14
```python
if self.central_focus:
    image = self.central_focus(image, is_normal=(label == 1))
```
- 폐렴 이미지에만 중앙 강조 적용
- Grad-CAM 시선은 조금 개선되었으나, 여전히 NORMAL 정확도 낮음
- 결과: 중앙 집중 기법은 현재 모델의 성능을 저하시킴
- 최종 모델은 Augmentation만 적용한 baseline 버전이 가장 안정적

---

### 7. NORMAL에만 강한 Augmentation 적용
#### 2024.05.14
```python
transform_normal = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor()
])
```
- 정상 이미지에만 다양한 변형을 적용해 모델이 정상 상태의 다양성을 학습하도록 유도
- 예측 결과가 극단적이지 않고, softmax 확률이 0.4 ~ 0.7 사이로 분포
- Grad-CAM과 confusion matrix를 통해 후속 성능 분석 예정
- 결과: 현 시점에서 가장 균형 잡힌 모델 생성

---