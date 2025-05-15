
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
### 8. Classification Report 분석
#### 2024.05.15
#### 테스트 이미지 : NORMAL 5장 / PNEUMONIA 5장  

```
              precision    recall  f1-score   support

      NORMAL       0.00      0.00      0.00         5
   PNEUMONIA       0.50      1.00      0.67         5

    accuracy                           0.50        10
   macro avg       0.25      0.50      0.33        10
weighted avg       0.25      0.50      0.33        10
```

#### NORMAL 클래스 성능
| 지표 | 해석 |
|------|------|
| **precision = 0.00** | 모델이 NORMAL로 예측한 이미지 중 실제로 NORMAL인 건 없음  
| **recall = 0.00** | 실제 NORMAL 이미지 5장 중 한 장도 맞히지 못함  
| **f1-score = 0.00** | precision과 recall 모두 0 → f1도 0  

**결론** : 모델은 정상(NORMAL)이라는 상황을 인식하지 못함

#### PNEUMONIA 클래스 성능
| 지표 | 해석 |
|------|------|
| **precision = 0.50** | PNEUMONIA로 예측한 것 중 절반만 진짜 PNEUMONIA  
| **recall = 1.00** | 모든 PNEUMONIA 이미지를 정확히 맞춤  
| **f1-score = 0.67** | 중간 수준의 종합 성능  

**결론** : 모든 이미지를 폐렴이라고 예측했기 때문에 recall은 높고 precision은 낮음

#### 전체 정확도 및 평균
| 지표 | 해석 |
|------|------|
| **accuracy = 0.50** | 전체 10장 중 절반만 맞춤 (실은 전부 폐렴으로 예측했기 때문)  
| **macro avg = 0.25 / 0.50** | 클래스 불균형 고려하지 않고 단순 평균  
| **weighted avg = 0.25 / 0.50** | 각 클래스 지원 수만큼 가중합 평균  

**결론** : 모델은 일방적으로 한 클래스만 예측 → 편향(Bias) 상태

> - 모델이 PNEUMONIA에는 민감하지만, **NORMAL은 무시**하는 상태 
> - **실제 사용 환경에서는 큰 문제**가 될 수 있음 (정상도 질병이라 예측)

---
### 9. ResNet18 + FocalLoss 적용 (threshold=0.8)
#### 2024.05.15
- 목적: PNEUMONIA 중심 예측을 완화하고 NORMAL recall 향상 시도
- 결과: 여전히 전체 이미지를 PNEUMONIA로 예측 (confusion matrix: all PNEUMONIA)
- classification report:

```
              precision    recall  f1-score   support
      NORMAL       0.00      0.00      0.00         5
   PNEUMONIA       0.50      1.00      0.67         5
    accuracy                           0.50        10
```

**결론** : FocalLoss만으로는 해결되지 않음. 근본적으로 NORMAL class 기준 부족.

#### Grad-CAM 시각화 (NORMAL 이미지)
- 결과: 모델이 폐 전체가 아닌 **심장/횡격막 하부 중앙**에만 집중
- 예측 결과: 해당 NORMAL 이미지를 PNEUMONIA로 잘못 분류함
- 시선이 잘못된 영역에 집중됨. 정상 폐 구조를 보지 못하고 있음  

---