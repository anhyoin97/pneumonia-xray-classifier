
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
- 폐렴, 정상 이미지 모두 비교적 잘 분류함
- Grad-CAM 시각화 결과: **폐 중심보다는 외곽에 집중**

---

### 2. 중앙 집중 학습 적용 (실패)
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
---


