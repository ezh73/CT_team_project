
# 🫁 CT 기반 폐 결절 악성 예측 모델 (ConvNeXt + CBAM + CAM-Alignment)

**Axial 2D 슬라이스 영상 기반 Attention 및 시각적 정렬(CAM-Alignment) 적용 모델**

---

## 📌 프로젝트 개요

CT 영상에서 발견되는 폐 결절 중 대부분은 양성이며, 악성은 10% 이하에 불과합니다.  
침습적 검사는 위험성이 높기 때문에, 영상 기반으로 악성 가능성을 자동으로 판단해  
**의사의 의사결정을 보조**하는 **AI 기반 분류 모델**의 필요성이 대두됩니다.

---

## 🎯 제안 목표

- **2D CT 슬라이스**를 입력으로 사용하여 전처리 단계를 최소화
- ConvNeXt-Tiny 백본에 **CBAM Attention**과 **CAM-Alignment Loss**를 추가
- 시각적 주목성과 해석 가능성을 함께 고려하는 **설명 가능한 AI 모델** 개발

---

## 🛠️ 사용 기술

### 📍 주요 모델 구성
- **Backbone**: ConvNeXt-Tiny
- **Attention**: CBAM (Convolutional Block Attention Module)
- **Alignment Loss**: CAM(Activation Map)과 결절 마스크 간 정렬 유도

### 📍 데이터 전처리
- LIDC-IDRI CT 데이터셋 (Axial 2D Slice)
- HU Windowing 적용
- 결절 위치에 대한 Binary Mask 활용

---

## 📊 변경된 Loss 계산 방식

```python
Loss = Class_Loss + λ_align × Align_Loss
# Class_Loss: 라벨별 가중 평균 CrossEntropy
# Align_Loss: CAM과 결절 마스크 간의 Binary CrossEntropy
```

---

## 🧪 성능 결과

| 모델 구성 | Accuracy |
|-----------|----------|
| ConvNeXt Only | 89.32% |
| + CBAM        | 90.44% |
| + CAM-Alignment | **91.50%** |

- λ_align=0.5일 때 가장 높은 성능을 달성
- Grad-CAM 시각화 결과에서 **결절 부위 집중도 증가** 확인

---

## 🧭 CAM-Alignment란?

CAM (Class Activation Map)은 모델이 입력에서 어떤 위치를 주목했는지를 시각화한 것입니다.  
`CAM-Alignment Loss`는 이 주목 영역이 **실제 결절 마스크와 일치하도록 학습**시키는 기법입니다.

👉 시각적 해석력(Explainability)을 높이고, 의료 영상의 **신뢰도**를 확보합니다.

---

## ⚠️ 한계점 및 향후 연구

- 단일 공공 데이터셋 (LIDC-IDRI)에 한정 → 다양한 병원 및 인구집단 대상 일반화 필요
- 2D 슬라이스 기반 → 결절의 **3D 연속성 정보** 반영 어려움
- 향후:
  - Polygon 또는 Volume 기반 마스크 활용한 지도 학습 확대
  - 다기관 데이터 적용 및 외부 테스트셋 기반 성능 검증

---

## ✅ 기대 효과

- CT 기반 폐 결절 악성 예측의 정확도 및 신뢰도 향상
- 불필요한 침습적 검사 감소 → 환자 부담 최소화
- 의료진의 빠르고 정확한 진단 보조 가능
- 조기 폐암 발견 및 치료율 향상에 기여

---

## 👨‍💻 팀 소개

- 김석호 (팀장) - [kimseockho93@gmail.com]  
- 석지원 (팀원) - [wb00102681@gmail.com]  
- 이지훈 (팀원) - [ezh737@gmail.com]
