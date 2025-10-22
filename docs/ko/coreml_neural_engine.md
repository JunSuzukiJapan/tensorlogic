# CoreML & Neural Engine 통합 가이드

이 가이드는 TensorLogic에서 CoreML 모델을 사용하고 Apple Neural Engine에서 고속 추론을 수행하는 방법을 설명합니다.

## CoreML과 Neural Engine 소개

### CoreML

- Apple의 독점 머신러닝 프레임워크
- iOS/macOS 전용 최적화
- Neural Engine, GPU, CPU를 자동으로 활용
- .mlmodel / .mlmodelc 형식

### Neural Engine

- Apple Silicon 전용 AI 전용 칩
- 최대 15.8 TOPS (M1 Pro/Max)
- 초저전력 소비 (GPU 대비 1/10 이하)
- f16 연산에 최적화

### TensorLogic과의 통합

- 모든 f16 연산 (Neural Engine 최적화)
- Metal GPU와 원활한 통합
- 자동 모델 형식 감지

## CoreML 모델 생성

CoreML 모델은 일반적으로 Python의 coreMLtools로 생성됩니다:

```python
import coremltools as ct
import torch

# PyTorch 모델 생성
model = MyModel()
model.eval()

# 추적 모델 생성
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# CoreML로 변환
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=example_input.shape)],
    convert_to="mlprogram",  # Neural Engine 최적화
    compute_precision=ct.precision.FLOAT16  # f16 정밀도
)

# 저장
mlmodel.save("model.mlpackage")
```

## TensorLogic에서 사용

### 1. CoreML 모델 로드 (macOS 전용)

```tensorlogic
model = load_model("model.mlpackage")
// 또는
model = load_model("model.mlmodelc")
```

### 2. 메타데이터 확인

```tensorlogic
print("Model format:", model.metadata.format)  // CoreML
print("Quantization:", model.metadata.quantization)  // F16
```

## Neural Engine 최적화 모범 사례

### 1. 데이터 타입: f16 사용

✅ 권장: `compute_precision=ct.precision.FLOAT16`
❌ 권장하지 않음: FLOAT32 (GPU에서 실행)

### 2. 모델 형식: mlprogram 형식 사용

✅ 권장: `convert_to="mlprogram"`
❌ 권장하지 않음: `convert_to="neuralnetwork"` (레거시 형식)

### 3. 배치 크기: 1이 최적

✅ 권장: `batch_size=1`
⚠️ 참고: `batch_size>1`은 GPU에서 실행될 수 있음

### 4. 입력 크기: 고정 크기가 최적

✅ 권장: `shape=[1, 3, 224, 224]`
⚠️ 참고: 가변 크기는 최적화가 제한됨

## 지원되는 연산

### Neural Engine에서 빠르게 실행되는 연산

- ✅ 합성곱 (conv2d, depthwise_conv)
- ✅ 완전 연결 레이어 (linear, matmul)
- ✅ 풀링 (max_pool, avg_pool)
- ✅ 정규화 (batch_norm, layer_norm)
- ✅ 활성화 함수 (relu, gelu, sigmoid, tanh)
- ✅ 요소별 연산 (add, mul, sub, div)

## 성능 비교

ResNet-50 추론 (224x224 이미지):

| 디바이스           | 지연시간 | 전력    | 효율성  |
|-------------------|---------|--------|---------|
| Neural Engine     | ~3ms    | ~0.5W  | 최고    |
| Metal GPU (M1)    | ~8ms    | ~5W    | 중간    |
| CPU (M1)          | ~50ms   | ~2W    | 낮음    |

## 모델 형식 선택

### 사용 사례별 권장 형식

**학습**: SafeTensors
- PyTorch 호환
- 가중치 저장/로드
- Metal GPU에서 학습

**추론 (iOS/macOS)**: CoreML
- Neural Engine 최적화
- 초저전력 소비
- 앱 통합

**추론 (범용)**: GGUF
- 양자화 지원
- 크로스 플랫폼
- 메모리 효율적

## 참고자료

- [CoreML 공식 문서](https://developer.apple.com/documentation/coreml)
- [coremltools](https://github.com/apple/coremltools)
- [Neural Engine 가이드](https://machinelearning.apple.com/research/neural-engine-transformers)
- [모델 로딩 가이드](model_loading.md)
