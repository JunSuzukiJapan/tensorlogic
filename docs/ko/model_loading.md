# 모델 로딩 가이드

이 문서는 PyTorch 및 HuggingFace 모델을 TensorLogic에서 로드하고 사용하는 방법을 설명합니다. SafeTensors 형식(PyTorch 호환) 및 GGUF 형식(양자화된 LLM)을 지원합니다.

## 기본 사용법

### 1. SafeTensors 모델 로드 (PyTorch에서 저장)

```tensorlogic
model = load_model("path/to/model.safetensors")
```

### 2. GGUF 모델 로드 (양자화된 LLM)

```tensorlogic
model = load_model("path/to/llama-7b-q4.gguf")
```

### 3. 모델에서 텐서 가져오기

```tensorlogic
weights = model.get_tensor("layer.0.weight")
bias = model.get_tensor("layer.0.bias")
```

## 실용 예제: 선형 레이어 추론

모델 가중치와 바이어스를 사용한 추론 수행:

```tensorlogic
fn forward(input: float16[N, D_in],
                 weights: float16[D_in, D_out],
                 bias: float16[D_out]) -> float16[N, D_out] {
    // 선형 변환: output = input @ weights + bias
    let output = input @ weights
    return output + bias
}
```

## PyTorch 모델 준비

Python을 사용하여 SafeTensors 형식으로 모델 저장:

```python
import torch
from safetensors.torch import save_file

# PyTorch 모델 생성
model = MyModel()

# 모델 가중치를 딕셔너리로 가져오기
tensors = {name: param for name, param in model.named_parameters()}

# SafeTensors 형식으로 저장
save_file(tensors, "model.safetensors")
```

그런 다음 TensorLogic에서 로드:

```tensorlogic
model = load_model("model.safetensors")
```

## 지원되는 형식

### 1. SafeTensors (.safetensors)

- PyTorch 및 HuggingFace 호환
- F32, F64, F16, BF16 데이터 타입 지원
- 모든 데이터가 자동으로 f16으로 변환됨
- Metal GPU에 직접 로드

### 2. GGUF (.gguf)

- llama.cpp 형식 양자화 모델
- Q4_0, Q8_0, F32, F16 지원
- Metal GPU에 직접 로드

### 3. CoreML (.mlmodel, .mlpackage)

- Apple Neural Engine 최적화 모델
- iOS/macOS 전용

## 완전한 선형 모델 예제

```tensorlogic
// 입력 데이터 (배치 크기 4, 특징 차원 3)
let X = tensor<float16>([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], device: gpu)

// 가중치 행렬 (3 x 2)
let W = tensor<float16>([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
], device: gpu)

// 바이어스 (2차원)
let b = tensor<float16>([0.01, 0.02], device: gpu)

// 추론 실행
let output = forward(X, W, b)

// 결과 출력
print("Output shape:", output.shape)
print("Output:", output)
```

## 모델 저장

TensorLogic 모델을 SafeTensors 형식으로 저장할 수 있습니다:

```tensorlogic
save_model(model, "output.safetensors")
```

이를 통해 PyTorch 및 HuggingFace와의 상호 운용성이 가능합니다.

## 중요 사항

- TensorLogic은 모든 연산을 f16으로 실행 (Metal GPU 최적화)
- 다른 데이터 타입은 로드 시 자동으로 f16으로 변환
- 정수 타입 (i8, i32 등)은 지원되지 않음 (부동소수점만)
- 대규모 모델은 자동으로 Metal GPU 메모리에 로드

## 관련 문서

- [GGUF 양자화 모델](gguf_quantization.md)
- [CoreML & Neural Engine](coreml_neural_engine.md)
- [시작 가이드](../claudedocs/getting_started.md)
