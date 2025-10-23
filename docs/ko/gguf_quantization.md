# GGUF 양자화 모델 가이드

이 문서는 GGUF 형식의 양자화 모델(llama.cpp 호환)을 TensorLogic에서 로드하고 사용하는 방법을 설명합니다.

## GGUF 형식에 대하여

GGUF (GGML Universal Format)는 llama.cpp 프로젝트에서 개발한 대규모 언어 모델을 위한 효율적인 양자화 형식입니다.

### 주요 특징

- 4-bit/8-bit 양자화를 통한 메모리 효율성 (최대 8배 압축)
- 블록 기반 양자화로 정확도 유지
- llama.cpp, Ollama, LM Studio 등과 호환

### TensorLogic이 지원하는 양자화 형식

- ✅ **Q4_0**: 4-bit 양자화 (최고 압축률)
- ✅ **Q8_0**: 8-bit 양자화 (정확도와 압축 균형)
- ✅ **F16**: 16-bit 부동소수점 (높은 정확도)
- ✅ **F32**: 32-bit 부동소수점 (최고 정확도)

## 기본 사용법

### 1. 양자화 모델 로드

자동으로 f16으로 역양자화되어 Metal GPU에 로드됨:

```tensorlogic
model = load_model("models/llama-7b-q4_0.gguf")
```

### 2. 모델에서 텐서 가져오기

```tensorlogic
embeddings = model.get_tensor("token_embd.weight")
output_weight = model.get_tensor("output.weight")
```

## 양자화 형식 선택하기

### Q4_0 (4-bit)

- **메모리**: 최소 사용량 (원본 모델의 ~1/8)
- **속도**: 가장 빠른 추론
- **정확도**: 약간의 정확도 저하 (일반적으로 허용 가능)
- **사용 사례**: 챗봇, 일반 텍스트 생성

### Q8_0 (8-bit)

- **메모리**: 중간 사용량 (원본 모델의 ~1/4)
- **속도**: 빠름
- **정확도**: 높음 (F16과 거의 동등)
- **사용 사례**: 고품질 생성, 코딩 어시스턴트

### F16 (16-bit)

- **메모리**: 원본 모델의 ~1/2
- **속도**: 표준
- **정확도**: TensorLogic 네이티브 형식, Metal GPU 최적화
- **사용 사례**: 최고 품질이 필요한 경우

## 실용 예제: 토큰 임베딩

```tensorlogic
// LLama 모델에서 토큰 임베딩 가져오기
embedding_table = model.get_tensor("token_embd.weight")
print("Embedding shape:", embedding_table.shape)  // [vocab_size, hidden_dim]

// 토큰 ID에서 임베딩 벡터 가져오기
fn get_token_embedding(embedding_table: float16[V, D],
                             token_id: int) -> float16[D] {
    return embedding_table[token_id, :]
}
```

## 양자화로 인한 메모리 절약

예시: LLama-7B 모델 (70억 파라미터):

| 형식       | 메모리 사용량 | 압축률 |
|-----------|-------------|--------|
| F32 (원본) | ~28 GB      | 1x     |
| F16       | ~14 GB      | 2x     |
| Q8_0      | ~7 GB       | 4x     |
| Q4_0      | ~3.5 GB     | 8x     |

TensorLogic은 모든 형식을 로드 시 f16으로 변환하고 Metal GPU에서 효율적으로 실행합니다.

## 모델 다운로드 및 설치

### 1. HuggingFace에서 GGUF 모델 다운로드

예시: https://huggingface.co/TheBloke

### 2. 권장 모델 (초보자용)

- **TinyLlama-1.1B-Chat-v1.0** (Q4_0: ~600MB)
- **Phi-2** (Q4_0: ~1.6GB)
- **Mistral-7B** (Q4_0: ~3.8GB)

### 3. TensorLogic에서 로드

```tensorlogic
model = load_model("path/to/model-q4_0.gguf")
```

## 중요 사항

- 양자화 모델은 읽기 전용 (TensorLogic에서 저장 불가)
- 학습에는 비양자화 모델 (F16/F32) 사용
- Q4/Q8은 추론 전용으로 최적화됨
- 모든 양자화 형식은 자동으로 f16으로 역양자화되어 GPU에 로드

## 참고자료

- [GGUF 사양](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [HuggingFace GGUF 모델](https://huggingface.co/TheBloke)
- [모델 로딩 가이드](model_loading.md)
