# TensorLogic API 참조

TensorLogic에서 사용 가능한 모든 작업에 대한 완전한 API 참조입니다.

## 목차

1. [텐서 생성](#텐서-생성)
2. [형상 작업](#형상-작업)
3. [수학 함수](#수학-함수)
4. [집계 작업](#집계-작업)
5. [활성화 함수](#활성화-함수)
6. [행렬 연산](#행렬-연산)
7. [정규화](#정규화)
8. [마스킹 작업](#마스킹-작업)
9. [인덱싱 작업](#인덱싱-작업)
10. [임베딩](#임베딩)
11. [샘플링](#샘플링)
12. [융합 작업](#융합-작업)
13. [최적화](#최적화)
14. [기타 작업](#기타-작업)
15. [연산자](#연산자)
16. [타입 정의](#타입-정의)

---

## 텐서 생성

### `zeros(shape: Array<Int>) -> Tensor`

0으로 채워진 텐서를 생성합니다.

**매개변수:**
- `shape`: 텐서 차원을 지정하는 배열

**반환:** 0으로 채워진 텐서

**예제:**
```tensorlogic
let z = zeros([2, 3])  // 2x3 크기의 0 텐서
```

---

### `ones(shape: Array<Int>) -> Tensor`

1로 채워진 텐서를 생성합니다.

**매개변수:**
- `shape`: 텐서 차원을 지정하는 배열

**반환:** 1로 채워진 텐서

**예제:**
```tensorlogic
let o = ones([2, 3])  // 2x3 크기의 1 텐서
```

---

### `positional_encoding(seq_len: Int, d_model: Int) -> Tensor`

Transformer를 위한 정현파 위치 인코딩을 생성합니다.

**매개변수:**
- `seq_len`: 시퀀스 길이
- `d_model`: 모델 차원

**반환:** `[seq_len, d_model]` 형상의 텐서

**수학적 정의:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**예제:**
```tensorlogic
let pe = positional_encoding(10, 512)
```

**사용 사례:**
- Transformer 모델
- 시퀀스-투-시퀀스 모델
- 어텐션 메커니즘

**참조:**
- arXiv:2510.12269 (표 1)
- "Attention is All You Need" (Vaswani et al., 2017)

---

## 형상 작업

### `reshape(tensor: Tensor, new_shape: Array<Int>) -> Tensor`

데이터를 보존하면서 텐서 형상을 변경합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `new_shape`: 목표 형상

**반환:** 형상이 변경된 텐서

**예제:**
```tensorlogic
let data = positional_encoding(6, 4)  // [6, 4]
let r = reshape(data, [3, 8])         // [3, 8]
```

**제약:**
- 전체 요소 수는 동일하게 유지되어야 함

---

### `flatten(tensor: Tensor) -> Tensor`

텐서를 1D로 평탄화합니다.

**매개변수:**
- `tensor`: 입력 텐서

**반환:** 1D 텐서

**예제:**
```tensorlogic
let data = positional_encoding(3, 4)  // [3, 4]
let f = flatten(data)                 // [12]
```

---

### `transpose(tensor: Tensor) -> Tensor`

2D 텐서를 전치합니다(축을 교환).

**매개변수:**
- `tensor`: 입력 2D 텐서

**반환:** 전치된 텐서

**예제:**
```tensorlogic
let t = transpose(positional_encoding(3, 4))  // [3,4] -> [4,3]
```

---

### `permute(tensor: Tensor, dims: Array<Int>) -> Tensor`

텐서 차원을 재정렬합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `dims`: 새로운 차원 순서

**반환:** 순열된 텐서

**예제:**
```tensorlogic
let p = permute(positional_encoding(6, 4), [1, 0])  // [6,4] -> [4,6]
```

---

### `unsqueeze(tensor: Tensor, dim: Int) -> Tensor`

지정된 위치에 크기 1의 차원을 추가합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `dim`: 새 차원을 삽입할 위치

**반환:** 차원이 추가된 텐서

**예제:**
```tensorlogic
let x = positional_encoding(3, 4)  // [3, 4]
let y = unsqueeze(x, 0)            // [1, 3, 4]
```

---

### `squeeze(tensor: Tensor) -> Tensor`

크기 1의 모든 차원을 제거합니다.

**매개변수:**
- `tensor`: 입력 텐서

**반환:** 크기 1의 차원이 제거된 텐서

**예제:**
```tensorlogic
let x = unsqueeze(positional_encoding(3, 4), 0)  // [1, 3, 4]
let y = squeeze(x)                                // [3, 4]
```

---

### `split(tensor: Tensor, sizes: Array<Int>, dim: Int) -> Array<Tensor>`

지정된 차원을 따라 텐서를 여러 텐서로 분할합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `sizes`: 각 분할 섹션의 크기
- `dim`: 분할할 차원

**반환:** 텐서 배열

**예제:**
```tensorlogic
let x = positional_encoding(10, 4)
let parts = split(x, [3, 3, 4], 0)  // 3개의 텐서: [3,4], [3,4], [4,4]
```

---

### `chunk(tensor: Tensor, chunks: Int, dim: Int) -> Array<Tensor>`

텐서를 지정된 수의 청크로 분할합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `chunks`: 청크 개수
- `dim`: 분할할 차원

**반환:** 텐서 배열

**예제:**
```tensorlogic
let x = positional_encoding(12, 4)
let parts = chunk(x, 3, 0)  // 각각 [4,4]인 3개의 텐서
```

---

## 수학 함수

### `exp(tensor: Tensor) -> Tensor`

요소별로 지수 함수를 적용합니다.

**수학적 정의:** `exp(x) = e^x`

**예제:**
```tensorlogic
let e = exp(positional_encoding(2, 3))
```

---

### `log(tensor: Tensor) -> Tensor`

요소별로 자연 로그를 적용합니다.

**수학적 정의:** `log(x) = ln(x)`

**예제:**
```tensorlogic
let l = log(exp(positional_encoding(2, 3)))
```

---

### `sqrt(tensor: Tensor) -> Tensor`

요소별로 제곱근을 적용합니다.

**수학적 정의:** `sqrt(x) = √x`

**예제:**
```tensorlogic
let sq = sqrt(positional_encoding(2, 2))
```

---

### `pow(tensor: Tensor, exponent: Number) -> Tensor`

텐서 요소를 지정된 거듭제곱으로 올립니다.

**수학적 정의:** `pow(x, n) = x^n`

**예제:**
```tensorlogic
let pw = pow(positional_encoding(2, 3), 2)
```

---

### `sin(tensor: Tensor) -> Tensor`

요소별로 사인 함수를 적용합니다.

**예제:**
```tensorlogic
let sn = sin(positional_encoding(2, 3))
```

---

### `cos(tensor: Tensor) -> Tensor`

요소별로 코사인 함수를 적용합니다.

**예제:**
```tensorlogic
let cs = cos(positional_encoding(2, 3))
```

---

### `tan(tensor: Tensor) -> Tensor`

요소별로 탄젠트 함수를 적용합니다.

**예제:**
```tensorlogic
let tn = tan(positional_encoding(2, 3))
```

---

## 집계 작업

### `sum(tensor: Tensor) -> Number`

모든 요소의 합계를 계산합니다.

**예제:**
```tensorlogic
let s = sum(positional_encoding(3, 4))
```

---

### `mean(tensor: Tensor) -> Number`

모든 요소의 평균을 계산합니다.

**예제:**
```tensorlogic
let m = mean(positional_encoding(3, 4))
```

---

### `max(tensor: Tensor) -> Number`

텐서의 최댓값을 반환합니다.

**예제:**
```tensorlogic
let mx = max(positional_encoding(4, 5))
```

---

### `min(tensor: Tensor) -> Number`

텐서의 최솟값을 반환합니다.

**예제:**
```tensorlogic
let mn = min(positional_encoding(4, 5))
```

---

### `argmax(tensor: Tensor, dim: Int) -> Tensor`

지정된 차원을 따라 최댓값의 인덱스를 반환합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `dim`: 최댓값을 찾을 차원

**반환:** 인덱스 텐서

**예제:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmax(x, 1)  // 차원 1을 따라 최댓값 인덱스
```

---

### `argmin(tensor: Tensor, dim: Int) -> Tensor`

지정된 차원을 따라 최솟값의 인덱스를 반환합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `dim`: 최솟값을 찾을 차원

**반환:** 인덱스 텐서

**예제:**
```tensorlogic
let x = positional_encoding(4, 5)
let idx = argmin(x, 1)  // 차원 1을 따라 최솟값 인덱스
```

---

## 활성화 함수

### `relu(tensor: Tensor) -> Tensor`

정류 선형 유닛 활성화.

**수학적 정의:** `relu(x) = max(0, x)`

**예제:**
```tensorlogic
let activated = relu(positional_encoding(3, 4))
```

---

### `sigmoid(tensor: Tensor) -> Tensor`

시그모이드 활성화 함수.

**수학적 정의:** `sigmoid(x) = 1 / (1 + e^(-x))`

**예제:**
```tensorlogic
let activated = sigmoid(positional_encoding(3, 4))
```

---

### `gelu(tensor: Tensor) -> Tensor`

가우시안 오류 선형 유닛 활성화(BERT, GPT에서 사용).

**수학적 정의:** 
```
gelu(x) = x * Φ(x)
여기서 Φ(x)는 표준 정규 분포의 누적 분포 함수
```

**예제:**
```tensorlogic
let g = gelu(positional_encoding(3, 4))
```

**사용 사례:**
- BERT, GPT 모델
- 현대적인 Transformer 아키텍처

---

### `tanh(tensor: Tensor) -> Tensor`

쌍곡 탄젠트 활성화.

**수학적 정의:** `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**예제:**
```tensorlogic
let th = tanh(positional_encoding(3, 4))
```

---

### `softmax(tensor: Tensor, dim: Int) -> Tensor`

지정된 차원을 따라 소프트맥스 정규화를 적용합니다.

**수학적 정의:**
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**매개변수:**
- `tensor`: 입력 텐서
- `dim`: 소프트맥스를 적용할 차원

**반환:** 확률 분포 텐서

**예제:**
```tensorlogic
let probs = softmax(positional_encoding(3, 4), 1)
```

**사용 사례:**
- 어텐션 메커니즘
- 분류 출력 레이어
- 확률 분포

---

## 행렬 연산

### `matmul(a: Tensor, b: Tensor) -> Tensor`

행렬 곱셈.

**매개변수:**
- `a`: 왼쪽 행렬
- `b`: 오른쪽 행렬

**반환:** 행렬 곱셈의 결과

**예제:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(4, 5)
let c = matmul(a, b)  // [3, 5]
```

---

## 정규화

### `layer_norm(tensor: Tensor, normalized_shape: Array<Int>, eps: Float) -> Tensor`

레이어 정규화를 적용합니다.

**수학적 정의:**
```
y = (x - E[x]) / sqrt(Var[x] + eps)
```

**매개변수:**
- `tensor`: 입력 텐서
- `normalized_shape`: 정규화할 형상
- `eps`: 수치 안정성을 위한 작은 값(기본값: 1e-5)

**예제:**
```tensorlogic
let normalized = layer_norm(positional_encoding(4, 512), [512], 1e-5)
```

**사용 사례:**
- Transformer 레이어
- 순환 신경망
- 심층 신경망

---

### `batch_norm(tensor: Tensor, running_mean: Tensor, running_var: Tensor, eps: Float) -> Tensor`

배치 정규화를 적용합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `running_mean`: 이동 평균
- `running_var`: 이동 분산
- `eps`: 수치 안정성을 위한 작은 값

**예제:**
```tensorlogic
let mean = zeros([512])
let var = ones([512])
let normalized = batch_norm(positional_encoding(32, 512), mean, var, 1e-5)
```

---

### `dropout(tensor: Tensor, p: Float) -> Tensor`

드롭아웃을 적용합니다(확률 p로 요소를 무작위로 0으로 설정).

**매개변수:**
- `tensor`: 입력 텐서
- `p`: 드롭아웃 확률(0.0에서 1.0)

**반환:** 무작위 요소가 0으로 설정된 텐서

**예제:**
```tensorlogic
let dropped = dropout(positional_encoding(3, 4), 0.1)
```

**사용 사례:**
- 훈련 정규화
- 과적합 방지
- 앙상블 학습

---

## 마스킹 작업

### `apply_attention_mask(tensor: Tensor, mask: Tensor) -> Tensor`

어텐션 마스크를 적용합니다(마스크된 위치를 -inf로 설정).

**매개변수:**
- `tensor`: 어텐션 점수 텐서
- `mask`: 이진 마스크(1 = 유지, 0 = 마스크)

**반환:** 마스크된 텐서

**예제:**
```tensorlogic
let scores = positional_encoding(4, 4)
let mask = ones([4, 4])
let masked_scores = apply_attention_mask(scores, mask)
```

**사용 사례:**
- Transformer 어텐션 메커니즘
- 시퀀스 마스킹
- 인과적(자기회귀적) 어텐션

---

### `padding_mask(lengths: Array<Int>, max_len: Int) -> Tensor`

가변 길이 시퀀스를 위한 패딩 마스크를 생성합니다.

**매개변수:**
- `lengths`: 실제 시퀀스 길이 배열
- `max_len`: 최대 시퀀스 길이

**반환:** 이진 마스크 텐서

**예제:**
```tensorlogic
let lengths = [3, 5, 2, 4]
let pad_mask = padding_mask(lengths, 5)
// 결과: [batch_size, max_len] 여기서 1 = 실제 토큰, 0 = 패딩
```

**사용 사례:**
- 가변 길이 시퀀스 처리
- 배치 처리
- 어텐션 마스킹

---

### `combine_masks(mask1: Tensor, mask2: Tensor) -> Tensor`

논리적 AND를 사용하여 두 마스크를 결합합니다.

**매개변수:**
- `mask1`: 첫 번째 마스크
- `mask2`: 두 번째 마스크

**반환:** 결합된 마스크

**예제:**
```tensorlogic
let pad_mask = padding_mask([3, 5, 2, 4], 5)
let mask2 = ones([4, 5])
let combined = combine_masks(pad_mask, mask2)
```

**사용 사례:**
- 패딩 및 어텐션 마스크 결합
- 다중 제약 마스킹
- 복잡한 어텐션 패턴

---

## 인덱싱 작업

### `gather(tensor: Tensor, dim: Int, indices: Tensor) -> Tensor`

인덱스를 사용하여 차원을 따라 값을 수집합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `dim`: 수집할 차원
- `indices`: 인덱스 텐서

**반환:** 수집된 텐서

**예제:**
```tensorlogic
let x = positional_encoding(5, 4)
let indices = argmax(x, 1)
let gathered = gather(x, 1, indices)
```

**사용 사례:**
- 토큰 선택
- 빔 서치
- 고급 인덱싱

---

### `index_select(tensor: Tensor, dim: Int, indices: Array<Int>) -> Tensor`

차원을 따라 지정된 인덱스의 요소를 선택합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `dim`: 선택할 차원
- `indices`: 인덱스 배열

**반환:** 선택된 텐서

**예제:**
```tensorlogic
let x = positional_encoding(10, 4)
let selected = index_select(x, 0, [0, 2, 5])  // 행 0, 2, 5 선택
```

---

## 임베딩

### `embedding(indices: TokenIDs, vocab_size: Int, embed_dim: Int) -> Tensor`

토큰 ID를 임베딩으로 변환합니다.

**매개변수:**
- `indices`: 토큰 ID 시퀀스
- `vocab_size`: 어휘 크기
- `embed_dim`: 임베딩 차원

**반환:** `[seq_len, embed_dim]` 형상의 임베딩 텐서

**예제:**
```tensorlogic
let token_ids = tokenize("Hello world")
let embeddings = embedding(token_ids, 50000, 512)
```

**사용 사례:**
- 단어 임베딩
- 토큰 표현
- 언어 모델

**참조:**
- arXiv:2510.12269 (표 1)

---

## 샘플링

### `top_k(logits: Tensor, k: Int) -> Tensor`

top-k 샘플링을 사용하여 토큰을 샘플링합니다.

**매개변수:**
- `logits`: 모델 출력 로짓
- `k`: 고려할 상위 토큰 수

**반환:** 샘플링된 토큰 ID

**예제:**
```tensorlogic
let logits = positional_encoding(1, 50000)  // [1, vocab_size]
let token = top_k(logits, 50)
```

**사용 사례:**
- 텍스트 생성
- 제어된 샘플링
- 다양한 출력

**참조:**
- arXiv:2510.12269 (표 2)

---

### `top_p(logits: Tensor, p: Float) -> Tensor`

nucleus(top-p) 샘플링을 사용하여 토큰을 샘플링합니다.

**매개변수:**
- `logits`: 모델 출력 로짓
- `p`: 누적 확률 임계값(0.0에서 1.0)

**반환:** 샘플링된 토큰 ID

**예제:**
```tensorlogic
let logits = positional_encoding(1, 50000)
let token = top_p(logits, 0.9)
```

**사용 사례:**
- 텍스트 생성
- 동적 어휘 선택
- 품질 제어 샘플링

**참조:**
- arXiv:2510.12269 (표 2)

---

## 융합 작업

융합 작업은 메모리 오버헤드와 커널 실행을 줄여 더 나은 성능을 위해 여러 작업을 결합합니다.

### `fused_add_relu(tensor: Tensor, other: Tensor) -> Tensor`

덧셈과 ReLU 활성화를 융합합니다.

**수학적 정의:** `fused_add_relu(x, y) = relu(x + y)`

**예제:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused1 = fused_add_relu(a, b)
```

**성능:** 별도 작업보다 ~1.5배 빠름

---

### `fused_mul_relu(tensor: Tensor, other: Tensor) -> Tensor`

곱셈과 ReLU 활성화를 융합합니다.

**수학적 정의:** `fused_mul_relu(x, y) = relu(x * y)`

**예제:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let fused2 = fused_mul_relu(a, b)
```

---

### `fused_affine(tensor: Tensor, scale: Tensor, bias: Tensor) -> Tensor`

아핀 변환(스케일 및 시프트)을 융합합니다.

**수학적 정의:** `fused_affine(x, s, b) = x * s + b`

**예제:**
```tensorlogic
let a = positional_encoding(3, 4)
let scale = ones([3, 4])
let bias = zeros([3, 4])
let affine_result = fused_affine(a, scale, bias)
```

**사용 사례:**
- 배치 정규화
- 레이어 정규화
- 사용자 정의 선형 변환

---

### `fused_gelu_linear(tensor: Tensor, weight: Tensor, bias: Tensor) -> Tensor`

GELU 활성화와 선형 변환을 융합합니다.

**수학적 정의:** `fused_gelu_linear(x, W, b) = linear(gelu(x), W, b)`

**예제:**
```tensorlogic
let input = positional_encoding(2, 4)
let weight = positional_encoding(4, 3)
let bias_vec = zeros([2, 3])
let gelu_linear = fused_gelu_linear(input, weight, bias_vec)
```

**사용 사례:**
- Transformer FFN 레이어
- BERT/GPT 아키텍처
- 성능 중요 경로

---

## 최적화

### `sgd_step(params: Tensor, gradients: Tensor, lr: Float) -> Tensor`

SGD 옵티마이저 스텝을 수행합니다.

**수학적 정의:** `params_new = params - lr * gradients`

**매개변수:**
- `params`: 현재 매개변수
- `gradients`: 계산된 그래디언트
- `lr`: 학습률

**반환:** 업데이트된 매개변수

**예제:**
```tensorlogic
learn {
    let updated = sgd_step(weights, gradients, 0.01)
}
```

---

### `adam_step(params: Tensor, gradients: Tensor, m: Tensor, v: Tensor, lr: Float, beta1: Float, beta2: Float, eps: Float) -> Tensor`

Adam 옵티마이저 스텝을 수행합니다.

**매개변수:**
- `params`: 현재 매개변수
- `gradients`: 계산된 그래디언트
- `m`: 1차 모멘트 추정
- `v`: 2차 모멘트 추정
- `lr`: 학습률
- `beta1`: 1차 모멘트 감소율(기본값: 0.9)
- `beta2`: 2차 모멘트 감소율(기본값: 0.999)
- `eps`: 수치 안정성 상수(기본값: 1e-8)

**반환:** 업데이트된 매개변수

**예제:**
```tensorlogic
learn {
    let updated = adam_step(weights, gradients, m, v, 0.001, 0.9, 0.999, 1e-8)
}
```

---

## 기타 작업

### `tokenize(text: String) -> TokenIDs`

텍스트를 토큰 ID 시퀀스로 변환합니다.

**매개변수:**
- `text`: 입력 텍스트 문자열

**반환:** TokenIDs (Vec<u32>)

**예제:**
```tensorlogic
let token_ids = tokenize("Hello world")
```

**사용 사례:**
- 텍스트 전처리
- 언어 모델 입력
- NLP 파이프라인

---

### `broadcast_to(tensor: Tensor, shape: Array<Int>) -> Tensor`

텐서를 지정된 형상으로 브로드캐스트합니다.

**매개변수:**
- `tensor`: 입력 텐서
- `shape`: 목표 형상

**반환:** 브로드캐스트된 텐서

**예제:**
```tensorlogic
let small = positional_encoding(1, 4)
let broadcasted = broadcast_to(small, [3, 4])
```

**사용 사례:**
- 형상 정렬
- 배치 작업
- 다른 형상을 가진 요소별 작업

---

## 연산자

TensorLogic은 표준 수학 연산자를 지원합니다:

### 산술 연산자
- `+` : 덧셈
- `-` : 뺄셈
- `*` : 요소별 곱셈
- `/` : 요소별 나눗셈

**예제:**
```tensorlogic
let a = positional_encoding(3, 4)
let b = positional_encoding(3, 4)
let c = a + b
let d = a * 2.0
```

### 비교 연산자
- `==` : 같음
- `!=` : 같지 않음
- `<`  : 미만
- `<=` : 이하
- `>`  : 초과
- `>=` : 이상

### 논리 연산자
- `&&` : 논리 AND
- `||` : 논리 OR
- `!`  : 논리 NOT

---

## 타입 정의

### Tensor
Metal Performance Shaders를 통한 GPU 가속을 지원하는 다차원 배열.

**속성:**
- 형상: 차원 배열
- 데이터: Float32 요소
- 장치: Metal GPU 장치

### TokenIDs
토큰 ID 시퀀스를 위한 특수 타입.

**정의:** `Vec<u32>`

**사용 사례:**
- 토큰화 결과
- 임베딩 조회
- 시퀀스 처리

### Number
숫자 값(Int 또는 Float).

**변형:**
- `Integer`: 64비트 부호 있는 정수
- `Float`: 64비트 부동 소수점

---

## 참조

- **TensorLogic 논문**: arXiv:2510.12269
- **Transformer 아키텍처**: "Attention is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GELU**: "Gaussian Error Linear Units (GELUs)" (Hendrycks and Gimpel, 2016)

---

## 관련 문서

- [시작 가이드](./getting_started.md)
- [언어 참조](./language_reference.md)
- [예제](../../examples/)
- [2025년 추가된 작업](../added_operations_2025.md)
- [TODO 목록](../TODO.md)

---

**최종 업데이트:** 2025-01-22

**TensorLogic 버전:** 0.1.1+

**총 작업 수:** 48개 함수 + 4개 연산자
