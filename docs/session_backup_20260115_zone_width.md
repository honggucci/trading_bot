# 세션 백업: Zone 폭 결정 방식 확정 (2026-01-15)

## 최종 결론

### 운영 룰 (고정)

```python
# Zone 폭 계산
zone_width = ATR(14) * 2.45

# TF별 clamp
CLAMP = {
    '1d':  (0.5%, 3.0%),
    '4h':  (0.5%, 3.0%),
    '1h':  (0.2%, 1.5%),
    '15m': (0.2%, 1.5%),
}

# Zone = Fib_Level ± zone_width
```

### 역할 분리

| 용도 | 지표 | 파라미터 |
|------|------|----------|
| **Zone 폭** | ATR | k = 2.45 |
| **리스크/사이징** | σ (EWMA) | λ = 0.94 |

---

## 실험 결과 요약

### 1. 공정 비교 (커버리지 50%로 맞춤)

| 모델 | 필요 k | Pearson | MAE |
|------|--------|---------|-----|
| **ATR** | **2.46** | **0.7055** | $725 |
| EWMA | 3.72 | 0.7026 | $738 |
| Yang-Zhang | 3.64 | 0.6761 | $754 |
| Realized | 4.16 | 0.6671 | $824 |

**결론**: ATR과 EWMA는 거의 동등 (차이 0.4%)

### 2. 힐베르트 amplitude 실험 (실패)

- Non-causal FFT 기반: Pearson 0.68 (룩어헤드 오염)
- Causal 버전: Pearson **0.32** (꼴찌)
- **결론**: 변동성 예측에 쓸 가치 없음

### 3. 헤지펀드식 σ vs ATR

| 상황 | σ 기반 | ATR 기반 |
|------|--------|----------|
| k=2.0 고정 | 커버리지 10~12% | 커버리지 48% |
| 커버리지 50% 맞춤 | k=3.7 필요 | k=2.5 필요 |
| 예측력 | 동등 | 동등 |

**결론**: σ가 구린 게 아니라 스케일링(√h)이 안 맞았던 것

---

## 왜 ATR을 선택했나

1. **단순함**: 파라미터 1개 (k=2.45)
2. **직관적**: ATR은 이미 "가격 범위" 단위
3. **검증됨**: 6년 데이터에서 커버리지 50%, Pearson 0.71

σ는 리스크 모듈에서 별도 사용:
- 포지션 사이징: `size ∝ 1 / σ_EWMA`
- 급변 레짐 감지: σ > 상위 10%일 때 Zone 상방 보정

---

## 코드 위치

| 파일 | 역할 |
|------|------|
| [src/context/volatility.py](src/context/volatility.py) | 변동성 추정치 (ATR, EWMA, YZ, RV) |
| [src/context/fib_levels.py](src/context/fib_levels.py) | Fib 레벨 + detrend |
| [config/fib_1w_anchor.json](config/fib_1w_anchor.json) | 1W Fib 앵커 (확정) |

### 실험 스크립트

| 스크립트 | 목적 |
|----------|------|
| `scripts/experiment_volatility_estimators.py` | ATR vs σ 초기 비교 (불공정) |
| `scripts/experiment_fair_comparison.py` | 동일 커버리지 공정 비교 |
| `scripts/experiment_vol_forecast_proper.py` | Causal 힐베르트 검증 |
| `scripts/experiment_fib_hilbert.py` | 힐베르트 amplitude 실험 |

---

## 운영 파라미터 (고정, 바꾸지 마)

```python
# Zone 폭
ATR_MULT = 2.45
ATR_WINDOW = 14

# Clamp (%)
CLAMP_HTF = (0.005, 0.03)  # 1D, 4H
CLAMP_LTF = (0.002, 0.015) # 1H, 15m

# 리스크 (별도 모듈)
EWMA_LAMBDA = 0.94
```

---

## 다음 작업

1. **트리거 품질 검증**: Zone 진입 후 CHoCH/BOS 승률
2. **레짐별 성과 분리**: 추세/횡보/급변
3. **σ 스위치 구현**: 급변 레짐에서만 Zone 상방 보정

---

## 교훈

### 틀렸던 것
- "σ는 별로다" → 공정 비교하면 ATR과 동등
- "힐베르트가 ATR 대체 가능" → Causal로 하면 꼴찌
- "헤지펀드식이 더 좋다" → Zone 폭 목적엔 ATR이 실용적

### 맞았던 것
- 1W Fib 앵커 확정 (Linear, $3,120 / $20,650)
- ATR k=2.4~2.5가 커버리지 50%에 적합
- σ는 리스크/사이징용으로 분리

---

## 핵심 철학

> "운영 파라미터는 고정하고, 트리거 품질을 깎아라."
>
> 지표 발명이 아니라 샘플 쌓기가 돈을 번다.
