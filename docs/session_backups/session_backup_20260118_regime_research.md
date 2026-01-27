# 세션 백업: 레짐 분류기 연구 (2026-01-18)

## 개요

볼츠만 분포, Hilbert Transform, RSI CoG 기반 레짐 분류기 연구 및 OOS 검증 수행.

---

## 1. 테스트한 접근법

### 1.1 Wave Regime (Hilbert Transform)
- **파일**: `src/regime/wave_regime.py`
- **원리**: FFT 기반 Hilbert Transform으로 위상/진폭 추출
- **문제 발견**: 전체 시리즈 FFT는 룩어헤드 바이어스 존재 (65% 매칭)
- **해결**: 슬라이딩 윈도우 Hilbert 구현 → 97% 매칭 (Causal)

### 1.2 Boltzmann Regime (z-score 기반)
- **파일**: `src/regime/boltzmann_regime.py`
- **원리**: 가격 z-score + momentum을 에너지 함수로 변환, Boltzmann 분포로 확률 계산
- **수식**:
  ```
  E_bull = -z_score - momentum_z * 0.5
  E_bear = +z_score + momentum_z * 0.5
  P(state) = exp(-E/T) / Z
  ```

### 1.3 RSI Center of Gravity
- **파일**: `scripts/test_rsi_cog.py`
- **원리**: RSI와 RSI 이동평균(CoG) 차이로 레짐 판단
- **문제**: 2025년 OOS에서 신호 반전 (1D 기준)

### 1.4 앙상블
- Boltzmann + HMM: 15m에서 악화
- Boltzmann + HMM (1H): STRONG_BULL만 개선
- RSI + Hilbert (1H): Hilbert 단독이 더 좋음

---

## 2. 타임프레임별 OOS 결과

| TF | 방식 | Forward | OOS BULL | OOS BEAR | IC | 평가 |
|----|------|---------|----------|----------|-----|------|
| **15m** | Boltzmann | 4H | -0.002% FAIL | +0.013% FAIL | -0.008 | ❌ |
| **1H** | Boltzmann | 1H | +0.004% OK | -0.002% OK | +0.007 | ⚠️ |
| **1H** | Hilbert (causal) | 1H | +0.037% OK | -0.018% OK | **+0.027** | ✅ 최고 |
| **1D** | Boltzmann | 2W | +0.37% OK | -0.16% OK | +0.038 | ⚠️ |
| **1D** | RSI CoG | All | FAIL | FAIL | -0.11 | ❌ |

---

## 3. 핵심 발견

### 3.1 Hilbert Wave (1H Causal)가 최고 성능
- **IC = +0.027** (테스트 중 최고)
- OOS에서 BULL/BEAR 양방향 정확
- 슬라이딩 윈도우로 룩어헤드 제거 완료

### 3.2 앙상블 효과 제한적
- 대부분 단독 신호보다 악화
- 1H에서 STRONG_BULL만 +10% 개선

### 3.3 신호 강도 약함
- 평균 수익률 차이: 0.01~0.04%
- IC: 0.007~0.038
- 단독 트레이딩 신호로 사용 불가 → **필터 용도**

### 3.4 타임프레임별 적합 용도
- **15m**: 레짐 효과 없음 → 트리거 전용
- **1H**: Hilbert 레짐 → 진입 Gate
- **1D**: Boltzmann → 2주 바이어스

---

## 4. 생성/수정된 파일

### 신규 생성
- `src/regime/boltzmann_regime.py` - Boltzmann 레짐 분류기
- `scripts/test_rsi_cog.py` - RSI CoG 테스트
- `scripts/test_1d_regime.py` - 1D 레짐 테스트
- `scripts/test_1h_regime.py` - 1H 레짐 테스트
- `scripts/test_ensemble_regime.py` - 15m 앙상블 테스트
- `scripts/test_ensemble_1h.py` - 1H 앙상블 테스트
- `scripts/test_rsi_hilbert_ensemble.py` - RSI+Hilbert 앙상블

### 수정
- `src/regime/wave_regime.py` - Causal 슬라이딩 Hilbert 추가
  - `classify_series_causal()` 메서드 추가
  - `sliding_hilbert()` 함수 추가

---

## 5. 권장 아키텍처

```
Layer 1 (Context): 1D Boltzmann → 2W 바이어스
Layer 3 (Gate): 1H Hilbert (causal) → 진입 허용/금지
Layer 4 (Trigger): 5m 흡수/실패 → 실제 진입 결정
```

### 진입 로직 예시
```python
# Long 진입 조건
if zone_entry and trigger_5m == 'absorption':
    if hilbert_1h.regime in ['BULL', 'RANGE']:
        execute_long()  # 허용
    else:
        skip()  # BEAR에서 Long 금지
```

---

## 6. 수식 정리

### Hilbert Transform (Causal)
```python
# 슬라이딩 윈도우 - 룩어헤드 없음
for t in range(window, N):
    segment = data[t-window+1 : t+1]  # 과거 window개만
    analytic = scipy.signal.hilbert(segment)
    phase[t] = angle(analytic[-1])
    amplitude[t] = abs(analytic[-1])

# 레짐 분류 (Mean-Reversion)
if amplitude_z < 0.5:
    regime = 'RANGE'
elif 225° ≤ phase < 315°:  # falling
    regime = 'BULL'  # 반등 예상
elif 45° ≤ phase < 135°:   # rising
    regime = 'BEAR'  # 하락 예상
```

### Boltzmann Distribution
```python
# 에너지 함수
E_bull = -z_score - momentum_z * 0.5
E_bear = +z_score + momentum_z * 0.5
E_range = |z_score| * 0.5 + |momentum_z| * 0.3

# 확률 (temperature T)
P(state) = exp(-E/T) / (exp(-E_bull/T) + exp(-E_bear/T) + exp(-E_range/T))
```

---

## 7. 다음 세션 작업

1. **HMM Gate에 Hilbert 통합**: `src/gate/hmm_entry_gate.py` 수정
2. **백테스트 검증**: Hilbert 필터 추가 시 성능 변화 측정
3. **롱/숏 비대칭 적용**: 롱은 엄격 (BULL만), 숏은 느슨 (BEAR+RANGE)

---

## 8. 테스트 실행 방법

```bash
# 1H Hilbert 테스트
python scripts/test_rsi_hilbert_ensemble.py

# 1D Boltzmann 테스트
python scripts/test_1d_regime.py

# 기존 테스트
python -m pytest tests/ -v
```

---

## 9. 결론

**레짐 분류는 단독 신호가 아닌 필터로 사용**

- 1H Hilbert (causal): 가장 효과적, IC=+0.027
- 극적인 개선은 아니지만 잘못된 방향 트레이드 감소 효과
- 기존 아키텍처 (Zone + Trigger)에 Gate로 추가 권장

---

*Generated: 2026-01-18*
*Project: trading_bot*
*RAG Embedding: trading_bot*
