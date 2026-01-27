# Session Backup - 2026-01-17 16:49

## 세션 요약: ΔRSI 검증 및 다이버전스 강도 변수 탐색

### 완료된 작업

#### 1. ΔRSI 검증 스크립트 작성 및 실행
- `scripts/validate_delta_rsi.py` 생성
- 2020-2025년 BTC 5m 데이터에서 다이버전스 이벤트 수집
- ΔRSI vs MFE (Maximum Favorable Excursion) 상관관계 분석

#### 2. 핵심 검증 결과

**통계 결과:**
- 샘플 수: 289개 (Bullish 137, Bearish 152)
- Spearman 상관계수: r = 0.0254
- p-value: 0.6666 (통계적으로 유의하지 않음)

**ΔRSI 빈별 수익 확률 P(NetMFE >= 0.5%):**
| ΔRSI 범위 | 확률 |
|-----------|------|
| 0-3       | 40%  |
| 3-5       | 31%  |
| 5-8       | 32%  |
| 8-12      | 41%  |
| 12-20     | 27%  |

**결론:**
- **ΔRSI는 유효한 예측 변수가 아님**
- 단조성 없음 (ΔRSI가 높다고 바운스가 커지지 않음)
- GPT 제안 "ΔRSI ≥ 8 필터"는 검증 결과 효과 없음 확인

### 생성/수정된 파일

1. **생성:**
   - `scripts/validate_delta_rsi.py` - ΔRSI 검증 스크립트
   - `data/analysis/divergence_events.parquet` - 검증 결과 데이터

### 다음 작업 제안

#### 새로운 검증 대상 변수 (우선순위순)

1. **trend_alignment (상위 TF 추세 일치)**
   - 기존 추세 필터에서 이미 효과 확인됨
   - 거부권(필터) + 사이즈 모듈레이션으로 활용

2. **zone_depth (Fib zone 근접도)**
   - "얼마나 깊게 들어왔는지" 측정
   - MR 전략에서 가장 중요한 변수일 가능성

3. **bars_between (두 피봇 간 시간 간격)**
   - 너무 짧으면 잡음, 너무 길면 구조 변화
   - 최소 간격으로 노이즈 제거

4. **volume_ratio (거래량 변화)**
   - 후순위, 데이터 품질에 따라 흔들림

#### 다음 검증 방법

```python
# 타겟
Y = 1 if NetMFE >= 0.5% else 0

# 검증 1: zone_depth bin
# depth를 5분위로 나누고 P(Y=1) 비교

# 검증 2: trend_alignment 조건부
# TrendOK vs TrendBad로 나누고 zone_depth 효과 비교

# 검증 3: bars_between
# (짧음/적정/김) 3구간으로 분류
```

#### 새 점수 체계 후보: DQS (Divergence Quality Score)

```
DQS = 0.45*zone_depth_score + 0.35*trend_alignment_score + 0.20*bars_between_score
```

### 핵심 교훈

1. **검증 없이 변수 추천 금지** - ΔRSI처럼 "경험상" 제안된 변수는 실제 데이터로 검증 필수
2. **올바른 순서**: 변수 검증 → 점수화 → 볼츠만/시그모이드 적용
3. **단조성 확인**: 변수가 유효하려면 빈별 확률이 단조 증가/감소해야 함

---

## 관련 컨텍스트

- 기존 백테스트: 97 trades, 33% win rate, RR=1.08, EV=-$4.06/trade
- 수수료 고려: 0.15-0.3% round trip → 최소 0.3-0.5% 바운스 필요

---

*Project: trading_bot*
*RAG Embedding: trading_bot*
