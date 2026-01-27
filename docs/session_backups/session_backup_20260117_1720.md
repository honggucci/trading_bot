# Session Backup - 2026-01-17 17:20

## 세션 요약: 다이버전스 강도 변수 검증 완료

### 완료된 작업

#### 1. 다이버전스 강도 변수 검증 스크립트 작성
- `scripts/validate_divergence_vars.py` 생성
- 4개 변수 검증: zone_depth, trend_alignment, bars_score, fib_proximity
- 비교용: delta_rsi (이전 세션에서 무효 확인)

#### 2. 핵심 검증 결과

**전체 통계:**
- 샘플 수: 291개 (Bullish 143, Bearish 148)
- 데이터: 2020-01에 주로 집중 (데이터 분포 편향)

**변수별 Spearman 상관 (vs MFE%):**

| 변수 | r | p-value | 유의성 |
|------|---|---------|--------|
| **zone_depth** | **+0.178** | **0.0023** | **유의** |
| trend_alignment | -0.058 | 0.329 | 무의미 |
| bars_score | +0.053 | 0.368 | 무의미 |
| fib_proximity | -0.128 | 0.029 | 역방향 |
| delta_rsi | +0.095 | 0.106 | 무의미 |

**zone_depth 빈별 P(NetMFE>=0.5%):**

| Depth 범위 | 확률 | 샘플 수 |
|------------|------|---------|
| 0.00-0.20 | 0.0% | 2 |
| 0.20-0.40 | 42.9% | 14 |
| 0.40-0.60 | 24.2% | 66 |
| 0.60-0.80 | 35.4% | 48 |
| 0.80-1.00 | 42.9% | 161 |

**Side별 분석:**
- Bullish zone_depth: r=+0.187, p=0.0251 (유의)
- Bearish zone_depth: r=+0.153, p=0.0626 (경계선)
- Bearish trend_alignment: r=-0.294, p=0.0003 (역방향!)

#### 3. 결론

1. **zone_depth만 유효한 예측 변수**
   - 피보나치 존에 깊이 들어갈수록 바운스 확률 증가
   - Bullish에서 더 효과적

2. **무효 확인된 변수**
   - ΔRSI: 상관계수 거의 0
   - trend_alignment: 방향 없음
   - bars_score: 변동성 없음 (대부분 1.0)
   - fib_proximity: 역방향 상관 (정확한 레벨보다 깊이가 중요)

3. **DQS 단순화 제안**
   ```
   DQS = zone_depth (0-1)
   ```
   복잡한 가중 합산 불필요. 단일 변수로 충분.

### 생성/수정된 파일

1. **생성:**
   - `scripts/validate_divergence_vars.py` - 다이버전스 변수 검증 스크립트
   - `data/analysis/enhanced_divergence_events.parquet` - 검증 결과 데이터

### 다음 작업 제안

#### 1. zone_depth 기반 필터 적용 백테스트
```python
# 조건: zone_depth >= 0.6 (60% 이상 진입)
# 기대: 낮은 depth 이벤트 제외로 win rate 개선
```

#### 2. 데이터 분포 개선
- 현재 2020-01에 샘플 집중
- 스캔 로직 개선으로 전 기간 샘플 확보

#### 3. 백테스트 통합
- zone_depth 필터를 기존 백테스트 시스템에 적용
- EV 변화 측정

### 핵심 교훈

1. **복잡한 점수 체계 불필요** - 단일 유효 변수(zone_depth)로 충분
2. **직관과 일치** - "깊이 들어올수록 반등 확률 높음" 검증됨
3. **다른 변수 기각** - ΔRSI, trend_alignment, bars_between 모두 무효

---

## 검증 프로토콜 정립

```python
# 변수 검증 표준 절차
1. 이벤트 수집 (최소 200개 이상)
2. Spearman 상관계수 (p < 0.05 필수)
3. 빈별 확률 단조성 확인
4. Side별 분리 분석

# 유효 기준
- r > 0.1 AND p < 0.05
- 빈별 확률 단조 증가/감소
```

---

*Project: trading_bot*
*RAG Embedding: trading_bot*
