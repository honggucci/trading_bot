# Session Backup - 2026-01-21 14:00

## 1. 완료된 작업

### PR-MODE49 LONG 전체 흐름 검토

**Step 1-3 완료 (이전 세션):**
- StochRSI 과매도 체크 (iloc[-1])
- Divergence 체크 (d[-2] lookahead 제거)
- Fib 레벨 근접성 (extended_ratios, fib_zone 모드)

**Step 4-6 검토 (현재 세션):**

#### Step 4: Prob Gate 메커니즘 분석
- `src/regime/upstream_scores.py`: Hilbert Transform → direction score
  - Detrend (48 EMA) → Hilbert analytic signal → phase
  - score = sin(phase + 45°) - sin(phase)
  - Amplitude weighting (tanh)

- `src/regime/prob_gate.py`: Temperature-scaled sigmoid
  - T = 1.2 + 0.6 × vol_z (clipped [0.7, 3.0])
  - p_bull = sigmoid(score / T)
  - action_code: +1 (p_bull >= 0.55), 0 (otherwise)

#### Step 5: RR Limit Entry
- entry_limit = signal_price - (tp_distance × entry_offset_ratio)
- entry_offset_ratio = 0.3 (EXP2 최적값)
- rr_limit_ttl = 6h, rr_limit_target = 2.0

#### Step 6: Position Management
- use_fib_based_sl: true (ATR×1.5는 fallback)
- tp_mode: trailing_only (TP = infinity)
- trailing_activation_atr: 2.0
- trailing_distance_atr: 1.0

---

## 2. 백테스트 결과

### MODE49 (2021-01-01 ~ 2024-12-31)

| Metric | Value |
|--------|-------|
| Total Trades | 70 |
| Win Rate | 60.0% (42W/28L) |
| Total PnL | $183.38 (+1.83%) |
| Avg PnL/Trade | $2.62 |
| RR Ratio | 0.86 |
| Max Win | $56.30 |
| Max Loss | -$33.35 |
| Final Equity | $10,183.38 |

### Exit Breakdown

| Exit Reason | Count | Win Rate | Avg PnL | Total PnL |
|-------------|-------|----------|---------|-----------|
| SL | 54 (77.1%) | 51.9% | $1.50 | $80.80 |
| 5m_Div_Partial | 14 | 100.0% | $10.89 | $152.47 |
| GateFlip | 2 | 0.0% | -$24.94 | -$49.89 |

### Prob Gate 통계

- 정상 통과 (LONG): 68개
- 거부 (NOT_LONG): 540개

### SL 분석

| bars_to_SL | Count | Ratio |
|------------|-------|-------|
| 0-3 bars (0-15m) | 20 | 37.0% |
| 4-6 bars (20-30m) | 3 | 5.6% |
| 7-12 bars (35-60m) | 7 | 13.0% |
| 13-24 bars (65-120m) | 9 | 16.7% |
| 25+ bars (125m+) | 15 | 27.8% |

---

## 3. 생성/수정된 파일

- `configs/mode49.json` - fib_zone 모드 설정
- `scripts/backtest_strategy_compare.py` - d[-2] lookahead 수정

---

## 4. 핵심 인사이트

1. **5m Div Partial 전략이 가장 효과적**: 100% 승률, 평균 +$10.89
2. **SL 빈도가 높음 (77.1%)**: 초기 진입 후 빠르게 역행하는 경우가 많음
3. **Prob Gate 필터링 효과**: 608개 시그널 중 68개만 통과 (88.8% 필터링)
4. **빠른 SL Hit**: 37%가 15분 내 SL (진입 타이밍 개선 필요)

---

## 5. 다음 작업 제안

1. **진입 타이밍 최적화**: 빠른 SL hit (37%) 줄이기 위한 추가 조건 검토
2. **5m Div Partial 강화**: 가장 효과적인 전략으로 확대 적용 고려
3. **trailing_activation_atr 조정**: 현재 2.0 → 1.5~1.8 테스트
4. **entry_offset_ratio 미세 조정**: 0.25~0.35 범위 추가 실험

---

## 6. RAG 임베딩 키워드

- trading_bot
- MODE49
- backtest
- prob_gate
- hilbert_score
- fib_zone
- trailing_only
- entry_offset_ratio

---

**Created**: 2026-01-21 14:00
**Project**: trading_bot
