# MODE63 Shadow Trade 연구 - 세션 백업

**일시**: 2026-01-21
**목적**: ProbGate 필터 효과성 검증 via Shadow Trade 분석
**RAG Project**: trading_bot

---

## 1. 연구 목표

ProbGate (확률 게이트) 필터가 실제로 손실을 방지하는지 정량적으로 검증

### Shadow Trade 개념
- **정의**: ProbGate에 의해 거부된 시그널이 실제로 실행되었다면 어떤 결과가 나왔을지 시뮬레이션
- **목적**: 필터의 효과성 측정 (필터가 없었다면 발생했을 손실 계산)

---

## 2. MODE63 설정 요약

```json
{
  "_description": "MODE 63: MODE49 + min_fib_gap_pct 필터",

  "use_min_fib_gap_filter": true,
  "min_fib_gap_pct": 0.5,
  "track_shadow_trades": true,

  "tp_mode": "trailing_only",
  "use_prob_gate": true,
  "prob_gate_thr_long": 0.55,

  "entry_offset_ratio": 0.3,
  "rr_entry_mode": "offset_ratio",

  "use_dynamic_fib": true,
  "dynamic_fib_tf": "1d",
  "dynamic_fib_space": "linear",

  "trailing_activation_atr": 2.0,
  "trailing_distance_atr": 1.0
}
```

---

## 3. 백테스트 결과 (2021-2024)

### 3.1 실제 거래 (Actual Trades)

| 지표 | 값 |
|------|-----|
| 총 거래 수 | 31 |
| 승률 | 51.6% (16W / 15L) |
| 총 PnL | **-$57.68** |
| 평균 이익 | $18.59 |
| 평균 손실 | $23.67 |
| RR Ratio | 0.79 |
| 최종 자본 | $9,942.32 |

### 3.2 Shadow Trades (ProbGate 거부된 시그널)

| 지표 | 값 |
|------|-----|
| 총 거래 수 | 171 |
| 승률 | 42.7% |
| 총 PnL | **-$704.47** |

### 3.3 비교 분석

| 항목 | 실제 | Shadow | 차이 |
|------|------|--------|------|
| 거래 수 | 31 | 171 | -140 (82% 필터링) |
| 승률 | 51.6% | 42.7% | +8.9%p 개선 |
| 총 손실 | -$57.68 | -$704.47 | **$646.79 방어** |

---

## 4. 핵심 발견사항

### 4.1 ProbGate 효과성: 검증됨

```
ProbGate가 필터링한 171개 시그널:
- 승률: 42.7% (낮음)
- 예상 손실: -$704.47
- 결론: ProbGate가 $704.47의 손실을 방지함
```

### 4.2 그럼에도 손실이 발생한 이유

```
문제: RR Ratio = 0.79 < 1.0

평균 이익: $18.59
평균 손실: $23.67
→ 이기면 $18.59, 지면 -$23.67

손익분기 승률 공식:
BE_WR = 1 / (1 + RR) = 1 / (1 + 0.79) = 55.9%

현재 승률: 51.6%
필요 승률: 55.9%
갭: -4.3%p

결론: 승률이 손익분기점에 4.3%p 미달
```

### 4.3 기대값 계산

```python
# 기대값 공식
EV = (Win_Rate × Avg_Win) - (Loss_Rate × Avg_Loss)
EV = (0.516 × $18.59) - (0.484 × $23.67)
EV = $9.59 - $11.46
EV = -$1.87 per trade

# 31 거래 × -$1.87 = -$57.97 (실제: -$57.68)
```

---

## 5. 결론

### ProbGate 판정: 효과적

| 평가 항목 | 결과 |
|----------|------|
| 손실 방어 | $646.79 (92% 방어) |
| 승률 개선 | +8.9%p |
| 거래 품질 | 향상됨 |

### 전략 전체 판정: 수익성 부족

| 문제점 | 현재 값 | 필요 값 |
|--------|---------|---------|
| RR Ratio | 0.79 | ≥1.0 |
| 승률 | 51.6% | ≥55.9% |

### 개선 방향

1. **RR Ratio 개선** (우선순위 높음)
   - SL 거리 축소 (현재 Fib 기반)
   - TP 개선 (trailing_only 검토)

2. **승률 개선**
   - 추가 필터 도입
   - prob_gate_thr_long 상향 (0.55 → 0.60)

3. **둘 중 하나 달성 시 수익성 확보**
   - RR=1.0 + 승률 51.6% → BE
   - RR=0.79 + 승률 56% → BE

---

## 6. 수정된 코드

### 6.1 AttributeError 수정 (Line 5181)

```python
# Before (에러)
actual_pnl = result.get('total_pnl_usd', 0)

# After (수정됨)
actual_pnl = result.summary().get('total_pnl_usd', 0)
```

**원인**: `BacktestResult`는 dataclass이며 `.get()` 메서드가 없음. `.summary()` 호출 필요.

---

## 7. 관련 파일

| 파일 | 역할 |
|------|------|
| `configs/mode63.json` | 챔피언 설정 (Shadow Trade 활성화) |
| `backtest_strategy_compare.py` | Shadow Trade 로직 구현 |
| `configs/mode49.json` | MODE63 기반 설정 |

---

## 8. 다음 단계 제안

1. **RR Ratio 개선 실험**
   - `fib_sl_fallback_mult` 조정 (1.5 → 1.2)
   - `trailing_distance_atr` 조정 (1.0 → 0.8)

2. **승률 개선 실험**
   - `prob_gate_thr_long`: 0.55 → 0.60
   - `min_fib_gap_pct`: 0.5 → 0.7

3. **새로운 MODE64 생성**
   - RR 또는 승률 개선 파라미터 적용
   - Shadow Trade 비교 분석 지속

---

**세션 종료**: 2026-01-21
**다음 작업**: RR Ratio 개선 또는 승률 개선 실험
