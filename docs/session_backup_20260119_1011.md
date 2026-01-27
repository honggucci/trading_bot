# Session Backup: PR4-R4b + PR4-R5 구현

**Date**: 2026-01-19 10:11
**Project**: trading_bot
**Session**: PR4 손실 원인 분석 및 Entry RR Gate + MFE Profit Lock 구현

---

## 완료된 작업

### 1. PR4-R4b: MFE Profit Lock (이전 세션)
- MFE >= 1.5 ATR 도달 시 50% 부분익절
- SL을 entry + buffer(0.1 ATR)로 이동
- MODE24 생성 및 테스트

### 2. PR4-R5: Entry RR Gate (이번 세션)
- 진입 전 RR_net 계산: `(TP1 - entry - fees) / (entry - SL + fees)`
- min_rr_net(1.2) 미만 시 진입 거부
- LONG/SHORT 양쪽에 구현

### 3. Combined Approach (MODE26)
- RR Gate + Profit Lock 결합
- 5개 기간 테스트 완료

---

## 테스트 결과 요약

| Mode | 설명 | Total PnL | 개선 | Trades |
|------|------|-----------|------|--------|
| MODE22 | Baseline | -$247.28 | - | 43 |
| MODE24 | + Profit Lock | -$182.89 | +$64 | 59 |
| MODE25 | + RR Gate | -$216.85 | +$30 | 29 |
| **MODE26** | **+ Both** | **-$164.13** | **+$83** | 37 |

### 기간별 결과
- **2021-Q1 (BULL)**: -$198 → -$76 (+$122 개선) - 가장 큰 개선
- **2021-11 (baseline)**: $118 → $45 (-$73 퇴보) - Trade-off
- **2023-Jul (RANGE)**: -$154 → -$118 (+$36 개선)
- **2025-Jan (OOS)**: -$12 → -$16 (-$4 퇴보)

### 핵심 지표 개선
- SL 청산 비율: 64% → 42% (목표 45% 달성)
- 트레이드 수: 43 → 37 (-14%)
- 트레이드당 손실: -$5.75 → -$4.44 (23% 개선)

---

## 생성/수정된 파일

### 새 파일
| 파일 | 용도 |
|------|------|
| `configs/mode25.json` | RR Gate only |
| `configs/mode26.json` | RR Gate + Profit Lock |
| `scripts/test_mode25_rr_gate.py` | RR Gate 단독 테스트 |
| `scripts/test_mode26_combined.py` | 전체 비교 테스트 |

### 수정된 파일
| 파일 | 변경 내용 |
|------|-----------|
| `scripts/backtest_strategy_compare.py` | RR Gate 로직 추가 (LONG/SHORT) |

---

## 구현 상세

### Entry RR Gate 로직 (backtest_strategy_compare.py)
```python
# === PR4-R5: Entry RR Gate ===
if self.config.use_rr_gate:
    tp_target = tp1 if self.config.rr_gate_use_tp1 else tp2
    total_cost_pct = (self.config.fee_bps + self.config.slippage_bps) * 2 / 10000
    fee_cost = entry_price * total_cost_pct
    profit_gross = tp_target - entry_price
    loss_gross = entry_price - sl
    profit_net = profit_gross - fee_cost
    loss_net = loss_gross + fee_cost
    rr_net = profit_net / loss_net if loss_net > 0 else 0

    if rr_net < self.config.min_rr_net:
        rr_gate_rejects['RR_GATE_LONG_LOW_RR'] += 1
        pending_long_signal = None
        continue
```

### Config 옵션
```python
# PR4-R5: Entry RR Gate
use_rr_gate: bool = False
min_rr_net: float = 1.2
rr_gate_use_tp1: bool = True

# PR4-R4b: MFE Profit Lock
use_breakeven: bool = False
be_mfe_atr: float = 1.5
be_partial_pct: float = 0.5
be_buffer_atr: float = 0.1
```

---

## 다음 작업 제안

1. **min_rr_net 최적화**: 1.2 외에 1.0, 1.5 테스트
2. **be_mfe_atr 최적화**: 1.5 외에 1.0, 2.0 테스트
3. **2021-11 퇴보 분석**: 왜 좋은 구간에서 성능이 떨어지는지 분석
4. **SHORT 진입 로직 개선**: 현재 LONG-only, SHORT 추가 검토
5. **Out-of-Sample 확장 테스트**: 더 많은 OOS 기간 검증

---

## 결론

**MODE26 (RR Gate + Profit Lock)이 현재 최적 설정**
- 총 PnL: +$83 개선 (-$247 → -$164)
- 손실 구간(2021-Q1)에서 가장 큰 효과
- Trade-off: 좋은 구간(2021-11)에서 일부 퇴보

근본적으로 전략이 여전히 손실(-$164)이므로, 추가적인 진입 로직 개선이나 다른 접근법 검토 필요.

---

## RAG 임베딩 정보
- **project**: trading_bot
- **session_id**: 20260119_1011
- **tags**: PR4, RR_Gate, Profit_Lock, MODE26, backtest
