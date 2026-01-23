# MODE78 매매 로직 검증 체크리스트

> **최우선 목표: 이 체크리스트의 모든 항목을 100% 완료**

---

## 진행률: 10/19 (53%)

---

## Step 1: 데이터 로딩 & Warmup

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1.1 | OHLCV 데이터 (5m, 15m, 1h, 4h, 1w) 올바르게 로딩? | ✅ | 완료 |
| 1.2 | Warmup 기간 동안 지표 초기화? | ⬜ | 미검증 |
| 1.3 | ATR 계산 올바름? | ⬜ | 미검증 |

---

## Step 2: Dynamic Fib Anchor (1W)

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 2.1 | ZigZag pivot 올바르게 감지? | ⚠️ | 스윙은 맞지만 너무 많음 (14개/년) |
| 2.2 | pivot 확정 후에만 사용? (lookahead 없음) | ✅ | 검증 완료 |
| 2.3 | Fib high/low가 합리적인 값? | ⬜ | 너무 자주 변경됨 → k 조정 필요 |

### 2.1 이슈: ZigZag 피봇 과다 감지

**현재 상황:**
- 54개 1W 바 (1년) → 14개 피봇 감지
- `reversal_atr_mult = 1.5`, ATR 평균 = $9,412
- Threshold $14k로 12~14% 작은 스윙까지 잡힘

**해결책: Hilbert 기반 동적 k**
- `src/regime/wave_regime.py`의 amplitude 활용
- 사이클 강하면 k↓, 약하면 k↑
- 목표: 14개 → 4-6개 피봇

---

## Step 3: 다이버전스 신호 감지 (5m)

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 3.1 | StochRSI 계산 올바름? | ✅ | 17.5% 과매도 비율 확인 |
| 3.2 | Oversold/Overbought 감지? | ✅ | state 모드 적용됨 |
| 3.3 | Regular/Hidden Div 계산? | ⬜ | 미검증 |
| 3.4 | 다이버전스 가격 계산? | ⬜ | 미검증 |

---

## Step 4: Fib Zone 확인

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 4.1 | Dynamic Fib 레벨 계산? | ⬜ | 미검증 |
| 4.2 | 가격이 Zone 내에 있는지 체크? | ⬜ | 미검증 |

---

## Step 5: Limit Order 생성

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 5.1 | Entry 가격 계산 (RR 2:1 역산)? | ✅ | `Entry = (TP + 2×SL) / 3` |
| 5.2 | SL 계산 (Micro SL 1H swing)? | ✅ | `swing_low - 0.5×ATR` |
| 5.3 | TP 계산 (ATR 기준)? | ✅ | TP1=2ATR, TP2=3ATR, TP3=4ATR |

---

## Step 6: Limit Order Fill

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 6.1 | Fill 조건 (bar['low'] <= entry_limit)? | ⬜ | 미검증 |
| 6.2 | Fill 가격 (min(entry_limit, open))? | ⬜ | 미검증 |

---

## Step 7: 포지션 관리

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 7.1 | SL Hit 체크? | 🔴 | **BUG: gap 시 낙관적 청산가** |
| 7.2 | TP Hit 체크? | ✅ | 정상 |
| 7.3 | Partial Exit (2R 40%)? | ✅ | 정상 |
| 7.4 | Trailing Stop? | ⬜ | 미검증 |

### 7.1 버그: SL 청산가 계산

**위치**: `scripts/backtest_strategy_compare.py` 라인 ~4181-4196

**현재 코드**:
```python
if bar['low'] <= long_position['sl']:
    exit_price = long_position['sl']  # SL 가격으로 청산
```

**문제**: Gap 발생 시 `bar['low'] < sl`이면 실제로는 `low`에서만 청산 가능

**수정 필요**:
```python
exit_price = max(long_position['sl'], bar['low'])  # gap 반영
```

---

## 상태 범례

| 기호 | 의미 |
|------|------|
| ✅ | 완료/정상 |
| ⚠️ | 문제 발견 (수정 필요) |
| 🔴 | 버그 발견 (즉시 수정) |
| ⬜ | 미검증 |

---

## 수정 완료된 버그 (이전 세션)

- ✅ initial_sl 미설정 → 수정 완료
- ✅ remaining 계산 불일치 → 수정 완료
- ✅ atr_15m 미정의 → 수정 완료
- ✅ dynfib_ratios 중복 → 수정 완료

---

## 업데이트 이력

- 2026-01-23: 최초 생성, 진행률 53%
