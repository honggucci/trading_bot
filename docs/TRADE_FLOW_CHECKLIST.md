# 매매 흐름 체크리스트 (MODE47 v2)

## 검증일: 2026-01-20 (최신 업데이트)

---

## 현재 설정 요약

| 항목 | 값 | 비고 |
|------|-----|------|
| `stoch_rsi_oversold` | **30** | 완화 (20→30) |
| `stoch_rsi_period` | 14 | 기본값 유지 |
| `early_exit_duration` | **8h** | 변경 (2h→8h) |
| `use_hilbert_filter` | **false** | 비활성화 |
| `use_trend_filter_1h` | **false** | 비활성화 |
| `use_prob_gate` | true | ProbGate만 사용 |
| `prob_gate_thr_long` | 0.55 | p_bull > 55% |
| `dynfib_lookback_bars` | **100** | 변경 (52→100) |
| Entry Offset | **0.3** | PR-FIB-ENTRY (TP거리의 30%) |

---

## 전체 흐름

```
1️⃣ Fib 레벨 계산
    └─ 1D ZigZag Anchor → L0 Only Log Fib (~1% 간격)
    └─ Fallback: Macro Fib (범위 밖일 때)

2️⃣ 진입 시그널 감지
    └─ StochRSI Oversold (< 30)
    └─ Fib 레벨 근처 (ATR tolerance)
    └─ Regular Divergence (Hidden Div 비활성화)

3️⃣ Entry Gate 통과
    └─ ProbGate만 사용 (p_bull > 0.55)
    └─ Hilbert/Trend Filter 비활성화

4️⃣ RR 계산 + Limit Order
    └─ Entry = div_price - (TP거리 × 0.3) [PR-FIB-ENTRY]
    └─ TP = 다음 Fib 레벨
    └─ SL = Fib 기반 또는 ATR Fallback

5️⃣ 포지션 진입
    └─ Limit Order Fill (low 터치)
    └─ TTL 6h (미체결 시 취소)

6️⃣ 포지션 관리
    └─ StaleLoss: 8h 경과 후 손실 시 조기 손절
    └─ 5m_Short_Div: 반대 다이버전스 청산
    └─ Trailing Stop: 5m Div 후 50% 청산 + 트레일링

7️⃣ 청산 + 기록
    └─ SL/TP/Early Exit
    └─ PnL 계산 (수수료 포함)
```

---

## 최신 백테스트 결과 (2021-01-01 ~ 2021-06-30)

### 현재 설정 (StochRSI<30, Offset 0.3, StaleLoss 8h)
| 지표 | 값 |
|------|-----|
| Trades | **46** |
| Total PnL | **+$288** |
| per Trade | **+$6.26** |
| Win Rate | **60.9%** |

### 비교 (이전 설정들)
| 설정 | Trades | PnL | $/Trade | WR |
|------|--------|-----|---------|-----|
| 기존 (Hilbert ON) | 126 | -$766 | -$6.08 | 36.5% |
| StochRSI<20, Offset 0.5 | 16 | +$150 | +$9.37 | 56.2% |
| Hidden Div 추가 | 34 | -$88 | -$2.59 | 35.3% |
| **현재 (v2)** | **46** | **+$288** | **+$6.26** | **60.9%** |

---

## 단계별 검증

### 1️⃣ Fib 레벨 계산

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| 1D ZigZag Anchor | [x] | LOW/HIGH 확정 pivot 사용 |
| L0 Only | [x] | 7개 레벨 (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0) |
| Gap ~1% | [x] | 15m 스윙매매에 적합한 TP 간격 |
| Macro Fallback | [x] | 범위 밖일 때 $3120~$143360 사용 |
| Log Space | [x] | Log Fib 계산 정확성 |
| lookback | [x] | **100** 1D 봉 (기존 52) |

---

### 2️⃣ 진입 시그널 감지

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| StochRSI < 30 | [x] | Oversold 조건 (완화됨) |
| Fib 근처 | [x] | fib_gap tolerance 모드 |
| Regular Divergence | [x] | 일반 다이버전스만 사용 |
| Hidden Divergence | [x] | **비활성화** (테스트 후 악화) |

---

### 3️⃣ Entry Gate 통과

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| Hilbert Filter | [x] | **비활성화** |
| Trend Filter 1H | [x] | **비활성화** |
| Prob Gate | [x] | p_bull > 0.55 (유일한 필터) |

**ProbGate 로직:**
```
1. 1H Hilbert Score 계산
2. Temperature = f(volatility)
3. p_bull = sigmoid(score / T)
4. p_bull > 0.55 → LONG 허용
```

---

### 4️⃣ RR 계산 + Limit Order

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| PR-FIB-ENTRY | [x] | Entry = div_price - (TP거리 × **0.3**) |
| TP = Fib 레벨 | [x] | 다음 Fib 레벨 자동 선택 |
| SL = Fib 기반 | [x] | prev_fib - buffer |
| RR >= 2.3 | [x] | 0.3 offset으로 자동 보장 |

---

### 5️⃣ 포지션 진입

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| Limit Fill | [x] | `bar['low'] <= entry_limit` 조건 체결 |
| TTL 6h | [x] | 24 bars (15m 기준) 후 미체결 취소 |

---

### 6️⃣ 포지션 관리

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| StaleLoss | [x] | **8h** 후 손실 시 조기 손절 (기존 2h) |
| GateFlip | [x] | 손실 중 ProbGate 방향 반전 → 청산 |
| 5m_Short_Div | [x] | 5분봉 반대 다이버전스 → 청산 |
| Trailing Stop | [x] | 5m Div 후 50% 청산 + 트레일링 |

---

### 7️⃣ 청산 + 기록

| 항목 | 상태 | 검증 내용 |
|------|------|----------|
| SL 청산 | [x] | `bar['low'] <= sl` 시 즉시 청산 |
| TP 청산 | [x] | TP1/2/3 순차 부분청산 |
| 수수료 | [x] | entry_cost + exit_cost + funding |

---

## 변경 이력

- 2026-01-19: 초안 작성
- 2026-01-20: 7단계 검증 완료
- 2026-01-20: **v2 업데이트**
  - `stoch_rsi_oversold`: 25 → 20 → **30** (완화)
  - `early_exit_duration`: 2h → **8h** (연장)
  - `use_hilbert_filter`: true → **false** (비활성화)
  - `use_trend_filter_1h`: true → **false** (비활성화)
  - `dynfib_lookback_bars`: 52 → **100** (확장)
  - Entry Offset: 0.5 → **0.3** (완화)
  - Hidden Divergence: 테스트 후 **비활성화** (성과 악화)
  - 결과: 46 trades, **+$288**, WR **60.9%**
