# MODE78 매매 로직 검증 체크리스트

> **최우선 목표: 이 체크리스트의 모든 항목을 100% 완료**

---

## 진행률: 20/20 (100%) ✅

---

## P0 신뢰성 체크 (2026-01-24)

**진행률: 6/6 PASS ✅**

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| P0-1 | Hilbert 인과성 | ✅ | 치팅 검출 테스트 PASS (Online != Batch 확인) |
| P0-2 | SL 갭 체결 모델 | ✅ | LONG `min(sl, open)`, SHORT `max(sl, open)` 검증 완료 |
| P0-3 | Entry/TP Invariant | ✅ | `offset_ratio` 모드로 정순 계산 |
| P0-4 | SL Source 단일화 | ✅ | `offset_ratio` 모드로 Micro SL 적용 |
| P0-5 | k-anchor 구현 | ✅ | extreme_ts 분리 저장 + atr_anchor 고정 + OHLC ALL PASS |
| P0-5b | Pending Reversal | ✅ | min_spacing 부족 시 pivot 손실 방지 (PENDING_START/CONFIRM 테스트 PASS) |

### P0-1 치팅 검출 테스트 결과

**테스트 스크립트**: `scripts/test_hilbert_causality.py`

**원리**: Online (prices[:t+1]) vs Batch (full prices) 비교
- 두 결과가 **달라야 정상** (같으면 미래정보 사용 의심)

**결과**:
```
hilbert        : PASS (Online != Batch) (avg_diff=1.012)
regime_switch  : PASS (Online != Batch) (avg_diff=0.450)
inverse        : PASS (Online != Batch) (avg_diff=1.168)
```

### P0-2 SHORT SL 검증 완료

**검증 위치**: `scripts/backtest_strategy_compare.py`
- 라인 4365: `exit_price = max(short_position['sl'], bar['open'])`
- 라인 6386: `exit_price = max(short_position['sl'], bar['open'])`

### P0-5 extreme_ts 분리 검증 결과 (2026-01-24)

**테스트 스크립트**: `scripts/show_zigzag_pivots.py`

**핵심 수정 사항:**
- ZigZag pivot은 2개의 timestamp를 가짐:
  - `extreme_ts`: 실제 고점/저점 발생 시점 (가격 기준)
  - `confirm_ts`: pivot 확정 시점 (전략 인지 시점)
- 이전 테스트는 `confirm_ts`를 날짜로, `extreme_price`를 가격으로 혼용 → OHLC 불일치

**수정 후 결과:**
```
#1: 2025-01-20 | HIGH | $109,588 | OHLC PASS (확정: 2025-04-14, lag 12주)
#2: 2025-04-14 | LOW  | $83,112  | OHLC PASS (확정: 2025-07-14, lag 13주)
#3: 2025-10-06 | HIGH | $126,200 | OHLC PASS (확정: 2025-11-17, lag 6주)
```

**Spacing 분석 (2종류):**
```
spacing_extreme (실제 스윙 간격): median 18.5주 >= 13주 목표 ✅
spacing_confirm (전략 인지 간격): median 15.5주
confirm_lag (확정 지연): median 9.5주, max 13.0주
```

**권장 설정**: `zigzag_min_spacing_weeks=13`

### P0-5b Pending Reversal 상태 머신 (2026-01-24)

**테스트 스크립트**: `scripts/test_pending_reversal.py`

**문제**: min_spacing 부족 시 `pass`만 하여 중요한 pivot 영구 손실

**해결책**: "Pending Reversal" 상태 머신
- reversal 감지 but spacing 부족 → **pending 상태** 진입
- pending 상태에서 반대 극점 추적, 취소/확정 조건 평가
- spacing 충족 시 확정

**수정 파일**: `src/context/dynamic_fib_anchor.py`
- `DynamicFibAnchorState`에 pending 필드 7개 추가
- `update_anchor_zigzag()`에서 pending 상태 전이 로직 구현

**테스트 결과**:
```
PENDING_START:   2 times
PENDING_CONFIRM: 2 times
PENDING_CANCEL:  0 times
OHLC: ALL PASS (3/3)
median(spacing_extreme) = 18.5w >= 13w
>>> P0-5b PASS
```

---

## Step 1: 데이터 로딩 & Warmup

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1.1 | OHLCV 데이터 (5m, 15m, 1h, 4h, 1w) 올바르게 로딩? | ✅ | 완료 |
| 1.2 | Warmup 기간 동안 지표 초기화? | ✅ | warmup=84 > atr_period=38, 신호 방지 체크 있음 |
| 1.3 | ATR 계산 올바름? | ✅ | talib.ATR, period=38, NaN 체크 정상 |

---

## Step 2: Dynamic Fib Anchor (1W)

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 2.1 | ZigZag pivot 올바르게 감지? | ✅ | P0-5로 해결: 14개→3개 (min_spacing 13주, OHLC ALL PASS) |
| 2.2 | pivot 확정 후에만 사용? (lookahead 없음) | ✅ | 검증 완료 |
| 2.3 | Fib high/low가 합리적인 값? | ✅ | P0-5 구현으로 median_spacing 13주 달성 |

### 2.1 이슈: ZigZag 피봇 과다 감지 → **해결됨**

**이전 상황:**
- 54개 1W 바 (1년) → 14개 피봇 감지
- `reversal_atr_mult = 1.5`, ATR 평균 = $9,412
- Threshold $14k로 12~14% 작은 스윙까지 잡힘

**해결책: Hilbert 기반 동적 k + min_spacing + extreme_ts 분리 (P0-5)**
- `compute_dynamic_k()` 함수 구현: `src/context/dynamic_fib_anchor.py`
- `min_spacing_weeks=13` 적용 (권장 설정)
- `extreme_ts` vs `confirm_ts` 분리 저장 (OHLC 정합 검증)
- **결과: 14개 → 3개 피봇, median(spacing_extreme) 18.5주, OHLC ALL PASS**

---

## Step 3: 다이버전스 신호 감지 (5m)

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 3.1 | StochRSI 계산 올바름? | ✅ | 17.5% 과매도 비율 확인 |
| 3.2 | Oversold/Overbought 감지? | ✅ | state 모드 적용됨 |
| 3.3 | Regular/Hidden Div 계산? | ✅ | divergence.py:92-126 검증 완료 |
| 3.4 | 다이버전스 가격 계산? | ✅ | binary search 경계가격 + d[-2] lookahead 방지 |

---

## Step 4: Fib Zone 확인

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 4.1 | Dynamic Fib 레벨 계산? | ✅ | get_dynamic_fib_levels() 검증 완료 |
| 4.2 | 가격이 Zone 내에 있는지 체크? | ✅ | is_near_dynamic_fib_level() 4가지 tolerance 모드 |

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
| 6.1 | Fill 조건 (bar['low'] <= entry_limit)? | ✅ | 검증 완료 (lines 3369, 4609) |
| 6.2 | Fill 가격 (min(entry_limit, open))? | ✅ | 보수적 모델 적용됨 (lines 3371, 4611) |

---

## Step 7: 포지션 관리

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 7.1 | SL Hit 체크? | ✅ | P0-2 확인: `min(sl, open)` 보수적 모델 적용됨 |
| 7.2 | TP Hit 체크? | ✅ | 정상 |
| 7.3 | Partial Exit (2R 40%)? | ✅ | 정상 |
| 7.4 | Trailing Stop? | ✅ | 2모드 지원 (ATR/R-based) lines 3891-4046 |

### 7.1 SL 청산가 계산 → **이미 수정됨 (P0-2)**

**위치**: `scripts/backtest_strategy_compare.py` 라인 ~4200

**현재 코드 (정상):**
```python
# LONG SL: 보수적 체결
exit_price = min(long_position['sl'], bar['open'])

# SHORT SL: 보수적 체결
exit_price = max(short_position['sl'], bar['open'])
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
- ✅ **entry_limit >= signal_price 해킹** → 스킵으로 수정 (2026-01-24)
  - 기존: `entry_limit = signal_price - 1` (RR 2.0 보장 깨짐, 실제 RR 4.0까지 발생)
  - 수정: RR 역산 결과가 signal_price보다 높으면 신호 스킵
  - 이유: 역산 공식상 불가능한 케이스는 땜질 금지, 신호 거부가 정답

---

## 업데이트 이력

- 2026-01-23 21:00: 최초 생성, 진행률 53%
- 2026-01-23 22:00: P0 신뢰성 체크 완료, P0-5 k-anchor 구현, 진행률 74%
  - P0-2 SL 갭 체결: 이미 `min(sl, open)` 적용 확인
  - P0-3 Entry/TP: `offset_ratio` 모드로 정순 계산 확인
  - P0-4 SL Source: Micro SL 정상 적용 확인
  - P0-5 k-anchor: `compute_dynamic_k()` 함수 구현, 14개→6개 피봇
- 2026-01-23 23:00: P0 5/5 PASS 완료, 진행률 79%
  - P0-1: 치팅 검출 테스트 추가 (`test_hilbert_causality.py`) - 3개 모드 PASS
  - P0-2: SHORT SL `max(sl, open)` 검증 완료 (라인 4365, 6386)
  - P0-5: min_spacing=13주로 조정 → median=13.0주 달성
  - 테스트 스크립트 추가: `test_min_spacing.py`, `test_hilbert_causality.py`
- 2026-01-24 00:00: P0-5 extreme_ts 버그 수정
  - **버그 발견**: confirm_ts와 extreme_price 혼용으로 OHLC 불일치
  - `DynamicFibAnchorState`에 `candidate_high_ts`, `candidate_low_ts` 필드 추가
  - `last_extreme_ts`, `last_confirm_ts`, `last_pivot_type` 필드 추가
  - `show_zigzag_pivots.py` 수정: OHLC 검증 ALL PASS (3/3)
  - spacing_extreme (실제 스윙 간격): median 18.5주 >= 13주 목표 달성
- 2026-01-24 01:00: P0-5b Pending Reversal 상태 머신 구현, 진행률 80%
  - **버그 발견**: min_spacing 부족 시 `pass`만 하여 pivot 영구 손실
  - `DynamicFibAnchorState`에 pending 필드 7개 추가
  - `update_anchor_zigzag()`에서 pending 상태 전이 로직 구현
  - `atr_anchor` 필드 추가 (pivot 확정 시 ATR 고정)
  - 테스트 스크립트 추가: `test_pending_reversal.py`
  - **결과**: PENDING_START 2회, PENDING_CONFIRM 2회, OHLC ALL PASS
- 2026-01-24 02:00: 전체 검증 완료, 진행률 100%
  - Step 1.2/1.3: Warmup 및 ATR 검증 완료
  - Step 3.3/3.4: Divergence 로직 검증 완료 (d[-2] lookahead 방지 확인)
  - Step 4.1/4.2: Dynamic Fib Zone 검증 완료 (4가지 tolerance 모드)
  - Step 6.1/6.2: Limit Fill 검증 완료 (보수적 모델)
  - Step 7.4: Trailing Stop 검증 완료 (ATR/R-based 2모드)
  - **버그 수정**: `entry_limit >= signal_price` 해킹 → 스킵으로 변경
    - 기존: `entry_limit = signal_price - 1` (RR 보장 깨짐)
    - 수정: `pending_long_signal = None; continue` (RR 불가시 스킵)
    - 위치: `scripts/backtest_strategy_compare.py` 라인 4569-4572
