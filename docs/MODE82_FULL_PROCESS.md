# MODE82 전체 프로세스 문서

> **작성일**: 2026-01-27
> **목적**: MODE82의 모든 모듈/함수/설정값/발견된 버그를 정리하여 다음 세션에서 빠르게 복구
> **핵심 파일**: `scripts/backtest_strategy_compare.py`, `configs/mode82.json`

---

## 1. MODE82 개요

- **전략**: Multi-TF Regime Aggregator (1H) 기반 Mean-Reversion
- **구조**: ZigZag(1W,1D) + ProbGate(4H,1H) 합의 → 레짐 분류 → StochRSI 과매도 진입
- **Trigger TF**: 1h (5m fallback OFF)
- **Anchor TF**: 1h

---

## 2. 전체 프로세스 흐름

```
1. 데이터 로드 (1h Parquet, UTC+8 → UTC 변환)
2. 지표 계산: RSI(14), StochRSI(period=26, k=3), ATR
3. Multi-TF Regime Aggregator → BULL/RANGE/BEAR
4. StochRSI 과매도 시그널 감지 (K <= 20)
5. Divergence 확인 (Regular/Hidden Bullish)
6. Entry 결정:
   - DivMid 모드: mid_entry 지정가 (현재 활성)
   - Fixed RR 모드: Fib 기반 SL → RR 2:1 역산 (DivMid OFF 시)
7. RR Limit Entry: TTL 168h 내 체결 대기
8. Position Sizing: risk_usd=$25 / SL거리, max $5000 notional
9. Exit 관리:
   - SL: 1 ATR (고정)
   - TP1: 1 ATR → 50% 부분청산 (BULL=trailing, RANGE/BEAR=100%청산)
   - Trailing: activation 1ATR, distance 0.5ATR
   - Overbought Exit: StochRSI >= 70
   - Early Exit: 24h 후 MFE < 0.3 ATR
   - Gate Flip Exit: ProbGate 반전 시
   - 15m Div Exit: 반대 다이버전스 감지
```

---

## 3. 핵심 파일 목록

### 3-1. 백테스트 엔진

| 파일 | 역할 |
|------|------|
| `scripts/backtest_strategy_compare.py` | **메인 백테스트** (~8000+ lines) |
| `configs/mode82.json` | MODE82 설정 파일 |

### 3-2. src/ 모듈

| 모듈 | 파일 | 역할 |
|------|------|------|
| **anchor** | `src/anchor/stochrsi.py` (291줄) | StochRSI 계산, 과매도 구간 탐지 |
| **anchor** | `src/anchor/exit_logic.py` | SL/TP 레벨 계산, TrailingState |
| **anchor** | `src/anchor/divergence.py` | RSI 다이버전스 탐지 |
| **anchor** | `src/anchor/confluence.py` | 다중 지표 컨플루언스 |
| **anchor** | `src/anchor/unified_signal.py` | V2 시그널 생성 |
| **regime** | `src/regime/multi_tf_regime.py` (439줄) | **Multi-TF Regime Aggregator** |
| **regime** | `src/regime/prob_gate.py` (921줄) | Temperature-Scaled ProbGate |
| **regime** | `src/regime/regime_strategy.py` (354줄) | 레짐별 파라미터 테이블 |
| **risk** | `src/risk/manager.py` (339줄) | RiskManager (circuit breaker) |
| **zone** | `src/zone/zone_div.py` (579줄) | 다이버전스 존 계산 |
| **zone** | `src/zone/builder.py` | Fib Zone 빌더 |
| **trigger** | `src/trigger/trigger_a.py` | Spring/UTAD 트리거 (5m) |
| **trigger** | `src/trigger/trigger_b.py` | Z-score 트리거 (5m) |
| **context** | `src/context/zigzag.py` | ZigZag 유틸리티 |
| **context** | `src/context/fibonacci.py` | Fibonacci 유틸리티 |
| **context** | `src/context/dynamic_fib_anchor.py` | ZigZagState (동적 Fib) |
| **gate** | `src/gate/hmm_entry_gate.py` | HMM Entry Gate |
| **utils** | `src/utils/timeframe.py` (305줄) | TF 변환 시스템 |
| **backtest** | `src/backtest/engine.py` (812줄) | V2/V3 백테스트 엔진 |

---

## 4. 핵심 함수 상세 (backtest_strategy_compare.py)

### 4-1. StochRSI 계산 (THE BUG)

**파일**: `scripts/backtest_strategy_compare.py`
**라인**: 1030-1065

```python
def calc_stoch_rsi(close, period=14, k_period=3, d_period=3, rsi_period=14):
    """TradingView Identical StochRSI %K 계산"""
    close = np.asarray(close, dtype=np.float64)
    rsi = talib.RSI(close, timeperiod=rsi_period)    # Step 1: RSI(rsi_period)
    rsi_s = pd.Series(rsi)
    lo = rsi_s.rolling(period, min_periods=period).min()   # Step 2: Stoch(period)
    hi = rsi_s.rolling(period, min_periods=period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi_s - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)  # ← fillna(0) 문제!
    k = stoch.rolling(k_period, min_periods=k_period).mean() * 100.0  # Step 3: K(k_period)
    return k.to_numpy()
```

**호출 위치** (라인 3679-3680, 현재 하드코딩됨):
```python
df['rsi'] = calc_rsi_wilder(df['close'].values, period=14)  # RSI(14) 하드코딩
df['stoch_k'] = calc_stoch_rsi(df['close'].values, period=config.stoch_rsi_period,
                                k_period=3, d_period=3, rsi_period=14)  # RSI(14) 하드코딩
```

**실제 호출 파라미터**: `rsi_period=14`, `period=26` (config.stoch_rsi_period), `k_period=3`

### 4-2. RSI 계산

**라인**: 1025-1028

```python
def calc_rsi_wilder(close, period=14):
    """Wilder's RSI 계산 (talib 사용)"""
    close = np.asarray(close, dtype=np.float64)
    return talib.RSI(close, timeperiod=period)
```

### 4-3. Config 클래스

**라인**: 233-849+

주요 기본값:
| 필드 | 기본값 | mode82 설정 |
|------|--------|------------|
| `rsi_period` | 26 | 14 |
| `stoch_rsi_period` | 26 | 26 |
| `stoch_rsi_oversold` | 20.0 | 20.0 |
| `stoch_rsi_overbought` | 80.0 | 70.0 |
| `stoch_signal_mode` | "state" | "state" |
| `trigger_tf` | "5m" | "1h" |
| `anchor_tf` | "15m" | "1h" |
| `use_regime_aggregator` | false | true |
| `use_div_mid_entry` | false | false (json에 없으나 backtest 기본) |
| `use_rr_limit_entry` | false | true |
| `rr_limit_target` | 2.0 | 2.0 |

### 4-4. 시그널 생성 (StochRSI 과매도)

**라인**: 4056-4101

```python
# prev_confirmed_stoch = 확정된 이전 15m 바의 StochK
prev_confirmed_stoch = df_15m_slice['stoch_k'].iloc[-2]

# state 모드: StochK <= oversold_thr → LONG 시그널
if stoch_signal_mode == "state":
    long_signal_triggered = (prev_confirmed_stoch <= oversold_thr)  # oversold_thr=20.0
```

**레짐별 StochRSI 임계값 오버라이드** (라인 4062-4068):
```python
if use_regime_aggregator:
    oversold_thr = config.regime_stoch_rsi.get(current_regime, 30.0)
    # BULL=20, RANGE=20, BEAR=20 (mode82.json 기준)
```

### 4-5. DivMid Entry

**라인**: 4906-4931 (DivMid 모드)
**라인**: 6730-6743 (DivMid entry logic)

```python
# DivMid: mid_entry = (break_price + div_price) / 2 ≈ 현재가
# SL = break_price - N×ATR
# TP = mid_entry + N×ATR (1:1 RR)
```

**mode82에서 DivMid는 OFF** (`use_div_mid_entry: false`)

### 4-6. Fixed RR Entry (Fib 기반)

**라인**: 4981-5025

```python
# DivMid OFF + DivBreak OFF → Fib 기반 SL/TP 자동 활성화
# 1. Dynamic Fib 레벨 수집
fib_levels = get_dynamic_fib_levels(...)

# 2. SL = 가격 아래 가장 가까운 Fib
prev_fib = fib_levels_below[0]
temp_sl = prev_fib - buffer

# 3. Entry 역산 (RR 2:1 보장)
entry_limit = (temp_tp + target_rr * temp_sl) / (1 + target_rr)
```

### 4-7. RR Limit Entry

**라인**: 4899-5165

```python
# TTL 168h 내 체결 대기
# fill_on = "low": bar['low'] <= entry_limit → 체결
# price_ref = "close": 기준가 = close
```

### 4-8. Position Sizing (Risk-Fixed)

**라인**: 878-905

```python
# qty = risk_usd_per_trade / sl_distance
# max: max_notional_usd ($5000)
# min SL distance: 0.1 ATR (qty 폭주 방지)
```

### 4-9. Exit Logic

#### Trailing Stop (라인 4260-4305)
```python
# activation: MFE >= 1.0 ATR
# distance: 고점 - 0.5 ATR
```

#### Overbought Exit (라인 4591-4611)
```python
# StochRSI >= 70 → 전량 청산
```

#### Early Exit (라인 4418-4557)
```python
# 24h 후 MFE < 0.3 ATR → 조기 청산
```

#### Gate Flip Exit (라인 4500-4520)
```python
# ProbGate SHORT 신호 연속 3+ bars + 손실 중 → 청산
```

#### 15m Div Exit (라인 4680-4710)
```python
# 반대 방향 다이버전스 감지 → 청산
```

### 4-10. Fib 기반 SL 계산

**라인**: 3264-3370 (`calc_fib_based_sl`)

```python
# 1. entry_fib = entry_price 근처(이하) Fib 레벨
# 2. prev_fib = entry_fib 바로 아래 Fib 레벨
# 3. fib_gap = entry_fib - prev_fib
# 4. buffer = fib_gap × buffer_ratio (1.0 = min(atr, fib_gap))
# 5. SL = prev_fib - buffer
```

---

## 5. Multi-TF Regime Aggregator 상세

### 5-1. 구조

```
src/regime/multi_tf_regime.py

RegimeAggConfig:
  weight_1w_zz: 0.40   ← 1W ZigZag
  weight_1d_zz: 0.25   ← 1D ZigZag
  weight_4h_pg: 0.20   ← 4H ProbGate
  weight_1h_pg: 0.15   ← 1H ProbGate

ZigZag Prior:
  1w_up: 0.75, 1w_down: 0.25
  1d_up: 0.65, 1d_down: 0.35
  unknown: 0.50

Hysteresis:
  bull_enter: 0.65, bull_exit: 0.55
  bear_enter: 0.35, bear_exit: 0.45
```

### 5-2. 계산 흐름

```python
# 1. ZigZag → Prior 변환
zz_to_prior(direction="up", tf="1w") → 0.75

# 2. ProbGate → Shrink 적용
shrink_to_half(p_bull, uncertainty) → 0.5 + (p - 0.5) * (1 - u)

# 3. 가중합
score = 0.40 * p_1w + 0.25 * p_1d + 0.20 * p_4h + 0.15 * p_1h

# 4. EMA 스무딩

# 5. Hysteresis → BULL/RANGE/BEAR
if score >= bull_enter(0.65): BULL
if score <= bear_enter(0.35): BEAR
else: RANGE (with exit thresholds)
```

### 5-3. 주요 함수

| 함수 | 라인 | 역할 |
|------|------|------|
| `zz_to_prior()` | 113-133 | ZigZag 방향 → 사전확률 변환 |
| `shrink_to_half()` | 136-153 | 불확실성으로 0.5 방향 축소 |
| `compute_uncertainty()` | 156-170 | Temperature → 불확실성 |
| `compute_regime_score()` | 173-214 | 가중합 계산 |
| `update_regime_with_hysteresis()` | 217-257 | 히스테리시스 적용 |
| `MultiTFRegimeAggregator.update()` | 281-341 | 메인 업데이트 루프 |

---

## 6. ProbGate 상세

**파일**: `src/regime/prob_gate.py` (921줄)

### 주요 함수

| 함수 | 라인 | 역할 |
|------|------|------|
| `compute_atr_pct()` | 31-76 | Wilder ATR% 계산 |
| `rolling_zscore()` | 136-168 | 인과적 z-score |
| `compute_temperature_vol()` | 260-294 | ATR 변동성 기반 온도 |
| `normalize_score()` | 346-373 | 점수 정규화 (zscore/tanh) |
| `prob_from_score()` | 380-394 | sigmoid(score / T) → P_bull |
| `gate_action()` | 397-424 | P → +1/0/-1 (LONG/FLAT/SHORT) |

### ProbabilityGate.compute() 흐름 (라인 791-894)

```
1. normalize_score → zscore or tanh
2. compute_temperature → vol/instability/fixed
3. prob_from_score → sigmoid(score_norm / T)
4. apply_calibration → shrink + bias
5. gate_action → threshold 비교 → action
```

---

## 7. 레짐별 전략 파라미터

**파일**: `src/regime/regime_strategy.py`

| 파라미터 | BULL | RANGE | BEAR |
|---------|------|-------|------|
| stoch_rsi_threshold | 20 | 20 | 20 |
| require_reclaim | false | true | true |
| tp_mode | trailing | quick_exit | quick_exit |
| tp_partial_pct | 0.50 | 1.00 | 1.00 |
| risk_mult | 1.0 | 0.3 | 0.3 |

---

## 8. configs/mode82.json 전체 설정

```json
{
  "trigger_tf": "1h",
  "anchor_tf": "1h",
  "use_5m_entry_fallback": false,

  "use_regime_aggregator": true,
  "regime_weights": { "1w_zz": 0.40, "1d_zz": 0.25, "4h_pg": 0.20, "1h_pg": 0.15 },
  "zz_prior": { "1w_up": 0.75, "1w_down": 0.25, "1d_up": 0.65, "1d_down": 0.35 },
  "hysteresis": { "bull_enter": 0.65, "bull_exit": 0.55, "bear_enter": 0.35, "bear_exit": 0.45 },

  "use_dynamic_fib": true,
  "dynamic_fib_tf": "1d",
  "dynamic_fib_space": "log",
  "dynfib_ratios": [0.236, 0.382, 0.5, 0.618, 0.786],

  "stoch_rsi_oversold": 20.0,
  "stoch_rsi_overbought": 70.0,
  "rsi_period": 14,
  "stoch_rsi_period": 26,
  "stoch_signal_mode": "state",

  "regime_stoch_rsi": { "BULL": 20.0, "RANGE": 20.0, "BEAR": 20.0 },
  "regime_rsi_threshold": { "BEAR": 100.0 },
  "regime_require_reclaim": { "BULL": false, "RANGE": true, "BEAR": true },

  "use_rr_limit_entry": true,
  "rr_limit_target": 2.0,
  "rr_limit_ttl": "168h",
  "rr_limit_max_atr_dist": 999.0,
  "rr_limit_fill_on": "low",

  "use_div_break_sl": false,
  "use_fib_based_sl": false,
  "fib_sl_buffer_ratio": 1.0,

  "use_div_mid_entry": false,

  "tp_mode": "atr",
  "tp_atr_mults": [1.0, 2.0, 3.0],
  "regime_tp_mode": { "BULL": "trailing", "RANGE": "quick_exit", "BEAR": "quick_exit" },
  "regime_tp_partial_pct": { "BULL": 0.50, "RANGE": 1.00, "BEAR": 1.00 },

  "use_trailing_stop": true,
  "trailing_activation_atr": 1.0,
  "trailing_distance_atr": 0.5,

  "use_overbought_exit": true,
  "use_early_exit": true,
  "early_exit_duration": "24h",
  "early_exit_mfe_mult": 0.3,
  "use_gate_flip_exit": true,
  "use_opposite_div_early": true,
  "use_15m_div_exit": true,

  "regime_risk_mult": { "BULL": 1.0, "RANGE": 0.3, "BEAR": 0.3 },

  "use_risk_fixed_sizing": true,
  "risk_usd_per_trade": 25.0,
  "max_notional_usd": 5000.0,

  "sl_atr_mult": 1.0,
  "atr_tf_for_risk": "15m",
  "atr_period": 38
}
```

---

## 9. src/anchor/stochrsi.py 상세

**별도 TradingView 방식 구현 (백테스트에서 미사용)**

| 함수 | 라인 | 역할 |
|------|------|------|
| `tv_stoch_rsi()` | 17-58 | TradingView 동일 StochRSI (RSI + Stoch + SMA) |
| `is_oversold()` | 61-63 | StochRSI %D <= threshold 체크 |
| `is_overbought()` | 66-68 | StochRSI %D >= threshold 체크 |
| `pick_oversold_segments()` | 71-121 | 연속 과매도 구간 탐지 → DataFrame |
| `pick_oversold_segment_with_rule()` | 124-199 | 과매도 구간 선택 규칙 |
| `extract_ref_from_segment()` | 211-242 | 구간에서 RefPoint 추출 (최저 close) |
| `extract_ref_lowest_close()` | 245-290 | lookback_bars 기반 RefPoint 추출 |

**RefPoint** dataclass (라인 202-208): `idx, ts, price, rsi`

---

## 10. src/zone/zone_div.py 상세

**다이버전스 존 계산**

| 함수 | 라인 | 역할 |
|------|------|------|
| `calc_rsi()` | 65-68 | talib RSI |
| `calc_stoch_rsi()` | 71-88 | **talib.STOCHRSI 사용** (backtest와 다름!) |
| `find_oversold_segments()` | 91-124 | 과매도 구간 탐지 |
| `find_overbought_segments()` | 127-155 | 과매수 구간 탐지 |
| `get_reference_from_segment()` | 158-183 | 구간 기준점 추출 |
| `calc_regular_bullish_boundary()` | 186-248 | 정규 상승 다이버전스 경계 |
| `calc_hidden_bullish_range()` | 251-317 | 히든 상승 다이버전스 범위 |
| `get_div_zones()` | 434-578 | 메인: 4종 다이버전스 존 계산 |

**DivZone** dataclass (라인 19-51): `side, kind, boundary_price, range_low, range_high, ref_price, confidence`

> **중요**: `zone_div.py`의 `calc_stoch_rsi()`는 `talib.STOCHRSI(close, timeperiod=stoch_period, ...)`를 사용하는 반면,
> `backtest_strategy_compare.py`의 `calc_stoch_rsi()`는 수동 구현(RSI + rolling Stoch + SMA)을 사용함 → **서로 다른 값 산출!**

---

## 11. 데이터 구조

### Parquet 파일 위치
```
data/bronze/binance/futures/BTC-USDT/
├── 1h/    ← MODE82 메인 데이터
├── 15m/
├── 5m/
├── 4h/
├── 1d/
└── 1w/
```

### 타임스탬프 변환
```python
# Parquet의 timestamp = UTC+8
df['utc'] = df['timestamp'] - pd.Timedelta(hours=8)
```

---

## 12. 발견된 버그/이슈 (CRITICAL)

### BUG-1: StochRSI 계산 불일치 (가장 심각)

**증상**: 2025-06-05 02:00 UTC 기준
- 백테스트 `calc_stoch_rsi(RSI=14, Stoch=26, K=3)` → **32.22**
- TradingView StochRSI(RSI길이=26, 스토캐스틱=26, K=3) → **43.29**
- `talib.STOCHRSI(timeperiod=14, fastk=3)` → **43.95** (가장 가까움)
- 수동 RSI(26)+Stoch(26)+K(3) → **23.89**

**원인 분석**:
1. `calc_stoch_rsi()`의 `fillna(0.0)` → NaN을 0으로 채워 Stochastic 계산 오염
2. `talib.STOCHRSI`는 `timeperiod`를 RSI와 Stoch 둘 다에 사용 (같은 값)
3. 수동 구현은 `rsi_period`과 `period`(stoch)를 분리 → 불일치 발생
4. TradingView의 정확한 warm-up 방식이 talib과 미세하게 다름

**영향**: 모든 과매도/과매수 시그널이 잘못된 값에 기반 → 진입 타이밍 전부 부정확

**해결 방향**:
- `talib.STOCHRSI(close, timeperiod=14, fastk_period=3)` 사용이 TV에 가장 근접 (오차 ~0.7pt)
- 또는 `src/anchor/stochrsi.py`의 `tv_stoch_rsi()` 함수 활용 (현재 백테스트 미사용)
- 현재 하드코딩된 `rsi_period=14`를 어떤 값으로 바꿔야 하는지 재검증 필요

### BUG-2: RSI period 혼란

**증상**: Config 기본값 `rsi_period=26`인데 mode82.json은 14로 오버라이드. 라인 3679-3680에서 하드코딩 14.

**히스토리**:
- 원래: `config.rsi_period` 사용 (26)
- 이전 세션: RSI(26)이 TV와 매칭된다고 오판 → `rsi_period=26`으로 변경
- 현재 세션: RSI(14) 하드코딩으로 복원 (하지만 StochRSI 자체가 잘못됨)

### BUG-3: zone_div.py vs backtest StochRSI 불일치

**증상**:
- `src/zone/zone_div.py:71-88` → `talib.STOCHRSI()` 사용
- `scripts/backtest_strategy_compare.py:1030-1065` → 수동 구현 (RSI + rolling)
- 같은 데이터에서 **다른 StochRSI 값** 산출

### BUG-4: DivMid 구조적 문제 (SL=TP=1ATR)

**증상**: DivMid entry는 현재가 근처에 지정가 → R=1ATR 양방향 대칭 → 동전 던지기
- 30건 중 13건 Pure SL (43%), 8건 BE stop (27%)
- Runner PnL 합계: +$5.62 (50% 포지션 거의 무가치)

---

## 13. 검증 스크립트 목록

| 파일 | 목적 |
|------|------|
| `scripts/stochrsi_tv_compare.py` | talib vs TV RSI 비교 (RMA 직접 구현) |
| `scripts/stochrsi_tv_compare2.py` | 1-bar shift 가설 + brute force |
| `scripts/stochrsi_tv_compare3.py` | 3-point 파라미터 brute force (RSI×Stoch×K) |
| `scripts/stochrsi_tv_verify_rsi26.py` | RSI(26)+Stoch(26) 검증 |

---

## 14. 백테스트 결과 히스토리

| 버전 | Trades | WR | RR | PnL | 비고 |
|------|--------|------|-----|------|------|
| 원본 (RSI26 default) | 30 | 53.2% | 0.32 | -$287 | config 기본값 |
| RSI26 StochRSI 강제 | 26 | 34.6% | 0.40 | -$346 | 악화 |
| RSI14 하드코딩 + BEAR=0.0 | 22 | 27.3% | 0.41 | -$384 | BEAR 차단 |
| RSI14 하드코딩 + BEAR=0.3 | 40 | 35.0% | 0.47 | -$510 | 최악 |

**결론**: StochRSI 계산 자체가 틀려서 어떤 파라미터 조합도 의미 없음. 먼저 StochRSI를 TV와 일치시킨 후 재백테스트 필요.

---

## 15. 다음 세션 TODO

1. **StochRSI 수정**: `calc_stoch_rsi()`를 `talib.STOCHRSI` 또는 `tv_stoch_rsi()` 기반으로 교체
2. **TV 매칭 검증**: 수정 후 2025-06-05 02:00에서 43.29 확인
3. **재백테스트**: 올바른 StochRSI로 MODE82 재실행
4. **DivMid vs Fib Entry**: DivMid OFF + fixed_rr Fib 진입 비교
5. **BEAR 차단 재평가**: StochRSI 수정 후 레짐별 결과 확인

---

## 16. 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────┐
│                  MODE82 Architecture                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Layer 1: Context TFs (1W/1D/4H/1H)                │
│  ├─ ZigZag (1W, 1D) → Direction Prior              │
│  ├─ ProbGate (4H, 1H) → P_bull + Temperature       │
│  └─ MultiTFRegimeAggregator → BULL/RANGE/BEAR      │
│       weighted sum + EMA + hysteresis               │
│                                                      │
│  Layer 2: Anchor TF (1H for MODE82)                 │
│  ├─ RSI(14) via talib.RSI                           │
│  ├─ StochRSI(period=26, k=3) ← **BUG: 값 불일치** │
│  ├─ Dynamic Fib (1D ZigZag)                         │
│  └─ Divergence Detection (Regular/Hidden)           │
│                                                      │
│  Layer 3: Signal Generation                         │
│  ├─ StochRSI <= 20 → LONG signal                   │
│  ├─ Divergence confirmation                         │
│  ├─ Regime filter (BEAR RSI < 100 = always pass)    │
│  └─ Reclaim filter (RANGE/BEAR only)                │
│                                                      │
│  Layer 4: Entry                                      │
│  ├─ RR Limit: entry = (TP + 2×SL) / 3 (RR 2:1)   │
│  ├─ TTL: 168h (7일)                                 │
│  ├─ Fill: bar_low <= entry_limit                    │
│  └─ Position: $25 risk / SL거리, max $5000          │
│                                                      │
│  Layer 5: Exit Management                           │
│  ├─ SL: 1 ATR (고정)                                │
│  ├─ TP1: 1 ATR (RANGE/BEAR: 100%, BULL: 50%)       │
│  ├─ Trailing: 1ATR activation, 0.5ATR distance      │
│  ├─ Overbought: StochRSI >= 70                      │
│  ├─ Early Exit: 24h + MFE < 0.3 ATR                │
│  ├─ Gate Flip: ProbGate 반전 3+ bars                │
│  └─ 15m Div Exit: 반대 다이버전스                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 17. 주요 라인 번호 요약 (backtest_strategy_compare.py)

| 기능 | 라인 |
|------|------|
| `calc_rsi_wilder()` | 1025-1028 |
| `calc_stoch_rsi()` | 1030-1065 |
| Config class | 233-849 |
| `calc_fib_based_sl()` | 3264-3370 |
| RSI/StochRSI 계산 (하드코딩) | 3679-3680 |
| Regime Aggregator init | 3903-3926 |
| Regime Aggregator update | 4001-4048 |
| StochRSI 임계값 오버라이드 | 4062-4068 |
| StochRSI signal 생성 | 4070-4101 |
| DivMid TP1 부분청산 | 4173-4202 |
| Trailing Stop | 4260-4305 |
| Early Exit (TimeStop/StaleLoss) | 4418-4557 |
| Gate Flip Exit | 4500-4520 |
| Overbought Exit | 4591-4611 |
| 5m/15m Div Exit | 4680-4710 |
| DivMid entry setup | 4906-4931 |
| Fixed RR Fib entry | 4981-5025 |
| RR Limit entry | 4899-5165 |
| Position sizing (risk-fixed) | 878-905 |
