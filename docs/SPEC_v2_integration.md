# 통합 명세 v2: Legacy(Zone) + WPCN(Gate/Execution)

## 0) Core Philosophy

- Market is fractal ranges (mean-reversion is default).
- Trend is a rare range-expansion event.
- Trade only at extremes (discount/premium) with absorption/failure confirmation.

---

## 1) Timeframe Roles (fixed terminology)

| Role | Timeframes | Purpose |
|------|------------|---------|
| **Context TFs** | 1W, 1D, 4H, 1H | fractal range position + regime agreement |
| **Anchor TF** | 15m | Zone builder (candidate price bands only) |
| **Trigger TF** | 5m | Absorption/Failure trigger (entry timing) |

### 1.1 Multi-TF Usage Rules (MUST - 위반 시 과적합/룩어헤드)

**올바른 멀티TF 사용 (살아남는 방식)**:
- 상위TF(1W~1H)는 **게이트/사전확률/사이징**으로만 사용
- 신호 생성은 **15m에서만**, 진입 확인은 **5m에서만**

**망하는 멀티TF 사용 (금지)**:
- "1W도 롱, 1D도 롱, 4H도 롱..." → 자기모순 + 과적합
- 상위TF 조건을 신호 생성에 섞기 → 파라미터 폭발

### 1.2 Multi-TF MUST 규칙 6개

| # | 규칙 | 설명 |
|---|------|------|
| 1 | **확정 캔들만** | 상위TF는 진행중 캔들(intrabar) 사용 금지 |
| 2 | **역할 제한** | 상위TF는 게이트/사이징/사전확률만 (신호 생성 금지) |
| 3 | **좌표 축약** | TF별 조건이 아닌 `context_score = Σ w_tf * z_tf`로 통합 |
| 4 | **레이어 분리** | 15m=Zone 생성, 5m=흡수/실패 트리거 |
| 5 | **반개구간 정렬** | 모든 TF는 [start, end) half-open window |
| 6 | **포지션 중 고정** | 상위TF 조건이 바뀌어도 포지션 중 정책/존 고정 |

### 1.3 상위TF 허용 용도 (이것만)

```python
@dataclass
class ContextTFUsage:
    # 허용 ✅
    trade_enable: bool      # 거래 허용/금지
    size_mult: float        # 포지션 크기 조절 (0.5~1.5)
    zone_score_prior: float # Zone 점수에 더할 사전확률

    # 금지 ❌
    # signal_direction: str  # 상위TF로 방향 결정 금지
    # entry_condition: bool  # 상위TF를 진입 조건으로 금지
```

---

## 2) Layer 1: Fractal Range Coordinates (Context TFs)

**목적**: "극단(할인/프리미엄)이 여러 해상도에서 동시에 발생했는지" 확인

For each tf in {1W, 1D, 4H, 1H, 15m} compute:

```python
center_tf: float  # anchored VWAP / rolling median / (optional) volume-profile POC
width_tf: float   # ATR / MAD / std
z_tf = (price - center_tf) / width_tf

balance_score_tf: float  # how range-like (low trend strength, low drift)
breakout_risk_tf: float  # risk of range break
```

### Context Agreement (단일 점수로 축약)

다수결이 아닌 **가중치 합**으로 한 번에 축약:

```python
# TF별 z-score를 단일 context_score로 축약
context_score = sum(w_tf * clamp(z_tf, -3, 3) for tf in TFs) / sum(w_tf)

# 가중치 예시 (장기 TF에 더 큰 가중치)
weights = {'1W': 0.4, '1D': 0.3, '4H': 0.2, '1H': 0.1}
```

- **LONG_BIAS**: `context_score <= -threshold` AND balance_score high AND breakout_risk low
- **SHORT_BIAS**: `context_score >= +threshold` similarly

---

## 3) Layer 2: 15m Zone Builder (Legacy as helper, NOT authority)

**Goal**: produce candidate zone band with confidence score, NOT a single deterministic level.

### 3.1 ZigZag & Fib (CAUSAL ONLY - 9-Persona Reviewed)

**HARD RULES**:
- ZigZag pivot must have `confirm_ts`.
- Fib swing (hi/lo) is computed ONLY from confirmed alternating pivot pair.
- **No pivot usage before confirmation.**
- **Fib는 "정답 레벨"이 아니라 근접도/밴드 피처로 취급**

```python
@dataclass
class ConfirmedPivot:
    idx: int
    price: float
    direction: int  # 1=high, -1=low
    confirm_ts: pd.Timestamp  # pivot 확정 시점
    ttl_ts: Optional[pd.Timestamp] = None  # 유효기간 (다음 pivot confirm까지)

# Fib 계산은 confirm_ts 이후에만
def fib_from_confirmed_pivots(pivot_a: ConfirmedPivot, pivot_b: ConfirmedPivot):
    # pivot_a.confirm_ts < pivot_b.confirm_ts 보장
    # current_ts <= pivot_b.ttl_ts 보장 (유효기간 체크)
    ...
```

### 3.1.1 Multi-Swing Consensus (퀀트 권고)

최신 스윙 1개에 올인하면 휩쏘우에 사망. **멀티 TF 스윙 합의**로 존 생성:

```python
@dataclass
class FibConsensus:
    # 15m/1h/4h 스윙별 Fib 레벨
    fib_15m: List[float]
    fib_1h: List[float]
    fib_4h: List[float]

    # 레벨 겹침 구간만 Zone 후보로 승격
    def get_confluence_zones(self, tolerance: float = 0.002) -> List[Tuple[float, float]]:
        # 2개 이상 TF에서 겹치는 레벨 밴드 반환
        ...
```

### 3.2 Oversold/Overbought Segments (StochRSI %D)

- Detect segments (<=20 or >=80) to locate "when" extremes occur.

### 3.3 Divergence (replace hard boundary with score)

**DO NOT** rely on bisection "feasible boundary" as primary.

Compute `divergence_strength` score using:
- price_extreme_delta
- rsi_extreme_delta
- time_gap
- ATR-normalized magnitude

```python
@dataclass
class DivergenceScore:
    regular_strength: float  # 0~1, 정규 다이버전스 강도
    hidden_strength: float   # 0~1, 히든 다이버전스 강도
    confidence: float        # 신뢰도
```

Optional: keep boundary calc only as auxiliary, with O(1) RSI update cache.

### 3.4 Zone Output (15m)

```python
@dataclass
class ZoneEvent:
    zone_low: float           # 하단 (밴드/호가 고려)
    zone_high: float          # 상단 (밴드/호가 고려)
    zone_score: float         # 0~1 (divergence + stoch + fib + context)
    invalid_level: float      # 레인지 붕괴 손절 라인
    target_center: float      # 회귀 목표 (center)
    direction: str            # 'long' | 'short'
    created_ts: pd.Timestamp
    ttl_ts: pd.Timestamp      # 유효기간 (다음 pivot confirm까지)

    # 리스크 매니저 권고: 포지션 없을 때만 존 변경
    is_locked: bool = False   # 포지션 보유 중이면 True
```

---

## 4) Layer 3: HMM Entry Gate (WPCN)

- Gate evaluated on **CONFIRMED 15m bar** (avoid 5m leaking future 15m state).
- Transition cooldown blocks entries when state delta > threshold.
- Long/Short permit by HMM state and trend strength.
- Size multiplier by VaR target (soft sizing).

### 4.1 Gate 병목 문제 및 개선 옵션

**현재 문제**: Gate가 "뒤에서 99% 차단" → 신호 생산 비용만 늘고 대부분 폐기

**개선 A (권장): Gate를 '앞'으로 끌어올리기**
```python
# 15m Zone 생성 단계에서 이미 Gate 선적용
if not hmm_gate.check_permit(direction, current_bar):
    return None  # Zone 자체를 생성하지 않음
```
→ 애초에 Zone을 덜 만들어서 낭비 제거

**개선 B: transition cooldown 발작 방지**
```python
# raw delta 대신 EMA로 완화
smoothed_delta = ema(posterior_delta, span=3)

# 또는 min_dwell_bars 추가
if bars_in_current_state < min_dwell_bars:
    return blocked

# 레인지 구간에서는 delta 완화
if balance_score > 0.7:
    effective_delta = transition_delta * 1.5  # 더 관대하게
```

```python
@dataclass
class HMMGateConfig:
    # Transition Cooldown
    transition_delta: float = 0.40
    cooldown_bars: int = 1

    # Soft Sizing
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Short Permit
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # Long Permit
    long_permit_enabled: bool = True
    long_permit_states: tuple = ('markup', 'accumulation', 're_accumulation')
```

---

## 5) Layer 4: 5m Trigger (MUST include absorption/failure)

**Entry allowed only if**:
1. price enters 15m zone band AND
2. at least one microstructure trigger holds

### Triggers (최소 1개 필수)

#### A) Spring/UTAD Reclaim
```
Long: low breaks zone_low → close reclaims above zone_low
Short: high breaks zone_high → close reclaims below zone_high
```

#### B) Failed Swing
```
Long: new low fails (higher low) → breaks prior minor high
Short: new high fails (lower high) → breaks prior minor low
```

#### C) Effort vs Result (Absorption)
```
spread/volume expands but close regains (lack of follow-through)
= 노력(스프레드/볼륨)은 큰데 결과(진행)가 안 나옴
```

#### D) (Optional) LTF Mini Divergence Score
```
5m에서 미니 다이버전스 점수 추가
```

---

## 6) Layer 5: Execution (WPCN 15m limit backtester/live)

### 6.1 5m 트리거 → 15m 실행 연결 옵션

**선택 1 (최소 변경, 빠름): 5m 트리거 → 15m 다음 봉에서 주문**
```python
# 5m에서 트리거 발생
if trigger_5m.fired:
    # 다음 15m 바 시작에 PendingOrder 생성
    next_15m_bar = align_to_15m(current_ts)
    create_pending_order(next_15m_bar, zone.zone_low, direction='long')
```

**선택 2 (정교, 권장): 체결 체크만 5m로 다운그레이드**
```python
# 주문 생성은 15m 기준
# 체결 체크만 5m high/low로 판단 (더 현실적)
def check_fill_5m(order, bar_5m):
    if bar_5m.low <= order.limit_price <= bar_5m.high:
        return Fill(price=order.limit_price, ts=bar_5m.ts)
```
→ 실제 진입 타이밍은 5m이므로 철학에 더 맞음

### 6.2 Execution Config

```python
@dataclass
class ExecutionConfig:
    # 지정가 주문
    pending_order_max_bars: int = 4   # 4봉(1시간) 미체결 시 취소
    fill_check_tf: str = '5m'         # '15m' or '5m' (선택 2면 '5m')

    # ATR 기반 SL/TP
    atr_period: int = 14
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 3.0

    # 최대 보유
    max_hold_bars: int = 32           # 32봉(8시간)

    # 비용
    maker_fee: float = 0.0002         # 지정가 0.02%
    taker_fee: float = 0.0005         # 시장가 0.05%
    slippage_pct: float = 0.0001      # 0.01%
```

---

## 7) Parameter Operation (9-Persona Reviewed)

**핵심 원칙**: "매 순간 최적화"가 아니라 **상태→파라미터 정책을 온라인으로 조금씩 업데이트**

### 7.1 필수 수정 6개 (이거 없으면 동적 파라미터 금지)

| # | 규칙 | 설명 |
|---|------|------|
| 1 | **Fast/Medium/Slow 계층 분리** | 매 순간 변경은 Fast(스케일/임계치)만 허용 |
| 2 | **One-step delay** | t 성과로 t 의사결정 수정 금지 (t+1 이후 반영) |
| 3 | **변화율 제한 + 포지션 중 고정** | 포지션 보유 중 파라미터 변경 금지 |
| 4 | **온라인 정책** | Optuna 금지 → 밴딧/베이지안/칼만 업데이트 |
| 5 | **비용+리스크 목표함수** | 수수료/슬리피지/펀딩 + variance/drawdown 패널티 |
| 6 | **챔피언/챌린저 승격** | 정책 변경은 OOS 검증 후 해시/로그 필수 |

### 7.2 킬스위치/가드레일

```python
@dataclass
class UpdateGuardrail:
    # 업데이트 중지 조건
    max_recent_losses: int = 3         # 최근 M거래 연속 손실 시 중지
    max_drawdown_pct: float = 0.05     # 5% DD 초과 시 중지
    regime_cooldown_bars: int = 4      # 레짐 전환 후 N봉 업데이트 금지

    # 롤백 조건
    oos_calmar_threshold: float = 0.5  # 2주 OOS Calmar 미달 시 롤백
    max_update_freq: int = 96          # 하루 최대 업데이트 횟수
```

### 7.3 파라미터 계층

```python
@dataclass
class ParamHierarchy:
    # FAST (매 바 변경 가능) - 스케일 계열만
    fast_params: Dict[str, float] = field(default_factory=lambda: {
        'width_multiplier': 1.0,      # k * width 형태
        'threshold_scale': 1.0,
    })

    # MEDIUM (레짐 변경 시만)
    medium_params: Dict[str, float] = field(default_factory=lambda: {
        'zig_depth': 0.01,
        'divergence_weight': 0.5,
    })

    # SLOW (챔피언/챌린저로만)
    slow_params: Dict[str, float] = field(default_factory=lambda: {
        'atr_period': 14,
        'fib_levels': [0.382, 0.5, 0.618],
    })
```

### 7.4 ParamPolicy (레짐 매핑)

```python
@dataclass
class ParamPolicy:
    regime: str                        # 'accumulation' | 'markup' | 'distribution' | 'markdown'
    policy_hash: str                   # 변경 추적용 해시
    created_ts: pd.Timestamp

    # 계층별 파라미터
    hierarchy: ParamHierarchy

    # 변경 제한
    max_change_rate: float = 0.1       # 10% 이상 급변 금지
    smoothing_factor: float = 0.8      # EMA 스무딩

    # 운영 KPI 제약 (퀀트 트레이더 권고)
    target_trades_per_day: Tuple[int, int] = (2, 8)  # 일 평균 거래 수 범위
    target_hold_bars: Tuple[int, int] = (4, 24)      # 평균 보유 시간 범위
```

---

## 8) Mandatory Validations (anti-illusion tests)

| Test | Description |
|------|-------------|
| **Causality** | No pivot usage before confirm_ts |
| **Window Alignment** | Half-open [start, end) - no boundary leakage |
| **Gate Integrity** | Uses only confirmed 15m state (no future info) |
| **Trigger Comparison** | Compare "zone-only" vs "zone+trigger" baseline |
| **Parameter Continuity** | Performance should be continuous w.r.t. param changes |

---

## 9) Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Trading Bot v2                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Context TFs: 1W/1D/4H/1H]                                     │
│  └─ Fractal Range Coordinates (z-score)                         │
│  └─ Trade Bias: LONG / SHORT / NEUTRAL                          │
│                    │                                            │
│                    ▼                                            │
│  [Anchor TF: 15m] ─────────────────────────────────────────     │
│  ├─ ZigZag (confirmed pivots only)                              │
│  ├─ Fib Levels                                                  │
│  ├─ StochRSI Oversold Segments                                  │
│  ├─ Divergence Score (NOT boundary)                             │
│  └─ OUTPUT: ZoneEvent (band + score + invalid)                  │
│                    │                                            │
│                    ▼                                            │
│  [HMM Entry Gate] ──────────────────────────────────────────    │
│  ├─ Transition Cooldown                                         │
│  ├─ Long/Short Permit                                           │
│  └─ Soft Sizing (VaR)                                           │
│                    │                                            │
│                    ▼                                            │
│  [Trigger TF: 5m] ──────────────────────────────────────────    │
│  ├─ Zone Entry Check                                            │
│  └─ Absorption/Failure Trigger (MUST)                           │
│       ├─ Spring/UTAD Reclaim                                    │
│       ├─ Failed Swing                                           │
│       └─ Effort vs Result                                       │
│                    │                                            │
│                    ▼                                            │
│  [Execution] ───────────────────────────────────────────────    │
│  ├─ Limit Order (pending 1~4 bars)                              │
│  ├─ ATR SL/TP                                                   │
│  ├─ Time-stop (32 bars)                                         │
│  └─ Cost Model (fees/slippage/funding)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9.1) WPCN 프로세스 흐름 (15분봉 기준)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        데이터 흐름 (15분봉 기준)                           │
└─────────────────────────────────────────────────────────────────────────┘

1. 데이터 로드
   └─ data/bronze/binance/futures/BTC-USDT/15m/*.parquet
                    │
                    ▼
2. 지표 계산
   ├─ ATR (변동성)
   ├─ RSI, Stoch RSI (모멘텀)
   ├─ Wyckoff Phase (accumulation/markup/distribution/markdown)
   └─ HMM Posterior (상태 확률)
                    │
                    ▼
3. 시그널 생성 (Zone + Trigger)
   └─ Zone 진입 + Absorption Trigger 충족 시
      예: zone_low 도달 + Spring Reclaim → LONG 시그널
                    │
                    ▼
4. HMM Entry Gate (필터링) ← WPCN v1.1에서 99% 차단되던 병목
   ├─ Transition Cooldown: delta > 0.40이면 대기
   ├─ Long Permit: markup/accumulation/re_accumulation 상태만 허용
   └─ Short Permit: P(markdown) > 0.60 & trend_strength < -0.10
                    │
                    ▼
5. Pending Order 생성 (지정가)
   ├─ limit_price = 시그널 가격 (zone_low 또는 zone_high)
   ├─ sl_price = entry - ATR × 2.0
   └─ tp_price = entry + ATR × 3.0
                    │
                    ▼
6. 체결 체크 (매 봉)
   ├─ bar_low <= limit_price <= bar_high → 체결!
   └─ 4봉(1시간) 초과 → 주문 취소
                    │
                    ▼
7. 포지션 관리
   ├─ SL 히트: bar_low <= sl_price (long)
   ├─ TP 히트: bar_high >= tp_price (long)
   └─ Time-stop: 32봉(8시간) 초과 → 시장가 청산
                    │
                    ▼
8. 결과
   └─ trades_df, equity_df, statistics
```

### 병목 해결 히스토리

| 버전 | transition_delta | 결과 |
|------|------------------|------|
| v1.1 | 0.20 (너무 민감) | 99.97% 차단, trades: 3~18개 |
| v1.2 | 0.40 (완화) | trades: 72~143개 (통계 가능) |

### WPCN 파일 역할

| 파일 | 역할 |
|------|------|
| `run_dlite_wfo_v2.py` | 진입점 - WFO 실행 |
| `futures_backtest_15m.py` | 15분봉 백테스터 (PendingOrder, Position15m) |
| `hmm_entry_gate.py` | HMM 기반 진입 필터 |
| `policy_v1_2_relaxed_cooldown.py` | 현재 최적 설정 |
| `invariants.py` | PnL 계산 검증 |

### WPCN에 추가할 모듈

| 모듈 | 역할 | 수정/신규 |
|------|------|----------|
| `fractal_context.py` | 1W~1H z-score/balance/breakout_risk | **신규** |
| `zone_builder_15m.py` | confirmed pivot 기반 zone + score | **신규** |
| `micro_trigger_5m.py` | Spring/Failed Swing/Absorption | **신규** |
| `param_policy.py` | Fast/Medium/Slow + smoothing | **신규** |
| `run_dlite_wfo_v2.py` | 5m→15m 리샘플 + 상위TF 리샘플 | 수정 |
| `hmm_entry_gate.py` | Gate 앞단 적용 또는 EMA/min_dwell | 수정 |
| `futures_backtest_15m.py` | fill_check를 5m로 옵션 추가 | 수정 |

---

## 9.3) 최소 변경 로드맵 (바로 성능 검증 가능한 순서)

| 순서 | 작업 | 효과 |
|------|------|------|
| 1 | **Layer 1 추가** (프랙탈 컨텍스트) | 거래 후보를 context_score로 사전 필터링 |
| 2 | **15m 시그널** = zone_score 기반 | Fib 매칭 → zone_score (연속값) |
| 3 | **5m 트리거** = Spring 회수 1개만 강제 | 흡수 확인 없이 진입 금지 |
| 4 | **Gate 앞단으로** | 낭비 제거 + EMA/min_dwell 발작 방지 |
| 5 | **fill_check 5m** | 체결 정교화 (선택적) |

각 단계마다 **백테스트 비교**:
- Before: 기존 WPCN v1.2 결과
- After: 해당 단계 적용 후 결과
- 지표: Sharpe, Calmar, trade 수, win_rate, avg_pnl

---

## 10) File Structure (target)

```
trading_bot/
├── docs/
│   ├── SPEC_v2_integration.md    # 이 문서
│   └── session_backups/
├── src/
│   ├── context/                  # Layer 1: Context TFs
│   │   ├── fractal_range.py
│   │   └── regime_detector.py
│   ├── zone/                     # Layer 2: Zone Builder
│   │   ├── zigzag.py             # confirm_ts 포함
│   │   ├── fibonacci.py
│   │   ├── divergence.py         # score 기반
│   │   └── zone_builder.py
│   ├── gate/                     # Layer 3: HMM Gate
│   │   ├── hmm_entry_gate.py
│   │   └── param_policy.py
│   ├── trigger/                  # Layer 4: 5m Trigger
│   │   ├── spring_utad.py
│   │   ├── failed_swing.py
│   │   └── absorption.py
│   ├── execution/                # Layer 5: Execution
│   │   ├── backtest_15m.py
│   │   ├── cost_model.py
│   │   └── position.py
│   └── utils/
├── tests/
│   ├── test_causality.py
│   ├── test_gate_integrity.py
│   └── test_trigger_comparison.py
├── config/
│   └── param_policies/
├── data/
└── legacy/                       # 기존 코드 (참조용)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | 2026-01-13 | Initial integration spec (Legacy + WPCN) |
| v2.1 | 2026-01-13 | 9-Persona Review 반영 (ParamPolicy 강화, Fib TTL/Consensus, Zone Lock) |
| v2.2 | 2026-01-13 | WPCN 프로세스 흐름 상세화 (8단계 + 병목 히스토리) |
| v2.3 | 2026-01-13 | Multi-TF 올바른 사용법 6규칙, Gate 개선 옵션, 5m→15m 연결, 최소 변경 로드맵 |

---

## Author

- Project: trading_bot
- RAG Embedding: trading_bot