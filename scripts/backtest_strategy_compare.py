# backtest_strategy_compare.py
# 전략 A vs 전략 B 비교 백테스트
# - 전략 A: 15m 진입 + 5m 청산 (반대 다이버전스)
# - 전략 B: Fib 레벨 기반 (L1 근처 진입, 다음 L1 TP)

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
import talib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Tuple

# 프로젝트 루트
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.context.cycle_anchor import get_fractal_fib_levels, get_nearby_fib_levels, FibLevel
from src.context.cycle_dynamics import CycleDynamics
from src.context.dynamic_fib_anchor import (
    DynamicFibAnchorState,
    update_anchor_zigzag,
    update_anchor_rolling,
    update_anchor_conditional,
    get_dynamic_fib_levels,
    create_initial_state
)
from src.regime.wave_regime import WaveRegimeClassifier

# ProbabilityGate v2 import
try:
    from src.regime.prob_gate import ProbabilityGate, ProbGateConfig
    from src.regime.upstream_scores import make_score_hilbert_1h, align_score_1h_to_15m
    PROB_GATE_AVAILABLE = True
except ImportError as e:
    PROB_GATE_AVAILABLE = False
    print(f"[WARN] ProbabilityGate not available: {e}")

# FeatureStore import (P5)
try:
    from src.features.feature_store import FeatureStore
    FEATURE_STORE_AVAILABLE = True
except ImportError as e:
    FEATURE_STORE_AVAILABLE = False
    print(f"[WARN] FeatureStore not available: {e}")

# =============================================================================
# 설정
# =============================================================================

import json
from dataclasses import asdict, fields


def load_mode_config(mode: int, config_dir: Path = None) -> dict:
    """
    Load mode configuration from JSON file.

    Args:
        mode: RUN_MODE number (e.g., 16)
        config_dir: Path to configs directory (default: ROOT/configs)

    Returns:
        Dictionary of config overrides

    Raises:
        FileNotFoundError: If mode file doesn't exist
    """
    if config_dir is None:
        config_dir = ROOT / "configs"

    mode_path = config_dir / f"mode{mode}.json"
    if not mode_path.exists():
        raise FileNotFoundError(f"Config file not found: {mode_path}")

    with open(mode_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove metadata fields (start with _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def apply_config_overrides(config, overrides: dict, strict: bool = True):
    """
    Apply overrides to config with FAIL-CLOSED validation.

    Args:
        config: Config dataclass instance
        overrides: Dictionary of field_name -> value
        strict: If True, raise on unknown keys (FAIL-CLOSED)

    Returns:
        Modified config

    Raises:
        ValueError: If strict=True and unknown keys found
    """
    # Get valid field names from dataclass
    valid_keys = {f.name for f in fields(config)}

    # Check for unknown keys (FAIL-CLOSED)
    unknown = set(overrides.keys()) - valid_keys
    if unknown and strict:
        raise ValueError(
            f"Unknown config keys in override: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_keys)}"
        )

    # Apply overrides
    for key, value in overrides.items():
        if key in valid_keys:
            setattr(config, key, value)

    return config


def build_config_from_mode(mode: int, cli_overrides: dict = None) -> 'Config':
    """
    Build Config from mode JSON file with optional CLI overrides.

    Order of precedence (lowest to highest):
    1. Config() defaults
    2. configs/mode{N}.json
    3. cli_overrides

    Args:
        mode: RUN_MODE number
        cli_overrides: Optional dict of CLI overrides

    Returns:
        Fully configured Config instance with derived values computed
    """
    config = Config()

    # Load and apply mode config
    try:
        mode_overrides = load_mode_config(mode)
        config = apply_config_overrides(config, mode_overrides)
    except FileNotFoundError:
        # Fall back to legacy if-else (temporary)
        pass

    # Apply CLI overrides
    if cli_overrides:
        config = apply_config_overrides(config, cli_overrides)

    # Recompute derived values after all overrides
    config.recompute_derived()

    return config


# Duration-to-bars 변환 헬퍼
def _duration_to_bars(duration: str, timeframe: str) -> int:
    """Convert duration string to bar count for given timeframe."""
    try:
        from src.utils.timeframe import duration_to_bars
        return duration_to_bars(duration, timeframe)
    except ImportError:
        # Fallback: manual calculation
        tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        dur_minutes = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080}
        import re
        match = re.match(r'^(\d+(?:\.\d+)?)\s*(m|h|d|w)$', duration.strip().lower())
        if not match:
            raise ValueError(f"Invalid duration: {duration}")
        value = float(match.group(1))
        unit = match.group(2)
        total_min = value * dur_minutes[unit]
        tf_min = tf_minutes.get(timeframe.lower(), 5)
        return max(1, int(round(total_min / tf_min)))


def _floor_by_tf(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """
    Floor timestamp to the start of the previous completed bar for given timeframe.

    Examples:
        _floor_by_tf(2021-01-01 10:23, '1h')  -> 2021-01-01 09:00 (완료된 1H 봉)
        _floor_by_tf(2021-01-01 10:23, '15m') -> 2021-01-01 10:00 (완료된 15m 봉)
        _floor_by_tf(2021-01-01 10:23, '4h')  -> 2021-01-01 04:00 (완료된 4H 봉)
    """
    tf_lower = timeframe.lower().strip()
    if tf_lower == '1m':
        return ts.floor('1min') - pd.Timedelta(minutes=1)
    elif tf_lower == '5m':
        return ts.floor('5min') - pd.Timedelta(minutes=5)
    elif tf_lower == '15m':
        return ts.floor('15min') - pd.Timedelta(minutes=15)
    elif tf_lower == '30m':
        return ts.floor('30min') - pd.Timedelta(minutes=30)
    elif tf_lower == '1h':
        return ts.floor('1h') - pd.Timedelta(hours=1)
    elif tf_lower == '4h':
        return ts.floor('4h') - pd.Timedelta(hours=4)
    elif tf_lower == '1d':
        return ts.floor('1D') - pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")


@dataclass
class Config:
    # === Timeframe Configuration (NEW) ===
    # TF 변경 시 duration 기반 설정이 자동으로 bar 수로 변환됨
    trigger_tf: str = '5m'   # 트리거 TF (진입/청산 판단)
    anchor_tf: str = '15m'   # 앵커 TF (다이버전스/Zone 생성)
    context_tf: str = '1h'   # 컨텍스트 TF (레짐/Hilbert)

    # === Duration-based config (NEW) ===
    # duration이 설정되면 해당 bar 수를 자동 계산
    # None이면 legacy bar 수 사용 (하위호환)
    cooldown_duration: Optional[str] = '1h'           # → cooldown_bars (1h=12 bars @5m)
    early_exit_duration: Optional[str] = '2h'         # → early_exit_time_bars (2h=24 bars @5m)
    cycle_lookback_duration: Optional[str] = '2d'     # → cycle_lookback (2d=192 bars @15m)
    zone_depth_duration: Optional[str] = '8h'         # → zone_depth_lookback (8h=96 bars @5m)
    atr_duration: Optional[str] = None                # → atr_period (설정 시 atr_period 덮어씀)

    # Context TF (1H/4H) 용 duration
    trend_lookback_1h_duration: Optional[str] = '20h' # → trend_lookback_1h (20h=20 bars @1h)
    trend_lookback_4h_duration: Optional[str] = '40h' # → trend_lookback_4h (40h=10 bars @4h)
    atr_pct_lookback_duration: Optional[str] = '100h' # → atr_pct_lookback (100h=100 bars @1h)

    # 자산 & 레버리지
    initial_capital: float = 10000.0
    margin_pct: float = 0.01          # 1%
    leverage: float = 25.0

    # === PR6: 리스크 고정 사이징 ===
    # use_risk_fixed_sizing=True이면 손실 금액 고정 방식으로 전환
    # qty = risk_usd / sl_distance (SL 거리에 따라 포지션 크기 조절)
    use_risk_fixed_sizing: bool = False    # 리스크 고정 사이징 사용
    risk_usd_per_trade: float = 25.0       # 트레이드당 최대 손실 (USD)
    max_notional_usd: float = 5000.0       # 최대 포지션 명목가 (폭주 방지)
    min_sl_distance_atr_mult: float = 0.1  # 최소 SL 거리 (ATR 배수, qty 폭주 방지)

    # === PR6.2: TP 분할 청산 ===
    # TP 도달 시 50% → 30% → 20% 순차 청산
    use_tp_split: bool = False             # TP 분할 사용
    tp_split_ratios: tuple = (0.5, 0.3, 0.2)  # TP1:50%, TP2:30%, TP3:20%

    # RSI 설정
    rsi_period: int = 14
    stoch_rsi_period: int = 26

    # === PR4-R0: StochRSI 임계값 파라미터화 ===
    # 하드코딩 20/80 제거 → Config로 제어
    stoch_rsi_oversold: float = 20.0       # 과매도 진입 임계값
    stoch_rsi_overbought: float = 80.0     # 과매수 진입 임계값

    # === PR4-R0: TP 모드 옵션 ===
    # "fib": 기존 Fib 레벨 기반 TP (RR 보장 안됨)
    # "atr": ATR 배수 기반 TP (RR 안정화)
    # "fib_rr": Fib 레벨 중 RR >= min_rr_net 만족하는 가장 가까운 TP 선택 (없으면 진입 거부)
    tp_mode: str = "fib"                   # "fib" | "atr" | "fib_rr"
    tp_atr_mults: tuple = (1.0, 1.5, 2.0)  # mode="atr"일 때 TP1/TP2/TP3 배수

    # === PR-B: tp_mode="fib_rr" 설정 ===
    # Fib 후보 중 RR_net >= min_rr_net 만족하는 가장 가까운 TP 선택
    fib_tp_candidates: int = 4             # Fib TP 후보 개수 (L1 레벨 N개)
    rr_tp_ref: str = "tp2"                 # RR 기준점: "tp1" | "tp2" | "expected"

    # === PR4-R2: LONG-only 모드 ===
    # SHORT 비활성화 (ProbGate가 p_bull 0.75+로 SHORT 차단하므로 명시적 비활성화)
    enable_short: bool = True              # False = LONG-only Champion mode

    # === PR4-R3: Regime Trade Permission ===
    # 레짐별 LONG 허용/차단 (drift_regime 기준)
    # 2021-11 분석: UPTREND +$117.95, RANGE -$96.32, DOWNTREND -$38.54
    regime_long_uptrend: bool = True       # UPTREND에서 LONG 허용 (87.5% WR)
    regime_long_range: bool = False        # RANGE에서 LONG 차단 (2×2+PR-RANGE-1 확정: 독약)
    regime_long_downtrend: bool = False    # DOWNTREND에서 LONG 차단 (42.9% WR, 손실 구간)

    # === PR-B: 필터 중복 제거 실험 ===
    # Regime 필터와 Drift 필터를 독립적으로 토글하여 중복 영향 분석
    use_regime_filter: bool = True         # False: 레짐 기반 진입 차단 비활성화
    use_drift_filter: bool = True          # False: 드리프트 기반 threshold 조정 비활성화

    # === PR-RANGE-1: RANGE 전용 정책 ===
    # RANGE에서 진입을 '희귀'하게 만드는 하드 가드레일
    # 2×2 실험 결과: RANGE 허용 시 $/trade -$3.68 → -$7.85 (2배 악화)
    use_range_policy: bool = False          # True: RANGE 전용 정책 활성화

    # (A) 레인지 폭 필터: 폭이 충분히 넓을 때만 MR 의미 있음
    range_width_min_atr: float = 4.0        # 레인지 폭 >= X * ATR 필요 (좁은 레인지 = 수수료 머신)

    # (B) 극단 진입 강제: 레인지 하단 근처에서만 LONG
    range_entry_zone_pct: float = 0.25      # 레인지 하단 25% 구간에서만 진입 허용

    # (C) RANGE 전용 RR 강화
    min_rr_net_range: float = 2.5           # RANGE에서 더 높은 RR 요구 (UPTREND는 2.0)

    # (D) RANGE 전용 쿨다운
    range_cooldown_bars: int = 48           # RANGE에서 진입 후 N bars 재진입 금지 (15m 기준, 48=12시간)

    # (E) RANGE 전용 ProfitLock
    range_profit_lock_mfe_atr: float = 1.0  # RANGE에서 MFE >= X * ATR 도달 시 BE/부분익절

    # === PR4-R4a: UPTREND 전용 추세 지속형 진입 ===
    # 2021-Q1 진단: 다이버전스 진입이 상승장에서 로컬 고점 매수 (MAE > MFE 65.3%)
    # UPTREND에서는 다이버전스 대신 추세 지속형 진입 사용
    uptrend_entry_mode: str = "divergence"  # "divergence" (기존) | "trend_continuation" (추세 지속형)
    # 추세 지속형 진입 조건:
    # 1) EMA slope > 0 (상승 추세 확인)
    # 2) StochRSI가 oversold에서 상향 크로스 (풀백 후 재개)
    # 3) close > EMA (가격이 EMA 위)
    uptrend_ema_period: int = 20            # EMA 기간
    uptrend_require_ema_above: bool = True  # close > EMA 조건 필수
    uptrend_require_slope_positive: bool = True  # EMA slope > 0 조건 필수
    # Step A: 오버트레이딩 방지 (진입 후 쿨다운)
    uptrend_entry_cooldown_bars: int = 0    # 진입 후 N bars 재진입 금지 (15m 기준, 24=6시간)
    # Step B: 추세 재개 확인 (고점 돌파)
    uptrend_require_breakout: bool = False  # close > high[-1] 조건 필수

    # 비용
    fee_bps: float = 4.0              # 0.04%
    slippage_bps: float = 5.0         # 0.05%
    funding_rate: float = 0.0001      # 0.01% per 8h

    # Fib 설정
    fib_max_depth: int = 1            # L0+L1 (41개 레벨)
    fib_tolerance_pct: float = 0.003  # 0.3% (±$180 at $60K, 커버리지 69%)
    fib_atr_mult: float = 1.5         # ATR 기반 tolerance (전략 A): tolerance = ATR * mult / price

    # === PR-FIB: Fibonacci Configuration ===
    # Log Fib: low * (high/low)^ratio → %기반 균등 밴드
    # Linear Fib: low + (high-low)*ratio → 절대가격 기반
    fib_space: str = "linear"             # "linear" | "log" (log가 BTC에 적합)
    fib_anchor_low: float = 3120.0        # Fib 앵커 저점 (2019 12월 저점)
    fib_anchor_high: float = 143360.0     # Fib 앵커 고점 (69K * 2.0782)
    fib_tolerance_mode: str = "pct"       # "pct" | "atr_pct" | "gap_frac"
    fib_tolerance_atr_mult: float = 1.0   # atr_pct 모드: tolerance = ATR * mult / price
    atr_tf_for_fib: str = "15m"           # Fib tolerance 계산용 ATR 타임프레임

    # === PR-ENTRY-RR2: RR Limit Entry (수동매매 방식) ===
    # 다이버전스 신호 발생 시 즉시 진입하지 않고,
    # RR >= target이 되는 entry_limit 가격을 역산해서 그 가격에서만 체결
    # 사용자의 "RR 2:1 되는 자리까지 기다렸다가 산다" 방식 구현
    use_rr_limit_entry: bool = False          # RR limit entry 사용
    rr_limit_target: float = 2.0              # 목표 RR (2.0 = 2:1)
    rr_limit_ttl: str = "6h"                  # 주문 유효 시간 (6시간 = 15m 기준 24 bars)
    rr_limit_max_atr_dist: float = 2.0        # entry_limit이 현재가에서 N ATR 이상 멀면 skip
    rr_limit_price_ref: str = "close"         # entry 기준점: "close" | "zone_low" (추후)
    rr_limit_fill_on: str = "low"             # 체결 조건: "low" (bar low <= limit) | "close"

    # === PR-DYN-FIB v2: 1W Dynamic Fib (ZigZag + Log) ===
    # Layer 1 (Macro): 1W Log Fib (정적 $3K~$143K)
    # Layer 2 (Dynamic): 1W ZigZag Log Fib (동적, Macro 갭 채움)
    # NOTE: 15m dynamic fib는 range가 너무 좁아 효과 없음 → 1W만 사용
    use_dynamic_fib: bool = False             # 동적 Fib 사용
    dynamic_fib_tf: str = "1w"                # 1W 고정 (15m은 효과 없음)
    dynamic_fib_space: str = "log"            # Log 스페이스 (1W range에서 효과 있음)
    dynamic_fib_mode: str = "zigzag"          # ZigZag 고정 (pivot 기반)
    dynfib_reversal_atr_mult: float = 1.5     # zigzag: pivot 확정 기준 ATR 배수
    dynfib_ratios: tuple = (0.236, 0.382, 0.5, 0.618, 0.786)  # 생성할 Fib 비율
    dynfib_lookback_bars: int = 52            # rolling 모드용 lookback (1W = 52주 = 1년)
    dynfib_min_swing_atr_mult: float = 1.5    # conditional 모드용 최소 스윙 배수
    dynfib_use_as: str = "tp_candidate"       # "tp_candidate" | "entry_filter" | "both"
    dynfib_confluence_with_macro: bool = False  # Macro Fib와 confluence 요구 (미사용)
    dynfib_confluence_tol: float = 0.005      # confluence 허용 오차 (미사용)

    # 쿨다운 설정 (손절 후 재진입 제한)
    # Legacy bar count (duration 우선)
    cooldown_bars: int = 12           # 5m * 12 = 1시간 쿨다운

    # ATR 기반 SL
    sl_atr_mult: float = 1.5          # SL = entry ± 1.5*ATR (bc3e19a 설정)
    atr_period: int = 14              # ATR 계산 기간 (bc3e19a: 14, atr_duration 미설정 시 사용)

    # === PR-FIB-SL: Fib 구조 기반 SL ===
    # SL = prev_fib - buffer, buffer = min(ATR, fib_gap)
    # fib_gap = trigger_fib - prev_fib (인접 Fib 레벨 간격)
    use_fib_based_sl: bool = False    # Fib 기반 SL 사용 여부
    fib_sl_fallback_mult: float = 1.5 # Fib 레벨 부족 시 폴백 ATR 배수

    # === PR-A: ATR TF for Risk ===
    # SL/TP/MFE/BE 계산에 사용할 ATR의 타임프레임
    # '5m': 기존 동작 (회귀 테스트용)
    # '15m': 15분봉 기준 ATR 사용 (의도된 동작)
    atr_tf_for_risk: str = '15m'      # 리스크 계산용 ATR 타임프레임

    # 동적 SL (Hilbert + FFT 사이클 기반)
    use_dynamic_sl: bool = False      # 동적 SL 사용 여부 (비교용: False)
    dynamic_sl_base: float = 1.5      # 기본 배수
    dynamic_sl_adj: float = 0.5       # 위상 조절 범위 (1.5 ~ 2.0)
    cycle_lookback: int = 200         # 사이클 분석 윈도우 (15m bars) (duration 우선)

    # === PR4-R4b: 브레이크이븐 + 부분익절 ===
    # MFE >= be_mfe_atr * ATR 도달 시:
    #   1) 포지션 be_partial_pct% 청산
    #   2) 나머지 SL = entry + be_buffer_atr * ATR
    use_breakeven: bool = False           # 브레이크이븐 사용
    be_mfe_atr: float = 1.0               # 트리거: MFE >= X * ATR
    be_partial_pct: float = 0.5           # 부분청산 비율 (0.5 = 50%)
    be_buffer_atr: float = 0.05           # SL 버퍼 (수수료/슬리피지 커버)

    # === PR4-R5: Entry RR Gate ===
    # 진입 전 RR(수수료 포함) 계산하여 기준 미달 시 진입 취소
    # RR = (TP1 - entry - fees) / (entry - SL + fees)
    use_rr_gate: bool = False             # RR Gate 사용
    min_rr_net: float = 1.2               # 최소 RR (1.2 = 손실 $1당 수익 $1.2 기대)
    rr_gate_use_tp1: bool = True          # TP1 기준 RR 계산 (False면 TP2)

    # === PR4-R6: Liquidation Mode (고레버리지 + 무손절 실험) ===
    # SL 대신 리퀴데이션 가격을 손절로 사용, RR≥2 진입 필터
    use_liq_as_stop: bool = False         # True: SL=liq_price (일반 SL 없음)
    leverage_mode: str = 'fixed'          # 'fixed' or 'dynamic'
    leverage_fixed: float = 10.0          # 고정 레버리지 (leverage_mode='fixed')
    leverage_max: float = 50.0            # 최대 레버리지 (leverage_mode='dynamic')
    leverage_min: float = 5.0             # 최소 레버리지 (leverage_mode='dynamic')
    liq_mmr: float = 0.004                # Maintenance Margin Rate (0.4% 기본)

    # Risk Guardrails (안전장치)
    use_isolated_margin: bool = True      # Isolated margin 가정
    margin_per_trade_pct: float = 0.01    # 트레이드당 증거금 = 계좌의 1%
    max_liq_events_period: int = 3        # 기간당 강제청산 N회 초과 시 중단
    min_liq_distance_atr: float = 2.0     # liq까지 최소 N*ATR 이상 필요

    # 추세 필터 (계단식 적용)
    use_trend_filter_1h: bool = False  # 1H 역행 금지
    use_trend_filter_4h: bool = False  # 4H 역행 금지
    use_atr_vol_filter: bool = False   # ATR 고변동성 사이즈 축소
    atr_vol_threshold: float = 80.0    # ATR percentile 임계값
    atr_vol_size_mult: float = 0.5     # 고변동성 시 사이즈 배수

    # Context TF lookback (duration 우선)
    trend_lookback_1h: int = 20        # 1H trend 계산 lookback
    trend_lookback_4h: int = 10        # 4H trend 계산 lookback
    atr_pct_lookback: int = 100        # ATR percentile lookback (1H)

    # Zone Depth 필터 (검증 완료: r=0.215, p≈0)
    use_zone_depth_filter: bool = False  # zone_depth >= 0.6 필터
    zone_depth_min: float = 0.6          # 최소 depth (이 미만 진입 금지)
    zone_depth_lookback: int = 100       # swing high/low 계산 lookback (duration 우선)

    # Hilbert 레짐 필터 (1H, causal, IC=+0.027) - LEGACY
    use_hilbert_filter: bool = False     # Hilbert 레짐 필터 사용

    # WaveRegime (Hilbert) params (duration-based on context_tf)
    wave_regime_detrend_duration: str = '2d'  # → detrend_period (2d=48 bars @1h)
    wave_regime_hilbert_window: int = 32
    hilbert_block_long_on_bear: bool = True   # Long: BEAR 레짐에서 차단
    hilbert_block_short_on_bull: bool = False # Short: BULL 레짐에서 차단 (느슨)

    # ProbabilityGate v2 (1H Hilbert score, IC=+0.062, OOS 검증)
    # RUN_MODE=7: Hilbert 필터 대체 (교체, AND 아님)
    use_prob_gate: bool = False          # ProbabilityGate v2 사용
    prob_gate_temp_mode: str = 'vol'     # 온도 모드 ('vol' or 'instability')
    prob_gate_p_shrink: float = 0.6      # 캘리브레이션 (0.6 = champion)
    prob_gate_thr_long: float = 0.55     # Long 허용 임계값 (p_bull > thr)
    prob_gate_thr_short: float = 0.60    # Short 허용 임계값 (p_bull < 1-thr) - 강화됨

    # ProbGate rolling windows (duration-based; converted on trigger_tf)
    prob_gate_atr_duration: str = '1d'   # → n_atr (1d=288 bars @5m)
    prob_gate_vol_duration: str = '2d'   # → vol_window (2d=576 bars @5m)

    # Dynamic Threshold (RUN_MODE=9)
    # 레짐/불확실성 기반 동적 threshold 조절
    prob_gate_use_dynamic_thr: bool = False   # 동적 threshold 사용
    prob_gate_dyn_thr_short_floor: float = 0.60  # SHORT thr 하한 (0.55로 안 내려가게)

    # PR4.2: SHORT 확신도 필터 (HOT 전용으로 축소)
    # conf = abs(p_bull - 0.5) * 2, 0~1 범위
    prob_gate_use_conf_filter: bool = False    # SHORT conf 필터 사용
    prob_gate_conf_min_short: float = 0.0      # 일반: OFF (conf_hot_add만 사용)
    prob_gate_T_hot: float = 1.8               # T >= 1.8이면 HOT
    prob_gate_conf_hot_add: float = 0.30       # HOT에서만 conf >= 0.30 필요

    # PR4.3: SHORT 타이밍 확인 (ret_n + EMA)
    # 목표: "빠른 SL" 문제 해결 - SHORT 진입 시 모멘텀+위치 확인
    prob_gate_use_short_timing: bool = False   # SHORT 타이밍 필터 사용
    prob_gate_short_ret_bars: int = 3          # 모멘텀 계산용 n bars
    prob_gate_short_ret_min: float = -0.0005   # ret_n < -0.05% 필요
    prob_gate_short_ema_period: int = 20       # close < EMA(20) 필요

    # PR4.4: 드리프트 기반 동적 thr_short (1H EMA200 레짐)
    # 핵심: HOT/COLD(변동성)가 아니라 UPTREND/DOWNTREND(드리프트)로 조절
    prob_gate_use_drift_thr: bool = False      # 드리프트 기반 동적 threshold
    prob_gate_drift_ema_period: int = 200      # 1H EMA 기간 (200 = 약 8일)
    # PR4.4.1: 히스테리시스 (whipsaw 방지)
    prob_gate_drift_enter_pct: float = 0.012   # 진입 ±1.2%
    prob_gate_drift_exit_pct: float = 0.008    # 탈출 ±0.8%
    prob_gate_drift_min_bars: int = 3          # 최소 3 bars 연속
    # PR4.4.1: EMA slope 조건 (방향성 확인)
    prob_gate_drift_use_slope: bool = True     # slope 조건 사용
    prob_gate_drift_slope_duration: str = '24h'  # → slope_bars (24h=24 bars @1h)
    prob_gate_drift_slope_bars: int = 24       # 24시간 기울기 (context_tf 기준, duration 우선)
    # Threshold 설정 (PR4.4.1: DOWN도 엄격하게 - 0.62→0.65)
    prob_gate_thr_short_uptrend: float = 0.70  # 상승장: SHORT 더 엄격
    prob_gate_thr_short_downtrend: float = 0.65  # 하락장: RANGE와 동일 (관대함 제거)
    prob_gate_thr_short_range: float = 0.65    # 횡보: 기본값

    # === Early Exit Rules (PR3.5) - SL 회피용 조기 청산 ===
    use_early_exit: bool = False          # Early Exit 전체 활성화
    # 1) TimeStop: N bars 내 MFE < threshold*ATR이면 청산
    early_exit_time_bars: int = 24        # 5m * 24 = 2시간 (duration 우선)
    early_exit_mfe_mult: float = 0.3      # MFE < 0.3*ATR이면 조기 청산
    # 2) GateFlip Exit: ProbGate 방향 반전 시 청산
    use_gate_flip_exit: bool = False      # Gate 방향 반전 시 청산
    # 3) Opposite Div Early Exit: 손실 상태에서 반대 다이버전스 시 SL 전 청산
    use_opposite_div_early: bool = False  # 손실 포지션 반대 Div 조기 청산
    # 4) 5m Div Exit: 5분봉 반대 다이버전스 시 전체 청산 (TP 도달 전)
    use_5m_div_exit: bool = True          # True=기존 동작, False=비활성화
    # 5) 15m Div Exit: 15분봉 반대 다이버전스 시 전체 청산
    use_15m_div_exit: bool = False        # 15m 다이버전스 청산

    # PR3.6: HOT 구간 Early Exit 강화 (불확실성 높을 때 더 공격적으로 청산)
    use_hot_early_exit: bool = False      # HOT 구간 강화 Early Exit
    early_exit_time_bars_hot: int = 18    # HOT: 5m * 18 = 1.5시간 (더 빠름)
    early_exit_mfe_mult_hot: float = 0.2  # HOT: MFE < 0.2*ATR (더 공격적)
    stale_loss_mult_hot: float = 0.2      # HOT: StaleLoss -0.2*ATR (더 빠른 손절)

    # === Trailing Stop (PR3.7) - 수익 보존용 트레일링 스탑 ===
    use_trailing_stop: bool = False       # Trailing Stop 활성화
    trailing_activation_atr: float = 1.0  # MFE가 N ATR 이상일 때 트레일링 시작
    trailing_distance_atr: float = 0.5    # 고점에서 N ATR 뒤에서 추적
    use_5m_div_trailing: bool = False     # 5m 다이버전스 발생 시 트레일링 즉시 활성화
    use_5m_div_partial_trailing: bool = False  # 5m 다이버전스 시 50% 청산 + 나머지 트레일링

    # 레짐 기반 Hidden Divergence 전략 (새로운 접근법)
    use_regime_hidden_strategy: bool = False  # 레짐 방향 + Hidden Divergence
    # BULL → Hidden Bullish만 Long
    # BEAR → Hidden Bearish만 Short
    # RANGE → 진입 안함

    # Hidden Divergence 추가 (Regular + Hidden 동시 사용)
    use_hidden_div_long: bool = False  # Regular Div + Hidden Bullish Div 동시 사용

    # === FeatureStore (P5) ===
    # Centralized feature computation with FAIL-CLOSED warmup validation
    use_feature_store: bool = False      # FeatureStore로 prob_gate 계산

    # === Warmup Configuration (auto-calculated) ===
    # Duration 기반 최소 warmup + 필요 lookback들의 max로 자동 산출
    warmup_duration: str = '6h'          # 최소 warmup 시간 (6h=72 bars @5m)
    warmup_margin_duration: str = '1h'   # 안전 마진 (1h=12 bars @5m)
    warmup_bars: int = 72                # trigger_tf 기준 (auto-calculated)
    warmup_bars_anchor: int = 24         # anchor_tf 기준 (auto-calculated)

    def __post_init__(self):
        """Convert duration strings to bar counts."""
        # cooldown: trigger_tf 기준
        if self.cooldown_duration:
            self.cooldown_bars = _duration_to_bars(self.cooldown_duration, self.trigger_tf)

        # early_exit: trigger_tf 기준
        if self.early_exit_duration:
            self.early_exit_time_bars = _duration_to_bars(self.early_exit_duration, self.trigger_tf)

        # cycle_lookback: anchor_tf 기준
        if self.cycle_lookback_duration:
            self.cycle_lookback = _duration_to_bars(self.cycle_lookback_duration, self.anchor_tf)

        # zone_depth: trigger_tf 기준
        if self.zone_depth_duration:
            self.zone_depth_lookback = _duration_to_bars(self.zone_depth_duration, self.trigger_tf)

        # atr_period: trigger_tf 기준
        if self.atr_duration:
            self.atr_period = _duration_to_bars(self.atr_duration, self.trigger_tf)

        # Context TF lookbacks
        if self.trend_lookback_1h_duration:
            self.trend_lookback_1h = _duration_to_bars(self.trend_lookback_1h_duration, '1h')
        if self.trend_lookback_4h_duration:
            self.trend_lookback_4h = _duration_to_bars(self.trend_lookback_4h_duration, '4h')
        if self.atr_pct_lookback_duration:
            self.atr_pct_lookback = _duration_to_bars(self.atr_pct_lookback_duration, '1h')

        # PR4.4.1: Drift slope window in bars (context_tf 기준)
        try:
            self.prob_gate_drift_slope_bars = _duration_to_bars(
                self.prob_gate_drift_slope_duration, self.context_tf
            )
        except Exception:
            # keep existing integer if duration parse fails
            pass

        # === Warmup auto-calculation ===
        # trigger_tf 기준 warmup: trigger_tf 전용 lookback들의 max
        duration_warmup = _duration_to_bars(self.warmup_duration, self.trigger_tf)
        margin_bars = _duration_to_bars(self.warmup_margin_duration, self.trigger_tf)

        # trigger_tf 전용 lookback들 (이 TF에서 직접 계산되는 지표들)
        required_lookbacks_trigger = [
            self.atr_period,                    # ATR 계산
            self.zone_depth_lookback,           # Zone depth
            self.cooldown_bars,                 # Cooldown
            self.early_exit_time_bars,          # Early exit
        ]
        # NOTE: trend_lookback_1h, trend_lookback_4h는 context_tf에서 계산되므로
        # trigger_tf warmup에 포함하지 않음 (해당 TF 데이터에서만 필요)

        self.warmup_bars = max(
            duration_warmup,
            max(required_lookbacks_trigger) if required_lookbacks_trigger else 0
        ) + margin_bars

        # anchor_tf 기준 warmup
        anchor_warmup = _duration_to_bars(self.warmup_duration, self.anchor_tf)
        anchor_margin = _duration_to_bars(self.warmup_margin_duration, self.anchor_tf)

        # anchor_tf 전용 lookbacks (다이버전스, StochRSI 등)
        required_lookbacks_anchor = [
            self.cycle_lookback,                # Cycle 분석
            self.rsi_period * 3,                # RSI + 여유 (다이버전스용)
            self.stoch_rsi_period * 2,          # StochRSI + 여유
        ]

        self.warmup_bars_anchor = max(
            anchor_warmup,
            max(required_lookbacks_anchor) if required_lookbacks_anchor else 0
        ) + anchor_margin

    def recompute_derived(self):
        """
        Recompute all derived values after config overrides.

        Call this after applying JSON/CLI overrides to ensure
        duration-based values are recalculated.
        """
        self.__post_init__()

    @property
    def margin_per_trade(self) -> float:
        return self.initial_capital * self.margin_pct

    @property
    def position_size(self) -> float:
        return self.margin_per_trade * self.leverage

    def entry_cost_pct(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10000

    def exit_cost_pct(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10000

    def funding_cost(self, hours: float) -> float:
        return self.funding_rate * (hours / 8)

    def calculate_qty(self, entry_price: float, sl_price: float, atr: float = 0.0) -> float:
        """
        PR6: 리스크 고정 사이징으로 포지션 수량 계산.

        Args:
            entry_price: 진입가
            sl_price: 손절가
            atr: 현재 ATR (min_sl_distance 계산용)

        Returns:
            qty: BTC 수량 (소수점)

        Notes:
            - use_risk_fixed_sizing=False면 기존 방식 (position_size / entry_price)
            - use_risk_fixed_sizing=True면 risk_usd / sl_distance
            - max_notional_usd, min_sl_distance로 폭주 방지
        """
        if not self.use_risk_fixed_sizing:
            # Legacy: 고정 포지션 크기
            return self.position_size / entry_price

        # PR6: 리스크 고정 사이징
        sl_distance = abs(entry_price - sl_price)

        # 최소 SL 거리 적용 (qty 폭주 방지)
        min_sl_distance = atr * self.min_sl_distance_atr_mult if atr > 0 else 0.0
        if min_sl_distance > 0 and sl_distance < min_sl_distance:
            sl_distance = min_sl_distance

        # SL 거리가 0이면 안전하게 거부
        if sl_distance <= 0:
            return 0.0

        # qty = risk_usd / sl_distance
        qty_raw = self.risk_usd_per_trade / sl_distance

        # max_notional 제한
        max_qty = self.max_notional_usd / entry_price
        qty = min(qty_raw, max_qty)

        # 레버리지/마진 제한 (기존 position_size를 상한으로)
        legacy_max_qty = self.position_size / entry_price
        qty = min(qty, legacy_max_qty)

        return qty

    def calculate_qty_info(self, entry_price: float, sl_price: float, atr: float = 0.0) -> dict:
        """
        PR6.3: 리스크 고정 사이징 + 상세 로깅 정보 반환.

        Returns:
            dict with keys:
                qty: 최종 수량
                sl_distance_raw: 원래 SL 거리
                sl_distance_final: clamp 적용 후 SL 거리
                clamped: bool - min_sl_distance 적용 여부
                sl_distance_atr: sl_distance / atr 비율
                notional: qty * entry_price
                risk_usd: 설정된 risk_usd_per_trade
                cap_reason: 'none' | 'max_notional' | 'legacy'
        """
        if not self.use_risk_fixed_sizing:
            qty = self.position_size / entry_price
            return {
                'qty': qty,
                'sl_distance_raw': abs(entry_price - sl_price),
                'sl_distance_final': abs(entry_price - sl_price),
                'clamped': False,
                'sl_distance_atr': abs(entry_price - sl_price) / atr if atr > 0 else 0.0,
                'notional': qty * entry_price,
                'risk_usd': 0.0,  # Legacy doesn't use risk_usd
                'cap_reason': 'legacy',
            }

        # PR6: 리스크 고정 사이징
        sl_distance_raw = abs(entry_price - sl_price)

        # 최소 SL 거리 적용 (qty 폭주 방지)
        min_sl_distance = atr * self.min_sl_distance_atr_mult if atr > 0 else 0.0
        clamped = min_sl_distance > 0 and sl_distance_raw < min_sl_distance
        sl_distance_final = max(sl_distance_raw, min_sl_distance) if clamped else sl_distance_raw

        # SL 거리가 0이면 안전하게 거부
        if sl_distance_final <= 0:
            return {
                'qty': 0.0,
                'sl_distance_raw': sl_distance_raw,
                'sl_distance_final': 0.0,
                'clamped': clamped,
                'sl_distance_atr': 0.0,
                'notional': 0.0,
                'risk_usd': self.risk_usd_per_trade,
                'cap_reason': 'zero_sl',
            }

        # qty = risk_usd / sl_distance
        qty_raw = self.risk_usd_per_trade / sl_distance_final

        # max_notional 제한
        max_qty = self.max_notional_usd / entry_price
        cap_reason = 'none'
        if qty_raw > max_qty:
            qty = max_qty
            cap_reason = 'max_notional'
        else:
            qty = qty_raw

        # NOTE: PR6에서는 max_notional_usd가 유일한 상한
        # legacy position_size cap은 use_risk_fixed_sizing=False일 때만 의미 있음
        # 따라서 여기서는 추가 cap 없음

        return {
            'qty': qty,
            'sl_distance_raw': sl_distance_raw,
            'sl_distance_final': sl_distance_final,
            'clamped': clamped,
            'sl_distance_atr': sl_distance_raw / atr if atr > 0 else 0.0,
            'notional': qty * entry_price,
            'risk_usd': self.risk_usd_per_trade,
            'cap_reason': cap_reason,
        }

    def calculate_pnl_usd(self, side: str, entry_price: float, exit_price: float, qty: float, hours: float = 0.0) -> tuple:
        """
        PR6: USD 기반 PnL 계산.

        Args:
            side: 'long' or 'short'
            entry_price: 진입가
            exit_price: 청산가
            qty: 포지션 수량
            hours: 보유 시간 (펀딩비 계산용)

        Returns:
            (pnl_usd, pnl_pct): USD 손익, % 손익
        """
        # Raw PnL
        if side == 'long':
            raw_pnl = (exit_price - entry_price) * qty
        else:
            raw_pnl = (entry_price - exit_price) * qty

        # 비용 (USD 기반)
        notional = qty * entry_price
        entry_cost = notional * self.entry_cost_pct()
        exit_cost = qty * exit_price * self.exit_cost_pct()
        funding = notional * self.funding_cost(hours) if hours > 0 else 0.0

        pnl_usd = raw_pnl - entry_cost - exit_cost - funding

        # pnl_pct (마진 대비)
        margin_used = notional / self.leverage
        pnl_pct = pnl_usd / margin_used if margin_used > 0 else 0.0

        return pnl_usd, pnl_pct


# Backward compatibility alias
BacktestConfig = Config


# =============================================================================
# RSI 계산 (talib 사용)
# =============================================================================
def calc_rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI 계산 (talib 사용)"""
    close = np.asarray(close, dtype=np.float64)
    return talib.RSI(close, timeperiod=period)

def calc_stoch_rsi(close: np.ndarray, period: int = 14, k_period: int = 3, d_period: int = 3) -> np.ndarray:
    """StochRSI %D 계산 (talib 사용)"""
    close = np.asarray(close, dtype=np.float64)
    fastk, fastd = talib.STOCHRSI(
        close,
        timeperiod=period,
        fastk_period=k_period,
        fastd_period=d_period,
        fastd_matype=0  # SMA
    )
    return fastd

def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 21) -> np.ndarray:
    """ATR 계산 (talib 사용)"""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ATR(high, low, close, timeperiod=period)

# =============================================================================
# PR4-R6: Liquidation Price 계산
# =============================================================================
def calc_liq_price(
    entry_price: float,
    side: str,
    leverage: float,
    mmr: float = 0.004
) -> float:
    """
    리퀴데이션 가격 계산 (근사치)

    Args:
        entry_price: 진입 가격
        side: 'long' or 'short'
        leverage: 레버리지 배수
        mmr: Maintenance Margin Rate (기본 0.4%)

    Returns:
        liq_price: 강제청산 가격

    근사 공식:
        - Long: liq = entry * (1 - 1/leverage + mmr)
        - Short: liq = entry * (1 + 1/leverage - mmr)
    """
    if leverage <= 0:
        leverage = 1.0  # 안전장치

    if side == 'long':
        # Long: 가격이 떨어지면 청산
        liq_price = entry_price * (1 - (1 / leverage) + mmr)
    else:
        # Short: 가격이 오르면 청산
        liq_price = entry_price * (1 + (1 / leverage) - mmr)

    return max(0.0, liq_price)  # 음수 방지


def calc_liq_distance_atr(
    entry_price: float,
    liq_price: float,
    atr: float
) -> float:
    """
    진입가에서 청산가까지의 거리를 ATR 단위로 계산

    Returns:
        liq_distance_atr: abs(entry - liq) / ATR
    """
    if atr <= 0:
        return float('inf')  # ATR 0이면 무한 거리
    return abs(entry_price - liq_price) / atr


def select_leverage_dynamic(
    entry_price: float,
    tp_price: float,
    side: str,
    atr: float,
    config: 'Config'
) -> Tuple[float, float, float]:
    """
    Dynamic leverage 선택: RR과 liq_distance 조건을 만족하는 최대 레버리지

    Returns:
        (leverage, liq_price, rr_net) or (0, 0, 0) if no valid leverage
    """
    total_cost_pct = (config.fee_bps + config.slippage_bps) * 2 / 10000
    fee_cost = entry_price * total_cost_pct

    # leverage_max -> leverage_min 순으로 탐색
    for lev in range(int(config.leverage_max), int(config.leverage_min) - 1, -1):
        liq_price = calc_liq_price(entry_price, side, lev, config.liq_mmr)
        liq_dist_atr = calc_liq_distance_atr(entry_price, liq_price, atr)

        # Risk 계산
        if side == 'long':
            risk = entry_price - liq_price
            reward = tp_price - entry_price
        else:
            risk = liq_price - entry_price
            reward = entry_price - tp_price

        # RR_net 계산
        profit_net = reward - fee_cost
        loss_net = risk + fee_cost
        rr_net = profit_net / loss_net if loss_net > 0 else 0

        # 조건 체크
        if liq_dist_atr >= config.min_liq_distance_atr and rr_net >= config.min_rr_net:
            return (float(lev), liq_price, rr_net)

    return (0.0, 0.0, 0.0)  # 조건 만족하는 레버리지 없음


# =============================================================================
# Zone Depth 계산 (검증 완료: r=0.215, p≈0)
# =============================================================================
def calc_zone_depth(
    close_arr: np.ndarray,
    current_idx: int,
    side: str,
    lookback: int = 100
) -> float:
    """
    Zone Depth 계산: Fib zone에 얼마나 깊이 들어갔는지

    Bullish: 저점(swing_low)에 가까울수록 높은 depth
    Bearish: 고점(swing_high)에 가까울수록 높은 depth

    Returns:
        0-1 범위 (1 = 완전히 극단에 도달)
    """
    if current_idx < lookback:
        return 0.5  # 데이터 부족시 중립

    start = current_idx - lookback
    window = close_arr[start:current_idx]

    swing_high = np.max(window)
    swing_low = np.min(window)
    current_price = close_arr[current_idx]

    price_range = swing_high - swing_low
    if price_range <= 0 or swing_high <= swing_low * 1.01:
        return 0.5  # 범위 부족시 중립

    if side == 'long':
        # Bullish: 저점에 가까울수록 depth 높음
        depth = 1.0 - (current_price - swing_low) / price_range
    else:
        # Bearish: 고점에 가까울수록 depth 높음
        depth = (current_price - swing_low) / price_range

    return max(0.0, min(1.0, depth))


def calc_zone_depth_size_mult(zone_depth: float, min_depth: float = 0.6) -> float:
    """
    Zone depth 기반 사이즈 배수 계산 (사이징 전용, 필터 아님)

    핵심 변경: 필터(진입 금지) → 사이징(작게 진입)
    - depth < min_depth: 0.25 (작게 진입, 완전 스킵 아님)
    - depth >= min_depth: 0.25 ~ 1.0 선형 스케일

    근거:
    - Total PnL 개선(-393 → -260)은 리스크 노출량 감소 효과
    - EV/Trade 악화(-4.06 → -4.19)는 필터가 엣지를 못 만듦
    - 따라서 "스킵"이 아니라 "노출량을 깊이에 비례해서 배분"

    Returns:
        0.25-1.0 사이즈 배수 (항상 진입, 크기만 조절)
    """
    if zone_depth < min_depth:
        return 0.25  # 얕은 구간: 작게 진입 (완전 스킵 아님)
    # 선형 스케일: 0.6 → 0.25, 1.0 → 1.0
    return 0.25 + 0.75 * ((zone_depth - min_depth) / (1.0 - min_depth))


# =============================================================================
# 추세 계산 함수
# =============================================================================
def calculate_trend(highs: np.ndarray, lows: np.ndarray, lookback: int = 20) -> str:
    """
    최근 N개 바의 추세 판단 (HH/HL vs LH/LL)
    - HH + HL: UPTREND
    - LH + LL: DOWNTREND
    - 혼합: SIDEWAYS
    """
    if len(highs) < lookback:
        return "UNKNOWN"

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]

    mid = lookback // 2

    first_half_high = recent_highs[:mid].max()
    second_half_high = recent_highs[mid:].max()
    first_half_low = recent_lows[:mid].min()
    second_half_low = recent_lows[mid:].min()

    higher_high = second_half_high > first_half_high
    higher_low = second_half_low > first_half_low
    lower_high = second_half_high < first_half_high
    lower_low = second_half_low < first_half_low

    if higher_high and higher_low:
        return "UPTREND"
    elif lower_high and lower_low:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"


def precompute_trend_column(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    데이터프레임에 추세 컬럼 미리 계산 (O(n) 한 번만)
    calculate_trend()와 동일한 결과를 내기 위해 현재 바 포함
    """
    n = len(df)
    trends = ['UNKNOWN'] * n

    highs = df['high'].values
    lows = df['low'].values

    for i in range(lookback - 1, n):
        # 현재 바(i)를 포함한 마지막 lookback개 바 사용 (calculate_trend와 동일)
        recent_highs = highs[i-lookback+1:i+1]
        recent_lows = lows[i-lookback+1:i+1]

        mid = lookback // 2
        first_half_high = recent_highs[:mid].max()
        second_half_high = recent_highs[mid:].max()
        first_half_low = recent_lows[:mid].min()
        second_half_low = recent_lows[mid:].min()

        higher_high = second_half_high > first_half_high
        higher_low = second_half_low > first_half_low
        lower_high = second_half_high < first_half_high
        lower_low = second_half_low < first_half_low

        if higher_high and higher_low:
            trends[i] = "UPTREND"
        elif lower_high and lower_low:
            trends[i] = "DOWNTREND"
        else:
            trends[i] = "SIDEWAYS"

    return pd.Series(trends, index=df.index)


def precompute_atr_percentile_column(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """
    ATR percentile 미리 계산
    calculate_atr_percentile()와 동일한 결과를 내기 위해 현재 바 포함
    """
    n = len(df)
    percentiles = [50.0] * n

    if 'atr' not in df.columns:
        return pd.Series(percentiles, index=df.index)

    atr_values = df['atr'].values

    for i in range(lookback - 1, n):
        # 현재 바(i)를 포함한 마지막 lookback개 바 사용 (calculate_atr_percentile와 동일)
        recent = atr_values[i-lookback+1:i+1]
        valid = recent[np.isfinite(recent)]
        if len(valid) < 10:
            continue
        current = atr_values[i]
        if np.isfinite(current):
            percentiles[i] = (valid < current).sum() / len(valid) * 100

    return pd.Series(percentiles, index=df.index)


def precompute_prob_gate(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    config: Config
) -> Optional[pd.DataFrame]:
    """
    ProbabilityGate v2 결과 pre-compute (5m 인덱스로 정렬)

    1H Hilbert score → 5m 정렬 → ProbabilityGate → action_code

    Args:
        df_1h: 1H 데이터 (timestamp index 또는 컬럼)
        df_5m: 5m 데이터 (timestamp index 또는 컬럼)
        config: Config with prob_gate settings

    Returns:
        DataFrame with [T, score_norm, p_bull, action_code, action_str, valid]
        indexed by 5m timestamp
    """
    if not PROB_GATE_AVAILABLE:
        return None

    try:
        # 1. df_1h 준비 (timestamp 컬럼 필요)
        df_1h_for_score = df_1h.copy()
        if isinstance(df_1h_for_score.index, pd.DatetimeIndex):
            if 'timestamp' not in df_1h_for_score.columns:
                df_1h_for_score = df_1h_for_score.reset_index()
                if 'index' in df_1h_for_score.columns:
                    df_1h_for_score = df_1h_for_score.rename(columns={'index': 'timestamp'})

        # 2. df_5m 준비 (timestamp index 필요)
        df_5m_indexed = df_5m.copy()
        if 'timestamp' in df_5m_indexed.columns and not isinstance(df_5m_indexed.index, pd.DatetimeIndex):
            df_5m_indexed['timestamp'] = pd.to_datetime(df_5m_indexed['timestamp'])
            df_5m_indexed = df_5m_indexed.set_index('timestamp').sort_index()

        # 3. 1H Hilbert score 계산
        score_1h = make_score_hilbert_1h(df_1h_for_score)

        # 4. 5m에 정렬 (forward-fill + shift(1) for 'open' semantics)
        score_5m = align_score_1h_to_15m(score_1h, df_5m_indexed, timestamp_semantics='open')

        # 5. ProbabilityGate 설정 (duration 기반 - TF 변경 시 자동 조정)
        # Dynamic threshold 설정 (RUN_MODE=9)
        from src.regime.prob_gate import DynamicThresholdConfig
        dyn_thr_cfg = None
        if config.prob_gate_use_dynamic_thr:
            dyn_thr_cfg = DynamicThresholdConfig(
                base_long=0.58,
                base_short=0.60,  # SHORT는 기본적으로 더 엄격
                conf_favor_delta=0.03,
                conf_against_delta=0.04,
                uncertainty_delta=0.04,
                thr_min=0.55,
                thr_max=0.70,
                ema_span=3,
            )

        gate_config = ProbGateConfig(
            temp_mode=config.prob_gate_temp_mode,
            p_shrink=config.prob_gate_p_shrink,
            thr_long=config.prob_gate_thr_long,
            thr_short=config.prob_gate_thr_short,
            # Duration 기반 설정 (5m TF)
            timeframe='5m',
            atr_duration='1d',    # -> 288 bars at 5m
            vol_duration='2d',    # -> 576 bars at 5m
            # Dynamic threshold
            use_dynamic_threshold=config.prob_gate_use_dynamic_thr,
            dynamic_thr_cfg=dyn_thr_cfg,
        )

        gate = ProbabilityGate(gate_config)

        # 6. Gate 계산
        result = gate.compute(
            score_5m.values,
            df_5m_indexed['close'].values,
            df_5m_indexed['high'].values,
            df_5m_indexed['low'].values
        )
        result.index = df_5m_indexed.index

        # 7. SHORT threshold floor 적용 (동적 thr 시에도 0.60 이하로 안 내려가게)
        if config.prob_gate_use_dynamic_thr and 'thr_short' in result.columns:
            floor = config.prob_gate_dyn_thr_short_floor
            result['thr_short'] = result['thr_short'].clip(lower=floor)
            # action 재계산 (thr_short가 바뀌었으므로)
            for i in range(len(result)):
                p = result['p_bull'].iloc[i]
                ts = result['thr_short'].iloc[i]
                if result['action_code'].iloc[i] == -1:  # SHORT
                    if p >= (1 - ts):  # 이제 SHORT 조건 불충족
                        result.loc[result.index[i], 'action_code'] = 0
                        result.loc[result.index[i], 'action_str'] = 'FLAT'

        # 8. 동적 threshold 통계 출력
        if config.prob_gate_use_dynamic_thr:
            thr_long = result['thr_long'].dropna()
            thr_short = result['thr_short'].dropna()
            print(f"\n[Dynamic Threshold Stats]")
            print(f"  thr_long:  mean={thr_long.mean():.4f}, std={thr_long.std():.4f}, "
                  f"min={thr_long.min():.4f}, max={thr_long.max():.4f}")
            print(f"  thr_short: mean={thr_short.mean():.4f}, std={thr_short.std():.4f}, "
                  f"min={thr_short.min():.4f}, max={thr_short.max():.4f}")
            print(f"  thr_long p50/p90: {thr_long.quantile(0.5):.4f} / {thr_long.quantile(0.9):.4f}")
            print(f"  thr_short p50/p90: {thr_short.quantile(0.5):.4f} / {thr_short.quantile(0.9):.4f}")

            # 9. Regime-based breakdown (bull/bear x hot/cold)
            p_bull = result['p_bull']
            T_col = result['T'] if 'T' in result.columns else None
            action_code = result['action_code']

            # Define regimes
            is_bull = p_bull >= 0.5
            is_bear = p_bull < 0.5
            if T_col is not None:
                T_median = T_col.median()
                is_hot = T_col > T_median  # Above median = hot (uncertain)
                is_cold = T_col <= T_median  # Below median = cold (confident)
            else:
                is_hot = pd.Series(False, index=result.index)
                is_cold = pd.Series(True, index=result.index)

            print(f"\n[Regime-based Breakdown]")
            print(f"  T median (hot/cold split): {T_col.median():.4f}" if T_col is not None else "  T column not available")

            # 4 quadrants: bull_hot, bull_cold, bear_hot, bear_cold
            regimes = {
                'BULL_HOT':  is_bull & is_hot,
                'BULL_COLD': is_bull & is_cold,
                'BEAR_HOT':  is_bear & is_hot,
                'BEAR_COLD': is_bear & is_cold,
            }

            for regime_name, mask in regimes.items():
                regime_count = mask.sum()
                long_count = (action_code[mask] == 1).sum()
                short_count = (action_code[mask] == -1).sum()
                flat_count = (action_code[mask] == 0).sum()
                thr_l_mean = thr_long[mask].mean() if mask.sum() > 0 else 0
                thr_s_mean = thr_short[mask].mean() if mask.sum() > 0 else 0
                print(f"  {regime_name:10s}: {regime_count:5d} bars | "
                      f"L:{long_count:4d} S:{short_count:4d} F:{flat_count:4d} | "
                      f"thr_L={thr_l_mean:.3f} thr_S={thr_s_mean:.3f}")

        # PR4.3: SHORT 타이밍 피처 추가 (EMA, ret_n)
        close = df_5m_indexed['close']
        ema_period = config.prob_gate_short_ema_period  # default 20
        ret_bars = config.prob_gate_short_ret_bars      # default 3

        result['ema_short'] = close.ewm(span=ema_period, adjust=False).mean()
        result['ret_n'] = close.pct_change(periods=ret_bars)
        result['close_5m'] = close.values  # 현재 close 저장

        # PR4.4.1: 1H EMA200 기반 드리프트 레짐 계산 (히스테리시스 + slope)
        # PR-B: use_drift_filter=False면 drift_regime 계산 스킵
        if config.use_drift_filter and config.prob_gate_use_drift_thr:
            # 1H close 추출
            df_1h_close = df_1h_for_score.set_index('timestamp')['close'] if 'timestamp' in df_1h_for_score.columns else df_1h_for_score['close']

            # EMA200 계산 (1H 기준)
            drift_ema_period = config.prob_gate_drift_ema_period  # default 200
            ema_1h = df_1h_close.ewm(span=drift_ema_period, adjust=False).mean()

            # PR4.4.1: EMA slope 계산 (24시간 = 24 bars at 1H)
            slope_bars = config.prob_gate_drift_slope_bars  # default 24
            ema_slope_1h = ema_1h / ema_1h.shift(slope_bars) - 1

            # 히스테리시스 파라미터
            enter_pct = config.prob_gate_drift_enter_pct  # 1.2% (진입 조건)
            exit_pct = config.prob_gate_drift_exit_pct    # 0.8% (탈출 조건)
            min_bars = config.prob_gate_drift_min_bars    # 3 bars 연속
            use_slope = config.prob_gate_drift_use_slope  # slope 조건 사용

            # 5m에 정렬 (완료된 1H봉 기준, lookahead 방지)
            drift_regime_5m = []
            current_regime = 'RANGE'  # 상태 머신: 현재 레짐
            candidate_regime = None   # 후보 레짐 (연속 카운트용)
            consecutive_count = 0     # 연속 카운트

            for ts_5m in df_5m_indexed.index:
                # 완료된 context_tf봉 (현재 trigger_tf 시점에서 알 수 있는 가장 최근 봉)
                ts_context = _floor_by_tf(ts_5m, config.context_tf)  # lookahead 방지

                if ts_context in ema_1h.index and ts_context in df_1h_close.index:
                    close_ctx = df_1h_close.loc[ts_context]
                    ema_val = ema_1h.loc[ts_context]
                    slope_val = ema_slope_1h.loc[ts_context] if ts_context in ema_slope_1h.index else 0

                    if ema_val > 0:
                        pct_diff = (close_ctx - ema_val) / ema_val

                        # === 히스테리시스 로직 ===
                        # 현재 레짐 유지 조건 (exit band 기준)
                        if current_regime == 'UPTREND':
                            # UP 유지: close > ema*(1+exit_pct)
                            if pct_diff >= exit_pct:
                                drift_regime_5m.append('UPTREND')
                                candidate_regime = None
                                consecutive_count = 0
                                continue
                        elif current_regime == 'DOWNTREND':
                            # DOWN 유지: close < ema*(1-exit_pct)
                            if pct_diff <= -exit_pct:
                                drift_regime_5m.append('DOWNTREND')
                                candidate_regime = None
                                consecutive_count = 0
                                continue

                        # 레짐 전환 후보 감지 (enter band 기준)
                        new_candidate = None
                        if pct_diff > enter_pct:
                            # UP 진입 후보 (slope 조건 확인)
                            if use_slope and slope_val <= 0:
                                new_candidate = 'RANGE'  # slope 불일치 → RANGE
                            else:
                                new_candidate = 'UPTREND'
                        elif pct_diff < -enter_pct:
                            # DOWN 진입 후보 (slope 조건 확인)
                            if use_slope and slope_val >= 0:
                                new_candidate = 'RANGE'  # slope 불일치 → RANGE
                            else:
                                new_candidate = 'DOWNTREND'
                        else:
                            new_candidate = 'RANGE'

                        # 연속 카운트 로직
                        if new_candidate == candidate_regime:
                            consecutive_count += 1
                        else:
                            candidate_regime = new_candidate
                            consecutive_count = 1

                        # min_bars 연속 시 레짐 전환
                        if consecutive_count >= min_bars and candidate_regime != current_regime:
                            current_regime = candidate_regime

                        drift_regime_5m.append(current_regime)
                    else:
                        drift_regime_5m.append('RANGE')
                else:
                    drift_regime_5m.append('RANGE')  # 데이터 없으면 기본값

            result['drift_regime'] = drift_regime_5m

            # 드리프트 레짐 통계 출력
            regime_counts = pd.Series(drift_regime_5m).value_counts()
            print(f"\n[Drift Regime Stats - PR4.4.1 Hysteresis]")
            print(f"  Enter: ±{enter_pct*100:.1f}% | Exit: ±{exit_pct*100:.1f}% | Min bars: {min_bars}")
            print(f"  Slope condition: {'ON' if use_slope else 'OFF'} ({slope_bars}h)")
            for regime, count in regime_counts.items():
                pct = count / len(drift_regime_5m) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")

        return result

    except Exception as e:
        print(f"[ERROR] precompute_prob_gate failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def precompute_prob_gate_via_store(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    config: Config
) -> Optional[pd.DataFrame]:
    """
    ProbabilityGate 결과를 FeatureStore를 통해 계산 (P5).

    FeatureStore의 장점:
      - 중복 계산 제거 (캐싱)
      - FAIL-CLOSED warmup validation
      - 정의 충돌 감지 (같은 이름, 다른 파라미터)

    Args:
        df_1h: 1H 데이터 (context TF)
        df_5m: 5m 데이터 (trigger TF)
        config: Config with prob_gate settings

    Returns:
        DataFrame with prob_gate_bundle columns, indexed by 5m timestamp
    """
    if not FEATURE_STORE_AVAILABLE:
        print("[WARN] FeatureStore not available, falling back to legacy precompute")
        return precompute_prob_gate(df_1h, df_5m, config)

    try:
        # 1. Prepare DataFrames with DatetimeIndex
        df_trigger = df_5m.copy()
        if 'timestamp' in df_trigger.columns and not isinstance(df_trigger.index, pd.DatetimeIndex):
            df_trigger['timestamp'] = pd.to_datetime(df_trigger['timestamp'])
            df_trigger = df_trigger.set_index('timestamp').sort_index()

        df_context = df_1h.copy()
        if 'timestamp' in df_context.columns and not isinstance(df_context.index, pd.DatetimeIndex):
            df_context['timestamp'] = pd.to_datetime(df_context['timestamp'])
            df_context = df_context.set_index('timestamp').sort_index()

        # 2. Create FeatureStore with duration_to_bars wired
        store = FeatureStore(config, duration_to_bars=_duration_to_bars)

        # 3. Attach DataFrames by role
        store.attach("trigger", df_trigger, timeframe=config.trigger_tf)
        store.attach("context", df_context, timeframe=config.context_tf)

        # 4. Register default features (hilbert, drift, ret_n, ema_short, prob_gate_bundle)
        store.register_default_prob_gate_bundle()

        # 5. Get prob_gate_bundle (computes all dependencies)
        print("[FeatureStore] Computing prob_gate_bundle...")
        result = store.get_df("prob_gate_bundle")

        # 6. Print validation stats
        if 'valid' in result.columns:
            valid_count = result['valid'].sum()
            total_count = len(result)
            print(f"  Computed: {total_count:,} bars, valid: {valid_count:,}")
            action_counts = result[result['valid']]['action_str'].value_counts()
            for action, cnt in action_counts.items():
                print(f"    {action}: {cnt:,} ({100*cnt/valid_count:.1f}%)")
        else:
            print(f"  Computed: {len(result):,} bars")

        return result

    except Exception as e:
        print(f"[ERROR] precompute_prob_gate_via_store failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to legacy
        print("[WARN] Falling back to legacy precompute_prob_gate")
        return precompute_prob_gate(df_1h, df_5m, config)


# GPT 진단: Pass Reasons 카운터
prob_gate_pass_reasons = {
    # FAIL-CLOSED rejects (이것들이 1이라도 있으면 백테스트 무효)
    "INDEX_MISSING_REJECT": 0,
    "WARMUP_REJECT": 0,
    "INVALID_ROW_REJECT": 0,
    # Normal passes
    "NORMAL_PASS_LONG": 0,
    "NORMAL_PASS_SHORT": 0,
}


def check_prob_gate_filter(
    side: str,
    current_time: pd.Timestamp,
    prob_gate_result: Optional[pd.DataFrame],
    config: Optional['Config'] = None
) -> Tuple[bool, str, float]:
    """
    ProbabilityGate v2 필터 체크

    Args:
        side: 'long' or 'short'
        current_time: 현재 5m 타임스탬프
        prob_gate_result: Pre-computed gate result
        config: Config 객체 (PR4.2 conf 필터용)

    Returns:
        (pass_filter, reject_reason, p_bull)
    """
    global prob_gate_pass_reasons

    if prob_gate_result is None:
        return True, "", 0.5

    # CRITICAL: FAIL-CLOSED - Gate row 누락 시 차단 (인덱스 불일치 감지용)
    if current_time not in prob_gate_result.index:
        prob_gate_pass_reasons["INDEX_MISSING_REJECT"] += 1
        return False, "PROB_GATE_INDEX_MISSING", 0.5

    row = prob_gate_result.loc[current_time]

    # CRITICAL: Warmup 구간 진입 차단 (초반 불완전 데이터 방지)
    if not row['valid']:
        prob_gate_pass_reasons["WARMUP_REJECT"] += 1
        return False, "PROB_GATE_WARMUP", 0.5

    p_bull = float(row['p_bull'])

    # FAIL-CLOSED: 잘못된 값도 차단
    if not np.isfinite(p_bull):
        prob_gate_pass_reasons["INVALID_ROW_REJECT"] += 1
        return False, "PROB_GATE_INVALID_ROW", 0.5

    action_code = int(row['action_code'])

    # Gate 로직: action_code와 side 일치 여부
    if side == 'long':
        if action_code == 1:  # LONG 허용
            prob_gate_pass_reasons["NORMAL_PASS_LONG"] += 1
            return True, "", p_bull
        else:
            return False, "PROB_GATE_NOT_LONG", p_bull
    else:  # short
        # === PR4.4: 드리프트 기반 동적 threshold ===
        # PR-B: use_drift_filter=False면 drift 기반 threshold 조정 스킵
        if config is not None and config.use_drift_filter and config.prob_gate_use_drift_thr:
            # 드리프트 레짐에 따른 thr_short 결정
            drift_regime = str(row.get('drift_regime', 'RANGE')) if 'drift_regime' in row.index else 'RANGE'

            if drift_regime == 'UPTREND':
                thr_short = config.prob_gate_thr_short_uptrend  # 0.70 (엄격)
            elif drift_regime == 'DOWNTREND':
                thr_short = config.prob_gate_thr_short_downtrend  # 0.62 (관대)
            else:  # RANGE
                thr_short = config.prob_gate_thr_short_range  # 0.65 (기본)

            # SHORT 조건 재계산: p_bull < (1 - thr_short)
            if p_bull >= (1 - thr_short):
                return False, f"PROB_GATE_NOT_SHORT_{drift_regime}", p_bull
        else:
            # 기존 로직: precompute에서 계산된 action_code 사용
            if action_code != -1:  # SHORT 불허용 (기존 threshold 기반)
                return False, "PROB_GATE_NOT_SHORT", p_bull

        # === PR4.2: SHORT conf 필터 ===
        if config is not None and config.prob_gate_use_conf_filter:
            # conf = abs(p_bull - 0.5) * 2, 0~1 범위
            conf = abs(p_bull - 0.5) * 2.0
            conf_min = config.prob_gate_conf_min_short

            # HOT 구간이면 conf 요구량 상향
            T = float(row.get('T', 1.0)) if 'T' in row.index else 1.0
            is_hot = T >= config.prob_gate_T_hot

            if is_hot:
                conf_min += config.prob_gate_conf_hot_add
                if conf < conf_min:
                    return False, "PROB_GATE_SHORT_LOW_CONF_HOT", p_bull
            else:
                if conf < conf_min:
                    return False, "PROB_GATE_SHORT_LOW_CONF", p_bull

        # === PR4.3: SHORT 타이밍 확인 (ret_n + EMA) ===
        if config is not None and config.prob_gate_use_short_timing:
            # 조건 A: 단기 모멘텀 확인 - ret_n < ret_min (음수)
            ret_n = float(row.get('ret_n', 0.0)) if 'ret_n' in row.index else 0.0
            ret_min = config.prob_gate_short_ret_min  # default -0.0005 (-0.05%)

            if not np.isfinite(ret_n) or ret_n >= ret_min:
                return False, "PROB_GATE_SHORT_NO_MOMENTUM", p_bull

            # 조건 B: 위치 확인 - close < EMA
            close = float(row.get('close_5m', 0.0)) if 'close_5m' in row.index else 0.0
            ema = float(row.get('ema_short', 0.0)) if 'ema_short' in row.index else 0.0

            if close <= 0 or ema <= 0:
                # 데이터 없으면 통과
                pass
            elif close >= ema:
                return False, "PROB_GATE_SHORT_ABOVE_EMA", p_bull

        prob_gate_pass_reasons["NORMAL_PASS_SHORT"] += 1
        return True, "", p_bull


def calculate_atr_percentile(atr_values: np.ndarray, lookback: int = 100) -> float:
    """현재 ATR이 최근 N개 바 중 몇 percentile인지"""
    if len(atr_values) < lookback:
        return 50.0

    recent = atr_values[-lookback:]
    valid = recent[np.isfinite(recent)]
    if len(valid) < 10:
        return 50.0

    current = atr_values[-1]
    if not np.isfinite(current):
        return 50.0

    percentile = (valid < current).sum() / len(valid) * 100
    return percentile


def check_trend_filter(
    side: str,
    trend_1h: str,
    trend_4h: str,
    use_1h_filter: bool,
    use_4h_filter: bool
) -> Tuple[bool, str]:
    """
    추세 필터 체크
    Returns: (pass_filter, reject_reason)
    """
    # 1H 필터
    if use_1h_filter:
        if side == "long" and trend_1h == "DOWNTREND":
            return False, "1H_DOWNTREND"
        if side == "short" and trend_1h == "UPTREND":
            return False, "1H_UPTREND"

    # 4H 필터
    if use_4h_filter:
        if side == "long" and trend_4h == "DOWNTREND":
            return False, "4H_DOWNTREND"
        if side == "short" and trend_4h == "UPTREND":
            return False, "4H_UPTREND"

    return True, ""


def check_hilbert_filter(
    side: str,
    hilbert_regime: str,
    block_long_on_bear: bool,
    block_short_on_bull: bool
) -> Tuple[bool, str]:
    """
    Hilbert 레짐 필터 체크
    Returns: (pass_filter, reject_reason)

    - Long: BEAR 레짐에서 차단 (가격이 EMA 위, 하락 예상)
    - Short: BULL 레짐에서 차단 (가격이 EMA 아래, 상승 예상)
    """
    if side == "long" and block_long_on_bear:
        if hilbert_regime == "BEAR":
            return False, "HILBERT_BEAR"

    if side == "short" and block_short_on_bull:
        if hilbert_regime == "BULL":
            return False, "HILBERT_BULL"

    return True, ""


def get_current_hilbert_regime(
    hilbert_regimes: Optional[pd.DataFrame],
    current_time: pd.Timestamp,
    context_tf: str = '1h'  # config.context_tf
) -> str:
    """
    현재 시간의 Hilbert 레짐 가져오기 (causal - 완료된 context_tf봉만 사용)
    Returns: 'BULL', 'BEAR', or 'RANGE'
    """
    if hilbert_regimes is None:
        return 'RANGE'

    # 완료된 context_tf봉 기준 (lookahead 방지)
    ts_ctx = _floor_by_tf(current_time, context_tf)

    if isinstance(hilbert_regimes.index, pd.DatetimeIndex):
        if ts_ctx in hilbert_regimes.index:
            regime = hilbert_regimes.loc[ts_ctx, 'regime']
            return str(regime) if pd.notna(regime) else 'RANGE'

        # 가장 가까운 이전 timestamp 찾기
        mask = hilbert_regimes.index <= ts_ctx
        if mask.any():
            closest = hilbert_regimes.index[mask][-1]
            regime = hilbert_regimes.loc[closest, 'regime']
            return str(regime) if pd.notna(regime) else 'RANGE'

    return 'RANGE'


# =============================================================================
# RSI 역산 함수
# =============================================================================
def _rsi_at_price(close_arr: np.ndarray, new_close: float, period: int = 14) -> float:
    """특정 가격에서의 RSI 계산"""
    close = close_arr.copy()
    close[-1] = new_close
    rsi = calc_rsi_wilder(close, period)
    return float(rsi[-1]) if np.isfinite(rsi[-1]) else np.nan

def needed_close_for_regular_bullish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
) -> Optional[float]:
    """Regular Bullish: 가격 < ref, RSI > ref

    동적 범위 탐색:
    1. U = ref_price (약간 아래)에서 rsi_U > ref_rsi 확인
    2. L을 점진적으로 낮추면서 rsi_L <= ref_rsi가 되는 지점 찾기
    3. L~U 범위에서 이분탐색
    """
    eps = 1e-8
    U = ref_price - max(eps, abs(ref_price) * 1e-6)

    if not np.isfinite(U) or U <= 0:
        return None

    # Step 1: U에서 RSI > ref_rsi 확인 (필수조건)
    rsi_U = _rsi_at_price(close_arr, U, rsi_period)
    if not np.isfinite(rsi_U) or rsi_U <= ref_rsi:
        return None

    # Step 2: 동적 범위 탐색 - L을 찾기 (rsi_L <= ref_rsi가 되는 지점)
    L = U * 0.99  # 1% 아래부터 시작
    found_L = False
    for _ in range(100):  # 최대 100번 확장
        if L <= 0:
            break
        rsi_L = _rsi_at_price(close_arr, L, rsi_period)
        if np.isfinite(rsi_L) and rsi_L <= ref_rsi:
            found_L = True
            break
        L *= 0.95  # 5%씩 더 낮춤

    if not found_L:
        # 범위 내에서 경계를 못 찾음 → 가장 낮은 가격 반환
        result = L * 1.001
        final_rsi = _rsi_at_price(close_arr, result, rsi_period)
        if np.isfinite(final_rsi) and final_rsi > ref_rsi and result < ref_price:
            return float(result)
        return None

    # Step 3: 이분탐색 (L~U 범위에서 경계 찾기)
    lo, hi = L, U
    for _ in range(60):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)
        if not np.isfinite(rsi_mid):
            lo = mid
            continue
        if rsi_mid > ref_rsi:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= 1e-6:
            break

    result = min(hi, U) * 1.001  # 경계보다 0.1% 위 → RSI > ref_rsi 확정
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi <= ref_rsi:
        return None
    return float(result) if result > 0 else None

def needed_close_for_regular_bearish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
) -> Optional[float]:
    """Regular Bearish: 가격 > ref, RSI < ref

    동적 범위 탐색:
    1. L = ref_price (약간 위)에서 rsi_L < ref_rsi 확인
    2. U를 점진적으로 높이면서 rsi_U >= ref_rsi가 되는 지점 찾기
    3. L~U 범위에서 이분탐색
    """
    eps = 1e-8
    L = ref_price + max(eps, abs(ref_price) * 1e-6)

    if not np.isfinite(L):
        return None

    # Step 1: L에서 RSI < ref_rsi 확인 (필수조건)
    rsi_L = _rsi_at_price(close_arr, L, rsi_period)
    if not np.isfinite(rsi_L) or rsi_L >= ref_rsi:
        return None

    # Step 2: 동적 범위 탐색 - U를 찾기 (rsi_U >= ref_rsi가 되는 지점)
    U = L * 1.01  # 1% 위부터 시작
    found_U = False
    for _ in range(100):  # 최대 100번 확장
        rsi_U = _rsi_at_price(close_arr, U, rsi_period)
        if np.isfinite(rsi_U) and rsi_U >= ref_rsi:
            found_U = True
            break
        U *= 1.05  # 5%씩 더 높임

    if not found_U:
        # 범위 내에서 경계를 못 찾음 → 가장 높은 가격 반환
        result = U * 0.999
        final_rsi = _rsi_at_price(close_arr, result, rsi_period)
        if np.isfinite(final_rsi) and final_rsi < ref_rsi and result > ref_price:
            return float(result)
        return None

    # Step 3: 이분탐색 (L~U 범위에서 경계 찾기)
    lo, hi = L, U
    for _ in range(60):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)
        if not np.isfinite(rsi_mid):
            hi = mid
            continue
        if rsi_mid < ref_rsi:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= 1e-6:
            break

    result = max(lo, L) * 0.999  # 경계보다 0.1% 아래 → RSI < ref_rsi 확정
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi >= ref_rsi:
        return None
    return float(result)


def needed_close_for_hidden_bullish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
) -> Optional[float]:
    """
    Hidden Bullish Divergence 성립 가격 역산
    조건: 가격 > ref_price (Higher Low), RSI < ref_rsi (Lower Low)

    동적 범위 탐색:
    1. L = ref_price (약간 위)에서 rsi_L < ref_rsi 확인
    2. U를 점진적으로 높이면서 rsi_U >= ref_rsi가 되는 지점 찾기
    3. L~U 범위에서 이분탐색
    """
    eps = 1e-8
    L = ref_price + max(eps, abs(ref_price) * 1e-6)

    if not np.isfinite(L):
        return None

    # Step 1: L에서 RSI < ref_rsi 확인 (필수조건)
    rsi_L = _rsi_at_price(close_arr, L, rsi_period)
    if not np.isfinite(rsi_L) or rsi_L >= ref_rsi:
        return None

    # Step 2: 동적 범위 탐색 - U를 찾기 (rsi_U >= ref_rsi가 되는 지점)
    U = L * 1.01  # 1% 위부터 시작
    found_U = False
    for _ in range(100):  # 최대 100번 확장
        rsi_U = _rsi_at_price(close_arr, U, rsi_period)
        if np.isfinite(rsi_U) and rsi_U >= ref_rsi:
            found_U = True
            break
        U *= 1.05  # 5%씩 더 높임

    if not found_U:
        # 범위 내에서 경계를 못 찾음 → 가장 높은 가격 반환
        result = U * 0.999
        final_rsi = _rsi_at_price(close_arr, result, rsi_period)
        if np.isfinite(final_rsi) and final_rsi < ref_rsi and result > ref_price:
            return float(result)
        return None

    # Step 3: 이분탐색 (L~U 범위에서 경계 찾기)
    lo, hi = L, U
    for _ in range(60):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)
        if not np.isfinite(rsi_mid):
            hi = mid
            continue
        if rsi_mid < ref_rsi:
            lo = mid  # RSI가 낮으니까 가격을 더 올려봐야 함
        else:
            hi = mid  # RSI가 높으면 가격을 낮춰야 함
        if abs(hi - lo) <= 1e-6:
            break

    result = max(lo, L) * 0.999  # 경계보다 0.1% 아래 → RSI < ref_rsi 확정
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi >= ref_rsi:
        return None
    return float(result)


def needed_close_for_hidden_bearish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
) -> Optional[float]:
    """
    Hidden Bearish Divergence 성립 가격 역산
    조건: 가격 < ref_price (Lower High), RSI > ref_rsi (Higher High)

    동적 범위 탐색:
    1. U = ref_price (약간 아래)에서 rsi_U > ref_rsi 확인
    2. L을 점진적으로 낮추면서 rsi_L <= ref_rsi가 되는 지점 찾기
    3. L~U 범위에서 이분탐색
    """
    eps = 1e-8
    U = ref_price - max(eps, abs(ref_price) * 1e-6)

    if not np.isfinite(U) or U <= 0:
        return None

    # Step 1: U에서 RSI > ref_rsi 확인 (필수조건)
    rsi_U = _rsi_at_price(close_arr, U, rsi_period)
    if not np.isfinite(rsi_U) or rsi_U <= ref_rsi:
        return None

    # Step 2: 동적 범위 탐색 - L을 찾기 (rsi_L <= ref_rsi가 되는 지점)
    L = U * 0.99  # 1% 아래부터 시작
    found_L = False
    for _ in range(100):  # 최대 100번 확장
        if L <= 0:
            break
        rsi_L = _rsi_at_price(close_arr, L, rsi_period)
        if np.isfinite(rsi_L) and rsi_L <= ref_rsi:
            found_L = True
            break
        L *= 0.95  # 5%씩 더 낮춤

    if not found_L:
        # 범위 내에서 경계를 못 찾음 → 가장 낮은 가격 반환
        result = L * 1.001
        final_rsi = _rsi_at_price(close_arr, result, rsi_period)
        if np.isfinite(final_rsi) and final_rsi > ref_rsi and result < ref_price:
            return float(result)
        return None

    # Step 3: 이분탐색 (L~U 범위에서 경계 찾기)
    lo, hi = L, U
    for _ in range(60):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)
        if not np.isfinite(rsi_mid):
            lo = mid
            continue
        if rsi_mid > ref_rsi:
            hi = mid  # RSI가 높으니까 가격을 더 낮춰봐야 함
        else:
            lo = mid  # RSI가 낮으면 가격을 올려야 함
        if abs(hi - lo) <= 1e-6:
            break

    result = min(hi, U) * 1.001  # 경계보다 0.1% 위 → RSI > ref_rsi 확정
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi <= ref_rsi:
        return None
    return float(result) if result > 0 else None


# =============================================================================
# 참조점 찾기
# =============================================================================
def find_oversold_reference(df: pd.DataFrame, lookback: int = 100, threshold: float = 20.0) -> Optional[Dict]:
    """최근 oversold 구간의 저점

    Args:
        df: OHLC + stoch_d + rsi DataFrame
        lookback: 참조 윈도우 크기
        threshold: 과매도 임계값 (PR4-R0: 하드코딩 제거)
    """
    if len(df) < 10:
        return None

    d = df['stoch_d'].values[-lookback:] if len(df) >= lookback else df['stoch_d'].values
    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] <= threshold:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= threshold:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 직전봉이 과매도일 때만 전 세그먼트 사용 (lookahead 방지: 현재봉은 미확정)
    if len(d) < 2:
        return None
    prev_oversold = np.isfinite(d[-2]) and d[-2] <= threshold
    if not prev_oversold:
        return None  # 직전봉이 과매도 아니면 참조점 없음

    if len(segments) < 2:
        return None  # 전 세그먼트가 없으면 참조점 없음

    seg = segments[-2]  # 전 세그먼트 (현재 세그먼트 제외)

    a, b = seg
    seg_close = close[a:b+1]
    seg_rsi = rsi[a:b+1]

    min_idx = np.argmin(seg_close)
    ref_price = float(seg_close[min_idx])
    ref_rsi = float(seg_rsi[min_idx])

    if not np.isfinite(ref_rsi):
        return None
    return {'ref_price': ref_price, 'ref_rsi': ref_rsi}

def find_overbought_reference(df: pd.DataFrame, lookback: int = 100, threshold: float = 80.0) -> Optional[Dict]:
    """최근 overbought 구간의 고점

    Args:
        df: OHLC + stoch_d + rsi DataFrame
        lookback: 참조 윈도우 크기
        threshold: 과매수 임계값 (PR4-R0: 하드코딩 제거)
    """
    if len(df) < 10:
        return None

    d = df['stoch_d'].values[-lookback:] if len(df) >= lookback else df['stoch_d'].values
    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] >= threshold:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] >= threshold:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 직전봉이 과매수일 때만 전 세그먼트 사용 (lookahead 방지: 현재봉은 미확정)
    if len(d) < 2:
        return None
    prev_overbought = np.isfinite(d[-2]) and d[-2] >= threshold
    if not prev_overbought:
        return None  # 직전봉이 과매수 아니면 참조점 없음

    if len(segments) < 2:
        return None  # 전 세그먼트가 없으면 참조점 없음

    seg = segments[-2]  # 전 세그먼트 (현재 세그먼트 제외)

    a, b = seg
    seg_close = close[a:b+1]
    seg_rsi = rsi[a:b+1]

    max_idx = np.argmax(seg_close)
    ref_price = float(seg_close[max_idx])
    ref_rsi = float(seg_rsi[max_idx])

    if not np.isfinite(ref_rsi):
        return None
    return {'ref_price': ref_price, 'ref_rsi': ref_rsi}


def find_swing_low_reference(df: pd.DataFrame, lookback: int = 50, min_bars_back: int = 5) -> Optional[Dict]:
    """
    Hidden Bullish Divergence용 스윙 저점 찾기
    현재 과매도 조건 없이 최근 스윙 저점을 찾음
    """
    if len(df) < 20:
        return None

    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values
    low = df['low'].values[-lookback:] if len(df) >= lookback else df['low'].values

    n = len(close)
    if n < min_bars_back + 5:
        return None

    # 최근 min_bars_back 이전의 데이터에서 스윙 저점 찾기
    search_range = close[:-min_bars_back]
    search_rsi = rsi[:-min_bars_back]
    search_low = low[:-min_bars_back]

    if len(search_range) < 5:
        return None

    # 간단한 스윙 저점: 좌우 2개 바보다 낮은 점
    swing_lows = []
    for i in range(2, len(search_low) - 2):
        if (search_low[i] < search_low[i-1] and search_low[i] < search_low[i-2] and
            search_low[i] < search_low[i+1] and search_low[i] < search_low[i+2]):
            swing_lows.append(i)

    if not swing_lows:
        # 스윙 저점이 없으면 가장 낮은 점 사용
        min_idx = np.argmin(search_low)
        if min_idx < 2 or min_idx >= len(search_low) - 2:
            return None
        swing_lows = [min_idx]

    # 가장 최근 스윙 저점 사용
    ref_idx = swing_lows[-1]
    ref_price = float(search_range[ref_idx])
    ref_rsi = float(search_rsi[ref_idx])

    if not np.isfinite(ref_rsi):
        return None

    return {'ref_price': ref_price, 'ref_rsi': ref_rsi}


def find_swing_high_reference(df: pd.DataFrame, lookback: int = 50, min_bars_back: int = 5) -> Optional[Dict]:
    """
    Hidden Bearish Divergence용 스윙 고점 찾기
    현재 과매수 조건 없이 최근 스윙 고점을 찾음
    """
    if len(df) < 20:
        return None

    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values
    high = df['high'].values[-lookback:] if len(df) >= lookback else df['high'].values

    n = len(close)
    if n < min_bars_back + 5:
        return None

    # 최근 min_bars_back 이전의 데이터에서 스윙 고점 찾기
    search_range = close[:-min_bars_back]
    search_rsi = rsi[:-min_bars_back]
    search_high = high[:-min_bars_back]

    if len(search_range) < 5:
        return None

    # 간단한 스윙 고점: 좌우 2개 바보다 높은 점
    swing_highs = []
    for i in range(2, len(search_high) - 2):
        if (search_high[i] > search_high[i-1] and search_high[i] > search_high[i-2] and
            search_high[i] > search_high[i+1] and search_high[i] > search_high[i+2]):
            swing_highs.append(i)

    if not swing_highs:
        # 스윙 고점이 없으면 가장 높은 점 사용
        max_idx = np.argmax(search_high)
        if max_idx < 2 or max_idx >= len(search_high) - 2:
            return None
        swing_highs = [max_idx]

    # 가장 최근 스윙 고점 사용
    ref_idx = swing_highs[-1]
    ref_price = float(search_range[ref_idx])
    ref_rsi = float(search_rsi[ref_idx])

    if not np.isfinite(ref_rsi):
        return None

    return {'ref_price': ref_price, 'ref_rsi': ref_rsi}


def find_oversold_reference_hybrid(
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    lookback: int = 100,
    threshold: float = 20.0
) -> Optional[Dict]:
    """
    15m StochRSI 세그먼트 + 5m RSI 레퍼런스
    - 15m StochRSI로 과매도 세그먼트 시간 구간 찾기
    - 해당 시간 구간의 5m 데이터에서 종가 최저점의 RSI를 레퍼런스로 사용

    Args:
        threshold: StochRSI 과매도 임계값 (기본값 20.0)
    """
    if len(df_15m) < 10 or len(df_5m) < 10:
        return None

    # 15m StochRSI 세그먼트 찾기
    d = df_15m['stoch_d'].values[-lookback:] if len(df_15m) >= lookback else df_15m['stoch_d'].values
    idx_15m = df_15m.index[-lookback:] if len(df_15m) >= lookback else df_15m.index

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] <= threshold:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= threshold:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 현재 과매도일 때만 전 세그먼트 사용
    current_oversold = np.isfinite(d[-1]) and d[-1] <= threshold
    if not current_oversold:
        return None

    if len(segments) < 2:
        return None

    # 전 세그먼트 시간 구간 추출
    seg = segments[-2]
    a, b = seg
    seg_start_time = idx_15m[a]
    seg_end_time = idx_15m[b]

    # 해당 시간 구간의 5m 데이터 필터링
    mask_5m = (df_5m.index >= seg_start_time) & (df_5m.index <= seg_end_time + pd.Timedelta(minutes=15))
    df_5m_seg = df_5m[mask_5m]

    if len(df_5m_seg) == 0:
        return None

    # 5m에서 종가 최저점의 RSI를 레퍼런스로 사용
    min_idx = df_5m_seg['close'].idxmin()
    ref_price = float(df_5m_seg.loc[min_idx, 'close'])
    ref_rsi = float(df_5m_seg.loc[min_idx, 'rsi'])

    if not np.isfinite(ref_rsi):
        return None
    return {'ref_price': ref_price, 'ref_rsi': ref_rsi}


def find_overbought_reference_hybrid(
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    lookback: int = 100,
    threshold: float = 80.0
) -> Optional[Dict]:
    """
    15m StochRSI 세그먼트 + 5m RSI 레퍼런스
    - 15m StochRSI로 과매수 세그먼트 시간 구간 찾기
    - 해당 시간 구간의 5m 데이터에서 종가 최고점의 RSI를 레퍼런스로 사용

    Args:
        threshold: StochRSI 과매수 임계값 (기본값 80.0)
    """
    if len(df_15m) < 10 or len(df_5m) < 10:
        return None

    # 15m StochRSI 세그먼트 찾기
    d = df_15m['stoch_d'].values[-lookback:] if len(df_15m) >= lookback else df_15m['stoch_d'].values
    idx_15m = df_15m.index[-lookback:] if len(df_15m) >= lookback else df_15m.index

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] >= threshold:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] >= threshold:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 현재 과매수일 때만 전 세그먼트 사용
    current_overbought = np.isfinite(d[-1]) and d[-1] >= threshold
    if not current_overbought:
        return None

    if len(segments) < 2:
        return None

    # 전 세그먼트 시간 구간 추출
    seg = segments[-2]
    a, b = seg
    seg_start_time = idx_15m[a]
    seg_end_time = idx_15m[b]

    # 해당 시간 구간의 5m 데이터 필터링
    mask_5m = (df_5m.index >= seg_start_time) & (df_5m.index <= seg_end_time + pd.Timedelta(minutes=15))
    df_5m_seg = df_5m[mask_5m]

    if len(df_5m_seg) == 0:
        return None

    # 5m에서 종가 최고점의 RSI를 레퍼런스로 사용
    max_idx = df_5m_seg['close'].idxmax()
    ref_price = float(df_5m_seg.loc[max_idx, 'close'])
    ref_rsi = float(df_5m_seg.loc[max_idx, 'rsi'])

    if not np.isfinite(ref_rsi):
        return None
    return {'ref_price': ref_price, 'ref_rsi': ref_rsi}


# =============================================================================
# Fib 바운더리 체크 (캐싱 적용)
# =============================================================================
# 전역 캐시: config별로 분리 저장 {cache_key: List[FibLevel]}
_FIB_LEVELS_CACHE: Dict[str, List[FibLevel]] = {}

def _get_fib_cache_key(config: Config = None) -> str:
    """Config 기반 캐시 키 생성"""
    if config is None:
        return "default_3120_143360_linear_1"
    return f"{config.fib_anchor_low}_{config.fib_anchor_high}_{config.fib_space}_{config.fib_max_depth}"

def _ensure_fib_cache(config: Config = None) -> str:
    """Fib 레벨 캐시 초기화 (config 지원)

    Returns:
        cache_key: 캐시 조회에 사용할 키
    """
    global _FIB_LEVELS_CACHE
    key = _get_fib_cache_key(config)

    if key not in _FIB_LEVELS_CACHE:
        if config is None:
            # Legacy 호환: 하드코딩된 값 사용
            _FIB_LEVELS_CACHE[key] = get_fractal_fib_levels(
                (3120, 143360), max_depth=1, space="linear"
            )
        else:
            # Config 기반: space + anchor 사용
            _FIB_LEVELS_CACHE[key] = get_fractal_fib_levels(
                (config.fib_anchor_low, config.fib_anchor_high),
                max_depth=config.fib_max_depth,
                space=config.fib_space
            )
        _FIB_LEVELS_CACHE[key].sort(key=lambda lvl: lvl.price)

    return key

def get_l1_boundary(price: float, config: Config = None) -> Optional[Tuple[float, float, FibLevel, FibLevel]]:
    """
    가격이 속한 L1 바운더리 반환 (캐시 사용)
    Returns: (lower_price, upper_price, lower_level, upper_level)
    """
    cache_key = _ensure_fib_cache(config)
    levels = _FIB_LEVELS_CACHE[cache_key]

    below = [lvl for lvl in levels if lvl.price < price]
    above = [lvl for lvl in levels if lvl.price > price]

    if not below or not above:
        return None

    lower = below[-1]  # 가장 가까운 아래 레벨
    upper = above[0]   # 가장 가까운 위 레벨

    return (lower.price, upper.price, lower, upper)

def is_near_boundary_edge(price: float, edge_pct: float = 0.15, config: Config = None) -> Tuple[bool, Optional[Tuple[float, float]], str]:
    """
    가격이 L1 바운더리의 극단(상단/하단 15%)에 있는지 체크

    Args:
        price: 체크할 가격
        edge_pct: 극단 범위 (0.15 = 상/하단 15%)
        config: Fib 설정 (None이면 기본값 사용)

    Returns:
        (극단 여부, (lower, upper), 위치='lower'|'upper'|'middle')
    """
    boundary = get_l1_boundary(price, config)
    if boundary is None:
        return False, None, 'none'

    lower_price, upper_price, _, _ = boundary
    range_size = upper_price - lower_price

    # 하단 극단: lower ~ lower + range*edge_pct
    lower_edge = lower_price + range_size * edge_pct
    # 상단 극단: upper - range*edge_pct ~ upper
    upper_edge = upper_price - range_size * edge_pct

    if price <= lower_edge:
        return True, (lower_price, upper_price), 'lower'
    elif price >= upper_edge:
        return True, (lower_price, upper_price), 'upper'
    else:
        return False, (lower_price, upper_price), 'middle'


def is_within_l1_boundary(price: float, config: Config = None) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """가격이 L1 바운더리 내에 있는지 체크 (하위 호환)"""
    boundary = get_l1_boundary(price, config)
    if boundary is None:
        return False, None

    lower_price, upper_price, _, _ = boundary
    is_within = lower_price <= price <= upper_price
    return is_within, (lower_price, upper_price)

def is_near_l1_level(
    price: float,
    atr: float = None,
    atr_mult: float = 1.0,
    tolerance_pct: float = None,
    config: Config = None
) -> Tuple[bool, Optional[FibLevel]]:
    """
    가격이 L1 레벨 근처인지 체크 (캐시 사용)

    Tolerance 모드 (config.fib_tolerance_mode):
    - "pct": 고정 % 기반 (config.fib_tolerance_pct)
    - "atr_pct": ATR 기반 (config.fib_tolerance_atr_mult * ATR / price)
    - "gap_frac": 인접 Fib 레벨 간격 기반 (미지원, fallback to pct)

    Args:
        price: 체크할 가격
        atr: ATR 값 (atr_pct 모드에서 필수)
        atr_mult: ATR 배수 (config 없을 때 사용, 기본 1.0)
        tolerance_pct: tolerance % (config 없을 때 사용)
        config: Fib 설정 (None이면 기존 동작)
    """
    cache_key = _ensure_fib_cache(config)
    levels = _FIB_LEVELS_CACHE[cache_key]

    l1_levels = [lvl for lvl in levels if lvl.depth <= 1]

    if not l1_levels:
        return False, None

    closest = min(l1_levels, key=lambda lvl: abs(lvl.price - price))

    # Tolerance 계산 (config 기반 or legacy)
    if config is not None:
        mode = config.fib_tolerance_mode
        if mode == "fib_gap":
            # fib_gap 모드: mult = 현재 Fib 구간의 gap (되돌림 변화량)
            # 예: 0.236~0.382 구간이면 mult = 0.382 - 0.236 = 0.146
            FIB_BOUNDARIES = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            fib_ratio = closest.fib_ratio if closest else 0.5
            # 해당 ratio가 속한 구간 찾기
            gap = 0.236  # 기본값
            for i in range(len(FIB_BOUNDARIES) - 1):
                if FIB_BOUNDARIES[i] <= fib_ratio <= FIB_BOUNDARIES[i + 1]:
                    gap = FIB_BOUNDARIES[i + 1] - FIB_BOUNDARIES[i]
                    break
            if atr is not None and atr > 0:
                effective_tolerance = (atr * gap) / price
            else:
                effective_tolerance = config.fib_tolerance_pct
        elif mode == "fib_ratio":
            # fib_ratio 모드: mult = 해당 Fib ratio (0.236~0.786)
            # 높은 ratio일수록 tolerance 넓음
            fib_ratio = closest.fib_ratio if closest else 0.5
            if atr is not None and atr > 0:
                effective_tolerance = (atr * fib_ratio) / price
            else:
                effective_tolerance = config.fib_tolerance_pct
        elif mode == "atr_pct":
            if atr is not None and atr > 0:
                effective_tolerance = (atr * config.fib_tolerance_atr_mult) / price
            else:
                # ATR이 없으면 fallback to pct
                effective_tolerance = config.fib_tolerance_pct
        elif mode == "gap_frac":
            # 인접 레벨 간격 기반 (구현 복잡, 일단 pct로 fallback)
            effective_tolerance = config.fib_tolerance_pct
        else:  # "pct" (기본)
            effective_tolerance = config.fib_tolerance_pct
    else:
        # Legacy 동작: 파라미터 기반
        if atr is not None and atr > 0:
            effective_tolerance = (atr * atr_mult) / price
        elif tolerance_pct is not None:
            effective_tolerance = tolerance_pct
        else:
            effective_tolerance = 0.01  # 기본값 1%

    distance_pct = abs(closest.price - price) / closest.price

    if distance_pct <= effective_tolerance:
        return True, closest
    return False, None

def get_next_l1_above(price: float, config: Config = None) -> Optional[FibLevel]:
    """현재 가격 위의 다음 L1 레벨"""
    cache_key = _ensure_fib_cache(config)
    levels = _FIB_LEVELS_CACHE[cache_key]
    above = [lvl for lvl in levels if lvl.price > price and lvl.depth <= 1]
    return above[0] if above else None

def get_next_l1_below(price: float, config: Config = None) -> Optional[FibLevel]:
    """현재 가격 아래의 다음 L1 레벨"""
    cache_key = _ensure_fib_cache(config)
    levels = _FIB_LEVELS_CACHE[cache_key]
    below = [lvl for lvl in levels if lvl.price < price and lvl.depth <= 1]
    return below[-1] if below else None


# =============================================================================
# PR-DYN-FIB: Dynamic Fib Level Check (for signal generation)
# =============================================================================
def is_near_dynamic_fib_level(
    price: float,
    dynfib_state,  # DynamicFibAnchorState
    atr: float,
    config: Config,
    ratios: tuple = (0.236, 0.382, 0.5, 0.618, 0.786)
) -> Tuple[bool, Optional[FibLevel]]:
    """
    가격이 Dynamic Fib 레벨 근처인지 체크

    Args:
        price: 체크할 가격
        dynfib_state: DynamicFibAnchorState (low, high 포함)
        atr: ATR 값
        config: Config (tolerance 설정 등)
        ratios: Fib 비율 튜플

    Returns:
        (is_near: bool, fib_level: Optional[FibLevel])
    """
    if dynfib_state is None or not dynfib_state.is_valid():
        return False, None

    # Dynamic Fib 레벨 계산
    dyn_levels = get_dynamic_fib_levels(
        dynfib_state.low,
        dynfib_state.high,
        ratios,
        space=config.dynamic_fib_space
    )

    if not dyn_levels:
        return False, None

    # 가장 가까운 레벨 찾기
    closest_price = min(dyn_levels, key=lambda lvl: abs(lvl - price))
    closest_idx = dyn_levels.index(closest_price)
    closest_ratio = ratios[closest_idx] if closest_idx < len(ratios) else 0.5

    # Tolerance 계산 (config 기반)
    if config.fib_tolerance_mode == "fib_gap":
        # fib_gap 모드: mult = 현재 Fib 구간의 gap (되돌림 변화량)
        FIB_BOUNDARIES = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        gap = 0.236  # 기본값
        for i in range(len(FIB_BOUNDARIES) - 1):
            if FIB_BOUNDARIES[i] <= closest_ratio <= FIB_BOUNDARIES[i + 1]:
                gap = FIB_BOUNDARIES[i + 1] - FIB_BOUNDARIES[i]
                break
        if atr is not None and atr > 0:
            effective_tolerance = (atr * gap) / price
        else:
            effective_tolerance = config.fib_tolerance_pct
    elif config.fib_tolerance_mode == "fib_ratio":
        # fib_ratio 모드: mult = 해당 Fib ratio (0.236~0.786)
        if atr is not None and atr > 0:
            effective_tolerance = (atr * closest_ratio) / price
        else:
            effective_tolerance = config.fib_tolerance_pct
    elif config.fib_tolerance_mode == "atr_pct":
        if atr is not None and atr > 0:
            effective_tolerance = (atr * config.fib_tolerance_atr_mult) / price
        else:
            effective_tolerance = config.fib_tolerance_pct
    else:
        effective_tolerance = config.fib_tolerance_pct

    distance_pct = abs(closest_price - price) / closest_price if closest_price > 0 else 1.0

    if distance_pct <= effective_tolerance:
        # FibLevel 객체 생성 (dynamic fib는 depth=0, cell=(0,0)으로 표시)
        fib_level = FibLevel(price=closest_price, fib_ratio=closest_ratio, depth=0, cell=(0, 0))
        return True, fib_level

    return False, None


def is_near_fib_level_combined(
    price: float,
    atr: float,
    config: Config,
    dynfib_state=None
) -> Tuple[bool, Optional[FibLevel], str]:
    """
    Macro Fib + Dynamic Fib 통합 체크 (dynfib_use_as == "both" 지원)

    Args:
        price: 체크할 가격
        atr: ATR 값
        config: Config
        dynfib_state: DynamicFibAnchorState (None이면 Macro만 체크)

    Returns:
        (is_near: bool, fib_level: Optional[FibLevel], source: "macro"|"dynfib"|"none")
    """
    # 1. Macro Fib 체크 (항상)
    is_near_macro, fib_level_macro = is_near_l1_level(price, atr=atr, config=config)

    # 2. Dynamic Fib 체크 (dynfib_use_as가 "both" 또는 "entry_filter"일 때)
    is_near_dyn = False
    fib_level_dyn = None
    if config.use_dynamic_fib and config.dynfib_use_as in ("both", "entry_filter"):
        if dynfib_state is not None and dynfib_state.is_valid():
            is_near_dyn, fib_level_dyn = is_near_dynamic_fib_level(
                price, dynfib_state, atr, config, config.dynfib_ratios
            )

    # 3. 결과 결정
    if is_near_macro and is_near_dyn:
        # 둘 다 매치 → 더 가까운 것 선택
        dist_macro = abs(fib_level_macro.price - price) if fib_level_macro else float('inf')
        dist_dyn = abs(fib_level_dyn.price - price) if fib_level_dyn else float('inf')
        if dist_dyn < dist_macro:
            return True, fib_level_dyn, "dynfib"
        return True, fib_level_macro, "macro"
    elif is_near_macro:
        return True, fib_level_macro, "macro"
    elif is_near_dyn:
        return True, fib_level_dyn, "dynfib"
    else:
        return False, None, "none"


# =============================================================================
# PR-ENTRY-RR2: RR Limit Entry Functions
# =============================================================================
def solve_entry_for_rr_net(
    tp: float,
    sl: float,
    target_rr: float,
    fee_bps: float = 4.0,
    slippage_bps: float = 5.0,
    side: str = "long"
) -> Optional[float]:
    """
    RR_net >= target_rr를 만족하는 entry 가격 역산

    수학 (LONG 기준, 수수료 무시 시):
    - RR = (TP - entry) / (entry - SL) >= R
    - TP - entry >= R * (entry - SL)
    - TP + R*SL >= (R+1) * entry
    - entry <= (TP + R*SL) / (R+1)

    수수료 포함 시 (fee_rate = total fees as fraction):
    - risk_net = (entry - SL) + entry * fee_rate
    - reward_net = (TP - entry) - entry * fee_rate
    - 이분탐색으로 정확한 entry 찾음

    Args:
        tp: Take Profit 가격
        sl: Stop Loss 가격
        target_rr: 목표 RR (예: 2.0)
        fee_bps: 수수료 bps (편도)
        slippage_bps: 슬리피지 bps (편도)
        side: "long" | "short"

    Returns:
        entry_limit 가격 (None이면 불가능)
    """
    if side == "long":
        if tp <= sl:
            return None  # Invalid: TP must be above SL for long

        # 수수료 비율 (왕복)
        fee_rate = (fee_bps + slippage_bps) * 2 / 10000

        # 수수료 무시 시 closed-form solution (상한)
        entry_max = (tp + target_rr * sl) / (target_rr + 1)

        if entry_max <= sl:
            return None  # 불가능: entry가 SL 아래

        # 수수료 포함 시 이분탐색
        def calc_rr_net(entry: float) -> float:
            risk = entry - sl
            reward = tp - entry
            fees = entry * fee_rate
            risk_net = risk + fees
            reward_net = reward - fees
            if risk_net <= 0:
                return -999
            return reward_net / risk_net

        # 이분탐색: sl < entry < entry_max 범위에서 RR=target 되는 entry 찾기
        lo, hi = sl + 0.01, entry_max
        for _ in range(50):  # 충분한 정밀도
            mid = (lo + hi) / 2
            rr = calc_rr_net(mid)
            if rr >= target_rr:
                lo = mid  # 더 높은 entry도 가능
            else:
                hi = mid  # entry를 낮춰야 함

        # 최종 검증
        entry_limit = lo
        if calc_rr_net(entry_limit) >= target_rr:
            return entry_limit
        return None

    else:  # short
        if sl <= tp:
            return None  # Invalid: SL must be above TP for short

        fee_rate = (fee_bps + slippage_bps) * 2 / 10000

        # 수수료 무시 시 closed-form (하한)
        # RR = (entry - TP) / (SL - entry) >= R
        # entry - TP >= R * (SL - entry)
        # entry + R*entry >= TP + R*SL
        # entry >= (TP + R*SL) / (R+1)
        entry_min = (tp + target_rr * sl) / (target_rr + 1)

        if entry_min >= sl:
            return None

        def calc_rr_net(entry: float) -> float:
            risk = sl - entry
            reward = entry - tp
            fees = entry * fee_rate
            risk_net = risk + fees
            reward_net = reward - fees
            if risk_net <= 0:
                return -999
            return reward_net / risk_net

        # 이분탐색
        lo, hi = entry_min, sl - 0.01
        for _ in range(50):
            mid = (lo + hi) / 2
            rr = calc_rr_net(mid)
            if rr >= target_rr:
                hi = mid  # 더 낮은 entry도 가능
            else:
                lo = mid  # entry를 높여야 함

        entry_limit = hi
        if calc_rr_net(entry_limit) >= target_rr:
            return entry_limit
        return None


def calc_fib_based_sl(
    entry_price: float,
    trigger_fib: float,
    atr: float,
    fib_levels: List[float],
    side: str = "long",
    fallback_atr_mult: float = 1.5
) -> Tuple[float, float, float]:
    """
    Fib 구조 기반 SL 계산.

    Logic (LONG):
        1. entry_fib = entry_price 근처(이하)의 Fib 레벨
        2. prev_fib = entry_fib 바로 아래 Fib 레벨
        3. fib_gap = entry_fib - prev_fib
        4. buffer = min(ATR, fib_gap)  # fib_gap 넘어가면 다음 레벨 침범
        5. SL = prev_fib - buffer

    Args:
        entry_price: 진입 가격
        trigger_fib: 시그널 트리거 Fib 레벨 (참고용)
        atr: ATR 값
        fib_levels: 정렬된 Fib 레벨 리스트 (오름차순)
        side: "long" or "short"
        fallback_atr_mult: Fib 레벨 부족 시 폴백 ATR 배수

    Returns:
        (sl_price, buffer, fib_gap)
    """
    if not fib_levels or len(fib_levels) < 2:
        # Fallback: ATR 기반
        if side == "long":
            return entry_price - atr * fallback_atr_mult, atr, 0
        else:
            return entry_price + atr * fallback_atr_mult, atr, 0

    sorted_levels = sorted(fib_levels)

    if side == "long":
        # 1. entry_price 이하의 Fib 레벨들 찾기 (entry_fib 후보)
        # tolerance = 0.5% of entry (NOT ATR) - floating point handling
        tolerance = entry_price * 0.005
        levels_at_or_below = [f for f in sorted_levels if f <= entry_price + tolerance]

        if len(levels_at_or_below) < 2:
            # Fib 레벨 2개 미만이면 폴백
            fib_gap = atr * fallback_atr_mult
            buffer = min(atr, fib_gap)
            sl = entry_price - atr * fallback_atr_mult
            return sl, buffer, fib_gap

        # 2. entry_fib = entry_price에 가장 가까운 Fib (이하)
        entry_fib = max(levels_at_or_below)

        # 3. prev_fib = entry_fib 바로 아래 Fib
        levels_below_entry_fib = [f for f in sorted_levels if f < entry_fib - 1e-6]
        if not levels_below_entry_fib:
            # entry_fib가 가장 낮은 Fib이면 폴백
            fib_gap = atr * fallback_atr_mult
            buffer = min(atr, fib_gap)
            sl = entry_fib - buffer
            return sl, buffer, fib_gap

        prev_fib = max(levels_below_entry_fib)

        # 4. fib_gap = entry_fib - prev_fib
        fib_gap = entry_fib - prev_fib

        # 5. buffer = min(ATR, fib_gap)
        buffer = min(atr, fib_gap) if fib_gap > 0 else atr * 0.5

        # 6. SL = prev_fib - buffer
        sl = prev_fib - buffer

        return sl, buffer, fib_gap

    else:  # short
        # 1. entry_price 이상의 Fib 레벨들 찾기 (entry_fib 후보)
        # tolerance = 0.5% of entry (NOT ATR) - floating point handling
        tolerance = entry_price * 0.005
        levels_at_or_above = [f for f in sorted_levels if f >= entry_price - tolerance]

        if len(levels_at_or_above) < 2:
            # Fib 레벨 2개 미만이면 폴백
            fib_gap = atr * fallback_atr_mult
            buffer = min(atr, fib_gap)
            sl = entry_price + atr * fallback_atr_mult
            return sl, buffer, fib_gap

        # 2. entry_fib = entry_price에 가장 가까운 Fib (이상)
        entry_fib = min(levels_at_or_above)

        # 3. prev_fib = entry_fib 바로 위 Fib
        levels_above_entry_fib = [f for f in sorted_levels if f > entry_fib + 1e-6]
        if not levels_above_entry_fib:
            # entry_fib가 가장 높은 Fib이면 폴백
            fib_gap = atr * fallback_atr_mult
            buffer = min(atr, fib_gap)
            sl = entry_fib + buffer
            return sl, buffer, fib_gap

        prev_fib = min(levels_above_entry_fib)

        # 4. fib_gap = prev_fib - entry_fib
        fib_gap = prev_fib - entry_fib

        # 5. buffer = min(ATR, fib_gap)
        buffer = min(atr, fib_gap) if fib_gap > 0 else atr * 0.5

        # 6. SL = prev_fib + buffer
        sl = prev_fib + buffer

        return sl, buffer, fib_gap


@dataclass
class LimitOrderResult:
    """Limit order 시뮬레이션 결과"""
    filled: bool
    fill_price: Optional[float]
    fill_ts: Optional[pd.Timestamp]
    fill_bar_idx: Optional[int]
    wait_bars: int
    reason: str  # "FILLED" | "TTL_EXPIRED" | "CANCELLED"


def simulate_limit_fill(
    df: pd.DataFrame,
    start_idx: int,
    entry_limit: float,
    ttl_bars: int,
    side: str = "long",
    fill_on: str = "low"
) -> LimitOrderResult:
    """
    Limit order 체결 시뮬레이션

    Args:
        df: OHLCV DataFrame (low, high, close 컬럼 필요)
        start_idx: 주문 시작 bar index
        entry_limit: limit 가격
        ttl_bars: 주문 유효 bar 수
        side: "long" | "short"
        fill_on: "low" (bar low <= limit) | "close" (close <= limit)

    Returns:
        LimitOrderResult
    """
    end_idx = min(start_idx + ttl_bars, len(df))

    for i in range(start_idx, end_idx):
        bar = df.iloc[i]

        if side == "long":
            # Long: bar low가 limit 이하면 체결
            if fill_on == "low":
                if bar['low'] <= entry_limit:
                    # 체결가: limit 또는 open 중 낮은 값 (gap down 시)
                    fill_price = min(entry_limit, bar['open'])
                    return LimitOrderResult(
                        filled=True,
                        fill_price=fill_price,
                        fill_ts=bar.name if hasattr(bar, 'name') else df.index[i],
                        fill_bar_idx=i,
                        wait_bars=i - start_idx,
                        reason="FILLED"
                    )
            else:  # fill_on == "close"
                if bar['close'] <= entry_limit:
                    return LimitOrderResult(
                        filled=True,
                        fill_price=bar['close'],
                        fill_ts=bar.name if hasattr(bar, 'name') else df.index[i],
                        fill_bar_idx=i,
                        wait_bars=i - start_idx,
                        reason="FILLED"
                    )
        else:  # short
            # Short: bar high가 limit 이상이면 체결
            if fill_on == "low":  # actually "high" for short
                if bar['high'] >= entry_limit:
                    fill_price = max(entry_limit, bar['open'])
                    return LimitOrderResult(
                        filled=True,
                        fill_price=fill_price,
                        fill_ts=bar.name if hasattr(bar, 'name') else df.index[i],
                        fill_bar_idx=i,
                        wait_bars=i - start_idx,
                        reason="FILLED"
                    )
            else:
                if bar['close'] >= entry_limit:
                    return LimitOrderResult(
                        filled=True,
                        fill_price=bar['close'],
                        fill_ts=bar.name if hasattr(bar, 'name') else df.index[i],
                        fill_bar_idx=i,
                        wait_bars=i - start_idx,
                        reason="FILLED"
                    )

    # TTL 만료
    return LimitOrderResult(
        filled=False,
        fill_price=None,
        fill_ts=None,
        fill_bar_idx=None,
        wait_bars=ttl_bars,
        reason="TTL_EXPIRED"
    )


# =============================================================================
# 데이터 로딩
# =============================================================================
def load_data(tf: str, start_date: str, end_date: str, config: Config) -> pd.DataFrame:
    """데이터 로딩 및 인디케이터 계산"""
    data_dir = ROOT / "data" / "bronze" / "binance" / "futures" / "BTC-USDT" / tf

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = list(data_dir.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')

    df = df.sort_index()
    df = df[start_date:end_date]

    # 인디케이터 (talib 사용)
    df['rsi'] = calc_rsi_wilder(df['close'].values, period=config.rsi_period)
    df['stoch_d'] = calc_stoch_rsi(df['close'].values, period=config.stoch_rsi_period, k_period=3, d_period=3)
    df['atr'] = calc_atr(df['high'].values, df['low'].values, df['close'].values, period=config.atr_period)

    return df

# =============================================================================
# 트레이드 결과
# =============================================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    side: Literal['long', 'short'] = 'long'
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    exit_reason: str = ""
    div_type: str = ""
    tf: str = ""
    # PR4 분석용 필드
    bars_held: int = 0           # 보유 기간 (5m bars)
    mfe: float = 0.0             # Maximum Favorable Excursion ($)
    mae: float = 0.0             # Maximum Adverse Excursion ($)
    mfe_first_6: float = 0.0     # 첫 6 bars 동안 MFE ($)
    mae_first_6: float = 0.0     # 첫 6 bars 동안 MAE ($)
    entry_atr: float = 0.0       # 진입 시 ATR
    # PR6.3: 사이징 로깅 필드
    sl_distance_raw: float = 0.0   # 원래 SL 거리 ($)
    sl_distance_atr: float = 0.0   # SL 거리 / ATR 비율
    clamped: bool = False          # min_sl_distance 적용 여부
    notional: float = 0.0          # 포지션 명목가 ($)
    cap_reason: str = ""           # none/max_notional/legacy

@dataclass
class BacktestResult:
    strategy_name: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> Dict:
        if not self.trades:
            return {
                'strategy': self.strategy_name,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl_usd': 0.0,
                'total_pnl_pct': 0.0,
                'avg_pnl_usd': 0.0,
                'max_win_usd': 0.0,
                'max_loss_usd': 0.0,
                'final_equity': 10000.0,
            }

        wins = [t for t in self.trades if t.pnl_usd > 0]
        losses = [t for t in self.trades if t.pnl_usd <= 0]

        total_pnl = sum(t.pnl_usd for t in self.trades)

        return {
            'strategy': self.strategy_name,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0,
            'total_pnl_usd': total_pnl,
            'total_pnl_pct': total_pnl / 10000 * 100,  # 초기자산 대비 %
            'avg_pnl_usd': np.mean([t.pnl_usd for t in self.trades]),
            'max_win_usd': max(t.pnl_usd for t in self.trades),
            'max_loss_usd': min(t.pnl_usd for t in self.trades),
            'final_equity': self.equity_curve[-1] if self.equity_curve else 10000,
        }

# =============================================================================
# 전략 A: 15m 진입 + 5m 청산 (반등 확인 후 시장가 진입)
# =============================================================================
class StrategyA:
    """
    전략 A: 15m 진입 + 5m 청산
    - 진입: 15m RSI 다이버전스 존 터치 + 반등 확인 후 시장가 진입
    - SL: ATR 기반
    - TP: 5m 반대 다이버전스
    """

    def __init__(self, config: Config):
        self.config = config
        self.name = "Strategy A: 15m Entry + 5m Exit (Bounce Confirm)"

        # 동적 SL을 위한 CycleDynamics 초기화
        if config.use_dynamic_sl:
            self.cycle_dynamics = CycleDynamics(
                lookback=config.cycle_lookback,
                min_period=10,
                max_period=100,
                base_sl_mult=config.dynamic_sl_base,
                sl_phase_adj=config.dynamic_sl_adj,
                use_continuous=True
            )
        else:
            self.cycle_dynamics = None

    def run(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
            df_1h: pd.DataFrame = None, df_4h: pd.DataFrame = None,
            prob_gate_result: pd.DataFrame = None) -> BacktestResult:
        result = BacktestResult(strategy_name=self.name)
        equity = self.config.initial_capital
        result.equity_curve.append(equity)

        # 롱/숏 포지션 독립 관리
        long_position = None
        short_position = None

        # 대기 신호 (존 터치됨, 반등 대기)
        pending_long_signal = None   # {'zone_price', 'boundary', 'atr', 'touched_time'}
        pending_short_signal = None

        # 쿨다운 카운터 (손절 후 재진입 제한)
        long_cooldown = 0
        short_cooldown = 0

        # 15m 바 상태 전환 추적 (과매도/과매수 진입 순간 감지)
        last_15m_bar_time = None
        prev_15m_stoch = 50.0  # 중립 시작
        long_signal_triggered = False   # 현재 15m 바에서 롱 신호 발생 여부
        short_signal_triggered = False  # 현재 15m 바에서 숏 신호 발생 여부

        # PR4-R4a: 추세 지속형 진입 쿨다운 추적
        last_trend_cont_entry_bar_idx = -9999  # 마지막 추세 지속형 진입 바 인덱스

        # PR3.6: HOT 구간 판단용 T median 계산
        T_median = None
        if self.config.use_hot_early_exit and prob_gate_result is not None and 'T' in prob_gate_result.columns:
            T_median = prob_gate_result['T'].median()
            print(f"  [PR3.6] HOT Early Exit enabled | T_median={T_median:.4f}")

        # 추세 필터 통계
        trend_filter_rejects = {'1H_DOWNTREND': 0, '1H_UPTREND': 0, '4H_DOWNTREND': 0, '4H_UPTREND': 0}
        hilbert_filter_rejects = {'HILBERT_BEAR': 0, 'HILBERT_BULL': 0}
        prob_gate_rejects = {
            # FAIL-CLOSED (이것들이 있으면 Gate 무결성 검증 필요)
            'PROB_GATE_INDEX_MISSING': 0,       # Gate row 누락 - 인덱스 불일치
            'PROB_GATE_WARMUP': 0,              # Warmup 구간 - 데이터 불완전
            'PROB_GATE_INVALID_ROW': 0,         # NaN/Inf 값
            # Normal rejects
            'PROB_GATE_NOT_LONG': 0,
            'PROB_GATE_NOT_SHORT': 0,
            # PR4.4: 드리프트 기반 SHORT reject
            'PROB_GATE_NOT_SHORT_UPTREND': 0,   # 상승장 SHORT reject (thr=0.70)
            'PROB_GATE_NOT_SHORT_DOWNTREND': 0, # 하락장 SHORT reject (thr=0.62)
            'PROB_GATE_NOT_SHORT_RANGE': 0,     # 횡보 SHORT reject (thr=0.65)
            # PR4.2/4.3
            'PROB_GATE_SHORT_LOW_CONF': 0,      # PR4.2: conf < conf_min_short
            'PROB_GATE_SHORT_LOW_CONF_HOT': 0,  # PR4.2: HOT에서 conf 부족
            'PROB_GATE_SHORT_NO_MOMENTUM': 0,   # PR4.3: ret_n >= ret_min
            'PROB_GATE_SHORT_ABOVE_EMA': 0,     # PR4.3: close >= EMA
        }
        # PR4-R3: 레짐별 거래 허용/차단 통계
        regime_rejects = {
            'REGIME_LONG_UPTREND': 0,    # UPTREND에서 차단 (config.regime_long_uptrend=False)
            'REGIME_LONG_RANGE': 0,      # RANGE에서 차단 (config.regime_long_range=False)
            'REGIME_LONG_DOWNTREND': 0,  # DOWNTREND에서 차단 (config.regime_long_downtrend=False)
        }
        atr_vol_size_cuts = 0
        # PR4-R5: RR Gate 통계
        rr_gate_rejects = {
            'RR_GATE_LONG_LOW_RR': 0,   # LONG 진입 RR 부족
            'RR_GATE_SHORT_LOW_RR': 0,  # SHORT 진입 RR 부족
        }

        # PR-DYN-FIB: 동적 Fib 앵커 상태 초기화
        dynfib_state = None
        df_1w = None  # 1W ZigZag용 데이터
        if self.config.use_dynamic_fib:
            dynfib_state = create_initial_state(self.config.dynamic_fib_mode)
            print(f"  [PR-DYN-FIB] Dynamic Fib enabled | mode={self.config.dynamic_fib_mode}, "
                  f"tf={self.config.dynamic_fib_tf}, lookback={self.config.dynfib_lookback_bars}, use_as={self.config.dynfib_use_as}")
            # 1W 타임프레임 사용 시 1W 데이터 로드
            if self.config.dynamic_fib_tf == "1w":
                try:
                    start_date = df_15m.index.min().strftime("%Y-%m-%d")
                    end_date = df_15m.index.max().strftime("%Y-%m-%d")
                    df_1w = load_data("1w", start_date, end_date, self.config)
                    print(f"  [PR-DYN-FIB] Loaded {len(df_1w)} 1W bars for ZigZag")
                except Exception as e:
                    print(f"  [PR-DYN-FIB] WARNING: Failed to load 1W data: {e}")
                    df_1w = None

        # Hilbert 레짐 계산 (1H, causal)
        hilbert_regimes = None
        if (self.config.use_hilbert_filter or self.config.use_regime_hidden_strategy) and df_1h is not None:
            detrend_period = _duration_to_bars(
                self.config.wave_regime_detrend_duration, self.config.context_tf
            )
            classifier = WaveRegimeClassifier(
                detrend_period=detrend_period,
                hilbert_window=self.config.wave_regime_hilbert_window
            )
            hilbert_regimes = classifier.classify_series_causal(df_1h['close'])
            # 레짐 분포 출력
            if hilbert_regimes is not None and 'regime' in hilbert_regimes.columns:
                regime_counts = hilbert_regimes['regime'].value_counts()
                print(f"\n[Hilbert Regime 분포]")
                for reg, cnt in regime_counts.items():
                    print(f"  {reg}: {cnt} bars ({100*cnt/len(hilbert_regimes):.1f}%)")

        warmup = self.config.warmup_bars
        warmup_anchor = self.config.warmup_bars_anchor
        total_bars = len(df_5m) - warmup
        for idx, i in enumerate(range(warmup, len(df_5m))):
            if idx % 1000 == 0:
                print(f"  A Progress: {idx}/{total_bars} ({100*idx/total_bars:.1f}%)", flush=True)
            bar = df_5m.iloc[i]
            current_time = df_5m.index[i]

            # 쿨다운 감소
            if long_cooldown > 0:
                long_cooldown -= 1
            if short_cooldown > 0:
                short_cooldown -= 1

            # 15m 데이터 슬라이스
            mask_15m = df_15m.index <= current_time
            if mask_15m.sum() < warmup_anchor:
                continue
            df_15m_slice = df_15m[mask_15m]
            close_arr_15m = df_15m_slice['close'].values.astype(float)

            # 15m 바 전환 감지
            current_15m_bar_time = df_15m_slice.index[-1]
            if current_15m_bar_time != last_15m_bar_time:
                # 새 15m 바 진입 - 상태 전환 체크
                current_stoch = df_15m_slice['stoch_d'].iloc[-1] if len(df_15m_slice) > 0 else 50.0
                if not np.isfinite(current_stoch):
                    current_stoch = 50.0

                # 과매도 진입 순간: 이전 > threshold → 현재 ≤ threshold
                oversold_thr = self.config.stoch_rsi_oversold
                overbought_thr = self.config.stoch_rsi_overbought
                long_signal_triggered = (prev_15m_stoch > oversold_thr and current_stoch <= oversold_thr)
                # 과매수 진입 순간: 이전 < threshold → 현재 ≥ threshold
                short_signal_triggered = (prev_15m_stoch < overbought_thr and current_stoch >= overbought_thr)

                prev_15m_stoch = current_stoch
                last_15m_bar_time = current_15m_bar_time
            else:
                # 동일 15m 바 내 후속 5m 바 - 신호 비활성화 (한 번만 체크)
                long_signal_triggered = False
                short_signal_triggered = False

            # 5m 데이터 슬라이스
            df_5m_slice = df_5m.iloc[:i+1]
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # === PR-A: ATR TF for Risk ===
            # atr_tf_for_risk에 따라 5m 또는 15m ATR 사용
            if self.config.atr_tf_for_risk == '15m':
                # 15m ATR 사용 (df_15m_slice는 이미 위에서 생성됨)
                atr_15m = df_15m_slice['atr'].iloc[-1] if 'atr' in df_15m_slice.columns and len(df_15m_slice) > 0 else None
                current_atr = atr_15m if atr_15m is not None and np.isfinite(atr_15m) else 500
            else:
                # 5m ATR 사용 (기존 동작, 회귀 테스트용)
                current_atr = bar['atr'] if 'atr' in bar and np.isfinite(bar['atr']) else 500

            # === PR-DYN-FIB: 동적 Fib 앵커 갱신 ===
            if self.config.use_dynamic_fib and dynfib_state is not None:
                # 1W ZigZag 모드
                if self.config.dynamic_fib_tf == "1w" and df_1w is not None:
                    mask_1w = df_1w.index <= current_time
                    if mask_1w.sum() >= 2:  # 최소 2개 1W 바 필요
                        df_1w_slice = df_1w[mask_1w]
                        i_1w = len(df_1w_slice) - 1
                        # 1W ATR 계산 (더 큰 스윙에 맞게)
                        atr_1w = df_1w_slice['atr'].iloc[-1] if 'atr' in df_1w_slice.columns else current_atr * 10
                        if self.config.dynamic_fib_mode == "zigzag":
                            dynfib_state = update_anchor_zigzag(
                                df_1w_slice, i_1w, dynfib_state, atr_1w,
                                self.config.dynfib_reversal_atr_mult
                            )
                        elif self.config.dynamic_fib_mode == "rolling":
                            low, high = update_anchor_rolling(
                                df_1w_slice, i_1w, self.config.dynfib_lookback_bars
                            )
                            dynfib_state.low = low
                            dynfib_state.high = high
                            dynfib_state.last_update_ts = current_time
                # 기존 15m 모드
                elif len(df_15m_slice) > 0:
                    i_15m = len(df_15m_slice) - 1
                    if self.config.dynamic_fib_mode == "zigzag":
                        dynfib_state = update_anchor_zigzag(
                            df_15m_slice, i_15m, dynfib_state, current_atr,
                            self.config.dynfib_reversal_atr_mult
                        )
                    elif self.config.dynamic_fib_mode == "rolling":
                        low, high = update_anchor_rolling(
                            df_15m_slice, i_15m, self.config.dynfib_lookback_bars
                        )
                        dynfib_state.low = low
                        dynfib_state.high = high
                        dynfib_state.last_update_ts = current_time
                    elif self.config.dynamic_fib_mode == "conditional":
                        low, high, updated = update_anchor_conditional(
                            df_15m_slice, i_15m, self.config.dynfib_lookback_bars,
                            current_atr, self.config.dynfib_min_swing_atr_mult, dynfib_state
                        )
                        if updated:
                            dynfib_state.low = low
                            dynfib_state.high = high
                            dynfib_state.last_update_ts = current_time

            # ===== 포지션 청산 체크 =====
            # === PR4 분석용: MFE/MAE/bars_held 추적 (항상 실행) ===
            if long_position is not None:
                long_position['mfe'] = max(long_position.get('mfe', long_position['entry_price']), bar['high'])
                long_position['mae'] = min(long_position.get('mae', long_position['entry_price']), bar['low'])
                bars_held = i - long_position.get('entry_bar_idx', i)
                long_position['bars_held'] = bars_held
                if bars_held <= 6:
                    long_position['mfe_first_6'] = max(long_position.get('mfe_first_6', long_position['entry_price']), bar['high'])
                    long_position['mae_first_6'] = min(long_position.get('mae_first_6', long_position['entry_price']), bar['low'])

                # === PR4-R4b: MFE 기반 브레이크이븐 + 부분익절 ===
                if self.config.use_breakeven and not long_position.get('be_triggered', False):
                    entry_atr = long_position.get('atr', current_atr)
                    mfe_move = long_position['mfe'] - long_position['entry_price']
                    be_threshold = self.config.be_mfe_atr * entry_atr

                    if mfe_move >= be_threshold:
                        # 1) 부분익절 (be_partial_pct%)
                        partial_exit_price = bar['high']  # MFE 도달 시점 가격
                        partial_ratio = self.config.be_partial_pct
                        trade = self._close_position_partial(
                            long_position, partial_exit_price, 'BE_Partial', current_time, partial_ratio
                        )
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)

                        # 2) SL을 entry + buffer로 이동
                        new_sl = long_position['entry_price'] + (self.config.be_buffer_atr * entry_atr)
                        long_position['sl'] = new_sl
                        long_position['be_triggered'] = True
                        long_position['remaining'] = 1.0 - partial_ratio

                        print(f"    [BE TRIGGERED] MFE ${mfe_move:.0f} >= {self.config.be_mfe_atr} ATR (${be_threshold:.0f})")
                        print(f"      Partial exit: {partial_ratio*100:.0f}% @ ${partial_exit_price:,.0f}")
                        print(f"      SL moved: ${long_position['entry_price']:,.0f} → ${new_sl:,.0f}")

                # === PR3.7: Trailing Stop (Long) ===
                if self.config.use_trailing_stop:
                    entry_atr = long_position.get('atr', current_atr)
                    mfe_move = long_position['mfe'] - long_position['entry_price']
                    activation_threshold = self.config.trailing_activation_atr * entry_atr

                    if mfe_move >= activation_threshold:
                        # 트레일링 활성화: 고점에서 N ATR 뒤에서 추적
                        trailing_sl = long_position['mfe'] - (self.config.trailing_distance_atr * entry_atr)

                        # 트레일링 SL은 기존 SL보다 높을 때만 업데이트 (Long)
                        if trailing_sl > long_position['sl']:
                            old_sl = long_position['sl']
                            long_position['sl'] = trailing_sl
                            if not long_position.get('trailing_activated', False):
                                long_position['trailing_activated'] = True
                                print(f"    [TRAILING ACTIVATED] MFE ${mfe_move:.0f} >= {self.config.trailing_activation_atr} ATR")
                            print(f"    [TRAILING SL] ${old_sl:,.0f} → ${trailing_sl:,.0f} (MFE: ${long_position['mfe']:,.0f})")

            if short_position is not None:
                short_position['mfe'] = min(short_position.get('mfe', short_position['entry_price']), bar['low'])
                short_position['mae'] = max(short_position.get('mae', short_position['entry_price']), bar['high'])
                bars_held = i - short_position.get('entry_bar_idx', i)
                short_position['bars_held'] = bars_held
                if bars_held <= 6:
                    short_position['mfe_first_6'] = min(short_position.get('mfe_first_6', short_position['entry_price']), bar['low'])
                    short_position['mae_first_6'] = max(short_position.get('mae_first_6', short_position['entry_price']), bar['high'])

                # === PR4-R4b: MFE 기반 브레이크이븐 + 부분익절 (Short) ===
                if self.config.use_breakeven and not short_position.get('be_triggered', False):
                    entry_atr = short_position.get('atr', current_atr)
                    mfe_move = short_position['entry_price'] - short_position['mfe']  # Short는 반대
                    be_threshold = self.config.be_mfe_atr * entry_atr

                    if mfe_move >= be_threshold:
                        # 1) 부분익절
                        partial_exit_price = bar['low']
                        partial_ratio = self.config.be_partial_pct
                        trade = self._close_position_partial(
                            short_position, partial_exit_price, 'BE_Partial', current_time, partial_ratio
                        )
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)

                        # 2) SL을 entry - buffer로 이동 (Short)
                        new_sl = short_position['entry_price'] - (self.config.be_buffer_atr * entry_atr)
                        short_position['sl'] = new_sl
                        short_position['be_triggered'] = True
                        short_position['remaining'] = 1.0 - partial_ratio

                        print(f"    [BE TRIGGERED SHORT] MFE ${mfe_move:.0f} >= {self.config.be_mfe_atr} ATR")
                        print(f"      Partial exit: {partial_ratio*100:.0f}% @ ${partial_exit_price:,.0f}")
                        print(f"      SL moved: ${short_position['entry_price']:,.0f} → ${new_sl:,.0f}")

                # === PR3.7: Trailing Stop (Short) ===
                if self.config.use_trailing_stop:
                    entry_atr = short_position.get('atr', current_atr)
                    mfe_move = short_position['entry_price'] - short_position['mfe']  # Short는 반대
                    activation_threshold = self.config.trailing_activation_atr * entry_atr

                    if mfe_move >= activation_threshold:
                        # 트레일링 활성화: 저점에서 N ATR 위에서 추적
                        trailing_sl = short_position['mfe'] + (self.config.trailing_distance_atr * entry_atr)

                        # 트레일링 SL은 기존 SL보다 낮을 때만 업데이트 (Short)
                        if trailing_sl < short_position['sl']:
                            old_sl = short_position['sl']
                            short_position['sl'] = trailing_sl
                            if not short_position.get('trailing_activated', False):
                                short_position['trailing_activated'] = True
                                print(f"    [TRAILING ACTIVATED SHORT] MFE ${mfe_move:.0f} >= {self.config.trailing_activation_atr} ATR")
                            print(f"    [TRAILING SL SHORT] ${old_sl:,.0f} → ${trailing_sl:,.0f} (MFE: ${short_position['mfe']:,.0f})")

            # === Early Exit (PR3.5): SL 전 조기 청산 ===
            if self.config.use_early_exit:
                # Long Early Exit
                if long_position is not None:
                    bars_held = long_position['bars_held']
                    entry_atr = long_position.get('atr', current_atr)
                    mfe_move = long_position['mfe'] - long_position['entry_price']
                    current_pnl = bar['close'] - long_position['entry_price']  # 현재 손익

                    early_exit_triggered = False
                    early_exit_reason = None

                    # PR3.6: HOT 구간 판단 (T > median)
                    is_hot = False
                    if self.config.use_hot_early_exit and T_median is not None and prob_gate_result is not None:
                        if current_time in prob_gate_result.index:
                            current_T = prob_gate_result.loc[current_time, 'T']
                            is_hot = current_T > T_median

                    # 파라미터 선택 (HOT vs NORMAL)
                    time_bars = self.config.early_exit_time_bars_hot if is_hot else self.config.early_exit_time_bars
                    mfe_mult = self.config.early_exit_mfe_mult_hot if is_hot else self.config.early_exit_mfe_mult
                    stale_mult = self.config.stale_loss_mult_hot if is_hot else 0.3

                    # 1a) TimeStop: N bars 후 MFE < threshold*ATR이면 조기 청산
                    if bars_held >= time_bars:
                        if mfe_move < mfe_mult * entry_atr:
                            early_exit_triggered = True
                            early_exit_reason = 'TimeStop_HOT' if is_hot else 'TimeStop'

                    # 1b) StaleLoss: N bars 후 유의미한 손실이면 조기 청산
                    if not early_exit_triggered and bars_held >= time_bars:
                        if current_pnl < -stale_mult * entry_atr:
                            early_exit_triggered = True
                            early_exit_reason = 'StaleLoss_HOT' if is_hot else 'StaleLoss'

                    # 2) GateFlip Exit: ProbGate 방향 반전 시 청산 (손실 상태에서만)
                    if self.config.use_gate_flip_exit and prob_gate_result is not None and current_pnl < 0 and not early_exit_triggered:
                        if i < len(prob_gate_result) and prob_gate_result.iloc[i]['action_code'] == -1:  # SHORT 전환
                            early_exit_triggered = True
                            early_exit_reason = 'GateFlip'

                    # 3) Opposite Div Early (손실 상태에서만): 반대 Div 시 SL 전 조기 탈출
                    if self.config.use_opposite_div_early and current_pnl < 0 and not early_exit_triggered:
                        if self._check_short_divergence(df_5m_slice):
                            early_exit_triggered = True
                            early_exit_reason = 'EarlyDiv'

                    if early_exit_triggered:
                        exit_price = bar['close']
                        remaining = long_position.get('remaining', 1.0)
                        trade = self._close_position_partial(long_position, exit_price, early_exit_reason, current_time, remaining)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        print(f"    [EARLY EXIT] {early_exit_reason} | bars={bars_held} | MFE=${mfe_move:.0f} | PnL=${trade.pnl_usd:.2f}")
                        long_position = None

                # Short Early Exit
                if short_position is not None:
                    bars_held = short_position['bars_held']
                    entry_atr = short_position.get('atr', current_atr)
                    mfe_move = short_position['entry_price'] - short_position['mfe']
                    current_pnl = short_position['entry_price'] - bar['close']  # 현재 손익

                    early_exit_triggered = False
                    early_exit_reason = None

                    # PR3.6: HOT 구간 판단 (T > median)
                    is_hot = False
                    if self.config.use_hot_early_exit and T_median is not None and prob_gate_result is not None:
                        if current_time in prob_gate_result.index:
                            current_T = prob_gate_result.loc[current_time, 'T']
                            is_hot = current_T > T_median

                    # 파라미터 선택 (HOT vs NORMAL)
                    time_bars = self.config.early_exit_time_bars_hot if is_hot else self.config.early_exit_time_bars
                    mfe_mult = self.config.early_exit_mfe_mult_hot if is_hot else self.config.early_exit_mfe_mult
                    stale_mult = self.config.stale_loss_mult_hot if is_hot else 0.3

                    # 1a) TimeStop
                    if bars_held >= time_bars:
                        if mfe_move < mfe_mult * entry_atr:
                            early_exit_triggered = True
                            early_exit_reason = 'TimeStop_HOT' if is_hot else 'TimeStop'

                    # 1b) StaleLoss: N bars 후 유의미한 손실이면 조기 청산
                    if not early_exit_triggered and bars_held >= time_bars:
                        if current_pnl < -stale_mult * entry_atr:
                            early_exit_triggered = True
                            early_exit_reason = 'StaleLoss_HOT' if is_hot else 'StaleLoss'

                    # 2) GateFlip Exit (손실 상태에서만)
                    if self.config.use_gate_flip_exit and prob_gate_result is not None and current_pnl < 0 and not early_exit_triggered:
                        if i < len(prob_gate_result) and prob_gate_result.iloc[i]['action_code'] == 1:  # LONG 전환
                            early_exit_triggered = True
                            early_exit_reason = 'GateFlip'

                    # 3) Opposite Div Early (손실 상태에서만)
                    if self.config.use_opposite_div_early and current_pnl < 0 and not early_exit_triggered:
                        if self._check_long_divergence(df_5m_slice):
                            early_exit_triggered = True
                            early_exit_reason = 'EarlyDiv'

                    if early_exit_triggered:
                        exit_price = bar['close']
                        remaining = short_position.get('remaining', 1.0)
                        trade = self._close_position_partial(short_position, exit_price, early_exit_reason, current_time, remaining)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        print(f"    [EARLY EXIT] {early_exit_reason} | bars={bars_held} | MFE=${mfe_move:.0f} | PnL=${trade.pnl_usd:.2f}")
                        short_position = None

            # Long 청산: SL → TP1 (50%) → TP2 (30%) → TP3 (20%) → 5m 다이버전스
            if long_position is not None:
                # SL 체크 (최우선) - PR4-R6: LIQ 모드면 'LIQ' 라벨
                if bar['low'] <= long_position['sl']:
                    exit_price = long_position['sl']
                    remaining = long_position.get('remaining', 1.0)
                    # Exit reason: LIQ > SL_BE > SL
                    if long_position.get('is_liq_mode', False):
                        exit_reason = 'LIQ'
                    elif long_position.get('be_triggered', False):
                        exit_reason = 'SL_BE'
                    else:
                        exit_reason = 'SL'
                    trade = self._close_position_partial(long_position, exit_price, exit_reason, current_time, remaining)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    long_cooldown = self.config.cooldown_bars
                    long_position = None
                # TP1 체크 (50% 청산 + SL→Breakeven)
                elif not long_position.get('tp1_hit', False) and bar['high'] >= long_position['tp1']:
                    exit_price = long_position['tp1']
                    # PR6.2: 설정된 비율로 부분 청산
                    tp1_ratio = self.config.tp_split_ratios[0] if self.config.use_tp_split else 0.5
                    trade = self._close_position_partial(long_position, exit_price, 'TP1', current_time, tp1_ratio)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    # SL을 Breakeven으로 이동, TP1 히트 플래그 설정
                    long_position['sl'] = long_position['entry_price']
                    long_position['tp1_hit'] = True
                    long_position['remaining'] = 1.0 - tp1_ratio
                    print(f"    [TP1 HIT] ${exit_price:,.0f} | SL→BE: ${long_position['sl']:,.0f} | Closed: {tp1_ratio*100:.0f}%")
                # TP2 체크
                elif long_position.get('tp1_hit', False) and not long_position.get('tp2_hit', False) and bar['high'] >= long_position['tp2']:
                    exit_price = long_position['tp2']
                    if self.config.use_tp_split:
                        # PR6.2: 3단계 분할 - TP2는 30%
                        tp2_ratio = self.config.tp_split_ratios[1]
                        trade = self._close_position_partial(long_position, exit_price, 'TP2', current_time, tp2_ratio)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        long_position['tp2_hit'] = True
                        long_position['remaining'] = self.config.tp_split_ratios[2]  # 남은 20%
                        print(f"    [TP2 HIT] ${exit_price:,.0f} | Closed: {tp2_ratio*100:.0f}%")
                    else:
                        # Legacy: 2단계 - TP2에서 나머지 전량 청산
                        trade = self._close_position_partial(long_position, exit_price, 'TP2', current_time, long_position['remaining'])
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        print(f"    [TP2 HIT] ${exit_price:,.0f}")
                        long_position = None
                # TP3 체크 (PR6.2: 3단계 분할 시에만)
                elif self.config.use_tp_split and long_position.get('tp2_hit', False) and bar['high'] >= long_position.get('tp3', float('inf')):
                    exit_price = long_position['tp3']
                    tp3_ratio = self.config.tp_split_ratios[2]
                    trade = self._close_position_partial(long_position, exit_price, 'TP3', current_time, tp3_ratio)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    print(f"    [TP3 HIT] ${exit_price:,.0f} | Closed: {tp3_ratio*100:.0f}%")
                    long_position = None
                # 5m 숏 다이버전스 → 부분청산+트레일링 / 트레일링 / 즉시 청산
                elif self.config.use_5m_div_exit and self._check_short_divergence(df_5m_slice):
                    if self.config.use_5m_div_partial_trailing and not long_position.get('div_partial_done', False):
                        # 50% 청산 + 나머지 트레일링
                        exit_price = bar['close']
                        trade = self._close_position_partial(long_position, exit_price, '5m_Div_Partial', current_time, 0.5)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        # 나머지 50%에 트레일링 활성화
                        entry_atr = long_position.get('entry_atr', 300)
                        trailing_sl = bar['high'] - (self.config.trailing_distance_atr * entry_atr)
                        if trailing_sl > long_position['sl']:
                            long_position['sl'] = trailing_sl
                        long_position['trailing_activated'] = True
                        long_position['div_partial_done'] = True
                        print(f"    [5m DIV -> 50% CLOSE + TRAILING] Closed 50% at ${exit_price:,.0f}, Trailing SL=${long_position['sl']:,.0f}")
                    elif self.config.use_5m_div_trailing and not long_position.get('trailing_activated', False):
                        # 트레일링 즉시 활성화 (MFE 조건 무시)
                        entry_atr = long_position.get('entry_atr', 300)
                        trailing_sl = bar['high'] - (self.config.trailing_distance_atr * entry_atr)
                        if trailing_sl > long_position['sl']:
                            long_position['sl'] = trailing_sl
                        long_position['trailing_activated'] = True
                        long_position['div_trailing_triggered'] = True
                        print(f"    [5m DIV -> TRAILING] Activated at ${bar['high']:,.0f}, SL=${long_position['sl']:,.0f}")
                    elif not long_position.get('div_partial_done', False):
                        # 즉시 청산 (부분청산 이후가 아닐 때만)
                        exit_price = bar['close']
                        remaining = long_position.get('remaining', 1.0)
                        trade = self._close_position_partial(long_position, exit_price, '5m_Short_Div', current_time, remaining)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        long_position = None
                # 15m 숏 다이버전스 → 롱 청산 (use_15m_div_exit=True 일 때만)
                elif self.config.use_15m_div_exit and self._check_short_divergence(df_15m_slice):
                    exit_price = bar['close']
                    remaining = long_position.get('remaining', 1.0)
                    trade = self._close_position_partial(long_position, exit_price, '15m_Short_Div', current_time, remaining)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    long_position = None

            # Short 청산: SL → TP1 (50%) → TP2 (30%) → TP3 (20%) → 5m 다이버전스
            if short_position is not None:
                # SL 체크 (최우선) - PR4-R6: LIQ 모드면 'LIQ' 라벨
                if bar['high'] >= short_position['sl']:
                    exit_price = short_position['sl']
                    remaining = short_position.get('remaining', 1.0)
                    # Exit reason: LIQ > SL_BE > SL
                    if short_position.get('is_liq_mode', False):
                        exit_reason = 'LIQ'
                    elif short_position.get('be_triggered', False):
                        exit_reason = 'SL_BE'
                    else:
                        exit_reason = 'SL'
                    trade = self._close_position_partial(short_position, exit_price, exit_reason, current_time, remaining)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    short_cooldown = self.config.cooldown_bars
                    short_position = None
                # TP1 체크 (50% 청산 + SL→Breakeven)
                elif not short_position.get('tp1_hit', False) and bar['low'] <= short_position['tp1']:
                    exit_price = short_position['tp1']
                    # PR6.2: 설정된 비율로 부분 청산
                    tp1_ratio = self.config.tp_split_ratios[0] if self.config.use_tp_split else 0.5
                    trade = self._close_position_partial(short_position, exit_price, 'TP1', current_time, tp1_ratio)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    # SL을 Breakeven으로 이동, TP1 히트 플래그 설정
                    short_position['sl'] = short_position['entry_price']
                    short_position['tp1_hit'] = True
                    short_position['remaining'] = 1.0 - tp1_ratio
                    print(f"    [TP1 HIT] ${exit_price:,.0f} | SL→BE: ${short_position['sl']:,.0f} | Closed: {tp1_ratio*100:.0f}%")
                # TP2 체크
                elif short_position.get('tp1_hit', False) and not short_position.get('tp2_hit', False) and bar['low'] <= short_position['tp2']:
                    exit_price = short_position['tp2']
                    if self.config.use_tp_split:
                        # PR6.2: 3단계 분할 - TP2는 30%
                        tp2_ratio = self.config.tp_split_ratios[1]
                        trade = self._close_position_partial(short_position, exit_price, 'TP2', current_time, tp2_ratio)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        short_position['tp2_hit'] = True
                        short_position['remaining'] = self.config.tp_split_ratios[2]  # 남은 20%
                        print(f"    [TP2 HIT] ${exit_price:,.0f} | Closed: {tp2_ratio*100:.0f}%")
                    else:
                        # Legacy: 2단계 - TP2에서 나머지 전량 청산
                        trade = self._close_position_partial(short_position, exit_price, 'TP2', current_time, short_position['remaining'])
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        print(f"    [TP2 HIT] ${exit_price:,.0f}")
                        short_position = None
                # TP3 체크 (PR6.2: 3단계 분할 시에만)
                elif self.config.use_tp_split and short_position.get('tp2_hit', False) and bar['low'] <= short_position.get('tp3', float('-inf')):
                    exit_price = short_position['tp3']
                    tp3_ratio = self.config.tp_split_ratios[2]
                    trade = self._close_position_partial(short_position, exit_price, 'TP3', current_time, tp3_ratio)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    print(f"    [TP3 HIT] ${exit_price:,.0f} | Closed: {tp3_ratio*100:.0f}%")
                    short_position = None
                # 5m 롱 다이버전스 → 트레일링 활성화 또는 숏 청산
                elif self.config.use_5m_div_exit and self._check_long_divergence(df_5m_slice):
                    if self.config.use_5m_div_trailing and not short_position.get('trailing_activated', False):
                        # 트레일링 즉시 활성화 (MFE 조건 무시)
                        entry_atr = short_position.get('entry_atr', 300)
                        trailing_sl = bar['low'] + (self.config.trailing_distance_atr * entry_atr)
                        if trailing_sl < short_position['sl']:
                            short_position['sl'] = trailing_sl
                        short_position['trailing_activated'] = True
                        short_position['div_trailing_triggered'] = True
                        print(f"    [5m DIV -> TRAILING SHORT] Activated at ${bar['low']:,.0f}, SL=${short_position['sl']:,.0f}")
                    else:
                        # 즉시 청산
                        exit_price = bar['close']
                        remaining = short_position.get('remaining', 1.0)
                        trade = self._close_position_partial(short_position, exit_price, '5m_Long_Div', current_time, remaining)
                        result.trades.append(trade)
                        equity += trade.pnl_usd
                        result.equity_curve.append(equity)
                        short_position = None
                # 15m 롱 다이버전스 → 숏 청산 (use_15m_div_exit=True 일 때만)
                elif self.config.use_15m_div_exit and self._check_long_divergence(df_15m_slice):
                    exit_price = bar['close']
                    remaining = short_position.get('remaining', 1.0)
                    trade = self._close_position_partial(short_position, exit_price, '15m_Long_Div', current_time, remaining)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    short_position = None

            # ===== 반등 확인 → 진입 =====
            # Long 반등 확인
            if pending_long_signal is not None and long_position is None:
                # === PR-ENTRY-RR2: RR Limit Entry 모드 ===
                if self.config.use_rr_limit_entry:
                    # TTL 계산 (duration → bars)
                    ttl_bars = _duration_to_bars(self.config.rr_limit_ttl, self.config.anchor_tf)
                    bars_since_touch = (current_time - pending_long_signal['touched_time']).total_seconds() / (
                        {'5m': 300, '15m': 900, '1h': 3600, '4h': 14400}.get(self.config.anchor_tf, 900)
                    )

                    if bars_since_touch > ttl_bars:
                        print(f"  [LONG LIMIT EXPIRED] {current_time} - TTL {ttl_bars} bars exceeded")
                        pending_long_signal = None  # 주문 만료
                    else:
                        # Limit fill 조건 체크 전에 entry_limit 계산 (아직 안했으면)
                        if 'entry_limit' not in pending_long_signal:
                            # SL/TP 미리 계산
                            signal_atr = pending_long_signal['atr']
                            signal_price = pending_long_signal['zone_price']
                            sl_mult = self.config.sl_atr_mult
                            temp_sl = signal_price - (signal_atr * sl_mult)

                            # TP 후보 계산 (fib_rr 방식)
                            fib_candidates = []
                            curr_p = signal_price
                            for _ in range(self.config.fib_tp_candidates + 1):
                                fib = get_next_l1_above(curr_p, self.config)
                                if fib:
                                    fib_candidates.append(fib.price)
                                    curr_p = fib.price + 1
                                else:
                                    break
                            fib_candidates = fib_candidates[1:] if len(fib_candidates) > 1 else fib_candidates
                            temp_tp = fib_candidates[0] if fib_candidates else signal_price + (signal_atr * 3.0)

                            # PR-FIB-ENTRY: Entry = div_price - (TP거리 × 0.5)
                            # TP까지 거리의 30% 아래에서 진입 → RR 2.3:1 자동 보장
                            tp_distance = temp_tp - signal_price
                            if tp_distance <= 0:
                                print(f"  [LONG LIMIT FAIL] {current_time} - TP <= signal_price")
                                pending_long_signal = None
                                continue

                            entry_limit = signal_price - (tp_distance * 0.3)

                            # SL도 entry 기준으로 재계산
                            temp_sl = entry_limit - (signal_atr * sl_mult)

                            # 현재가에서 너무 멀면 skip
                            dist_atr = (bar['close'] - entry_limit) / signal_atr
                            if dist_atr > self.config.rr_limit_max_atr_dist:
                                print(f"  [LONG LIMIT SKIP] {current_time} - Entry limit ${entry_limit:,.0f} is {dist_atr:.1f} ATR below (max={self.config.rr_limit_max_atr_dist})")
                                pending_long_signal = None
                                continue

                            # pending_signal에 limit 정보 저장
                            pending_long_signal['entry_limit'] = entry_limit
                            pending_long_signal['pre_sl'] = temp_sl
                            pending_long_signal['pre_tp'] = temp_tp
                            print(f"  [LONG LIMIT ORDER] {current_time} - Entry=${entry_limit:,.0f}, SL=${temp_sl:,.0f}, TP=${temp_tp:,.0f}")

                        # Limit fill 체크
                        entry_limit = pending_long_signal['entry_limit']
                        fill_on = self.config.rr_limit_fill_on

                        if fill_on == "low" and bar['low'] <= entry_limit:
                            # Fill! 진입 가격 = min(entry_limit, open)
                            fill_price = min(entry_limit, bar['open'])
                            pending_long_signal['fill_price'] = fill_price
                            is_limit_filled = True
                        elif fill_on == "close" and bar['close'] <= entry_limit:
                            pending_long_signal['fill_price'] = bar['close']
                            is_limit_filled = True
                        else:
                            is_limit_filled = False

                        if not is_limit_filled:
                            continue  # 다음 bar 기다림

                        # 이하 필터 체크 진행 (기존 로직)
                        is_bullish_candle = True  # limit fill이면 양봉 조건 스킵

                else:
                    # === 기존: 양봉 기반 진입 ===
                    is_bullish_candle = bar['close'] > bar['open']
                    # 신호 유효기간 체크 (2봉 이내)
                    bars_since_touch = (current_time - pending_long_signal['touched_time']).total_seconds() / 300
                    if bars_since_touch > 3:
                        pending_long_signal = None  # 신호 만료
                        continue

                if pending_long_signal is not None and (is_bullish_candle if not self.config.use_rr_limit_entry else True):
                    # === 추세 필터 체크 (Precomputed 컬럼 사용) ===
                    trend_1h = "UNKNOWN"
                    trend_4h = "UNKNOWN"

                    if df_1h is not None and 'trend' in df_1h.columns:
                        df_1h_valid = df_1h[df_1h.index <= current_time]
                        if len(df_1h_valid) > 0:
                            trend_1h = df_1h_valid['trend'].iloc[-1]

                    if df_4h is not None and 'trend' in df_4h.columns:
                        df_4h_valid = df_4h[df_4h.index <= current_time]
                        if len(df_4h_valid) > 0:
                            trend_4h = df_4h_valid['trend'].iloc[-1]

                    pass_filter, reject_reason = check_trend_filter(
                        'long', trend_1h, trend_4h,
                        self.config.use_trend_filter_1h,
                        self.config.use_trend_filter_4h
                    )

                    if not pass_filter:
                        trend_filter_rejects[reject_reason] += 1
                        print(f"  [LONG REJECTED] {current_time} - {reject_reason} (1H:{trend_1h}, 4H:{trend_4h})")
                        pending_long_signal = None
                        continue

                    # === Hilbert 레짐 필터 (context_tf, causal) ===
                    if self.config.use_hilbert_filter and hilbert_regimes is not None:
                        # 완료된 context_tf봉 기준 (lookahead 방지)
                        ts_ctx = _floor_by_tf(current_time, self.config.context_tf)
                        if ts_ctx in hilbert_regimes.index:
                            hilbert_regime = str(hilbert_regimes.loc[ts_ctx, 'regime'])
                        else:
                            mask = hilbert_regimes.index <= ts_ctx
                            if mask.any():
                                hilbert_regime = str(hilbert_regimes.loc[hilbert_regimes.index[mask][-1], 'regime'])
                            else:
                                hilbert_regime = 'RANGE'

                        pass_hilbert, hilbert_reason = check_hilbert_filter(
                            'long', hilbert_regime,
                            self.config.hilbert_block_long_on_bear,
                            self.config.hilbert_block_short_on_bull
                        )

                        if not pass_hilbert:
                            hilbert_filter_rejects[hilbert_reason] += 1
                            print(f"  [LONG REJECTED] {current_time} - {hilbert_reason} (Hilbert:{hilbert_regime})")
                            pending_long_signal = None
                            continue

                    # === ProbabilityGate v2 필터 (1H Hilbert → 5m, causal) ===
                    if self.config.use_prob_gate and prob_gate_result is not None:
                        pass_gate, gate_reason, p_bull = check_prob_gate_filter(
                            'long', current_time, prob_gate_result, self.config
                        )
                        if not pass_gate:
                            prob_gate_rejects[gate_reason] += 1
                            print(f"  [LONG REJECTED] {current_time} - {gate_reason} (p_bull={p_bull:.3f})")
                            pending_long_signal = None
                            continue

                    # === PR4-R3: Regime Trade Permission 필터 ===
                    # drift_regime 기반 LONG 허용/차단
                    # PR-B: use_regime_filter=False면 이 필터 전체 스킵
                    # 레짐 초기화 (fib_rr에서 RANGE 전용 RR 사용을 위해)
                    current_drift_regime = 'UNKNOWN'
                    if prob_gate_result is not None and 'drift_regime' in prob_gate_result.columns:
                        pg_mask = prob_gate_result.index <= current_time
                        if pg_mask.any():
                            current_drift_regime = prob_gate_result.loc[pg_mask, 'drift_regime'].iloc[-1]

                    if self.config.use_regime_filter and current_drift_regime != 'UNKNOWN':
                        # 레짐별 LONG 허용 체크
                        regime_allowed = True
                        if current_drift_regime == 'UPTREND' and not self.config.regime_long_uptrend:
                            regime_allowed = False
                            regime_rejects['REGIME_LONG_UPTREND'] += 1
                        elif current_drift_regime == 'RANGE' and not self.config.regime_long_range:
                            regime_allowed = False
                            regime_rejects['REGIME_LONG_RANGE'] += 1
                        elif current_drift_regime == 'DOWNTREND' and not self.config.regime_long_downtrend:
                            regime_allowed = False
                            regime_rejects['REGIME_LONG_DOWNTREND'] += 1

                        if not regime_allowed:
                            print(f"  [LONG REJECTED] {current_time} - REGIME_{current_drift_regime} blocked")
                            pending_long_signal = None
                            continue

                        # === PR-RANGE-1: RANGE 전용 정책 ===
                        # RANGE에서 진입 조건을 더 엄격하게
                        if current_drift_regime == 'RANGE' and self.config.use_range_policy:
                            range_policy_pass = True
                            range_reject_reason = None

                            # (A) 레인지 폭 필터
                            if len(df_15m_slice) >= 50:
                                lookback = min(200, len(df_15m_slice))
                                range_high = df_15m_slice['high'].iloc[-lookback:].max()
                                range_low = df_15m_slice['low'].iloc[-lookback:].min()
                                range_width = range_high - range_low
                                current_atr_15m = df_15m_slice['atr'].iloc[-1] if 'atr' in df_15m_slice.columns else 500

                                if range_width < self.config.range_width_min_atr * current_atr_15m:
                                    range_policy_pass = False
                                    range_reject_reason = f"RANGE_WIDTH_NARROW ({range_width:.0f} < {self.config.range_width_min_atr * current_atr_15m:.0f})"

                                # (B) 극단 진입 강제: 레인지 하단 구간에서만 LONG
                                if range_policy_pass:
                                    current_price = bar['close']
                                    entry_zone_threshold = range_low + (range_width * self.config.range_entry_zone_pct)
                                    if current_price > entry_zone_threshold:
                                        range_policy_pass = False
                                        range_reject_reason = f"RANGE_NOT_AT_BOTTOM (price {current_price:.0f} > threshold {entry_zone_threshold:.0f})"

                            # (D) RANGE 전용 쿨다운
                            if range_policy_pass and hasattr(self, 'last_range_entry_bar_idx'):
                                bars_since = i - self.last_range_entry_bar_idx
                                if bars_since < self.config.range_cooldown_bars:
                                    range_policy_pass = False
                                    range_reject_reason = f"RANGE_COOLDOWN ({bars_since}/{self.config.range_cooldown_bars})"

                            if not range_policy_pass:
                                range_policy_rejects = getattr(self, 'range_policy_rejects', {})
                                range_policy_rejects[range_reject_reason.split('(')[0].strip()] = range_policy_rejects.get(range_reject_reason.split('(')[0].strip(), 0) + 1
                                self.range_policy_rejects = range_policy_rejects
                                print(f"  [LONG REJECTED] {current_time} - {range_reject_reason}")
                                pending_long_signal = None
                                continue

                    # === ATR 변동성 필터 (Precomputed 컬럼 사용) ===
                    size_mult = 1.0
                    if self.config.use_atr_vol_filter and df_1h is not None and 'atr_pct' in df_1h.columns:
                        df_1h_valid = df_1h[df_1h.index <= current_time]
                        if len(df_1h_valid) > 0:
                            atr_pct = df_1h_valid['atr_pct'].iloc[-1]
                            if atr_pct > self.config.atr_vol_threshold:
                                size_mult = self.config.atr_vol_size_mult
                                atr_vol_size_cuts += 1

                    # === Zone Depth 사이징 (필터→사이징 전환) ===
                    # 더 이상 진입 금지하지 않음. 대신 depth에 비례해 포지션 크기 조절
                    zone_depth = 0.5  # 기본값
                    if self.config.use_zone_depth_filter:
                        current_idx = len(df_5m_slice) - 1
                        zone_depth = calc_zone_depth(
                            close_arr_5m, current_idx, 'long',
                            lookback=self.config.zone_depth_lookback
                        )
                        depth_size_mult = calc_zone_depth_size_mult(zone_depth, self.config.zone_depth_min)
                        # 필터 없음: 항상 진입, 크기만 조절 (0.25 ~ 1.0)
                        size_mult *= depth_size_mult

                    # 반등 확인! 진입
                    # PR-ENTRY-RR2: limit entry면 fill_price 사용
                    if self.config.use_rr_limit_entry and 'fill_price' in pending_long_signal:
                        entry_price = pending_long_signal['fill_price']
                        atr = pending_long_signal['atr']

                        # PR-FIB-SL: Fib 구조 기반 SL
                        if self.config.use_fib_based_sl and 'fib_level' in pending_long_signal:
                            trigger_fib = pending_long_signal['fib_level'].price
                            # Dynamic Fib 레벨 수집
                            fib_levels = []
                            if self.config.use_dynamic_fib and dynfib_state is not None and dynfib_state.is_valid():
                                fib_levels = get_dynamic_fib_levels(
                                    dynfib_state.low, dynfib_state.high,
                                    self.config.dynfib_ratios,
                                    space=self.config.dynamic_fib_space
                                )
                            sl, sl_buffer, sl_fib_gap = calc_fib_based_sl(
                                entry_price, trigger_fib, atr, fib_levels,
                                side="long", fallback_atr_mult=self.config.fib_sl_fallback_mult
                            )
                            print(f"  [LONG LIMIT FILLED] Entry=${entry_price:,.0f} | SL=${sl:,.0f} (Fib-based, buffer=${sl_buffer:.0f}, gap=${sl_fib_gap:.0f})")
                        else:
                            # BUG FIX: SL을 fill_price 기준으로 재계산 (pre_sl은 signal_price 기준이라 틀림)
                            sl = entry_price - (atr * self.config.sl_atr_mult)
                            print(f"  [LONG LIMIT FILLED] Entry=${entry_price:,.0f} (limit order)")
                    else:
                        entry_price = bar['close']
                        atr = pending_long_signal['atr']

                        # PR-FIB-SL: Fib 구조 기반 SL
                        if self.config.use_fib_based_sl and 'fib_level' in pending_long_signal:
                            trigger_fib = pending_long_signal['fib_level'].price
                            # Dynamic Fib 레벨 수집
                            fib_levels = []
                            if self.config.use_dynamic_fib and dynfib_state is not None and dynfib_state.is_valid():
                                fib_levels = get_dynamic_fib_levels(
                                    dynfib_state.low, dynfib_state.high,
                                    self.config.dynfib_ratios,
                                    space=self.config.dynamic_fib_space
                                )
                            sl, sl_buffer, sl_fib_gap = calc_fib_based_sl(
                                entry_price, trigger_fib, atr, fib_levels,
                                side="long", fallback_atr_mult=self.config.fib_sl_fallback_mult
                            )
                            print(f"  [LONG ENTRY] Entry=${entry_price:,.0f} | SL=${sl:,.0f} (Fib-based, buffer=${sl_buffer:.0f}, gap=${sl_fib_gap:.0f})")
                        else:
                            # 동적 SL 배수 계산
                            if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                                cycle_result = self.cycle_dynamics.analyze(close_arr_15m)
                                sl_mult = cycle_result['dynamic_sl_mult']
                            else:
                                sl_mult = self.config.sl_atr_mult

                            # ATR 기반 SL
                            sl = entry_price - (atr * sl_mult)

                    # PR4-R0: TP 모드에 따라 계산
                    fib_rr_rejected = False  # fib_rr 모드에서 RR 기준 미달 시 True
                    if self.config.tp_mode == "atr":
                        # ATR 배수 기반 TP (RR 안정화)
                        tp_mults = self.config.tp_atr_mults
                        tp1 = entry_price + (atr * tp_mults[0])
                        tp2 = entry_price + (atr * tp_mults[1])
                        tp3 = entry_price + (atr * tp_mults[2])
                    elif self.config.tp_mode == "fib_rr":
                        # === PR-B: Fib + RR 필터 모드 ===
                        # Fib 후보 중 RR_net >= min_rr_net 만족하는 가장 가까운 TP 선택
                        # 없으면 진입 거부
                        num_candidates = self.config.fib_tp_candidates
                        fib_candidates = []
                        current_price = entry_price
                        for _ in range(num_candidates + 1):  # +1 for skipping first
                            fib = get_next_l1_above(current_price, self.config)
                            if fib:
                                fib_candidates.append(fib.price)
                                current_price = fib.price + 1
                            else:
                                break

                        # 첫번째 Fib 스킵 (너무 가까움)
                        fib_candidates = fib_candidates[1:] if len(fib_candidates) > 1 else fib_candidates

                        # === PR-DYN-FIB v2: 동적 Fib 레벨 추가 (LONG) ===
                        if self.config.use_dynamic_fib and dynfib_state is not None and dynfib_state.is_valid():
                            dyn_levels = get_dynamic_fib_levels(
                                dynfib_state.low, dynfib_state.high,
                                self.config.dynfib_ratios,
                                space=self.config.dynamic_fib_space
                            )
                            # entry_price 위의 레벨만 TP 후보로 추가
                            dyn_above = [p for p in dyn_levels if p > entry_price]
                            # DEBUG: dynfib 상태 출력
                            if hasattr(self.config, '_dynfib_debug') and self.config._dynfib_debug:
                                print(f"  [DYNFIB-LONG] entry={entry_price:.2f}, anchor=({dynfib_state.low:.2f}, {dynfib_state.high:.2f})")
                                print(f"    dyn_levels={len(dyn_levels)}, dyn_above={len(dyn_above)}, macro={len(fib_candidates)}")
                            if self.config.dynfib_use_as == "tp_candidate":
                                # Macro Fib + Dynamic Fib 합침
                                fib_candidates = sorted(set(fib_candidates + dyn_above))

                        # RR_net 계산하여 유효한 TP 찾기
                        risk = entry_price - sl
                        total_cost_pct = (self.config.fee_bps + self.config.slippage_bps) * 2 / 10000
                        fee_cost = entry_price * total_cost_pct
                        valid_tps = []

                        # PR-RANGE-1 (C): RANGE에서 더 높은 RR 요구
                        min_rr_threshold = self.config.min_rr_net
                        if current_drift_regime == 'RANGE' and self.config.use_range_policy:
                            min_rr_threshold = self.config.min_rr_net_range

                        for fib_tp in fib_candidates:
                            reward = fib_tp - entry_price
                            profit_net = reward - fee_cost
                            loss_net = risk + fee_cost
                            rr = profit_net / loss_net if loss_net > 0 else 0
                            if rr >= min_rr_threshold:
                                valid_tps.append((fib_tp, rr))

                        if not valid_tps:
                            # 유효한 TP 없음 - 진입 거부
                            fib_rr_rejected = True
                            rr_gate_rejects['RR_FAIL_FIB_LONG'] = rr_gate_rejects.get('RR_FAIL_FIB_LONG', 0) + 1
                            print(f"  [LONG REJECTED] {current_time} - FIB_RR: No Fib level meets RR>={min_rr_threshold}")
                            pending_long_signal = None
                            continue

                        # 가장 가까운(첫번째) 유효 TP 선택
                        selected_tp, selected_rr = valid_tps[0]
                        tp1 = selected_tp
                        # DEBUG: TP 선택 출력
                        if hasattr(self.config, '_tp_debug') and self.config._tp_debug:
                            print(f"  [TP-DEBUG LONG] Entry=${entry_price:.0f} | TP1=${tp1:.0f} | RR={selected_rr:.2f} | valid_tps={len(valid_tps)}")
                            if self.config.use_dynamic_fib and dynfib_state and dynfib_state.is_valid():
                                dyn_levels = get_dynamic_fib_levels(dynfib_state.low, dynfib_state.high, self.config.dynfib_ratios, space=self.config.dynamic_fib_space)
                                is_dyn = tp1 in dyn_levels
                                print(f"    Anchor=({dynfib_state.low:.0f}, {dynfib_state.high:.0f}) | Range=${dynfib_state.get_range():.0f} | TP_from={'DYN' if is_dyn else 'MACRO'}")
                        # TP2, TP3는 다음 유효 TP 또는 ATR 폴백
                        tp2 = valid_tps[1][0] if len(valid_tps) > 1 else entry_price + (atr * 3.5)
                        tp3 = valid_tps[2][0] if len(valid_tps) > 2 else entry_price + (atr * 5.0)
                        rr_net = selected_rr  # fib_rr 모드에서는 이미 RR 계산됨
                    else:
                        # Fib TP: TP1 (두번째 L1), TP2 (세번째 L1), TP3 (네번째 L1) - 확장된 타겟
                        fib1 = get_next_l1_above(entry_price, self.config)
                        fib2 = get_next_l1_above(fib1.price + 1, self.config) if fib1 else None
                        fib3 = get_next_l1_above(fib2.price + 1, self.config) if fib2 else None
                        fib4 = get_next_l1_above(fib3.price + 1, self.config) if fib3 else None
                        # TP1 = 두번째 L1 (첫번째 스킵)
                        tp1 = fib2.price if fib2 else entry_price + (atr * 2.0)
                        # TP2 = 세번째 L1 또는 ATR 기반
                        tp2 = fib3.price if fib3 else entry_price + (atr * 3.5)
                        # TP3 = 네번째 L1 또는 ATR 기반 (PR6.2)
                        tp3 = fib4.price if fib4 else entry_price + (atr * 5.0)

                    # === PR4-R5/R6: Entry RR Gate (SL or Liquidation 기반) ===
                    leverage_used = 1.0
                    liq_price = None
                    rr_net = 0.0

                    if self.config.use_liq_as_stop:
                        # PR4-R6: Liquidation Mode - SL 대신 liq_price 사용
                        tp_target = tp1 if self.config.rr_gate_use_tp1 else tp2

                        if self.config.leverage_mode == 'dynamic':
                            # Dynamic: RR+liq_distance 조건 만족하는 최대 레버리지
                            leverage_used, liq_price, rr_net = select_leverage_dynamic(
                                entry_price, tp_target, 'long', atr, self.config
                            )
                            if leverage_used == 0:
                                rr_gate_rejects['RR_GATE_LONG_LOW_RR'] += 1
                                print(f"  [LONG REJECTED] {current_time} - LIQ_RR_GATE: No valid leverage found")
                                pending_long_signal = None
                                continue
                        else:
                            # Fixed leverage
                            leverage_used = self.config.leverage_fixed
                            liq_price = calc_liq_price(entry_price, 'long', leverage_used, self.config.liq_mmr)
                            liq_dist_atr = calc_liq_distance_atr(entry_price, liq_price, atr)

                            # min_liq_distance_atr 체크
                            if liq_dist_atr < self.config.min_liq_distance_atr:
                                rr_gate_rejects['RR_GATE_LONG_LOW_RR'] += 1
                                print(f"  [LONG REJECTED] {current_time} - LIQ_DIST: {liq_dist_atr:.2f} < min={self.config.min_liq_distance_atr}")
                                pending_long_signal = None
                                continue

                            # RR_net 계산 (liq_price 기반)
                            total_cost_pct = (self.config.fee_bps + self.config.slippage_bps) * 2 / 10000
                            fee_cost = entry_price * total_cost_pct
                            profit_gross = tp_target - entry_price
                            loss_gross = entry_price - liq_price
                            profit_net = profit_gross - fee_cost
                            loss_net = loss_gross + fee_cost
                            rr_net = profit_net / loss_net if loss_net > 0 else 0

                            if rr_net < self.config.min_rr_net:
                                rr_gate_rejects['RR_GATE_LONG_LOW_RR'] += 1
                                print(f"  [LONG REJECTED] {current_time} - LIQ_RR_GATE: RR_net={rr_net:.2f} < min={self.config.min_rr_net}")
                                pending_long_signal = None
                                continue

                        # Liquidation mode: SL을 liq_price로 덮어쓰기
                        sl = liq_price

                    elif self.config.use_rr_gate:
                        # PR4-R5: 기존 SL 기반 RR Gate
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
                            print(f"  [LONG REJECTED] {current_time} - RR_GATE: RR_net={rr_net:.2f} < min={self.config.min_rr_net}")
                            pending_long_signal = None
                            continue

                    # 모든 트레이드 로그 (디버그용)
                    print(f"  [LONG ENTRY] {current_time}")

                    # PR-RANGE-1 (D): RANGE 진입 시 쿨다운 트래커 업데이트
                    if current_drift_regime == 'RANGE':
                        self.last_range_entry_bar_idx = i

                    if self.config.use_liq_as_stop:
                        print(f"    Entry: ${entry_price:,.0f} | LIQ: ${liq_price:,.0f} | TP1: ${tp1:,.0f} | Lev: {leverage_used:.0f}x | RR: {rr_net:.2f}")
                    else:
                        print(f"    Entry: ${entry_price:,.0f} | SL: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f} | TP3: ${tp3:,.0f}")
                    print(f"    Trend: 1H={trend_1h}, 4H={trend_4h} | Size: {size_mult:.2f}x | Depth: {zone_depth:.2f}")
                    if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                        print(f"    Cycle: {cycle_result['phase_degrees']:.1f}° ({cycle_result['cycle_state']}) | SL Mult: {sl_mult:.3f}")
                    else:
                        print(f"    Cycle: N/A | SL Mult: {sl_mult:.3f} (fixed)")

                    # PR6: 리스크 고정 사이징 + PR6.3 로깅
                    qty_info = self.config.calculate_qty_info(entry_price, sl, atr)
                    qty = qty_info['qty']

                    long_position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'tp3': tp3,  # PR6.2: TP3 추가
                        'atr': atr,  # 트레일링 스탑용
                        'qty': qty,  # PR6: 포지션 수량
                        'remaining': 1.0,  # 100% 남음
                        'size_mult': size_mult,  # ATR 변동성 사이즈 배수
                        # Early Exit 추적용
                        'entry_bar_idx': i,  # 진입 바 인덱스
                        'mfe': entry_price,  # Maximum Favorable Excursion (롱: 최고가)
                        # PR4 분석용
                        'mae': entry_price,  # Maximum Adverse Excursion (롱: 최저가)
                        'mfe_first_6': entry_price,  # 첫 6 bars MFE
                        'mae_first_6': entry_price,  # 첫 6 bars MAE
                        'bars_held': 0,
                        # PR6.3: 사이징 로깅
                        'sl_distance_raw': qty_info['sl_distance_raw'],
                        'sl_distance_atr': qty_info['sl_distance_atr'],
                        'clamped': qty_info['clamped'],
                        'notional': qty_info['notional'],
                        'cap_reason': qty_info['cap_reason'],
                        # PR4-R6: Liquidation Mode
                        'leverage': leverage_used,
                        'liq_price': liq_price,
                        'is_liq_mode': self.config.use_liq_as_stop,
                    }
                    pending_long_signal = None

            # Short 반등 확인: 음봉 (close < open)
            if pending_short_signal is not None and short_position is None:
                is_bearish_candle = bar['close'] < bar['open']
                bars_since_touch = (current_time - pending_short_signal['touched_time']).total_seconds() / 300
                if bars_since_touch > 3:
                    pending_short_signal = None  # 신호 만료
                elif is_bearish_candle:
                    # === 추세 필터 체크 (Precomputed 컬럼 사용) ===
                    trend_1h = "UNKNOWN"
                    trend_4h = "UNKNOWN"

                    if df_1h is not None and 'trend' in df_1h.columns:
                        df_1h_valid = df_1h[df_1h.index <= current_time]
                        if len(df_1h_valid) > 0:
                            trend_1h = df_1h_valid['trend'].iloc[-1]

                    if df_4h is not None and 'trend' in df_4h.columns:
                        df_4h_valid = df_4h[df_4h.index <= current_time]
                        if len(df_4h_valid) > 0:
                            trend_4h = df_4h_valid['trend'].iloc[-1]

                    pass_filter, reject_reason = check_trend_filter(
                        'short', trend_1h, trend_4h,
                        self.config.use_trend_filter_1h,
                        self.config.use_trend_filter_4h
                    )

                    if not pass_filter:
                        trend_filter_rejects[reject_reason] += 1
                        print(f"  [SHORT REJECTED] {current_time} - {reject_reason} (1H:{trend_1h}, 4H:{trend_4h})")
                        pending_short_signal = None
                        continue

                    # === Hilbert 레짐 필터 (context_tf, causal) ===
                    if self.config.use_hilbert_filter and hilbert_regimes is not None:
                        # 완료된 context_tf봉 기준 (lookahead 방지)
                        ts_ctx = _floor_by_tf(current_time, self.config.context_tf)
                        if ts_ctx in hilbert_regimes.index:
                            hilbert_regime = str(hilbert_regimes.loc[ts_ctx, 'regime'])
                        else:
                            mask = hilbert_regimes.index <= ts_ctx
                            if mask.any():
                                hilbert_regime = str(hilbert_regimes.loc[hilbert_regimes.index[mask][-1], 'regime'])
                            else:
                                hilbert_regime = 'RANGE'

                        pass_hilbert, hilbert_reason = check_hilbert_filter(
                            'short', hilbert_regime,
                            self.config.hilbert_block_long_on_bear,
                            self.config.hilbert_block_short_on_bull
                        )

                        if not pass_hilbert:
                            hilbert_filter_rejects[hilbert_reason] += 1
                            print(f"  [SHORT REJECTED] {current_time} - {hilbert_reason} (Hilbert:{hilbert_regime})")
                            pending_short_signal = None
                            continue

                    # === ProbabilityGate v2 필터 (1H Hilbert → 5m, causal) ===
                    if self.config.use_prob_gate and prob_gate_result is not None:
                        pass_gate, gate_reason, p_bull = check_prob_gate_filter(
                            'short', current_time, prob_gate_result, self.config
                        )
                        if not pass_gate:
                            prob_gate_rejects[gate_reason] += 1
                            print(f"  [SHORT REJECTED] {current_time} - {gate_reason} (p_bull={p_bull:.3f})")
                            pending_short_signal = None
                            continue

                    # === ATR 변동성 필터 (Precomputed 컬럼 사용) ===
                    size_mult = 1.0
                    if self.config.use_atr_vol_filter and df_1h is not None and 'atr_pct' in df_1h.columns:
                        df_1h_valid = df_1h[df_1h.index <= current_time]
                        if len(df_1h_valid) > 0:
                            atr_pct = df_1h_valid['atr_pct'].iloc[-1]
                            if atr_pct > self.config.atr_vol_threshold:
                                size_mult = self.config.atr_vol_size_mult
                                atr_vol_size_cuts += 1

                    # === Zone Depth 사이징 (필터→사이징 전환) ===
                    # 더 이상 진입 금지하지 않음. 대신 depth에 비례해 포지션 크기 조절
                    zone_depth = 0.5  # 기본값
                    if self.config.use_zone_depth_filter:
                        current_idx = len(df_5m_slice) - 1
                        zone_depth = calc_zone_depth(
                            close_arr_5m, current_idx, 'short',
                            lookback=self.config.zone_depth_lookback
                        )
                        depth_size_mult = calc_zone_depth_size_mult(zone_depth, self.config.zone_depth_min)
                        # 필터 없음: 항상 진입, 크기만 조절 (0.25 ~ 1.0)
                        size_mult *= depth_size_mult

                    # 반등 확인! 시장가 진입
                    entry_price = bar['close']
                    atr = pending_short_signal['atr']

                    # 동적 SL 배수 계산
                    if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                        cycle_result = self.cycle_dynamics.analyze(close_arr_15m)
                        sl_mult = cycle_result['dynamic_sl_mult']
                    else:
                        sl_mult = self.config.sl_atr_mult

                    # ATR 기반 SL
                    sl = entry_price + (atr * sl_mult)

                    # PR4-R0: TP 모드에 따라 계산
                    fib_rr_rejected = False  # fib_rr 모드에서 RR 기준 미달 시 True
                    if self.config.tp_mode == "atr":
                        # ATR 배수 기반 TP (RR 안정화)
                        tp_mults = self.config.tp_atr_mults
                        tp1 = entry_price - (atr * tp_mults[0])
                        tp2 = entry_price - (atr * tp_mults[1])
                        tp3 = entry_price - (atr * tp_mults[2])
                    elif self.config.tp_mode == "fib_rr":
                        # === PR-B: Fib + RR 필터 모드 (SHORT) ===
                        # Fib 후보 중 RR_net >= min_rr_net 만족하는 가장 가까운 TP 선택
                        # 없으면 진입 거부
                        num_candidates = self.config.fib_tp_candidates
                        fib_candidates = []
                        current_price = entry_price
                        for _ in range(num_candidates + 1):  # +1 for skipping first
                            fib = get_next_l1_below(current_price, self.config)
                            if fib:
                                fib_candidates.append(fib.price)
                                current_price = fib.price - 1
                            else:
                                break

                        # 첫번째 Fib 스킵 (너무 가까움)
                        fib_candidates = fib_candidates[1:] if len(fib_candidates) > 1 else fib_candidates

                        # === PR-DYN-FIB v2: 동적 Fib 레벨 추가 (SHORT) ===
                        if self.config.use_dynamic_fib and dynfib_state is not None and dynfib_state.is_valid():
                            dyn_levels = get_dynamic_fib_levels(
                                dynfib_state.low, dynfib_state.high,
                                self.config.dynfib_ratios,
                                space=self.config.dynamic_fib_space
                            )
                            # entry_price 아래의 레벨만 TP 후보로 추가 (SHORT용)
                            dyn_below = [p for p in dyn_levels if p < entry_price]
                            if self.config.dynfib_use_as == "tp_candidate":
                                # Macro Fib + Dynamic Fib 합침
                                fib_candidates = sorted(set(fib_candidates + dyn_below), reverse=True)

                        # RR_net 계산하여 유효한 TP 찾기
                        risk = sl - entry_price  # SHORT: SL이 entry 위
                        total_cost_pct = (self.config.fee_bps + self.config.slippage_bps) * 2 / 10000
                        fee_cost = entry_price * total_cost_pct
                        valid_tps = []

                        for fib_tp in fib_candidates:
                            reward = entry_price - fib_tp  # SHORT: TP가 entry 아래
                            profit_net = reward - fee_cost
                            loss_net = risk + fee_cost
                            rr = profit_net / loss_net if loss_net > 0 else 0
                            if rr >= self.config.min_rr_net:
                                valid_tps.append((fib_tp, rr))

                        if not valid_tps:
                            # 유효한 TP 없음 - 진입 거부
                            fib_rr_rejected = True
                            rr_gate_rejects['RR_FAIL_FIB_SHORT'] = rr_gate_rejects.get('RR_FAIL_FIB_SHORT', 0) + 1
                            print(f"  [SHORT REJECTED] {current_time} - FIB_RR: No Fib level meets RR>={self.config.min_rr_net}")
                            pending_short_signal = None
                            continue

                        # 가장 가까운(첫번째) 유효 TP 선택
                        selected_tp, selected_rr = valid_tps[0]
                        tp1 = selected_tp
                        # TP2, TP3는 다음 유효 TP 또는 ATR 폴백
                        tp2 = valid_tps[1][0] if len(valid_tps) > 1 else entry_price - (atr * 3.5)
                        tp3 = valid_tps[2][0] if len(valid_tps) > 2 else entry_price - (atr * 5.0)
                        rr_net = selected_rr  # fib_rr 모드에서는 이미 RR 계산됨
                    else:
                        # Fib TP: TP1 (두번째 L1 아래), TP2 (세번째 L1 아래) - 확장된 타겟
                        fib1 = get_next_l1_below(entry_price, self.config)
                        fib2 = get_next_l1_below(fib1.price - 1, self.config) if fib1 else None
                        fib3 = get_next_l1_below(fib2.price - 1, self.config) if fib2 else None
                        fib4 = get_next_l1_below(fib3.price - 1, self.config) if fib3 else None
                        # TP1 = 두번째 L1 (첫번째 스킵)
                        tp1 = fib2.price if fib2 else entry_price - (atr * 2.0)
                        # TP2 = 세번째 L1 또는 ATR 기반
                        tp2 = fib3.price if fib3 else entry_price - (atr * 3.5)
                        # TP3 = 네번째 L1 또는 ATR 기반 (PR6.2)
                        tp3 = fib4.price if fib4 else entry_price - (atr * 5.0)

                    # === PR4-R5/R6: Entry RR Gate (SL or Liquidation 기반) ===
                    leverage_used = 1.0
                    liq_price = None
                    rr_net = 0.0

                    if self.config.use_liq_as_stop:
                        # PR4-R6: Liquidation Mode - SL 대신 liq_price 사용
                        tp_target = tp1 if self.config.rr_gate_use_tp1 else tp2

                        if self.config.leverage_mode == 'dynamic':
                            # Dynamic: RR+liq_distance 조건 만족하는 최대 레버리지
                            leverage_used, liq_price, rr_net = select_leverage_dynamic(
                                entry_price, tp_target, 'short', atr, self.config
                            )
                            if leverage_used == 0:
                                rr_gate_rejects['RR_GATE_SHORT_LOW_RR'] += 1
                                print(f"  [SHORT REJECTED] {current_time} - LIQ_RR_GATE: No valid leverage found")
                                pending_short_signal = None
                                continue
                        else:
                            # Fixed leverage
                            leverage_used = self.config.leverage_fixed
                            liq_price = calc_liq_price(entry_price, 'short', leverage_used, self.config.liq_mmr)
                            liq_dist_atr = calc_liq_distance_atr(entry_price, liq_price, atr)

                            # min_liq_distance_atr 체크
                            if liq_dist_atr < self.config.min_liq_distance_atr:
                                rr_gate_rejects['RR_GATE_SHORT_LOW_RR'] += 1
                                print(f"  [SHORT REJECTED] {current_time} - LIQ_DIST: {liq_dist_atr:.2f} < min={self.config.min_liq_distance_atr}")
                                pending_short_signal = None
                                continue

                            # RR_net 계산 (liq_price 기반)
                            total_cost_pct = (self.config.fee_bps + self.config.slippage_bps) * 2 / 10000
                            fee_cost = entry_price * total_cost_pct
                            profit_gross = entry_price - tp_target
                            loss_gross = liq_price - entry_price
                            profit_net = profit_gross - fee_cost
                            loss_net = loss_gross + fee_cost
                            rr_net = profit_net / loss_net if loss_net > 0 else 0

                            if rr_net < self.config.min_rr_net:
                                rr_gate_rejects['RR_GATE_SHORT_LOW_RR'] += 1
                                print(f"  [SHORT REJECTED] {current_time} - LIQ_RR_GATE: RR_net={rr_net:.2f} < min={self.config.min_rr_net}")
                                pending_short_signal = None
                                continue

                        # Liquidation mode: SL을 liq_price로 덮어쓰기
                        sl = liq_price

                    elif self.config.use_rr_gate:
                        # PR4-R5: 기존 SL 기반 RR Gate
                        tp_target = tp1 if self.config.rr_gate_use_tp1 else tp2
                        total_cost_pct = (self.config.fee_bps + self.config.slippage_bps) * 2 / 10000
                        fee_cost = entry_price * total_cost_pct
                        profit_gross = entry_price - tp_target
                        loss_gross = sl - entry_price
                        profit_net = profit_gross - fee_cost
                        loss_net = loss_gross + fee_cost
                        rr_net = profit_net / loss_net if loss_net > 0 else 0

                        if rr_net < self.config.min_rr_net:
                            rr_gate_rejects['RR_GATE_SHORT_LOW_RR'] += 1
                            print(f"  [SHORT REJECTED] {current_time} - RR_GATE: RR_net={rr_net:.2f} < min={self.config.min_rr_net}")
                            pending_short_signal = None
                            continue

                    # 모든 트레이드 로그 (디버그용)
                    print(f"  [SHORT ENTRY] {current_time}")
                    if self.config.use_liq_as_stop:
                        print(f"    Entry: ${entry_price:,.0f} | LIQ: ${liq_price:,.0f} | TP1: ${tp1:,.0f} | Lev: {leverage_used:.0f}x | RR: {rr_net:.2f}")
                    else:
                        print(f"    Entry: ${entry_price:,.0f} | SL: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f} | TP3: ${tp3:,.0f}")
                    print(f"    Trend: 1H={trend_1h}, 4H={trend_4h} | Size: {size_mult:.2f}x | Depth: {zone_depth:.2f}")
                    if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                        print(f"    Cycle: {cycle_result['phase_degrees']:.1f}° ({cycle_result['cycle_state']}) | SL Mult: {sl_mult:.3f}")
                    else:
                        print(f"    Cycle: N/A | SL Mult: {sl_mult:.3f} (fixed)")

                    # PR6: 리스크 고정 사이징 + PR6.3 로깅
                    qty_info = self.config.calculate_qty_info(entry_price, sl, atr)
                    qty = qty_info['qty']

                    short_position = {
                        'side': 'short',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'tp3': tp3,  # PR6.2: TP3 추가
                        'atr': atr,  # 트레일링 스탑용
                        'qty': qty,  # PR6: 포지션 수량
                        'remaining': 1.0,  # 100% 남음
                        'size_mult': size_mult,  # ATR 변동성 사이즈 배수
                        # Early Exit 추적용
                        'entry_bar_idx': i,  # 진입 바 인덱스
                        'mfe': entry_price,  # Maximum Favorable Excursion (숏: 최저가)
                        # PR4 분석용
                        'mae': entry_price,  # Maximum Adverse Excursion (숏: 최고가)
                        'mfe_first_6': entry_price,  # 첫 6 bars MFE
                        'mae_first_6': entry_price,  # 첫 6 bars MAE
                        'bars_held': 0,
                        # PR6.3: 사이징 로깅
                        'sl_distance_raw': qty_info['sl_distance_raw'],
                        'sl_distance_atr': qty_info['sl_distance_atr'],
                        'clamped': qty_info['clamped'],
                        'notional': qty_info['notional'],
                        'cap_reason': qty_info['cap_reason'],
                        # PR4-R6: Liquidation Mode
                        'leverage': leverage_used,
                        'liq_price': liq_price,
                        'is_liq_mode': self.config.use_liq_as_stop,
                    }
                    pending_short_signal = None

            # ===== 새 신호 체크 (존 터치) =====
            # 15m 먼저 체크, 없으면 5m fallback
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # Long 신호 체크
            if long_position is None and pending_long_signal is None and long_cooldown == 0:
                long_price = None
                signal_tf = self.config.anchor_tf
                div_type = 'Regular'

                # === 레짐 기반 Hidden Divergence 전략 ===
                if self.config.use_regime_hidden_strategy:
                    current_regime = get_current_hilbert_regime(hilbert_regimes, current_time, self.config.context_tf)
                    # BULL 레짐에서만 Hidden Bullish로 Long
                    if current_regime == 'BULL':
                        ref = find_swing_low_reference(df_15m_slice)
                        if ref:
                            long_price = needed_close_for_hidden_bullish(
                                close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                            )
                            div_type = 'Hidden'
                    # BEAR/RANGE에서는 Long 스킵
                else:
                    # === 과매도 구간 체크 (진입 순간 + 진행 중) ===
                    is_in_oversold = (prev_15m_stoch <= self.config.stoch_rsi_oversold)

                    # PR4-R4a: UPTREND에서 추세 지속형 진입 모드 체크
                    current_drift_regime = 'RANGE'  # 기본값
                    if prob_gate_result is not None and 'drift_regime' in prob_gate_result.columns:
                        pg_mask = prob_gate_result.index <= current_time
                        if pg_mask.any():
                            current_drift_regime = prob_gate_result.loc[pg_mask, 'drift_regime'].iloc[-1]

                    use_trend_continuation = (
                        current_drift_regime == 'UPTREND' and
                        self.config.uptrend_entry_mode == "trend_continuation"
                    )

                    if use_trend_continuation and long_signal_triggered:
                        # === PR4-R4a: UPTREND 추세 지속형 진입 ===
                        # 조건: StochRSI oversold 크로스 + EMA 조건 + 쿨다운 + 브레이크아웃
                        entry_allowed = True
                        reject_reason = None

                        # Step A: 쿨다운 체크 (오버트레이딩 방지)
                        cooldown_bars = self.config.uptrend_entry_cooldown_bars
                        if cooldown_bars > 0:
                            bars_since_last = i - last_trend_cont_entry_bar_idx
                            if bars_since_last < cooldown_bars:
                                entry_allowed = False
                                reject_reason = f"COOLDOWN ({bars_since_last}/{cooldown_bars})"

                        # EMA 계산 (15m 데이터 사용)
                        ema_period = self.config.uptrend_ema_period
                        if entry_allowed and len(df_15m_slice) >= ema_period:
                            ema_values = df_15m_slice['close'].ewm(span=ema_period, adjust=False).mean()
                            current_ema = ema_values.iloc[-1]
                            current_close = df_15m_slice['close'].iloc[-1]

                            # 조건 1: close > EMA
                            if self.config.uptrend_require_ema_above and current_close <= current_ema:
                                entry_allowed = False
                                reject_reason = "CLOSE_BELOW_EMA"

                            # 조건 2: EMA slope > 0 (최근 3봉 기준)
                            if entry_allowed and self.config.uptrend_require_slope_positive and len(ema_values) >= 3:
                                ema_slope = ema_values.iloc[-1] - ema_values.iloc[-3]
                                if ema_slope <= 0:
                                    entry_allowed = False
                                    reject_reason = "EMA_SLOPE_NEGATIVE"

                            # Step B: 브레이크아웃 확인 (close > high[-1])
                            if entry_allowed and self.config.uptrend_require_breakout and len(df_15m_slice) >= 2:
                                prev_high = df_15m_slice['high'].iloc[-2]
                                if current_close <= prev_high:
                                    entry_allowed = False
                                    reject_reason = "NO_BREAKOUT"

                            if entry_allowed:
                                # 진입가: 현재 close (다이버전스 가격 아님)
                                long_price = current_close

                                # Fib 레벨 근접 체크 (Macro + Dynamic Fib 통합)
                                is_near, fib_level, fib_src = is_near_fib_level_combined(long_price, current_atr, self.config, dynfib_state)
                                if is_near and fib_level:
                                    pending_long_signal = {
                                        'zone_price': long_price,
                                        'fib_level': fib_level,
                                        'atr': current_atr,
                                        'touched_time': current_time,
                                    }
                                    signal_tf = "15m_TrendCont"
                                    div_type = "TrendContinuation"
                                    # 쿨다운 트래커 업데이트
                                    last_trend_cont_entry_bar_idx = i
                                    print(f"  [LONG SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                    print(f"    Entry Price: ${long_price:,.0f} | EMA: ${current_ema:,.0f}")
                                    print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio}) [{fib_src}]")
                        elif entry_allowed:
                            reject_reason = "INSUFFICIENT_DATA"

                    elif long_signal_triggered or is_in_oversold:
                        # === 기존: 다이버전스 진입 (RANGE/DOWNTREND 또는 divergence 모드) ===
                        # === 1) 15m 다이버전스 진입 시도 ===
                        ref = find_oversold_reference(df_15m_slice, threshold=self.config.stoch_rsi_oversold)
                        if ref:
                            long_price = needed_close_for_regular_bullish(
                                close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                            )

                            # 15m 진입 시도
                            if long_price and long_price > 0:
                                is_near, fib_level, fib_src = is_near_fib_level_combined(long_price, current_atr, self.config, dynfib_state)
                                if is_near and fib_level and bar['low'] <= long_price:
                                    pending_long_signal = {
                                        'zone_price': long_price,
                                        'fib_level': fib_level,
                                        'atr': current_atr,
                                        'touched_time': current_time,
                                    }
                                    signal_tf = self.config.anchor_tf
                                    print(f"  [LONG SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                    print(f"    Div Price: ${long_price:,.0f} | Bar Low: ${bar['low']:,.0f}")
                                    print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio}) [{fib_src}]")

                        # === 2) 15m 실패 시 5m fallback (15m 세그먼트 → 5m REF) ===
                        if pending_long_signal is None:
                            ref_5m = find_oversold_reference_hybrid(df_15m_slice, df_5m_slice)
                            if ref_5m:
                                long_price_5m = needed_close_for_regular_bullish(
                                    close_arr_5m, ref_5m['ref_price'], ref_5m['ref_rsi'], self.config.rsi_period
                                )

                                # 5m 진입 시도
                                if long_price_5m and long_price_5m > 0:
                                    is_near, fib_level, fib_src = is_near_fib_level_combined(long_price_5m, current_atr, self.config, dynfib_state)
                                    if is_near and fib_level and bar['low'] <= long_price_5m:
                                        pending_long_signal = {
                                            'zone_price': long_price_5m,
                                            'fib_level': fib_level,
                                            'atr': current_atr,
                                            'touched_time': current_time,
                                        }
                                        signal_tf = self.config.trigger_tf
                                        print(f"  [LONG SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                        print(f"    Div Price: ${long_price_5m:,.0f} | Bar Low: ${bar['low']:,.0f}")
                                        print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio}) [{fib_src}]")

                        # === 3) Regular 실패 시 Hidden Bullish Divergence 시도 ===
                        if pending_long_signal is None and self.config.use_hidden_div_long:
                            ref_hidden = find_swing_low_reference(df_15m_slice)
                            if ref_hidden:
                                hidden_price = needed_close_for_hidden_bullish(
                                    close_arr_15m, ref_hidden['ref_price'], ref_hidden['ref_rsi'], self.config.rsi_period
                                )
                                if hidden_price and hidden_price > 0:
                                    is_near, fib_level, fib_src = is_near_fib_level_combined(hidden_price, current_atr, self.config, dynfib_state)
                                    if is_near and fib_level and bar['low'] <= hidden_price:
                                        pending_long_signal = {
                                            'zone_price': hidden_price,
                                            'fib_level': fib_level,
                                            'atr': current_atr,
                                            'touched_time': current_time,
                                        }
                                        div_type = 'Hidden'
                                        signal_tf = self.config.anchor_tf
                                        print(f"  [LONG SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                        print(f"    Div Price: ${hidden_price:,.0f} | Bar Low: ${bar['low']:,.0f}")
                                        print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio}) [{fib_src}]")

            # Short 신호 체크 (PR4-R2: enable_short=False면 스킵)
            if self.config.enable_short and short_position is None and pending_short_signal is None and short_cooldown == 0:
                short_price = None
                signal_tf = self.config.anchor_tf
                div_type = 'Regular'

                # === 레짐 기반 Hidden Divergence 전략 ===
                if self.config.use_regime_hidden_strategy:
                    current_regime = get_current_hilbert_regime(hilbert_regimes, current_time, self.config.context_tf)
                    # BEAR 레짐에서만 Hidden Bearish로 Short
                    if current_regime == 'BEAR':
                        ref = find_swing_high_reference(df_15m_slice)
                        if ref:
                            short_price = needed_close_for_hidden_bearish(
                                close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                            )
                            div_type = 'Hidden'
                    # BULL/RANGE에서는 Short 스킵
                else:
                    # === 과매수 구간 체크 (진입 순간 + 진행 중) ===
                    is_in_overbought = (prev_15m_stoch >= self.config.stoch_rsi_overbought)

                    if short_signal_triggered or is_in_overbought:
                        # === 1) 15m 다이버전스 진입 시도 ===
                        ref = find_overbought_reference(df_15m_slice, threshold=self.config.stoch_rsi_overbought)
                        if ref:
                            short_price = needed_close_for_regular_bearish(
                                close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                            )

                            # 15m 진입 시도
                            if short_price and short_price > 0:
                                is_near, fib_level, fib_src = is_near_fib_level_combined(short_price, current_atr, self.config, dynfib_state)
                                if is_near and fib_level and bar['high'] >= short_price:
                                    pending_short_signal = {
                                        'zone_price': short_price,
                                        'fib_level': fib_level,
                                        'atr': current_atr,
                                        'touched_time': current_time,
                                    }
                                    signal_tf = self.config.anchor_tf
                                    print(f"  [SHORT SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                    print(f"    Div Price: ${short_price:,.0f} | Bar High: ${bar['high']:,.0f}")
                                    print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio}) [{fib_src}]")

                        # === 2) 15m 실패 시 5m fallback (15m 세그먼트 → 5m REF) ===
                        if pending_short_signal is None:
                            ref_5m = find_overbought_reference_hybrid(df_15m_slice, df_5m_slice)
                            if ref_5m:
                                short_price_5m = needed_close_for_regular_bearish(
                                    close_arr_5m, ref_5m['ref_price'], ref_5m['ref_rsi'], self.config.rsi_period
                                )

                                # 5m 진입 시도
                                if short_price_5m and short_price_5m > 0:
                                    is_near, fib_level, fib_src = is_near_fib_level_combined(short_price_5m, current_atr, self.config, dynfib_state)
                                    if is_near and fib_level and bar['high'] >= short_price_5m:
                                        pending_short_signal = {
                                            'zone_price': short_price_5m,
                                            'fib_level': fib_level,
                                            'atr': current_atr,
                                            'touched_time': current_time,
                                        }
                                        signal_tf = self.config.trigger_tf
                                        print(f"  [SHORT SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                        print(f"    Div Price: ${short_price_5m:,.0f} | Bar High: ${bar['high']:,.0f}")
                                        print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio}) [{fib_src}]")

        # 미청산 포지션 정리
        if long_position:
            trade = self._close_position(long_position, df_5m.iloc[-1]['close'], 'EOD', df_5m.index[-1])
            result.trades.append(trade)
            equity += trade.pnl_usd
            result.equity_curve.append(equity)

        if short_position:
            trade = self._close_position(short_position, df_5m.iloc[-1]['close'], 'EOD', df_5m.index[-1])
            result.trades.append(trade)
            equity += trade.pnl_usd
            result.equity_curve.append(equity)

        # 추세 필터 통계 출력
        total_rejects = sum(trend_filter_rejects.values())
        if total_rejects > 0 or atr_vol_size_cuts > 0:
            print(f"\n  [Trend Filter Stats]")
            for reason, count in trend_filter_rejects.items():
                if count > 0:
                    print(f"    {reason}: {count} rejected")
            if atr_vol_size_cuts > 0:
                print(f"    ATR Vol Size Cuts: {atr_vol_size_cuts}")

        # Hilbert 필터 통계 출력
        total_hilbert_rejects = sum(hilbert_filter_rejects.values())
        if total_hilbert_rejects > 0:
            print(f"\n  [Hilbert Filter Stats]")
            for reason, count in hilbert_filter_rejects.items():
                if count > 0:
                    print(f"    {reason}: {count} rejected")

        # ProbabilityGate v2 통계 출력
        total_gate_rejects = sum(prob_gate_rejects.values())
        print(f"\n  [ProbabilityGate v2 Stats]")
        print(f"    Total Rejects: {total_gate_rejects}")
        for reason, count in prob_gate_rejects.items():
            print(f"    {reason}: {count} rejected")

        # GPT 진단: Pass Reasons 출력
        print(f"\n  [ProbabilityGate v2 PASS Reasons]")
        for k, v in prob_gate_pass_reasons.items():
            print(f"    {k}: {v}")

        # PR4-R3: Regime Trade Permission 통계 출력
        total_regime_rejects = sum(regime_rejects.values())
        if total_regime_rejects > 0:
            print(f"\n  [PR4-R3 Regime Filter Stats]")
            print(f"    Total Rejects: {total_regime_rejects}")
            for reason, count in regime_rejects.items():
                if count > 0:
                    print(f"    {reason}: {count} rejected")

        # PR4-R5: RR Gate 통계 출력
        total_rr_rejects = sum(rr_gate_rejects.values())
        if total_rr_rejects > 0:
            print(f"\n  [PR4-R5 RR Gate Stats]")
            print(f"    Total Rejects: {total_rr_rejects}")
            for reason, count in rr_gate_rejects.items():
                if count > 0:
                    print(f"    {reason}: {count} rejected")

        return result

    def _check_long_divergence(self, df: pd.DataFrame) -> bool:
        """5m 롱 다이버전스 체크 - Regular만 사용"""
        ref = find_oversold_reference(df, threshold=self.config.stoch_rsi_oversold)
        if not ref:
            return False
        close_arr = df['close'].values.astype(float)
        # Regular만 사용 (Hidden 제거)
        price = needed_close_for_regular_bullish(close_arr, ref['ref_price'], ref['ref_rsi'])
        if not price or price <= 0:
            return False
        # 현재가가 다이버전스 형성 가격 이하일 때 롱 다이버전스 확정
        return close_arr[-1] <= price

    def _check_short_divergence(self, df: pd.DataFrame) -> bool:
        """5m 숏 다이버전스 체크 - Regular만 사용"""
        ref = find_overbought_reference(df, threshold=self.config.stoch_rsi_overbought)
        if not ref:
            return False
        close_arr = df['close'].values.astype(float)
        # Regular만 사용 (Hidden 제거)
        price = needed_close_for_regular_bearish(close_arr, ref['ref_price'], ref['ref_rsi'])
        if not price or price <= 0:
            return False
        # 현재가가 다이버전스 형성 가격 이상일 때 숏 다이버전스 확정
        return close_arr[-1] >= price

    def _close_position(self, position: Dict, exit_price: float, reason: str, exit_time) -> Trade:
        """포지션 청산 및 Trade 생성"""
        hours = (exit_time - position['entry_time']).total_seconds() / 3600

        # PR6: 리스크 고정 사이징 사용 시 qty 기반 계산
        if self.config.use_risk_fixed_sizing and 'qty' in position:
            qty = position['qty']
            pnl_usd, pnl_pct = self.config.calculate_pnl_usd(
                position['side'], position['entry_price'], exit_price, qty, hours
            )
            leveraged_pnl_pct = pnl_pct
        else:
            # Legacy: % 기반 계산
            if position['side'] == 'long':
                raw_pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:
                raw_pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

            costs = self.config.entry_cost_pct() + self.config.exit_cost_pct()
            costs += self.config.funding_cost(hours)
            net_pnl_pct = raw_pnl_pct - costs

            leveraged_pnl_pct = net_pnl_pct * self.config.leverage
            pnl_usd = self.config.margin_per_trade * leveraged_pnl_pct

        # PR4 분석용 데이터 추출
        bars_held = position.get('bars_held', 0)
        entry_atr = position.get('atr', 0.0)

        # MFE/MAE 계산 ($ 단위)
        if position['side'] == 'long':
            mfe = (position.get('mfe', position['entry_price']) - position['entry_price'])
            mae = (position['entry_price'] - position.get('mae', position['entry_price']))
            mfe_first_6 = (position.get('mfe_first_6', position['entry_price']) - position['entry_price'])
            mae_first_6 = (position['entry_price'] - position.get('mae_first_6', position['entry_price']))
        else:  # short
            mfe = (position['entry_price'] - position.get('mfe', position['entry_price']))
            mae = (position.get('mae', position['entry_price']) - position['entry_price'])
            mfe_first_6 = (position['entry_price'] - position.get('mfe_first_6', position['entry_price']))
            mae_first_6 = (position.get('mae_first_6', position['entry_price']) - position['entry_price'])

        return Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            pnl_pct=leveraged_pnl_pct,
            pnl_usd=pnl_usd,
            exit_reason=reason,
            bars_held=bars_held,
            mfe=mfe,
            mae=mae,
            mfe_first_6=mfe_first_6,
            mae_first_6=mae_first_6,
            entry_atr=entry_atr,
            # PR6.3: 사이징 로깅
            sl_distance_raw=position.get('sl_distance_raw', 0.0),
            sl_distance_atr=position.get('sl_distance_atr', 0.0),
            clamped=position.get('clamped', False),
            notional=position.get('notional', 0.0),
            cap_reason=position.get('cap_reason', ''),
        )

    def _close_position_partial(self, position: Dict, exit_price: float, reason: str, exit_time, close_ratio: float = 1.0) -> Trade:
        """포지션 부분 청산 (Partial TP용)"""
        hours = (exit_time - position['entry_time']).total_seconds() / 3600

        # PR6: 리스크 고정 사이징 사용 시 qty 기반 계산
        if self.config.use_risk_fixed_sizing and 'qty' in position:
            # 부분 청산할 수량 계산
            partial_qty = position['qty'] * close_ratio
            pnl_usd, pnl_pct = self.config.calculate_pnl_usd(
                position['side'], position['entry_price'], exit_price, partial_qty, hours
            )
            leveraged_pnl_pct = pnl_pct
        else:
            # Legacy: % 기반 계산
            if position['side'] == 'long':
                raw_pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:
                raw_pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

            # 비용 계산 (비례 적용)
            costs = (self.config.entry_cost_pct() + self.config.exit_cost_pct()) * close_ratio
            costs += self.config.funding_cost(hours) * close_ratio

            net_pnl_pct = raw_pnl_pct - costs

            # 레버리지 적용 (비례 적용)
            leveraged_pnl_pct = net_pnl_pct * self.config.leverage
            pnl_usd = self.config.margin_per_trade * leveraged_pnl_pct * close_ratio
            pnl_pct = leveraged_pnl_pct * close_ratio

        # PR4 분석용 데이터 추출
        bars_held = position.get('bars_held', 0)
        entry_atr = position.get('atr', 0.0)

        # MFE/MAE 계산 ($ 단위)
        if position['side'] == 'long':
            mfe = (position.get('mfe', position['entry_price']) - position['entry_price'])
            mae = (position['entry_price'] - position.get('mae', position['entry_price']))
            mfe_first_6 = (position.get('mfe_first_6', position['entry_price']) - position['entry_price'])
            mae_first_6 = (position['entry_price'] - position.get('mae_first_6', position['entry_price']))
        else:  # short
            mfe = (position['entry_price'] - position.get('mfe', position['entry_price']))
            mae = (position.get('mae', position['entry_price']) - position['entry_price'])
            mfe_first_6 = (position['entry_price'] - position.get('mfe_first_6', position['entry_price']))
            mae_first_6 = (position.get('mae_first_6', position['entry_price']) - position['entry_price'])

        return Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            pnl_pct=pnl_pct,  # PR6: 두 경로 모두 pnl_pct 직접 사용
            pnl_usd=pnl_usd,
            exit_reason=reason,
            bars_held=bars_held,
            mfe=mfe,
            mae=mae,
            mfe_first_6=mfe_first_6,
            mae_first_6=mae_first_6,
            entry_atr=entry_atr,
            # PR6.3: 사이징 로깅
            sl_distance_raw=position.get('sl_distance_raw', 0.0),
            sl_distance_atr=position.get('sl_distance_atr', 0.0),
            clamped=position.get('clamped', False),
            notional=position.get('notional', 0.0),
            cap_reason=position.get('cap_reason', ''),
        )

# =============================================================================
# 전략 B: Fib 레벨 기반 (반등 확인 후 시장가 진입)
# =============================================================================
class StrategyB:
    """
    전략 B: Fib 레벨 기반
    - 진입: 15m RSI 다이버전스 존 터치 + 반등 확인 후 시장가 진입
    - SL: ATR 기반
    - TP: 다음 L1 레벨
    """

    def __init__(self, config: Config):
        self.config = config
        self.name = "Strategy B: Fib Level (Bounce Confirm)"

    def run(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> BacktestResult:
        result = BacktestResult(strategy_name=self.name)
        equity = self.config.initial_capital
        result.equity_curve.append(equity)

        long_position = None
        short_position = None

        # 대기 신호 (존 터치됨, 반등 대기)
        pending_long_signal = None   # {'zone_price', 'boundary', 'atr', 'tp', 'touched_time'}
        pending_short_signal = None

        # 쿨다운 카운터
        long_cooldown = 0
        short_cooldown = 0

        warmup = self.config.warmup_bars
        warmup_anchor = self.config.warmup_bars_anchor
        total_bars = len(df_5m) - warmup
        for idx, i in enumerate(range(warmup, len(df_5m))):
            if idx % 1000 == 0:
                print(f"  B Progress: {idx}/{total_bars} ({100*idx/total_bars:.1f}%)", flush=True)
            bar = df_5m.iloc[i]
            current_time = df_5m.index[i]

            # 쿨다운 감소
            if long_cooldown > 0:
                long_cooldown -= 1
            if short_cooldown > 0:
                short_cooldown -= 1

            mask_15m = df_15m.index <= current_time
            if mask_15m.sum() < warmup_anchor:
                continue
            df_15m_slice = df_15m[mask_15m]
            close_arr_15m = df_15m_slice['close'].values.astype(float)

            # 5m 데이터 슬라이스
            df_5m_slice = df_5m.iloc[:i+1]

            # === PR-A: ATR TF for Risk ===
            if self.config.atr_tf_for_risk == '15m':
                atr_15m = df_15m_slice['atr'].iloc[-1] if 'atr' in df_15m_slice.columns and len(df_15m_slice) > 0 else None
                current_atr = atr_15m if atr_15m is not None and np.isfinite(atr_15m) else 500
            else:
                current_atr = bar['atr'] if 'atr' in bar and np.isfinite(bar['atr']) else 500

            # ===== 포지션 청산 체크 =====
            # Long 청산
            if long_position is not None:
                exit_price = None
                exit_reason = None

                # SL 체크
                if bar['low'] <= long_position['sl']:
                    exit_price = long_position['sl']
                    exit_reason = 'SL'
                # TP 체크 (다음 L1 레벨)
                elif bar['high'] >= long_position['tp']:
                    exit_price = long_position['tp']
                    exit_reason = 'TP'

                if exit_price:
                    trade = self._close_position(long_position, exit_price, exit_reason, current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    if exit_reason == 'SL':
                        long_cooldown = self.config.cooldown_bars
                    long_position = None

            # Short 청산
            if short_position is not None:
                exit_price = None
                exit_reason = None

                # SL 체크
                if bar['high'] >= short_position['sl']:
                    exit_price = short_position['sl']
                    exit_reason = 'SL'
                # TP 체크
                elif bar['low'] <= short_position['tp']:
                    exit_price = short_position['tp']
                    exit_reason = 'TP'

                if exit_price:
                    trade = self._close_position(short_position, exit_price, exit_reason, current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    if exit_reason == 'SL':
                        short_cooldown = self.config.cooldown_bars
                    short_position = None

            # ===== 반등 확인 → 진입 =====
            # Long 반등 확인: 양봉 (close > open)
            if pending_long_signal is not None and long_position is None:
                is_bullish_candle = bar['close'] > bar['open']
                bars_since_touch = (current_time - pending_long_signal['touched_time']).total_seconds() / 300
                if bars_since_touch > 3:
                    pending_long_signal = None
                elif is_bullish_candle:
                    entry_price = bar['close']
                    atr = pending_long_signal['atr']
                    tp = pending_long_signal['tp']

                    # ATR 기반 SL
                    sl = entry_price - (atr * self.config.sl_atr_mult)

                    # PR6: 리스크 고정 사이징 + PR6.3 로깅
                    qty_info = self.config.calculate_qty_info(entry_price, sl, atr)
                    qty = qty_info['qty']

                    long_position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp': tp,
                        'atr': atr,
                        'qty': qty,  # PR6
                        # PR6.3: 사이징 로깅
                        'sl_distance_raw': qty_info['sl_distance_raw'],
                        'sl_distance_atr': qty_info['sl_distance_atr'],
                        'clamped': qty_info['clamped'],
                        'notional': qty_info['notional'],
                        'cap_reason': qty_info['cap_reason'],
                    }
                    pending_long_signal = None

            # Short 반등 확인: 음봉 (close < open)
            if pending_short_signal is not None and short_position is None:
                is_bearish_candle = bar['close'] < bar['open']
                bars_since_touch = (current_time - pending_short_signal['touched_time']).total_seconds() / 300
                if bars_since_touch > 3:
                    pending_short_signal = None
                elif is_bearish_candle:
                    entry_price = bar['close']
                    atr = pending_short_signal['atr']
                    tp = pending_short_signal['tp']

                    # PR-FIB-SL: Fib 구조 기반 SL
                    if self.config.use_fib_based_sl and 'fib_level' in pending_short_signal:
                        trigger_fib = pending_short_signal['fib_level'].price
                        # Dynamic Fib 레벨 수집
                        fib_levels = []
                        if self.config.use_dynamic_fib and dynfib_state is not None and dynfib_state.is_valid():
                            fib_levels = get_dynamic_fib_levels(
                                dynfib_state.low, dynfib_state.high,
                                self.config.dynfib_ratios,
                                space=self.config.dynamic_fib_space
                            )
                        sl, sl_buffer, sl_fib_gap = calc_fib_based_sl(
                            entry_price, trigger_fib, atr, fib_levels,
                            side="short", fallback_atr_mult=self.config.fib_sl_fallback_mult
                        )
                        print(f"  [SHORT ENTRY] Entry=${entry_price:,.0f} | SL=${sl:,.0f} (Fib-based, buffer=${sl_buffer:.0f}, gap=${sl_fib_gap:.0f})")
                    else:
                        # ATR 기반 SL
                        sl = entry_price + (atr * self.config.sl_atr_mult)

                    # PR6: 리스크 고정 사이징 + PR6.3 로깅
                    qty_info = self.config.calculate_qty_info(entry_price, sl, atr)
                    qty = qty_info['qty']

                    short_position = {
                        'side': 'short',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp': tp,
                        'atr': atr,
                        'qty': qty,  # PR6
                        # PR6.3: 사이징 로깅
                        'sl_distance_raw': qty_info['sl_distance_raw'],
                        'sl_distance_atr': qty_info['sl_distance_atr'],
                        'clamped': qty_info['clamped'],
                        'notional': qty_info['notional'],
                        'cap_reason': qty_info['cap_reason'],
                    }
                    pending_short_signal = None

            # ===== 새 신호 체크 (존 터치) =====
            # 15m 먼저 체크, 없으면 5m fallback
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # Long 신호 체크 - 바운더리 하단 극단에서만
            if long_position is None and pending_long_signal is None and long_cooldown == 0:
                # 1) 15m 다이버전스 체크 (Regular + Hidden)
                ref = find_oversold_reference(df_15m_slice, threshold=self.config.stoch_rsi_oversold)
                long_price = None

                if ref:
                    reg_price = needed_close_for_regular_bullish(
                        close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                    )
                    hid_price = needed_close_for_hidden_bullish(
                        close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                    )
                    candidates = [p for p in [reg_price, hid_price] if p and p > 0]
                    long_price = min(candidates) if candidates else None

                # 2) 15m에 없으면 5m fallback
                if long_price is None or long_price <= 0:
                    ref = find_oversold_reference(df_5m_slice, threshold=self.config.stoch_rsi_oversold)
                    if ref:
                        reg_price = needed_close_for_regular_bullish(
                            close_arr_5m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                        )
                        hid_price = needed_close_for_hidden_bullish(
                            close_arr_5m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                        )
                        candidates = [p for p in [reg_price, hid_price] if p and p > 0]
                        long_price = min(candidates) if candidates else None

                if long_price and long_price > 0:
                    is_near, fib_level = is_near_l1_level(long_price, atr=current_atr, config=self.config)
                    if is_near and fib_level:
                        tp_level = get_next_l1_above(long_price, self.config)
                        if tp_level:
                            # 존 터치 체크
                            if bar['low'] <= long_price:
                                pending_long_signal = {
                                    'zone_price': long_price,
                                    'fib_level': fib_level,
                                    'atr': current_atr,
                                    'tp': tp_level.price,
                                    'touched_time': current_time,
                                }

            # Short 신호 체크 - Fib 레벨 근처 (PR4-R2: enable_short=False면 스킵)
            if self.config.enable_short and short_position is None and pending_short_signal is None and short_cooldown == 0:
                # 1) 15m 다이버전스 체크 (Regular + Hidden)
                ref = find_overbought_reference(df_15m_slice, threshold=self.config.stoch_rsi_overbought)
                short_price = None

                if ref:
                    reg_price = needed_close_for_regular_bearish(
                        close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                    )
                    hid_price = needed_close_for_hidden_bearish(
                        close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                    )
                    candidates = [p for p in [reg_price, hid_price] if p and p > 0]
                    short_price = max(candidates) if candidates else None

                # 2) 15m에 없으면 5m fallback
                if short_price is None or short_price <= 0:
                    ref = find_overbought_reference(df_5m_slice, threshold=self.config.stoch_rsi_overbought)
                    if ref:
                        reg_price = needed_close_for_regular_bearish(
                            close_arr_5m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                        )
                        hid_price = needed_close_for_hidden_bearish(
                            close_arr_5m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                        )
                        candidates = [p for p in [reg_price, hid_price] if p and p > 0]
                        short_price = max(candidates) if candidates else None

                if short_price and short_price > 0:
                    is_near, fib_level = is_near_l1_level(short_price, atr=current_atr, config=self.config)
                    if is_near and fib_level:
                        tp_level = get_next_l1_below(short_price, self.config)
                        if tp_level:
                            # 존 터치 체크
                            if bar['high'] >= short_price:
                                pending_short_signal = {
                                    'zone_price': short_price,
                                    'fib_level': fib_level,
                                    'atr': current_atr,
                                    'tp': tp_level.price,
                                    'touched_time': current_time,
                                }

        # 미청산 포지션 정리
        if long_position:
            trade = self._close_position(long_position, df_5m.iloc[-1]['close'], 'EOD', df_5m.index[-1])
            result.trades.append(trade)
            equity += trade.pnl_usd
            result.equity_curve.append(equity)

        if short_position:
            trade = self._close_position(short_position, df_5m.iloc[-1]['close'], 'EOD', df_5m.index[-1])
            result.trades.append(trade)
            equity += trade.pnl_usd
            result.equity_curve.append(equity)

        return result

    def _close_position(self, position: Dict, exit_price: float, reason: str, exit_time) -> Trade:
        """포지션 청산 및 Trade 생성"""
        if position['side'] == 'long':
            raw_pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            raw_pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        costs = self.config.entry_cost_pct() + self.config.exit_cost_pct()
        hours = (exit_time - position['entry_time']).total_seconds() / 3600
        costs += self.config.funding_cost(hours)

        net_pnl_pct = raw_pnl_pct - costs
        leveraged_pnl_pct = net_pnl_pct * self.config.leverage
        pnl_usd = self.config.margin_per_trade * leveraged_pnl_pct

        return Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            pnl_pct=leveraged_pnl_pct,
            pnl_usd=pnl_usd,
            exit_reason=reason,
            # PR6.3: 사이징 로깅
            sl_distance_raw=position.get('sl_distance_raw', 0.0),
            sl_distance_atr=position.get('sl_distance_atr', 0.0),
            clamped=position.get('clamped', False),
            notional=position.get('notional', 0.0),
            cap_reason=position.get('cap_reason', ''),
        )

# =============================================================================
# Legacy RUN_MODE Config (점진적 JSON 전환 중 - 추후 제거 예정)
# =============================================================================
def _apply_legacy_run_mode(config: 'Config', run_mode: int) -> 'Config':
    """
    Legacy RUN_MODE if-else 블록.
    JSON config가 없는 모드용 fallback.
    새 모드는 configs/mode{N}.json으로 추가할 것.
    """
    # 계단식 필터 적용 (RUN_MODE 1-5)
    if run_mode >= 1 and run_mode < 6:
        config.use_trend_filter_1h = True
    if run_mode >= 2 and run_mode < 6:
        config.use_trend_filter_4h = True
    if run_mode >= 3 and run_mode < 6:
        config.use_atr_vol_filter = True
    if run_mode >= 4 and run_mode < 6:
        config.use_zone_depth_filter = True
    if run_mode >= 5 and run_mode < 6:
        config.use_hilbert_filter = True
        config.hilbert_block_long_on_bear = True
        config.hilbert_block_short_on_bull = False

    # RUN_MODE 6: 레짐 기반 Hidden Divergence 전략
    if run_mode == 6:
        config.use_regime_hidden_strategy = True
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_hilbert_filter = False

    # RUN_MODE 7: ProbabilityGate v2 (Hilbert 필터 교체)
    if run_mode == 7:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.60
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False

    # RUN_MODE 8: ProbabilityGate v2 + Early Exit (PR3.5)
    if run_mode == 8:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.60
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True

    # RUN_MODE 9: ProbabilityGate v2 + Early Exit + Dynamic Threshold
    if run_mode == 9:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.60
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_dynamic_thr = True
        config.prob_gate_dyn_thr_short_floor = 0.60

    # RUN_MODE 10: PR3.6 - HOT 구간 Early Exit 강화
    if run_mode == 10:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.60
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.use_hot_early_exit = True
        config.early_exit_time_bars_hot = 18
        config.early_exit_mfe_mult_hot = 0.2
        config.stale_loss_mult_hot = 0.2

    # RUN_MODE 11: PR4.1 - SHORT threshold 강화 (0.60 → 0.65)
    if run_mode == 11:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.65
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True

    # RUN_MODE 12: PR4.2-A - conf_min_short만 (HOT 강화 없음)
    if run_mode == 12:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.65
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_conf_filter = True
        config.prob_gate_conf_min_short = 0.40
        config.prob_gate_conf_hot_add = 0.0

    # RUN_MODE 13: PR4.2-B - conf_min_short + HOT 강화
    if run_mode == 13:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.65
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_conf_filter = True
        config.prob_gate_conf_min_short = 0.40
        config.prob_gate_T_hot = 1.8
        config.prob_gate_conf_hot_add = 0.10

    # RUN_MODE 14: PR4.3 - SHORT 타이밍 확인 (ret_n + EMA)
    if run_mode == 14:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.65
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_conf_filter = True
        config.prob_gate_conf_min_short = 0.0
        config.prob_gate_T_hot = 1.8
        config.prob_gate_conf_hot_add = 0.30
        config.prob_gate_use_short_timing = True
        config.prob_gate_short_ret_bars = 3
        config.prob_gate_short_ret_min = -0.0005
        config.prob_gate_short_ema_period = 20

    # RUN_MODE 15: PR4.4 - 드리프트 기반 동적 thr_short (1H EMA200)
    if run_mode == 15:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.65
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_drift_thr = True
        config.prob_gate_drift_ema_period = 200
        config.prob_gate_drift_range_pct = 0.01
        config.prob_gate_thr_short_uptrend = 0.70
        config.prob_gate_thr_short_downtrend = 0.62
        config.prob_gate_thr_short_range = 0.65

    # RUN_MODE 16: PR4.3 + PR4.4.1 결합 (JSON으로 이관됨 - fallback용)
    if run_mode == 16:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.prob_gate_thr_short = 0.65
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_short_timing = True
        config.prob_gate_short_ret_bars = 3
        config.prob_gate_short_ret_min = -0.0005
        config.prob_gate_short_ema_period = 20
        config.prob_gate_use_drift_thr = True
        config.prob_gate_drift_ema_period = 200
        config.prob_gate_drift_enter_pct = 0.012
        config.prob_gate_drift_exit_pct = 0.008
        config.prob_gate_drift_min_bars = 3
        config.prob_gate_drift_use_slope = True
        config.prob_gate_drift_slope_duration = '24h'
        config.prob_gate_drift_slope_bars = _duration_to_bars(
            config.prob_gate_drift_slope_duration, config.context_tf
        )
        config.prob_gate_thr_short_uptrend = 0.70
        config.prob_gate_thr_short_downtrend = 0.65
        config.prob_gate_thr_short_range = 0.65

    # RUN_MODE 20: LONG-only Champion (PR4-R2)
    # - SHORT 비활성화 (p_bull 0.75+로 구조적 차단됨)
    # - StochRSI 25/75 (baseline 개선)
    # - LONG 최적화에 집중
    if run_mode == 20:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        # PR4-R2: LONG-only
        config.enable_short = False
        # PR4-R1: StochRSI 25/75 (baseline 개선)
        config.stoch_rsi_oversold = 25.0
        config.stoch_rsi_overbought = 75.0

    # RUN_MODE 21: SHORT threshold 실험 (PR4-R2 Research)
    # - SHORT threshold를 계단식으로 완화
    # - thr_short_experiment 파라미터로 외부에서 조절
    # - 기본값 0.50 (기존 0.65에서 완화)
    if run_mode == 21:
        config.use_prob_gate = True
        config.prob_gate_temp_mode = 'vol'
        config.prob_gate_p_shrink = 0.6
        config.prob_gate_thr_long = 0.55
        # SHORT threshold 완화 (실험용)
        config.prob_gate_thr_short = 0.50  # 기본 실험값
        config.prob_gate_thr_short_uptrend = 0.50
        config.prob_gate_thr_short_downtrend = 0.50
        config.prob_gate_thr_short_range = 0.50
        config.use_hilbert_filter = False
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_early_exit = True
        config.early_exit_time_bars = 24
        config.early_exit_mfe_mult = 0.3
        config.use_gate_flip_exit = True
        config.use_opposite_div_early = True
        config.prob_gate_use_drift_thr = True
        config.prob_gate_drift_ema_period = 200
        config.prob_gate_drift_enter_pct = 0.012
        config.prob_gate_drift_exit_pct = 0.008
        config.prob_gate_drift_min_bars = 3
        config.prob_gate_drift_use_slope = True
        config.prob_gate_drift_slope_duration = '24h'
        config.prob_gate_drift_slope_bars = _duration_to_bars(
            config.prob_gate_drift_slope_duration, config.context_tf
        )

    return config


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("전략 비교 백테스트")
    print("=" * 70)

    # === 계단식 실험 설정 ===
    # Run 0: Baseline (모든 필터 OFF)
    # Run 1: +1H 역행 금지
    # Run 2: +4H 역행 금지
    # Run 3: +ATR high vol size cut
    # Run 4: +Zone Depth 필터 (검증 완료: r=0.215, p≈0)
    # Run 5: +Hilbert 레짐 필터 (IC=+0.027, Long: BEAR에서 차단) - LEGACY
    # Run 6: 레짐 기반 Hidden Divergence (BULL→Long, BEAR→Short)
    # Run 7: ProbabilityGate v2 (IC=+0.062, Hilbert 교체, OOS 검증 완료)
    # Run 8: ProbabilityGate v2 + Early Exit (PR3.5)

    parser = argparse.ArgumentParser(description='Backtest Strategy Compare')
    parser.add_argument('--run-mode', type=int, default=7, help='RUN_MODE (0-16)')
    parser.add_argument('--start', type=str, default='2021-11-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2021-11-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--use-json', action='store_true', help='Force JSON config loading')
    args = parser.parse_args()
    RUN_MODE = args.run_mode  # 명령줄에서 지정 (기본값 7)

    # =========================================================================
    # Config Loading: JSON 우선, fallback to legacy if-else
    # =========================================================================
    config_loaded_from_json = False
    config_json_path = ROOT / "configs" / f"mode{RUN_MODE}.json"

    if config_json_path.exists() or args.use_json:
        try:
            config = build_config_from_mode(RUN_MODE)
            config_loaded_from_json = True
            print(f"[Config] Loaded from JSON: {config_json_path}")
        except FileNotFoundError as e:
            print(f"[Config] JSON not found, using legacy if-else: {e}")
            config = Config()
        except ValueError as e:
            print(f"[Config] JSON validation error (FAIL-CLOSED): {e}")
            sys.exit(1)
    else:
        config = Config()
        print(f"[Config] Using legacy if-else (no JSON for mode {RUN_MODE})")

    # =========================================================================
    # Legacy if-else 블록 (JSON이 없는 모드용)
    # =========================================================================
    if not config_loaded_from_json:
        config = _apply_legacy_run_mode(config, RUN_MODE)
        config.recompute_derived()  # Duration → bars 재계산

    print(f"\n[설정]")
    print(f"  초기 자산: ${config.initial_capital:,.0f}")
    print(f"  마진/트레이드: ${config.margin_per_trade:,.0f} ({config.margin_pct:.0%})")
    print(f"  레버리지: {config.leverage}x")
    print(f"  포지션 사이즈: ${config.position_size:,.0f}")
    print(f"\n[필터 설정] RUN_MODE={RUN_MODE}")
    print(f"  1H 역행 금지: {config.use_trend_filter_1h}")
    print(f"  4H 역행 금지: {config.use_trend_filter_4h}")
    print(f"  ATR Vol 사이즈 컷: {config.use_atr_vol_filter}")
    print(f"  Zone Depth 필터: {config.use_zone_depth_filter} (min={config.zone_depth_min})")
    print(f"  Hilbert 레짐 필터: {config.use_hilbert_filter}")
    print(f"  레짐 기반 Hidden Div: {config.use_regime_hidden_strategy}")
    print(f"  ProbabilityGate v2: {config.use_prob_gate}")
    if config.use_prob_gate:
        print(f"    - temp_mode: {config.prob_gate_temp_mode}")
        print(f"    - p_shrink: {config.prob_gate_p_shrink}")
        print(f"    - thr_long/short: {config.prob_gate_thr_long}/{config.prob_gate_thr_short}")
    print(f"  SHORT Conf Filter (PR4.2): {config.prob_gate_use_conf_filter}")
    if config.prob_gate_use_conf_filter:
        print(f"    - conf_min_short: {config.prob_gate_conf_min_short}")
        print(f"    - T_hot: {config.prob_gate_T_hot}, conf_hot_add: {config.prob_gate_conf_hot_add}")
    print(f"  SHORT Timing Filter (PR4.3): {config.prob_gate_use_short_timing}")
    if config.prob_gate_use_short_timing:
        print(f"    - ret_bars: {config.prob_gate_short_ret_bars}, ret_min: {config.prob_gate_short_ret_min*100:.2f}%")
        print(f"    - EMA period: {config.prob_gate_short_ema_period}")
    print(f"  Drift-based Threshold (PR4.4.1): {config.prob_gate_use_drift_thr}")
    if config.prob_gate_use_drift_thr:
        print(f"    - 1H EMA period: {config.prob_gate_drift_ema_period}")
        print(f"    - Hysteresis: enter ±{config.prob_gate_drift_enter_pct*100:.1f}%, exit ±{config.prob_gate_drift_exit_pct*100:.1f}%, min {config.prob_gate_drift_min_bars} bars")
        print(f"    - Slope: {'ON' if config.prob_gate_drift_use_slope else 'OFF'} ({config.prob_gate_drift_slope_duration} = {config.prob_gate_drift_slope_bars} bars @ {config.context_tf})")
        print(f"    - thr_short: UP={config.prob_gate_thr_short_uptrend}, DOWN={config.prob_gate_thr_short_downtrend}, RANGE={config.prob_gate_thr_short_range}")
    print(f"  Early Exit (PR3.5): {config.use_early_exit}")
    if config.use_early_exit:
        print(f"    - TimeStop: {config.early_exit_time_bars} bars, MFE < {config.early_exit_mfe_mult}*ATR")
        print(f"    - GateFlip Exit: {config.use_gate_flip_exit}")
        print(f"    - Opposite Div Early: {config.use_opposite_div_early}")
    print(f"  Dynamic Threshold: {config.prob_gate_use_dynamic_thr}")
    if config.prob_gate_use_dynamic_thr:
        print(f"    - SHORT thr floor: {config.prob_gate_dyn_thr_short_floor}")
    print(f"  HOT Early Exit (PR3.6): {config.use_hot_early_exit}")
    if config.use_hot_early_exit:
        print(f"    - HOT TimeStop: {config.early_exit_time_bars_hot} bars, MFE < {config.early_exit_mfe_mult_hot}*ATR")
        print(f"    - HOT StaleLoss: -{config.stale_loss_mult_hot}*ATR")

    START = args.start
    END = args.end

    print(f"\n[기간] {START} ~ {END}")

    # === P0: 재현성 잠금 - config JSON 덤프 ===
    import json
    from dataclasses import asdict
    config_dict = asdict(config)
    config_json_path = f"logs/config_mode{RUN_MODE}_{START}_{END}.json"
    try:
        import os
        os.makedirs("logs", exist_ok=True)
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'run_mode': RUN_MODE,
                'start': START,
                'end': END,
                'config': config_dict,
                'derived': {
                    'cooldown_bars': config.cooldown_bars,
                    'early_exit_time_bars': config.early_exit_time_bars,
                    'trigger_tf': config.trigger_tf,
                    'anchor_tf': config.anchor_tf,
                    'context_tf': config.context_tf,
                }
            }, f, indent=2, default=str)
        print(f"\n[재현성] Config 저장: {config_json_path}")
    except Exception as e:
        print(f"\n[경고] Config 저장 실패: {e}")

    # 데이터 로딩 (config TF 기반 - 하드코딩 제거)
    print(f"\n데이터 로딩 중...")
    print(f"  trigger_tf={config.trigger_tf}, anchor_tf={config.anchor_tf}, context_tf={config.context_tf}")
    df_anchor = load_data(config.anchor_tf, START, END, config)   # 15m (anchor)
    df_trigger = load_data(config.trigger_tf, START, END, config)  # 5m (trigger)
    # Legacy alias (하위호환 - 추후 제거 예정)
    df_15m = df_anchor
    df_5m = df_trigger

    # Context TF 데이터 로드 (추세 필터 + Hilbert + ProbabilityGate용)
    df_context = None  # 1H (context)
    df_context_high = None  # 4H (higher context)
    prob_gate_result = None  # ProbabilityGate 결과 (trigger_tf 인덱스)

    need_context_data = (config.use_trend_filter_1h or config.use_trend_filter_4h or
                         config.use_atr_vol_filter or config.use_hilbert_filter or
                         config.use_regime_hidden_strategy or config.use_prob_gate)

    if need_context_data:
        df_context = load_data(config.context_tf, START, END, config)  # 1H
        df_context_high = load_data('4h', START, END, config)  # 4H (TODO: config.context_tf_high)
        # Legacy alias
        df_1h = df_context
        df_4h = df_context_high
        # 1H ATR 계산 (없으면 추가) - 21h lookback at 1H
        if 'atr' not in df_1h.columns:
            df_1h['atr'] = calc_atr(
                df_1h['high'].values,
                df_1h['low'].values,
                df_1h['close'].values,
                config.trend_lookback_1h  # Use config (default 20~21 bars at 1H)
            )
        # === Precompute 추세/ATR 컬럼 (O(n) 한 번만) ===
        df_1h['trend'] = precompute_trend_column(df_1h, lookback=config.trend_lookback_1h)
        df_4h['trend'] = precompute_trend_column(df_4h, lookback=config.trend_lookback_4h)
        df_1h['atr_pct'] = precompute_atr_percentile_column(df_1h, lookback=config.atr_pct_lookback)
        print(f"  1h: {len(df_1h)} bars (trend/atr_pct precomputed, trend_lookback={config.trend_lookback_1h})")
        print(f"  4h: {len(df_4h)} bars (trend precomputed, trend_lookback={config.trend_lookback_4h})")

        # === ProbabilityGate v2 pre-compute ===
        if config.use_prob_gate and PROB_GATE_AVAILABLE:
            print(f"\n[ProbabilityGate v2] Pre-computing...")
            # P5: Use FeatureStore if enabled
            if config.use_feature_store and FEATURE_STORE_AVAILABLE:
                prob_gate_result = precompute_prob_gate_via_store(df_1h, df_5m, config)
            else:
                prob_gate_result = precompute_prob_gate(df_1h, df_5m, config)
            if prob_gate_result is not None:
                valid_count = prob_gate_result['valid'].sum()
                print(f"  Computed: {len(prob_gate_result):,} bars, valid: {valid_count:,}")
                # 액션 분포 출력
                action_counts = prob_gate_result[prob_gate_result['valid']]['action_str'].value_counts()
                for action, cnt in action_counts.items():
                    print(f"    {action}: {cnt:,} ({100*cnt/valid_count:.1f}%)")

                # === GPT 진단: Sanity Check ===
                r = prob_gate_result
                print("\n[ProbGate Precompute Sanity]")
                print(f"  len_5m: {len(df_5m)}, len_gate: {len(r)}")
                print(f"  valid_rate: {float(r['valid'].mean()):.4f}")
                print(f"  p_bull: min={r['p_bull'].min():.4f}, max={r['p_bull'].max():.4f}, mean={r['p_bull'].mean():.4f}, std={r['p_bull'].std():.4f}")
                print(f"  T: min={r['T'].min():.4f}, max={r['T'].max():.4f}, mean={r['T'].mean():.4f}, std={r['T'].std():.4f}")
                print(f"  action_code_counts: {r['action_code'].value_counts().to_dict()}")
                print(f"  NaN p_bull: {int(r['p_bull'].isna().sum())}, NaN T: {int(r['T'].isna().sum())}")
            else:
                print(f"  [WARN] ProbabilityGate computation failed, disabling filter")
                config.use_prob_gate = False

    print(f"  15m: {len(df_15m)} bars")
    print(f"  5m: {len(df_5m)} bars")

    # 전략 A 실행
    print(f"\n{'='*70}")
    print("전략 A: 15m 진입 + 5m 청산")
    print("='*70")

    strategy_a = StrategyA(config)
    result_a = strategy_a.run(df_15m, df_5m, df_1h, df_4h, prob_gate_result)

    summary_a = result_a.summary()
    for k, v in summary_a.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # === Strategy A RR 분석 (바로 출력) ===
    longs_a = [t for t in result_a.trades if t.side == 'long']
    shorts_a = [t for t in result_a.trades if t.side == 'short']
    wins_a = [t for t in result_a.trades if t.pnl_usd > 0]
    losses_a = [t for t in result_a.trades if t.pnl_usd <= 0]

    print(f"\n{'='*70}")
    print("RR (Risk/Reward) 분석 - Strategy A")
    print("='*70")

    if wins_a and losses_a:
        avg_win = sum(t.pnl_usd for t in wins_a) / len(wins_a)
        avg_loss = abs(sum(t.pnl_usd for t in losses_a) / len(losses_a))
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        win_rate = len(wins_a) / len(result_a.trades)
        loss_rate = 1 - win_rate
        expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)
        breakeven_wr = avg_loss / (avg_win + avg_loss) * 100

        print(f"\n[전체]")
        print(f"  승리 트레이드: {len(wins_a)}개")
        print(f"  패배 트레이드: {len(losses_a)}개")
        print(f"  평균 수익 (Avg Win):  ${avg_win:.2f}")
        print(f"  평균 손실 (Avg Loss): ${avg_loss:.2f}")
        print(f"  RR Ratio: {rr_ratio:.2f}")
        print(f"  기대값 (EV): ${expected_value:.2f}/트레이드")
        print(f"  손익분기 승률: {breakeven_wr:.1f}%")
        print(f"  현재 승률: {win_rate*100:.1f}%")
        print(f"  승률 마진: {win_rate*100 - breakeven_wr:+.1f}%p")

        # LONG RR
        long_wins = [t for t in longs_a if t.pnl_usd > 0]
        long_losses = [t for t in longs_a if t.pnl_usd <= 0]
        if long_wins and long_losses:
            l_avg_win = sum(t.pnl_usd for t in long_wins) / len(long_wins)
            l_avg_loss = abs(sum(t.pnl_usd for t in long_losses) / len(long_losses))
            l_rr = l_avg_win / l_avg_loss if l_avg_loss > 0 else 0
            l_wr = len(long_wins) / len(longs_a)
            l_ev = (l_wr * l_avg_win) - ((1-l_wr) * l_avg_loss)
            l_be = l_avg_loss / (l_avg_win + l_avg_loss) * 100
            print(f"\n[LONG]")
            print(f"  트레이드: {len(longs_a)}개 (W:{len(long_wins)}/L:{len(long_losses)})")
            print(f"  평균 수익: ${l_avg_win:.2f} | 평균 손실: ${l_avg_loss:.2f}")
            print(f"  RR Ratio: {l_rr:.2f}")
            print(f"  승률: {l_wr*100:.1f}% (손익분기: {l_be:.1f}%)")
            print(f"  기대값: ${l_ev:.2f}/트레이드")
            print(f"  상태: {'[+] 양의 기대값' if l_ev > 0 else '[-] 음의 기대값'}")

        # SHORT RR
        short_wins = [t for t in shorts_a if t.pnl_usd > 0]
        short_losses = [t for t in shorts_a if t.pnl_usd <= 0]
        if short_wins and short_losses:
            s_avg_win = sum(t.pnl_usd for t in short_wins) / len(short_wins)
            s_avg_loss = abs(sum(t.pnl_usd for t in short_losses) / len(short_losses))
            s_rr = s_avg_win / s_avg_loss if s_avg_loss > 0 else 0
            s_wr = len(short_wins) / len(shorts_a)
            s_ev = (s_wr * s_avg_win) - ((1-s_wr) * s_avg_loss)
            s_be = s_avg_loss / (s_avg_win + s_avg_loss) * 100
            print(f"\n[SHORT]")
            print(f"  트레이드: {len(shorts_a)}개 (W:{len(short_wins)}/L:{len(short_losses)})")
            print(f"  평균 수익: ${s_avg_win:.2f} | 평균 손실: ${s_avg_loss:.2f}")
            print(f"  RR Ratio: {s_rr:.2f}")
            print(f"  승률: {s_wr*100:.1f}% (손익분기: {s_be:.1f}%)")
            print(f"  기대값: ${s_ev:.2f}/트레이드")
            print(f"  상태: {'[+] 양의 기대값' if s_ev > 0 else '[-] 음의 기대값'}")

        # 연속 기록
        print(f"\n[연속 기록]")
        max_cl, max_cw, cl, cw = 0, 0, 0, 0
        for t in result_a.trades:
            if t.pnl_usd > 0:
                cw += 1; cl = 0; max_cw = max(max_cw, cw)
            else:
                cl += 1; cw = 0; max_cl = max(max_cl, cl)
        print(f"  최대 연속 손실: {max_cl}회")
        print(f"  최대 연속 승리: {max_cw}회")

        # 손실/수익 분포
        loss_amts = sorted([abs(t.pnl_usd) for t in losses_a])
        win_amts = sorted([t.pnl_usd for t in wins_a])
        print(f"\n[손실 분포] 최소=${min(loss_amts):.2f} | 중간=${loss_amts[len(loss_amts)//2]:.2f} | 최대=${max(loss_amts):.2f}")
        print(f"[수익 분포] 최소=${min(win_amts):.2f} | 중간=${win_amts[len(win_amts)//2]:.2f} | 최대=${max(win_amts):.2f}")

    # === Exit Reason 상세 분석 ===
    print(f"\n{'='*70}")
    print("Exit Reason 상세 분석 - Strategy A")
    print("='*70")

    exit_stats = {}
    for t in result_a.trades:
        r = t.exit_reason or 'UNKNOWN'
        if r not in exit_stats:
            exit_stats[r] = {'count': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        exit_stats[r]['count'] += 1
        exit_stats[r]['total_pnl'] += t.pnl_usd
        if t.pnl_usd > 0:
            exit_stats[r]['wins'] += 1
        else:
            exit_stats[r]['losses'] += 1

    print(f"\n{'Exit Reason':<20} {'Count':>6} {'W/L':>10} {'WinRate':>8} {'Avg PnL':>10} {'Total':>12}")
    print("-" * 70)
    for r in sorted(exit_stats.keys()):
        s = exit_stats[r]
        wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
        avg_pnl = s['total_pnl'] / s['count'] if s['count'] > 0 else 0
        print(f"{r:<20} {s['count']:>6} {s['wins']:>4}/{s['losses']:<4} {wr:>7.1f}% ${avg_pnl:>9.2f} ${s['total_pnl']:>11.2f}")

    # Exit Type Summary
    total_trades = len(result_a.trades)
    total_sl = exit_stats.get('SL', {}).get('count', 0)
    total_tp1 = exit_stats.get('TP1', {}).get('count', 0)
    total_tp2 = exit_stats.get('TP2', {}).get('count', 0)
    total_tp3 = exit_stats.get('TP3', {}).get('count', 0)  # PR6.2
    total_div_5m = exit_stats.get('5m_Short_Div', {}).get('count', 0) + exit_stats.get('5m_Long_Div', {}).get('count', 0)
    total_div_15m = exit_stats.get('15m_Short_Div', {}).get('count', 0) + exit_stats.get('15m_Long_Div', {}).get('count', 0)
    total_eod = exit_stats.get('EOD', {}).get('count', 0)

    print(f"\n[Exit Type Summary]")
    print(f"  SL Hit:         {total_sl:>3} ({total_sl/total_trades*100:.1f}%)")
    print(f"  TP1 Hit:        {total_tp1:>3} ({total_tp1/total_trades*100:.1f}%)")
    print(f"  TP2 Hit:        {total_tp2:>3} ({total_tp2/total_trades*100:.1f}%)")
    print(f"  TP3 Hit:        {total_tp3:>3} ({total_tp3/total_trades*100:.1f}%)")  # PR6.2
    print(f"  5m Divergence:  {total_div_5m:>3} ({total_div_5m/total_trades*100:.1f}%)")
    print(f"  15m Divergence: {total_div_15m:>3} ({total_div_15m/total_trades*100:.1f}%)")
    print(f"  EOD:            {total_eod:>3} ({total_eod/total_trades*100:.1f}%)")

    # === PR4: SL 트레이드 상세 분석 ===
    sl_trades = [t for t in result_a.trades if t.exit_reason == 'SL']
    if sl_trades:
        print(f"\n{'='*70}")
        print("PR4 분석: SL 트레이드 상세")
        print("='*70")

        # bars_held 분포
        bars_held_list = [t.bars_held for t in sl_trades]
        print(f"\n[1] bars_to_SL 분포 (5m bars)")
        buckets = [(0, 3), (4, 6), (7, 12), (13, 24), (25, 999)]
        for lo, hi in buckets:
            cnt = len([b for b in bars_held_list if lo <= b <= hi])
            pct = cnt / len(sl_trades) * 100
            label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
            bar_time = f"({lo*5}m~{hi*5}m)" if hi < 999 else f"({lo*5}m+)"
            print(f"  {label:>6} bars {bar_time:>15}: {cnt:>3} ({pct:>5.1f}%)")

        # MFE/MAE first 6 bars
        print(f"\n[2] 첫 6 bars MFE/MAE 분석 (진입 직후 움직임)")
        mfe_6_list = [t.mfe_first_6 for t in sl_trades]
        mae_6_list = [t.mae_first_6 for t in sl_trades]
        atr_list = [t.entry_atr for t in sl_trades if t.entry_atr > 0]

        if mfe_6_list:
            avg_mfe_6 = sum(mfe_6_list) / len(mfe_6_list)
            avg_mae_6 = sum(mae_6_list) / len(mae_6_list)
            avg_atr = sum(atr_list) / len(atr_list) if atr_list else 500
            print(f"  Avg MFE (first 6): ${avg_mfe_6:>7.1f} ({avg_mfe_6/avg_atr:.2f} ATR)")
            print(f"  Avg MAE (first 6): ${avg_mae_6:>7.1f} ({avg_mae_6/avg_atr:.2f} ATR)")

            # MFE가 거의 없고 MAE만 큰 경우 (진입 후 바로 역행)
            no_breath = len([i for i, (mfe, mae) in enumerate(zip(mfe_6_list, mae_6_list))
                            if mfe < avg_atr * 0.1 and mae > avg_atr * 0.3])
            print(f"  '숨 못 쉰' 트레이드 (MFE<0.1ATR, MAE>0.3ATR): {no_breath}/{len(sl_trades)} ({no_breath/len(sl_trades)*100:.1f}%)")

        # 전체 MFE/MAE (보유 기간 전체)
        print(f"\n[3] 전체 MFE/MAE 분석")
        mfe_list = [t.mfe for t in sl_trades]
        mae_list = [t.mae for t in sl_trades]
        if mfe_list:
            avg_mfe = sum(mfe_list) / len(mfe_list)
            avg_mae = sum(mae_list) / len(mae_list)
            print(f"  Avg Total MFE: ${avg_mfe:>7.1f}")
            print(f"  Avg Total MAE: ${avg_mae:>7.1f}")

        # LONG vs SHORT 비교
        print(f"\n[4] SL 트레이드: LONG vs SHORT")
        sl_longs = [t for t in sl_trades if t.side == 'long']
        sl_shorts = [t for t in sl_trades if t.side == 'short']
        for side, trades in [('LONG', sl_longs), ('SHORT', sl_shorts)]:
            if trades:
                avg_bars = sum(t.bars_held for t in trades) / len(trades)
                avg_mfe_6 = sum(t.mfe_first_6 for t in trades) / len(trades)
                avg_mae_6 = sum(t.mae_first_6 for t in trades) / len(trades)
                print(f"  {side}: {len(trades)}개 | avg_bars={avg_bars:.1f} | mfe_6=${avg_mfe_6:.1f} | mae_6=${avg_mae_6:.1f}")

    # === PR6.3: 사이징 분석 리포트 ===
    if config.use_risk_fixed_sizing:
        print(f"\n{'='*70}")
        print("PR6.3 분석: Risk-Fixed Sizing 상세")
        print("='*70")

        # 전체 트레이드 통계
        all_trades = result_a.trades
        trades_with_sizing = [t for t in all_trades if t.sl_distance_raw > 0]

        if trades_with_sizing:
            # Clamp rate
            clamped_count = sum(1 for t in trades_with_sizing if t.clamped)
            clamp_rate = clamped_count / len(trades_with_sizing) * 100
            print(f"\n[1] Clamp Rate (min_sl_distance 적용)")
            print(f"  Total: {len(trades_with_sizing)}개 | Clamped: {clamped_count}개 ({clamp_rate:.1f}%)")

            # sl_distance/atr 분포 (p10, p50, p90)
            sl_atr_vals = sorted([t.sl_distance_atr for t in trades_with_sizing])
            p10 = sl_atr_vals[int(len(sl_atr_vals) * 0.1)] if len(sl_atr_vals) > 10 else sl_atr_vals[0]
            p50 = sl_atr_vals[len(sl_atr_vals) // 2]
            p90 = sl_atr_vals[int(len(sl_atr_vals) * 0.9)] if len(sl_atr_vals) > 10 else sl_atr_vals[-1]
            print(f"\n[2] SL Distance / ATR 분포")
            print(f"  p10: {p10:.2f} | p50: {p50:.2f} | p90: {p90:.2f}")

            # LONG vs SHORT 분리
            longs = [t for t in trades_with_sizing if t.side == 'long']
            shorts = [t for t in trades_with_sizing if t.side == 'short']

            print(f"\n[3] 방향별 분석")
            for side_name, side_trades in [('LONG', longs), ('SHORT', shorts)]:
                if side_trades:
                    side_clamped = sum(1 for t in side_trades if t.clamped)
                    side_clamp_rate = side_clamped / len(side_trades) * 100
                    side_sl_atr = sorted([t.sl_distance_atr for t in side_trades])
                    side_p50 = side_sl_atr[len(side_sl_atr) // 2]
                    side_avg_notional = sum(t.notional for t in side_trades) / len(side_trades)
                    side_sl_loss = [t for t in side_trades if t.exit_reason == 'SL']
                    sl_loss_rate = len(side_sl_loss) / len(side_trades) * 100 if side_trades else 0
                    print(f"  {side_name}: {len(side_trades)}개 | clamp: {side_clamp_rate:.1f}% | sl_atr(p50): {side_p50:.2f} | avg_notional: ${side_avg_notional:.0f} | SL_loss: {sl_loss_rate:.1f}%")

            # Cap Reason 분포
            cap_reasons = {}
            for t in trades_with_sizing:
                reason = t.cap_reason or 'none'
                cap_reasons[reason] = cap_reasons.get(reason, 0) + 1
            print(f"\n[4] Cap Reason 분포")
            for reason, count in sorted(cap_reasons.items()):
                print(f"  {reason}: {count}개 ({count/len(trades_with_sizing)*100:.1f}%)")

            # Clamped 트레이드 vs Non-Clamped 성과 비교
            clamped_trades = [t for t in trades_with_sizing if t.clamped]
            non_clamped = [t for t in trades_with_sizing if not t.clamped]
            if clamped_trades and non_clamped:
                print(f"\n[5] Clamped vs Non-Clamped 성과")
                clamped_pnl = sum(t.pnl_usd for t in clamped_trades)
                non_clamped_pnl = sum(t.pnl_usd for t in non_clamped)
                clamped_sl_rate = sum(1 for t in clamped_trades if t.exit_reason == 'SL') / len(clamped_trades) * 100
                non_clamped_sl_rate = sum(1 for t in non_clamped if t.exit_reason == 'SL') / len(non_clamped) * 100
                print(f"  Clamped ({len(clamped_trades)}개): PnL ${clamped_pnl:.2f} | SL_rate: {clamped_sl_rate:.1f}%")
                print(f"  Non-Clamped ({len(non_clamped)}개): PnL ${non_clamped_pnl:.2f} | SL_rate: {non_clamped_sl_rate:.1f}%")

    print(f"\n{'='*70}")
    print("Strategy B 스킵 (시간 절약)")
    print("='*70")
    return  # Strategy B 스킵

    # 전략 B 실행
    print(f"\n{'='*70}")
    print("전략 B: Fib 레벨 기반")
    print("='*70")

    strategy_b = StrategyB(config)
    result_b = strategy_b.run(df_15m, df_5m)

    summary_b = result_b.summary()
    for k, v in summary_b.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 비교 요약
    print(f"\n{'='*70}")
    print("비교 요약")
    print("='*70")
    print(f"{'전략':<30} {'트레이드':<10} {'승률':<10} {'총 PnL ($)':<15} {'최종 자산 ($)':<15}")
    print("-" * 80)
    print(f"{'A: 15m+5m':<30} {summary_a['total_trades']:<10} {summary_a['win_rate']:.1%}{'':5} ${summary_a['total_pnl_usd']:>10,.2f}{'':5} ${summary_a['final_equity']:>10,.2f}")
    print(f"{'B: Fib Level':<30} {summary_b['total_trades']:<10} {summary_b['win_rate']:.1%}{'':5} ${summary_b['total_pnl_usd']:>10,.2f}{'':5} ${summary_b['final_equity']:>10,.2f}")

    # 트레이드 상세 분석 (전체)
    print(f"\n{'='*70}")
    print("전략 A 트레이드 전체")
    print("='*70")
    for i, t in enumerate(result_a.trades):
        pnl_str = f"+${t.pnl_usd:.2f}" if t.pnl_usd > 0 else f"-${abs(t.pnl_usd):.2f}"
        print(f"{i+1}. {t.side.upper():<5} | Entry: ${t.entry_price:,.0f} | Exit: ${t.exit_price:,.0f} | {t.exit_reason:<15} | {pnl_str}")

    # LONG/SHORT 분리 분석
    longs = [t for t in result_a.trades if t.side == 'long']
    shorts = [t for t in result_a.trades if t.side == 'short']

    print(f"\n{'='*70}")
    print("방향별 분석")
    print("='*70")
    if longs:
        long_wins = len([t for t in longs if t.pnl_usd > 0])
        long_pnl = sum(t.pnl_usd for t in longs)
        print(f"LONG: {len(longs)}개, 승/패: {long_wins}/{len(longs)-long_wins}, 승률: {long_wins/len(longs)*100:.1f}%, PnL: ${long_pnl:+.2f}")
    if shorts:
        short_wins = len([t for t in shorts if t.pnl_usd > 0])
        short_pnl = sum(t.pnl_usd for t in shorts)
        print(f"SHORT: {len(shorts)}개, 승/패: {short_wins}/{len(shorts)-short_wins}, 승률: {short_wins/len(shorts)*100:.1f}%, PnL: ${short_pnl:+.2f}")

    # 청산 사유별 분석 (상세)
    print(f"\n{'='*70}")
    print("청산 사유 상세 분석 - Strategy A")
    print("='*70")

    # Exit reason별 통계 수집
    exit_stats = {}
    for t in result_a.trades:
        r = t.exit_reason or 'UNKNOWN'
        if r not in exit_stats:
            exit_stats[r] = {'count': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'pnls': []}
        exit_stats[r]['count'] += 1
        exit_stats[r]['total_pnl'] += t.pnl_usd
        exit_stats[r]['pnls'].append(t.pnl_usd)
        if t.pnl_usd > 0:
            exit_stats[r]['wins'] += 1
        else:
            exit_stats[r]['losses'] += 1

    print(f"\n{'Exit Reason':<20} {'Count':>6} {'W/L':>10} {'WinRate':>8} {'Avg PnL':>10} {'Total':>12}")
    print("-" * 70)
    for r in sorted(exit_stats.keys()):
        s = exit_stats[r]
        wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
        avg_pnl = s['total_pnl'] / s['count'] if s['count'] > 0 else 0
        print(f"{r:<20} {s['count']:>6} {s['wins']:>4}/{s['losses']:<4} {wr:>7.1f}% ${avg_pnl:>9.2f} ${s['total_pnl']:>11.2f}")

    # 요약
    total_sl = exit_stats.get('SL', {}).get('count', 0)
    total_tp = exit_stats.get('TP1', {}).get('count', 0) + exit_stats.get('TP2', {}).get('count', 0) + exit_stats.get('TP3', {}).get('count', 0)
    total_div_5m = exit_stats.get('5m_Short_Div', {}).get('count', 0) + exit_stats.get('5m_Long_Div', {}).get('count', 0)
    total_div_15m = exit_stats.get('15m_Short_Div', {}).get('count', 0) + exit_stats.get('15m_Long_Div', {}).get('count', 0)
    total_eod = exit_stats.get('EOD', {}).get('count', 0)

    print(f"\n[Exit Type Summary]")
    print(f"  SL Hit:         {total_sl:>3} ({total_sl/len(result_a.trades)*100:.1f}%)")
    print(f"  TP Hit (1+2):   {total_tp:>3} ({total_tp/len(result_a.trades)*100:.1f}%)")
    print(f"  5m Divergence:  {total_div_5m:>3} ({total_div_5m/len(result_a.trades)*100:.1f}%)")
    print(f"  15m Divergence: {total_div_15m:>3} ({total_div_15m/len(result_a.trades)*100:.1f}%)")
    print(f"  EOD:            {total_eod:>3} ({total_eod/len(result_a.trades)*100:.1f}%)")

    # === RR (Risk/Reward) 분석 ===
    print(f"\n{'='*70}")
    print("RR (Risk/Reward) 분석 - Strategy A")
    print("='*70")

    wins = [t for t in result_a.trades if t.pnl_usd > 0]
    losses = [t for t in result_a.trades if t.pnl_usd <= 0]

    if wins and losses:
        avg_win = sum(t.pnl_usd for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl_usd for t in losses) / len(losses))
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        win_rate = len(wins) / len(result_a.trades)
        loss_rate = 1 - win_rate
        expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)

        print(f"\n[전체]")
        print(f"  승리 트레이드: {len(wins)}개")
        print(f"  패배 트레이드: {len(losses)}개")
        print(f"  평균 수익 (Avg Win):  ${avg_win:.2f}")
        print(f"  평균 손실 (Avg Loss): ${avg_loss:.2f}")
        print(f"  RR Ratio: {rr_ratio:.2f}")
        print(f"  기대값 (EV): ${expected_value:.2f}/트레이드")

        # 손익분기 승률 계산
        breakeven_wr = avg_loss / (avg_win + avg_loss) * 100
        print(f"  손익분기 승률: {breakeven_wr:.1f}%")
        print(f"  현재 승률: {win_rate*100:.1f}%")
        print(f"  승률 마진: {win_rate*100 - breakeven_wr:+.1f}%p")

    # LONG/SHORT 분리 RR 분석
    for side_name, side_trades in [("LONG", longs), ("SHORT", shorts)]:
        if not side_trades:
            continue

        side_wins = [t for t in side_trades if t.pnl_usd > 0]
        side_losses = [t for t in side_trades if t.pnl_usd <= 0]

        if side_wins and side_losses:
            s_avg_win = sum(t.pnl_usd for t in side_wins) / len(side_wins)
            s_avg_loss = abs(sum(t.pnl_usd for t in side_losses) / len(side_losses))
            s_rr = s_avg_win / s_avg_loss if s_avg_loss > 0 else 0
            s_wr = len(side_wins) / len(side_trades)
            s_ev = (s_wr * s_avg_win) - ((1-s_wr) * s_avg_loss)
            s_breakeven = s_avg_loss / (s_avg_win + s_avg_loss) * 100

            print(f"\n[{side_name}]")
            print(f"  트레이드: {len(side_trades)}개 (W:{len(side_wins)}/L:{len(side_losses)})")
            print(f"  평균 수익: ${s_avg_win:.2f} | 평균 손실: ${s_avg_loss:.2f}")
            print(f"  RR Ratio: {s_rr:.2f}")
            print(f"  승률: {s_wr*100:.1f}% (손익분기: {s_breakeven:.1f}%)")
            print(f"  기대값: ${s_ev:.2f}/트레이드")
            print(f"  상태: {'[+] 양의 기대값' if s_ev > 0 else '[-] 음의 기대값'}")

    # 최대 연속 손실/승리
    print(f"\n[연속 기록]")
    max_consec_loss = 0
    max_consec_win = 0
    current_loss = 0
    current_win = 0
    for t in result_a.trades:
        if t.pnl_usd > 0:
            current_win += 1
            current_loss = 0
            max_consec_win = max(max_consec_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_consec_loss = max(max_consec_loss, current_loss)
    print(f"  최대 연속 손실: {max_consec_loss}회")
    print(f"  최대 연속 승리: {max_consec_win}회")

    # 손실 분포 분석
    print(f"\n[손실 분포]")
    loss_amounts = [abs(t.pnl_usd) for t in losses]
    if loss_amounts:
        loss_amounts.sort()
        median_loss = loss_amounts[len(loss_amounts)//2]
        print(f"  최소 손실: ${min(loss_amounts):.2f}")
        print(f"  중간 손실: ${median_loss:.2f}")
        print(f"  최대 손실: ${max(loss_amounts):.2f}")
        print(f"  평균 손실: ${sum(loss_amounts)/len(loss_amounts):.2f}")

    # 수익 분포 분석
    print(f"\n[수익 분포]")
    win_amounts = [t.pnl_usd for t in wins]
    if win_amounts:
        win_amounts.sort()
        median_win = win_amounts[len(win_amounts)//2]
        print(f"  최소 수익: ${min(win_amounts):.2f}")
        print(f"  중간 수익: ${median_win:.2f}")
        print(f"  최대 수익: ${max(win_amounts):.2f}")
        print(f"  평균 수익: ${sum(win_amounts)/len(win_amounts):.2f}")

if __name__ == "__main__":
    main()
