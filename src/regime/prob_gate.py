"""
Probability Gate Module (v2)
============================

Temperature-Scaled Sigmoid 기반 확률 게이트.
물리학 흉내가 아닌, 실용적인 확신도 조절 모듈.

핵심 원칙:
- 뜨거운 장 (vol↑) → T↑ → 확률 0.5로 눌림 (과신 방지)
- 차가운 장 (vol↓) → T↓ → 확률 극단 (확신 허용)

사용법:
```python
from src.regime.prob_gate import ProbabilityGate, ProbGateConfig

gate = ProbabilityGate()
result = gate.compute(score_raw, close, high, low)
# result: DataFrame with [T, score_norm, p_bull, action_code, action_str, valid]
```
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Union, Literal


# ============================================================================
# Utility Functions
# ============================================================================

def compute_atr_pct(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 96,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Wilder ATR% (표준 방식)

    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR[t] = (ATR[t-1]*(n-1) + TR[t]) / n  (Wilder smoothing)
    ATR% = ATR / (close + eps)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 96 = 1 day for 15m)
        eps: Small value to prevent division by zero

    Returns:
        ATR% array (same length as input)
    """
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # First TR
    tr[0] = high[0] - low[0]

    # TR for rest
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Wilder smoothing for ATR
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # ATR%
    atr_pct = atr / (close + eps)
    return atr_pct


def rolling_mean(x: np.ndarray, window: int, min_periods: Optional[int] = None) -> np.ndarray:
    """
    Rolling mean (causal)

    Args:
        x: Input array
        window: Window size
        min_periods: Minimum periods required (default: window)

    Returns:
        Rolling mean array (NaN for warmup period)
    """
    if min_periods is None:
        min_periods = window

    n = len(x)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = x[start:i+1]
        valid = segment[np.isfinite(segment)]
        if len(valid) >= min_periods:
            result[i] = np.mean(valid)

    return result


def rolling_std(x: np.ndarray, window: int, min_periods: Optional[int] = None, eps: float = 1e-10) -> np.ndarray:
    """
    Rolling standard deviation (causal)

    Args:
        x: Input array
        window: Window size
        min_periods: Minimum periods required (default: window)
        eps: Small value added to std to prevent division by zero

    Returns:
        Rolling std array (NaN for warmup period)
    """
    if min_periods is None:
        min_periods = window

    n = len(x)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = x[start:i+1]
        valid = segment[np.isfinite(segment)]
        if len(valid) >= min_periods:
            result[i] = np.std(valid, ddof=1) + eps

    return result


def rolling_zscore(x: np.ndarray, window: int = 192, min_periods: Optional[int] = None, eps: float = 1e-10) -> np.ndarray:
    """
    Rolling z-score (CAUSAL)

    - min_periods = window (기본값) → warmup 구간은 NaN
    - std=0 방지: (std + eps)
    - 현재 시점 t는 [t-window+1 : t+1] 범위만 사용

    Args:
        x: Input array
        window: Window size for mean/std calculation
        min_periods: Minimum periods (default: window)
        eps: Small value added to std

    Returns:
        Z-score array (NaN for warmup period)
    """
    if min_periods is None:
        min_periods = window

    n = len(x)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = x[start:i+1]
        valid = segment[np.isfinite(segment)]
        if len(valid) >= min_periods:
            mean = np.mean(valid)
            std = np.std(valid, ddof=1) + eps
            result[i] = (x[i] - mean) / std

    return result


def ema(x: np.ndarray, span: int = 12) -> np.ndarray:
    """
    Exponential moving average (causal)

    Args:
        x: Input array
        span: EMA span

    Returns:
        EMA array
    """
    alpha = 2.0 / (span + 1.0)
    n = len(x)
    result = np.zeros(n)

    # Find first valid value
    first_valid_idx = 0
    for i in range(n):
        if np.isfinite(x[i]):
            first_valid_idx = i
            result[i] = x[i]
            break
        else:
            result[i] = np.nan

    # EMA calculation
    for i in range(first_valid_idx + 1, n):
        if np.isfinite(x[i]):
            if np.isfinite(result[i-1]):
                result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = x[i]
        else:
            result[i] = result[i-1] if np.isfinite(result[i-1]) else np.nan

    return result


def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """
    Overflow-safe sigmoid

    x = clip(x, -60, 60)  # exp overflow 방지
    return 1 / (1 + exp(-x))

    Args:
        x: Input array

    Returns:
        Sigmoid output in [0, 1]
    """
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def apply_probability_calibration(
    p: np.ndarray,
    shrink: float = 1.0,
    bias: float = 0.0,
    clip_eps: float = 0.0
) -> np.ndarray:
    """Post-hoc probability calibration (simple, monotonic).

    목적:
    - 과신(극단 확률)을 줄이기 위해 확률을 0.5로 수축(shrink)
    - 순위(=IC)를 유지한 채, Brier/ECE 개선을 노림

    수식:
        p' = clip(0.5 + shrink*(p-0.5) + bias, clip_eps, 1-clip_eps)

    주의:
    - shrink > 0 여야 단조(monotonic) 변환 → IC 순위 유지
    - shrink < 1 : 더 보수적(0.5 쪽으로)
    - bias는 base-rate 보정용(보통 0 유지)
    """
    if not np.isfinite(shrink) or shrink <= 0:
        raise ValueError(f"shrink must be > 0, got {shrink}")
    p2 = 0.5 + shrink * (p - 0.5) + bias
    if clip_eps > 0:
        p2 = np.clip(p2, clip_eps, 1.0 - clip_eps)
    else:
        p2 = np.clip(p2, 0.0, 1.0)
    return p2


# ============================================================================
# Temperature Computation
# ============================================================================

def compute_temperature_vol(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    n_atr: int = 96,
    W: int = 192,
    a: float = 1.2,
    b: float = 0.6,
    T_min: float = 0.7,
    T_max: float = 3.0,
    ema_span: int = 12
) -> np.ndarray:
    """
    Option A: 변동성 기반 온도 계산

    뜨거운 장(vol↑) → T↑ → 확률 0.5로 눌림 (과신 방지)
    차가운 장(vol↓) → T↓ → 확률 극단 (확신 허용)

    Args:
        close, high, low: Price arrays
        n_atr: ATR calculation period
        W: Z-score window for volatility
        a, b: T = a + b * vol_z
        T_min, T_max: Temperature bounds
        ema_span: EMA smoothing span

    Returns:
        Temperature array
    """
    vol_pct = compute_atr_pct(high, low, close, period=n_atr)
    vol_z = rolling_zscore(vol_pct, window=W)
    T_raw = a + b * vol_z
    T_clipped = np.clip(T_raw, T_min, T_max)
    T_smooth = ema(T_clipped, span=ema_span)
    return T_smooth


def compute_temperature_instability(
    score_norm: np.ndarray,
    W: int = 96,
    W_ref: int = 384,
    T_min: float = 0.7,
    T_max: float = 3.0,
    ema_span: int = 12
) -> np.ndarray:
    """
    Option B: 신호 불안정성 기반 온도 계산

    W_ref > W 필수 (분모가 더 느리게 움직여야 "지금 불안정" 감지)

    Args:
        score_norm: Normalized score array
        W: Rolling std window
        W_ref: Reference window (should be > W, typically 4*W)
        T_min, T_max: Temperature bounds
        ema_span: EMA smoothing span

    Returns:
        Temperature array
    """
    sig_std = rolling_std(score_norm, window=W)
    std_ref = rolling_mean(sig_std, window=W_ref)
    T_raw = sig_std / (std_ref + 1e-10)
    T_clipped = np.clip(T_raw, T_min, T_max)
    T_smooth = ema(T_clipped, span=ema_span)
    return T_smooth


def compute_temperature_fixed(n: int, T_fixed: float = 1.5) -> np.ndarray:
    """
    Baseline: 고정 온도

    Args:
        n: Array length
        T_fixed: Fixed temperature value

    Returns:
        Constant temperature array
    """
    return np.full(n, T_fixed)


# ============================================================================
# Score Normalization
# ============================================================================

def normalize_score(
    score_raw: np.ndarray,
    method: str = 'zscore',
    window: int = 192,
    clip_val: float = 3.0,
    tanh_scale: float = 2.0
) -> np.ndarray:
    """
    스코어 정규화 (scale 안정화)

    주의: score_raw는 upstream에서 방향성(추세/모멘텀) 기반으로 만들어야 함
    이 함수는 scale 안정화 역할만 함

    Args:
        score_raw: Raw directional score
        method: 'zscore' or 'tanh'
        window: Z-score window
        clip_val: Clip value for zscore method
        tanh_scale: Scale for tanh method

    Returns:
        Normalized score array
    """
    if method == 'zscore':
        z = rolling_zscore(score_raw, window=window)
        return np.clip(z, -clip_val, clip_val)
    else:  # tanh
        return np.tanh(score_raw / tanh_scale)


# ============================================================================
# Probability and Action
# ============================================================================

def prob_from_score(score_norm: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    정규화된 스코어 → Bull 확률

    P(bull) = sigmoid_stable(score / T)

    Args:
        score_norm: Normalized score
        T: Temperature array

    Returns:
        P(bull) in [0, 1]
    """
    x = score_norm / (T + 1e-10)
    return sigmoid_stable(x)


def gate_action(p_bull: np.ndarray, thr_long: float = 0.55, thr_short: float = 0.55) -> tuple:
    """
    확률 → Gate Action

    Args:
        p_bull: Bull probability
        thr_long: Threshold for LONG (p_bull > thr_long)
        thr_short: Threshold for SHORT (p_bull < 1 - thr_short)

    Returns:
        (action_code, action_str) where:
        - action_code: +1 (LONG), 0 (FLAT), -1 (SHORT)
        - action_str: 'LONG', 'FLAT', 'SHORT'
    """
    n = len(p_bull)
    action_code = np.zeros(n, dtype=int)
    action_str = np.array(['FLAT'] * n, dtype=object)

    long_mask = p_bull > thr_long
    short_mask = p_bull < (1 - thr_short)

    action_code[long_mask] = 1
    action_code[short_mask] = -1

    action_str[long_mask] = 'LONG'
    action_str[short_mask] = 'SHORT'

    return action_code, action_str


# ============================================================================
# Dynamic Threshold (레짐 기반)
# ============================================================================

@dataclass
class DynamicThresholdConfig:
    """
    동적 Threshold 설정

    핵심 원리:
    - Bull 레짐: LONG 관대, SHORT 엄격
    - Bear 레짐: SHORT 관대, LONG 엄격
    - 불확실성(u) 높으면 양쪽 다 엄격 (SL 제조 구간 회피)
    """
    # Base thresholds
    base_long: float = 0.58
    base_short: float = 0.60  # SHORT는 기본적으로 더 엄격

    # Confidence adjustment (conf = abs(p_bull - 0.5) * 2)
    # 우세 방향은 완화, 반대 방향은 엄격
    conf_favor_delta: float = 0.03   # 우세 방향 완화량
    conf_against_delta: float = 0.04  # 반대 방향 엄격량

    # Uncertainty adjustment (u = clip((T - 1) / 1, 0, 1))
    # 높을수록 양쪽 다 엄격
    uncertainty_delta: float = 0.04

    # Bounds
    thr_min: float = 0.55
    thr_max: float = 0.70

    # Hysteresis (EMA smoothing to prevent flip-flopping)
    ema_span: int = 3  # 1H bars 기준 (15m라면 12)


def compute_dynamic_thresholds(
    p_bull: np.ndarray,
    T: np.ndarray,
    cfg: Optional[DynamicThresholdConfig] = None
) -> tuple:
    """
    레짐 기반 동적 Threshold 계산

    Args:
        p_bull: Bull probability (1H 레짐 기준)
        T: Temperature array

    Returns:
        (thr_long, thr_short) arrays with EMA smoothing
    """
    if cfg is None:
        cfg = DynamicThresholdConfig()

    n = len(p_bull)
    thr_long_raw = np.zeros(n)
    thr_short_raw = np.zeros(n)

    for i in range(n):
        p = p_bull[i] if np.isfinite(p_bull[i]) else 0.5
        t = T[i] if np.isfinite(T[i]) else 1.5

        # Confidence: 레짐 확신도 (0~1)
        conf = abs(p - 0.5) * 2

        # Uncertainty: 시장 불확실성 (0~1)
        u = np.clip((t - 1.0) / 1.0, 0.0, 1.0)

        if p >= 0.5:  # Bull regime
            # LONG: 관대하게 (conf만큼 낮춤), 불확실하면 엄격하게 (u만큼 높임)
            thr_long_raw[i] = cfg.base_long - cfg.conf_favor_delta * conf + cfg.uncertainty_delta * u
            # SHORT: 엄격하게 (conf만큼 높임), 불확실하면 더 엄격하게
            thr_short_raw[i] = cfg.base_short + cfg.conf_against_delta * conf + cfg.uncertainty_delta * u
        else:  # Bear regime
            # LONG: 엄격하게
            thr_long_raw[i] = cfg.base_long + cfg.conf_against_delta * conf + cfg.uncertainty_delta * u
            # SHORT: 관대하게
            thr_short_raw[i] = cfg.base_short - cfg.conf_favor_delta * conf + cfg.uncertainty_delta * u

    # Clip to bounds
    thr_long_raw = np.clip(thr_long_raw, cfg.thr_min, cfg.thr_max)
    thr_short_raw = np.clip(thr_short_raw, cfg.thr_min, cfg.thr_max)

    # Hysteresis: EMA smoothing to prevent flip-flopping
    thr_long = ema(thr_long_raw, span=cfg.ema_span)
    thr_short = ema(thr_short_raw, span=cfg.ema_span)

    return thr_long, thr_short


def gate_action_dynamic(
    p_bull: np.ndarray,
    thr_long: np.ndarray,
    thr_short: np.ndarray
) -> tuple:
    """
    동적 Threshold 적용 Gate Action

    Args:
        p_bull: Bull probability
        thr_long: Dynamic LONG threshold array
        thr_short: Dynamic SHORT threshold array

    Returns:
        (action_code, action_str)
    """
    n = len(p_bull)
    action_code = np.zeros(n, dtype=int)
    action_str = np.array(['FLAT'] * n, dtype=object)

    for i in range(n):
        if not np.isfinite(p_bull[i]):
            continue

        tl = thr_long[i] if np.isfinite(thr_long[i]) else 0.55
        ts = thr_short[i] if np.isfinite(thr_short[i]) else 0.55

        if p_bull[i] > tl:
            action_code[i] = 1
            action_str[i] = 'LONG'
        elif p_bull[i] < (1 - ts):
            action_code[i] = -1
            action_str[i] = 'SHORT'

    return action_code, action_str


# ============================================================================
# Evaluation Utilities (SciPy 의존성 없음)
# ============================================================================

def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman IC (SciPy 없이 구현)

    1. rank 변환
    2. Pearson corr 계산

    Args:
        x, y: Input arrays

    Returns:
        Spearman correlation coefficient
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()

    denom = np.sqrt((rx**2).mean()) * np.sqrt((ry**2).mean())
    return float((rx * ry).mean() / (denom + 1e-12))


def eval_ic(p_bull: np.ndarray, close: np.ndarray, horizons: list = [1, 4, 16]) -> Dict[str, float]:
    """
    다중 horizon Spearman IC

    fwd_ret_h = log(close[t+h]) - log(close[t])  # log-return

    Args:
        p_bull: Bull probability array
        close: Close price array
        horizons: List of forward horizons (bars)

    Returns:
        Dict with IC@{h}bar for each horizon
    """
    results = {}
    for h in horizons:
        if len(close) <= h:
            results[f'IC@{h}bar'] = np.nan
            continue
        fwd_ret = np.log(close[h:]) - np.log(close[:-h])
        ic = spearman_ic(p_bull[:-h], fwd_ret)
        results[f'IC@{h}bar'] = ic
    return results


def eval_brier(p_bull: np.ndarray, actual_up: np.ndarray) -> float:
    """
    Brier score = mean((p - label)^2)

    Lower is better. Perfect = 0, Random = 0.25

    Args:
        p_bull: Predicted probability
        actual_up: Actual outcome (1 if up, 0 if down)

    Returns:
        Brier score
    """
    mask = np.isfinite(p_bull) & np.isfinite(actual_up)
    return float(np.mean((p_bull[mask] - actual_up[mask])**2))


def eval_calibration(p_bull: np.ndarray, actual_up: np.ndarray, n_bins: int = 10) -> Dict:
    """
    ECE-like calibration

    Returns:
        dict: {
            'bins': [(count, mean_p, hit_rate, gap), ...],
            'ece': weighted_abs_gap_sum
        }
    """
    mask = np.isfinite(p_bull) & np.isfinite(actual_up)
    p = p_bull[mask]
    y = actual_up[mask]

    if len(p) == 0:
        return {'bins': [], 'ece': np.nan}

    bins = []
    ece = 0.0
    total = len(p)

    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        if i == n_bins - 1:
            in_bin = (p >= lo) & (p <= hi)
        else:
            in_bin = (p >= lo) & (p < hi)

        count = in_bin.sum()
        if count > 0:
            mean_p = p[in_bin].mean()
            hit_rate = y[in_bin].mean()
            gap = abs(mean_p - hit_rate)
            bins.append((int(count), float(mean_p), float(hit_rate), float(gap)))
            ece += (count / total) * gap
        else:
            bins.append((0, (lo + hi) / 2, np.nan, np.nan))

    return {'bins': bins, 'ece': float(ece)}


# ============================================================================
# Main Class
# ============================================================================

@dataclass
class ProbGateConfig:
    """
    Probability Gate Configuration

    Supports both duration-based and legacy bar-count configuration.

    Duration-based (recommended):
        config = ProbGateConfig(
            timeframe='5m',
            atr_duration='1d',   # -> n_atr=288 at 5m
            vol_duration='2d',   # -> vol_window=576 at 5m
        )

    Legacy (backward compatible):
        config = ProbGateConfig(
            n_atr=96,        # Hardcoded bar count
            vol_window=192,
        )
    """

    # === Duration-based config (NEW) ===
    # If duration is set, it overrides the corresponding bar count
    timeframe: str = '15m'  # Operating timeframe for duration conversion
    atr_duration: Optional[str] = None      # e.g., "1d" -> n_atr
    vol_duration: Optional[str] = None      # e.g., "2d" -> vol_window
    score_duration: Optional[str] = None    # e.g., "2d" -> score_window
    instability_duration: Optional[str] = None      # -> instability_window
    instability_ref_duration: Optional[str] = None  # -> instability_ref_window

    # === Temperature mode ===
    temp_mode: str = 'vol'  # 'vol', 'instability', or 'fixed'
    T_fixed: float = 1.5    # Used when temp_mode='fixed'

    # === Temperature (Option A: volatility) ===
    # Legacy bar counts (used if duration not set)
    n_atr: int = 96         # ATR period (15m: 96 = 1 day)
    vol_window: int = 192   # Z-score window for volatility
    T_a: float = 1.2        # T = a + b * vol_z
    T_b: float = 0.6
    T_min: float = 0.7
    T_max: float = 3.0
    T_ema_span: int = 12

    # === Temperature (Option B: instability) ===
    instability_window: int = 96
    instability_ref_window: int = 384  # 4*W

    # === Score normalization ===
    score_method: str = 'zscore'  # 'zscore' or 'tanh'
    score_window: int = 192
    score_clip: float = 3.0
    score_tanh_scale: float = 2.0

    # === Gate thresholds ===
    thr_long: float = 0.55
    thr_short: float = 0.55

    # === Dynamic threshold ===
    use_dynamic_threshold: bool = False
    dynamic_thr_cfg: Optional[DynamicThresholdConfig] = None

    # === Probability calibration (post-hoc) ===
    # - shrink < 1.0: 확률을 0.5로 수축 → 과신 완화(Brier/ECE 개선용)
    # - shrink > 1.0: 확률을 더 극단으로(일반적으로 비권장)
    # - bias: base-rate 보정(기본 0 유지)
    # NOTE: shrink > 0이면 단조 변환이므로 IC 순위는 유지됨.
    # CHAMPION (1H OOS): p_shrink=0.6 → IC +24%, Brier -5.7%, ECE -28.6% vs Baseline
    p_shrink: float = 1.0  # 1H 권장: 0.6
    p_bias: float = 0.0
    p_clip_eps: float = 0.0

    def __post_init__(self):
        """Resolve duration strings to bar counts."""
        try:
            from ..utils.timeframe import duration_to_bars
        except ImportError:
            # Fallback if utils not available yet
            return

        if self.atr_duration is not None:
            self.n_atr = duration_to_bars(self.atr_duration, self.timeframe)
        if self.vol_duration is not None:
            self.vol_window = duration_to_bars(self.vol_duration, self.timeframe)
        if self.score_duration is not None:
            self.score_window = duration_to_bars(self.score_duration, self.timeframe)
        if self.instability_duration is not None:
            self.instability_window = duration_to_bars(self.instability_duration, self.timeframe)
        if self.instability_ref_duration is not None:
            self.instability_ref_window = duration_to_bars(self.instability_ref_duration, self.timeframe)


class ProbabilityGate:
    """
    Temperature-Scaled Probability Gate

    Usage:
        gate = ProbabilityGate()
        result = gate.compute(score_raw, close, high, low)

    Result columns:
        - T: Temperature
        - score_norm: Normalized score
        - p_bull: Bull probability [0, 1]
        - thr_long: LONG threshold (static or dynamic)
        - thr_short: SHORT threshold (static or dynamic)
        - action_code: +1 (LONG), 0 (FLAT), -1 (SHORT)
        - action_str: 'LONG', 'FLAT', 'SHORT'
        - valid: True after warmup (no NaN)
    """

    def __init__(self, cfg: Optional[ProbGateConfig] = None):
        self.cfg = cfg or ProbGateConfig()

    def compute(
        self,
        score_raw: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series]
    ) -> pd.DataFrame:
        """
        Compute probability gate

        Args:
            score_raw: Raw directional score (from upstream)
            close, high, low: Price data

        Returns:
            DataFrame with T, score_norm, p_bull, action_code, action_str, valid
        """
        # Extract index if pandas
        index = None
        if isinstance(close, pd.Series):
            index = close.index

        # Convert to numpy
        score_raw = np.asarray(score_raw, dtype=float)
        close = np.asarray(close, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        n = len(close)
        cfg = self.cfg

        # 1. Normalize score
        score_norm = normalize_score(
            score_raw,
            method=cfg.score_method,
            window=cfg.score_window,
            clip_val=cfg.score_clip,
            tanh_scale=cfg.score_tanh_scale
        )

        # 2. Compute temperature
        if cfg.temp_mode == 'vol':
            T = compute_temperature_vol(
                close, high, low,
                n_atr=cfg.n_atr,
                W=cfg.vol_window,
                a=cfg.T_a,
                b=cfg.T_b,
                T_min=cfg.T_min,
                T_max=cfg.T_max,
                ema_span=cfg.T_ema_span
            )
        elif cfg.temp_mode == 'instability':
            T = compute_temperature_instability(
                score_norm,
                W=cfg.instability_window,
                W_ref=cfg.instability_ref_window,
                T_min=cfg.T_min,
                T_max=cfg.T_max,
                ema_span=cfg.T_ema_span
            )
        else:  # fixed
            T = compute_temperature_fixed(n, cfg.T_fixed)

        # 3. Compute probability
        p_bull = prob_from_score(score_norm, T)
        # Post-hoc calibration (PR3): reduce overconfidence without changing rank ordering
        if (cfg.p_shrink != 1.0) or (cfg.p_bias != 0.0) or (cfg.p_clip_eps != 0.0):
            p_bull = apply_probability_calibration(
                p_bull,
                shrink=cfg.p_shrink,
                bias=cfg.p_bias,
                clip_eps=cfg.p_clip_eps,
            )

        # 4. Compute action (static or dynamic threshold)
        if cfg.use_dynamic_threshold:
            dyn_cfg = cfg.dynamic_thr_cfg or DynamicThresholdConfig()
            thr_long_arr, thr_short_arr = compute_dynamic_thresholds(p_bull, T, dyn_cfg)
            action_code, action_str = gate_action_dynamic(p_bull, thr_long_arr, thr_short_arr)
        else:
            thr_long_arr = np.full(n, cfg.thr_long)
            thr_short_arr = np.full(n, cfg.thr_short)
            action_code, action_str = gate_action(p_bull, cfg.thr_long, cfg.thr_short)

        # 5. Valid mask (after warmup)
        valid = np.isfinite(T) & np.isfinite(score_norm) & np.isfinite(p_bull)

        # Build DataFrame
        result = pd.DataFrame({
            'T': T,
            'score_norm': score_norm,
            'p_bull': p_bull,
            'thr_long': thr_long_arr,
            'thr_short': thr_short_arr,
            'action_code': action_code,
            'action_str': action_str,
            'valid': valid
        })

        if index is not None:
            result.index = index

        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def create_simple_direction_score(close: np.ndarray, momentum_period: int = 20) -> np.ndarray:
    """
    간단한 방향성 스코어 생성 (테스트용)

    실제 사용 시에는 upstream에서 더 정교한 스코어 사용 권장

    Args:
        close: Close price array
        momentum_period: Momentum calculation period

    Returns:
        Direction score (positive = bullish, negative = bearish)
    """
    n = len(close)
    score = np.zeros(n)

    for i in range(momentum_period, n):
        score[i] = (close[i] / close[i - momentum_period] - 1) * 100

    return score
