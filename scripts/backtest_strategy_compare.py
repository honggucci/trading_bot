# backtest_strategy_compare.py
# 전략 A vs 전략 B 비교 백테스트
# - 전략 A: 15m 진입 + 5m 청산 (반대 다이버전스)
# - 전략 B: Fib 레벨 기반 (L1 근처 진입, 다음 L1 TP)

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Tuple

# 프로젝트 루트
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.context.cycle_anchor import get_fractal_fib_levels, get_nearby_fib_levels, FibLevel

# =============================================================================
# 설정
# =============================================================================
@dataclass
class Config:
    # 자산 & 레버리지
    initial_capital: float = 10000.0
    margin_pct: float = 0.01          # 1%
    leverage: float = 25.0

    # RSI 설정
    rsi_period: int = 14
    stoch_rsi_period: int = 26

    # 비용
    fee_bps: float = 4.0              # 0.04%
    slippage_bps: float = 5.0         # 0.05%
    funding_rate: float = 0.0001      # 0.01% per 8h

    # Fib 설정
    fib_max_depth: int = 1            # L0 + L1
    fib_tolerance_pct: float = 0.01   # 1% (전략 B용, 고정값)
    fib_atr_mult: float = 1.5         # ATR 기반 tolerance (전략 A): tolerance = ATR * mult / price

    # 쿨다운 설정 (손절 후 재진입 제한)
    cooldown_bars: int = 12           # 5m * 12 = 1시간 쿨다운

    # ATR 기반 SL
    sl_atr_mult: float = 1.5          # SL = entry ± 1.5*ATR (bc3e19a 설정)

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

# =============================================================================
# RSI 계산 (순수 Python)
# =============================================================================
def calc_rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI 계산"""
    n = len(close)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

def calc_stoch_rsi(rsi: np.ndarray, period: int = 14, k_period: int = 3) -> np.ndarray:
    """StochRSI %D 계산"""
    n = len(rsi)
    stoch_d = np.full(n, np.nan)

    for i in range(period + k_period - 1, n):
        window = rsi[i-period+1:i+1]
        if np.all(np.isfinite(window)):
            min_rsi = np.min(window)
            max_rsi = np.max(window)
            if max_rsi > min_rsi:
                stoch_k = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
            else:
                stoch_k = 50.0
        else:
            continue
        stoch_d[i] = stoch_k

    return stoch_d

def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 21) -> np.ndarray:
    """ATR 계산 (21선 고정)"""
    n = len(close)
    atr = np.full(n, np.nan)

    if n < period + 1:
        return atr

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr

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
    """Regular Bullish: 가격 < ref, RSI > ref"""
    eps = 1e-8
    U = ref_price - max(eps, abs(ref_price) * 1e-6)
    L = U * 0.9

    if not np.isfinite(U) or L >= U or L <= 0:
        return None

    rsi_U = _rsi_at_price(close_arr, U, rsi_period)
    if not np.isfinite(rsi_U) or rsi_U <= ref_rsi:
        return None

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

    result = min(hi, U)
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
    """Regular Bearish: 가격 > ref, RSI < ref"""
    eps = 1e-8
    L = ref_price + max(eps, abs(ref_price) * 1e-6)
    U = L * 1.1

    if not np.isfinite(L) or L >= U:
        return None

    rsi_L = _rsi_at_price(close_arr, L, rsi_period)
    if not np.isfinite(rsi_L) or rsi_L >= ref_rsi:
        return None

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

    result = max(lo, L)
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
    """
    eps = 1e-8
    L = ref_price + max(eps, abs(ref_price) * 1e-6)
    U = L * 1.1

    if not np.isfinite(L) or L >= U:
        return None

    # L에서 RSI가 ref_rsi보다 낮아야 함
    rsi_L = _rsi_at_price(close_arr, L, rsi_period)
    if not np.isfinite(rsi_L) or rsi_L >= ref_rsi:
        return None

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

    result = max(lo, L)
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
    """
    eps = 1e-8
    U = ref_price - max(eps, abs(ref_price) * 1e-6)
    L = U * 0.9

    if not np.isfinite(U) or L >= U or L <= 0:
        return None

    # U에서 RSI가 ref_rsi보다 높아야 함
    rsi_U = _rsi_at_price(close_arr, U, rsi_period)
    if not np.isfinite(rsi_U) or rsi_U <= ref_rsi:
        return None

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

    result = min(hi, U)
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi <= ref_rsi:
        return None
    return float(result) if result > 0 else None


# =============================================================================
# 참조점 찾기
# =============================================================================
def find_oversold_reference(df: pd.DataFrame, lookback: int = 100) -> Optional[Dict]:
    """최근 oversold 구간의 저점"""
    if len(df) < 10:
        return None

    d = df['stoch_d'].values[-lookback:] if len(df) >= lookback else df['stoch_d'].values
    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] <= 20.0:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= 20.0:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 현재 과매도일 때만 전 세그먼트 사용
    current_oversold = np.isfinite(d[-1]) and d[-1] <= 20.0
    if not current_oversold:
        return None  # 현재 과매도 아니면 참조점 없음

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

def find_overbought_reference(df: pd.DataFrame, lookback: int = 100) -> Optional[Dict]:
    """최근 overbought 구간의 고점"""
    if len(df) < 10:
        return None

    d = df['stoch_d'].values[-lookback:] if len(df) >= lookback else df['stoch_d'].values
    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] >= 80.0:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] >= 80.0:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 현재 과매수일 때만 전 세그먼트 사용
    current_overbought = np.isfinite(d[-1]) and d[-1] >= 80.0
    if not current_overbought:
        return None  # 현재 과매수 아니면 참조점 없음

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


def find_oversold_reference_hybrid(
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    lookback: int = 100
) -> Optional[Dict]:
    """
    15m StochRSI 세그먼트 + 5m RSI 레퍼런스
    - 15m StochRSI로 과매도 세그먼트 시간 구간 찾기
    - 해당 시간 구간의 5m 데이터에서 종가 최저점의 RSI를 레퍼런스로 사용
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
        if np.isfinite(d[i]) and d[i] <= 20.0:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= 20.0:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 현재 과매도일 때만 전 세그먼트 사용
    current_oversold = np.isfinite(d[-1]) and d[-1] <= 20.0
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
    lookback: int = 100
) -> Optional[Dict]:
    """
    15m StochRSI 세그먼트 + 5m RSI 레퍼런스
    - 15m StochRSI로 과매수 세그먼트 시간 구간 찾기
    - 해당 시간 구간의 5m 데이터에서 종가 최고점의 RSI를 레퍼런스로 사용
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
        if np.isfinite(d[i]) and d[i] >= 80.0:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] >= 80.0:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]
    if not segments:
        return None

    # 현재 과매수일 때만 전 세그먼트 사용
    current_overbought = np.isfinite(d[-1]) and d[-1] >= 80.0
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
# 전역 캐시: 한번만 계산
_FIB_LEVELS_CACHE: List[FibLevel] = []

def _ensure_fib_cache():
    """Fib 레벨 캐시 초기화"""
    global _FIB_LEVELS_CACHE
    if not _FIB_LEVELS_CACHE:
        # 1W Fib 앵커 확장 범위 (Fib 0 ~ 8.0)
        # FIB_0=3120, RANGE=17530, MAX=8 → $3,120 ~ $143,360
        _FIB_LEVELS_CACHE = get_fractal_fib_levels((3120, 143360), max_depth=1)
        _FIB_LEVELS_CACHE.sort(key=lambda lvl: lvl.price)

def get_l1_boundary(price: float) -> Optional[Tuple[float, float, FibLevel, FibLevel]]:
    """
    가격이 속한 L1 바운더리 반환 (캐시 사용)
    Returns: (lower_price, upper_price, lower_level, upper_level)
    """
    _ensure_fib_cache()

    below = [lvl for lvl in _FIB_LEVELS_CACHE if lvl.price < price]
    above = [lvl for lvl in _FIB_LEVELS_CACHE if lvl.price > price]

    if not below or not above:
        return None

    lower = below[-1]  # 가장 가까운 아래 레벨
    upper = above[0]   # 가장 가까운 위 레벨

    return (lower.price, upper.price, lower, upper)

def is_near_boundary_edge(price: float, edge_pct: float = 0.15) -> Tuple[bool, Optional[Tuple[float, float]], str]:
    """
    가격이 L1 바운더리의 극단(상단/하단 15%)에 있는지 체크

    Args:
        price: 체크할 가격
        edge_pct: 극단 범위 (0.15 = 상/하단 15%)

    Returns:
        (극단 여부, (lower, upper), 위치='lower'|'upper'|'middle')
    """
    boundary = get_l1_boundary(price)
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


def is_within_l1_boundary(price: float) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """가격이 L1 바운더리 내에 있는지 체크 (하위 호환)"""
    boundary = get_l1_boundary(price)
    if boundary is None:
        return False, None

    lower_price, upper_price, _, _ = boundary
    is_within = lower_price <= price <= upper_price
    return is_within, (lower_price, upper_price)

def is_near_l1_level(price: float, atr: float = None, atr_mult: float = 1.0, tolerance_pct: float = None) -> Tuple[bool, Optional[FibLevel]]:
    """
    가격이 L1 레벨 근처인지 체크 (캐시 사용)

    ATR 기반 tolerance:
    - tolerance = (ATR * atr_mult) / price
    - 가격이 높아지면 ATR도 커지지만, 비율은 유지됨
    - atr_mult로 민감도 조절 (기본 1.0 = 1 ATR 범위)
    """
    _ensure_fib_cache()

    l1_levels = [lvl for lvl in _FIB_LEVELS_CACHE if lvl.depth <= 1]

    if not l1_levels:
        return False, None

    closest = min(l1_levels, key=lambda lvl: abs(lvl.price - price))

    # ATR 기반 tolerance 계산
    if atr is not None and atr > 0:
        # tolerance = ATR * mult / price (가격 대비 ATR 비율)
        effective_tolerance = (atr * atr_mult) / price
    elif tolerance_pct is not None:
        effective_tolerance = tolerance_pct
    else:
        effective_tolerance = 0.01  # 기본값 1%

    distance_pct = abs(closest.price - price) / closest.price

    if distance_pct <= effective_tolerance:
        return True, closest
    return False, None

def get_next_l1_above(price: float) -> Optional[FibLevel]:
    """현재 가격 위의 다음 L1 레벨"""
    _ensure_fib_cache()
    above = [lvl for lvl in _FIB_LEVELS_CACHE if lvl.price > price and lvl.depth <= 1]
    return above[0] if above else None

def get_next_l1_below(price: float) -> Optional[FibLevel]:
    """현재 가격 아래의 다음 L1 레벨"""
    _ensure_fib_cache()
    below = [lvl for lvl in _FIB_LEVELS_CACHE if lvl.price < price and lvl.depth <= 1]
    return below[-1] if below else None

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

    # 인디케이터
    df['rsi'] = calc_rsi_wilder(df['close'].values, period=config.rsi_period)
    df['stoch_d'] = calc_stoch_rsi(df['rsi'].values, period=config.stoch_rsi_period, k_period=3)
    df['atr'] = calc_atr(df['high'].values, df['low'].values, df['close'].values, period=21)

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

    def run(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> BacktestResult:
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

        total_bars = len(df_5m) - 50
        for idx, i in enumerate(range(50, len(df_5m))):
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
            if mask_15m.sum() < 50:
                continue
            df_15m_slice = df_15m[mask_15m]

            # 15m 바 전환 감지
            current_15m_bar_time = df_15m_slice.index[-1]
            if current_15m_bar_time != last_15m_bar_time:
                # 새 15m 바 진입 - 상태 전환 체크
                current_stoch = df_15m_slice['stoch_d'].iloc[-1] if len(df_15m_slice) > 0 else 50.0
                if not np.isfinite(current_stoch):
                    current_stoch = 50.0

                # 과매도 진입 순간: 이전 > 20 → 현재 ≤ 20
                long_signal_triggered = (prev_15m_stoch > 20.0 and current_stoch <= 20.0)
                # 과매수 진입 순간: 이전 < 80 → 현재 ≥ 80
                short_signal_triggered = (prev_15m_stoch < 80.0 and current_stoch >= 80.0)

                prev_15m_stoch = current_stoch
                last_15m_bar_time = current_15m_bar_time
            else:
                # 동일 15m 바 내 후속 5m 바 - 신호 비활성화 (한 번만 체크)
                long_signal_triggered = False
                short_signal_triggered = False

            # 5m 데이터 슬라이스
            df_5m_slice = df_5m.iloc[:i+1]

            # 현재 ATR (5m ATR 사용)
            current_atr = bar['atr'] if 'atr' in bar and np.isfinite(bar['atr']) else 500

            # ===== 포지션 청산 체크 =====
            # Long 청산: SL 또는 5m 숏 다이버전스
            if long_position is not None:
                # SL 체크
                if bar['low'] <= long_position['sl']:
                    exit_price = long_position['sl']
                    trade = self._close_position(long_position, exit_price, 'SL', current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    long_cooldown = self.config.cooldown_bars
                    long_position = None
                # 5m 숏 다이버전스 → 롱 청산 (Regular + Hidden)
                elif self._check_short_divergence(df_5m_slice):
                    exit_price = bar['close']
                    trade = self._close_position(long_position, exit_price, '5m_Short_Div', current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    long_position = None

            # Short 청산: SL 또는 5m 롱 다이버전스
            if short_position is not None:
                # SL 체크
                if bar['high'] >= short_position['sl']:
                    exit_price = short_position['sl']
                    trade = self._close_position(short_position, exit_price, 'SL', current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    short_cooldown = self.config.cooldown_bars
                    short_position = None
                # 5m 롱 다이버전스 → 숏 청산 (Regular + Hidden)
                elif self._check_long_divergence(df_5m_slice):
                    exit_price = bar['close']
                    trade = self._close_position(short_position, exit_price, '5m_Long_Div', current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    short_position = None

            # ===== 반등 확인 → 진입 =====
            # Long 반등 확인: 양봉 (close > open)
            if pending_long_signal is not None and long_position is None:
                is_bullish_candle = bar['close'] > bar['open']
                # 신호 유효기간 체크 (2봉 이내)
                bars_since_touch = (current_time - pending_long_signal['touched_time']).total_seconds() / 300
                if bars_since_touch > 3:
                    pending_long_signal = None  # 신호 만료
                elif is_bullish_candle:
                    # 반등 확인! 시장가 진입
                    entry_price = bar['close']
                    atr = pending_long_signal['atr']

                    # ATR 기반 SL
                    sl = entry_price - (atr * self.config.sl_atr_mult)

                    # Fib TP: TP1 (첫번째 L1), TP2 (두번째 L1)
                    fib1 = get_next_l1_above(entry_price)
                    tp1 = fib1.price if fib1 else entry_price * 1.01
                    fib2 = get_next_l1_above(tp1 + 1) if fib1 else None
                    tp2 = fib2.price if fib2 else entry_price * 1.02

                    if len(result.trades) < 5:  # 처음 5개만 로그
                        print(f"  [LONG ENTRY] {current_time}")
                        print(f"    Entry: ${entry_price:,.0f} | SL: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f}")
                        print(f"    Candle: O=${bar['open']:,.0f} C=${bar['close']:,.0f} (Bullish)")

                    long_position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'atr': atr,  # 트레일링 스탑용
                        'remaining': 1.0,  # 100% 남음
                    }
                    pending_long_signal = None

            # Short 반등 확인: 음봉 (close < open)
            if pending_short_signal is not None and short_position is None:
                is_bearish_candle = bar['close'] < bar['open']
                bars_since_touch = (current_time - pending_short_signal['touched_time']).total_seconds() / 300
                if bars_since_touch > 3:
                    pending_short_signal = None  # 신호 만료
                elif is_bearish_candle:
                    # 반등 확인! 시장가 진입
                    entry_price = bar['close']
                    atr = pending_short_signal['atr']

                    # ATR 기반 SL
                    sl = entry_price + (atr * self.config.sl_atr_mult)

                    # Fib TP: TP1 (첫번째 L1 아래), TP2 (두번째 L1 아래)
                    fib1 = get_next_l1_below(entry_price)
                    tp1 = fib1.price if fib1 else entry_price * 0.99
                    fib2 = get_next_l1_below(tp1 - 1) if fib1 else None
                    tp2 = fib2.price if fib2 else entry_price * 0.98

                    short_position = {
                        'side': 'short',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'atr': atr,  # 트레일링 스탑용
                        'remaining': 1.0,  # 100% 남음
                    }
                    pending_short_signal = None

            # ===== 새 신호 체크 (존 터치) =====
            # 15m 먼저 체크, 없으면 5m fallback
            close_arr_15m = df_15m_slice['close'].values.astype(float)
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # Long 신호 체크 - 과매도 진입 순간에만 다이버전스 체크
            if long_position is None and pending_long_signal is None and long_cooldown == 0:
                long_price = None
                signal_tf = '15m'

                # 과매도 진입 순간 (상태 전환)에만 다이버전스 체크
                if long_signal_triggered:
                    # 15m에서 레퍼런스 찾기
                    ref = find_oversold_reference(df_15m_slice)
                    if ref:
                        # Regular만 사용 (Hidden 제거)
                        long_price = needed_close_for_regular_bullish(
                            close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                        )

                if long_price and long_price > 0:
                    # 고정 1% tolerance로 원복 (bc3e19a 설정)
                    is_near, fib_level = is_near_l1_level(long_price, tolerance_pct=0.01)
                    if is_near and fib_level:
                        # 존 터치 체크
                        if bar['low'] <= long_price:
                            pending_long_signal = {
                                'zone_price': long_price,
                                'fib_level': fib_level,
                                'atr': current_atr,
                                'touched_time': current_time,
                            }
                            if len(result.trades) < 5:  # 처음 5개만 로그
                                print(f"  [LONG SIGNAL] {current_time} ({signal_tf})")
                                print(f"    Div Price: ${long_price:,.0f} | Bar Low: ${bar['low']:,.0f}")
                                print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio})")

            # Short 신호 체크 - 과매수 진입 순간에만 다이버전스 체크
            if short_position is None and pending_short_signal is None and short_cooldown == 0:
                short_price = None
                signal_tf = '15m'

                # 과매수 진입 순간 (상태 전환)에만 다이버전스 체크
                if short_signal_triggered:
                    # 15m에서 레퍼런스 찾기
                    ref = find_overbought_reference(df_15m_slice)
                    if ref:
                        # Regular만 사용 (Hidden 제거)
                        short_price = needed_close_for_regular_bearish(
                            close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                        )

                if short_price and short_price > 0:
                    # 고정 1% tolerance로 원복 (bc3e19a 설정)
                    is_near, fib_level = is_near_l1_level(short_price, tolerance_pct=0.01)
                    if is_near and fib_level:
                        # 존 터치 체크
                        if bar['high'] >= short_price:
                            pending_short_signal = {
                                'zone_price': short_price,
                                'fib_level': fib_level,
                                'atr': current_atr,
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

    def _check_long_divergence(self, df: pd.DataFrame) -> bool:
        """5m 롱 다이버전스 체크 - Regular만 사용"""
        ref = find_oversold_reference(df)
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
        ref = find_overbought_reference(df)
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
        if position['side'] == 'long':
            raw_pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            raw_pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        # 비용 계산
        costs = self.config.entry_cost_pct() + self.config.exit_cost_pct()
        hours = (exit_time - position['entry_time']).total_seconds() / 3600
        costs += self.config.funding_cost(hours)

        net_pnl_pct = raw_pnl_pct - costs

        # 레버리지 적용
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
        )

    def _close_position_partial(self, position: Dict, exit_price: float, reason: str, exit_time, close_ratio: float = 1.0) -> Trade:
        """포지션 부분 청산 (Partial TP용)"""
        if position['side'] == 'long':
            raw_pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            raw_pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        # 비용 계산 (비례 적용)
        costs = (self.config.entry_cost_pct() + self.config.exit_cost_pct()) * close_ratio
        hours = (exit_time - position['entry_time']).total_seconds() / 3600
        costs += self.config.funding_cost(hours) * close_ratio

        net_pnl_pct = raw_pnl_pct - costs

        # 레버리지 적용 (비례 적용)
        leveraged_pnl_pct = net_pnl_pct * self.config.leverage
        pnl_usd = self.config.margin_per_trade * leveraged_pnl_pct * close_ratio

        return Trade(
            entry_time=position['entry_time'],
            exit_time=exit_time,
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            pnl_pct=leveraged_pnl_pct * close_ratio,
            pnl_usd=pnl_usd,
            exit_reason=reason,
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

        total_bars = len(df_5m) - 50
        for idx, i in enumerate(range(50, len(df_5m))):
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
            if mask_15m.sum() < 50:
                continue
            df_15m_slice = df_15m[mask_15m]

            # 5m 데이터 슬라이스
            df_5m_slice = df_5m.iloc[:i+1]

            # 현재 ATR
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

                    long_position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp': tp,
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

                    # ATR 기반 SL
                    sl = entry_price + (atr * self.config.sl_atr_mult)

                    short_position = {
                        'side': 'short',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp': tp,
                    }
                    pending_short_signal = None

            # ===== 새 신호 체크 (존 터치) =====
            # 15m 먼저 체크, 없으면 5m fallback
            close_arr_15m = df_15m_slice['close'].values.astype(float)
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # Long 신호 체크 - 바운더리 하단 극단에서만
            if long_position is None and pending_long_signal is None and long_cooldown == 0:
                # 1) 15m 다이버전스 체크 (Regular + Hidden)
                ref = find_oversold_reference(df_15m_slice)
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
                    ref = find_oversold_reference(df_5m_slice)
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
                    is_near, fib_level = is_near_l1_level(long_price, tolerance_pct=0.01)
                    if is_near and fib_level:
                        tp_level = get_next_l1_above(long_price)
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

            # Short 신호 체크 - Fib 레벨 근처
            if short_position is None and pending_short_signal is None and short_cooldown == 0:
                # 1) 15m 다이버전스 체크 (Regular + Hidden)
                ref = find_overbought_reference(df_15m_slice)
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
                    ref = find_overbought_reference(df_5m_slice)
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
                    is_near, fib_level = is_near_l1_level(short_price, tolerance_pct=0.01)
                    if is_near and fib_level:
                        tp_level = get_next_l1_below(short_price)
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
        )

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("전략 비교 백테스트")
    print("=" * 70)

    config = Config()

    print(f"\n[설정]")
    print(f"  초기 자산: ${config.initial_capital:,.0f}")
    print(f"  마진/트레이드: ${config.margin_per_trade:,.0f} ({config.margin_pct:.0%})")
    print(f"  레버리지: {config.leverage}x")
    print(f"  포지션 사이즈: ${config.position_size:,.0f}")

    START = "2021-11-01"
    END = "2021-11-30"

    print(f"\n[기간] {START} ~ {END}")

    # 데이터 로딩
    print(f"\n데이터 로딩 중...")
    df_15m = load_data('15m', START, END, config)
    df_5m = load_data('5m', START, END, config)
    print(f"  15m: {len(df_15m)} bars")
    print(f"  5m: {len(df_5m)} bars")

    # 전략 A 실행
    print(f"\n{'='*70}")
    print("전략 A: 15m 진입 + 5m 청산")
    print("='*70")

    strategy_a = StrategyA(config)
    result_a = strategy_a.run(df_15m, df_5m)

    summary_a = result_a.summary()
    for k, v in summary_a.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

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

    # 청산 사유별 분석
    print(f"\n{'='*70}")
    print("청산 사유 분석")
    print("='*70")
    for strat_name, trades in [("A", result_a.trades), ("B", result_b.trades)]:
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"{strat_name}: {reasons}")

if __name__ == "__main__":
    main()
