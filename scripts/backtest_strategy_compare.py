# backtest_strategy_compare.py
# 전략 A vs 전략 B 비교 백테스트
# - 전략 A: 15m 진입 + 5m 청산 (반대 다이버전스)
# - 전략 B: Fib 레벨 기반 (L1 근처 진입, 다음 L1 TP)

from __future__ import annotations

import sys
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
from src.regime.wave_regime import WaveRegimeClassifier

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
    fib_max_depth: int = 1            # L0+L1 (41개 레벨)
    fib_tolerance_pct: float = 0.003  # 0.3% (±$180 at $60K, 커버리지 69%)
    fib_atr_mult: float = 1.5         # ATR 기반 tolerance (전략 A): tolerance = ATR * mult / price

    # 쿨다운 설정 (손절 후 재진입 제한)
    cooldown_bars: int = 12           # 5m * 12 = 1시간 쿨다운

    # ATR 기반 SL
    sl_atr_mult: float = 1.5          # SL = entry ± 1.5*ATR (bc3e19a 설정)

    # 동적 SL (Hilbert + FFT 사이클 기반)
    use_dynamic_sl: bool = False      # 동적 SL 사용 여부 (비교용: False)
    dynamic_sl_base: float = 1.5      # 기본 배수
    dynamic_sl_adj: float = 0.5       # 위상 조절 범위 (1.5 ~ 2.0)
    cycle_lookback: int = 200         # 사이클 분석 윈도우 (15m bars)

    # 추세 필터 (계단식 적용)
    use_trend_filter_1h: bool = False  # 1H 역행 금지
    use_trend_filter_4h: bool = False  # 4H 역행 금지
    use_atr_vol_filter: bool = False   # ATR 고변동성 사이즈 축소
    atr_vol_threshold: float = 80.0    # ATR percentile 임계값
    atr_vol_size_mult: float = 0.5     # 고변동성 시 사이즈 배수

    # Zone Depth 필터 (검증 완료: r=0.215, p≈0)
    use_zone_depth_filter: bool = False  # zone_depth >= 0.6 필터
    zone_depth_min: float = 0.6          # 최소 depth (이 미만 진입 금지)
    zone_depth_lookback: int = 100       # swing high/low 계산 lookback (5m bars)

    # Hilbert 레짐 필터 (1H, causal, IC=+0.027)
    use_hilbert_filter: bool = False     # Hilbert 레짐 필터 사용
    hilbert_block_long_on_bear: bool = True   # Long: BEAR 레짐에서 차단
    hilbert_block_short_on_bull: bool = False # Short: BULL 레짐에서 차단 (느슨)

    # 레짐 기반 Hidden Divergence 전략 (새로운 접근법)
    use_regime_hidden_strategy: bool = False  # 레짐 방향 + Hidden Divergence
    # BULL → Hidden Bullish만 Long
    # BEAR → Hidden Bearish만 Short
    # RANGE → 진입 안함

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
    current_time: pd.Timestamp
) -> str:
    """
    현재 시간의 Hilbert 레짐 가져오기 (causal - 완료된 1H봉만 사용)
    Returns: 'BULL', 'BEAR', or 'RANGE'
    """
    if hilbert_regimes is None:
        return 'RANGE'

    # 완료된 1H봉 기준 (lookahead 방지)
    ts_1h = current_time.floor('1h') - pd.Timedelta(hours=1)

    if isinstance(hilbert_regimes.index, pd.DatetimeIndex):
        if ts_1h in hilbert_regimes.index:
            regime = hilbert_regimes.loc[ts_1h, 'regime']
            return str(regime) if pd.notna(regime) else 'RANGE'

        # 가장 가까운 이전 timestamp 찾기
        mask = hilbert_regimes.index <= ts_1h
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
        _FIB_LEVELS_CACHE = get_fractal_fib_levels((3120, 143360), max_depth=1)  # L0+L1
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

    # 인디케이터 (talib 사용)
    df['rsi'] = calc_rsi_wilder(df['close'].values, period=config.rsi_period)
    df['stoch_d'] = calc_stoch_rsi(df['close'].values, period=config.stoch_rsi_period, k_period=3, d_period=3)
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
            df_1h: pd.DataFrame = None, df_4h: pd.DataFrame = None) -> BacktestResult:
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

        # 추세 필터 통계
        trend_filter_rejects = {'1H_DOWNTREND': 0, '1H_UPTREND': 0, '4H_DOWNTREND': 0, '4H_UPTREND': 0}
        hilbert_filter_rejects = {'HILBERT_BEAR': 0, 'HILBERT_BULL': 0}
        atr_vol_size_cuts = 0

        # Hilbert 레짐 계산 (1H, causal)
        hilbert_regimes = None
        if (self.config.use_hilbert_filter or self.config.use_regime_hidden_strategy) and df_1h is not None:
            classifier = WaveRegimeClassifier(detrend_period=48, hilbert_window=32)
            hilbert_regimes = classifier.classify_series_causal(df_1h['close'])
            # 레짐 분포 출력
            if hilbert_regimes is not None and 'regime' in hilbert_regimes.columns:
                regime_counts = hilbert_regimes['regime'].value_counts()
                print(f"\n[Hilbert Regime 분포]")
                for reg, cnt in regime_counts.items():
                    print(f"  {reg}: {cnt} bars ({100*cnt/len(hilbert_regimes):.1f}%)")

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
            close_arr_15m = df_15m_slice['close'].values.astype(float)

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
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # 현재 ATR (5m ATR 사용)
            current_atr = bar['atr'] if 'atr' in bar and np.isfinite(bar['atr']) else 500

            # ===== 포지션 청산 체크 =====
            # Long 청산: SL → TP1 (50%) → TP2 → 5m 다이버전스
            if long_position is not None:
                # SL 체크 (최우선)
                if bar['low'] <= long_position['sl']:
                    exit_price = long_position['sl']
                    trade = self._close_position(long_position, exit_price, 'SL', current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    long_cooldown = self.config.cooldown_bars
                    long_position = None
                # TP1 체크 (50% 청산 + SL→Breakeven)
                elif not long_position.get('tp1_hit', False) and bar['high'] >= long_position['tp1']:
                    exit_price = long_position['tp1']
                    # 50% 부분 청산
                    trade = self._close_position_partial(long_position, exit_price, 'TP1', current_time, 0.5)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    # SL을 Breakeven으로 이동, TP1 히트 플래그 설정
                    long_position['sl'] = long_position['entry_price']
                    long_position['tp1_hit'] = True
                    long_position['remaining'] = 0.5
                    print(f"    [TP1 HIT] ${exit_price:,.0f} | SL→BE: ${long_position['sl']:,.0f}")
                # TP2 체크 (나머지 전량 청산)
                elif long_position.get('tp1_hit', False) and bar['high'] >= long_position['tp2']:
                    exit_price = long_position['tp2']
                    trade = self._close_position_partial(long_position, exit_price, 'TP2', current_time, long_position['remaining'])
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    print(f"    [TP2 HIT] ${exit_price:,.0f}")
                    long_position = None
                # 5m 숏 다이버전스 → 롱 청산 (TP1 이후 또는 TP 미도달 시)
                elif self._check_short_divergence(df_5m_slice):
                    exit_price = bar['close']
                    remaining = long_position.get('remaining', 1.0)
                    trade = self._close_position_partial(long_position, exit_price, '5m_Short_Div', current_time, remaining)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    long_position = None

            # Short 청산: SL → TP1 (50%) → TP2 → 5m 다이버전스
            if short_position is not None:
                # SL 체크 (최우선)
                if bar['high'] >= short_position['sl']:
                    exit_price = short_position['sl']
                    trade = self._close_position(short_position, exit_price, 'SL', current_time)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    short_cooldown = self.config.cooldown_bars
                    short_position = None
                # TP1 체크 (50% 청산 + SL→Breakeven)
                elif not short_position.get('tp1_hit', False) and bar['low'] <= short_position['tp1']:
                    exit_price = short_position['tp1']
                    # 50% 부분 청산
                    trade = self._close_position_partial(short_position, exit_price, 'TP1', current_time, 0.5)
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    # SL을 Breakeven으로 이동, TP1 히트 플래그 설정
                    short_position['sl'] = short_position['entry_price']
                    short_position['tp1_hit'] = True
                    short_position['remaining'] = 0.5
                    print(f"    [TP1 HIT] ${exit_price:,.0f} | SL→BE: ${short_position['sl']:,.0f}")
                # TP2 체크 (나머지 전량 청산)
                elif short_position.get('tp1_hit', False) and bar['low'] <= short_position['tp2']:
                    exit_price = short_position['tp2']
                    trade = self._close_position_partial(short_position, exit_price, 'TP2', current_time, short_position['remaining'])
                    result.trades.append(trade)
                    equity += trade.pnl_usd
                    result.equity_curve.append(equity)
                    print(f"    [TP2 HIT] ${exit_price:,.0f}")
                    short_position = None
                # 5m 롱 다이버전스 → 숏 청산 (TP1 이후 또는 TP 미도달 시)
                elif self._check_long_divergence(df_5m_slice):
                    exit_price = bar['close']
                    remaining = short_position.get('remaining', 1.0)
                    trade = self._close_position_partial(short_position, exit_price, '5m_Long_Div', current_time, remaining)
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

                    # === Hilbert 레짐 필터 (1H, causal) ===
                    if self.config.use_hilbert_filter and hilbert_regimes is not None:
                        # 완료된 1H봉 기준 (lookahead 방지)
                        ts_1h = current_time.floor('1h') - pd.Timedelta(hours=1)
                        if ts_1h in hilbert_regimes.index:
                            hilbert_regime = str(hilbert_regimes.loc[ts_1h, 'regime'])
                        else:
                            mask = hilbert_regimes.index <= ts_1h
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

                    # 반등 확인! 시장가 진입
                    entry_price = bar['close']
                    atr = pending_long_signal['atr']

                    # 동적 SL 배수 계산
                    if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                        cycle_result = self.cycle_dynamics.analyze(close_arr_15m)
                        sl_mult = cycle_result['dynamic_sl_mult']
                    else:
                        sl_mult = self.config.sl_atr_mult

                    # ATR 기반 SL
                    sl = entry_price - (atr * sl_mult)

                    # Fib TP: TP1 (두번째 L1), TP2 (세번째 L1) - 확장된 타겟
                    fib1 = get_next_l1_above(entry_price)
                    fib2 = get_next_l1_above(fib1.price + 1) if fib1 else None
                    fib3 = get_next_l1_above(fib2.price + 1) if fib2 else None
                    # TP1 = 두번째 L1 (첫번째 스킵)
                    tp1 = fib2.price if fib2 else entry_price + (atr * 2.0)
                    # TP2 = 세번째 L1 또는 ATR 기반
                    tp2 = fib3.price if fib3 else entry_price + (atr * 3.5)

                    # 모든 트레이드 로그 (디버그용)
                    print(f"  [LONG ENTRY] {current_time}")
                    print(f"    Entry: ${entry_price:,.0f} | SL: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f}")
                    print(f"    Trend: 1H={trend_1h}, 4H={trend_4h} | Size: {size_mult:.2f}x | Depth: {zone_depth:.2f}")
                    if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                        print(f"    Cycle: {cycle_result['phase_degrees']:.1f}° ({cycle_result['cycle_state']}) | SL Mult: {sl_mult:.3f}")
                    else:
                        print(f"    Cycle: N/A | SL Mult: {sl_mult:.3f} (fixed)")

                    long_position = {
                        'side': 'long',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'atr': atr,  # 트레일링 스탑용
                        'remaining': 1.0,  # 100% 남음
                        'size_mult': size_mult,  # ATR 변동성 사이즈 배수
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

                    # === Hilbert 레짐 필터 (1H, causal) ===
                    if self.config.use_hilbert_filter and hilbert_regimes is not None:
                        # 완료된 1H봉 기준 (lookahead 방지)
                        ts_1h = current_time.floor('1h') - pd.Timedelta(hours=1)
                        if ts_1h in hilbert_regimes.index:
                            hilbert_regime = str(hilbert_regimes.loc[ts_1h, 'regime'])
                        else:
                            mask = hilbert_regimes.index <= ts_1h
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

                    # Fib TP: TP1 (두번째 L1 아래), TP2 (세번째 L1 아래) - 확장된 타겟
                    fib1 = get_next_l1_below(entry_price)
                    fib2 = get_next_l1_below(fib1.price - 1) if fib1 else None
                    fib3 = get_next_l1_below(fib2.price - 1) if fib2 else None
                    # TP1 = 두번째 L1 (첫번째 스킵)
                    tp1 = fib2.price if fib2 else entry_price - (atr * 2.0)
                    # TP2 = 세번째 L1 또는 ATR 기반
                    tp2 = fib3.price if fib3 else entry_price - (atr * 3.5)

                    # 모든 트레이드 로그 (디버그용)
                    print(f"  [SHORT ENTRY] {current_time}")
                    print(f"    Entry: ${entry_price:,.0f} | SL: ${sl:,.0f} | TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f}")
                    print(f"    Trend: 1H={trend_1h}, 4H={trend_4h} | Size: {size_mult:.2f}x | Depth: {zone_depth:.2f}")
                    if self.cycle_dynamics and len(df_15m_slice) >= self.config.cycle_lookback:
                        print(f"    Cycle: {cycle_result['phase_degrees']:.1f}° ({cycle_result['cycle_state']}) | SL Mult: {sl_mult:.3f}")
                    else:
                        print(f"    Cycle: N/A | SL Mult: {sl_mult:.3f} (fixed)")

                    short_position = {
                        'side': 'short',
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'atr': atr,  # 트레일링 스탑용
                        'remaining': 1.0,  # 100% 남음
                        'size_mult': size_mult,  # ATR 변동성 사이즈 배수
                    }
                    pending_short_signal = None

            # ===== 새 신호 체크 (존 터치) =====
            # 15m 먼저 체크, 없으면 5m fallback
            close_arr_5m = df_5m_slice['close'].values.astype(float)

            # Long 신호 체크
            if long_position is None and pending_long_signal is None and long_cooldown == 0:
                long_price = None
                signal_tf = '15m'
                div_type = 'Regular'

                # === 레짐 기반 Hidden Divergence 전략 ===
                if self.config.use_regime_hidden_strategy:
                    current_regime = get_current_hilbert_regime(hilbert_regimes, current_time)
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
                    is_in_oversold = (prev_15m_stoch <= 20.0)

                    if long_signal_triggered or is_in_oversold:
                        # === 1) 15m 다이버전스 진입 시도 ===
                        ref = find_oversold_reference(df_15m_slice)
                        if ref:
                            long_price = needed_close_for_regular_bullish(
                                close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                            )

                            # 15m 진입 시도
                            if long_price and long_price > 0:
                                is_near, fib_level = is_near_l1_level(long_price, tolerance_pct=self.config.fib_tolerance_pct)
                                if is_near and fib_level and bar['low'] <= long_price:
                                    pending_long_signal = {
                                        'zone_price': long_price,
                                        'fib_level': fib_level,
                                        'atr': current_atr,
                                        'touched_time': current_time,
                                    }
                                    signal_tf = '15m'
                                    print(f"  [LONG SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                    print(f"    Div Price: ${long_price:,.0f} | Bar Low: ${bar['low']:,.0f}")
                                    print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio})")

                        # === 2) 15m 실패 시 5m fallback (15m 세그먼트 → 5m REF) ===
                        if pending_long_signal is None:
                            ref_5m = find_oversold_reference_hybrid(df_15m_slice, df_5m_slice)
                            if ref_5m:
                                long_price_5m = needed_close_for_regular_bullish(
                                    close_arr_5m, ref_5m['ref_price'], ref_5m['ref_rsi'], self.config.rsi_period
                                )

                                # 5m 진입 시도
                                if long_price_5m and long_price_5m > 0:
                                    is_near, fib_level = is_near_l1_level(long_price_5m, tolerance_pct=self.config.fib_tolerance_pct)
                                    if is_near and fib_level and bar['low'] <= long_price_5m:
                                        pending_long_signal = {
                                            'zone_price': long_price_5m,
                                            'fib_level': fib_level,
                                            'atr': current_atr,
                                            'touched_time': current_time,
                                        }
                                        signal_tf = '5m'
                                        print(f"  [LONG SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                        print(f"    Div Price: ${long_price_5m:,.0f} | Bar Low: ${bar['low']:,.0f}")
                                        print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio})")

            # Short 신호 체크
            if short_position is None and pending_short_signal is None and short_cooldown == 0:
                short_price = None
                signal_tf = '15m'
                div_type = 'Regular'

                # === 레짐 기반 Hidden Divergence 전략 ===
                if self.config.use_regime_hidden_strategy:
                    current_regime = get_current_hilbert_regime(hilbert_regimes, current_time)
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
                    is_in_overbought = (prev_15m_stoch >= 80.0)

                    if short_signal_triggered or is_in_overbought:
                        # === 1) 15m 다이버전스 진입 시도 ===
                        ref = find_overbought_reference(df_15m_slice)
                        if ref:
                            short_price = needed_close_for_regular_bearish(
                                close_arr_15m, ref['ref_price'], ref['ref_rsi'], self.config.rsi_period
                            )

                            # 15m 진입 시도
                            if short_price and short_price > 0:
                                is_near, fib_level = is_near_l1_level(short_price, tolerance_pct=self.config.fib_tolerance_pct)
                                if is_near and fib_level and bar['high'] >= short_price:
                                    pending_short_signal = {
                                        'zone_price': short_price,
                                        'fib_level': fib_level,
                                        'atr': current_atr,
                                        'touched_time': current_time,
                                    }
                                    signal_tf = '15m'
                                    print(f"  [SHORT SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                    print(f"    Div Price: ${short_price:,.0f} | Bar High: ${bar['high']:,.0f}")
                                    print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio})")

                        # === 2) 15m 실패 시 5m fallback (15m 세그먼트 → 5m REF) ===
                        if pending_short_signal is None:
                            ref_5m = find_overbought_reference_hybrid(df_15m_slice, df_5m_slice)
                            if ref_5m:
                                short_price_5m = needed_close_for_regular_bearish(
                                    close_arr_5m, ref_5m['ref_price'], ref_5m['ref_rsi'], self.config.rsi_period
                                )

                                # 5m 진입 시도
                                if short_price_5m and short_price_5m > 0:
                                    is_near, fib_level = is_near_l1_level(short_price_5m, tolerance_pct=self.config.fib_tolerance_pct)
                                    if is_near and fib_level and bar['high'] >= short_price_5m:
                                        pending_short_signal = {
                                            'zone_price': short_price_5m,
                                            'fib_level': fib_level,
                                            'atr': current_atr,
                                            'touched_time': current_time,
                                        }
                                        signal_tf = '5m'
                                        print(f"  [SHORT SIGNAL] {current_time} ({signal_tf}, {div_type})")
                                        print(f"    Div Price: ${short_price_5m:,.0f} | Bar High: ${bar['high']:,.0f}")
                                        print(f"    Fib Level: ${fib_level.price:,.0f} ({fib_level.fib_ratio})")

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
            close_arr_15m = df_15m_slice['close'].values.astype(float)

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
                    is_near, fib_level = is_near_l1_level(long_price, tolerance_pct=self.config.fib_tolerance_pct)
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
                    is_near, fib_level = is_near_l1_level(short_price, tolerance_pct=self.config.fib_tolerance_pct)
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

    # === 계단식 실험 설정 ===
    # Run 0: Baseline (모든 필터 OFF)
    # Run 1: +1H 역행 금지
    # Run 2: +4H 역행 금지
    # Run 3: +ATR high vol size cut
    # Run 4: +Zone Depth 필터 (검증 완료: r=0.215, p≈0)
    # Run 5: +Hilbert 레짐 필터 (IC=+0.027, Long: BEAR에서 차단)
    # Run 6: 레짐 기반 Hidden Divergence (BULL→Long, BEAR→Short)
    RUN_MODE = 0  # 0, 1, 2, 3, 4, 5, 6 중 선택

    config = Config()

    # 계단식 필터 적용
    if RUN_MODE >= 1 and RUN_MODE < 6:
        config.use_trend_filter_1h = True
    if RUN_MODE >= 2 and RUN_MODE < 6:
        config.use_trend_filter_4h = True
    if RUN_MODE >= 3 and RUN_MODE < 6:
        config.use_atr_vol_filter = True
    if RUN_MODE >= 4 and RUN_MODE < 6:
        config.use_zone_depth_filter = True
    if RUN_MODE >= 5 and RUN_MODE < 6:
        config.use_hilbert_filter = True
        config.hilbert_block_long_on_bear = True
        config.hilbert_block_short_on_bull = False  # Short은 느슨하게

    # RUN_MODE 6: 레짐 기반 Hidden Divergence 전략
    if RUN_MODE == 6:
        config.use_regime_hidden_strategy = True
        # 기존 필터들 OFF (레짐이 방향 결정)
        config.use_trend_filter_1h = False
        config.use_trend_filter_4h = False
        config.use_hilbert_filter = False

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

    START = "2021-11-01"
    END = "2021-11-30"  # 2021년 11월 테스트

    print(f"\n[기간] {START} ~ {END}")

    # 데이터 로딩
    print(f"\n데이터 로딩 중...")
    df_15m = load_data('15m', START, END, config)
    df_5m = load_data('5m', START, END, config)

    # 1H/4H 데이터 로드 (추세 필터 + Hilbert용)
    df_1h = None
    df_4h = None
    if config.use_trend_filter_1h or config.use_trend_filter_4h or config.use_atr_vol_filter or config.use_hilbert_filter or config.use_regime_hidden_strategy:
        df_1h = load_data('1h', START, END, config)
        df_4h = load_data('4h', START, END, config)
        # 1H ATR 계산 (없으면 추가)
        if 'atr' not in df_1h.columns:
            df_1h['atr'] = calc_atr(
                df_1h['high'].values,
                df_1h['low'].values,
                df_1h['close'].values,
                21
            )
        # === Precompute 추세/ATR 컬럼 (O(n) 한 번만) ===
        df_1h['trend'] = precompute_trend_column(df_1h, lookback=20)
        df_4h['trend'] = precompute_trend_column(df_4h, lookback=10)
        df_1h['atr_pct'] = precompute_atr_percentile_column(df_1h, lookback=100)
        print(f"  1h: {len(df_1h)} bars (trend/atr_pct precomputed)")
        print(f"  4h: {len(df_4h)} bars (trend precomputed)")

    print(f"  15m: {len(df_15m)} bars")
    print(f"  5m: {len(df_5m)} bars")

    # 전략 A 실행
    print(f"\n{'='*70}")
    print("전략 A: 15m 진입 + 5m 청산")
    print("='*70")

    strategy_a = StrategyA(config)
    result_a = strategy_a.run(df_15m, df_5m, df_1h, df_4h)

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

    # 청산 사유별 분석
    print(f"\n{'='*70}")
    print("청산 사유 분석")
    print("='*70")
    for strat_name, trades in [("A", result_a.trades), ("B", result_b.trades)]:
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print(f"{strat_name}: {reasons}")

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
