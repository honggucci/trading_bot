"""
ZigZag utilities - Legacy stubs
===============================

ZigZag pivot detection을 위한 stub 함수들.
실제 구현은 dynamic_fib_anchor.py의 ZigZagState를 사용합니다.

Note: 이 모듈은 legacy_pipeline.py 호환성을 위해 존재합니다.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any


@dataclass
class SwingInfo:
    """스윙 정보 (Legacy 호환용)"""
    swing_low: float
    swing_high: float
    low_idx: int
    high_idx: int
    direction: str  # "up" or "down"


def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's ATR 계산

    Args:
        df: OHLC DataFrame (high, low, close 컬럼 필요)
        period: ATR 기간

    Returns:
        ATR Series
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr


def zigzag_pivots(
    df: pd.DataFrame,
    *,
    up_pct: float = 0.02,
    down_pct: float = 0.02,
    atr_period: int = 14,
    atr_mult: float = 2.0,
    threshold_mode: str = 'or',
    min_bars: int = 5,
) -> pd.DataFrame:
    """
    ZigZag 피봇 찾기 (Legacy stub)

    Note: 실제 구현은 dynamic_fib_anchor.py의 ZigZagState를 사용하세요.

    Returns:
        DataFrame with pivot columns
    """
    # Stub - 빈 피봇 반환
    result = df.copy()
    result['pivot_type'] = None
    result['pivot_price'] = np.nan
    return result


def get_latest_swing(
    df: pd.DataFrame,
    min_swing_atr: float = 1.0,
) -> Optional[SwingInfo]:
    """
    최근 스윙 정보 추출 (Legacy stub)

    Returns:
        SwingInfo or None
    """
    if len(df) < 20:
        return None

    # 간단한 최근 고/저점 찾기
    lookback = min(100, len(df))
    recent = df.tail(lookback)

    low_idx = recent['low'].idxmin()
    high_idx = recent['high'].idxmax()

    swing_low = df.loc[low_idx, 'low']
    swing_high = df.loc[high_idx, 'high']

    # 방향 결정 (최근 피봇 기준)
    direction = "up" if df.index.get_loc(low_idx) < df.index.get_loc(high_idx) else "down"

    return SwingInfo(
        swing_low=swing_low,
        swing_high=swing_high,
        low_idx=df.index.get_loc(low_idx),
        high_idx=df.index.get_loc(high_idx),
        direction=direction,
    )


def add_pivot_columns(df: pd.DataFrame, pivots: pd.DataFrame) -> pd.DataFrame:
    """
    피봇 컬럼 추가 (Legacy stub)

    Returns:
        DataFrame with added columns
    """
    result = df.copy()
    if 'pivot_type' not in result.columns:
        result['pivot_type'] = None
    if 'pivot_price' not in result.columns:
        result['pivot_price'] = np.nan
    return result


__all__ = [
    'SwingInfo',
    'wilder_atr',
    'zigzag_pivots',
    'get_latest_swing',
    'add_pivot_columns',
]
