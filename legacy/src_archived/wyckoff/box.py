"""
Wyckoff Box Engine
==================

Wyckoff 박스 계산 및 프리즈 로직.

Origin: WPCN wpcn/_03_common/_03_wyckoff/box.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .types import Theta


@dataclass
class BoxState:
    """Wyckoff 박스 상태"""
    high: float = np.nan
    low: float = np.nan
    mid: float = np.nan
    width: float = np.nan
    frozen_until_i: int = -1


def box_engine_freeze(df: pd.DataFrame, theta: Theta) -> pd.DataFrame:
    """
    Wyckoff 박스 계산 (프리즈 포함)

    Args:
        df: OHLCV DataFrame
        theta: Theta 설정

    Returns:
        DataFrame with box_high, box_low, box_mid, box_width
    """
    hi_roll = df["high"].rolling(theta.box_L, min_periods=theta.box_L).max()
    lo_roll = df["low"].rolling(theta.box_L, min_periods=theta.box_L).min()
    width_roll = hi_roll - lo_roll

    # Pre-compute rolling median
    med_bw_roll = width_roll.rolling(theta.box_L, min_periods=1).median()

    # Use numpy arrays for faster iteration
    hi_arr = hi_roll.values
    lo_arr = lo_roll.values
    bw_arr = width_roll.values
    med_arr = med_bw_roll.values
    n = len(df)

    # Pre-allocate output arrays
    out_high = np.full(n, np.nan)
    out_low = np.full(n, np.nan)
    out_mid = np.full(n, np.nan)
    out_width = np.full(n, np.nan)

    state_high, state_low, state_mid, state_width = np.nan, np.nan, np.nan, np.nan
    frozen_until_i = -1

    for i in range(n):
        if np.isnan(hi_arr[i]) or np.isnan(lo_arr[i]):
            continue

        if i <= frozen_until_i:
            out_high[i] = state_high
            out_low[i] = state_low
            out_mid[i] = state_mid
            out_width[i] = state_width
            continue

        bh = hi_arr[i]
        bl = lo_arr[i]
        bw = bw_arr[i]
        bm = (bh + bl) / 2.0
        med_bw = med_arr[i]

        box_candidate = (bw <= 1.5 * med_bw) if med_bw > 0 else False

        if box_candidate:
            state_high, state_low, state_mid, state_width = bh, bl, bm, bw
            frozen_until_i = i + theta.m_freeze
        else:
            state_high, state_low, state_mid, state_width = bh, bl, bm, bw
            frozen_until_i = -1

        out_high[i] = state_high
        out_low[i] = state_low
        out_mid[i] = state_mid
        out_width[i] = state_width

    out = pd.DataFrame({
        "box_high": out_high,
        "box_low": out_low,
        "box_mid": out_mid,
        "box_width": out_width
    }, index=df.index)

    return out