# -*- coding: utf-8 -*-
"""
Volatility - ATR 계산
=====================

Zone Width 계산용 ATR.
"""
import numpy as np
from typing import Optional


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 21
) -> np.ndarray:
    """
    Average True Range (ATR)

    Args:
        high, low, close: 가격 시계열
        window: 롤링 윈도우 (기본 21)

    Returns:
        ATR 시계열 (달러 단위)
    """
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    n = len(tr)
    atr_values = np.full(n, np.nan)

    for i in range(window - 1, n):
        atr_values[i] = np.mean(tr[i - window + 1:i + 1])

    return atr_values


def atr_to_zone_width(
    atr_values: np.ndarray,
    fib_price: float,
    k: float = 2.75,
    min_pct: float = 0.002,
    max_pct: float = 0.015
) -> np.ndarray:
    """
    ATR → Zone Width 변환

    Formula:
        width = ATR * k
        width = clamp(width, fib_price * min_pct, fib_price * max_pct)

    Args:
        atr_values: ATR 시계열 (달러)
        fib_price: Fib 레벨 가격 (clamp 기준)
        k: ATR 배수
        min_pct: 최소 폭 (%)
        max_pct: 최대 폭 (%)

    Returns:
        Zone Width (달러 단위)
    """
    width = atr_values * k

    min_width = fib_price * min_pct
    max_width = fib_price * max_pct

    return np.clip(width, min_width, max_width)
