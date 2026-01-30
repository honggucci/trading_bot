"""
Technical Indicators
====================

ATR, RSI, ADX 등 기술적 지표 계산 함수.
talib 라이브러리 사용.

Origin: WPCN wpcn/_03_common/_02_features/indicators.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import talib


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range 계산"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range 계산 (talib 사용)"""
    high = df["high"].astype(np.float64).values
    low = df["low"].astype(np.float64).values
    close = df["close"].astype(np.float64).values
    result = talib.ATR(high, low, close, timeperiod=length)
    return pd.Series(result, index=df.index)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index 계산 (talib 사용)"""
    close_arr = close.astype(np.float64).values
    result = talib.RSI(close_arr, timeperiod=length)
    return pd.Series(result, index=close.index)


def stoch_rsi(
    close: pd.Series,
    rsi_len: int = 14,
    stoch_len: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
):
    """Stochastic RSI 계산 (talib 사용)"""
    close_arr = close.astype(np.float64).values
    fastk, fastd = talib.STOCHRSI(
        close_arr,
        timeperiod=stoch_len,
        fastk_period=smooth_k,
        fastd_period=smooth_d,
        fastd_matype=0  # SMA
    )
    return pd.Series(fastk, index=close.index), pd.Series(fastd, index=close.index)


def zscore(close: pd.Series, length: int = 20) -> pd.Series:
    """Z-Score 계산"""
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return (close - ma) / sd.replace(0, np.nan)


def slope(close: pd.Series, length: int = 20) -> pd.Series:
    """선형 회귀 기울기 계산"""
    x = np.arange(length)
    x = x - x.mean()
    denom = (x**2).sum()

    def _sl(y):
        y = np.asarray(y)
        y = y - y.mean()
        return float((x * y).sum() / denom)

    return close.rolling(length, min_periods=length).apply(_sl, raw=False)


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average Directional Index 계산 (talib 사용)"""
    high = df["high"].astype(np.float64).values
    low = df["low"].astype(np.float64).values
    close = df["close"].astype(np.float64).values
    result = talib.ADX(high, low, close, timeperiod=length)
    return pd.Series(result, index=df.index)


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (talib 사용)"""
    close_arr = close.astype(np.float64).values
    result = talib.EMA(close_arr, timeperiod=span)
    return pd.Series(result, index=close.index)


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average (talib 사용)"""
    close_arr = close.astype(np.float64).values
    result = talib.SMA(close_arr, timeperiod=length)
    return pd.Series(result, index=close.index)