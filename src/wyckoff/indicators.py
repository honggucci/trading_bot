"""
Technical Indicators
====================

ATR, RSI, ADX 등 기술적 지표 계산 함수.

Origin: WPCN wpcn/_03_common/_02_features/indicators.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd


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
    """Average True Range 계산"""
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index 계산 (Wilder's smoothing)"""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1/length, adjust=False).mean() / (down.ewm(alpha=1/length, adjust=False).mean() + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def stoch_rsi(
    close: pd.Series,
    rsi_len: int = 14,
    stoch_len: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
):
    """Stochastic RSI 계산"""
    r = rsi(close, rsi_len)
    r_min = r.rolling(stoch_len, min_periods=stoch_len).min()
    r_max = r.rolling(stoch_len, min_periods=stoch_len).max()
    st = (r - r_min) / (r_max - r_min + 1e-12)
    k = st.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d


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
    """Average Directional Index 계산"""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_ + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_ + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx_ = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx_


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return close.ewm(span=span, adjust=False).mean()


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return close.rolling(length, min_periods=length).mean()