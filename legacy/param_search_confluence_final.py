from __future__ import annotations
# -*- coding: utf-8 -*-
# === SAFE PANDAS BOOTSTRAP ===
import os

import pandas as pd

# 1) Bring in ALL original logic
from param_search_confluence_v0 import *  # noqa

# 2) TradingView-identical StochRSI + helper overrides
import numpy as np
try:
    import talib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False

def _rsi_wilder_numpy(close, period: int = 14):
    import numpy as _np
    close = _np.asarray(close, dtype=float)
    n = len(close)
    out = _np.full(n, _np.nan, dtype=float)
    if n < period + 1:
        return out
    deltas = _np.diff(close)
    gains = _np.where(deltas > 0, deltas, 0.0)
    losses = _np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = (avg_gain / avg_loss) if avg_loss != 0 else _np.inf
    out[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        g = gains[i-1]; l = losses[i-1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def tv_stoch_rsi(close, *, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    s = pd.Series(close, dtype='float64')
    if _HAS_TALIB:
        rsi_vals = talib.RSI(s.to_numpy(), timeperiod=int(rsi_len))
    else:
        rsi_vals = _rsi_wilder_numpy(s.to_numpy(), period=int(rsi_len))
    rsi = pd.Series(rsi_vals, index=s.index)
    lo = rsi.rolling(int(stoch_len), min_periods=int(stoch_len)).min()
    hi = rsi.rolling(int(stoch_len), min_periods=int(stoch_len)).max()
    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)
    k = stoch.rolling(int(k_len), min_periods=int(k_len)).mean() * 100.0
    d = k.rolling(int(d_len), min_periods=int(d_len)).mean()
    return k.to_numpy(), d.to_numpy()

def ensure_stochD_inplace(df, col='close_stoch', rsi_period=14, stochrsi_period=14, fastd=3, fastk=3, force=False):
    need = force or (col not in df.columns) or df[col].isna().all()
    prev = df.attrs.get('_stoch_params', None)
    curp = (int(stochrsi_period), int(fastd), int(fastk))
    if (not need) and prev is not None and prev != curp:
        need = True
    if not need:
        return
    k, d = tv_stoch_rsi(
        df['close'].astype(float).values,
        rsi_len=int(rsi_period),
        stoch_len=int(stochrsi_period),
        k_len=int(fastk),
        d_len=int(fastd)
    )
    df[col] = d
    df.attrs['_stoch_params'] = curp

def _compute_stochD(_df, stochrsi_period, fastd):
    fk = 3
    _, d = tv_stoch_rsi(
        _df['close'].astype(float).values,
        rsi_len=14,
        stoch_len=int(stochrsi_period),
        k_len=int(fk),
        d_len=int(fastd)
    )
    _df['close_stoch'] = d
