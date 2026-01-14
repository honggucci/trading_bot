# param_search_confluence_v2.py
# Top-Down Report v2 — selective Fibonacci levels & divergence-gated display
# (R-162 ~ R162 전 범위 계산·필터링·표시 + 다이버전스 없으면 미표시)

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import talib
from contextlib import redirect_stdout

# =========================
# 0) Utilities
# =========================
def _ts_series(df: pd.DataFrame):
    return df["datetime"] if "datetime" in df.columns else df.index

# TV-like StochRSI (NumPy)
def _rsi_wilder_numpy(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1: return out
    deltas = np.diff(close)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = (avg_gain / avg_loss) if avg_loss != 0 else np.inf
    out[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        g = gains[i-1]; l = losses[i-1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = 100.0 if avg_loss == 0 else (100.0 - 100.0 / (1.0 + (avg_gain/avg_loss)))
    return out

def _sma(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float); n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if win <= 0 or n < win: return out
    csum = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    cnt  = np.cumsum(np.where(np.isfinite(x), 1.0, 0.0))
    out[win-1:] = (csum[win-1:] - np.r_[0.0, csum[:-win]]) / (cnt[win-1:] - np.r_[0.0, cnt[:-win]])
    return out

def tv_stoch_rsi_numpy(close, *, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    c = np.asarray(close, dtype=float)
    rsi = _rsi_wilder_numpy(c, period=int(rsi_len))
    n = len(c); L = int(stoch_len)
    lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    if n >= L:
        for i in range(L-1, n):
            seg = rsi[i-L+1:i+1]
            if np.all(np.isfinite(seg)):
                lo[i] = np.min(seg); hi[i] = np.max(seg)
    denom = (hi - lo)
    stoch = np.where(np.isfinite(denom) & (denom != 0.0), (rsi - lo) / denom, np.nan)
    k = _sma(stoch, int(k_len)) * 100.0
    d = _sma(k,     int(d_len))
    return k, d

def ensure_stochD_inplace(df,
                          col: str = 'close_stoch',
                          stochrsi_period: int = 14,
                          fastd: int = 3,
                          fastd_matype: int = 0,
                          force: bool = True) -> None:
    # Talib 기반 (호환)
    _, d = talib.STOCHRSI(
        df['close'].astype(float).values,
        timeperiod=int(stochrsi_period),
        fastk_period=int(stochrsi_period),
        fastd_period=int(fastd),
        fastd_matype=int(fastd_matype),
    )
    df[col] = d

# =========================
# 1) Param space & volatility scaling
# =========================
def make_param_space_for_tf(tf: str):
    tf = tf.lower()
    if tf == '1m':
        up_dn = (0.003, 0.008); atrp = (7, 21); mult = (1.0, 2.0); mb = (2, 6)
    elif tf == '5m':
        up_dn = (0.003, 0.012); atrp = (14, 28); mult = (1.2, 2.2); mb = (3, 8)
    elif tf == '15m':
        up_dn = (0.007, 0.03);  atrp = (14, 35); mult = (1.4, 2.6); mb = (5, 12)
    elif tf == '1h':
        up_dn = (0.010, 0.040); atrp = (14, 35); mult = (1.4, 2.8); mb = (6, 12)
    elif tf == '4h':
        up_dn = (0.020, 0.060); atrp = (21, 42); mult = (1.6, 3.0); mb = (7, 14)
    elif tf == '1w':
        up_dn = (0.040, 0.120); atrp = (21, 55); mult = (1.8, 3.2); mb = (6, 12)
    else:  # 1d 등
        up_dn = (0.030, 0.080); atrp = (21, 42); mult = (1.6, 3.0); mb = (7, 14)
    return {
        # 인디케이터(튜닝 후보)
        'stochrsi_period': {'type':'int','min':7,'max':34},
        'stochrsi_fastd':  {'type':'int','min':2,'max':6},
        'atr_period':      {'type':'int','min': atrp[0],'max': atrp[1]},
        # 지그재그 파라미터 범위
        'zig_up_pct':        {'type':'float','min': up_dn[0],'max': up_dn[1]},
        'zig_down_pct':      {'type':'float','min': up_dn[0],'max': up_dn[1]},
        'zig_atr_mult':      {'type':'float','min': mult[0],'max': mult[1]},
        'zig_min_bars':      {'type':'int','min': mb[0],'max':  mb[1]},
        'zig_atr_period':    {'type':'int','min': atrp[0],'max': atrp[1]},
        'zig_threshold_mode':{'type':'choice','choices':['or','and','max']},
    }

def compute_vol_stats(df: pd.DataFrame, atr_period: int = 21, lookback: int = 400):
    d = df.tail(int(lookback)).copy()
    close = d["close"].astype(float).to_numpy()
    high  = d["high"].astype(float).to_numpy()
    low   = d["low"].astype(float).to_numpy()
    open_ = d.get("open", d["close"]).astype(float).to_numpy()
    atr = talib.ATR(high, low, close, timeperiod=int(atr_period))
    eps = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        natr  = np.where(close > eps, atr / close, np.nan)
        body  = np.where(close > eps, np.abs(close - open_) / close, np.nan)
        rang  = np.where(close > eps, (high - low) / close, np.nan)
    return float(np.nanmedian(natr)), float(np.nanmedian(body)), float(np.nanmedian(rang))

def atr_scaled_param_space(df_top, base_space,
                           atr_period=21, lookback=400,
                           updn_k_low=1.4, updn_k_high=3.2,
                           mult_bias=1.0, mult_gamma=0.0,
                           min_bars_lo_shift=-2, min_bars_hi_shift=+2):
    ps = {k:(v.copy() if isinstance(v,dict) else v) for k,v in base_space.items()}
    natr_med, body_med, range_med = compute_vol_stats(df_top, atr_period=atr_period, lookback=lookback)
    vol_score = 0.6*natr_med + 0.2*body_med + 0.2*range_med
    def _reshape(key):
        lo0, hi0 = ps[key]['min'], ps[key]['max']
        lo = max(lo0, vol_score*updn_k_low)
        hi = min(hi0, vol_score*updn_k_high)
        if hi < lo:
            mid=(lo0+hi0)/2; lo,hi=mid*0.8, mid*1.2
        ps[key]['min'], ps[key]['max'] = lo, hi
    _reshape('zig_up_pct'); _reshape('zig_down_pct')
    if 'zig_min_bars' in ps:
        lo0, hi0 = ps['zig_min_bars']['min'], ps['zig_min_bars']['max']
        ps['zig_min_bars']['min'] = int(max(1, lo0 + min_bars_lo_shift))
        ps['zig_min_bars']['max'] = int(max(ps['zig_min_bars']['min'], hi0 + min_bars_hi_shift))
    return ps, dict(natr_med=natr_med, body_med=body_med, range_med=range_med, vol_score=vol_score)

# =========================
# 2) ZigZag & Fibonacci
# =========================
def wilder_atr(high, low, close, period=14):
    high = np.asarray(high, dtype=float)
    low  = np.asarray(low, dtype=float)
    close= np.asarray(close, dtype=float)
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high-low, np.maximum(np.abs(high-prev_close), np.abs(low-prev_close)))
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

def zigzag_meaningful_v2(
    close, high=None, low=None,
    up_pct=0.05, down_pct=0.05,
    atr_period=14, atr_mult=2.0,
    threshold_mode='or', use_hl=True,
    min_bars=5, min_swing_atr=1.0, finalize_last=False
):
    price = np.asarray(close, dtype=float)
    if use_hl:
        if high is None or low is None: raise ValueError("use_hl=True면 high/low 필요")
        high = np.asarray(high, dtype=float); low = np.asarray(low, dtype=float)
    else:
        high = price; low = price
    try:
        atr = talib.ATR(high.astype(float), low.astype(float), price.astype(float), timeperiod=int(atr_period))
    except Exception:
        atr = wilder_atr(high, low, price, period=int(atr_period))

    def moved_up(ext_val, i):
        pct_ok = (high[i] - ext_val) / max(ext_val, 1e-12) >= up_pct
        atr_ok = (high[i] - ext_val) >= atr_mult * atr[i]
        if threshold_mode == 'and': return pct_ok and atr_ok
        if threshold_mode == 'max': return (high[i] - ext_val) >= max(up_pct*ext_val, atr_mult*atr[i])
        return pct_ok or atr_ok
    def moved_down(ext_val, i):
        pct_ok = (ext_val - low[i]) / max(ext_val, 1e-12) >= down_pct
        atr_ok = (ext_val - low[i]) >= atr_mult * atr[i]
        if threshold_mode == 'and': return pct_ok and atr_ok
        if threshold_mode == 'max': return (ext_val - low[i]) >= max(down_pct*ext_val, atr_mult*atr[i])
        return pct_ok or atr_ok

    n=len(price); pivots=np.zeros(n, dtype=int); trend=0; ext_idx=0; ext_val=price[0]
    for i in range(1, n):
        if trend == 0:
            if moved_up(ext_val, i):   trend=1; ext_idx=i; ext_val=high[i]; continue
            if moved_down(ext_val, i): trend=-1; ext_idx=i; ext_val=low[i];  continue
        elif trend==1:
            if high[i] > ext_val: ext_idx, ext_val = i, high[i]
            if moved_down(ext_val, i):
                pivots[ext_idx]=1; trend=-1; ext_idx, ext_val = i, low[i]
        else:
            if low[i] < ext_val:  ext_idx, ext_val = i, low[i]
            if moved_up(ext_val, i):
                pivots[ext_idx]=-1; trend=1; ext_idx, ext_val = i, high[i]
    if finalize_last and trend != 0:
        pivots[ext_idx] = 1 if trend == 1 else -1

    # prune alternation & min bars & min swing ATR
    def extreme_at(idx, sign): return high[idx] if sign == 1 else low[idx]
    idxs = np.where(pivots!=0)[0].tolist()
    j=1
    while j < len(idxs):
        a,b = idxs[j-1], idxs[j]
        if pivots[a]==pivots[b]:
            if pivots[a]==1:
                keep = a if high[a]>=high[b] else b
            else:
                keep = a if low[a]<=low[b] else b
            drop = b if keep==a else a
            pivots[drop]=0
            idxs.pop(j if drop==b else j-1)
            j=max(1,j-1)
        else: j+=1
    changed=True
    while changed and len(idxs)>=3:
        changed=False; k=1
        while k < len(idxs)-1:
            a,b,c = idxs[k-1], idxs[k], idxs[k+1]
            if (b-a)<min_bars or (c-b)<min_bars:
                pivots[b]=0; idxs.pop(k); changed=True; continue
            amp1 = abs(extreme_at(b, pivots[b]) - extreme_at(a, pivots[a]))
            amp2 = abs(extreme_at(c, pivots[c]) - extreme_at(b, pivots[b]))
            thr  = min_swing_atr * atr[b]
            if min(amp1, amp2) < thr:
                pivots[b]=0; idxs.pop(k); changed=True; continue
            k+=1
    return pivots, atr

def latest_alternating_pivots(df, pivot_col='pivot', pivot_price_col='pivot_price'):
    p = df[pivot_col].to_numpy()
    idxs = np.flatnonzero((p!=0) & np.isfinite(p))
    if len(idxs) < 2: raise ValueError("피벗이 2개 미만")
    for k in range(len(idxs)-1, 0, -1):
        i2, i1 = idxs[k], idxs[k-1]
        s2, s1 = int(p[i2]), int(p[i1])
        if s1 != s2:
            start_pos=i1; end_pos=i2
            start_price = float(df[pivot_price_col].iat[i1])
            end_price   = float(df[pivot_price_col].iat[i2])
            direction = 'up' if (s1==-1 and s2==1) else 'down'
            return {
                'start_pos':start_pos,'end_pos':end_pos,
                'start_ts':df.index[start_pos],'end_ts':df.index[end_pos],
                'start_sign':s1,'end_sign':s2,
                'start_price':start_price,'end_price':end_price,
                'direction':direction
            }
    raise ValueError("교대 피벗 쌍을 찾지 못함")

def fib_from_latest(
    df, pivot_col='pivot', pivot_price_col='pivot_price',
    retracements=(0.236,0.382,0.5,0.618,0.786),
    extensions=(1.272,1.414,1.618),
    include_extremes=True,
    ext_side='auto'
):
    sw = latest_alternating_pivots(df, pivot_col, pivot_price_col)
    if sw['direction']=='up':
        lo, hi = sw['start_price'], sw['end_price']
        anchors={'0':hi,'1':lo}
        rets = {r: hi - (hi-lo)*r for r in retracements}
        exts = {e: hi + (hi-lo)*e for e in extensions} if ext_side!='below' else {e: lo - (hi-lo)*e for e in extensions}
        gp   = (hi - (hi-lo)*0.65, hi - (hi-lo)*0.618)
    else:
        hi, lo = sw['start_price'], sw['end_price']
        anchors={'0':lo,'1':hi}
        rets = {r: lo + (hi-lo)*r for r in retracements}
        exts = {e: lo - (hi-lo)*e for e in extensions} if ext_side!='above' else {e: hi + (hi-lo)*e for e in extensions}
        gp   = (lo + (hi-lo)*0.618, lo + (hi-lo)*0.65)
    if include_extremes:
        rets = {**rets, 0.0: float(anchors['0']), 1.0: float(anchors['1'])}
    gp = (min(gp), max(gp))
    return {'swing': sw, 'anchors': anchors, 'retracements': rets, 'extensions': exts, 'golden_pocket': gp}

# =========================
# 3) Divergence helpers
# =========================
def pick_oversold_segment_D_with_current_rule(df, d_col='close_stoch', oversold=20.0, auto_scale=True, prefer_current=False):
    d = df[d_col].astype(float).to_numpy()
    if auto_scale:
        maxv = np.nanmax(d)
        thr = oversold/100.0 if maxv <= 1.0 else oversold
    else:
        thr = oversold
    n=len(d); segs=[]; i=n-1
    while i>=0:
        if np.isfinite(d[i]) and d[i] <= thr:
            b=i; a=i
            while a-1>=0 and np.isfinite(d[a-1]) and d[a-1] <= thr: a-=1
            segs.append((a,b)); i=a-1
        else: i-=1
    segs=segs[::-1]
    if not segs: return (None, thr, 'no_segment')
    cur=d[-1]
    if np.isfinite(cur) and cur <= thr:
        if prefer_current: return (segs[-1], thr, 'current_oversold_use_current')
        return (segs[-2], thr, 'current_oversold_use_previous') if len(segs)>=2 else (None, thr, 'current_oversold_but_no_previous')
    else:
        return (segs[-1], thr, 'current_not_oversold_use_latest')

def ref_from_segment_min_close(df, seg, *, rsi_col='rsi'):
    a,b = seg; sub = df.iloc[a:b+1]
    idx_min = sub['close'].idxmin()
    iloc_min = df.index.get_loc(idx_min)
    return {'ref_idx': iloc_min, 'ref_ts': idx_min,
            'ref_price': float(df.at[idx_min, 'close']),
            'ref_rsi': float(df.at[idx_min, rsi_col])}

def needed_close_regular_now(df, *, ref_price, ref_rsi, rsi_period=14,
                             lower_bound=None, eps=1e-8, tol=1e-6, max_iter=60):
    close = df['close'].to_numpy(float).copy()
    def rsi_last(x):
        close[-1]=x
        return talib.RSI(close, timeperiod=int(rsi_period))[-1]
    U = ref_price - max(eps, abs(ref_price)*1e-6)
    L = (U - max(1e-6, abs(U)*0.10)) if lower_bound is None else min(lower_bound, U - max(1e-6, abs(U)*0.001))
    if not np.isfinite(U) or L >= U: return None
    rU = rsi_last(U)
    if (not np.isfinite(rU)) or rU <= ref_rsi: return None
    lo, hi = L, U
    for _ in range(max_iter):
        mid=(lo+hi)/2; rmid=rsi_last(mid)
        if not np.isfinite(rmid): lo=mid; continue
        if rmid > ref_rsi: hi=mid
        else: lo=mid
        if abs(hi-lo)<=tol: break
    return float(min(hi, U))

def feasible_close_range_hidden_now(df, *, ref_price, ref_rsi, rsi_period=14,
                                    upper_bound=None, eps=1e-8, tol=1e-6, max_iter=60):
    close = df['close'].to_numpy(float).copy()
    def rsi_last(x):
        close[-1]=x
        return talib.RSI(close, timeperiod=int(rsi_period))[-1]
    L = ref_price + max(eps, abs(ref_price)*1e-6)
    U = (L + max(1e-6, abs(L)*0.10)) if upper_bound is None else max(upper_bound, L + max(1e-6, abs(L)*0.001))
    if not np.isfinite(L) or L >= U: return None
    rL = rsi_last(L)
    if (not np.isfinite(rL)) or rL >= ref_rsi: return None
    lo, hi = L, U
    for _ in range(max_iter):
        mid=(lo+hi)/2; rmid=rsi_last(mid)
        if not np.isfinite(rmid): hi=mid; continue
        if rmid < ref_rsi: lo=mid
        else: hi=mid
        if abs(hi-lo)<=tol: break
    xmax = float(lo)
    if xmax <= L: return None
    return (float(L), xmax)

def _segment_window_half_open(df_top, seg):
    ts = _ts_series(df_top); a,b = seg
    start = pd.Timestamp(ts.iloc[a])
    if b+1 < len(ts):
        end_excl = pd.Timestamp(ts.iloc[b+1])
    else:
        diffs = ts.diff().dropna()
        step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
        end_excl = pd.Timestamp(ts.iloc[b]) + step
    return start, end_excl

def ref_from_down_window_by_oversold(df_, start_ts, end_ts_exclusive, * ,
                                     rsi_col='rsi', d_col='close_stoch',
                                     oversold=20, auto_scale=True, prefer_current=True):
    ts4 = _ts_series(df_)
    win = df_.loc[(ts4 >= pd.Timestamp(start_ts)) & (ts4 < pd.Timestamp(end_ts_exclusive))].copy()
    if win.empty: raise ValueError("DOWN 창이 비었습니다.")
    seg, thr_used, reason = pick_oversold_segment_D_with_current_rule(
        win, d_col=d_col, oversold=oversold, auto_scale=auto_scale, prefer_current=prefer_current
    )
    if seg is None: raise ValueError("오버솔드 세그먼트 없음")
    a,b = seg; sub = win.iloc[a:b+1]
    idx_min = sub['close'].idxmin()
    ref_ts = idx_min
    ref_price = float(win.at[idx_min, 'close'])
    ref_rsi   = float(win.at[idx_min, rsi_col])
    iloc = df_.index.get_loc(idx_min)
    return {'ref_iloc': iloc, 'ref_ts': ref_ts, 'ref_price': ref_price, 'ref_rsi': ref_rsi}

def ref_from_down_window(df_, start_ts, end_ts_exclusive, *, rsi_col='rsi', prefer='close'):
    ts4 = _ts_series(df_)
    sub = df_.loc[(ts4 >= pd.Timestamp(start_ts)) & (ts4 < pd.Timestamp(end_ts_exclusive))]
    if sub.empty: raise ValueError("DOWN 창이 비었습니다.")
    col = 'close' if str(prefer).lower()=='close' else 'low'
    idx_min  = sub[col].astype(float).idxmin()
    ref_ts   = ts4.loc[idx_min]
    ref_price= float(df_.loc[idx_min, 'close'])
    ref_rsi  = float(df_.loc[idx_min, rsi_col]) if rsi_col in df_.columns else float('nan')
    ref_iloc = int(df_.index.get_loc(idx_min))
    return {'ref_iloc': ref_iloc, 'ref_ts': ref_ts, 'ref_price': ref_price, 'ref_rsi': ref_rsi}

# =========================
# 4) Fibonacci boundaries (±R162), ranking, printing
# =========================
FIB_RATIOS_FULL = (
    -1.618,-1.414,-1.272,-1.000,-0.786,-0.618,-0.500,-0.382,-0.236,
     0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.414, 1.618
)

def _atr_at_end(df, res, atr_col='atr'):
    end_pos = res['swing']['end_pos']
    return float(pd.Series(df[atr_col]).iloc[:end_pos+1].dropna().iloc[-1])

def build_fib_ratio_boundaries(df, res, * ,
                               atr_col='atr',
                               ratios=FIB_RATIOS_FULL,
                               k=1.0, mode='linear', gamma=1.0,
                               max_half_mult=None):
    atr_end = _atr_at_end(df, res, atr_col=atr_col)
    sw = res['swing']
    if sw['direction'] == 'up':
        lo, hi = sw['start_price'], sw['end_price']
    else:
        hi, lo = sw['start_price'], sw['end_price']
    def g(r):
        if mode == 'sqrt': return np.sqrt(r)
        if mode == 'pow' : return r**gamma
        return r
    out=[]
    for r in ratios:
        r=float(r)
        if sw['direction'] == 'up':
            if r < 0.0:   lvl = lo - (hi - lo) * abs(r)
            elif r < 1.0: lvl = hi - (hi - lo) * r
            elif r == 1.: lvl = float(res['anchors']['1'])
            else:         lvl = hi + (hi - lo) * r
        else:
            if r < 0.0:   lvl = hi + (hi - lo) * abs(r)
            elif r < 1.0: lvl = lo + (hi - lo) * r
            elif r == 1.: lvl = float(res['anchors']['1'])
            else:         lvl = lo - (hi - lo) * r
        half = float(k * g(abs(r)) * atr_end)
        if max_half_mult is not None: half = min(half, float(max_half_mult * atr_end))
        out.append({'ratio': r, 'label': f'R{int(round(r*100))}', 'level': lvl,
                    'low': lvl-half, 'high': lvl+half, 'half': half})
    return out, atr_end

def make_level_map_from_res(res, ratios):
    out = {}
    sw = res['swing']
    if sw['direction']=='up':
        lo, hi = sw['start_price'], sw['end_price']
    else:
        hi, lo = sw['start_price'], sw['end_price']
    for r in ratios:
        r=float(r)
        if r == 0.0: price=float(res['anchors']['0'])
        elif r == 1.0: price=float(res['anchors']['1'])
        else:
            if sw['direction']=='up':
                if r<0.0:   price=lo-(hi-lo)*abs(r)
                elif r<1.0: price=hi-(hi-lo)*r
                else:       price=hi+(hi-lo)*r
            else:
                if r<0.0:   price=hi+(hi-lo)*abs(r)
                elif r<1.0: price=lo+(hi-lo)*r
                else:       price=lo-(hi-lo)*r
        out[f"R{int(round(r*100))}"]=float(price)
    return out

def _score_level(b, *, price_now, atr_end, need_reg=None, hid_range=None,
                 w_prox=1.0, w_reg=0.6, w_hid=1.0, w_width=-0.15):
    low, high = float(b['low']), float(b['high'])
    center = float(b.get('level', (low + high)/2.0))
    half   = float(b['half'])
    dist_atr = abs(center - price_now) / max(1e-9, atr_end)
    prox = np.exp(-dist_atr)
    reg_bonus=0.0
    if need_reg is not None and np.isfinite(need_reg):
        if low <= float(need_reg) <= high: reg_bonus=1.0
    hid_bonus=0.0
    if hid_range is not None:
        Lh,Xh = float(hid_range[0]), float(hid_range[1])
        lo=max(low,Lh); hi=min(high,Xh)
        if hi>lo:
            overlap=(hi-lo)/max(1e-9,(high-low))
            hid_bonus=0.5+0.5*np.clip(overlap,0,1)
    width_pen = half / max(1e-9, atr_end)
    return float(w_prox*prox + w_reg*reg_bonus + w_hid*hid_bonus + w_width*width_pen)

def filter_and_rank_bounds(df, res, * ,
                           fib_ratios=None, fib_k=1.0, fib_mode='linear', fib_max_half_mult=1.0,
                           need_reg=None, hid_range=None,
                           top_k=None, min_spacing_atr=0.6, prefer_higher=False,
                           atr_col='atr', require_divergence=True):
    bounds, atr_end = build_fib_ratio_boundaries(
        df, res, atr_col=atr_col,
        ratios=FIB_RATIOS_FULL, k=fib_k, mode=fib_mode, max_half_mult=fib_max_half_mult
    )
    price_now = float(df['close'].astype(float).iloc[-1])

    if require_divergence:
        filtered=[]
        for b in bounds:
            low, high = float(b['low']), float(b['high'])
            has_reg = (need_reg is not None and np.isfinite(need_reg) and low <= float(need_reg) <= high)
            has_hid = False
            if hid_range is not None:
                Lh,Xh = float(hid_range[0]), float(hid_range[1])
                lo=max(low,Lh); hi=min(high,Xh)
                has_hid = (hi>lo)
            if has_reg or has_hid: filtered.append(b)
        bounds = filtered
    if not bounds: return [], atr_end

    for b in bounds:
        b['score'] = _score_level(b, price_now=price_now, atr_end=atr_end, need_reg=need_reg, hid_range=hid_range)
        b['ratio'] = float(b.get('ratio', 0.0))

    prefer_set = set([float(r) for r in (fib_ratios or ())])
    for b in bounds:
        if b['ratio'] in prefer_set: b['score'] += 0.05

    bounds.sort(key=lambda x: (x['score'], x['ratio'] if prefer_higher else -x['ratio']), reverse=True)

    chosen, centers = [], []
    for b in bounds:
        c = float(b.get('level', (b['low']+b['high'])/2))
        if all(abs(c-cc) >= (min_spacing_atr*atr_end) for cc in centers):
            chosen.append(b); centers.append(c)
        if top_k and len(chosen) >= int(top_k): break

    chosen = sorted(chosen, key=lambda b: float(b.get('ratio', 0.0)))
    return chosen, atr_end

def print_level_divergence_hits_mobile(df, bounds, *, need_reg=None, hid_range=None, rsi_period=14, level_map=None):
    IND="      "; STAR_IND="     "
    def rsi_at(price):
        close = df['close'].astype(float).to_numpy().copy()
        close[-1]=float(price)
        r = talib.RSI(close, timeperiod=int(rsi_period))
        return float(r[-1])
    def rsi_range_on(p_lo, p_hi, samples=5):
        xs = np.linspace(p_lo, p_hi, max(2, int(samples)))
        vals = [rsi_at(x) for x in xs]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals: return (None, None)
        return float(min(vals)), float(max(vals))
    def fmtP(x): return f"{float(x):,.6f}"
    def fmtR(x): return f"{float(x):.6f}"
    bounds = sorted(bounds, key=lambda b: float(b.get('ratio', 0.0)))
    for b in bounds:
        lbl=b['label']; low, high = float(b['low']), float(b['high'])
        ctr = float(level_map.get(lbl, b.get('level', (low+high)/2.0))) if level_map else float(b.get('level',(low+high)/2.0))
        print(f"{lbl} : PRC {fmtP(ctr)}")
        rsi_lo = rsi_at(low); rsi_hi = rsi_at(high)
        print(f"{IND}· CLU [{fmtP(low)}–{fmtP(high)}]")
        print(f"{IND}· RSI  [{fmtR(rsi_lo)}–{fmtR(rsi_hi)}]")
        print(f"{IND}- REG: —")
        if hid_range is not None:
            Lh,Xh = float(hid_range[0]), float(hid_range[1])
            hid_lo, hid_hi = max(low,Lh), min(high,Xh)
            if hid_hi>hid_lo:
                star_label="★ HID: "
                print(f"{STAR_IND}{star_label}C[{fmtP(hid_lo)}–{fmtP(hid_hi)}]")
                cont_indent = STAR_IND + " " * len(star_label)
                rmin, rmax = rsi_range_on(hid_lo, hid_hi)
                print(f"{cont_indent}R[{fmtR(rmin)}–{fmtR(rmax)}]")
            else:
                print(f"{IND}- HID: —")
        else:
            print(f"{IND}- HID: —")

# =========================
# 5) Profiles & Mobile printer v2
# =========================
FIB_PROFILES = {
    'scalp':   {'fib_ratios': (1.272,1.0,0.786,0.618,0.5,0.382,0.236,0.0), 'fib_k':0.8, 'fib_max_half_mult':0.8},
    'intraday':{'fib_ratios': (1.414,1.272,1.0,0.786,0.618,0.5,0.382,0.236,0.0), 'fib_k':0.9, 'fib_max_half_mult':0.9},
    'swing':   {'fib_ratios': (1.618,1.414,1.272,1.0,0.786,0.618,0.5,0.382,0.236,0.0), 'fib_k':1.0, 'fib_max_half_mult':1.2},
}

def print_top_down_reports_mobile_v2(
    df, df_, res, * ,
    rsi_period_daily=14, rsi_period_4h=14,
    profile='intraday', top_k=None, min_spacing_atr=0.6,
    fib_ratios=None, fib_k=None, fib_mode='linear', fib_max_half_mult=None,
    stoch_oversold=25, **tags
):
    top_tag  = tags.get('top_tag',  'TOP')
    down_tag = tags.get('down_tag', 'DOWN')
    prof = FIB_PROFILES.get(profile, FIB_PROFILES['intraday'])
    fib_ratios = fib_ratios or prof['fib_ratios']
    fib_k = fib_k if fib_k is not None else prof['fib_k']
    fib_max_half_mult = fib_max_half_mult if fib_max_half_mult is not None else prof['fib_max_half_mult']

    print(f"---{top_tag}---")
    seg, _, _ = pick_oversold_segment_D_with_current_rule(df, d_col='close_stoch', oversold=20, auto_scale=False)
    if seg is None:
        print("[no top segment]")
    else:
        start_top, end_top_excl = _segment_window_half_open(df, seg)
        ts_top = _ts_series(df)
        last_included_top = ts_top[(ts_top >= start_top) & (ts_top < end_top_excl)].max()
        print(f"window={start_top} ~ {last_included_top}")
        refD = ref_from_segment_min_close(df, seg, rsi_col='rsi')
        print(f"REF: price={refD['ref_price']:.6f} | RSI={refD['ref_rsi']:.3f}")
        need_reg_D  = needed_close_regular_now(df,  ref_price=refD['ref_price'], ref_rsi=refD['ref_rsi'], rsi_period=rsi_period_daily)
        hid_range_D = feasible_close_range_hidden_now(df, ref_price=refD['ref_price'], ref_rsi=refD['ref_rsi'], rsi_period=rsi_period_daily)
        bounds_top, _ = filter_and_rank_bounds(
            df, res, fib_ratios=fib_ratios, fib_k=fib_k, fib_mode=fib_mode, fib_max_half_mult=fib_max_half_mult,
            need_reg=need_reg_D, hid_range=hid_range_D, top_k=top_k, min_spacing_atr=min_spacing_atr,
            prefer_higher=False, require_divergence=True
        )
        if not bounds_top:
            print("[no divergent levels on TOP]")
        else:
            level_map_top = make_level_map_from_res(res, ratios=FIB_RATIOS_FULL)
            print_level_divergence_hits_mobile(df, bounds_top, need_reg=need_reg_D, hid_range=hid_range_D,
                                               rsi_period=rsi_period_daily, level_map=level_map_top)

    print(f"\n---{down_tag}---")
    if seg is None:
        print("[skip: no top window]"); return
    d_start, d_end_excl = _segment_window_half_open(df, seg)
    ts4 = _ts_series(df_); last_included_4h = ts4[(ts4 >= d_start) & (ts4 < d_end_excl)].max()
    print(f"window={d_start} ~ {last_included_4h}")
    try:
        ref4 = ref_from_down_window_by_oversold(df_, d_start, d_end_excl, rsi_col='rsi', d_col='close_stoch',
                                                oversold=stoch_oversold, auto_scale=False, prefer_current=True)
    except Exception:
        try:
            ref4 = ref_from_down_window_by_oversold(df_, d_start, d_end_excl, rsi_col='rsi', d_col='close_stoch',
                                                    oversold=min(stoch_oversold+5,35), auto_scale=True, prefer_current=True)
        except Exception:
            ref4 = ref_from_down_window(df_, d_start, d_end_excl, rsi_col='rsi', prefer='close')
    print(f"REF: price={ref4['ref_price']:.6f} | RSI={ref4['ref_rsi']:.3f}")
    need_reg_4h  = needed_close_regular_now(df_, ref_price=ref4['ref_price'], ref_rsi=ref4['ref_rsi'], rsi_period=rsi_period_4h)
    hid_range_4h = feasible_close_range_hidden_now(df_, ref_price=ref4['ref_price'], ref_rsi=ref4['ref_rsi'], rsi_period=rsi_period_4h)
    bounds_down, _ = filter_and_rank_bounds(
        df, res, fib_ratios=fib_ratios, fib_k=fib_k, fib_mode=fib_mode, fib_max_half_mult=fib_max_half_mult,
        need_reg=need_reg_4h, hid_range=hid_range_4h, top_k=top_k, min_spacing_atr=min_spacing_atr,
        prefer_higher=False, require_divergence=True
    )
    if not bounds_down:
        print("[no divergent levels on DOWN]")
    else:
        level_map_down = make_level_map_from_res(res, ratios=FIB_RATIOS_FULL)
        print_level_divergence_hits_mobile(df_, bounds_down, need_reg=need_reg_4h, hid_range=hid_range_4h,
                                           rsi_period=rsi_period_4h, level_map=level_map_down)

# =========================
# 6) Minimal SelfTuner (safe defaults)
# =========================
class SelfTuner:
    def __init__(self, param_space, zig_params=None, FIB_CFG=None, search_cfg=None,
                 iters=12, batch=8, T0=0.6, T_min=0.08,
                 retune_every=200, vol_window=100, vol_jump=0.55,
                 stoch_col='close_stoch', oversold_thr=20, auto_scale=True,
                 seed=12345, stability=None):
        self.param_space = param_space
        self.zig_params  = dict(zig_params or {})
        self.best = None
        self.stoch_col = stoch_col

    def _ensure_best(self):
        if self.best is None:
            zp = self.zig_params
            self.best = {
                "stochrsi_period": 14,
                "stochrsi_fastd": 3,
                "atr_period": int(zp.get("atr_period", 21)),
                "zig_up_pct": float(zp.get("up_pct", 0.04)),
                "zig_down_pct": float(zp.get("down_pct", 0.04)),
                "zig_atr_mult": float(zp.get("atr_mult", 1.8)),
                "zig_min_bars": int(zp.get("min_bars", 7)),
                "zig_atr_period": int(zp.get("atr_period", 21)),
                "zig_threshold_mode": str(zp.get("threshold_mode", "max")),
                "min_swing_atr": float(zp.get("min_swing_atr", 1.0)),
            }

    def step(self, df, df_, print_header=False):
        self._ensure_best()
        # 최소한의 stoch 보정만 수행
        ensure_stochD_inplace(df,  col=self.stoch_col,
                              stochrsi_period=int(self.best["stochrsi_period"]),
                              fastd=int(self.best["stochrsi_fastd"]), fastd_matype=0, force=True)
        ensure_stochD_inplace(df_, col=self.stoch_col,
                              stochrsi_period=int(self.best["stochrsi_period"]),
                              fastd=int(self.best["stochrsi_fastd"]), fastd_matype=0, force=True)
        return dict(self.best)

# =========================
# 7) Single-entry runner used by the bot
# =========================
def run_top_down_report_from_df__selective(
    df_top: pd.DataFrame,
    df_down: pd.DataFrame,
    ticker: str,
    top_time: str,
    down_time: str,
    *,
    tuner_kwargs: dict | None = None,
    profile: str = 'intraday',
    top_k: int | None = 5,
    min_spacing_atr: float = 0.6,
) -> str:
    df = df_top.copy(); df_ = df_down.copy()

    def _sanitize_stochd_inplace(d, col='close_stoch'):
        if col not in d.columns: return
        s = pd.to_numeric(d[col], errors='coerce')
        if s.isna().mean() > 0.2: s = s.interpolate(limit_direction='both')
        if np.nanmax(s.to_numpy()) <= 1.0: s = s * 100.0
        d[col] = s.clip(0, 100)
    _sanitize_stochd_inplace(df, 'close_stoch'); _sanitize_stochd_inplace(df_, 'close_stoch')

    base_space = make_param_space_for_tf(top_time)
    param_space, _vol = atr_scaled_param_space(df, base_space, atr_period=21, lookback=400)

    SEARCH_CFG = dict(sample_last=600, step=10)
    zig_params = dict(up_pct=0.04, down_pct=0.04, atr_period=21, atr_mult=1.8,
                      threshold_mode='max', min_bars=7, min_swing_atr=1.0)

    prof = FIB_PROFILES.get(profile, FIB_PROFILES['intraday'])
    FIB_CFG = dict(fib_ratios=prof['fib_ratios'], fib_k=prof['fib_k'],
                   fib_mode='linear', fib_max_half_mult=prof['fib_max_half_mult'])
    STABILITY = {'lambda_stick': 0.85}

    tk = dict(param_space=param_space, zig_params=zig_params, FIB_CFG=FIB_CFG, search_cfg=SEARCH_CFG,
              iters=12, batch=8, T0=0.6, T_min=0.08, retune_every=200, vol_window=100, vol_jump=0.55,
              stoch_col='close_stoch', oversold_thr=20, auto_scale=True, seed=12345, stability=STABILITY)
    if tuner_kwargs: tk.update(tuner_kwargs)

    tuner = SelfTuner(**tk)
    best_now = tuner.step(df, df_, print_header=False)
    best_params = dict(best_now)

    # Indicators
    RP=14; AP=int(best_params.get('atr_period', 21))
    df['rsi']  = talib.RSI(df['close'].astype(float).values,  timeperiod=RP)
    df_['rsi'] = talib.RSI(df_['close'].astype(float).values, timeperiod=RP)
    df['atr']  = talib.ATR(df['high'].values, df['low'].values, df['close'].values,  timeperiod=AP)
    df_['atr'] = talib.ATR(df_['high'].values, df_['low'].values, df_['close'].values, timeperiod=AP)
    ensure_stochD_inplace(df,  col='close_stoch',
        stochrsi_period=int(best_params.get('stochrsi_period', 14)),
        fastd=int(best_params.get('stochrsi_fastd', 3)),
        fastd_matype=0, force=True
    )
    ensure_stochD_inplace(df_, col='close_stoch',
        stochrsi_period=int(best_params.get('stochrsi_period', 14)),
        fastd=int(best_params.get('stochrsi_fastd', 3)),
        fastd_matype=0, force=True
    )

    # ZigZag with best
    def _tuned_zig(base, best):
        z=dict(base or {})
        if 'zig_up_pct' in best: z['up_pct']=float(best['zig_up_pct'])
        if 'zig_down_pct' in best: z['down_pct']=float(best['zig_down_pct'])
        if 'zig_atr_mult' in best: z['atr_mult']=float(best['zig_atr_mult'])
        if 'zig_min_bars' in best: z['min_bars']=int(best['zig_min_bars'])
        if 'zig_atr_period'in best: z['atr_period']=int(best['zig_atr_period'])
        if 'zig_threshold_mode'in best: z['threshold_mode']=str(best['zig_threshold_mode'])
        if 'min_swing_atr' not in z: z['min_swing_atr']=1.0
        return z

    zp = _tuned_zig(zig_params, best_params)
    pivots,_ = zigzag_meaningful_v2(
        close=df['close'], high=df['high'], low=df['low'],
        up_pct=zp.get('up_pct',0.04), down_pct=zp.get('down_pct',0.04),
        atr_period=zp.get('atr_period',21), atr_mult=zp.get('atr_mult',1.8),
        threshold_mode=zp.get('threshold_mode','max'),
        use_hl=True, min_bars=max(5,int(zp.get('min_bars',7))),
        min_swing_atr=0.8, finalize_last=True
    )
    df['pivot']=pivots
    df['pivot_price']=np.where(df['pivot']==1, df['high'],
                        np.where(df['pivot']==-1, df['low'], np.nan))

    res = fib_from_latest(df, pivot_col='pivot', pivot_price_col='pivot_price',
                          include_extremes=True, ext_side='auto')

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"[ticker]    : {ticker}")
        print(f"[top_time]  : {top_time}")
        print(f"[down_time] : {down_time}\n")
        print_top_down_reports_mobile_v2(
            df, df_, res,
            symbol=ticker, top_tag=top_time, down_tag=down_time,
            rsi_period_daily=14, rsi_period_4h=14,
            profile=profile, top_k=top_k, min_spacing_atr=min_spacing_atr,
            stoch_oversold=25
        )
    return buf.getvalue().strip()

# (옵션) 기존 이름과의 호환성: 기존 bot 코드가 run_top_down_report_from_df(...)를 import 한다면 아래 alias 사용
def run_top_down_report_from_df(df_top, df_down, ticker, top_time, down_time, **kw):
    return run_top_down_report_from_df__selective(df_top, df_down, ticker, top_time, down_time, **kw)
