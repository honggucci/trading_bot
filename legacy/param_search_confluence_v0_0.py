from __future__ import annotations

import pandas as pd
import numpy as np
import talib
from typing import Optional, Tuple, Dict, List, Any
from contextlib import redirect_stdout
import io

# 파일 상단 어딘가 공용 상수
FIB_RATIOS_FULL = (
    -1.618, -1.414, -1.272, -1.000, -0.786, -0.618, -0.500, -0.382, -0.236,
     0.000,  0.236,  0.382,  0.500,  0.618,  0.786,  1.000,  1.272,  1.414,  1.618
)

def _atr_at_end(df, res, atr_col='atr'):
    end_pos = res['swing']['end_pos']
    return float(pd.Series(df[atr_col]).iloc[:end_pos+1].dropna().iloc[-1])

def realize_window(cand, *, warmup=60, gap=30, is_min=400, oos_min=150):
    """
    후보 파라미터(cand)에서 실제 윈도우 설정을 계산.
    반환: dict(sample_last, OOS, stride, step) 또는 None(불가능)
    """
    N   = int(cand.get('N_total', 1000))
    r   = float(cand.get('OOS_ratio', 0.20))
    sm  = str(cand.get('stride_mode', 'oos')).lower()

    OOS = max(oos_min, int(round(N * r)))
    IS  = N - warmup - gap - OOS
    if IS < is_min:
        return None  # 불가능 조합

    stride = OOS if sm == 'oos' else max(1, OOS // 2)
    # 기존 평가기가 sample_last/step만 쓰므로 매핑
    step   = max(1, stride // 20)  # 촘촘/성능 밸런스(원하면 조정)
    return {'sample_last': N, 'OOS': OOS, 'IS': IS, 'gap': gap, 'stride': stride, 'step': step}

def window_reliability_penalty(oos_len, trades_oos=None, *, alpha=0.15, beta=2.0, trades_min=15):
    """
    OOS 길이가 짧을수록 점수 감점; OOS 거래수 부족 시 추가 감점.
    """
    pen = alpha / max(oos_len**0.5, 1.0)
    if trades_oos is not None and trades_oos < trades_min:
        pen += beta
    return float(pen)


## 1) Core helpers — ATR, ZigZag, Fibonacci
# 1) 우리가 쓸 타임프레임만 지원 (5m, 15m, 1h, 4h, 1d)
def make_param_space_for_tf(tf: str):
    tf = tf.lower()
    if tf == '1m':                     # ← 추가
        # 초단기라 변동폭을 더 작게, 스윙은 더 촘촘하게
        up_dn = (0.003, 0.008)        # 0.15% ~ 0.8%
        atrp  = (7, 21)
        mult  = (1.0, 2.0)
        mb    = (2, 6)
    elif tf == '5m':
        up_dn = (0.003, 0.012)   # 0.3% ~ 1.2%
        atrp  = (14, 28)
        mult  = (1.2, 2.2)
        mb    = (3, 8)
    elif tf == '15m':
        up_dn = (0.007, 0.03)    # 0.7% ~ 3.0%
        atrp  = (14, 35)
        mult  = (1.4, 2.6)
        mb    = (5, 12)
    elif tf == '1h':
        up_dn = (0.010, 0.040)   # 1.0% ~ 4.0%
        atrp  = (14, 35)
        mult  = (1.4, 2.8)
        mb    = (6, 12)
    elif tf == '4h':
        up_dn = (0.020, 0.060)   # 2.0% ~ 6.0%
        atrp  = (21, 42)
        mult  = (1.6, 3.0)
        mb    = (7, 14)
        
    elif tf == '1w':  # ← 주봉 전용 권장 범위
        up_dn = (0.040, 0.120)   # 4% ~ 12%
        atrp  = (21, 55)
        mult  = (1.8, 3.2)
        mb    = (6, 12)
        
    else:  # '1d' 등
        up_dn = (0.030, 0.080)   # 3.0% ~ 8.0%
        atrp  = (21, 42)
        mult  = (1.6, 3.0)
        mb    = (7, 14)

    return {
        # 인디케이터(원하면 범위 줄여도 됨)
        'stochrsi_period': {'type':'int','min':7,'max':34},
        'stochrsi_fastd':  {'type':'int','min':2,'max':6},
        'atr_period':      {'type':'int','min': atrp[0],'max': atrp[1]},
        # 지그재그
        'zig_up_pct':        {'type':'float','min': up_dn[0],'max': up_dn[1]},
        'zig_down_pct':      {'type':'float','min': up_dn[0],'max': up_dn[1]},
        'zig_atr_mult':      {'type':'float','min': mult[0],'max': mult[1]},
        'zig_min_bars':      {'type':'int','min': mb[0],'max':  mb[1]},
        'zig_atr_period':    {'type':'int','min': atrp[0],'max': atrp[1]},
        'zig_threshold_mode':{'type':'choice','choices':['or','and','max']},
    }

# 2) TOP/DOWN 순서 보정 (혹시 순서를 거꾸로 넣어도 TOP=큰 타임프레임으로 고정)
def canonicalize_pair(t1: str, t2: str):
    order = {'1m':0, '5m':1, '15m':2, '1h':3, '4h':4, '1d':5, '1w':6}
    a, b = t1.lower(), t2.lower()
    if order[a] >= order[b]:
        return a, b   # 이미 a가 TOP(더 큼)
    else:
        return b, a   # b를 TOP으로 승격

# 3) 변동성 기반으로 TOP 타임프레임의 up/down 허용범위 미세 조정
def vol_scaled_param_space(df_top, base_tf_space, vol_lookback=300):
    close = df_top['close'].astype(float).to_numpy()
    high  = df_top['high'].astype(float).to_numpy()
    low   = df_top['low'].astype(float).to_numpy()
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high-low, np.maximum(np.abs(high-prev_close), np.abs(low-prev_close)))
    rel = tr / np.maximum(close, 1e-9)
    v = float(np.nanmedian(rel[-vol_lookback:]))  # 예: 0.004 = 0.4%

    k_lo, k_hi = 1.2, 3.0
    up_lo = max(base_tf_space['zig_up_pct']['min'],  v*k_lo)
    up_hi = min(base_tf_space['zig_up_pct']['max'],  v*k_hi)
    dn_lo = max(base_tf_space['zig_down_pct']['min'],v*k_lo)
    dn_hi = min(base_tf_space['zig_down_pct']['max'],v*k_hi)

    ps = dict(base_tf_space)
    ps['zig_up_pct']   = {'type':'float','min': up_lo,'max': up_hi}
    ps['zig_down_pct'] = {'type':'float','min': dn_lo,'max': dn_hi}
    return ps

def compute_vol_stats(df, atr_period=21, lookback=400):
    """
    df: 반드시 ['high','low','close','open'] 포함
    반환:
      natr_med   : median(ATR/close)   → %단위 (예: 0.004 = 0.4%)
      body_med   : median(|close-open|/close)
      range_med  : median((high-low)/close)
    """
    d = df.tail(lookback).copy()
    h = d['high'].astype(float).to_numpy()
    l = d['low'].astype(float).to_numpy()
    c = d['close'].astype(float).to_numpy()
    o = d['open'].astype(float).to_numpy()

    # ATR (Wilder) – talib 우선, 실패시 수동
    try:
        atr = talib.ATR(h, l, c, timeperiod=int(atr_period))
    except Exception:
        prev_close = np.r_[c[0], c[:-1]]
        tr = np.maximum(h-l, np.maximum(np.abs(h-prev_close), np.abs(l-prev_close)))
        atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean().values

    natr = atr / np.maximum(c, 1e-9)
    body = np.abs(c - o) / np.maximum(c, 1e-9)
    rng  = (h - l)      / np.maximum(c, 1e-9)

    # robust central tendency
    natr_med  = float(np.nanmedian(natr))
    body_med  = float(np.nanmedian(body))
    range_med = float(np.nanmedian(rng))
    return natr_med, body_med, range_med

def atr_scaled_param_space(df_top, base_space,
                           atr_period=21,
                           lookback=400,
                           # up/down 퍼센트 범위를 nATR 기반으로 재설정할 때 곱해줄 계수
                           updn_k_low=1.4, updn_k_high=3.2,
                           # atr_mult도 nATR에 따라 미세조정 (기본범위 안에서)
                           mult_bias=1.0, mult_gamma=0.0,
                           # min_bars도 변동성에 따라 살짝 조정
                           min_bars_lo_shift=-2, min_bars_hi_shift=+2):
    """
    - base_space: make_param_space_for_tf(top_tf) 결과(dict)
    - nATR가 높을수록 up/down 임계치 범위를 넓히고(더 큰 변동 허용),
      nATR가 낮을수록 범위를 좁힌다.
    - atr_mult는 보수적으로: mult_bias * (nATR / ref)^gamma (gamma=0이면 고정)
    - min_bars는 변동성 높으면 소폭 증가(+), 낮으면 소폭 감소(-) 정도로만.
    """
    ps = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base_space.items()}

    # 1) 변동성 통계
    natr_med, body_med, range_med = compute_vol_stats(df_top, atr_period=atr_period, lookback=lookback)
    # 하나의 스칼라로 요약 (원하면 가중치 조정 가능)
    vol_score = 0.6 * natr_med + 0.2 * body_med + 0.2 * range_med   # 0.0~?
    # 기준 nATR (타임프레임별 대략값; 필요시 tf별 상수로 관리 가능)
    ref_natr = 0.004  # 0.4% 정도를 기준치로 가정
    scale = np.clip(vol_score / max(ref_natr, 1e-9), 0.5, 2.5)  # 보호용 클램프

    # 2) zig_up_pct / zig_down_pct 범위 재설정
    #    base_min/base_max를 기준으로 nATR*계수 범위를 CLAMP
    def _reshape_updn(key):
        lo0, hi0 = ps[key]['min'], ps[key]['max']
        lo = max(lo0, vol_score * updn_k_low)
        hi = min(hi0, vol_score * updn_k_high)
        if hi < lo:  # 극단적 케이스 보호
            mid = (lo0 + hi0)/2
            lo, hi = mid*0.8, mid*1.2
        ps[key]['min'], ps[key]['max'] = lo, hi

    _reshape_updn('zig_up_pct')
    _reshape_updn('zig_down_pct')

    # 3) zig_atr_mult 범위도 소폭 보정 (선택)
    #    mult ≈ base * (scale^gamma), gamma=0이면 그대로
    if 'zig_atr_mult' in ps and mult_gamma != 0.0:
        lo0, hi0 = ps['zig_atr_mult']['min'], ps['zig_atr_mult']['max']
        base_mid = (lo0 + hi0)/2
        adj_mid  = mult_bias * base_mid * (scale ** mult_gamma)
        # mid 주변으로 ±20% 정도의 창을 주고 base 범위와 교집합
        lo = max(lo0, adj_mid * 0.8)
        hi = min(hi0, adj_mid * 1.2)
        if hi > lo:
            ps['zig_atr_mult']['min'], ps['zig_atr_mult']['max'] = lo, hi

    # 4) min_bars 약간 조정 (변동성 높으면 더 신중하게)
    if 'zig_min_bars' in ps:
        lo0, hi0 = ps['zig_min_bars']['min'], ps['zig_min_bars']['max']
        # scale>1 → 변동성 높음 → hi 쪽으로 살짝 이동
        lo = int(np.clip(lo0 + min_bars_lo_shift, 1, hi0))
        hi = int(np.clip(hi0 + min_bars_hi_shift*min(scale,1.5), lo, hi0+5))
        ps['zig_min_bars']['min'], ps['zig_min_bars']['max'] = lo, hi

    return ps, dict(natr_med=natr_med, body_med=body_med, range_med=range_med, vol_score=vol_score)

def vol_scaled_param_space(df, base_tf_space, vol_lookback=300):
    close = df['close'].astype(float).to_numpy()
    high  = df['high'].astype(float).to_numpy()
    low   = df['low'].astype(float).to_numpy()

    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high-low, np.maximum(np.abs(high-prev_close), np.abs(low-prev_close)))
    rel = tr / np.maximum(close, 1e-9)
    v = float(np.nanmedian(rel[-vol_lookback:]))  # 예: 0.004 = 0.4%

    k_lo, k_hi = 1.2, 3.0  # 변동성 배수 밴드
    up_lo = max(base_tf_space['zig_up_pct']['min'],  v*k_lo)
    up_hi = min(base_tf_space['zig_up_pct']['max'],  v*k_hi)
    dn_lo = max(base_tf_space['zig_down_pct']['min'],v*k_lo)
    dn_hi = min(base_tf_space['zig_down_pct']['max'],v*k_hi)

    ps = dict(base_tf_space)
    ps['zig_up_pct']   = {'type':'float','min': up_lo,'max': up_hi}
    ps['zig_down_pct'] = {'type':'float','min': dn_lo,'max': dn_hi}
    return ps

def wilder_atr(high, low, close, period=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    return atr

def zigzag_meaningful_v2(
    close, high=None, low=None,
    up_pct=0.05, down_pct=0.05,
    atr_period=14, atr_mult=2.0,
    threshold_mode='or',          # 'or' | 'and' | 'max'
    use_hl=True,
    min_bars=5,
    min_swing_atr=1.0,
    finalize_last=False
):
    price = np.asarray(close, dtype=float)
    if use_hl:
        if high is None or low is None:
            raise ValueError("use_hl=True면 high/low가 필요합니다.")
        high = np.asarray(high, dtype=float)
        low  = np.asarray(low, dtype=float)
    else:
        high = price; low = price

    # ATR
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

    n = len(price)
    pivots = np.zeros(n, dtype=int)
    trend = 0
    ext_idx = 0
    ext_val = price[0]

    for i in range(1, n):
        if trend == 0:
            if moved_up(ext_val, i):
                trend = 1; ext_idx = i; ext_val = high[i]; continue
            if moved_down(ext_val, i):
                trend = -1; ext_idx = i; ext_val = low[i];  continue
        elif trend == 1:
            if high[i] > ext_val:
                ext_idx, ext_val = i, high[i]
            if moved_down(ext_val, i):
                pivots[ext_idx] = 1; trend = -1; ext_idx, ext_val = i, low[i]
        else:
            if low[i] < ext_val:
                ext_idx, ext_val = i, low[i]
            if moved_up(ext_val, i):
                pivots[ext_idx] = -1; trend = 1; ext_idx, ext_val = i, high[i]

    if finalize_last and trend != 0:
        pivots[ext_idx] = 1 if trend == 1 else -1

    # Pruning: alternate & min swing
    def extreme_at(idx, sign): return high[idx] if sign == 1 else low[idx]
    idxs = np.where(pivots != 0)[0].tolist()
    j = 1
    while j < len(idxs):
        a, b = idxs[j-1], idxs[j]
        if pivots[a] == pivots[b]:
            if pivots[a] == 1:
                keep = a if high[a] >= high[b] else b
            else:
                keep = a if low[a]  <= low[b]  else b
            drop = b if keep == a else a
            pivots[drop] = 0
            idxs.pop(j if drop == b else j-1)
            j = max(1, j-1)
        else:
            j += 1

    changed = True
    while changed and len(idxs) >= 3:
        changed = False
        k = 1
        while k < len(idxs)-1:
            a, b, c = idxs[k-1], idxs[k], idxs[k+1]
            sign_b = pivots[b]
            if (b - a) < min_bars or (c - b) < min_bars:
                pivots[b] = 0; idxs.pop(k); changed = True; continue
            amp1 = abs(extreme_at(b, sign_b) - extreme_at(a, pivots[a]))
            amp2 = abs(extreme_at(c, pivots[c]) - extreme_at(b, sign_b))
            thr  = min_swing_atr * atr[b]
            if min(amp1, amp2) < thr:
                pivots[b] = 0; idxs.pop(k); changed = True; continue
            k += 1
    return pivots, atr
# ===== Regime probability (Energy view) =====
def _ema(x, n):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([])
    a = 2.0/(n+1.0)
    out = np.empty_like(x); out[:] = np.nan
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a*x[i] + (1-a)*out[i-1]
    return out

def compute_regime_probs(df,
                         price_col='close',
                         vol_col='volume',
                         ret_win=48,         # 수익률/속도창
                         vol_win=96,         # 변동성(온도)창
                         mass_win=96):       # 거래량(질량)창
    """
    Maxwell–Boltzmann(속도 v=|r|), Black–Scholes(온도 T~σ^2), Volume(질량 m)로
    에너지 E=m*v^2, '온도' T~σ^2 ⇒ p_energy = 1-exp(-E/T) 형태를 구성.
    보수적 추세는 EMA slope/ADX 대용으로 fast/slow EMA 기울기를 사용.
    """
    import numpy as np

    c = df[price_col].astype(float).to_numpy()
    if len(c) < max(ret_win, vol_win, mass_win)+5:
        return {'p_up':0.33, 'p_rng':0.34, 'p_dn':0.33, 'trend':0.0}

    # 로그수익률
    r = np.diff(np.log(c))
    r = np.r_[0.0, r]

    # 속도 v = |r| (ret_win)
    v = np.abs(r)
    v_avg = _ema(v, ret_win)
    v_last = float(np.nan_to_num(v_avg[-1], nan=0.0))

    # 온도 T ~ σ^2 (vol_win)
    sig = pd.Series(r).rolling(vol_win).std().to_numpy()
    T = float(np.nan_to_num(sig[-1], nan=0.0)**2)
    T = max(T, 1e-8)

    # 질량 m = 거래량 / EMA(거래량) (mass_win)  (1.0이 중립)
    if vol_col in df.columns:
        vol = df[vol_col].astype(float).to_numpy()
        vema = _ema(vol, mass_win)
        m = float(np.nan_to_num(vol[-1] / max(vema[-1], 1e-12), nan=1.0))
        m = np.clip(m, 0.2, 5.0)  # 과도한 폭주 방지
    else:
        m = 1.0

    # 에너지와 에너지 기반 확률
    E = m * (v_last ** 2)
    p_energy = 1.0 - np.exp(-E / T)     # 0~1 (변동성 대비 에너지 과다하면 1에 가까움)

    # 보수적 추세 score s ∈ [-1,1]: fast/slow EMA 기울기/괴리
    fast = _ema(c, 20); slow = _ema(c, 60)
    slope = (fast[-1] - fast[-2]) / max(fast[-2], 1e-12)
    gap   = (fast[-1] / max(slow[-1],1e-12)) - 1.0
    s = np.tanh( 2.0*slope + 3.0*gap )  # 꽤 보수적 필터

    # 결합(정규화)
    p_up  = ((s+1.0)/2.0) * p_energy
    p_dn  = ((1.0 - (s+1.0)/2.0)) * p_energy
    p_rng = 1.0 - p_energy
    Z = max(1e-9, p_up + p_dn + p_rng)
    return {'p_up':p_up/Z, 'p_rng':p_rng/Z, 'p_dn':p_dn/Z, 'trend':float(s)}

def regime_gate_from_probs(probs, *, up_thr=0.58, dn_thr=0.58):
    """
    보수적 게이트: 상승만 허용/하락만 허용/횡보(둘 다 금지) 판정
    """
    u, d, r = probs['p_up'], probs['p_dn'], probs['p_rng']
    if u >= up_thr and u >= d:
        return 'up'
    if d >= dn_thr and d >= u:
        return 'down'
    return 'range'

def latest_alternating_pivots(
    df,
    pivot_col='pivot',
    pivot_price_col='pivot_price',
    *,
    allow_synthetic=True,
    synthetic_lookback=200
):
    """
    - 기본: 마지막 피벗부터 거꾸로 훑어 서로 다른 부호(+1/-1)인 두 개를 반환
    - 실패 시(교대 못 찾음, 피벗 1개 이하):
        allow_synthetic=True 이면 원시 high/low를 이용해 '반대 극값'을 합성하여 반환
        (크래시 방지)
    반환 dict:
      {start_pos, end_pos, start_ts, end_ts,
       start_sign, end_sign, start_price, end_price, direction}
    """
    import numpy as np
    pvals = df[pivot_col].to_numpy()
    idxs = np.flatnonzero(np.isfinite(pvals) & (pvals != 0))
    if len(idxs) < 2:
        # 피벗이 2개가 안되면 합성 처리로 넘김
        if not allow_synthetic:
            raise ValueError("확정 피벗이 2개 이상 필요합니다.")
        # 합성으로 진행
        return _synthetic_swing_from_raw(df, pivot_sign_hint=None, lookback=synthetic_lookback)

    # 마지막에서부터 다른 부호를 찾는다
    for k in range(len(idxs)-1, 0, -1):
        i2 = idxs[k]; i1 = idxs[k-1]
        s1, s2 = int(pvals[i1]), int(pvals[i2])
        if s1 == s2:
            # 같은 부호면 더 이전으로 계속 검색
            continue
        # 교대 쌍 확보
        p1 = float(df[pivot_price_col].iat[i1])
        p2 = float(df[pivot_price_col].iat[i2])
        if s1 == -1 and s2 == 1:
            direction = 'up'
        elif s1 == 1 and s2 == -1:
            direction = 'down'
        else:
            # 이론상 여길 타면 안되지만, 방어적으로 continue
            continue
        return {
            'start_pos': i1, 'end_pos': i2,
            'start_ts': df.index[i1], 'end_ts': df.index[i2],
            'start_sign': s1, 'end_sign': s2,
            'start_price': p1, 'end_price': p2,
            'direction': direction
        }

    # 여기까지 왔다는 건 피벗이 2개 이상인데도 교대가 안 된 케이스(같은 부호가 연속)
    if not allow_synthetic:
        raise ValueError("교대(고↔저) 피벗 쌍을 찾을 수 없습니다.")
    # 마지막 피벗의 반대 극값을 원시 high/low에서 합성
    last_idx = idxs[-1]
    last_sign = int(pvals[last_idx])
    return _synthetic_swing_from_raw(df, pivot_sign_hint=last_sign, lookback=synthetic_lookback)


def _synthetic_swing_from_raw(df, pivot_sign_hint=None, lookback=200):
    """
    최근 구간의 원시 high/low로 '반대 극값'을 합성해 최소 한 쌍의 스윙을 만든다.
    pivot_sign_hint:
      - 마지막 피벗 부호(+1 고점 / -1 저점)를 넘기면 그 반대부호 극값을 찾아 합성
      - None이면 단순히 최근 구간의 최저/최고를 시간순으로 배치
    """
    import numpy as np
    n = len(df)
    a = max(0, n - int(lookback))
    highs = df['high'].astype(float).to_numpy()
    lows  = df['low'].astype(float).to_numpy()

    # 최근 창
    H = highs[a:]
    L = lows[a:]

    # 시간 인덱스
    idx_off = a

    # 기본 극값
    loc_max = int(np.nanargmax(H)); val_max = float(H[loc_max]); pos_max = idx_off + loc_max
    loc_min = int(np.nanargmin(L)); val_min = float(L[loc_min]); pos_min = idx_off + loc_min

    # 힌트가 있으면 반대 극값을 먼저 찾는다
    if pivot_sign_hint == 1:
        # 마지막이 고점(+1)이었다면, 그 이전에서 저점(-1)을 찾아 교대 구성
        # 시간상 pos_min < pos_max 이면 up-swing, 반대면 down-swing
        if pos_min < pos_max:
            # up swing
            return {
                'start_pos': pos_min, 'end_pos': pos_max,
                'start_ts': df.index[pos_min], 'end_ts': df.index[pos_max],
                'start_sign': -1, 'end_sign': 1,
                'start_price': val_min, 'end_price': val_max,
                'direction': 'up'
            }
        else:
            # down swing (시간 순서 반대면 뒤집어서 반환)
            return {
                'start_pos': pos_max, 'end_pos': pos_min,
                'start_ts': df.index[pos_max], 'end_ts': df.index[pos_min],
                'start_sign': 1, 'end_sign': -1,
                'start_price': val_max, 'end_price': val_min,
                'direction': 'down'
            }
    elif pivot_sign_hint == -1:
        # 마지막이 저점(-1)이었다면 반대로 고점(+1)을 매칭
        if pos_max < pos_min:
            # down swing
            return {
                'start_pos': pos_max, 'end_pos': pos_min,
                'start_ts': df.index[pos_max], 'end_ts': df.index[pos_min],
                'start_sign': 1, 'end_sign': -1,
                'start_price': val_max, 'end_price': val_min,
                'direction': 'down'
            }
        else:
            # up swing
            return {
                'start_pos': pos_min, 'end_pos': pos_max,
                'start_ts': df.index[pos_min], 'end_ts': df.index[pos_max],
                'start_sign': -1, 'end_sign': 1,
                'start_price': val_min, 'end_price': val_max,
                'direction': 'up'
            }
    else:
        # 힌트가 없으면 단순·안전: 시간 순서대로 작은 쪽이 start가 되도록 구성
        if min(pos_min, pos_max) == pos_min:
            dir_ = 'up' if pos_min < pos_max else 'down'
            return {
                'start_pos': pos_min, 'end_pos': pos_max,
                'start_ts': df.index[pos_min], 'end_ts': df.index[pos_max],
                'start_sign': -1, 'end_sign': 1,
                'start_price': val_min, 'end_price': val_max,
                'direction': 'up' if pos_min < pos_max else 'down'
            }
        else:
            # pos_max가 더 앞
            return {
                'start_pos': pos_max, 'end_pos': pos_min,
                'start_ts': df.index[pos_max], 'end_ts': df.index[pos_min],
                'start_sign': 1, 'end_sign': -1,
                'start_price': val_max, 'end_price': val_min,
                'direction': 'down' if pos_max < pos_min else 'up'
            }

# 

def fib_from_latest(
    df, pivot_col='pivot', pivot_price_col='pivot_price',
    retracements=(0.236, 0.382, 0.5, 0.618, 0.786),
    extensions=(1.272, 1.414, 1.618),
    include_extremes=True,
    ext_side='above'   # ← 기본은 'auto' (상승=위, 하락=아래)
):
    swing = latest_alternating_pivots(df, pivot_col, pivot_price_col)

    if swing['direction'] == 'up':
        lo, hi = swing['start_price'], swing['end_price']
        d = (hi - lo)
        # ✅ TV: R0=hi, R100=lo
        anchors = {'0': hi, '1': lo}
        # 되돌림(고점에서 r만큼 ↓)
        rets = {r: hi - d * r for r in retracements}
        # 확장
        if ext_side == 'below':
            exts = {e: lo - d * e for e in extensions}
        else:  # 'auto' 또는 'above' => 위로
            exts = {e: hi + d * e for e in extensions}
        # 골든포켓(0.618~0.65, 고점 기준)
        gp = (hi - d * 0.65, hi - d * 0.618)

    else:  # 'down'
        hi, lo = swing['start_price'], swing['end_price']
        d = (hi - lo)
        # ✅ TV: R0=lo, R100=hi
        anchors = {'0': lo, '1': hi}
        # 되돌림(저점에서 r만큼 ↑)
        rets = {r: lo + d * r for r in retracements}
        # 확장
        if ext_side == 'above':
            exts = {e: hi + d * e for e in extensions}
        else:  # 'auto' 또는 'below' => 아래로
            exts = {e: lo - d * e for e in extensions}
        # 골든포켓(0.618~0.65, 저점 기준)
        gp = (lo + d * 0.618, lo + d * 0.65)

    if include_extremes:
        rets = {**rets, 0.0: float(anchors['0']), 1.0: float(anchors['1'])}
    gp = (min(gp), max(gp))

    return {
        'swing': swing,
        'anchors': anchors,          # ← R0/R100이 TV 기준으로 고정됨
        'retracements': rets,        # ← r<1.0일 때 여기서 가져감
        'extensions': exts,          # ← r>1.0일 때 여기서 가져감
        'golden_pocket': gp
    }






## 2) Divergence & Boundary tools
def _ts_series(df):
    return df['datetime'] if 'datetime' in df.columns else df.index

def pick_oversold_segment_D_with_current_rule(df, d_col='close_stoch', oversold=20.0, auto_scale=True,prefer_current=False):
    d = df[d_col].astype(float).to_numpy()
    if auto_scale:
        maxv = np.nanmax(d)
        thr = oversold/100.0 if maxv <= 1.0 else oversold
    else:
        thr = oversold
    n = len(d); segs = []; i = n-1
    while i >= 0:
        if np.isfinite(d[i]) and d[i] <= thr:
            b = i; a = i
            while a-1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= thr:
                a -= 1
            segs.append((a,b)); i = a - 1
        else:
            i -= 1
    segs = segs[::-1]
    if not segs: return (None, thr, 'no_segment')
    cur = d[-1]
    if np.isfinite(cur) and cur <= thr:
        if prefer_current:
            return (segs[-1], thr, 'current_oversold_use_current')
        return (segs[-2], thr, 'current_oversold_use_previous') if len(segs)>=2 \
               else (None, thr, 'current_oversold_but_no_previous')
    else:
        return (segs[-1], thr, 'current_not_oversold_use_latest')

def ref_from_segment_min_close(df, seg, *, rsi_col='rsi'):
    a, b = seg; sub = df.iloc[a:b+1]
    idx_min = sub['close'].idxmin()
    iloc_min = df.index.get_loc(idx_min)
    return {'ref_idx': iloc_min, 'ref_ts': idx_min,
            'ref_price': float(df.at[idx_min, 'close']),
            'ref_rsi': float(df.at[idx_min, rsi_col])}


def ref_from_down_window_by_oversold(df_, start_ts, end_ts_exclusive, *,
                                     rsi_col='rsi', d_col='close_stoch',
                                     oversold=20, auto_scale=True, prefer_current=False):
    """
    DOWN 창 [start, end)에서 StochRSI ≤ oversold 인 세그먼트 중
    (현재가 오버솔드면 직전 세그먼트)에서 '종가가 가장 낮은 봉'의 (가격, RSI, 시각)을 반환.
    """
    ts4 = _ts_series(df_)
    start_ts = pd.Timestamp(start_ts); end_ts_exclusive = pd.Timestamp(end_ts_exclusive)
    win = df_.loc[(ts4 >= start_ts) & (ts4 < end_ts_exclusive)].copy()
    if win.empty:
        raise ValueError("DOWN 창이 비었습니다. 시간 범위를 확인하세요.")

    seg, thr_used, reason = pick_oversold_segment_D_with_current_rule(
        win, d_col=d_col, oversold=oversold, auto_scale=auto_scale, prefer_current=prefer_current
    )
    if seg is None:
        # 창 안에 오버솔드 세그먼트가 없으면 '정규 다이버전스 기준점 없음'
        raise ValueError("창 내 StochRSI≤20 세그먼트가 없습니다.")

    ref = ref_from_segment_min_close(win, seg, rsi_col=rsi_col)
    # ref는 win(=원본 df_ 슬라이스)의 인덱스를 그대로 가지므로 그대로 반환해도 OK
    return ref  # {'ref_idx'/iloc, 'ref_ts', 'ref_price', 'ref_rsi'}


def needed_close_regular_now(df, *, ref_price, ref_rsi, rsi_period=14,
                             lower_bound=None, eps=1e-8, tol=1e-6, max_iter=60):
    close = df['close'].to_numpy(float).copy()
    def rsi_last(x):
        close[-1] = x
        return talib.RSI(close, timeperiod=int(rsi_period))[-1]
    U = ref_price - max(eps, abs(ref_price)*1e-6)
    if lower_bound is None:
        L = U - max(1e-6, abs(U)*0.10)
    else:
        L = min(lower_bound, U - max(1e-6, abs(U)*0.001))
    if not np.isfinite(U) or L >= U:
        return None
    rU = rsi_last(U)
    if (not np.isfinite(rU)) or rU <= ref_rsi:
        return None
    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi)/2.0
        rmid = rsi_last(mid)
        if not np.isfinite(rmid): lo = mid; continue
        if rmid > ref_rsi: hi = mid
        else: lo = mid
        if abs(hi - lo) <= tol: break
    return float(min(hi, U))

def feasible_close_range_hidden_now(df, *, ref_price, ref_rsi, rsi_period=14,
                                    upper_bound=None, eps=1e-8, tol=1e-6, max_iter=60):
    close = df['close'].to_numpy(float).copy()
    def rsi_last(x):
        close[-1] = x
        return talib.RSI(close, timeperiod=int(rsi_period))[-1]
    L = ref_price + max(eps, abs(ref_price)*1e-6)
    if upper_bound is None:
        U = L + max(1e-6, abs(L)*0.10)
    else:
        U = max(upper_bound, L + max(1e-6, abs(L)*0.001))
    if not np.isfinite(L) or L >= U:
        return None
    rL = rsi_last(L)
    if (not np.isfinite(rL)) or rL >= ref_rsi:
        return None
    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi)/2.0
        rmid = rsi_last(mid)
        if not np.isfinite(rmid): hi = mid; continue
        if rmid < ref_rsi: lo = mid
        else: hi = mid
        if abs(hi - lo) <= tol: break
    xmax = float(lo)
    if xmax <= L: return None
    return (float(L), xmax)

def _rsi_at_price(df, price, rsi_period=14):
    close = df['close'].astype(float).to_numpy().copy()
    close[-1] = float(price)
    r = talib.RSI(close, timeperiod=int(rsi_period))
    return float(r[-1])

def _atr_at_end(df, res, atr_col='atr'):
    end_pos = res['swing']['end_pos']
    return float(pd.Series(df[atr_col]).iloc[:end_pos+1].dropna().iloc[-1])

def build_fib_ratio_boundaries(
    df, res, * ,
    atr_col='atr',
    ratios=(0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618),
    k=1.0, mode='linear', gamma=1.0,
    max_half_mult=None,
    min_half_atr=0.25,   # ★ 추가: 앵커 최소 폭 (ATR×이 값)
    r_floor=0.12         # ★ 추가: r가 0/1 근처면 이 값으로 대체
):
    atr_end = _atr_at_end(df, res, atr_col=atr_col)
    rets = res['retracements']

    def _nearest_key(x): 
        return min(rets.keys(), key=lambda kk: abs(float(kk)-float(x)))

    def _g(r):
        # r을 '거리'로 표준화: 되돌림은 r, 확장은 (r-1)
        if r < 1.0:
            r_eff = max(float(r), r_floor)          # R0도 r_floor로 폭 부여
        elif r == 1.0:
            r_eff = r_floor                         # R100도 최소 폭
        else:
            r_eff = max(float(r) - 1.0, r_floor)    # 확장은 (r-1) 기준
        if mode == 'sqrt': return np.sqrt(r_eff)
        if mode == 'pow' : return r_eff**gamma
        return r_eff

    bounds = []
    for r in ratios:
        r = float(r)
        # 기준 레벨(center) 계산
        if r < 1.0:
            rk = r if r in rets else _nearest_key(r)
            lvl = float(rets[rk])
        elif r == 1.0:
            lvl = float(res['anchors']['1'])
        else:
            exts = res.get('extensions', {})
            if exts:
                if r in exts: lvl = float(exts[r])
                else:
                    rk = min(exts.keys(), key=lambda kk: abs(float(kk)-r))
                    lvl = float(exts[rk])
            else:
                sw = res['swing']
                if sw['direction'] == 'up':
                    lo, hi = sw['start_price'], sw['end_price']
                    lvl = float(hi + (hi - lo) * r)
                else:
                    hi, lo = sw['start_price'], sw['end_price']
                    lvl = float(lo - (hi - lo) * r)

        # 폭(half) 계산: 최소폭 보장
        half = float(k * _g(r) * atr_end)
        half = max(half, float(min_half_atr * atr_end))   # ★ 최소폭 적용
        if max_half_mult is not None:
            half = min(half, float(max_half_mult * atr_end))

        bounds.append({
            'ratio': r,
            'label': f'R{int(round(r*100))}',
            'level': lvl,
            'low':   lvl - half,
            'high':  lvl + half,
            'half':  half
        })
    return bounds, atr_end



def make_level_map_from_res(res, ratios):
    out = {}
    rets = res['retracements']
    exts = res.get('extensions', {})

    for r in ratios:
        r = float(r)
        if r == 0.0:
            price = float(res['anchors']['0'])
        elif r == 1.0:
            price = float(res['anchors']['1'])
        elif r < 1.0:
            # retracement에서 조회(가까운 키 허용)
            if r in rets:
                price = float(rets[r])
            else:
                rk = min(rets.keys(), key=lambda kk: abs(float(kk)-r))
                price = float(rets[rk])
        else:
            # extension에서 조회(가까운 키 허용) — 없으면 안전 외삽
            if exts:
                if r in exts:
                    price = float(exts[r])
                else:
                    rk = min(exts.keys(), key=lambda kk: abs(float(kk)-r))
                    price = float(exts[rk])
            else:
                sw = res['swing']
                if sw['direction'] == 'up':
                    lo, hi = sw['start_price'], sw['end_price']
                    price = float(hi + (hi - lo) * r)
                else:
                    hi, lo = sw['start_price'], sw['end_price']
                    price = float(lo - (hi - lo) * r)

        out[f"R{int(round(r*100))}"] = price
    return out





def print_level_divergence_hits(df, bounds, *, need_reg=None, hid_range=None,
                                rsi_period=14, level_map=None, show_center=True):
    import numpy as np
    if need_reg is not None:
        try: need_reg = float(need_reg)
        except: need_reg = None
    if hid_range is not None:
        try:
            Lh, Xh = float(hid_range[0]), float(hid_range[1])
            if not (np.isfinite(Lh) and np.isfinite(Xh)): hid_range = None
            else: hid_range = (Lh, Xh)
        except: hid_range = None

    def _rsi_range_on_interval(df, L, H, rsi_period=14, samples=5):
        L = float(L); H = float(H)
        if not (np.isfinite(L) and np.isfinite(H)) or H <= L:
            return (None, None)
        xs = np.linspace(L, H, max(2, int(samples)))
        vals = [ _rsi_at_price(df, x, rsi_period) for x in xs ]
        vals = [ v for v in vals if np.isfinite(v) ]
        if not vals: return (None, None)
        return (float(np.min(vals)), float(np.max(vals)))

    print("\n=== LEVEL × DIVERGENCE CHECK ===")
    for b in bounds:
        lbl, low, high = b['label'], float(b['low']), float(b['high'])
        ctr = float(level_map[lbl]) if (level_map and lbl in level_map) else float(b.get('level', (low+high)/2))
        rsi_lo = _rsi_at_price(df, low,  rsi_period)
        rsi_hi = _rsi_at_price(df, high, rsi_period)
        print(f"{lbl}: Price {ctr:,.2f} -> [{low:,.2f} ~ {high:,.2f}] | RSI[{rsi_lo:.2f} ~ {rsi_hi:.2f}]")
        print(f"   · RSI→Price (boundary): {rsi_lo:.2f}→{low:,.2f} , {rsi_hi:.2f}→{high:,.2f}")
        # Regular
        reg_hit, reg_rng = False, None
        if need_reg is not None and np.isfinite(need_reg):
            hi = min(high, need_reg)
            if hi > low: reg_hit, reg_rng = True, (low, hi)
        # Hidden
        hid_hit, hid_rng = False, None
        if hid_range is not None:
            Lh, Xh = hid_range
            lo = max(low, Lh); hi = min(high, Xh)
            if hi > lo: hid_hit, hid_rng = True, (lo, hi)
        # BOTH
        both_rng = None
        if reg_rng and hid_rng:
            lo = max(reg_rng[0], hid_rng[0]); hi = min(reg_rng[1], hid_rng[1])
            if hi > lo: both_rng = (lo, hi)
        # Print
        if reg_rng:
            Rmin, Rmax = _rsi_range_on_interval(df, reg_rng[0], reg_rng[1], rsi_period)
            print(f"   ★ Regular : YES  overlap={reg_rng[0]:,.2f}~{reg_rng[1]:,.2f}  |  RSI@Regular[{Rmin:.2f} ~ {Rmax:.2f}]")
        else:
            why = ("need_reg=None" if need_reg is None else (f"need_reg≤low ({need_reg:,.2f}≤{low:,.2f})" if need_reg <= low else "no-overlap"))
            print(f"   - Regular : no  ({why})")
        if hid_rng:
            Rmin, Rmax = _rsi_range_on_interval(df, hid_rng[0], hid_rng[1], rsi_period)
            print(f"   ★ Hidden  : YES  overlap={hid_rng[0]:,.2f}~{hid_rng[1]:,.2f}  |  RSI@Hidden [{Rmin:.2f} ~ {Rmax:.2f}]")
        else:
            print("   - Hidden  : no  (no-overlap)")
        if both_rng:
            Rmin, Rmax = _rsi_range_on_interval(df, both_rng[0], both_rng[1], rsi_period)
            print(f"   - BOTH    : {both_rng[0]:,.2f} ~ {both_rng[1]:,.2f}  |  RSI@BOTH   [{Rmin:.2f} ~ {Rmax:.2f}]")

## 3) Parameterization — Indicators & Boltzmann search
# === 추가: 순수 NumPy 버전 TV StochRSI ===


def _rsi_wilder_numpy(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return out
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
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if win <= 0 or n < win:
        return out
    csum = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    cnt  = np.cumsum(np.where(np.isfinite(x), 1.0, 0.0))
    out[win-1:] = (csum[win-1:] - np.r_[0.0, csum[:-win]]) / (cnt[win-1:] - np.r_[0.0, cnt[:-win]])
    return out

def tv_stoch_rsi_numpy(close, *, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    """
    TradingView Pine 기준:
      rsi  = ta.rsi(close, rsi_len)
      lo   = lowest(rsi, stoch_len)
      hi   = highest(rsi, stoch_len)
      stoch= (rsi - lo) / (hi - lo)     // hi==lo이면 0
      K    = sma(stoch, k_len) * 100
      D    = sma(K, d_len)
    초기 구간은 TV처럼 NaN이 납니다.
    """
    c = np.asarray(close, dtype=float)
    n = len(c)
    rsi = _rsi_wilder_numpy(c, period=int(rsi_len))

    # rolling lowest / highest (NaN 포함 구간은 NaN)
    lo = np.full(n, np.nan, dtype=float)
    hi = np.full(n, np.nan, dtype=float)
    L = int(stoch_len)
    if n >= L:
        from collections import deque
        dq_min, dq_max = deque(), deque()
        for i in range(n):
            v = rsi[i]
            # NaN 처리: 윈도우 내에 NaN 있으면 결과 NaN
            # 간단히: 인덱스 유지 deque + 별도 NaN 카운트
            # 여기서는 빠르게: 창 범위 충분할 때만 진행, 창 안에 NaN 있으면 NaN
            # 윈도우 시작
            while dq_min and dq_min[0] <= i-L: dq_min.popleft()
            while dq_max and dq_max[0] <= i-L: dq_max.popleft()
            # v가 NaN이면 그냥 푸시만 하지 않음(인덱스는 유지 못하므로 후단 처리)
            if np.isfinite(v):
                while dq_min and rsi[dq_min[-1]] >= v: dq_min.pop()
                while dq_max and rsi[dq_max[-1]] <= v: dq_max.pop()
                dq_min.append(i); dq_max.append(i)

            if i >= L-1:
                # 윈도우 내에 NaN이 있으면 dq가 창을 모두 대표 못할 수 있음 → 체크
                seg = rsi[i-L+1:i+1]
                if np.all(np.isfinite(seg)):
                    lo[i] = rsi[dq_min[0]]
                    hi[i] = rsi[dq_max[0]]
                else:
                    lo[i] = np.nan
                    hi[i] = np.nan

    denom = (hi - lo)
    stoch = np.where(np.isfinite(denom) & (denom != 0.0), (rsi - lo) / denom, 0.0)
    # 초기 구간(hi/lo NaN) → NaN 유지
    stoch[np.isnan(hi) | np.isnan(lo)] = np.nan

    k = _sma(stoch, int(k_len)) * 100.0
    d = _sma(k,     int(d_len))
    return k, d

# 기존 ensure_stochD_inplace(...) 교체
def ensure_stochD_inplace(df,
                          col: str = 'close_stoch',
                          stochrsi_period: int = 14,
                          fastd: int = 3,
                          fastd_matype: int = 0,
                          force: bool = False) -> None:
    _, d = talib.STOCHRSI(
        df['close'].astype(float).values,
        timeperiod=int(stochrsi_period),
        fastk_period=int(stochrsi_period),  # TI 방식
        fastd_period=int(fastd),
        fastd_matype=int(fastd_matype),
    )
    df[col] = d
    df.attrs['_stoch_params'] = (int(stochrsi_period), int(fastd), int(stochrsi_period), int(fastd_matype))





# =========================
# 1) 샘플러(choice/int/float 지원)
# =========================
def _sample_params(param_space, around=None, scale=0.35, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    out = {}
    for k, spec in param_space.items():
        t = spec.get('type', 'float')
        if t == 'choice':
            choices = spec['choices']
            if (around is None) or (k not in around) or (rng.random() < 0.25):
                v = rng.choice(choices)
            else:
                v = around[k]
        elif t == 'int':
            lo, hi = int(spec['min']), int(spec['max'])
            if (around is None) or (k not in around):
                v = int(rng.integers(lo, hi+1))
            else:
                mu = int(around[k]); sigma = max(1, int((hi-lo)*scale))
                v = int(np.clip(int(rng.normal(mu, sigma)), lo, hi))
        else:  # float
            lo, hi = float(spec['min']), float(spec['max'])
            if (around is None) or (k not in around):
                v = float(rng.uniform(lo, hi))
            else:
                mu = float(around[k]); sigma = max(1e-4, (hi-lo)*scale)
                v = float(np.clip(rng.normal(mu, sigma), lo, hi))
        out[k] = v
    return out


# =========================
# 2) 후보 적용 & 점수 계산
#    (네가 이미 정의해둔 함수들 사용)
#    - zigzag_meaningful_v2
#    - fib_from_latest
#    - build_fib_ratio_boundaries
#    - pick_oversold_segment_D_with_current_rule
#    - ref_from_segment_min_close
#    - needed_close_regular_now
#    - feasible_close_range_hidden_now
#    - fib_divergence_confluence
# =========================
# 기존 _compute_stochD(...) 교체
def _compute_stochD(_df, stochrsi_period, fastd):
    ensure_stochD_inplace(
        _df,
        col='close_stoch',
        stochrsi_period=int(stochrsi_period),
        fastd=int(fastd),
        fastd_matype=0,
        force=True
    )
def fib_divergence_confluence(
    df,
    res,
    zones,
    *,
    need_reg=None,
    hid_range=None,
    atr_col='atr',
    sort_by='score'
):
    """
    zones: [{'low':float, 'high':float, 'strength':1}, ...]  ← build_fib_ratio_boundaries로 만든 밴드들 대응
    반환: [{ 'label': str, 'low':float, 'high':float, 'center':float, 'score':float, 'k':dict(...) }, ...]
    """
    import numpy as np

    # 안전 처리
    if not zones:
        return []

    # ATR(말단) 대용: 최신 ATR
    atr = df[atr_col].astype(float).to_numpy()
    atr_now = float(np.nan if len(atr)==0 else np.nan_to_num(atr[-1], nan=np.nan))
    if not np.isfinite(atr_now) or atr_now <= 0:
        # 최소 보호
        atr_now = float(np.nanmax(np.asarray(atr))) if np.isfinite(np.nanmax(np.asarray(atr))) else 1.0

    # 피보 스윙 정보/골든포켓
    sw = res.get('swing', {})
    gp = res.get('golden_pocket', None)

    def _overlap(a1, a2, b1, b2):
        lo = max(a1, b1); hi = min(a2, b2)
        return (lo, hi, max(0.0, hi - lo))

    need_reg_val = None
    if need_reg is not None:
        try:
            need_reg_val = float(need_reg)
        except:
            need_reg_val = None

    hid_rng = None
    if hid_range is not None:
        try:
            Lh, Xh = float(hid_range[0]), float(hid_range[1])
            if np.isfinite(Lh) and np.isfinite(Xh) and (Xh > Lh):
                hid_rng = (Lh, Xh)
        except:
            hid_rng = None

    rows = []
    for z in zones:
        low = float(z['low']); high = float(z['high'])
        center = 0.5*(low+high)
        width  = max(1e-12, high-low)
        base_strength = float(z.get('strength', 1.0))

        # 1) 존 자체 스코어: 폭이 너무 좁거나(=잡음), 너무 넓으면(=모호) 감소
        #    폭을 ATR 기준으로 정규화
        w_norm = np.clip(width/atr_now, 0.25, 4.0)  # 0.25~4.0 ATR 사이로 클램프
        s_zone = 1.0 - abs(np.log(w_norm)) * 0.25   # ~1점에서 ± 가감

        # 2) 정규 다이버전스 가능 영역과의 겹침 (need_reg: 상한)
        s_reg = 0.0
        if need_reg_val is not None:
            lo, hi, ov = _overlap(low, high, -1e18, need_reg_val)
            if ov > 0:
                # 겹치는 폭이 넓을수록 우수. 폭/width 비율
                s_reg = (ov/width)

        # 3) 히든 다이버전스 가능 영역과의 겹침
        s_hid = 0.0
        if hid_rng is not None:
            Lh, Xh = hid_rng
            lo, hi, ov = _overlap(low, high, Lh, Xh)
            if ov > 0:
                s_hid = (ov/width)

        # 4) 골든포켓 근접 보너스(있으면)
        s_gp = 0.0
        if gp is not None:
            gp_lo, gp_hi = float(gp[0]), float(gp[1])
            # 센터가 GP 안이면 보너스, 거리가 멀수록 감소
            if gp_lo <= center <= gp_hi:
                s_gp = 0.15
            else:
                # 거리 기반 작은 보너스
                dist = min(abs(center-gp_lo), abs(center-gp_hi))
                s_gp = max(0.0, 0.10 - 0.10 * (dist / (2.0*atr_now)))  # 최대 0.1

        # 5) 방향 일치(스윙방향) 보너스(보수적으로 가중치 작게)
        s_dir = 0.0
        if sw:
            direction = sw.get('direction', None)
            # 상승 스윙이면 되돌림(센터<최근가)쪽 존, 하락 스윙이면 반대쪽 존을 약간 선호
            price_now = float(df['close'].astype(float).iloc[-1])
            if direction == 'up':
                s_dir = 0.05 if center <= price_now else 0.0
            elif direction == 'down':
                s_dir = 0.05 if center >= price_now else 0.0

        # 총 스코어
        score = (1.0 * s_zone) + (0.9 * s_reg) + (0.7 * s_hid) + s_gp + s_dir
        score *= base_strength

        rows.append({
            'label': f'[{low:.6f}~{high:.6f}]',
            'low': low, 'high': high, 'center': center,
            'score': float(score),
            'k': {
                'zone': s_zone, 'reg': s_reg, 'hid': s_hid,
                'gp': s_gp, 'dir': s_dir, 'width_atr': width/atr_now
            }
        })

    if sort_by:
        rows.sort(key=lambda r: r.get(sort_by, 0.0), reverse=True)
    return rows


def _apply_candidate_and_score(df, cand, *, zig_params, sample_last=300, step=10,
                               fib_ratios=(0.786,0.618,0.5,0.382,0.236),
                               fib_k=1.0, fib_mode='linear', fib_max_half_mult=1.2,
                               tf_hint: str = '15m'):
    # 0) 길이 체크
    N = min(len(df), int(sample_last))
    if N < 120:
        return -1e9

    # 1) 풀해상도에서 지표 계산
    tail = df.tail(N).copy()

    # RSI(14)
    if 'rsi' not in tail.columns or tail['rsi'].isna().all():
        tail['rsi'] = talib.RSI(tail['close'].astype(float).values, timeperiod=14)

    # ATR(후보)
    atr_period = int(cand.get('atr_period', 21))
    tail['atr'] = talib.ATR(
        tail['high'].astype(float).values,
        tail['low'].astype(float).values,
        tail['close'].astype(float).values,
        timeperiod=atr_period
    )

    # StochRSI D는 항상 후보값으로 재계산
    sp = int(cand.get('stochrsi_period', 14))
    fd = int(cand.get('stochrsi_fastd', 3))
    _compute_stochD(tail, sp, fd)
 
    # 2) 다운샘플
    slc = tail.iloc[::int(step)].copy()

    # 3) ZigZag 후보 적용
    zp = dict(zig_params or {})
    if 'zig_up_pct'         in cand: zp['up_pct']        = float(cand['zig_up_pct'])
    if 'zig_down_pct'       in cand: zp['down_pct']      = float(cand['zig_down_pct'])
    if 'zig_atr_mult'       in cand: zp['atr_mult']      = float(cand['zig_atr_mult'])
    if 'zig_min_bars'       in cand: zp['min_bars']      = int(cand['zig_min_bars'])
    if 'zig_atr_period'     in cand: zp['atr_period']    = int(cand['zig_atr_period'])
    if 'zig_threshold_mode' in cand: zp['threshold_mode']= str(cand['zig_threshold_mode'])

    try:
        pivots, _ = zigzag_meaningful_v2(
            close=slc['close'], high=slc['high'], low=slc['low'],
            up_pct=zp.get('up_pct',0.04), down_pct=zp.get('down_pct',0.04),
            atr_period=zp.get('atr_period',21), atr_mult=zp.get('atr_mult',1.8),
            threshold_mode=zp.get('threshold_mode','max'),
            use_hl=True, min_bars=zp.get('min_bars',7),
            min_swing_atr=1.0, finalize_last=False
        )
        slc['pivot'] = pivots
        slc['pivot_price'] = np.where(slc['pivot']==1, slc['high'],
                               np.where(slc['pivot']==-1, slc['low'], np.nan))
        # ── [안전장치] 피벗 교대 검증
        idxs = np.flatnonzero(slc['pivot'].to_numpy() != 0)
        def _alternating_ok(_idxs):
            if len(_idxs) < 2: return False
            return int(slc['pivot'].iloc[_idxs[-1]]) != int(slc['pivot'].iloc[_idxs[-2]])

        if (len(idxs) < 2) or (not _alternating_ok(idxs)):
            # 파라미터 완화해서 한 번 더 시도
            piv2, _ = zigzag_meaningful_v2(
                close=slc['close'], high=slc['high'], low=slc['low'],
                up_pct=max(1e-6, zp.get('up_pct',0.04) * 0.7),        # 30% 완화
                down_pct=max(1e-6, zp.get('down_pct',0.04) * 0.7),
                atr_period=zp.get('atr_period',21),
                atr_mult=max(0.5,  zp.get('atr_mult',1.8) * 0.8),     # 20% 완화
                threshold_mode='or',                                   # 민감도 ↑
                use_hl=True,
                min_bars=max(3, int(zp.get('min_bars',7)) - 2),       # 최소 바 수 ↓
                min_swing_atr=0.8,                                    # 스윙 필터 ↓
                finalize_last=True                                    # 마지막 피벗 확정
            )
            slc['pivot'] = piv2
            slc['pivot_price'] = np.where(slc['pivot']==1, slc['high'],
                                np.where(slc['pivot']==-1, slc['low'], np.nan))
            idxs = np.flatnonzero(slc['pivot'].to_numpy() != 0)
            if (len(idxs) < 2) or (not _alternating_ok(idxs)):
                return -1e9  # 여전히 안되면 해당 후보 탈락
        # 4) 피보나치/존
        res = fib_from_latest(slc, pivot_col='pivot', pivot_price_col='pivot_price',
                              include_extremes=True, ext_side='above')
        bounds, _ = build_fib_ratio_boundaries(
            slc, res, atr_col='atr',
            ratios=fib_ratios, k=fib_k, mode=fib_mode, max_half_mult=fib_max_half_mult
        )
        zones = [{'low':b['low'], 'high':b['high'], 'strength':1} for b in bounds]

        # 5) 오버솔드 세그먼트 → ref
        seg, _, _ = pick_oversold_segment_D_with_current_rule(
            slc, d_col='close_stoch', oversold=20, auto_scale=True
        )
        if seg is None:
            return -1e9
        ref = ref_from_segment_min_close(slc, seg, rsi_col='rsi')

        need_reg  = needed_close_regular_now(
            slc, ref_price=ref['ref_price'], ref_rsi=ref['ref_rsi'], rsi_period=14
        )
        hid_range = feasible_close_range_hidden_now(
            slc, ref_price=ref['ref_price'], ref_rsi=ref['ref_rsi'], rsi_period=14
        )

        rows = fib_divergence_confluence(
            slc, res, zones, need_reg=need_reg, hid_range=hid_range, atr_col='atr', sort_by='score'
        )
        if not rows:
            return -1e9
        score = float(np.mean([r['score'] for r in rows]))

        # === (추가) 에너지-레짐 기반 보수적 추세 게이트 ===
        probs = compute_regime_probs(slc, price_col='close', vol_col='volume' if 'volume' in slc.columns else None)
        gate  = regime_gate_from_probs(probs, up_thr=0.60, dn_thr=0.60)
        swing_dir = res['swing']['direction']  # 'up' or 'down'

        # 추세-스윙 불일치 시 강한 감점(보수적)
        if gate == 'up' and swing_dir == 'down':
            score -= 0.40
        elif gate == 'down' and swing_dir == 'up':
            score -= 0.40
        elif gate == 'range':
            score -= 0.15  # 횡보면 살짝 감점

        # === (추가) '민감 파라미터 → 거래 빈도 ↑' 유도 가산점 (소량)
        try:
            recent = slc.tail(200)
            px = recent['close'].astype(float).to_numpy()
            hits = 0
            for b in bounds:
                lo, hi = float(b['low']), float(b['high'])
                hits += int(np.sum((px >= lo) & (px <= hi)))
            # 목표 히트(거래 트리거) 수치로 유도 (너무 크면 노이즈라 과도 가산은 금지)
            target = 14.0
            # 부족할수록 penalty, 목표 부근이면 보너스 약간
            score += 0.01 * (min(hits, target) - target*0.7)  # -0.3~+0.3 근처 범위
        except Exception:
            pass
        
        # 6) 스윙/ATR 비율 패널티 (TF 힌트별 밴드)
        try:
            swing = res['swing']
            swing_amp = abs(swing['end_price'] - swing['start_price'])
            end_pos = swing['end_pos']
            atr_end = float(pd.Series(slc['atr']).iloc[:end_pos+1].dropna().iloc[-1])
            ratio = swing_amp / max(atr_end, 1e-9)

            if tf_hint in ('5m','15m'):
                lo, hi = 8.0, 25.0
            elif tf_hint in ('1h',):
                lo, hi = 6.0, 18.0
            else:
                lo, hi = 5.0, 15.0

            if ratio < lo: penalty = (lo - ratio) / lo
            elif ratio > hi: penalty = (ratio - hi) / hi
            else: penalty = 0.0

            score -= 0.5 * penalty
        except Exception:
            pass

        if not np.isfinite(score):
            return -1e9
        return score

    except Exception:
        return -1e9


def _score_candidate(df, cand, *, windows, zig_params, fib_cfg, param_space, prior=None, stick_lambda=0.0):
    scores = []
    for w in windows:
        s = _apply_candidate_and_score(
            df, cand,
            zig_params=zig_params,
            sample_last=w['sample_last'],
            step=w['step'],
            fib_ratios=fib_cfg['fib_ratios'],
            fib_k=fib_cfg['fib_k'],
            fib_mode=fib_cfg['fib_mode'],
            fib_max_half_mult=fib_cfg['fib_max_half_mult']
        )
        scores.append(s)

    base = float(np.mean(scores))

    # 스틱니스(이전 best에서 멀어질수록 페널티)
    if prior:
        dist = 0.0
        for k, spec in param_space.items():
            if (k in cand) and (k in prior):
                if spec.get('type') == 'choice':
                    continue
                if ('min' in spec) and ('max' in spec):
                    rng = (spec['max'] - spec['min']) or 1.0
                    dist += ((float(cand[k]) - float(prior[k])) / rng) ** 2
        base -= stick_lambda * dist

    return base


# =========================
# 3) 볼츠만 탐색 본체
# =========================
def boltzmann_search_confluence(df, param_space,
                                iters=15, batch=10, T0=0.6, T_min=0.08,
                                sample_last=300, step=10,
                                fib_ratios=(0.786,0.618,0.5,0.382,0.236),
                                fib_k=1.0, fib_mode='linear', fib_max_half_mult=1.2,
                                zig_params=None,
                                windows=None, seed=None, prior=None, stick_lambda=0.0):
    rng = np.random.default_rng(seed)
    # >>> 추가: 지역 변수 초기화 <<<
    best = None              # (params_dict, meta_dict) 형태로 저장할 예정
    hist = []                # [(params_dict, {'score': ...}), ...]
    best_score = -1e9        # 현재까지 최고 점수
    if not windows:
        windows = [{'sample_last': sample_last, 'step': step}]
    fib_cfg = dict(fib_ratios=fib_ratios, fib_k=fib_k, fib_mode=fib_mode, fib_max_half_mult=fib_max_half_mult)

    for it in range(1, iters+1):
        # 후보군
        if best is None:
            cands = [_sample_params(param_space, around=None, scale=0.35, rng=rng) for _ in range(batch)]
        else:
            around = best[0]  # best = (params, meta)
            cands  = [_sample_params(param_space, around=around, scale=0.25, rng=rng) for _ in range(batch)]

        batch_scores = []
        for cand in cands:
            wcfg = realize_window(cand, warmup=60, gap=30, is_min=400, oos_min=150)
            if wcfg is None:
                s = -1e9
            else:
                s = _apply_candidate_and_score(
                    df, cand, zig_params=zig_params,
                    sample_last=wcfg['sample_last'],
                    step=wcfg['step'],
                    fib_ratios=fib_ratios, fib_k=fib_k, fib_mode=fib_mode, fib_max_half_mult=fib_max_half_mult
                )
                s -= window_reliability_penalty(wcfg['OOS'])
                if cand.get('N_total', 1000) == 1000 and s > -1e8:
                    s += 0.05
            batch_scores.append((cand, {'score': s, 'window': wcfg}))



        # 정렬 및 베스트 갱신
        batch_scores.sort(key=lambda x: x[1]['score'], reverse=True)
        if (best is None) or (batch_scores[0][1]['score'] > best[1]['score']):
            best = batch_scores[0]
        hist.extend(batch_scores)

        # 로그
        T = max(T_min, T0 * (0.85 ** (it-1)))
        batch_avg = float(np.mean([m['score'] for _, m in batch_scores]))
        top_prob  = min(1.0, np.exp((batch_scores[0][1]['score'] - batch_avg) / max(1e-6, T)))
        print(f"[Iter {it}/{iters}] T={T:.3f}  best_total={best[1]['score']:.3f}  batch_avg={batch_avg:.3f}  top_prob={top_prob:.2f}")

    # >>> 추가: 안전 정렬/리턴 <<<
    hist_sorted = sorted(hist, key=lambda x: x[1]['score'], reverse=True) if hist else []
    if best is None:
        best = ({}, {'score': -1e9})  # 아무 후보도 채택 못했을 때의 안전값
    return best, hist_sorted


# =========================
# 4) 최종 적용용: 인디케이터/지그재그 생성기
# =========================
def _apply_params_inplace(df, *,
                          rsi_period=14, atr_period=21,
                          maybe_recompute_stoch=False,
                          stochrsi_period=14, stochrsi_fastd=3):
    df['rsi'] = talib.RSI(df['close'].astype(float).values, timeperiod=int(rsi_period))
    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values,
                          timeperiod=int(atr_period))
    if maybe_recompute_stoch:
        ensure_stochD_inplace(
            df,
            col='close_stoch',
            stochrsi_period=int(stochrsi_period),
            fastd=int(stochrsi_fastd),
            fastd_matype=0,
            force=True
        )

def _apply_zigzag_inplace(df,
                          up_pct, down_pct,
                          atr_period, atr_mult,
                          threshold_mode,
                          min_bars,
                          min_swing_atr=1.0,
                          finalize_last=False):
    """
    df에 pivot / pivot_price를 생성.
    """
    pivots, _ = zigzag_meaningful_v2(
        close=df['close'], high=df['high'], low=df['low'],
        up_pct=float(up_pct), down_pct=float(down_pct),
        atr_period=int(atr_period), atr_mult=float(atr_mult),
        threshold_mode=str(threshold_mode),
        use_hl=True,
        min_bars=int(min_bars),
        min_swing_atr=float(min_swing_atr),
        finalize_last=bool(finalize_last)
    )
    df['pivot'] = pivots
    df['pivot_price'] = np.where(df['pivot']==1, df['high'],
                         np.where(df['pivot']==-1, df['low'], np.nan))


# =========================
# 5) SelfTuner
# =========================
class SelfTuner:
    def __init__(self,
                 param_space,
                 zig_params=None,
                 FIB_CFG=None,
                 search_cfg=None,
                 iters=15, batch=10, T0=0.6, T_min=0.08,
                 retune_every=50, vol_window=50, vol_jump=0.35,
                 stoch_col='close_stoch', oversold_thr=20, auto_scale=True,
                 seed=42,
                 stability=None):

        self.param_space   = param_space
        # ↓ 두 줄 추가/수정 (둘 중 하나만 써도 되지만 통일을 위해 둘 다 세팅)
        self.zig_defaults  = dict(zig_params or {})   # 탐색/기본지그재그 파라미터
        self.zig_params    = dict(zig_params or {})   # 필요하면 기존 이름도 유지

        self.FIB_CFG       = dict(FIB_CFG or {})
        self.search_cfg    = dict(search_cfg or {})

        self.iters = iters; self.batch = batch
        self.T0 = T0; self.T_min = T_min

        self.retune_every = retune_every
        self.vol_window   = vol_window
        self.vol_jump     = vol_jump

        self.stoch_col    = stoch_col
        self.oversold_thr = oversold_thr
        self.auto_scale   = auto_scale

        self.best = None
        self.last_tune_pos = -1
        self.seed = seed
        self.stability = stability or {'lambda_stick': 0.5}
        
    # SelfTuner 클래스 안에 추가
    def _ensure_best_initialized(self):
        """처음 step 호출 시 self.best가 None이면 기본값 채움"""
        if getattr(self, "best", None) is None:
            zp = getattr(self, "zig_params", {}) or {}
            self.best = {
                # StochRSI D 기본
                "stochrsi_period": 14,
                "stochrsi_fastd": 3,
                # ATR 기본
                "atr_period": max(14, int(zp.get("atr_period", 21))),
                # ZigZag 기본(초기 시드)
                "zig_up_pct": float(zp.get("up_pct", 0.04)),
                "zig_down_pct": float(zp.get("down_pct", 0.04)),
                "zig_atr_mult": float(zp.get("atr_mult", 1.8)),
                "zig_min_bars": int(zp.get("min_bars", 7)),
                "zig_atr_period": int(zp.get("atr_period", 21)),
                "zig_threshold_mode": str(zp.get("threshold_mode", "max")),
            }


        
    def _need_retune(self, df):
        # 1) 초기엔 반드시 튜닝
        if self.best is None:
            return True, "first_run"
        # 2) 일정 주기마다
        if len(df) - self.last_tune_pos >= self.retune_every:
            return True, "periodic"
        # 3) 변동성 급증 감지(간단 버전)
        tail = df['close'].tail(self.vol_window).pct_change().dropna()
        if not tail.empty and tail.std() > self.vol_jump * max(1e-9, tail.rolling(10).std().mean()):
            return True, "vol_spike"
        return False, ""

    def _tune(self, df):
        windows = [
            {'sample_last': self.search_cfg.get('sample_last', 300), 'step': self.search_cfg.get('step', 10)},
            {'sample_last': max(300, self.search_cfg.get('sample_last', 300)), 'step': max(5, self.search_cfg.get('step', 10)//2)}
        ]
        best, hist_sorted = boltzmann_search_confluence(
            df, self.param_space,
            iters=self.iters, batch=self.batch, T0=self.T0, T_min=self.T_min,
            zig_params=self.zig_defaults,
            fib_ratios=self.FIB_CFG['fib_ratios'],
            fib_k=self.FIB_CFG['fib_k'], fib_mode=self.FIB_CFG['fib_mode'], fib_max_half_mult=self.FIB_CFG['fib_max_half_mult'],
            windows=windows,                 # ← 여러 윈도우
            seed=self.seed,                  # ← 고정 시드
            prior=self.best,                 # ← 이전 best (초기엔 None)
            stick_lambda=self.stability.get('lambda_stick', 0.5)
        )
        self.best = best[0]


    def _apply_best_to_df(self, df):
        bp = self.best or {}
        # RSI(고정 14)
        df['rsi'] = talib.RSI(df['close'].astype(float).values, timeperiod=14)

        # ATR: 최적값 반영
        ap = int(bp.get('atr_period', 21))
        df['atr'] = talib.ATR(
            df['high'].astype(float).values,
            df['low'].astype(float).values,
            df['close'].astype(float).values,
            timeperiod=ap
        )

        # SelfTuner._apply_best_to_df 내부
        sp = int(bp.get('stochrsi_period', 14))
        fd = int(bp.get('stochrsi_fastd', 3))
        ensure_stochD_inplace(
            df, col=self.stoch_col,
            stochrsi_period=sp, fastd=fd,
            fastd_matype=0, force=True
        )

        # ZigZag: 최적값 반영
        zp = dict(self.zig_defaults)
        if 'zig_up_pct'         in bp: zp['up_pct']        = float(bp['zig_up_pct'])
        if 'zig_down_pct'       in bp: zp['down_pct']      = float(bp['zig_down_pct'])
        if 'zig_atr_mult'       in bp: zp['atr_mult']      = float(bp['zig_atr_mult'])
        if 'zig_min_bars'       in bp: zp['min_bars']      = int(bp['zig_min_bars'])
        if 'zig_atr_period'     in bp: zp['atr_period']    = int(bp['zig_atr_period'])
        if 'zig_threshold_mode' in bp: zp['threshold_mode']= str(bp['zig_threshold_mode'])

        pivots, _ = zigzag_meaningful_v2(
            close=df['close'], high=df['high'], low=df['low'],
            up_pct=zp.get('up_pct',0.04), down_pct=zp.get('down_pct',0.04),
            atr_period=zp.get('atr_period',ap), atr_mult=zp.get('atr_mult',1.8),
            threshold_mode=zp.get('threshold_mode','max'),
            use_hl=True, min_bars=zp.get('min_bars',7),
            min_swing_atr=1.0, finalize_last=False
        )
        df['pivot'] = pivots
        df['pivot_price'] = np.where(df['pivot']==1, df['high'],
                            np.where(df['pivot']==-1, df['low'], np.nan))




    def step(self, df, df_, print_header=True):
        self._ensure_best_initialized()
        # SelfTuner.step(...) 내부 맨 앞, 튜닝/평가 전에
        ensure_stochD_inplace(df, col=self.stoch_col,
                              stochrsi_period=int(self.best.get('stochrsi_period', 14)),
                              fastd=int(self.best.get('stochrsi_fastd', 3)),
                              fastd_matype=0,
                              force=True)

        need, reason = self._need_retune(df)
        if need:
            self._tune(df)
            if print_header:
                print(f"[AUTOTUNE] params => {self.best}  |  reason={reason}  |  idx={len(df)}")

        # 최적 파라미터를 전체 df에 반영(피벗 포함)
        self._apply_best_to_df(df)

        # (선택) 여기서 리포트까지 찍고 싶다면, 네가 만든 top_down_report를 쓰면 됨.
        try:
            res = fib_from_latest(df, pivot_col='pivot', pivot_price_col='pivot_price',
                                  include_extremes=True, ext_side='above')
            # 일봉 리포트(원하면 활성화)
            # top_down_report(df, df_, res, d_col=self.stoch_col, oversold=20, auto_scale=self.auto_scale,
            #                 rsi_period_daily=14, rsi_period_4h=14,
            #                 use_low_4h=False,
            #                 fib_ratios=self.FIB_CFG['fib_ratios'],
            #                 fib_k=self.FIB_CFG['fib_k'],
            #                 fib_mode=self.FIB_CFG['fib_mode'],
            #                 fib_max_half_mult=self.FIB_CFG['fib_max_half_mult'])
        except Exception as e:
            print("[WARN] reporting skipped:", e)

        return dict(self.best)  # 현재 적용 파라미터 반환

## 4) TOP/DOWN report
def _daily_segment_window(df, seg): # 이건 이제 사용하지 않으나 일단 남김
    a, b = seg; ts = _ts_series(df)
    return ts.iloc[a], ts.iloc[b]




# 이해했어요. 지금 DOWN 창이 … ~ 09:00까지만 잡혀서 05:00 캔들이 빠지고 있어요. 해결 핵심은:

# TOP 세그먼트 (a, b)의 다운 창을 [start, next_top_open) (끝 배제) 으로 잡는 것

# 즉, end_exclusive = top_ts[b+1] (다음 TOP 캔들 오픈시각)

# DOWN 필터는 ts4 >= start and ts4 < end_exclusive

# 출력할 때는 실제 포함된 DOWN 캔들 중 마지막 시각을 찍어야 05:00이 보입니다.

# 1) TOP 세그먼트를 half-open 창으로 변환

def _segment_window_half_open(df_top, seg):
    """
    TOP 세그먼트 (a,b) -> [start, next_top_open) 윈도우로 변환
    (끝 배제이므로 DOWN은 다음 TOP 바 직전까지만 포함됨)
    """
    ts = _ts_series(df_top)  # Series or DatetimeIndex
    a, b = seg
    # 공통 위치 접근
    def at_pos(x, i):
        return x[i] if isinstance(x, pd.DatetimeIndex) else x.iloc[i]
    start = pd.Timestamp(at_pos(ts, a))
    if b + 1 < len(ts):
        end_excl = pd.Timestamp(at_pos(ts, b+1))
    else:
        ts_series = ts.to_series(index=ts) if isinstance(ts, pd.DatetimeIndex) else ts
        step = (ts_series.diff().dropna().median() or pd.Timedelta(days=1))
        end_excl = pd.Timestamp(at_pos(ts, b)) + step
    return start, end_excl


# 2) 창 매핑: “TOP 세그먼트의 실제 시간창 안에서” DOWN의 최저 종가 시각을 쓰기
# ref_from_down_window를 [start, end) 반닫힘 창을 쓰고, close 기준으로 잡도록 고쳐요. (원하면 prefer='low'도 지원하게 해 둠)
def ref_from_down_window(df_, start_ts, end_ts_exclusive, *, rsi_col='rsi', prefer='close'):
    """
    DOWN 창에서 종가(min) 시각/RSI를 참조점으로 선택.
    창은 [start_ts, end_ts_exclusive) (끝 배제)로 필터링.
    """
    ts4 = _ts_series(df_)
    start_ts = pd.Timestamp(start_ts); end_ts_exclusive = pd.Timestamp(end_ts_exclusive)

    mask = (ts4 >= start_ts) & (ts4 < end_ts_exclusive)
    sub = df_.loc[mask]
    if sub.empty:
        raise ValueError("DOWN 창이 비었습니다. 시간 범위/인덱스를 확인하세요.")

    col = 'close' if str(prefer).lower() == 'close' else 'low'
    idx_min  = sub[col].astype(float).idxmin()
    ref_ts   = ts4.loc[idx_min]
    ref_price= float(df_.loc[idx_min, 'close'])  # RSI는 종가 기준
    ref_rsi  = float(df_.loc[idx_min, rsi_col]) if rsi_col in df_.columns else float('nan')
    ref_iloc = int(df_.index.get_loc(idx_min))
    return {'ref_iloc': ref_iloc, 'ref_ts': ref_ts, 'ref_price': ref_price, 'ref_rsi': ref_rsi}


def print_top_down_reports(df, df_, res, *, rsi_period_daily=14, rsi_period_4h=14,
                           fib_ratios=(1.0,0.786,0.618,0.5,0.382,0.236,0.0), fib_k=1.0, fib_mode='linear', fib_max_half_mult=1.2,
                           use_low_4h=False, stoch_oversold=25,  # 새로 추가
                            stoch_overbought=75, # 새로 추가
                            **tags):
    top_tag  = tags.get('top_tag',  '1d')
    down_tag = tags.get('down_tag', '4h')
    bounds, atr_end = build_fib_ratio_boundaries(
        df, res, atr_col='atr', ratios=fib_ratios, k=fib_k, mode=fib_mode, max_half_mult=fib_max_half_mult
    )
    level_map = make_level_map_from_res(res, ratios=fib_ratios)

    print("\n---TOP---")
    seg, thr_used, reason = pick_oversold_segment_D_with_current_rule(
    df, d_col='close_stoch', oversold=20, auto_scale=False)
    if seg is None:
        print("[WARN] 일봉 세그먼트를 찾지 못함")
    else:
        refD = ref_from_segment_min_close(df, seg, rsi_col='rsi')
        need_reg_D  = needed_close_regular_now(df,  ref_price=refD['ref_price'], ref_rsi=refD['ref_rsi'], rsi_period=rsi_period_daily)
        hid_range_D = feasible_close_range_hidden_now(df, ref_price=refD['ref_price'], ref_rsi=refD['ref_rsi'], rsi_period=rsi_period_daily)
        print(f"[SEG] {seg}, thr_used={thr_used}, reason={reason}")
        print(f"[REF] price={refD['ref_price']:.6f}, rsi={refD['ref_rsi']:.3f}, ts={refD['ref_ts']}")
        print(f"[REG]  need_reg={need_reg_D}")
        print(f"[HID]  hid_range={hid_range_D}")
        print(f"[ATR(end)] {atr_end:.6f}")
        print_level_divergence_hits(df, bounds, need_reg=need_reg_D, hid_range=hid_range_D,
                                    rsi_period=rsi_period_daily, level_map=level_map, show_center=True)

    # 3) TOP→DOWN 창 넘길 때 그대로 쓰기

    # 이미 TOP에서 오버솔드 세그먼트를 구해 d_start, d_end = _daily_segment_window(df, seg)로 창을 가져오죠.
    # 이걸 그대로 DOWN에 적용하면 됩니다. print_top_down_reports 안의 호출부만 바꿔요
    print("\n---DOWN---")
    if seg is None:
        print("[INFO] 일봉 세그먼트가 없어 DOWN 스킵")
        return

    # ❶ TOP 세그먼트 → [start, next_top_open) 윈도우
    d_start, d_end_excl = _segment_window_half_open(df, seg)

    # ❷ ref 추출 (끝 배제)
    try:
        # 1차: 사용자가 넘긴 stoch_oversold 반영
        ref4 = ref_from_down_window_by_oversold(
            df_, d_start, d_end_excl,
            rsi_col='rsi', d_col='close_stoch',
            oversold=stoch_oversold, auto_scale=False,  # <-- False
            prefer_current=True
        )
    except Exception:
        try:
            # 2차: 임계값 살짝 완화 (예: +5)
            ref4 = ref_from_down_window_by_oversold(
                df_, d_start, d_end_excl,
                rsi_col='rsi', d_col='close_stoch',
                oversold=min(stoch_oversold + 5, 35), auto_scale=True, prefer_current=True
            )
        except Exception:
            # 3차: 오버솔드 없어도 최소 종가 기준으로 다운 리포트는 반드시 나오게
            ref4 = ref_from_down_window(
                df_, d_start, d_end_excl, rsi_col='rsi', prefer='close'
            )


    # ❸ 보기 좋게 "실제 포함된 마지막 DOWN 캔들 시각"을 출력
    ts4 = _ts_series(df_)
    mask = (ts4 >= d_start) & (ts4 < d_end_excl)
    last_included = ts4[mask].max() if mask.any() else d_start

    need_reg_4h  = needed_close_regular_now(df_,  ref_price=ref4['ref_price'], ref_rsi=ref4['ref_rsi'], rsi_period=rsi_period_4h)
    hid_range_4h = feasible_close_range_hidden_now(df_, ref_price=ref4['ref_price'], ref_rsi=ref4['ref_rsi'], rsi_period=rsi_period_4h)

    print(f"[SEG_] used top seg={seg}  window={d_start} ~ {last_included}")
    print(f"[REF_] price={ref4['ref_price']:.6f}, rsi={ref4['ref_rsi']:.3f}, ts={ref4['ref_ts']}")
    print(f"[REG_] need_reg={need_reg_4h}")
    print(f"[HID_] hid_range={hid_range_4h}")

    print_level_divergence_hits(
        df_, bounds, need_reg=need_reg_4h, hid_range=hid_range_4h,
        rsi_period=rsi_period_4h, level_map=level_map, show_center=True
    )


## 5) ⚙️ Configure search / Load data

# df  = pd.read_csv('daily.csv', parse_dates=['datetime']).set_index('datetime')   # 예시
# df_ = pd.read_csv('4h.csv',    parse_dates=['datetime']).set_index('datetime')   # 예시

# 필수 컬럼 점검 (이미 메모리에 df/df_가 있다면 무시)
# assert {'close','high','low'}.issubset(df.columns)
# assert {'close','high','low'}.issubset(df_.columns)


## 6) 🔍 Run Boltzmann search (find best params)

# === NEW: 데이터프레임을 받아 최적화+리포트까지 한 번에 ===
def run_top_down_report_from_df(
    df_top: pd.DataFrame,
    df_down: pd.DataFrame,
    ticker: str,
    top_time: str,
    down_time: str,
    *,
    tuner_kwargs: dict | None = None,
) -> str:
    """
    bot.py에서 이미 로딩/전처리한 df_top, df_down을 받아
    - 파라미터 튜닝(SelfTuner)
    - 피벗/피보
    - TOP/DOWN 리포트 출력을 하나의 문자열로 반환
    """

    # 0) 안전 복사 + 기본 보정
    df = df_top.copy()
    df_ = df_down.copy()

    def _sanitize_stochd_inplace(d, col='close_stoch'):
        if col not in d.columns:
            return
        s = pd.to_numeric(d[col], errors='coerce')
        if s.isna().mean() > 0.2:
            s = s.interpolate(limit_direction='both')
        if np.nanmax(s.to_numpy()) <= 1.0:
            s = s * 100.0
        d[col] = s.clip(0, 100)

    _sanitize_stochd_inplace(df,  'close_stoch')
    _sanitize_stochd_inplace(df_, 'close_stoch')

    # 1) TF별 기본 탐색범위 → ATR기반 미세 조정(상단 TF 기준)
    base_space = make_param_space_for_tf(top_time)
    # 2) ATR 기반 미세 조정 → 여기서 'param_space'가 생성됩니다
    param_space, _vol = atr_scaled_param_space(
        df, base_space,
        atr_period=21, lookback=400,
        updn_k_low=1.4, updn_k_high=3.2,
        mult_bias=1.0, mult_gamma=0.0,
        min_bars_lo_shift=-1, min_bars_hi_shift=+2
    )

    # (예) run_top_down_report_from_df 안, param_space 구성 직후
    param_space.update({
        'N_total':     {'type':'choice','choices':[600, 800, 1000, 1200]},
        'OOS_ratio':   {'type':'choice','choices':[0.15, 0.20, 0.25]},
        'stride_mode': {'type':'choice','choices':['oos','half']},  # stride = OOS 또는 OOS//2
    })

    # === ✅ 바로 여기! 민감 탐색 허용 패치 === --> 보수적일때 주석
    if 'zig_up_pct' in param_space:
        param_space['zig_up_pct']['min'] = max(1e-4, param_space['zig_up_pct']['min'] * 0.7)
    if 'zig_down_pct' in param_space:
        param_space['zig_down_pct']['min'] = max(1e-4, param_space['zig_down_pct']['min'] * 0.7)
    if 'zig_atr_mult' in param_space:
        param_space['zig_atr_mult']['min'] = max(0.5, param_space['zig_atr_mult']['min'] * 0.8)
    if 'zig_min_bars' in param_space:
        param_space['zig_min_bars']['min'] = max(3, int(param_space['zig_min_bars']['min']) - 2)
    if 'zig_threshold_mode' in param_space and 'choices' in param_space['zig_threshold_mode']:
        ch = list(param_space['zig_threshold_mode']['choices'])
        if 'or' not in ch: ch.append('or')
        param_space['zig_threshold_mode']['choices'] = ch
# up_pct/down_pct : 퍼센트 임계치 (기준 extreme 대비 변화율)
# → 낮출수록 민감(피벗이 자주), 높일수록 보수적.

# atr_mult + atr_period : 변동성(ATR) 기반 임계치
# → 장이 출렁일 때는 단순 퍼센트 대신 “ATR만큼 움직였나”를 같이 봐서 거짓 시그널 필터.

# threshold_mode : 'or'|'and'|'max'

# or: 퍼센트 또는 ATR 기준 중 하나만 만족해도 전환(가장 민감)

# and: 둘 다 만족해야 전환(가장 보수적)

# max: max(up_pct*price, atr_mult*ATR)만큼 이동해야 전환(중간)

# min_bars : 피벗 간 최소 봉수(너무 촘촘하면 제거)

# min_swing_atr : 피벗-피벗 사이의 진폭이 ATR×k 미만이면 제거(노이즈 컷)

# finalize_last : 마지막 스윙을 강제로 확정(리포트에 최신 고점/저점 반영하기 좋음)

    # 2) 지그재그/피보 설정
    SEARCH_CFG = dict(sample_last=600, step=10)
    zig_params = dict(
        up_pct=0.04, down_pct=0.04,
        atr_period=21, atr_mult=1.8,
        threshold_mode='max',
        min_bars=7, min_swing_atr=1.0
    )
    FIB_CFG = dict(
        fib_ratios=(1.618, 1.414, 1.272, 1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0),
        fib_k=0.9,                # 보수적 밴드(기존 1.0 -> 0.9 권장)
        fib_mode='linear',
        fib_max_half_mult=0.9     # 밴드 상한 (ATR×0.9)
    )
    STABILITY = {'lambda_stick': 0.85}
    # 보수적
    # tk = dict(
    #     param_space=param_space,
    #     zig_params=zig_params,
    #     FIB_CFG=FIB_CFG,
    #     search_cfg=SEARCH_CFG,
    #     iters=12, batch=8, T0=0.6, T_min=0.08,
    #     retune_every=200, vol_window=100, vol_jump=0.55,
    #     stoch_col='close_stoch', oversold_thr=20, auto_scale=True,
    #     seed=12345,
    #     stability=STABILITY
    # )

    # 민감 
    tk = dict(
        param_space=param_space,
        zig_params={**zig_params,
            'up_pct': zig_params.get('up_pct', 0.04) * 0.7,      # 예) 0.028
            'down_pct': zig_params.get('down_pct', 0.04) * 0.7,  # 예) 0.028
            'atr_mult': max(0.5, zig_params.get('atr_mult', 1.8) * 0.8),  # 예) 1.44
            'threshold_mode': 'or',      # 민감
            'min_bars': max(3, zig_params.get('min_bars', 7) - 2),
            'min_swing_atr': 0.8,
        },
        FIB_CFG={**FIB_CFG, 'fib_k': 0.85, 'fib_max_half_mult': 0.8},
        search_cfg={**SEARCH_CFG,
            'sample_last': max(400, SEARCH_CFG.get('sample_last', 600)),
            'step': max(5, int(SEARCH_CFG.get('step', 10) * 0.7)),
        },
        iters=16, batch=10, T0=0.8, T_min=0.06,
        retune_every=80,
        vol_window=80, vol_jump=0.35,
        stoch_col='close_stoch', oversold_thr=20, auto_scale=True,
        seed=12345,
        stability={**STABILITY, 'lambda_stick': 0.30})
    if tuner_kwargs:
        tk.update(tuner_kwargs)

    tuner = SelfTuner(**tk)

    # 3) 튜닝 실행
    best_now = tuner.step(df, df_, print_header=False)   # df(상단TF) 기반으로 튜닝
    best_params = dict(best_now)

    # 4) 튠 결과로 지그/피보 계산 (상단 TF)
    #   - 리포트에서는 ATR 레벨폭을 쓰므로, ATR/RSI 14를 튠값으로 재계산
    RP = 14
    AP = int(best_params.get('atr_period', 21))
    df['rsi']  = talib.RSI(df['close'].astype(float).values,  timeperiod=RP)
    df_['rsi'] = talib.RSI(df_['close'].astype(float).values, timeperiod=RP)
    df['atr']  = talib.ATR(df['high'].values, df['low'].values, df['close'].values,  timeperiod=AP)
    df_['atr'] = talib.ATR(df_['high'].values, df_['low'].values, df_['close'].values, timeperiod=AP)
    ensure_stochD_inplace(df,  col='close_stoch',
        stochrsi_period=int(best_params.get('stochrsi_period', 14)),
        fastd=int(best_params.get('stochrsi_fastd', 3)),
        fastd_matype=0,
        force=True
    )
    ensure_stochD_inplace(df_, col='close_stoch',
        stochrsi_period=int(best_params.get('stochrsi_period', 14)),
        fastd=int(best_params.get('stochrsi_fastd', 3)),
        fastd_matype=0,
        force=True
    )

    def _tuned_zig(base, best):
        z = dict(base or {})
        if 'zig_up_pct'         in best: z['up_pct']        = float(best['zig_up_pct'])
        if 'zig_down_pct'       in best: z['down_pct']      = float(best['zig_down_pct'])
        if 'zig_atr_mult'       in best: z['atr_mult']      = float(best['zig_atr_mult'])
        if 'zig_min_bars'       in best: z['min_bars']      = int(best['zig_min_bars'])
        if 'zig_atr_period'     in best: z['atr_period']    = int(best['zig_atr_period'])
        if 'zig_threshold_mode' in best: z['threshold_mode']= str(best['zig_threshold_mode'])
        if 'min_swing_atr' not in z: z['min_swing_atr'] = 1.0
        return z

    zp = _tuned_zig(zig_params, best_params)

    pivots, _ = zigzag_meaningful_v2(
        close=df['close'], high=df['high'], low=df['low'],
        up_pct=zp.get('up_pct',0.04), down_pct=zp.get('down_pct',0.04),
        atr_period=zp.get('atr_period',21), atr_mult=zp.get('atr_mult',1.8),
        threshold_mode=zp.get('threshold_mode','max'),
        use_hl=True, min_bars=max(5, int(zp.get('min_bars',7))),
        min_swing_atr=0.8,                 # ← 살짝 완화
        finalize_last=True                 # ← 마지막 피벗 확정
    )

    df['pivot'] = pivots
    df['pivot_price'] = np.where(df['pivot']==1, df['high'],
                          np.where(df['pivot']==-1, df['low'], np.nan))

    res = fib_from_latest(
            df, pivot_col='pivot', pivot_price_col='pivot_price',
            include_extremes=True,
            ext_side='above'     # ⬅️ 확장을 항상 위로
            )

    # 5) 헤더(튜닝 결과) + 본문(리포트) 캡처
    header = print_params_pretty_to_str(best_params, zp, ticker, top_time, down_time, rsi_period_fixed=14)

    buf = io.StringIO()
    
    # 완전 보고서용
    # with redirect_stdout(buf):
    #     print_top_down_reports(
    #         df, df_, res,
    #         top_tag=top_time, down_tag=down_time,
    #         rsi_period_daily=14, rsi_period_4h=14,
    #         fib_ratios=(1.618, 1.414, 1.272, 1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0),
    #         fib_k=0.9, fib_mode='linear', fib_max_half_mult=0.9,
    #         use_low_4h=False
    #     )
    
    # 모바일용
    with redirect_stdout(buf):
        print(f"[ticker]    : {ticker}")
        print(f"[top_time]  : {top_time}")
        print(f"[down_time] : {down_time}\n")
        print_top_down_reports_mobile(
            df, df_, res,
            symbol=ticker, top_tag=top_time, down_tag=down_time,
            rsi_period_daily=14, rsi_period_4h=14,
            fib_ratios=(1.618, 1.414, 1.272, 1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0),
            fib_k=1.0, fib_mode='linear', fib_max_half_mult=1.2,
            stoch_oversold=25
        )
    body = buf.getvalue().strip()
    return body


def print_params_pretty_to_str(best_params: dict, zig_params_now: dict,
                               symbol: str, top_tf: str, down_tf: str,
                               rsi_period_fixed: int = 14) -> str:
    sp  = int(best_params.get('stochrsi_period', 14))
    fd  = int(best_params.get('stochrsi_fastd', 3))
    ap  = int(best_params.get('atr_period', 21))

    z   = zig_params_now or {}
    up  = z.get('up_pct')
    dn  = z.get('down_pct')
    am  = z.get('atr_mult')
    n  = best_params.get('N_total', '-')
    rr = best_params.get('OOS_ratio', '-')
    sm = best_params.get('stride_mode', '-')
    def pct(x): return f"{x*100:.2f}%" if isinstance(x,(int,float)) else "-"

    lines = []
    lines.append("[Window]")
    lines.append(f"  • N_total           : {n}")
    lines.append(f"  • OOS_ratio         : {rr}")
    lines.append(f"  • stride_mode       : {sm}")
    lines.append(f"[ticker]    : {symbol}")
    lines.append(f"[top_time]  : {top_tf}")
    lines.append(f"[down_time] : {down_tf}\n")
    lines.append("=== Tuned Parameters ===")
    lines.append("[Indicators]")
    lines.append(f"  • RSI period        : {rsi_period_fixed} (fixed)")
    lines.append(f"  • StochRSI period   : {sp}")
    lines.append(f"  • StochRSI fastD    : {fd}")
    lines.append(f"  • ATR period        : {ap}\n")
    lines.append("[ZigZag]")
    lines.append(f"  • up_pct            : {pct(up)}")
    lines.append(f"  • down_pct          : {pct(dn)}")
    lines.append(f"  • atr_period        : {z.get('atr_period','-')}")
    lines.append(f"  • atr_mult          : {am:.4f}" if isinstance(am,(int,float)) else "  • atr_mult          : -")
    lines.append(f"  • threshold_mode    : {z.get('threshold_mode','-')}")
    lines.append(f"  • min_bars          : {z.get('min_bars','-')}")
    lines.append(f"  • min_swing_atr     : {z.get('min_swing_atr','-')}")
    return "\n".join(lines)


# 모바일용 보고서
# ========== 모바일 리포트(요청 포맷) ==========
# ── 모바일용: 단문·가독성 포맷 (레벨별 2줄 + REG/HID)
def print_level_divergence_hits_mobile(
    df, bounds, *, need_reg=None, hid_range=None, rsi_period=14, level_map=None
):
    IND = "      "      # 6칸
    STAR_IND = "     "  # 5칸

    def rsi_at(price):
        close = df['close'].astype(float).to_numpy().copy()
        close[-1] = float(price)
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

    # 오름차순(비율) 정렬
    bounds = sorted(bounds, key=lambda b: float(b.get('ratio', 0.0)))

    # need_reg/hid_range 정규화
    need_reg_val = None
    if need_reg is not None:
        try:
            need_reg_val = float(need_reg)
        except:
            need_reg_val = None
    hid_rng = None
    if hid_range is not None:
        try:
            Lh, Xh = float(hid_range[0]), float(hid_range[1])
            if np.isfinite(Lh) and np.isfinite(Xh) and (Xh > Lh):
                hid_rng = (Lh, Xh)
        except:
            hid_rng = None

    for b in bounds:
        lbl = b['label']          # e.g. "R62"
        low, high = float(b['low']), float(b['high'])
        ctr = float(level_map.get(lbl, b.get('level', (low + high) / 2.0))) if level_map else float(b.get('level', (low + high) / 2.0))

        # 헤더
        print(f"{lbl} : PRC {fmtP(ctr)}")
        # CLU / RSI
        rsi_lo = rsi_at(low); rsi_hi = rsi_at(high)
        print(f"{IND}· CLU [{fmtP(low)}–{fmtP(high)}]")
        print(f"{IND}· RSI  [{fmtR(rsi_lo)}–{fmtR(rsi_hi)}]")

        # ── REG: boundary[low, high] ∩ (-∞, need_reg] 가 존재하면 히트
        reg_hit = False
        reg_lo = reg_hi = None
        if need_reg_val is not None:
            hi = min(high, need_reg_val)
            if hi > low:
                reg_hit = True
                reg_lo, reg_hi = low, hi

        if reg_hit:
            rmin, rmax = rsi_range_on(reg_lo, reg_hi)
            # 1줄: 가격 범위
            print(f"{STAR_IND}★ REG: C[{fmtP(reg_lo)}–{fmtP(reg_hi)}]")
            # 2줄: RSI 범위
            cont_indent = STAR_IND + " " * len("★ REG: ")
            print(f"{cont_indent}R[{fmtR(rmin)}–{fmtR(rmax)}]")
        else:
            print(f"{IND}- REG: —")

        # ── HID: boundary[low, high] ∩ hid_range[Lh, Xh] 가 존재하면 히트
        if hid_rng is not None:
            Lh, Xh = hid_rng
            lo = max(low, Lh); hi = min(high, Xh)
            if hi > lo:
                rmin, rmax = rsi_range_on(lo, hi)
                print(f"{STAR_IND}★ HID: C[{fmtP(lo)}–{fmtP(hi)}]")
                cont_indent = STAR_IND + " " * len("★ HID: ")
                print(f"{cont_indent}R[{fmtR(rmin)}–{fmtR(rmax)}]")
            else:
                print(f"{IND}- HID: —")
        else:
            print(f"{IND}- HID: —")




def print_top_down_reports_mobile(df, df_, res, * ,
                                  rsi_period_daily=14, rsi_period_4h=14,
                                  # 확장 포함: 1.618, 1.414, 1.272 추가
                                  fib_ratios=(1.618,1.414,1.272,1.0,0.786,0.618,0.5,0.382,0.236,0.0),
                                  fib_k=1.0, fib_mode='linear', fib_max_half_mult=1.2,
                                  stoch_oversold=25, **tags):
    top_tag  = tags.get('top_tag',  'TOP')
    down_tag = tags.get('down_tag', 'DOWN')

    # 공통 레벨/경계
    bounds, _ = build_fib_ratio_boundaries(
        df, res, atr_col='atr', ratios=fib_ratios, k=fib_k, mode=fib_mode, max_half_mult=fib_max_half_mult
    )
    level_map = make_level_map_from_res(res, ratios=fib_ratios)  # ← 확장 레벨 포함 센터값

    # === TOP ===
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

        print_level_divergence_hits_mobile(
            df, bounds, need_reg=need_reg_D, hid_range=hid_range_D,
            rsi_period=rsi_period_daily, level_map=level_map
        )

    # === DOWN ===
    print(f"\n---{down_tag}---")
    if seg is None:
        print("[skip: no top window]")
        return

    d_start, d_end_excl = _segment_window_half_open(df, seg)
    ts4 = _ts_series(df_)
    last_included_4h = ts4[(ts4 >= d_start) & (ts4 < d_end_excl)].max()
    print(f"window={d_start} ~ {last_included_4h}")

    # 오버솔드 우선 REF, 실패 시 완화 → 최저 종가
    try:
        ref4 = ref_from_down_window_by_oversold(
            df_, d_start, d_end_excl, rsi_col='rsi', d_col='close_stoch',
            oversold=stoch_oversold, auto_scale=False, prefer_current=True
        )
    except Exception:
        try:
            ref4 = ref_from_down_window_by_oversold(
                df_, d_start, d_end_excl, rsi_col='rsi', d_col='close_stoch',
                oversold=min(stoch_oversold+5, 35), auto_scale=True, prefer_current=True
            )
        except Exception:
            ref4 = ref_from_down_window(df_, d_start, d_end_excl, rsi_col='rsi', prefer='close')

    print(f"REF: price={ref4['ref_price']:.6f} | RSI={ref4['ref_rsi']:.3f}")

    need_reg_4h  = needed_close_regular_now(df_, ref_price=ref4['ref_price'], ref_rsi=ref4['ref_rsi'], rsi_period=rsi_period_4h)
    hid_range_4h = feasible_close_range_hidden_now(df_, ref_price=ref4['ref_price'], ref_rsi=ref4['ref_rsi'], rsi_period=rsi_period_4h)

    print_level_divergence_hits_mobile(
        df_, bounds, need_reg=need_reg_4h, hid_range=hid_range_4h,
        rsi_period=rsi_period_4h, level_map=level_map
    )
