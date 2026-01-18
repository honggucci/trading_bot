"""
TradingView Identical StochRSI Implementation
=============================================

TradingView와 동일한 StochRSI 계산.
talib 라이브러리 사용.

Origin: param_search_confluence_final.py
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
import talib


def tv_stoch_rsi(
    close: np.ndarray,
    *,
    rsi_len: int = 14,
    stoch_len: int = 14,
    k_len: int = 3,
    d_len: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TradingView Identical StochRSI

    Args:
        close: 종가 배열
        rsi_len: RSI 기간 (default: 14)
        stoch_len: Stochastic 기간 (default: 14)
        k_len: %K smoothing 기간 (default: 3)
        d_len: %D smoothing 기간 (default: 3)

    Returns:
        (k, d): %K와 %D 배열 (0-100 스케일)
    """
    s = pd.Series(close, dtype='float64')

    # RSI 계산 (talib 사용)
    rsi_vals = talib.RSI(s.to_numpy(), timeperiod=int(rsi_len))
    rsi = pd.Series(rsi_vals, index=s.index)

    # Stochastic transformation
    lo = rsi.rolling(int(stoch_len), min_periods=int(stoch_len)).min()
    hi = rsi.rolling(int(stoch_len), min_periods=int(stoch_len)).max()

    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)

    # %K = SMA of stoch
    k = stoch.rolling(int(k_len), min_periods=int(k_len)).mean() * 100.0

    # %D = SMA of %K
    d = k.rolling(int(d_len), min_periods=int(d_len)).mean()

    return k.to_numpy(), d.to_numpy()


def is_oversold(stoch_d: float, threshold: float = 20.0) -> bool:
    """StochRSI %D가 oversold 영역인지 확인"""
    return stoch_d <= threshold


def is_overbought(stoch_d: float, threshold: float = 80.0) -> bool:
    """StochRSI %D가 overbought 영역인지 확인"""
    return stoch_d >= threshold


def pick_oversold_segments(
    df: pd.DataFrame,
    d_col: str = 'stoch_d',
    oversold: float = 20.0,
) -> pd.DataFrame:
    """
    StochRSI %D가 oversold인 연속 구간 찾기

    Args:
        df: DataFrame with stoch_d column
        d_col: %D 컬럼명
        oversold: oversold 임계치

    Returns:
        DataFrame with segment info (start_idx, end_idx, min_d)
    """
    if d_col not in df.columns:
        return pd.DataFrame()

    d_values = df[d_col].values
    is_os = d_values <= oversold

    segments = []
    in_segment = False
    start_idx = 0

    for i, os in enumerate(is_os):
        if os and not in_segment:
            in_segment = True
            start_idx = i
        elif not os and in_segment:
            in_segment = False
            segment_d = d_values[start_idx:i]
            segments.append({
                'start_idx': start_idx,
                'end_idx': i - 1,
                'min_d': segment_d.min(),
                'duration': i - start_idx,
            })

    # 마지막 세그먼트 처리
    if in_segment:
        segment_d = d_values[start_idx:]
        segments.append({
            'start_idx': start_idx,
            'end_idx': len(d_values) - 1,
            'min_d': segment_d.min(),
            'duration': len(d_values) - start_idx,
        })

    return pd.DataFrame(segments)


def pick_oversold_segment_with_rule(
    df: pd.DataFrame,
    d_col: str = 'stoch_d',
    oversold: float = 20.0,
    *,
    auto_scale: bool = True,
    prefer_current: bool = False,
) -> Tuple[Optional[Tuple[int, int]], float, str]:
    """
    레거시 로직: 현재가 oversold면 직전 세그먼트 사용

    규칙:
    - 현재 StochRSI %D <= oversold → 최신 세그먼트 건너뛰고 "직전" 세그먼트 선택
    - 현재 StochRSI %D > oversold → 가장 최근의 oversold 세그먼트 선택

    Args:
        df: DataFrame with stoch_d column
        d_col: %D 컬럼명
        oversold: oversold 임계치
        auto_scale: True면 0-1 스케일 자동 감지
        prefer_current: True면 현재 oversold여도 현재 세그먼트 사용

    Returns:
        (segment, threshold, reason)
        segment: (start_idx, end_idx) 또는 None
        threshold: 사용된 임계치
        reason: 선택 이유
    """
    if d_col not in df.columns:
        return None, oversold, 'no_stoch_column'

    d = df[d_col].astype(float).to_numpy()

    # 스케일 자동 감지
    if auto_scale:
        maxv = np.nanmax(d)
        thr = oversold / 100.0 if maxv <= 1.0 else oversold
    else:
        thr = oversold

    n = len(d)

    # 뒤에서부터 oversold 세그먼트 찾기
    segs = []
    i = n - 1
    while i >= 0:
        if np.isfinite(d[i]) and d[i] <= thr:
            b = i  # end
            a = i  # start
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= thr:
                a -= 1
            segs.append((a, b))
            i = a - 1
        else:
            i -= 1

    segs = segs[::-1]  # 시간순으로 정렬

    if not segs:
        return None, thr, 'no_segment'

    cur = d[-1]

    # 현재가 oversold인지 확인
    if np.isfinite(cur) and cur <= thr:
        if prefer_current:
            return segs[-1], thr, 'current_oversold_use_current'

        # 직전 세그먼트 사용
        if len(segs) >= 2:
            return segs[-2], thr, 'current_oversold_use_previous'
        else:
            return None, thr, 'current_oversold_but_no_previous'
    else:
        # 가장 최근 세그먼트 사용
        return segs[-1], thr, 'current_not_oversold_use_latest'


@dataclass
class RefPoint:
    """기준점 (REF) 정보"""
    idx: int  # DataFrame 내 위치
    ts: pd.Timestamp
    price: float
    rsi: float


def extract_ref_from_segment(
    df: pd.DataFrame,
    segment: Tuple[int, int],
    *,
    close_col: str = 'close',
    rsi_col: str = 'rsi',
) -> RefPoint:
    """
    세그먼트 내에서 최저 종가를 가진 봉의 정보 추출 (REF)

    Args:
        df: DataFrame
        segment: (start_idx, end_idx)
        close_col: 종가 컬럼명
        rsi_col: RSI 컬럼명

    Returns:
        RefPoint (기준점 정보)
    """
    a, b = segment
    sub = df.iloc[a:b+1]

    # 최저 종가 봉 찾기
    idx_min = sub[close_col].idxmin()
    iloc_min = df.index.get_loc(idx_min)

    return RefPoint(
        idx=iloc_min,
        ts=idx_min,
        price=float(df.at[idx_min, close_col]),
        rsi=float(df.at[idx_min, rsi_col]) if rsi_col in df.columns else np.nan,
    )