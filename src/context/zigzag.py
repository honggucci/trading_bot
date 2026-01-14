"""
ZigZag Pivot Detection
======================

스윙 고점/저점을 찾아 피보나치 되돌림 계산의 기반이 되는 피벗 포인트 추출.

핵심 원리:
- up_pct/down_pct: 최소 변동폭 (퍼센트 기준)
- atr_mult: ATR 기반 최소 변동폭
- threshold_mode: 'or' (둘 중 하나), 'and' (둘 다), 'max' (둘 중 큰 것)
- min_bars: 피벗 간 최소 간격
- min_swing_atr: 스윙 폭이 ATR의 몇 배 이상이어야 유효한지

Origin: param_search_confluence_v0.py의 zigzag_meaningful_v2()
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Literal
from dataclasses import dataclass


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR 계산"""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    return atr


def zigzag_pivots(
    close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    *,
    up_pct: float = 0.05,
    down_pct: float = 0.05,
    atr_period: int = 14,
    atr_mult: float = 2.0,
    threshold_mode: Literal['or', 'and', 'max'] = 'or',
    use_hl: bool = True,
    min_bars: int = 5,
    min_swing_atr: float = 1.0,
    finalize_last: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ZigZag 피벗 감지

    Args:
        close: 종가 배열
        high: 고가 배열 (use_hl=True일 때 필수)
        low: 저가 배열 (use_hl=True일 때 필수)
        up_pct: 상승 전환 최소 퍼센트
        down_pct: 하락 전환 최소 퍼센트
        atr_period: ATR 계산 기간
        atr_mult: ATR 배수
        threshold_mode: 'or' | 'and' | 'max'
        use_hl: True면 고/저가 사용, False면 종가만 사용
        min_bars: 피벗 간 최소 바 수
        min_swing_atr: 최소 스윙 폭 (ATR 배수)
        finalize_last: 마지막 피벗을 확정할지 여부

    Returns:
        (pivots, atr): pivots[i] = +1 (고점), -1 (저점), 0 (비피벗)
    """
    price = np.asarray(close, dtype=float)
    if use_hl:
        if high is None or low is None:
            raise ValueError("use_hl=True면 high/low가 필요합니다.")
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
    else:
        high = price.copy()
        low = price.copy()

    # ATR 계산
    atr = wilder_atr(high, low, price, period=atr_period)

    def moved_up(ext_val: float, i: int) -> bool:
        pct_ok = (high[i] - ext_val) / max(ext_val, 1e-12) >= up_pct
        atr_ok = (high[i] - ext_val) >= atr_mult * atr[i]
        if threshold_mode == 'and':
            return pct_ok and atr_ok
        if threshold_mode == 'max':
            return (high[i] - ext_val) >= max(up_pct * ext_val, atr_mult * atr[i])
        return pct_ok or atr_ok  # 'or'

    def moved_down(ext_val: float, i: int) -> bool:
        pct_ok = (ext_val - low[i]) / max(ext_val, 1e-12) >= down_pct
        atr_ok = (ext_val - low[i]) >= atr_mult * atr[i]
        if threshold_mode == 'and':
            return pct_ok and atr_ok
        if threshold_mode == 'max':
            return (ext_val - low[i]) >= max(down_pct * ext_val, atr_mult * atr[i])
        return pct_ok or atr_ok

    n = len(price)
    pivots = np.zeros(n, dtype=int)
    trend = 0  # 0: 초기, 1: 상승, -1: 하락
    ext_idx = 0
    ext_val = price[0]

    # 1차 스캔: 피벗 후보 찾기
    for i in range(1, n):
        if trend == 0:
            if moved_up(ext_val, i):
                trend = 1
                ext_idx = i
                ext_val = high[i]
                continue
            if moved_down(ext_val, i):
                trend = -1
                ext_idx = i
                ext_val = low[i]
                continue
        elif trend == 1:  # 상승 추세
            if high[i] > ext_val:
                ext_idx, ext_val = i, high[i]
            if moved_down(ext_val, i):
                pivots[ext_idx] = 1  # 고점 확정
                trend = -1
                ext_idx, ext_val = i, low[i]
        else:  # trend == -1, 하락 추세
            if low[i] < ext_val:
                ext_idx, ext_val = i, low[i]
            if moved_up(ext_val, i):
                pivots[ext_idx] = -1  # 저점 확정
                trend = 1
                ext_idx, ext_val = i, high[i]

    if finalize_last and trend != 0:
        pivots[ext_idx] = 1 if trend == 1 else -1

    # 2차: 같은 부호 연속 시 더 극단적인 것만 유지
    def extreme_at(idx: int, sign: int) -> float:
        return high[idx] if sign == 1 else low[idx]

    idxs = np.where(pivots != 0)[0].tolist()
    j = 1
    while j < len(idxs):
        a, b = idxs[j-1], idxs[j]
        if pivots[a] == pivots[b]:
            if pivots[a] == 1:  # 둘 다 고점
                keep = a if high[a] >= high[b] else b
            else:  # 둘 다 저점
                keep = a if low[a] <= low[b] else b
            drop = b if keep == a else a
            pivots[drop] = 0
            idxs.pop(j if drop == b else j-1)
            j = max(1, j - 1)
        else:
            j += 1

    # 3차: min_bars, min_swing_atr 필터
    changed = True
    while changed and len(idxs) >= 3:
        changed = False
        k = 1
        while k < len(idxs) - 1:
            a, b, c = idxs[k-1], idxs[k], idxs[k+1]
            sign_b = pivots[b]

            # 최소 바 수 체크
            if (b - a) < min_bars or (c - b) < min_bars:
                pivots[b] = 0
                idxs.pop(k)
                changed = True
                continue

            # 최소 스윙 폭 체크
            amp1 = abs(extreme_at(b, sign_b) - extreme_at(a, pivots[a]))
            amp2 = abs(extreme_at(c, pivots[c]) - extreme_at(b, sign_b))
            thr = min_swing_atr * atr[b]
            if min(amp1, amp2) < thr:
                pivots[b] = 0
                idxs.pop(k)
                changed = True
                continue

            k += 1

    return pivots, atr


@dataclass
class SwingInfo:
    """스윙 정보"""
    start_idx: int
    end_idx: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    start_sign: int  # +1: 고점, -1: 저점
    end_sign: int
    start_price: float
    end_price: float
    direction: Literal['up', 'down']


def get_latest_swing(
    df: pd.DataFrame,
    pivots: np.ndarray,
    *,
    pivot_price_col: Optional[str] = None,
) -> Optional[SwingInfo]:
    """
    마지막 교대 피벗 쌍에서 스윙 정보 추출

    Args:
        df: DataFrame (index가 timestamp)
        pivots: zigzag_pivots() 결과
        pivot_price_col: 피벗 가격 컬럼 (없으면 high/low에서 추출)

    Returns:
        SwingInfo 또는 None
    """
    idxs = np.where(pivots != 0)[0]
    if len(idxs) < 2:
        return None

    # 마지막에서 교대 쌍 찾기
    for k in range(len(idxs) - 1, 0, -1):
        i1, i2 = idxs[k-1], idxs[k]
        s1, s2 = int(pivots[i1]), int(pivots[i2])

        if s1 == s2:
            continue  # 같은 부호면 스킵

        # 가격 추출
        if pivot_price_col and pivot_price_col in df.columns:
            p1 = float(df[pivot_price_col].iloc[i1])
            p2 = float(df[pivot_price_col].iloc[i2])
        else:
            p1 = float(df['high'].iloc[i1]) if s1 == 1 else float(df['low'].iloc[i1])
            p2 = float(df['high'].iloc[i2]) if s2 == 1 else float(df['low'].iloc[i2])

        direction = 'up' if s1 == -1 and s2 == 1 else 'down'

        return SwingInfo(
            start_idx=i1,
            end_idx=i2,
            start_ts=df.index[i1],
            end_ts=df.index[i2],
            start_sign=s1,
            end_sign=s2,
            start_price=p1,
            end_price=p2,
            direction=direction,
        )

    return None


def add_pivot_columns(df: pd.DataFrame, pivots: np.ndarray) -> None:
    """
    DataFrame에 pivot 컬럼 추가 (inplace)

    추가 컬럼:
    - pivot: +1 (고점), -1 (저점), 0 (비피벗)
    - pivot_price: 피벗 가격 (고점이면 high, 저점이면 low)
    """
    df['pivot'] = pivots
    df['pivot_price'] = np.where(
        pivots == 1, df['high'],
        np.where(pivots == -1, df['low'], np.nan)
    )
