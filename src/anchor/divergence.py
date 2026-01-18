"""
RSI Divergence Detection
========================

RSI와 가격의 다이버전스 감지.

Types:
- Regular Bullish (REG↑): 가격 Lower Low + RSI Higher Low → 반등 신호
- Hidden Bullish (HID↑): 가격 Higher Low + RSI Lower Low → 추세 지속 신호

Origin: param_search_confluence_v0.py
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Literal
from dataclasses import dataclass

import talib


@dataclass
class DivergenceResult:
    """다이버전스 감지 결과"""
    type: Literal['regular', 'hidden', 'none']
    score: float  # 0-1 (신뢰도)
    price_ref: float  # 기준 가격
    price_current: float  # 현재 가격
    rsi_ref: float  # 기준 RSI
    rsi_current: float  # 현재 RSI
    bars_apart: int  # 기준점과의 거리 (바 수)


def calc_rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI 계산 (talib 사용)"""
    close = np.asarray(close, dtype=np.float64)
    return talib.RSI(close, timeperiod=period)


def detect_bullish_divergence(
    close: np.ndarray,
    rsi: np.ndarray,
    ref_idx: int,
    current_idx: int,
    min_bars: int = 5,
    max_bars: int = 50,
) -> DivergenceResult:
    """
    Bullish Divergence 감지 (REG↑ / HID↑)

    Args:
        close: 종가 배열
        rsi: RSI 배열
        ref_idx: 기준점 인덱스 (이전 저점)
        current_idx: 현재 인덱스
        min_bars: 최소 거리 (바 수)
        max_bars: 최대 거리 (바 수)

    Returns:
        DivergenceResult
    """
    bars_apart = current_idx - ref_idx

    # 거리 검증
    if bars_apart < min_bars or bars_apart > max_bars:
        return DivergenceResult(
            type='none',
            score=0.0,
            price_ref=close[ref_idx],
            price_current=close[current_idx],
            rsi_ref=rsi[ref_idx],
            rsi_current=rsi[current_idx],
            bars_apart=bars_apart,
        )

    price_ref = close[ref_idx]
    price_current = close[current_idx]
    rsi_ref = rsi[ref_idx]
    rsi_current = rsi[current_idx]

    # NaN 체크
    if np.isnan(rsi_ref) or np.isnan(rsi_current):
        return DivergenceResult(
            type='none',
            score=0.0,
            price_ref=price_ref,
            price_current=price_current,
            rsi_ref=rsi_ref if not np.isnan(rsi_ref) else 0.0,
            rsi_current=rsi_current if not np.isnan(rsi_current) else 0.0,
            bars_apart=bars_apart,
        )

    # Regular Bullish: 가격 LL + RSI HL
    is_regular = (price_current < price_ref) and (rsi_current > rsi_ref)

    # Hidden Bullish: 가격 HL + RSI LL
    is_hidden = (price_current > price_ref) and (rsi_current < rsi_ref)

    if is_regular:
        # Score: RSI 상승폭 + 가격 하락폭 반영
        rsi_strength = (rsi_current - rsi_ref) / max(rsi_ref, 1)
        price_weakness = (price_ref - price_current) / price_ref
        score = min(1.0, (rsi_strength + price_weakness) / 2 + 0.5)
        return DivergenceResult(
            type='regular',
            score=score,
            price_ref=price_ref,
            price_current=price_current,
            rsi_ref=rsi_ref,
            rsi_current=rsi_current,
            bars_apart=bars_apart,
        )

    if is_hidden:
        # Score: RSI 하락폭 + 가격 상승폭 반영
        rsi_weakness = (rsi_ref - rsi_current) / max(rsi_ref, 1)
        price_strength = (price_current - price_ref) / price_ref
        score = min(1.0, (rsi_weakness + price_strength) / 2 + 0.5)
        return DivergenceResult(
            type='hidden',
            score=score,
            price_ref=price_ref,
            price_current=price_current,
            rsi_ref=rsi_ref,
            rsi_current=rsi_current,
            bars_apart=bars_apart,
        )

    return DivergenceResult(
        type='none',
        score=0.0,
        price_ref=price_ref,
        price_current=price_current,
        rsi_ref=rsi_ref,
        rsi_current=rsi_current,
        bars_apart=bars_apart,
    )


def find_best_divergence(
    df: pd.DataFrame,
    current_idx: int,
    rsi_col: str = 'rsi',
    close_col: str = 'close',
    lookback: int = 50,
    min_bars: int = 5,
) -> DivergenceResult:
    """
    과거 lookback 구간에서 가장 강한 Bullish Divergence 찾기

    Args:
        df: DataFrame with rsi and close columns
        current_idx: 현재 인덱스
        rsi_col: RSI 컬럼명
        close_col: 종가 컬럼명
        lookback: 탐색 구간
        min_bars: 최소 거리

    Returns:
        가장 강한 DivergenceResult
    """
    close = df[close_col].values
    rsi = df[rsi_col].values

    best_result = DivergenceResult(
        type='none',
        score=0.0,
        price_ref=0.0,
        price_current=close[current_idx],
        rsi_ref=0.0,
        rsi_current=rsi[current_idx] if not np.isnan(rsi[current_idx]) else 0.0,
        bars_apart=0,
    )

    start_idx = max(0, current_idx - lookback)

    for ref_idx in range(start_idx, current_idx - min_bars + 1):
        result = detect_bullish_divergence(
            close, rsi, ref_idx, current_idx,
            min_bars=min_bars, max_bars=lookback,
        )

        if result.score > best_result.score:
            best_result = result

    return best_result


def compute_divergence_score(
    df: pd.DataFrame,
    current_idx: int,
    rsi_col: str = 'rsi',
    close_col: str = 'close',
    lookback: int = 50,
) -> float:
    """
    Divergence Score 계산 (0-1)

    MVP용 단순화 버전.
    Regular와 Hidden 중 더 강한 신호 반환.
    """
    result = find_best_divergence(
        df, current_idx,
        rsi_col=rsi_col,
        close_col=close_col,
        lookback=lookback,
    )
    return result.score


# ============================================================
# 레거시 로직: REF 기준 Divergence 경계 계산
# ============================================================

from functools import lru_cache


def _rsi_at_price_factory(df: pd.DataFrame, rsi_period: int = 14):
    """
    특정 가격에서의 RSI를 계산하는 팩토리 함수

    마지막 봉의 종가를 주어진 가격으로 대체하여 RSI 계산.
    캐시로 성능 최적화.
    """
    base = df['close'].astype(float).to_numpy()

    @lru_cache(maxsize=256)
    def _inner(price: float) -> float:
        arr = base.copy()
        arr[-1] = float(price)
        return float(calc_rsi_wilder(arr, period=int(rsi_period))[-1])

    return _inner


def check_divergence_at_current(
    df: pd.DataFrame,
    ref_price: float,
    ref_rsi: float,
    *,
    rsi_period: int = 14,
) -> Dict[str, any]:
    """
    현재 봉에서 REF 대비 Divergence 판정

    Args:
        df: DataFrame with close column
        ref_price: 기준점 가격
        ref_rsi: 기준점 RSI
        rsi_period: RSI 기간

    Returns:
        {
            'close_now': 현재 종가,
            'rsi_now': 현재 RSI,
            'is_reg_up': Regular Bullish 성립 여부,
            'is_hid_up': Hidden Bullish 성립 여부,
        }
    """
    close_now = float(df['close'].astype(float).iloc[-1])
    rsi_last = _rsi_at_price_factory(df, rsi_period=rsi_period)
    rsi_now = float(rsi_last(close_now))

    return {
        'close_now': close_now,
        'rsi_now': rsi_now,
        'is_reg_up': (close_now < float(ref_price)) and (rsi_now > float(ref_rsi)),
        'is_hid_up': (close_now > float(ref_price)) and (rsi_now < float(ref_rsi)),
    }


def needed_close_for_regular(
    df: pd.DataFrame,
    ref_price: float,
    ref_rsi: float,
    *,
    rsi_period: int = 14,
    lower_bound: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Optional[float]:
    """
    Regular Bullish Divergence가 성립하는 최소 가격 찾기

    조건: price < ref_price AND RSI(price) > ref_rsi

    이분 탐색으로 경계 가격을 찾는다.

    Args:
        df: DataFrame with close column
        ref_price: 기준점 가격
        ref_rsi: 기준점 RSI
        rsi_period: RSI 기간
        lower_bound: 탐색 하한 (없으면 ref_price의 90%에서 시작)
        tol: 수렴 허용 오차
        max_iter: 최대 반복 횟수

    Returns:
        Regular divergence가 성립하는 최소 가격, 또는 None
    """
    rsi_at = _rsi_at_price_factory(df, rsi_period=rsi_period)

    # 탐색 범위: [L, U] where U < ref_price
    U = float(ref_price) - 1e-8
    L = float(lower_bound) if lower_bound else U * 0.90

    # U에서 RSI > ref_rsi인지 먼저 체크
    if rsi_at(U) <= float(ref_rsi):
        return None  # U에서도 안 되면 불가능

    # L에서 체크
    if rsi_at(L) > float(ref_rsi):
        return L  # L에서도 성립하면 L 반환

    # 이분 탐색
    for _ in range(max_iter):
        if (U - L) < tol:
            break
        M = (L + U) / 2.0
        if rsi_at(M) > float(ref_rsi):
            U = M  # RSI가 높으면 더 낮은 가격으로
        else:
            L = M  # RSI가 낮으면 더 높은 가격으로

    return U if rsi_at(U) > float(ref_rsi) else None


def feasible_range_for_hidden(
    df: pd.DataFrame,
    ref_price: float,
    ref_rsi: float,
    *,
    rsi_period: int = 14,
    upper_bound: Optional[float] = None,
    scan_pts: int = 20,
) -> Optional[Tuple[float, float]]:
    """
    Hidden Bullish Divergence가 성립하는 가격 범위 찾기

    조건: price > ref_price AND RSI(price) < ref_rsi

    Args:
        df: DataFrame with close column
        ref_price: 기준점 가격
        ref_rsi: 기준점 RSI
        rsi_period: RSI 기간
        upper_bound: 탐색 상한 (없으면 ref_price의 110%까지)
        scan_pts: 스캔 포인트 수

    Returns:
        (하한, 상한) 또는 None
    """
    rsi_at = _rsi_at_price_factory(df, rsi_period=rsi_period)

    L = float(ref_price) + 1e-8
    X = float(upper_bound) if upper_bound else L * 1.10

    # 스캔
    valid_prices = []
    for i in range(scan_pts + 1):
        p = L + (X - L) * i / scan_pts
        if rsi_at(p) < float(ref_rsi):
            valid_prices.append(p)

    if not valid_prices:
        return None

    return (min(valid_prices), max(valid_prices))