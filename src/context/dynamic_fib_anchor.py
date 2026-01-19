"""
PR-DYN-FIB: 15m Dynamic Fibonacci Anchor

15m 단기 스윙의 동적 앵커를 계산하여 Linear Fib 레벨 생성.
TP 후보 확장에 사용 (진입 필터로 사용 시 트레이드 수 사망 위험).

Anchor 갱신 모드:
- zigzag: ZigZag pivot 확정 시 갱신 (가장 정석, 후행)
- rolling: N봉 rolling high/low (단순, 휩쏘 취약)
- conditional: range >= N*ATR 일 때만 갱신 (노이즈 억제)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


@dataclass
class DynamicFibAnchorState:
    """동적 Fib 앵커 상태"""
    low: float = 0.0                        # 앵커 저점
    high: float = 0.0                       # 앵커 고점
    last_update_ts: Optional[pd.Timestamp] = None  # 마지막 갱신 시각
    mode: str = "rolling"                   # "zigzag" | "rolling" | "conditional"
    direction: str = "unknown"              # "up" | "down" | "unknown" (zigzag용)
    last_pivot_price: float = 0.0           # 마지막 확정 pivot (zigzag용)
    last_pivot_ts: Optional[pd.Timestamp] = None   # 마지막 pivot 시각

    def is_valid(self) -> bool:
        """앵커가 유효한지 확인"""
        return self.low > 0 and self.high > 0 and self.high > self.low


def update_anchor_rolling(
    df: pd.DataFrame,
    i: int,
    lookback_bars: int = 96
) -> Tuple[float, float]:
    """
    Rolling N봉 high/low 방식으로 앵커 갱신.

    Args:
        df: OHLCV DataFrame (15m)
        i: 현재 인덱스
        lookback_bars: lookback 봉 수 (기본 96 = 24h)

    Returns:
        (low, high) 튜플
    """
    start_idx = max(0, i - lookback_bars + 1)
    end_idx = i + 1

    window = df.iloc[start_idx:end_idx]

    low = window['low'].min()
    high = window['high'].max()

    return low, high


def update_anchor_conditional(
    df: pd.DataFrame,
    i: int,
    lookback_bars: int,
    atr: float,
    min_swing_mult: float = 1.5,
    prev_state: Optional[DynamicFibAnchorState] = None
) -> Tuple[float, float, bool]:
    """
    Conditional 방식: range >= min_swing_mult * ATR 일 때만 앵커 갱신.

    Args:
        df: OHLCV DataFrame (15m)
        i: 현재 인덱스
        lookback_bars: lookback 봉 수
        atr: 현재 ATR
        min_swing_mult: 최소 스윙 배수 (기본 1.5)
        prev_state: 이전 상태 (갱신 안 할 경우 유지)

    Returns:
        (low, high, updated) 튜플 - updated는 갱신 여부
    """
    # Rolling 방식으로 후보 계산
    low, high = update_anchor_rolling(df, i, lookback_bars)
    swing_range = high - low

    min_swing = min_swing_mult * atr

    # 조건 충족 시 갱신
    if swing_range >= min_swing:
        return low, high, True

    # 조건 미충족: 이전 상태 유지
    if prev_state and prev_state.is_valid():
        return prev_state.low, prev_state.high, False

    # 이전 상태가 없으면 현재 값 사용
    return low, high, False


def update_anchor_zigzag(
    df: pd.DataFrame,
    i: int,
    state: DynamicFibAnchorState,
    atr: float,
    reversal_mult: float = 1.5,
    min_bars_between_pivots: int = 3
) -> DynamicFibAnchorState:
    """
    ZigZag pivot 확정 방식으로 앵커 갱신.

    ATR 기반 reversal 조건:
    - 방향이 "up"이고 현재 low가 마지막 pivot에서 reversal_mult * ATR 이상 하락하면
      → 마지막 고점을 pivot으로 확정, 방향을 "down"으로 전환
    - 반대도 동일

    Args:
        df: OHLCV DataFrame (15m)
        i: 현재 인덱스
        state: 현재 상태
        atr: 현재 ATR
        reversal_mult: reversal 조건 ATR 배수 (기본 1.5)
        min_bars_between_pivots: pivot 간 최소 봉 수 (기본 3)

    Returns:
        갱신된 DynamicFibAnchorState
    """
    if i < 1:
        return state

    bar = df.iloc[i]
    current_high = bar['high']
    current_low = bar['low']
    current_ts = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else bar.get('timestamp', pd.Timestamp.now())

    reversal_threshold = reversal_mult * atr

    # 초기화: 첫 상태 설정
    if state.direction == "unknown" or state.last_pivot_price == 0:
        # 초기 방향 결정: 최근 N봉 추세
        lookback = min(20, i + 1)
        start_idx = max(0, i - lookback + 1)
        window = df.iloc[start_idx:i+1]

        first_close = window.iloc[0]['close']
        last_close = window.iloc[-1]['close']

        if last_close > first_close:
            # 상승 추세 → 저점부터 시작
            state.direction = "up"
            state.last_pivot_price = window['low'].min()
            state.low = state.last_pivot_price
            state.high = current_high
        else:
            # 하락 추세 → 고점부터 시작
            state.direction = "down"
            state.last_pivot_price = window['high'].max()
            state.high = state.last_pivot_price
            state.low = current_low

        state.last_pivot_ts = current_ts
        state.last_update_ts = current_ts
        return state

    # 방향별 pivot 확정 로직
    if state.direction == "up":
        # 상승 중: 새로운 고점 추적
        if current_high > state.high:
            state.high = current_high
            state.last_update_ts = current_ts

        # Reversal 조건: 마지막 고점에서 threshold 이상 하락
        if state.high - current_low >= reversal_threshold:
            # 고점 확정 → 방향 전환
            state.last_pivot_price = state.high
            state.last_pivot_ts = current_ts
            state.direction = "down"
            state.low = current_low  # 새로운 저점 추적 시작
            state.last_update_ts = current_ts

    elif state.direction == "down":
        # 하락 중: 새로운 저점 추적
        if current_low < state.low:
            state.low = current_low
            state.last_update_ts = current_ts

        # Reversal 조건: 마지막 저점에서 threshold 이상 상승
        if current_high - state.low >= reversal_threshold:
            # 저점 확정 → 방향 전환
            state.last_pivot_price = state.low
            state.last_pivot_ts = current_ts
            state.direction = "up"
            state.high = current_high  # 새로운 고점 추적 시작
            state.last_update_ts = current_ts

    return state


def get_dynamic_fib_levels(
    low: float,
    high: float,
    ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    include_extensions: bool = False
) -> List[float]:
    """
    동적 Fib 레벨 가격 리스트 생성 (Linear).

    Args:
        low: 앵커 저점
        high: 앵커 고점
        ratios: Fib 비율 (기본 0.236~0.786)
        include_extensions: 확장 레벨 포함 여부 (1.272, 1.618)

    Returns:
        가격 리스트 (오름차순 정렬)
    """
    if low <= 0 or high <= 0 or high <= low:
        return []

    swing_range = high - low
    levels = []

    # Retracement 레벨 (저점에서 올라가는 방향)
    for r in ratios:
        price = low + swing_range * r
        levels.append(price)

    # Extension 레벨 (옵션)
    if include_extensions:
        extension_ratios = (1.0, 1.272, 1.618)
        for r in extension_ratios:
            price = low + swing_range * r
            levels.append(price)

    # 정렬 및 중복 제거
    levels = sorted(set(levels))

    return levels


def get_dynamic_fib_levels_from_state(
    state: DynamicFibAnchorState,
    ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    include_extensions: bool = False
) -> List[float]:
    """
    DynamicFibAnchorState에서 Fib 레벨 생성.

    Args:
        state: 앵커 상태
        ratios: Fib 비율
        include_extensions: 확장 레벨 포함 여부

    Returns:
        가격 리스트 (유효하지 않으면 빈 리스트)
    """
    if not state.is_valid():
        return []

    return get_dynamic_fib_levels(
        state.low, state.high, ratios, include_extensions
    )


def check_confluence(
    price: float,
    macro_levels: List[float],
    dyn_levels: List[float],
    tolerance: float = 0.005
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Macro Fib 레벨과 Dynamic Fib 레벨의 confluence 체크.

    Args:
        price: 현재가
        macro_levels: Macro Fib 레벨 리스트
        dyn_levels: Dynamic Fib 레벨 리스트
        tolerance: 허용 오차 비율 (기본 0.5%)

    Returns:
        (is_confluence, macro_level, dyn_level) 튜플
        - is_confluence: confluence 여부
        - macro_level: 가까운 macro 레벨 (None이면 없음)
        - dyn_level: 가까운 dyn 레벨 (None이면 없음)
    """
    def find_nearest(levels: List[float], target: float) -> Optional[float]:
        if not levels:
            return None
        nearest = min(levels, key=lambda x: abs(x - target))
        if abs(nearest - target) / target <= tolerance:
            return nearest
        return None

    macro_near = find_nearest(macro_levels, price)
    dyn_near = find_nearest(dyn_levels, price)

    is_confluence = macro_near is not None and dyn_near is not None

    return is_confluence, macro_near, dyn_near


def filter_tp_by_confluence(
    tp_candidates: List[float],
    macro_levels: List[float],
    dyn_levels: List[float],
    tolerance: float = 0.005,
    require_confluence: bool = False
) -> List[float]:
    """
    TP 후보를 confluence 기준으로 필터링.

    Args:
        tp_candidates: TP 후보 가격 리스트
        macro_levels: Macro Fib 레벨 리스트
        dyn_levels: Dynamic Fib 레벨 리스트
        tolerance: 허용 오차 비율
        require_confluence: True면 confluence인 TP만 반환

    Returns:
        필터링된 TP 리스트
    """
    if not require_confluence:
        return tp_candidates

    filtered = []
    for tp in tp_candidates:
        is_conf, _, _ = check_confluence(tp, macro_levels, dyn_levels, tolerance)
        if is_conf:
            filtered.append(tp)

    return filtered if filtered else tp_candidates  # 없으면 원본 반환


# Utility: 상태 초기화
def create_initial_state(mode: str = "rolling") -> DynamicFibAnchorState:
    """초기 상태 생성"""
    return DynamicFibAnchorState(
        low=0.0,
        high=0.0,
        last_update_ts=None,
        mode=mode,
        direction="unknown",
        last_pivot_price=0.0,
        last_pivot_ts=None
    )
