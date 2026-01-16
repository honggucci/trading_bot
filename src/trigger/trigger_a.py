"""
A급 트리거 - 5m Trigger TF
===========================

Zone 진입 시 필수 확인 트리거.

트리거 종류:
1. Spring/UTAD Reclaim - 스윙 실패 후 되돌림
2. Failed Swing - 직전 스윙 돌파 실패
3. Effort vs Result (Absorption) - 볼륨 대비 가격 움직임 비대칭
"""

from dataclasses import dataclass
from typing import Optional, Literal, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class TriggerResult:
    """트리거 결과"""
    triggered: bool
    trigger_type: str  # 'spring', 'utad', 'failed_swing', 'absorption', 'none'
    side: Literal['long', 'short', 'none']
    confidence: float = 1.0
    reason: str = ''


# =============================================================================
# 1. Spring / UTAD Reclaim
# =============================================================================

def find_swing_lows(
    low: np.ndarray,
    lookback: int = 5,
) -> List[int]:
    """스윙 저점 찾기 (좌우 lookback 봉보다 낮은 점)"""
    swings = []
    n = len(low)

    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_swing = False
                break
        if is_swing:
            swings.append(i)

    return swings


def find_swing_highs(
    high: np.ndarray,
    lookback: int = 5,
) -> List[int]:
    """스윙 고점 찾기 (좌우 lookback 봉보다 높은 점)"""
    swings = []
    n = len(high)

    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_swing = False
                break
        if is_swing:
            swings.append(i)

    return swings


def check_spring(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    swing_lookback: int = 5,
    reclaim_bars: int = 3,
) -> TriggerResult:
    """
    Spring 트리거 (Long)

    조건:
    1. 최근 스윙 저점 아래로 돌파 (false breakdown)
    2. 빠르게 되돌아와서 스윙 저점 위로 회복 (reclaim)

    Args:
        df: OHLCV DataFrame
        lookback_bars: 스윙 저점 탐색 범위
        swing_lookback: 스윙 포인트 확인용 좌우 봉 수
        reclaim_bars: 회복 확인 봉 수
    """
    if len(df) < lookback_bars + swing_lookback:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='데이터 부족',
        )

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # 최근 구간에서 스윙 저점 찾기
    search_start = max(0, len(df) - lookback_bars - swing_lookback)
    search_end = len(df) - reclaim_bars

    swing_lows = find_swing_lows(low[search_start:search_end], swing_lookback)

    if not swing_lows:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='스윙 저점 없음',
        )

    # 가장 최근 스윙 저점
    last_swing_idx = search_start + swing_lows[-1]
    swing_low_price = low[last_swing_idx]

    # 스윙 저점 이후 데이터
    after_swing = df.iloc[last_swing_idx + 1:]

    if len(after_swing) < reclaim_bars:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='회복 확인 데이터 부족',
        )

    # 조건 1: 스윙 저점 아래로 돌파했는가?
    broke_below = any(after_swing['low'].values < swing_low_price)

    if not broke_below:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='스윙 저점 미돌파',
        )

    # 조건 2: 현재 가격이 스윙 저점 위로 회복했는가?
    current_close = close[-1]
    reclaimed = current_close > swing_low_price

    # 조건 3: 최근 N봉이 상승 중인가?
    recent_closes = close[-reclaim_bars:]
    rising = all(recent_closes[i] <= recent_closes[i + 1] for i in range(len(recent_closes) - 1))

    if reclaimed and rising:
        return TriggerResult(
            triggered=True,
            trigger_type='spring',
            side='long',
            confidence=1.0,
            reason=f'Spring: 스윙 저점 {swing_low_price:.2f} 돌파 후 회복',
        )

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='회복 미완료',
    )


def check_utad(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    swing_lookback: int = 5,
    reclaim_bars: int = 3,
) -> TriggerResult:
    """
    UTAD (Upthrust After Distribution) 트리거 (Short)

    조건:
    1. 최근 스윙 고점 위로 돌파 (false breakout)
    2. 빠르게 되돌아와서 스윙 고점 아래로 회복 (rejection)
    """
    if len(df) < lookback_bars + swing_lookback:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='데이터 부족',
        )

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # 최근 구간에서 스윙 고점 찾기
    search_start = max(0, len(df) - lookback_bars - swing_lookback)
    search_end = len(df) - reclaim_bars

    swing_highs = find_swing_highs(high[search_start:search_end], swing_lookback)

    if not swing_highs:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='스윙 고점 없음',
        )

    # 가장 최근 스윙 고점
    last_swing_idx = search_start + swing_highs[-1]
    swing_high_price = high[last_swing_idx]

    # 스윙 고점 이후 데이터
    after_swing = df.iloc[last_swing_idx + 1:]

    if len(after_swing) < reclaim_bars:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='확인 데이터 부족',
        )

    # 조건 1: 스윙 고점 위로 돌파했는가?
    broke_above = any(after_swing['high'].values > swing_high_price)

    if not broke_above:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='스윙 고점 미돌파',
        )

    # 조건 2: 현재 가격이 스윙 고점 아래로 회복했는가?
    current_close = close[-1]
    rejected = current_close < swing_high_price

    # 조건 3: 최근 N봉이 하락 중인가?
    recent_closes = close[-reclaim_bars:]
    falling = all(recent_closes[i] >= recent_closes[i + 1] for i in range(len(recent_closes) - 1))

    if rejected and falling:
        return TriggerResult(
            triggered=True,
            trigger_type='utad',
            side='short',
            confidence=1.0,
            reason=f'UTAD: 스윙 고점 {swing_high_price:.2f} 돌파 후 rejection',
        )

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='rejection 미완료',
    )


# =============================================================================
# 2. Failed Swing
# =============================================================================

def check_failed_swing_low(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    tolerance_pct: float = 0.002,
) -> TriggerResult:
    """
    Failed Swing Low 트리거 (Long)

    조건:
    1. 두 번째 저점이 첫 번째 저점보다 높거나 비슷 (higher low)
    2. 두 번째 저점 테스트 후 반등
    """
    if len(df) < lookback_bars:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='데이터 부족',
        )

    low = df['low'].values
    close = df['close'].values

    recent_low = low[-lookback_bars:]

    # 최저점 2개 찾기
    sorted_indices = np.argsort(recent_low)

    if len(sorted_indices) < 2:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='저점 부족',
        )

    # 첫 번째, 두 번째 최저점 인덱스
    first_low_idx = sorted_indices[0]
    second_low_idx = sorted_indices[1]

    # 시간순 정렬 (먼저 온 것이 첫 번째)
    if first_low_idx > second_low_idx:
        first_low_idx, second_low_idx = second_low_idx, first_low_idx

    first_low_price = recent_low[first_low_idx]
    second_low_price = recent_low[second_low_idx]

    # 조건: 두 번째 저점이 첫 번째보다 높거나 비슷
    tolerance = first_low_price * tolerance_pct
    is_higher_low = second_low_price >= first_low_price - tolerance

    # 조건: 두 번째 저점이 최근에 있고, 현재 반등 중
    is_recent = second_low_idx >= lookback_bars - 5
    current_close = close[-1]
    is_bouncing = current_close > second_low_price

    if is_higher_low and is_recent and is_bouncing:
        return TriggerResult(
            triggered=True,
            trigger_type='failed_swing',
            side='long',
            confidence=0.9,
            reason=f'Failed Swing Low: {first_low_price:.2f} → {second_low_price:.2f} (higher low)',
        )

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='Failed Swing 조건 미충족',
    )


def check_failed_swing_high(
    df: pd.DataFrame,
    lookback_bars: int = 20,
    tolerance_pct: float = 0.002,
) -> TriggerResult:
    """
    Failed Swing High 트리거 (Short)

    조건:
    1. 두 번째 고점이 첫 번째 고점보다 낮거나 비슷 (lower high)
    2. 두 번째 고점 테스트 후 하락
    """
    if len(df) < lookback_bars:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='데이터 부족',
        )

    high = df['high'].values
    close = df['close'].values

    recent_high = high[-lookback_bars:]

    # 최고점 2개 찾기
    sorted_indices = np.argsort(recent_high)[::-1]  # 내림차순

    if len(sorted_indices) < 2:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='고점 부족',
        )

    # 첫 번째, 두 번째 최고점 인덱스
    first_high_idx = sorted_indices[0]
    second_high_idx = sorted_indices[1]

    # 시간순 정렬
    if first_high_idx > second_high_idx:
        first_high_idx, second_high_idx = second_high_idx, first_high_idx

    first_high_price = recent_high[first_high_idx]
    second_high_price = recent_high[second_high_idx]

    # 조건: 두 번째 고점이 첫 번째보다 낮거나 비슷
    tolerance = first_high_price * tolerance_pct
    is_lower_high = second_high_price <= first_high_price + tolerance

    # 조건: 두 번째 고점이 최근에 있고, 현재 하락 중
    is_recent = second_high_idx >= lookback_bars - 5
    current_close = close[-1]
    is_dropping = current_close < second_high_price

    if is_lower_high and is_recent and is_dropping:
        return TriggerResult(
            triggered=True,
            trigger_type='failed_swing',
            side='short',
            confidence=0.9,
            reason=f'Failed Swing High: {first_high_price:.2f} → {second_high_price:.2f} (lower high)',
        )

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='Failed Swing 조건 미충족',
    )


# =============================================================================
# 3. Effort vs Result (Absorption)
# =============================================================================

def check_absorption_long(
    df: pd.DataFrame,
    lookback_bars: int = 10,
    vol_threshold: float = 1.5,
    price_threshold: float = 0.3,
) -> TriggerResult:
    """
    Absorption 트리거 (Long)

    조건:
    - 높은 볼륨 + 작은 가격 하락 = 매도 흡수
    - 이후 반등 시작

    Args:
        vol_threshold: 평균 볼륨 대비 배수 (기본 1.5)
        price_threshold: 볼륨 대비 가격 움직임 비율 (기본 0.3)
    """
    if 'volume' not in df.columns:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='볼륨 데이터 없음',
        )

    if len(df) < lookback_bars + 20:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='데이터 부족',
        )

    close = df['close'].values
    volume = df['volume'].values

    # 평균 볼륨
    avg_vol = np.mean(volume[-lookback_bars - 20:-lookback_bars])
    if avg_vol <= 0:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='볼륨 계산 불가',
        )

    # 최근 구간 분석
    recent_vol = volume[-lookback_bars:]
    recent_close = close[-lookback_bars:]

    # 볼륨 스파이크 봉 찾기
    high_vol_bars = []
    for i in range(len(recent_vol)):
        if recent_vol[i] >= avg_vol * vol_threshold:
            high_vol_bars.append(i)

    if not high_vol_bars:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='볼륨 스파이크 없음',
        )

    # 볼륨 스파이크 봉에서 가격 변화 확인
    for bar_idx in high_vol_bars:
        if bar_idx == 0:
            continue

        price_change_pct = (recent_close[bar_idx] - recent_close[bar_idx - 1]) / recent_close[bar_idx - 1]
        vol_ratio = recent_vol[bar_idx] / avg_vol

        # 볼륨은 높은데 가격 하락이 작음 = absorption
        if price_change_pct < 0 and abs(price_change_pct) < price_threshold / vol_ratio:
            # 이후 반등 확인
            if bar_idx < len(recent_close) - 2:
                following_closes = recent_close[bar_idx + 1:]
                if len(following_closes) >= 2 and following_closes[-1] > recent_close[bar_idx]:
                    return TriggerResult(
                        triggered=True,
                        trigger_type='absorption',
                        side='long',
                        confidence=0.85,
                        reason=f'Absorption: 볼륨 {vol_ratio:.1f}x, 가격변화 {price_change_pct:.2%}',
                    )

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='Absorption 조건 미충족',
    )


def check_absorption_short(
    df: pd.DataFrame,
    lookback_bars: int = 10,
    vol_threshold: float = 1.5,
    price_threshold: float = 0.3,
) -> TriggerResult:
    """
    Absorption 트리거 (Short)

    조건:
    - 높은 볼륨 + 작은 가격 상승 = 매수 흡수
    - 이후 하락 시작
    """
    if 'volume' not in df.columns:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='볼륨 데이터 없음',
        )

    if len(df) < lookback_bars + 20:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='데이터 부족',
        )

    close = df['close'].values
    volume = df['volume'].values

    # 평균 볼륨
    avg_vol = np.mean(volume[-lookback_bars - 20:-lookback_bars])
    if avg_vol <= 0:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='볼륨 계산 불가',
        )

    # 최근 구간 분석
    recent_vol = volume[-lookback_bars:]
    recent_close = close[-lookback_bars:]

    # 볼륨 스파이크 봉 찾기
    high_vol_bars = []
    for i in range(len(recent_vol)):
        if recent_vol[i] >= avg_vol * vol_threshold:
            high_vol_bars.append(i)

    if not high_vol_bars:
        return TriggerResult(
            triggered=False,
            trigger_type='none',
            side='none',
            reason='볼륨 스파이크 없음',
        )

    # 볼륨 스파이크 봉에서 가격 변화 확인
    for bar_idx in high_vol_bars:
        if bar_idx == 0:
            continue

        price_change_pct = (recent_close[bar_idx] - recent_close[bar_idx - 1]) / recent_close[bar_idx - 1]
        vol_ratio = recent_vol[bar_idx] / avg_vol

        # 볼륨은 높은데 가격 상승이 작음 = distribution
        if price_change_pct > 0 and price_change_pct < price_threshold / vol_ratio:
            # 이후 하락 확인
            if bar_idx < len(recent_close) - 2:
                following_closes = recent_close[bar_idx + 1:]
                if len(following_closes) >= 2 and following_closes[-1] < recent_close[bar_idx]:
                    return TriggerResult(
                        triggered=True,
                        trigger_type='absorption',
                        side='short',
                        confidence=0.85,
                        reason=f'Distribution: 볼륨 {vol_ratio:.1f}x, 가격변화 {price_change_pct:.2%}',
                    )

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='Distribution 조건 미충족',
    )


# =============================================================================
# 통합 API
# =============================================================================

def check_trigger(
    df: pd.DataFrame,
    side: Literal['long', 'short'],
    **kwargs,
) -> TriggerResult:
    """
    모든 트리거 확인 (A급용)

    Args:
        df: OHLCV DataFrame (Trigger TF)
        side: 'long' 또는 'short'
        **kwargs: 개별 트리거 파라미터

    Returns:
        TriggerResult (하나라도 충족되면 triggered=True)
    """
    if side == 'long':
        triggers = [
            check_spring(df, **{k: v for k, v in kwargs.items() if k in ['lookback_bars', 'swing_lookback', 'reclaim_bars']}),
            check_failed_swing_low(df, **{k: v for k, v in kwargs.items() if k in ['lookback_bars', 'tolerance_pct']}),
            check_absorption_long(df, **{k: v for k, v in kwargs.items() if k in ['lookback_bars', 'vol_threshold', 'price_threshold']}),
        ]
    else:
        triggers = [
            check_utad(df, **{k: v for k, v in kwargs.items() if k in ['lookback_bars', 'swing_lookback', 'reclaim_bars']}),
            check_failed_swing_high(df, **{k: v for k, v in kwargs.items() if k in ['lookback_bars', 'tolerance_pct']}),
            check_absorption_short(df, **{k: v for k, v in kwargs.items() if k in ['lookback_bars', 'vol_threshold', 'price_threshold']}),
        ]

    # 우선순위: Spring/UTAD > Failed Swing > Absorption
    for result in triggers:
        if result.triggered:
            return result

    return TriggerResult(
        triggered=False,
        trigger_type='none',
        side='none',
        reason='모든 트리거 미충족',
    )


def trigger_ok(
    df: pd.DataFrame,
    side: Literal['long', 'short'],
    **kwargs,
) -> bool:
    """단순 API: 트리거 충족 여부만 반환"""
    result = check_trigger(df, side, **kwargs)
    return result.triggered
