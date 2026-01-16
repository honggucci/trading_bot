"""
B급 대체 트리거
===============

Divergence 없이 Fib Zone만 있을 때 사용하는 트리거.

권장 1순위: Z-score Revert
- 가격이 MA 대비 과확장 → 되돌림 시작 신호
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd


@dataclass
class ZScoreTrigger:
    """Z-score 기반 트리거 결과"""
    triggered: bool
    side: Literal['long', 'short', 'none']
    z_score: float
    ma_price: float
    std_price: float
    threshold: float
    reason: str = ''


def calc_zscore(
    close: np.ndarray,
    window: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score 계산

    z = (price - MA) / std

    Returns:
        (z_score, ma, std)
    """
    close = np.asarray(close, dtype=float)
    n = len(close)

    ma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    z = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = close[i - window + 1:i + 1]
        ma[i] = np.mean(window_data)
        std[i] = np.std(window_data, ddof=1)

        if std[i] > 0:
            z[i] = (close[i] - ma[i]) / std[i]

    return z, ma, std


def check_zscore_revert(
    df: pd.DataFrame,
    long_threshold: float = -2.0,
    short_threshold: float = 2.0,
    revert_bars: int = 2,
    window: int = 20,
) -> ZScoreTrigger:
    """
    Z-score Revert 트리거 확인

    조건:
    - Long: z-score가 threshold 이하로 갔다가 되돌아오기 시작
    - Short: z-score가 threshold 이상으로 갔다가 되돌아오기 시작

    Args:
        df: OHLCV DataFrame
        long_threshold: Long 트리거 threshold (음수, 예: -2.0)
        short_threshold: Short 트리거 threshold (양수, 예: 2.0)
        revert_bars: 되돌림 확인 봉 수
        window: MA/std 계산 윈도우

    Returns:
        ZScoreTrigger
    """
    close = df['close'].astype(float).values

    if len(close) < window + revert_bars:
        return ZScoreTrigger(
            triggered=False,
            side='none',
            z_score=np.nan,
            ma_price=np.nan,
            std_price=np.nan,
            threshold=0,
            reason='데이터 부족',
        )

    z, ma, std = calc_zscore(close, window)

    current_z = z[-1]
    ma_now = ma[-1]
    std_now = std[-1]

    if not np.isfinite(current_z):
        return ZScoreTrigger(
            triggered=False,
            side='none',
            z_score=np.nan,
            ma_price=ma_now,
            std_price=std_now,
            threshold=0,
            reason='Z-score 계산 불가',
        )

    # === Long 트리거 확인 ===
    # 조건: 최근 N봉 내 z < threshold 였다가 현재 z가 상승 중
    recent_z = z[-revert_bars - 1:-1]  # 현재 제외 최근 N봉

    if np.any(recent_z < long_threshold):
        # 과거에 과매도였음
        # 현재 z가 상승 중인지 (되돌림)
        if current_z > recent_z[-1]:
            return ZScoreTrigger(
                triggered=True,
                side='long',
                z_score=current_z,
                ma_price=ma_now,
                std_price=std_now,
                threshold=long_threshold,
                reason=f'Z-score revert (과매도 {min(recent_z):.2f} → {current_z:.2f})',
            )

    # === Short 트리거 확인 ===
    if np.any(recent_z > short_threshold):
        # 과거에 과매수였음
        # 현재 z가 하락 중인지 (되돌림)
        if current_z < recent_z[-1]:
            return ZScoreTrigger(
                triggered=True,
                side='short',
                z_score=current_z,
                ma_price=ma_now,
                std_price=std_now,
                threshold=short_threshold,
                reason=f'Z-score revert (과매수 {max(recent_z):.2f} → {current_z:.2f})',
            )

    return ZScoreTrigger(
        triggered=False,
        side='none',
        z_score=current_z,
        ma_price=ma_now,
        std_price=std_now,
        threshold=long_threshold if current_z < 0 else short_threshold,
        reason='트리거 조건 미충족',
    )


def alt_trigger_ok(
    df: pd.DataFrame,
    side: Literal['long', 'short'],
    **kwargs,
) -> bool:
    """
    B급 대체 트리거 충족 여부 (단순 API)

    Args:
        df: OHLCV DataFrame
        side: 'long' 또는 'short'
        **kwargs: check_zscore_revert 파라미터

    Returns:
        True if 트리거 충족
    """
    result = check_zscore_revert(df, **kwargs)

    if not result.triggered:
        return False

    return result.side == side


# =============================================================================
# 추가 트리거 (나중에 확장용)
# =============================================================================

def check_candle_reversal(
    df: pd.DataFrame,
    side: Literal['long', 'short'],
    min_body_ratio: float = 0.5,
) -> bool:
    """
    2-bar 캔들 반전 패턴

    Long: 음봉 → 양봉 (양봉이 음봉 몸통 50% 이상 회복)
    Short: 양봉 → 음봉 (음봉이 양봉 몸통 50% 이상 하락)
    """
    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_open = float(prev['open'])
    prev_close = float(prev['close'])
    curr_open = float(curr['open'])
    curr_close = float(curr['close'])

    prev_body = prev_close - prev_open
    curr_body = curr_close - curr_open

    if side == 'long':
        # 음봉 → 양봉
        if prev_body >= 0 or curr_body <= 0:
            return False
        # 양봉이 음봉 몸통의 min_body_ratio 이상 회복
        recovery_ratio = curr_body / abs(prev_body)
        return recovery_ratio >= min_body_ratio

    else:  # short
        # 양봉 → 음봉
        if prev_body <= 0 or curr_body >= 0:
            return False
        # 음봉이 양봉 몸통의 min_body_ratio 이상 하락
        drop_ratio = abs(curr_body) / prev_body
        return drop_ratio >= min_body_ratio


def check_volume_spike(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    window: int = 20,
) -> bool:
    """
    볼륨 스파이크 확인

    현재 볼륨이 최근 평균의 multiplier 배 이상
    """
    if 'volume' not in df.columns:
        return False

    if len(df) < window:
        return False

    vol = df['volume'].astype(float).values
    avg_vol = np.mean(vol[-window - 1:-1])

    if avg_vol <= 0:
        return False

    return vol[-1] >= avg_vol * multiplier
