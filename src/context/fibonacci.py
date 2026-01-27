"""
Fibonacci utilities - Re-export from cycle_anchor + legacy stubs
================================================================

FibLevel class를 cycle_anchor.py에서 재수출하고,
legacy 코드 호환을 위한 stub 함수들을 제공합니다.

Note: Fib는 이제 ZigZag에서만 사용되고, SL/TP에는 사용되지 않습니다.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

# Re-export from cycle_anchor
from .cycle_anchor import FibLevel, get_fractal_fib_levels

# Type alias for backward compatibility
FibLevels = List[FibLevel]


def calc_fib_levels(swing: Any, fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786)) -> FibLevels:
    """
    스윙 정보에서 Fib 레벨 계산 (Legacy stub)

    Args:
        swing: SwingInfo (swing_low, swing_high 속성 필요)
        fib_ratios: Fib 비율들

    Returns:
        FibLevels (List[FibLevel])
    """
    # SwingInfo가 없는 경우 빈 리스트 반환
    if swing is None:
        return []

    # swing에서 low/high 추출
    swing_low = getattr(swing, 'swing_low', None) or getattr(swing, 'low', 0)
    swing_high = getattr(swing, 'swing_high', None) or getattr(swing, 'high', 0)

    if swing_low >= swing_high:
        return []

    swing_range = swing_high - swing_low
    levels = []

    for ratio in fib_ratios:
        price = swing_low + (swing_range * ratio)
        levels.append(FibLevel(
            fib_ratio=ratio,
            price=price,
            depth=0,
            cell=(0, 1),
        ))

    return levels


def build_fib_zones(
    fib_levels: FibLevels,
    atr: float,
    *,
    fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    min_half_atr: float = 0.3,
    max_half_mult: float = 1.0,
) -> List[FibLevel]:
    """
    Fib 레벨에 ATR 기반 Zone Width 적용 (Legacy stub)

    Note: Zone Width는 실제로 FibLevel에 저장되지 않음.
          이 함수는 호환성을 위해 입력을 그대로 반환합니다.

    Args:
        fib_levels: calc_fib_levels() 결과
        atr: ATR 값
        fib_ratios: Fib 비율들 (사용 안함)
        min_half_atr: 최소 half-width (ATR 배수)
        max_half_mult: 최대 half-width 배수

    Returns:
        List[FibLevel] - 입력을 그대로 반환
    """
    # 호환성을 위해 입력을 그대로 반환
    # Zone Width는 이제 사용되지 않음
    return fib_levels


__all__ = [
    'FibLevel',
    'FibLevels',
    'calc_fib_levels',
    'build_fib_zones',
    'get_fractal_fib_levels',
]
