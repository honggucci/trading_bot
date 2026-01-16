"""
Trigger Layer - 5m 진입 트리거
==============================

Zone 진입 시 추가 확인 트리거.

Components:
- trigger_a.py: A급 트리거 (Spring/UTAD, Failed Swing, Absorption)
- trigger_b.py: B급 대체 트리거 (Z-score revert)
"""

from .trigger_a import (
    TriggerResult,
    check_spring,
    check_utad,
    check_failed_swing_low,
    check_failed_swing_high,
    check_absorption_long,
    check_absorption_short,
    check_trigger,
    trigger_ok,
)

from .trigger_b import (
    ZScoreTrigger,
    calc_zscore,
    check_zscore_revert,
    alt_trigger_ok,
    check_candle_reversal,
    check_volume_spike,
)

__all__ = [
    # A급 트리거
    'TriggerResult',
    'check_spring',
    'check_utad',
    'check_failed_swing_low',
    'check_failed_swing_high',
    'check_absorption_long',
    'check_absorption_short',
    'check_trigger',
    'trigger_ok',
    # B급 트리거
    'ZScoreTrigger',
    'calc_zscore',
    'check_zscore_revert',
    'alt_trigger_ok',
    'check_candle_reversal',
    'check_volume_spike',
]
