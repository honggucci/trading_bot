"""
Anchor Layer - Divergence Detection
====================================

MODE82에서 사용하는 다이버전스 로직.
- divergence.py: RSI divergence detection + 경계 계산 (REG/HID)
"""
try:
    from .divergence import (
        DivergenceResult,
        calc_rsi_wilder,
        detect_bullish_divergence,
        find_best_divergence,
        compute_divergence_score,
        check_divergence_at_current,
        needed_close_for_regular,
        feasible_range_for_hidden,
    )
except ImportError:
    pass

__all__ = [
    # Divergence
    'DivergenceResult',
    'calc_rsi_wilder',
    'detect_bullish_divergence',
    'find_best_divergence',
    'compute_divergence_score',
    'check_divergence_at_current',
    'needed_close_for_regular',
    'feasible_range_for_hidden',
]
