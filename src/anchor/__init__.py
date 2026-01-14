"""
Anchor Layer - 15m Divergence + StochRSI + Confluence
=====================================================

레거시 로직 완전 구현 + HMM 통합:
- stochrsi.py: TradingView identical StochRSI + Oversold 세그먼트 + REF 추출
- divergence.py: RSI divergence detection + 경계 계산 (REG/HID)
- confluence.py: Fib + Divergence Confluence 점수 계산
- legacy_pipeline.py: 전체 Confluence 분석 파이프라인
- unified_signal.py: Legacy + HMM 통합 신호 (V2)
"""
from .stochrsi import (
    tv_stoch_rsi,
    is_oversold,
    is_overbought,
    pick_oversold_segments,
    pick_oversold_segment_with_rule,
    extract_ref_from_segment,
    RefPoint,
)
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
from .confluence import (
    calc_zone_confluence_scores,
    find_best_entry_zone,
    is_price_in_confluence_zone,
    ZoneScore,
)
from .legacy_pipeline import (
    analyze_confluence,
    should_enter_long,
    ConfluenceResult,
)
from .unified_signal import (
    UnifiedSignal,
    check_unified_long_signal,
    check_unified_short_signal,
    check_unified_long_signal_v2,
    check_unified_short_signal_v2,
    REGIME_PARAMS,
    classify_regime_from_state,
)
from .exit_logic import (
    ExitLevels,
    ExitSignal,
    ExitReason,
    TrailingState,
    Position,
    calc_exit_levels,
    check_exit_signal,
    manage_position,
    calc_position_size,
    format_exit_levels,
)
from .unified_signal_v3 import (
    UnifiedSignalV3,
    check_unified_long_signal_v3,
    check_unified_short_signal_v3,
    calc_mtf_boost,
    determine_regime,
    format_signal_v3,
)

__all__ = [
    # StochRSI
    'tv_stoch_rsi',
    'is_oversold',
    'is_overbought',
    'pick_oversold_segments',
    'pick_oversold_segment_with_rule',
    'extract_ref_from_segment',
    'RefPoint',

    # Divergence
    'DivergenceResult',
    'calc_rsi_wilder',
    'detect_bullish_divergence',
    'find_best_divergence',
    'compute_divergence_score',
    'check_divergence_at_current',
    'needed_close_for_regular',
    'feasible_range_for_hidden',

    # Confluence
    'calc_zone_confluence_scores',
    'find_best_entry_zone',
    'is_price_in_confluence_zone',
    'ZoneScore',

    # Legacy Pipeline
    'analyze_confluence',
    'should_enter_long',
    'ConfluenceResult',

    # Unified Signal V2
    'UnifiedSignal',
    'check_unified_long_signal',
    'check_unified_short_signal',
    'check_unified_long_signal_v2',
    'check_unified_short_signal_v2',
    'REGIME_PARAMS',
    'classify_regime_from_state',

    # Exit Logic
    'ExitLevels',
    'ExitSignal',
    'ExitReason',
    'TrailingState',
    'Position',
    'calc_exit_levels',
    'check_exit_signal',
    'manage_position',
    'calc_position_size',
    'format_exit_levels',

    # Unified Signal V3
    'UnifiedSignalV3',
    'check_unified_long_signal_v3',
    'check_unified_short_signal_v3',
    'calc_mtf_boost',
    'determine_regime',
    'format_signal_v3',
]