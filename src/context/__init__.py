"""
Context Layer - Multi-TF Analysis
=================================

ZigZag 피벗 감지 및 피보나치 레벨 계산.

Components:
- zigzag.py: 스윙 고점/저점 감지
- fibonacci.py: 피보나치 되돌림/확장 레벨
- cycle_anchor.py: 비트코인 사이클 기반 Fib 앵커
"""
from .zigzag import (
    zigzag_pivots,
    get_latest_swing,
    add_pivot_columns,
    SwingInfo,
    wilder_atr,
)
from .fibonacci import (
    calc_fib_levels,
    build_fib_zones,
    is_in_zone,
    is_in_golden_pocket,
    find_nearest_zone,
    find_containing_zones,
    FibLevel,
    FibLevels,
    DEFAULT_RETRACEMENTS,
    DEFAULT_EXTENSIONS,
)
from .cycle_anchor import (
    CycleAnchor,
    CycleData,
    BTC_CYCLES,
    CURRENT_CYCLE,
    get_btc_cycle_anchor,
    get_fib_levels,
    get_current_cycle_position,
    normalize_price_to_cycle,
    denormalize_price_from_cycle,
)
from .multi_tf_fib import (
    MultiTFFibSystem,
    FibHierarchy,
    TFFibLevel,
    ZigZagParams,
    ZigZagOptimizer,
    DEFAULT_ZIGZAG_PARAMS,
    build_multi_tf_fib,
    find_fib_confluence,
)
from .tf_predictor import (
    TFPredictor,
    SwingPrediction,
    TradingSignal,
    PredictionZone,
    SwingBias,
    ZoneType,
    predict_btc_move,
    get_btc_signal,
    analyze_btc,
)

__all__ = [
    # ZigZag
    'zigzag_pivots',
    'get_latest_swing',
    'add_pivot_columns',
    'SwingInfo',
    'wilder_atr',

    # Fibonacci
    'calc_fib_levels',
    'build_fib_zones',
    'is_in_zone',
    'is_in_golden_pocket',
    'find_nearest_zone',
    'find_containing_zones',
    'FibLevel',
    'FibLevels',
    'DEFAULT_RETRACEMENTS',
    'DEFAULT_EXTENSIONS',

    # Cycle Anchor
    'CycleAnchor',
    'CycleData',
    'BTC_CYCLES',
    'CURRENT_CYCLE',
    'get_btc_cycle_anchor',
    'get_fib_levels',
    'get_current_cycle_position',
    'normalize_price_to_cycle',
    'denormalize_price_from_cycle',

    # Multi-TF Fib
    'MultiTFFibSystem',
    'FibHierarchy',
    'TFFibLevel',
    'ZigZagParams',
    'ZigZagOptimizer',
    'DEFAULT_ZIGZAG_PARAMS',
    'build_multi_tf_fib',
    'find_fib_confluence',

    # TF Predictor
    'TFPredictor',
    'SwingPrediction',
    'TradingSignal',
    'PredictionZone',
    'SwingBias',
    'ZoneType',
    'predict_btc_move',
    'get_btc_signal',
    'analyze_btc',
]