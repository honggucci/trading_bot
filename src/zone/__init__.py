"""
Zone Module
===========

Fib Zone + Divergence Zone + Cluster Zone

구조:
- builder.py: Fib Zone 생성
- zone_div.py: Divergence Zone 계산
- zone_cluster.py: Cluster Zone = Fib ∩ Divergence
- entry_grade.py: Entry Grade A/B
"""

from .builder import (
    ZoneBuilder,
    FibZone,
    fib_to_price,
    price_to_fib,
    analyze_zone_overlaps,
    FIB_0,
    FIB_1,
    RANGE,
)

from .zone_div import (
    DivZone,
    OversoldSegment,
    calc_rsi,
    calc_stoch_rsi,
    find_oversold_segments,
    find_overbought_segments,
    get_reference_from_segment,
    calc_regular_bullish_boundary,
    calc_hidden_bullish_range,
    calc_regular_bearish_boundary,
    calc_hidden_bearish_range,
    get_div_zones,
)

from .zone_cluster import (
    ClusterZone,
    FibOnlyZone,
    intersect_fib_div,
    get_cluster_zones,
    get_fib_only_zones,
    get_all_entry_zones,
)

from .entry_grade import (
    EntrySignal,
    GradeConfig,
    calc_entry_grade,
    select_best_zone,
    get_entry_signal,
    filter_zones_by_price,
    check_price_in_zone,
)

__all__ = [
    # builder
    'ZoneBuilder',
    'FibZone',
    'fib_to_price',
    'price_to_fib',
    'analyze_zone_overlaps',
    'FIB_0',
    'FIB_1',
    'RANGE',
    # zone_div
    'DivZone',
    'OversoldSegment',
    'calc_rsi',
    'calc_stoch_rsi',
    'find_oversold_segments',
    'find_overbought_segments',
    'get_reference_from_segment',
    'calc_regular_bullish_boundary',
    'calc_hidden_bullish_range',
    'calc_regular_bearish_boundary',
    'calc_hidden_bearish_range',
    'get_div_zones',
    # zone_cluster
    'ClusterZone',
    'FibOnlyZone',
    'intersect_fib_div',
    'get_cluster_zones',
    'get_fib_only_zones',
    'get_all_entry_zones',
    # entry_grade
    'EntrySignal',
    'GradeConfig',
    'calc_entry_grade',
    'select_best_zone',
    'get_entry_signal',
    'filter_zones_by_price',
    'check_price_in_zone',
]
