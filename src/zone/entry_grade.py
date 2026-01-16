"""
Entry Grade Calculator
======================

Zone 기반 진입 등급 및 사이즈 결정.

Grade A: Fib + Divergence (Cluster Zone) → base_size
Grade B: Fib only + 대체 트리거 → 0.25~0.5 size
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Union, Tuple
import numpy as np
import pandas as pd

from .zone_cluster import ClusterZone, FibOnlyZone


@dataclass
class EntrySignal:
    """진입 신호"""
    # Zone 정보
    zone: Union[ClusterZone, FibOnlyZone]
    grade: Literal['A', 'B']

    # 방향
    side: Literal['long', 'short']

    # 사이즈
    size_mult: float  # 0.25 ~ 1.0

    # 진입 가격대
    entry_low: float
    entry_high: float

    # 트리거 충족 여부
    trigger_ok: bool

    # 메타
    confidence: float = 1.0
    reason: str = ''

    @property
    def entry_center(self) -> float:
        return (self.entry_low + self.entry_high) / 2


@dataclass
class GradeConfig:
    """등급별 설정"""
    # A급 사이즈
    grade_a_size: float = 1.0

    # B급 사이즈 (대체 트리거 충족 시)
    grade_b_size: float = 0.5
    grade_b_min_size: float = 0.25

    # B급 트리거 요구
    grade_b_require_trigger: bool = True


def calc_entry_grade(
    zone: Union[ClusterZone, FibOnlyZone],
    trigger_ok: bool,
    config: Optional[GradeConfig] = None,
) -> EntrySignal:
    """
    Zone과 트리거 상태로 진입 등급/사이즈 결정

    Args:
        zone: ClusterZone (A급) 또는 FibOnlyZone (B급)
        trigger_ok: 대체 트리거 충족 여부 (B급에서 필수)
        config: 등급 설정

    Returns:
        EntrySignal
    """
    if config is None:
        config = GradeConfig()

    # 등급 결정
    if isinstance(zone, ClusterZone):
        grade = 'A'
        size_mult = config.grade_a_size
        entry_low = zone.zone_low
        entry_high = zone.zone_high
        side = zone.side
        confidence = zone.confidence
        reason = f"Fib({zone.fib_zone.fib_ratio:.3f}) + Div({zone.div_zone.kind})"
        signal_trigger_ok = True  # A급은 트리거 없어도 진입 가능

    else:  # FibOnlyZone
        grade = 'B'

        if config.grade_b_require_trigger and not trigger_ok:
            # B급인데 트리거 미충족 → 진입 불가
            size_mult = 0.0
            signal_trigger_ok = False
            reason = "B급 트리거 미충족"
        else:
            size_mult = config.grade_b_size
            signal_trigger_ok = trigger_ok
            reason = f"Fib({zone.fib_zone.fib_ratio:.3f}) only"

        entry_low = zone.zone_low
        entry_high = zone.zone_high
        side = zone.side
        confidence = 0.7  # B급은 신뢰도 감소

    return EntrySignal(
        zone=zone,
        grade=grade,
        side=side,
        size_mult=size_mult,
        entry_low=entry_low,
        entry_high=entry_high,
        trigger_ok=signal_trigger_ok,
        confidence=confidence,
        reason=reason,
    )


def select_best_zone(
    a_zones: List[ClusterZone],
    b_zones: List[FibOnlyZone],
    current_price: float,
    side_filter: Optional[Literal['long', 'short']] = None,
) -> Optional[Union[ClusterZone, FibOnlyZone]]:
    """
    가장 좋은 진입 Zone 선택

    우선순위:
    1. A급 (Cluster) 중 가장 가까운 것
    2. B급 (Fib only) 중 가장 가까운 것

    Args:
        a_zones: A급 ClusterZone 리스트
        b_zones: B급 FibOnlyZone 리스트
        current_price: 현재가
        side_filter: 특정 방향만 필터링

    Returns:
        선택된 Zone (없으면 None)
    """
    # Side 필터
    if side_filter:
        a_zones = [z for z in a_zones if z.side == side_filter]
        b_zones = [z for z in b_zones if z.side == side_filter]

    # A급 우선
    if a_zones:
        # 가장 가까운 A급
        return min(a_zones, key=lambda z: abs(z.center - current_price))

    # B급
    if b_zones:
        return min(b_zones, key=lambda z: abs(z.center - current_price))

    return None


def get_entry_signal(
    a_zones: List[ClusterZone],
    b_zones: List[FibOnlyZone],
    current_price: float,
    b_trigger_ok: bool = False,
    side_filter: Optional[Literal['long', 'short']] = None,
    config: Optional[GradeConfig] = None,
) -> Optional[EntrySignal]:
    """
    최종 진입 신호 생성

    Args:
        a_zones: A급 Zone 리스트
        b_zones: B급 Zone 리스트
        current_price: 현재가
        b_trigger_ok: B급 대체 트리거 충족 여부
        side_filter: 방향 필터
        config: 등급 설정

    Returns:
        EntrySignal (진입 가능한 Zone 없으면 None)
    """
    zone = select_best_zone(a_zones, b_zones, current_price, side_filter)

    if zone is None:
        return None

    trigger_ok = True if isinstance(zone, ClusterZone) else b_trigger_ok

    signal = calc_entry_grade(zone, trigger_ok, config)

    # 사이즈가 0이면 진입 불가
    if signal.size_mult <= 0:
        return None

    return signal


def filter_zones_by_price(
    a_zones: List[ClusterZone],
    b_zones: List[FibOnlyZone],
    current_price: float,
    side: Literal['long', 'short'],
) -> Tuple[List[ClusterZone], List[FibOnlyZone]]:
    """
    현재가 위치 기준 Zone 필터링

    Long: 현재가 아래에 있는 Zone (지지)
    Short: 현재가 위에 있는 Zone (저항)

    Args:
        a_zones: A급 Zone
        b_zones: B급 Zone
        current_price: 현재가
        side: 방향

    Returns:
        필터링된 (A급, B급) 튜플
    """
    if side == 'long':
        # Long: Zone이 현재가 아래에 있어야 함
        filtered_a = [z for z in a_zones if z.zone_high <= current_price and z.side == 'long']
        filtered_b = [z for z in b_zones if z.zone_high <= current_price and z.side == 'long']
    else:
        # Short: Zone이 현재가 위에 있어야 함
        filtered_a = [z for z in a_zones if z.zone_low >= current_price and z.side == 'short']
        filtered_b = [z for z in b_zones if z.zone_low >= current_price and z.side == 'short']

    return filtered_a, filtered_b


def check_price_in_zone(
    current_price: float,
    a_zones: List[ClusterZone],
    b_zones: List[FibOnlyZone],
) -> Optional[Union[ClusterZone, FibOnlyZone]]:
    """
    현재가가 어떤 Zone 안에 있는지 확인

    Returns:
        현재가가 속한 Zone (A급 우선)
    """
    # A급 먼저 확인
    for zone in a_zones:
        if zone.contains(current_price):
            return zone

    # B급 확인
    for zone in b_zones:
        if zone.contains(current_price):
            return zone

    return None
