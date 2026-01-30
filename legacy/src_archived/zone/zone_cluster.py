"""
Cluster Zone Calculator
=======================

Fib Zone과 Divergence Zone의 교집합 계산.

Cluster Zone = Fib Zone ∩ Divergence Zone
→ 독립 근거 2개 겹침 = 강한 진입 후보
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple
import pandas as pd

from .builder import FibZone
from .zone_div import DivZone


@dataclass
class ClusterZone:
    """
    Cluster Zone = Fib ∩ Divergence

    진입 후보 가격대 (타이밍은 Trigger가 결정)
    """
    # 교집합 범위
    zone_low: float
    zone_high: float

    # 방향
    side: Literal['long', 'short']

    # 구성 요소
    fib_zone: FibZone
    div_zone: DivZone

    # 등급 (A급 = Fib + Div 겹침)
    grade: Literal['A', 'B'] = 'A'

    # 메타
    confidence: float = 1.0

    @property
    def center(self) -> float:
        return (self.zone_low + self.zone_high) / 2

    @property
    def width(self) -> float:
        return self.zone_high - self.zone_low

    def contains(self, price: float) -> bool:
        """가격이 Cluster Zone 내에 있는지"""
        return self.zone_low <= price <= self.zone_high


def intersect_fib_div(
    fib_zone: FibZone,
    div_zone: DivZone,
) -> Optional[ClusterZone]:
    """
    Fib Zone과 Div Zone의 교집합 계산

    Args:
        fib_zone: FibZone (fib_low ~ fib_high)
        div_zone: DivZone (boundary 또는 range)

    Returns:
        ClusterZone if 교집합 존재, else None
    """
    # Fib Zone 범위
    fib_low = fib_zone.zone_low
    fib_high = fib_zone.zone_high

    # Div Zone 범위 추출
    if div_zone.kind == 'regular':
        if div_zone.boundary_price is None:
            return None

        if div_zone.side == 'long':
            # Regular Bullish: price <= boundary
            div_low = fib_low  # Fib 하단부터
            div_high = div_zone.boundary_price
        else:
            # Regular Bearish: price >= boundary
            div_low = div_zone.boundary_price
            div_high = fib_high  # Fib 상단까지

    else:  # hidden
        if div_zone.range_low is None or div_zone.range_high is None:
            return None
        div_low = div_zone.range_low
        div_high = div_zone.range_high

    # 교집합 계산
    intersect_low = max(fib_low, div_low)
    intersect_high = min(fib_high, div_high)

    # 교집합이 없으면 None
    if intersect_low >= intersect_high:
        return None

    # Side 결정 (Div Zone의 side 따름)
    side = div_zone.side

    # Fib Zone 위치와 Div side 일치 확인
    # Long은 Fib 하단에서, Short은 Fib 상단에서
    fib_center = fib_zone.fib_price
    cluster_center = (intersect_low + intersect_high) / 2

    if side == 'long' and cluster_center > fib_center:
        # Long인데 Fib 상단에 있으면 신뢰도 감소
        confidence = 0.7
    elif side == 'short' and cluster_center < fib_center:
        # Short인데 Fib 하단에 있으면 신뢰도 감소
        confidence = 0.7
    else:
        confidence = 1.0

    return ClusterZone(
        zone_low=intersect_low,
        zone_high=intersect_high,
        side=side,
        fib_zone=fib_zone,
        div_zone=div_zone,
        grade='A',
        confidence=confidence,
    )


def get_cluster_zones(
    fib_zones: List[FibZone],
    div_zones: List[DivZone],
    current_price: Optional[float] = None,
    max_distance_pct: float = 0.05,
) -> List[ClusterZone]:
    """
    모든 Fib Zone과 Div Zone의 교집합 계산

    Args:
        fib_zones: FibZone 리스트
        div_zones: DivZone 리스트
        current_price: 현재가 (가까운 Zone 우선 정렬용)
        max_distance_pct: 현재가 대비 최대 거리 (%) - 너무 먼 Zone 제외

    Returns:
        ClusterZone 리스트 (거리순 정렬)
    """
    clusters = []

    for fib in fib_zones:
        for div in div_zones:
            cluster = intersect_fib_div(fib, div)
            if cluster is not None:
                # 현재가 대비 거리 필터
                if current_price is not None:
                    distance_pct = abs(cluster.center - current_price) / current_price
                    if distance_pct > max_distance_pct:
                        continue

                clusters.append(cluster)

    # 현재가 기준 거리순 정렬
    if current_price is not None:
        clusters.sort(key=lambda c: abs(c.center - current_price))

    return clusters


@dataclass
class FibOnlyZone:
    """
    B급 Zone = Fib만 있고 Divergence 없음
    대체 트리거 필요
    """
    fib_zone: FibZone
    side: Literal['long', 'short']
    grade: Literal['B'] = 'B'
    size_mult: float = 0.5  # 기본 사이즈의 50%

    @property
    def zone_low(self) -> float:
        return self.fib_zone.zone_low

    @property
    def zone_high(self) -> float:
        return self.fib_zone.zone_high

    @property
    def center(self) -> float:
        return self.fib_zone.fib_price

    def contains(self, price: float) -> bool:
        return self.fib_zone.contains(price)


def get_fib_only_zones(
    fib_zones: List[FibZone],
    cluster_zones: List[ClusterZone],
    current_price: float,
    max_distance_pct: float = 0.05,
) -> List[FibOnlyZone]:
    """
    Cluster Zone에 포함되지 않은 Fib Zone 추출 (B급)

    Args:
        fib_zones: 전체 FibZone 리스트
        cluster_zones: A급 ClusterZone 리스트
        current_price: 현재가
        max_distance_pct: 최대 거리

    Returns:
        FibOnlyZone 리스트 (B급)
    """
    # Cluster에 사용된 Fib Zone 추출
    used_fibs = {id(c.fib_zone) for c in cluster_zones}

    b_zones = []
    for fib in fib_zones:
        if id(fib) in used_fibs:
            continue

        # 거리 필터
        distance_pct = abs(fib.fib_price - current_price) / current_price
        if distance_pct > max_distance_pct:
            continue

        # Side 결정 (Fib 레벨 대비 현재가 위치)
        if current_price > fib.fib_price:
            # 현재가가 Fib 위 → Long 방향 (아래에서 지지)
            side = 'long'
        else:
            # 현재가가 Fib 아래 → Short 방향 (위에서 저항)
            side = 'short'

        b_zones.append(FibOnlyZone(
            fib_zone=fib,
            side=side,
        ))

    # 거리순 정렬
    b_zones.sort(key=lambda z: abs(z.center - current_price))

    return b_zones


def get_all_entry_zones(
    fib_zones: List[FibZone],
    div_zones: List[DivZone],
    current_price: float,
    max_distance_pct: float = 0.05,
) -> Tuple[List[ClusterZone], List[FibOnlyZone]]:
    """
    A급 (Cluster)과 B급 (Fib only) Zone 모두 반환

    Returns:
        (A급 ClusterZone 리스트, B급 FibOnlyZone 리스트)
    """
    # A급: Cluster Zone
    a_zones = get_cluster_zones(
        fib_zones, div_zones, current_price, max_distance_pct
    )

    # B급: Fib only (Cluster에 없는 것)
    b_zones = get_fib_only_zones(
        fib_zones, a_zones, current_price, max_distance_pct
    )

    return a_zones, b_zones
