"""
Fibonacci Retracement/Extension Levels
======================================

ZigZag 스윙을 기반으로 피보나치 되돌림/확장 레벨 계산.

핵심 개념:
- 되돌림 (Retracement): 0.236, 0.382, 0.5, 0.618, 0.786
- 확장 (Extension): 1.272, 1.414, 1.618
- 골든 포켓 (Golden Pocket): 0.618 ~ 0.65 구간

TradingView 기준:
- Up Swing: R0 = 고점, R100 = 저점 (고점에서 저점 방향으로 되돌림)
- Down Swing: R0 = 저점, R100 = 고점 (저점에서 고점 방향으로 되돌림)

Origin: param_search_confluence_v0.py의 fib_from_latest()
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .zigzag import SwingInfo


# 기본 피보나치 비율
DEFAULT_RETRACEMENTS = (0.236, 0.382, 0.5, 0.618, 0.786)
DEFAULT_EXTENSIONS = (1.272, 1.414, 1.618)


@dataclass
class FibLevel:
    """개별 피보나치 레벨"""
    ratio: float
    label: str  # "R24", "R38", "R62" 등
    price: float
    low: float  # 존 하단
    high: float  # 존 상단
    half_width: float  # 존 폭의 절반


@dataclass
class FibLevels:
    """피보나치 레벨 집합"""
    swing: SwingInfo
    anchors: Dict[str, float]  # {'0': R0가격, '1': R100가격}
    retracements: Dict[float, float]  # {비율: 가격}
    extensions: Dict[float, float]
    golden_pocket: Tuple[float, float]  # (하단, 상단)
    levels: List[FibLevel] = field(default_factory=list)


def calc_fib_levels(
    swing: SwingInfo,
    *,
    retracements: Tuple[float, ...] = DEFAULT_RETRACEMENTS,
    extensions: Tuple[float, ...] = DEFAULT_EXTENSIONS,
    include_extremes: bool = True,
    ext_side: str = 'auto',
) -> FibLevels:
    """
    스윙 정보로부터 피보나치 레벨 계산

    Args:
        swing: SwingInfo (zigzag에서 추출)
        retracements: 되돌림 비율들
        extensions: 확장 비율들
        include_extremes: 0.0, 1.0 포함 여부
        ext_side: 확장 방향 ('auto', 'above', 'below')

    Returns:
        FibLevels
    """
    # 확장 방향 자동 결정
    if ext_side == 'auto':
        ext_side = 'below' if swing.direction == 'down' else 'above'

    if swing.direction == 'up':
        # Up Swing: 저점 → 고점
        lo, hi = swing.start_price, swing.end_price
        d = hi - lo
        # TradingView: R0 = 고점, R100 = 저점
        anchors = {'0': hi, '1': lo}

        # 되돌림 (고점에서 r만큼 내려감)
        rets = {r: hi - d * r for r in retracements}

        # 확장
        if ext_side == 'below':
            exts = {e: lo - d * (e - 1.0) for e in extensions}
        else:
            exts = {e: hi + d * (e - 1.0) for e in extensions}

        # 골든 포켓 (0.618 ~ 0.65)
        gp = (hi - d * 0.65, hi - d * 0.618)

    else:  # down swing
        # Down Swing: 고점 → 저점
        hi, lo = swing.start_price, swing.end_price
        d = hi - lo
        # TradingView: R0 = 저점, R100 = 고점
        anchors = {'0': lo, '1': hi}

        # 되돌림 (저점에서 r만큼 올라감)
        rets = {r: lo + d * r for r in retracements}

        # 확장
        if ext_side == 'above':
            exts = {e: hi + d * (e - 1.0) for e in extensions}
        else:
            exts = {e: lo - d * (e - 1.0) for e in extensions}

        # 골든 포켓
        gp = (lo + d * 0.618, lo + d * 0.65)

    # 극값 포함
    if include_extremes:
        rets = {**rets, 0.0: float(anchors['0']), 1.0: float(anchors['1'])}

    gp = (min(gp), max(gp))

    return FibLevels(
        swing=swing,
        anchors=anchors,
        retracements=rets,
        extensions=exts,
        golden_pocket=gp,
    )


def build_fib_zones(
    fib_levels: FibLevels,
    atr: float,
    *,
    fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    min_half_atr: float = 0.25,
    max_half_mult: float = 0.8,
) -> List[FibLevel]:
    """
    피보나치 레벨에 존(Zone) 폭 부여

    Args:
        fib_levels: FibLevels
        atr: 현재 ATR 값
        fib_ratios: 존으로 만들 비율들
        min_half_atr: 최소 존 폭 (ATR 배수)
        max_half_mult: 최대 존 폭 (ATR 배수)

    Returns:
        List[FibLevel]
    """
    levels = []

    for r in fib_ratios:
        # 되돌림에서 가격 찾기
        if r in fib_levels.retracements:
            price = fib_levels.retracements[r]
        elif r in fib_levels.extensions:
            price = fib_levels.extensions[r]
        else:
            continue

        # 존 폭 계산
        base = max(min_half_atr * atr, 1e-12)
        half = base * max(0.12, float(r) if r <= 1.0 else (r - 1.0))
        half = min(half, max_half_mult * atr)

        label = f"R{int(round(r * 100))}"

        levels.append(FibLevel(
            ratio=r,
            label=label,
            price=price,
            low=price - half,
            high=price + half,
            half_width=half,
        ))

    fib_levels.levels = levels
    return levels


def is_in_zone(price: float, zone: FibLevel) -> bool:
    """가격이 존 안에 있는지 확인"""
    return zone.low <= price <= zone.high


def is_in_golden_pocket(price: float, fib_levels: FibLevels) -> bool:
    """가격이 골든 포켓 안에 있는지 확인"""
    gp_low, gp_high = fib_levels.golden_pocket
    return gp_low <= price <= gp_high


def find_nearest_zone(price: float, zones: List[FibLevel]) -> Optional[FibLevel]:
    """가격에 가장 가까운 존 찾기"""
    if not zones:
        return None

    best = None
    best_dist = float('inf')

    for zone in zones:
        dist = abs(price - zone.price)
        if dist < best_dist:
            best_dist = dist
            best = zone

    return best


def find_containing_zones(price: float, zones: List[FibLevel]) -> List[FibLevel]:
    """가격을 포함하는 모든 존 찾기"""
    return [z for z in zones if is_in_zone(price, z)]