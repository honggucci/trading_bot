"""
Cycle Anchor - Bitcoin Halving Cycle Based Fibonacci
=====================================================

비트코인 반감기 사이클 기반 Fibonacci 앵커 시스템.

핵심 사상:
- "전 사이클 고점 = 현 사이클 저점 (지지선)"
- 폭락 시 전 사이클 ATH가 지지선 역할
- 사이클 고점/저점으로 Fib 0/1 고정

사용법:
```python
from src.context.cycle_anchor import get_btc_cycle_anchor, get_fib_levels

# 현재 사이클 앵커 가져오기
anchor = get_btc_cycle_anchor()

# Fib 레벨 계산
levels = get_fib_levels(anchor.cycle_low, anchor.cycle_high)

# 현재가 위치
position = anchor.get_position(92000)  # 0.69 (69%)

# 폭락 시 지지선
support = anchor.get_crash_support()  # $69,000
```
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


# ============================================================================
# Bitcoin Cycle Data (Hardcoded - 사이클 이론 기반)
# ============================================================================

@dataclass
class CycleData:
    """단일 사이클 데이터"""
    cycle_num: int
    halving_date: str
    cycle_low: float
    cycle_low_date: str
    cycle_high: float
    cycle_high_date: str

    @property
    def range(self) -> float:
        return self.cycle_high - self.cycle_low

    @property
    def multiplier(self) -> float:
        """저점 대비 고점 배수"""
        return self.cycle_high / self.cycle_low if self.cycle_low > 0 else 0


# 비트코인 역대 사이클 데이터
BTC_CYCLES: Dict[int, CycleData] = {
    1: CycleData(
        cycle_num=1,
        halving_date="2012-11-28",
        cycle_low=2,
        cycle_low_date="2011-11-18",
        cycle_high=1200,
        cycle_high_date="2013-11-29",
    ),
    2: CycleData(
        cycle_num=2,
        halving_date="2016-07-09",
        cycle_low=200,
        cycle_low_date="2015-01-14",
        cycle_high=20000,
        cycle_high_date="2017-12-17",
    ),
    3: CycleData(
        cycle_num=3,
        halving_date="2020-05-11",
        cycle_low=3200,
        cycle_low_date="2018-12-15",
        cycle_high=69000,
        cycle_high_date="2021-11-10",
    ),
    4: CycleData(
        cycle_num=4,
        halving_date="2024-04-20",
        cycle_low=15500,
        cycle_low_date="2022-11-21",
        cycle_high=126296,
        cycle_high_date="2025-10-06",
    ),
}

# 현재 사이클 번호
CURRENT_CYCLE = 4


# ============================================================================
# Cycle Anchor
# ============================================================================

@dataclass
class CycleAnchor:
    """
    사이클 기반 Fibonacci 앵커

    사이클 이론:
    - Fib 0 = 현 사이클 저점 (≈ 전 사이클 고점)
    - Fib 1 = 현 사이클 고점
    - 폭락 지지선 = 전 사이클 고점
    """
    # 현 사이클
    cycle_num: int
    cycle_low: float
    cycle_high: float
    cycle_low_date: str
    cycle_high_date: str

    # 전 사이클 (폭락 지지선용)
    prev_cycle_high: float
    prev_cycle_high_date: str

    # 메타
    halving_date: str
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_position(self, price: float) -> float:
        """
        현재가의 사이클 내 위치 (0~1)

        0 = 사이클 저점
        1 = 사이클 고점
        >1 = 신고점 돌파
        """
        if self.cycle_high == self.cycle_low:
            return 0.5
        return (price - self.cycle_low) / (self.cycle_high - self.cycle_low)

    def get_price_at_fib(self, fib_level: float) -> float:
        """
        Fib 레벨에 해당하는 가격

        Args:
            fib_level: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0 등
        """
        return self.cycle_low + (self.cycle_high - self.cycle_low) * fib_level

    def get_crash_support(self) -> float:
        """
        폭락 시 예상 지지선 (전 사이클 고점)

        사이클 이론: "전 사이클 ATH = 현 사이클 바닥"
        """
        return self.prev_cycle_high

    def get_expected_crash_low(self, crash_pct: float = 0.75) -> float:
        """
        사이클 종료 후 예상 저점

        역사적으로 사이클 고점에서 70-80% 하락
        기본값: 75% 하락 (0.25 유지)

        Args:
            crash_pct: 하락률 (0.7 = 70%, 0.8 = 80%)

        Returns:
            예상 저점 가격
        """
        return self.cycle_high * (1 - crash_pct)

    def get_wyckoff_zones(self) -> Dict[str, Tuple[float, float]]:
        """
        Wyckoff 패턴 기반 존 (사이클 종료 후)

        사이클 고점에서 70-80% 하락 → Wyckoff Accumulation 시작
        """
        high = self.cycle_high
        return {
            "markdown_zone": (high * 0.5, high * 1.0),      # 하락 추세
            "accumulation_zone": (high * 0.2, high * 0.35), # 매집 구간 (70-80% 하락)
            "spring_zone": (high * 0.15, high * 0.25),      # Spring (추가 하락 후 반등)
            "markup_zone": (high * 0.3, high * 0.5),        # 상승 시작
        }

    def get_all_fib_levels(
        self,
        ratios: Tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0),
    ) -> Dict[str, float]:
        """모든 Fib 레벨 가격 반환"""
        return {
            f"{r:.3f}": self.get_price_at_fib(r)
            for r in ratios
        }

    def get_zone(self, price: float, tolerance: float = 0.02) -> Optional[str]:
        """
        현재가가 위치한 Fib 존 반환

        Args:
            price: 현재가
            tolerance: 존 판정 허용 오차 (2%)

        Returns:
            "0.382", "0.5", "0.618" 등 또는 None
        """
        position = self.get_position(price)
        key_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        for level in key_levels:
            if abs(position - level) <= tolerance:
                return f"{level:.3f}"
        return None

    def is_near_crash_support(self, price: float, tolerance_pct: float = 0.05) -> bool:
        """
        현재가가 폭락 지지선(전 사이클 고점) 근처인지

        Args:
            price: 현재가
            tolerance_pct: 허용 오차 (5%)
        """
        support = self.get_crash_support()
        return abs(price - support) / support <= tolerance_pct


# ============================================================================
# Factory Functions
# ============================================================================

def get_btc_cycle_anchor(cycle_num: Optional[int] = None) -> CycleAnchor:
    """
    비트코인 사이클 앵커 가져오기

    Args:
        cycle_num: 사이클 번호 (None이면 현재 사이클)

    Returns:
        CycleAnchor 객체
    """
    if cycle_num is None:
        cycle_num = CURRENT_CYCLE

    current = BTC_CYCLES[cycle_num]
    prev = BTC_CYCLES.get(cycle_num - 1)

    return CycleAnchor(
        cycle_num=current.cycle_num,
        cycle_low=current.cycle_low,
        cycle_high=current.cycle_high,
        cycle_low_date=current.cycle_low_date,
        cycle_high_date=current.cycle_high_date,
        prev_cycle_high=prev.cycle_high if prev else 0,
        prev_cycle_high_date=prev.cycle_high_date if prev else "",
        halving_date=current.halving_date,
    )


def get_fib_levels(
    low: float,
    high: float,
    ratios: Tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0),
    include_extensions: bool = True,
) -> Dict[str, float]:
    """
    Fibonacci 레벨 계산

    Args:
        low: Fib 0 (저점)
        high: Fib 1 (고점)
        ratios: 되돌림 비율
        include_extensions: 확장 레벨 포함 여부

    Returns:
        {레벨: 가격} 딕셔너리
    """
    range_size = high - low
    levels = {}

    for r in ratios:
        levels[f"{r:.3f}"] = low + range_size * r

    if include_extensions:
        extensions = (1.272, 1.618, 2.0, 2.618)
        for ext in extensions:
            levels[f"{ext:.3f}"] = low + range_size * ext

    return levels


def get_current_cycle_position(price: float, symbol: str = "BTC") -> Dict[str, any]:
    """
    현재가의 사이클 위치 정보 반환

    Args:
        price: 현재가
        symbol: 심볼 (현재 BTC만 지원)

    Returns:
        {
            "position": 0.69,
            "zone": "0.618",
            "crash_support": 69000,
            "near_support": False,
            ...
        }
    """
    if symbol.upper() not in ("BTC", "BTC-USDT", "BTCUSDT"):
        raise ValueError(f"Unsupported symbol: {symbol}. Only BTC supported.")

    anchor = get_btc_cycle_anchor()

    return {
        "symbol": "BTC",
        "price": price,
        "cycle_num": anchor.cycle_num,
        "position": anchor.get_position(price),
        "zone": anchor.get_zone(price),
        "cycle_low": anchor.cycle_low,
        "cycle_high": anchor.cycle_high,
        "crash_support": anchor.get_crash_support(),
        "near_support": anchor.is_near_crash_support(price),
        "fib_levels": anchor.get_all_fib_levels(),
    }


# ============================================================================
# Integration with Spectral Analysis
# ============================================================================

def normalize_price_to_cycle(
    prices: np.ndarray,
    anchor: Optional[CycleAnchor] = None,
) -> np.ndarray:
    """
    가격을 사이클 범위로 정규화 (0~1)

    스펙트럴 분석 전처리용:
    - 사이클 저점 = 0
    - 사이클 고점 = 1

    Args:
        prices: 가격 배열
        anchor: CycleAnchor (None이면 현재 BTC 사이클)

    Returns:
        정규화된 가격 (0~1 범위, 범위 초과 가능)
    """
    if anchor is None:
        anchor = get_btc_cycle_anchor()

    return (prices - anchor.cycle_low) / (anchor.cycle_high - anchor.cycle_low)


def denormalize_price_from_cycle(
    normalized: np.ndarray,
    anchor: Optional[CycleAnchor] = None,
) -> np.ndarray:
    """
    정규화된 가격을 실제 가격으로 복원

    Args:
        normalized: 정규화된 가격 (0~1)
        anchor: CycleAnchor

    Returns:
        실제 가격
    """
    if anchor is None:
        anchor = get_btc_cycle_anchor()

    return normalized * (anchor.cycle_high - anchor.cycle_low) + anchor.cycle_low


# ============================================================================
# ETH Cycle Data (참고용 - 아직 미검증)
# ============================================================================

ETH_CYCLES: Dict[int, CycleData] = {
    1: CycleData(
        cycle_num=1,
        halving_date="N/A",  # ETH는 반감기 없음
        cycle_low=80,
        cycle_low_date="2018-12-15",
        cycle_high=4800,
        cycle_high_date="2021-11-10",
    ),
    2: CycleData(
        cycle_num=2,
        halving_date="N/A",
        cycle_low=880,
        cycle_low_date="2022-06-18",
        cycle_high=4500,  # 추정 (미확정)
        cycle_high_date="2025-??-??",
    ),
}


def get_eth_cycle_anchor(cycle_num: int = 2) -> CycleAnchor:
    """ETH 사이클 앵커 (미검증, 참고용)"""
    current = ETH_CYCLES[cycle_num]
    prev = ETH_CYCLES.get(cycle_num - 1)

    return CycleAnchor(
        cycle_num=current.cycle_num,
        cycle_low=current.cycle_low,
        cycle_high=current.cycle_high,
        cycle_low_date=current.cycle_low_date,
        cycle_high_date=current.cycle_high_date,
        prev_cycle_high=prev.cycle_high if prev else 0,
        prev_cycle_high_date=prev.cycle_high_date if prev else "",
        halving_date=current.halving_date,
    )
