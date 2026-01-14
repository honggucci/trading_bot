"""
Multi-Timeframe Fibonacci System
================================

계층적 Fibonacci 시스템.

구조:
- L0 (Cycle): 사이클 고점/저점 (고정)
- L1 (Weekly): 사이클 내 주봉 스윙
- L2 (Daily): 주봉 스윙 내 일봉 스윙
- L3 (4H): 일봉 스윙 내 4시간봉 스윙
- L4 (1H/15m): 실행 레벨

핵심 원리:
- 상위 TF Fib가 하위 TF의 범위를 제약
- "Fib within Fib" - 프랙탈 구조
- ZigZag 파라미터는 TF별로 최적화

사용법:
```python
from src.context.multi_tf_fib import MultiTFFibSystem

system = MultiTFFibSystem()

# 계층적 Fib 계산
hierarchy = system.build_hierarchy(
    df_1d=df_daily,
    df_4h=df_4h,
    df_1h=df_1h,
)

# 현재가 위치 (모든 레벨)
positions = system.get_positions(price=92000)

# Confluence 존 찾기
zones = system.find_confluence_zones(tolerance=0.02)
```
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .cycle_anchor import get_btc_cycle_anchor, CycleAnchor
from .zigzag import zigzag_pivots, get_latest_swing, SwingInfo


# ============================================================================
# ZigZag Parameters per Timeframe
# ============================================================================

@dataclass
class ZigZagParams:
    """타임프레임별 ZigZag 파라미터"""
    up_pct: float
    down_pct: float
    atr_mult: float
    min_bars: int
    min_swing_atr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'up_pct': self.up_pct,
            'down_pct': self.down_pct,
            'atr_mult': self.atr_mult,
            'min_bars': self.min_bars,
            'min_swing_atr': self.min_swing_atr,
        }


# 타임프레임별 기본 ZigZag 파라미터
# - 상위 TF: 더 큰 움직임만 감지
# - 하위 TF: 작은 움직임도 감지
DEFAULT_ZIGZAG_PARAMS: Dict[str, ZigZagParams] = {
    '1W': ZigZagParams(up_pct=0.10, down_pct=0.10, atr_mult=3.0, min_bars=4, min_swing_atr=2.0),
    '1D': ZigZagParams(up_pct=0.05, down_pct=0.05, atr_mult=2.5, min_bars=5, min_swing_atr=1.5),
    '4H': ZigZagParams(up_pct=0.03, down_pct=0.03, atr_mult=2.0, min_bars=6, min_swing_atr=1.2),
    '1H': ZigZagParams(up_pct=0.02, down_pct=0.02, atr_mult=1.5, min_bars=5, min_swing_atr=1.0),
    '15m': ZigZagParams(up_pct=0.01, down_pct=0.01, atr_mult=1.2, min_bars=5, min_swing_atr=0.8),
}


# ============================================================================
# Fib Level at Each Timeframe
# ============================================================================

@dataclass
class TFFibLevel:
    """단일 타임프레임의 Fib 정보"""
    timeframe: str
    fib_low: float
    fib_high: float
    swing: Optional[SwingInfo]
    swing_direction: str  # 'up' or 'down'

    @property
    def range(self) -> float:
        return self.fib_high - self.fib_low

    def get_position(self, price: float) -> float:
        """가격의 Fib 위치 (0~1)"""
        if self.range == 0:
            return 0.5
        return (price - self.fib_low) / self.range

    def get_price_at_fib(self, ratio: float) -> float:
        """Fib 비율에 해당하는 가격"""
        return self.fib_low + self.range * ratio

    def get_fib_levels(
        self,
        ratios: Tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0),
    ) -> Dict[str, float]:
        """모든 Fib 레벨"""
        return {f"{r:.3f}": self.get_price_at_fib(r) for r in ratios}


@dataclass
class FibHierarchy:
    """계층적 Fib 구조"""
    cycle: CycleAnchor
    levels: Dict[str, TFFibLevel] = field(default_factory=dict)

    def get_position_all(self, price: float) -> Dict[str, float]:
        """모든 TF에서의 Fib 위치"""
        result = {
            'cycle': self.cycle.get_position(price),
        }
        for tf, level in self.levels.items():
            result[tf] = level.get_position(price)
        return result

    def find_confluence_zones(
        self,
        tolerance: float = 0.02,
        min_tf_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        여러 TF에서 겹치는 Fib 레벨 찾기

        Args:
            tolerance: 레벨 간 허용 오차 (2%)
            min_tf_count: 최소 겹침 TF 수

        Returns:
            [{"price": 85000, "fib_ratio": 0.618, "timeframes": ["1D", "4H"]}]
        """
        # 모든 TF의 Fib 레벨 수집
        all_levels = []

        # 사이클 레벨
        for ratio in (0.236, 0.382, 0.5, 0.618, 0.786):
            price = self.cycle.get_price_at_fib(ratio)
            all_levels.append({
                'price': price,
                'ratio': ratio,
                'tf': 'cycle',
            })

        # 각 TF 레벨
        for tf, level in self.levels.items():
            for ratio in (0.236, 0.382, 0.5, 0.618, 0.786):
                price = level.get_price_at_fib(ratio)
                all_levels.append({
                    'price': price,
                    'ratio': ratio,
                    'tf': tf,
                })

        # Confluence 찾기
        confluence_zones = []
        used = set()

        for i, lvl1 in enumerate(all_levels):
            if i in used:
                continue

            cluster = [lvl1]
            tfs = {lvl1['tf']}

            for j, lvl2 in enumerate(all_levels):
                if j <= i or j in used:
                    continue

                # 가격 근접 체크
                if abs(lvl2['price'] - lvl1['price']) / lvl1['price'] <= tolerance:
                    cluster.append(lvl2)
                    tfs.add(lvl2['tf'])
                    used.add(j)

            if len(tfs) >= min_tf_count:
                avg_price = np.mean([c['price'] for c in cluster])
                confluence_zones.append({
                    'price': avg_price,
                    'timeframes': list(tfs),
                    'levels': cluster,
                    'strength': len(tfs),
                })

        # 강도순 정렬
        confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
        return confluence_zones


# ============================================================================
# Multi-TF Fib System
# ============================================================================

class MultiTFFibSystem:
    """
    다중 타임프레임 Fibonacci 시스템

    계층 구조:
    - Cycle (고정) → 1W → 1D → 4H → 1H → 15m
    - 각 레벨은 상위 레벨의 범위 내에서 Fib 계산
    """

    def __init__(
        self,
        zigzag_params: Optional[Dict[str, ZigZagParams]] = None,
        symbol: str = "BTC",
    ):
        """
        Args:
            zigzag_params: TF별 ZigZag 파라미터 (None이면 기본값)
            symbol: 심볼
        """
        self.params = zigzag_params or DEFAULT_ZIGZAG_PARAMS.copy()
        self.symbol = symbol
        self.hierarchy: Optional[FibHierarchy] = None

    def build_hierarchy(
        self,
        dataframes: Dict[str, pd.DataFrame],
        cycle_anchor: Optional[CycleAnchor] = None,
    ) -> FibHierarchy:
        """
        계층적 Fib 구조 생성

        Args:
            dataframes: {"1D": df_daily, "4H": df_4h, ...}
            cycle_anchor: 사이클 앵커 (None이면 BTC 기본값)

        Returns:
            FibHierarchy
        """
        if cycle_anchor is None:
            cycle_anchor = get_btc_cycle_anchor()

        hierarchy = FibHierarchy(cycle=cycle_anchor)

        # 상위 TF부터 순서대로 처리
        tf_order = ['1W', '1D', '4H', '1H', '15m']
        parent_low = cycle_anchor.cycle_low
        parent_high = cycle_anchor.cycle_high

        for tf in tf_order:
            if tf not in dataframes:
                continue

            df = dataframes[tf]
            if len(df) < 50:
                continue

            params = self.params.get(tf, DEFAULT_ZIGZAG_PARAMS['1D'])

            # ZigZag 실행
            pivots, _ = zigzag_pivots(
                df['close'].values,
                df['high'].values,
                df['low'].values,
                **params.to_dict(),
            )

            # 최신 스윙 추출
            swing = get_latest_swing(df, pivots)

            if swing:
                # 상위 범위 내로 클램핑
                fib_low = max(swing.start_price, parent_low) if swing.direction == 'up' else max(swing.end_price, parent_low)
                fib_high = min(swing.end_price, parent_high) if swing.direction == 'up' else min(swing.start_price, parent_high)

                # 저점이 고점보다 크면 스왑
                if fib_low > fib_high:
                    fib_low, fib_high = fib_high, fib_low

                level = TFFibLevel(
                    timeframe=tf,
                    fib_low=fib_low,
                    fib_high=fib_high,
                    swing=swing,
                    swing_direction=swing.direction,
                )
                hierarchy.levels[tf] = level

                # 다음 TF의 부모 범위 업데이트
                parent_low = fib_low
                parent_high = fib_high

        self.hierarchy = hierarchy
        return hierarchy

    def get_positions(self, price: float) -> Dict[str, float]:
        """모든 TF에서의 현재가 위치"""
        if self.hierarchy is None:
            raise ValueError("build_hierarchy()를 먼저 호출하세요")
        return self.hierarchy.get_position_all(price)

    def find_confluence_zones(
        self,
        tolerance: float = 0.02,
        min_tf_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """Confluence 존 찾기"""
        if self.hierarchy is None:
            raise ValueError("build_hierarchy()를 먼저 호출하세요")
        return self.hierarchy.find_confluence_zones(tolerance, min_tf_count)

    def get_all_fib_levels(self) -> Dict[str, Dict[str, float]]:
        """모든 TF의 Fib 레벨"""
        if self.hierarchy is None:
            raise ValueError("build_hierarchy()를 먼저 호출하세요")

        result = {
            'cycle': self.hierarchy.cycle.get_all_fib_levels(),
        }
        for tf, level in self.hierarchy.levels.items():
            result[tf] = level.get_fib_levels()

        return result


# ============================================================================
# ZigZag Parameter Optimizer
# ============================================================================

class ZigZagOptimizer:
    """
    ZigZag 파라미터 최적화

    목표:
    - 의미 있는 스윙만 감지 (노이즈 제거)
    - TF별 최적 파라미터 찾기
    - 사이클 범위 내에서 안정적인 피벗 생성
    """

    def __init__(self, cycle_anchor: Optional[CycleAnchor] = None):
        self.cycle_anchor = cycle_anchor or get_btc_cycle_anchor()

    def optimize_for_tf(
        self,
        df: pd.DataFrame,
        tf: str,
        target_pivots_per_period: int = 10,
    ) -> ZigZagParams:
        """
        특정 TF에 대한 ZigZag 파라미터 최적화

        Args:
            df: OHLCV DataFrame
            tf: 타임프레임
            target_pivots_per_period: 기간당 목표 피벗 수

        Returns:
            최적화된 ZigZagParams
        """
        prices = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 가격 범위를 사이클로 정규화
        cycle_range = self.cycle_anchor.cycle_high - self.cycle_anchor.cycle_low
        price_range = np.max(high) - np.min(low)
        range_ratio = price_range / cycle_range

        # TF별 기본값에서 시작
        base = DEFAULT_ZIGZAG_PARAMS.get(tf, DEFAULT_ZIGZAG_PARAMS['1D'])

        # 파라미터 후보군 생성
        up_pct_candidates = [base.up_pct * m for m in [0.5, 0.75, 1.0, 1.25, 1.5]]
        atr_mult_candidates = [base.atr_mult * m for m in [0.75, 1.0, 1.25]]

        best_params = base
        best_score = float('inf')

        for up_pct in up_pct_candidates:
            for atr_mult in atr_mult_candidates:
                params = ZigZagParams(
                    up_pct=up_pct,
                    down_pct=up_pct,  # 대칭
                    atr_mult=atr_mult,
                    min_bars=base.min_bars,
                    min_swing_atr=base.min_swing_atr,
                )

                pivots, _ = zigzag_pivots(
                    prices, high, low,
                    **params.to_dict(),
                )

                pivot_count = np.sum(pivots != 0)
                expected = len(df) / (len(df) / target_pivots_per_period)

                # 목표 피벗 수와의 차이
                score = abs(pivot_count - target_pivots_per_period)

                # 피벗이 너무 적거나 많으면 패널티
                if pivot_count < 3:
                    score += 100
                if pivot_count > target_pivots_per_period * 3:
                    score += 50

                if score < best_score:
                    best_score = score
                    best_params = params

        return best_params

    def optimize_all_tfs(
        self,
        dataframes: Dict[str, pd.DataFrame],
    ) -> Dict[str, ZigZagParams]:
        """
        모든 TF에 대한 ZigZag 파라미터 최적화

        Args:
            dataframes: {"1D": df, "4H": df, ...}

        Returns:
            {"1D": ZigZagParams, ...}
        """
        # TF별 목표 피벗 수
        target_pivots = {
            '1W': 5,   # 주봉은 적게
            '1D': 8,
            '4H': 12,
            '1H': 15,
            '15m': 20,
        }

        optimized = {}
        for tf, df in dataframes.items():
            if len(df) < 50:
                continue
            target = target_pivots.get(tf, 10)
            optimized[tf] = self.optimize_for_tf(df, tf, target)

        return optimized


# ============================================================================
# Convenience Functions
# ============================================================================

def build_multi_tf_fib(
    dataframes: Dict[str, pd.DataFrame],
    symbol: str = "BTC",
) -> FibHierarchy:
    """
    간편 함수: 다중 TF Fib 계층 생성

    사용법:
    ```python
    hierarchy = build_multi_tf_fib({
        "1D": df_daily,
        "4H": df_4h,
        "1H": df_1h,
    })

    # 모든 TF에서의 현재 위치
    positions = hierarchy.get_position_all(92000)

    # Confluence 존
    zones = hierarchy.find_confluence_zones()
    ```
    """
    system = MultiTFFibSystem(symbol=symbol)
    return system.build_hierarchy(dataframes)


def find_fib_confluence(
    dataframes: Dict[str, pd.DataFrame],
    tolerance: float = 0.02,
    min_tf_count: int = 2,
) -> List[Dict[str, Any]]:
    """
    간편 함수: Fib Confluence 존 찾기

    Returns:
        [{"price": 85000, "timeframes": ["cycle", "1D", "4H"], "strength": 3}]
    """
    system = MultiTFFibSystem()
    system.build_hierarchy(dataframes)
    return system.find_confluence_zones(tolerance, min_tf_count)