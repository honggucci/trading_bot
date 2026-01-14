# -*- coding: utf-8 -*-
"""
Timeframe Predictor - Hierarchical ZigZag Prediction
=====================================================

상위 TF 스윙으로 하위 TF 방향 예측.

핵심 원리:
1. 주봉 스윙 방향 = 전체 편향 (Bias)
2. 일봉 스윙 = 주봉 범위 내에서 움직임
3. 4H/1H/15m = 점진적으로 세부 움직임

예측 가능한 것:
- 스윙 방향 (상승/하락 편향)
- 예상 타겟 존 (Fib 레벨)
- Confluence 존 (반전 가능성 높은 구간)

예측 불가능한 것:
- 정확한 타이밍
- 정확한 가격
- 외부 충격 (뉴스, 블랙스완)

사용법:
```python
from src.context.tf_predictor import TFPredictor

predictor = TFPredictor()
predictor.load_data()  # Binance에서 데이터 로드

# 현재 상태 분석
state = predictor.analyze_current_state()

# 예측
prediction = predictor.predict_next_move()

# 트레이딩 신호
signal = predictor.get_trading_signal()
```
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd

from .cycle_anchor import get_btc_cycle_anchor, CycleAnchor
from .multi_tf_fib import (
    MultiTFFibSystem,
    FibHierarchy,
    TFFibLevel,
    ZigZagParams,
    DEFAULT_ZIGZAG_PARAMS,
)
from .zigzag import zigzag_pivots, get_latest_swing, SwingInfo


class SwingBias(Enum):
    """스윙 편향"""
    STRONG_BULLISH = "strong_bullish"   # 모든 TF 상승
    BULLISH = "bullish"                  # 다수 TF 상승
    NEUTRAL = "neutral"                  # 혼조
    BEARISH = "bearish"                  # 다수 TF 하락
    STRONG_BEARISH = "strong_bearish"   # 모든 TF 하락


class ZoneType(Enum):
    """존 타입"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    CONFLUENCE = "confluence"
    GOLDEN_POCKET = "golden_pocket"  # 0.618-0.65
    CYCLE_SUPPORT = "cycle_support"  # 전 사이클 고점


@dataclass
class PredictionZone:
    """예측 존"""
    price: float
    zone_type: ZoneType
    timeframes: List[str]
    strength: int  # 겹치는 TF 수
    distance_pct: float  # 현재가 대비 거리 (%)
    fib_ratio: Optional[float] = None


@dataclass
class SwingPrediction:
    """스윙 예측"""
    bias: SwingBias
    current_price: float

    # 상위 TF 스윙 정보
    weekly_swing: Optional[str] = None  # up/down
    daily_swing: Optional[str] = None
    h4_swing: Optional[str] = None

    # 예상 타겟
    upside_targets: List[PredictionZone] = field(default_factory=list)
    downside_targets: List[PredictionZone] = field(default_factory=list)

    # Confluence 존
    confluence_zones: List[PredictionZone] = field(default_factory=list)

    # 가장 가까운 지지/저항
    nearest_support: Optional[PredictionZone] = None
    nearest_resistance: Optional[PredictionZone] = None

    # 신뢰도
    confidence: float = 0.0


@dataclass
class TradingSignal:
    """트레이딩 신호"""
    action: str  # "long", "short", "wait"
    confidence: float
    entry_zone: Optional[PredictionZone] = None
    stop_loss: Optional[float] = None
    take_profit: List[float] = field(default_factory=list)
    reason: str = ""


class TFPredictor:
    """
    타임프레임 예측기

    상위 TF 스윙을 기반으로 하위 TF 방향 예측.
    """

    # TF 가중치 (상위 TF일수록 높음)
    TF_WEIGHTS = {
        'cycle': 5.0,
        '1W': 4.0,
        '1D': 3.0,
        '4H': 2.0,
        '1H': 1.5,
        '15m': 1.0,
    }

    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.fib_system = MultiTFFibSystem()
        self.hierarchy: Optional[FibHierarchy] = None
        self.current_price: float = 0.0

    def load_data(self, dataframes: Optional[Dict[str, pd.DataFrame]] = None):
        """
        데이터 로드

        Args:
            dataframes: 미리 로드된 데이터 (None이면 Binance에서 가져옴)
        """
        if dataframes:
            self.dataframes = dataframes
        else:
            self._fetch_from_binance()

        # Fib 계층 구조 빌드
        self.hierarchy = self.fib_system.build_hierarchy(self.dataframes)

        # 현재가 업데이트
        if '1H' in self.dataframes and len(self.dataframes['1H']) > 0:
            self.current_price = self.dataframes['1H']['close'].iloc[-1]
        elif '4H' in self.dataframes and len(self.dataframes['4H']) > 0:
            self.current_price = self.dataframes['4H']['close'].iloc[-1]

    def _fetch_from_binance(self):
        """Binance에서 데이터 가져오기"""
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt required: pip install ccxt")

        exchange = ccxt.binance({'enableRateLimit': True})

        tf_map = {
            '1W': ('1w', 100),
            '1D': ('1d', 365),
            '4H': ('4h', 500),
            '1H': ('1h', 500),
            '15m': ('15m', 500),
        }

        for tf, (binance_tf, limit) in tf_map.items():
            try:
                ohlcv = exchange.fetch_ohlcv(self.symbol, binance_tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self.dataframes[tf] = df
            except Exception as e:
                print(f"[WARN] Failed to fetch {tf}: {e}")

        # 현재가
        ticker = exchange.fetch_ticker(self.symbol)
        self.current_price = ticker['last']

    def analyze_current_state(self) -> Dict[str, Any]:
        """
        현재 상태 분석

        Returns:
            {
                "price": 95600,
                "cycle_position": 0.723,
                "swings": {"1W": "down", "1D": "down", "4H": "up"},
                "bias": "neutral",
                "positions": {"cycle": 0.72, "1W": 0.28, ...}
            }
        """
        if self.hierarchy is None:
            raise ValueError("load_data() 먼저 호출")

        # 스윙 방향
        swings = {}
        for tf, level in self.hierarchy.levels.items():
            swings[tf] = level.swing_direction

        # Bias 계산
        bias = self._calculate_bias(swings)

        # 모든 TF 위치
        positions = self.fib_system.get_positions(self.current_price)

        return {
            "price": self.current_price,
            "cycle_position": self.hierarchy.cycle.get_position(self.current_price),
            "swings": swings,
            "bias": bias.value,
            "positions": positions,
        }

    def _calculate_bias(self, swings: Dict[str, str]) -> SwingBias:
        """스윙 방향으로 편향 계산"""
        up_score = 0.0
        down_score = 0.0

        for tf, direction in swings.items():
            weight = self.TF_WEIGHTS.get(tf, 1.0)
            if direction == 'up':
                up_score += weight
            else:
                down_score += weight

        total = up_score + down_score
        if total == 0:
            return SwingBias.NEUTRAL

        ratio = up_score / total

        if ratio >= 0.8:
            return SwingBias.STRONG_BULLISH
        elif ratio >= 0.6:
            return SwingBias.BULLISH
        elif ratio >= 0.4:
            return SwingBias.NEUTRAL
        elif ratio >= 0.2:
            return SwingBias.BEARISH
        else:
            return SwingBias.STRONG_BEARISH

    def predict_next_move(self) -> SwingPrediction:
        """
        다음 움직임 예측

        Returns:
            SwingPrediction
        """
        if self.hierarchy is None:
            raise ValueError("load_data() 먼저 호출")

        state = self.analyze_current_state()

        # Confluence 존 찾기
        confluence_zones = self._find_prediction_zones()

        # 가장 가까운 지지/저항
        supports = [z for z in confluence_zones if z.distance_pct < 0]
        resistances = [z for z in confluence_zones if z.distance_pct > 0]

        nearest_support = min(supports, key=lambda x: abs(x.distance_pct)) if supports else None
        nearest_resistance = min(resistances, key=lambda x: abs(x.distance_pct)) if resistances else None

        # Upside/Downside 타겟
        upside_targets = sorted([z for z in resistances], key=lambda x: x.distance_pct)[:3]
        downside_targets = sorted([z for z in supports], key=lambda x: x.distance_pct, reverse=True)[:3]

        # 신뢰도 계산
        confidence = self._calculate_confidence(state, confluence_zones)

        return SwingPrediction(
            bias=SwingBias(state['bias']),
            current_price=self.current_price,
            weekly_swing=state['swings'].get('1W'),
            daily_swing=state['swings'].get('1D'),
            h4_swing=state['swings'].get('4H'),
            upside_targets=upside_targets,
            downside_targets=downside_targets,
            confluence_zones=[z for z in confluence_zones if z.strength >= 2],
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            confidence=confidence,
        )

    def _find_prediction_zones(self) -> List[PredictionZone]:
        """예측 존 찾기"""
        zones = []

        # 사이클 Fib 레벨
        cycle = self.hierarchy.cycle
        for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
            price = cycle.get_price_at_fib(ratio)
            distance = (price - self.current_price) / self.current_price * 100

            zone_type = ZoneType.SUPPORT if distance < 0 else ZoneType.RESISTANCE
            if 0.618 <= ratio <= 0.65:
                zone_type = ZoneType.GOLDEN_POCKET

            zones.append(PredictionZone(
                price=price,
                zone_type=zone_type,
                timeframes=['cycle'],
                strength=1,
                distance_pct=distance,
                fib_ratio=ratio,
            ))

        # 각 TF의 Fib 레벨
        for tf, level in self.hierarchy.levels.items():
            for ratio in [0.382, 0.5, 0.618, 0.786]:
                price = level.get_price_at_fib(ratio)
                distance = (price - self.current_price) / self.current_price * 100

                zone_type = ZoneType.SUPPORT if distance < 0 else ZoneType.RESISTANCE

                zones.append(PredictionZone(
                    price=price,
                    zone_type=zone_type,
                    timeframes=[tf],
                    strength=1,
                    distance_pct=distance,
                    fib_ratio=ratio,
                ))

        # Confluence 병합
        zones = self._merge_confluence_zones(zones)

        # 전 사이클 고점 (Crash Support)
        crash_support = cycle.get_crash_support()
        distance = (crash_support - self.current_price) / self.current_price * 100
        zones.append(PredictionZone(
            price=crash_support,
            zone_type=ZoneType.CYCLE_SUPPORT,
            timeframes=['cycle'],
            strength=3,  # 강력한 지지
            distance_pct=distance,
        ))

        return sorted(zones, key=lambda x: abs(x.distance_pct))

    def _merge_confluence_zones(
        self,
        zones: List[PredictionZone],
        tolerance: float = 0.02,
    ) -> List[PredictionZone]:
        """근접 존 병합 (Confluence)"""
        if not zones:
            return []

        merged = []
        used = set()

        for i, z1 in enumerate(zones):
            if i in used:
                continue

            cluster = [z1]
            tfs = set(z1.timeframes)

            for j, z2 in enumerate(zones):
                if j <= i or j in used:
                    continue

                # 가격 근접 체크
                if abs(z2.price - z1.price) / z1.price <= tolerance:
                    cluster.append(z2)
                    tfs.update(z2.timeframes)
                    used.add(j)

            if len(tfs) >= 2:
                # Confluence!
                avg_price = np.mean([z.price for z in cluster])
                distance = (avg_price - self.current_price) / self.current_price * 100

                merged.append(PredictionZone(
                    price=avg_price,
                    zone_type=ZoneType.CONFLUENCE,
                    timeframes=list(tfs),
                    strength=len(tfs),
                    distance_pct=distance,
                ))
            else:
                merged.append(z1)

        return merged

    def _calculate_confidence(
        self,
        state: Dict[str, Any],
        zones: List[PredictionZone],
    ) -> float:
        """신뢰도 계산"""
        confidence = 0.5  # 기본값

        # 1. 상위 TF 일치도
        swings = state['swings']
        up_count = sum(1 for d in swings.values() if d == 'up')
        down_count = len(swings) - up_count
        alignment = max(up_count, down_count) / len(swings) if swings else 0
        confidence += alignment * 0.2

        # 2. Confluence 강도
        strong_zones = [z for z in zones if z.strength >= 3]
        if strong_zones:
            confidence += 0.1

        # 3. 현재 위치가 Fib 레벨 근처
        cycle_pos = state['cycle_position']
        key_levels = [0.382, 0.5, 0.618, 0.786]
        near_level = any(abs(cycle_pos - lvl) < 0.03 for lvl in key_levels)
        if near_level:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_trading_signal(self) -> TradingSignal:
        """
        트레이딩 신호 생성

        Returns:
            TradingSignal
        """
        prediction = self.predict_next_move()

        # 기본값: 대기
        signal = TradingSignal(
            action="wait",
            confidence=prediction.confidence,
            reason="Waiting for setup",
        )

        # 1. 강한 Confluence 존 근처에서 반전 기대
        for zone in prediction.confluence_zones:
            if abs(zone.distance_pct) <= 2.0 and zone.strength >= 3:
                if zone.distance_pct < 0:
                    # 지지선 근처 → Long 고려
                    signal = TradingSignal(
                        action="long",
                        confidence=prediction.confidence,
                        entry_zone=zone,
                        stop_loss=zone.price * 0.97,  # 3% 아래
                        take_profit=[
                            prediction.nearest_resistance.price if prediction.nearest_resistance else zone.price * 1.05,
                        ],
                        reason=f"Near strong confluence support ({zone.strength} TFs)",
                    )
                else:
                    # 저항선 근처 → Short 고려
                    signal = TradingSignal(
                        action="short",
                        confidence=prediction.confidence,
                        entry_zone=zone,
                        stop_loss=zone.price * 1.03,  # 3% 위
                        take_profit=[
                            prediction.nearest_support.price if prediction.nearest_support else zone.price * 0.95,
                        ],
                        reason=f"Near strong confluence resistance ({zone.strength} TFs)",
                    )
                break

        # 2. Bias 기반 방향 결정
        if signal.action == "wait":
            if prediction.bias in [SwingBias.STRONG_BULLISH, SwingBias.BULLISH]:
                if prediction.nearest_support and abs(prediction.nearest_support.distance_pct) <= 5.0:
                    signal = TradingSignal(
                        action="long",
                        confidence=prediction.confidence * 0.8,
                        entry_zone=prediction.nearest_support,
                        stop_loss=prediction.nearest_support.price * 0.95,
                        take_profit=[t.price for t in prediction.upside_targets[:2]],
                        reason="Bullish bias, near support",
                    )
            elif prediction.bias in [SwingBias.STRONG_BEARISH, SwingBias.BEARISH]:
                if prediction.nearest_resistance and abs(prediction.nearest_resistance.distance_pct) <= 5.0:
                    signal = TradingSignal(
                        action="short",
                        confidence=prediction.confidence * 0.8,
                        entry_zone=prediction.nearest_resistance,
                        stop_loss=prediction.nearest_resistance.price * 1.05,
                        take_profit=[t.price for t in prediction.downside_targets[:2]],
                        reason="Bearish bias, near resistance",
                    )

        return signal

    def print_analysis(self):
        """분석 결과 출력"""
        state = self.analyze_current_state()
        prediction = self.predict_next_move()
        signal = self.get_trading_signal()

        print("=" * 60)
        print(f"TF Predictor Analysis - {self.symbol}")
        print("=" * 60)

        print(f"\nCurrent Price: ${self.current_price:,.2f}")
        print(f"Cycle Position: {state['cycle_position']:.1%}")
        print(f"Bias: {prediction.bias.value.upper()}")
        print(f"Confidence: {prediction.confidence:.1%}")

        print("\n=== Swing Directions ===")
        for tf, direction in state['swings'].items():
            arrow = "[UP]" if direction == 'up' else "[DN]"
            print(f"  {tf}: {arrow} {direction.upper()}")

        print("\n=== Key Levels ===")

        if prediction.nearest_resistance:
            r = prediction.nearest_resistance
            print(f"  Nearest Resistance: ${r.price:,.0f} ({r.distance_pct:+.1f}%)")
            if r.timeframes:
                print(f"    TFs: {r.timeframes} (strength: {r.strength})")

        if prediction.nearest_support:
            s = prediction.nearest_support
            print(f"  Nearest Support: ${s.price:,.0f} ({s.distance_pct:+.1f}%)")
            if s.timeframes:
                print(f"    TFs: {s.timeframes} (strength: {s.strength})")

        print("\n=== Confluence Zones ===")
        for i, zone in enumerate(prediction.confluence_zones[:5]):
            print(f"  {i+1}. ${zone.price:,.0f} ({zone.distance_pct:+.1f}%)")
            print(f"     TFs: {zone.timeframes} (strength: {zone.strength})")

        print("\n=== Upside Targets ===")
        for t in prediction.upside_targets[:3]:
            print(f"  ${t.price:,.0f} ({t.distance_pct:+.1f}%)")

        print("\n=== Downside Targets ===")
        for t in prediction.downside_targets[:3]:
            print(f"  ${t.price:,.0f} ({t.distance_pct:+.1f}%)")

        print("\n=== Trading Signal ===")
        print(f"  Action: {signal.action.upper()}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Reason: {signal.reason}")
        if signal.entry_zone:
            print(f"  Entry Zone: ${signal.entry_zone.price:,.0f}")
        if signal.stop_loss:
            print(f"  Stop Loss: ${signal.stop_loss:,.0f}")
        if signal.take_profit:
            print(f"  Take Profit: {[f'${p:,.0f}' for p in signal.take_profit]}")


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_btc_move() -> SwingPrediction:
    """BTC 움직임 예측 (편의 함수)"""
    predictor = TFPredictor("BTC/USDT")
    predictor.load_data()
    return predictor.predict_next_move()


def get_btc_signal() -> TradingSignal:
    """BTC 트레이딩 신호 (편의 함수)"""
    predictor = TFPredictor("BTC/USDT")
    predictor.load_data()
    return predictor.get_trading_signal()


def analyze_btc():
    """BTC 분석 출력 (편의 함수)"""
    predictor = TFPredictor("BTC/USDT")
    predictor.load_data()
    predictor.print_analysis()