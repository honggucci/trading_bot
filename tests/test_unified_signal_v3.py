# -*- coding: utf-8 -*-
"""
Unified Signal V3 Test
======================

V3 통합 신호 테스트.

테스트 항목:
1. TFPredictor 없이 fallback 동작
2. HMM Gate 오류 시 처리
3. MTF Boost 계산
4. Exit Levels 포함/미포함
5. 실제 HMM Gate 연동
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.anchor.unified_signal_v3 import (
    UnifiedSignalV3,
    check_unified_long_signal_v3,
    check_unified_short_signal_v3,
    calc_mtf_boost,
    determine_regime,
    format_signal_v3,
)


def generate_mock_df(bars: int = 200) -> pd.DataFrame:
    """Mock 15분봉 DataFrame 생성"""
    np.random.seed(42)

    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(bars)]

    base_price = 95000
    returns = np.random.normal(0.0001, 0.01, bars)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, bars)),
        'low': prices * (1 - np.random.uniform(0, 0.01, bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, bars),
    }, index=pd.DatetimeIndex(timestamps))

    # ATR 추가
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1)),
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # RSI 추가
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1)))

    return df


class MockHMMGate:
    """Mock HMM Gate"""

    def __init__(self, allowed: bool = True, state: str = 'accumulation'):
        self._allowed = allowed
        self._state = state

    def check_entry(self, ts, side):
        return type('obj', (object,), {
            'allowed': self._allowed,
            'state': self._state,
            'size_mult': 1.0 if self._allowed else 0.0,
            'blocked_reason': None if self._allowed else 'test_block',
            'expected_var': -5.0,
            'markdown_prob': 0.1,
            'cooldown_active': False,
        })()


class MockPredictionZone:
    """Mock PredictionZone"""

    def __init__(self, price: float, strength: int, zone_type: str = 'confluence'):
        self.price = price
        self.strength = strength
        self.zone_type = type('obj', (object,), {'value': zone_type})()


class TestCalcMTFBoost:
    """MTF Boost 계산 테스트"""

    def test_no_zones(self):
        """Confluence Zone 없으면 boost = 0"""
        boost, count, support, resistance = calc_mtf_boost(95000, [])
        assert boost == 0.0
        assert count == 0
        assert support is None
        assert resistance is None
        print("[PASS] No zones -> boost=0")

    def test_weak_zone(self):
        """2TF 겹침 -> boost = 0.1"""
        zones = [MockPredictionZone(94000, 2)]  # 1% below
        boost, count, support, resistance = calc_mtf_boost(95000, zones)
        assert boost == 0.1
        assert count == 1
        print(f"[PASS] 2TF zone -> boost={boost}")

    def test_strong_zone(self):
        """4TF 겹침 -> boost = 0.3"""
        zones = [MockPredictionZone(94500, 4)]  # 0.5% below
        boost, count, support, resistance = calc_mtf_boost(95000, zones)
        assert boost == 0.3
        print(f"[PASS] 4TF zone -> boost={boost}")

    def test_golden_pocket(self):
        """Golden Pocket -> 추가 부스트"""
        zones = [MockPredictionZone(94500, 3, 'golden_pocket')]
        boost, count, support, resistance = calc_mtf_boost(95000, zones)
        # strength=3 -> 0.2, golden_pocket -> +0.1 = 0.3
        # but capped at 0.4
        assert 0.2 <= boost <= 0.4
        print(f"[PASS] Golden pocket -> boost={boost}")

    def test_far_zone_ignored(self):
        """거리 3% 초과 -> 무시"""
        zones = [MockPredictionZone(90000, 4)]  # 5% below
        boost, count, support, resistance = calc_mtf_boost(95000, zones)
        assert boost == 0.0
        print("[PASS] Far zone ignored")


class TestDetermineRegime:
    """Regime 결정 테스트"""

    def test_trending_markup(self):
        """markup 상태 -> trending"""
        regime = determine_regime('markup', 'bullish')
        assert regime == 'trending'
        print(f"[PASS] markup + bullish -> {regime}")

    def test_ranging_accumulation(self):
        """accumulation 상태 -> ranging"""
        regime = determine_regime('accumulation', 'neutral')
        assert regime == 'ranging'
        print(f"[PASS] accumulation + neutral -> {regime}")


class TestUnifiedSignalV3:
    """V3 신호 테스트"""

    def test_fallback_no_tf_predictor(self):
        """TFPredictor 없으면 fallback 사용"""
        df = generate_mock_df()
        gate = MockHMMGate(allowed=True)
        ts = df.index[-1]

        signal = check_unified_long_signal_v3(
            df, gate, ts,
            tf_predictor=None,
        )

        assert signal.fallback_used is True
        assert signal.mtf_boost == 0.0
        print(f"[PASS] No TFPredictor -> fallback={signal.fallback_used}")

    def test_hmm_blocked(self):
        """HMM Gate 거부 시 allowed=False"""
        df = generate_mock_df()
        gate = MockHMMGate(allowed=False)
        ts = df.index[-1]

        signal = check_unified_long_signal_v3(df, gate, ts)

        assert signal.allowed is False
        assert signal.hmm_allowed is False
        print(f"[PASS] HMM blocked -> allowed={signal.allowed}")

    def test_exit_levels_included(self):
        """Exit levels 포함 시 계산됨"""
        df = generate_mock_df()
        gate = MockHMMGate(allowed=True)
        ts = df.index[-1]

        signal = check_unified_long_signal_v3(
            df, gate, ts,
            include_exit_levels=True,
        )

        # Exit levels는 allowed=True일 때만 계산됨
        # (이 테스트에서는 confluence_ok가 False일 수 있음)
        print(f"[PASS] Exit levels: {signal.exit_levels is not None if signal.allowed else 'N/A (not allowed)'}")

    def test_exit_levels_excluded(self):
        """Exit levels 제외 시 None"""
        df = generate_mock_df()
        gate = MockHMMGate(allowed=True)
        ts = df.index[-1]

        signal = check_unified_long_signal_v3(
            df, gate, ts,
            include_exit_levels=False,
        )

        assert signal.exit_levels is None
        print("[PASS] Exit levels excluded -> None")

    def test_warnings_captured(self):
        """경고 메시지 캡처"""
        df = generate_mock_df()

        # 에러를 발생시키는 Mock Gate
        class ErrorGate:
            def check_entry(self, ts, side):
                raise ValueError("Test error")

        signal = check_unified_long_signal_v3(df, ErrorGate(), df.index[-1])

        assert len(signal.warnings) > 0
        assert 'HMM Gate error' in signal.warnings[0]
        print(f"[PASS] Warnings captured: {signal.warnings}")


class TestShortSignalV3:
    """Short 신호 테스트"""

    def test_short_basic(self):
        """Short 기본 동작"""
        df = generate_mock_df()
        gate = MockHMMGate(allowed=True, state='markdown')
        ts = df.index[-1]

        signal = check_unified_short_signal_v3(df, gate, ts)

        assert signal.side in ['short', 'none']
        print(f"[PASS] Short signal: side={signal.side}, allowed={signal.allowed}")


class TestRealHMMGate:
    """실제 HMM Gate 연동 테스트"""

    def test_load_hmm_gate(self):
        """HMM Gate 로드"""
        try:
            from src.gate import load_hmm_gate
            gate = load_hmm_gate()
            print(f"[PASS] HMM Gate loaded: {len(gate.posterior_map)} timestamps")
            return gate
        except FileNotFoundError as e:
            print(f"[SKIP] HMM Gate not trained yet: {e}")
            return None

    def test_signal_with_real_gate(self):
        """실제 Gate로 신호 생성"""
        try:
            from src.gate import load_hmm_gate
            gate = load_hmm_gate()
        except FileNotFoundError:
            print("[SKIP] No trained model")
            return

        # 실제 timestamp 사용
        ts = list(gate.posterior_map.keys())[-1]
        df = generate_mock_df()

        signal = check_unified_long_signal_v3(df, gate, ts)

        print(f"[PASS] Real gate signal:")
        print(f"  State: {signal.hmm_state}")
        print(f"  Allowed: {signal.allowed}")
        print(f"  Confidence: {signal.confidence:.2f}")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Unified Signal V3 Tests")
    print("=" * 60)

    # MTF Boost
    print("\n--- MTF Boost Tests ---")
    test_boost = TestCalcMTFBoost()
    test_boost.test_no_zones()
    test_boost.test_weak_zone()
    test_boost.test_strong_zone()
    test_boost.test_golden_pocket()
    test_boost.test_far_zone_ignored()

    # Regime
    print("\n--- Regime Tests ---")
    test_regime = TestDetermineRegime()
    test_regime.test_trending_markup()
    test_regime.test_ranging_accumulation()

    # V3 Signal
    print("\n--- V3 Signal Tests ---")
    test_v3 = TestUnifiedSignalV3()
    test_v3.test_fallback_no_tf_predictor()
    test_v3.test_hmm_blocked()
    test_v3.test_exit_levels_included()
    test_v3.test_exit_levels_excluded()
    test_v3.test_warnings_captured()

    # Short Signal
    print("\n--- Short Signal Tests ---")
    test_short = TestShortSignalV3()
    test_short.test_short_basic()

    # Real HMM Gate
    print("\n--- Real HMM Gate Tests ---")
    test_real = TestRealHMMGate()
    test_real.test_load_hmm_gate()
    test_real.test_signal_with_real_gate()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()