"""
Multi-TF Regime Aggregator 테스트
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unittest
import pandas as pd
from src.regime.multi_tf_regime import (
    RegimeConfig,
    RegimeState,
    MultiTFRegimeAggregator,
    zz_to_prior,
    shrink_to_half,
    compute_uncertainty,
    compute_regime_score,
    update_regime_with_hysteresis,
    can_long_in_regime,
)
from src.regime.regime_strategy import (
    REGIME_PARAMS,
    get_regime_params,
    get_stoch_rsi_threshold,
    get_risk_mult,
    get_tp_mode,
    check_entry_conditions,
    get_tp_config,
)


class TestZZToPrior(unittest.TestCase):
    """ZigZag → Prior 변환 테스트"""

    def setUp(self):
        self.config = RegimeConfig()

    def test_1w_up(self):
        """1W up → 0.75"""
        p = zz_to_prior("up", "1w", self.config)
        self.assertEqual(p, 0.75)

    def test_1w_down(self):
        """1W down → 0.25"""
        p = zz_to_prior("down", "1w", self.config)
        self.assertEqual(p, 0.25)

    def test_1d_up(self):
        """1D up → 0.65"""
        p = zz_to_prior("up", "1d", self.config)
        self.assertEqual(p, 0.65)

    def test_1d_down(self):
        """1D down → 0.35"""
        p = zz_to_prior("down", "1d", self.config)
        self.assertEqual(p, 0.35)

    def test_unknown(self):
        """unknown → 0.50"""
        p = zz_to_prior("unknown", "1w", self.config)
        self.assertEqual(p, 0.50)


class TestShrinkToHalf(unittest.TestCase):
    """Shrink 함수 테스트"""

    def test_no_uncertainty(self):
        """u=0 → 수축 없음"""
        p = shrink_to_half(0.80, 0.0)
        self.assertAlmostEqual(p, 0.80, places=4)

    def test_half_uncertainty(self):
        """u=0.5 → 절반 수축"""
        p = shrink_to_half(0.80, 0.5)
        self.assertAlmostEqual(p, 0.65, places=4)

    def test_full_uncertainty(self):
        """u=1.0 → 0.5로 수축"""
        p = shrink_to_half(0.80, 1.0)
        self.assertAlmostEqual(p, 0.50, places=4)

    def test_below_half(self):
        """p < 0.5 도 정상 수축"""
        p = shrink_to_half(0.20, 0.5)
        self.assertAlmostEqual(p, 0.35, places=4)


class TestComputeUncertainty(unittest.TestCase):
    """Uncertainty 계산 테스트"""

    def test_T_equals_ref(self):
        """T=1.0 → u=0"""
        u = compute_uncertainty(1.0, T_ref=1.0)
        self.assertAlmostEqual(u, 0.0, places=4)

    def test_T_above_ref(self):
        """T=1.5 → u=0.5"""
        u = compute_uncertainty(1.5, T_ref=1.0, T_range=1.0)
        self.assertAlmostEqual(u, 0.5, places=4)

    def test_T_max(self):
        """T=2.0 → u=1.0 (capped)"""
        u = compute_uncertainty(2.0, T_ref=1.0, T_range=1.0)
        self.assertAlmostEqual(u, 1.0, places=4)


class TestHysteresis(unittest.TestCase):
    """Hysteresis 테스트"""

    def setUp(self):
        self.config = RegimeConfig()

    def test_bull_stays_bull(self):
        """Bull에서 score=0.60 → Bull 유지"""
        regime = update_regime_with_hysteresis("BULL", 0.60, self.config)
        self.assertEqual(regime, "BULL")

    def test_bull_to_range(self):
        """Bull에서 score=0.50 → Range로 전환"""
        regime = update_regime_with_hysteresis("BULL", 0.50, self.config)
        self.assertEqual(regime, "RANGE")

    def test_bull_to_bear(self):
        """Bull에서 score=0.30 → Bear로 전환"""
        regime = update_regime_with_hysteresis("BULL", 0.30, self.config)
        self.assertEqual(regime, "BEAR")

    def test_bear_stays_bear(self):
        """Bear에서 score=0.40 → Bear 유지"""
        regime = update_regime_with_hysteresis("BEAR", 0.40, self.config)
        self.assertEqual(regime, "BEAR")

    def test_bear_to_range(self):
        """Bear에서 score=0.50 → Range로 전환"""
        regime = update_regime_with_hysteresis("BEAR", 0.50, self.config)
        self.assertEqual(regime, "RANGE")

    def test_range_to_bull(self):
        """Range에서 score=0.70 → Bull로 전환"""
        regime = update_regime_with_hysteresis("RANGE", 0.70, self.config)
        self.assertEqual(regime, "BULL")

    def test_range_to_bear(self):
        """Range에서 score=0.30 → Bear로 전환"""
        regime = update_regime_with_hysteresis("RANGE", 0.30, self.config)
        self.assertEqual(regime, "BEAR")

    def test_range_stays_range(self):
        """Range에서 score=0.50 → Range 유지"""
        regime = update_regime_with_hysteresis("RANGE", 0.50, self.config)
        self.assertEqual(regime, "RANGE")


class TestAggregator(unittest.TestCase):
    """Aggregator 통합 테스트"""

    def setUp(self):
        self.aggregator = MultiTFRegimeAggregator()

    def test_all_bullish(self):
        """모든 신호 강세 → BULL"""
        state = self.aggregator.update(
            zz_1w_direction="up",
            zz_1d_direction="up",
            pg_4h_p_bull=0.80,
            pg_4h_T=1.0,
            pg_1h_p_bull=0.75,
            pg_1h_T=1.0,
        )
        self.assertEqual(state.regime, "BULL")
        self.assertGreater(state.score, 0.65)

    def test_all_bearish(self):
        """모든 신호 약세 → BEAR"""
        state = self.aggregator.update(
            zz_1w_direction="down",
            zz_1d_direction="down",
            pg_4h_p_bull=0.20,
            pg_4h_T=1.0,
            pg_1h_p_bull=0.25,
            pg_1h_T=1.0,
        )
        self.assertEqual(state.regime, "BEAR")
        self.assertLess(state.score, 0.35)

    def test_mixed_signals(self):
        """혼합 신호 → RANGE"""
        state = self.aggregator.update(
            zz_1w_direction="up",
            zz_1d_direction="down",
            pg_4h_p_bull=0.55,
            pg_4h_T=1.5,
            pg_1h_p_bull=0.45,
            pg_1h_T=1.5,
        )
        # 혼합이면 보통 RANGE
        self.assertIn(state.regime, ["RANGE", "BULL"])

    def test_hysteresis_transition(self):
        """연속 업데이트 시 hysteresis 동작"""
        # 초기: 강세
        self.aggregator.update(
            zz_1w_direction="up", zz_1d_direction="up",
            pg_4h_p_bull=0.75, pg_4h_T=1.0,
            pg_1h_p_bull=0.70, pg_1h_T=1.0,
        )
        self.assertEqual(self.aggregator.get_regime(), "BULL")

        # 약간 하락해도 BULL 유지 (hysteresis)
        self.aggregator.update(
            zz_1w_direction="up", zz_1d_direction="up",
            pg_4h_p_bull=0.55, pg_4h_T=1.0,
            pg_1h_p_bull=0.55, pg_1h_T=1.0,
        )
        self.assertEqual(self.aggregator.get_regime(), "BULL")


class TestRegimeStrategy(unittest.TestCase):
    """레짐 전략 파라미터 테스트"""

    def test_bull_params(self):
        """BULL 파라미터"""
        params = get_regime_params("BULL")
        self.assertEqual(params["stoch_rsi_threshold"], 30.0)
        self.assertEqual(params["risk_mult"], 1.0)
        self.assertEqual(params["tp_mode"], "trailing")
        self.assertFalse(params["require_reclaim"])

    def test_range_params(self):
        """RANGE 파라미터 (BEAR와 동일 - 보수적)"""
        params = get_regime_params("RANGE")
        self.assertEqual(params["stoch_rsi_threshold"], 20.0)
        self.assertEqual(params["risk_mult"], 0.3)
        self.assertEqual(params["tp_mode"], "quick_exit")
        self.assertTrue(params["require_reclaim"])

    def test_bear_params(self):
        """BEAR 파라미터"""
        params = get_regime_params("BEAR")
        self.assertEqual(params["stoch_rsi_threshold"], 20.0)
        self.assertEqual(params["risk_mult"], 0.3)
        self.assertTrue(params["require_reclaim"])


class TestEntryConditions(unittest.TestCase):
    """Entry 조건 체크 테스트"""

    def test_bull_entry_ok(self):
        """BULL에서 StochRSI=25 → 허용"""
        result = check_entry_conditions("BULL", stoch_rsi=25.0)
        self.assertTrue(result.overall_ok)

    def test_bull_entry_fail(self):
        """BULL에서 StochRSI=35 → 불허"""
        result = check_entry_conditions("BULL", stoch_rsi=35.0)
        self.assertFalse(result.overall_ok)

    def test_bear_entry_without_reclaim(self):
        """BEAR에서 StochRSI=8, reclaim 없음 → 불허"""
        result = check_entry_conditions("BEAR", stoch_rsi=8.0, reclaim_confirmed=False)
        self.assertFalse(result.overall_ok)
        self.assertIn("reclaim", result.reason.lower())

    def test_bear_entry_with_reclaim(self):
        """BEAR에서 StochRSI=8, reclaim 있음 → 허용"""
        result = check_entry_conditions("BEAR", stoch_rsi=8.0, reclaim_confirmed=True)
        self.assertTrue(result.overall_ok)


class TestCanLongInRegime(unittest.TestCase):
    """롱 허용 여부 테스트"""

    def test_bull_long_allowed(self):
        """BULL에서 StochRSI=25 → 허용"""
        allowed, reason = can_long_in_regime("BULL", stoch_rsi=25.0)
        self.assertTrue(allowed)

    def test_bear_long_conditional(self):
        """BEAR에서 reclaim 없음 → 불허"""
        allowed, reason = can_long_in_regime("BEAR", stoch_rsi=8.0, reclaim_confirmed=False)
        self.assertFalse(allowed)

    def test_bear_long_with_reclaim(self):
        """BEAR에서 reclaim 있음 → 허용"""
        allowed, reason = can_long_in_regime("BEAR", stoch_rsi=8.0, reclaim_confirmed=True)
        self.assertTrue(allowed)


def run_tests():
    """테스트 실행"""
    print("=" * 60)
    print("Multi-TF Regime Aggregator Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 테스트 클래스 추가
    suite.addTests(loader.loadTestsFromTestCase(TestZZToPrior))
    suite.addTests(loader.loadTestsFromTestCase(TestShrinkToHalf))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeUncertainty))
    suite.addTests(loader.loadTestsFromTestCase(TestHysteresis))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestRegimeStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestEntryConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestCanLongInRegime))

    # 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
