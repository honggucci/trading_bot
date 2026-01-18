"""
Tests for Probability Gate Module (v2)
======================================

7 필수 테스트:
1. test_prob_range - p_bull must be in [0, 1]
2. test_monotonic_score - Higher score => higher p_bull (T fixed)
3. test_temperature_effect - Higher T => p closer to 0.5
4. test_clipping_and_ema_stability - No NaNs after warmup
5. test_gate_action_thresholds - LONG/SHORT/FLAT based on thresholds
6. test_temperature_bounds - T must stay within [T_min, T_max]
7. test_vol_spike_increases_T - Volatility spike → T increase
"""
import numpy as np
import pytest

from src.regime.prob_gate import (
    ProbabilityGate,
    ProbGateConfig,
    compute_atr_pct,
    rolling_zscore,
    sigmoid_stable,
    compute_temperature_vol,
    prob_from_score,
    gate_action,
    spearman_ic,
    eval_ic,
    eval_brier,
    eval_calibration,
    create_simple_direction_score,
    apply_probability_calibration,
    DynamicThresholdConfig,
    compute_dynamic_thresholds,
    gate_action_dynamic,
)


def generate_synthetic_ohlc(n: int = 500, seed: int = 42) -> dict:
    """
    테스트용 합성 OHLC 데이터 생성

    Returns:
        dict with 'open', 'high', 'low', 'close' arrays
    """
    np.random.seed(seed)

    # Random walk for close
    returns = np.random.randn(n) * 0.01  # 1% daily vol
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    open_ = close * (1 + np.random.randn(n) * 0.003)

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    }


class TestProbRange:
    """Test 1: p_bull must be in [0, 1] for all valid points"""

    def test_prob_range_vol_mode(self):
        """Option A: Volatility-based temperature"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        cfg = ProbGateConfig(temp_mode='vol')
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        valid_mask = result['valid'].values
        p_bull = result['p_bull'].values[valid_mask]

        assert len(p_bull) > 0, "Should have valid points after warmup"
        assert np.all(p_bull >= 0.0), "p_bull must be >= 0"
        assert np.all(p_bull <= 1.0), "p_bull must be <= 1"

    def test_prob_range_fixed_mode(self):
        """Baseline: Fixed temperature"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        cfg = ProbGateConfig(temp_mode='fixed', T_fixed=1.5)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        valid_mask = result['valid'].values
        p_bull = result['p_bull'].values[valid_mask]

        assert np.all(p_bull >= 0.0), "p_bull must be >= 0"
        assert np.all(p_bull <= 1.0), "p_bull must be <= 1"

    def test_prob_range_instability_mode(self):
        """Option B: Instability-based temperature"""
        data = generate_synthetic_ohlc(600)  # Need more data for instability warmup
        score_raw = create_simple_direction_score(data['close'])

        cfg = ProbGateConfig(temp_mode='instability')
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        valid_mask = result['valid'].values
        p_bull = result['p_bull'].values[valid_mask]

        if len(p_bull) > 0:  # May have no valid points with small data
            assert np.all(p_bull >= 0.0), "p_bull must be >= 0"
            assert np.all(p_bull <= 1.0), "p_bull must be <= 1"


class TestMonotonicScore:
    """Test 2: Higher score => higher p_bull (T fixed)"""

    def test_monotonic_with_fixed_T(self):
        """With fixed T, higher score_norm should give higher p_bull"""
        T = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        score_norm = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        p_bull = prob_from_score(score_norm, T)

        # Check strictly increasing
        for i in range(1, len(p_bull)):
            assert p_bull[i] > p_bull[i-1], f"p_bull should increase with score: p[{i}]={p_bull[i]}, p[{i-1}]={p_bull[i-1]}"

    def test_monotonic_extreme_scores(self):
        """Extreme scores should give extreme probabilities"""
        T = np.array([1.0, 1.0])
        score_high = np.array([10.0])
        score_low = np.array([-10.0])

        p_high = prob_from_score(score_high, T[:1])
        p_low = prob_from_score(score_low, T[:1])

        assert p_high[0] > 0.99, f"Very high score should give p > 0.99, got {p_high[0]}"
        assert p_low[0] < 0.01, f"Very low score should give p < 0.01, got {p_low[0]}"


class TestTemperatureEffect:
    """Test 3: Higher T => p closer to 0.5 for same score"""

    def test_higher_T_more_uncertain(self):
        """Higher temperature should push probability toward 0.5"""
        score_norm = np.array([2.0, 2.0, 2.0])
        T_low = np.array([0.5])
        T_mid = np.array([1.5])
        T_high = np.array([3.0])

        p_low = prob_from_score(score_norm[:1], T_low)
        p_mid = prob_from_score(score_norm[:1], T_mid)
        p_high = prob_from_score(score_norm[:1], T_high)

        # All should be > 0.5 (positive score)
        assert p_low[0] > 0.5
        assert p_mid[0] > 0.5
        assert p_high[0] > 0.5

        # Higher T should be closer to 0.5
        dist_low = abs(p_low[0] - 0.5)
        dist_mid = abs(p_mid[0] - 0.5)
        dist_high = abs(p_high[0] - 0.5)

        assert dist_low > dist_mid > dist_high, \
            f"Higher T should bring p closer to 0.5: {dist_low:.4f} > {dist_mid:.4f} > {dist_high:.4f}"

    def test_zero_score_gives_half(self):
        """Score = 0 should give p = 0.5 regardless of T"""
        score_zero = np.array([0.0, 0.0, 0.0])
        T_values = np.array([0.5, 1.5, 3.0])

        for T_val in T_values:
            p = prob_from_score(np.array([0.0]), np.array([T_val]))
            assert abs(p[0] - 0.5) < 0.001, f"score=0 should give p=0.5, got {p[0]} with T={T_val}"


class TestClippingAndEmaStability:
    """Test 4: No NaNs after warmup, limited jumps"""

    def test_no_nans_after_warmup(self):
        """After warmup period, there should be no NaN values"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        cfg = ProbGateConfig(temp_mode='vol', vol_window=192)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        # After warmup (vol_window + some buffer), should have valid values
        warmup = cfg.vol_window + 50
        T_after = result['T'].values[warmup:]
        p_after = result['p_bull'].values[warmup:]
        score_after = result['score_norm'].values[warmup:]

        assert not np.any(np.isnan(T_after)), "T should not have NaN after warmup"
        assert not np.any(np.isnan(p_after)), "p_bull should not have NaN after warmup"
        assert not np.any(np.isnan(score_after)), "score_norm should not have NaN after warmup"

    def test_ema_limits_jumps(self):
        """EMA smoothing should limit sudden jumps in temperature"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        cfg = ProbGateConfig(temp_mode='vol', T_ema_span=12)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        valid_mask = result['valid'].values
        T = result['T'].values[valid_mask]

        if len(T) > 1:
            # Calculate absolute differences
            T_diff = np.abs(np.diff(T))
            max_jump = np.max(T_diff)

            # With EMA smoothing, jumps should be limited
            # T range is [0.7, 3.0], so max possible range is 2.3
            # EMA with span=12 should smooth significantly
            assert max_jump < 1.0, f"EMA should limit T jumps, max jump was {max_jump}"


class TestGateActionThresholds:
    """Test 5: LONG/SHORT/FLAT based on thresholds"""

    def test_long_threshold(self):
        """p_bull > thr_long => LONG"""
        p_bull = np.array([0.6, 0.7, 0.8])
        thr_long = 0.55
        thr_short = 0.55

        action_code, action_str = gate_action(p_bull, thr_long, thr_short)

        assert np.all(action_code == 1), "All should be LONG"
        assert np.all(action_str == 'LONG'), "All should be 'LONG'"

    def test_short_threshold(self):
        """p_bull < (1 - thr_short) => SHORT"""
        p_bull = np.array([0.4, 0.3, 0.2])
        thr_long = 0.55
        thr_short = 0.55  # 1 - 0.55 = 0.45, so p < 0.45 is SHORT

        action_code, action_str = gate_action(p_bull, thr_long, thr_short)

        assert np.all(action_code == -1), "All should be SHORT"
        assert np.all(action_str == 'SHORT'), "All should be 'SHORT'"

    def test_flat_zone(self):
        """p_bull in [1-thr_short, thr_long] => FLAT"""
        p_bull = np.array([0.48, 0.50, 0.52])  # In between
        thr_long = 0.55
        thr_short = 0.55

        action_code, action_str = gate_action(p_bull, thr_long, thr_short)

        assert np.all(action_code == 0), "All should be FLAT"
        assert np.all(action_str == 'FLAT'), "All should be 'FLAT'"

    def test_mixed_actions(self):
        """Mixed p_bull values should give corresponding actions"""
        p_bull = np.array([0.3, 0.5, 0.7])  # SHORT, FLAT, LONG
        thr_long = 0.55
        thr_short = 0.55

        action_code, action_str = gate_action(p_bull, thr_long, thr_short)

        assert action_code[0] == -1 and action_str[0] == 'SHORT'
        assert action_code[1] == 0 and action_str[1] == 'FLAT'
        assert action_code[2] == 1 and action_str[2] == 'LONG'


class TestTemperatureBounds:
    """Test 6: T must stay within [T_min, T_max] after warmup"""

    def test_temperature_within_bounds_vol_mode(self):
        """Option A: T must be in [T_min, T_max]"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        T_min, T_max = 0.7, 3.0
        cfg = ProbGateConfig(temp_mode='vol', T_min=T_min, T_max=T_max, vol_window=192)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        # After warmup
        warmup = cfg.vol_window
        T = result['T'].values[warmup:]
        T_valid = T[np.isfinite(T)]

        if len(T_valid) > 0:
            assert np.nanmin(T_valid) >= T_min - 1e-9, \
                f"T should be >= T_min ({T_min}), got min={np.nanmin(T_valid)}"
            assert np.nanmax(T_valid) <= T_max + 1e-9, \
                f"T should be <= T_max ({T_max}), got max={np.nanmax(T_valid)}"

    def test_temperature_within_bounds_instability_mode(self):
        """Option B: T must be in [T_min, T_max]"""
        data = generate_synthetic_ohlc(800)
        score_raw = create_simple_direction_score(data['close'])

        T_min, T_max = 0.5, 2.5
        cfg = ProbGateConfig(
            temp_mode='instability',
            T_min=T_min,
            T_max=T_max,
            instability_window=96,
            instability_ref_window=384
        )
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        # After instability warmup (needs ref_window)
        warmup = cfg.instability_ref_window + cfg.score_window
        T = result['T'].values[warmup:]
        T_valid = T[np.isfinite(T)]

        if len(T_valid) > 0:
            assert np.nanmin(T_valid) >= T_min - 1e-9, \
                f"T should be >= T_min ({T_min}), got min={np.nanmin(T_valid)}"
            assert np.nanmax(T_valid) <= T_max + 1e-9, \
                f"T should be <= T_max ({T_max}), got max={np.nanmax(T_valid)}"

    def test_custom_bounds(self):
        """Custom T bounds should be respected"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        T_min, T_max = 1.0, 2.0  # Narrower bounds
        cfg = ProbGateConfig(temp_mode='vol', T_min=T_min, T_max=T_max)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, data['close'], data['high'], data['low'])

        valid_mask = result['valid'].values
        T = result['T'].values[valid_mask]

        if len(T) > 0:
            assert np.min(T) >= T_min - 1e-9
            assert np.max(T) <= T_max + 1e-9


class TestVolSpikeIncreasesT:
    """Test 7: Volatility spike → T increase (Option A 핵심 테스트)"""

    def test_vol_spike_increases_temperature(self):
        """
        인위적으로 vol spike 구간 만들어서 T 상승 검증
        이거 없으면 "온도 효과 없음" 버그 못 잡음
        """
        np.random.seed(123)
        n = 600

        # Create base price with normal volatility
        returns_normal = np.random.randn(n) * 0.005  # Low vol
        close = 100 * np.exp(np.cumsum(returns_normal))
        high = close * (1 + np.abs(np.random.randn(n)) * 0.003)
        low = close * (1 - np.abs(np.random.randn(n)) * 0.003)

        # Inject vol spike in middle section (bars 300-400)
        spike_start, spike_end = 300, 400
        spike_multiplier = 5.0  # 5x volatility

        for i in range(spike_start, spike_end):
            spike_return = np.random.randn() * 0.005 * spike_multiplier
            close[i] = close[i-1] * np.exp(spike_return)
            high[i] = close[i] * (1 + np.abs(np.random.randn()) * 0.003 * spike_multiplier)
            low[i] = close[i] * (1 - np.abs(np.random.randn()) * 0.003 * spike_multiplier)

        score_raw = create_simple_direction_score(close)

        cfg = ProbGateConfig(temp_mode='vol', vol_window=50, T_ema_span=6)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, close, high, low)

        T = result['T'].values

        # Compare average T before spike vs during spike
        # Need to account for warmup and ATR lag
        warmup = cfg.vol_window + cfg.n_atr
        pre_spike_region = slice(warmup, spike_start - 20)
        spike_region = slice(spike_start + 30, spike_end - 10)

        T_pre = T[pre_spike_region]
        T_spike = T[spike_region]

        T_pre_valid = T_pre[np.isfinite(T_pre)]
        T_spike_valid = T_spike[np.isfinite(T_spike)]

        if len(T_pre_valid) > 10 and len(T_spike_valid) > 10:
            mean_T_pre = np.mean(T_pre_valid)
            mean_T_spike = np.mean(T_spike_valid)

            assert mean_T_spike > mean_T_pre, \
                f"Vol spike should increase T: pre={mean_T_pre:.3f}, spike={mean_T_spike:.3f}"

    def test_low_vol_decreases_temperature(self):
        """Low volatility period should have lower temperature"""
        np.random.seed(456)
        n = 600

        # Create base price with higher volatility
        returns_high = np.random.randn(n) * 0.02  # High vol
        close = 100 * np.exp(np.cumsum(returns_high))
        high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
        low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

        # Inject low vol section (bars 300-400)
        calm_start, calm_end = 300, 400
        calm_multiplier = 0.2  # 20% of normal vol

        for i in range(calm_start, calm_end):
            calm_return = np.random.randn() * 0.02 * calm_multiplier
            close[i] = close[i-1] * np.exp(calm_return)
            high[i] = close[i] * (1 + np.abs(np.random.randn()) * 0.01 * calm_multiplier)
            low[i] = close[i] * (1 - np.abs(np.random.randn()) * 0.01 * calm_multiplier)

        score_raw = create_simple_direction_score(close)

        cfg = ProbGateConfig(temp_mode='vol', vol_window=50, T_ema_span=6)
        gate = ProbabilityGate(cfg)
        result = gate.compute(score_raw, close, high, low)

        T = result['T'].values

        warmup = cfg.vol_window + cfg.n_atr
        pre_calm_region = slice(warmup, calm_start - 20)
        calm_region = slice(calm_start + 30, calm_end - 10)

        T_pre = T[pre_calm_region]
        T_calm = T[calm_region]

        T_pre_valid = T_pre[np.isfinite(T_pre)]
        T_calm_valid = T_calm[np.isfinite(T_calm)]

        if len(T_pre_valid) > 10 and len(T_calm_valid) > 10:
            mean_T_pre = np.mean(T_pre_valid)
            mean_T_calm = np.mean(T_calm_valid)

            assert mean_T_calm < mean_T_pre, \
                f"Low vol should decrease T: pre={mean_T_pre:.3f}, calm={mean_T_calm:.3f}"


class TestUtilityFunctions:
    """Additional tests for utility functions"""

    def test_sigmoid_stable_no_overflow(self):
        """sigmoid_stable should handle extreme values without overflow"""
        extreme_values = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
        result = sigmoid_stable(extreme_values)

        assert not np.any(np.isnan(result)), "sigmoid should not produce NaN"
        assert not np.any(np.isinf(result)), "sigmoid should not produce Inf"
        assert np.all(result >= 0.0) and np.all(result <= 1.0), "sigmoid must be in [0, 1]"

    def test_rolling_zscore_warmup(self):
        """rolling_zscore should return NaN during warmup"""
        x = np.arange(100).astype(float)
        window = 20

        z = rolling_zscore(x, window=window)

        # First (window-1) values should be NaN
        assert np.all(np.isnan(z[:window-1])), "Warmup period should be NaN"
        # After warmup should be valid
        assert np.all(np.isfinite(z[window:])), "After warmup should be finite"

    def test_spearman_ic_calculation(self):
        """spearman_ic should match expected correlation"""
        # Perfect positive rank correlation
        x = np.array([1, 2, 3, 4, 5]).astype(float)
        y = np.array([10, 20, 30, 40, 50]).astype(float)

        ic = spearman_ic(x, y)
        assert abs(ic - 1.0) < 0.01, f"Perfect correlation should be ~1.0, got {ic}"

        # Perfect negative rank correlation
        y_neg = np.array([50, 40, 30, 20, 10]).astype(float)
        ic_neg = spearman_ic(x, y_neg)
        assert abs(ic_neg - (-1.0)) < 0.01, f"Perfect negative should be ~-1.0, got {ic_neg}"

    def test_atr_pct_positive(self):
        """ATR% should always be positive"""
        data = generate_synthetic_ohlc(200)
        atr_pct = compute_atr_pct(data['high'], data['low'], data['close'])

        assert np.all(atr_pct >= 0), "ATR% must be non-negative"
        assert np.all(np.isfinite(atr_pct)), "ATR% should not have NaN/Inf"


class TestEvaluationFunctions:
    """Tests for evaluation utilities"""

    def test_eval_brier_range(self):
        """Brier score should be in [0, 1]"""
        p_bull = np.array([0.2, 0.5, 0.8, 0.3, 0.7])
        actual_up = np.array([0, 1, 1, 0, 1])

        brier = eval_brier(p_bull, actual_up)
        assert 0 <= brier <= 1, f"Brier score should be in [0, 1], got {brier}"

    def test_eval_brier_perfect(self):
        """Perfect predictions should give Brier = 0"""
        p_bull = np.array([0.0, 1.0, 0.0, 1.0])
        actual_up = np.array([0, 1, 0, 1])

        brier = eval_brier(p_bull, actual_up)
        assert brier < 0.01, f"Perfect predictions should give Brier ~0, got {brier}"

    def test_eval_calibration_structure(self):
        """Calibration result should have expected structure"""
        np.random.seed(789)
        p_bull = np.random.rand(100)
        actual_up = (np.random.rand(100) > 0.5).astype(float)

        cal = eval_calibration(p_bull, actual_up, n_bins=10)

        assert 'bins' in cal, "Should have 'bins' key"
        assert 'ece' in cal, "Should have 'ece' key"
        assert len(cal['bins']) == 10, "Should have 10 bins"
        assert 0 <= cal['ece'] <= 1, f"ECE should be in [0, 1], got {cal['ece']}"

    def test_eval_ic_horizons(self):
        """eval_ic should return IC for all horizons"""
        np.random.seed(101)
        n = 200
        p_bull = np.random.rand(n)
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        ic_results = eval_ic(p_bull, close, horizons=[1, 4, 16])

        assert 'IC@1bar' in ic_results
        assert 'IC@4bar' in ic_results
        assert 'IC@16bar' in ic_results

        for key, ic in ic_results.items():
            if np.isfinite(ic):
                assert -1 <= ic <= 1, f"{key} should be in [-1, 1], got {ic}"


class TestProbabilityCalibration:
    """PR3: post-hoc probability calibration behavior"""

    def test_calibration_shrink_moves_toward_half_and_preserves_order(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        p2 = apply_probability_calibration(p, shrink=0.6, bias=0.0, clip_eps=0.0)

        # Moves toward 0.5 (reduces confidence)
        assert np.all(np.abs(p2 - 0.5) <= np.abs(p - 0.5) + 1e-12)

        # Preserves order (monotonic transform)
        assert np.all(np.diff(p2) > 0)

    def test_calibration_invalid_shrink_raises(self):
        p = np.array([0.2, 0.8])
        with pytest.raises(ValueError):
            _ = apply_probability_calibration(p, shrink=0.0)


class TestDynamicThreshold:
    """Tests for Dynamic Threshold (레짐 기반 임계값 조절)"""

    def test_bull_regime_favors_long(self):
        """Bull 레짐 (p_bull > 0.5): LONG 관대, SHORT 엄격"""
        p_bull = np.array([0.7, 0.7, 0.7])  # Bull regime
        T = np.array([1.0, 1.0, 1.0])  # T=1.0 → u=0 (no uncertainty effect)

        cfg = DynamicThresholdConfig(
            base_long=0.58,
            base_short=0.60,
            conf_favor_delta=0.03,
            conf_against_delta=0.04,
            uncertainty_delta=0.0,  # Disable for this test
            ema_span=1  # No smoothing for test
        )
        thr_long, thr_short = compute_dynamic_thresholds(p_bull, T, cfg)

        # Bull regime: LONG should be more lenient (lower), SHORT stricter (higher)
        assert thr_long[-1] < cfg.base_long, \
            f"Bull regime: LONG thr should decrease, got {thr_long[-1]} vs base {cfg.base_long}"
        assert thr_short[-1] > cfg.base_short, \
            f"Bull regime: SHORT thr should increase, got {thr_short[-1]} vs base {cfg.base_short}"

    def test_bear_regime_favors_short(self):
        """Bear 레짐 (p_bull < 0.5): SHORT 관대, LONG 엄격"""
        p_bull = np.array([0.3, 0.3, 0.3])  # Bear regime
        T = np.array([1.0, 1.0, 1.0])  # T=1.0 → u=0 (no uncertainty effect)

        cfg = DynamicThresholdConfig(
            base_long=0.58,
            base_short=0.60,
            conf_favor_delta=0.03,
            conf_against_delta=0.04,
            uncertainty_delta=0.0,  # Disable for this test
            ema_span=1
        )
        thr_long, thr_short = compute_dynamic_thresholds(p_bull, T, cfg)

        # Bear regime: LONG should be stricter (higher), SHORT more lenient (lower)
        assert thr_long[-1] > cfg.base_long, \
            f"Bear regime: LONG thr should increase, got {thr_long[-1]} vs base {cfg.base_long}"
        assert thr_short[-1] < cfg.base_short, \
            f"Bear regime: SHORT thr should decrease, got {thr_short[-1]} vs base {cfg.base_short}"

    def test_high_uncertainty_tightens_both(self):
        """불확실성(T↑) 높으면 양쪽 다 엄격해짐"""
        p_bull = np.array([0.7, 0.7])  # Bull regime
        T_low = np.array([1.0, 1.0])   # Low uncertainty (T=1)
        T_high = np.array([2.5, 2.5])  # High uncertainty (T=2.5)

        cfg = DynamicThresholdConfig(uncertainty_delta=0.04, ema_span=1)

        thr_long_low, thr_short_low = compute_dynamic_thresholds(p_bull, T_low, cfg)
        thr_long_high, thr_short_high = compute_dynamic_thresholds(p_bull, T_high, cfg)

        # High T should make both thresholds higher (stricter)
        assert thr_long_high[-1] > thr_long_low[-1], \
            "High uncertainty should tighten LONG threshold"
        assert thr_short_high[-1] > thr_short_low[-1], \
            "High uncertainty should tighten SHORT threshold"

    def test_threshold_bounds(self):
        """Threshold는 [thr_min, thr_max] 범위 내"""
        n = 100
        p_bull = np.random.rand(n)  # Random regime
        T = np.random.uniform(0.7, 3.0, n)

        cfg = DynamicThresholdConfig(thr_min=0.55, thr_max=0.70, ema_span=3)
        thr_long, thr_short = compute_dynamic_thresholds(p_bull, T, cfg)

        valid_long = thr_long[np.isfinite(thr_long)]
        valid_short = thr_short[np.isfinite(thr_short)]

        assert np.all(valid_long >= cfg.thr_min - 1e-9), \
            f"thr_long min={np.min(valid_long)}, expected >= {cfg.thr_min}"
        assert np.all(valid_long <= cfg.thr_max + 1e-9), \
            f"thr_long max={np.max(valid_long)}, expected <= {cfg.thr_max}"
        assert np.all(valid_short >= cfg.thr_min - 1e-9), \
            f"thr_short min={np.min(valid_short)}, expected >= {cfg.thr_min}"
        assert np.all(valid_short <= cfg.thr_max + 1e-9), \
            f"thr_short max={np.max(valid_short)}, expected <= {cfg.thr_max}"

    def test_hysteresis_smooths_jumps(self):
        """EMA smoothing으로 threshold의 갑작스런 변화 완화"""
        n = 50
        # Oscillating regime
        p_bull = np.array([0.7 if i % 2 == 0 else 0.3 for i in range(n)]).astype(float)
        T = np.full(n, 1.5)

        # No smoothing
        cfg_no_ema = DynamicThresholdConfig(ema_span=1)
        thr_long_raw, _ = compute_dynamic_thresholds(p_bull, T, cfg_no_ema)

        # With smoothing
        cfg_ema = DynamicThresholdConfig(ema_span=5)
        thr_long_smooth, _ = compute_dynamic_thresholds(p_bull, T, cfg_ema)

        # Calculate variance of changes
        diff_raw = np.abs(np.diff(thr_long_raw[10:]))
        diff_smooth = np.abs(np.diff(thr_long_smooth[10:]))

        assert np.mean(diff_smooth) < np.mean(diff_raw), \
            "EMA should reduce threshold oscillation"

    def test_gate_action_dynamic_applies_per_bar(self):
        """Dynamic threshold가 각 bar별로 적용되는지 확인"""
        p_bull = np.array([0.60, 0.60, 0.60])

        # Different thresholds per bar
        thr_long = np.array([0.55, 0.58, 0.65])  # First passes, second passes, third fails
        thr_short = np.array([0.55, 0.55, 0.55])

        action_code, action_str = gate_action_dynamic(p_bull, thr_long, thr_short)

        assert action_code[0] == 1, "p=0.60 > thr=0.55 should be LONG"
        assert action_code[1] == 1, "p=0.60 > thr=0.58 should be LONG"
        assert action_code[2] == 0, "p=0.60 < thr=0.65 should be FLAT"

    def test_integration_with_probability_gate(self):
        """ProbabilityGate에서 use_dynamic_threshold=True 통합 테스트"""
        data = generate_synthetic_ohlc(500)
        score_raw = create_simple_direction_score(data['close'])

        # Static threshold
        cfg_static = ProbGateConfig(temp_mode='vol', use_dynamic_threshold=False)
        gate_static = ProbabilityGate(cfg_static)
        result_static = gate_static.compute(score_raw, data['close'], data['high'], data['low'])

        # Dynamic threshold
        cfg_dynamic = ProbGateConfig(
            temp_mode='vol',
            use_dynamic_threshold=True,
            dynamic_thr_cfg=DynamicThresholdConfig()
        )
        gate_dynamic = ProbabilityGate(cfg_dynamic)
        result_dynamic = gate_dynamic.compute(score_raw, data['close'], data['high'], data['low'])

        # Both should have required columns
        assert 'thr_long' in result_static.columns
        assert 'thr_short' in result_static.columns
        assert 'thr_long' in result_dynamic.columns
        assert 'thr_short' in result_dynamic.columns

        # Static should have constant thresholds
        valid_static = result_static[result_static['valid']]
        if len(valid_static) > 0:
            assert np.allclose(valid_static['thr_long'].values, cfg_static.thr_long), \
                "Static should have constant thr_long"

        # Dynamic should have varying thresholds
        valid_dynamic = result_dynamic[result_dynamic['valid']]
        if len(valid_dynamic) > 10:
            thr_long_std = valid_dynamic['thr_long'].std()
            assert thr_long_std > 0, "Dynamic should have varying thr_long"
