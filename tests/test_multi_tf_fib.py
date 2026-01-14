# -*- coding: utf-8 -*-
"""
Multi-TF Fib System Integration Test
====================================

Test targets:
1. CycleAnchor - Cycle-based Fib anchor
2. MultiTFFibSystem - Hierarchical Fib system
3. ZigZagOptimizer - TF-specific parameter optimization
4. Confluence Zone detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test target modules
from src.context.cycle_anchor import (
    get_btc_cycle_anchor,
    get_fib_levels,
    get_current_cycle_position,
    BTC_CYCLES,
    CycleAnchor,
)
from src.context.multi_tf_fib import (
    MultiTFFibSystem,
    ZigZagOptimizer,
    DEFAULT_ZIGZAG_PARAMS,
    build_multi_tf_fib,
    find_fib_confluence,
)


def generate_mock_ohlcv(
    start_price: float = 50000,
    n_bars: int = 500,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate mock OHLCV data for testing"""
    np.random.seed(seed)

    closes = [start_price]
    for _ in range(n_bars - 1):
        change = np.random.randn() * volatility + trend
        closes.append(closes[-1] * (1 + change))

    closes = np.array(closes)

    # Generate OHLC
    highs = closes * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    lows = closes * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    volumes = np.random.randint(100, 10000, n_bars).astype(float)

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1h')

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    }, index=dates)


class TestCycleAnchor:
    """CycleAnchor Tests"""

    def test_btc_cycles_data(self):
        """Validate BTC cycle data"""
        print("\n=== BTC Cycles Data ===")

        for cycle_num, data in BTC_CYCLES.items():
            crash_pct = 1 - (data.cycle_low / BTC_CYCLES.get(cycle_num - 1, data).cycle_high) if cycle_num > 1 else 0
            print(f"Cycle {cycle_num}:")
            print(f"  Low: ${data.cycle_low:,.0f} ({data.cycle_low_date})")
            print(f"  High: ${data.cycle_high:,.0f} ({data.cycle_high_date})")
            print(f"  Multiplier: {data.multiplier:.1f}x")
            if cycle_num > 1:
                prev_high = BTC_CYCLES[cycle_num - 1].cycle_high
                actual_crash = (prev_high - data.cycle_low) / prev_high * 100
                print(f"  Crash from prev high: {actual_crash:.1f}%")

        # Validate cycle theory: 70-90% crash
        for cycle_num in [2, 3, 4]:
            prev_high = BTC_CYCLES[cycle_num - 1].cycle_high
            curr_low = BTC_CYCLES[cycle_num].cycle_low
            crash_pct = (prev_high - curr_low) / prev_high * 100
            assert 70 <= crash_pct <= 90, f"Cycle {cycle_num} crash {crash_pct:.1f}% outside 70-90% range"

        print("\n[PASS] Cycle theory validated (70-90% crash)")
        return True

    def test_cycle_anchor_creation(self):
        """Test CycleAnchor creation"""
        print("\n=== Cycle Anchor Creation ===")

        anchor = get_btc_cycle_anchor()

        print(f"Current Cycle: {anchor.cycle_num}")
        print(f"Cycle Low: ${anchor.cycle_low:,.0f}")
        print(f"Cycle High: ${anchor.cycle_high:,.0f}")
        print(f"Prev Cycle High (Crash Support): ${anchor.prev_cycle_high:,.0f}")

        assert anchor.cycle_num == 4
        assert anchor.cycle_low == 15500
        assert anchor.prev_cycle_high == 69000

        print("\n[PASS] CycleAnchor creation test passed")
        return True

    def test_fib_position(self):
        """Test Fib position calculation"""
        print("\n=== Fib Position Test ===")

        anchor = get_btc_cycle_anchor()

        test_prices = [15500, 50000, 69000, 92000, 126296]

        for price in test_prices:
            position = anchor.get_position(price)
            zone = anchor.get_zone(price)
            print(f"${price:,.0f}: Position = {position:.3f} ({position*100:.1f}%), Zone = {zone}")

        # Low = 0, High = 1
        assert anchor.get_position(anchor.cycle_low) == 0.0
        assert anchor.get_position(anchor.cycle_high) == 1.0

        print("\n[PASS] Fib position calculation test passed")
        return True

    def test_crash_support(self):
        """Test crash support level"""
        print("\n=== Crash Support Test ===")

        anchor = get_btc_cycle_anchor()

        crash_support = anchor.get_crash_support()
        expected_crash_low = anchor.get_expected_crash_low(0.75)

        print(f"Crash Support (Prev ATH): ${crash_support:,.0f}")
        print(f"Expected Crash Low (75%): ${expected_crash_low:,.0f}")

        assert crash_support == 69000

        print("\n[PASS] Crash support test passed")
        return True

    def test_wyckoff_zones(self):
        """Test Wyckoff zones"""
        print("\n=== Wyckoff Zones Test ===")

        anchor = get_btc_cycle_anchor()
        zones = anchor.get_wyckoff_zones()

        for zone_name, (low, high) in zones.items():
            print(f"{zone_name}: ${low:,.0f} - ${high:,.0f}")

        print("\n[PASS] Wyckoff zones test passed")
        return True


class TestMultiTFFib:
    """MultiTFFibSystem Tests"""

    def test_default_zigzag_params(self):
        """Test default ZigZag parameters"""
        print("\n=== Default ZigZag Params ===")

        for tf, params in DEFAULT_ZIGZAG_PARAMS.items():
            print(f"{tf}: up_pct={params.up_pct:.2%}, atr_mult={params.atr_mult}, min_bars={params.min_bars}")

        # Higher TF should detect larger movements
        assert DEFAULT_ZIGZAG_PARAMS['1W'].up_pct > DEFAULT_ZIGZAG_PARAMS['1D'].up_pct
        assert DEFAULT_ZIGZAG_PARAMS['1D'].up_pct > DEFAULT_ZIGZAG_PARAMS['4H'].up_pct

        print("\n[PASS] ZigZag parameter hierarchy verified")
        return True

    def test_build_hierarchy(self):
        """Test Fib hierarchy building"""
        print("\n=== Build Fib Hierarchy ===")

        # Generate mock data
        dataframes = {
            '1D': generate_mock_ohlcv(start_price=50000, n_bars=365, volatility=0.03, trend=0.001),
            '4H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.015, trend=0.0005),
            '1H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.008, trend=0.0002),
        }

        system = MultiTFFibSystem()
        hierarchy = system.build_hierarchy(dataframes)

        print(f"Cycle: ${hierarchy.cycle.cycle_low:,.0f} - ${hierarchy.cycle.cycle_high:,.0f}")

        for tf, level in hierarchy.levels.items():
            print(f"{tf}: ${level.fib_low:,.0f} - ${level.fib_high:,.0f} ({level.swing_direction})")

        print("\n[PASS] Fib hierarchy build test passed")
        return True

    def test_confluence_zones(self):
        """Test confluence zone detection"""
        print("\n=== Confluence Zones ===")

        dataframes = {
            '1D': generate_mock_ohlcv(start_price=50000, n_bars=365, volatility=0.03, trend=0.001, seed=42),
            '4H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.015, trend=0.0005, seed=43),
            '1H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.008, trend=0.0002, seed=44),
        }

        zones = find_fib_confluence(dataframes, tolerance=0.03, min_tf_count=2)

        print(f"Found {len(zones)} confluence zones:")
        for i, zone in enumerate(zones[:5]):  # Top 5 only
            print(f"  {i+1}. ${zone['price']:,.0f} - TFs: {zone['timeframes']} (strength: {zone['strength']})")

        print("\n[PASS] Confluence zone detection test passed")
        return True

    def test_position_all_tfs(self):
        """Test position calculation across all TFs"""
        print("\n=== Position All TFs ===")

        dataframes = {
            '1D': generate_mock_ohlcv(start_price=50000, n_bars=365, volatility=0.03, trend=0.001),
            '4H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.015, trend=0.0005),
        }

        system = MultiTFFibSystem()
        system.build_hierarchy(dataframes)

        test_price = 75000
        positions = system.get_positions(test_price)

        print(f"Price: ${test_price:,}")
        for tf, pos in positions.items():
            print(f"  {tf}: {pos:.3f} ({pos*100:.1f}%)")

        print("\n[PASS] All TF position calculation test passed")
        return True


class TestZigZagOptimizer:
    """ZigZag Optimizer Tests"""

    def test_optimize_single_tf(self):
        """Test single TF optimization"""
        print("\n=== ZigZag Optimizer (Single TF) ===")

        df = generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.02)

        optimizer = ZigZagOptimizer()
        optimized = optimizer.optimize_for_tf(df, '4H', target_pivots_per_period=10)

        default = DEFAULT_ZIGZAG_PARAMS['4H']

        print(f"Default: up_pct={default.up_pct:.3f}, atr_mult={default.atr_mult}")
        print(f"Optimized: up_pct={optimized.up_pct:.3f}, atr_mult={optimized.atr_mult}")

        print("\n[PASS] Single TF optimization test passed")
        return True

    def test_optimize_all_tfs(self):
        """Test all TF optimization"""
        print("\n=== ZigZag Optimizer (All TFs) ===")

        dataframes = {
            '1D': generate_mock_ohlcv(start_price=50000, n_bars=365, volatility=0.03),
            '4H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.015),
            '1H': generate_mock_ohlcv(start_price=50000, n_bars=500, volatility=0.008),
        }

        optimizer = ZigZagOptimizer()
        optimized = optimizer.optimize_all_tfs(dataframes)

        for tf, params in optimized.items():
            default = DEFAULT_ZIGZAG_PARAMS.get(tf)
            if default:
                print(f"{tf}: up_pct {default.up_pct:.3f} -> {params.up_pct:.3f}")

        print("\n[PASS] All TF optimization test passed")
        return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Multi-TF Fib System Integration Test")
    print("=" * 60)

    results = []

    # CycleAnchor tests
    anchor_tests = TestCycleAnchor()
    results.append(("BTC Cycles Data", anchor_tests.test_btc_cycles_data()))
    results.append(("Cycle Anchor Creation", anchor_tests.test_cycle_anchor_creation()))
    results.append(("Fib Position", anchor_tests.test_fib_position()))
    results.append(("Crash Support", anchor_tests.test_crash_support()))
    results.append(("Wyckoff Zones", anchor_tests.test_wyckoff_zones()))

    # MultiTFFib tests
    mtf_tests = TestMultiTFFib()
    results.append(("Default ZigZag Params", mtf_tests.test_default_zigzag_params()))
    results.append(("Build Hierarchy", mtf_tests.test_build_hierarchy()))
    results.append(("Confluence Zones", mtf_tests.test_confluence_zones()))
    results.append(("Position All TFs", mtf_tests.test_position_all_tfs()))

    # ZigZag Optimizer tests
    opt_tests = TestZigZagOptimizer()
    results.append(("Optimize Single TF", opt_tests.test_optimize_single_tf()))
    results.append(("Optimize All TFs", opt_tests.test_optimize_all_tfs()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)