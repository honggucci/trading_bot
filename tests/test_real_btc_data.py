# -*- coding: utf-8 -*-
"""
Real BTC Data Test - Multi-TF Fib System
=========================================

Binance에서 실제 BTC 데이터를 가져와서 테스트.
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
    get_current_cycle_position,
    BTC_CYCLES,
)
from src.context.multi_tf_fib import (
    MultiTFFibSystem,
    ZigZagOptimizer,
    DEFAULT_ZIGZAG_PARAMS,
    find_fib_confluence,
)

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("[WARNING] ccxt not installed. Run: pip install ccxt")


def fetch_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 500) -> pd.DataFrame:
    """Binance에서 OHLCV 데이터 가져오기"""
    if not HAS_CCXT:
        raise ImportError("ccxt required")

    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df


def test_current_btc_position():
    """현재 BTC 가격의 사이클 위치 테스트"""
    print("\n" + "=" * 60)
    print("Real BTC Data Test - Current Position")
    print("=" * 60)

    if not HAS_CCXT:
        print("[SKIP] ccxt not installed")
        return False

    # 현재가 가져오기
    exchange = ccxt.binance()
    ticker = exchange.fetch_ticker("BTC/USDT")
    current_price = ticker['last']

    print(f"\nCurrent BTC Price: ${current_price:,.2f}")
    print(f"24h Change: {ticker['percentage']:.2f}%")

    # 사이클 위치 분석
    anchor = get_btc_cycle_anchor()
    position = anchor.get_position(current_price)
    zone = anchor.get_zone(current_price)
    crash_support = anchor.get_crash_support()

    print(f"\n=== Cycle Analysis (Cycle {anchor.cycle_num}) ===")
    print(f"Cycle Low: ${anchor.cycle_low:,.0f}")
    print(f"Cycle High: ${anchor.cycle_high:,.0f}")
    print(f"Current Position: {position:.3f} ({position*100:.1f}%)")
    print(f"Zone: {zone}")
    print(f"Crash Support (Prev ATH): ${crash_support:,.0f}")

    # Fib 레벨별 가격
    print(f"\n=== Fib Levels ===")
    fib_levels = anchor.get_all_fib_levels()
    for level, price in fib_levels.items():
        marker = " <-- CURRENT" if abs(float(level) - position) < 0.03 else ""
        print(f"  {level}: ${price:,.0f}{marker}")

    # Wyckoff Zones (사이클 종료 후 대비)
    print(f"\n=== Wyckoff Zones (Post-Cycle) ===")
    zones = anchor.get_wyckoff_zones()
    for zone_name, (low, high) in zones.items():
        print(f"  {zone_name}: ${low:,.0f} - ${high:,.0f}")

    return True


def test_multi_tf_confluence():
    """Multi-TF Confluence 테스트"""
    print("\n" + "=" * 60)
    print("Real BTC Data Test - Multi-TF Confluence")
    print("=" * 60)

    if not HAS_CCXT:
        print("[SKIP] ccxt not installed")
        return False

    print("\nFetching data from Binance...")

    # 여러 타임프레임 데이터 가져오기
    dataframes = {}

    tf_map = {
        '1W': ('1w', 100),
        '1D': ('1d', 365),
        '4H': ('4h', 500),
        '1H': ('1h', 500),
    }

    for tf, (binance_tf, limit) in tf_map.items():
        try:
            df = fetch_ohlcv("BTC/USDT", binance_tf, limit)
            dataframes[tf] = df
            print(f"  {tf}: {len(df)} candles ({df.index[0].date()} ~ {df.index[-1].date()})")
        except Exception as e:
            print(f"  {tf}: Failed - {e}")

    if len(dataframes) < 2:
        print("[FAIL] Not enough data")
        return False

    # Multi-TF Fib 시스템 구축
    print("\n=== Building Fib Hierarchy ===")
    system = MultiTFFibSystem()
    hierarchy = system.build_hierarchy(dataframes)

    print(f"Cycle: ${hierarchy.cycle.cycle_low:,.0f} - ${hierarchy.cycle.cycle_high:,.0f}")

    for tf, level in hierarchy.levels.items():
        print(f"{tf}: ${level.fib_low:,.0f} - ${level.fib_high:,.0f} ({level.swing_direction})")

    # 현재가 위치
    exchange = ccxt.binance()
    ticker = exchange.fetch_ticker("BTC/USDT")
    current_price = ticker['last']

    print(f"\n=== Current Price Position ===")
    print(f"Price: ${current_price:,.2f}")

    positions = system.get_positions(current_price)
    for tf, pos in positions.items():
        status = ""
        if 0.6 <= pos <= 0.65:
            status = " (Golden Pocket!)"
        elif 0.78 <= pos <= 0.8:
            status = " (0.786 level)"
        elif 0.5 <= pos <= 0.52:
            status = " (50% level)"
        print(f"  {tf}: {pos:.3f} ({pos*100:.1f}%){status}")

    # Confluence Zones 찾기
    print(f"\n=== Confluence Zones ===")
    zones = system.find_confluence_zones(tolerance=0.02, min_tf_count=2)

    if zones:
        print(f"Found {len(zones)} confluence zones:")
        for i, zone in enumerate(zones[:10]):
            distance = (zone['price'] - current_price) / current_price * 100
            direction = "above" if distance > 0 else "below"
            print(f"  {i+1}. ${zone['price']:,.0f} ({abs(distance):.1f}% {direction})")
            print(f"      TFs: {zone['timeframes']} (strength: {zone['strength']})")
    else:
        print("No confluence zones found")

    return True


def test_zigzag_on_real_data():
    """실제 데이터에서 ZigZag 최적화 테스트"""
    print("\n" + "=" * 60)
    print("Real BTC Data Test - ZigZag Optimization")
    print("=" * 60)

    if not HAS_CCXT:
        print("[SKIP] ccxt not installed")
        return False

    print("\nFetching data...")

    dataframes = {}
    tf_map = {
        '1D': ('1d', 365),
        '4H': ('4h', 500),
        '1H': ('1h', 500),
    }

    for tf, (binance_tf, limit) in tf_map.items():
        try:
            df = fetch_ohlcv("BTC/USDT", binance_tf, limit)
            dataframes[tf] = df
        except Exception as e:
            print(f"  {tf}: Failed - {e}")

    if not dataframes:
        print("[FAIL] No data")
        return False

    # ZigZag 최적화
    print("\n=== ZigZag Optimization ===")
    optimizer = ZigZagOptimizer()
    optimized = optimizer.optimize_all_tfs(dataframes)

    print("\nDefault vs Optimized:")
    for tf, params in optimized.items():
        default = DEFAULT_ZIGZAG_PARAMS.get(tf)
        if default:
            print(f"\n{tf}:")
            print(f"  up_pct:   {default.up_pct:.3f} -> {params.up_pct:.3f}")
            print(f"  atr_mult: {default.atr_mult:.1f} -> {params.atr_mult:.1f}")

    return True


def test_historical_validation():
    """과거 사이클 데이터 검증"""
    print("\n" + "=" * 60)
    print("Historical Cycle Validation")
    print("=" * 60)

    print("\n=== BTC Halving Cycles ===")

    for cycle_num, data in BTC_CYCLES.items():
        print(f"\nCycle {cycle_num}:")
        print(f"  Halving: {data.halving_date}")
        print(f"  Low: ${data.cycle_low:,.0f} ({data.cycle_low_date})")
        print(f"  High: ${data.cycle_high:,.0f} ({data.cycle_high_date})")
        print(f"  Multiplier: {data.multiplier:.1f}x")

        if cycle_num > 1:
            prev_high = BTC_CYCLES[cycle_num - 1].cycle_high
            crash = (prev_high - data.cycle_low) / prev_high * 100
            print(f"  Crash from prev ATH: {crash:.1f}%")

            # Fib 레벨 계산
            anchor = get_btc_cycle_anchor(cycle_num)
            levels = anchor.get_all_fib_levels()
            print(f"  Key Fib Levels:")
            for ratio in ['0.382', '0.500', '0.618']:
                print(f"    {ratio}: ${levels[ratio]:,.0f}")

    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Real BTC Data Integration Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # 1. 현재 BTC 위치
    try:
        results.append(("Current BTC Position", test_current_btc_position()))
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append(("Current BTC Position", False))

    # 2. Multi-TF Confluence
    try:
        results.append(("Multi-TF Confluence", test_multi_tf_confluence()))
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append(("Multi-TF Confluence", False))

    # 3. ZigZag 최적화
    try:
        results.append(("ZigZag Optimization", test_zigzag_on_real_data()))
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append(("ZigZag Optimization", False))

    # 4. 과거 사이클 검증
    try:
        results.append(("Historical Validation", test_historical_validation()))
    except Exception as e:
        print(f"[ERROR] {e}")
        results.append(("Historical Validation", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL/SKIP]"
        print(f"  {status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)