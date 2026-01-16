#!/usr/bin/env python3
"""
12월 1일 하루 매매 기록 확인용 스크립트
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_strategy_compare import (
    Config, StrategyA, StrategyB, load_data
)

def main():
    print("=" * 70)
    print("12월 1일 매매 기록 확인")
    print("=" * 70)

    config = Config()

    # 2021년 11월 1일
    START = "2021-11-01"
    END = "2021-11-01"

    print(f"\n[기간] {START}")

    # 데이터 로딩
    print(f"\n데이터 로딩 중...")
    df_15m = load_data('15m', START, END, config)
    df_5m = load_data('5m', START, END, config)
    print(f"  15m: {len(df_15m)} bars")
    print(f"  5m: {len(df_5m)} bars")

    # 전략 A 실행
    print("\n" + "=" * 70)
    print("전략 A: 15m 진입 + 5m 청산")
    print("=" * 70)

    strategy_a = StrategyA(config)
    result_a = strategy_a.run(df_15m, df_5m)

    print(f"\n[전략 A 트레이드 목록]")
    print("-" * 90)
    print(f"{'#':>3} | {'방향':<6} | {'진입시간':<20} | {'진입가':>12} | {'청산가':>12} | {'청산사유':<15} | {'손익':>10}")
    print("-" * 90)

    for i, t in enumerate(result_a.trades, 1):
        direction = "LONG" if t.side == "long" else "SHORT"
        entry_time = t.entry_time.strftime("%Y-%m-%d %H:%M")
        pnl_str = f"+${t.pnl_usd:.2f}" if t.pnl_usd > 0 else f"-${abs(t.pnl_usd):.2f}"
        print(f"{i:>3} | {direction:<6} | {entry_time:<20} | ${t.entry_price:>10,.0f} | ${t.exit_price:>10,.0f} | {t.exit_reason:<15} | {pnl_str:>10}")

    print("-" * 90)
    total_a = sum(t.pnl_usd for t in result_a.trades)
    print(f"{'합계':>3} | {'':6} | {'':<20} | {'':<12} | {'':<12} | {'':<15} | ${total_a:>9.2f}")

    # 전략 B 실행
    print("\n" + "=" * 70)
    print("전략 B: Fib 레벨 기반")
    print("=" * 70)

    strategy_b = StrategyB(config)
    result_b = strategy_b.run(df_15m, df_5m)

    print(f"\n[전략 B 트레이드 목록]")
    print("-" * 90)
    print(f"{'#':>3} | {'방향':<6} | {'진입시간':<20} | {'진입가':>12} | {'청산가':>12} | {'청산사유':<15} | {'손익':>10}")
    print("-" * 90)

    for i, t in enumerate(result_b.trades, 1):
        direction = "LONG" if t.side == "long" else "SHORT"
        entry_time = t.entry_time.strftime("%Y-%m-%d %H:%M")
        pnl_str = f"+${t.pnl_usd:.2f}" if t.pnl_usd > 0 else f"-${abs(t.pnl_usd):.2f}"
        print(f"{i:>3} | {direction:<6} | {entry_time:<20} | ${t.entry_price:>10,.0f} | ${t.exit_price:>10,.0f} | {t.exit_reason:<15} | {pnl_str:>10}")

    print("-" * 90)
    total_b = sum(t.pnl_usd for t in result_b.trades)
    print(f"{'합계':>3} | {'':6} | {'':<20} | {'':<12} | {'':<12} | {'':<15} | ${total_b:>9.2f}")

if __name__ == "__main__":
    main()
