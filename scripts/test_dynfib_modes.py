#!/usr/bin/env python3
"""
PR-DYN-FIB Test: MODE43 vs MODE44 vs MODE45 vs MODE46 (4-Way Comparison)

비교 대상:
- MODE43: Champion (Log Fib + ATR tolerance + RR limit entry)
- MODE44: + ZigZag Dynamic Fib
- MODE45: + Rolling Dynamic Fib
- MODE46: + Conditional Dynamic Fib

테스트 구간 (고정 4구간):
- 2021-Q1: BULL market
- 2021-Nov: BULL → BEAR transition
- 2022-May: BEAR market
- 2023-Jul: RANGE market

리포트 지표:
- trades, WR, total_pnl, $/trade
- avg_win, avg_loss, realized RR
- Dynamic Fib 효과 분석
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import io
import numpy as np
from contextlib import redirect_stdout
from typing import List, Dict, Any
from collections import defaultdict

from backtest_strategy_compare import (
    build_config_from_mode, load_data,
    precompute_trend_column, _duration_to_bars, StrategyA, Trade, Config
)
from src.features.feature_store import FeatureStore


def run_backtest(config: Config, start: str, end: str) -> tuple:
    """백테스트 실행 후 (trades, 로그) 반환"""
    df_15m = load_data('15m', start, end, config)
    df_5m = load_data('5m', start, end, config)
    df_1h = load_data('1h', start, end, config)
    df_4h = load_data('4h', start, end, config)
    df_1h['trend'] = precompute_trend_column(df_1h, lookback=20)
    df_4h['trend'] = precompute_trend_column(df_4h, lookback=10)

    store = FeatureStore(config, duration_to_bars=_duration_to_bars)
    store.attach('trigger', df_5m, timeframe='5m')
    store.attach('anchor', df_15m, timeframe='15m')
    store.attach('context', df_1h, timeframe='1h')
    store.register_default_prob_gate_bundle()
    prob_gate_result = store.get_df('prob_gate_bundle')

    strategy = StrategyA(config)

    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        result = strategy.run(df_15m, df_5m, df_1h, df_4h, prob_gate_result)

    logs = log_buffer.getvalue()
    return [t for t in result.trades if t.side == 'long'], logs


def calc_metrics(trades: List[Trade]) -> Dict[str, Any]:
    """트레이드 리스트에서 메트릭 계산"""
    if not trades:
        return {
            'count': 0, 'win_rate': 0, 'total_pnl': 0,
            'avg_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
            'realized_rr': 0
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]

    total_pnl = sum(t.pnl_usd for t in trades)
    avg_pnl = total_pnl / len(trades)
    avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0
    realized_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'count': len(trades),
        'win_rate': len(wins) / len(trades) * 100,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'realized_rr': realized_rr
    }


def parse_limit_stats(logs: str) -> Dict[str, Any]:
    """로그에서 limit order 통계 추출"""
    stats = {
        'limit_orders': 0,
        'limit_filled': 0,
        'limit_expired': 0,
    }

    for line in logs.split('\n'):
        if '[LONG LIMIT ORDER]' in line:
            stats['limit_orders'] += 1
        elif '[LONG LIMIT FILLED]' in line:
            stats['limit_filled'] += 1
        elif '[LONG LIMIT EXPIRED]' in line:
            stats['limit_expired'] += 1

    if stats['limit_orders'] > 0:
        stats['fill_rate'] = stats['limit_filled'] / stats['limit_orders'] * 100
    else:
        stats['fill_rate'] = 0

    return stats


def main():
    print("=" * 120)
    print("PR-DYN-FIB Test: MODE43 vs MODE44 vs MODE45 vs MODE46 (4-Way Comparison)")
    print("=" * 120)
    print()
    print("Hypothesis: Dynamic Fib will provide better TP candidates, improving $/trade")
    print()

    # 테스트 구간
    periods = [
        ('2021-Q1', '2021-01-01', '2021-03-31', 'BULL'),
        ('2021-Nov', '2021-10-01', '2021-12-31', 'BULL->BEAR'),
        ('2022-May', '2022-04-01', '2022-06-30', 'BEAR'),
        ('2023-Jul', '2023-06-01', '2023-08-31', 'RANGE'),
    ]

    modes = [43, 44, 45, 46]
    mode_names = {
        43: "MODE43 (Champion)",
        44: "MODE44 (ZigZag)",
        45: "MODE45 (Rolling)",
        46: "MODE46 (Conditional)",
    }

    # 결과 저장
    all_results = defaultdict(dict)  # {period: {mode: metrics}}
    all_limit_stats = defaultdict(dict)  # {period: {mode: limit_stats}}

    # 각 구간별 테스트
    for period_name, start, end, regime in periods:
        print("=" * 120)
        print(f"[{period_name}] {regime} ({start} ~ {end})")
        print("-" * 120)

        for mode in modes:
            config = build_config_from_mode(mode)
            print(f"  Running {mode_names[mode]}...", end=" ")

            try:
                trades, logs = run_backtest(config, start, end)
                metrics = calc_metrics(trades)
                limit_stats = parse_limit_stats(logs)

                all_results[period_name][mode] = metrics
                all_limit_stats[period_name][mode] = limit_stats

                print(f"{metrics['count']} trades, ${metrics['total_pnl']:+.2f}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results[period_name][mode] = {'count': 0, 'total_pnl': 0, 'avg_pnl': 0}
                all_limit_stats[period_name][mode] = {}

        print()

    # 결과 비교 테이블
    print("=" * 120)
    print("Results Comparison")
    print("=" * 120)
    print()

    header = f"{'Period':<12} | {'Mode':<20} | {'Tr':>4} | {'WR%':>6} | {'PnL':>10} | {'$/Tr':>8} | {'AvgWin':>8} | {'AvgLoss':>9} | {'RR':>5}"
    print(header)
    print("-" * 120)

    for period_name, start, end, regime in periods:
        for i, mode in enumerate(modes):
            m = all_results[period_name].get(mode, {})
            if m.get('count', 0) > 0:
                row = f"{period_name if i == 0 else '':<12} | {mode_names[mode]:<20} | {m['count']:>4} | {m['win_rate']:>5.1f}% | ${m['total_pnl']:>8.2f} | ${m['avg_pnl']:>6.2f} | ${m['avg_win']:>6.2f} | ${m['avg_loss']:>8.2f} | {m['realized_rr']:>5.2f}"
            else:
                row = f"{period_name if i == 0 else '':<12} | {mode_names[mode]:<20} | No trades or error"
            print(row)

            # Limit stats for reference
            ls = all_limit_stats[period_name].get(mode, {})
            if ls.get('limit_orders', 0) > 0:
                print(f"{'':>12} |   Limit Stats      | Orders={ls['limit_orders']}, Filled={ls['limit_filled']}, Expired={ls.get('limit_expired', 0)}, Fill%={ls['fill_rate']:.0f}%")

        print()

    # 전체 요약
    print("=" * 120)
    print("SUMMARY (All Periods)")
    print("=" * 120)
    print()

    for mode in modes:
        total_trades = sum(all_results[p][mode].get('count', 0) for p, _, _, _ in periods)
        total_pnl = sum(all_results[p][mode].get('total_pnl', 0) for p, _, _, _ in periods)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # 승률 계산 (가중 평균)
        wins = sum(all_results[p][mode].get('count', 0) * all_results[p][mode].get('win_rate', 0) / 100
                   for p, _, _, _ in periods)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        print(f"{mode_names[mode]}:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Avg $/Trade: ${avg_pnl:.2f}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print()

    # 최종 판정
    print("=" * 120)
    print("VERDICT")
    print("=" * 120)
    print()

    # 최고 $/trade 찾기
    mode_avg_pnls = {}
    for mode in modes:
        total_trades = sum(all_results[p][mode].get('count', 0) for p, _, _, _ in periods)
        total_pnl = sum(all_results[p][mode].get('total_pnl', 0) for p, _, _, _ in periods)
        mode_avg_pnls[mode] = total_pnl / total_trades if total_trades > 0 else -9999

    best_mode = max(mode_avg_pnls, key=mode_avg_pnls.get)
    print(f"Best $/Trade: {mode_names[best_mode]} (${mode_avg_pnls[best_mode]:.2f})")
    print()

    # Champion(MODE43)과 비교
    champion_avg = mode_avg_pnls[43]
    for mode in [44, 45, 46]:
        diff = mode_avg_pnls[mode] - champion_avg
        if diff > 0:
            print(f"[DYNAMIC FIB HELPS] {mode_names[mode]} is ${diff:.2f}/trade better than MODE43")
        elif diff < 0:
            print(f"[DYNAMIC FIB HURTS] {mode_names[mode]} is ${abs(diff):.2f}/trade worse than MODE43")
        else:
            print(f"[NO CHANGE] {mode_names[mode]} same as MODE43")

    print()


if __name__ == "__main__":
    main()
