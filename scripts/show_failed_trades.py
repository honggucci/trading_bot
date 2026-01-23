#!/usr/bin/env python3
"""
MODE78 실패 매매 예시 분석
- SL로 청산된 최근 매매 5건 상세 출력
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from scripts.backtest_strategy_compare import BacktestConfig, StrategyA, load_data

# Config 로드
config_path = ROOT / "configs" / "mode78.json"
with open(config_path, encoding='utf-8') as f:
    config_dict = json.load(f)
config = BacktestConfig(**{k: v for k, v in config_dict.items() if not k.startswith('_')})

# 데이터 로드
START_DATE = "2025-01-01"
END_DATE = "2026-01-23"

print("데이터 로딩 중...")
df_15m = load_data("15m", START_DATE, END_DATE, config)
df_5m = load_data("5m", START_DATE, END_DATE, config)
df_1h = load_data("1h", START_DATE, END_DATE, config)
df_4h = load_data("4h", START_DATE, END_DATE, config)
print(f"  15m: {len(df_15m)} bars, 5m: {len(df_5m)} bars")
print(f"  1h: {len(df_1h)} bars, 4h: {len(df_4h)} bars")

# 백테스트 실행
print("\n백테스트 실행 중...")
strategy = StrategyA(config)
result = strategy.run(df_15m, df_5m, df_1h=df_1h, df_4h=df_4h)

# SL로 청산된 매매 필터링
sl_trades = [t for t in result.trades if t.exit_reason == 'SL']

print("=" * 80)
print(f"SL 청산 매매 분석 (총 {len(sl_trades)}건 중 최근 5건)")
print("=" * 80)

# 최근 5건
for i, trade in enumerate(sl_trades[-5:], 1):
    entry = trade.entry_price
    sl = trade.sl
    tp1 = trade.tp1 if hasattr(trade, 'tp1') else None
    exit_price = trade.exit_price
    pnl = trade.pnl_usd

    # 거리 계산
    sl_dist = entry - sl
    sl_dist_pct = (sl_dist / entry) * 100

    # MFE (최대 유리 이동) - trade에 저장되어 있으면 사용
    mfe = getattr(trade, 'mfe', None)
    mae = getattr(trade, 'mae', None)

    print(f"\n[Trade #{i}] {trade.entry_time}")
    print(f"  Side: {trade.side}")
    print(f"  Entry:  ${entry:,.0f}")
    print(f"  SL:     ${sl:,.0f} (거리: ${sl_dist:,.0f}, {sl_dist_pct:.2f}%)")
    if tp1:
        tp_dist = tp1 - entry
        tp_dist_pct = (tp_dist / entry) * 100
        print(f"  TP1:    ${tp1:,.0f} (거리: ${tp_dist:,.0f}, {tp_dist_pct:.2f}%)")
    print(f"  Exit:   ${exit_price:,.0f} @ {trade.exit_time}")
    print(f"  PnL:    ${pnl:,.2f}")
    if mfe is not None:
        print(f"  MFE:    ${mfe:,.0f} (최대 유리)")
    if mae is not None:
        print(f"  MAE:    ${mae:,.0f} (최대 불리)")

    # 실제 가격 데이터에서 MFE/MAE 확인
    # (trade 객체에 없으면 별도 계산 필요)

print("\n" + "=" * 80)
print("요약 통계")
print("=" * 80)

if sl_trades:
    sl_distances = [(t.entry_price - t.sl) for t in sl_trades]
    sl_dist_pcts = [(t.entry_price - t.sl) / t.entry_price * 100 for t in sl_trades]

    print(f"  평균 SL 거리: ${sum(sl_distances)/len(sl_distances):,.0f} ({sum(sl_dist_pcts)/len(sl_dist_pcts):.2f}%)")
    print(f"  최소 SL 거리: ${min(sl_distances):,.0f} ({min(sl_dist_pcts):.2f}%)")
    print(f"  최대 SL 거리: ${max(sl_distances):,.0f} ({max(sl_dist_pcts):.2f}%)")
else:
    print("  SL 청산된 매매 없음")

# 전체 매매 요약
print("\n" + "=" * 80)
print("전체 매매 요약")
print("=" * 80)
total_trades = len(result.trades)
tp_trades = [t for t in result.trades if 'TP' in (t.exit_reason or '')]
print(f"  총 매매: {total_trades}건")
print(f"  SL 청산: {len(sl_trades)}건 ({len(sl_trades)/total_trades*100:.1f}%)" if total_trades else "  SL 청산: 0건")
print(f"  TP 청산: {len(tp_trades)}건 ({len(tp_trades)/total_trades*100:.1f}%)" if total_trades else "  TP 청산: 0건")
