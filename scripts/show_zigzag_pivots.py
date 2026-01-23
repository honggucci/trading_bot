#!/usr/bin/env python3
"""P0-5 검증: ZigZag pivot extreme_ts vs confirm_ts 분리 테스트

핵심 검증:
1. OHLC 정합: df.loc[extreme_ts, 'high/low'] == pivot_price
2. spacing_extreme: 실제 스윙 간격 (연 2-4회 목표 기준)
3. spacing_confirm: 전략 앵커 갱신 간격
4. confirm_lag: confirm_ts - extreme_ts 분포
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.context.dynamic_fib_anchor import (
    DynamicFibAnchorState,
    update_anchor_zigzag,
)

# Load data
ROOT = Path(__file__).parent.parent
data_dir = ROOT / "data" / "bronze" / "binance" / "futures" / "BTC-USDT" / "1w"

files = list(data_dir.glob("**/*.parquet"))
dfs = [pd.read_parquet(f) for f in files]
df_1w = pd.concat(dfs, ignore_index=True)

if 'datetime' in df_1w.columns:
    df_1w['datetime'] = pd.to_datetime(df_1w['datetime'])
    df_1w = df_1w.set_index('datetime')
elif 'timestamp' in df_1w.columns:
    df_1w['datetime'] = pd.to_datetime(df_1w['timestamp'], unit='ms')
    df_1w = df_1w.set_index('datetime')

df_1w = df_1w.sort_index()

# ATR
def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values

df_1w['atr'] = calc_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)

df = df_1w['2025-01-01':'2026-01-23'].copy()

print("="*80)
print("P0-5 검증: ZigZag extreme_ts vs confirm_ts 분리 테스트")
print("="*80)
print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"총 1W 바: {len(df)}")
print()

# Run with min_spacing=13
state = DynamicFibAnchorState(mode='zigzag')
pivots = []

for i in range(len(df)):
    bar = df.iloc[i]
    atr = bar['atr'] if pd.notna(bar['atr']) else 5000

    old_pivot_count = state.pivot_count

    state = update_anchor_zigzag(
        df, i, state, atr,
        reversal_mult=2.0,
        k_mode='hilbert',
        min_spacing_weeks=13.0,
        k_min=1.8,
        k_max=4.5,
        k_fixed=2.0,
        strength_ref=0.8
    )

    if state.pivot_count > old_pivot_count and state.last_pivot_type:
        # P0-5 FIX: extreme_ts와 confirm_ts 모두 기록
        pivots.append({
            'extreme_ts': state.last_extreme_ts,
            'confirm_ts': state.last_confirm_ts,
            'type': state.last_pivot_type,
            'price': state.high if state.last_pivot_type == 'HIGH' else state.low,
            'k': state.k_anchor
        })

print("="*80)
print("감지된 Pivots (extreme_ts vs confirm_ts 분리)")
print("="*80)
print(f"{'#':>3} | {'Type':^5} | {'Extreme Date':^12} | {'Confirm Date':^12} | {'Lag(weeks)':>10} | {'Price':>12} | {'OHLC Check':^10}")
print("-"*90)

ohlc_pass = 0
ohlc_fail = 0

for idx, p in enumerate(pivots):
    extreme_ts = p['extreme_ts']
    confirm_ts = p['confirm_ts']

    # Lag 계산
    if extreme_ts and confirm_ts:
        lag_days = (confirm_ts - extreme_ts).days
        lag_weeks = lag_days / 7.0
    else:
        lag_weeks = None

    # OHLC 검증
    if extreme_ts and extreme_ts in df.index:
        if p['type'] == 'HIGH':
            actual_ohlc = df.loc[extreme_ts, 'high']
        else:
            actual_ohlc = df.loc[extreme_ts, 'low']

        # float 비교 (소수점 차이 허용)
        if abs(actual_ohlc - p['price']) < 1.0:
            ohlc_check = "PASS"
            ohlc_pass += 1
        else:
            ohlc_check = f"FAIL ({actual_ohlc:.0f})"
            ohlc_fail += 1
    else:
        ohlc_check = "N/A"
        ohlc_fail += 1

    extreme_date = extreme_ts.date() if extreme_ts else "N/A"
    confirm_date = confirm_ts.date() if confirm_ts else "N/A"
    lag_str = f"{lag_weeks:.1f}" if lag_weeks is not None else "N/A"

    print(f"  {idx+1:>2} | {p['type']:^5} | {str(extreme_date):^12} | {str(confirm_date):^12} | {lag_str:>10} | ${p['price']:>10,.0f} | {ohlc_check:^10}")

print("-"*90)
print(f"총 pivots: {len(pivots)}")
print(f"OHLC 정합: {ohlc_pass} PASS / {ohlc_fail} FAIL")

# Spacing 계산 (2종류)
if len(pivots) > 1:
    spacing_extreme = []
    spacing_confirm = []
    lags = []

    for j in range(1, len(pivots)):
        # extreme 기준 spacing
        if pivots[j]['extreme_ts'] and pivots[j-1]['extreme_ts']:
            days_e = (pivots[j]['extreme_ts'] - pivots[j-1]['extreme_ts']).days
            spacing_extreme.append(days_e / 7.0)

        # confirm 기준 spacing
        if pivots[j]['confirm_ts'] and pivots[j-1]['confirm_ts']:
            days_c = (pivots[j]['confirm_ts'] - pivots[j-1]['confirm_ts']).days
            spacing_confirm.append(days_c / 7.0)

        # lag
        if pivots[j]['extreme_ts'] and pivots[j]['confirm_ts']:
            lag_d = (pivots[j]['confirm_ts'] - pivots[j]['extreme_ts']).days
            lags.append(lag_d / 7.0)

    print()
    print("="*80)
    print("Spacing 분석 (2종류)")
    print("="*80)

    if spacing_extreme:
        print(f"spacing_extreme (실제 스윙 간격): {[f'{s:.1f}' for s in spacing_extreme]}")
        print(f"  median: {np.median(spacing_extreme):.1f}주")
        print(f"  mean: {np.mean(spacing_extreme):.1f}주")

    if spacing_confirm:
        print()
        print(f"spacing_confirm (전략 인지 간격): {[f'{s:.1f}' for s in spacing_confirm]}")
        print(f"  median: {np.median(spacing_confirm):.1f}주")
        print(f"  mean: {np.mean(spacing_confirm):.1f}주")

    if lags:
        print()
        print(f"confirm_lag (확정 지연): {[f'{l:.1f}' for l in lags]}")
        print(f"  median: {np.median(lags):.1f}주")
        print(f"  mean: {np.mean(lags):.1f}주")
        print(f"  max: {np.max(lags):.1f}주")

print()
print("="*80)
print("최종 앵커")
print("="*80)
print(f"  Low:  ${state.low:,.0f}")
print(f"  High: ${state.high:,.0f}")
print(f"  Direction: {state.direction}")

# 결과 요약
print()
print("="*80)
print("P0-5 검증 결과")
print("="*80)

target_met = False
if spacing_extreme:
    median_extreme = np.median(spacing_extreme)
    if median_extreme >= 13.0:
        print(f"  median(spacing_extreme) = {median_extreme:.1f}주 >= 13주 목표: PASS")
        target_met = True
    else:
        print(f"  median(spacing_extreme) = {median_extreme:.1f}주 < 13주 목표: FAIL")

if ohlc_pass > 0 and ohlc_fail == 0:
    print(f"  OHLC 정합 검증: ALL PASS ({ohlc_pass}/{ohlc_pass})")
else:
    print(f"  OHLC 정합 검증: {ohlc_fail} FAIL (재확인 필요)")

if target_met and ohlc_fail == 0:
    print()
    print("  >>> P0-5 PASS: extreme_ts 분리 저장 정상 동작")
else:
    print()
    print("  >>> P0-5 FAIL: 추가 수정 필요")
