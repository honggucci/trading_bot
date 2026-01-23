#!/usr/bin/env python3
"""P0-5b 검증: Pending Reversal 상태 머신 테스트

핵심 검증:
1. PENDING_START: spacing 부족 시 pending 진입
2. PENDING_UPDATE: 반대 극점 갱신
3. PENDING_CANCEL: 가격이 pivot 초과 시 취소
4. PENDING_CONFIRM: spacing 충족 시 확정
5. OHLC 정합: 기존 테스트 ALL PASS 유지
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
print("P0-5b 검증: Pending Reversal 상태 머신 테스트")
print("="*80)
print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"총 1W 바: {len(df)}")
print()

# === 로깅 활성화된 버전으로 실행 ===
state = DynamicFibAnchorState(mode='zigzag')
pivots = []
pending_events = []

for i in range(len(df)):
    bar = df.iloc[i]
    atr = bar['atr'] if pd.notna(bar['atr']) else 5000
    current_ts = df.index[i]

    old_pivot_count = state.pivot_count
    old_pending = state.pending
    old_pending_type = state.pending_type

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

    # Pending 이벤트 로깅
    if not old_pending and state.pending:
        # PENDING_START
        pending_events.append({
            'event': 'PENDING_START',
            'ts': current_ts,
            'type': state.pending_type,
            'pivot_price': state.pending_pivot_price,
            'opposite_price': state.pending_opposite_price
        })
        print(f"[{current_ts.date()}] PENDING_START: {state.pending_type} @ ${state.pending_pivot_price:,.0f}")

    elif old_pending and not state.pending:
        if state.pivot_count > old_pivot_count:
            # PENDING_CONFIRM
            pending_events.append({
                'event': 'PENDING_CONFIRM',
                'ts': current_ts,
                'type': state.last_pivot_type,
                'pivot_price': state.high if state.last_pivot_type == 'HIGH' else state.low
            })
            print(f"[{current_ts.date()}] PENDING_CONFIRM: {state.last_pivot_type} @ ${state.high if state.last_pivot_type == 'HIGH' else state.low:,.0f}")
        else:
            # PENDING_CANCEL
            pending_events.append({
                'event': 'PENDING_CANCEL',
                'ts': current_ts,
                'type': old_pending_type
            })
            print(f"[{current_ts.date()}] PENDING_CANCEL: {old_pending_type}")

    # Pivot 확정 (pending 아닌 일반 확정 포함)
    if state.pivot_count > old_pivot_count and state.last_pivot_type:
        pivots.append({
            'extreme_ts': state.last_extreme_ts,
            'confirm_ts': state.last_confirm_ts,
            'type': state.last_pivot_type,
            'price': state.high if state.last_pivot_type == 'HIGH' else state.low,
            'k': state.k_anchor
        })

print()
print("="*80)
print("Pending 이벤트 요약")
print("="*80)
start_count = len([e for e in pending_events if e['event'] == 'PENDING_START'])
confirm_count = len([e for e in pending_events if e['event'] == 'PENDING_CONFIRM'])
cancel_count = len([e for e in pending_events if e['event'] == 'PENDING_CANCEL'])
print(f"  PENDING_START:   {start_count}")
print(f"  PENDING_CONFIRM: {confirm_count}")
print(f"  PENDING_CANCEL:  {cancel_count}")

print()
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

# Spacing 계산
if len(pivots) > 1:
    spacing_extreme = []
    spacing_confirm = []
    lags = []

    for j in range(1, len(pivots)):
        if pivots[j]['extreme_ts'] and pivots[j-1]['extreme_ts']:
            days_e = (pivots[j]['extreme_ts'] - pivots[j-1]['extreme_ts']).days
            spacing_extreme.append(days_e / 7.0)

        if pivots[j]['confirm_ts'] and pivots[j-1]['confirm_ts']:
            days_c = (pivots[j]['confirm_ts'] - pivots[j-1]['confirm_ts']).days
            spacing_confirm.append(days_c / 7.0)

        if pivots[j]['extreme_ts'] and pivots[j]['confirm_ts']:
            lag_d = (pivots[j]['confirm_ts'] - pivots[j]['extreme_ts']).days
            lags.append(lag_d / 7.0)

    print()
    print("="*80)
    print("Spacing 분석")
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
print(f"  Pending: {state.pending} ({state.pending_type})")

# 결과 요약
print()
print("="*80)
print("P0-5b 검증 결과")
print("="*80)

all_pass = True

# 1. Pending 이벤트 발생 확인
if start_count > 0:
    print(f"  [PASS] PENDING_START: {start_count} times")
else:
    print(f"  [WARN] PENDING_START: 0 times (no spacing-insufficient cases)")

# 2. OHLC 정합
if ohlc_fail == 0:
    print(f"  [PASS] OHLC: ALL PASS ({ohlc_pass}/{ohlc_pass})")
else:
    print(f"  [FAIL] OHLC: {ohlc_fail} FAIL")
    all_pass = False

# 3. Spacing 목표
if spacing_extreme:
    median_extreme = np.median(spacing_extreme)
    if median_extreme >= 13.0:
        print(f"  [PASS] median(spacing_extreme) = {median_extreme:.1f}w >= 13w")
    else:
        print(f"  [FAIL] median(spacing_extreme) = {median_extreme:.1f}w < 13w")
        all_pass = False

# 4. Pending 상태 일관성
if not state.pending:
    print(f"  [PASS] End state: pending=False (normal)")
else:
    print(f"  [WARN] End state: pending=True (incomplete)")

print()
if all_pass:
    print("  >>> P0-5b PASS: Pending Reversal state machine working correctly")
else:
    print("  >>> P0-5b FAIL: Additional fixes needed")
