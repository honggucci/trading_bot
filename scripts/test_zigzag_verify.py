#!/usr/bin/env python3
"""ZigZag 피봇 검증 - 실제 스윙인지 주변 데이터로 확인"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# 데이터 로드
ROOT = Path(r"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot")
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
df = df_1w['2025-01-01':'2026-01-23'].copy()

# 검증할 피봇들 (ZigZag 결과에서)
pivots_to_verify = [
    {'type': 'LOW', 'price': 74508, 'confirm_date': '2025-04-21'},
    {'type': 'HIGH', 'price': 126200, 'confirm_date': '2025-10-06'},
    {'type': 'LOW', 'price': 80600, 'confirm_date': '2026-01-12'},
    {'type': 'HIGH', 'price': 116400, 'confirm_date': '2025-11-03'},
]

print("="*90)
print("ZigZag 피봇 검증 - 주변 봉 데이터로 실제 스윙 확인")
print("="*90)

for pivot in pivots_to_verify:
    print()
    print(f"{'='*90}")
    print(f"[{pivot['type']}] ${pivot['price']:,} (확정일: {pivot['confirm_date']})")
    print(f"{'='*90}")

    # 해당 가격이 찍힌 봉 찾기
    if pivot['type'] == 'LOW':
        target_bars = df[df['low'] == pivot['price']]
    else:
        target_bars = df[df['high'] == pivot['price']]

    if len(target_bars) == 0:
        # 정확히 일치하지 않으면 ±500 범위에서 찾기
        if pivot['type'] == 'LOW':
            target_bars = df[(df['low'] >= pivot['price'] - 500) & (df['low'] <= pivot['price'] + 500)]
        else:
            target_bars = df[(df['high'] >= pivot['price'] - 500) & (df['high'] <= pivot['price'] + 500)]

    if len(target_bars) == 0:
        print("  해당 가격의 봉을 찾을 수 없음")
        continue

    # 여러 봉이 있으면 confirm_date 기준으로 가장 가까운 봉 선택
    confirm_date = pd.Timestamp(pivot['confirm_date'])
    if len(target_bars) > 1:
        # confirm_date 이전이면서 가장 가까운 봉 선택
        before_confirm = target_bars[target_bars.index <= confirm_date]
        if len(before_confirm) > 0:
            target_date = before_confirm.index[-1]  # confirm_date 직전 봉
        else:
            target_date = target_bars.index[0]
    else:
        target_date = target_bars.index[0]
    target_idx = df.index.get_loc(target_date)

    print(f"실제 발생일: {target_date.date()}")
    print()

    # 전후 3봉씩 보여주기
    start_idx = max(0, target_idx - 3)
    end_idx = min(len(df), target_idx + 4)

    print(f"{'날짜':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}  검증")
    print("-"*70)

    for i in range(start_idx, end_idx):
        bar = df.iloc[i]
        bar_date = df.index[i]

        # 스윙 검증
        is_target = (i == target_idx)

        if pivot['type'] == 'LOW':
            # 저점 검증: 앞뒤 봉보다 낮아야 함
            if is_target:
                prev_low = df.iloc[i-1]['low'] if i > 0 else float('inf')
                next_low = df.iloc[i+1]['low'] if i < len(df)-1 else float('inf')
                is_valid_swing = bar['low'] < prev_low and bar['low'] < next_low
                marker = f"<< TARGET {'[O] SWING' if is_valid_swing else '[X] NOT SWING'}"
            else:
                marker = ""
        else:
            # 고점 검증: 앞뒤 봉보다 높아야 함
            if is_target:
                prev_high = df.iloc[i-1]['high'] if i > 0 else 0
                next_high = df.iloc[i+1]['high'] if i < len(df)-1 else 0
                is_valid_swing = bar['high'] > prev_high and bar['high'] > next_high
                marker = f"<< TARGET {'[O] SWING' if is_valid_swing else '[X] NOT SWING'}"
            else:
                marker = ""

        print(f"{bar_date.date()!s:<12} ${bar['open']:>9,.0f} ${bar['high']:>9,.0f} ${bar['low']:>9,.0f} ${bar['close']:>9,.0f}  {marker}")

print()
print("="*90)
print("검증 기준: 저점은 앞뒤 봉보다 낮아야 하고, 고점은 앞뒤 봉보다 높아야 함")
print("="*90)
