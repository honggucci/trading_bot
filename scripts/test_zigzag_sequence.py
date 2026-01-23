#!/usr/bin/env python3
"""ZigZag 스윙 시퀀스 분석 - 저점/고점 교대 확인"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.context.dynamic_fib_anchor import create_initial_state, update_anchor_zigzag

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

# ATR 계산
def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values

df_1w['atr'] = calc_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)

df = df_1w['2025-01-01':'2026-01-23'].copy()

print("="*80)
print("ZigZag 스윙 시퀀스 분석")
print("="*80)
print()

# ZigZag 시뮬레이션하며 피봇 변경 추적
state = create_initial_state("zigzag")
reversal_atr_mult = 1.5

pivot_events = []
for i in range(len(df)):
    bar = df.iloc[i]
    atr = bar['atr'] if pd.notna(bar['atr']) else 5000

    old_low = state.low
    old_high = state.high
    old_dir = state.direction

    state = update_anchor_zigzag(df, i, state, atr, reversal_atr_mult)

    # 저점 변경
    if state.low != old_low:
        pivot_events.append({
            'date': df.index[i],
            'type': 'LOW',
            'price': state.low,
            'direction': state.direction
        })

    # 고점 변경
    if state.high != old_high:
        pivot_events.append({
            'date': df.index[i],
            'type': 'HIGH',
            'price': state.high,
            'direction': state.direction
        })

# 피봇 이벤트를 시퀀스로 출력
print("피봇 시퀀스 (저점 ↔ 고점 교대):")
print("-"*80)
print(f"{'날짜':<12} {'타입':<6} {'가격':>12} {'방향':<6}")
print("-"*80)

for evt in pivot_events:
    marker = "★" if evt['price'] == 126200 else ""
    print(f"{evt['date'].date()!s:<12} {evt['type']:<6} ${evt['price']:>10,.0f} {evt['direction']:<6} {marker}")

# 절대 최고점 $126,200 직전 저점 찾기
print()
print("="*80)
print("절대 최고점 $126,200 분석")
print("="*80)

# $126,200 직전의 저점 이벤트 찾기
for i, evt in enumerate(pivot_events):
    if evt['price'] == 126200 and evt['type'] == 'HIGH':
        print(f"고점 $126,200 확정일: {evt['date'].date()}")

        # 직전 저점 찾기
        for j in range(i-1, -1, -1):
            if pivot_events[j]['type'] == 'LOW':
                prev_low = pivot_events[j]
                print(f"직전 저점: ${prev_low['price']:,.0f} ({prev_low['date'].date()})")
                break
        break

# 실제 데이터에서 날짜 확인
print()
print("="*80)
print("실제 1W 바 데이터에서 확인")
print("="*80)

# $126,200 고점이 찍힌 봉
high_126200 = df[df['high'] == 126200]
if len(high_126200) > 0:
    for ts, row in high_126200.iterrows():
        print(f"$126,200 고점 봉: {ts.date()} | O=${row['open']:,.0f} H=${row['high']:,.0f} L=${row['low']:,.0f} C=${row['close']:,.0f}")

# $107,255 저점이 찍힌 봉 찾기
low_107255 = df[(df['low'] >= 107000) & (df['low'] <= 108000)]
print()
print("$107,000~$108,000 범위 저점 봉:")
for ts, row in low_107255.iterrows():
    print(f"  {ts.date()} | O=${row['open']:,.0f} H=${row['high']:,.0f} L=${row['low']:,.0f} C=${row['close']:,.0f}")
