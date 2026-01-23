#!/usr/bin/env python3
"""ZigZag 피봇의 실제 날짜 확인"""

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

# 2025-01-01부터 2026-01-23까지
df = df_1w['2025-01-01':'2026-01-23'].copy()

# 최종 앵커 값
final_low = 80600
final_high = 116400

print("="*70)
print("ZigZag 피봇 실제 날짜 확인")
print("="*70)
print()

# Low가 찍힌 실제 날짜 찾기
low_bars = df[df['low'] == final_low]
if len(low_bars) > 0:
    print(f"Low ${final_low:,.0f} 가 찍힌 날짜:")
    for ts, row in low_bars.iterrows():
        print(f"  {ts.date()} | O=${row['open']:,.0f} H=${row['high']:,.0f} L=${row['low']:,.0f} C=${row['close']:,.0f}")
else:
    # 정확히 일치하지 않으면 가장 가까운 값 찾기
    idx = (df['low'] - final_low).abs().idxmin()
    row = df.loc[idx]
    print(f"Low ${final_low:,.0f} 에 가장 가까운 봉:")
    print(f"  {idx.date()} | Low=${row['low']:,.0f}")

print()

# High가 찍힌 실제 날짜 찾기
high_bars = df[df['high'] == final_high]
if len(high_bars) > 0:
    print(f"High ${final_high:,.0f} 가 찍힌 날짜:")
    for ts, row in high_bars.iterrows():
        print(f"  {ts.date()} | O=${row['open']:,.0f} H=${row['high']:,.0f} L=${row['low']:,.0f} C=${row['close']:,.0f}")
else:
    # 정확히 일치하지 않으면 가장 가까운 값 찾기
    idx = (df['high'] - final_high).abs().idxmin()
    row = df.loc[idx]
    print(f"High ${final_high:,.0f} 에 가장 가까운 봉:")
    print(f"  {idx.date()} | High=${row['high']:,.0f}")

print()
print("="*70)
print("전체 1W 바에서 최저/최고점")
print("="*70)
min_low_idx = df['low'].idxmin()
max_high_idx = df['high'].idxmax()

print(f"최저점: {min_low_idx.date()} | Low=${df.loc[min_low_idx]['low']:,.0f}")
print(f"최고점: {max_high_idx.date()} | High=${df.loc[max_high_idx]['high']:,.0f}")
