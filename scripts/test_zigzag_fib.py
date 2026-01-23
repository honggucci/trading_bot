#!/usr/bin/env python3
"""ZigZag Fibonacci 단위 테스트 - 1W 피봇 감지 검증"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Dynamic Fib 모듈 임포트
from src.context.dynamic_fib_anchor import (
    create_initial_state, update_anchor_zigzag, get_dynamic_fib_levels
)

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
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

df_1w['atr'] = calc_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)

# 2025-01-01부터 2026-01-23까지 (1년치)
df = df_1w['2025-01-01':'2026-01-23'].copy()

print("="*70)
print("ZigZag Fibonacci 단위 테스트 (1W, 2025-01-01 ~ 2026-01-23)")
print("="*70)
print(f"총 1W 바: {len(df)}")
print()

# 원본 OHLC 출력
print("1W 바 데이터 (처음 10개):")
print("-"*70)
for i, (ts, row) in enumerate(df.head(10).iterrows()):
    print(f"  {ts.date()} | O=${row['open']:,.0f} H=${row['high']:,.0f} L=${row['low']:,.0f} C=${row['close']:,.0f}")

print()

# ZigZag state 초기화 및 업데이트
state = create_initial_state("zigzag")
reversal_atr_mult = 1.5

pivots = []
for i in range(len(df)):
    bar = df.iloc[i]
    atr = bar['atr'] if pd.notna(bar['atr']) else 5000

    old_low = state.low
    old_high = state.high
    old_dir = state.direction

    state = update_anchor_zigzag(df, i, state, atr, reversal_atr_mult)

    # pivot 변경 감지
    if state.low != old_low or state.high != old_high:
        pivots.append({
            'ts': df.index[i],
            'low': state.low,
            'high': state.high,
            'direction': state.direction,
            'pivot_count': state.pivot_count
        })

print("피봇 히스토리:")
print("-"*70)
for p in pivots:
    print(f"  {p['ts'].date()} | Low=${p['low']:,.0f} | High=${p['high']:,.0f} | Dir={p['direction']} | Pivots={p['pivot_count']}")

print()
print("="*70)
print("최종 ZigZag Fib Anchor")
print("="*70)
print(f"  Low:  ${state.low:,.0f}")
print(f"  High: ${state.high:,.0f}")
print(f"  Range: ${state.high - state.low:,.0f} ({100*(state.high - state.low)/state.low:.1f}%)")
print(f"  Direction: {state.direction}")
print(f"  Pivot Count: {state.pivot_count}")
print()

# Fib 레벨 계산
ratios = [-2.0, -1.618, -1.0, -0.618, -0.382, -0.236, 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0]
levels = get_dynamic_fib_levels(state.low, state.high, ratios, space="linear")

print("="*70)
print("Dynamic Fib 레벨 (Linear)")
print("="*70)
for ratio, level in zip(ratios, levels):
    marker = ""
    if ratio == 0:
        marker = " <-- LOW"
    elif ratio == 1.0:
        marker = " <-- HIGH"
    elif ratio == 0.618:
        marker = " <-- Golden Ratio"
    print(f"  {ratio:6.3f}: ${level:,.0f}{marker}")

# 2026-01-23 기준 현재가 위치
current_price = df.iloc[-1]['close']
print()
print(f"현재가 (2026-01-23): ${current_price:,.0f}")

# 현재가가 어느 Fib 레벨 사이에 있는지 확인
for i in range(len(levels) - 1):
    if levels[i] <= current_price <= levels[i+1]:
        print(f"  → {ratios[i]:.3f} (${levels[i]:,.0f}) ~ {ratios[i+1]:.3f} (${levels[i+1]:,.0f}) 구간")
        break
