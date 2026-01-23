#!/usr/bin/env python3
"""StochRSI 단위 테스트 - 과매도 감지 검증"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import talib

# 데이터 로드
ROOT = Path(r"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot")
data_dir = ROOT / "data" / "bronze" / "binance" / "futures" / "BTC-USDT" / "15m"

files = list(data_dir.glob("**/*.parquet"))
dfs = [pd.read_parquet(f) for f in files]
df_15m = pd.concat(dfs, ignore_index=True)

if 'datetime' in df_15m.columns:
    df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
    df_15m = df_15m.set_index('datetime')
elif 'timestamp' in df_15m.columns:
    df_15m['datetime'] = pd.to_datetime(df_15m['timestamp'], unit='ms')
    df_15m = df_15m.set_index('datetime')

df_15m = df_15m.sort_index()

# 2026년 1월 데이터만
df = df_15m['2026-01-01':'2026-01-23'].copy()

# StochRSI 계산 (talib) - 전체 데이터로 계산 후 슬라이스
close_full = df_15m['close'].values.astype(np.float64)
fastk_full, fastd_full = talib.STOCHRSI(close_full, timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0)
df_15m['stoch_d'] = fastd_full

# 슬라이스
df = df_15m['2026-01-01':'2026-01-23'].copy()

# 과매도 구간 찾기 (stoch_d <= 20)
oversold_mask = df['stoch_d'] <= 20
oversold_bars = df[oversold_mask]

print("="*70)
print("StochRSI 단위 테스트 (2026-01-01 ~ 01-23)")
print("="*70)
print(f"총 15m 바: {len(df)}")
print(f"과매도 바 (stoch_d <= 20): {len(oversold_bars)} ({100*len(oversold_bars)/len(df):.1f}%)")
print()

# 과매도 구간 시작점 찾기 (transition into oversold)
df['prev_stoch_d'] = df['stoch_d'].shift(1)
df['oversold_start'] = (df['stoch_d'] <= 20) & (df['prev_stoch_d'] > 20)
oversold_starts = df[df['oversold_start']]

print(f"과매도 진입 횟수: {len(oversold_starts)}")
print()
print("과매도 진입 시점 (처음 10개):")
print("-"*70)
for i, (ts, row) in enumerate(oversold_starts.head(10).iterrows()):
    print(f"  {ts} | close=${row['close']:,.0f} | stoch_d={row['stoch_d']:.1f} (prev={row['prev_stoch_d']:.1f})")

# 과매도 상태인 봉 (state mode용)
df['is_oversold'] = df['stoch_d'] <= 20
oversold_state_bars = df[df['is_oversold']]

print()
print("="*70)
print("과매도 상태 봉 (state mode 기준)")
print("="*70)
print(f"과매도 봉 수: {len(oversold_state_bars)} ({100*len(oversold_state_bars)/len(df):.1f}%)")
print()
print("과매도 봉 (처음 15개):")
print("-"*70)
for i, (ts, row) in enumerate(oversold_state_bars.head(15).iterrows()):
    print(f"  {ts} | close=${row['close']:,.0f} | stoch_d={row['stoch_d']:.1f}")

print()
print("="*70)
print("전체 StochRSI 분포")
print("="*70)
print(f"  Min: {df['stoch_d'].min():.1f}")
print(f"  Max: {df['stoch_d'].max():.1f}")
print(f"  Mean: {df['stoch_d'].mean():.1f}")
print(f"  Median: {df['stoch_d'].median():.1f}")
print()
print("구간별 분포:")
bins = [0, 20, 40, 60, 80, 100]
for i in range(len(bins)-1):
    mask = (df['stoch_d'] >= bins[i]) & (df['stoch_d'] < bins[i+1])
    cnt = mask.sum()
    print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {cnt:4d} ({100*cnt/len(df):5.1f}%)")
