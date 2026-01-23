#!/usr/bin/env python3
"""P0-1 치팅 검출 테스트

핵심 원리:
- Online (시점별 slice): prices[:t+1]로 k 계산
- Batch (full series): 전체 prices로 k 계산
- 두 결과가 **달라야 정상** (같으면 미래정보를 봤을 가능성)

PASS 조건:
1. Online과 Batch 결과가 다름 (diff > 0)
2. Online 방식만 사용하면 미래정보 없음
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.context.dynamic_fib_anchor import compute_dynamic_k

# 데이터 로드
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

# ATR 계산
def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values

df_1w['atr'] = calc_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)

df = df_1w['2025-01-01':'2026-01-23'].copy()
prices_full = df['close'].values

print("="*70)
print("P0-1 치팅 검출 테스트: Online vs Batch")
print("="*70)
print()

# 테스트 시점들 (전체의 25%, 50%, 75%, 100%)
test_points = [len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]

results = []
for mode in ["hilbert", "regime_switch", "inverse"]:
    print(f"Mode: {mode}")
    print("-"*50)

    diffs = []
    for i in test_points:
        atr = df['atr'].iloc[i]

        # Online: prices[:i+1] (현재 시점까지만)
        prices_online = prices_full[:i+1]
        k_online = compute_dynamic_k(
            prices_online, atr, mode=mode,
            k_min=1.8, k_max=4.5, k_fixed=2.0, strength_ref=0.8,
            ampz_thr=0.5, k_cycle=2.0, k_trend=3.8
        )

        # Batch: 전체 prices
        k_batch = compute_dynamic_k(
            prices_full, atr, mode=mode,
            k_min=1.8, k_max=4.5, k_fixed=2.0, strength_ref=0.8,
            ampz_thr=0.5, k_cycle=2.0, k_trend=3.8
        )

        diff = abs(k_online - k_batch)
        diffs.append(diff)

        date = df.index[i].date()
        status = "DIFF" if diff > 0.01 else "SAME"
        print(f"  t={i:3} ({date}) | online={k_online:.3f} | batch={k_batch:.3f} | diff={diff:.3f} | {status}")

    # 전체 diff가 0인 경우 (모두 같음) = 미래정보 유출 가능성
    all_same = all(d < 0.01 for d in diffs)
    result = "FAIL (미래정보 유출 가능)" if all_same else "PASS (Online != Batch)"
    results.append((mode, result, diffs))

    print(f"  >>> {result}")
    print()

print("="*70)
print("최종 결과")
print("="*70)
for mode, result, diffs in results:
    avg_diff = np.mean(diffs)
    print(f"  {mode:<15}: {result} (avg_diff={avg_diff:.3f})")

print()
print("="*70)
print("해석 가이드")
print("="*70)
print("""
- PASS (Online != Batch): Online 방식이 전체 시계열을 보지 않음 (정상)
- FAIL (미래정보 유출 가능): Online과 Batch 결과가 같음 (치팅 의심)

참고: Hilbert Transform은 본질적으로 전역 연산이므로,
sliding window 방식이 아니면 미래정보가 섞일 수 있음.
현재 구현은 WaveRegimeClassifier.classify()를 사용하며,
이것이 causal인지 확인 필요.
""")
