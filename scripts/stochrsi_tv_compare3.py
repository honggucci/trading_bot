"""
StochRSI TradingView 파라미터 역추적
가설: TradingView Stochastic Length != 26
"""
import pandas as pd
import numpy as np
import talib
from pathlib import Path

# 1h 데이터 로드
data_dir = Path(r'c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\bronze\binance\futures\BTC-USDT\1h')
files = sorted(data_dir.glob('**/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['utc'] = df['timestamp'] - pd.Timedelta(hours=8)
df = df.sort_values('utc').reset_index(drop=True)

close_arr = df['close'].values.astype(np.float64)

def stoch_rsi_k(rsi_series, stoch_period, k_smooth=3):
    lo = rsi_series.rolling(stoch_period, min_periods=stoch_period).min()
    hi = rsi_series.rolling(stoch_period, min_periods=stoch_period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi_series - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)
    k = stoch.rolling(k_smooth, min_periods=k_smooth).mean() * 100.0
    return k

# TV 참조값 3개 (모두 검증)
tv_vals = {
    '2025-06-04 20:00': 17.99,
    '2025-06-04 21:00': 10.40,
    '2025-06-04 22:00': 14.46,
}

tv_indices = {}
for utc_str in tv_vals:
    rows = df[df['utc'].dt.strftime('%Y-%m-%d %H:%M') == utc_str]
    if len(rows) > 0:
        tv_indices[utc_str] = rows.index[0]

# Brute force: RSI period x Stochastic period x K smooth
print('=' * 90)
print('3-point fit: TV 20:00=17.99, 21:00=10.40, 22:00=14.46')
print('=' * 90)
print()

best_total_gap = 999
best_params = None
results = []

for rsi_p in [14]:
    rsi = talib.RSI(close_arr, timeperiod=rsi_p)
    rsi_s = pd.Series(rsi)

    for stoch_p in range(10, 51):
        for k_s in [1, 3, 5]:
            kk = stoch_rsi_k(rsi_s, stoch_p, k_s)

            total_gap = 0
            vals = {}
            for utc_str, tv in tv_vals.items():
                idx = tv_indices[utc_str]
                computed = kk.iloc[idx]
                vals[utc_str] = computed
                total_gap += abs(computed - tv)

            if total_gap < 5.0:  # close matches only
                results.append((rsi_p, stoch_p, k_s, total_gap, vals.copy()))

            if total_gap < best_total_gap:
                best_total_gap = total_gap
                best_params = (rsi_p, stoch_p, k_s, total_gap, vals.copy())

# Sort by total gap
results.sort(key=lambda x: x[3])

print('  Top 10 matches (sorted by total gap):')
print('  {:>4s} {:>6s} {:>3s} | {:>8s} | {:>15s} {:>15s} {:>15s}'.format(
    'RSI', 'Stoch', 'K', 'TotalGap', '20:00 (TV=17.99)', '21:00 (TV=10.40)', '22:00 (TV=14.46)'))
print('  ' + '-' * 85)

for rsi_p, stoch_p, k_s, gap, vals in results[:10]:
    v20 = vals['2025-06-04 20:00']
    v21 = vals['2025-06-04 21:00']
    v22 = vals['2025-06-04 22:00']
    print('  {:>4d} {:>6d} {:>3d} | {:>8.2f} | {:>7.2f} ({:+.2f}) {:>7.2f} ({:+.2f}) {:>7.2f} ({:+.2f})'.format(
        rsi_p, stoch_p, k_s, gap, v20, v20-17.99, v21, v21-10.40, v22, v22-14.46))

print()
print('  BEST: RSI={}, Stoch={}, K={}, Total gap={:.2f}'.format(
    best_params[0], best_params[1], best_params[2], best_params[3]))
for utc_str, tv in tv_vals.items():
    computed = best_params[4][utc_str]
    print('    {} : computed={:.2f}, TV={:.2f}, gap={:+.2f}'.format(
        utc_str, computed, tv, computed - tv))

# 현재 config (RSI=14, Stoch=26, K=3)
print()
print('  현재 config (RSI=14, Stoch=26, K=3):')
rsi14 = talib.RSI(close_arr, timeperiod=14)
rsi_s14 = pd.Series(rsi14)
k26 = stoch_rsi_k(rsi_s14, 26, 3)
total_gap_26 = 0
for utc_str, tv in tv_vals.items():
    idx = tv_indices[utc_str]
    computed = k26.iloc[idx]
    gap = computed - tv
    total_gap_26 += abs(gap)
    print('    {} : computed={:.2f}, TV={:.2f}, gap={:+.2f}'.format(
        utc_str, computed, tv, gap))
print('    Total gap: {:.2f}'.format(total_gap_26))

# ===== 추가: 다른 날짜도 검증 =====
print()
print('=' * 90)
print('Best 파라미터로 다른 시간대 예측값 (검증용)')
print('=' * 90)
print()

bp_rsi, bp_stoch, bp_k = best_params[0], best_params[1], best_params[2]
rsi_best = talib.RSI(close_arr, timeperiod=bp_rsi)
rsi_s_best = pd.Series(rsi_best)
k_best = stoch_rsi_k(rsi_s_best, bp_stoch, bp_k)
df['k_best'] = k_best.values

# 06-04 ~ 06-05 전체
mask = (df['utc'] >= '2025-06-04 16:00') & (df['utc'] <= '2025-06-05 06:00')
subset = df[mask]

print('  RSI={}, Stoch={}, K={} 기준 예측값:'.format(bp_rsi, bp_stoch, bp_k))
print('  (TradingView에서 아래 시간대 StochRSI 값 확인하면 검증 가능)')
print()

for _, row in subset.iterrows():
    utc_str = row['utc'].strftime('%Y-%m-%d %H:%M')
    tv = tv_vals.get(utc_str, None)
    marker = ' <-- TV ref' if tv is not None else ''
    tv_str = ' (TV={:.2f})'.format(tv) if tv else ''
    print('  {} | Close={:>10,.1f} | K={:>6.2f}{}{}'.format(
        utc_str, row['close'], row['k_best'], tv_str, marker))
