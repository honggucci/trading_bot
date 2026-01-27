"""
StochRSI 검증: RSI(26) + Stoch(26) + K(3) = TradingView?
"""
import pandas as pd
import numpy as np
import talib
from pathlib import Path

data_dir = Path(r'c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\bronze\binance\futures\BTC-USDT\1h')
files = sorted(data_dir.glob('**/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['utc'] = df['timestamp'] - pd.Timedelta(hours=8)
df = df.sort_values('utc').reset_index(drop=True)

close = df['close'].values.astype(np.float64)

def stoch_rsi_k(close_arr, rsi_period, stoch_period, k_smooth=3):
    rsi = talib.RSI(close_arr, timeperiod=rsi_period)
    rsi_s = pd.Series(rsi)
    lo = rsi_s.rolling(stoch_period, min_periods=stoch_period).min()
    hi = rsi_s.rolling(stoch_period, min_periods=stoch_period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi_s - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)
    k = stoch.rolling(k_smooth, min_periods=k_smooth).mean() * 100.0
    return k

# 현재 코드: RSI(14) + Stoch(26)
k_r14_s26 = stoch_rsi_k(close, rsi_period=14, stoch_period=26)
# TradingView: RSI(26) + Stoch(26)
k_r26_s26 = stoch_rsi_k(close, rsi_period=26, stoch_period=26)

df['k_r14_s26'] = k_r14_s26.values
df['k_r26_s26'] = k_r26_s26.values

tv_vals = {
    '2025-06-04 20:00': 17.99,
    '2025-06-04 21:00': 10.40,
    '2025-06-04 22:00': 14.46,
}

print('=' * 85)
print('StochRSI TradingView Match Test')
print('TV: RSI(26) + Stoch(26) + K(3)')
print('=' * 85)
print()

mask = (df['utc'] >= '2025-06-04 18:00') & (df['utc'] <= '2025-06-05 06:00')
subset = df[mask]

for _, row in subset.iterrows():
    utc_str = row['utc'].strftime('%Y-%m-%d %H:%M')
    tv = tv_vals.get(utc_str, None)
    marker = ''
    tv_str = ''
    if tv is not None:
        tv_str = '  TV={:.2f}'.format(tv)
        g14 = row['k_r14_s26'] - tv
        g26 = row['k_r26_s26'] - tv
        marker = '  gap(r14)={:+.2f}  gap(r26)={:+.2f}'.format(g14, g26)

    print('  {} | r14s26={:>6.2f} | r26s26={:>6.2f}{}{}'.format(
        utc_str, row['k_r14_s26'], row['k_r26_s26'], tv_str, marker))

print()
print('=== 3-point gap summary ===')
total_r14 = 0
total_r26 = 0
for utc_str, tv in tv_vals.items():
    rows = subset[subset['utc'].dt.strftime('%Y-%m-%d %H:%M') == utc_str]
    if len(rows) == 0:
        continue
    r = rows.iloc[0]
    g14 = abs(r['k_r14_s26'] - tv)
    g26 = abs(r['k_r26_s26'] - tv)
    total_r14 += g14
    total_r26 += g26
    print('  {}:  RSI(14)+Stoch(26) gap={:.2f}  |  RSI(26)+Stoch(26) gap={:.2f}'.format(
        utc_str, g14, g26))

print()
print('  Total gap RSI(14): {:.2f}'.format(total_r14))
print('  Total gap RSI(26): {:.2f}'.format(total_r26))
print()
if total_r26 < total_r14:
    print('  >>> RSI(26)+Stoch(26) WIN! gap {:.1f}x smaller'.format(total_r14 / total_r26))
