"""
StochRSI TradingView vs talib 비교 스크립트
근본 원인 분석: ~5pt 갭
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

close = df['close'].values.astype(np.float64)

# RSI(14) via talib
rsi14 = talib.RSI(close, timeperiod=14)
rsi_s = pd.Series(rsi14)

def stoch_rsi_k(rsi_series, stoch_period, k_smooth=3):
    """StochRSI %K = SMA(Stoch(RSI, period), k_smooth)"""
    lo = rsi_series.rolling(stoch_period, min_periods=stoch_period).min()
    hi = rsi_series.rolling(stoch_period, min_periods=stoch_period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi_series - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)
    k = stoch.rolling(k_smooth, min_periods=k_smooth).mean() * 100.0
    return k

# 여러 stochastic period로 비교
k_26 = stoch_rsi_k(rsi_s, 26, 3)
k_14 = stoch_rsi_k(rsi_s, 14, 3)

df['stoch_k_26'] = k_26.values
df['stoch_k_14'] = k_14.values
df['rsi14'] = rsi14

# TradingView 참조값 (user 제공)
tv_vals = {
    '2025-06-04 20:00': 17.99,
    '2025-06-04 21:00': 10.40,
    '2025-06-04 22:00': 14.46,
}

# ===== TEST 1: Stochastic Period 차이 =====
print('=' * 80)
print('TEST 1: Stochastic Period 차이 (26 vs 14)')
print('=' * 80)
print()

mask = (df['utc'] >= '2025-06-04 18:00') & (df['utc'] <= '2025-06-05 06:00')
subset = df[mask][['utc', 'close', 'rsi14', 'stoch_k_26', 'stoch_k_14']]

for _, row in subset.iterrows():
    utc_str = row['utc'].strftime('%Y-%m-%d %H:%M')
    tv = tv_vals.get(utc_str, None)

    line = '  {} | Close={:>10,.1f} | RSI={:>6.2f} | K(26)={:>6.2f}'.format(
        utc_str, row['close'], row['rsi14'], row['stoch_k_26'])
    line += ' | K(14)={:>6.2f}'.format(row['stoch_k_14'])

    if tv is not None:
        line += ' | TV={:.2f} | gap(26)={:+.1f} | gap(14)={:+.1f}'.format(
            tv, row['stoch_k_26'] - tv, row['stoch_k_14'] - tv)
    print(line)

print()

# ===== TEST 2: TradingView 방식 RSI (RMA) 직접 구현 =====
print('=' * 80)
print('TEST 2: TradingView Pine Script RMA 기반 RSI 직접 구현')
print('=' * 80)
print()

def rma(values, period):
    """TradingView ta.rma (= Wilder's EMA)
    First value = SMA, then EMA with alpha=1/period
    """
    alpha = 1.0 / period
    result = np.full(len(values), np.nan)

    # Find first non-NaN index
    start = 0
    for i in range(len(values)):
        if not np.isnan(values[i]):
            start = i
            break

    # First value = SMA of first `period` valid values
    valid_count = 0
    sma_sum = 0.0
    sma_idx = -1
    for i in range(start, len(values)):
        if not np.isnan(values[i]):
            sma_sum += values[i]
            valid_count += 1
            if valid_count == period:
                sma_idx = i
                break

    if sma_idx == -1:
        return result

    result[sma_idx] = sma_sum / period

    # EMA from there
    for i in range(sma_idx + 1, len(values)):
        if not np.isnan(values[i]):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        else:
            result[i] = result[i - 1]

    return result

def tv_rsi(close_arr, period=14):
    """TradingView Pine Script ta.rsi 구현
    change = close - close[1]
    up = rma(max(change, 0), period)
    down = rma(max(-change, 0), period)
    rsi = 100 - 100 / (1 + up/down)
    """
    change = np.diff(close_arr, prepend=np.nan)
    up = np.where(change > 0, change, 0.0)
    down = np.where(change < 0, -change, 0.0)
    up[0] = np.nan
    down[0] = np.nan

    avg_up = rma(up, period)
    avg_down = rma(down, period)

    rs = np.where(avg_down != 0, avg_up / avg_down, 100.0)
    rsi_val = np.where(np.isnan(avg_up) | np.isnan(avg_down), np.nan, 100 - 100 / (1 + rs))
    return rsi_val

# TV RSI 계산
tv_rsi14 = tv_rsi(close, 14)
tv_rsi_s = pd.Series(tv_rsi14)

# TV 방식 StochRSI
tv_k_26 = stoch_rsi_k(tv_rsi_s, 26, 3)
tv_k_14 = stoch_rsi_k(tv_rsi_s, 14, 3)

df['tv_rsi14'] = tv_rsi14
df['tv_k_26'] = tv_k_26.values
df['tv_k_14'] = tv_k_14.values

subset2 = df[mask][['utc', 'close', 'rsi14', 'tv_rsi14', 'stoch_k_26', 'tv_k_26', 'stoch_k_14', 'tv_k_14']]

print('RSI 비교 (talib vs TV-RMA):')
for _, row in subset2.iterrows():
    utc_str = row['utc'].strftime('%Y-%m-%d %H:%M')
    rsi_diff = row['tv_rsi14'] - row['rsi14'] if not np.isnan(row['tv_rsi14']) else float('nan')
    print('  {} | talib RSI={:.2f} | TV RSI={:.2f} | diff={:+.2f}'.format(
        utc_str, row['rsi14'], row['tv_rsi14'], rsi_diff))

print()
print('StochRSI %K 비교:')
print('  {:>20s} | {:>12s} | {:>12s} | {:>12s} | {:>12s} | {:>6s}'.format(
    'UTC', 'talib K(26)', 'TV-RMA K(26)', 'talib K(14)', 'TV-RMA K(14)', 'TV ref'))
print('  ' + '-' * 85)

for _, row in subset2.iterrows():
    utc_str = row['utc'].strftime('%Y-%m-%d %H:%M')
    tv = tv_vals.get(utc_str, None)
    tv_str = '{:.2f}'.format(tv) if tv is not None else ''

    print('  {:>20s} | {:>12.2f} | {:>12.2f} | {:>12.2f} | {:>12.2f} | {:>6s}'.format(
        utc_str, row['stoch_k_26'], row['tv_k_26'], row['stoch_k_14'], row['tv_k_14'], tv_str))

# TV 참조값과의 갭 비교
print()
print('=== TV 참조값과의 갭 (절대값 작을수록 일치) ===')
for utc_str, tv in tv_vals.items():
    row = subset2[subset2['utc'].dt.strftime('%Y-%m-%d %H:%M') == utc_str]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    print('  {}  TV={:.2f}'.format(utc_str, tv))
    print('    talib K(26): {:.2f}  gap={:+.2f}'.format(r['stoch_k_26'], r['stoch_k_26'] - tv))
    print('    TV-RMA K(26): {:.2f}  gap={:+.2f}'.format(r['tv_k_26'], r['tv_k_26'] - tv))
    print('    talib K(14): {:.2f}  gap={:+.2f}'.format(r['stoch_k_14'], r['stoch_k_14'] - tv))
    print('    TV-RMA K(14): {:.2f}  gap={:+.2f}'.format(r['tv_k_14'], r['tv_k_14'] - tv))
    print()

# ===== TEST 3: 데이터 시작점 영향 =====
print('=' * 80)
print('TEST 3: 데이터 시작점 (warm-up) 영향')
print('=' * 80)
first_utc = df['utc'].iloc[0]
last_utc = df['utc'].iloc[-1]
total_bars = len(df)
print('  데이터 범위: {} ~ {}'.format(first_utc, last_utc))
print('  총 바 수: {:,}'.format(total_bars))
print('  RSI warm-up: 14 bars')
print('  Stoch(26) warm-up: 26 bars')
print('  K smooth: 3 bars')
print('  최소 warm-up: 14 + 26 + 3 = 43 bars')
print()

# talib RSI가 처음 valid한 인덱스
first_valid_rsi = np.argmax(~np.isnan(rsi14))
print('  talib RSI first valid index: {} (bar {})'.format(first_valid_rsi, first_valid_rsi))
print('  TV RMA RSI first valid index: {}'.format(np.argmax(~np.isnan(tv_rsi14))))

# RSI 수렴 확인 (마지막 1000바에서)
rsi_diff_last1000 = rsi14[-1000:] - tv_rsi14[-1000:]
print('  RSI diff (last 1000 bars): mean={:.4f}, max={:.4f}, min={:.4f}'.format(
    np.nanmean(rsi_diff_last1000), np.nanmax(rsi_diff_last1000), np.nanmin(rsi_diff_last1000)))
