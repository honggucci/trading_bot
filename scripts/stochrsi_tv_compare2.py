"""
StochRSI TradingView 갭 원인 분석 2
가설: 데이터 시프트 (UTC+8 변환 오류 or candle open/close 혼동)
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
df = df.sort_values('timestamp').reset_index(drop=True)

# 원본 timestamp 그대로 + UTC 변환
df['utc'] = df['timestamp'] - pd.Timedelta(hours=8)

# ===== TEST 1: 실제 Close 가격 비교 =====
print('=' * 80)
print('TEST 1: 데이터 Close 가격 확인')
print('       TradingView에서 같은 봉의 Close 가격과 일치하는지?')
print('=' * 80)
print()

# UTC 기준으로 필터
mask = (df['utc'] >= '2025-06-04 16:00') & (df['utc'] <= '2025-06-05 04:00')
subset = df[mask][['timestamp', 'utc', 'open', 'high', 'low', 'close']]

print('  {:>20s} → {:>20s} | {:>10s} {:>10s} {:>10s} {:>10s}'.format(
    'Raw TS (UTC+8)', 'UTC', 'Open', 'High', 'Low', 'Close'))
print('  ' + '-' * 95)
for _, row in subset.iterrows():
    raw = row['timestamp'].strftime('%Y-%m-%d %H:%M')
    utc = row['utc'].strftime('%Y-%m-%d %H:%M')
    print('  {:>20s} -> {:>20s} | {:>10,.1f} {:>10,.1f} {:>10,.1f} {:>10,.1f}'.format(
        raw, utc, row['open'], row['high'], row['low'], row['close']))

print()
print('  >> TradingView BTCUSDT.P 1h 차트에서 위 UTC 시간의 OHLC를 대조하세요.')
print('  >> Close가 다르면 = 데이터 자체가 다름 (심볼/소스 차이)')
print('  >> Close가 같으면 = 계산 로직 차이 (warm-up 등)')

# ===== TEST 2: 시프트 가설 테스트 =====
print()
print('=' * 80)
print('TEST 2: 1-bar 시프트 가설 (timestamp가 close time일 가능성)')
print('=' * 80)
print()

# 가설: timestamp = candle close time (not open)
# 즉, UTC+8 04:00 timestamp = 03:00~04:00 캔들 = UTC -8h = UTC 20:00
# → 정상적으로 UTC 20:00 캔들의 close
# vs
# 가설2: timestamp = candle open time
# 즉, UTC+8 04:00 timestamp = 04:00~05:00 캔들 = UTC -8h = UTC 20:00
# → 이건 실제로 UTC 20:00에 열린 캔들 = TradingView의 20:00 캔들
# 두 경우 다 같은 결과 (close는 그 캔들의 마지막 가격)

# BUT: 만약 timestamp가 close time이고 candle은 1h 전에 시작했다면:
# raw ts 2025-06-05 04:00 (UTC+8) = 이 캔들은 03:00~04:00 UTC+8 = UTC 19:00~20:00
# 우리가 utc = 04:00 - 8h = 20:00 으로 표시
# 실제로는 UTC 19:00에 열린 캔들 → TradingView의 19:00 봉
# → 1bar shift!

# 1-bar shifted 계산
df_shift = df.copy()
df_shift['utc_shifted'] = df_shift['utc'] - pd.Timedelta(hours=1)  # 1봉 앞당김

close_arr = df_shift['close'].values.astype(np.float64)
rsi14 = talib.RSI(close_arr, timeperiod=14)
rsi_s = pd.Series(rsi14)

def stoch_rsi_k(rsi_series, stoch_period, k_smooth=3):
    lo = rsi_series.rolling(stoch_period, min_periods=stoch_period).min()
    hi = rsi_series.rolling(stoch_period, min_periods=stoch_period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    stoch = (rsi_series - lo) / denom
    stoch = stoch.clip(lower=0.0, upper=1.0).fillna(0.0)
    k = stoch.rolling(k_smooth, min_periods=k_smooth).mean() * 100.0
    return k

k_26_shift = stoch_rsi_k(rsi_s, 26, 3)
df_shift['k26_shifted'] = k_26_shift.values

# 원래 (shift 없음)도 같이 계산
close_orig = df['close'].values.astype(np.float64)
rsi14_orig = talib.RSI(close_orig, timeperiod=14)
rsi_s_orig = pd.Series(rsi14_orig)
k_26_orig = stoch_rsi_k(rsi_s_orig, 26, 3)
df['k26'] = k_26_orig.values

tv_vals = {
    '2025-06-04 20:00': 17.99,
    '2025-06-04 21:00': 10.40,
    '2025-06-04 22:00': 14.46,
}

# Shifted에서 비교
print('  원래 UTC vs 1bar-shifted UTC 비교:')
print()

for utc_str, tv in tv_vals.items():
    # 원래: utc 칼럼 매칭
    orig_row = df[df['utc'].dt.strftime('%Y-%m-%d %H:%M') == utc_str]
    # Shifted: utc_shifted 칼럼 매칭
    shift_row = df_shift[df_shift['utc_shifted'].dt.strftime('%Y-%m-%d %H:%M') == utc_str]

    k_orig = orig_row.iloc[0]['k26'] if len(orig_row) > 0 else float('nan')
    k_shift = shift_row.iloc[0]['k26_shifted'] if len(shift_row) > 0 else float('nan')

    print('  TV {} = {:.2f}'.format(utc_str, tv))
    print('    Original K(26) = {:.2f}  gap={:+.2f}'.format(k_orig, k_orig - tv))
    print('    Shifted  K(26) = {:.2f}  gap={:+.2f}'.format(k_shift, k_shift - tv))
    print()

# ===== TEST 3: 다양한 파라미터 조합 시뮬레이션 =====
print('=' * 80)
print('TEST 3: 파라미터 조합 brute-force (TV=17.99 기준)')
print('=' * 80)
print()

close_arr = df['close'].values.astype(np.float64)
target_utc = '2025-06-04 20:00'
target_val = 17.99

# RSI periods to try
rsi_periods = [14]
stoch_periods = [14, 20, 25, 26, 30]
k_smooths = [1, 3, 5]

best_gap = 999
best_params = None

idx = df[df['utc'].dt.strftime('%Y-%m-%d %H:%M') == target_utc].index[0]

for rsi_p in rsi_periods:
    rsi = talib.RSI(close_arr, timeperiod=rsi_p)
    rsi_ss = pd.Series(rsi)

    for stoch_p in stoch_periods:
        for k_s in k_smooths:
            kk = stoch_rsi_k(rsi_ss, stoch_p, k_s)
            val = kk.iloc[idx]
            gap = abs(val - target_val)
            if gap < best_gap:
                best_gap = gap
                best_params = (rsi_p, stoch_p, k_s, val)

            if gap < 3.0:
                print('  RSI={}, Stoch={}, K={}: K={:.2f}  gap={:+.2f}'.format(
                    rsi_p, stoch_p, k_s, val, val - target_val))

print()
print('  Best match: RSI={}, Stoch={}, K={} -> K={:.2f}  gap={:+.2f}'.format(
    best_params[0], best_params[1], best_params[2], best_params[3], best_params[3] - target_val))

# ===== TEST 4: RSI 자체 값 비교 (TradingView RSI 검증) =====
print()
print('=' * 80)
print('TEST 4: Raw RSI(14) 값 확인 (TradingView RSI 인디케이터와 비교용)')
print('=' * 80)
print()

rsi14_final = talib.RSI(close_arr, timeperiod=14)
df['rsi14'] = rsi14_final

mask2 = (df['utc'] >= '2025-06-04 18:00') & (df['utc'] <= '2025-06-05 02:00')
for _, row in df[mask2].iterrows():
    utc_str = row['utc'].strftime('%Y-%m-%d %H:%M')
    print('  {} | Close={:>10,.1f} | RSI(14)={:.2f}'.format(
        utc_str, row['close'], row['rsi14']))

print()
print('  >> TradingView에서 RSI(14) 값을 확인하세요.')
print('  >> RSI가 동일하면 = Stochastic 변환 문제')
print('  >> RSI가 다르면 = Close 가격 or 데이터 소스 문제')
