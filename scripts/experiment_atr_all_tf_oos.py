"""
모든 TF에서 IS vs OOS 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import brentq


def load_data(tf):
    data_path = Path(rf"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\bronze\binance\futures\BTC-USDT\{tf}")
    all_dfs = []
    for year_dir in sorted(data_path.iterdir()):
        if year_dir.is_dir():
            for f in sorted(year_dir.glob("*.parquet")):
                all_dfs.append(pd.read_parquet(f))
    if not all_dfs:
        return None
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def atr(high, low, close, window):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    n = len(tr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.mean(tr[i - window + 1:i + 1])
    return out


def future_range(high, low, lookahead):
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(n - lookahead):
        out[i] = np.max(high[i+1:i+1+lookahead]) - np.min(low[i+1:i+1+lookahead])
    return out


def find_k(vol, fr, target=0.50):
    valid = ~(np.isnan(vol) | np.isnan(fr))
    v, r = vol[valid], fr[valid]
    if len(v) == 0:
        return 2.5
    try:
        return brentq(lambda k: (v * k * 2 >= r).mean() - target, 0.1, 20.0)
    except:
        return 2.5


def run():
    print("=" * 80)
    print("모든 TF - IS vs OOS 검증")
    print("=" * 80)

    tf_config = {
        '1d': 5,
        '4h': 12,
        '1h': 24,
        '15m': 32,
    }

    windows = [13, 21, 34, 55, 89]

    for tf, la in tf_config.items():
        print(f"\n{'='*80}")
        print(f"TF: {tf.upper()}, lookahead: {la}봉")
        print("=" * 80)

        df = load_data(tf)
        if df is None:
            continue

        split = int(len(df) * 0.8)
        is_df, oos_df = df.iloc[:split], df.iloc[split:]

        h_is, l_is, c_is = is_df['high'].values, is_df['low'].values, is_df['close'].values
        h_oos, l_oos, c_oos = oos_df['high'].values, oos_df['low'].values, oos_df['close'].values

        fr_is = future_range(h_is, l_is, la)
        fr_oos = future_range(h_oos, l_oos, la)

        results = []
        for w in windows:
            atr_is = atr(h_is, l_is, c_is, w)
            atr_oos = atr(h_oos, l_oos, c_oos, w)

            k = find_k(atr_is, fr_is)

            zone_is = atr_is * k * 2
            zone_oos = atr_oos * k * 2

            valid_is = ~(np.isnan(zone_is) | np.isnan(fr_is))
            valid_oos = ~(np.isnan(zone_oos) | np.isnan(fr_oos))

            corr_is, _ = pearsonr(zone_is[valid_is], fr_is[valid_is])
            corr_oos, _ = pearsonr(zone_oos[valid_oos], fr_oos[valid_oos])

            results.append({
                'w': w,
                'k': k,
                'is': corr_is,
                'oos': corr_oos,
                'drop': (corr_is - corr_oos) / corr_is * 100,
            })

        print(f"\n{'W':>5} {'k':>6} {'IS':>8} {'OOS':>8} {'Drop':>7}")
        print("-" * 40)
        for r in results:
            print(f"{r['w']:>5} {r['k']:>6.2f} {r['is']:>8.4f} {r['oos']:>8.4f} {r['drop']:>6.1f}%")

        # IS vs OOS 최적 비교
        is_best = max(results, key=lambda x: x['is'])
        oos_best = max(results, key=lambda x: x['oos'])

        print(f"\nIS 최적: ATR({is_best['w']})")
        print(f"OOS 최적: ATR({oos_best['w']})")

        if is_best['w'] == oos_best['w']:
            print(f"-> 일치! ATR({is_best['w']}) 채택")
        else:
            # OOS 기준으로 선택
            print(f"-> 불일치. OOS 기준 ATR({oos_best['w']}) 채택")


if __name__ == "__main__":
    run()
