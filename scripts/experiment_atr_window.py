"""
ATR 윈도우 최적화 실험

윈도우 후보: 7, 14, 21, 34, 50, 89 (피보나치 수열 포함)

평가: 동일 커버리지(50%)에서 Pearson, MAE 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import brentq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_1h_data() -> pd.DataFrame:
    data_path = Path(r"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\bronze\binance\futures\BTC-USDT\1h")

    all_dfs = []
    for year_dir in sorted(data_path.iterdir()):
        if year_dir.is_dir():
            for parquet_file in sorted(year_dir.glob("*.parquet")):
                df = pd.read_parquet(parquet_file)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    combined = combined.drop_duplicates(subset='timestamp', keep='first')

    return combined


def atr(high, low, close, window):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    n = len(tr)
    atr_values = np.full(n, np.nan)

    for i in range(window - 1, n):
        atr_values[i] = np.mean(tr[i - window + 1:i + 1])

    return atr_values


def calculate_future_range(df, lookahead=24):
    future_high = df['high'].rolling(lookahead).max().shift(-lookahead)
    future_low = df['low'].rolling(lookahead).min().shift(-lookahead)
    return future_high - future_low


def find_k_for_coverage(vol, future_range, target=0.50):
    valid = ~(np.isnan(vol) | np.isnan(future_range))
    v = vol[valid]
    r = future_range[valid]

    def coverage_at_k(k):
        return (v * k * 2 >= r).mean() - target

    try:
        return brentq(coverage_at_k, 0.1, 20.0)
    except:
        return 2.5


def run_experiment():
    print("=" * 70)
    print("ATR 윈도우 최적화 실험")
    print("=" * 70)

    df = load_1h_data()
    print(f"데이터: {len(df):,}개 1시간봉")

    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    df['future_range'] = calculate_future_range(df, lookahead=24)
    future_range = df['future_range'].values

    # 윈도우 후보 (피보나치 수열 포함)
    windows = [5, 8, 13, 14, 21, 34, 50, 89]

    print("\n" + "=" * 70)
    print("윈도우별 성능 비교 (커버리지 50% 맞춤)")
    print("=" * 70)

    results = []
    for w in windows:
        atr_values = atr(h, l, c, window=w)

        # 커버리지 50%에 필요한 k 찾기
        k = find_k_for_coverage(atr_values, future_range, target=0.50)

        # 성능 평가
        zone_width = atr_values * k * 2
        valid = ~(np.isnan(zone_width) | np.isnan(future_range))

        corr, _ = pearsonr(zone_width[valid], future_range[valid])
        mae = np.mean(np.abs(zone_width[valid] - future_range[valid]))

        # 비율 분포
        ratio = zone_width[valid] / future_range[valid]
        ratio_10 = np.percentile(ratio, 10)
        ratio_90 = np.percentile(ratio, 90)
        spread = ratio_90 - ratio_10

        results.append({
            'window': w,
            'k': k,
            'pearson': corr,
            'mae': mae,
            'spread': spread,
        })

        print(f"\n  ATR({w:>2}):")
        print(f"    k={k:.3f}, Pearson={corr:.4f}, MAE=${mae:,.0f}")
        print(f"    10-90%ile 폭={spread:.3f}")

    # 순위
    print("\n" + "=" * 70)
    print("순위 (Pearson 기준)")
    print("=" * 70)

    results.sort(key=lambda x: x['pearson'], reverse=True)

    for i, r in enumerate(results, 1):
        print(f"  {i}. ATR({r['window']:>2}): Pearson={r['pearson']:.4f}, k={r['k']:.3f}")

    best = results[0]
    print(f"\n최적 윈도우: ATR({best['window']}), k={best['k']:.2f}")

    # ATR(14) 대비 비교
    atr14 = next(r for r in results if r['window'] == 14)
    diff = (best['pearson'] - atr14['pearson']) / atr14['pearson'] * 100

    print(f"\nATR(14) 대비: {diff:+.1f}%")

    if abs(diff) < 2:
        print("-> 차이 미미, ATR(14) 유지해도 무방")
    elif diff > 5:
        print(f"-> ATR({best['window']})로 변경 권장")
    else:
        print("-> 약간 개선, 변경 고려")

    return results


if __name__ == "__main__":
    run_experiment()
