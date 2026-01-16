"""
ATR 윈도우 최적화 - 멀티타임프레임

각 TF별로 최적 윈도우 찾기
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import brentq


def load_data(tf: str) -> pd.DataFrame:
    data_path = Path(rf"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\bronze\binance\futures\BTC-USDT\{tf}")

    all_dfs = []
    for year_dir in sorted(data_path.iterdir()):
        if year_dir.is_dir():
            for parquet_file in sorted(year_dir.glob("*.parquet")):
                df = pd.read_parquet(parquet_file)
                all_dfs.append(df)

    if not all_dfs:
        return None

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


def calculate_future_range(high, low, lookahead):
    n = len(high)
    future_range = np.full(n, np.nan)

    for i in range(n - lookahead):
        future_range[i] = np.max(high[i+1:i+1+lookahead]) - np.min(low[i+1:i+1+lookahead])

    return future_range


def find_k_for_coverage(vol, future_range, target=0.50):
    valid = ~(np.isnan(vol) | np.isnan(future_range))
    v = vol[valid]
    r = future_range[valid]

    if len(v) == 0:
        return 2.5

    def coverage_at_k(k):
        return (v * k * 2 >= r).mean() - target

    try:
        return brentq(coverage_at_k, 0.1, 20.0)
    except:
        return 2.5


def run_experiment():
    print("=" * 70)
    print("ATR 윈도우 최적화 - 멀티타임프레임")
    print("=" * 70)

    # TF별 설정
    # lookahead: Zone 폭이 커버해야 할 향후 봉 수
    tf_config = {
        '1d': {'lookahead': 5},    # 5일
        '4h': {'lookahead': 12},   # 2일
        '1h': {'lookahead': 24},   # 1일
        '15m': {'lookahead': 32},  # 8시간
    }

    # 피보나치 수열 윈도우
    windows = [8, 13, 21, 34, 55, 89]

    all_results = {}

    for tf, config in tf_config.items():
        print(f"\n{'='*70}")
        print(f"TF: {tf.upper()}")
        print(f"{'='*70}")

        df = load_data(tf)
        if df is None:
            print(f"  데이터 없음")
            continue

        print(f"  데이터: {len(df):,}개 봉")

        h = df['high'].values
        l = df['low'].values
        c = df['close'].values

        future_range = calculate_future_range(h, l, config['lookahead'])

        results = []
        for w in windows:
            atr_values = atr(h, l, c, window=w)

            k = find_k_for_coverage(atr_values, future_range, target=0.50)

            zone_width = atr_values * k * 2
            valid = ~(np.isnan(zone_width) | np.isnan(future_range))

            if valid.sum() < 100:
                continue

            corr, _ = pearsonr(zone_width[valid], future_range[valid])
            mae = np.mean(np.abs(zone_width[valid] - future_range[valid]))

            results.append({
                'window': w,
                'k': k,
                'pearson': corr,
                'mae': mae,
            })

        if not results:
            continue

        # 정렬
        results.sort(key=lambda x: x['pearson'], reverse=True)

        print(f"\n  순위 (Pearson 기준):")
        for i, r in enumerate(results[:3], 1):
            print(f"    {i}. ATR({r['window']:>2}): Pearson={r['pearson']:.4f}, k={r['k']:.3f}")

        best = results[0]
        all_results[tf] = best
        print(f"\n  최적: ATR({best['window']}), k={best['k']:.2f}")

    # 전체 요약
    print("\n" + "=" * 70)
    print("전체 요약")
    print("=" * 70)

    print("\n  TF별 최적 ATR 윈도우:")
    for tf, result in all_results.items():
        # 실제 시간으로 환산
        tf_minutes = {'1d': 1440, '4h': 240, '1h': 60, '15m': 15}
        hours = result['window'] * tf_minutes[tf] / 60
        print(f"    {tf:>3}: ATR({result['window']:>2}) = {hours:>5.1f}시간, k={result['k']:.2f}")

    # 패턴 분석
    print("\n  패턴:")
    windows_list = [r['window'] for r in all_results.values()]

    if len(set(windows_list)) == 1:
        print(f"    모든 TF에서 ATR({windows_list[0]})가 최적")
    else:
        print(f"    TF별로 최적 윈도우가 다름")
        print(f"    윈도우: {windows_list}")

    return all_results


if __name__ == "__main__":
    run_experiment()
