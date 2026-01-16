"""
ATR 윈도우 비판적 검증

의심:
1. Pearson만 보면 길수록 유리 (스무딩)
2. 반응성(lag)은 어떤가?
3. OOS에서도 유효한가?
4. 다른 목적함수(MAE, 캘리브레이션)에서는?
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

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

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


def calculate_lag(atr_values, future_range):
    """
    ATR 변화가 실제 변동성 변화를 얼마나 늦게 반영하는가
    교차상관으로 최적 lag 찾기
    """
    valid = ~(np.isnan(atr_values) | np.isnan(future_range))
    a = atr_values[valid]
    r = future_range[valid]

    # 정규화
    a = (a - a.mean()) / a.std()
    r = (r - r.mean()) / r.std()

    # 교차상관 (lag 0~20)
    best_lag = 0
    best_corr = 0

    for lag in range(0, 21):
        if lag == 0:
            corr = np.corrcoef(a, r)[0, 1]
        else:
            corr = np.corrcoef(a[:-lag], r[lag:])[0, 1]

        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def run_experiment():
    print("=" * 70)
    print("ATR 윈도우 비판적 검증")
    print("=" * 70)

    # 15m만 집중 분석 (가장 의심스러운 케이스)
    tf = '15m'
    lookahead = 32  # 8시간

    print(f"\nTF: {tf}, lookahead: {lookahead}봉")

    df = load_data(tf)
    if df is None:
        print("데이터 없음")
        return

    # 80/20 분할
    split_idx = int(len(df) * 0.8)
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()

    print(f"IS: {len(df_is):,}봉, OOS: {len(df_oos):,}봉")

    h_is, l_is, c_is = df_is['high'].values, df_is['low'].values, df_is['close'].values
    h_oos, l_oos, c_oos = df_oos['high'].values, df_oos['low'].values, df_oos['close'].values

    future_is = calculate_future_range(h_is, l_is, lookahead)
    future_oos = calculate_future_range(h_oos, l_oos, lookahead)

    windows = [8, 13, 21, 34, 55, 89, 144]

    print("\n" + "=" * 70)
    print("1. IS vs OOS 비교 (과적합 체크)")
    print("=" * 70)

    results = []
    for w in windows:
        atr_is = atr(h_is, l_is, c_is, w)
        atr_oos = atr(h_oos, l_oos, c_oos, w)

        # IS에서 k 찾기
        k = find_k_for_coverage(atr_is, future_is, target=0.50)

        # IS 성능
        zone_is = atr_is * k * 2
        valid_is = ~(np.isnan(zone_is) | np.isnan(future_is))
        corr_is, _ = pearsonr(zone_is[valid_is], future_is[valid_is])
        mae_is = np.mean(np.abs(zone_is[valid_is] - future_is[valid_is]))

        # OOS 성능 (IS에서 찾은 k 사용)
        zone_oos = atr_oos * k * 2
        valid_oos = ~(np.isnan(zone_oos) | np.isnan(future_oos))
        corr_oos, _ = pearsonr(zone_oos[valid_oos], future_oos[valid_oos])
        mae_oos = np.mean(np.abs(zone_oos[valid_oos] - future_oos[valid_oos]))

        # 커버리지 (OOS)
        coverage_oos = (zone_oos[valid_oos] >= future_oos[valid_oos]).mean() * 100

        # Lag
        lag, _ = calculate_lag(atr_is, future_is)

        results.append({
            'window': w,
            'k': k,
            'corr_is': corr_is,
            'corr_oos': corr_oos,
            'corr_drop': (corr_is - corr_oos) / corr_is * 100,
            'mae_is': mae_is,
            'mae_oos': mae_oos,
            'coverage_oos': coverage_oos,
            'lag': lag,
        })

    print(f"\n{'Window':>8} {'k':>6} {'IS Corr':>10} {'OOS Corr':>10} {'Drop%':>8} {'Lag':>5} {'OOS Cov':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['window']:>8} {r['k']:>6.2f} {r['corr_is']:>10.4f} {r['corr_oos']:>10.4f} "
              f"{r['corr_drop']:>7.1f}% {r['lag']:>5} {r['coverage_oos']:>7.1f}%")

    # ==========================================================================
    # 2. 최적 윈도우 판정 (OOS 기준)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. 최적 윈도우 판정")
    print("=" * 70)

    # IS 기준 순위
    is_ranked = sorted(results, key=lambda x: x['corr_is'], reverse=True)
    print("\nIS 기준 TOP3:")
    for i, r in enumerate(is_ranked[:3], 1):
        print(f"  {i}. ATR({r['window']}): {r['corr_is']:.4f}")

    # OOS 기준 순위
    oos_ranked = sorted(results, key=lambda x: x['corr_oos'], reverse=True)
    print("\nOOS 기준 TOP3:")
    for i, r in enumerate(oos_ranked[:3], 1):
        print(f"  {i}. ATR({r['window']}): {r['corr_oos']:.4f}")

    # IS vs OOS 일치 여부
    is_best = is_ranked[0]['window']
    oos_best = oos_ranked[0]['window']

    print(f"\nIS 최적: ATR({is_best})")
    print(f"OOS 최적: ATR({oos_best})")

    if is_best == oos_best:
        print("-> 일치! 과적합 우려 낮음")
    else:
        print("-> 불일치! IS에서 과적합 가능성")

    # ==========================================================================
    # 3. Lag vs Pearson 트레이드오프
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. Lag vs Pearson 트레이드오프")
    print("=" * 70)

    print("\n짧은 윈도우: 빠른 반응, 낮은 상관")
    print("긴 윈도우: 느린 반응, 높은 상관")

    # Lag가 0인 것 중 OOS 상관 최고
    low_lag = [r for r in results if r['lag'] <= 2]
    if low_lag:
        best_low_lag = max(low_lag, key=lambda x: x['corr_oos'])
        print(f"\nLag <= 2 중 OOS 최고: ATR({best_low_lag['window']}), OOS Corr={best_low_lag['corr_oos']:.4f}")

    # ==========================================================================
    # 4. 결론
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. 결론")
    print("=" * 70)

    # OOS 기준으로 판정
    best = oos_ranked[0]
    second = oos_ranked[1]

    diff = (best['corr_oos'] - second['corr_oos']) / second['corr_oos'] * 100

    print(f"\nOOS 1등: ATR({best['window']}), Corr={best['corr_oos']:.4f}")
    print(f"OOS 2등: ATR({second['window']}), Corr={second['corr_oos']:.4f}")
    print(f"차이: {diff:.1f}%")

    if diff < 2:
        print("\n-> 1등과 2등 차이 미미 (<2%)")
        print(f"   Lag 고려하면 더 짧은 ATR({min(best['window'], second['window'])}) 선택 권장")
    else:
        print(f"\n-> ATR({best['window']}) 선택")

    return results


if __name__ == "__main__":
    run_experiment()
