"""
공정 비교: 동일 커버리지로 맞춘 후 비교

문제:
- ATR(k=2.4)은 48% 커버리지
- σ 기반(k=2.0)은 10~12% 커버리지
→ 비교가 불공정

해결:
- 각 모델의 k를 튜닝해서 커버리지 50%로 맞춤
- 그 상태에서 Pearson, MAE, 캘리브레이션 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import brentq
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from context.volatility import (
    atr,
    ewma_vol_from_prices,
    yang_zhang_volatility,
    realized_volatility,
)


def load_1h_data() -> pd.DataFrame:
    """1시간봉 데이터 로드"""
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


def calculate_future_range(df: pd.DataFrame, lookahead: int = 24) -> pd.Series:
    """향후 N봉 동안의 실제 가격 범위"""
    future_high = df['high'].rolling(lookahead).max().shift(-lookahead)
    future_low = df['low'].rolling(lookahead).min().shift(-lookahead)
    return future_high - future_low


def find_k_for_coverage(vol_estimate, price, future_range, target_coverage=0.50, is_dollar=False):
    """
    목표 커버리지를 달성하는 k 찾기

    커버리지 = (zone_width * 2 >= future_range)인 비율
    """
    valid_mask = ~(np.isnan(vol_estimate) | np.isnan(future_range))
    vol_valid = vol_estimate[valid_mask]
    price_valid = price[valid_mask]
    range_valid = future_range[valid_mask]

    def coverage_at_k(k):
        if is_dollar:
            # ATR: width = ATR * k
            zone_width = vol_valid * k
        else:
            # σ 기반: width = P * σ * k (√h는 k에 포함)
            zone_width = price_valid * vol_valid * k

        coverage = (zone_width * 2 >= range_valid).mean()
        return coverage - target_coverage

    # k 범위에서 이분탐색
    try:
        k_optimal = brentq(coverage_at_k, 0.1, 20.0)
    except ValueError:
        # 범위 내에서 해가 없으면 끝값 사용
        if coverage_at_k(0.1) > 0:
            k_optimal = 0.1
        else:
            k_optimal = 20.0

    return k_optimal


def run_experiment():
    print("=" * 70)
    print("공정 비교: 동일 커버리지(50%)로 맞춘 후 비교")
    print("=" * 70)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    df = load_1h_data()
    print(f"로드 완료: {len(df):,}개 1시간봉")

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    # 변동성 추정치
    df['atr_14'] = atr(h, l, c, window=14)
    df['ewma_vol'] = ewma_vol_from_prices(c, lambda_=0.94)
    df['yz_vol'] = yang_zhang_volatility(o, h, l, c, window=14)
    df['rv_14'] = realized_volatility(c, window=14)

    # 향후 24봉 실제 범위
    df['future_range'] = calculate_future_range(df, lookahead=24)

    # NaN 제거
    df_clean = df.dropna().copy()
    print(f"유효 데이터: {len(df_clean):,}개")

    future_range = df_clean['future_range'].values
    price = df_clean['close'].values

    # ==========================================================================
    # 1. 각 모델의 k 찾기 (커버리지 50% 맞추기)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. 커버리지 50% 맞추는 k 찾기")
    print("=" * 70)

    models = {
        'atr': {'vol': df_clean['atr_14'].values, 'is_dollar': True},
        'ewma': {'vol': df_clean['ewma_vol'].values, 'is_dollar': False},
        'yz': {'vol': df_clean['yz_vol'].values, 'is_dollar': False},
        'rv': {'vol': df_clean['rv_14'].values, 'is_dollar': False},
    }

    optimal_k = {}
    for name, data in models.items():
        k = find_k_for_coverage(
            data['vol'], price, future_range,
            target_coverage=0.50, is_dollar=data['is_dollar']
        )
        optimal_k[name] = k

        # 검증
        if data['is_dollar']:
            zone_width = data['vol'] * k
        else:
            zone_width = price * data['vol'] * k

        actual_coverage = (zone_width * 2 >= future_range).mean() * 100
        print(f"  {name}: k={k:.3f}, 실제 커버리지={actual_coverage:.1f}%")

    # ==========================================================================
    # 2. 동일 커버리지에서 성능 비교
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. 동일 커버리지(50%)에서 성능 비교")
    print("=" * 70)

    results = []
    for name, data in models.items():
        k = optimal_k[name]

        if data['is_dollar']:
            zone_width = data['vol'] * k
        else:
            zone_width = price * data['vol'] * k

        # 양방향 Zone 폭
        zone_total = zone_width * 2

        # Pearson 상관 (Zone 폭 vs 실제 범위)
        corr, _ = pearsonr(zone_total, future_range)

        # MAE (Zone 폭 - 실제 범위)
        mae = np.mean(np.abs(zone_total - future_range))

        # 비율 (Zone / 실제)
        ratio = zone_total / future_range
        ratio_median = np.median(ratio)
        ratio_10 = np.percentile(ratio, 10)
        ratio_90 = np.percentile(ratio, 90)

        # 과소추정 비율 (Zone < 실제)
        underestimate = (zone_total < future_range).mean() * 100

        results.append({
            'model': name,
            'k': k,
            'corr': corr,
            'mae': mae,
            'ratio_median': ratio_median,
            'ratio_10': ratio_10,
            'ratio_90': ratio_90,
            'underestimate': underestimate,
        })

        print(f"\n  {name} (k={k:.3f}):")
        print(f"    Pearson: {corr:.4f}")
        print(f"    MAE: ${mae:,.0f}")
        print(f"    비율 중앙값: {ratio_median:.3f}")
        print(f"    비율 10~90%ile: {ratio_10:.3f} ~ {ratio_90:.3f}")
        print(f"    과소추정: {underestimate:.1f}%")

    # ==========================================================================
    # 3. 순위
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. 순위 (Pearson 기준)")
    print("=" * 70)

    results.sort(key=lambda x: x['corr'], reverse=True)

    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['model']}: Pearson={r['corr']:.4f}, k={r['k']:.3f}")

    # ==========================================================================
    # 4. 캘리브레이션 (비율 분포)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. 캘리브레이션 분석")
    print("=" * 70)

    print("\n이상적 캘리브레이션: 비율 중앙값 ≈ 1.0, 분산 작음")
    print("\n  모델별 비율 분포 (Zone/실제):")

    for r in results:
        spread = r['ratio_90'] - r['ratio_10']
        cal_score = abs(r['ratio_median'] - 1.0) + spread / 10  # 낮을수록 좋음

        print(f"  {r['model']:>6}: 중앙값={r['ratio_median']:.3f}, "
              f"10-90%ile 폭={spread:.3f}, 캘리브레이션 점수={cal_score:.3f}")

    # ==========================================================================
    # 5. 결론
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. 결론")
    print("=" * 70)

    best_corr = results[0]
    best_cal = min(results, key=lambda x: abs(x['ratio_median'] - 1.0))

    print(f"\n  예측력(Pearson) 1등: {best_corr['model']} ({best_corr['corr']:.4f})")
    print(f"  캘리브레이션 1등: {best_cal['model']} (중앙값={best_cal['ratio_median']:.3f})")

    # ATR vs 나머지 차이
    atr_result = next(r for r in results if r['model'] == 'atr')
    sigma_results = [r for r in results if r['model'] != 'atr']

    atr_corr = atr_result['corr']
    avg_sigma_corr = np.mean([r['corr'] for r in sigma_results])

    diff_pct = (atr_corr - avg_sigma_corr) / avg_sigma_corr * 100

    print(f"\n  ATR Pearson: {atr_corr:.4f}")
    print(f"  σ 평균 Pearson: {avg_sigma_corr:.4f}")
    print(f"  ATR 우위: {diff_pct:+.1f}%")

    if diff_pct > 10:
        print("\n  → ATR이 σ 대비 유의미하게 우수 (10% 이상)")
    elif diff_pct > 5:
        print("\n  → ATR이 σ 대비 약간 우수 (5~10%)")
    else:
        print("\n  → ATR과 σ는 거의 동등 (5% 이내)")
        print("    σ 기반도 충분히 사용 가능")

    # 필요한 k 비교
    print("\n  동일 커버리지(50%)에 필요한 k:")
    for name in ['atr', 'ewma', 'yz', 'rv']:
        k = optimal_k[name]
        if name == 'atr':
            print(f"    {name}: {k:.2f}x ATR")
        else:
            # σ의 k는 √h를 포함한 값
            # k = 약 4.x는 √24 ≈ 4.9에 가까움
            equiv_h = (k / 2.0) ** 2  # k=2.0일 때 h=1 가정
            print(f"    {name}: {k:.2f}x σ (≈ 2.0 × √{equiv_h:.0f})")

    return results


if __name__ == "__main__":
    run_experiment()
