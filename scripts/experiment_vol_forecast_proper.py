"""
변동성 예측력 검증 (A단계) - 제대로 된 버전

목표: 다음 h바의 실현변동성을 예측

후보 모델:
1. ATR
2. EWMA σ (RiskMetrics식)
3. Yang-Zhang σ
4. Realized Vol (rolling)
5. Hilbert amp (causal 버전)

평가: MAE, MSE, 상관계수 (OOS 검증)

핵심: 룩어헤드 없이 t 시점 정보만으로 t+h 변동성 예측
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
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


def hilbert_causal(signal: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Causal 힐베르트 amplitude (FIR 근사)

    Non-causal FFT 기반 hilbert() 대신,
    rolling window로 causal 근사.

    방법: 롤링 윈도우 내에서만 FFT 힐베르트 적용
    → 미래 데이터 사용 안 함
    """
    from scipy.signal import hilbert

    n = len(signal)
    amplitude = np.full(n, np.nan)

    for i in range(window - 1, n):
        # t-window+1 ~ t 까지만 사용 (미래 데이터 없음)
        segment = signal[i - window + 1:i + 1]
        analytic = hilbert(segment)
        # 마지막 값만 사용 (현재 시점)
        amplitude[i] = np.abs(analytic[-1])

    return amplitude


def calculate_future_realized_vol(returns: np.ndarray, h: int = 24) -> np.ndarray:
    """
    향후 h봉의 실현변동성 (정답 레이블)

    RV = sqrt(sum(r_i^2)) for i in [t+1, t+h]
    """
    n = len(returns)
    future_rv = np.full(n, np.nan)

    for i in range(n - h):
        # t+1 ~ t+h (미래 h봉)
        future_returns = returns[i + 1:i + 1 + h]
        future_rv[i] = np.sqrt(np.sum(future_returns ** 2))

    return future_rv


def run_experiment():
    print("=" * 70)
    print("변동성 예측력 검증 (A단계) - 룩어헤드 없는 버전")
    print("=" * 70)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    df = load_1h_data()
    print(f"로드 완료: {len(df):,}개 1시간봉")

    o = df['open'].values
    h_price = df['high'].values
    l_price = df['low'].values
    c = df['close'].values

    # Log returns
    returns = np.diff(np.log(c), prepend=np.nan)
    returns[0] = 0

    # ==========================================================================
    # 정답 레이블: 향후 h봉 실현변동성
    # ==========================================================================
    h = 24  # 24봉 (1H * 24 = 1일)

    print(f"\n정답 레이블: 향후 {h}봉 실현변동성")
    future_rv = calculate_future_realized_vol(returns, h=h)

    # ==========================================================================
    # 예측 모델들 (t 시점 정보만 사용)
    # ==========================================================================
    print("\n예측 모델 계산 중...")

    # 1. ATR (달러 → 비율로 변환)
    atr_14 = atr(h_price, l_price, c, window=14)
    atr_ratio = atr_14 / c  # ATR을 비율로 변환

    # 2. EWMA σ
    ewma_vol = ewma_vol_from_prices(c, lambda_=0.94)

    # 3. Yang-Zhang σ
    yz_vol = yang_zhang_volatility(o, h_price, l_price, c, window=14)

    # 4. Realized Vol (과거 window)
    rv_14 = realized_volatility(c, window=14)

    # 5. Hilbert amplitude (CAUSAL 버전)
    print("  Causal 힐베르트 계산 중 (느림)...")

    # Fib 스케일 변환 (detrend)
    FIB_0 = 3120
    RANGE = 17530
    fib_scale = (c - FIB_0) / RANGE

    # 1차 차분 (추세 제거)
    fib_diff = np.diff(fib_scale, prepend=fib_scale[0])

    # Causal 힐베르트
    hilbert_amp = hilbert_causal(fib_diff, window=50)

    # 달러 → 비율 변환
    hilbert_ratio = (hilbert_amp * RANGE) / c

    # ==========================================================================
    # 예측력 평가 (전체 기간)
    # ==========================================================================
    df_eval = pd.DataFrame({
        'future_rv': future_rv,
        'atr_ratio': atr_ratio,
        'ewma_vol': ewma_vol,
        'yz_vol': yz_vol,
        'rv_14': rv_14,
        'hilbert_ratio': hilbert_ratio,
    }).dropna()

    print(f"\n평가 데이터: {len(df_eval):,}개")

    print("\n" + "=" * 70)
    print("예측력 평가 (전체 기간)")
    print("=" * 70)

    models = ['atr_ratio', 'ewma_vol', 'yz_vol', 'rv_14', 'hilbert_ratio']

    results = []
    for model in models:
        pred = df_eval[model].values
        actual = df_eval['future_rv'].values

        # 스케일 맞추기 (예측값을 실제값 스케일로)
        # 단순 선형 회귀로 스케일 조정
        slope = np.cov(pred, actual)[0, 1] / np.var(pred)
        intercept = np.mean(actual) - slope * np.mean(pred)
        pred_scaled = slope * pred + intercept

        mae = np.mean(np.abs(pred_scaled - actual))
        mse = np.mean((pred_scaled - actual) ** 2)
        rmse = np.sqrt(mse)
        corr, _ = pearsonr(pred, actual)

        results.append({
            'model': model,
            'corr': corr,
            'mae': mae,
            'rmse': rmse,
        })

        print(f"\n  {model}:")
        print(f"    상관계수: {corr:.4f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")

    # ==========================================================================
    # OOS 검증 (마지막 20% 데이터)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("OOS 검증 (마지막 20% 데이터)")
    print("=" * 70)

    split_idx = int(len(df_eval) * 0.8)
    df_oos = df_eval.iloc[split_idx:]

    print(f"\nOOS 데이터: {len(df_oos):,}개")

    oos_results = []
    for model in models:
        pred = df_oos[model].values
        actual = df_oos['future_rv'].values

        # 스케일 조정 (IS 데이터로 학습한 걸 OOS에 적용해야 하지만, 단순화)
        slope = np.cov(pred, actual)[0, 1] / np.var(pred)
        intercept = np.mean(actual) - slope * np.mean(pred)
        pred_scaled = slope * pred + intercept

        mae = np.mean(np.abs(pred_scaled - actual))
        corr, _ = pearsonr(pred, actual)

        oos_results.append({
            'model': model,
            'corr': corr,
            'mae': mae,
        })

        print(f"\n  {model}:")
        print(f"    OOS 상관계수: {corr:.4f}")
        print(f"    OOS MAE: {mae:.6f}")

    # ==========================================================================
    # 순위 및 결론
    # ==========================================================================
    print("\n" + "=" * 70)
    print("순위 (OOS 상관계수 기준)")
    print("=" * 70)

    oos_results.sort(key=lambda x: x['corr'], reverse=True)

    for i, r in enumerate(oos_results, 1):
        print(f"  {i}. {r['model']}: corr={r['corr']:.4f}")

    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    best = oos_results[0]['model']
    hilbert_rank = next(i for i, r in enumerate(oos_results, 1) if r['model'] == 'hilbert_ratio')
    hilbert_corr = next(r['corr'] for r in oos_results if r['model'] == 'hilbert_ratio')
    best_corr = oos_results[0]['corr']

    if hilbert_rank == 1:
        print(f"\n힐베르트가 1등! (corr={hilbert_corr:.4f})")
        print("→ Zone 폭에 활용 가치 있음")
    elif hilbert_corr >= best_corr * 0.9:
        print(f"\n힐베르트 {hilbert_rank}등 (corr={hilbert_corr:.4f})")
        print(f"1등({best})과 10% 이내 차이")
        print("→ 보조 지표로 활용 가능")
    else:
        print(f"\n힐베르트 {hilbert_rank}등 (corr={hilbert_corr:.4f})")
        print(f"1등 {best} (corr={best_corr:.4f})에 크게 뒤짐")
        print("→ 예쁜 수학이었을 뿐. ATR/EWMA 쓰셈.")

    return df_eval


if __name__ == "__main__":
    run_experiment()
