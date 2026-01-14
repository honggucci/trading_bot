"""
HMM Emission Features
=====================

5차원 Emission Features 계산.

Features:
1. price_position: 박스 내 가격 위치 (0-1)
2. volume_ratio: log(vol/ma_vol) robust z-score
3. volatility: ATR/close * 100
4. trend_strength: (EMA50-EMA200)/close * 100
5. range_width_norm: box_width / ATR

Origin: WPCN run_step8_hmm_train.py compute_emission_features()
"""
import numpy as np
import pandas as pd
from typing import Optional

try:
    from ..context.zigzag import wilder_atr
except ImportError:
    # Fallback: 직접 ATR 계산
    def wilder_atr(high, low, close, period=14):
        """Wilder's ATR 계산"""
        import numpy as np
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        atr_arr = np.zeros(len(close))
        atr_arr[period-1] = np.mean(tr[1:period])
        for i in range(period, len(close)):
            atr_arr[i] = (atr_arr[i-1] * (period - 1) + tr[i]) / period
        return atr_arr


def compute_emission_features(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    box_lookback: int = 24,
    ema_short: int = 50,
    ema_long: int = 200,
    vol_ma_period: int = 20,
) -> pd.DataFrame:
    """
    5차원 Emission Features 계산

    Args:
        df: OHLCV DataFrame
        atr_period: ATR 기간
        box_lookback: 박스 계산용 lookback
        ema_short: 단기 EMA 기간
        ema_long: 장기 EMA 기간
        vol_ma_period: Volume MA 기간

    Returns:
        DataFrame with columns:
        - price_position: 박스 내 가격 위치 (0-1)
        - volume_ratio: log(vol/ma_vol) robust z-score
        - volatility: ATR/close * 100
        - trend_strength: (EMA50-EMA200)/close * 100
        - range_width_norm: box_width / ATR
    """
    # ATR 계산
    _atr = wilder_atr(
        df['high'].values,
        df['low'].values,
        df['close'].values,
        period=atr_period,
    )

    # EMA
    ema_s = df['close'].ewm(span=ema_short, adjust=False).mean()
    ema_l = df['close'].ewm(span=ema_long, adjust=False).mean()

    # Volume MA
    vol_ma = df['volume'].rolling(vol_ma_period, min_periods=5).mean()

    # 간단한 박스 (lookback 기간의 고저)
    box_high = df['high'].rolling(box_lookback, min_periods=5).max()
    box_low = df['low'].rolling(box_lookback, min_periods=5).min()
    box_width = box_high - box_low

    features = pd.DataFrame(index=df.index)

    # 1. Price Position: (close - box_low) / box_width
    features['price_position'] = (df['close'] - box_low) / (box_width + 1e-10)
    features['price_position'] = features['price_position'].clip(0, 1)

    # 2. Volume Ratio: log(volume / ma_volume)
    vol_ratio = df['volume'] / (vol_ma + 1e-10)
    vol_ratio_raw = np.log(vol_ratio + 1e-10)
    # Robust z-score (median/MAD)
    median = pd.Series(vol_ratio_raw).rolling(100, min_periods=20).median()
    mad = (pd.Series(vol_ratio_raw) - median).abs().rolling(100, min_periods=20).median()
    features['volume_ratio'] = (vol_ratio_raw - median) / (mad * 1.4826 + 1e-10)
    features['volume_ratio'] = features['volume_ratio'].clip(-3, 3)

    # 3. Volatility: ATR / close * 100
    features['volatility'] = (_atr / df['close'].values) * 100
    features['volatility'] = features['volatility'].clip(0, 10)

    # 4. Trend Strength: (EMA_short - EMA_long) / close * 100
    features['trend_strength'] = ((ema_s - ema_l) / df['close']) * 100
    features['trend_strength'] = features['trend_strength'].clip(-10, 10)

    # 5. Range Width Norm: box_width / ATR
    features['range_width_norm'] = box_width / (_atr + 1e-10)
    features['range_width_norm'] = features['range_width_norm'].clip(0, 10)

    return features


FEATURE_COLS = [
    'price_position',
    'volume_ratio',
    'volatility',
    'trend_strength',
    'range_width_norm',
]
