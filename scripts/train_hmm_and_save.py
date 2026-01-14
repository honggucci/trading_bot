# -*- coding: utf-8 -*-
"""
HMM Training & Posterior Map Export
===================================

WPCN HMM 학습 코드 기반으로 posterior_map 생성 및 저장.

출력:
- models/hmm_model.pkl: 학습된 GaussianHMM
- models/posterior_map.pkl: {timestamp: posterior_array}
- models/features_df.pkl: 15m features DataFrame
"""
import sys
import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# trading_bot 경로 추가
TRADING_BOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(TRADING_BOT_PATH / "src"))

# HMM 모듈 import
from hmm.states import HMM_STATES, STATE_TO_IDX, IDX_TO_STATE, UNCERTAINTY_THRESHOLD
from hmm.features import FEATURE_COLS
from hmm.train import train_hmm, compute_initial_params

# 출력 디렉토리
OUTPUT_DIR = TRADING_BOT_PATH / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_btc_data(days: int = 365) -> pd.DataFrame:
    """Binance에서 BTC 15분봉 데이터 가져오기"""
    try:
        import ccxt
    except ImportError:
        print("[ERROR] ccxt not installed. Run: pip install ccxt")
        return pd.DataFrame()

    print(f"Fetching BTC/USDT 15m data ({days} days)...")
    exchange = ccxt.binance({'enableRateLimit': True})

    all_data = []
    limit = 1000
    total_bars = days * 96  # 15분봉 = 하루 96개

    # 과거부터 현재까지
    since = exchange.parse8601((datetime.now() - pd.Timedelta(days=days)).isoformat())

    while len(all_data) < total_bars:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # 다음 batch
            print(f"  Fetched {len(all_data)} bars...")
        except Exception as e:
            print(f"  Error: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    print(f"Total: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """HMM emission features 계산"""
    df = df.copy()

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1)),
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Volume MA
    df['volume_ma'] = df['volume'].rolling(20).mean()

    # Features
    # 1. price_position: 최근 박스 내 위치
    box_high = df['high'].rolling(24).max()
    box_low = df['low'].rolling(24).min()
    box_width = box_high - box_low
    df['price_position'] = (df['close'] - box_low) / box_width.replace(0, 1)
    df['price_position'] = df['price_position'].clip(0, 1)

    # 2. volume_ratio: log(volume / ma_volume)
    df['volume_ratio'] = np.log1p(df['volume'] / df['volume_ma'].replace(0, 1))
    df['volume_ratio'] = df['volume_ratio'].clip(-3, 3)

    # 3. volatility: ATR / close * 100
    df['volatility'] = df['atr'] / df['close'] * 100

    # 4. trend_strength: (EMA50 - EMA200) / close * 100
    df['trend_strength'] = (df['ema50'] - df['ema200']) / df['close'] * 100

    # 5. range_width_norm: box_width / ATR
    df['range_width_norm'] = box_width / df['atr'].replace(0, 1)
    df['range_width_norm'] = df['range_width_norm'].clip(0, 10)

    return df


def create_rule_labels(df: pd.DataFrame) -> np.ndarray:
    """Rule-based Wyckoff 라벨 생성 (약지도 초기화용)"""
    labels = []

    for i in range(len(df)):
        row = df.iloc[i]
        trend = row.get('trend_strength', 0)
        pos = row.get('price_position', 0.5)
        vol_ratio = row.get('volume_ratio', 0)

        # 간단한 규칙 기반 라벨링
        if trend > 2.0:  # 강한 상승 추세
            label = 'markup'
        elif trend < -2.0:  # 강한 하락 추세
            label = 'markdown'
        elif pos > 0.7 and vol_ratio > 0.5:  # 고점 부근 + 거래량 증가
            label = 'distribution'
        elif pos < 0.3 and vol_ratio > 0.5:  # 저점 부근 + 거래량 증가
            label = 'accumulation'
        elif pos > 0.5 and trend > 0:
            label = 're_distribution' if trend < 1.0 else 'markup'
        elif pos < 0.5 and trend < 0:
            label = 're_accumulation' if trend > -1.0 else 'markdown'
        else:
            # 불확실하면 가장 가까운 상태
            label = 'accumulation' if pos < 0.5 else 'distribution'

        labels.append(label)

    return np.array(labels)


def train_and_save():
    """HMM 학습 및 저장"""
    print("=" * 60)
    print("HMM Training Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    df = fetch_btc_data(days=365)
    if df.empty:
        print("[ERROR] No data fetched")
        return False

    # 2. Features 계산
    print("\nComputing features...")
    df = compute_features(df)

    # NaN 제거
    df = df.dropna()
    print(f"After dropna: {len(df)} bars")

    # 3. Rule-based 라벨 생성
    print("\nCreating rule-based labels...")
    labels = create_rule_labels(df)

    # 라벨 분포
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} ({c/len(labels)*100:.1f}%)")

    # 4. Feature matrix 추출
    feature_cols = ['price_position', 'volume_ratio', 'volatility', 'trend_strength', 'range_width_norm']
    X = df[feature_cols].values
    print(f"\nFeature matrix: {X.shape}")

    # 5. HMM 학습
    print("\nTraining HMM...")
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("[ERROR] hmmlearn not installed. Run: pip install hmmlearn")
        return False

    model = train_hmm(X, labels, lengths=[len(X)], n_iter=100, sticky=True)

    if model is None:
        print("[ERROR] HMM training failed")
        return False

    print("[OK] HMM trained successfully")

    # 6. Posterior 계산
    print("\nComputing posteriors...")
    posteriors = model.predict_proba(X)
    decoded_states = model.predict(X)

    # posterior_map 생성
    posterior_map = {}
    for i, ts in enumerate(df.index):
        posterior_map[ts] = posteriors[i]

    print(f"Posterior map size: {len(posterior_map)}")

    # 7. 저장
    print("\nSaving models...")

    # HMM 모델
    with open(OUTPUT_DIR / "hmm_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"  [OK] {OUTPUT_DIR / 'hmm_model.pkl'}")

    # Posterior map
    with open(OUTPUT_DIR / "posterior_map.pkl", 'wb') as f:
        pickle.dump(posterior_map, f)
    print(f"  [OK] {OUTPUT_DIR / 'posterior_map.pkl'}")

    # Features DataFrame
    features_df = df[feature_cols + ['trend_strength', 'atr']].copy()
    with open(OUTPUT_DIR / "features_df.pkl", 'wb') as f:
        pickle.dump(features_df, f)
    print(f"  [OK] {OUTPUT_DIR / 'features_df.pkl'}")

    # 8. 검증
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    # 상태 분포
    decoded_labels = [IDX_TO_STATE.get(s, 'range') for s in decoded_states]
    unique, counts = np.unique(decoded_labels, return_counts=True)
    print("\nDecoded state distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} ({c/len(decoded_labels)*100:.1f}%)")

    # 최대 확률 분포
    max_probs = posteriors.max(axis=1)
    print(f"\nMax probability stats:")
    print(f"  Mean: {max_probs.mean():.3f}")
    print(f"  Min: {max_probs.min():.3f}")
    print(f"  Uncertain (< {UNCERTAINTY_THRESHOLD}): {(max_probs < UNCERTAINTY_THRESHOLD).sum()} bars")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = train_and_save()
    sys.exit(0 if success else 1)