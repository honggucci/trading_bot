"""
BTC 데이터 업데이트 스크립트
============================

기존 parquet 데이터에서 마지막 timestamp 확인 후
바이낸스 API로 최신 데이터 가져와서 병합.

사용법:
    python scripts/update_btc_data.py

환경변수:
    BINANCE_API_KEY (선택)
    BINANCE_API_SECRET (선택)
"""
import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

# 기본 경로
BASE_PATH = Path(__file__).parent.parent / "data" / "bronze" / "binance" / "futures" / "BTC-USDT"

# 바이낸스 API
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"

# TF → 바이낸스 interval
TF_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

# TF → ms 변환
TF_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}


def get_last_timestamp(tf: str) -> Optional[pd.Timestamp]:
    """해당 TF의 마지막 timestamp 가져오기"""
    tf_path = BASE_PATH / tf
    if not tf_path.exists():
        return None

    # 최신 연도/월 찾기
    years = sorted([d for d in os.listdir(tf_path) if d.isdigit()], reverse=True)
    if not years:
        return None

    for year in years:
        year_path = tf_path / year
        months = sorted([m.replace('.parquet', '') for m in os.listdir(year_path) if m.endswith('.parquet')], reverse=True)
        if months:
            latest_file = year_path / f"{months[0]}.parquet"
            df = pd.read_parquet(latest_file)
            if 'timestamp' in df.columns:
                return pd.to_datetime(df['timestamp'].iloc[-1])
    return None


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List[dict]:
    """바이낸스 API에서 klines 가져오기"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit,
    }

    try:
        resp = requests.get(BINANCE_FUTURES_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] API 호출 실패: {e}")
        return []


def fetch_all_klines(symbol: str, interval: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """start_ts부터 end_ts까지 모든 klines 가져오기"""
    all_data = []
    current_start = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    while current_start < end_ms:
        klines = fetch_klines(symbol, interval, current_start, end_ms)
        if not klines:
            break

        all_data.extend(klines)

        # 다음 시작점
        last_open_time = klines[-1][0]
        current_start = last_open_time + 1

        # Rate limit
        time.sleep(0.1)

        if len(klines) < 1000:
            break

    if not all_data:
        return pd.DataFrame()

    # DataFrame 변환
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # 필요한 컬럼만
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.drop('open_time', axis=1)

    # 타입 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 컬럼 순서
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return df


def save_to_parquet(df: pd.DataFrame, tf: str):
    """월별로 parquet 저장"""
    if df.empty:
        return

    # 월별 그룹화
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    for (year, month), group in df.groupby(['year', 'month']):
        year_path = BASE_PATH / tf / str(year)
        year_path.mkdir(parents=True, exist_ok=True)

        parquet_file = year_path / f"{month:02d}.parquet"

        # 기존 데이터와 병합
        group_data = group.drop(['year', 'month'], axis=1).copy()

        if parquet_file.exists():
            existing = pd.read_parquet(parquet_file)
            combined = pd.concat([existing, group_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            combined = combined.sort_values('timestamp').reset_index(drop=True)
        else:
            combined = group_data.sort_values('timestamp').reset_index(drop=True)

        combined.to_parquet(parquet_file, index=False)
        print(f"    Saved: {parquet_file} ({len(combined)} rows)")


def update_timeframe(tf: str, end_date: datetime):
    """특정 TF 업데이트"""
    print(f"\n[{tf}] 업데이트 중...")

    last_ts = get_last_timestamp(tf)
    if last_ts is None:
        print(f"  기존 데이터 없음. 스킵.")
        return

    print(f"  마지막 데이터: {last_ts}")

    # 다음 봉부터 시작
    tf_ms = TF_MS.get(tf, 60000)
    start_ts = last_ts + pd.Timedelta(milliseconds=tf_ms)
    end_ts = pd.Timestamp(end_date)

    if start_ts >= end_ts:
        print(f"  이미 최신. 스킵.")
        return

    print(f"  가져올 범위: {start_ts} ~ {end_ts}")

    # 데이터 가져오기
    interval = TF_MAP.get(tf, tf)
    df = fetch_all_klines("BTCUSDT", interval, start_ts, end_ts)

    if df.empty:
        print(f"  새 데이터 없음.")
        return

    print(f"  가져온 데이터: {len(df)}개 봉")

    # 저장
    save_to_parquet(df, tf)


def main():
    print("=" * 60)
    print("BTC 데이터 업데이트")
    print("=" * 60)

    # 목표 날짜 (2026-01-15 23:59:59)
    end_date = datetime(2026, 1, 16, 0, 0, 0)

    print(f"\n목표: {end_date}")

    # 모든 TF 업데이트
    for tf in ["5m", "15m", "1h", "4h", "1d", "1w"]:
        update_timeframe(tf, end_date)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
