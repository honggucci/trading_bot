"""
Zone Coverage 모니터링

재학습 트리거 조건:
- 3개월 롤링 Coverage가 40% 이하 or 60% 이상이면 경고
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_config():
    config_path = Path(r"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\config\zone_width.json")
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


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


def check_coverage():
    config = load_config()

    tf_lookahead = {
        '1d': 5,
        '4h': 12,
        '1h': 24,
        '15m': 32,
    }

    # 3개월 = 약 2160봉 (15m), 540봉 (1h), 540봉 (4h), 90봉 (1d)
    rolling_bars = {
        '1d': 90,
        '4h': 540,
        '1h': 540,
        '15m': 2160,
    }

    print("=" * 60)
    print("Zone Coverage 모니터링")
    print("=" * 60)
    print(f"경고 기준: Coverage < 40% or > 60%")
    print()

    alerts = []

    for tf in ['15m', '1h', '4h', '1d']:
        tf_config = config['tf_params'].get(tf)
        if not tf_config or tf_config.get('atr_window') is None:
            continue

        df = load_data(tf)
        if df is None:
            continue

        window = tf_config['atr_window']
        k = tf_config['k']
        la = tf_lookahead[tf]

        h, l, c = df['high'].values, df['low'].values, df['close'].values

        atr_vals = atr(h, l, c, window)
        fr = future_range(h, l, la)

        zone_width = atr_vals * k * 2

        # 최근 3개월만
        recent = rolling_bars[tf]
        zone_recent = zone_width[-recent:]
        fr_recent = fr[-recent:]

        valid = ~(np.isnan(zone_recent) | np.isnan(fr_recent))
        if valid.sum() < 10:
            continue

        coverage = (zone_recent[valid] >= fr_recent[valid]).mean() * 100

        status = "OK"
        if coverage < 40 or coverage > 60:
            status = "ALERT"
            alerts.append(tf)

        print(f"{tf:>4}: Coverage = {coverage:5.1f}%  [{status}]")

    print()
    if alerts:
        print("=" * 60)
        print("재학습 검토 필요:")
        for tf in alerts:
            print(f"  - {tf}")
        print()
        print("실행: python scripts/experiment_atr_all_tf_oos.py")
        print("=" * 60)
        return False
    else:
        print("모든 TF 정상 범위 내")
        return True


if __name__ == "__main__":
    check_coverage()
