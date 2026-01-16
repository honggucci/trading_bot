"""
Zone + Divergence 백테스트 엔진
================================

멀티 TF 구조:
- 15m 매매: Anchor=15m, Trigger=5m
- 1H 매매: Anchor=1H, Trigger=15m
- 4H 매매: Anchor=4H, Trigger=1H
- 1D 매매: Anchor=1D, Trigger=4H

시작일: 2021-10-01
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Tuple
from datetime import datetime

from src.zone import (
    ZoneBuilder,
    FibZone,
    get_div_zones,
    get_all_entry_zones,
    get_entry_signal,
    ClusterZone,
    FibOnlyZone,
    DivZone,
    calc_rsi,
)
from src.trigger import alt_trigger_ok, trigger_ok
from src.context import atr


# =============================================================================
# 데이터 로드
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "bronze" / "binance" / "futures" / "BTC-USDT"


def load_tf_data(tf: str, start_date: str = "2021-10-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """TF별 데이터 로드"""
    tf_dir = DATA_DIR / tf
    if not tf_dir.exists():
        raise FileNotFoundError(f"TF directory not found: {tf_dir}")

    dfs = []
    for year_dir in sorted(tf_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for parquet_file in sorted(year_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {parquet_file}: {e}")

    if not dfs:
        raise ValueError(f"No data found for TF={tf}")

    df = pd.concat(dfs, ignore_index=True)

    # timestamp 컬럼을 인덱스로 설정
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

    df = df.sort_index()

    # 날짜 필터
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]

    # 컬럼 정리
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna()

    return df


# =============================================================================
# 백테스트 결과
# =============================================================================

@dataclass
class Trade:
    """단일 거래"""
    entry_ts: pd.Timestamp
    entry_price: float
    side: Literal['long', 'short']
    grade: Literal['A', 'B']
    size_mult: float

    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ''
    pnl_pct: float = 0.0

    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0


@dataclass
class BacktestResult:
    """백테스트 결과"""
    tf: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def a_grade_trades(self) -> int:
        return len([t for t in self.trades if t.grade == 'A'])

    @property
    def b_grade_trades(self) -> int:
        return len([t for t in self.trades if t.grade == 'B'])

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = len([t for t in self.trades if t.pnl_pct > 0])
        return wins / len(self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.pnl_pct for t in self.trades])

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    def summary(self) -> Dict:
        return {
            'tf': self.tf,
            'total_trades': self.total_trades,
            'a_grade': self.a_grade_trades,
            'b_grade': self.b_grade_trades,
            'win_rate': f"{self.win_rate:.1%}",
            'avg_pnl': f"{self.avg_pnl:.2%}",
            'total_pnl': f"{self.total_pnl:.2%}",
        }


# =============================================================================
# 백테스트 엔진
# =============================================================================

@dataclass
class BacktestConfig:
    """백테스트 설정"""
    # ATR 기반 SL/TP
    atr_period: int = 21
    atr_sl_mult: float = 1.5
    atr_tp1_mult: float = 2.5  # 1.5 → 2.5 (R:R 개선)
    atr_tp2_mult: float = 2.5
    atr_tp3_mult: float = 3.5

    # 최대 보유 시간 (봉 수)
    max_hold_bars: int = 32

    # Zone 설정
    max_distance_pct: float = 0.03  # 현재가 ±3% 내 Zone만

    # B급 트리거
    zscore_long_threshold: float = -2.0
    zscore_short_threshold: float = 2.0


class ZoneDivBacktester:
    """Zone + Divergence 백테스터"""

    def __init__(
        self,
        anchor_tf: str,
        trigger_tf: str,
        config: Optional[BacktestConfig] = None,
    ):
        self.anchor_tf = anchor_tf
        self.trigger_tf = trigger_tf
        self.config = config or BacktestConfig()
        self.zone_builder = ZoneBuilder()

    def run(self, start_date: str = "2021-10-01", end_date: str = "2025-12-31") -> BacktestResult:
        """백테스트 실행"""
        print(f"\n[{self.anchor_tf}] 백테스트 시작 ({start_date} ~ {end_date})")

        # 데이터 로드
        df_anchor = load_tf_data(self.anchor_tf, start_date, end_date)
        df_trigger = load_tf_data(self.trigger_tf, start_date, end_date)

        print(f"  Anchor ({self.anchor_tf}): {len(df_anchor)} bars")
        print(f"  Trigger ({self.trigger_tf}): {len(df_trigger)} bars")

        # ATR 계산
        df_anchor['atr'] = atr(
            df_anchor['high'].values,
            df_anchor['low'].values,
            df_anchor['close'].values,
            window=self.config.atr_period,
        )

        # RSI 계산
        df_anchor['rsi'] = calc_rsi(df_anchor['close'].values, period=14)

        result = BacktestResult(tf=self.anchor_tf)
        position: Optional[Trade] = None
        warmup = max(100, self.config.atr_period * 2)

        # 메인 루프 (Anchor TF 기준)
        anchor_timestamps = df_anchor.index[warmup:]

        for i, ts in enumerate(anchor_timestamps):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(anchor_timestamps)}")

            # 현재 Anchor 데이터
            anchor_idx = df_anchor.index.get_loc(ts)
            df_up_to_now = df_anchor.iloc[:anchor_idx + 1]

            current_price = float(df_anchor.loc[ts, 'close'])
            current_atr = float(df_anchor.loc[ts, 'atr'])

            if not np.isfinite(current_atr) or current_atr <= 0:
                continue

            # === 포지션 관리 ===
            if position is not None:
                # Trigger TF에서 SL/TP 체크
                trigger_bars = df_trigger[
                    (df_trigger.index > position.entry_ts) &
                    (df_trigger.index <= ts)
                ]

                for t_ts, t_bar in trigger_bars.iterrows():
                    t_high = float(t_bar['high'])
                    t_low = float(t_bar['low'])

                    # SL 체크
                    if position.side == 'long' and t_low <= position.sl_price:
                        position.exit_ts = t_ts
                        position.exit_price = position.sl_price
                        position.exit_reason = 'SL'
                        position.pnl_pct = (position.sl_price - position.entry_price) / position.entry_price * position.size_mult
                        result.trades.append(position)
                        position = None
                        break
                    elif position.side == 'short' and t_high >= position.sl_price:
                        position.exit_ts = t_ts
                        position.exit_price = position.sl_price
                        position.exit_reason = 'SL'
                        position.pnl_pct = (position.entry_price - position.sl_price) / position.entry_price * position.size_mult
                        result.trades.append(position)
                        position = None
                        break

                    # TP1 체크 (간단화: TP1만 체크)
                    if position.side == 'long' and t_high >= position.tp1_price:
                        position.exit_ts = t_ts
                        position.exit_price = position.tp1_price
                        position.exit_reason = 'TP1'
                        position.pnl_pct = (position.tp1_price - position.entry_price) / position.entry_price * position.size_mult
                        result.trades.append(position)
                        position = None
                        break
                    elif position.side == 'short' and t_low <= position.tp1_price:
                        position.exit_ts = t_ts
                        position.exit_price = position.tp1_price
                        position.exit_reason = 'TP1'
                        position.pnl_pct = (position.entry_price - position.tp1_price) / position.entry_price * position.size_mult
                        result.trades.append(position)
                        position = None
                        break

                # Time-stop
                if position is not None:
                    bars_held = anchor_idx - df_anchor.index.get_loc(
                        df_anchor.index[df_anchor.index <= position.entry_ts][-1]
                    )
                    if bars_held >= self.config.max_hold_bars:
                        position.exit_ts = ts
                        position.exit_price = current_price
                        position.exit_reason = 'TIME'
                        if position.side == 'long':
                            position.pnl_pct = (current_price - position.entry_price) / position.entry_price * position.size_mult
                        else:
                            position.pnl_pct = (position.entry_price - current_price) / position.entry_price * position.size_mult
                        result.trades.append(position)
                        position = None

                continue  # 포지션 있으면 신규 진입 안함

            # === 신규 진입 ===

            # Fib Zone 계산
            atr_dict = {self.anchor_tf: current_atr}
            price_range = (current_price * 0.9, current_price * 1.1)

            try:
                fib_zones = self.zone_builder.build_zones(
                    current_atr=atr_dict,
                    max_depth=1,
                    price_range=price_range,
                    timeframe=self.anchor_tf,
                )
            except Exception:
                continue

            # Divergence Zone 계산
            try:
                div_zones = get_div_zones(df_up_to_now, rsi_period=14)
            except Exception:
                div_zones = []

            # Cluster Zone 계산
            a_zones, b_zones = get_all_entry_zones(
                fib_zones, div_zones, current_price,
                max_distance_pct=self.config.max_distance_pct,
            )

            # Trigger TF 데이터 (현재 Anchor 봉까지)
            df_trigger_now = df_trigger[df_trigger.index <= ts]
            if len(df_trigger_now) < 30:
                continue

            # A급 트리거 확인 (Trigger TF 사용)
            a_trigger_long = trigger_ok(df_trigger_now, 'long')
            a_trigger_short = trigger_ok(df_trigger_now, 'short')

            # B급 트리거 확인 (Anchor TF 사용)
            b_trigger_long = alt_trigger_ok(
                df_up_to_now, 'long',
                long_threshold=self.config.zscore_long_threshold,
            )
            b_trigger_short = alt_trigger_ok(
                df_up_to_now, 'short',
                short_threshold=self.config.zscore_short_threshold,
            )

            # Long 신호 확인
            signal = get_entry_signal(
                a_zones, b_zones, current_price,
                b_trigger_ok=b_trigger_long,
                side_filter='long',
            )

            # A급이면 A급 트리거 필수
            if signal is not None and signal.grade == 'A' and not a_trigger_long:
                signal = None

            if signal is None:
                # Short 신호 확인
                signal = get_entry_signal(
                    a_zones, b_zones, current_price,
                    b_trigger_ok=b_trigger_short,
                    side_filter='short',
                )

                # A급이면 A급 트리거 필수
                if signal is not None and signal.grade == 'A' and not a_trigger_short:
                    signal = None

            if signal is None:
                continue

            # 현재가가 Zone 안에 있는지 확인
            if not (signal.entry_low <= current_price <= signal.entry_high):
                continue

            # 진입
            if signal.side == 'long':
                sl_price = current_price - current_atr * self.config.atr_sl_mult
                tp1_price = current_price + current_atr * self.config.atr_tp1_mult
                tp2_price = current_price + current_atr * self.config.atr_tp2_mult
                tp3_price = current_price + current_atr * self.config.atr_tp3_mult
            else:
                sl_price = current_price + current_atr * self.config.atr_sl_mult
                tp1_price = current_price - current_atr * self.config.atr_tp1_mult
                tp2_price = current_price - current_atr * self.config.atr_tp2_mult
                tp3_price = current_price - current_atr * self.config.atr_tp3_mult

            position = Trade(
                entry_ts=ts,
                entry_price=current_price,
                side=signal.side,
                grade=signal.grade,
                size_mult=signal.size_mult,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                tp3_price=tp3_price,
            )

        # 미청산 포지션 처리
        if position is not None:
            last_price = float(df_anchor.iloc[-1]['close'])
            position.exit_ts = df_anchor.index[-1]
            position.exit_price = last_price
            position.exit_reason = 'END'
            if position.side == 'long':
                position.pnl_pct = (last_price - position.entry_price) / position.entry_price * position.size_mult
            else:
                position.pnl_pct = (position.entry_price - last_price) / position.entry_price * position.size_mult
            result.trades.append(position)

        return result


# =============================================================================
# 메인
# =============================================================================

def main():
    """멀티 TF 백테스트 실행"""
    print("=" * 70)
    print("Zone + Divergence 백테스트")
    print("=" * 70)

    # TF 설정
    tf_pairs = [
        ('1h', '15m'),    # 1H 매매: Anchor=1H, Trigger=15m
        # ('4h', '1h'),     # 4H 매매
        # ('1d', '4h'),     # 1D 매매
        # ('15m', '5m'),  # 15m 매매 (데이터 많아서 느림)
    ]

    results = []

    for anchor_tf, trigger_tf in tf_pairs:
        backtester = ZoneDivBacktester(anchor_tf, trigger_tf)
        result = backtester.run(start_date="2021-10-01", end_date="2022-03-31")

        print(f"\n[{anchor_tf}] 결과:")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")

        results.append(result)

    # A급 vs B급 비교
    print("\n" + "=" * 70)
    print("A급 vs B급 비교")
    print("=" * 70)

    for result in results:
        a_trades = [t for t in result.trades if t.grade == 'A']
        b_trades = [t for t in result.trades if t.grade == 'B']

        a_win = len([t for t in a_trades if t.pnl_pct > 0]) / len(a_trades) if a_trades else 0
        b_win = len([t for t in b_trades if t.pnl_pct > 0]) / len(b_trades) if b_trades else 0

        a_avg = np.mean([t.pnl_pct for t in a_trades]) if a_trades else 0
        b_avg = np.mean([t.pnl_pct for t in b_trades]) if b_trades else 0

        print(f"\n[{result.tf}]")
        print(f"  A급: {len(a_trades)} trades, WR {a_win:.1%}, Avg {a_avg:.2%}")
        print(f"  B급: {len(b_trades)} trades, WR {b_win:.1%}, Avg {b_avg:.2%}")


if __name__ == "__main__":
    main()
