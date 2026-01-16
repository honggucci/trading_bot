"""
Zone + Divergence 백테스트 v2
============================

wpcn-backtester 비용 모델 적용:
- Slippage: 5 bps
- Fee: 4 bps (taker)
- Funding: 0.01% per 8h

멀티 TF 구조:
- 15m 매매: Anchor=15m, Trigger=5m
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
# 비용 모델 (wpcn 참고)
# =============================================================================

@dataclass
class CostModel:
    """비용 모델"""
    fee_bps: float = 4.0          # Taker fee (0.04%)
    slippage_bps: float = 5.0     # Slippage (0.05%)
    funding_rate: float = 0.0001  # 0.01% per 8h
    funding_interval_bars_5m: int = 96  # 8h = 96 * 5m bars
    funding_interval_bars_15m: int = 32  # 8h = 32 * 15m bars

    def one_way_cost_pct(self) -> float:
        """편도 비용 (%)"""
        return (self.fee_bps + self.slippage_bps) / 10000

    def round_trip_cost_pct(self) -> float:
        """왕복 비용 (%)"""
        return self.one_way_cost_pct() * 2

    def apply_entry_cost(self, price: float, side: str) -> float:
        """진입 비용 적용"""
        cost_pct = self.one_way_cost_pct()
        if side == 'long':
            return price * (1 + cost_pct)  # 비싸게 삼
        else:
            return price * (1 - cost_pct)  # 싸게 팔았다 침

    def apply_exit_cost(self, price: float, side: str) -> float:
        """청산 비용 적용"""
        cost_pct = self.one_way_cost_pct()
        if side == 'long':
            return price * (1 - cost_pct)  # 싸게 팔림
        else:
            return price * (1 + cost_pct)  # 비싸게 사야됨

    def calc_funding_cost(self, position_value: float, hours_held: float, side: str) -> float:
        """펀딩비 계산 (Long은 지불, Short은 수취)"""
        funding_periods = hours_held / 8.0
        funding_cost = position_value * self.funding_rate * funding_periods
        if side == 'long':
            return funding_cost  # Long은 지불
        else:
            return -funding_cost  # Short은 수취


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
    entry_price: float  # 실제 체결가 (비용 포함)
    raw_entry_price: float  # 원래 가격
    side: Literal['long', 'short']
    grade: Literal['A', 'B']
    size_mult: float
    trigger_type: str = ''

    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None  # 실제 체결가 (비용 포함)
    raw_exit_price: Optional[float] = None  # 원래 가격
    exit_reason: str = ''
    pnl_pct: float = 0.0
    funding_cost: float = 0.0

    sl_price: float = 0.0
    tp1_price: float = 0.0


@dataclass
class BacktestResult:
    """백테스트 결과"""
    tf: str
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def long_trades(self) -> int:
        return len([t for t in self.trades if t.side == 'long'])

    @property
    def short_trades(self) -> int:
        return len([t for t in self.trades if t.side == 'short'])

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

    @property
    def total_funding(self) -> float:
        return sum(t.funding_cost for t in self.trades)

    def summary(self) -> Dict:
        return {
            'tf': self.tf,
            'total_trades': self.total_trades,
            'long': self.long_trades,
            'short': self.short_trades,
            'a_grade': self.a_grade_trades,
            'b_grade': self.b_grade_trades,
            'win_rate': f"{self.win_rate:.1%}",
            'avg_pnl': f"{self.avg_pnl:.2%}",
            'total_pnl': f"{self.total_pnl:.2%}",
            'funding': f"{self.total_funding:.2%}",
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
    atr_tp_mult: float = 2.5  # R:R = 1:1.67

    # 최대 보유 시간 (봉 수)
    max_hold_bars: int = 48  # 15m * 48 = 12시간

    # Zone 설정
    max_distance_pct: float = 0.02  # 현재가 ±2% 내 Zone만

    # 비용
    cost_model: CostModel = field(default_factory=CostModel)


class ZoneDivBacktesterV2:
    """Zone + Divergence 백테스터 v2"""

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

        cost = self.config.cost_model

        # 메인 루프 (Anchor TF 기준)
        anchor_timestamps = df_anchor.index[warmup:]

        for i, ts in enumerate(anchor_timestamps):
            if i % 500 == 0:
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
                    t_close = float(t_bar['close'])

                    # SL 체크
                    if position.side == 'long' and t_low <= position.sl_price:
                        exit_price = cost.apply_exit_cost(position.sl_price, 'long')
                        position.exit_ts = t_ts
                        position.exit_price = exit_price
                        position.raw_exit_price = position.sl_price
                        position.exit_reason = 'SL'

                        # 보유 시간 계산
                        hours_held = (t_ts - position.entry_ts).total_seconds() / 3600
                        position.funding_cost = cost.calc_funding_cost(1.0, hours_held, 'long') * position.size_mult

                        position.pnl_pct = ((exit_price - position.entry_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
                        result.trades.append(position)
                        position = None
                        break

                    elif position.side == 'short' and t_high >= position.sl_price:
                        exit_price = cost.apply_exit_cost(position.sl_price, 'short')
                        position.exit_ts = t_ts
                        position.exit_price = exit_price
                        position.raw_exit_price = position.sl_price
                        position.exit_reason = 'SL'

                        hours_held = (t_ts - position.entry_ts).total_seconds() / 3600
                        position.funding_cost = cost.calc_funding_cost(1.0, hours_held, 'short') * position.size_mult

                        position.pnl_pct = ((position.entry_price - exit_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
                        result.trades.append(position)
                        position = None
                        break

                    # TP 체크
                    if position.side == 'long' and t_high >= position.tp1_price:
                        exit_price = cost.apply_exit_cost(position.tp1_price, 'long')
                        position.exit_ts = t_ts
                        position.exit_price = exit_price
                        position.raw_exit_price = position.tp1_price
                        position.exit_reason = 'TP'

                        hours_held = (t_ts - position.entry_ts).total_seconds() / 3600
                        position.funding_cost = cost.calc_funding_cost(1.0, hours_held, 'long') * position.size_mult

                        position.pnl_pct = ((exit_price - position.entry_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
                        result.trades.append(position)
                        position = None
                        break

                    elif position.side == 'short' and t_low <= position.tp1_price:
                        exit_price = cost.apply_exit_cost(position.tp1_price, 'short')
                        position.exit_ts = t_ts
                        position.exit_price = exit_price
                        position.raw_exit_price = position.tp1_price
                        position.exit_reason = 'TP'

                        hours_held = (t_ts - position.entry_ts).total_seconds() / 3600
                        position.funding_cost = cost.calc_funding_cost(1.0, hours_held, 'short') * position.size_mult

                        position.pnl_pct = ((position.entry_price - exit_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
                        result.trades.append(position)
                        position = None
                        break

                # Time-stop
                if position is not None:
                    bars_held = anchor_idx - df_anchor.index.get_loc(
                        df_anchor.index[df_anchor.index <= position.entry_ts][-1]
                    )
                    if bars_held >= self.config.max_hold_bars:
                        exit_price = cost.apply_exit_cost(current_price, position.side)
                        position.exit_ts = ts
                        position.exit_price = exit_price
                        position.raw_exit_price = current_price
                        position.exit_reason = 'TIME'

                        hours_held = (ts - position.entry_ts).total_seconds() / 3600
                        position.funding_cost = cost.calc_funding_cost(1.0, hours_held, position.side) * position.size_mult

                        if position.side == 'long':
                            position.pnl_pct = ((exit_price - position.entry_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
                        else:
                            position.pnl_pct = ((position.entry_price - exit_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
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

            # Trigger TF 데이터
            df_trigger_now = df_trigger[df_trigger.index <= ts]
            if len(df_trigger_now) < 30:
                continue

            # A급 트리거 확인 (Trigger TF)
            a_trigger_long = trigger_ok(df_trigger_now, 'long')
            a_trigger_short = trigger_ok(df_trigger_now, 'short')

            # B급 트리거 확인 (Anchor TF)
            b_trigger_long = alt_trigger_ok(df_up_to_now, 'long')
            b_trigger_short = alt_trigger_ok(df_up_to_now, 'short')

            # Long/Short 둘 다 확인
            signal = None
            trigger_type = ''

            # Long 신호 확인
            long_signal = get_entry_signal(
                a_zones, b_zones, current_price,
                b_trigger_ok=b_trigger_long,
                side_filter='long',
            )

            if long_signal is not None:
                if long_signal.grade == 'A' and a_trigger_long:
                    signal = long_signal
                    trigger_type = 'A-trigger'
                elif long_signal.grade == 'B':
                    signal = long_signal
                    trigger_type = 'B-zscore'

            # Short 신호 확인
            short_signal = get_entry_signal(
                a_zones, b_zones, current_price,
                b_trigger_ok=b_trigger_short,
                side_filter='short',
            )

            if short_signal is not None and signal is None:
                if short_signal.grade == 'A' and a_trigger_short:
                    signal = short_signal
                    trigger_type = 'A-trigger'
                elif short_signal.grade == 'B':
                    signal = short_signal
                    trigger_type = 'B-zscore'

            if signal is None:
                continue

            # 현재가가 Zone 안에 있는지 확인
            if not (signal.entry_low <= current_price <= signal.entry_high):
                continue

            # 진입 (비용 적용)
            entry_price = cost.apply_entry_cost(current_price, signal.side)

            if signal.side == 'long':
                sl_price = current_price - current_atr * self.config.atr_sl_mult
                tp_price = current_price + current_atr * self.config.atr_tp_mult
            else:
                sl_price = current_price + current_atr * self.config.atr_sl_mult
                tp_price = current_price - current_atr * self.config.atr_tp_mult

            position = Trade(
                entry_ts=ts,
                entry_price=entry_price,
                raw_entry_price=current_price,
                side=signal.side,
                grade=signal.grade,
                size_mult=signal.size_mult,
                trigger_type=trigger_type,
                sl_price=sl_price,
                tp1_price=tp_price,
            )

        # 미청산 포지션 처리
        if position is not None:
            last_price = float(df_anchor.iloc[-1]['close'])
            exit_price = cost.apply_exit_cost(last_price, position.side)
            position.exit_ts = df_anchor.index[-1]
            position.exit_price = exit_price
            position.raw_exit_price = last_price
            position.exit_reason = 'END'

            hours_held = (position.exit_ts - position.entry_ts).total_seconds() / 3600
            position.funding_cost = cost.calc_funding_cost(1.0, hours_held, position.side) * position.size_mult

            if position.side == 'long':
                position.pnl_pct = ((exit_price - position.entry_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
            else:
                position.pnl_pct = ((position.entry_price - exit_price) / position.raw_entry_price - position.funding_cost) * position.size_mult
            result.trades.append(position)

        return result


def analyze_by_market_regime(result: BacktestResult, df: pd.DataFrame) -> Dict:
    """시장 상황별 분석 (상승장/하락장)"""
    # 20일 MA 기준 레짐 분류
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['regime'] = np.where(df['close'] > df['ma20'], 'uptrend', 'downtrend')

    uptrend_trades = []
    downtrend_trades = []

    for trade in result.trades:
        if trade.entry_ts in df.index:
            regime = df.loc[trade.entry_ts, 'regime']
            if regime == 'uptrend':
                uptrend_trades.append(trade)
            else:
                downtrend_trades.append(trade)

    def calc_stats(trades):
        if not trades:
            return {'count': 0, 'wr': 0, 'avg_pnl': 0, 'total_pnl': 0}
        wins = len([t for t in trades if t.pnl_pct > 0])
        return {
            'count': len(trades),
            'wr': wins / len(trades),
            'avg_pnl': np.mean([t.pnl_pct for t in trades]),
            'total_pnl': sum(t.pnl_pct for t in trades),
            'long': len([t for t in trades if t.side == 'long']),
            'short': len([t for t in trades if t.side == 'short']),
        }

    return {
        'uptrend': calc_stats(uptrend_trades),
        'downtrend': calc_stats(downtrend_trades),
    }


# =============================================================================
# 메인
# =============================================================================

def main():
    """백테스트 실행"""
    print("=" * 70)
    print("Zone + Divergence 백테스트 v2 (비용 모델 적용)")
    print("=" * 70)

    # 15m 매매 (1개월 테스트)
    START = "2021-11-01"
    END = "2021-11-30"

    backtester = ZoneDivBacktesterV2('15m', '5m')
    result = backtester.run(start_date=START, end_date=END)

    print(f"\n[15m] 결과:")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    # Long vs Short
    print("\n" + "-" * 50)
    print("Long vs Short 비교")
    print("-" * 50)

    long_trades = [t for t in result.trades if t.side == 'long']
    short_trades = [t for t in result.trades if t.side == 'short']

    def stats(trades, name):
        if not trades:
            return f"{name}: 0 trades"
        wins = len([t for t in trades if t.pnl_pct > 0])
        wr = wins / len(trades)
        avg = np.mean([t.pnl_pct for t in trades])
        total = sum(t.pnl_pct for t in trades)
        return f"{name}: {len(trades)} trades, WR {wr:.1%}, Avg {avg:.2%}, Total {total:.2%}"

    print(stats(long_trades, "Long"))
    print(stats(short_trades, "Short"))

    # A급 vs B급
    print("\n" + "-" * 50)
    print("A급 vs B급 비교")
    print("-" * 50)

    a_trades = [t for t in result.trades if t.grade == 'A']
    b_trades = [t for t in result.trades if t.grade == 'B']

    print(stats(a_trades, "A급"))
    print(stats(b_trades, "B급"))

    # 시장 상황별
    print("\n" + "-" * 50)
    print("시장 상황별 분석 (MA20 기준)")
    print("-" * 50)

    df = load_tf_data('15m', START, END)
    regime_stats = analyze_by_market_regime(result, df)

    for regime, s in regime_stats.items():
        if s['count'] > 0:
            print(f"{regime}: {s['count']} trades (L:{s['long']}/S:{s['short']}), "
                  f"WR {s['wr']:.1%}, Avg {s['avg_pnl']:.2%}, Total {s['total_pnl']:.2%}")

    # Exit 이유별
    print("\n" + "-" * 50)
    print("Exit 이유별")
    print("-" * 50)

    for reason in ['TP', 'SL', 'TIME', 'END']:
        reason_trades = [t for t in result.trades if t.exit_reason == reason]
        if reason_trades:
            print(stats(reason_trades, reason))


if __name__ == "__main__":
    main()
