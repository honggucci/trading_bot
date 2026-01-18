# backtest_divergence_v3.py
# RSI 다이버전스 지정가 백테스트
# - 롱/숏 독립적으로 15m → 5m fallback
# - Regular/Hidden 둘 다 체크
# - 비용 모델 (wpcn 참고)

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import talib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Tuple

# 프로젝트 루트
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Fibonacci import
from src.context.cycle_anchor import get_nearby_fib_levels, FibLevel

# =============================================================================
# 비용 모델 (wpcn 참고)
# =============================================================================
@dataclass
class CostModel:
    fee_bps: float = 4.0          # 0.04% maker/taker
    slippage_bps: float = 5.0     # 0.05% 슬리피지
    funding_rate: float = 0.0001  # 0.01% per 8h

    def entry_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10000

    def exit_cost(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10000

    def funding_cost(self, hours: float) -> float:
        return self.funding_rate * (hours / 8)

# =============================================================================
# RSI 계산 (talib 사용)
# =============================================================================
def calc_rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI 계산 (talib 사용)"""
    close = np.asarray(close, dtype=np.float64)
    return talib.RSI(close, timeperiod=period)

def _rsi_at_price(close_arr: np.ndarray, new_close: float, period: int = 14) -> float:
    """특정 가격에서의 RSI 계산"""
    close = close_arr.copy()
    close[-1] = new_close
    rsi = calc_rsi_wilder(close, period)
    return float(rsi[-1]) if np.isfinite(rsi[-1]) else np.nan

# =============================================================================
# RSI 역산 함수들
# =============================================================================
def needed_close_for_regular_bullish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    tol: float = 1e-6,
    max_iter: int = 60
) -> Optional[float]:
    """
    Regular Bullish Divergence 성립 가격 역산
    조건: 가격 < ref_price (Lower Low), RSI > ref_rsi (Higher Low)
    """
    eps = 1e-8
    U = ref_price - max(eps, abs(ref_price) * 1e-6)
    L = U * 0.9

    if not np.isfinite(U) or L >= U or L <= 0:
        return None

    rsi_U = _rsi_at_price(close_arr, U, rsi_period)
    if not np.isfinite(rsi_U) or rsi_U <= ref_rsi:
        return None

    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)

        if not np.isfinite(rsi_mid):
            lo = mid
            continue

        if rsi_mid > ref_rsi:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) <= tol:
            break

    result = min(hi, U)
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi <= ref_rsi:
        return None

    return float(result) if result > 0 else None

def needed_close_for_regular_bearish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    tol: float = 1e-6,
    max_iter: int = 60
) -> Optional[float]:
    """
    Regular Bearish Divergence 성립 가격 역산
    조건: 가격 > ref_price (Higher High), RSI < ref_rsi (Lower High)
    """
    eps = 1e-8
    L = ref_price + max(eps, abs(ref_price) * 1e-6)
    U = L * 1.1

    if not np.isfinite(L) or L >= U:
        return None

    rsi_L = _rsi_at_price(close_arr, L, rsi_period)
    if not np.isfinite(rsi_L) or rsi_L >= ref_rsi:
        return None

    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)

        if not np.isfinite(rsi_mid):
            hi = mid
            continue

        if rsi_mid < ref_rsi:
            lo = mid
        else:
            hi = mid

        if abs(hi - lo) <= tol:
            break

    result = max(lo, L)
    final_rsi = _rsi_at_price(close_arr, result, rsi_period)
    if not np.isfinite(final_rsi) or final_rsi >= ref_rsi:
        return None

    return float(result)

def feasible_range_for_hidden_bullish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    tol: float = 1e-6,
    max_iter: int = 60
) -> Optional[Tuple[float, float]]:
    """
    Hidden Bullish Divergence 성립 가격 범위 역산
    조건: 가격 > ref_price (Higher Low), RSI < ref_rsi (Lower Low)
    """
    eps = 1e-8
    L = ref_price + max(eps, abs(ref_price) * 1e-6)
    U = L * 1.1

    if not np.isfinite(L) or L >= U:
        return None

    rsi_L = _rsi_at_price(close_arr, L, rsi_period)
    if not np.isfinite(rsi_L) or rsi_L >= ref_rsi:
        return None

    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)

        if not np.isfinite(rsi_mid):
            hi = mid
            continue

        if rsi_mid < ref_rsi:
            lo = mid
        else:
            hi = mid

        if abs(hi - lo) <= tol:
            break

    x_max = lo
    if x_max <= L:
        return None

    return (float(L), float(x_max))

def feasible_range_for_hidden_bearish(
    close_arr: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    tol: float = 1e-6,
    max_iter: int = 60
) -> Optional[Tuple[float, float]]:
    """
    Hidden Bearish Divergence 성립 가격 범위 역산
    조건: 가격 < ref_price (Lower High), RSI > ref_rsi (Higher High)
    """
    eps = 1e-8
    U = ref_price - max(eps, abs(ref_price) * 1e-6)
    L = U * 0.9

    if not np.isfinite(U) or L >= U or L <= 0:
        return None

    rsi_U = _rsi_at_price(close_arr, U, rsi_period)
    if not np.isfinite(rsi_U) or rsi_U <= ref_rsi:
        return None

    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rsi_mid = _rsi_at_price(close_arr, mid, rsi_period)

        if not np.isfinite(rsi_mid):
            lo = mid
            continue

        if rsi_mid > ref_rsi:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) <= tol:
            break

    x_min = hi
    if x_min >= U:
        return None

    return (float(x_min), float(U))

# =============================================================================
# 참조점 찾기 (StochRSI oversold/overbought)
# =============================================================================
def find_oversold_reference(
    df: pd.DataFrame,
    stoch_col: str = 'stoch_d',
    oversold: float = 20.0,
    lookback: int = 100
) -> Optional[Dict]:
    """최근 oversold 구간의 저점을 참조점으로 반환"""
    if len(df) < 10:
        return None

    d = df[stoch_col].values[-lookback:] if len(df) >= lookback else df[stoch_col].values
    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] <= oversold:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] <= oversold:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]

    if not segments:
        return None

    current_oversold = np.isfinite(d[-1]) and d[-1] <= oversold

    if current_oversold and len(segments) >= 2:
        seg = segments[-2]
    elif current_oversold:
        return None
    else:
        seg = segments[-1]

    a, b = seg
    seg_close = close[a:b+1]
    seg_rsi = rsi[a:b+1]

    min_idx = np.argmin(seg_close)
    ref_price = float(seg_close[min_idx])
    ref_rsi = float(seg_rsi[min_idx])

    if not np.isfinite(ref_rsi):
        return None

    return {'ref_price': ref_price, 'ref_rsi': ref_rsi, 'segment': seg}

def find_overbought_reference(
    df: pd.DataFrame,
    stoch_col: str = 'stoch_d',
    overbought: float = 80.0,
    lookback: int = 100
) -> Optional[Dict]:
    """최근 overbought 구간의 고점을 참조점으로 반환"""
    if len(df) < 10:
        return None

    d = df[stoch_col].values[-lookback:] if len(df) >= lookback else df[stoch_col].values
    close = df['close'].values[-lookback:] if len(df) >= lookback else df['close'].values
    rsi = df['rsi'].values[-lookback:] if len(df) >= lookback else df['rsi'].values

    n = len(d)
    segments = []
    i = n - 1

    while i >= 0:
        if np.isfinite(d[i]) and d[i] >= overbought:
            b = i
            a = i
            while a - 1 >= 0 and np.isfinite(d[a-1]) and d[a-1] >= overbought:
                a -= 1
            segments.append((a, b))
            i = a - 1
        else:
            i -= 1

    segments = segments[::-1]

    if not segments:
        return None

    current_overbought = np.isfinite(d[-1]) and d[-1] >= overbought

    if current_overbought and len(segments) >= 2:
        seg = segments[-2]
    elif current_overbought:
        return None
    else:
        seg = segments[-1]

    a, b = seg
    seg_close = close[a:b+1]
    seg_rsi = rsi[a:b+1]

    max_idx = np.argmax(seg_close)
    ref_price = float(seg_close[max_idx])
    ref_rsi = float(seg_rsi[max_idx])

    if not np.isfinite(ref_rsi):
        return None

    return {'ref_price': ref_price, 'ref_rsi': ref_rsi, 'segment': seg}

# =============================================================================
# Fibonacci 근접 체크
# =============================================================================
def is_near_fib_level(
    price: float,
    tolerance_pct: float = 0.02,  # 2% 허용 오차
    max_depth: int = 1,
) -> Tuple[bool, Optional[FibLevel]]:
    """
    가격이 Fib 레벨 근처인지 체크

    Args:
        price: 체크할 가격
        tolerance_pct: 허용 오차 (2% = 0.02)
        max_depth: 프랙탈 깊이 (0=L0, 1=L0+L1)

    Returns:
        (근접 여부, 가장 가까운 FibLevel)
    """
    nearby = get_nearby_fib_levels(price, count=3, max_depth=max_depth)

    all_levels = nearby.get('above', []) + nearby.get('below', [])
    if not all_levels:
        return False, None

    # 가장 가까운 레벨 찾기
    closest = min(all_levels, key=lambda lvl: abs(lvl.price - price))
    distance_pct = abs(closest.price - price) / closest.price

    if distance_pct <= tolerance_pct:
        return True, closest

    return False, None


# =============================================================================
# 다이버전스 가격 계산
# =============================================================================
@dataclass
class DivergencePrice:
    long_regular: Optional[float] = None
    long_hidden: Optional[Tuple[float, float]] = None
    short_regular: Optional[float] = None
    short_hidden: Optional[Tuple[float, float]] = None
    tf: str = ""

def calculate_divergence_prices(df: pd.DataFrame, rsi_period: int = 14) -> DivergencePrice:
    """현재 시점에서 4가지 다이버전스 가격 계산"""
    result = DivergencePrice()

    if len(df) < rsi_period + 10:
        return result

    close_arr = df['close'].values.astype(float)

    ref_long = find_oversold_reference(df)
    if ref_long:
        result.long_regular = needed_close_for_regular_bullish(
            close_arr, ref_long['ref_price'], ref_long['ref_rsi'], rsi_period
        )
        result.long_hidden = feasible_range_for_hidden_bullish(
            close_arr, ref_long['ref_price'], ref_long['ref_rsi'], rsi_period
        )

    ref_short = find_overbought_reference(df)
    if ref_short:
        result.short_regular = needed_close_for_regular_bearish(
            close_arr, ref_short['ref_price'], ref_short['ref_rsi'], rsi_period
        )
        result.short_hidden = feasible_range_for_hidden_bearish(
            close_arr, ref_short['ref_price'], ref_short['ref_rsi'], rsi_period
        )

    return result

# =============================================================================
# 트레이드 & 결과
# =============================================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    side: Literal['long', 'short'] = 'long'
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    div_type: str = ""
    tf: str = ""

@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)

    def summary(self) -> Dict:
        if not self.trades:
            return {'total_trades': 0}
        wins = [t for t in self.trades if t.pnl > 0]
        return {
            'total_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades),
            'total_pnl': sum(t.pnl_pct for t in self.trades),
            'avg_pnl': np.mean([t.pnl_pct for t in self.trades]),
            'max_win': max(t.pnl_pct for t in self.trades),
            'max_loss': min(t.pnl_pct for t in self.trades),
        }

# =============================================================================
# 인디케이터 계산 (talib 사용)
# =============================================================================
def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 계산 (talib 사용)"""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return talib.ATR(high, low, close, timeperiod=period)

def calc_stoch_rsi(close: np.ndarray, period: int = 14, k_period: int = 3, d_period: int = 3) -> np.ndarray:
    """StochRSI %D 계산 (talib 사용)"""
    close = np.asarray(close, dtype=np.float64)
    fastk, fastd = talib.STOCHRSI(
        close,
        timeperiod=period,
        fastk_period=k_period,
        fastd_period=d_period,
        fastd_matype=0  # SMA
    )
    return fastd

# =============================================================================
# 데이터 로딩
# =============================================================================
def load_data(tf: str, start_date: str, end_date: str) -> pd.DataFrame:
    """데이터 로딩 및 인디케이터 계산 (순수 Python)"""
    # data/bronze/binance/futures/BTC-USDT/{tf}/YYYY/MM.parquet
    data_dir = ROOT / "data" / "bronze" / "binance" / "futures" / "BTC-USDT" / tf

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 모든 parquet 파일 로딩
    files = list(data_dir.glob("**/*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')

    df = df.sort_index()
    df = df[start_date:end_date]

    # RSI (Wilder) - talib 사용
    df['rsi'] = calc_rsi_wilder(df['close'].values, period=14)

    # StochRSI (26선) - talib 사용
    df['stoch_d'] = calc_stoch_rsi(df['close'].values, period=26, k_period=3, d_period=3)

    # ATR - talib 사용
    df['atr'] = calc_atr(df['high'].values, df['low'].values, df['close'].values, period=14)

    return df

# =============================================================================
# 백테스터
# =============================================================================
class DivergenceBacktester:
    def __init__(
        self,
        top_tf: str = '15m',
        down_tf: str = '5m',
        rsi_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        fib_filter: bool = True,           # Fib 필터 ON/OFF
        fib_tolerance_pct: float = 0.02,   # 2% 허용 오차
        fib_max_depth: int = 1,            # L0 + L1
    ):
        self.top_tf = top_tf
        self.down_tf = down_tf
        self.rsi_period = rsi_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.fib_filter = fib_filter
        self.fib_tolerance_pct = fib_tolerance_pct
        self.fib_max_depth = fib_max_depth
        self.cost_model = CostModel()

        # 통계
        self.fib_filtered_count = 0

    def run(self, start_date: str, end_date: str) -> BacktestResult:
        """백테스트 실행"""
        print(f"Loading {self.top_tf} data...")
        df_top = load_data(self.top_tf, start_date, end_date)

        print(f"Loading {self.down_tf} data...")
        df_down = load_data(self.down_tf, start_date, end_date)

        print(f"Top TF: {len(df_top)} bars, Down TF: {len(df_down)} bars")

        result = BacktestResult()
        position = None

        for i in range(50, len(df_down)):
            bar = df_down.iloc[i]
            current_time = df_down.index[i]

            # 포지션 체크 (SL/TP)
            if position is not None:
                exit_price = None
                exit_reason = None

                if position['side'] == 'long':
                    if bar['low'] <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL'
                    elif bar['high'] >= position['tp']:
                        exit_price = position['tp']
                        exit_reason = 'TP'
                else:
                    if bar['high'] >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL'
                    elif bar['low'] <= position['tp']:
                        exit_price = position['tp']
                        exit_reason = 'TP'

                if exit_price:
                    if position['side'] == 'long':
                        raw_pnl = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        raw_pnl = (position['entry_price'] - exit_price) / position['entry_price']

                    costs = self.cost_model.entry_cost() + self.cost_model.exit_cost()
                    hours = (current_time - position['entry_time']).total_seconds() / 3600
                    costs += self.cost_model.funding_cost(hours)
                    net_pnl = raw_pnl - costs

                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        side=position['side'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        pnl=net_pnl * position['entry_price'],
                        pnl_pct=net_pnl,
                        exit_reason=exit_reason,
                        div_type=position['div_type'],
                        tf=position['tf']
                    )
                    result.trades.append(trade)
                    position = None

            # 새 진입
            if position is None:
                mask_top = df_top.index <= current_time
                if mask_top.sum() < 50:
                    continue

                df_top_slice = df_top[mask_top]
                df_down_slice = df_down.iloc[:i+1]

                # Long 가격: 15m → 5m fallback
                long_price = None
                long_tf = None
                long_type = None
                div_5m = None

                div_15m = calculate_divergence_prices(df_top_slice, self.rsi_period)
                div_15m.tf = '15m'

                if div_15m.long_regular is not None:
                    long_price = div_15m.long_regular
                    long_tf = '15m'
                    long_type = 'regular'
                elif div_15m.long_hidden is not None:
                    long_price = div_15m.long_hidden[0]
                    long_tf = '15m'
                    long_type = 'hidden'
                else:
                    div_5m = calculate_divergence_prices(df_down_slice, self.rsi_period)
                    div_5m.tf = '5m'
                    if div_5m.long_regular is not None:
                        long_price = div_5m.long_regular
                        long_tf = '5m'
                        long_type = 'regular'
                    elif div_5m.long_hidden is not None:
                        long_price = div_5m.long_hidden[0]
                        long_tf = '5m'
                        long_type = 'hidden'

                # Short 가격: 15m → 5m fallback
                short_price = None
                short_tf = None
                short_type = None

                if div_15m.short_regular is not None:
                    short_price = div_15m.short_regular
                    short_tf = '15m'
                    short_type = 'regular'
                elif div_15m.short_hidden is not None:
                    short_price = div_15m.short_hidden[1]
                    short_tf = '15m'
                    short_type = 'hidden'
                else:
                    if div_5m is None:
                        div_5m = calculate_divergence_prices(df_down_slice, self.rsi_period)
                    if div_5m.short_regular is not None:
                        short_price = div_5m.short_regular
                        short_tf = '5m'
                        short_type = 'regular'
                    elif div_5m.short_hidden is not None:
                        short_price = div_5m.short_hidden[1]
                        short_tf = '5m'
                        short_type = 'hidden'

                # ATR
                atr = bar['atr'] if np.isfinite(bar['atr']) else df_down['atr'].iloc[:i].dropna().iloc[-1]

                # Fib 필터 적용
                long_fib_ok = True
                short_fib_ok = True
                long_fib_level = None
                short_fib_level = None

                if self.fib_filter:
                    if long_price is not None and long_price > 0:
                        long_fib_ok, long_fib_level = is_near_fib_level(
                            long_price, self.fib_tolerance_pct, self.fib_max_depth
                        )
                        if not long_fib_ok:
                            self.fib_filtered_count += 1

                    if short_price is not None and short_price > 0:
                        short_fib_ok, short_fib_level = is_near_fib_level(
                            short_price, self.fib_tolerance_pct, self.fib_max_depth
                        )
                        if not short_fib_ok:
                            self.fib_filtered_count += 1

                # Long 체결 (Fib 필터 통과 시)
                if long_price is not None and long_price > 0 and long_fib_ok:
                    if bar['low'] <= long_price:
                        position = {
                            'side': 'long',
                            'entry_price': long_price,
                            'entry_time': current_time,
                            'sl': long_price - self.sl_atr_mult * atr,
                            'tp': long_price + self.tp_atr_mult * atr,
                            'div_type': long_type,
                            'tf': long_tf,
                            'fib_level': long_fib_level.fib_ratio if long_fib_level else None
                        }
                        continue

                # Short 체결 (Fib 필터 통과 시)
                if short_price is not None and short_price > 0 and short_fib_ok:
                    if bar['high'] >= short_price:
                        position = {
                            'side': 'short',
                            'entry_price': short_price,
                            'entry_time': current_time,
                            'sl': short_price + self.sl_atr_mult * atr,
                            'tp': short_price - self.tp_atr_mult * atr,
                            'div_type': short_type,
                            'tf': short_tf,
                            'fib_level': short_fib_level.fib_ratio if short_fib_level else None
                        }

        # 미청산 포지션
        if position is not None:
            last_bar = df_down.iloc[-1]
            last_time = df_down.index[-1]
            exit_price = last_bar['close']

            if position['side'] == 'long':
                raw_pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                raw_pnl = (position['entry_price'] - exit_price) / position['entry_price']

            costs = self.cost_model.entry_cost() + self.cost_model.exit_cost()
            hours = (last_time - position['entry_time']).total_seconds() / 3600
            costs += self.cost_model.funding_cost(hours)
            net_pnl = raw_pnl - costs

            trade = Trade(
                entry_time=position['entry_time'],
                exit_time=last_time,
                side=position['side'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                pnl=net_pnl * position['entry_price'],
                pnl_pct=net_pnl,
                exit_reason='EOD',
                div_type=position['div_type'],
                tf=position['tf']
            )
            result.trades.append(trade)

        return result

# =============================================================================
# 분석 함수들
# =============================================================================
def analyze_by_side(result: BacktestResult) -> Dict:
    long_trades = [t for t in result.trades if t.side == 'long']
    short_trades = [t for t in result.trades if t.side == 'short']

    def stats(trades, name):
        if not trades:
            return f"{name}: 0 trades"
        wins = [t for t in trades if t.pnl > 0]
        return (f"{name}: {len(trades)} trades, "
                f"WR {len(wins)/len(trades):.1%}, "
                f"Total {sum(t.pnl_pct for t in trades):.2%}")

    return {'long': stats(long_trades, 'Long'), 'short': stats(short_trades, 'Short')}

def analyze_by_tf(result: BacktestResult) -> Dict:
    tf_15m = [t for t in result.trades if t.tf == '15m']
    tf_5m = [t for t in result.trades if t.tf == '5m']

    def stats(trades, name):
        if not trades:
            return f"{name}: 0 trades"
        wins = [t for t in trades if t.pnl > 0]
        return (f"{name}: {len(trades)} trades, "
                f"WR {len(wins)/len(trades):.1%}, "
                f"Total {sum(t.pnl_pct for t in trades):.2%}")

    return {'15m': stats(tf_15m, '15m'), '5m': stats(tf_5m, '5m')}

def analyze_by_div_type(result: BacktestResult) -> Dict:
    regular = [t for t in result.trades if t.div_type == 'regular']
    hidden = [t for t in result.trades if t.div_type == 'hidden']

    def stats(trades, name):
        if not trades:
            return f"{name}: 0 trades"
        wins = [t for t in trades if t.pnl > 0]
        return (f"{name}: {len(trades)} trades, "
                f"WR {len(wins)/len(trades):.1%}, "
                f"Total {sum(t.pnl_pct for t in trades):.2%}")

    return {'regular': stats(regular, 'Regular'), 'hidden': stats(hidden, 'Hidden')}

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("RSI Divergence 백테스트 v3.1")
    print("- 15m → 5m Fallback")
    print("- Long/Short 독립")
    print("- Regular/Hidden 모두 체크")
    print("- Fibonacci Boundary 필터 추가")
    print("=" * 70)

    START = "2021-11-01"
    END = "2021-11-30"

    # Fib 필터 파라미터
    FIB_FILTER = True           # Fib 필터 ON
    FIB_TOLERANCE_PCT = 0.02    # 2% 허용 오차
    FIB_MAX_DEPTH = 1           # L0 + L1

    backtester = DivergenceBacktester(
        top_tf='15m',
        down_tf='5m',
        rsi_period=14,
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        fib_filter=FIB_FILTER,
        fib_tolerance_pct=FIB_TOLERANCE_PCT,
        fib_max_depth=FIB_MAX_DEPTH,
    )

    print(f"\nFib Filter: {'ON' if FIB_FILTER else 'OFF'}")
    print(f"Fib Tolerance: {FIB_TOLERANCE_PCT:.1%}")
    print(f"Fib Max Depth: L{FIB_MAX_DEPTH}\n")

    result = backtester.run(START, END)

    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    summary = result.summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Fib 필터 통계
    if backtester.fib_filter:
        print(f"\n  [Fib Filter Stats]")
        print(f"  Filtered by Fib: {backtester.fib_filtered_count} entries blocked")

    print("\n" + "-" * 50)
    print("Long vs Short")
    print("-" * 50)
    for k, v in analyze_by_side(result).items():
        print(f"  {v}")

    print("\n" + "-" * 50)
    print("15m vs 5m (TF별)")
    print("-" * 50)
    for k, v in analyze_by_tf(result).items():
        print(f"  {v}")

    print("\n" + "-" * 50)
    print("Regular vs Hidden")
    print("-" * 50)
    for k, v in analyze_by_div_type(result).items():
        print(f"  {v}")

    print("\n" + "-" * 50)
    print("Exit 이유별")
    print("-" * 50)

    exit_reasons = {}
    for t in result.trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {'count': 0, 'pnl': 0}
        exit_reasons[r]['count'] += 1
        exit_reasons[r]['pnl'] += t.pnl_pct

    for reason, stats in exit_reasons.items():
        print(f"  {reason}: {stats['count']} trades, Total {stats['pnl']:.2%}")

if __name__ == "__main__":
    main()
