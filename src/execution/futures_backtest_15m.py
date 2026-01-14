"""
선물 백테스트 엔진 v2.6.15 - 15분봉 지정가 주문 전용
======================================================

Config 기반 파라미터 로드 (v2.6.15):
- get_config_for_symbol(symbol): config에서 FuturesConfig15m 로드
- 환경변수 오버라이드 지원

핵심 로직:
1. 시그널 발생 → PendingOrder 생성 (limit_price = 시그널 가격)
2. 다음 1~4봉 동안 bar_low <= limit_price <= bar_high 체크
3. 체결 시 SL/TP 즉시 설정 (ATR 기반 지정가)
4. 32봉 초과 → 시장가 청산 (time-stop)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd


@dataclass
class FuturesConfig15m:
    """15분봉 지정가 백테스트 설정"""
    leverage: float = 3.0
    margin_mode: str = 'isolated'

    # 포지션 사이징
    position_pct: float = 0.03  # 자본의 3%

    # ATR 기반 SL/TP
    atr_period: int = 14
    atr_sl_mult: float = 1.5    # SL = entry ± ATR * 1.5
    atr_tp_mult: float = 2.5    # TP = entry ± ATR * 2.5

    # 지정가 주문 유효기간
    pending_order_max_bars: int = 4   # 4봉(1시간) 미체결 시 취소

    # 최대 보유 시간 (펀딩비 회피)
    max_hold_bars: int = 32           # 32봉 = 8시간

    # 비용
    maker_fee: float = 0.0002         # 지정가 수수료 0.02%
    taker_fee: float = 0.0005         # 시장가 수수료 0.05%
    slippage_pct: float = 0.0001      # 슬리피지 0.01%

    # 시그널 필터
    fib_tolerance: float = 0.02       # 피보나치 레벨 허용 오차 2%
    entry_cooldown: int = 6           # 진입 후 쿨다운 (봉)


def get_config_for_symbol(symbol: Optional[str] = None) -> "FuturesConfig15m":
    """
    심볼별 FuturesConfig15m 로드 (config 기반)

    Args:
        symbol: 심볼명 (예: "BTC-USDT"). None이면 환경변수 TRADING_SYMBOL 사용

    Returns:
        FuturesConfig15m 객체

    사용법:
        cfg = get_config_for_symbol("BTC-USDT")
        cfg = get_config_for_symbol()  # TRADING_SYMBOL 환경변수 사용
    """
    if symbol is None:
        symbol = os.getenv("TRADING_SYMBOL", "BTC-USDT")

    try:
        from src.config import load_symbol_config
        config = load_symbol_config(symbol)
        exe = config.execution
        lev = config.leverage

        return FuturesConfig15m(
            leverage=lev.default,
            margin_mode=lev.margin_mode,
            atr_period=exe.atr_period,
            atr_sl_mult=exe.atr_sl_mult,
            atr_tp_mult=exe.atr_tp_mult,
            max_hold_bars=exe.max_hold_bars,
            pending_order_max_bars=exe.pending_order_max_bars,
            maker_fee=exe.maker_fee,
            taker_fee=exe.taker_fee,
            slippage_pct=exe.slippage,
        )
    except ImportError:
        # config 모듈 없으면 기본값 반환
        return FuturesConfig15m()


@dataclass
class PendingOrder:
    """미체결 지정가 주문"""
    side: Literal['long', 'short']
    limit_price: float        # 지정가
    target_qty: float         # 목표 수량
    margin: float             # 증거금
    sl_price: float           # SL 지정가
    tp_price: float           # TP 지정가
    signal_bar: int           # 시그널 발생 봉
    signal_time: pd.Timestamp # 시그널 시간
    max_bars: int = 4         # 유효 봉 수
    atr_val: float = 0.0      # ATR 값 (디버깅용)

    def is_expired(self, current_bar: int) -> bool:
        """주문 만료 체크"""
        return (current_bar - self.signal_bar) > self.max_bars

    def check_fill(self, bar_low: float, bar_high: float) -> bool:
        """체결 조건 체크: 지정가가 봉의 범위 내에 있는지"""
        return bar_low <= self.limit_price <= bar_high


@dataclass
class Position15m:
    """15분봉 포지션"""
    side: Literal['long', 'short']
    entry_price: float
    qty: float
    margin: float
    leverage: float
    entry_bar: int
    entry_time: pd.Timestamp
    sl_price: float
    tp_price: float

    def check_sl_hit(self, bar_low: float, bar_high: float) -> bool:
        """SL 체결 체크"""
        if self.side == 'long':
            return bar_low <= self.sl_price
        else:  # short
            return bar_high >= self.sl_price

    def check_tp_hit(self, bar_low: float, bar_high: float) -> bool:
        """TP 체결 체크"""
        if self.side == 'long':
            return bar_high >= self.tp_price
        else:  # short
            return bar_low <= self.tp_price

    def check_time_stop(self, current_bar: int, max_hold: int) -> bool:
        """시간 청산 체크"""
        return (current_bar - self.entry_bar) >= max_hold

    def calc_pnl(self, exit_price: float) -> float:
        """PnL 계산 (qty는 롱 양수, 숏 음수)"""
        if self.side == 'long':
            return (exit_price - self.entry_price) * self.qty
        else:  # short (qty가 음수이므로 abs 사용)
            return (self.entry_price - exit_price) * abs(self.qty)


def calc_atr_15m(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """15분봉 ATR 계산 (shift(1) 적용 - lookahead 방지)"""
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(period).mean()
    return atr.shift(1)  # 완성된 봉의 ATR만 사용


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def resample_5m_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 15분봉 리샘플링"""
    return df_5m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
