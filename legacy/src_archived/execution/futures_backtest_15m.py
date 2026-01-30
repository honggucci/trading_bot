"""
선물 백테스트 엔진 v2.7.0 - 15분봉 지정가 주문 전용
======================================================

v2.7.0 변경사항:
- 레버리지: 3x → 25x
- TP 분할: 단일 → 3단계 (50/30/20)
- pending_order_max_bars: 4봉 → 32봉 (8시간)

Config 기반 파라미터 로드:
- get_config_for_symbol(symbol): config에서 FuturesConfig15m 로드
- 환경변수 오버라이드 지원

핵심 로직:
1. 시그널 발생 → PendingOrder 생성 (limit_price = 다이버전스 존 중심)
2. 다음 1~32봉 동안 bar_low <= limit_price <= bar_high 체크
3. 체결 시 SL/TP1/TP2/TP3 즉시 설정 (ATR 기반 지정가)
4. TP1 체결 → 50% 청산, TP2 체결 → 30% 청산, TP3 체결 → 20% 청산
5. 32봉 초과 → 시장가 청산 (time-stop)
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
    leverage: float = 25.0            # 레버리지 (3x → 25x)
    margin_mode: str = 'isolated'

    # 포지션 사이징
    position_pct: float = 0.03  # 자본의 3%

    # ATR 기반 SL/TP (3단계 분할)
    atr_period: int = 14
    atr_sl_mult: float = 1.5          # SL = entry ± ATR * 1.5
    atr_tp1_mult: float = 1.5         # TP1 = entry ± ATR * 1.5 (50%)
    atr_tp2_mult: float = 2.5         # TP2 = entry ± ATR * 2.5 (30%)
    atr_tp3_mult: float = 3.5         # TP3 = entry ± ATR * 3.5 (20%)
    tp_split: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # 분할 비율

    # 지정가 주문 유효기간
    pending_order_max_bars: int = 32  # 32봉(8시간) 미체결 시 취소

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
    tp1_price: float          # TP1 지정가 (50%)
    tp2_price: float          # TP2 지정가 (30%)
    tp3_price: float          # TP3 지정가 (20%)
    signal_bar: int           # 시그널 발생 봉
    signal_time: pd.Timestamp # 시그널 시간
    max_bars: int = 32        # 유효 봉 수 (8시간)
    atr_val: float = 0.0      # ATR 값 (디버깅용)

    def is_expired(self, current_bar: int) -> bool:
        """주문 만료 체크"""
        return (current_bar - self.signal_bar) > self.max_bars

    def check_fill(self, bar_low: float, bar_high: float) -> bool:
        """체결 조건 체크: 지정가가 봉의 범위 내에 있는지"""
        return bar_low <= self.limit_price <= bar_high


@dataclass
class Position15m:
    """15분봉 포지션 (3단계 TP 분할)"""
    side: Literal['long', 'short']
    entry_price: float
    qty: float
    margin: float
    leverage: float
    entry_bar: int
    entry_time: pd.Timestamp
    sl_price: float
    tp1_price: float          # TP1 (50%)
    tp2_price: float          # TP2 (30%)
    tp3_price: float          # TP3 (20%)
    tp1_filled: bool = False  # TP1 체결 여부
    tp2_filled: bool = False  # TP2 체결 여부
    remaining_qty: float = 1.0  # 남은 수량 비율 (1.0 = 100%)

    def check_sl_hit(self, bar_low: float, bar_high: float) -> bool:
        """SL 체결 체크"""
        if self.side == 'long':
            return bar_low <= self.sl_price
        else:  # short
            return bar_high >= self.sl_price

    def check_tp1_hit(self, bar_low: float, bar_high: float) -> bool:
        """TP1 체결 체크"""
        if self.tp1_filled:
            return False
        if self.side == 'long':
            return bar_high >= self.tp1_price
        else:  # short
            return bar_low <= self.tp1_price

    def check_tp2_hit(self, bar_low: float, bar_high: float) -> bool:
        """TP2 체결 체크"""
        if self.tp2_filled:
            return False
        if self.side == 'long':
            return bar_high >= self.tp2_price
        else:  # short
            return bar_low <= self.tp2_price

    def check_tp3_hit(self, bar_low: float, bar_high: float) -> bool:
        """TP3 체결 체크 (전량 청산)"""
        if self.side == 'long':
            return bar_high >= self.tp3_price
        else:  # short
            return bar_low <= self.tp3_price

    def check_time_stop(self, current_bar: int, max_hold: int) -> bool:
        """시간 청산 체크"""
        return (current_bar - self.entry_bar) >= max_hold

    def calc_pnl(self, exit_price: float, exit_qty_ratio: float = 1.0) -> float:
        """PnL 계산

        Args:
            exit_price: 청산 가격
            exit_qty_ratio: 청산 비율 (0.5 = 50%)

        Returns:
            PnL (USD)
        """
        exit_qty = self.qty * exit_qty_ratio
        if self.side == 'long':
            return (exit_price - self.entry_price) * exit_qty
        else:  # short (qty가 음수이므로 abs 사용)
            return (self.entry_price - exit_price) * abs(exit_qty)


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
