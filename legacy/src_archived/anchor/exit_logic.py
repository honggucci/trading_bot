# -*- coding: utf-8 -*-
"""
Exit Logic - SL/TP/Trailing Stop
================================

청산 로직 모듈.

핵심 원칙:
1. ATR 기반 동적 Stop Loss
2. Multi-level Take Profit (1차 50%, 2차 30%, 3차 20%)
3. Trailing Stop (수익 보호)
4. Confluence Zone 기반 TP 설정

사용법:
```python
from src.anchor.exit_logic import calc_exit_levels, ExitLevels

exit_levels = calc_exit_levels(
    entry_price=95000,
    side='long',
    atr=1500,
    confluence_zones=prediction.confluence_zones,
)

print(f"SL: ${exit_levels.stop_loss:,.0f}")
print(f"TP1: ${exit_levels.take_profit_1:,.0f} (50%)")
print(f"TP2: ${exit_levels.take_profit_2:,.0f} (30%)")
```
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple
from enum import Enum
import numpy as np


class ExitReason(Enum):
    """청산 사유"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class ExitLevels:
    """청산 레벨"""
    # Entry 정보
    entry_price: float
    side: Literal['long', 'short']

    # Stop Loss
    stop_loss: float
    stop_loss_pct: float  # % 거리

    # Take Profit (Multi-level)
    take_profit_1: float  # 1차 (50% 청산)
    take_profit_2: float  # 2차 (30% 청산)
    take_profit_3: float  # 3차 (20% 청산)
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float

    # Trailing Stop
    trailing_activation: float  # 활성화 가격 (TP1 도달 후)
    trailing_distance_pct: float  # 트레일링 거리 (%)

    # Risk:Reward
    risk_reward_1: float  # TP1 기준 R:R
    risk_reward_2: float  # TP2 기준 R:R

    # ATR 정보
    atr: float
    atr_mult_sl: float

    # 근거
    sl_reason: str = ""
    tp_reason: str = ""


@dataclass
class TrailingState:
    """트레일링 스탑 상태"""
    is_active: bool = False
    highest_price: float = 0.0  # Long: 최고가
    lowest_price: float = float('inf')  # Short: 최저가
    current_stop: float = 0.0

    def update(self, current_price: float, side: str, trailing_pct: float) -> Tuple[bool, float]:
        """
        트레일링 스탑 업데이트

        Returns:
            (triggered, new_stop)
        """
        if side == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.current_stop = current_price * (1 - trailing_pct / 100)

            if current_price <= self.current_stop:
                return True, self.current_stop
        else:  # short
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                self.current_stop = current_price * (1 + trailing_pct / 100)

            if current_price >= self.current_stop:
                return True, self.current_stop

        return False, self.current_stop


@dataclass
class ExitSignal:
    """청산 신호"""
    should_exit: bool
    reason: ExitReason
    exit_price: float
    exit_pct: float  # 청산 비율 (0.5 = 50%)
    pnl_pct: float  # 예상 손익 %
    message: str = ""


# =============================================================================
# Exit Level Calculation
# =============================================================================

def calc_exit_levels(
    entry_price: float,
    side: Literal['long', 'short'],
    atr: float,
    *,
    # SL 설정
    sl_atr_mult: float = 2.0,
    sl_min_pct: float = 1.0,
    sl_max_pct: float = 5.0,
    # TP 설정
    tp1_rr: float = 1.5,  # Risk:Reward 1.5x
    tp2_rr: float = 2.5,  # Risk:Reward 2.5x
    tp3_rr: float = 4.0,  # Risk:Reward 4.0x
    # Trailing 설정
    trailing_pct: float = 1.5,
    # Confluence 기반 TP
    confluence_zones: Optional[List] = None,
) -> ExitLevels:
    """
    청산 레벨 계산

    Args:
        entry_price: 진입가
        side: 'long' or 'short'
        atr: ATR 값
        sl_atr_mult: SL ATR 배수
        sl_min_pct: 최소 SL %
        sl_max_pct: 최대 SL %
        tp1_rr: TP1 Risk:Reward
        tp2_rr: TP2 Risk:Reward
        tp3_rr: TP3 Risk:Reward
        trailing_pct: 트레일링 거리 %
        confluence_zones: Confluence Zone 리스트 (TP 조정용)

    Returns:
        ExitLevels
    """
    # 1. ATR 기반 SL 계산
    sl_distance = atr * sl_atr_mult
    sl_pct = sl_distance / entry_price * 100

    # SL % 제한
    sl_pct = max(sl_min_pct, min(sl_max_pct, sl_pct))
    sl_distance = entry_price * sl_pct / 100

    if side == 'long':
        stop_loss = entry_price - sl_distance
    else:
        stop_loss = entry_price + sl_distance

    # 2. R:R 기반 TP 계산
    risk_amount = sl_distance

    if side == 'long':
        tp1 = entry_price + risk_amount * tp1_rr
        tp2 = entry_price + risk_amount * tp2_rr
        tp3 = entry_price + risk_amount * tp3_rr
    else:
        tp1 = entry_price - risk_amount * tp1_rr
        tp2 = entry_price - risk_amount * tp2_rr
        tp3 = entry_price - risk_amount * tp3_rr

    # 3. Confluence Zone 기반 TP 조정 (선택적)
    tp_reason = f"R:R based ({tp1_rr}/{tp2_rr}/{tp3_rr})"

    if confluence_zones:
        adjusted_tp1, adjusted_tp2, adj_reason = _adjust_tp_by_confluence(
            entry_price, side, tp1, tp2, confluence_zones
        )
        if adjusted_tp1:
            tp1 = adjusted_tp1
            tp_reason = adj_reason
        if adjusted_tp2:
            tp2 = adjusted_tp2

    # 4. TP %s
    tp1_pct = abs(tp1 - entry_price) / entry_price * 100
    tp2_pct = abs(tp2 - entry_price) / entry_price * 100
    tp3_pct = abs(tp3 - entry_price) / entry_price * 100

    # 5. Trailing 활성화 가격 (TP1 도달 시)
    trailing_activation = tp1

    return ExitLevels(
        entry_price=entry_price,
        side=side,

        stop_loss=stop_loss,
        stop_loss_pct=sl_pct,

        take_profit_1=tp1,
        take_profit_2=tp2,
        take_profit_3=tp3,
        tp1_pct=tp1_pct,
        tp2_pct=tp2_pct,
        tp3_pct=tp3_pct,

        trailing_activation=trailing_activation,
        trailing_distance_pct=trailing_pct,

        risk_reward_1=tp1_rr,
        risk_reward_2=tp2_rr,

        atr=atr,
        atr_mult_sl=sl_atr_mult,

        sl_reason=f"ATR({atr:.0f}) x {sl_atr_mult} = {sl_pct:.1f}%",
        tp_reason=tp_reason,
    )


def _adjust_tp_by_confluence(
    entry_price: float,
    side: str,
    tp1: float,
    tp2: float,
    zones: List,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Confluence Zone 기반 TP 조정

    - TP1/TP2 근처에 강한 저항/지지가 있으면 그 앞에서 청산
    """
    adjusted_tp1 = None
    adjusted_tp2 = None
    reason = ""

    for zone in zones:
        # PredictionZone 객체 가정
        zone_price = getattr(zone, 'price', None)
        zone_strength = getattr(zone, 'strength', 1)

        if zone_price is None:
            continue

        if side == 'long':
            # TP1 조정: 강한 저항이 TP1 이전에 있으면
            if entry_price < zone_price < tp1 and zone_strength >= 3:
                adjusted_tp1 = zone_price * 0.995  # 저항 약간 아래
                reason = f"Confluence resistance at ${zone_price:,.0f}"
        else:
            # Short TP1 조정: 강한 지지가 TP1 이전에 있으면
            if tp1 < zone_price < entry_price and zone_strength >= 3:
                adjusted_tp1 = zone_price * 1.005  # 지지 약간 위
                reason = f"Confluence support at ${zone_price:,.0f}"

    return adjusted_tp1, adjusted_tp2, reason


# =============================================================================
# Exit Signal Check
# =============================================================================

def check_exit_signal(
    current_price: float,
    exit_levels: ExitLevels,
    trailing_state: Optional[TrailingState] = None,
    position_pct: float = 1.0,  # 남은 포지션 비율
) -> Optional[ExitSignal]:
    """
    청산 신호 체크

    Args:
        current_price: 현재가
        exit_levels: 청산 레벨
        trailing_state: 트레일링 상태
        position_pct: 남은 포지션 비율 (1.0 = 100%)

    Returns:
        ExitSignal if should exit, else None
    """
    entry = exit_levels.entry_price
    side = exit_levels.side

    # PnL 계산
    if side == 'long':
        pnl_pct = (current_price - entry) / entry * 100
    else:
        pnl_pct = (entry - current_price) / entry * 100

    # 1. Stop Loss 체크
    if side == 'long' and current_price <= exit_levels.stop_loss:
        return ExitSignal(
            should_exit=True,
            reason=ExitReason.STOP_LOSS,
            exit_price=exit_levels.stop_loss,
            exit_pct=1.0,  # 전량 청산
            pnl_pct=-exit_levels.stop_loss_pct,
            message=f"Stop Loss triggered at ${exit_levels.stop_loss:,.0f}",
        )
    elif side == 'short' and current_price >= exit_levels.stop_loss:
        return ExitSignal(
            should_exit=True,
            reason=ExitReason.STOP_LOSS,
            exit_price=exit_levels.stop_loss,
            exit_pct=1.0,
            pnl_pct=-exit_levels.stop_loss_pct,
            message=f"Stop Loss triggered at ${exit_levels.stop_loss:,.0f}",
        )

    # 2. Take Profit 1 (50% 청산)
    if position_pct > 0.5:
        if side == 'long' and current_price >= exit_levels.take_profit_1:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT_1,
                exit_price=exit_levels.take_profit_1,
                exit_pct=0.5,
                pnl_pct=exit_levels.tp1_pct,
                message=f"TP1 hit at ${exit_levels.take_profit_1:,.0f} (+{exit_levels.tp1_pct:.1f}%)",
            )
        elif side == 'short' and current_price <= exit_levels.take_profit_1:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT_1,
                exit_price=exit_levels.take_profit_1,
                exit_pct=0.5,
                pnl_pct=exit_levels.tp1_pct,
                message=f"TP1 hit at ${exit_levels.take_profit_1:,.0f} (+{exit_levels.tp1_pct:.1f}%)",
            )

    # 3. Take Profit 2 (30% 청산)
    if 0.2 < position_pct <= 0.5:
        if side == 'long' and current_price >= exit_levels.take_profit_2:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT_2,
                exit_price=exit_levels.take_profit_2,
                exit_pct=0.6,  # 남은 50% 중 60% = 30%
                pnl_pct=exit_levels.tp2_pct,
                message=f"TP2 hit at ${exit_levels.take_profit_2:,.0f} (+{exit_levels.tp2_pct:.1f}%)",
            )
        elif side == 'short' and current_price <= exit_levels.take_profit_2:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT_2,
                exit_price=exit_levels.take_profit_2,
                exit_pct=0.6,
                pnl_pct=exit_levels.tp2_pct,
                message=f"TP2 hit at ${exit_levels.take_profit_2:,.0f} (+{exit_levels.tp2_pct:.1f}%)",
            )

    # 4. Take Profit 3 (전량 청산)
    if position_pct <= 0.2:
        if side == 'long' and current_price >= exit_levels.take_profit_3:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT_3,
                exit_price=exit_levels.take_profit_3,
                exit_pct=1.0,
                pnl_pct=exit_levels.tp3_pct,
                message=f"TP3 hit at ${exit_levels.take_profit_3:,.0f} (+{exit_levels.tp3_pct:.1f}%)",
            )
        elif side == 'short' and current_price <= exit_levels.take_profit_3:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT_3,
                exit_price=exit_levels.take_profit_3,
                exit_pct=1.0,
                pnl_pct=exit_levels.tp3_pct,
                message=f"TP3 hit at ${exit_levels.take_profit_3:,.0f} (+{exit_levels.tp3_pct:.1f}%)",
            )

    # 5. Trailing Stop (TP1 이후 활성화)
    if trailing_state and trailing_state.is_active:
        triggered, stop_price = trailing_state.update(
            current_price, side, exit_levels.trailing_distance_pct
        )
        if triggered:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                exit_price=stop_price,
                exit_pct=1.0,  # 남은 전량
                pnl_pct=pnl_pct,
                message=f"Trailing stop triggered at ${stop_price:,.0f}",
            )

    return None


# =============================================================================
# Position Manager
# =============================================================================

@dataclass
class Position:
    """포지션 정보"""
    entry_price: float
    side: Literal['long', 'short']
    size: float  # 계약 수량
    remaining_pct: float = 1.0  # 남은 비율
    exit_levels: Optional[ExitLevels] = None
    trailing_state: Optional[TrailingState] = None

    # 실현 손익
    realized_pnl: float = 0.0

    def activate_trailing(self):
        """TP1 도달 후 트레일링 활성화"""
        if self.trailing_state is None:
            self.trailing_state = TrailingState()

        self.trailing_state.is_active = True
        self.trailing_state.highest_price = self.entry_price
        self.trailing_state.lowest_price = self.entry_price

        if self.exit_levels:
            self.trailing_state.current_stop = self.exit_levels.stop_loss


def manage_position(
    position: Position,
    current_price: float,
) -> Optional[ExitSignal]:
    """
    포지션 관리 - 청산 신호 체크

    Args:
        position: 포지션 정보
        current_price: 현재가

    Returns:
        ExitSignal if should exit
    """
    if position.exit_levels is None:
        return None

    signal = check_exit_signal(
        current_price,
        position.exit_levels,
        position.trailing_state,
        position.remaining_pct,
    )

    if signal:
        # TP1 도달 시 트레일링 활성화
        if signal.reason == ExitReason.TAKE_PROFIT_1:
            position.activate_trailing()
            # 50% 청산 -> 남은 비율 = 0.5
            position.realized_pnl += signal.pnl_pct * signal.exit_pct
            position.remaining_pct = 1.0 - signal.exit_pct
        elif signal.reason == ExitReason.TAKE_PROFIT_2:
            # 남은 50% 중 60% 청산 = 30%
            exit_amount = position.remaining_pct * signal.exit_pct
            position.realized_pnl += signal.pnl_pct * exit_amount
            position.remaining_pct -= exit_amount
        elif signal.reason == ExitReason.TAKE_PROFIT_3:
            # 잔량 전부 청산
            position.realized_pnl += signal.pnl_pct * position.remaining_pct
            position.remaining_pct = 0.0
        elif signal.reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP]:
            position.realized_pnl += signal.pnl_pct * position.remaining_pct
            position.remaining_pct = 0.0

    return signal


# =============================================================================
# Convenience Functions
# =============================================================================

def calc_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    포지션 사이즈 계산 (리스크 기반)

    Args:
        equity: 총 자본
        risk_pct: 리스크 % (예: 1.0 = 1%)
        entry_price: 진입가
        stop_loss: 손절가

    Returns:
        계약 수량
    """
    risk_amount = equity * risk_pct / 100
    price_risk = abs(entry_price - stop_loss)

    if price_risk == 0:
        return 0.0

    return risk_amount / price_risk


def format_exit_levels(levels: ExitLevels) -> str:
    """청산 레벨 포맷팅"""
    lines = [
        "=" * 50,
        f"Exit Levels ({levels.side.upper()})",
        "=" * 50,
        f"Entry: ${levels.entry_price:,.0f}",
        "",
        f"Stop Loss: ${levels.stop_loss:,.0f} (-{levels.stop_loss_pct:.1f}%)",
        f"  Reason: {levels.sl_reason}",
        "",
        f"TP1 (50%): ${levels.take_profit_1:,.0f} (+{levels.tp1_pct:.1f}%) R:R={levels.risk_reward_1}",
        f"TP2 (30%): ${levels.take_profit_2:,.0f} (+{levels.tp2_pct:.1f}%) R:R={levels.risk_reward_2}",
        f"TP3 (20%): ${levels.take_profit_3:,.0f} (+{levels.tp3_pct:.1f}%)",
        f"  Reason: {levels.tp_reason}",
        "",
        f"Trailing: {levels.trailing_distance_pct}% (activates at TP1)",
        "=" * 50,
    ]
    return "\n".join(lines)
