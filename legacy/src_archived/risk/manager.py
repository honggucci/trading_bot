# -*- coding: utf-8 -*-
"""
Risk Manager - Position & Loss Limits
======================================

리스크 관리 모듈.

핵심 기능:
1. Max Position Size - 계좌 대비 최대 포지션
2. Daily Loss Limit - 일일 최대 손실
3. Concurrent Position Limit - 동시 포지션 수
4. Circuit Breaker - 연속 손실 시 거래 중단

사용법:
```python
from src.risk.manager import RiskManager, RiskConfig

manager = RiskManager(
    equity=100000,
    config=RiskConfig(
        max_position_pct=10.0,
        daily_loss_limit_pct=3.0,
    )
)

# 진입 전 체크
ok, reason = manager.can_open_position(size=0.1, price=95000)
if not ok:
    print(f"Blocked: {reason}")

# 거래 기록
manager.record_trade(pnl=-500)

# 일일 한도 체크
if manager.is_daily_limit_reached():
    print("Daily limit reached!")
```
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, date
import numpy as np


@dataclass
class RiskConfig:
    """리스크 설정"""
    # Position Limits
    max_position_pct: float = 10.0  # 계좌 대비 최대 포지션 %
    max_position_usd: float = 50000.0  # 절대 최대 포지션 USD
    max_concurrent_positions: int = 3  # 동시 포지션 수

    # Loss Limits
    daily_loss_limit_pct: float = 3.0  # 일일 최대 손실 %
    weekly_loss_limit_pct: float = 10.0  # 주간 최대 손실 %
    per_trade_loss_limit_pct: float = 1.0  # 거래당 최대 손실 %

    # Circuit Breaker
    consecutive_loss_limit: int = 3  # 연속 손실 횟수
    cooldown_minutes: int = 60  # 쿨다운 시간 (분)

    # Size Limits
    min_size: float = 0.001  # 최소 거래량
    max_leverage: float = 5.0  # 최대 레버리지


@dataclass
class DailyStats:
    """일일 통계"""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0


class RiskManager:
    """리스크 관리자"""

    def __init__(
        self,
        equity: float,
        config: Optional[RiskConfig] = None,
    ):
        """
        Args:
            equity: 초기 자본
            config: 리스크 설정
        """
        self.initial_equity = equity
        self.equity = equity
        self.config = config or RiskConfig()

        # 상태
        self.open_positions: int = 0
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None

        # 일일 통계
        self.daily_stats = DailyStats(
            date=date.today(),
            peak_equity=equity,
        )

        # 거래 히스토리
        self.trade_history: List[float] = []

    def reset_daily(self):
        """일일 통계 리셋"""
        today = date.today()
        if self.daily_stats.date != today:
            self.daily_stats = DailyStats(
                date=today,
                peak_equity=self.equity,
            )

    def can_open_position(
        self,
        size: float,
        price: float,
        side: str = 'long',
    ) -> Tuple[bool, str]:
        """
        포지션 진입 가능 여부 체크

        Args:
            size: 포지션 크기 (계약 수량)
            price: 현재가
            side: 'long' or 'short'

        Returns:
            (allowed, reason)
        """
        self.reset_daily()

        # 1. Circuit Breaker 체크
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.now()).seconds // 60
            return False, f"Circuit breaker active ({remaining}m remaining)"

        # 2. 동시 포지션 수 체크
        if self.open_positions >= self.config.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.config.max_concurrent_positions}) reached"

        # 3. 포지션 크기 체크
        position_value = size * price

        # 계좌 대비 %
        position_pct = position_value / self.equity * 100
        if position_pct > self.config.max_position_pct:
            return False, f"Position size {position_pct:.1f}% exceeds limit {self.config.max_position_pct}%"

        # 절대 금액
        if position_value > self.config.max_position_usd:
            return False, f"Position ${position_value:,.0f} exceeds limit ${self.config.max_position_usd:,.0f}"

        # 최소 크기
        if size < self.config.min_size:
            return False, f"Size {size} below minimum {self.config.min_size}"

        # 4. 일일 손실 한도 체크
        if self.is_daily_limit_reached():
            return False, f"Daily loss limit ({self.config.daily_loss_limit_pct}%) reached"

        # 5. 연속 손실 체크
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            return False, f"Consecutive losses ({self.consecutive_losses}) reached limit"

        return True, ""

    def record_trade(
        self,
        pnl: float,
        is_closed: bool = True,
    ):
        """
        거래 기록

        Args:
            pnl: 손익 (USD)
            is_closed: 포지션 종료 여부
        """
        self.reset_daily()

        # 자본 업데이트
        self.equity += pnl
        self.trade_history.append(pnl)

        # 일일 통계 업데이트
        self.daily_stats.trades += 1
        self.daily_stats.pnl += pnl

        if pnl > 0:
            self.daily_stats.wins += 1
            self.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.consecutive_losses += 1

        # Peak equity 업데이트
        if self.equity > self.daily_stats.peak_equity:
            self.daily_stats.peak_equity = self.equity

        # Drawdown 계산
        drawdown = (self.daily_stats.peak_equity - self.equity) / self.daily_stats.peak_equity * 100
        self.daily_stats.max_drawdown = max(self.daily_stats.max_drawdown, drawdown)

        # 포지션 카운트
        if is_closed:
            self.open_positions = max(0, self.open_positions - 1)

        # Circuit Breaker 체크
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            self._activate_circuit_breaker()

    def open_position(self):
        """포지션 오픈 기록"""
        self.open_positions += 1

    def close_position(self):
        """포지션 클로즈 기록"""
        self.open_positions = max(0, self.open_positions - 1)

    def is_daily_limit_reached(self) -> bool:
        """일일 손실 한도 도달 여부"""
        self.reset_daily()

        if self.daily_stats.pnl >= 0:
            return False

        loss_pct = abs(self.daily_stats.pnl) / self.initial_equity * 100
        return loss_pct >= self.config.daily_loss_limit_pct

    def _activate_circuit_breaker(self):
        """Circuit Breaker 활성화"""
        from datetime import timedelta
        self.cooldown_until = datetime.now() + timedelta(minutes=self.config.cooldown_minutes)

    def reset_circuit_breaker(self):
        """Circuit Breaker 리셋"""
        self.cooldown_until = None
        self.consecutive_losses = 0

    def get_max_position_size(self, price: float) -> float:
        """
        허용된 최대 포지션 크기 계산

        Args:
            price: 현재가

        Returns:
            최대 계약 수량
        """
        # 계좌 대비 % 기준
        max_by_pct = self.equity * self.config.max_position_pct / 100 / price

        # 절대 금액 기준
        max_by_usd = self.config.max_position_usd / price

        return min(max_by_pct, max_by_usd)

    def get_position_size_by_risk(
        self,
        price: float,
        stop_loss_pct: float,
    ) -> float:
        """
        리스크 기반 포지션 크기 계산

        Args:
            price: 현재가
            stop_loss_pct: 손절 거리 %

        Returns:
            계약 수량
        """
        # 거래당 리스크
        risk_per_trade = self.equity * self.config.per_trade_loss_limit_pct / 100

        # SL 거리
        sl_distance = price * stop_loss_pct / 100

        if sl_distance <= 0:
            return 0.0

        size = risk_per_trade / sl_distance

        # 최대 크기 제한
        max_size = self.get_max_position_size(price)
        return min(size, max_size)

    def get_status(self) -> dict:
        """현재 상태 반환"""
        self.reset_daily()

        return {
            'equity': self.equity,
            'initial_equity': self.initial_equity,
            'pnl_pct': (self.equity - self.initial_equity) / self.initial_equity * 100,
            'open_positions': self.open_positions,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self.cooldown_until is not None and datetime.now() < self.cooldown_until,
            'daily_stats': {
                'date': str(self.daily_stats.date),
                'trades': self.daily_stats.trades,
                'wins': self.daily_stats.wins,
                'losses': self.daily_stats.losses,
                'pnl': self.daily_stats.pnl,
                'win_rate': self.daily_stats.wins / self.daily_stats.trades * 100 if self.daily_stats.trades > 0 else 0,
                'max_drawdown': self.daily_stats.max_drawdown,
            },
        }

    def format_status(self) -> str:
        """상태 포맷팅"""
        status = self.get_status()
        daily = status['daily_stats']

        lines = [
            "=" * 50,
            "Risk Manager Status",
            "=" * 50,
            f"Equity: ${status['equity']:,.0f} ({status['pnl_pct']:+.2f}%)",
            f"Open Positions: {status['open_positions']}/{self.config.max_concurrent_positions}",
            f"Consecutive Losses: {status['consecutive_losses']}/{self.config.consecutive_loss_limit}",
            f"Circuit Breaker: {'ACTIVE' if status['circuit_breaker_active'] else 'OFF'}",
            "",
            f"--- Daily ({daily['date']}) ---",
            f"Trades: {daily['trades']} (W:{daily['wins']} L:{daily['losses']})",
            f"Win Rate: {daily['win_rate']:.1f}%",
            f"PnL: ${daily['pnl']:+,.0f}",
            f"Max Drawdown: {daily['max_drawdown']:.2f}%",
            "=" * 50,
        ]

        return "\n".join(lines)
