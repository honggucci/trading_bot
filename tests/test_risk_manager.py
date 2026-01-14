# -*- coding: utf-8 -*-
"""
Risk Manager Tests
==================

RiskManager 단위 테스트.
"""
import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# trading_bot 경로 추가
TRADING_BOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(TRADING_BOT_PATH / "src"))

from risk import RiskManager, RiskConfig


class TestRiskConfig:
    """RiskConfig 기본값 테스트"""

    def test_default_values(self):
        cfg = RiskConfig()
        assert cfg.max_position_pct == 10.0
        assert cfg.daily_loss_limit_pct == 3.0
        assert cfg.consecutive_loss_limit == 3
        assert cfg.cooldown_minutes == 60
        assert cfg.min_size == 0.001

    def test_custom_values(self):
        cfg = RiskConfig(
            max_position_pct=5.0,
            daily_loss_limit_pct=2.0,
            consecutive_loss_limit=5,
        )
        assert cfg.max_position_pct == 5.0
        assert cfg.daily_loss_limit_pct == 2.0
        assert cfg.consecutive_loss_limit == 5


class TestRiskManagerInit:
    """RiskManager 초기화 테스트"""

    def test_init_default_config(self):
        rm = RiskManager(equity=100000)
        assert rm.equity == 100000
        assert rm.initial_equity == 100000
        assert rm.config.max_position_pct == 10.0

    def test_init_custom_config(self):
        cfg = RiskConfig(max_position_pct=5.0)
        rm = RiskManager(equity=50000, config=cfg)
        assert rm.equity == 50000
        assert rm.config.max_position_pct == 5.0


class TestCanOpenPosition:
    """can_open_position() 테스트"""

    def test_normal_position(self):
        """정상 포지션 허용"""
        rm = RiskManager(equity=100000)
        # 10000 USD 포지션 (10% of 100000) - 정확히 limit
        ok, reason = rm.can_open_position(size=0.1, price=100000)
        assert ok is True
        assert reason == ""

    def test_position_too_large_pct(self):
        """계좌 대비 % 초과"""
        rm = RiskManager(equity=100000)
        # 15000 USD 포지션 (15% of 100000) - limit 10% 초과
        ok, reason = rm.can_open_position(size=0.15, price=100000)
        assert ok is False
        assert "exceeds limit" in reason

    def test_position_too_large_usd(self):
        """절대 금액 초과"""
        cfg = RiskConfig(max_position_usd=10000)
        rm = RiskManager(equity=1000000, config=cfg)
        # 20000 USD 포지션 - limit 10000 초과
        ok, reason = rm.can_open_position(size=0.2, price=100000)
        assert ok is False
        assert "exceeds limit $" in reason

    def test_position_too_small(self):
        """최소 크기 미만"""
        rm = RiskManager(equity=100000)
        ok, reason = rm.can_open_position(size=0.0001, price=100000)
        assert ok is False
        assert "below minimum" in reason

    def test_max_concurrent_positions(self):
        """동시 포지션 수 초과"""
        cfg = RiskConfig(max_concurrent_positions=2)
        rm = RiskManager(equity=100000, config=cfg)
        rm.open_positions = 2  # 이미 2개 오픈
        ok, reason = rm.can_open_position(size=0.01, price=100000)
        assert ok is False
        assert "Max concurrent" in reason

    def test_consecutive_loss_limit(self):
        """연속 손실 초과"""
        cfg = RiskConfig(consecutive_loss_limit=3)
        rm = RiskManager(equity=100000, config=cfg)
        rm.consecutive_losses = 3
        ok, reason = rm.can_open_position(size=0.01, price=100000)
        assert ok is False
        assert "Consecutive losses" in reason

    def test_daily_loss_limit(self):
        """일일 손실 한도 초과"""
        cfg = RiskConfig(daily_loss_limit_pct=3.0)
        rm = RiskManager(equity=100000, config=cfg)
        # 3000 USD 손실 기록 (3% of initial)
        rm.daily_stats.pnl = -3000
        ok, reason = rm.can_open_position(size=0.01, price=100000)
        assert ok is False
        assert "Daily loss limit" in reason


class TestCircuitBreaker:
    """Circuit Breaker 테스트"""

    def test_circuit_breaker_activation(self):
        """연속 손실 시 Circuit Breaker 활성화"""
        cfg = RiskConfig(consecutive_loss_limit=3, cooldown_minutes=30)
        rm = RiskManager(equity=100000, config=cfg)

        # 3번 연속 손실
        rm.record_trade(pnl=-100)
        rm.record_trade(pnl=-100)
        rm.record_trade(pnl=-100)

        # Circuit breaker 활성화 확인
        assert rm.cooldown_until is not None
        assert rm.cooldown_until > datetime.now()

    def test_circuit_breaker_blocks_trading(self):
        """Circuit Breaker 활성화 시 거래 차단"""
        cfg = RiskConfig(consecutive_loss_limit=2, cooldown_minutes=30)
        rm = RiskManager(equity=100000, config=cfg)

        rm.record_trade(pnl=-100)
        rm.record_trade(pnl=-100)

        ok, reason = rm.can_open_position(size=0.01, price=100000)
        assert ok is False
        assert "Circuit breaker" in reason

    def test_circuit_breaker_reset(self):
        """Circuit Breaker 수동 리셋"""
        cfg = RiskConfig(consecutive_loss_limit=2)
        rm = RiskManager(equity=100000, config=cfg)

        rm.record_trade(pnl=-100)
        rm.record_trade(pnl=-100)

        rm.reset_circuit_breaker()
        assert rm.cooldown_until is None
        assert rm.consecutive_losses == 0


class TestRecordTrade:
    """record_trade() 테스트"""

    def test_winning_trade(self):
        """수익 거래 기록"""
        rm = RiskManager(equity=100000)
        rm.record_trade(pnl=500)

        assert rm.equity == 100500
        assert rm.daily_stats.wins == 1
        assert rm.daily_stats.losses == 0
        assert rm.daily_stats.pnl == 500

    def test_losing_trade(self):
        """손실 거래 기록"""
        rm = RiskManager(equity=100000)
        rm.record_trade(pnl=-300)

        assert rm.equity == 99700
        assert rm.daily_stats.wins == 0
        assert rm.daily_stats.losses == 1
        assert rm.daily_stats.pnl == -300

    def test_consecutive_losses_count(self):
        """연속 손실 카운트"""
        rm = RiskManager(equity=100000)
        rm.record_trade(pnl=-100)
        rm.record_trade(pnl=-100)
        assert rm.consecutive_losses == 2

        # 수익 시 리셋
        rm.record_trade(pnl=100)
        assert rm.consecutive_losses == 0

    def test_drawdown_calculation(self):
        """Drawdown 계산"""
        rm = RiskManager(equity=100000)

        # 먼저 수익
        rm.record_trade(pnl=2000)  # equity = 102000, peak = 102000

        # 손실
        rm.record_trade(pnl=-1000)  # equity = 101000, peak 유지

        # Drawdown = (102000 - 101000) / 102000 * 100 ≈ 0.98%
        assert rm.daily_stats.max_drawdown > 0
        assert rm.daily_stats.max_drawdown < 1.0


class TestPositionSizing:
    """포지션 사이징 테스트"""

    def test_get_max_position_size(self):
        """최대 포지션 크기 계산"""
        cfg = RiskConfig(max_position_pct=10.0, max_position_usd=50000)
        rm = RiskManager(equity=100000, config=cfg)

        # 10% of 100000 = 10000 USD
        # max_by_pct = 10000 / 100000 (price) = 0.1 BTC
        # max_by_usd = 50000 / 100000 (price) = 0.5 BTC
        # min(0.1, 0.5) = 0.1
        max_size = rm.get_max_position_size(price=100000)
        assert max_size == 0.1

    def test_get_position_size_by_risk(self):
        """리스크 기반 포지션 크기 계산"""
        cfg = RiskConfig(per_trade_loss_limit_pct=1.0)
        rm = RiskManager(equity=100000, config=cfg)

        # risk_per_trade = 100000 * 1% = 1000 USD
        # stop_loss_pct = 2%
        # sl_distance = 100000 * 2% = 2000 USD
        # size = 1000 / 2000 = 0.5 BTC
        # But max_size might limit this
        size = rm.get_position_size_by_risk(price=100000, stop_loss_pct=2.0)
        assert size > 0
        assert size <= rm.get_max_position_size(100000)

    def test_position_size_zero_sl(self):
        """SL이 0일 때"""
        rm = RiskManager(equity=100000)
        size = rm.get_position_size_by_risk(price=100000, stop_loss_pct=0.0)
        assert size == 0.0


class TestDailyStats:
    """일일 통계 테스트"""

    def test_win_rate(self):
        """승률 계산"""
        rm = RiskManager(equity=100000)
        rm.record_trade(pnl=100)  # win
        rm.record_trade(pnl=100)  # win
        rm.record_trade(pnl=-50)  # loss

        status = rm.get_status()
        assert status['daily_stats']['win_rate'] == pytest.approx(66.67, rel=0.01)

    def test_format_status(self):
        """상태 포맷팅"""
        rm = RiskManager(equity=100000)
        rm.record_trade(pnl=500)

        status_str = rm.format_status()
        assert "Risk Manager Status" in status_str
        assert "Equity:" in status_str
        assert "Open Positions:" in status_str


class TestOpenClosePosition:
    """포지션 오픈/클로즈 카운트 테스트"""

    def test_open_position(self):
        """포지션 오픈"""
        rm = RiskManager(equity=100000)
        rm.open_position()
        assert rm.open_positions == 1

    def test_close_position(self):
        """포지션 클로즈"""
        rm = RiskManager(equity=100000)
        rm.open_positions = 2
        rm.close_position()
        assert rm.open_positions == 1

    def test_close_position_floor(self):
        """클로즈 시 0 이하로 내려가지 않음"""
        rm = RiskManager(equity=100000)
        rm.close_position()
        rm.close_position()
        assert rm.open_positions == 0


class TestPessimistScenarios:
    """Pessimist: 최악의 시나리오 테스트"""

    def test_consecutive_losses_circuit_breaker(self):
        """연속 손실 → Circuit Breaker 작동"""
        rm = RiskManager(equity=100000, config=RiskConfig(consecutive_loss_limit=3))

        # 3연속 손실
        rm.record_trade(pnl=-500, is_closed=True)
        rm.record_trade(pnl=-500, is_closed=True)
        rm.record_trade(pnl=-500, is_closed=True)

        # Circuit breaker 활성화 (cooldown_until 설정됨)
        assert rm.cooldown_until is not None
        can_open, reason = rm.can_open_position(0.01, 95000, 'long')
        assert can_open is False
        assert "Circuit" in reason or "breaker" in reason

    def test_daily_loss_limit_exceeded(self):
        """일일 손실 한도 초과"""
        rm = RiskManager(equity=100000, config=RiskConfig(daily_loss_limit_pct=3.0))

        # 3000불 손실 (3%)
        rm.record_trade(pnl=-3000, is_closed=True)

        can_open, reason = rm.can_open_position(0.01, 95000, 'long')
        assert can_open is False
        assert "daily" in reason.lower() or "loss" in reason.lower()

    def test_extreme_equity_drawdown(self):
        """극단적 equity 감소 (90% 손실)"""
        rm = RiskManager(equity=100000)

        # 90% 손실 시뮬레이션
        rm.equity = 10000

        # 여전히 동작해야 함 (crash 방지)
        max_size = rm.get_max_position_size(95000)
        assert max_size > 0
        assert max_size < 0.1  # 매우 작은 포지션만 허용

    def test_all_limits_triggered(self):
        """모든 제한이 동시에 트리거"""
        rm = RiskManager(
            equity=100000,
            config=RiskConfig(
                max_position_pct=5.0,
                daily_loss_limit_pct=2.0,
                consecutive_loss_limit=2,
            )
        )

        # 연속 손실
        rm.record_trade(pnl=-1000, is_closed=True)
        rm.record_trade(pnl=-1500, is_closed=True)  # 2연속 + 2500불 손실

        # 모두 트리거됨 (cooldown_until 설정됨)
        assert rm.cooldown_until is not None
        can_open, reason = rm.can_open_position(0.1, 95000, 'long')
        assert can_open is False

    def test_zero_equity_handled(self):
        """equity가 0일 때 crash 방지"""
        rm = RiskManager(equity=0)

        # Division by zero 방지
        try:
            max_size = rm.get_max_position_size(95000)
            assert max_size == 0 or max_size >= 0  # 에러 없이 0 이상
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError should not occur")

    def test_negative_pnl_streak(self):
        """10연속 손실 시나리오"""
        rm = RiskManager(equity=100000, config=RiskConfig(consecutive_loss_limit=5))

        # 10연속 손실
        for i in range(10):
            rm.record_trade(pnl=-100, is_closed=True)

        # 상태 확인 (cooldown 활성화됨)
        assert rm.consecutive_losses >= 5
        assert rm.cooldown_until is not None
        assert rm.daily_stats.trades == 10
        assert rm.daily_stats.losses == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])