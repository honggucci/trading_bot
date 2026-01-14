# -*- coding: utf-8 -*-
"""
Exit Logic Test
===============

청산 로직 단위 테스트.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.anchor.exit_logic import (
    ExitLevels,
    ExitSignal,
    ExitReason,
    TrailingState,
    Position,
    calc_exit_levels,
    check_exit_signal,
    manage_position,
    calc_position_size,
    format_exit_levels,
)


class TestCalcExitLevels:
    """calc_exit_levels() 테스트"""

    def test_long_basic(self):
        """Long 기본 케이스"""
        levels = calc_exit_levels(
            entry_price=95000,
            side='long',
            atr=1500,
            sl_atr_mult=2.0,
        )

        assert levels.entry_price == 95000
        assert levels.side == 'long'

        # SL = entry - ATR * mult = 95000 - 3000 = 92000
        assert levels.stop_loss == pytest.approx(92000, rel=0.01)

        # TP1 = entry + risk * 1.5 = 95000 + 3000 * 1.5 = 99500
        assert levels.take_profit_1 == pytest.approx(99500, rel=0.01)

        # TP2 = entry + risk * 2.5 = 95000 + 3000 * 2.5 = 102500
        assert levels.take_profit_2 == pytest.approx(102500, rel=0.01)

        print(f"[PASS] Long basic: SL=${levels.stop_loss:,.0f}, TP1=${levels.take_profit_1:,.0f}")

    def test_short_basic(self):
        """Short 기본 케이스"""
        levels = calc_exit_levels(
            entry_price=95000,
            side='short',
            atr=1500,
            sl_atr_mult=2.0,
        )

        # SL = entry + ATR * mult = 95000 + 3000 = 98000
        assert levels.stop_loss == pytest.approx(98000, rel=0.01)

        # TP1 = entry - risk * 1.5 = 95000 - 3000 * 1.5 = 90500
        assert levels.take_profit_1 == pytest.approx(90500, rel=0.01)

        print(f"[PASS] Short basic: SL=${levels.stop_loss:,.0f}, TP1=${levels.take_profit_1:,.0f}")

    def test_sl_min_max_clamp(self):
        """SL % 최소/최대 제한"""
        # 매우 작은 ATR -> min_pct 적용
        levels = calc_exit_levels(
            entry_price=100000,
            side='long',
            atr=100,  # 0.2% 거리
            sl_atr_mult=2.0,
            sl_min_pct=1.0,
        )

        # 0.2% < 1.0% 이므로 1% 적용
        assert levels.stop_loss_pct == pytest.approx(1.0, rel=0.1)

        # 매우 큰 ATR -> max_pct 적용
        levels2 = calc_exit_levels(
            entry_price=100000,
            side='long',
            atr=5000,  # 10% 거리
            sl_atr_mult=2.0,
            sl_max_pct=5.0,
        )

        # 10% > 5% 이므로 5% 적용
        assert levels2.stop_loss_pct == pytest.approx(5.0, rel=0.1)

        print(f"[PASS] SL clamp: min={levels.stop_loss_pct:.1f}%, max={levels2.stop_loss_pct:.1f}%")

    def test_risk_reward_ratio(self):
        """R:R 비율 확인"""
        levels = calc_exit_levels(
            entry_price=100000,
            side='long',
            atr=2000,
            sl_atr_mult=2.0,
            tp1_rr=2.0,
            tp2_rr=3.0,
            tp3_rr=5.0,
        )

        risk = abs(levels.entry_price - levels.stop_loss)
        reward1 = abs(levels.take_profit_1 - levels.entry_price)
        reward2 = abs(levels.take_profit_2 - levels.entry_price)

        assert reward1 / risk == pytest.approx(2.0, rel=0.01)
        assert reward2 / risk == pytest.approx(3.0, rel=0.01)

        print(f"[PASS] R:R verified: TP1 R:R={reward1/risk:.1f}, TP2 R:R={reward2/risk:.1f}")


class TestCheckExitSignal:
    """check_exit_signal() 테스트"""

    @pytest.fixture
    def long_levels(self):
        return calc_exit_levels(
            entry_price=95000,
            side='long',
            atr=1500,
        )

    def test_stop_loss_triggered(self, long_levels):
        """SL 트리거"""
        signal = check_exit_signal(
            current_price=91000,  # SL 아래
            exit_levels=long_levels,
        )

        assert signal is not None
        assert signal.should_exit is True
        assert signal.reason == ExitReason.STOP_LOSS
        assert signal.exit_pct == 1.0  # 전량 청산

        print(f"[PASS] SL triggered: {signal.message}")

    def test_tp1_triggered(self, long_levels):
        """TP1 트리거"""
        signal = check_exit_signal(
            current_price=100000,  # TP1 위
            exit_levels=long_levels,
            position_pct=1.0,
        )

        assert signal is not None
        assert signal.reason == ExitReason.TAKE_PROFIT_1
        assert signal.exit_pct == 0.5  # 50% 청산

        print(f"[PASS] TP1 triggered: {signal.message}")

    def test_tp2_triggered(self, long_levels):
        """TP2 트리거 (포지션 50% 남음)"""
        signal = check_exit_signal(
            current_price=103000,  # TP2 위
            exit_levels=long_levels,
            position_pct=0.5,  # TP1 이후
        )

        assert signal is not None
        assert signal.reason == ExitReason.TAKE_PROFIT_2
        assert signal.exit_pct == 0.6  # 남은 50% 중 60% = 30%

        print(f"[PASS] TP2 triggered: {signal.message}")

    def test_no_signal(self, long_levels):
        """청산 조건 미충족"""
        signal = check_exit_signal(
            current_price=96000,  # 중간
            exit_levels=long_levels,
        )

        assert signal is None

        print("[PASS] No signal in middle zone")


class TestTrailingState:
    """TrailingState 테스트"""

    def test_long_trailing_update(self):
        """Long 트레일링 업데이트"""
        state = TrailingState(is_active=True)
        state.highest_price = 100000
        state.current_stop = 98500  # 1.5% 아래

        # 가격 상승 -> 스탑 따라감
        triggered, new_stop = state.update(102000, 'long', 1.5)
        assert triggered is False
        assert new_stop == pytest.approx(102000 * 0.985, rel=0.01)
        assert state.highest_price == 102000

        # 가격 하락 -> 스탑 유지
        triggered, new_stop = state.update(101000, 'long', 1.5)
        assert triggered is False
        assert state.highest_price == 102000  # 변동 없음

        # 스탑 도달 -> 트리거
        triggered, stop_price = state.update(state.current_stop - 100, 'long', 1.5)
        assert triggered is True

        print("[PASS] Long trailing stop works correctly")

    def test_short_trailing_update(self):
        """Short 트레일링 업데이트"""
        state = TrailingState(is_active=True)
        state.lowest_price = 95000
        state.current_stop = 96425  # 1.5% 위

        # 가격 하락 -> 스탑 따라감
        triggered, new_stop = state.update(93000, 'short', 1.5)
        assert triggered is False
        assert new_stop == pytest.approx(93000 * 1.015, rel=0.01)
        assert state.lowest_price == 93000

        print("[PASS] Short trailing stop works correctly")


class TestPositionManagement:
    """Position 관리 테스트"""

    def test_position_lifecycle(self):
        """포지션 라이프사이클"""
        levels = calc_exit_levels(
            entry_price=95000,
            side='long',
            atr=1500,
        )

        position = Position(
            entry_price=95000,
            side='long',
            size=0.1,
            exit_levels=levels,
        )

        # 1. 초기 상태
        assert position.remaining_pct == 1.0
        assert position.realized_pnl == 0.0

        # 2. TP1 도달
        signal = manage_position(position, 100000)
        assert signal.reason == ExitReason.TAKE_PROFIT_1
        assert position.remaining_pct == pytest.approx(0.5, rel=0.1)
        assert position.trailing_state.is_active is True

        # 3. TP2 도달
        signal = manage_position(position, 103000)
        assert signal.reason == ExitReason.TAKE_PROFIT_2
        assert position.remaining_pct < 0.3

        print(f"[PASS] Position lifecycle: remaining={position.remaining_pct:.1%}, realized={position.realized_pnl:.1f}%")

    def test_position_lifecycle_short(self):
        """Short 포지션 라이프사이클"""
        levels = calc_exit_levels(
            entry_price=95000,
            side='short',
            atr=1500,
        )

        position = Position(
            entry_price=95000,
            side='short',
            size=0.1,
            exit_levels=levels,
        )

        # 1. 초기 상태
        assert position.remaining_pct == 1.0
        assert position.realized_pnl == 0.0

        # 2. TP1 도달 (Short이므로 가격 하락)
        signal = manage_position(position, 90000)
        assert signal.reason == ExitReason.TAKE_PROFIT_1
        assert position.remaining_pct == pytest.approx(0.5, rel=0.1)
        assert position.trailing_state.is_active is True

        # 3. TP2 도달 (더 하락)
        signal = manage_position(position, 87000)
        assert signal.reason == ExitReason.TAKE_PROFIT_2
        assert position.remaining_pct < 0.3

        print(f"[PASS] Short position lifecycle: remaining={position.remaining_pct:.1%}, realized={position.realized_pnl:.1f}%")


class TestCalcPositionSize:
    """포지션 사이즈 계산 테스트"""

    def test_risk_based_sizing(self):
        """리스크 기반 사이즈 계산"""
        size = calc_position_size(
            equity=100000,
            risk_pct=1.0,  # 1% 리스크
            entry_price=95000,
            stop_loss=92000,  # $3000 거리
        )

        # 리스크 금액 = 100000 * 1% = 1000
        # 가격 리스크 = 3000
        # 사이즈 = 1000 / 3000 = 0.333
        assert size == pytest.approx(0.333, rel=0.01)

        print(f"[PASS] Position size: {size:.4f} BTC for 1% risk")

    def test_zero_risk_distance(self):
        """리스크 거리 0 처리"""
        size = calc_position_size(
            equity=100000,
            risk_pct=1.0,
            entry_price=95000,
            stop_loss=95000,  # 같은 가격
        )

        assert size == 0.0

        print("[PASS] Zero risk distance handled")


class TestFormatExitLevels:
    """포맷팅 테스트"""

    def test_format_output(self):
        """출력 포맷"""
        levels = calc_exit_levels(
            entry_price=95000,
            side='long',
            atr=1500,
        )

        output = format_exit_levels(levels)

        assert "Exit Levels (LONG)" in output
        assert "Stop Loss:" in output
        assert "TP1 (50%):" in output
        assert "Trailing:" in output

        print("[PASS] Format output:")
        print(output)


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Exit Logic Unit Tests")
    print("=" * 60)

    # calc_exit_levels
    test_levels = TestCalcExitLevels()
    test_levels.test_long_basic()
    test_levels.test_short_basic()
    test_levels.test_sl_min_max_clamp()
    test_levels.test_risk_reward_ratio()

    # check_exit_signal
    levels = calc_exit_levels(entry_price=95000, side='long', atr=1500)
    test_signal = TestCheckExitSignal()
    test_signal.test_stop_loss_triggered(levels)
    test_signal.test_tp1_triggered(levels)
    test_signal.test_tp2_triggered(levels)
    test_signal.test_no_signal(levels)

    # TrailingState
    test_trailing = TestTrailingState()
    test_trailing.test_long_trailing_update()
    test_trailing.test_short_trailing_update()

    # Position management
    test_pos = TestPositionManagement()
    test_pos.test_position_lifecycle()

    # Position sizing
    test_size = TestCalcPositionSize()
    test_size.test_risk_based_sizing()
    test_size.test_zero_risk_distance()

    # Format
    test_fmt = TestFormatExitLevels()
    test_fmt.test_format_output()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()