# -*- coding: utf-8 -*-
"""
Backtest Engine Test
====================

백테스트 엔진 단위 테스트.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    Trade,
    MockHMMGate,
)


def generate_mock_data(
    days: int = 30,
    start_price: float = 95000,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Mock 15분봉 데이터 생성"""
    bars_per_day = 96  # 24 * 4
    total_bars = days * bars_per_day

    # 타임스탬프
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(total_bars)]

    # 가격 생성 (랜덤 워크 + 약간의 트렌드)
    np.random.seed(42)
    returns = np.random.normal(0.0001, volatility / np.sqrt(bars_per_day), total_bars)
    prices = start_price * np.cumprod(1 + returns)

    # OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, volatility / 2, total_bars)),
        'low': prices * (1 - np.random.uniform(0, volatility / 2, total_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, total_bars),
    }, index=pd.DatetimeIndex(timestamps))

    return df


class TestBacktestEngine:
    """BacktestEngine 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        engine = BacktestEngine(initial_equity=50000)
        assert engine.initial_equity == 50000
        assert engine.equity == 50000
        print("[PASS] Engine initialization")

    def test_reset(self):
        """리셋 테스트"""
        engine = BacktestEngine()
        engine.equity = 50000
        engine.trades = [Trade(entry_time=datetime.now(), entry_price=100, side='long', size=1)]

        engine.reset()
        assert engine.equity == engine.initial_equity
        assert len(engine.trades) == 0
        print("[PASS] Engine reset")

    def test_mock_data_generation(self):
        """Mock 데이터 생성 테스트"""
        df = generate_mock_data(days=7)
        assert len(df) == 7 * 96
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        print(f"[PASS] Mock data: {len(df)} bars, price range ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")

    def test_run_simple(self):
        """간단한 백테스트 실행"""
        df = generate_mock_data(days=30)
        engine = BacktestEngine(initial_equity=100000)

        # V2로 실행 (Mock 데이터이므로 MockHMMGate 사용)
        result = engine.run(df, signal_version='v2', use_mock_hmm=True)

        assert isinstance(result, BacktestResult)
        assert result.initial_equity == 100000
        assert len(result.equity_curve) > 0
        print(f"[PASS] Simple backtest: {result.total_trades} trades, final equity ${result.equity_curve[-1]:,.0f}")

    def test_result_metrics(self):
        """결과 메트릭 계산 테스트"""
        # 가짜 거래로 결과 생성
        engine = BacktestEngine(initial_equity=100000)

        # 수동으로 거래 추가
        engine.trades = [
            Trade(entry_time=datetime.now(), entry_price=95000, side='long', size=0.1,
                  exit_price=96000, pnl=100, pnl_pct=0.1),
            Trade(entry_time=datetime.now(), entry_price=96000, side='long', size=0.1,
                  exit_price=95500, pnl=-50, pnl_pct=-0.05),
            Trade(entry_time=datetime.now(), entry_price=95500, side='long', size=0.1,
                  exit_price=97000, pnl=150, pnl_pct=0.15),
        ]
        engine.equity_curve = [100000, 100100, 100050, 100200]

        df = generate_mock_data(days=1)
        result = engine._calculate_results('v2', df)

        assert result.total_trades == 3
        assert result.winning_trades == 2
        assert result.losing_trades == 1
        assert result.win_rate == pytest.approx(66.67, rel=0.1)
        print(f"[PASS] Metrics: {result.total_trades} trades, {result.win_rate:.1f}% win rate")

    def test_compare_results(self):
        """V2 vs V3 비교 테스트"""
        v2 = BacktestResult(
            total_trades=50,
            win_rate=55.0,
            total_pnl_pct=10.5,
            sharpe_ratio=1.2,
            max_drawdown_pct=8.0,
            profit_factor=1.5,
        )
        v3 = BacktestResult(
            total_trades=45,
            win_rate=62.0,
            total_pnl_pct=15.3,
            sharpe_ratio=1.8,
            max_drawdown_pct=6.5,
            profit_factor=2.1,
        )

        comparison = BacktestEngine.compare(v2, v3)

        assert len(comparison['metric']) == 9
        assert 'V3' in comparison['winner']  # V3가 더 나은 메트릭 있어야 함
        print("[PASS] Comparison test")

    def test_print_report(self):
        """리포트 출력 테스트"""
        result = BacktestResult(
            total_trades=30,
            winning_trades=18,
            losing_trades=12,
            total_pnl=5000,
            total_pnl_pct=5.0,
            win_rate=60.0,
            max_drawdown=2000,
            max_drawdown_pct=2.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            profit_factor=1.8,
            avg_win=0.5,
            avg_loss=-0.3,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            trading_days=30,
            signal_version='v3',
            initial_equity=100000,
        )

        print("\n")
        BacktestEngine.print_report(result)
        print("[PASS] Report printed successfully")

    def test_generate_signal_v2(self):
        """V2 신호 생성 직접 테스트"""
        df = generate_mock_data(days=30)
        engine = BacktestEngine(initial_equity=100000)

        # ATR 추가
        df = engine._add_atr(df.copy())

        # Mock HMM Gate
        hmm_gate = MockHMMGate()
        current_time = df.index[150]

        # _generate_signal 직접 호출
        signal = engine._generate_signal(
            df.iloc[:151],
            hmm_gate,
            current_time,
            version='v2',
            tf_predictor=None,
        )

        # 신호 구조 검증
        assert signal is None or isinstance(signal, dict)
        if signal:
            assert 'allowed' in signal
            assert 'side' in signal
            assert 'confidence' in signal
            assert 'size_mult' in signal
            print(f"[PASS] V2 signal: allowed={signal['allowed']}, side={signal['side']}, conf={signal['confidence']:.2f}")
        else:
            print("[PASS] V2 signal: None (no signal condition)")

    def test_generate_signal_graceful_fallback(self):
        """빈 데이터에서 graceful fallback"""
        engine = BacktestEngine(initial_equity=100000)

        # 빈 DataFrame
        empty_df = pd.DataFrame()
        hmm_gate = MockHMMGate()

        signal = engine._generate_signal(
            empty_df,
            hmm_gate,
            datetime.now(),
            version='v2',
            tf_predictor=None,
        )

        # None 또는 allowed=False
        if signal is None:
            print("[PASS] Signal generation returns None on empty data")
        else:
            assert signal['allowed'] is False
            print("[PASS] Signal generation returns allowed=False on empty data")


def run_all_tests():
    """모든 테스트 실행"""
    import pytest

    print("=" * 60)
    print("Backtest Engine Unit Tests")
    print("=" * 60)

    test = TestBacktestEngine()
    test.test_initialization()
    test.test_reset()
    test.test_mock_data_generation()
    test.test_run_simple()

    # pytest 필요한 테스트
    try:
        test.test_result_metrics()
    except NameError:
        print("[SKIP] pytest not imported for approx test")

    test.test_compare_results()
    test.test_print_report()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    import pytest
    run_all_tests()