# -*- coding: utf-8 -*-
"""
Timeframe Module Tests
======================

Tests for src/utils/timeframe.py
"""
import pytest
from src.utils.timeframe import (
    TimeframeSpec,
    Duration,
    duration_to_bars,
    TIMEFRAME_MINUTES,
    MINUTES_PER_DAY,
    TRADING_DAYS_PER_YEAR,
    TF_5M,
    TF_15M,
    TF_1H,
    TF_4H,
    TF_1D,
    TF_1W,
    TF_HIERARCHY,
    get_lower_timeframe,
    get_higher_timeframe,
    get_fallback_chain,
)


class TestTimeframeSpec:
    """TimeframeSpec 테스트"""

    def test_from_string_15m(self):
        tf = TimeframeSpec.from_string("15m")
        assert tf.name == "15m"
        assert tf.minutes == 15

    def test_from_string_1h(self):
        tf = TimeframeSpec.from_string("1h")
        assert tf.name == "1h"
        assert tf.minutes == 60

    def test_from_string_case_insensitive(self):
        tf = TimeframeSpec.from_string("1H")
        assert tf.name == "1h"
        assert tf.minutes == 60

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            TimeframeSpec.from_string("invalid")

    def test_bars_per_day_15m(self):
        tf = TimeframeSpec.from_string("15m")
        assert tf.bars_per_day == 96  # 1440 / 15 = 96

    def test_bars_per_day_5m(self):
        tf = TimeframeSpec.from_string("5m")
        assert tf.bars_per_day == 288  # 1440 / 5 = 288

    def test_bars_per_day_1h(self):
        tf = TimeframeSpec.from_string("1h")
        assert tf.bars_per_day == 24  # 1440 / 60 = 24

    def test_bars_per_year(self):
        tf = TimeframeSpec.from_string("15m")
        expected = TRADING_DAYS_PER_YEAR * 96
        assert tf.bars_per_year == expected

    def test_str(self):
        tf = TimeframeSpec.from_string("15m")
        assert str(tf) == "15m"


class TestDuration:
    """Duration 테스트"""

    def test_parse_1d(self):
        d = Duration.parse("1d")
        assert d.value == 1
        assert d.unit == "d"

    def test_parse_2d(self):
        d = Duration.parse("2d")
        assert d.value == 2
        assert d.unit == "d"

    def test_parse_6h(self):
        d = Duration.parse("6h")
        assert d.value == 6
        assert d.unit == "h"

    def test_parse_30m(self):
        d = Duration.parse("30m")
        assert d.value == 30
        assert d.unit == "m"

    def test_parse_1w(self):
        d = Duration.parse("1w")
        assert d.value == 1
        assert d.unit == "w"

    def test_parse_float(self):
        d = Duration.parse("1.5d")
        assert d.value == 1.5
        assert d.unit == "d"

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            Duration.parse("invalid")

    def test_parse_empty(self):
        with pytest.raises(ValueError):
            Duration.parse("")

    def test_total_minutes_1d(self):
        d = Duration.parse("1d")
        assert d.total_minutes == 1440

    def test_total_minutes_6h(self):
        d = Duration.parse("6h")
        assert d.total_minutes == 360

    def test_total_minutes_1w(self):
        d = Duration.parse("1w")
        assert d.total_minutes == 10080

    def test_to_bars_1d_15m(self):
        d = Duration.parse("1d")
        assert d.to_bars(TF_15M) == 96

    def test_to_bars_2d_15m(self):
        d = Duration.parse("2d")
        assert d.to_bars(TF_15M) == 192

    def test_to_bars_1d_5m(self):
        d = Duration.parse("1d")
        assert d.to_bars(TF_5M) == 288

    def test_to_bars_2d_5m(self):
        d = Duration.parse("2d")
        assert d.to_bars(TF_5M) == 576

    def test_to_bars_6h_1h(self):
        d = Duration.parse("6h")
        assert d.to_bars(TF_1H) == 6

    def test_to_bars_minimum_1(self):
        """Very short duration should return at least 1 bar"""
        d = Duration.parse("1m")
        assert d.to_bars(TF_1D) >= 1

    def test_str(self):
        d = Duration.parse("1d")
        assert str(d) == "1d"

    def test_str_float(self):
        d = Duration.parse("1.5d")
        assert str(d) == "1.5d"


class TestDurationToBars:
    """duration_to_bars() 함수 테스트"""

    def test_1d_to_15m(self):
        """1 day = 96 bars at 15m"""
        assert duration_to_bars("1d", "15m") == 96

    def test_2d_to_15m(self):
        """2 days = 192 bars at 15m"""
        assert duration_to_bars("2d", "15m") == 192

    def test_1d_to_5m(self):
        """1 day = 288 bars at 5m"""
        assert duration_to_bars("1d", "5m") == 288

    def test_2d_to_5m(self):
        """2 days = 576 bars at 5m"""
        assert duration_to_bars("2d", "5m") == 576

    def test_6h_to_1h(self):
        """6 hours = 6 bars at 1h"""
        assert duration_to_bars("6h", "1h") == 6

    def test_48h_to_1h(self):
        """48 hours = 48 bars at 1h (2 days)"""
        assert duration_to_bars("48h", "1h") == 48

    def test_1w_to_1d(self):
        """1 week = 7 bars at 1d"""
        assert duration_to_bars("1w", "1d") == 7

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            duration_to_bars("invalid", "15m")

    def test_invalid_timeframe(self):
        with pytest.raises(ValueError):
            duration_to_bars("1d", "invalid")


class TestPredefinedTimeframes:
    """Pre-defined timeframe constants"""

    def test_TF_5M(self):
        assert TF_5M.name == "5m"
        assert TF_5M.minutes == 5
        assert TF_5M.bars_per_day == 288

    def test_TF_15M(self):
        assert TF_15M.name == "15m"
        assert TF_15M.minutes == 15
        assert TF_15M.bars_per_day == 96

    def test_TF_1H(self):
        assert TF_1H.name == "1h"
        assert TF_1H.minutes == 60
        assert TF_1H.bars_per_day == 24

    def test_TF_4H(self):
        assert TF_4H.name == "4h"
        assert TF_4H.minutes == 240
        assert TF_4H.bars_per_day == 6

    def test_TF_1D(self):
        assert TF_1D.name == "1d"
        assert TF_1D.minutes == 1440
        assert TF_1D.bars_per_day == 1


class TestMagicNumberReplacement:
    """Magic number 대체 검증 - 기존 하드코딩 값과 일치 확인"""

    def test_prob_gate_n_atr_15m(self):
        """ProbGateConfig default: n_atr=96 (15m, 1 day)"""
        assert duration_to_bars("1d", "15m") == 96

    def test_prob_gate_vol_window_15m(self):
        """ProbGateConfig default: vol_window=192 (15m, 2 days)"""
        assert duration_to_bars("2d", "15m") == 192

    def test_backtest_n_atr_5m(self):
        """backtest_strategy_compare: 96*3=288 (5m, 1 day)"""
        assert duration_to_bars("1d", "5m") == 288

    def test_backtest_vol_window_5m(self):
        """backtest_strategy_compare: 192*3=576 (5m, 2 days)"""
        assert duration_to_bars("2d", "5m") == 576

    def test_hilbert_detrend_span(self):
        """HilbertScoreConfig: detrend_span=48 (1H, 2 days)"""
        assert duration_to_bars("2d", "1h") == 48

    def test_instability_window_15m(self):
        """ProbGateConfig: instability_window=96 (15m, 1 day)"""
        assert duration_to_bars("1d", "15m") == 96

    def test_instability_ref_window_15m(self):
        """ProbGateConfig: instability_ref_window=384 (15m, 4 days)"""
        assert duration_to_bars("4d", "15m") == 384


class TestTimeframeHierarchy:
    """Timeframe hierarchy and fallback tests"""

    def test_hierarchy_order(self):
        """TF_HIERARCHY is ordered from highest to lowest"""
        assert TF_HIERARCHY == ['1w', '1d', '4h', '1h', '15m', '5m', '1m']

    def test_hierarchy_length(self):
        """Hierarchy contains all main timeframes"""
        assert len(TF_HIERARCHY) == 7


class TestGetLowerTimeframe:
    """get_lower_timeframe() tests - TF fallback logic"""

    def test_4h_to_1h(self):
        """4h → 1h"""
        assert get_lower_timeframe('4h') == '1h'

    def test_1h_to_15m(self):
        """1h → 15m"""
        assert get_lower_timeframe('1h') == '15m'

    def test_15m_to_5m(self):
        """15m → 5m (다이버전스 폴백)"""
        assert get_lower_timeframe('15m') == '5m'

    def test_5m_to_1m(self):
        """5m → 1m"""
        assert get_lower_timeframe('5m') == '1m'

    def test_1m_returns_none(self):
        """1m is lowest, returns None"""
        assert get_lower_timeframe('1m') is None

    def test_1d_to_4h(self):
        """1d → 4h"""
        assert get_lower_timeframe('1d') == '4h'

    def test_1w_to_1d(self):
        """1w → 1d"""
        assert get_lower_timeframe('1w') == '1d'

    def test_case_insensitive(self):
        """Case insensitive input"""
        assert get_lower_timeframe('4H') == '1h'
        assert get_lower_timeframe('1H') == '15m'

    def test_invalid_tf_raises(self):
        """Invalid timeframe raises ValueError"""
        with pytest.raises(ValueError):
            get_lower_timeframe('2h')


class TestGetHigherTimeframe:
    """get_higher_timeframe() tests"""

    def test_15m_to_1h(self):
        """15m → 1h"""
        assert get_higher_timeframe('15m') == '1h'

    def test_1h_to_4h(self):
        """1h → 4h"""
        assert get_higher_timeframe('1h') == '4h'

    def test_4h_to_1d(self):
        """4h → 1d"""
        assert get_higher_timeframe('4h') == '1d'

    def test_1d_to_1w(self):
        """1d → 1w"""
        assert get_higher_timeframe('1d') == '1w'

    def test_1w_returns_none(self):
        """1w is highest, returns None"""
        assert get_higher_timeframe('1w') is None

    def test_5m_to_15m(self):
        """5m → 15m"""
        assert get_higher_timeframe('5m') == '15m'

    def test_1m_to_5m(self):
        """1m → 5m"""
        assert get_higher_timeframe('1m') == '5m'

    def test_case_insensitive(self):
        """Case insensitive input"""
        assert get_higher_timeframe('15M') == '1h'

    def test_invalid_tf_raises(self):
        """Invalid timeframe raises ValueError"""
        with pytest.raises(ValueError):
            get_higher_timeframe('3h')


class TestGetFallbackChain:
    """get_fallback_chain() tests - divergence fallback chain"""

    def test_1h_to_5m(self):
        """1h fallback to 5m: ['1h', '15m', '5m']"""
        assert get_fallback_chain('1h', '5m') == ['1h', '15m', '5m']

    def test_15m_to_5m(self):
        """15m fallback to 5m: ['15m', '5m']"""
        assert get_fallback_chain('15m', '5m') == ['15m', '5m']

    def test_4h_to_15m(self):
        """4h fallback to 15m: ['4h', '1h', '15m']"""
        assert get_fallback_chain('4h', '15m') == ['4h', '1h', '15m']

    def test_1d_to_1h(self):
        """1d fallback to 1h: ['1d', '4h', '1h']"""
        assert get_fallback_chain('1d', '1h') == ['1d', '4h', '1h']

    def test_full_chain_1w_to_1m(self):
        """Full fallback chain from 1w to 1m"""
        assert get_fallback_chain('1w', '1m') == ['1w', '1d', '4h', '1h', '15m', '5m', '1m']

    def test_same_tf(self):
        """Same TF returns single element"""
        assert get_fallback_chain('15m', '15m') == ['15m']

    def test_default_min_tf(self):
        """Default min_tf is 1m"""
        assert get_fallback_chain('15m') == ['15m', '5m', '1m']

    def test_case_insensitive(self):
        """Case insensitive input"""
        assert get_fallback_chain('4H', '15M') == ['4h', '1h', '15m']

    def test_invalid_tf_raises(self):
        """Invalid tf raises ValueError"""
        with pytest.raises(ValueError):
            get_fallback_chain('2h', '1m')

    def test_invalid_min_tf_raises(self):
        """Invalid min_tf raises ValueError"""
        with pytest.raises(ValueError):
            get_fallback_chain('1h', '2m')

    def test_min_higher_than_tf_raises(self):
        """min_tf higher than tf raises ValueError"""
        with pytest.raises(ValueError):
            get_fallback_chain('15m', '1h')
