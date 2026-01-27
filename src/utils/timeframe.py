"""
Timeframe Configuration Module
==============================

Provides timeframe-agnostic configuration:
- TimeframeSpec: Defines bar duration
- Duration: Human-readable time periods
- duration_to_bars(): Single conversion point

Usage:
    from src.utils.timeframe import duration_to_bars, TimeframeSpec

    # Convert "1 day" to bar count for 15m timeframe
    bars = duration_to_bars("1d", "15m")  # -> 96

    # Same duration for 5m timeframe
    bars_5m = duration_to_bars("1d", "5m")  # -> 288

Magic Number Reference (for documentation):
    15m: 1d = 96 bars, 2d = 192 bars
    5m:  1d = 288 bars, 2d = 576 bars
    1h:  1d = 24 bars, 2d = 48 bars
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict
import re


# Predefined timeframe mappings (minutes per bar)
TIMEFRAME_MINUTES: Dict[str, int] = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '6h': 360,
    '12h': 720,
    '1d': 1440,
    '1w': 10080,
}

# Common constants
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440  # 24 * 60
MINUTES_PER_WEEK = 10080  # 7 * 1440
TRADING_DAYS_PER_YEAR = 252  # For Sharpe ratio annualization


@dataclass(frozen=True)
class TimeframeSpec:
    """Immutable timeframe specification."""
    name: str
    minutes: int

    @classmethod
    def from_string(cls, tf: str) -> TimeframeSpec:
        """
        Parse timeframe string like '15m', '1h', '1d'.

        Args:
            tf: Timeframe string (e.g., '15m', '1h', '4h', '1d')

        Returns:
            TimeframeSpec instance

        Raises:
            ValueError: If timeframe is not recognized
        """
        tf_lower = tf.lower().strip()
        if tf_lower not in TIMEFRAME_MINUTES:
            valid = list(TIMEFRAME_MINUTES.keys())
            raise ValueError(f"Unknown timeframe: '{tf}'. Valid options: {valid}")
        return cls(name=tf_lower, minutes=TIMEFRAME_MINUTES[tf_lower])

    @property
    def bars_per_day(self) -> float:
        """Number of bars in 24 hours (1440 minutes)."""
        return MINUTES_PER_DAY / self.minutes

    @property
    def bars_per_year(self) -> float:
        """For annualization (trading days = 252)."""
        return TRADING_DAYS_PER_YEAR * self.bars_per_day

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Duration:
    """
    Time duration that converts to bars for any timeframe.

    Examples:
        Duration(1, 'd').to_bars(TF_15M)  # -> 96
        Duration(2, 'd').to_bars(TF_5M)   # -> 576
        Duration(6, 'h').to_bars(TF_1H)   # -> 6
    """
    value: float
    unit: Literal['m', 'h', 'd', 'w']

    @property
    def total_minutes(self) -> int:
        """Total duration in minutes."""
        multipliers = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080}
        return int(self.value * multipliers[self.unit])

    def to_bars(self, tf: TimeframeSpec) -> int:
        """
        Convert duration to bar count for given timeframe.

        Args:
            tf: TimeframeSpec defining bar duration

        Returns:
            Number of bars (minimum 1)
        """
        bars = self.total_minutes / tf.minutes
        return max(1, int(round(bars)))

    @classmethod
    def parse(cls, s: str) -> Duration:
        """
        Parse duration string like '1d', '2d', '6h', '30m'.

        Args:
            s: Duration string (e.g., '1d', '2d', '6h', '30m', '1w')

        Returns:
            Duration instance

        Raises:
            ValueError: If format is invalid
        """
        match = re.match(r'^(\d+(?:\.\d+)?)\s*(m|h|d|w)$', s.strip().lower())
        if not match:
            raise ValueError(
                f"Invalid duration format: '{s}'. "
                "Expected format: '1d', '6h', '30m', '1w', etc."
            )
        return cls(value=float(match.group(1)), unit=match.group(2))

    def __str__(self) -> str:
        if self.value == int(self.value):
            return f"{int(self.value)}{self.unit}"
        return f"{self.value}{self.unit}"


def duration_to_bars(duration: str, timeframe: str) -> int:
    """
    Convert duration string to bar count for a specific timeframe.

    This is the SINGLE conversion point for the codebase.
    Use this instead of hardcoding magic numbers like 96, 192, 288.

    Args:
        duration: Time period string (e.g., "1d", "2d", "6h", "30m")
        timeframe: Bar timeframe string (e.g., "15m", "5m", "1h")

    Returns:
        Number of bars

    Examples:
        >>> duration_to_bars("1d", "15m")
        96
        >>> duration_to_bars("2d", "15m")
        192
        >>> duration_to_bars("1d", "5m")
        288
        >>> duration_to_bars("2d", "5m")
        576
        >>> duration_to_bars("6h", "1h")
        6
        >>> duration_to_bars("48h", "1h")
        48

    Raises:
        ValueError: If duration or timeframe format is invalid
    """
    d = Duration.parse(duration)
    tf = TimeframeSpec.from_string(timeframe)
    return d.to_bars(tf)


# Pre-defined commonly used timeframes (convenience)
TF_1M = TimeframeSpec.from_string('1m')
TF_5M = TimeframeSpec.from_string('5m')
TF_15M = TimeframeSpec.from_string('15m')
TF_30M = TimeframeSpec.from_string('30m')
TF_1H = TimeframeSpec.from_string('1h')
TF_4H = TimeframeSpec.from_string('4h')
TF_2H = TimeframeSpec.from_string('2h')
TF_1D = TimeframeSpec.from_string('1d')
TF_1W = TimeframeSpec.from_string('1w')


# ============================================================
# Timeframe Hierarchy (for fallback logic)
# ============================================================
# Ordered from highest (1w) to lowest (1m)
TF_HIERARCHY: list[str] = ['1w', '1d', '4h', '2h', '1h', '15m', '5m', '1m']


def get_lower_timeframe(tf: str) -> str | None:
    """
    Get the next lower timeframe in the hierarchy.

    Fallback logic: 4h → 1h → 15m → 5m → 1m

    Args:
        tf: Current timeframe string (e.g., '15m', '1h', '4h')

    Returns:
        Next lower timeframe string, or None if already at lowest (1m)

    Examples:
        >>> get_lower_timeframe('4h')
        '1h'
        >>> get_lower_timeframe('1h')
        '15m'
        >>> get_lower_timeframe('15m')
        '5m'
        >>> get_lower_timeframe('1m')
        None
    """
    tf_lower = tf.lower().strip()
    if tf_lower not in TF_HIERARCHY:
        raise ValueError(f"Unknown timeframe: '{tf}'. Valid: {TF_HIERARCHY}")

    idx = TF_HIERARCHY.index(tf_lower)
    if idx >= len(TF_HIERARCHY) - 1:
        return None  # Already at lowest (1m)
    return TF_HIERARCHY[idx + 1]


def get_higher_timeframe(tf: str) -> str | None:
    """
    Get the next higher timeframe in the hierarchy.

    Args:
        tf: Current timeframe string (e.g., '15m', '1h', '4h')

    Returns:
        Next higher timeframe string, or None if already at highest (1w)

    Examples:
        >>> get_higher_timeframe('15m')
        '1h'
        >>> get_higher_timeframe('1h')
        '4h'
        >>> get_higher_timeframe('1w')
        None
    """
    tf_lower = tf.lower().strip()
    if tf_lower not in TF_HIERARCHY:
        raise ValueError(f"Unknown timeframe: '{tf}'. Valid: {TF_HIERARCHY}")

    idx = TF_HIERARCHY.index(tf_lower)
    if idx <= 0:
        return None  # Already at highest (1w)
    return TF_HIERARCHY[idx - 1]


def get_fallback_chain(tf: str, min_tf: str = '1m') -> list[str]:
    """
    Get the full fallback chain from current TF down to min_tf.

    Useful for divergence fallback: try 15m → 5m → 1m until success.

    Args:
        tf: Starting timeframe
        min_tf: Minimum timeframe to fall back to (default: '1m')

    Returns:
        List of timeframes from current to min_tf (inclusive)

    Examples:
        >>> get_fallback_chain('1h', '5m')
        ['1h', '15m', '5m']
        >>> get_fallback_chain('15m', '5m')
        ['15m', '5m']
        >>> get_fallback_chain('4h', '15m')
        ['4h', '1h', '15m']
    """
    tf_lower = tf.lower().strip()
    min_lower = min_tf.lower().strip()

    if tf_lower not in TF_HIERARCHY:
        raise ValueError(f"Unknown timeframe: '{tf}'. Valid: {TF_HIERARCHY}")
    if min_lower not in TF_HIERARCHY:
        raise ValueError(f"Unknown min_tf: '{min_tf}'. Valid: {TF_HIERARCHY}")

    start_idx = TF_HIERARCHY.index(tf_lower)
    end_idx = TF_HIERARCHY.index(min_lower)

    if start_idx > end_idx:
        raise ValueError(f"min_tf '{min_tf}' is higher than tf '{tf}'")

    return TF_HIERARCHY[start_idx:end_idx + 1]
