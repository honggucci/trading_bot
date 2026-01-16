"""
Divergence Zone Calculator
==========================

RSI 다이버전스가 성립하는 가격 구간(경계/범위) 계산.

핵심 원칙:
- 다이버전스를 "신호(점)"가 아니라 "가격 영역(Zone)"으로 만든다
- 이분탐색으로 "이 가격이면 다이버전스 성립" 바운더리 계산
"""

from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple
import numpy as np
import pandas as pd


@dataclass
class DivZone:
    """다이버전스 Zone 표준 출력"""
    side: Literal['long', 'short']
    kind: Literal['regular', 'hidden']

    # 바운더리 (단일 경계 또는 구간)
    boundary_price: Optional[float]  # 이하/이상이면 성립 (Regular용)
    range_low: Optional[float]       # 구간 하단 (Hidden용)
    range_high: Optional[float]      # 구간 상단 (Hidden용)

    # 참조점 정보
    ref_price: float
    ref_rsi: float
    ref_ts: pd.Timestamp
    ref_idx: int

    # 메타
    confidence: float = 1.0          # 강도 점수 (0~1)

    def contains(self, price: float) -> bool:
        """가격이 다이버전스 Zone 내에 있는지"""
        if self.kind == 'regular':
            if self.boundary_price is None:
                return False
            if self.side == 'long':
                return price <= self.boundary_price
            else:  # short
                return price >= self.boundary_price
        else:  # hidden
            if self.range_low is None or self.range_high is None:
                return False
            return self.range_low <= price <= self.range_high


@dataclass
class OversoldSegment:
    """과매도/과매수 세그먼트"""
    start_idx: int
    end_idx: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    threshold: float
    side: Literal['oversold', 'overbought']


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 계산 (Wilder 방식)"""
    close = np.asarray(close, dtype=float)
    n = len(close)
    out = np.full(n, np.nan, dtype=float)

    if n < period + 1:
        return out

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    rs = (avg_gain / avg_loss) if avg_loss != 0 else np.inf
    out[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = 100.0 if avg_loss == 0 else (100.0 - 100.0 / (1.0 + (avg_gain / avg_loss)))

    return out


def calc_stoch_rsi(
    close: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """StochRSI %K, %D 계산"""
    rsi = calc_rsi(close, period=rsi_period)
    n = len(close)

    # Stoch of RSI
    stoch = np.full(n, np.nan)
    for i in range(stoch_period - 1, n):
        window = rsi[i - stoch_period + 1:i + 1]
        if np.all(np.isfinite(window)):
            lo, hi = np.min(window), np.max(window)
            if hi - lo > 0:
                stoch[i] = (rsi[i] - lo) / (hi - lo) * 100

    # SMA smoothing
    def sma(x, w):
        out = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            window = x[i - w + 1:i + 1]
            if np.all(np.isfinite(window)):
                out[i] = np.mean(window)
        return out

    k = sma(stoch, k_period)
    d = sma(k, d_period)

    return k, d


def find_oversold_segments(
    stoch_d: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float = 20.0,
) -> List[OversoldSegment]:
    """
    StochRSI %D 기준 과매도 세그먼트 탐지

    Returns:
        시간순 정렬된 과매도 세그먼트 리스트
    """
    n = len(stoch_d)
    segments = []
    i = 0

    while i < n:
        if np.isfinite(stoch_d[i]) and stoch_d[i] <= threshold:
            start = i
            while i < n and np.isfinite(stoch_d[i]) and stoch_d[i] <= threshold:
                i += 1
            end = i - 1

            segments.append(OversoldSegment(
                start_idx=start,
                end_idx=end,
                start_ts=timestamps[start],
                end_ts=timestamps[end],
                threshold=threshold,
                side='oversold',
            ))
        else:
            i += 1

    return segments


def find_overbought_segments(
    stoch_d: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float = 80.0,
) -> List[OversoldSegment]:
    """StochRSI %D 기준 과매수 세그먼트 탐지"""
    n = len(stoch_d)
    segments = []
    i = 0

    while i < n:
        if np.isfinite(stoch_d[i]) and stoch_d[i] >= threshold:
            start = i
            while i < n and np.isfinite(stoch_d[i]) and stoch_d[i] >= threshold:
                i += 1
            end = i - 1

            segments.append(OversoldSegment(
                start_idx=start,
                end_idx=end,
                start_ts=timestamps[start],
                end_ts=timestamps[end],
                threshold=threshold,
                side='overbought',
            ))
        else:
            i += 1

    return segments


def get_reference_from_segment(
    df: pd.DataFrame,
    segment: OversoldSegment,
    rsi_col: str = 'rsi',
) -> dict:
    """
    세그먼트에서 참조점 추출 (최저/최고 종가 지점)

    Returns:
        {'ref_idx', 'ref_ts', 'ref_price', 'ref_rsi'}
    """
    sub = df.iloc[segment.start_idx:segment.end_idx + 1]

    if segment.side == 'oversold':
        idx_extreme = sub['close'].idxmin()
    else:  # overbought
        idx_extreme = sub['close'].idxmax()

    ref_idx = df.index.get_loc(idx_extreme)

    return {
        'ref_idx': ref_idx,
        'ref_ts': idx_extreme,
        'ref_price': float(df.at[idx_extreme, 'close']),
        'ref_rsi': float(df.at[idx_extreme, rsi_col]) if rsi_col in df.columns else np.nan,
    }


def calc_regular_bullish_boundary(
    close: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    lower_bound: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Optional[float]:
    """
    Regular Bullish 다이버전스 바운더리 계산 (이분탐색)

    Regular Bullish: price < ref_price AND rsi > ref_rsi
    → "현재가가 여기 이하면 RSI가 ref_rsi보다 높아짐"

    Returns:
        바운더리 가격 (이 가격 이하에서 Regular Bullish 성립)
        None이면 다이버전스 불가
    """
    close = close.copy()
    eps = 1e-8

    def rsi_at(price: float) -> float:
        close[-1] = price
        rsi = calc_rsi(close, period=rsi_period)
        return rsi[-1]

    # 상한: ref_price보다 약간 낮은 가격
    U = ref_price - max(eps, abs(ref_price) * 1e-6)

    # 하한
    if lower_bound is None:
        L = U - max(1e-6, abs(U) * 0.10)
    else:
        L = min(lower_bound, U - max(1e-6, abs(U) * 0.001))

    if not np.isfinite(U) or L >= U:
        return None

    # U에서 RSI가 ref_rsi보다 높아야 다이버전스 가능
    rU = rsi_at(U)
    if not np.isfinite(rU) or rU <= ref_rsi:
        return None

    # 이분탐색: RSI = ref_rsi 되는 가격 찾기
    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rmid = rsi_at(mid)

        if not np.isfinite(rmid):
            lo = mid
            continue

        if rmid > ref_rsi:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) <= tol:
            break

    return float(min(hi, U))


def calc_hidden_bullish_range(
    close: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    upper_bound: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Optional[Tuple[float, float]]:
    """
    Hidden Bullish 다이버전스 범위 계산 (이분탐색)

    Hidden Bullish: price > ref_price AND rsi < ref_rsi
    → "현재가가 이 범위면 RSI가 ref_rsi보다 낮아짐"

    Returns:
        (range_low, range_high) 범위
        None이면 다이버전스 불가
    """
    close = close.copy()
    eps = 1e-8

    def rsi_at(price: float) -> float:
        close[-1] = price
        rsi = calc_rsi(close, period=rsi_period)
        return rsi[-1]

    # 하한: ref_price보다 약간 높은 가격
    L = ref_price + max(eps, abs(ref_price) * 1e-6)

    # 상한
    if upper_bound is None:
        U = L + max(1e-6, abs(L) * 0.10)
    else:
        U = max(upper_bound, L + max(1e-6, abs(L) * 0.001))

    if not np.isfinite(L) or L >= U:
        return None

    # L에서 RSI가 ref_rsi보다 낮아야 다이버전스 가능
    rL = rsi_at(L)
    if not np.isfinite(rL) or rL >= ref_rsi:
        return None

    # 이분탐색: RSI = ref_rsi 되는 상한 가격 찾기
    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rmid = rsi_at(mid)

        if not np.isfinite(rmid):
            hi = mid
            continue

        if rmid < ref_rsi:
            lo = mid
        else:
            hi = mid

        if abs(hi - lo) <= tol:
            break

    xmax = float(lo)
    if xmax <= L:
        return None

    return (float(L), xmax)


def calc_regular_bearish_boundary(
    close: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    upper_bound: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Optional[float]:
    """
    Regular Bearish 다이버전스 바운더리 계산

    Regular Bearish: price > ref_price AND rsi < ref_rsi
    → "현재가가 여기 이상이면 RSI가 ref_rsi보다 낮아짐"

    Returns:
        바운더리 가격 (이 가격 이상에서 Regular Bearish 성립)
    """
    close = close.copy()
    eps = 1e-8

    def rsi_at(price: float) -> float:
        close[-1] = price
        rsi = calc_rsi(close, period=rsi_period)
        return rsi[-1]

    # 하한: ref_price보다 약간 높은 가격
    L = ref_price + max(eps, abs(ref_price) * 1e-6)

    # 상한
    if upper_bound is None:
        U = L + max(1e-6, abs(L) * 0.10)
    else:
        U = max(upper_bound, L + max(1e-6, abs(L) * 0.001))

    if not np.isfinite(L) or L >= U:
        return None

    # L에서 RSI가 ref_rsi보다 낮아야 다이버전스 가능
    rL = rsi_at(L)
    if not np.isfinite(rL) or rL >= ref_rsi:
        return None

    return float(L)


def calc_hidden_bearish_range(
    close: np.ndarray,
    ref_price: float,
    ref_rsi: float,
    rsi_period: int = 14,
    lower_bound: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> Optional[Tuple[float, float]]:
    """
    Hidden Bearish 다이버전스 범위 계산

    Hidden Bearish: price < ref_price AND rsi > ref_rsi
    → "현재가가 이 범위면 RSI가 ref_rsi보다 높아짐"

    Returns:
        (range_low, range_high) 범위
    """
    close = close.copy()
    eps = 1e-8

    def rsi_at(price: float) -> float:
        close[-1] = price
        rsi = calc_rsi(close, period=rsi_period)
        return rsi[-1]

    # 상한: ref_price보다 약간 낮은 가격
    U = ref_price - max(eps, abs(ref_price) * 1e-6)

    # 하한
    if lower_bound is None:
        L = U - max(1e-6, abs(U) * 0.10)
    else:
        L = min(lower_bound, U - max(1e-6, abs(U) * 0.001))

    if not np.isfinite(U) or L >= U:
        return None

    # U에서 RSI가 ref_rsi보다 높아야 다이버전스 가능
    rU = rsi_at(U)
    if not np.isfinite(rU) or rU <= ref_rsi:
        return None

    # 이분탐색: RSI = ref_rsi 되는 하한 가격 찾기
    lo, hi = L, U
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rmid = rsi_at(mid)

        if not np.isfinite(rmid):
            lo = mid
            continue

        if rmid > ref_rsi:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) <= tol:
            break

    xmin = float(hi)
    if xmin >= U:
        return None

    return (xmin, float(U))


def get_div_zones(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_rsi_period: int = 14,
    oversold_threshold: float = 20.0,
    overbought_threshold: float = 80.0,
    use_current_segment: bool = False,
) -> List[DivZone]:
    """
    현재 시점에서 가능한 모든 다이버전스 Zone 계산

    Args:
        df: OHLCV DataFrame (index=timestamp)
        rsi_period: RSI 기간
        stoch_rsi_period: StochRSI 기간
        oversold_threshold: 과매도 기준 (StochRSI %D)
        overbought_threshold: 과매수 기준 (StochRSI %D)
        use_current_segment: True면 현재 진행중인 세그먼트 사용, False면 직전 완료된 세그먼트 사용

    Returns:
        DivZone 리스트
    """
    close = df['close'].astype(float).values

    # RSI 계산
    if 'rsi' not in df.columns:
        df = df.copy()
        df['rsi'] = calc_rsi(close, period=rsi_period)

    # StochRSI 계산
    if 'stoch_d' not in df.columns:
        df = df.copy()
        _, stoch_d = calc_stoch_rsi(close, rsi_period=stoch_rsi_period)
        df['stoch_d'] = stoch_d
    else:
        stoch_d = df['stoch_d'].values

    timestamps = df.index
    div_zones = []

    # === Long Side (과매도 세그먼트 기반) ===
    oversold_segs = find_oversold_segments(stoch_d, timestamps, oversold_threshold)

    if oversold_segs:
        # 현재가 과매도 상태인지 확인
        current_oversold = np.isfinite(stoch_d[-1]) and stoch_d[-1] <= oversold_threshold

        if current_oversold and not use_current_segment:
            # 현재 과매도 중이면 직전 세그먼트 사용
            if len(oversold_segs) >= 2:
                seg = oversold_segs[-2]
            else:
                seg = None
        else:
            seg = oversold_segs[-1] if oversold_segs else None

        if seg:
            ref = get_reference_from_segment(df, seg, rsi_col='rsi')

            # Regular Bullish
            reg_boundary = calc_regular_bullish_boundary(
                close, ref['ref_price'], ref['ref_rsi'], rsi_period
            )
            if reg_boundary is not None:
                div_zones.append(DivZone(
                    side='long',
                    kind='regular',
                    boundary_price=reg_boundary,
                    range_low=None,
                    range_high=None,
                    ref_price=ref['ref_price'],
                    ref_rsi=ref['ref_rsi'],
                    ref_ts=ref['ref_ts'],
                    ref_idx=ref['ref_idx'],
                ))

            # Hidden Bullish
            hid_range = calc_hidden_bullish_range(
                close, ref['ref_price'], ref['ref_rsi'], rsi_period
            )
            if hid_range is not None:
                div_zones.append(DivZone(
                    side='long',
                    kind='hidden',
                    boundary_price=None,
                    range_low=hid_range[0],
                    range_high=hid_range[1],
                    ref_price=ref['ref_price'],
                    ref_rsi=ref['ref_rsi'],
                    ref_ts=ref['ref_ts'],
                    ref_idx=ref['ref_idx'],
                ))

    # === Short Side (과매수 세그먼트 기반) ===
    overbought_segs = find_overbought_segments(stoch_d, timestamps, overbought_threshold)

    if overbought_segs:
        current_overbought = np.isfinite(stoch_d[-1]) and stoch_d[-1] >= overbought_threshold

        if current_overbought and not use_current_segment:
            if len(overbought_segs) >= 2:
                seg = overbought_segs[-2]
            else:
                seg = None
        else:
            seg = overbought_segs[-1] if overbought_segs else None

        if seg:
            ref = get_reference_from_segment(df, seg, rsi_col='rsi')

            # Regular Bearish
            reg_boundary = calc_regular_bearish_boundary(
                close, ref['ref_price'], ref['ref_rsi'], rsi_period
            )
            if reg_boundary is not None:
                div_zones.append(DivZone(
                    side='short',
                    kind='regular',
                    boundary_price=reg_boundary,
                    range_low=None,
                    range_high=None,
                    ref_price=ref['ref_price'],
                    ref_rsi=ref['ref_rsi'],
                    ref_ts=ref['ref_ts'],
                    ref_idx=ref['ref_idx'],
                ))

            # Hidden Bearish
            hid_range = calc_hidden_bearish_range(
                close, ref['ref_price'], ref['ref_rsi'], rsi_period
            )
            if hid_range is not None:
                div_zones.append(DivZone(
                    side='short',
                    kind='hidden',
                    boundary_price=None,
                    range_low=hid_range[0],
                    range_high=hid_range[1],
                    ref_price=ref['ref_price'],
                    ref_rsi=ref['ref_rsi'],
                    ref_ts=ref['ref_ts'],
                    ref_idx=ref['ref_idx'],
                ))

    return div_zones
