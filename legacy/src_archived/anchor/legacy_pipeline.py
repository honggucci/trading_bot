"""
Legacy Pipeline - 전체 Confluence 분석 파이프라인
=================================================

레거시 로직의 전체 흐름을 통합한 파이프라인.

흐름:
1. ZigZag → 스윙 고점/저점 찾기
2. Fib 레벨 계산 → 되돌림/확장 존 생성
3. StochRSI %D <= 20 세그먼트 → REF (기준점) 추출
4. REF 기준 Divergence 경계 계산 (REG/HID)
5. Fib Zone + Divergence Confluence 점수 계산
6. 최적 진입 존 찾기

사용 예시:
```python
from src.anchor.legacy_pipeline import analyze_confluence

result = analyze_confluence(df_15m)
if result.best_zone and result.is_in_zone:
    print(f"진입 신호! Zone: {result.best_zone.zone.label}, Score: {result.best_zone.score:.2f}")
```
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from ..context.zigzag import zigzag_pivots, get_latest_swing, add_pivot_columns, SwingInfo
from ..context.fibonacci import calc_fib_levels, build_fib_zones, FibLevels, FibLevel

from .stochrsi import (
    tv_stoch_rsi,
    pick_oversold_segment_with_rule,
    extract_ref_from_segment,
    RefPoint,
)
from .divergence import (
    calc_rsi_wilder,
    check_divergence_at_current,
    needed_close_for_regular,
    feasible_range_for_hidden,
)
from .confluence import (
    calc_zone_confluence_scores,
    find_best_entry_zone,
    is_price_in_confluence_zone,
    ZoneScore,
)


@dataclass
class ConfluenceResult:
    """Confluence 분석 결과"""
    # 기본 정보
    success: bool
    error: Optional[str] = None

    # 스윙 정보
    swing: Optional[SwingInfo] = None
    fib_levels: Optional[FibLevels] = None

    # REF (기준점)
    ref: Optional[RefPoint] = None
    ref_reason: str = ""

    # Divergence 정보
    is_reg_now: bool = False  # 현재 Regular Divergence 성립
    is_hid_now: bool = False  # 현재 Hidden Divergence 성립
    need_reg: Optional[float] = None  # Regular 경계 가격
    hid_range: Optional[Tuple[float, float]] = None  # Hidden 범위

    # Confluence 존
    zones: List[FibLevel] = field(default_factory=list)
    zone_scores: List[ZoneScore] = field(default_factory=list)
    best_zone: Optional[ZoneScore] = None

    # 진입 판단
    is_in_zone: bool = False
    current_zone: Optional[ZoneScore] = None


def analyze_confluence(
    df: pd.DataFrame,
    *,
    # ZigZag 파라미터
    zig_up_pct: float = 0.02,
    zig_down_pct: float = 0.02,
    zig_atr_period: int = 14,
    zig_atr_mult: float = 2.0,
    zig_threshold_mode: str = 'or',
    zig_min_bars: int = 5,
    zig_min_swing_atr: float = 1.0,
    # StochRSI 파라미터
    stochrsi_period: int = 14,
    stochrsi_k_len: int = 3,
    stochrsi_d_len: int = 3,
    oversold: float = 20.0,
    # Fib 파라미터
    fib_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    fib_min_half_atr: float = 0.25,
    fib_max_half_mult: float = 0.8,
    # Confluence 파라미터
    min_score: float = 0.3,
) -> ConfluenceResult:
    """
    전체 Confluence 분석 파이프라인

    Args:
        df: OHLCV DataFrame (index가 timestamp)
        zig_*: ZigZag 파라미터
        stochrsi_*: StochRSI 파라미터
        oversold: Oversold 임계치
        fib_*: 피보나치 파라미터
        min_score: 최소 Confluence 점수

    Returns:
        ConfluenceResult
    """
    result = ConfluenceResult(success=False)

    try:
        # 필수 컬럼 확인
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            result.error = f"Missing columns: {missing}"
            return result

        # 데이터 복사
        df = df.copy()

        # 1. RSI 계산
        if 'rsi' not in df.columns or df['rsi'].isna().all():
            df['rsi'] = calc_rsi_wilder(df['close'].values)

        # 2. StochRSI 계산
        if 'stoch_d' not in df.columns or df['stoch_d'].isna().all():
            k, d = tv_stoch_rsi(
                df['close'].values,
                rsi_len=stochrsi_period,
                stoch_len=stochrsi_period,
                k_len=stochrsi_k_len,
                d_len=stochrsi_d_len,
            )
            df['stoch_k'] = k
            df['stoch_d'] = d

        # 3. ZigZag 피벗 감지
        pivots, atr = zigzag_pivots(
            df['close'].values,
            df['high'].values,
            df['low'].values,
            up_pct=zig_up_pct,
            down_pct=zig_down_pct,
            atr_period=zig_atr_period,
            atr_mult=zig_atr_mult,
            threshold_mode=zig_threshold_mode,
            min_bars=zig_min_bars,
            min_swing_atr=zig_min_swing_atr,
        )
        df['atr'] = atr
        add_pivot_columns(df, pivots)

        # 4. 스윙 정보 추출
        swing = get_latest_swing(df, pivots)
        if swing is None:
            result.error = "No valid swing found"
            return result
        result.swing = swing

        # 5. Fib 레벨 계산
        fib_levels = calc_fib_levels(swing)
        atr_now = float(df['atr'].dropna().iloc[-1])
        zones = build_fib_zones(fib_levels, atr_now, fib_ratios=fib_ratios,
                                min_half_atr=fib_min_half_atr, max_half_mult=fib_max_half_mult)
        result.fib_levels = fib_levels
        result.zones = zones

        # 6. Oversold 세그먼트에서 REF 추출
        segment, thr, reason = pick_oversold_segment_with_rule(
            df, d_col='stoch_d', oversold=oversold
        )
        result.ref_reason = reason

        if segment is None:
            result.error = f"No oversold segment: {reason}"
            return result

        ref = extract_ref_from_segment(df, segment)
        result.ref = ref

        # 7. 현재 Divergence 판정
        div_check = check_divergence_at_current(df, ref.price, ref.rsi)
        result.is_reg_now = div_check['is_reg_up']
        result.is_hid_now = div_check['is_hid_up']

        # 8. Divergence 경계 계산
        need_reg = needed_close_for_regular(df, ref.price, ref.rsi)
        hid_range = feasible_range_for_hidden(df, ref.price, ref.rsi)
        result.need_reg = need_reg
        result.hid_range = hid_range

        # 9. Confluence 점수 계산
        zone_scores = calc_zone_confluence_scores(
            df, fib_levels, zones,
            need_reg=need_reg,
            hid_range=hid_range,
        )
        result.zone_scores = zone_scores

        # 10. 현재가가 존 안에 있는지 확인
        current_price = float(df['close'].iloc[-1])
        is_in, current_zone = is_price_in_confluence_zone(current_price, zone_scores, min_score)
        result.is_in_zone = is_in
        result.current_zone = current_zone

        # 11. 최적 진입 존
        best = find_best_entry_zone(zone_scores, current_price, min_score=min_score, atr=atr_now)
        result.best_zone = best

        result.success = True
        return result

    except Exception as e:
        result.error = str(e)
        return result


def should_enter_long(result: ConfluenceResult, min_score: float = 0.3) -> bool:
    """
    Long 진입 여부 판단

    조건:
    1. Confluence 분석 성공
    2. 현재가가 Confluence Zone 안에 있음
    3. Zone 점수가 min_score 이상
    4. (선택) Regular 또는 Hidden Divergence 성립

    Args:
        result: analyze_confluence() 결과
        min_score: 최소 점수

    Returns:
        Long 진입 여부
    """
    if not result.success:
        return False

    if not result.is_in_zone:
        return False

    if result.current_zone is None:
        return False

    if result.current_zone.score < min_score:
        return False

    # Divergence 조건 (선택적으로 강화 가능)
    # if not (result.is_reg_now or result.is_hid_now):
    #     return False

    return True
