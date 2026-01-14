"""
Fib + Divergence Confluence Scoring
===================================

피보나치 존과 다이버전스 경계의 겹침을 점수화.

핵심 개념:
- Fib Zone: 피보나치 되돌림 레벨 주변의 지지/저항 구간
- Regular Divergence: price < ref_price AND RSI > ref_rsi → 반등 신호
- Hidden Divergence: price > ref_price AND RSI < ref_rsi → 추세 지속
- Confluence: Fib Zone과 Divergence 영역이 겹치면 강한 신호

8개 컴포넌트 점수:
1. s_zone: 존 품질 (폭 vs ATR)
2. s_reg: Regular divergence 겹침
3. s_hid: Hidden divergence 겹침
4. s_gp: Golden Pocket (0.618~0.65) 보너스
5. s_dir: 스윙 방향 정합
6. s_gamma: 옵션 그릭스 (감마) - 선택적
7. s_delta: 옵션 그릭스 (델타) - 선택적
8. s_vwap: VWAP 근접도 - 선택적

Origin: param_search_confluence_v0.py의 fib_divergence_confluence()
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..context.fibonacci import FibLevel, FibLevels


@dataclass
class ZoneScore:
    """개별 존의 Confluence 점수"""
    zone: FibLevel
    score: float
    components: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


def _overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> Tuple[float, float, float]:
    """두 구간의 겹침 계산"""
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    ov = max(0.0, hi - lo)
    return lo, hi, ov


def calc_zone_confluence_scores(
    df: pd.DataFrame,
    fib_levels: FibLevels,
    zones: List[FibLevel],
    *,
    need_reg: Optional[float] = None,  # Regular divergence 경계 가격
    hid_range: Optional[Tuple[float, float]] = None,  # Hidden divergence 범위
    atr_col: str = 'atr',
    close_col: str = 'close',
    sort_by: str = 'score',
    weights: Optional[Dict[str, float]] = None,
) -> List[ZoneScore]:
    """
    Fib Zone들의 Confluence 점수 계산

    Args:
        df: DataFrame with price data
        fib_levels: FibLevels (피보나치 레벨 정보)
        zones: List[FibLevel] (점수 계산할 존들)
        need_reg: Regular divergence 경계 가격 (이 가격 이하에서 REG 성립)
        hid_range: Hidden divergence 범위 (lo, hi) (이 범위 안에서 HID 성립)
        atr_col: ATR 컬럼명
        close_col: 종가 컬럼명
        sort_by: 정렬 기준 ('score', 'center', 'ratio')
        weights: 컴포넌트 가중치 (없으면 기본값 사용)

    Returns:
        List[ZoneScore] (점수순 정렬)
    """
    # 기본 가중치
    if weights is None:
        weights = {
            'w_zone': 0.30,
            'w_reg': 0.25,
            'w_hid': 0.15,
            'w_gp': 0.10,
            'w_dir': 0.10,
            'w_gamma': 0.05,
            'w_delta': 0.05,
            'w_vwap': 0.00,  # VWAP 비활성화
        }

    # ATR 가져오기
    if atr_col in df.columns:
        atr_now = float(df[atr_col].dropna().iloc[-1])
    else:
        atr_now = float(df[close_col].std())  # 대체

    # 현재가
    price_now = float(df[close_col].iloc[-1])

    # 골든 포켓
    gp = fib_levels.golden_pocket

    # 스윙 방향
    swing_dir = fib_levels.swing.direction

    results = []

    for zone in zones:
        low = zone.low
        high = zone.high
        center = zone.price
        width = max(1e-12, high - low)

        # 1. 존 품질 (폭 vs ATR)
        w_norm = np.clip(width / atr_now, 0.25, 4.0)
        s_zone = 1.0 - abs(np.log(w_norm)) * 0.25

        # 2. Regular Divergence 겹침
        s_reg = 0.0
        if need_reg is not None:
            # need_reg 이하의 가격에서 REG 성립
            _, _, ov = _overlap(low, high, -1e18, need_reg)
            if ov > 0:
                s_reg = min(1.0, ov / width)

        # 3. Hidden Divergence 겹침
        s_hid = 0.0
        if hid_range is not None:
            hid_lo, hid_hi = hid_range
            _, _, ov = _overlap(low, high, hid_lo, hid_hi)
            if ov > 0:
                s_hid = min(1.0, ov / width)

        # 4. 골든 포켓 보너스
        s_gp = 0.0
        if gp is not None:
            gp_lo, gp_hi = gp
            if gp_lo <= center <= gp_hi:
                s_gp = 0.15
            else:
                dist = min(abs(center - gp_lo), abs(center - gp_hi))
                s_gp = max(0.0, 0.10 - 0.10 * (dist / (2.0 * atr_now)))

        # 5. 스윙 방향 정합
        s_dir = 0.0
        if swing_dir == 'up':
            s_dir = 0.05 if center <= price_now else 0.0
        elif swing_dir == 'down':
            s_dir = 0.05 if center >= price_now else 0.0

        # 6, 7. 그릭스 (간소화 - 거리 기반)
        dist_atr = abs(center - price_now) / max(atr_now, 1e-12)
        s_gamma = max(0.0, 1.0 - dist_atr * 0.2)  # 가까울수록 높음
        s_delta = 0.5  # 중립

        # 8. VWAP (미구현 - 0)
        s_vwap = 0.0

        # 가중 합계
        components = np.array([s_zone, s_reg, s_hid, s_gp, s_dir, s_gamma, s_delta, s_vwap])
        w_vec = np.array([
            weights['w_zone'],
            weights['w_reg'],
            weights['w_hid'],
            weights['w_gp'],
            weights['w_dir'],
            weights['w_gamma'],
            weights['w_delta'],
            weights['w_vwap'],
        ])

        score = float(np.dot(w_vec, components))

        results.append(ZoneScore(
            zone=zone,
            score=score,
            components={
                'zone': s_zone,
                'reg': s_reg,
                'hid': s_hid,
                'gp': s_gp,
                'dir': s_dir,
                'gamma': s_gamma,
                'delta': s_delta,
                'vwap': s_vwap,
            },
            details={
                'width_atr': width / atr_now,
                'dist_atr': dist_atr,
            },
        ))

    # 정렬
    if sort_by == 'score':
        results.sort(key=lambda x: x.score, reverse=True)
    elif sort_by == 'center':
        results.sort(key=lambda x: x.zone.price)
    elif sort_by == 'ratio':
        results.sort(key=lambda x: x.zone.ratio)

    return results


def find_best_entry_zone(
    zone_scores: List[ZoneScore],
    current_price: float,
    *,
    min_score: float = 0.3,
    max_distance_atr: float = 2.0,
    atr: float = 1.0,
) -> Optional[ZoneScore]:
    """
    최적 진입 존 찾기

    Args:
        zone_scores: calc_zone_confluence_scores() 결과
        current_price: 현재가
        min_score: 최소 점수
        max_distance_atr: 최대 거리 (ATR 배수)
        atr: ATR 값

    Returns:
        최적 ZoneScore 또는 None
    """
    for zs in zone_scores:
        if zs.score < min_score:
            continue

        dist = abs(zs.zone.price - current_price)
        if dist > max_distance_atr * atr:
            continue

        # 현재가가 존 안에 있거나 가까우면 반환
        if zs.zone.low <= current_price <= zs.zone.high:
            return zs

    return None


def is_price_in_confluence_zone(
    current_price: float,
    zone_scores: List[ZoneScore],
    min_score: float = 0.3,
) -> Tuple[bool, Optional[ZoneScore]]:
    """
    현재가가 Confluence Zone 안에 있는지 확인

    Args:
        current_price: 현재가
        zone_scores: calc_zone_confluence_scores() 결과
        min_score: 최소 점수

    Returns:
        (안에 있는지, ZoneScore 또는 None)
    """
    for zs in zone_scores:
        if zs.score < min_score:
            continue

        if zs.zone.low <= current_price <= zs.zone.high:
            return True, zs

    return False, None
