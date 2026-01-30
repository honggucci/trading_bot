"""
MODE80 Helper Functions
=======================

Committee Final Spec 기반 헬퍼 함수들:
- check_reclaim_confirmed: DOWN 모드 reclaim 확인
- rr_feasible_gate: RR 2 Gate 검증
- AttemptTracker: 재진입 루프 방지

Author: Claude Code
Date: 2026-01-24
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# ============================================================
# 1. Reclaim Confirmation (DOWN 모드 필수)
# ============================================================

def get_1h_swing_high_at_time(
    df_1h: pd.DataFrame,
    ref_time: pd.Timestamp,
    lookback_bars: int = 10,
) -> Optional[float]:
    """
    특정 시점의 1H swing high 찾기

    Args:
        df_1h: 1H DataFrame (index=datetime)
        ref_time: 기준 시점
        lookback_bars: 룩백 기간 (1H 바 개수)

    Returns:
        swing high 가격 또는 None
    """
    if df_1h is None or len(df_1h) == 0:
        return None

    # ref_time 이전 데이터만
    mask = df_1h.index <= ref_time
    df_slice = df_1h[mask]

    if len(df_slice) < 3:
        return None

    # 최근 lookback_bars 내 고점 찾기
    df_recent = df_slice.tail(lookback_bars)

    # 간단한 swing high: 이전/이후보다 높은 고점
    highs = df_recent['high'].values
    for i in range(len(highs) - 2, 0, -1):  # 역순으로 찾기
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            return float(highs[i])

    # swing high 못 찾으면 최근 고점 사용
    return float(df_recent['high'].max())


def check_reclaim_confirmed(
    df_1h: pd.DataFrame,
    current_time: pd.Timestamp,
    key_level: float,
    close_count: int = 1,
) -> Dict[str, any]:
    """
    1H close가 key_level 위로 N회 마감했는지 확인

    Args:
        df_1h: 1H DataFrame
        current_time: 현재 시점
        key_level: 리클레임 기준 레벨
        close_count: 필요한 연속 마감 횟수 (default: 1)

    Returns:
        {
            'confirmed': bool,
            'consecutive_closes': int,
            'last_close': float,
            'key_level': float,
        }
    """
    result = {
        'confirmed': False,
        'consecutive_closes': 0,
        'last_close': None,
        'key_level': key_level,
    }

    if df_1h is None or len(df_1h) == 0:
        return result

    # current_time 이전의 확정된 1H 바만 사용 (룩어헤드 방지)
    mask = df_1h.index < current_time
    df_slice = df_1h[mask]

    if len(df_slice) == 0:
        return result

    # 최근 N개 바 체크
    closes = df_slice['close'].tail(close_count + 2).values
    result['last_close'] = float(closes[-1]) if len(closes) > 0 else None

    # 연속 마감 횟수 카운트 (최신부터)
    consecutive = 0
    for close in reversed(closes):
        if close > key_level:
            consecutive += 1
        else:
            break

    result['consecutive_closes'] = consecutive
    result['confirmed'] = consecutive >= close_count

    return result


# ============================================================
# 2. RR Feasible Gate
# ============================================================

def find_nearest_resistance(
    current_price: float,
    fib_levels: List[Tuple[float, float]],  # [(ratio, price), ...]
    df_1h: pd.DataFrame = None,
    current_time: pd.Timestamp = None,
    lookback_bars: int = 20,
) -> Optional[float]:
    """
    현재 가격 위의 가장 가까운 저항 레벨 찾기

    우선순위:
    1. 1H swing high (최근 20봉)
    2. 위쪽 Fib 레벨

    Args:
        current_price: 현재 가격
        fib_levels: Fib 레벨 리스트 [(ratio, price), ...]
        df_1h: 1H DataFrame (optional)
        current_time: 현재 시점 (optional)
        lookback_bars: 1H 룩백 기간

    Returns:
        저항 레벨 가격 또는 None
    """
    resistances = []

    # 1. Fib 레벨에서 현재가 위의 레벨들
    for ratio, price in sorted(fib_levels, key=lambda x: x[1]):
        if price > current_price:
            resistances.append(price)

    # 2. 1H swing high 추가
    if df_1h is not None and current_time is not None:
        swing_high = get_1h_swing_high_at_time(df_1h, current_time, lookback_bars)
        if swing_high is not None and swing_high > current_price:
            resistances.append(swing_high)

    # 가장 가까운 저항
    if resistances:
        return min(resistances)
    return None


def rr_feasible_gate(
    entry_price: float,
    sl_price: float,
    tp_candidate: float,
    rr_min: float = 1.8,
    rr_max: float = 2.2,
    soft_range: bool = True,
) -> Dict[str, any]:
    """
    RR Feasible Gate: 진입 전 RR >= rr_min 검증

    Args:
        entry_price: 진입 가격
        sl_price: SL 가격
        tp_candidate: TP 후보 (가장 가까운 저항)
        rr_min: 최소 RR (default: 1.8)
        rr_max: 최대 RR (default: 2.2, soft range용)
        soft_range: True면 rr_min-rr_max 범위 허용

    Returns:
        {
            'pass': bool,
            'r': float (리스크 단위),
            'tp_2r': float (2R 목표가),
            'tp_candidate': float,
            'actual_rr': float,
            'reason': str,
        }
    """
    # R = entry - SL (롱 기준)
    r = entry_price - sl_price

    if r <= 0:
        return {
            'pass': False,
            'r': 0,
            'tp_2r': entry_price,
            'tp_candidate': tp_candidate,
            'actual_rr': 0,
            'reason': 'invalid_sl',
        }

    # 2R 목표가
    tp_2r = entry_price + 2 * r

    # 실제 RR 계산
    if tp_candidate and tp_candidate > entry_price:
        actual_rr = (tp_candidate - entry_price) / r
    else:
        actual_rr = 0

    # Gate 판정
    if soft_range:
        # soft range: rr_min 이상이면 PASS
        is_pass = actual_rr >= rr_min
        reason = 'pass' if is_pass else f'rr_{actual_rr:.2f}_below_{rr_min}'
    else:
        # hard: 정확히 2.0 이상
        is_pass = actual_rr >= 2.0
        reason = 'pass' if is_pass else f'rr_{actual_rr:.2f}_below_2.0'

    return {
        'pass': is_pass,
        'r': r,
        'tp_2r': tp_2r,
        'tp_candidate': tp_candidate,
        'actual_rr': actual_rr,
        'reason': reason,
    }


# ============================================================
# 3. Attempt Tracker (재진입 루프 방지)
# ============================================================

@dataclass
class AttemptTracker:
    """
    재진입 루프 방지용 Attempt Tracker

    기능:
    - Per-level attempt cap (레벨당 진입 횟수 제한)
    - Cooldown (손절 후 재진입 금지 기간)
    """
    cap_per_level: int = 1  # 레벨당 최대 진입 횟수
    cooldown_bars: int = 16  # 손절 후 쿨다운 (15m 바 기준)

    # 내부 상태
    _attempts: Dict[str, int] = field(default_factory=dict)  # level_key -> count
    _cooldown_until_bar: int = -1  # 쿨다운 종료 바 인덱스
    _last_sl_bar: int = -1  # 마지막 손절 바 인덱스

    def get_level_key(self, fib_ratio: float, tolerance: float = 0.05) -> str:
        """Fib 레벨을 키로 변환 (tolerance 내 동일 레벨 처리)"""
        # 0.05 단위로 반올림
        rounded = round(fib_ratio / tolerance) * tolerance
        return f"fib_{rounded:.3f}"

    def can_enter(self, fib_ratio: float, current_bar_idx: int) -> Tuple[bool, str]:
        """
        진입 가능 여부 확인

        Returns:
            (can_enter, reason)
        """
        # 1. 쿨다운 체크
        if current_bar_idx < self._cooldown_until_bar:
            remaining = self._cooldown_until_bar - current_bar_idx
            return False, f'cooldown_active_{remaining}_bars'

        # 2. Attempt cap 체크
        level_key = self.get_level_key(fib_ratio)
        current_attempts = self._attempts.get(level_key, 0)

        if current_attempts >= self.cap_per_level:
            return False, f'attempt_cap_reached_{level_key}'

        return True, 'ok'

    def record_entry(self, fib_ratio: float) -> None:
        """진입 기록"""
        level_key = self.get_level_key(fib_ratio)
        self._attempts[level_key] = self._attempts.get(level_key, 0) + 1

    def record_sl_exit(self, current_bar_idx: int) -> None:
        """손절 기록 (쿨다운 시작)"""
        self._last_sl_bar = current_bar_idx
        self._cooldown_until_bar = current_bar_idx + self.cooldown_bars

    def reset_level(self, fib_ratio: float) -> None:
        """특정 레벨 attempt 리셋"""
        level_key = self.get_level_key(fib_ratio)
        self._attempts[level_key] = 0

    def reset_all(self) -> None:
        """전체 상태 리셋"""
        self._attempts.clear()
        self._cooldown_until_bar = -1
        self._last_sl_bar = -1

    def get_stats(self) -> Dict[str, any]:
        """현재 상태 반환"""
        return {
            'attempts': dict(self._attempts),
            'cooldown_until_bar': self._cooldown_until_bar,
            'last_sl_bar': self._last_sl_bar,
        }


# ============================================================
# 4. UP/DOWN Mode Utilities
# ============================================================

def get_swing_direction(dynfib_state) -> str:
    """
    ZigZag direction에서 UP/DOWN 모드 결정

    Args:
        dynfib_state: DynamicFibState (direction 속성 필요)

    Returns:
        'up' (Pullback) 또는 'down' (Bounce)
        direction 없으면 'up' (기본값, 보수적)
    """
    if dynfib_state is None:
        return 'up'  # fallback

    direction = getattr(dynfib_state, 'direction', None)
    if direction is None:
        return 'up'  # fallback

    # 'up' = last confirmed pivot is LOW (상승 중 조정)
    # 'down' = last confirmed pivot is HIGH (하락 중 반등)
    return direction if direction in ('up', 'down') else 'up'


def is_in_l0_zone(
    current_price: float,
    fib_low: float,
    fib_high: float,
    swing_dir: str,
    zone_up_min: float = 0.382,
    zone_up_max: float = 0.618,
    zone_down_min: float = 0.618,
    zone_down_max: float = 0.786,
) -> Tuple[bool, float, str]:
    """
    L0 (1W Fib) 허용 구역 체크

    Args:
        current_price: 현재 가격
        fib_low: Fib 앵커 저점
        fib_high: Fib 앵커 고점
        swing_dir: 'up' 또는 'down'
        zone_up_min/max: UP 모드 허용 구역
        zone_down_min/max: DOWN 모드 허용 구역

    Returns:
        (in_zone, current_fib_ratio, reason)
    """
    fib_range = fib_high - fib_low
    if fib_range <= 0:
        return False, 0.0, 'invalid_fib_range'

    # 현재 가격의 Fib 비율
    current_ratio = (current_price - fib_low) / fib_range

    # 모드별 허용 구역
    if swing_dir == 'up':
        in_zone = zone_up_min <= current_ratio <= zone_up_max
        reason = f'up_zone_{zone_up_min}-{zone_up_max}'
    else:  # down
        in_zone = zone_down_min <= current_ratio <= zone_down_max
        reason = f'down_zone_{zone_down_min}-{zone_down_max}'

    return in_zone, current_ratio, reason


def get_fib_levels(
    fib_low: float,
    fib_high: float,
    ratios: List[float],
) -> List[Tuple[float, float]]:
    """
    Fib 레벨 계산

    Args:
        fib_low: 앵커 저점
        fib_high: 앵커 고점
        ratios: Fib 비율 리스트

    Returns:
        [(ratio, price), ...] 정렬된 리스트
    """
    fib_range = fib_high - fib_low
    if fib_range <= 0:
        return []

    levels = []
    for ratio in ratios:
        price = fib_low + fib_range * ratio
        levels.append((ratio, price))

    return sorted(levels, key=lambda x: x[1])


def get_lower_fib_level(
    entry_price: float,
    fib_levels: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """
    Entry 바로 아래 Fib 레벨 찾기

    Returns:
        (ratio, price) 또는 None
    """
    lower_fib = None
    for ratio, price in reversed(fib_levels):
        if price < entry_price:
            lower_fib = (ratio, price)
            break
    return lower_fib


# ============================================================
# 5. 2R Partial Exit 계산
# ============================================================

def calc_partial_exit_qty(
    total_qty: float,
    swing_dir: str,
    partial_pct_up: float = 0.50,
    partial_pct_down: float = 0.70,
) -> float:
    """
    2R 도달 시 부분청산 수량 계산

    Args:
        total_qty: 전체 포지션 수량
        swing_dir: 'up' 또는 'down'
        partial_pct_up: UP 모드 청산 비율 (default: 50%)
        partial_pct_down: DOWN 모드 청산 비율 (default: 70%)

    Returns:
        청산할 수량
    """
    pct = partial_pct_down if swing_dir == 'down' else partial_pct_up
    return total_qty * pct


# ============================================================
# 6. 로그 필드 생성
# ============================================================

def create_trade_log(
    mode: str,
    l0_fib_ratio: float,
    in_l0_zone: bool,
    div_type: str,
    reclaim_required: bool,
    reclaim_pass: bool,
    entry: float,
    sl: float,
    r: float,
    tp_2r: float,
    tp_candidate: float,
    rr_feasible_pass: bool,
    attempt_count_level: int,
    attempt_count_swing: int,
    cooldown_active: bool,
) -> Dict[str, any]:
    """
    필수 로그 필드 생성 (QA/검증용)
    """
    return {
        'mode': mode,
        'l0_fib_ratio': l0_fib_ratio,
        'in_l0_zone': in_l0_zone,
        'div_type': div_type,
        'reclaim_required': reclaim_required,
        'reclaim_pass': reclaim_pass,
        'entry': entry,
        'sl': sl,
        'r': r,
        'tp_2r': tp_2r,
        'tp_candidate': tp_candidate,
        'rr_feasible_pass': rr_feasible_pass,
        'attempt_count_level': attempt_count_level,
        'attempt_count_swing': attempt_count_swing,
        'cooldown_active': cooldown_active,
    }
