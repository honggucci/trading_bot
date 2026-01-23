"""
PR-DYN-FIB v2: 1W Dynamic Fibonacci Anchor (Log Space)

1W ZigZag pivot 기반 동적 앵커로 Log Fib 레벨 생성.
Macro Log Fib($3K~$143K)의 갭을 채우는 용도.

NOTE: 15m dynamic Fib는 range가 너무 좁아(~$1K) 효과 없음 → 1W만 사용.

Anchor 갱신 모드:
- zigzag: ZigZag pivot 확정 시 갱신 (1W 권장)
- rolling: N봉 rolling high/low (단순, 휩쏘 취약)
- conditional: range >= N*ATR 일 때만 갱신 (노이즈 억제)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

# Lazy import to avoid circular dependency
_wave_classifier = None

def _get_wave_classifier(hilbert_window: int = 64):
    """Lazy import of WaveRegimeClassifier to avoid circular dependency."""
    global _wave_classifier
    if _wave_classifier is None or _wave_classifier.hilbert_window != hilbert_window:
        from ..regime.wave_regime import WaveRegimeClassifier
        _wave_classifier = WaveRegimeClassifier(hilbert_window=hilbert_window)
    return _wave_classifier


@dataclass
class DynamicFibAnchorState:
    """동적 Fib 앵커 상태 (v3: extreme_ts 분리 저장)"""
    # === 확정된 pivot 쌍 (Fib 계산에 사용) ===
    low: float = 0.0                        # 확정된 저점 (= confirmed_low)
    high: float = 0.0                       # 확정된 고점 (= confirmed_high)

    # === 현재 추적 중인 상태 ===
    running_extreme: float = 0.0            # 현재 추적 중인 극점 (미확정)
    direction: str = "unknown"              # "up" | "down" | "unknown"

    # === 메타데이터 ===
    last_update_ts: Optional[pd.Timestamp] = None  # 마지막 갱신 시각
    last_pivot_ts: Optional[pd.Timestamp] = None   # 마지막 pivot 확정 시각 (기존 호환)
    mode: str = "rolling"                   # "zigzag" | "rolling" | "conditional"
    pivot_count: int = 0                    # 확정된 pivot 수

    # === P0-5: k-anchor for ZigZag reversal threshold ===
    k_anchor: float = 1.5                   # 현재 swing에 적용되는 k (pivot 확정 시 갱신)
    atr_anchor: float = 0.0                 # pivot 확정 시점의 ATR (다음 pivot까지 고정)

    # === P0-5 FIX: extreme vs confirm timestamp 분리 ===
    # candidate: running_extreme이 갱신될 때의 timestamp (아직 미확정)
    candidate_high_ts: Optional[pd.Timestamp] = None  # 고점 후보 발생 시점
    candidate_low_ts: Optional[pd.Timestamp] = None   # 저점 후보 발생 시점

    # last confirmed pivot의 상세 정보
    last_extreme_ts: Optional[pd.Timestamp] = None   # 극값 발생 시점 (가격이 찍힌 시점)
    last_confirm_ts: Optional[pd.Timestamp] = None   # 확정 시점 (reversal 감지 시점)
    last_pivot_type: str = ""                        # "HIGH" or "LOW"

    # === P0-5b: Pending Reversal 상태 (min_spacing 부족 시) ===
    pending: bool = False                               # pending 상태 여부
    pending_type: str = ""                              # "HIGH" or "LOW" (pending pivot 타입)
    pending_pivot_price: float = 0.0                    # pending pivot 가격
    pending_extreme_ts: Optional[pd.Timestamp] = None   # pending pivot extreme 발생 시점
    pending_start_ts: Optional[pd.Timestamp] = None     # pending 상태 진입 시점
    pending_opposite_price: float = 0.0                 # pending 중 반대 방향 극점
    pending_opposite_ts: Optional[pd.Timestamp] = None  # 반대 방향 극점 시점

    def is_valid(self) -> bool:
        """앵커가 유효한지 확인 (최소 2개 pivot 확정)"""
        return self.low > 0 and self.high > 0 and self.high > self.low and self.pivot_count >= 2

    def get_range(self) -> float:
        """현재 앵커 range 반환"""
        if self.is_valid():
            return self.high - self.low
        return 0.0


def compute_dynamic_k(
    prices: np.ndarray,
    atr: float,
    mode: str = "fixed",
    k_fixed: float = 1.5,
    k_min: float = 1.8,
    k_max: float = 4.5,
    k_base: float = 3.0,
    strength_ref: float = 0.8,
    ampz_thr: float = 0.5,
    k_cycle: float = 2.0,
    k_trend: float = 3.8
) -> float:
    """
    Hilbert amplitude 기반 동적 k 계산 (P0-5).

    부호 관계 (필수):
    - 사이클 강함 (amplitude 큼) → k 작아야 (민감)
    - 사이클 약함/추세/노이즈 → k 커야 (둔감)

    Args:
        prices: 가격 배열 (pivot 시점까지만!)
        atr: 현재 ATR
        mode: "fixed" | "regime_switch" | "hilbert" | "inverse"
        k_fixed: fixed 모드 k값 (기본 1.5)
        k_min: k 최소값 (사이클 강할 때)
        k_max: k 최대값 (사이클 약할 때)
        k_base: inverse 모드 기준 k
        strength_ref: cycle_strength 기준값
        ampz_thr: regime_switch 모드 amplitude_z 임계값
        k_cycle: regime_switch 모드 사이클 장 k
        k_trend: regime_switch 모드 추세/노이즈 장 k

    Returns:
        계산된 k 값
    """
    if mode == "fixed":
        return k_fixed

    if len(prices) < 20:
        return k_fixed  # 데이터 부족 시 기본값

    try:
        # Hilbert amplitude 추출 (causal - pivot 시점까지만)
        classifier = _get_wave_classifier(hilbert_window=min(64, len(prices)))
        state = classifier.classify(prices)

        cycle_strength = state.amplitude / atr if atr > 0 else 1.0

        if mode == "regime_switch":
            # 사이클 강하면 k 작게, 약하면 k 크게
            return k_cycle if state.amplitude_z >= ampz_thr else k_trend

        elif mode == "hilbert":
            # 연속형: 강할수록 k 작아짐 (단조 감소)
            # s ∈ [0, 1] 정규화
            s = np.clip(cycle_strength / strength_ref, 0.0, 1.0)
            # k = k_max - s * (k_max - k_min)
            # s=0 (약함) → k=k_max, s=1 (강함) → k=k_min
            return k_max - s * (k_max - k_min)

        elif mode == "inverse":
            # 역비례형: k = k_base * (ref / strength)
            eps = 1e-6
            return np.clip(k_base * (strength_ref / (cycle_strength + eps)), k_min, k_max)

    except Exception as e:
        # Hilbert 계산 실패 시 기본값
        print(f"[compute_dynamic_k] Hilbert failed: {e}, using k_fixed={k_fixed}")
        return k_fixed

    return k_fixed


def update_anchor_rolling(
    df: pd.DataFrame,
    i: int,
    lookback_bars: int = 96
) -> Tuple[float, float]:
    """
    Rolling N봉 high/low 방식으로 앵커 갱신.

    Args:
        df: OHLCV DataFrame (15m)
        i: 현재 인덱스
        lookback_bars: lookback 봉 수 (기본 96 = 24h)

    Returns:
        (low, high) 튜플
    """
    start_idx = max(0, i - lookback_bars + 1)
    end_idx = i + 1

    window = df.iloc[start_idx:end_idx]

    low = window['low'].min()
    high = window['high'].max()

    return low, high


def update_anchor_conditional(
    df: pd.DataFrame,
    i: int,
    lookback_bars: int,
    atr: float,
    min_swing_mult: float = 1.5,
    prev_state: Optional[DynamicFibAnchorState] = None
) -> Tuple[float, float, bool]:
    """
    Conditional 방식: range >= min_swing_mult * ATR 일 때만 앵커 갱신.

    Args:
        df: OHLCV DataFrame (15m)
        i: 현재 인덱스
        lookback_bars: lookback 봉 수
        atr: 현재 ATR
        min_swing_mult: 최소 스윙 배수 (기본 1.5)
        prev_state: 이전 상태 (갱신 안 할 경우 유지)

    Returns:
        (low, high, updated) 튜플 - updated는 갱신 여부
    """
    # Rolling 방식으로 후보 계산
    low, high = update_anchor_rolling(df, i, lookback_bars)
    swing_range = high - low

    min_swing = min_swing_mult * atr

    # 조건 충족 시 갱신
    if swing_range >= min_swing:
        return low, high, True

    # 조건 미충족: 이전 상태 유지
    if prev_state and prev_state.is_valid():
        return prev_state.low, prev_state.high, False

    # 이전 상태가 없으면 현재 값 사용
    return low, high, False


def update_anchor_zigzag(
    df: pd.DataFrame,
    i: int,
    state: DynamicFibAnchorState,
    atr: float,
    reversal_mult: float = 1.5,
    min_bars_between_pivots: int = 3,
    # === P0-5: Dynamic k 파라미터 ===
    k_mode: str = "fixed",
    k_fixed: float = 1.5,
    k_min: float = 1.8,
    k_max: float = 4.5,
    k_base: float = 3.0,
    strength_ref: float = 0.8,
    ampz_thr: float = 0.5,
    k_cycle: float = 2.0,
    k_trend: float = 3.8,
    min_spacing_weeks: float = 0.0,  # 0이면 비활성
) -> DynamicFibAnchorState:
    """
    ZigZag pivot 확정 방식으로 앵커 갱신 (v3: k-anchor + min_spacing).

    핵심 로직:
    - 상승 중: 고점 추적 (running_extreme)
    - reversal 감지 시: 고점 확정 (high = running_extreme), 방향 전환
    - 하락 중: 저점 추적 (running_extreme)
    - reversal 감지 시: 저점 확정 (low = running_extreme), 방향 전환

    v2 수정: 확정된 low/high는 리셋하지 않고 유지.
    v3 수정 (P0-5): k-anchor 고정 + min_spacing 제약

    k-anchor 규칙:
    - pivot 확정 시점에 k 계산하여 k_anchor로 저장
    - 다음 pivot까지 k_anchor 유지 (매 bar 변경 금지)

    min_spacing 규칙:
    - pivot 간 최소 간격 미만이면 pivot 확정 거부
    - 단, running_extreme은 계속 추적

    Args:
        df: OHLCV DataFrame (1W 권장)
        i: 현재 인덱스
        state: 현재 상태
        atr: 현재 ATR
        reversal_mult: reversal 조건 ATR 배수 (기본 1.5, k_mode=fixed일 때만 사용)
        min_bars_between_pivots: pivot 간 최소 봉 수 (기본 3)
        k_mode: dynamic k 모드 ("fixed" | "regime_switch" | "hilbert" | "inverse")
        k_fixed ~ k_trend: compute_dynamic_k() 파라미터
        min_spacing_weeks: pivot 간 최소 간격 (주 단위, 0이면 비활성)

    Returns:
        갱신된 DynamicFibAnchorState
    """
    if i < 1:
        return state

    bar = df.iloc[i]
    current_high = bar['high']
    current_low = bar['low']
    current_ts = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else bar.get('timestamp', pd.Timestamp.now())

    # P0-5: k_anchor 기반 reversal threshold (매 bar 변경 금지)
    # k_mode="fixed"이면 reversal_mult 사용, 아니면 state.k_anchor 사용
    if k_mode == "fixed":
        effective_k = reversal_mult
    else:
        effective_k = state.k_anchor if state.k_anchor > 0 else reversal_mult

    # ATR anchor: pivot 확정 시점의 ATR을 사용 (매 bar 변동 방지)
    effective_atr = state.atr_anchor if state.atr_anchor > 0 else atr
    reversal_threshold = effective_k * effective_atr

    # === 초기화: 첫 상태 설정 ===
    if state.direction == "unknown" or state.pivot_count == 0:
        # 초기 방향 결정: 최근 N봉 추세
        lookback = min(20, i + 1)
        start_idx = max(0, i - lookback + 1)
        window = df.iloc[start_idx:i+1]

        window_low = window['low'].min()
        window_high = window['high'].max()
        first_close = window.iloc[0]['close']
        last_close = window.iloc[-1]['close']

        # FIX: 초기화 시 low/high 모두 설정하여 즉시 valid 상태로 만듦
        # 이후 reversal이 발생하면 더 정확한 pivot으로 갱신됨
        state.low = window_low
        state.high = window_high
        state.pivot_count = 2  # 초기화로 2개 pivot 확정 (window의 high/low)

        if last_close > first_close:
            # 상승 추세 → 고점 추적 시작
            state.direction = "up"
            state.running_extreme = current_high
            state.candidate_high_ts = current_ts  # P0-5 FIX: 초기 candidate 설정
        else:
            # 하락 추세 → 저점 추적 시작
            state.direction = "down"
            state.running_extreme = current_low
            state.candidate_low_ts = current_ts   # P0-5 FIX: 초기 candidate 설정

        # P0-5: 초기 k_anchor 계산
        if k_mode != "fixed":
            prices = df['close'].iloc[:i+1].values
            state.k_anchor = compute_dynamic_k(
                prices, atr, mode=k_mode, k_fixed=k_fixed,
                k_min=k_min, k_max=k_max, k_base=k_base,
                strength_ref=strength_ref, ampz_thr=ampz_thr,
                k_cycle=k_cycle, k_trend=k_trend
            )
        else:
            state.k_anchor = reversal_mult

        # ATR anchor 초기화: 현재 ATR을 다음 pivot까지 고정
        state.atr_anchor = atr

        state.last_pivot_ts = current_ts
        state.last_update_ts = current_ts
        return state

    # === P0-5b: Pending 상태 처리 (매 bar 먼저 체크) ===
    if state.pending:
        # spacing 재계산 (last_pivot_ts 기준)
        spacing_from_last = 0.0
        if state.last_pivot_ts is not None:
            try:
                spacing_from_last = (current_ts - state.last_pivot_ts).days / 7.0
            except:
                pass

        if state.pending_type == "HIGH":
            # === HIGH pending 처리 ===
            # 4a) 취소 조건: 가격이 pending pivot 초과
            if current_high > state.pending_pivot_price:
                # print(f"[ZigZag] PENDING_CANCEL: HIGH, price {current_high:.0f} > pending {state.pending_pivot_price:.0f}")
                state.pending = False
                state.pending_type = ""
                # 새 고점 추적 재시작
                state.running_extreme = current_high
                state.candidate_high_ts = current_ts
                state.last_update_ts = current_ts
            else:
                # 4b) 반대 방향 극점 갱신 (저점 추적)
                if current_low < state.pending_opposite_price:
                    state.pending_opposite_price = current_low
                    state.pending_opposite_ts = current_ts
                    # print(f"[ZigZag] PENDING_UPDATE: HIGH opposite low @ ${current_low:,.0f}")

                # 4c) spacing 충족 AND reversal 유효 → 확정!
                reversal_still_valid = (state.pending_pivot_price - state.pending_opposite_price >= reversal_threshold)

                if spacing_from_last >= min_spacing_weeks and reversal_still_valid:
                    # print(f"[ZigZag] PENDING_CONFIRM: HIGH @ ${state.pending_pivot_price:,.0f} after {spacing_from_last:.1f}w")
                    # 고점 확정
                    state.high = state.pending_pivot_price
                    state.last_extreme_ts = state.pending_extreme_ts
                    state.last_confirm_ts = current_ts
                    state.last_pivot_ts = current_ts
                    state.last_pivot_type = "HIGH"
                    state.pivot_count += 1

                    # 방향 전환 + 저점 추적 시작
                    state.direction = "down"
                    state.running_extreme = state.pending_opposite_price
                    state.candidate_low_ts = state.pending_opposite_ts
                    state.last_update_ts = current_ts

                    # k_anchor 갱신
                    if k_mode != "fixed":
                        prices = df['close'].iloc[:i+1].values
                        state.k_anchor = compute_dynamic_k(
                            prices, atr, mode=k_mode, k_fixed=k_fixed,
                            k_min=k_min, k_max=k_max, k_base=k_base,
                            strength_ref=strength_ref, ampz_thr=ampz_thr,
                            k_cycle=k_cycle, k_trend=k_trend
                        )
                    state.atr_anchor = atr

                    # pending 상태 클리어
                    state.pending = False
                    state.pending_type = ""
                    state.pending_pivot_price = 0.0
                    state.pending_extreme_ts = None
                    state.pending_start_ts = None
                    state.pending_opposite_price = 0.0
                    state.pending_opposite_ts = None

        elif state.pending_type == "LOW":
            # === LOW pending 처리 ===
            # 4a) 취소 조건: 가격이 pending pivot 미만
            if current_low < state.pending_pivot_price:
                # print(f"[ZigZag] PENDING_CANCEL: LOW, price {current_low:.0f} < pending {state.pending_pivot_price:.0f}")
                state.pending = False
                state.pending_type = ""
                # 새 저점 추적 재시작
                state.running_extreme = current_low
                state.candidate_low_ts = current_ts
                state.last_update_ts = current_ts
            else:
                # 4b) 반대 방향 극점 갱신 (고점 추적)
                if current_high > state.pending_opposite_price:
                    state.pending_opposite_price = current_high
                    state.pending_opposite_ts = current_ts
                    # print(f"[ZigZag] PENDING_UPDATE: LOW opposite high @ ${current_high:,.0f}")

                # 4c) spacing 충족 AND reversal 유효 → 확정!
                reversal_still_valid = (state.pending_opposite_price - state.pending_pivot_price >= reversal_threshold)

                if spacing_from_last >= min_spacing_weeks and reversal_still_valid:
                    # print(f"[ZigZag] PENDING_CONFIRM: LOW @ ${state.pending_pivot_price:,.0f} after {spacing_from_last:.1f}w")
                    # 저점 확정
                    state.low = state.pending_pivot_price
                    state.last_extreme_ts = state.pending_extreme_ts
                    state.last_confirm_ts = current_ts
                    state.last_pivot_ts = current_ts
                    state.last_pivot_type = "LOW"
                    state.pivot_count += 1

                    # 방향 전환 + 고점 추적 시작
                    state.direction = "up"
                    state.running_extreme = state.pending_opposite_price
                    state.candidate_high_ts = state.pending_opposite_ts
                    state.last_update_ts = current_ts

                    # k_anchor 갱신
                    if k_mode != "fixed":
                        prices = df['close'].iloc[:i+1].values
                        state.k_anchor = compute_dynamic_k(
                            prices, atr, mode=k_mode, k_fixed=k_fixed,
                            k_min=k_min, k_max=k_max, k_base=k_base,
                            strength_ref=strength_ref, ampz_thr=ampz_thr,
                            k_cycle=k_cycle, k_trend=k_trend
                        )
                    state.atr_anchor = atr

                    # pending 상태 클리어
                    state.pending = False
                    state.pending_type = ""
                    state.pending_pivot_price = 0.0
                    state.pending_extreme_ts = None
                    state.pending_start_ts = None
                    state.pending_opposite_price = 0.0
                    state.pending_opposite_ts = None

    # === 방향별 pivot 확정 로직 (pending 중이 아닐 때만) ===
    if not state.pending:
        if state.direction == "up":
            # 상승 중: 새로운 고점 추적
            if current_high > state.running_extreme:
                state.running_extreme = current_high
                state.candidate_high_ts = current_ts  # P0-5 FIX: extreme 발생 시점 저장
                state.last_update_ts = current_ts

            # Reversal 조건: running_extreme에서 threshold 이상 하락
            if state.running_extreme - current_low >= reversal_threshold:
                # P0-5: min_spacing 검사
                spacing_ok = True
                spacing_weeks = 0.0
                if min_spacing_weeks > 0 and state.last_pivot_ts is not None:
                    try:
                        spacing_days = (current_ts - state.last_pivot_ts).days
                        spacing_weeks = spacing_days / 7.0
                        if spacing_weeks < min_spacing_weeks:
                            spacing_ok = False
                    except:
                        pass

                if spacing_ok:
                    # 고점 확정 → 방향 전환
                    state.high = state.running_extreme  # 고점 확정
                    state.pivot_count += 1

                    # P0-5 FIX: extreme vs confirm 시점 분리 저장
                    state.last_extreme_ts = state.candidate_high_ts
                    state.last_confirm_ts = current_ts
                    state.last_pivot_ts = current_ts
                    state.last_pivot_type = "HIGH"

                    state.direction = "down"
                    state.running_extreme = current_low
                    state.candidate_low_ts = current_ts
                    state.last_update_ts = current_ts

                    # k_anchor 갱신
                    if k_mode != "fixed":
                        prices = df['close'].iloc[:i+1].values
                        state.k_anchor = compute_dynamic_k(
                            prices, atr, mode=k_mode, k_fixed=k_fixed,
                            k_min=k_min, k_max=k_max, k_base=k_base,
                            strength_ref=strength_ref, ampz_thr=ampz_thr,
                            k_cycle=k_cycle, k_trend=k_trend
                        )
                    state.atr_anchor = atr

                else:
                    # P0-5b: spacing 부족 → pending 상태 진입
                    # print(f"[ZigZag] PENDING_START: HIGH @ ${state.running_extreme:,.0f}, spacing={spacing_weeks:.1f}w < {min_spacing_weeks}w")
                    state.pending = True
                    state.pending_type = "HIGH"
                    state.pending_pivot_price = state.running_extreme
                    state.pending_extreme_ts = state.candidate_high_ts
                    state.pending_start_ts = current_ts
                    state.pending_opposite_price = current_low
                    state.pending_opposite_ts = current_ts

        elif state.direction == "down":
            # 하락 중: 새로운 저점 추적
            if current_low < state.running_extreme:
                state.running_extreme = current_low
                state.candidate_low_ts = current_ts
                state.last_update_ts = current_ts

            # Reversal 조건: running_extreme에서 threshold 이상 상승
            if current_high - state.running_extreme >= reversal_threshold:
                # P0-5: min_spacing 검사
                spacing_ok = True
                spacing_weeks = 0.0
                if min_spacing_weeks > 0 and state.last_pivot_ts is not None:
                    try:
                        spacing_days = (current_ts - state.last_pivot_ts).days
                        spacing_weeks = spacing_days / 7.0
                        if spacing_weeks < min_spacing_weeks:
                            spacing_ok = False
                    except:
                        pass

                if spacing_ok:
                    # 저점 확정 → 방향 전환
                    state.low = state.running_extreme
                    state.pivot_count += 1

                    # P0-5 FIX: extreme vs confirm 시점 분리 저장
                    state.last_extreme_ts = state.candidate_low_ts
                    state.last_confirm_ts = current_ts
                    state.last_pivot_ts = current_ts
                    state.last_pivot_type = "LOW"

                    state.direction = "up"
                    state.running_extreme = current_high
                    state.candidate_high_ts = current_ts
                    state.last_update_ts = current_ts

                    # k_anchor 갱신
                    if k_mode != "fixed":
                        prices = df['close'].iloc[:i+1].values
                        state.k_anchor = compute_dynamic_k(
                            prices, atr, mode=k_mode, k_fixed=k_fixed,
                            k_min=k_min, k_max=k_max, k_base=k_base,
                            strength_ref=strength_ref, ampz_thr=ampz_thr,
                            k_cycle=k_cycle, k_trend=k_trend
                        )
                    state.atr_anchor = atr

                else:
                    # P0-5b: spacing 부족 → pending 상태 진입
                    # print(f"[ZigZag] PENDING_START: LOW @ ${state.running_extreme:,.0f}, spacing={spacing_weeks:.1f}w < {min_spacing_weeks}w")
                    state.pending = True
                    state.pending_type = "LOW"
                    state.pending_pivot_price = state.running_extreme
                    state.pending_extreme_ts = state.candidate_low_ts
                    state.pending_start_ts = current_ts
                    state.pending_opposite_price = current_high
                    state.pending_opposite_ts = current_ts

    return state


def get_dynamic_fib_levels(
    low: float,
    high: float,
    ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    include_extensions: bool = False,
    space: str = "linear",
    direction: str = "up"
) -> List[float]:
    """
    동적 Fib 레벨 가격 리스트 생성 (Linear only).

    방향에 따라 0/1 기준점 변경:
    - direction="up" (상승 스윙): 0=low, 1=high
    - direction="down" (하락 스윙): 0=high, 1=low

    확장 레벨 지원 (-2 ~ +2.618):
    - r < 0: 스윙 시작점 아래/위 확장 (SL용)
    - 0 <= r <= 1: 기본 retracement 범위
    - r > 1: 스윙 끝점 위/아래 확장 (TP용)

    Args:
        low: 앵커 저점
        high: 앵커 고점
        ratios: Fib 비율 (기본 0.236~0.786, 음수/1 초과 가능)
        include_extensions: 확장 레벨 포함 여부 (1.272, 1.618) - 레거시 호환
        space: "linear" (기본값, log는 사용하지 않음)
        direction: "up" (상승 스윙) or "down" (하락 스윙)

    Returns:
        가격 리스트 (오름차순 정렬)
    """
    if low <= 0 or high <= 0 or high <= low:
        return []

    levels = []
    all_ratios = list(ratios)

    # Extension 레벨 (옵션) - 레거시 호환
    if include_extensions:
        all_ratios.extend([1.0, 1.272, 1.618])

    swing_range = high - low

    # Linear space only (log 사용 안 함)
    # 방향에 따라 계산 방식 변경: "0=스윙 시작, 1=스윙 끝"
    for r in all_ratios:
        if direction == "up":
            # 상승 스윙: 0=low, 1=high
            price = low + swing_range * r
        else:
            # 하락 스윙: 0=high, 1=low
            price = high - swing_range * r
        levels.append(price)

    # 정렬 및 중복 제거
    levels = sorted(set(levels))

    return levels


def get_dynamic_fib_levels_from_state(
    state: DynamicFibAnchorState,
    ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    include_extensions: bool = False,
    space: str = "linear",
    direction: str = None
) -> List[float]:
    """
    DynamicFibAnchorState에서 Fib 레벨 생성.

    Args:
        state: 앵커 상태
        ratios: Fib 비율
        include_extensions: 확장 레벨 포함 여부
        space: "linear" (기본값, log는 사용하지 않음)
        direction: "up" or "down" (None이면 state.direction 사용)

    Returns:
        가격 리스트 (유효하지 않으면 빈 리스트)
    """
    if not state.is_valid():
        return []

    # direction이 지정되지 않으면 state에서 가져옴
    swing_direction = direction if direction is not None else state.direction

    return get_dynamic_fib_levels(
        state.low, state.high, ratios, include_extensions, space, swing_direction
    )


def check_confluence(
    price: float,
    macro_levels: List[float],
    dyn_levels: List[float],
    tolerance: float = 0.005
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Macro Fib 레벨과 Dynamic Fib 레벨의 confluence 체크.

    Args:
        price: 현재가
        macro_levels: Macro Fib 레벨 리스트
        dyn_levels: Dynamic Fib 레벨 리스트
        tolerance: 허용 오차 비율 (기본 0.5%)

    Returns:
        (is_confluence, macro_level, dyn_level) 튜플
        - is_confluence: confluence 여부
        - macro_level: 가까운 macro 레벨 (None이면 없음)
        - dyn_level: 가까운 dyn 레벨 (None이면 없음)
    """
    def find_nearest(levels: List[float], target: float) -> Optional[float]:
        if not levels:
            return None
        nearest = min(levels, key=lambda x: abs(x - target))
        if abs(nearest - target) / target <= tolerance:
            return nearest
        return None

    macro_near = find_nearest(macro_levels, price)
    dyn_near = find_nearest(dyn_levels, price)

    is_confluence = macro_near is not None and dyn_near is not None

    return is_confluence, macro_near, dyn_near


def filter_tp_by_confluence(
    tp_candidates: List[float],
    macro_levels: List[float],
    dyn_levels: List[float],
    tolerance: float = 0.005,
    require_confluence: bool = False
) -> List[float]:
    """
    TP 후보를 confluence 기준으로 필터링.

    Args:
        tp_candidates: TP 후보 가격 리스트
        macro_levels: Macro Fib 레벨 리스트
        dyn_levels: Dynamic Fib 레벨 리스트
        tolerance: 허용 오차 비율
        require_confluence: True면 confluence인 TP만 반환

    Returns:
        필터링된 TP 리스트
    """
    if not require_confluence:
        return tp_candidates

    filtered = []
    for tp in tp_candidates:
        is_conf, _, _ = check_confluence(tp, macro_levels, dyn_levels, tolerance)
        if is_conf:
            filtered.append(tp)

    return filtered if filtered else tp_candidates  # 없으면 원본 반환


# Utility: 상태 초기화
def create_initial_state(mode: str = "rolling") -> DynamicFibAnchorState:
    """초기 상태 생성"""
    return DynamicFibAnchorState(
        low=0.0,
        high=0.0,
        running_extreme=0.0,
        direction="unknown",
        last_update_ts=None,
        last_pivot_ts=None,
        mode=mode,
        pivot_count=0
    )


# =============================================================================
# Fractal Fibonacci (L0 + L1)
# =============================================================================

def get_fractal_fib_levels(
    low: float,
    high: float,
    l0_ratios: Tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0),
    l1_ratios: Tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786),
    space: str = "linear"
) -> List[float]:
    """
    L0 + L1 프랙탈 Fib 레벨 생성 (Linear only).

    L0: 전체 범위 (low ~ high)
    L1: 각 L0 세그먼트를 다시 Fib로 분할

    Args:
        low: 앵커 저점
        high: 앵커 고점
        l0_ratios: L0 비율 (기본 0~1 전체)
        l1_ratios: L1 세분화 비율
        space: "linear" (기본값, log는 사용하지 않음)

    Returns:
        모든 L0 + L1 레벨 (정렬, 중복 제거)
    """
    if low <= 0 or high <= 0 or high <= low:
        return []

    levels = set()
    swing_range = high - low

    # L0 레벨 계산 (linear only)
    l0_prices = []
    for r in l0_ratios:
        price = low + swing_range * r
        l0_prices.append(price)
        levels.add(price)

    # L1 레벨 계산 (각 L0 세그먼트 내)
    for i in range(len(l0_prices) - 1):
        seg_low = l0_prices[i]
        seg_high = l0_prices[i + 1]
        seg_range = seg_high - seg_low
        for r in l1_ratios:
            price = seg_low + seg_range * r
            levels.add(price)

    return sorted(levels)


# =============================================================================
# Fallback Logic: ZigZag primary, Macro fallback
# =============================================================================

# Macro Fib 앵커 (정적, BTC 역사적 범위)
MACRO_FIB_LOW = 3120.0
MACRO_FIB_HIGH = 143360.0


def get_fib_levels_with_fallback(
    price: float,
    zigzag_state: Optional[DynamicFibAnchorState],
    use_fractal: bool = True,
    space: str = "linear"
) -> Tuple[List[float], str]:
    """
    ZigZag 우선, 범위 밖이면 Macro Fallback (L0+L1 포함).

    Args:
        price: 현재 가격
        zigzag_state: ZigZag 앵커 상태 (None이면 Macro 사용)
        use_fractal: L1 프랙탈 포함 여부 (기본 True)
        space: "linear" (기본값, log는 사용하지 않음)

    Returns:
        (levels, source) 튜플
        - levels: Fib 레벨 리스트
        - source: "zigzag" | "macro" | "macro_fallback"

    Fallback 조건:
        1. zigzag_state가 None이거나 is_valid() == False → Macro
        2. price > zigzag.high → Macro (위로 벗어남)
        3. price < zigzag.low → Macro (아래로 벗어남, 반등용)
    """
    # 1. ZigZag 유효성 체크
    if zigzag_state is None or not zigzag_state.is_valid():
        if use_fractal:
            levels = get_fractal_fib_levels(MACRO_FIB_LOW, MACRO_FIB_HIGH, space=space)
        else:
            levels = get_dynamic_fib_levels(
                MACRO_FIB_LOW, MACRO_FIB_HIGH,
                include_extensions=True, space=space
            )
        return levels, "macro"

    zz_low, zz_high = zigzag_state.low, zigzag_state.high

    # 2. 가격이 ZigZag 범위 내 → ZigZag 사용
    if zz_low <= price <= zz_high:
        if use_fractal:
            levels = get_fractal_fib_levels(zz_low, zz_high, space=space)
        else:
            levels = get_dynamic_fib_levels(
                zz_low, zz_high,
                include_extensions=True, space=space
            )
        return levels, "zigzag"

    # 3. 가격이 ZigZag 위로 벗어남 → Macro로 TP 탐색
    if price > zz_high:
        if use_fractal:
            macro_levels = get_fractal_fib_levels(MACRO_FIB_LOW, MACRO_FIB_HIGH, space=space)
        else:
            macro_levels = get_dynamic_fib_levels(
                MACRO_FIB_LOW, MACRO_FIB_HIGH,
                include_extensions=True, space=space
            )
        # 현재가 위의 레벨만 반환 (TP용)
        levels = [lvl for lvl in macro_levels if lvl > price]
        return levels, "macro_fallback"

    # 4. 가격이 ZigZag 아래로 벗어남 → Macro로 지지선 탐색 (반등용!)
    if price < zz_low:
        if use_fractal:
            macro_levels = get_fractal_fib_levels(MACRO_FIB_LOW, MACRO_FIB_HIGH, space=space)
        else:
            macro_levels = get_dynamic_fib_levels(
                MACRO_FIB_LOW, MACRO_FIB_HIGH,
                include_extensions=True, space=space
            )
        # 현재가 아래의 레벨만 반환 (지지/반등용)
        levels = [lvl for lvl in macro_levels if lvl < price]
        return levels, "macro_fallback"

    # Fallthrough (shouldn't reach here)
    return [], "unknown"
