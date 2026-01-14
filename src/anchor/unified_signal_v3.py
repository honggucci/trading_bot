# -*- coding: utf-8 -*-
"""
Unified Signal V3 - Legacy + HMM + TFPredictor Integration
==========================================================

V3 변경사항:
- TFPredictor Multi-TF Confluence Boost
- 엣지케이스 방어코드 추가
- Exit Logic 통합

핵심 흐름:
1. TFPredictor → Multi-TF Bias + Confluence Zones
2. Legacy Confluence → 15m Zone + Divergence Score
3. MTF Boost → 여러 TF Fib 겹침 시 점수 부스트
4. HMM Gate → Soft Sizing + Cooldown
5. Exit Logic → SL/TP/Trailing

엣지케이스 처리:
- TFPredictor 실패 → Legacy only fallback
- Confluence Zone 0개 → MTF Boost = 0
- HMM vs TF Bias 충돌 → HMM 우선 (Wyckoff 기반)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, List, Tuple
import pandas as pd
import numpy as np
import logging

from .legacy_pipeline import analyze_confluence, ConfluenceResult
from .unified_signal import REGIME_PARAMS, classify_regime_from_state
from .exit_logic import calc_exit_levels, ExitLevels
from .stochrsi import tv_stoch_rsi, is_oversold, is_overbought

# 로깅
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnifiedSignalV3:
    """통합 신호 결과 V3"""
    # 기본 신호
    allowed: bool
    side: Literal['long', 'short', 'none']
    confidence: float  # 0-1 (Legacy + MTF Boost)
    size_mult: float  # HMM soft sizing

    # Legacy Confluence
    legacy_score: float
    is_in_zone: bool
    zone_label: Optional[str]

    # MTF Boost
    mtf_boost: float
    mtf_confluence_count: int
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]

    # TF Bias
    tf_bias: str  # 'bullish', 'bearish', 'neutral'
    tf_bias_confidence: float

    # HMM
    hmm_state: str
    hmm_allowed: bool
    hmm_blocked_reason: Optional[str]

    # Divergence
    is_reg_now: bool
    is_hid_now: bool

    # Regime
    regime: str

    # Exit Levels
    exit_levels: Optional[ExitLevels] = None

    # 에러/경고
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False

    # 디버그
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MTF Boost Calculation
# =============================================================================

def calc_mtf_boost(
    current_price: float,
    confluence_zones: List,
    *,
    distance_threshold: float = 3.0,
) -> Tuple[float, int, Optional[float], Optional[float]]:
    """
    MTF Confluence Boost 계산

    Args:
        current_price: 현재가
        confluence_zones: PredictionZone 리스트
        distance_threshold: 부스트 적용 거리 (%)

    Returns:
        (boost, confluence_count, nearest_support, nearest_resistance)
    """
    if not confluence_zones:
        return 0.0, 0, None, None

    boost = 0.0
    confluence_count = 0
    supports = []
    resistances = []

    for zone in confluence_zones:
        zone_price = getattr(zone, 'price', None)
        zone_strength = getattr(zone, 'strength', 1)

        if zone_price is None:
            continue

        distance_pct = (zone_price - current_price) / current_price * 100

        if abs(distance_pct) <= distance_threshold:
            confluence_count += 1

            # 부스트 계산
            if zone_strength >= 4:
                boost = max(boost, 0.3)
            elif zone_strength >= 3:
                boost = max(boost, 0.2)
            elif zone_strength >= 2:
                boost = max(boost, 0.1)

            # Golden Pocket 추가 부스트
            zone_type = getattr(zone, 'zone_type', None)
            if zone_type and hasattr(zone_type, 'value') and zone_type.value == 'golden_pocket':
                boost += 0.1

        # 지지/저항 분류
        if distance_pct < 0:
            supports.append(zone_price)
        else:
            resistances.append(zone_price)

    # 가장 가까운 지지/저항
    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    return min(boost, 0.4), confluence_count, nearest_support, nearest_resistance


# =============================================================================
# Regime Determination
# =============================================================================

def determine_regime(
    hmm_state: str,
    tf_bias: str,
) -> str:
    """
    HMM 상태 + TF Bias로 Regime 결정

    - 둘 다 일치하면 해당 regime
    - 충돌 시 HMM 우선 (Wyckoff 기반)
    """
    hmm_regime = classify_regime_from_state(hmm_state)

    if tf_bias in ['strong_bullish', 'bullish', 'strong_bearish', 'bearish']:
        tf_regime = 'trending'
    else:
        tf_regime = 'ranging'

    # HMM 우선
    return hmm_regime


# =============================================================================
# Main Signal Functions
# =============================================================================

def check_unified_long_signal_v3(
    df_15m: pd.DataFrame,
    hmm_gate,
    ts_15m: pd.Timestamp,
    *,
    tf_predictor=None,
    regime: Optional[str] = None,
    min_confluence_score: Optional[float] = None,
    require_divergence: bool = False,
    include_exit_levels: bool = True,
) -> UnifiedSignalV3:
    """
    V3: Legacy + HMM + TFPredictor Long Signal

    엣지케이스 처리:
    1. TFPredictor 없음/실패 → Legacy only fallback
    2. Confluence Zone 없음 → MTF Boost = 0
    3. HMM 거부 → allowed = False

    Args:
        df_15m: 15분봉 DataFrame
        hmm_gate: HMMEntryGate 인스턴스
        ts_15m: 15분봉 타임스탬프
        tf_predictor: TFPredictor 인스턴스 (선택)
        regime: 레짐 강제 지정
        min_confluence_score: 최소 점수 (None이면 레짐 기본값)
        require_divergence: Divergence 필수 여부
        include_exit_levels: Exit 레벨 계산 여부

    Returns:
        UnifiedSignalV3
    """
    warnings = []
    fallback_used = False

    # 1. HMM Gate 체크 (먼저 - state 필요)
    try:
        gate_decision = hmm_gate.check_entry(ts_15m, 'long')
    except Exception as e:
        logger.warning(f"HMM Gate error: {e}")
        warnings.append(f"HMM Gate error: {e}")
        # 에러 시 안전하게 진입 차단 (Short과 동일)
        gate_decision = type('obj', (object,), {
            'allowed': False,
            'state': 'unknown',
            'size_mult': 0.0,
            'blocked_reason': f"HMM Gate error: {e}",
            'expected_var': 0.02,
            'markdown_prob': 0.0,
            'cooldown_active': False,
        })()

    # 2. TFPredictor 분석 (있으면)
    tf_bias = 'neutral'
    tf_bias_confidence = 0.0
    confluence_zones = []
    mtf_boost = 0.0
    mtf_confluence_count = 0
    nearest_support = None
    nearest_resistance = None

    if tf_predictor is not None:
        try:
            # TFPredictor가 초기화되어 있는지 확인
            if hasattr(tf_predictor, 'hierarchy') and tf_predictor.hierarchy is not None:
                prediction = tf_predictor.predict_next_move()
                tf_bias = prediction.bias.value if hasattr(prediction.bias, 'value') else str(prediction.bias)
                tf_bias_confidence = prediction.confidence
                confluence_zones = prediction.confluence_zones or []

                # MTF Boost 계산
                current_price = float(df_15m['close'].iloc[-1])
                mtf_boost, mtf_confluence_count, nearest_support, nearest_resistance = calc_mtf_boost(
                    current_price, confluence_zones
                )
            else:
                warnings.append("TFPredictor not initialized (hierarchy is None)")
                fallback_used = True
        except Exception as e:
            logger.warning(f"TFPredictor error: {e}")
            warnings.append(f"TFPredictor error: {e}")
            fallback_used = True
    else:
        fallback_used = True

    # 3. Regime 결정
    if regime is None:
        regime = determine_regime(gate_decision.state, tf_bias)

    # 4. 레짐별 파라미터
    params = REGIME_PARAMS.get(regime, REGIME_PARAMS['ranging'])
    if min_confluence_score is None:
        min_confluence_score = params['min_confluence_score']

    # 5. Legacy Confluence 분석
    try:
        confluence_result = analyze_confluence(
            df_15m,
            fib_ratios=params['fib_ratios'],
            min_score=min_confluence_score,
        )
    except Exception as e:
        logger.warning(f"Confluence analysis error: {e}")
        warnings.append(f"Confluence analysis error: {e}")
        confluence_result = ConfluenceResult(success=False, error=str(e))

    # 6. Legacy 점수 추출
    legacy_score = 0.0
    is_in_zone = False
    zone_label = None
    is_reg_now = False
    is_hid_now = False

    if confluence_result.success:
        is_in_zone = confluence_result.is_in_zone
        is_reg_now = confluence_result.is_reg_now
        is_hid_now = confluence_result.is_hid_now

        if confluence_result.current_zone:
            legacy_score = confluence_result.current_zone.score
            zone_label = confluence_result.current_zone.zone.label

    # 7. 최종 Confidence = Legacy + MTF Boost
    final_confidence = legacy_score + mtf_boost

    # 8. 조건 체크
    confluence_ok = is_in_zone and final_confidence >= min_confluence_score
    hmm_ok = gate_decision.allowed
    divergence_ok = True if not require_divergence else (is_reg_now or is_hid_now)

    # 9. 최종 결정
    all_conditions_met = confluence_ok and hmm_ok and divergence_ok

    # 10. Exit Levels 계산 (진입 허용 시)
    exit_levels = None
    if all_conditions_met and include_exit_levels:
        try:
            current_price = float(df_15m['close'].iloc[-1])

            # ATR fallback with warning
            if 'atr' in df_15m.columns:
                atr = float(df_15m['atr'].iloc[-1])
            else:
                atr = current_price * 0.02
                warnings.append(f"ATR column missing, using 2% fallback (${atr:,.0f})")

            exit_levels = calc_exit_levels(
                entry_price=current_price,
                side='long',
                atr=atr,
                confluence_zones=confluence_zones,
            )
        except Exception as e:
            warnings.append(f"Exit levels error: {e}")

    return UnifiedSignalV3(
        allowed=all_conditions_met,
        side='long' if all_conditions_met else 'none',
        confidence=final_confidence if all_conditions_met else 0.0,
        size_mult=gate_decision.size_mult if all_conditions_met else 0.0,

        legacy_score=legacy_score,
        is_in_zone=is_in_zone,
        zone_label=zone_label,

        mtf_boost=mtf_boost,
        mtf_confluence_count=mtf_confluence_count,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,

        tf_bias=tf_bias,
        tf_bias_confidence=tf_bias_confidence,

        hmm_state=gate_decision.state,
        hmm_allowed=gate_decision.allowed,
        hmm_blocked_reason=gate_decision.blocked_reason,

        is_reg_now=is_reg_now,
        is_hid_now=is_hid_now,

        regime=regime,

        exit_levels=exit_levels,

        warnings=warnings,
        fallback_used=fallback_used,

        details={
            'params': params,
            'conditions': {
                'confluence_ok': confluence_ok,
                'hmm_ok': hmm_ok,
                'divergence_ok': divergence_ok,
            },
            'gate_decision': {
                'state': gate_decision.state,
                'expected_var': getattr(gate_decision, 'expected_var', None),
                'markdown_prob': getattr(gate_decision, 'markdown_prob', None),
                'cooldown_active': getattr(gate_decision, 'cooldown_active', None),
            },
        },
    )


def check_unified_short_signal_v3(
    df_15m: pd.DataFrame,
    hmm_gate,
    ts_15m: pd.Timestamp,
    *,
    tf_predictor=None,
    regime: Optional[str] = None,
    include_exit_levels: bool = True,
) -> UnifiedSignalV3:
    """
    V3: Legacy + HMM + TFPredictor Short Signal

    Short 조건:
    1. HMM Gate: short permit (markdown_prob > 60%)
    2. StochRSI overbought
    3. TF Bias: bearish (optional boost)
    """
    warnings = []
    fallback_used = False

    # 1. HMM Gate 체크
    try:
        gate_decision = hmm_gate.check_entry(ts_15m, 'short')
    except Exception as e:
        logger.warning(f"HMM Gate error: {e}")
        warnings.append(f"HMM Gate error: {e}")
        gate_decision = type('obj', (object,), {
            'allowed': False,
            'state': 'unknown',
            'size_mult': 0.0,
            'blocked_reason': str(e),
        })()

    # 2. TFPredictor (있으면)
    tf_bias = 'neutral'
    tf_bias_confidence = 0.0
    mtf_boost = 0.0
    mtf_confluence_count = 0
    nearest_support = None
    nearest_resistance = None

    if tf_predictor is not None:
        try:
            if hasattr(tf_predictor, 'hierarchy') and tf_predictor.hierarchy is not None:
                prediction = tf_predictor.predict_next_move()
                tf_bias = prediction.bias.value if hasattr(prediction.bias, 'value') else str(prediction.bias)
                tf_bias_confidence = prediction.confidence

                # Bearish bias면 boost
                if tf_bias in ['bearish', 'strong_bearish']:
                    mtf_boost = 0.15
            else:
                fallback_used = True
        except Exception as e:
            warnings.append(f"TFPredictor error: {e}")
            fallback_used = True
    else:
        fallback_used = True

    # 3. Regime 결정
    if regime is None:
        regime = determine_regime(gate_decision.state, tf_bias)

    params = REGIME_PARAMS.get(regime, REGIME_PARAMS['ranging'])

    # 4. StochRSI 계산
    if 'stoch_d' in df_15m.columns:
        stochrsi_d = float(df_15m['stoch_d'].iloc[-1])
    else:
        close_arr = df_15m['close'].values
        k, d = tv_stoch_rsi(close_arr)
        stochrsi_d = float(d[-1])

    stochrsi_ok = is_overbought(stochrsi_d, params['overbought_threshold'])

    # 5. 조건
    hmm_ok = gate_decision.allowed

    # 6. 최종 결정
    all_conditions_met = hmm_ok and stochrsi_ok

    # Short confidence = StochRSI / 100 + MTF Boost
    confidence = (stochrsi_d / 100 + mtf_boost) if all_conditions_met else 0.0

    # 7. Exit Levels
    exit_levels = None
    if all_conditions_met and include_exit_levels:
        try:
            current_price = float(df_15m['close'].iloc[-1])

            # ATR fallback with warning
            if 'atr' in df_15m.columns:
                atr = float(df_15m['atr'].iloc[-1])
            else:
                atr = current_price * 0.02
                warnings.append(f"ATR column missing, using 2% fallback (${atr:,.0f})")

            exit_levels = calc_exit_levels(
                entry_price=current_price,
                side='short',
                atr=atr,
            )
        except Exception as e:
            warnings.append(f"Exit levels error: {e}")

    return UnifiedSignalV3(
        allowed=all_conditions_met,
        side='short' if all_conditions_met else 'none',
        confidence=confidence,
        size_mult=gate_decision.size_mult if all_conditions_met else 0.0,

        legacy_score=stochrsi_d / 100,
        is_in_zone=False,  # Short은 zone 미사용
        zone_label=None,

        mtf_boost=mtf_boost,
        mtf_confluence_count=mtf_confluence_count,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,

        tf_bias=tf_bias,
        tf_bias_confidence=tf_bias_confidence,

        hmm_state=gate_decision.state,
        hmm_allowed=gate_decision.allowed,
        hmm_blocked_reason=getattr(gate_decision, 'blocked_reason', None),

        is_reg_now=False,
        is_hid_now=False,

        regime=regime,

        exit_levels=exit_levels,

        warnings=warnings,
        fallback_used=fallback_used,

        details={
            'stochrsi_d': stochrsi_d,
            'stochrsi_ok': stochrsi_ok,
            'params': params,
        },
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def format_signal_v3(signal: UnifiedSignalV3) -> str:
    """V3 신호 포맷팅"""
    lines = [
        "=" * 60,
        f"Unified Signal V3 - {signal.side.upper()}",
        "=" * 60,
        "",
        f"Allowed: {signal.allowed}",
        f"Confidence: {signal.confidence:.2f} (Legacy: {signal.legacy_score:.2f} + MTF: {signal.mtf_boost:.2f})",
        f"Size Mult: {signal.size_mult:.2f}",
        "",
        "--- Legacy Confluence ---",
        f"In Zone: {signal.is_in_zone}",
        f"Zone Label: {signal.zone_label or 'N/A'}",
        f"Divergence: REG={signal.is_reg_now}, HID={signal.is_hid_now}",
        "",
        "--- MTF Analysis ---",
        f"TF Bias: {signal.tf_bias} ({signal.tf_bias_confidence:.1%})",
        f"MTF Confluence Count: {signal.mtf_confluence_count}",
        f"Nearest Support: ${signal.nearest_support:,.0f}" if signal.nearest_support else "Nearest Support: N/A",
        f"Nearest Resistance: ${signal.nearest_resistance:,.0f}" if signal.nearest_resistance else "Nearest Resistance: N/A",
        "",
        "--- HMM Gate ---",
        f"State: {signal.hmm_state}",
        f"Allowed: {signal.hmm_allowed}",
        f"Blocked Reason: {signal.hmm_blocked_reason or 'None'}",
        "",
        f"Regime: {signal.regime}",
        f"Fallback Used: {signal.fallback_used}",
    ]

    if signal.warnings:
        lines.append("")
        lines.append("--- Warnings ---")
        for w in signal.warnings:
            lines.append(f"  - {w}")

    if signal.exit_levels:
        lines.append("")
        lines.append("--- Exit Levels ---")
        lines.append(f"SL: ${signal.exit_levels.stop_loss:,.0f} (-{signal.exit_levels.stop_loss_pct:.1f}%)")
        lines.append(f"TP1: ${signal.exit_levels.take_profit_1:,.0f} (+{signal.exit_levels.tp1_pct:.1f}%)")
        lines.append(f"TP2: ${signal.exit_levels.take_profit_2:,.0f} (+{signal.exit_levels.tp2_pct:.1f}%)")

    lines.append("=" * 60)
    return "\n".join(lines)
