"""
Unified Signal V2 - Legacy Confluence + HMM Integration
========================================================

Legacy Confluence 분석 + HMM Entry Gate 통합 신호.

V2 변경사항:
- find_best_divergence() → analyze_confluence() (REF 기반)
- Fib Zone + Divergence + Confluence Score 활용
- 레짐별 파라미터 적응 (Option D)

핵심 아이디어:
- Legacy: "어디서" 진입할지 (Confluence Zone)
- HMM: "얼마나" 진입할지 (VaR 기반 Sizing)

5-Persona Review:
- 회의적: REF 기반 Divergence가 더 정확
- 비판적: Confluence Score로 노이즈 필터링
- 완벽주의자: 레짐별 파라미터 적응
- 악마의 변호인: HMM soft sizing으로 리스크 관리
- 낙관론자: 두 시스템의 시너지
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Tuple
import pandas as pd
import numpy as np

from .legacy_pipeline import analyze_confluence, ConfluenceResult
from .stochrsi import tv_stoch_rsi, is_oversold, is_overbought


# ============================================================================
# Regime-based Parameters (Option D)
# ============================================================================

REGIME_PARAMS = {
    'ranging': {
        'fib_ratios': (0.618, 0.786),      # 깊은 되돌림
        'min_confluence_score': 0.3,        # 낮은 임계치
        'oversold_threshold': 20.0,
        'overbought_threshold': 80.0,
    },
    'trending': {
        'fib_ratios': (0.382, 0.5),         # 얕은 되돌림
        'min_confluence_score': 0.5,        # 높은 임계치
        'oversold_threshold': 15.0,         # 더 극단적 oversold
        'overbought_threshold': 85.0,
    },
}


def classify_regime_from_state(state: str) -> str:
    """HMM 상태에서 레짐 분류"""
    if state in ('accumulation', 'distribution', 're_accumulation', 're_distribution', 'range', 'unknown'):
        return 'ranging'
    else:  # markup, markdown
        return 'trending'


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class UnifiedSignal:
    """통합 신호 결과 V2"""
    allowed: bool
    side: Literal['long', 'short', 'none']
    confidence: float  # 0-1 (Confluence Score)
    size_mult: float  # HMM soft sizing

    # Confluence 결과
    confluence_score: float
    is_in_zone: bool
    zone_label: Optional[str]

    # HMM 결과
    hmm_state: str
    hmm_allowed: bool
    hmm_blocked_reason: Optional[str]

    # Divergence 결과
    is_reg_now: bool  # Regular Divergence 성립
    is_hid_now: bool  # Hidden Divergence 성립

    # StochRSI 결과
    stochrsi_d: float
    stochrsi_condition: bool  # oversold/overbought 조건

    # 레짐
    regime: str

    # 디버그 정보
    details: Dict[str, Any]


# ============================================================================
# Signal Check Functions V2
# ============================================================================

def check_unified_long_signal_v2(
    df_15m: pd.DataFrame,
    hmm_gate,  # HMMEntryGate instance
    ts_15m: pd.Timestamp,
    *,
    regime: Optional[str] = None,
    min_confluence_score: Optional[float] = None,
    require_divergence: bool = False,
) -> UnifiedSignal:
    """
    V2: Legacy Confluence + HMM Long Signal

    조건:
    1. Legacy Confluence: 현재가가 Confluence Zone 안에 있고 score >= min_score
    2. HMM Gate: allowed=True (state permit + sizing)
    3. (선택) Divergence: is_reg_now OR is_hid_now

    Args:
        df_15m: 15분봉 DataFrame
        hmm_gate: HMMEntryGate 인스턴스
        ts_15m: 15분봉 타임스탬프
        regime: 레짐 ('ranging' or 'trending'), None이면 HMM에서 추론
        min_confluence_score: 최소 Confluence 점수, None이면 레짐 기본값
        require_divergence: True면 Divergence 조건 필수

    Returns:
        UnifiedSignal
    """
    # 1. HMM Gate 체크 (먼저 - state 필요)
    gate_decision = hmm_gate.check_entry(ts_15m, 'long')

    # 레짐 결정
    if regime is None:
        regime = classify_regime_from_state(gate_decision.state)

    # 레짐별 파라미터
    params = REGIME_PARAMS.get(regime, REGIME_PARAMS['ranging'])
    if min_confluence_score is None:
        min_confluence_score = params['min_confluence_score']

    # 2. Legacy Confluence 분석
    confluence_result = analyze_confluence(
        df_15m,
        fib_ratios=params['fib_ratios'],
        min_score=min_confluence_score,
    )

    # 기본값 설정
    confluence_ok = False
    confluence_score = 0.0
    is_in_zone = False
    zone_label = None
    is_reg_now = False
    is_hid_now = False
    stochrsi_d = 50.0
    stochrsi_ok = False

    if confluence_result.success:
        is_in_zone = confluence_result.is_in_zone
        is_reg_now = confluence_result.is_reg_now
        is_hid_now = confluence_result.is_hid_now

        if confluence_result.current_zone:
            confluence_score = confluence_result.current_zone.score
            zone_label = confluence_result.current_zone.zone.label
            confluence_ok = is_in_zone and confluence_score >= min_confluence_score

        # StochRSI 확인
        if 'stoch_d' in df_15m.columns:
            stochrsi_d = float(df_15m['stoch_d'].iloc[-1])
        stochrsi_ok = is_oversold(stochrsi_d, params['oversold_threshold'])

    # 3. HMM Gate 결과
    hmm_ok = gate_decision.allowed

    # 4. Divergence 조건 (선택적)
    divergence_ok = True
    if require_divergence:
        divergence_ok = is_reg_now or is_hid_now

    # 5. 최종 결정
    all_conditions_met = confluence_ok and hmm_ok and divergence_ok

    # Confidence = Confluence Score
    confidence = confluence_score if all_conditions_met else 0.0

    return UnifiedSignal(
        allowed=all_conditions_met,
        side='long' if all_conditions_met else 'none',
        confidence=confidence,
        size_mult=gate_decision.size_mult if all_conditions_met else 0.0,

        confluence_score=confluence_score,
        is_in_zone=is_in_zone,
        zone_label=zone_label,

        hmm_state=gate_decision.state,
        hmm_allowed=gate_decision.allowed,
        hmm_blocked_reason=gate_decision.blocked_reason,

        is_reg_now=is_reg_now,
        is_hid_now=is_hid_now,

        stochrsi_d=stochrsi_d,
        stochrsi_condition=stochrsi_ok,

        regime=regime,

        details={
            'confluence_result': {
                'success': confluence_result.success,
                'error': confluence_result.error,
                'ref_price': confluence_result.ref.price if confluence_result.ref else None,
                'ref_rsi': confluence_result.ref.rsi if confluence_result.ref else None,
                'need_reg': confluence_result.need_reg,
                'hid_range': confluence_result.hid_range,
            },
            'gate_decision': {
                'state': gate_decision.state,
                'expected_var': gate_decision.expected_var,
                'markdown_prob': gate_decision.markdown_prob,
                'cooldown_active': gate_decision.cooldown_active,
            },
            'params': params,
            'conditions': {
                'confluence_ok': confluence_ok,
                'hmm_ok': hmm_ok,
                'divergence_ok': divergence_ok,
                'stochrsi_ok': stochrsi_ok,
            },
        },
    )


def check_unified_short_signal_v2(
    df_15m: pd.DataFrame,
    hmm_gate,
    ts_15m: pd.Timestamp,
    *,
    regime: Optional[str] = None,
) -> UnifiedSignal:
    """
    V2: Legacy Confluence + HMM Short Signal

    조건:
    1. HMM Gate: short permit (markdown_prob > 60% + trend_strength < -0.10)
    2. StochRSI overbought
    3. (Phase 2) Bearish Divergence

    Note: Short Divergence는 Phase 2에서 추가 예정
    """
    # 1. HMM Gate 체크
    gate_decision = hmm_gate.check_entry(ts_15m, 'short')

    # 레짐 결정
    if regime is None:
        regime = classify_regime_from_state(gate_decision.state)

    params = REGIME_PARAMS.get(regime, REGIME_PARAMS['ranging'])

    # 2. StochRSI 계산
    if 'stoch_d' in df_15m.columns:
        stochrsi_d = float(df_15m['stoch_d'].iloc[-1])
    else:
        close_arr = df_15m['close'].values
        k, d = tv_stoch_rsi(close_arr)
        df_15m['stoch_k'] = k
        df_15m['stoch_d'] = d
        stochrsi_d = float(d[-1])

    stochrsi_ok = is_overbought(stochrsi_d, params['overbought_threshold'])

    # 3. HMM Gate 결과
    hmm_ok = gate_decision.allowed

    # 4. 최종 결정
    all_conditions_met = hmm_ok and stochrsi_ok

    # Short은 StochRSI로 confidence 계산
    if all_conditions_met:
        confidence = stochrsi_d / 100
    else:
        confidence = 0.0

    return UnifiedSignal(
        allowed=all_conditions_met,
        side='short' if all_conditions_met else 'none',
        confidence=confidence,
        size_mult=gate_decision.size_mult if all_conditions_met else 0.0,

        confluence_score=0.0,  # Short은 confluence 미사용
        is_in_zone=False,
        zone_label=None,

        hmm_state=gate_decision.state,
        hmm_allowed=gate_decision.allowed,
        hmm_blocked_reason=gate_decision.blocked_reason,

        is_reg_now=False,
        is_hid_now=False,

        stochrsi_d=stochrsi_d,
        stochrsi_condition=stochrsi_ok,

        regime=regime,

        details={
            'gate_decision': {
                'state': gate_decision.state,
                'expected_var': gate_decision.expected_var,
                'markdown_prob': gate_decision.markdown_prob,
                'cooldown_active': gate_decision.cooldown_active,
            },
            'params': params,
            'conditions': {
                'hmm_ok': hmm_ok,
                'stochrsi_ok': stochrsi_ok,
            },
        },
    )


# ============================================================================
# Legacy Compatibility (V1 API)
# ============================================================================

def check_unified_long_signal(
    df_15m: pd.DataFrame,
    current_idx: int,
    hmm_gate,
    ts_15m: pd.Timestamp,
    *,
    stochrsi_threshold: float = 20.0,
    divergence_threshold: float = 0.5,
    rsi_col: str = 'rsi',
    close_col: str = 'close',
) -> UnifiedSignal:
    """
    V1 호환용 래퍼 (deprecated)

    V2 사용 권장: check_unified_long_signal_v2()
    """
    return check_unified_long_signal_v2(
        df_15m, hmm_gate, ts_15m,
        require_divergence=False,
    )


def check_unified_short_signal(
    df_15m: pd.DataFrame,
    current_idx: int,
    hmm_gate,
    ts_15m: pd.Timestamp,
    *,
    stochrsi_threshold: float = 80.0,
    close_col: str = 'close',
) -> UnifiedSignal:
    """
    V1 호환용 래퍼 (deprecated)

    V2 사용 권장: check_unified_short_signal_v2()
    """
    return check_unified_short_signal_v2(df_15m, hmm_gate, ts_15m)
