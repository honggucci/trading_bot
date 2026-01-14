"""
Regime-based Parameters
=======================

HMM 레짐별 Legacy Confluence 파라미터.

핵심 아이디어 (Option D):
- HMM이 "언제"가 아닌 "어떻게" 트레이딩할지 결정
- 횡보장: 깊은 피보나치, 낮은 임계치
- 추세장: 얕은 피보나치, 높은 임계치

GPT Strategist 추천 (2026-01):
- 핵심 철학 "시장은 본질적으로 항상 횡보"와 가장 정합적
- HMM을 진입 필터가 아닌 파라미터 조정자로 사용
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class RegimeParams:
    """레짐별 파라미터"""
    # Fibonacci
    fib_ratios: Tuple[float, ...]
    fib_min_half_atr: float = 0.25
    fib_max_half_mult: float = 0.8

    # Confluence
    min_confluence_score: float = 0.3

    # StochRSI
    oversold_threshold: float = 20.0
    overbought_threshold: float = 80.0

    # Execution
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.5


# 레짐별 파라미터 정의
RANGING_PARAMS = RegimeParams(
    fib_ratios=(0.618, 0.786),      # 깊은 되돌림
    min_confluence_score=0.3,        # 낮은 임계치
    oversold_threshold=20.0,
    overbought_threshold=80.0,
    atr_sl_mult=1.5,
    atr_tp_mult=2.5,
)

TRENDING_PARAMS = RegimeParams(
    fib_ratios=(0.382, 0.5),         # 얕은 되돌림
    min_confluence_score=0.5,        # 높은 임계치
    oversold_threshold=15.0,         # 더 극단적 oversold
    overbought_threshold=85.0,
    atr_sl_mult=2.0,                 # 추세장에서 더 넓은 SL
    atr_tp_mult=3.0,
)


REGIME_PARAMS: Dict[str, RegimeParams] = {
    'ranging': RANGING_PARAMS,
    'trending': TRENDING_PARAMS,
}


def classify_regime(state: str) -> str:
    """
    HMM 상태를 레짐으로 분류

    Args:
        state: HMM 상태명 (accumulation, markup, etc.)

    Returns:
        'ranging' or 'trending'
    """
    if state in ('accumulation', 'distribution', 're_accumulation', 're_distribution', 'range', 'unknown'):
        return 'ranging'
    else:  # markup, markdown
        return 'trending'


def get_regime_params(state: str) -> RegimeParams:
    """
    HMM 상태에 맞는 레짐 파라미터 반환

    Args:
        state: HMM 상태명

    Returns:
        RegimeParams
    """
    regime = classify_regime(state)
    return REGIME_PARAMS.get(regime, RANGING_PARAMS)


def get_fib_ratios_for_regime(regime: str) -> Tuple[float, ...]:
    """레짐별 피보나치 비율 반환"""
    params = REGIME_PARAMS.get(regime, RANGING_PARAMS)
    return params.fib_ratios


def get_min_score_for_regime(regime: str) -> float:
    """레짐별 최소 Confluence 점수 반환"""
    params = REGIME_PARAMS.get(regime, RANGING_PARAMS)
    return params.min_confluence_score