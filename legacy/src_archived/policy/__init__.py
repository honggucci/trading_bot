"""
Policy Layer - HMM Policy + Regime Parameters
==============================================

Components:
- hmm_policy.py: HMM Entry Gate 정책 (VaR, Cooldown, Permit)
- regime_params.py: 레짐별 Legacy Confluence 파라미터
"""
from .hmm_policy import (
    HMMPolicyConfig,
    DEFAULT_HMM_POLICY,
    AGGRESSIVE_HMM_POLICY,
    CONSERVATIVE_HMM_POLICY,
)
from .regime_params import (
    RegimeParams,
    RANGING_PARAMS,
    TRENDING_PARAMS,
    REGIME_PARAMS,
    classify_regime,
    get_regime_params,
    get_fib_ratios_for_regime,
    get_min_score_for_regime,
)

__all__ = [
    # HMM Policy
    'HMMPolicyConfig',
    'DEFAULT_HMM_POLICY',
    'AGGRESSIVE_HMM_POLICY',
    'CONSERVATIVE_HMM_POLICY',

    # Regime Params
    'RegimeParams',
    'RANGING_PARAMS',
    'TRENDING_PARAMS',
    'REGIME_PARAMS',
    'classify_regime',
    'get_regime_params',
    'get_fib_ratios_for_regime',
    'get_min_score_for_regime',
]