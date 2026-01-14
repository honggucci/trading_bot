"""
HMM Module - 6-State Wyckoff Hidden Markov Model
=================================================

Wyckoff 기반 6상태 HMM 구현.

Components:
- states.py: 상태 정의, VaR 값, 레짐 분류
- features.py: 5차원 Emission Features 계산
- train.py: HMM 학습 및 추론

Origin: WPCN backtester (wpcn-backtester-cli-noflask)
"""
from .states import (
    HMM_STATES,
    STATE_TO_IDX,
    IDX_TO_STATE,
    VAR5_BY_STATE,
    UNCERTAINTY_THRESHOLD,
    STICKY_KAPPA,
    classify_regime,
    get_long_permit_states,
    get_short_permit_states,
)
from .features import (
    compute_emission_features,
    FEATURE_COLS,
)
from .train import (
    HMMTrainResult,
    compute_initial_params,
    train_hmm,
    hungarian_matching,
    decode_with_uncertainty,
    create_posterior_map,
)

__all__ = [
    # States
    'HMM_STATES',
    'STATE_TO_IDX',
    'IDX_TO_STATE',
    'VAR5_BY_STATE',
    'UNCERTAINTY_THRESHOLD',
    'STICKY_KAPPA',
    'classify_regime',
    'get_long_permit_states',
    'get_short_permit_states',

    # Features
    'compute_emission_features',
    'FEATURE_COLS',

    # Training
    'HMMTrainResult',
    'compute_initial_params',
    'train_hmm',
    'hungarian_matching',
    'decode_with_uncertainty',
    'create_posterior_map',
]
