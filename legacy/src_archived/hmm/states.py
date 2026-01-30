"""
HMM States Definition
=====================

6-State Wyckoff HMM 상태 정의 및 VaR 값.

Origin: WPCN run_step8_hmm_train.py + run_step9_hmm_risk_filter.py
"""
from typing import Dict, Tuple

# 6 Wyckoff-based HMM States
HMM_STATES = [
    'accumulation',      # 축적 (상승 시작점)
    're_accumulation',   # 재축적 (횡보 바닥)
    'distribution',      # 배분 (하락 시작점)
    're_distribution',   # 재배분 (횡보 천정)
    'markup',           # 상승추세
    'markdown',         # 하락추세
]

STATE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(HMM_STATES)}
IDX_TO_STATE: Dict[int, str] = {i: s for i, s in enumerate(HMM_STATES)}

# 상태별 VaR 5% (OOS 검증 완료 - WPCN)
VAR5_BY_STATE: Dict[str, float] = {
    'accumulation': -5.56,      # 가장 안전
    're_accumulation': -6.91,
    'distribution': -5.63,
    're_distribution': -8.85,
    'markup': -7.16,
    'markdown': -10.52,         # 가장 위험
    'range': -7.0,              # 불확실 (중간값)
}

# 불확실성 임계치 (max_prob < 이 값이면 'range')
UNCERTAINTY_THRESHOLD = 0.45

# Sticky HMM 설정
STICKY_KAPPA = 10.0  # 자기전이 강화 계수


def classify_regime(state: str) -> str:
    """
    HMM 상태를 레짐으로 분류

    - ranging: 횡보장 (accumulation, distribution, re_*)
    - trending: 추세장 (markup, markdown)

    Args:
        state: HMM 상태명

    Returns:
        'ranging' or 'trending'
    """
    if state in ('accumulation', 'distribution', 're_accumulation', 're_distribution', 'range'):
        return 'ranging'
    else:  # markup, markdown
        return 'trending'


def get_long_permit_states() -> Tuple[str, ...]:
    """Long 진입 허용 상태 (기본값)"""
    return ('markup', 'accumulation', 're_accumulation')


def get_short_permit_states() -> Tuple[str, ...]:
    """Short 진입 허용 상태 (기본값)"""
    return ('markdown',)
