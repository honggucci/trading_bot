"""
HMM Training & Inference
========================

6-State Wyckoff HMM 학습 및 추론.

Features:
- Weakly-supervised initialization (라벨 기반 초기화)
- Sticky HMM (자기전이 강화)
- Hungarian matching (상태-라벨 최적 매칭)
- Uncertainty-based range labeling

Origin: WPCN run_step8_hmm_train.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment

from .states import (
    HMM_STATES, STATE_TO_IDX, IDX_TO_STATE,
    UNCERTAINTY_THRESHOLD, STICKY_KAPPA,
)
from .features import FEATURE_COLS


@dataclass
class HMMTrainResult:
    """HMM 학습 결과"""
    model: Any  # GaussianHMM
    mapping: Dict[int, str]  # HMM state idx → label
    accuracy: float
    metrics: Dict[str, Any]


def compute_initial_params(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    라벨 기반 초기 파라미터 계산 (약지도)

    Args:
        X: emission features (T, D)
        y: labels (T,)
        alpha: smoothing parameter

    Returns:
        means: (n_states, n_features)
        covars: (n_states, n_features) diagonal
        transmat: (n_states, n_states) row-normalized
        startprob: (n_states,)
    """
    n_states = len(HMM_STATES)
    n_features = X.shape[1]

    means = np.zeros((n_states, n_features))
    covars = np.zeros((n_states, n_features))

    for state, idx in STATE_TO_IDX.items():
        mask = (y == state)
        Xi = X[mask]
        if len(Xi) < 10:
            # 샘플 부족시 전체 평균 사용
            means[idx] = X.mean(axis=0)
            covars[idx] = X.var(axis=0) + 1e-4
        else:
            means[idx] = Xi.mean(axis=0)
            covars[idx] = Xi.var(axis=0) + 1e-4

    # 전이행렬
    trans = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(y) - 1):
        a, b = y[t], y[t+1]
        if a in STATE_TO_IDX and b in STATE_TO_IDX:
            trans[STATE_TO_IDX[a], STATE_TO_IDX[b]] += 1.0

    trans = trans + alpha  # smoothing
    trans = trans / trans.sum(axis=1, keepdims=True)

    # 시작 확률 (균등)
    startprob = np.ones(n_states) / n_states

    return means, covars, trans, startprob


def train_hmm(
    X: np.ndarray,
    y: np.ndarray,
    lengths: Optional[List[int]] = None,
    *,
    n_iter: int = 100,
    sticky: bool = True,
    kappa: float = STICKY_KAPPA,
) -> Optional[Any]:
    """
    약지도 초기화 HMM 학습

    Args:
        X: emission features (T, D)
        y: labels (T,)
        lengths: segment lengths (optional)
        n_iter: max iterations
        sticky: if True, use sticky HMM
        kappa: sticky prior strength

    Returns:
        Trained GaussianHMM model or None
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("[ERROR] hmmlearn not installed. Run: pip install hmmlearn")
        return None

    n_states = len(HMM_STATES)

    # 초기 파라미터
    means, covars, transmat, startprob = compute_initial_params(X, y)

    # Sticky HMM: 자기전이에 kappa 가중치 추가
    if sticky:
        for i in range(n_states):
            transmat[i, i] += kappa
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        tol=1e-4,
        random_state=42,
        init_params="",  # 초기값 보호
        params="stmc",   # 모두 학습
        verbose=False,
    )

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    if lengths is None:
        lengths = [len(X)]

    model.fit(X, lengths)

    return model


def hungarian_matching(
    hmm_states: np.ndarray,
    rule_labels: np.ndarray,
    n_hmm_states: int = 6,
) -> Tuple[Dict[int, str], np.ndarray, float]:
    """
    Hungarian algorithm으로 HMM state ↔ rule label 최적 매칭

    Args:
        hmm_states: HMM decoded states (0~5)
        rule_labels: rule-based labels (strings)
        n_hmm_states: number of HMM states

    Returns:
        mapping: {hmm_state_idx: rule_label_name}
        remapped_labels: HMM states remapped to rule labels
        accuracy_after: accuracy after matching
    """
    label_to_idx = {lbl: i for i, lbl in enumerate(HMM_STATES)}

    # Confusion matrix
    confusion = np.zeros((n_hmm_states, len(HMM_STATES)), dtype=int)
    for hs, rl in zip(hmm_states, rule_labels):
        if rl in label_to_idx:
            confusion[hs, label_to_idx[rl]] += 1

    # Hungarian algorithm
    cost_matrix = -confusion
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping
    mapping = {}
    for hmm_idx, label_idx in zip(row_ind, col_ind):
        mapping[hmm_idx] = HMM_STATES[label_idx]

    # Remap
    remapped_labels = np.array([mapping.get(s, 'range') for s in hmm_states])

    # Accuracy
    accuracy = (remapped_labels == rule_labels).mean()

    return mapping, remapped_labels, accuracy


def decode_with_uncertainty(
    model,
    X: np.ndarray,
    mapping: Dict[int, str],
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HMM 디코딩 + 불확실성 기반 range 라벨링

    Args:
        model: trained HMM
        X: emission features
        mapping: Hungarian mapping
        threshold: uncertainty threshold

    Returns:
        decoded_states: (T,) decoded state indices
        decoded_labels: (T,) decoded state labels (including 'range')
        posteriors: (T, n_states) posterior probabilities
    """
    posteriors = model.predict_proba(X)
    max_probs = posteriors.max(axis=1)

    decoded_states = model.predict(X)
    decoded_labels = np.array([mapping.get(s, 'range') for s in decoded_states])

    # 불확실성 높으면 range
    uncertain_mask = max_probs < threshold
    decoded_labels[uncertain_mask] = 'range'

    return decoded_states, decoded_labels, posteriors


def create_posterior_map(
    df_15m: pd.DataFrame,
    features: pd.DataFrame,
    model,
    mapping: Dict[int, str],
    min_lookback: int = 200,
) -> Dict[pd.Timestamp, np.ndarray]:
    """
    15분봉별 posterior 확률 맵 생성

    Args:
        df_15m: 15분봉 DataFrame
        features: emission features DataFrame
        model: trained HMM
        mapping: Hungarian mapping
        min_lookback: 제외할 초기 구간

    Returns:
        {timestamp: posterior_array}
    """
    # Valid mask
    valid_mask = ~features[FEATURE_COLS].isna().any(axis=1).values
    valid_mask[:min_lookback] = False

    posterior_map = {}

    # Sequential decoding (online inference)
    for i, (ts, row) in enumerate(features.iterrows()):
        if not valid_mask[i]:
            continue

        X_i = row[FEATURE_COLS].values.reshape(1, -1)
        try:
            posterior = model.predict_proba(X_i)[0]
            posterior_map[ts] = posterior
        except:
            continue

    return posterior_map
