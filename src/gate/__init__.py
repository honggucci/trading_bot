"""Gate Layer - HMM Entry Filter"""
import pickle
from pathlib import Path
from typing import Optional

from .hmm_entry_gate import (
    HMMGateConfig,
    GateDecision,
    HMMEntryGate,
    HMM_STATES,
    VAR5_BY_STATE,
)


def load_hmm_gate(
    models_dir: Optional[Path] = None,
    cfg: Optional[HMMGateConfig] = None,
) -> HMMEntryGate:
    """
    저장된 pkl 파일에서 HMMEntryGate 로드

    Args:
        models_dir: 모델 디렉토리 (기본: src/gate/../models)
        cfg: Gate 설정 (기본값 사용)

    Returns:
        HMMEntryGate 인스턴스
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent / "models"

    if cfg is None:
        cfg = HMMGateConfig()

    # Load posterior_map
    # SECURITY NOTE: pickle.load can execute arbitrary code if the file is tampered.
    # Only load from trusted local sources. Never load user-provided pickle files.
    posterior_path = models_dir / "posterior_map.pkl"
    if not posterior_path.exists():
        raise FileNotFoundError(f"posterior_map.pkl not found at {posterior_path}")

    with open(posterior_path, 'rb') as f:
        posterior_map = pickle.load(f)

    # Load features_df
    features_path = models_dir / "features_df.pkl"
    if not features_path.exists():
        raise FileNotFoundError(f"features_df.pkl not found at {features_path}")

    with open(features_path, 'rb') as f:
        features_df = pickle.load(f)

    return HMMEntryGate(
        posterior_map=posterior_map,
        features_df=features_df,
        cfg=cfg,
    )


__all__ = [
    'HMMGateConfig',
    'GateDecision',
    'HMMEntryGate',
    'HMM_STATES',
    'VAR5_BY_STATE',
    'load_hmm_gate',
]
