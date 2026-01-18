"""
Regime Classification Module
============================

시장 레짐 분류 모듈.

- WaveRegimeClassifier: Hilbert 기반 3-state (BULL/BEAR/RANGE)
- HMM: 기존 6-state Wyckoff 기반 (deprecated)
"""
from src.regime.wave_regime import (
    WaveRegimeClassifier,
    WaveState,
    RegimeType,
    get_wave_regime,
    get_wave_state,
)

__all__ = [
    'WaveRegimeClassifier',
    'WaveState',
    'RegimeType',
    'get_wave_regime',
    'get_wave_state',
]
