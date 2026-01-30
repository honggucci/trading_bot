"""
Wyckoff Module - Phase Detection & Box Engine
==============================================

Wyckoff 기반 시장 구조 분석 모듈.

Components:
- types.py: Theta, BacktestConfig 등 설정 dataclass
- indicators.py: ATR, RSI, ADX 등 기술적 지표
- box.py: Wyckoff 박스 엔진
- phases.py: Wyckoff 페이즈 감지
- spectral.py: 스펙트럴 분석 (FFT, Wavelet, Convolution)

Origin: WPCN backtester (wpcn-backtester-cli-noflask)
"""
from .types import Theta, get_default_theta
from .indicators import atr, rsi, zscore, adx, stoch_rsi, true_range
from .box import box_engine_freeze, BoxState
from .phases import (
    detect_wyckoff_phase,
    get_direction_ratios,
    WyckoffPhase,
    AccumulationSignal,
    DIRECTION_RATIOS,
)
from .spectral import (
    SpectralAnalyzer,
    SpectralResult,
    HarmonicComponent,
    PredictionResult,
    harmonic_predict,
    wavelet_denoise,
    analyze_cycles,
)

__all__ = [
    # Types
    'Theta',
    'get_default_theta',

    # Indicators
    'atr',
    'rsi',
    'zscore',
    'adx',
    'stoch_rsi',
    'true_range',

    # Box Engine
    'box_engine_freeze',
    'BoxState',

    # Phases
    'detect_wyckoff_phase',
    'get_direction_ratios',
    'WyckoffPhase',
    'AccumulationSignal',
    'DIRECTION_RATIOS',

    # Spectral Analysis
    'SpectralAnalyzer',
    'SpectralResult',
    'HarmonicComponent',
    'PredictionResult',
    'harmonic_predict',
    'wavelet_denoise',
    'analyze_cycles',
]
