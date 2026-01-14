"""
Spectral Analysis Module
========================

통신이론 기반 스펙트럴 분석 도구.

Components:
- Fourier 다중 하모닉 분해 및 합성
- Wavelet 다중 스케일 분해
- Convolution 필터링

Origin: 통신이론 (푸리에 급수, 합성곱, 웨이블릿)

사용 예시:
```python
from src.wyckoff.spectral import SpectralAnalyzer

analyzer = SpectralAnalyzer()

# 다중 하모닉 분해
harmonics = analyzer.decompose_harmonics(prices, n_harmonics=5)

# 미래 예측 (하모닉 외삽)
predicted = analyzer.harmonic_extrapolate(prices, predict_bars=10)

# Wavelet 디노이징
denoised = analyzer.wavelet_denoise(prices)

# 사이클 합성
synthesized = analyzer.synthesize_cycles(prices)
```
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft, ifft, fftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HarmonicComponent:
    """단일 하모닉 성분"""
    frequency: float      # 주파수 (cycles per bar)
    period: int           # 주기 (bars)
    amplitude: float      # 진폭
    phase: float          # 위상 (radians)
    power: float          # 파워 (amplitude^2)
    power_ratio: float    # 전체 파워 대비 비율


@dataclass
class SpectralResult:
    """스펙트럴 분석 결과"""
    dominant_period: int
    dominant_power_ratio: float
    harmonics: List[HarmonicComponent]
    hilbert_phase: float
    hilbert_position: str  # 'bottom', 'rising', 'top', 'falling'
    entropy: float

    def get_cycle_lengths(self) -> List[int]:
        """상위 하모닉들의 주기 반환"""
        return [h.period for h in self.harmonics]


@dataclass
class PredictionResult:
    """예측 결과"""
    predicted_prices: np.ndarray
    confidence: float
    dominant_cycle: int
    extrapolation_bars: int


# ============================================================================
# Spectral Analyzer
# ============================================================================

class SpectralAnalyzer:
    """
    통신이론 기반 스펙트럴 분석기

    오실로스코프 비유:
    - 기본파 (f) → 장기 추세
    - 2차 하모닉 (2f) → 중기 사이클
    - 3차 하모닉 (3f) → 단기 사이클
    - 고주파 노이즈 → 틱 노이즈 (필터링)
    """

    def __init__(self, max_harmonics: int = 10, min_period: int = 5):
        """
        Args:
            max_harmonics: 분석할 최대 하모닉 수
            min_period: 최소 주기 (bars)
        """
        self.max_harmonics = max_harmonics
        self.min_period = min_period

    # ========================================================================
    # Core: Fourier Decomposition
    # ========================================================================

    def decompose_harmonics(
        self,
        prices: np.ndarray,
        n_harmonics: int = 5,
        detrend: bool = True,
    ) -> List[HarmonicComponent]:
        """
        다중 하모닉 분해 (푸리에 급수)

        f(t) = Σ A_n * cos(2πf_n*t + φ_n)

        Args:
            prices: 가격 배열
            n_harmonics: 추출할 하모닉 수
            detrend: 선형 추세 제거 여부

        Returns:
            상위 n개 HarmonicComponent 리스트
        """
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) < self.min_period * 2:
            return []

        # 디트렌드
        if detrend:
            trend = np.linspace(prices[0], prices[-1], len(prices))
            signal = prices - trend
        else:
            signal = prices.copy()

        # FFT
        n = len(signal)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n)

        # 양의 주파수만 (대칭이므로)
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_fft = fft_result[pos_mask]

        # 최소 주기 필터
        period_mask = (1 / pos_freqs) >= self.min_period
        pos_freqs = pos_freqs[period_mask]
        pos_fft = pos_fft[period_mask]

        if len(pos_fft) == 0:
            return []

        # 파워 계산
        power = np.abs(pos_fft) ** 2
        total_power = np.sum(power)

        # 상위 n개 선택
        top_indices = np.argsort(power)[-n_harmonics:][::-1]

        harmonics = []
        for idx in top_indices:
            freq = pos_freqs[idx]
            amp = np.abs(pos_fft[idx]) * 2 / n  # 정규화
            phase = np.angle(pos_fft[idx])
            pwr = power[idx]

            harmonics.append(HarmonicComponent(
                frequency=freq,
                period=int(round(1 / freq)) if freq > 0 else n,
                amplitude=amp,
                phase=phase,
                power=pwr,
                power_ratio=pwr / total_power if total_power > 0 else 0,
            ))

        return harmonics

    def synthesize_from_harmonics(
        self,
        harmonics: List[HarmonicComponent],
        n_points: int,
        include_trend: bool = False,
        trend_slope: float = 0.0,
        trend_intercept: float = 0.0,
    ) -> np.ndarray:
        """
        하모닉 성분들로부터 신호 재합성

        Args:
            harmonics: HarmonicComponent 리스트
            n_points: 출력 길이
            include_trend: 트렌드 포함 여부
            trend_slope: 트렌드 기울기
            trend_intercept: 트렌드 절편

        Returns:
            합성된 신호
        """
        t = np.arange(n_points)
        signal = np.zeros(n_points)

        for h in harmonics:
            # f(t) = A * cos(2πft + φ)
            signal += h.amplitude * np.cos(2 * np.pi * h.frequency * t + h.phase)

        if include_trend:
            signal += trend_slope * t + trend_intercept

        return signal

    # ========================================================================
    # Core: Harmonic Extrapolation (Prediction)
    # ========================================================================

    def harmonic_extrapolate(
        self,
        prices: np.ndarray,
        predict_bars: int = 10,
        n_harmonics: int = 5,
        confidence_decay: float = 0.9,
    ) -> PredictionResult:
        """
        하모닉 외삽을 통한 미래 가격 예측

        원리:
        1. 과거 데이터에서 하모닉 분해
        2. 각 하모닉을 미래로 외삽
        3. 외삽된 하모닉들을 재합성

        Args:
            prices: 가격 배열
            predict_bars: 예측할 봉 수
            n_harmonics: 사용할 하모닉 수
            confidence_decay: 봉당 신뢰도 감소율

        Returns:
            PredictionResult
        """
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]
        n = len(prices)

        if n < self.min_period * 2:
            return PredictionResult(
                predicted_prices=np.full(predict_bars, prices[-1] if len(prices) > 0 else 0),
                confidence=0.0,
                dominant_cycle=0,
                extrapolation_bars=predict_bars,
            )

        # 트렌드 추출
        trend_slope = (prices[-1] - prices[0]) / (n - 1) if n > 1 else 0
        trend_intercept = prices[0]
        trend = trend_slope * np.arange(n) + trend_intercept
        detrended = prices - trend

        # 하모닉 분해
        harmonics = self.decompose_harmonics(detrended, n_harmonics=n_harmonics, detrend=False)

        if not harmonics:
            # 하모닉 없으면 트렌드만으로 예측
            future_t = np.arange(n, n + predict_bars)
            predicted = trend_slope * future_t + trend_intercept
            return PredictionResult(
                predicted_prices=predicted,
                confidence=0.3,
                dominant_cycle=0,
                extrapolation_bars=predict_bars,
            )

        # 미래 시점에서 하모닉 합성
        future_t = np.arange(n, n + predict_bars)
        predicted_detrended = np.zeros(predict_bars)

        for h in harmonics:
            predicted_detrended += h.amplitude * np.cos(
                2 * np.pi * h.frequency * future_t + h.phase
            )

        # 트렌드 추가
        future_trend = trend_slope * future_t + trend_intercept
        predicted = predicted_detrended + future_trend

        # 신뢰도 계산 (지배력 기반 + 시간 감쇠)
        dominant_power_ratio = harmonics[0].power_ratio if harmonics else 0
        base_confidence = min(0.8, dominant_power_ratio * 2)  # 최대 80%
        avg_confidence = base_confidence * (confidence_decay ** (predict_bars / 2))

        return PredictionResult(
            predicted_prices=predicted,
            confidence=avg_confidence,
            dominant_cycle=harmonics[0].period if harmonics else 0,
            extrapolation_bars=predict_bars,
        )

    # ========================================================================
    # Wavelet Analysis
    # ========================================================================

    def wavelet_decompose(
        self,
        prices: np.ndarray,
        wavelet: str = 'db4',
        levels: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Wavelet 다중 스케일 분해

        Args:
            prices: 가격 배열
            wavelet: 웨이블릿 종류 ('db4', 'haar', 'sym5', etc.)
            levels: 분해 레벨 수

        Returns:
            {
                'approximation': 저주파 성분 (장기 추세),
                'details': [고주파 성분들] (단기→장기 순),
                'coefficients': 원본 계수들
            }
        """
        if not HAS_PYWT:
            return {
                'approximation': prices.copy(),
                'details': [],
                'coefficients': [prices.copy()],
            }

        prices = np.asarray(prices, dtype=float)

        # 다중 레벨 분해
        coeffs = pywt.wavedec(prices, wavelet, level=levels)

        # 각 레벨 재구성
        details = []
        for i in range(1, len(coeffs)):
            # 해당 레벨만 유지하고 나머지 0으로
            detail_coeffs = [np.zeros_like(c) for c in coeffs]
            detail_coeffs[i] = coeffs[i]
            detail = pywt.waverec(detail_coeffs, wavelet)[:len(prices)]
            details.append(detail)

        # 근사 (저주파)
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        approximation = pywt.waverec(approx_coeffs, wavelet)[:len(prices)]

        return {
            'approximation': approximation,
            'details': details,
            'coefficients': coeffs,
        }

    def wavelet_denoise(
        self,
        prices: np.ndarray,
        wavelet: str = 'db4',
        levels: int = 4,
        noise_reduction: float = 0.5,
    ) -> np.ndarray:
        """
        Wavelet 기반 노이즈 제거

        고주파 성분(노이즈)을 감쇠시켜 깨끗한 신호 추출

        Args:
            prices: 가격 배열
            wavelet: 웨이블릿 종류
            levels: 분해 레벨
            noise_reduction: 노이즈 감쇠율 (0=유지, 1=제거)

        Returns:
            디노이즈된 가격
        """
        if not HAS_PYWT:
            # Fallback: 단순 이동평균
            window = max(3, len(prices) // 20)
            return pd.Series(prices).rolling(window, center=True, min_periods=1).mean().values

        prices = np.asarray(prices, dtype=float)
        coeffs = pywt.wavedec(prices, wavelet, level=levels)

        # 고주파 성분 감쇠 (레벨이 높을수록 더 많이 감쇠)
        for i in range(1, len(coeffs)):
            decay = noise_reduction * (i / len(coeffs))  # 점진적 감쇠
            coeffs[i] = coeffs[i] * (1 - decay)

        # 재합성
        denoised = pywt.waverec(coeffs, wavelet)
        return denoised[:len(prices)]

    # ========================================================================
    # Convolution Filtering
    # ========================================================================

    def convolve_filter(
        self,
        prices: np.ndarray,
        kernel: str = 'gaussian',
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        합성곱 필터링

        Args:
            prices: 가격 배열
            kernel: 커널 종류 ('gaussian', 'uniform', 'triangular', 'custom')
            kernel_size: 커널 크기
            sigma: 가우시안 표준편차

        Returns:
            필터링된 가격
        """
        prices = np.asarray(prices, dtype=float)

        # 커널 생성
        if kernel == 'gaussian':
            x = np.arange(kernel_size) - kernel_size // 2
            k = np.exp(-x**2 / (2 * sigma**2))
        elif kernel == 'uniform':
            k = np.ones(kernel_size)
        elif kernel == 'triangular':
            k = np.concatenate([
                np.arange(1, kernel_size // 2 + 2),
                np.arange(kernel_size // 2, 0, -1)
            ])[:kernel_size]
        else:
            k = np.ones(kernel_size)

        # 정규화
        k = k / np.sum(k)

        # 합성곱 (same 모드로 길이 유지)
        filtered = np.convolve(prices, k, mode='same')

        return filtered

    def bandpass_filter(
        self,
        prices: np.ndarray,
        low_period: int = 10,
        high_period: int = 50,
    ) -> np.ndarray:
        """
        대역통과 필터 (특정 주기 범위만 통과)

        Args:
            prices: 가격 배열
            low_period: 최소 주기 (고주파 차단)
            high_period: 최대 주기 (저주파 차단)

        Returns:
            필터링된 가격
        """
        prices = np.asarray(prices, dtype=float)
        n = len(prices)

        if n < low_period * 2:
            return prices.copy()

        # FFT
        fft_result = np.fft.fft(prices)
        freqs = np.fft.fftfreq(n)

        # 주기를 주파수로 변환
        low_freq = 1 / high_period if high_period > 0 else 0
        high_freq = 1 / low_period if low_period > 0 else 0.5

        # 대역통과 마스크
        mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        mask[0] = True  # DC 성분 유지

        # 필터링
        filtered_fft = fft_result * mask

        # IFFT
        filtered = np.fft.ifft(filtered_fft).real

        return filtered

    # ========================================================================
    # Full Analysis
    # ========================================================================

    def analyze(
        self,
        prices: np.ndarray,
        n_harmonics: int = 5,
    ) -> SpectralResult:
        """
        종합 스펙트럴 분석

        Args:
            prices: 가격 배열
            n_harmonics: 분석할 하모닉 수

        Returns:
            SpectralResult
        """
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        # 하모닉 분해
        harmonics = self.decompose_harmonics(prices, n_harmonics=n_harmonics)

        # 지배 주기
        dominant_period = harmonics[0].period if harmonics else 20
        dominant_power = harmonics[0].power_ratio if harmonics else 0

        # Hilbert 위상
        hilbert_phase, hilbert_position = self._compute_hilbert_phase(prices)

        # 엔트로피
        entropy = self._compute_entropy(prices)

        return SpectralResult(
            dominant_period=dominant_period,
            dominant_power_ratio=dominant_power,
            harmonics=harmonics,
            hilbert_phase=hilbert_phase,
            hilbert_position=hilbert_position,
            entropy=entropy,
        )

    def _compute_hilbert_phase(self, prices: np.ndarray) -> Tuple[float, str]:
        """Hilbert Transform 위상 계산"""
        if not HAS_SCIPY or len(prices) < 10:
            return 0.0, 'unknown'

        # 디트렌드
        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))

        # Hilbert Transform
        analytic = scipy_signal.hilbert(detrended)
        phase = np.angle(analytic[-1])

        # 위상 → 위치
        normalized = (phase + np.pi) % (2 * np.pi)
        if normalized < np.pi / 4 or normalized >= 7 * np.pi / 4:
            position = 'bottom'
        elif normalized < 3 * np.pi / 4:
            position = 'rising'
        elif normalized < 5 * np.pi / 4:
            position = 'top'
        else:
            position = 'falling'

        return phase, position

    def _compute_entropy(self, prices: np.ndarray, bins: int = 20) -> float:
        """Shannon Entropy 계산"""
        returns = np.diff(prices) / prices[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) < 10:
            return 0.5

        try:
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist[hist > 0]
            hist = hist / np.sum(hist)

            if HAS_SCIPY:
                from scipy.stats import entropy
                return entropy(hist) / np.log(bins)
            else:
                return -np.sum(hist * np.log(hist)) / np.log(bins)
        except:
            return 0.5


# ============================================================================
# Convenience Functions
# ============================================================================

def harmonic_predict(
    prices: np.ndarray,
    predict_bars: int = 10,
    n_harmonics: int = 5,
) -> PredictionResult:
    """
    간편 함수: 하모닉 외삽 예측

    사용 예시:
    ```python
    result = harmonic_predict(prices, predict_bars=10)
    print(f"예측: {result.predicted_prices}")
    print(f"신뢰도: {result.confidence:.2%}")
    ```
    """
    analyzer = SpectralAnalyzer()
    return analyzer.harmonic_extrapolate(prices, predict_bars, n_harmonics)


def wavelet_denoise(
    prices: np.ndarray,
    noise_reduction: float = 0.5,
) -> np.ndarray:
    """
    간편 함수: Wavelet 디노이징

    사용 예시:
    ```python
    clean_prices = wavelet_denoise(prices, noise_reduction=0.7)
    ```
    """
    analyzer = SpectralAnalyzer()
    return analyzer.wavelet_denoise(prices, noise_reduction=noise_reduction)


def analyze_cycles(
    prices: np.ndarray,
    n_harmonics: int = 5,
) -> SpectralResult:
    """
    간편 함수: 사이클 분석

    사용 예시:
    ```python
    result = analyze_cycles(prices)
    print(f"지배 주기: {result.dominant_period}봉")
    print(f"현재 위치: {result.hilbert_position}")
    ```
    """
    analyzer = SpectralAnalyzer()
    return analyzer.analyze(prices, n_harmonics)
