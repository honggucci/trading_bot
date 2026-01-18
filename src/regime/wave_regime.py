"""
Wave-based Regime Classifier
============================

Hilbert Transform 기반 3-State 레짐 분류기.

상태:
- BULL: falling 위상 (225°~315°) + 강한 진폭 (Mean-Reversion)
- BEAR: rising 위상 (45°~135°) + 강한 진폭
- RANGE: 나머지 (약한 진폭 또는 top/bottom)

특징:
- 룩어헤드 없음 (Causal)
- 15m 실시간 통합 가능
- HMM 대비 단순하고 해석 가능

사용법:
```python
from src.regime.wave_regime import WaveRegimeClassifier

classifier = WaveRegimeClassifier()
regime = classifier.classify(prices)  # 'BULL', 'BEAR', 'RANGE'

# 또는 전체 시리즈 (Causal - 룩어헤드 없음)
regimes = classifier.classify_series_causal(df['close'])
```
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Literal
from dataclasses import dataclass
from scipy import signal as scipy_signal

# Numba JIT (optional - fallback to pure numpy if not available)
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


RegimeType = Literal['BULL', 'BEAR', 'RANGE']


# ============================================================================
# Causal Sliding Window Hilbert Transform
# ============================================================================

def _sliding_hilbert_numpy(
    detrended: np.ndarray,
    window: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window Hilbert Transform (Pure NumPy, Causal)

    각 시점에서 최근 window 데이터만 사용하여 Hilbert 계산.
    룩어헤드 없음.

    Args:
        detrended: 디트렌드된 가격 배열
        window: Hilbert 윈도우 크기

    Returns:
        (phases, amplitudes) - 각 시점의 위상과 진폭
    """
    n = len(detrended)
    phases = np.zeros(n)
    amplitudes = np.zeros(n)

    for i in range(window, n):
        # 최근 window만 사용 (Causal)
        segment = detrended[i - window + 1:i + 1]

        # Hilbert Transform
        analytic = scipy_signal.hilbert(segment)

        # 마지막 값 (현재 시점)
        phases[i] = np.angle(analytic[-1])
        amplitudes[i] = np.abs(analytic[-1])

    return phases, amplitudes


if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def _hilbert_fft_numba(segment: np.ndarray) -> Tuple[float, float]:
        """
        Numba JIT Hilbert Transform (단일 윈도우)

        FFT 기반 Hilbert 구현 (scipy 없이)
        """
        n = len(segment)

        # FFT
        fft_result = np.fft.fft(segment)

        # Hilbert: 음수 주파수 제거, 양수 주파수 2배
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = 1
            h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2

        analytic_fft = fft_result * h
        analytic = np.fft.ifft(analytic_fft)

        # 마지막 값
        last = analytic[-1]
        phase = np.arctan2(last.imag, last.real)
        amplitude = np.sqrt(last.real**2 + last.imag**2)

        return phase, amplitude

    @jit(nopython=True, parallel=True, cache=True)
    def _sliding_hilbert_numba(
        detrended: np.ndarray,
        window: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window Hilbert Transform (Numba JIT, Causal)

        병렬 처리로 빠른 계산.
        """
        n = len(detrended)
        phases = np.zeros(n)
        amplitudes = np.zeros(n)

        for i in prange(window, n):
            segment = detrended[i - window + 1:i + 1].copy()
            phase, amplitude = _hilbert_fft_numba(segment)
            phases[i] = phase
            amplitudes[i] = amplitude

        return phases, amplitudes


def sliding_hilbert(
    detrended: np.ndarray,
    window: int = 64,
    use_numba: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window Hilbert Transform (Causal)

    Numba 사용 가능 시 JIT 버전 사용, 아니면 NumPy 버전.

    Args:
        detrended: 디트렌드된 가격 배열
        window: Hilbert 윈도우 크기
        use_numba: Numba JIT 사용 여부

    Returns:
        (phases, amplitudes) - 각 시점의 위상과 진폭
    """
    if use_numba and HAS_NUMBA:
        return _sliding_hilbert_numba(detrended, window)
    else:
        return _sliding_hilbert_numpy(detrended, window)


@dataclass
class WaveState:
    """파동 상태"""
    phase: float           # 위상 (radians, -π ~ π)
    phase_deg: float       # 위상 (degrees, 0 ~ 360)
    amplitude: float       # 진폭 (정규화)
    amplitude_z: float     # 진폭 z-score
    regime: RegimeType     # 레짐
    position: str          # 'bottom', 'rising', 'top', 'falling'


class WaveRegimeClassifier:
    """
    Hilbert 기반 3-State 레짐 분류기

    원리:
    1. 가격에서 추세 제거 (EMA detrend)
    2. Hilbert Transform으로 위상/진폭 추출
    3. 위상 + 진폭으로 레짐 분류
    """

    def __init__(
        self,
        detrend_period: int = 89,
        hilbert_window: int = 64,
        phase_smooth: int = 5,
        amp_threshold: float = 0.5,
    ):
        """
        Args:
            detrend_period: EMA 디트렌드 기간
            hilbert_window: Hilbert 계산 윈도우 (최소 필요 봉 수)
            phase_smooth: 위상 스무딩 EMA 기간
            amp_threshold: RANGE 판정 임계값 (z-score)
        """
        self.detrend_period = detrend_period
        self.hilbert_window = hilbert_window
        self.phase_smooth = phase_smooth
        self.amp_threshold = amp_threshold

    def _detrend(self, prices: np.ndarray) -> np.ndarray:
        """EMA 기반 디트렌드"""
        if len(prices) < self.detrend_period:
            # 데이터 부족 시 선형 디트렌드
            trend = np.linspace(prices[0], prices[-1], len(prices))
            return prices - trend

        # EMA 계산
        alpha = 2 / (self.detrend_period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return prices - ema

    def _hilbert_causal(self, signal: np.ndarray) -> Tuple[float, float]:
        """
        Causal Hilbert Transform (룩어헤드 없음)

        Returns:
            (phase, amplitude) - 마지막 시점의 값
        """
        if len(signal) < self.hilbert_window:
            return 0.0, 0.0

        # 최근 window만 사용
        window = signal[-self.hilbert_window:]

        # Hilbert Transform
        analytic = scipy_signal.hilbert(window)

        # 마지막 시점
        last = analytic[-1]
        phase = np.angle(last)
        amplitude = np.abs(last)

        return phase, amplitude

    def _phase_to_position(self, phase_deg: float) -> str:
        """위상을 위치로 변환"""
        if 315 <= phase_deg or phase_deg < 45:
            return 'bottom'
        elif 45 <= phase_deg < 135:
            return 'rising'
        elif 135 <= phase_deg < 225:
            return 'top'
        else:  # 225 <= phase_deg < 315
            return 'falling'

    def _classify_regime(
        self,
        phase_deg: float,
        amplitude_z: float
    ) -> RegimeType:
        """
        위상 + 진폭으로 레짐 분류

        Mean-Reversion 관점:
        - falling (225°~315°) = 가격이 EMA 아래 = 반등 예상 = BULL
        - rising (45°~135°) = 가격이 EMA 위 = 하락 예상 = BEAR
        """
        # 진폭이 약하면 RANGE
        if amplitude_z < self.amp_threshold:
            return 'RANGE'

        # Mean-Reversion: 위상 반전 해석
        if 225 <= phase_deg < 315:  # falling = 매수 기회
            return 'BULL'
        elif 45 <= phase_deg < 135:  # rising = 매도 기회
            return 'BEAR'
        else:
            return 'RANGE'

    def classify(self, prices: np.ndarray) -> WaveState:
        """
        단일 시점 레짐 분류

        Args:
            prices: 가격 배열 (최소 hilbert_window 이상)

        Returns:
            WaveState
        """
        prices = np.asarray(prices, dtype=float)

        if len(prices) < self.hilbert_window:
            return WaveState(
                phase=0.0,
                phase_deg=0.0,
                amplitude=0.0,
                amplitude_z=0.0,
                regime='RANGE',
                position='unknown',
            )

        # 디트렌드
        detrended = self._detrend(prices)

        # Hilbert (Causal)
        phase, amplitude = self._hilbert_causal(detrended)

        # 위상을 0~360도로 변환
        phase_deg = (np.degrees(phase) + 360) % 360

        # 진폭 z-score (최근 window 기준)
        recent_amps = []
        for i in range(max(0, len(prices) - 100), len(prices)):
            if i >= self.hilbert_window:
                _, amp = self._hilbert_causal(detrended[:i+1])
                recent_amps.append(amp)

        if len(recent_amps) > 10:
            amp_mean = np.mean(recent_amps)
            amp_std = np.std(recent_amps)
            amplitude_z = (amplitude - amp_mean) / amp_std if amp_std > 0 else 0
        else:
            amplitude_z = 0.0

        # 레짐 분류
        regime = self._classify_regime(phase_deg, amplitude_z)
        position = self._phase_to_position(phase_deg)

        return WaveState(
            phase=phase,
            phase_deg=phase_deg,
            amplitude=amplitude,
            amplitude_z=amplitude_z,
            regime=regime,
            position=position,
        )

    def classify_series(
        self,
        prices: pd.Series,
        min_periods: int = None,
    ) -> pd.DataFrame:
        """
        전체 시리즈 레짐 분류

        Args:
            prices: 가격 시리즈
            min_periods: 최소 시작 기간 (default: hilbert_window)

        Returns:
            DataFrame with columns: phase, phase_deg, amplitude,
                                   amplitude_z, regime, position
        """
        if min_periods is None:
            min_periods = self.hilbert_window

        prices_arr = prices.values
        n = len(prices_arr)

        results = []

        for i in range(n):
            if i < min_periods:
                results.append({
                    'phase': np.nan,
                    'phase_deg': np.nan,
                    'amplitude': np.nan,
                    'amplitude_z': np.nan,
                    'regime': 'RANGE',
                    'position': 'unknown',
                })
            else:
                state = self.classify(prices_arr[:i+1])
                results.append({
                    'phase': state.phase,
                    'phase_deg': state.phase_deg,
                    'amplitude': state.amplitude,
                    'amplitude_z': state.amplitude_z,
                    'regime': state.regime,
                    'position': state.position,
                })

        df = pd.DataFrame(results, index=prices.index)
        return df

    def classify_series_fast(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """
        빠른 시리즈 분류 (벡터화)

        Note: 약간의 룩어헤드 있음 (전체 EMA/std 사용)
              백테스트용으로만 사용
        """
        prices_arr = prices.values.astype(float)
        n = len(prices_arr)

        # 전체 디트렌드
        ema = pd.Series(prices_arr).ewm(span=self.detrend_period).mean().values
        detrended = prices_arr - ema

        # 전체 Hilbert
        analytic = scipy_signal.hilbert(detrended)
        phases = np.angle(analytic)
        amplitudes = np.abs(analytic)

        # 위상 스무딩
        phases_smooth = pd.Series(phases).ewm(span=self.phase_smooth).mean().values

        # 진폭 z-score (롤링)
        amp_series = pd.Series(amplitudes)
        amp_mean = amp_series.rolling(100, min_periods=20).mean()
        amp_std = amp_series.rolling(100, min_periods=20).std()
        amplitude_z = ((amplitudes - amp_mean) / amp_std).fillna(0).values

        # 위상 → 도
        phase_deg = (np.degrees(phases_smooth) + 360) % 360

        # 레짐 분류 (벡터화) - Mean-Reversion
        regimes = np.where(
            amplitude_z < self.amp_threshold,
            'RANGE',
            np.where(
                (phase_deg >= 225) & (phase_deg < 315),  # falling = BULL
                'BULL',
                np.where(
                    (phase_deg >= 45) & (phase_deg < 135),  # rising = BEAR
                    'BEAR',
                    'RANGE'
                )
            )
        )

        # 위치
        positions = np.where(
            (phase_deg >= 315) | (phase_deg < 45), 'bottom',
            np.where(
                (phase_deg >= 45) & (phase_deg < 135), 'rising',
                np.where(
                    (phase_deg >= 135) & (phase_deg < 225), 'top',
                    'falling'
                )
            )
        )

        df = pd.DataFrame({
            'phase': phases_smooth,
            'phase_deg': phase_deg,
            'amplitude': amplitudes,
            'amplitude_z': amplitude_z,
            'regime': regimes,
            'position': positions,
        }, index=prices.index)

        return df

    def classify_series_causal(
        self,
        prices: pd.Series,
        use_numba: bool = True,
    ) -> pd.DataFrame:
        """
        Causal 시리즈 분류 (룩어헤드 없음, 벡터화)

        Sliding window Hilbert를 사용하여 각 시점에서
        미래 데이터를 사용하지 않음.

        Args:
            prices: 가격 시리즈
            use_numba: Numba JIT 사용 여부 (설치 시)

        Returns:
            DataFrame with columns: phase, phase_deg, amplitude,
                                   amplitude_z, regime, position
        """
        prices_arr = prices.values.astype(float)
        n = len(prices_arr)

        # Causal EMA detrend (각 시점에서 과거만 사용)
        alpha = 2 / (self.detrend_period + 1)
        ema = np.zeros(n)
        ema[0] = prices_arr[0]
        for i in range(1, n):
            ema[i] = alpha * prices_arr[i] + (1 - alpha) * ema[i-1]
        detrended = prices_arr - ema

        # Causal Sliding Hilbert
        phases, amplitudes = sliding_hilbert(
            detrended,
            window=self.hilbert_window,
            use_numba=use_numba
        )

        # Causal phase smoothing (EMA)
        phases_smooth = np.zeros(n)
        alpha_phase = 2 / (self.phase_smooth + 1)
        # 위상은 circular 특성 - sin/cos 분리 스무딩
        sin_phase = np.sin(phases)
        cos_phase = np.cos(phases)
        sin_smooth = np.zeros(n)
        cos_smooth = np.zeros(n)
        sin_smooth[0] = sin_phase[0]
        cos_smooth[0] = cos_phase[0]
        for i in range(1, n):
            sin_smooth[i] = alpha_phase * sin_phase[i] + (1 - alpha_phase) * sin_smooth[i-1]
            cos_smooth[i] = alpha_phase * cos_phase[i] + (1 - alpha_phase) * cos_smooth[i-1]
        phases_smooth = np.arctan2(sin_smooth, cos_smooth)

        # Causal amplitude z-score (rolling)
        amp_series = pd.Series(amplitudes)
        amp_mean = amp_series.rolling(100, min_periods=20).mean()
        amp_std = amp_series.rolling(100, min_periods=20).std()
        amplitude_z = ((amplitudes - amp_mean) / amp_std).fillna(0).values

        # 위상 → 도
        phase_deg = (np.degrees(phases_smooth) + 360) % 360

        # 레짐 분류 (벡터화) - Mean-Reversion
        regimes = np.where(
            amplitude_z < self.amp_threshold,
            'RANGE',
            np.where(
                (phase_deg >= 225) & (phase_deg < 315),  # falling = BULL
                'BULL',
                np.where(
                    (phase_deg >= 45) & (phase_deg < 135),  # rising = BEAR
                    'BEAR',
                    'RANGE'
                )
            )
        )

        # 위치
        positions = np.where(
            (phase_deg >= 315) | (phase_deg < 45), 'bottom',
            np.where(
                (phase_deg >= 45) & (phase_deg < 135), 'rising',
                np.where(
                    (phase_deg >= 135) & (phase_deg < 225), 'top',
                    'falling'
                )
            )
        )

        # 초기 window 기간은 RANGE로 설정
        regimes[:self.hilbert_window] = 'RANGE'
        positions[:self.hilbert_window] = 'unknown'

        df = pd.DataFrame({
            'phase': phases_smooth,
            'phase_deg': phase_deg,
            'amplitude': amplitudes,
            'amplitude_z': amplitude_z,
            'regime': regimes,
            'position': positions,
        }, index=prices.index)

        return df


# ============================================================================
# Convenience Functions
# ============================================================================

def get_wave_regime(prices: np.ndarray) -> str:
    """
    간편 함수: 현재 레짐 반환

    Returns:
        'BULL', 'BEAR', 'RANGE'
    """
    classifier = WaveRegimeClassifier()
    state = classifier.classify(prices)
    return state.regime


def get_wave_state(prices: np.ndarray) -> WaveState:
    """
    간편 함수: 현재 파동 상태 반환
    """
    classifier = WaveRegimeClassifier()
    return classifier.classify(prices)
