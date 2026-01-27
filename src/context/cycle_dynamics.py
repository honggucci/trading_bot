"""
Cycle Dynamics Module
- FFT: 지배적 사이클 주기 추출
- Hilbert Transform: 현재 사이클 위상 감지
- 동적 SL/파라미터 조절
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, Dict
import pandas as pd


def compute_hilbert_phase(prices: np.ndarray, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hilbert Transform으로 순간 위상과 진폭 계산

    Args:
        prices: 가격 배열
        detrend: 추세 제거 여부

    Returns:
        (phase, amplitude) - 위상은 -π ~ π, 진폭은 절대값
    """
    if len(prices) < 10:
        return np.zeros(len(prices)), np.zeros(len(prices))

    # 1) 추세 제거 (선택적)
    if detrend:
        # 선형 추세 제거
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        trend = np.polyval(coeffs, x)
        detrended = prices - trend
    else:
        detrended = prices

    # 2) Hilbert Transform
    analytic_signal = signal.hilbert(detrended)

    # 3) 순간 위상 (Instantaneous Phase)
    phase = np.angle(analytic_signal)  # -π ~ π

    # 4) 순간 진폭 (Envelope)
    amplitude = np.abs(analytic_signal)

    return phase, amplitude


def extract_dominant_period_fft(prices: np.ndarray, min_period: int = 5, max_period: int = 200) -> Tuple[float, float]:
    """
    FFT로 지배적 사이클 주기 추출

    Args:
        prices: 가격 배열
        min_period: 최소 주기 (bars)
        max_period: 최대 주기 (bars)

    Returns:
        (dominant_period, power) - 주기(bars)와 파워
    """
    n = len(prices)
    if n < max_period:
        return 0.0, 0.0

    # 1) 추세 제거
    x = np.arange(n)
    coeffs = np.polyfit(x, prices, 1)
    trend = np.polyval(coeffs, x)
    detrended = prices - trend

    # 2) 윈도우 적용 (spectral leakage 감소)
    windowed = detrended * np.hanning(n)

    # 3) FFT
    yf = fft(windowed)
    xf = fftfreq(n, 1)  # 1 bar = 1 unit

    # 4) 파워 스펙트럼 (양의 주파수만)
    positive_freq_idx = np.where(xf > 0)[0]
    freqs = xf[positive_freq_idx]
    power = np.abs(yf[positive_freq_idx]) ** 2

    # 5) 주기 범위 필터링
    periods = 1 / freqs
    valid_idx = (periods >= min_period) & (periods <= max_period)

    if not np.any(valid_idx):
        return 0.0, 0.0

    filtered_periods = periods[valid_idx]
    filtered_power = power[valid_idx]

    # 6) 지배적 주기 (최대 파워)
    max_idx = np.argmax(filtered_power)
    dominant_period = filtered_periods[max_idx]
    dominant_power = filtered_power[max_idx]

    return dominant_period, dominant_power


def get_cycle_state(phase: float) -> str:
    """
    위상에 따른 사이클 상태 분류

    위상 구간:
    - Bottom Zone: -π ~ -π/2 (반등 준비)
    - Rising: -π/2 ~ 0 (상승 중)
    - Top Zone: 0 ~ π/2 (하락 준비)
    - Falling: π/2 ~ π (하락 중)
    """
    if -np.pi <= phase < -np.pi/2:
        return "BOTTOM"  # 바닥권 - SL 넓게
    elif -np.pi/2 <= phase < 0:
        return "RISING"  # 상승 중
    elif 0 <= phase < np.pi/2:
        return "TOP"     # 천장권 - SL 넓게
    else:
        return "FALLING" # 하락 중


def compute_dynamic_sl_mult(
    phase: float,
    base_mult: float = 1.5,
    phase_adjustment: float = 0.5
) -> float:
    """
    위상 기반 동적 SL 배수 계산

    Args:
        phase: 현재 위상 (-π ~ π)
        base_mult: 기본 SL 배수
        phase_adjustment: 위상에 따른 조절 범위

    Returns:
        동적 SL 배수

    로직:
    - 바닥권(BOTTOM): base + adjustment (SL 넓게, 반등 기대)
    - 천장권(TOP): base + adjustment (SL 넓게, 하락 기대)
    - 상승/하락 중: base (표준)
    """
    state = get_cycle_state(phase)

    if state in ["BOTTOM", "TOP"]:
        # 극단 구간에서는 SL 넓게 (노이즈 많음)
        return base_mult + phase_adjustment
    else:
        # 추세 구간에서는 표준 SL
        return base_mult


def compute_dynamic_sl_mult_continuous(
    phase: float,
    base_mult: float = 1.5,
    max_adjustment: float = 0.5
) -> float:
    """
    연속적인 위상 기반 동적 SL 배수 (부드러운 전환)

    cos(phase)를 사용:
    - phase = 0 (천장) → cos = 1 → SL 최대
    - phase = ±π (바닥) → cos = -1 → SL 최대
    - phase = ±π/2 (중간) → cos = 0 → SL 기본

    abs(cos)를 사용하면:
    - 0, ±π에서 최대
    - ±π/2에서 최소
    """
    # |cos(phase)|로 극단 구간에서 최대값
    adjustment = abs(np.cos(phase)) * max_adjustment
    return base_mult + adjustment


class CycleDynamics:
    """사이클 기반 동적 파라미터 관리자"""

    def __init__(
        self,
        lookback: int = 200,        # FFT 분석 윈도우
        min_period: int = 10,       # 최소 사이클 주기 (bars)
        max_period: int = 100,      # 최대 사이클 주기 (bars)
        base_sl_mult: float = 1.5,
        sl_phase_adj: float = 0.5,
        use_continuous: bool = True  # 연속적 조절 사용
    ):
        self.lookback = lookback
        self.min_period = min_period
        self.max_period = max_period
        self.base_sl_mult = base_sl_mult
        self.sl_phase_adj = sl_phase_adj
        self.use_continuous = use_continuous

        # 캐시
        self._last_period = None
        self._last_phase = None

    def analyze(self, prices: np.ndarray) -> Dict:
        """
        가격 데이터 분석

        Returns:
            {
                'dominant_period': float,  # 지배적 주기 (bars)
                'current_phase': float,    # 현재 위상 (-π ~ π)
                'cycle_state': str,        # BOTTOM/RISING/TOP/FALLING
                'dynamic_sl_mult': float,  # 동적 SL 배수
                'amplitude': float,        # 현재 진폭
            }
        """
        if len(prices) < self.lookback:
            # 데이터 부족 시 기본값
            return {
                'dominant_period': 0,
                'current_phase': 0,
                'cycle_state': 'UNKNOWN',
                'dynamic_sl_mult': self.base_sl_mult,
                'amplitude': 0,
            }

        # 최근 lookback 데이터만 사용
        recent = prices[-self.lookback:]

        # 1) FFT로 지배적 주기 추출
        period, power = extract_dominant_period_fft(
            recent, self.min_period, self.max_period
        )

        # 2) Hilbert로 위상/진폭
        phase_arr, amp_arr = compute_hilbert_phase(recent)
        current_phase = phase_arr[-1]
        current_amp = amp_arr[-1]

        # 3) 사이클 상태
        state = get_cycle_state(current_phase)

        # 4) 동적 SL 배수
        if self.use_continuous:
            sl_mult = compute_dynamic_sl_mult_continuous(
                current_phase, self.base_sl_mult, self.sl_phase_adj
            )
        else:
            sl_mult = compute_dynamic_sl_mult(
                current_phase, self.base_sl_mult, self.sl_phase_adj
            )

        # 캐시 업데이트
        self._last_period = period
        self._last_phase = current_phase

        return {
            'dominant_period': period,
            'current_phase': current_phase,
            'phase_degrees': np.degrees(current_phase),
            'cycle_state': state,
            'dynamic_sl_mult': sl_mult,
            'amplitude': current_amp,
            'power': power,
        }

    def get_sl_multiplier(self, prices: np.ndarray) -> float:
        """간편 인터페이스: SL 배수만 반환"""
        result = self.analyze(prices)
        return result['dynamic_sl_mult']


# ============================================================
# 테스트 함수
# ============================================================

def test_cycle_dynamics():
    """간단한 테스트"""
    # 합성 사이클 데이터 생성
    np.random.seed(42)
    n = 500
    t = np.arange(n)

    # 50-bar 주기 사인파 + 노이즈 + 추세
    period = 50
    price = 100 + 0.01 * t + 5 * np.sin(2 * np.pi * t / period) + np.random.randn(n) * 0.5

    # 분석
    cd = CycleDynamics(lookback=200, min_period=20, max_period=100)
    result = cd.analyze(price)

    print("=== Cycle Dynamics Test ===")
    print(f"True Period: {period}")
    print(f"Detected Period: {result['dominant_period']:.1f}")
    print(f"Current Phase: {result['phase_degrees']:.1f}°")
    print(f"Cycle State: {result['cycle_state']}")
    print(f"Dynamic SL Mult: {result['dynamic_sl_mult']:.3f}")
    print(f"Amplitude: {result['amplitude']:.3f}")

    return result


if __name__ == "__main__":
    test_cycle_dynamics()
