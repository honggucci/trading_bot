"""
Cycle Target Calculator - 수학적 사이클 고점 예측
=================================================

FFT, Boltzmann, Black-Scholes, Hilbert 이론을 조합하여
저점 대비 고점 목표가를 계산하는 모듈.

핵심 이론:
1. FFT (Fast Fourier Transform) - 사이클 주기 추출
2. Boltzmann Distribution - 가격 분포 확률 모델
3. Black-Scholes - 변동성 기반 목표가 범위
4. Hilbert Transform - 사이클 위상 및 진폭

4th Cycle Target 결과:
- Conservative: $126,000 (Fib 7.0)
- Moderate: $200,000 (Fib 11.236)
- Optimistic: $254,000 (Fib 14.326)

사용법:
```python
from src.context.cycle_target import CycleTargetCalculator

calc = CycleTargetCalculator()

# 과거 사이클 데이터로 다음 사이클 고점 예측
targets = calc.predict_cycle_high(
    cycle_lows=[3120, 15500],      # 과거 사이클 저점들
    cycle_highs=[20650, 69000],    # 과거 사이클 고점들
    current_cycle_low=15500,       # 현재 사이클 저점
)

print(f"Conservative: ${targets.conservative:,.0f}")
print(f"Moderate: ${targets.moderate:,.0f}")
print(f"Optimistic: ${targets.optimistic:,.0f}")
```
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft, ifft, fftfreq
    from scipy.stats import norm, lognorm
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CycleTargets:
    """사이클 목표가 결과"""
    conservative: float    # 보수적 목표가 (하위 25%)
    moderate: float        # 중간 목표가 (중앙값)
    optimistic: float      # 낙관적 목표가 (상위 75%)

    # 상세 정보
    fft_dominant_period: int           # FFT 지배 주기 (bars)
    fft_amplitude_ratio: float         # FFT 진폭 비율
    boltzmann_mean: float              # Boltzmann 평균
    boltzmann_std: float               # Boltzmann 표준편차
    bs_implied_vol: float              # Black-Scholes 내재 변동성
    hilbert_phase: float               # Hilbert 현재 위상
    hilbert_cycle_position: str        # Hilbert 사이클 위치

    # Fib 레벨
    conservative_fib: float
    moderate_fib: float
    optimistic_fib: float

    # 신뢰도
    confidence: float


@dataclass
class CycleData:
    """사이클 데이터"""
    cycle_num: int
    low: float
    high: float
    ratio: float  # high / low
    duration_days: Optional[int] = None


# ============================================================================
# Cycle Target Calculator
# ============================================================================

class CycleTargetCalculator:
    """
    수학적 사이클 고점 예측기

    4가지 이론 조합:
    1. FFT → 사이클 주기 + 진폭 패턴 추출
    2. Boltzmann → 가격 분포의 통계적 모델
    3. Black-Scholes → 변동성 기반 범위 계산
    4. Hilbert → 현재 사이클 위치 파악
    """

    # 1W Fib 앵커 (고정값)
    FIB_0 = 3120      # 2018년 저점
    FIB_1 = 20650     # 2017/18년 고점
    FIB_RANGE = FIB_1 - FIB_0  # 17530

    def __init__(
        self,
        fib_0: float = 3120,
        fib_1: float = 20650,
    ):
        """
        Args:
            fib_0: Fib 0 기준 가격 (저점)
            fib_1: Fib 1 기준 가격 (고점)
        """
        self.fib_0 = fib_0
        self.fib_1 = fib_1
        self.fib_range = fib_1 - fib_0

    # ========================================================================
    # Main: Predict Cycle High
    # ========================================================================

    def predict_cycle_high(
        self,
        cycle_lows: List[float],
        cycle_highs: List[float],
        current_cycle_low: float,
        weights: Tuple[float, float, float, float] = (0.3, 0.25, 0.25, 0.2),
    ) -> CycleTargets:
        """
        다음 사이클 고점 예측

        Args:
            cycle_lows: 과거 사이클 저점들 [cycle1_low, cycle2_low, ...]
            cycle_highs: 과거 사이클 고점들 [cycle1_high, cycle2_high, ...]
            current_cycle_low: 현재 사이클 저점
            weights: (FFT, Boltzmann, Black-Scholes, Hilbert) 가중치

        Returns:
            CycleTargets
        """
        if len(cycle_lows) != len(cycle_highs):
            raise ValueError("cycle_lows와 cycle_highs 길이가 같아야 합니다")

        if len(cycle_lows) < 2:
            raise ValueError("최소 2개 이상의 사이클 데이터 필요")

        # 사이클 비율 계산 (high / low)
        ratios = [h / l for h, l in zip(cycle_highs, cycle_lows)]

        # 1. FFT 분석 - 비율의 주기성
        fft_result = self._fft_analysis(ratios)

        # 2. Boltzmann 분포 - 비율의 통계적 분포
        boltz_result = self._boltzmann_analysis(ratios)

        # 3. Black-Scholes - 변동성 기반 범위
        bs_result = self._black_scholes_analysis(cycle_highs, cycle_lows)

        # 4. Hilbert - 현재 사이클 위상
        hilbert_result = self._hilbert_analysis(cycle_highs)

        # 가중 평균으로 최종 비율 계산
        w_fft, w_boltz, w_bs, w_hilbert = weights

        # 각 모델의 예측 비율
        fft_ratio = fft_result['predicted_ratio']
        boltz_ratios = boltz_result['predicted_ratios']  # (conservative, moderate, optimistic)
        bs_ratios = bs_result['predicted_ratios']
        hilbert_ratio = hilbert_result['predicted_ratio']

        # 가중 평균
        conservative_ratio = (
            w_fft * fft_ratio * 0.8 +
            w_boltz * boltz_ratios[0] +
            w_bs * bs_ratios[0] +
            w_hilbert * hilbert_ratio * 0.85
        )

        moderate_ratio = (
            w_fft * fft_ratio +
            w_boltz * boltz_ratios[1] +
            w_bs * bs_ratios[1] +
            w_hilbert * hilbert_ratio
        )

        optimistic_ratio = (
            w_fft * fft_ratio * 1.2 +
            w_boltz * boltz_ratios[2] +
            w_bs * bs_ratios[2] +
            w_hilbert * hilbert_ratio * 1.15
        )

        # 목표가 계산
        conservative = current_cycle_low * conservative_ratio
        moderate = current_cycle_low * moderate_ratio
        optimistic = current_cycle_low * optimistic_ratio

        # Fib 레벨 변환
        conservative_fib = self._price_to_fib(conservative)
        moderate_fib = self._price_to_fib(moderate)
        optimistic_fib = self._price_to_fib(optimistic)

        # 신뢰도 계산
        confidence = self._calc_confidence(ratios, fft_result, boltz_result)

        return CycleTargets(
            conservative=conservative,
            moderate=moderate,
            optimistic=optimistic,

            fft_dominant_period=fft_result['dominant_period'],
            fft_amplitude_ratio=fft_result['amplitude_ratio'],
            boltzmann_mean=boltz_result['mean'],
            boltzmann_std=boltz_result['std'],
            bs_implied_vol=bs_result['implied_vol'],
            hilbert_phase=hilbert_result['phase'],
            hilbert_cycle_position=hilbert_result['position'],

            conservative_fib=conservative_fib,
            moderate_fib=moderate_fib,
            optimistic_fib=optimistic_fib,

            confidence=confidence,
        )

    # ========================================================================
    # 1. FFT Analysis - 사이클 주기 추출
    # ========================================================================

    def _fft_analysis(self, ratios: List[float]) -> Dict[str, Any]:
        """
        FFT로 사이클 비율의 주기성 분석

        원리:
        - 과거 사이클 비율들의 패턴 추출
        - 지배 주기로 다음 비율 외삽
        """
        ratios = np.array(ratios)
        n = len(ratios)

        if n < 3:
            # 데이터 부족 시 단순 평균
            mean_ratio = np.mean(ratios)
            return {
                'predicted_ratio': mean_ratio * 1.1,  # 10% 성장 가정
                'dominant_period': n,
                'amplitude_ratio': 0.0,
            }

        # 디트렌드 (선형 추세 제거)
        trend = np.linspace(ratios[0], ratios[-1], n)
        detrended = ratios - trend

        # FFT
        fft_result = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(n)

        # 양의 주파수만
        pos_mask = freqs > 0
        if not np.any(pos_mask):
            return {
                'predicted_ratio': np.mean(ratios),
                'dominant_period': n,
                'amplitude_ratio': 0.0,
            }

        pos_freqs = freqs[pos_mask]
        pos_fft = fft_result[pos_mask]
        power = np.abs(pos_fft) ** 2

        # 지배 주파수
        dominant_idx = np.argmax(power)
        dominant_freq = pos_freqs[dominant_idx]
        dominant_period = int(round(1 / dominant_freq)) if dominant_freq > 0 else n

        # 진폭 비율
        total_power = np.sum(power)
        amplitude_ratio = power[dominant_idx] / total_power if total_power > 0 else 0

        # 다음 포인트 외삽
        next_t = n
        amplitude = np.abs(pos_fft[dominant_idx]) * 2 / n
        phase = np.angle(pos_fft[dominant_idx])

        # 하모닉 합성으로 다음 값 예측
        extrapolated_detrend = amplitude * np.cos(2 * np.pi * dominant_freq * next_t + phase)

        # 트렌드 외삽
        trend_slope = (ratios[-1] - ratios[0]) / (n - 1) if n > 1 else 0
        extrapolated_trend = ratios[-1] + trend_slope

        predicted_ratio = extrapolated_trend + extrapolated_detrend

        # 비율이 너무 낮으면 최소값 적용
        predicted_ratio = max(predicted_ratio, min(ratios) * 0.8)

        return {
            'predicted_ratio': predicted_ratio,
            'dominant_period': dominant_period,
            'amplitude_ratio': amplitude_ratio,
        }

    # ========================================================================
    # 2. Boltzmann Distribution - 가격 분포 모델
    # ========================================================================

    def _boltzmann_analysis(self, ratios: List[float]) -> Dict[str, Any]:
        """
        Boltzmann 분포 기반 비율 예측

        원리:
        - 사이클 비율을 에너지 레벨로 모델링
        - 통계역학적 분포로 확률 계산
        - P(E) ∝ exp(-E / kT)
        """
        ratios = np.array(ratios)

        # 로그 변환 (비율은 로그정규 분포 따름)
        log_ratios = np.log(ratios)

        # 평균과 표준편차
        mean_log = np.mean(log_ratios)
        std_log = np.std(log_ratios) if len(log_ratios) > 1 else 0.3

        # 표준편차가 너무 작으면 최소값 적용
        std_log = max(std_log, 0.1)

        # Boltzmann 분포 파라미터
        # kT (온도 파라미터) = 표준편차^2
        kT = std_log ** 2

        # 분위수 계산
        if HAS_SCIPY:
            dist = lognorm(s=std_log, scale=np.exp(mean_log))
            conservative = dist.ppf(0.25)   # 25%
            moderate = dist.ppf(0.50)       # 50% (중앙값)
            optimistic = dist.ppf(0.75)     # 75%
        else:
            # Fallback: 단순 계산
            conservative = np.exp(mean_log - std_log)
            moderate = np.exp(mean_log)
            optimistic = np.exp(mean_log + std_log)

        return {
            'mean': np.exp(mean_log),
            'std': std_log,
            'kT': kT,
            'predicted_ratios': (conservative, moderate, optimistic),
        }

    # ========================================================================
    # 3. Black-Scholes - 변동성 기반 범위
    # ========================================================================

    def _black_scholes_analysis(
        self,
        highs: List[float],
        lows: List[float],
    ) -> Dict[str, Any]:
        """
        Black-Scholes 모델 기반 목표가 범위

        원리:
        - 과거 변동성으로 내재 변동성 추정
        - 로그정규 분포 가정
        - d1, d2 계산으로 확률 범위
        """
        highs = np.array(highs)
        lows = np.array(lows)
        ratios = highs / lows

        # 로그 수익률 변동성
        log_returns = np.diff(np.log(highs))
        if len(log_returns) > 0:
            vol = np.std(log_returns) * np.sqrt(4)  # 연간화 (4 사이클/16년)
        else:
            vol = 1.0  # 기본 100% 변동성

        # 최소/최대 변동성 제한
        vol = np.clip(vol, 0.5, 2.0)

        # Black-Scholes 스타일 범위 계산
        # S0 = 현재 비율의 평균
        S0 = np.mean(ratios)
        T = 1.0  # 1 사이클

        # d1, d2 계산
        if HAS_SCIPY:
            # 정규분포 분위수
            z_conservative = norm.ppf(0.25)  # -0.674
            z_moderate = 0
            z_optimistic = norm.ppf(0.75)    # +0.674

            # GBM 기반 예측
            conservative = S0 * np.exp((vol ** 2 / 2) * T + vol * np.sqrt(T) * z_conservative)
            moderate = S0 * np.exp((vol ** 2 / 2) * T)
            optimistic = S0 * np.exp((vol ** 2 / 2) * T + vol * np.sqrt(T) * z_optimistic)
        else:
            # Fallback
            conservative = S0 * 0.8
            moderate = S0
            optimistic = S0 * 1.2

        return {
            'implied_vol': vol,
            'predicted_ratios': (conservative, moderate, optimistic),
        }

    # ========================================================================
    # 4. Hilbert Transform - 사이클 위상
    # ========================================================================

    def _hilbert_analysis(self, highs: List[float]) -> Dict[str, Any]:
        """
        Hilbert Transform으로 사이클 위상 분석

        원리:
        - 순시 위상 및 진폭 계산
        - 현재 사이클 위치 파악 (상승/하락/바닥/천장)
        - 다음 고점까지의 예상 비율
        """
        highs = np.array(highs)
        n = len(highs)

        if n < 3 or not HAS_SCIPY:
            return {
                'phase': 0.0,
                'position': 'unknown',
                'predicted_ratio': np.mean(highs[1:] / highs[:-1]) if n > 1 else 1.0,
            }

        # 디트렌드
        detrended = highs - np.linspace(highs[0], highs[-1], n)

        # Hilbert Transform
        analytic = scipy_signal.hilbert(detrended)

        # 순시 위상
        phase = np.angle(analytic[-1])

        # 순시 진폭
        amplitude = np.abs(analytic[-1])
        mean_amplitude = np.mean(np.abs(analytic))

        # 위상 → 위치 매핑
        normalized_phase = (phase + np.pi) % (2 * np.pi)

        if normalized_phase < np.pi / 4 or normalized_phase >= 7 * np.pi / 4:
            position = 'bottom'
            ratio_mult = 1.15  # 바닥에서 상승 기대
        elif normalized_phase < 3 * np.pi / 4:
            position = 'rising'
            ratio_mult = 1.05
        elif normalized_phase < 5 * np.pi / 4:
            position = 'top'
            ratio_mult = 0.95  # 천장에서 하락 기대
        else:
            position = 'falling'
            ratio_mult = 1.0

        # 진폭 기반 예측 비율
        amp_ratio = amplitude / mean_amplitude if mean_amplitude > 0 else 1.0

        # 과거 비율의 평균에 위상 조정
        if n > 1:
            growth_ratios = highs[1:] / highs[:-1]
            base_ratio = np.mean(growth_ratios)
        else:
            base_ratio = 1.0

        predicted_ratio = base_ratio * ratio_mult * (1 + 0.1 * (amp_ratio - 1))

        return {
            'phase': phase,
            'position': position,
            'amplitude_ratio': amp_ratio,
            'predicted_ratio': max(predicted_ratio, 0.5),  # 최소 0.5배
        }

    # ========================================================================
    # Utility Functions
    # ========================================================================

    def _price_to_fib(self, price: float) -> float:
        """가격 → Fib 레벨"""
        return (price - self.fib_0) / self.fib_range

    def _fib_to_price(self, fib_level: float) -> float:
        """Fib 레벨 → 가격"""
        return self.fib_0 + (fib_level * self.fib_range)

    def _calc_confidence(
        self,
        ratios: List[float],
        fft_result: Dict,
        boltz_result: Dict,
    ) -> float:
        """예측 신뢰도 계산"""
        # 데이터 포인트 수 기반
        n_factor = min(len(ratios) / 5, 1.0)

        # FFT 주기성 강도
        fft_factor = fft_result['amplitude_ratio']

        # Boltzmann 분산 (낮을수록 안정적)
        std_factor = max(0, 1 - boltz_result['std'])

        # 가중 평균
        confidence = 0.4 * n_factor + 0.3 * fft_factor + 0.3 * std_factor

        return min(max(confidence, 0.1), 0.95)


# ============================================================================
# Convenience Functions
# ============================================================================

def predict_btc_cycle_high(
    current_cycle_low: float = 15500,
) -> CycleTargets:
    """
    BTC 4th Cycle 고점 예측 (편의 함수)

    과거 사이클 데이터:
    - 1st: Low $3,120 → High $20,650 (x6.6)
    - 2nd: Low $3,200 → High $13,800 (x4.3)
    - 3rd: Low $15,500 → High $69,000 (x4.5)

    Args:
        current_cycle_low: 현재 사이클 저점 (default: $15,500)

    Returns:
        CycleTargets
    """
    calc = CycleTargetCalculator()

    return calc.predict_cycle_high(
        cycle_lows=[3120, 3200, 15500],
        cycle_highs=[20650, 13800, 69000],
        current_cycle_low=current_cycle_low,
    )


def get_fib_targets() -> Dict[str, float]:
    """
    주요 Fib 레벨 목표가 반환

    Returns:
        {
            "fib_7.0_conservative": 126000,
            "fib_11.236_moderate": 200088,
            "fib_14.326_optimistic": 254236,
        }
    """
    calc = CycleTargetCalculator()

    return {
        "fib_7.0_conservative": calc._fib_to_price(7.0),
        "fib_11.236_moderate": calc._fib_to_price(11.236),
        "fib_14.326_optimistic": calc._fib_to_price(14.326),
    }


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BTC 4th Cycle Target Prediction")
    print("=" * 60)

    # 예측 실행
    targets = predict_btc_cycle_high(current_cycle_low=15500)

    print(f"\nCycle Low: $15,500")
    print(f"\n--- Targets ---")
    print(f"Conservative: ${targets.conservative:,.0f} (Fib {targets.conservative_fib:.3f})")
    print(f"Moderate:     ${targets.moderate:,.0f} (Fib {targets.moderate_fib:.3f})")
    print(f"Optimistic:   ${targets.optimistic:,.0f} (Fib {targets.optimistic_fib:.3f})")

    print(f"\n--- Analysis Details ---")
    print(f"FFT Dominant Period: {targets.fft_dominant_period} cycles")
    print(f"FFT Amplitude Ratio: {targets.fft_amplitude_ratio:.2%}")
    print(f"Boltzmann Mean: {targets.boltzmann_mean:.2f}x")
    print(f"Boltzmann Std: {targets.boltzmann_std:.2f}")
    print(f"Black-Scholes Implied Vol: {targets.bs_implied_vol:.2%}")
    print(f"Hilbert Phase: {targets.hilbert_phase:.2f} rad")
    print(f"Hilbert Position: {targets.hilbert_cycle_position}")
    print(f"\nConfidence: {targets.confidence:.2%}")

    print("\n--- Reference Fib Levels ---")
    fib_targets = get_fib_targets()
    for key, price in fib_targets.items():
        print(f"{key}: ${price:,.0f}")

    print("=" * 60)
