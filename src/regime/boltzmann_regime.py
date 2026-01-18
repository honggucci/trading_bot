"""
Boltzmann Regime Classifier
===========================

통계역학 기반 3-State 레짐 분류기.

원리:
- 시장을 열역학 시스템으로 모델링
- 각 상태(BULL/BEAR/RANGE)의 에너지 함수 정의
- Boltzmann 분포로 상태 확률 계산: P(state) = exp(-E/T) / Z

장점:
- 완전히 Causal (미래 데이터 불필요)
- 온라인 업데이트 가능
- 해석 가능한 에너지 함수
- Temperature로 불확실성 반영

사용법:
```python
from src.regime.boltzmann_regime import BoltzmannRegimeClassifier

classifier = BoltzmannRegimeClassifier()
regime, probs = classifier.classify(prices)  # 'BULL', 'BEAR', 'RANGE'

# 전체 시리즈
result_df = classifier.classify_series(df['close'])
```
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Literal, Optional
from dataclasses import dataclass


RegimeType = Literal['BULL', 'BEAR', 'RANGE']


@dataclass
class BoltzmannState:
    """Boltzmann 상태"""
    regime: RegimeType
    prob_bull: float
    prob_bear: float
    prob_range: float
    energy_bull: float
    energy_bear: float
    energy_range: float
    temperature: float
    z_score: float
    momentum: float


class BoltzmannRegimeClassifier:
    """
    Boltzmann 분포 기반 레짐 분류기

    에너지 함수 (Mean-Reversion 관점):
    - E_bull = -z_score - momentum_z  (낮을수록 BULL 확률 높음)
    - E_bear = +z_score + momentum_z  (낮을수록 BEAR 확률 높음)
    - E_range = |z_score| + |momentum_z|  (중립일수록 RANGE)

    Temperature:
    - 변동성 기반 (높을수록 불확실성)
    """

    def __init__(
        self,
        ma_period: int = 89,
        momentum_period: int = 14,
        vol_period: int = 20,
        base_temperature: float = 1.0,
        vol_temp_scale: float = 2.0,
    ):
        """
        Args:
            ma_period: 이동평균 기간 (z-score 계산용)
            momentum_period: 모멘텀 계산 기간
            vol_period: 변동성 계산 기간
            base_temperature: 기본 온도
            vol_temp_scale: 변동성→온도 스케일
        """
        self.ma_period = ma_period
        self.momentum_period = momentum_period
        self.vol_period = vol_period
        self.base_temperature = base_temperature
        self.vol_temp_scale = vol_temp_scale

    def _compute_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Causal 피처 계산 (과거 데이터만 사용)
        """
        n = len(prices)

        if n < self.ma_period:
            return {
                'z_score': 0.0,
                'momentum': 0.0,
                'momentum_z': 0.0,
                'volatility': 1.0,
                'temperature': self.base_temperature,
            }

        # EMA (Causal)
        alpha = 2 / (self.ma_period + 1)
        ema = prices[0]
        for i in range(1, n):
            ema = alpha * prices[i] + (1 - alpha) * ema

        # 표준편차 (최근 vol_period)
        recent_prices = prices[-min(self.vol_period, n):]
        std = np.std(recent_prices)
        if std == 0:
            std = 1e-8

        # Z-score: (현재가 - EMA) / std
        z_score = (prices[-1] - ema) / std

        # 모멘텀: 최근 momentum_period 수익률
        if n >= self.momentum_period:
            momentum = (prices[-1] / prices[-self.momentum_period] - 1) * 100
        else:
            momentum = 0.0

        # 모멘텀 z-score
        if n >= self.momentum_period * 2:
            returns = []
            for i in range(self.momentum_period, min(n, self.momentum_period * 5)):
                ret = (prices[-i+self.momentum_period-1] / prices[-i-1] - 1) * 100
                returns.append(ret)
            if len(returns) > 1:
                mom_mean = np.mean(returns)
                mom_std = np.std(returns)
                momentum_z = (momentum - mom_mean) / mom_std if mom_std > 0 else 0
            else:
                momentum_z = 0.0
        else:
            momentum_z = 0.0

        # 변동성 (ATR-like)
        if n >= 2:
            recent = prices[-min(self.vol_period, n):]
            returns = np.diff(recent) / recent[:-1]
            volatility = np.std(returns) * np.sqrt(252 * 24 * 4)  # 연간화 (15m 기준)
        else:
            volatility = 0.1

        # Temperature = base + vol * scale
        temperature = self.base_temperature + volatility * self.vol_temp_scale

        return {
            'z_score': z_score,
            'momentum': momentum,
            'momentum_z': np.clip(momentum_z, -3, 3),  # 극단값 제한
            'volatility': volatility,
            'temperature': max(temperature, 0.1),  # 최소값
        }

    def _compute_energies(
        self,
        z_score: float,
        momentum_z: float,
    ) -> Tuple[float, float, float]:
        """
        각 상태의 에너지 계산

        Mean-Reversion 관점:
        - BULL: z_score가 낮을수록(oversold), momentum이 낮을수록 에너지 낮음
        - BEAR: z_score가 높을수록(overbought), momentum이 높을수록 에너지 낮음
        - RANGE: 중립일수록 에너지 낮음
        """
        # 에너지 함수 (낮을수록 해당 상태일 확률 높음)
        e_bull = -z_score - momentum_z * 0.5  # oversold + 하락모멘텀 = 반등 기회
        e_bear = +z_score + momentum_z * 0.5  # overbought + 상승모멘텀 = 하락 기회
        e_range = abs(z_score) * 0.5 + abs(momentum_z) * 0.3  # 중립일수록 낮음

        return e_bull, e_bear, e_range

    def _boltzmann_probs(
        self,
        e_bull: float,
        e_bear: float,
        e_range: float,
        temperature: float,
    ) -> Tuple[float, float, float]:
        """
        Boltzmann 분포로 확률 계산

        P(state) = exp(-E / T) / Z
        """
        # exp(-E/T) 계산 (오버플로우 방지)
        max_e = max(-e_bull, -e_bear, -e_range)

        exp_bull = np.exp((-e_bull - max_e) / temperature)
        exp_bear = np.exp((-e_bear - max_e) / temperature)
        exp_range = np.exp((-e_range - max_e) / temperature)

        # 분배 함수 Z
        Z = exp_bull + exp_bear + exp_range

        # 확률
        p_bull = exp_bull / Z
        p_bear = exp_bear / Z
        p_range = exp_range / Z

        return p_bull, p_bear, p_range

    def classify(self, prices: np.ndarray) -> BoltzmannState:
        """
        단일 시점 분류

        Args:
            prices: 가격 배열 (최근 데이터)

        Returns:
            BoltzmannState
        """
        prices = np.asarray(prices, dtype=float)

        # 피처 계산
        features = self._compute_features(prices)

        # 에너지 계산
        e_bull, e_bear, e_range = self._compute_energies(
            features['z_score'],
            features['momentum_z'],
        )

        # Boltzmann 확률
        p_bull, p_bear, p_range = self._boltzmann_probs(
            e_bull, e_bear, e_range,
            features['temperature'],
        )

        # 최고 확률 상태
        probs = {'BULL': p_bull, 'BEAR': p_bear, 'RANGE': p_range}
        regime = max(probs, key=probs.get)

        return BoltzmannState(
            regime=regime,
            prob_bull=p_bull,
            prob_bear=p_bear,
            prob_range=p_range,
            energy_bull=e_bull,
            energy_bear=e_bear,
            energy_range=e_range,
            temperature=features['temperature'],
            z_score=features['z_score'],
            momentum=features['momentum'],
        )

    def classify_series(
        self,
        prices: pd.Series,
        min_periods: int = None,
    ) -> pd.DataFrame:
        """
        전체 시리즈 분류 (Causal - 룩어헤드 없음)

        Args:
            prices: 가격 시리즈
            min_periods: 최소 시작 기간

        Returns:
            DataFrame with regime, probabilities, energies, etc.
        """
        if min_periods is None:
            min_periods = self.ma_period

        prices_arr = prices.values.astype(float)
        n = len(prices_arr)

        # 결과 배열
        regimes = np.empty(n, dtype=object)
        p_bulls = np.zeros(n)
        p_bears = np.zeros(n)
        p_ranges = np.zeros(n)
        e_bulls = np.zeros(n)
        e_bears = np.zeros(n)
        e_ranges = np.zeros(n)
        temperatures = np.zeros(n)
        z_scores = np.zeros(n)
        momentums = np.zeros(n)

        for i in range(n):
            if i < min_periods:
                regimes[i] = 'RANGE'
                p_ranges[i] = 1.0
                continue

            # 현재 시점까지의 데이터만 사용 (Causal)
            state = self.classify(prices_arr[:i+1])

            regimes[i] = state.regime
            p_bulls[i] = state.prob_bull
            p_bears[i] = state.prob_bear
            p_ranges[i] = state.prob_range
            e_bulls[i] = state.energy_bull
            e_bears[i] = state.energy_bear
            e_ranges[i] = state.energy_range
            temperatures[i] = state.temperature
            z_scores[i] = state.z_score
            momentums[i] = state.momentum

        return pd.DataFrame({
            'regime': regimes,
            'prob_bull': p_bulls,
            'prob_bear': p_bears,
            'prob_range': p_ranges,
            'energy_bull': e_bulls,
            'energy_bear': e_bears,
            'energy_range': e_ranges,
            'temperature': temperatures,
            'z_score': z_scores,
            'momentum': momentums,
        }, index=prices.index)

    def classify_series_fast(
        self,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """
        빠른 벡터화 버전 (Causal)

        EMA, rolling std 등을 벡터화하여 빠르게 계산
        """
        prices_arr = prices.values.astype(float)
        n = len(prices_arr)

        # Causal EMA
        alpha = 2 / (self.ma_period + 1)
        ema = np.zeros(n)
        ema[0] = prices_arr[0]
        for i in range(1, n):
            ema[i] = alpha * prices_arr[i] + (1 - alpha) * ema[i-1]

        # Rolling std
        std = pd.Series(prices_arr).rolling(self.vol_period, min_periods=2).std().fillna(1).values

        # Z-score
        z_scores = (prices_arr - ema) / np.maximum(std, 1e-8)

        # Momentum
        momentum = np.zeros(n)
        for i in range(self.momentum_period, n):
            momentum[i] = (prices_arr[i] / prices_arr[i - self.momentum_period] - 1) * 100

        # Momentum z-score (rolling)
        mom_series = pd.Series(momentum)
        mom_mean = mom_series.rolling(self.momentum_period * 3, min_periods=self.momentum_period).mean()
        mom_std = mom_series.rolling(self.momentum_period * 3, min_periods=self.momentum_period).std()
        momentum_z = ((momentum - mom_mean) / mom_std.replace(0, 1)).fillna(0).values
        momentum_z = np.clip(momentum_z, -3, 3)

        # Volatility
        returns = np.zeros(n)
        returns[1:] = np.diff(prices_arr) / prices_arr[:-1]
        vol = pd.Series(returns).rolling(self.vol_period, min_periods=2).std().fillna(0.01).values
        vol = vol * np.sqrt(252 * 24 * 4)

        # Temperature
        temperature = self.base_temperature + vol * self.vol_temp_scale
        temperature = np.maximum(temperature, 0.1)

        # Energies (vectorized)
        e_bull = -z_scores - momentum_z * 0.5
        e_bear = +z_scores + momentum_z * 0.5
        e_range = np.abs(z_scores) * 0.5 + np.abs(momentum_z) * 0.3

        # Boltzmann probabilities (vectorized)
        max_neg_e = np.maximum(-e_bull, np.maximum(-e_bear, -e_range))

        exp_bull = np.exp((-e_bull - max_neg_e) / temperature)
        exp_bear = np.exp((-e_bear - max_neg_e) / temperature)
        exp_range = np.exp((-e_range - max_neg_e) / temperature)

        Z = exp_bull + exp_bear + exp_range

        p_bull = exp_bull / Z
        p_bear = exp_bear / Z
        p_range = exp_range / Z

        # Regime = max probability
        probs = np.vstack([p_bull, p_bear, p_range])
        regime_idx = np.argmax(probs, axis=0)
        regime_map = {0: 'BULL', 1: 'BEAR', 2: 'RANGE'}
        regimes = np.array([regime_map[i] for i in regime_idx])

        # 초기 기간
        regimes[:self.ma_period] = 'RANGE'

        return pd.DataFrame({
            'regime': regimes,
            'prob_bull': p_bull,
            'prob_bear': p_bear,
            'prob_range': p_range,
            'energy_bull': e_bull,
            'energy_bear': e_bear,
            'energy_range': e_range,
            'temperature': temperature,
            'z_score': z_scores,
            'momentum': momentum,
        }, index=prices.index)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_boltzmann_regime(prices: np.ndarray) -> str:
    """현재 레짐 반환"""
    classifier = BoltzmannRegimeClassifier()
    state = classifier.classify(prices)
    return state.regime


def get_boltzmann_state(prices: np.ndarray) -> BoltzmannState:
    """현재 Boltzmann 상태 반환"""
    classifier = BoltzmannRegimeClassifier()
    return classifier.classify(prices)
