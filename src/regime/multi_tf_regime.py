"""
Multi-TF Regime Aggregator (MODE82)
====================================

ZigZag(1W, 1D) + ProbGate(4H, 1H) 합의 기반 레짐 분류.

핵심 원칙:
- ZigZag direction → prior 확률 변환
- ProbGate p_bull → uncertainty로 수축
- Weighted sum → regime score
- Hysteresis → BULL/RANGE/BEAR 분류

Usage:
```python
from src.regime.multi_tf_regime import MultiTFRegimeAggregator, RegimeConfig

config = RegimeConfig()
aggregator = MultiTFRegimeAggregator(config)

state = aggregator.update(
    zz_1w_direction="up",
    zz_1d_direction="up",
    pg_4h_p_bull=0.65,
    pg_4h_T=1.2,
    pg_1h_p_bull=0.58,
    pg_1h_T=1.5,
    timestamp=current_ts
)

print(state.regime)  # "BULL", "RANGE", or "BEAR"
print(state.score)   # 0.0 ~ 1.0
```
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
import pandas as pd
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RegimeConfig:
    """
    Multi-TF Regime Aggregator 설정

    15m 매매용 (Execution TF = 15m)
    - Context TFs: 1H, 15m, 5m
    """
    # === ZigZag 가중치 (15m 매매용: 1H, 15m) ===
    weight_1h_zz: float = 0.45   # 1H ZigZag (대세)
    weight_15m_zz: float = 0.25  # 15m ZigZag (중기)

    # === ProbGate 가중치 (5m) ===
    weight_5m_pg: float = 0.30   # 5m ProbGate (단기 모멘텀)

    # === ZigZag → Prior 변환 ===
    prior_1h_up: float = 0.80    # 1H up → 0.80
    prior_1h_down: float = 0.20  # 1H down → 0.20
    prior_15m_up: float = 0.70   # 15m up → 0.70
    prior_15m_down: float = 0.30 # 15m down → 0.30
    prior_unknown: float = 0.50

    # === Hysteresis 임계값 ===
    bull_enter: float = 0.60
    bull_exit: float = 0.50
    bear_enter: float = 0.40
    bear_exit: float = 0.50

    # === EMA Smoothing (optional) ===
    score_ema_span: int = 3  # 0이면 smoothing 안함


# =============================================================================
# State
# =============================================================================

@dataclass
class RegimeState:
    """
    레짐 상태
    """
    regime: str = "RANGE"  # "BULL" | "RANGE" | "BEAR"
    score: float = 0.5     # 0.0 ~ 1.0
    score_raw: float = 0.5 # smoothing 전 score

    # 컴포넌트별 상세
    components: Dict = field(default_factory=dict)

    # 메타데이터
    last_change_ts: Optional[pd.Timestamp] = None
    bars_in_regime: int = 0

    def to_dict(self) -> dict:
        """Dictionary 변환"""
        return {
            'regime': self.regime,
            'score': self.score,
            'score_raw': self.score_raw,
            'components': self.components,
            'last_change_ts': self.last_change_ts,
            'bars_in_regime': self.bars_in_regime,
        }


# =============================================================================
# Core Functions
# =============================================================================

def zz_to_prior(direction: str, tf: str, config: RegimeConfig) -> float:
    """
    ZigZag direction → prior 확률 변환

    Args:
        direction: "up" | "down" | "unknown"
        tf: "1h" | "15m" (15m 매매용)
        config: RegimeConfig

    Returns:
        prior 확률 (0.0 ~ 1.0)
    """
    if direction == "unknown":
        return config.prior_unknown

    if tf == "1h":
        return config.prior_1h_up if direction == "up" else config.prior_1h_down
    elif tf == "15m":
        return config.prior_15m_up if direction == "up" else config.prior_15m_down
    else:
        return config.prior_unknown


def shrink_to_half(p: float, u: float) -> float:
    """
    Uncertainty에 따라 확률을 0.5로 수축

    Args:
        p: 원래 확률 (0.0 ~ 1.0)
        u: uncertainty (0.0 = 확신, 1.0 = 불확신)

    Returns:
        수축된 확률

    Examples:
        shrink_to_half(0.80, 0.0) → 0.80 (확신)
        shrink_to_half(0.80, 0.5) → 0.65 (절반 수축)
        shrink_to_half(0.80, 1.0) → 0.50 (완전 불확신)
    """
    u = np.clip(u, 0.0, 1.0)
    return 0.5 + (p - 0.5) * (1.0 - u)


def compute_uncertainty(T: float, T_ref: float = 1.0, T_range: float = 1.0) -> float:
    """
    Temperature → Uncertainty 변환

    ProbGate의 Temperature가 높을수록 불확실

    Args:
        T: 현재 Temperature
        T_ref: 기준 Temperature (보통 1.0)
        T_range: 정규화 범위

    Returns:
        uncertainty (0.0 ~ 1.0)
    """
    return np.clip((T - T_ref) / T_range, 0.0, 1.0)


def compute_regime_score(
    zz_1h_direction: str,
    zz_15m_direction: str,
    pg_5m_p_bull: float,
    pg_5m_T: float,
    config: RegimeConfig
) -> tuple:
    """
    Multi-TF 레짐 스코어 계산 (15m 매매용)

    Args:
        zz_1h_direction: 1H ZigZag direction
        zz_15m_direction: 15m ZigZag direction
        pg_5m_p_bull: 5m ProbGate p_bull
        pg_5m_T: 5m ProbGate Temperature

    Returns:
        (score, components_dict)
    """
    # 1. ZigZag → Prior
    p_1h = zz_to_prior(zz_1h_direction, "1h", config)
    p_15m = zz_to_prior(zz_15m_direction, "15m", config)

    # 2. ProbGate → Shrink by uncertainty
    u_5m = compute_uncertainty(pg_5m_T)
    p_5m = shrink_to_half(pg_5m_p_bull, u_5m)

    # 3. Weighted sum
    score = (
        config.weight_1h_zz * p_1h +
        config.weight_15m_zz * p_15m +
        config.weight_5m_pg * p_5m
    )

    # 4. 컴포넌트 상세
    components = {
        'zz_1h': {'direction': zz_1h_direction, 'prior': p_1h, 'weight': config.weight_1h_zz},
        'zz_15m': {'direction': zz_15m_direction, 'prior': p_15m, 'weight': config.weight_15m_zz},
        'pg_5m': {'p_bull': pg_5m_p_bull, 'T': pg_5m_T, 'u': u_5m, 'p_shrunk': p_5m, 'weight': config.weight_5m_pg},
    }

    return score, components


def update_regime_with_hysteresis(
    current_regime: str,
    score: float,
    config: RegimeConfig
) -> str:
    """
    Hysteresis를 적용한 레짐 전환

    Args:
        current_regime: 현재 레짐 ("BULL" | "RANGE" | "BEAR")
        score: 새로운 스코어 (0.0 ~ 1.0)
        config: RegimeConfig

    Returns:
        새로운 레짐
    """
    if current_regime == "BULL":
        # Bull에서 나가려면 bull_exit 아래로 떨어져야 함
        if score < config.bull_exit:
            if score <= config.bear_exit:
                return "BEAR"
            else:
                return "RANGE"
        return "BULL"

    elif current_regime == "BEAR":
        # Bear에서 나가려면 bear_exit 위로 올라가야 함
        if score > config.bear_exit:
            if score >= config.bull_enter:
                return "BULL"
            else:
                return "RANGE"
        return "BEAR"

    else:  # RANGE
        # Range에서 Bull/Bear로 진입하려면 enter 임계값 필요
        if score >= config.bull_enter:
            return "BULL"
        elif score <= config.bear_enter:
            return "BEAR"
        return "RANGE"


# =============================================================================
# Aggregator Class
# =============================================================================

class MultiTFRegimeAggregator:
    """
    Multi-TF Regime Aggregator

    ZigZag(1W, 1D) + ProbGate(4H, 1H) 합의 기반 레짐 분류
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.state = RegimeState()
        self._score_history = []  # EMA 계산용

    def reset(self):
        """상태 초기화"""
        self.state = RegimeState()
        self._score_history = []

    def update(
        self,
        zz_1h_direction: str = "unknown",
        zz_15m_direction: str = "unknown",
        pg_5m_p_bull: float = 0.5,
        pg_5m_T: float = 1.0,
        timestamp: Optional[pd.Timestamp] = None
    ) -> RegimeState:
        """
        레짐 상태 업데이트 (15m 매매용)

        Args:
            zz_1h_direction: 1H ZigZag direction ("up" | "down" | "unknown")
            zz_15m_direction: 15m ZigZag direction
            pg_5m_p_bull: 5m ProbGate p_bull (0.0 ~ 1.0)
            pg_5m_T: 5m ProbGate Temperature
            timestamp: 현재 시각

        Returns:
            업데이트된 RegimeState
        """
        # 1. Raw score 계산
        score_raw, components = compute_regime_score(
            zz_1h_direction, zz_15m_direction,
            pg_5m_p_bull, pg_5m_T,
            self.config
        )

        # 2. EMA smoothing (optional)
        if self.config.score_ema_span > 0:
            self._score_history.append(score_raw)
            # Keep only recent values for EMA
            max_history = self.config.score_ema_span * 3
            if len(self._score_history) > max_history:
                self._score_history = self._score_history[-max_history:]

            # Simple EMA
            alpha = 2.0 / (self.config.score_ema_span + 1.0)
            score = score_raw
            for i, s in enumerate(self._score_history[:-1]):
                score = alpha * score_raw + (1 - alpha) * score
        else:
            score = score_raw

        # 3. Hysteresis 적용하여 레짐 결정
        prev_regime = self.state.regime
        new_regime = update_regime_with_hysteresis(prev_regime, score, self.config)

        # 4. 상태 업데이트
        if new_regime != prev_regime:
            self.state.last_change_ts = timestamp
            self.state.bars_in_regime = 1
        else:
            self.state.bars_in_regime += 1

        self.state.regime = new_regime
        self.state.score = score
        self.state.score_raw = score_raw
        self.state.components = components

        return self.state

    def get_regime(self) -> str:
        """현재 레짐 반환"""
        return self.state.regime

    def get_score(self) -> float:
        """현재 스코어 반환"""
        return self.state.score

    def is_bull(self) -> bool:
        """Bull 레짐 여부"""
        return self.state.regime == "BULL"

    def is_bear(self) -> bool:
        """Bear 레짐 여부"""
        return self.state.regime == "BEAR"

    def is_range(self) -> bool:
        """Range 레짐 여부"""
        return self.state.regime == "RANGE"


# =============================================================================
# Utility Functions
# =============================================================================

def regime_to_risk_mult(regime: str) -> float:
    """
    레짐 → 리스크 배수 변환

    Args:
        regime: "BULL" | "RANGE" | "BEAR"

    Returns:
        리스크 배수 (0.0 ~ 1.0)
    """
    if regime == "BULL":
        return 1.0
    elif regime == "RANGE":
        return 0.5
    else:  # BEAR
        return 0.3


def regime_to_stoch_threshold(regime: str) -> float:
    """
    레짐 → StochRSI 임계값 변환

    Args:
        regime: "BULL" | "RANGE" | "BEAR"

    Returns:
        StochRSI oversold 임계값
    """
    if regime == "BULL":
        return 30.0
    elif regime == "RANGE":
        return 20.0
    else:  # BEAR
        return 10.0


def can_long_in_regime(
    regime: str,
    stoch_rsi: float,
    reclaim_confirmed: bool = False
) -> tuple:
    """
    레짐에서 롱 허용 여부 판단

    Args:
        regime: "BULL" | "RANGE" | "BEAR"
        stoch_rsi: 현재 StochRSI 값
        reclaim_confirmed: 1H reclaim 확인 여부 (Bear에서 필요)

    Returns:
        (allowed, reason)
    """
    threshold = regime_to_stoch_threshold(regime)

    if regime == "BULL":
        if stoch_rsi <= threshold:
            return True, f"BULL: StochRSI {stoch_rsi:.1f} <= {threshold}"
        return False, f"BULL: StochRSI {stoch_rsi:.1f} > {threshold}"

    elif regime == "RANGE":
        if stoch_rsi <= threshold:
            return True, f"RANGE: StochRSI {stoch_rsi:.1f} <= {threshold}"
        return False, f"RANGE: StochRSI {stoch_rsi:.1f} > {threshold}"

    else:  # BEAR
        if stoch_rsi <= threshold and reclaim_confirmed:
            return True, f"BEAR: StochRSI {stoch_rsi:.1f} <= {threshold} AND reclaim OK"
        elif stoch_rsi > threshold:
            return False, f"BEAR: StochRSI {stoch_rsi:.1f} > {threshold}"
        else:
            return False, f"BEAR: StochRSI OK but reclaim NOT confirmed"
