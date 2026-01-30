"""
HMM Entry Gate Module (v2.7.0)
===============================

WPCN에서 복사 + trading_bot 독립 버전

Entry 시점에서 HMM 필터를 적용하는 모듈.
기존 apply_hmm_filter()의 후처리 방식을 대체.

핵심 원칙:
- HMM permit/transition cooldown/soft sizing을 Entry 결정 시점에 적용
- permit 미충족이면 트레이드 자체를 만들지 않음
- sizing은 entry에서 결정된 size로 포지션에 고정

v2.7.0 변경:
- 1H Hilbert 레짐 필터 추가 (IC=+0.027, causal)
- Long: BEAR 레짐에서 block
- Short: BULL 레짐에서 block (옵션)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import pandas as pd
import numpy as np

# Hilbert Regime Classifier
from src.regime.wave_regime import WaveRegimeClassifier

# HMM States
HMM_STATES = [
    'accumulation',
    're_accumulation',
    'distribution',
    're_distribution',
    'markup',
    'markdown',
]
STATE_TO_IDX = {s: i for i, s in enumerate(HMM_STATES)}
IDX_TO_STATE = {i: s for i, s in enumerate(HMM_STATES)}

# 상태별 VaR5% (OOS 검증 완료)
VAR5_BY_STATE = {
    'accumulation': -5.56,
    're_accumulation': -6.91,
    'distribution': -5.63,
    're_distribution': -8.85,
    'markup': -7.16,
    'markdown': -10.52,
    'range': -7.0,
}


@dataclass
class HMMGateConfig:
    """HMM Entry Gate 설정"""
    # Transition Cooldown
    transition_delta: float = 0.40  # v1.2 완화된 값
    cooldown_bars: int = 1

    # Soft Sizing
    var_target: float = -5.0
    size_min: float = 0.25
    size_max: float = 1.25

    # Short Permit (markdown + 강한 하락추세에서만 숏)
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # Long Permit (허용 상태에서만 롱)
    long_permit_enabled: bool = True
    long_permit_states: Tuple[str, ...] = ('markup', 'accumulation', 're_accumulation')

    # Long Filter (레거시 - long_permit_enabled=False일 때 사용)
    long_filter_enabled: bool = False
    long_filter_max_markdown_prob: float = 0.40

    # Hilbert Regime Filter (1H, causal, IC=+0.027)
    hilbert_filter_enabled: bool = True
    hilbert_detrend_period: int = 48   # 1H bars (48시간)
    hilbert_window: int = 32           # Hilbert 윈도우
    hilbert_block_long_on_bear: bool = True   # Long: BEAR에서 block
    hilbert_block_short_on_bull: bool = False  # Short: BULL에서 block (느슨)


@dataclass
class GateDecision:
    """Entry Gate 결정 결과"""
    allowed: bool
    size_mult: float
    blocked_reason: Optional[str]
    cooldown_active: bool

    # 추가 메타데이터
    state: str
    expected_var: float
    markdown_prob: float
    transition_delta: float = 0.0
    hilbert_regime: str = 'RANGE'  # 1H Hilbert 레짐


class HMMEntryGate:
    """
    HMM 기반 Entry Gate

    사용법:
    1. 백테스트 시작 전 초기화 (posterior_map 전달)
    2. 매 entry 시도 전 check_entry() 호출
    3. allowed=False면 entry skip
    """

    def __init__(
        self,
        posterior_map: Dict[pd.Timestamp, np.ndarray],
        features_df: pd.DataFrame,
        cfg: HMMGateConfig,
        var5_by_state: Optional[Dict[str, float]] = None,
        prices_1h: Optional[pd.Series] = None,
    ):
        """
        Args:
            posterior_map: {15m_timestamp: posterior_array}
            features_df: 15m features (trend_strength 등)
            cfg: Gate 설정
            var5_by_state: 상태별 VaR5% (없으면 기본값 사용)
            prices_1h: 1H 종가 시리즈 (Hilbert 필터용, optional)
        """
        self.posterior_map = posterior_map
        self.features_df = features_df
        self.cfg = cfg
        self.var5_by_state = var5_by_state or VAR5_BY_STATE

        # 상태 추적
        self.prev_posterior: Optional[np.ndarray] = None
        self.cooldown_until: Optional[pd.Timestamp] = None
        self.last_processed_ts: Optional[pd.Timestamp] = None

        # Hilbert Regime (1H)
        self._hilbert_regimes: Optional[pd.DataFrame] = None
        if cfg.hilbert_filter_enabled and prices_1h is not None:
            classifier = WaveRegimeClassifier(
                detrend_period=cfg.hilbert_detrend_period,
                hilbert_window=cfg.hilbert_window,
            )
            self._hilbert_regimes = classifier.classify_series_causal(prices_1h)

        # 통계
        self.stats = {
            'total_checks': 0,
            'blocked_by_transition': 0,
            'blocked_by_short_permit': 0,
            'blocked_by_long_permit': 0,
            'blocked_by_long_filter': 0,
            'blocked_by_hilbert': 0,
            'allowed': 0,
        }

        # Bar-level decision 캐시 (성능 최적화)
        self._decision_cache: Dict[pd.Timestamp, GateDecision] = {}

    def _get_posterior(self, ts_15m: pd.Timestamp) -> Optional[np.ndarray]:
        """15m 타임스탬프에 해당하는 posterior 가져오기"""
        if ts_15m in self.posterior_map:
            return self.posterior_map[ts_15m]

        # 가장 가까운 이전 timestamp 찾기
        earlier = [t for t in self.posterior_map.keys() if t <= ts_15m]
        if earlier:
            closest = max(earlier)
            return self.posterior_map[closest]

        return None

    def _get_trend_strength(self, ts_15m: pd.Timestamp) -> float:
        """15m 타임스탬프의 trend_strength 가져오기"""
        if ts_15m in self.features_df.index and 'trend_strength' in self.features_df.columns:
            val = self.features_df.loc[ts_15m, 'trend_strength']
            if not pd.isna(val):
                return float(val)
        return 0.0

    def _get_hilbert_regime(self, ts: pd.Timestamp) -> str:
        """1H Hilbert 레짐 가져오기 (causal - 완료된 1H봉만 사용)"""
        if self._hilbert_regimes is None:
            return 'RANGE'

        # 5m/15m → 완료된 1H봉 (lookahead 방지)
        ts_1h_current = ts.floor('1h')
        ts_1h = ts_1h_current - pd.Timedelta(hours=1)

        # DatetimeIndex인 경우
        if isinstance(self._hilbert_regimes.index, pd.DatetimeIndex):
            if ts_1h in self._hilbert_regimes.index:
                regime = self._hilbert_regimes.loc[ts_1h, 'regime']
                return str(regime) if pd.notna(regime) else 'RANGE'

            # 가장 가까운 이전 timestamp 찾기
            mask = self._hilbert_regimes.index <= ts_1h
            if mask.any():
                closest = self._hilbert_regimes.index[mask][-1]
                regime = self._hilbert_regimes.loc[closest, 'regime']
                return str(regime) if pd.notna(regime) else 'RANGE'
        else:
            # 인덱스가 datetime이 아닌 경우 - 마지막 유효값 반환
            last_regime = self._hilbert_regimes['regime'].iloc[-1]
            return str(last_regime) if pd.notna(last_regime) else 'RANGE'

        return 'RANGE'

    def _compute_bar_decision(self, ts_15m: pd.Timestamp) -> GateDecision:
        """특정 15m bar에 대한 gate 결정 계산"""
        posterior = self._get_posterior(ts_15m)

        if posterior is None:
            return GateDecision(
                allowed=True,
                size_mult=1.0,
                blocked_reason=None,
                cooldown_active=False,
                state='unknown',
                expected_var=-7.0,
                markdown_prob=0.0,
            )

        # 상태 추출
        state_idx = int(posterior.argmax())
        state = IDX_TO_STATE.get(state_idx, 'range')

        # Expected VaR (posterior-weighted)
        var5_vec = np.array([
            self.var5_by_state.get(IDX_TO_STATE.get(i, 'range'), -7.0)
            for i in range(len(posterior))
        ])
        expected_var = float((posterior * var5_vec).sum())

        # Markdown probability
        markdown_idx = STATE_TO_IDX.get('markdown', 5)
        markdown_prob = float(posterior[markdown_idx]) if markdown_idx < len(posterior) else 0.0

        # Transition delta
        delta = 0.0
        if self.prev_posterior is not None:
            delta = float(np.abs(posterior - self.prev_posterior).max())

        # Transition Cooldown
        cooldown_active = (
            self.cooldown_until is not None and
            ts_15m < self.cooldown_until
        )

        if not cooldown_active and delta > self.cfg.transition_delta:
            self.cooldown_until = ts_15m + pd.Timedelta(minutes=15 * self.cfg.cooldown_bars)
            cooldown_active = True

        if cooldown_active:
            return GateDecision(
                allowed=False,
                size_mult=0.0,
                blocked_reason=f'transition_cooldown (delta={delta:.3f})',
                cooldown_active=True,
                state=state,
                expected_var=expected_var,
                markdown_prob=markdown_prob,
                transition_delta=delta,
            )

        # Soft Sizing
        raw_size = abs(self.cfg.var_target) / max(abs(expected_var), 1e-6)
        size_mult = float(np.clip(raw_size, self.cfg.size_min, self.cfg.size_max))

        # 이전 posterior 업데이트
        self.prev_posterior = posterior.copy()

        return GateDecision(
            allowed=True,
            size_mult=size_mult,
            blocked_reason=None,
            cooldown_active=False,
            state=state,
            expected_var=expected_var,
            markdown_prob=markdown_prob,
            transition_delta=delta,
        )

    def check_entry(
        self,
        ts: pd.Timestamp,
        side: Literal['long', 'short'],
    ) -> GateDecision:
        """Entry 시도에 대한 gate 결정"""
        self.stats['total_checks'] += 1

        # 5m → 15m 변환 (완성된 15분봉만 사용 - lookahead 방지)
        ts_15m_current = ts.floor('15min')
        ts_15m = ts_15m_current - pd.Timedelta(minutes=15)

        # 캐시 체크
        if ts_15m not in self._decision_cache:
            self._decision_cache[ts_15m] = self._compute_bar_decision(ts_15m)

        base_decision = self._decision_cache[ts_15m]

        # Hilbert 레짐 가져오기 (1H, causal)
        hilbert_regime = self._get_hilbert_regime(ts)

        if not base_decision.allowed:
            self.stats['blocked_by_transition'] += 1
            return GateDecision(
                allowed=False,
                size_mult=0.0,
                blocked_reason=base_decision.blocked_reason,
                cooldown_active=base_decision.cooldown_active,
                state=base_decision.state,
                expected_var=base_decision.expected_var,
                markdown_prob=base_decision.markdown_prob,
                transition_delta=base_decision.transition_delta,
                hilbert_regime=hilbert_regime,
            )

        # Short Permit
        if side == 'short' and self.cfg.short_permit_enabled:
            trend_strength = self._get_trend_strength(ts_15m)
            short_allowed = (
                base_decision.markdown_prob > self.cfg.short_permit_min_markdown_prob and
                trend_strength < self.cfg.short_permit_min_trend_strength
            )

            if not short_allowed:
                self.stats['blocked_by_short_permit'] += 1
                return GateDecision(
                    allowed=False,
                    size_mult=0.0,
                    blocked_reason=f'short_permit (markdown_prob={base_decision.markdown_prob:.2f})',
                    cooldown_active=False,
                    state=base_decision.state,
                    expected_var=base_decision.expected_var,
                    markdown_prob=base_decision.markdown_prob,
                    transition_delta=base_decision.transition_delta,
                    hilbert_regime=hilbert_regime,
                )

        # Long Permit
        if side == 'long' and self.cfg.long_permit_enabled:
            long_allowed = base_decision.state in self.cfg.long_permit_states

            if not long_allowed:
                self.stats['blocked_by_long_permit'] += 1
                return GateDecision(
                    allowed=False,
                    size_mult=0.0,
                    blocked_reason=f'long_permit (state={base_decision.state})',
                    cooldown_active=False,
                    state=base_decision.state,
                    expected_var=base_decision.expected_var,
                    markdown_prob=base_decision.markdown_prob,
                    transition_delta=base_decision.transition_delta,
                    hilbert_regime=hilbert_regime,
                )

        # Hilbert Regime Filter (1H)
        if self.cfg.hilbert_filter_enabled:
            # Long: BEAR 레짐에서 block
            if side == 'long' and self.cfg.hilbert_block_long_on_bear:
                if hilbert_regime == 'BEAR':
                    self.stats['blocked_by_hilbert'] += 1
                    return GateDecision(
                        allowed=False,
                        size_mult=0.0,
                        blocked_reason=f'hilbert_bear (regime={hilbert_regime})',
                        cooldown_active=False,
                        state=base_decision.state,
                        expected_var=base_decision.expected_var,
                        markdown_prob=base_decision.markdown_prob,
                        transition_delta=base_decision.transition_delta,
                        hilbert_regime=hilbert_regime,
                    )

            # Short: BULL 레짐에서 block (옵션)
            if side == 'short' and self.cfg.hilbert_block_short_on_bull:
                if hilbert_regime == 'BULL':
                    self.stats['blocked_by_hilbert'] += 1
                    return GateDecision(
                        allowed=False,
                        size_mult=0.0,
                        blocked_reason=f'hilbert_bull (regime={hilbert_regime})',
                        cooldown_active=False,
                        state=base_decision.state,
                        expected_var=base_decision.expected_var,
                        markdown_prob=base_decision.markdown_prob,
                        transition_delta=base_decision.transition_delta,
                        hilbert_regime=hilbert_regime,
                    )

        self.stats['allowed'] += 1
        # 최종 결과에 hilbert_regime 추가
        return GateDecision(
            allowed=base_decision.allowed,
            size_mult=base_decision.size_mult,
            blocked_reason=base_decision.blocked_reason,
            cooldown_active=base_decision.cooldown_active,
            state=base_decision.state,
            expected_var=base_decision.expected_var,
            markdown_prob=base_decision.markdown_prob,
            transition_delta=base_decision.transition_delta,
            hilbert_regime=hilbert_regime,
        )

    def get_stats(self) -> Dict:
        """통계 반환"""
        return {
            **self.stats,
            'block_rate': (
                (self.stats['total_checks'] - self.stats['allowed']) /
                max(self.stats['total_checks'], 1)
            ),
        }

    def reset(self):
        """상태 초기화"""
        self.prev_posterior = None
        self.cooldown_until = None
        self._decision_cache.clear()
        self.stats = {k: 0 for k in self.stats}
