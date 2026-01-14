"""
HMM Risk Filter Policy
======================

HMM 기반 진입 필터 정책 설정.

Policy v1.2 - Transition Cooldown 완화:
- transition_delta: 0.40 (큰 상태 전이만 cooldown 유발)
- cooldown_bars: 1 (대기 시간 단축)

Origin: wpcn-backtester-cli-noflask/policy_v1_2_relaxed_cooldown.py
"""
from dataclasses import dataclass
from typing import Tuple
import hashlib
import json


@dataclass(frozen=True)
class HMMPolicyConfig:
    """
    HMM Risk Filter Policy 설정
    """
    # A. Soft Sizing (VaR 기반)
    var_target: float = -5.0      # 목표 VaR
    size_min: float = 0.25        # 최소 사이즈 배수
    size_max: float = 1.25        # 최대 사이즈 배수

    # B. Transition Cooldown
    transition_delta: float = 0.40  # 상태 전이 임계치
    cooldown_bars: int = 1          # 쿨다운 바 수

    # C. Short Permit (markdown + 강한 하락추세에서만 숏)
    short_permit_enabled: bool = True
    short_permit_min_markdown_prob: float = 0.60
    short_permit_min_trend_strength: float = -0.10

    # D. Long Permit (허용 상태에서만 롱)
    long_permit_enabled: bool = True
    long_permit_states: Tuple[str, ...] = ('markup', 'accumulation', 're_accumulation')

    # Legacy (비활성화)
    long_filter_enabled: bool = False
    long_filter_max_markdown_prob: float = 0.40

    def to_dict(self) -> dict:
        return {
            'var_target': self.var_target,
            'size_min': self.size_min,
            'size_max': self.size_max,
            'transition_delta': self.transition_delta,
            'cooldown_bars': self.cooldown_bars,
            'short_permit_enabled': self.short_permit_enabled,
            'short_permit_min_markdown_prob': self.short_permit_min_markdown_prob,
            'short_permit_min_trend_strength': self.short_permit_min_trend_strength,
            'long_permit_enabled': self.long_permit_enabled,
            'long_permit_states': list(self.long_permit_states),
            'long_filter_enabled': self.long_filter_enabled,
        }

    def config_hash(self) -> str:
        """설정 해시 (변경 감지용)"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


# Default Policy - v1.2 Relaxed Cooldown
DEFAULT_HMM_POLICY = HMMPolicyConfig()


# Aggressive Policy - No Cooldown
AGGRESSIVE_HMM_POLICY = HMMPolicyConfig(
    transition_delta=1.0,  # 사실상 off
    cooldown_bars=0,
)


# Conservative Policy - Strict Cooldown
CONSERVATIVE_HMM_POLICY = HMMPolicyConfig(
    transition_delta=0.20,
    cooldown_bars=2,
)