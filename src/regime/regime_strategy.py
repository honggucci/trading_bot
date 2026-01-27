"""
Regime-based Strategy Parameters (MODE82)
==========================================

레짐별 전략 파라미터 테이블.

핵심 원칙:
- BULL: 적극적 롱, 넓은 StochRSI, Trailing TP
- RANGE: 엄격 롱, 중간 StochRSI, 고정 RR
- BEAR: 조건부 롱 (StochRSI<10 + reclaim), 빠른 청산

Usage:
```python
from src.regime.regime_strategy import (
    REGIME_PARAMS,
    get_regime_params,
    get_entry_conditions,
    get_tp_mode,
)

params = get_regime_params("BULL")
print(params['stoch_rsi_threshold'])  # 30
print(params['tp_mode'])              # "trailing"
```
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal


# =============================================================================
# Regime Parameters Table
# =============================================================================

REGIME_PARAMS: Dict[str, Dict[str, Any]] = {
    "BULL": {
        # Entry 조건
        "stoch_rsi_threshold": 30.0,      # StochRSI < 30
        "rsi_threshold": 40.0,            # RSI < 40 (optional)
        "require_reclaim": False,         # Reclaim 불필요
        "long_allowed": True,

        # TP 설정
        "tp_mode": "trailing",            # Trailing Stop
        "tp_rr_target": 2.0,              # 최소 RR
        "tp_partial_pct": 0.50,           # 2R에서 50% 청산
        "trailing_activation_atr": 2.0,
        "trailing_distance_atr": 1.0,

        # 리스크
        "risk_mult": 1.0,                 # 기본 리스크

        # SL
        "sl_mode": "fib_lower",           # 기본 Fib SL
    },

    "RANGE": {
        # Entry 조건 (BEAR와 동일 - 보수적 접근)
        "stoch_rsi_threshold": 20.0,      # StochRSI < 20 (보수적)
        "rsi_threshold": 25.0,
        "require_reclaim": True,          # Reclaim 필수!
        "long_allowed": "conditional",    # 조건부

        # TP 설정 (BEAR와 동일)
        "tp_mode": "quick_exit",          # 빠른 청산
        "tp_rr_target": 1.5,              # 낮은 RR 허용
        "tp_partial_pct": 1.0,            # TP 도달 시 전량 청산
        "trailing_activation_atr": None,
        "trailing_distance_atr": None,

        # 리스크 (BEAR와 동일)
        "risk_mult": 0.3,                 # 30% 리스크

        # SL
        "sl_mode": "tight",               # 타이트한 SL
    },

    "BEAR": {
        # Entry 조건
        "stoch_rsi_threshold": 20.0,      # StochRSI < 20 (보수적)
        "rsi_threshold": 25.0,
        "require_reclaim": True,          # Reclaim 필수!
        "long_allowed": "conditional",    # 조건부

        # TP 설정
        "tp_mode": "quick_exit",          # 빠른 청산
        "tp_rr_target": 1.5,              # 낮은 RR 허용
        "tp_partial_pct": 1.0,            # TP 도달 시 전량 청산
        "trailing_activation_atr": None,
        "trailing_distance_atr": None,

        # 리스크
        "risk_mult": 0.3,                 # 30% 리스크

        # SL
        "sl_mode": "tight",               # 타이트한 SL
    },
}


# =============================================================================
# Getter Functions
# =============================================================================

def get_regime_params(regime: str) -> Dict[str, Any]:
    """
    레짐별 전체 파라미터 반환

    Args:
        regime: "BULL" | "RANGE" | "BEAR"

    Returns:
        파라미터 딕셔너리
    """
    return REGIME_PARAMS.get(regime, REGIME_PARAMS["RANGE"])


def get_stoch_rsi_threshold(regime: str) -> float:
    """레짐별 StochRSI 임계값"""
    return get_regime_params(regime)["stoch_rsi_threshold"]


def get_rsi_threshold(regime: str) -> float:
    """레짐별 RSI 임계값"""
    return get_regime_params(regime)["rsi_threshold"]


def get_risk_mult(regime: str) -> float:
    """레짐별 리스크 배수"""
    return get_regime_params(regime)["risk_mult"]


def get_tp_mode(regime: str) -> str:
    """레짐별 TP 모드"""
    return get_regime_params(regime)["tp_mode"]


def get_tp_partial_pct(regime: str) -> float:
    """레짐별 부분청산 비율"""
    return get_regime_params(regime)["tp_partial_pct"]


def requires_reclaim(regime: str) -> bool:
    """레짐에서 Reclaim 필요 여부"""
    return get_regime_params(regime)["require_reclaim"]


def is_long_allowed(regime: str) -> bool:
    """레짐에서 롱 허용 여부 (conditional은 True)"""
    allowed = get_regime_params(regime)["long_allowed"]
    return allowed == True or allowed == "conditional"


def is_long_conditional(regime: str) -> bool:
    """레짐에서 롱이 조건부인지"""
    return get_regime_params(regime)["long_allowed"] == "conditional"


# =============================================================================
# Entry Condition Checker
# =============================================================================

@dataclass
class EntryConditions:
    """Entry 조건 체크 결과"""
    regime: str
    stoch_rsi_ok: bool
    rsi_ok: bool
    reclaim_ok: bool
    overall_ok: bool
    reason: str


def check_entry_conditions(
    regime: str,
    stoch_rsi: float,
    rsi: Optional[float] = None,
    reclaim_confirmed: bool = False
) -> EntryConditions:
    """
    레짐에서 Entry 조건 체크

    Args:
        regime: 현재 레짐
        stoch_rsi: 현재 StochRSI
        rsi: 현재 RSI (optional)
        reclaim_confirmed: 1H reclaim 확인 여부

    Returns:
        EntryConditions 결과
    """
    params = get_regime_params(regime)

    # 1. StochRSI 체크
    stoch_threshold = params["stoch_rsi_threshold"]
    stoch_rsi_ok = stoch_rsi <= stoch_threshold

    # 2. RSI 체크 (optional)
    rsi_threshold = params["rsi_threshold"]
    rsi_ok = True if rsi is None else rsi <= rsi_threshold

    # 3. Reclaim 체크
    require_reclaim = params["require_reclaim"]
    reclaim_ok = True if not require_reclaim else reclaim_confirmed

    # 4. Long 허용 여부
    long_allowed = params["long_allowed"]

    # 5. Overall 판단
    if long_allowed == False:
        overall_ok = False
        reason = f"{regime}: Long not allowed"
    elif not stoch_rsi_ok:
        overall_ok = False
        reason = f"{regime}: StochRSI {stoch_rsi:.1f} > {stoch_threshold}"
    elif not rsi_ok:
        overall_ok = False
        reason = f"{regime}: RSI {rsi:.1f} > {rsi_threshold}"
    elif not reclaim_ok:
        overall_ok = False
        reason = f"{regime}: Reclaim required but not confirmed"
    else:
        overall_ok = True
        reason = f"{regime}: All conditions met"

    return EntryConditions(
        regime=regime,
        stoch_rsi_ok=stoch_rsi_ok,
        rsi_ok=rsi_ok,
        reclaim_ok=reclaim_ok,
        overall_ok=overall_ok,
        reason=reason
    )


# =============================================================================
# Position Sizing
# =============================================================================

def adjust_position_size(
    base_size: float,
    regime: str
) -> float:
    """
    레짐에 따라 포지션 크기 조정

    Args:
        base_size: 기본 포지션 크기
        regime: 현재 레짐

    Returns:
        조정된 포지션 크기
    """
    risk_mult = get_risk_mult(regime)
    return base_size * risk_mult


def adjust_risk_usd(
    base_risk_usd: float,
    regime: str
) -> float:
    """
    레짐에 따라 리스크 USD 조정

    Args:
        base_risk_usd: 기본 리스크 (USD)
        regime: 현재 레짐

    Returns:
        조정된 리스크 (USD)
    """
    risk_mult = get_risk_mult(regime)
    return base_risk_usd * risk_mult


# =============================================================================
# TP Configuration
# =============================================================================

@dataclass
class TPConfig:
    """TP 설정"""
    mode: str                               # "trailing" | "fixed_rr" | "quick_exit"
    rr_target: float                        # RR 목표
    partial_pct: float                      # 부분청산 비율 (0.0 ~ 1.0)
    trailing_activation_atr: Optional[float]
    trailing_distance_atr: Optional[float]


def get_tp_config(regime: str) -> TPConfig:
    """
    레짐별 TP 설정 반환

    Args:
        regime: 현재 레짐

    Returns:
        TPConfig
    """
    params = get_regime_params(regime)

    return TPConfig(
        mode=params["tp_mode"],
        rr_target=params["tp_rr_target"],
        partial_pct=params["tp_partial_pct"],
        trailing_activation_atr=params["trailing_activation_atr"],
        trailing_distance_atr=params["trailing_distance_atr"],
    )


# =============================================================================
# Summary Functions
# =============================================================================

def get_regime_summary(regime: str) -> str:
    """
    레짐 요약 문자열 반환

    Args:
        regime: 현재 레짐

    Returns:
        요약 문자열
    """
    params = get_regime_params(regime)

    return (
        f"{regime}: "
        f"StochRSI<{params['stoch_rsi_threshold']:.0f}, "
        f"Risk={params['risk_mult']*100:.0f}%, "
        f"TP={params['tp_mode']}"
        f"{' [RECLAIM_REQ]' if params['require_reclaim'] else ''}"
    )


def print_all_regime_params():
    """모든 레짐 파라미터 출력"""
    print("=" * 60)
    print("Regime Parameters (MODE82)")
    print("=" * 60)

    for regime in ["BULL", "RANGE", "BEAR"]:
        params = get_regime_params(regime)
        print(f"\n[{regime}]")
        print(f"  StochRSI: < {params['stoch_rsi_threshold']}")
        print(f"  RSI: < {params['rsi_threshold']}")
        print(f"  Reclaim: {'Required' if params['require_reclaim'] else 'Not required'}")
        print(f"  Long: {params['long_allowed']}")
        print(f"  Risk: {params['risk_mult']*100:.0f}%")
        print(f"  TP Mode: {params['tp_mode']}")
        print(f"  TP Partial: {params['tp_partial_pct']*100:.0f}%")

    print("\n" + "=" * 60)
