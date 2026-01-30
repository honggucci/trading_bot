"""
Wyckoff Types
=============

Wyckoff 분석에 필요한 설정 및 타입 정의.

Origin: WPCN wpcn/_03_common/_01_core/types.py

Config 연동 (v2.6.15):
- get_theta_for_symbol(symbol): config에서 심볼별 Theta 로드
- 환경변수 TRADING_SYMBOL로 기본 심볼 지정 가능
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass(frozen=True)
class Theta:
    """
    와이코프 신호 감지 파라미터

    박스 + reclaim 계약:
    - box_L: 박스 길이 (15m*24 = 6시간)
    - N_reclaim: reclaim 허용 기간 (봉 수)
    - m_bw: 박스폭 비율 (reclaim 레벨)
    """
    # box + reclaim contract
    pivot_lr: int         # 피벗 좌우 바 (스윙 감지)
    box_L: int            # 박스 길이
    m_freeze: int         # 박스 프리즈 기간
    atr_len: int          # ATR 계산 기간
    x_atr: float          # 돌파 임계값 (ATR 배수)
    m_bw: float           # 박스폭 비율 (reclaim 레벨)
    N_reclaim: int        # reclaim 허용 기간 (봉 수)

    # execution realism knobs
    N_fill: int           # 주문 체결 허용 기간
    F_min: float          # Fill Probability Minimum (0.0~1.0)


def get_default_theta() -> Theta:
    """기본 Theta 설정 반환 (하드코딩 fallback)"""
    return Theta(
        pivot_lr=5,
        box_L=24,         # 15m * 24 = 6시간
        atr_len=14,
        x_atr=0.5,
        m_bw=0.3,
        m_freeze=6,
        N_reclaim=4,
        N_fill=4,
        F_min=0.5,
    )


def get_theta_for_symbol(symbol: Optional[str] = None) -> Theta:
    """
    심볼별 Theta 설정 로드 (config 기반)

    Args:
        symbol: 심볼명 (예: "BTC-USDT"). None이면 환경변수 TRADING_SYMBOL 사용

    Returns:
        Theta 객체

    사용법:
        theta = get_theta_for_symbol("BTC-USDT")
        theta = get_theta_for_symbol()  # TRADING_SYMBOL 환경변수 사용
    """
    if symbol is None:
        symbol = os.getenv("TRADING_SYMBOL", "BTC-USDT")

    try:
        from src.config import load_symbol_config
        config = load_symbol_config(symbol)
        wyc = config.wyckoff

        return Theta(
            pivot_lr=wyc.pivot_lr,
            box_L=wyc.box_L,
            atr_len=wyc.atr_len,
            x_atr=wyc.x_atr,
            m_bw=wyc.m_bw,
            m_freeze=wyc.m_freeze,
            N_reclaim=wyc.N_reclaim,
            N_fill=wyc.N_fill,
            F_min=wyc.F_min,
        )
    except ImportError:
        # config 모듈 없으면 기본값 반환
        return get_default_theta()