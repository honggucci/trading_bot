"""
Config Loader
=============

YAML 기반 트레이딩 파라미터 로더.

사용법:
    from src.config import load_symbol_config, get_leverage

    # 심볼별 설정 로드
    config = load_symbol_config("BTC-USDT")
    print(config.leverage.default)  # 10
    print(config.execution.atr_sl_mult)  # 2.0

    # 특정 파라미터만
    leverage = get_leverage("BTC-USDT")  # 10

환경변수 오버라이드:
    TRADING_LEVERAGE=15        # 레버리지 강제
    TRADING_DRY_RUN=true       # 드라이런 모드
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
SYMBOLS_DIR = CONFIG_DIR / "symbols"


@dataclass
class ExecutionParams:
    """실행 파라미터"""
    atr_period: int = 14
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 3.0
    max_hold_bars: int = 32
    pending_order_max_bars: int = 4
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage: float = 0.0001


@dataclass
class LeverageParams:
    """레버리지 파라미터"""
    default: int = 3
    max: int = 25
    margin_mode: str = "isolated"


@dataclass
class WyckoffParams:
    """Wyckoff (Theta) 파라미터"""
    pivot_lr: int = 5
    box_L: int = 20
    atr_len: int = 14
    x_atr: float = 0.5
    m_bw: float = 0.3
    m_freeze: int = 5
    N_reclaim: int = 3
    N_fill: int = 3
    F_min: float = 0.5


@dataclass
class TradingConfig:
    """통합 트레이딩 설정"""
    symbol: str
    enabled: bool = True
    leverage: LeverageParams = field(default_factory=LeverageParams)
    execution: ExecutionParams = field(default_factory=ExecutionParams)
    wyckoff: WyckoffParams = field(default_factory=WyckoffParams)
    raw: Dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: Dict) -> Dict:
    if os.getenv("TRADING_LEVERAGE"):
        config.setdefault("leverage", {})
        config["leverage"]["default"] = int(os.getenv("TRADING_LEVERAGE"))
    if os.getenv("TRADING_DRY_RUN", "").lower() in ("true", "1"):
        config["dry_run"] = True
    return config


def load_config() -> Dict[str, Any]:
    """기본 설정 로드"""
    return _apply_env_overrides(_load_yaml(CONFIG_DIR / "default.yaml"))


def load_symbol_config(symbol: str) -> TradingConfig:
    """심볼별 설정 로드 (default + symbol override + env)"""
    base = load_config()
    symbol_cfg = _load_yaml(SYMBOLS_DIR / f"{symbol}.yaml")
    merged = _apply_env_overrides(_deep_merge(base, symbol_cfg))

    lev = merged.get("leverage", {})
    exe = merged.get("execution", {})
    wyc = merged.get("wyckoff", {})
    fee = merged.get("fees", {})

    return TradingConfig(
        symbol=symbol,
        enabled=merged.get("enabled", True),
        leverage=LeverageParams(
            default=lev.get("default", 3),
            max=lev.get("max", 25),
            margin_mode=lev.get("margin_mode", "isolated"),
        ),
        execution=ExecutionParams(
            atr_period=exe.get("atr_period", 14),
            atr_sl_mult=exe.get("atr_sl_mult", 2.0),
            atr_tp_mult=exe.get("atr_tp_mult", 3.0),
            max_hold_bars=exe.get("max_hold_bars", 32),
            pending_order_max_bars=exe.get("pending_order_max_bars", 4),
            maker_fee=fee.get("maker", 0.0002),
            taker_fee=fee.get("taker", 0.0005),
            slippage=fee.get("slippage", 0.0001),
        ),
        wyckoff=WyckoffParams(
            pivot_lr=wyc.get("pivot_lr", 5),
            box_L=wyc.get("box_L", 20),
            atr_len=wyc.get("atr_len", 14),
            x_atr=wyc.get("x_atr", 0.5),
            m_bw=wyc.get("m_bw", 0.3),
            m_freeze=wyc.get("m_freeze", 5),
            N_reclaim=wyc.get("N_reclaim", 3),
            N_fill=wyc.get("N_fill", 3),
            F_min=wyc.get("F_min", 0.5),
        ),
        raw=merged,
    )


def get_leverage(symbol: str, mode: str = "default") -> int:
    """레버리지 조회"""
    config = load_symbol_config(symbol)
    return config.leverage.max if mode == "max" else config.leverage.default


def get_execution_params(symbol: str) -> ExecutionParams:
    """실행 파라미터 조회"""
    return load_symbol_config(symbol).execution


def get_wyckoff_params(symbol: str) -> WyckoffParams:
    """Wyckoff 파라미터 조회"""
    return load_symbol_config(symbol).wyckoff


def list_symbols(enabled_only: bool = True) -> List[str]:
    """설정된 심볼 목록"""
    if not SYMBOLS_DIR.exists():
        return []
    symbols = []
    for f in SYMBOLS_DIR.glob("*.yaml"):
        if enabled_only:
            if load_symbol_config(f.stem).enabled:
                symbols.append(f.stem)
        else:
            symbols.append(f.stem)
    return sorted(symbols)