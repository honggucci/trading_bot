# -*- coding: utf-8 -*-
"""
Risk Management Module
======================

리스크 관리 모듈.

핵심 기능:
- Max Position Size - 계좌 대비 최대 포지션
- Daily Loss Limit - 일일 최대 손실
- Concurrent Position Limit - 동시 포지션 수
- Circuit Breaker - 연속 손실 시 거래 중단
- Risk-based Position Sizing - 손절 거리 기반 포지션 계산

사용법:
```python
from src.risk import RiskManager, RiskConfig

manager = RiskManager(
    equity=100000,
    config=RiskConfig(
        max_position_pct=10.0,
        daily_loss_limit_pct=3.0,
    )
)

# 진입 전 체크
ok, reason = manager.can_open_position(size=0.1, price=95000)
if not ok:
    print(f"Blocked: {reason}")

# 리스크 기반 사이징
size = manager.get_position_size_by_risk(price=95000, stop_loss_pct=2.0)
```
"""
from .manager import (
    RiskConfig,
    DailyStats,
    RiskManager,
)

__all__ = [
    'RiskConfig',
    'DailyStats',
    'RiskManager',
]