# Trading Bot

멀티타임프레임 프랙탈 레인지 기반 Mean-Reversion 트레이딩 시스템

## Core Philosophy

- **시장은 본질적으로 항상 횡보(레인지)다**
- 모든 타임프레임(1W~5m)은 프랙탈 레인지의 서로 다른 해상도
- 추세는 레인지 확장의 특수 이벤트
- 레인지 극단(할인/프리미엄)에서 흡수/실패 확인 후 Mean-Reversion

---

## Architecture

```
Context TFs (1W/1D/4H/1H)  →  Anchor TF (15m)  →  Trigger TF (5m)
     │                            │                    │
     │                            │                    └─ 흡수/실패 트리거
     │                            └─ Zone 생성 (가격대)
     └─ Trade Bias (방향성)
```

### Layer 구조

| Layer | TF | 역할 |
|-------|-----|------|
| 1. Context | 1W, 1D, 4H, 1H | 프랙탈 레인지 좌표 (z-score) |
| 2. Zone | 15m | ZigZag + Fib + Divergence Score → Zone 생성 |
| 3. Gate | 15m | HMM Entry Gate (Permit/Cooldown/Sizing) |
| 4. Trigger | 5m | Spring/Failed Swing/Absorption 확인 |
| 5. Execution | 15m | 지정가 주문 + ATR SL/TP |

---

## Project Structure

```
trading_bot/
├── docs/
│   ├── SPEC_v2_integration.md    # 통합 명세
│   └── session_backups/
├── src/
│   ├── context/                  # Layer 1
│   ├── zone/                     # Layer 2
│   ├── gate/                     # Layer 3
│   ├── trigger/                  # Layer 4
│   ├── execution/                # Layer 5
│   └── utils/
├── tests/
├── config/
├── data/
└── legacy/                       # 기존 코드 (참조용)
```

---

## Key Components

### From Legacy (trading_bot/legacy)
- ZigZag 피벗 탐지 (confirm_ts 필수)
- 피보나치 레벨 산출
- StochRSI 과매도 구간 탐지
- RSI 다이버전스 점수 (바운더리 → 점수로 변경)

### From WPCN (wpcn-backtester)
- HMM Entry Gate (Transition Cooldown, Permit, Soft Sizing)
- 15분봉 지정가 백테스터
- ATR 기반 SL/TP
- 비용 모델 (수수료/슬리피지/펀딩비)

---

## Critical Rules

1. **ZigZag confirm_ts**: pivot 확정 전 사용 금지 (룩어헤드 방지)
2. **Divergence Score**: 바운더리 이분탐색 → 점수 기반으로 전환
3. **ParamPolicy**: 매 바 튜닝 금지, 레짐→파라미터 매핑
4. **5m 트리거 필수**: Zone 진입만으론 부족, 흡수/실패 확인 필수

---

## Related Projects

| Project | Path | 역할 |
|---------|------|------|
| trading_bot | 현재 | 통합 시스템 |
| legacy | trading_bot/legacy | 다이버전스 로직 원본 |
| wpcn-backtester | ../wpcn-backtester-cli-noflask | 백테스터 + HMM Gate |
| hattz_empire | ../hattz_empire | AI 오케스트레이션 |

---

## RAG Embedding

- **Project Name**: `trading_bot`
- 모든 대화/산출물은 이 프로젝트명으로 임베딩

---

## Version

- **Current**: v2.0 (Integration Spec)
- **Date**: 2026-01-13

---

## Author

- Hattz (하홍구)