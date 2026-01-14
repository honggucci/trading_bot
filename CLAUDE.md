# Trading Bot - Claude Code Instructions

## Project Info

- **Project Name**: trading_bot
- **RAG Embedding**: trading_bot (모든 대화/산출물 임베딩 시 이 이름 사용)
- **Working Directory**: `C:\Users\hahonggu\Desktop\coin_master\projects\trading_bot`

---

## Core Philosophy

시장은 본질적으로 **항상 횡보(레인지)**다.
- 모든 TF(1W~5m)는 프랙탈 레인지의 서로 다른 해상도
- 추세는 레인지 확장의 특수 이벤트
- 레인지 극단에서 Mean-Reversion을 먹는 방식이 기본

---

## Timeframe Terminology (고정)

| Role | TFs | Purpose |
|------|-----|---------|
| **Context TFs** | 1W, 1D, 4H, 1H | 프랙탈 레인지 좌표 + 레짐 합의 |
| **Anchor TF** | 15m | Zone 생성 (후보 가격대) |
| **Trigger TF** | 5m | 흡수/실패 트리거 (진입 타이밍) |

---

## Architecture Layers

```
Layer 1: Context (1W/1D/4H/1H)
    └─ Fractal Range Coordinates (z-score)
    └─ Trade Bias: LONG / SHORT / NEUTRAL

Layer 2: Zone Builder (15m)
    └─ ZigZag (confirmed pivots only)
    └─ Fib Levels
    └─ Divergence Score (NOT boundary)
    └─ OUTPUT: ZoneEvent

Layer 3: HMM Entry Gate (15m)
    └─ Transition Cooldown
    └─ Long/Short Permit
    └─ Soft Sizing (VaR)

Layer 4: Trigger (5m)
    └─ Zone Entry Check
    └─ Absorption/Failure Trigger (MUST)

Layer 5: Execution
    └─ Limit Order
    └─ ATR SL/TP
    └─ Cost Model
```

---

## Critical Rules (위반 시 버그) - 9-Persona Reviewed

### 1. ZigZag confirm_ts + TTL
- pivot은 `confirm_ts` 이후에만 유효
- Fib swing은 confirmed pivot 쌍으로만 계산
- **pivot 확정 전 사용 금지 (룩어헤드)**
- **Fib TTL**: 다음 pivot confirm까지만 유효

### 2. Divergence Score + Fib as Feature
- 이분탐색 "feasible boundary" 방식 폐기
- `divergence_strength` 점수로 대체
- **Fib는 "정답 레벨"이 아니라 근접도/밴드 피처**
- Multi-TF 스윙 합의(15m/1h/4h)로 존 생성

### 3. ParamPolicy (6개 필수 규칙)
| # | 규칙 |
|---|------|
| 1 | Fast/Medium/Slow 계층 분리 |
| 2 | One-step delay (t→t+1) |
| 3 | 포지션 보유 중 파라미터 고정 |
| 4 | Optuna 금지 → 온라인 정책 |
| 5 | 비용+리스크 목표함수 |
| 6 | 챔피언/챌린저 승격 + 해시 |

### 4. 5m Trigger 필수
- Zone 진입만으론 진입 금지
- 최소 1개 흡수/실패 트리거 필요:
  - Spring/UTAD Reclaim
  - Failed Swing
  - Effort vs Result (Absorption)

### 5. HMM Gate Integrity
- 15m bar 확정 기준으로만 Gate 평가
- 5m에서 15m 미래값 사용 금지

### 6. Zone Lock (리스크 매니저)
- 포지션 보유 중 Zone 변경 금지
- 존 변경은 flat 상태에서만

---

## Related Projects

| Project | Path | 역할 |
|---------|------|------|
| legacy | ./legacy | 다이버전스 로직 원본 |
| wpcn-backtester | ../wpcn-backtester-cli-noflask | 백테스터 + HMM Gate |
| hattz_empire | ../../hattz_empire | AI 오케스트레이션 (RAG) |

---

## File Structure

```
trading_bot/
├── docs/
│   ├── SPEC_v2_integration.md
│   └── session_backups/
├── src/
│   ├── context/          # Layer 1
│   ├── zone/             # Layer 2
│   ├── gate/             # Layer 3
│   ├── trigger/          # Layer 4
│   ├── execution/        # Layer 5
│   └── utils/
├── tests/
├── config/
├── data/
└── legacy/
```

---

## Git Repository

- **Remote**: https://github.com/hattz/trading_bot.git
- **Main Branch**: main

---

## Session Backup (10턴 규칙)

**10턴마다 자동 계층백업 필수:**

1. `docs/session_backup_YYYYMMDD_HHMM.md` 생성
2. 백업 내용:
   - 완료된 작업 요약
   - 생성/수정된 파일 목록
   - 테스트 결과
   - 다음 작업 제안
3. hattz_empire 서버 실행 중이면 RAG 임베딩 API 호출
4. RAG 임베딩: project=trading_bot

---

## 테스트 현황 (v3.2)

```
tests/
├── test_exit_logic.py         - 15 tests
├── test_multi_tf_fib.py       - 11 tests
├── test_real_btc_data.py      -  4 tests
├── test_risk_manager.py       - 32 tests (+6 Pessimist scenarios)
├── test_unified_signal_v3.py  - 15 tests
└── test_backtest_engine.py    -  9 tests (+2 signal generation)
─────────────────────────────────────────────
Total                          - 86 tests
```

실행: `python -m pytest tests/ -v`

---

## 학습된 모델

```
models/
├── hmm_model.pkl      # 6-state GaussianHMM
├── posterior_map.pkl  # 34,981 timestamps (15m)
└── features_df.pkl    # emission features
```

학습: `python scripts/train_hmm_and_save.py`

---

## Version

- v2.0 (2026-01-13): Integration Spec 완성
- v2.1 (2026-01-13): 9-Persona Review 반영
- v3.0 (2026-01-14): Legacy + HMM + TFPredictor 통합
- v3.1 (2026-01-14): RiskManager 통합, 77 tests 완료
- v3.2 (2026-01-14): 9-Persona Critical Review 완료, 86 tests