# Trading Bot - 전체 로직 체크리스트

## 검증 일시: 2026-01-15

---

## 1. Core Philosophy

- [ ] 시장은 본질적으로 **횡보(레인지)**
- [ ] 모든 TF는 프랙탈 레인지의 서로 다른 해상도
- [ ] 레인지 극단에서 **Mean-Reversion** 전략

---

## 2. Multi-TF 트레이딩 구조

| Anchor TF | Trigger TF | Zone Width | k |
|-----------|------------|------------|---|
| 15m | 5m | 15m ATR(21) * 2.75 | 2.75 |
| 1H | 15m | 1H ATR(21) * 2.4 | 2.4 |
| 4H | 1H | 4H ATR(21) * 1.65 | 1.65 |
| 1D | 4H | 1D ATR(89) * 1.0 | 1.0 |

- [ ] 각 TF별 독립 트레이딩 가능
- [ ] Fib 좌표계는 **1W 기준 고정** (모든 TF 공유)

---

## 3. Layer 1: Context (1W/1D/4H/1H)

### 확정 사항
- [ ] **1W Fib Anchor (고정)**
  - Fib 0: $3,120
  - Fib 1: $20,650
  - Range: $17,530
  - Source: Binance Futures

### 기능
- [ ] Fractal Range Coordinates (z-score)
- [ ] Trade Bias 결정: LONG / SHORT / NEUTRAL
- [ ] HTF 맥락 제공 (A/B 등급 판단)

---

## 4. Layer 2: Zone Builder

### 확정 사항
- [ ] **Fib 좌표계 = 1W 기준 고정 (수학적)**
- [ ] **프랙탈 깊이 = L0만 사용** (L1 이상은 너무 밀집)
- [ ] **Zone Width = TF별 ATR * k**
- [ ] **Clamp = Fib 레벨 가격 기준 %**
  - 15m: min 0.2%, max 1.5%

### 파일
- `src/zone/builder.py`
- `config/zone_width.json`

### 검증 항목
- [ ] `FIB_0 = 3120` (고정)
- [ ] `FIB_1 = 20650` (고정)
- [ ] `RANGE = 17530` (고정)
- [ ] `EXTENDED_FIB_MAX = 8` (Fib 8.0 = $143,360)
- [ ] Zone Width clamp가 `fib_price` 기준인지 확인

### 출력
- [ ] ZoneEvent (FibZone 리스트)
- [ ] Confluence Zone (겹치는 Zone 병합)

---

## 5. Layer 3: HMM Entry Gate (15m)

### 기능
- [ ] Transition Cooldown
- [ ] Long/Short Permit
- [ ] Soft Sizing (VaR)

### Critical Rules
- [ ] 15m bar 확정 기준으로만 Gate 평가
- [ ] 5m에서 15m 미래값 사용 금지

### 파일
- `models/hmm_model.pkl`
- `models/posterior_map.pkl`

---

## 6. Layer 4: Trigger (5m / 15m / 1H / 4H)

### 진입 조건 (필수)
- [ ] Zone 진입만으론 진입 금지
- [ ] 최소 1개 흡수/실패 트리거 필요:
  - [ ] Spring/UTAD Reclaim
  - [ ] Failed Swing
  - [ ] Effort vs Result (Absorption)

### Trigger TF 매핑
| Anchor TF | Trigger TF |
|-----------|------------|
| 15m | 5m |
| 1H | 15m |
| 4H | 1H |
| 1D | 4H |

---

## 7. Layer 5: Execution

### 기능
- [ ] Limit Order
- [ ] ATR SL/TP
- [ ] Cost Model

### Critical Rules
- [ ] Zone Lock: 포지션 보유 중 Zone 변경 금지
- [ ] 존 변경은 flat 상태에서만

---

## 8. Critical Rules 체크리스트

### 8.1 ZigZag confirm_ts + TTL
- [ ] pivot은 `confirm_ts` 이후에만 유효
- [ ] Fib swing은 confirmed pivot 쌍으로만 계산
- [ ] pivot 확정 전 사용 금지 (룩어헤드)
- [ ] Fib TTL: 다음 pivot confirm까지만 유효

### 8.2 Divergence Score
- [ ] 이분탐색 "feasible boundary" 방식 **폐기**
- [ ] `divergence_strength` 점수로 대체
- [ ] Fib는 "정답 레벨"이 아니라 근접도/밴드 피처

### 8.3 ParamPolicy (6개 필수 규칙)
- [ ] Fast/Medium/Slow 계층 분리
- [ ] One-step delay (t→t+1)
- [ ] 포지션 보유 중 파라미터 고정
- [ ] Optuna 금지 → 온라인 정책
- [ ] 비용+리스크 목표함수
- [ ] 챔피언/챌린저 승격 + 해시

### 8.4 HMM Gate Integrity
- [ ] 15m bar 확정 기준으로만 Gate 평가
- [ ] 5m에서 15m 미래값 사용 금지

### 8.5 Zone Lock
- [ ] 포지션 보유 중 Zone 변경 금지
- [ ] 존 변경은 flat 상태에서만

---

## 9. Zone Width 파라미터 (확정, DO NOT CHANGE)

### Config: `config/zone_width.json`

| TF | ATR Window | k | Role |
|----|------------|---|------|
| 1W | null | null | Fib 좌표계만 |
| 1D | 89 | 1.0 | HTF Filter (예외: OOS -29% 손실로 ATR(89) 사용) |
| 4H | 21 | 1.65 | Context Filter |
| 1H | 21 | 2.4 | Context Filter |
| 15m | 21 | 2.75 | Zone Generator |

### Clamp (15m 기준)
- min_pct: 0.2%
- max_pct: 1.5%

### Recalibration Protocol
- Trigger: 3개월 롤링 Coverage < 40% or > 60%
- Change Condition: OOS Pearson +10% 이상 개선시만
- DO NOT CHANGE FOR: 몇 번 손실, 느낌, 새 논문

---

## 10. 테스트 현황

```
tests/
├── test_exit_logic.py         - 15 tests
├── test_multi_tf_fib.py       - 11 tests
├── test_real_btc_data.py      -  4 tests
├── test_risk_manager.py       - 32 tests
├── test_unified_signal_v3.py  - 15 tests
└── test_backtest_engine.py    -  9 tests
─────────────────────────────────────────────
Total                          - 86 tests
```

실행: `python -m pytest tests/ -v`

---

## 11. 파일 구조

```
src/
├── context/          # Layer 1: Context TF
│   ├── fib_levels.py      # 1W Fib 좌표계
│   ├── multi_tf_fib.py    # (기존, 동적 ZigZag - 참고용)
│   └── volatility.py      # ATR 계산
│
├── zone/             # Layer 2: Zone Builder
│   ├── builder.py         # ZoneBuilder, FibZone
│   └── __init__.py
│
├── gate/             # Layer 3: HMM Gate
├── trigger/          # Layer 4: Trigger
├── execution/        # Layer 5: Execution
└── utils/
```

---

## 12. 다음 작업 (TODO)

- [ ] Layer 3 (HMM Gate) 통합 검증
- [ ] Layer 4 (Trigger) 구현/검증
- [ ] 백테스트 실행 + 결과 분석
- [ ] 5m Trigger Quality 개선

---

## 버전

- v1.0 (2026-01-15): Zone Width 파라미터 확정 후 체크리스트 생성
