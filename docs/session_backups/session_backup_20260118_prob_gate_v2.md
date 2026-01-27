# Session Backup - Probability Gate v2 구현 완료

**Date**: 2026-01-18
**Project**: trading_bot
**RAG Embedding Tag**: trading_bot

---

## 완료된 작업

### PR1: Core Probability Gate ✅
- `src/regime/prob_gate.py` 신규 생성
- Temperature-scaled sigmoid: P(bull) = sigmoid(score / T)
- 2가지 온도 모드: vol (변동성 기반), instability (신호 불안정성 기반)
- Wilder ATR, rolling z-score, overflow-safe sigmoid 구현
- SciPy 의존성 없는 Spearman IC 구현

### PR2: Upstream Scores ✅
- `src/regime/upstream_scores.py` 신규 생성
- Hilbert 1H score builder
- HMM score builder (예정)
- 1H→15m alignment (ffill + shift(1) for 'open' semantics)

### PR3: Probability Calibration ✅
- `apply_probability_calibration()` 함수 추가
- p_shrink 파라미터로 과신(overconfidence) 감소
- **Champion Config 확정**: v2-vol + p_shrink=0.6

---

## OOS 평가 결과 (1H Hilbert Score)

| Model | IC@1 | IC@4 | IC@16 | Brier | ECE |
|-------|------|------|-------|-------|-----|
| Baseline (T=1.5) | 0.0420 | 0.0499 | 0.1153 | 0.2834 | 0.1519 |
| **v2-vol-s60** | **0.0446** | **0.0620** | **0.1207** | **0.2673** | **0.1084** |

### 개선율 (v2-vol-s60 vs Baseline)
- IC@4: **+24.2%**
- Brier: **-5.7%** (낮을수록 좋음)
- ECE: **-28.6%** (낮을수록 좋음)

---

## 생성/수정된 파일

| 파일 | 작업 | 라인 |
|------|------|------|
| `src/regime/prob_gate.py` | 신규 | ~350 |
| `src/regime/upstream_scores.py` | 신규 | ~150 |
| `tests/test_prob_gate.py` | 신규 | ~300 |
| `scripts/test_prob_gate_oos.py` | 신규 | ~400 |

---

## 테스트 결과

```
tests/test_prob_gate.py: 28 tests PASSED
- TestProbRange
- TestMonotonicScore
- TestTemperatureEffect
- TestTemperatureBounds
- TestVolSpikeIncreasesT
- TestGateActionThresholds
- TestClippingAndEmaStability
- TestProbabilityCalibration (PR3 추가)
```

---

## 핵심 발견

### 1H Score가 15m보다 효과적인 이유
1. **Signal Aging**: 1H→15m ffill 시 최대 59분 지연
2. **Horizon Mismatch**: 1H 스코어는 1H+ 수익률 예측에 적합
3. **15m 노이즈**: 고빈도 데이터의 낮은 SNR

### Calibration이 필요한 이유
- T-scaling만으로는 확률이 극단으로 치우침
- p_shrink로 0.5 방향으로 수축 → Brier/ECE 개선
- 단조 변환이므로 IC rank order 보존

---

## 다음 작업 (우선순위)

### 1. Backtest Wiring (긴급)
- ProbabilityGate를 BacktestEngine에 AND 필터로 연결
- Gate ON vs OFF 비교로 실제 PnL 효과 검증
- 파일: `src/backtest/engine.py` 수정

### 2. PR4: 15m-Specific Score
- 1H gate는 필터로만 사용
- 15m 전용 스코어 설계 (trigger-level)

### 3. TP Split (50/30/20)
- `src/execution/futures_backtest_15m.py` 수정
- 3단계 분할 청산

### 4. 1W Fib Anchor
- `src/context/cycle_anchor.py` 통합

---

## Champion Config

```python
# src/regime/prob_gate.py
ProbGateConfig(
    temp_mode='vol',
    n_atr=96,
    vol_window=192,
    T_a=1.2,
    T_b=0.6,
    T_min=0.7,
    T_max=3.0,
    score_window=192,
    p_shrink=0.6,  # CHAMPION
    thr_long=0.55,
    thr_short=0.55,
)
```

---

## 관련 컨텍스트

- Plan 파일: `C:\Users\hahonggu\.claude\plans\frolicking-launching-ocean.md`
- 기존 Boltzmann 레짐: IC=-0.008 (실패) → Prob Gate v2로 대체
- 물리학 흉내 X → Softmax + Temperature 노브 (확신도 조절)

---

## Version
- v2.0: 2026-01-18 - PR3 완료, Champion Config 확정
