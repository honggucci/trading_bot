# MODE47 v2 세션 백업

## 세션 정보
- **날짜**: 2026-01-20
- **프로젝트**: trading_bot
- **목표**: MODE47 설정 최적화 및 트레이드 수 증가

---

## 완료된 작업

### 1. 설정 변경
| 항목 | 기존 | 변경 | 이유 |
|------|------|------|------|
| `stoch_rsi_oversold` | 20 | **30** | 트레이드 수 증가 |
| `stoch_rsi_period` | 25 | **14** | 기본값 복원 |
| `early_exit_duration` | 2h | **8h** | StaleLoss 여유 확보 |
| `use_hilbert_filter` | true | **false** | ProbGate만 사용 |
| `use_trend_filter_1h` | true | **false** | 필터 단순화 |
| `dynfib_lookback_bars` | 52 | **100** | 스윙 탐색 범위 확대 |
| Entry Offset | 0.5 | **0.3** | Fill Rate 향상 |

### 2. Hidden Divergence 테스트
- **추가**: `use_hidden_div_long` 옵션 구현
- **결과**: 34 trades, -$88, WR 35.3% (악화)
- **결론**: **비활성화** (`use_hidden_div_long: false`)

### 3. ProbGate 로직 확인
- **실제 로직**: Hilbert Transform + Temperature-Scaled Sigmoid
- **NOT HMM**: 이전 문서의 "HMM 기반" 설명은 오류
- **공식**: `p_bull = sigmoid(Hilbert_score / Temperature)`

---

## 백테스트 결과 비교

| 설정 | Trades | PnL | $/Trade | WR |
|------|--------|-----|---------|-----|
| 기존 (Hilbert ON) | 126 | -$766 | -$6.08 | 36.5% |
| StochRSI<20, Offset 0.5 | 16 | +$150 | +$9.37 | 56.2% |
| Hidden Div 추가 | 34 | -$88 | -$2.59 | 35.3% |
| **현재 (v2)** | **46** | **+$288** | **+$6.26** | **60.9%** |

---

## 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `configs/mode47.json` | 설정 최적화 (StochRSI, StaleLoss, 필터 등) |
| `scripts/backtest_strategy_compare.py` | Hidden Div 로직 추가, Entry Offset 0.3 |
| `docs/TRADE_FLOW_CHECKLIST.md` | v2 업데이트 |

---

## 핵심 발견

### 1. 트레이드 수 문제
- **문제**: 6개월에 16개 트레이드 (너무 적음)
- **원인**: StochRSI<20, Entry Offset 0.5가 너무 엄격
- **해결**: StochRSI<30, Offset 0.3으로 완화 → 46 트레이드

### 2. Hidden Divergence 실패
- Regular + Hidden 동시 사용 시 **노이즈 증가**
- Win Rate 56.2% → 35.3% 하락
- **결론**: Regular Divergence만 사용

### 3. StaleLoss 조정
- 2h → 8h로 연장
- 빠른 손절이 오히려 손실 증가시킬 수 있음

### 4. ProbGate 작동 원리
```
1️⃣ 1H Hilbert Score 계산 (Hilbert Transform)
2️⃣ Temperature = f(ATR volatility)
   - 변동성↑ → T↑ → 확률 0.5로 눌림
   - 변동성↓ → T↓ → 극단 확률 허용
3️⃣ p_bull = sigmoid(score / T)
4️⃣ p_bull > 0.55 → LONG 허용
```

---

## 다음 작업 제안

1. **8h StaleLoss 백테스트**: 변경 후 결과 확인
2. **ProbGate 임계값 조정**: 0.55 → 0.50 테스트
3. **다른 기간 테스트**: 2021 하반기, 2022년 등
4. **실시간 페이퍼 트레이딩**: 설정 검증

---

## 태그
#MODE47 #v2 #StochRSI #ProbGate #HilbertTransform #EntryOffset #StaleLoss
