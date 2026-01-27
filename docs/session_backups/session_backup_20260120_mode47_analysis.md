# MODE47 Trade Flow 검증 - 세션 백업

## 세션 정보
- **날짜**: 2026-01-20
- **프로젝트**: trading_bot
- **목표**: MODE47 Trade Flow 7단계 검증

---

## 완료된 작업

### Step 1: Fib 레벨 계산 [완료]
- 1D ZigZag Anchor 정상 작동
- L0 Only (7개 레벨)
- Log Space 계산 정확

### Step 2: 진입 시그널 감지 [완료]
- StochRSI < 25: 정상 (19.4%)
- Fib Tolerance: atr_pct 모드 정상
- Divergence: 157개 감지

### Step 3: Entry Gate [완료]
- **설정 변경**: drift_regime → Hilbert Filter
- Hilbert 차단: 196개 (BEAR)
- ProbGate 차단: 1,186개 (p_bull < 0.55)
- Gate 통과: 125개

### Step 4: RR 계산 + Limit Order [완료]
- TP = Fib 레벨 (RR >= 2.0)
- SL = entry_price - (ATR * 1.5) - BUG FIX 적용됨
- **문제 발견**: TP1 거리 5-13% (너무 멂)

---

## 핵심 발견사항

### 백테스트 결과 (2021-01-01 ~ 06-30)
| 지표 | 값 |
|------|-----|
| Total Trades | 126 |
| Win Rate | 36.5% |
| Total PnL | **-$765.80** |
| $/Trade | -$6.08 |

### Exit별 분석
| Exit Type | 횟수 | Win Rate | 평균 PnL | 총 PnL |
|-----------|------|----------|----------|--------|
| **5m_Short_Div** | 46 | **97.8%** | +$24.57 | +$1,130 |
| SL | 48 | 0% | -$29.44 | -$1,413 |
| StaleLoss | 29 | 0% | -$16.68 | -$484 |
| TP1 | 1 | 100% | +$33.59 | +$34 |

### TP/SL 거리 분석
| 지표 | 값 | 문제 |
|------|-----|------|
| SL 거리 | 1.5-2.5% | 정상 |
| TP1 거리 | **5-13%** | **너무 멂** |
| TP1 히트율 | **1%** | 126개 중 1개만 도달 |

### MFE 분석
- 86%의 트레이드가 **한때 수익**이었음
- 14%만 "바로 손실" (straight to loss)
- 문제: **수익을 지키지 못함** (TP 너무 멂, Trailing Stop 없음)

---

## 근본 원인

```
문제: TP가 5-13% 거리 → 거의 도달 못함
현실: 가격이 2-3% 오르면 반전 → SL 또는 StaleLoss

수익: 5m_Short_Div +$24.57 × 46회 = +$1,130
손실: SL -$29.44 × 48회 = -$1,413
      StaleLoss -$16.68 × 29회 = -$484
──────────────────────────────────────────
총합: -$766
```

---

## 해결 옵션 (고도화용)

### Option A: 5m_Short_Div 전용 [선택됨]
- TP 제거, 5m 반대 다이버전스로만 청산
- 97.8% 승률 활용

### Option B: Trailing Stop 도입 [미래 고도화]
- MFE 50% 도달 시 트레일링 스탑 활성화
- 수익 보존

### Option C: SL 축소
- sl_atr_mult: 1.5 → 1.0
- 손실 감소

### Option D: TP 축소 (ATR 기반)
- tp_mode: "atr"
- tp_atr_mults: [2.0, 3.0, 4.0]

---

## 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `configs/mode47.json` | Hilbert Filter 활성화 |
| `scripts/backtest_strategy_compare.py` | SL 재계산 버그 수정 (line 3599-3600) |
| `docs/TRADE_FLOW_CHECKLIST.md` | 검증 결과 업데이트 |

---

## 다음 작업

1. **Option A 구현**: 5m_Short_Div 전용 전략 테스트
2. Step 5-7 검증 (포지션 진입/관리/청산)
3. **미래 고도화**: Trailing Stop 구현 (Option B)

---

## 태그
#MODE47 #TradeFlow #EntryGate #Hilbert #ProbGate #5mShortDiv #TP문제 #MFE분석
