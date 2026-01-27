# MODE47 전략 분석 요약 (GPT 리뷰용)

## 1. 전략 개요

**목적**: BTC/USDT 15m 스윙매매 (Long Only)

**핵심 로직**:
1. 1D ZigZag 기반 Dynamic Fib 레벨 생성
2. StochRSI < 30 (과매도) + Fib 레벨 근처 + RSI Divergence → 시그널
3. ProbGate 필터 (p_bull > 0.55)
4. Limit Order로 더 좋은 진입가 대기 (Entry Offset 30%)
5. Fib 기반 SL, 다음 Fib 레벨을 TP로 설정

---

## 2. 현재 설정 (configs/mode47.json)

```json
{
  "stoch_rsi_oversold": 30.0,
  "stoch_rsi_period": 14,
  "early_exit_duration": "8h",
  "use_hilbert_filter": false,
  "use_trend_filter_1h": false,
  "use_prob_gate": true,
  "prob_gate_thr_long": 0.55,
  "use_rr_limit_entry": true,
  "rr_limit_ttl": "6h",
  "dynfib_lookback_bars": 100,
  "use_trailing_stop": true,
  "trailing_activation_atr": 2.0
}
```

Entry Offset: **0.3** (TP 거리의 30% 아래에서 Limit 진입)

---

## 3. 백테스트 결과 (2021-01-01 ~ 2021-06-30)

### 현재 설정
| 지표 | 값 |
|------|-----|
| Trades | 46 |
| Total PnL | +$288 |
| $/Trade | +$6.26 |
| Win Rate | 60.9% |

### 비교 테스트 결과

| 테스트 | Trades | PnL | WR | 결론 |
|--------|--------|-----|-----|------|
| ProbGate OFF | 224 | -$765 | 41.5% | 필요함 |
| prob_gate_thr: 0.45 | 82 | +$245 | 52.4% | 0.55 유지 |
| Market Order (no limit) | 132 | -$64 | 42.4% | Limit 유지 |
| **현재 설정** | **46** | **+$288** | **60.9%** | 최적 |

---

## 4. 필터 퍼널 분석

```
476개 시그널 (Divergence + Fib + StochRSI)
    │
    ▼ ProbGate (p_bull > 0.55)
199개 통과 (58% 차단)
    │
    ▼ Limit Order 체결 (6h TTL)
46개 트레이드 (82% 미체결/만료)
```

---

## 5. 청산 분석 (핵심 문제점)

### Exit Reason Breakdown
| 청산 이유 | 횟수 | WR | Avg PnL |
|-----------|------|-----|---------|
| 5m_Div_Partial (Trailing) | 11 | 100% | +$13.74 |
| SL Hit | 35 | 48.6% | +$3.06 |

### 문제점
- **TP 도달: 0%** (TP1/TP2/TP3 한 번도 안 침)
- 모든 이익은 Trailing Stop에서 나옴
- SL이 너무 타이트 (0.3%~2%)
- TP가 너무 멀음 (5%~6%)

### 샘플 트레이드
| Entry | SL | TP1 | SL dist | TP dist | 문제 |
|-------|-----|-----|---------|---------|------|
| $33,979 | $33,872 | $35,842 | 0.3% | 5.5% | SL 너무 타이트 |
| $35,290 | $35,309 | $37,029 | -0.05% | 4.9% | SL > Entry (버그?) |

---

## 6. ProbGate 로직 (참고)

```
1. 1H Hilbert Score 계산 (Hilbert Transform)
2. Temperature = f(ATR volatility)
   - 변동성↑ → T↑ → 확률 0.5로 눌림
3. p_bull = sigmoid(Hilbert_score / Temperature)
4. p_bull > 0.55 → LONG 허용
```

---

## 7. 질문/개선 방향

1. **SL/TP 비대칭 문제**
   - SL이 너무 타이트해서 노이즈에 걸림
   - TP는 너무 멀어서 영영 못 닿음
   - 어떻게 조정해야 할까?

2. **WR 60%의 한계**
   - 엄청난 필터를 거쳐도 60%밖에 안 됨
   - 근본적인 시그널 품질 문제?

3. **Trailing Stop 의존**
   - 이익이 전부 Trailing에서 나옴
   - TP 없이 Trailing만으로 갈 수도?

4. **트레이드 수 vs 품질**
   - 6개월에 46개 (월 ~8개)
   - 더 많은 트레이드가 필요한가?

---

## 8. 코드 구조 (참고)

- `configs/mode47.json`: 설정 파일
- `scripts/backtest_strategy_compare.py`: 백테스트 엔진
- `src/context/dynamic_fib_anchor.py`: 1D ZigZag Fib 계산
- `src/regime/prob_gate.py`: ProbGate 로직

---

## 9. 요청사항

위 분석을 바탕으로:
1. SL/TP 설정 개선안 제안
2. 추가로 테스트해볼 만한 아이디어
3. 전략 로직의 근본적인 개선점
