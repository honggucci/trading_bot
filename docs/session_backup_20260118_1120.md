# Session Backup - 2026-01-18 11:20

## 완료된 작업

### 1. Option B 구현 (5m Fallback 로직)
**파일**: `scripts/backtest_strategy_compare.py`

5m fallback에서 15m StochRSI 세그먼트를 5m 타임스탬프로 변환하여 REF를 찾도록 수정:

| Side | Line | 변경 내용 |
|------|------|----------|
| Long | 1619 | `find_oversold_reference_hybrid(df_15m_slice, df_5m_slice)` |
| Short | 1687 | `find_overbought_reference_hybrid(df_15m_slice, df_5m_slice)` |

**hybrid 함수 동작 방식**:
1. 15m StochRSI로 과매도/과매수 세그먼트 시간 구간 찾기
2. 해당 시간 구간의 5m 데이터 필터링
3. 5m에서 종가 최저점/최고점의 가격/RSI를 REF로 사용
4. 5m close array로 다이버전스 가격 계산

### 2. 백테스트 실행 (2021-11, RUN_MODE=0)

**결과**:
| 항목 | 값 |
|------|-----|
| 총 트레이드 | 144 |
| 승률 | 38.9% |
| 총 PnL | -$470 (-4.7%) |
| RR Ratio | 0.96 |

| Side | Trades | Win Rate | EV |
|------|--------|----------|-----|
| Long | 82 | 30.5% | -$5.27 |
| Short | 62 | 50.0% | -$0.62 |

**분석**:
- 2021년 11월은 하락장 (BTC $69k → $57k)
- Long 승률 30.5%가 핵심 문제 (역추세 진입)
- Short은 50% 승률로 손익분기

---

## 생성/수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `scripts/backtest_strategy_compare.py` | 5m fallback hybrid 함수 적용, RUN_MODE=0 |

---

## 핵심 로직 흐름

```
15m StochRSI ≤ 20 (과매도 진입)
    ↓
1) 15m 다이버전스 진입 시도
   - find_oversold_reference(df_15m_slice) → REF
   - needed_close_for_regular_bullish() → div_price
   - is_near_l1_level() → Fib 검증
   - bar['low'] ≤ div_price → Zone Touch
   - 성공 시 pending_long_signal, signal_tf='15m'

    ↓ (15m 실패 시)

2) 5m fallback (Option B)
   - find_oversold_reference_hybrid(df_15m, df_5m)
     → 15m 세그먼트 시간 → 5m 데이터 필터 → 5m REF
   - needed_close_for_regular_bullish(close_arr_5m, ...)
   - 성공 시 pending_long_signal, signal_tf='5m'
```

---

## 다음 작업 제안

1. **RUN_MODE=5 테스트**: Hilbert 레짐 필터 적용 (BEAR에서 Long 차단)
2. **다른 기간 테스트**: 상승장 (2021-01 등)에서 성능 확인
3. **파라미터 튜닝**: StochRSI 임계값, Fib tolerance 조정

---

## RAG Embedding

- **Project**: trading_bot
- **Keywords**: StochRSI, RSI Divergence, 5m fallback, Option B, hybrid reference, backtest
