# Session Backup - 2026-01-16 15:51 (Updated)

## 작업 요약

### 목표
- RSI Divergence + Fibonacci 트레이딩 전략 최적화
- 이전 수익 모델 (bc3e19a: 16 trades, +$28.92) 재현

### bc3e19a 설정 분석 결과

**이전 대화 transcript (fa57fc86-16b6-4a2b-9467-5cecb39307fe.jsonl) 분석:**
- Line 3247에서 +$28.92 결과 발생
- Line 3779에서 sl_atr_mult 1.5→2.4 변경 (bc3e19a는 변경 전이므로 1.5)

**bc3e19a 정확한 설정:**
```python
# Config
sl_atr_mult: float = 1.5          # SL = entry ± 1.5*ATR
rsi_period: int = 14
stoch_rsi_period: int = 26

# ATR 계산
ATR period = 14

# 다이버전스
- Regular 다이버전스만 사용 (Hidden 제외)
- find_oversold_reference (원래 함수 사용, hybrid 아님)

# 진입 로직
- 15m StochRSI 상태 전환 감지 (과매도 진입 순간)
- Fib Level 근처 체크 (±1%)
- 3봉 이내 반등 확인 후 시장가 진입

# 청산 로직
- SL: ATR * 1.5
- TP: 5m Regular 다이버전스
```

### 테스트 결과 비교

| 버전 | 트레이드 | 승률 | PnL | SL 차이 |
|------|---------|------|-----|---------|
| **bc3e19a (목표)** | 16 | 50% | +$28.92 | $172 |
| **현재 (재현)** | 16 | 43.8% | +$13.14 | $164 |
| Hidden 포함 | 16 | 31.2% | -$60.92 | - |
| hybrid 함수 | 4 | 50% | +$26.28 | - |

### 차이점 분석
- 첫 번째 SHORT 트레이드:
  - bc3e19a: 5m_Long_Div 청산 (+$2.43)
  - 현재: SL 청산 (-$13.04)
- SL이 $8 더 타이트해서 SL에 먼저 걸림
- 나머지 트레이드는 동일

### 수정된 파일
- `scripts/backtest_strategy_compare.py`
  - sl_atr_mult: 2.4 → 1.5
  - ATR period: 21 → 14
  - Hidden 다이버전스 제거
  - hybrid 함수 → find_oversold_reference

## Hidden 다이버전스 분석

**결론: Hidden은 노이즈가 많아서 사용하지 않음**

- Regular Bullish: 가격↓ RSI↑ → 명확한 반전 신호
- Hidden Bullish: 가격↑ RSI↓ → "추세 지속"이지만 실전에서 가짜 신호 많음

| 설정 | 승률 | PnL |
|------|------|-----|
| Regular만 | 50% | +$28 |
| Regular+Hidden | 31% | -$60 |

## 다음 작업
- 다른 기간 (2021-12, 2022-01 등) 백테스트
- 파라미터 민감도 분석
