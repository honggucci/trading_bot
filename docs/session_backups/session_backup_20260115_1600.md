# Session Backup - 2026-01-15 16:00

## 세션 주제
**Zone Width 파라미터 최종 확정 + 재학습 프로토콜**

---

## 1. 완료된 작업

### 1.1 ATR 윈도우 최적화
- 피보나치 수열 윈도우 테스트: [8, 13, 21, 34, 55, 89]
- IS/OOS 분할 검증 (80/20)으로 과적합 방지
- TF별 최적 윈도우 결정

### 1.2 Zone Width 파라미터 확정

| TF | ATR Window | k | Role |
|----|------------|---|------|
| 1W | - | - | fib_coordinate_only |
| 1D | 89 | 1.0 | htf_filter |
| 4H | 21 | 1.65 | context_filter |
| 1H | 21 | 2.4 | context_filter |
| 15m | 21 | 2.75 | zone_generator |

### 1.3 1W Fib Anchor 확정
```
Fib 0: $3,120
Fib 1: $20,650
Range: $17,530
Scale: Linear
Source: Binance Futures (고정)
```

### 1.4 변동성 추정기 비교 실험
- ATR vs EWMA vs Yang-Zhang vs Realized Vol
- Fair comparison (50% coverage 정규화)
- 결과: ATR ≈ EWMA (0.7055 vs 0.7026)
- ATR 채택 (단순성)

### 1.5 Hilbert 진폭 실험
- Causal 버전 구현 후 테스트
- 결과: OOS 최하위 (Pearson 0.318)
- **폐기**

### 1.6 재학습 프로토콜 수립
- 트리거: 3개월 롤링 Coverage < 40% or > 60%
- 변경 조건: OOS Pearson +10% 이상 개선시만
- 모니터링 스크립트 생성

---

## 2. 생성/수정된 파일

### 신규 생성
| 파일 | 설명 |
|------|------|
| `config/zone_width.json` | Zone Width 운영 설정 v2.0.0 |
| `src/context/volatility.py` | 변동성 추정기 (ATR, EWMA, YZ, RV) |
| `scripts/experiment_atr_window_multi_tf.py` | 멀티TF ATR 윈도우 테스트 |
| `scripts/experiment_atr_window_critical.py` | IS/OOS 비판적 검증 |
| `scripts/experiment_atr_all_tf_oos.py` | 전체 TF OOS 검증 |
| `scripts/monitor_zone_coverage.py` | Coverage 모니터링 |

### 수정
| 파일 | 변경 내용 |
|------|----------|
| `src/context/fib_levels.py` | `fib_detrend()`, `fib_hilbert_amplitude()` 추가 |

---

## 3. 검증 결과

### Coverage 모니터링 (최근 3개월)
```
15m: 46.3% [OK]
 1h: 41.7% [OK]
 4h: 41.7% [OK]
 1d: 31.8% [ALERT - HTF 필터 역할이라 허용]
```

### OOS Correlation
```
1d_atr89: 0.22
4h_atr21: 0.29
1h_atr21: 0.40
15m_atr21: 0.42
```

---

## 4. 핵심 결정 사항

### 채택
1. ATR 기반 Zone Width (EWMA 대비 단순)
2. 1W는 Fib 좌표계만 (Zone Width 계산 제외)
3. 4H/1H는 Context Filter만 (트리거로 사용 금지)
4. 15m만 Zone Generator 역할

### 폐기
1. Hilbert 진폭 기반 변동성 예측
2. EMA detrend (Fib detrend로 대체)
3. 1W Zone Width 계산

---

## 5. 다음 작업 제안

### 즉시 (P0)
- [ ] 5m 트리거 품질 개선
  - Spring/UTAD Reclaim
  - Failed Swing
  - Effort vs Result (Absorption)

### 단기 (P1)
- [ ] 50~200 트레이드 샘플 축적
- [ ] A급/B급 세팅별 승률 분석

### 중기 (P2)
- [ ] HMM Gate + Zone 통합 테스트
- [ ] 실거래 시뮬레이션

---

## 6. GPT 피드백 요약

1. **1W 제외**: 찬성 (365봉, OOS 73봉 = 검증 불가)
2. **OOS 음수 원인**: 데이터 부족 + 레이블 정의/레짐 변화/시차 가능성
3. **4H/1H 역할**: Context Filter만 (존이 너무 많아지면 아무거나 트레이드)
4. **결론**: "또 수학 더 늘리면 돈 대신 파일만 늘어난다"

---

## 7. 운영 규칙

### Zone Width 변경 금지 사유
- 몇 번 손실
- 느낌
- 새 논문

### Zone Width 변경 허용 사유
- 3개월 Coverage 이탈 + OOS 재검증에서 +10% 개선

---

## 8. 관련 문서
- `config/zone_width.json` - 운영 설정
- `CLAUDE.md` - 프로젝트 규칙
- `docs/SPEC_v2_integration.md` - 통합 스펙

---

**작성**: Claude Code
**프로젝트**: trading_bot
**버전**: v3.3
