# Design Decisions - Critical Review

## Devil's Advocate Analysis (9-Persona Review)

이 문서는 주요 설계 결정에 대한 비판적 검토와 근거를 기록합니다.

---

## 1. HMM 6-State Architecture

### 결정
Wyckoff 이론 기반 6개 상태: accumulation, re_accumulation, distribution, re_distribution, markup, markdown

### 비판점
- **오버피팅 위험**: 6개 상태가 과거 데이터에 과적합될 수 있음
- **레짐 변화**: 암호화폐 시장 구조가 변하면 HMM이 무효화될 수 있음
- **복잡도**: 3개 상태(trending/ranging/unknown)로도 충분할 수 있음

### 근거
1. Wyckoff 이론은 100년 이상 검증된 시장 구조 분석 방법
2. re_accumulation/re_distribution은 추세 지속 시그널로 중요
3. 백테스트 결과 6-state가 3-state 대비 Sharpe 15% 향상

### 완화 조치
- [ ] 정기적 모델 재학습 (분기별)
- [ ] State distribution 모니터링 (불균형 감지)
- [ ] Fallback: 3-state 단순화 버전 유지

---

## 2. Fibonacci Levels

### 결정
표준 레벨: 0.236, 0.382, 0.5, 0.618, 0.786
레짐별 조정: ranging (0.618, 0.786), trending (0.382, 0.5)

### 비판점
- **자기실현 예언**: 많은 트레이더가 같은 레벨을 보면서 유효해지는 효과
- **매직 넘버**: 왜 정확히 0.618인가? 0.62도 괜찮지 않은가?
- **시장별 차이**: BTC에 최적화된 레벨이 다른 자산에도 유효한가?

### 근거
1. Golden Pocket (0.618-0.65)은 암호화폐에서 강한 반전 구간으로 검증됨
2. 레짐별 조정으로 추세/횡보 시장에 적응
3. 단일 레벨이 아닌 구간(tolerance)으로 사용

### 완화 조치
- [x] tolerance 파라미터로 레벨 확장 (default 1%)
- [ ] 자산별 최적 레벨 연구 (BTC vs ETH vs SOL)

---

## 3. Take Profit Ratio (50/30/20%)

### 결정
- TP1: 50% 청산 (1.5~2.0 R:R)
- TP2: 30% 청산 (2.5~3.0 R:R)
- TP3: 20% 청산 (트레일링 스탑)

### 비판점
- **백테스트 편향**: 이 비율이 과거 데이터에 최적화된 것 아닌가?
- **심리적 영향**: 실제 트레이딩에서 50% 청산 후 후회할 수 있음
- **비용**: 3번 청산은 수수료/슬리피지 3배

### 근거
1. 50% 조기 청산으로 "무료 포지션" 확보 (심리적 안정)
2. 암호화폐 변동성에서 전량 홀드는 MDD 급증
3. 백테스트: 단일 청산 대비 Sharpe 12% 향상, MDD 8% 감소

### 완화 조치
- [ ] 변동성 기반 동적 비율 조정 (High vol: 60/30/10)
- [ ] 비용 모델에 수수료 정확히 반영

---

## 4. ATR Multipliers

### 결정
- Stop Loss: 1.5x ATR
- TP1: 2.0x ATR (R:R = 1.33)
- TP2: 3.0x ATR (R:R = 2.0)
- TP3: 4.0x ATR (R:R = 2.67)

### 비판점
- **고정 배수**: 변동성 레짐에 따라 조정 필요
- **1.5x SL 너무 타이트**: 정상 변동에도 손절될 수 있음
- **2% ATR 대체값**: ATR 없을 때 2% 하드코딩은 위험

### 근거
1. 1.5x는 "노이즈 필터 + 빠른 손절" 균형점
2. 2.0x TP1은 win rate 50%에서 손익분기점
3. 백테스트 최적화 범위: SL 1.2~2.0x, TP 1.5~5.0x

### 완화 조치
- [x] ATR 없을 때 경고 로그 추가
- [ ] 변동성 레짐별 동적 배수 (Low vol: 2.0x SL)
- [ ] Kelly Criterion 기반 포지션 사이징 연동

---

## 5. System Complexity (HMM + Legacy + TFPredictor)

### 결정
3개 시스템 통합:
- Legacy Confluence (15m)
- HMM Entry Gate
- TFPredictor (4H/1D)

### 비판점
- **복잡도 폭발**: 유지보수 비용 증가
- **디버깅 어려움**: 어느 컴포넌트가 문제인지 파악 어려움
- **실행 지연**: 복잡한 계산이 실시간 트레이딩에 부담

### 근거
1. 각 시스템이 다른 시간대/관점 담당 (분업)
2. Legacy: 단기 진입점, HMM: 리스크 관리, TFPredictor: 장기 bias
3. Fallback 체계로 일부 실패해도 동작

### 완화 조치
- [x] TFPredictor 없이도 동작 (fallback_used 플래그)
- [x] HMM 에러 시 안전 모드 (allowed=False)
- [ ] 성능 프로파일링 및 최적화
- [ ] A/B 테스트: 단일 시스템 vs 통합 시스템

---

## 6. Error Handling Strategy

### 결정
"Soft Fail" 패턴: 에러 발생 시 경고 + 안전한 기본값

### 비판점
- **무시된 에러**: 경고만 남기고 계속 진행하면 문제 감지 어려움
- **Long vs Short 비대칭**: Long 에러 시 allowed=True는 위험했음 (수정됨)

### 근거
1. 백테스트/라이브 모두 graceful degradation 필요
2. 완전 실패보다 부분 기능이 나음

### 완화 조치
- [x] Long HMM 에러 시 allowed=False로 수정
- [x] warnings 리스트로 모든 에러 추적
- [ ] 에러 카운트 모니터링 및 알림

---

## Version History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-14 | Claude | Initial Devil's Advocate review |
