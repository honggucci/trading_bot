# Session Backup - 2026-01-23 21:00

## 세션 요약

### 완료된 작업

1. **StochRSI 단위 테스트** (`scripts/test_stoch_rsi.py`)
   - 과매도 비율: 17.5% (252/1,441 바)
   - state 모드 적용 확인

2. **ZigZag 단위 테스트** (`scripts/test_zigzag_fib.py`)
   - 1년치 1W 데이터에서 14개 피봇 감지
   - 최종 앵커: Low=$80,600, High=$116,400

3. **ZigZag 피봇 검증** (`scripts/test_zigzag_verify.py`)
   - 4개 피봇 모두 실제 스윙으로 확인
   - confirm_date 기준 봉 선택 로직 수정

4. **ZigZag 피봇 과다 문제 분석**
   - 문제: 14개/년 (너무 많음, 기대: 4-6개)
   - 원인: `reversal_atr_mult=1.5` → threshold $14k로 12-14% 작은 스윙까지 잡힘
   - 해결책: Hilbert amplitude 기반 동적 k 계산

5. **체크리스트 영구 저장**
   - 파일: `docs/MODE78_CHECKLIST.md`
   - 진행률: 10/19 (53%)

---

## 생성/수정된 파일

### 새 파일
- `scripts/test_stoch_rsi.py` - StochRSI 단위 테스트
- `scripts/test_zigzag_fib.py` - ZigZag Fib 단위 테스트
- `scripts/test_zigzag_dates.py` - 피봇 날짜 확인
- `scripts/test_zigzag_sequence.py` - 스윙 시퀀스 분석
- `scripts/test_zigzag_verify.py` - 피봇 스윙 검증
- `docs/MODE78_CHECKLIST.md` - 검증 체크리스트 (영구 저장)

### 수정된 파일
- `configs/mode78.json` - stoch_signal_mode: "state"로 변경

---

## 핵심 체크리스트 진행 상황

| Step | 항목 | 상태 |
|------|------|------|
| 1.1 | OHLCV 데이터 로딩 | ✅ |
| 1.2 | Warmup 지표 초기화 | ⬜ |
| 1.3 | ATR 계산 | ⬜ |
| 2.1 | ZigZag pivot 감지 | ⚠️ 너무 많음 |
| 2.2 | pivot lookahead 없음 | ✅ |
| 2.3 | Fib high/low 합리성 | ⬜ k 조정 필요 |
| 3.1 | StochRSI 계산 | ✅ |
| 3.2 | Oversold/Overbought | ✅ |
| 3.3 | Regular/Hidden Div | ⬜ |
| 3.4 | 다이버전스 가격 | ⬜ |
| 4.1 | Dynamic Fib 레벨 | ⬜ |
| 4.2 | Zone 체크 | ⬜ |
| 5.1 | Entry 계산 (RR 2:1) | ✅ |
| 5.2 | SL 계산 (Micro SL) | ✅ |
| 5.3 | TP 계산 (ATR) | ✅ |
| 6.1 | Fill 조건 | ⬜ |
| 6.2 | Fill 가격 | ⬜ |
| 7.1 | SL Hit | 🔴 버그 |
| 7.2 | TP Hit | ✅ |
| 7.3 | Partial Exit | ✅ |
| 7.4 | Trailing Stop | ⬜ |

---

## 다음 작업

1. **Step 2.3: Dynamic ZigZag k 구현**
   - `src/context/dynamic_fib_anchor.py`에 `compute_dynamic_k()` 함수 추가
   - Hilbert amplitude 기반 동적 k 계산
   - 목표: 피봇 14개 → 4-6개

2. **Step 7.1: SL 버그 수정**
   - gap 시 `exit_price = max(sl, bar['low'])` 적용

---

## 핵심 발견

### ZigZag k 동적 계산 원리 (Hilbert 기반)

```python
# 진폭/ATR 비율 기반
cycle_strength = A_t / ATR_1W
k_t = clip(q * cycle_strength, k_min, k_max)

# 또는 레짐 스위치
if amplitude_z >= amp_threshold:  # 사이클 장
    k = 1.5  # 민감
else:  # 추세/노이즈 장
    k = 3.0  # 둔감
```

- k는 주기(frequency)가 아니라 **진폭(amplitude)**에서 도출
- 기존 `src/regime/wave_regime.py`의 `WaveRegimeClassifier` 활용

---

## 테스트 결과

### StochRSI (2026-01-01 ~ 01-23)
- 총 15m 바: 1,441
- 과매도 바 (stoch_d <= 20): 252 (17.5%)
- 과매도 진입 횟수: 114

### ZigZag (2025-01-01 ~ 2026-01-23)
- 총 1W 바: 54
- 피봇 수: 14 (너무 많음)
- ATR 평균: $9,412
- Threshold (1.5×ATR): $14,118

---

## RAG 임베딩 정보

- **Project**: trading_bot
- **Session**: 2026-01-23 ZigZag 단위 테스트 및 체크리스트 정리
- **Key Topics**: ZigZag, Hilbert amplitude, dynamic k, StochRSI, MODE78 checklist
