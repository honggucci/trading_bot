# MODE78 세션 백업 (2026-01-24)

## 완료된 작업 요약

### P0 신뢰성 체크 (6/6 PASS)
1. **P0-1 Hilbert 인과성**: `classify_series_causal()` 사용 확인
2. **P0-2 SL 갭 체결**: LONG `min(sl, open)`, SHORT `max(sl, open)` 보수적 모델
3. **P0-3 Entry/TP Invariant**: `offset_ratio` 모드로 정순 계산
4. **P0-4 SL Source 단일화**: Micro SL 적용
5. **P0-5 k-anchor 구현**: extreme_ts 분리 저장 + atr_anchor 고정
6. **P0-5b Pending Reversal**: min_spacing 부족 시 pending 상태 머신

### ZigZag 피봇 최적화
- 14개 → 3개 피봇 (min_spacing 13주)
- OHLC 정합 ALL PASS
- median(spacing_extreme) = 18.5주

### 매매 로직 검증 (20/20 항목 완료)
- Step 1: 데이터 로딩 & Warmup
- Step 2: Dynamic Fib Anchor (1W)
- Step 3: 다이버전스 신호 감지 (15m)
- Step 4: Fib Zone 확인
- Step 5: Limit Order 생성
- Step 6: Limit Order Fill
- Step 7: 포지션 관리

---

## 생성/수정된 파일 목록

### Config
- `configs/mode78.json` - MODE78 v2 설정 (L0 컨텍스트 + L1 실행)

### Scripts
- `scripts/show_zigzag_pivots.py` - ZigZag pivot extreme_ts vs confirm_ts 분리 테스트
- `scripts/show_failed_trades.py` - SL 청산 매매 분석
- `scripts/test_hilbert_causality.py` - Hilbert 인과성 테스트
- `scripts/test_pending_reversal.py` - Pending Reversal 상태 머신 테스트

### Source
- `src/context/dynamic_fib_anchor.py` - 수정됨
  - `DynamicFibAnchorState`에 pending 필드 7개 추가
  - `compute_dynamic_k()` 함수 구현
  - `update_anchor_zigzag()`에 pending 상태 전이 로직

### Docs
- `docs/MODE78_CHECKLIST.md` - 검증 체크리스트 (100% 완료)

---

## MODE78 v2 Config (최종)

```json
{
  "_description": "MODE 78 v2: L0 컨텍스트 + L1 실행",
  "use_fib_based_sl": false,
  "use_micro_sl": true,
  "micro_sl_tf": "1h",
  "micro_sl_buffer_mult": 0.5,
  "use_5m_div_exit": false,
  "use_15m_div_exit": true,
  "tp_min_partial_pct": 0.5,
  "rr_limit_ttl": "168h"
}
```

---

## 발견된 문제점 (MODE79로 해결 예정)

### 스케일 불일치 문제
- **문제**: 15m 다이버전스 신호 + 1H swing SL (0.3~0.7%)이 노이즈에 자주 털림
- **해결책**: Fib 레벨을 청산가로 사용 + 격리 마진으로 리스크 관리

### MODE79 핵심 개념
- Entry: 15m 과매도 다이버전스 + Fib 레벨 근처
- SL: 바로 아래 Fib = 청산가
- 레버리지: 100% / (Entry - 아래Fib 거리%)
- 마진: 계좌의 2% (격리)

---

## 테스트 결과

### P0-1 Hilbert 치팅 검출
```
hilbert        : PASS (Online != Batch) (avg_diff=1.012)
regime_switch  : PASS (Online != Batch) (avg_diff=0.450)
inverse        : PASS (Online != Batch) (avg_diff=1.168)
```

### P0-5 ZigZag Pivot
```
#1: 2025-01-20 | HIGH | $109,588 | OHLC PASS
#2: 2025-04-14 | LOW  | $83,112  | OHLC PASS
#3: 2025-10-06 | HIGH | $126,200 | OHLC PASS
spacing_extreme: median 18.5주 >= 13주 목표
```

### P0-5b Pending Reversal
```
PENDING_START:   2 times
PENDING_CONFIRM: 2 times
PENDING_CANCEL:  0 times
OHLC: ALL PASS (3/3)
```

---

## 다음 작업

MODE79 구현:
1. `configs/mode79.json` 생성
2. Fib 거리/레버리지 계산 함수 추가
3. 격리 마진 포지션 계산 함수 추가
4. Entry/Exit 로직에 MODE79 분기 추가

---

## 백업 날짜
2026-01-24 (금)
