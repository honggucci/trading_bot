# Session Backup: PR-DYN-FIB 구현

**날짜**: 2026-01-19
**프로젝트**: trading_bot
**RAG Embedding**: trading_bot

---

## 완료된 작업

### 1. PR-FIB (이전 세션에서 완료)
- 1W 정적 Log Fibonacci 지원
- ATR-based tolerance 계산
- MODE42 생성

### 2. PR-ENTRY-RR2 (이전 세션에서 완료)
- RR Limit Entry 구현 (수동매매 방식)
- `solve_entry_for_rr_net()` 함수
- `simulate_limit_fill()` 함수
- MODE43 생성 (현재 챔피언, +$1.30/trade)

### 3. PR-DYN-FIB (이번 세션)
- **15m 동적 Linear Fibonacci** 구현
- 3가지 앵커 갱신 모드: ZigZag, Rolling, Conditional
- TP 후보 확장에 동적 레벨 추가

---

## 생성/수정된 파일

### 신규 생성

| 파일 | 용도 |
|------|------|
| `src/context/dynamic_fib_anchor.py` | 동적 Fib 앵커 모듈 |
| `configs/mode44.json` | ZigZag 동적 Fib 모드 |
| `configs/mode45.json` | Rolling 동적 Fib 모드 |
| `configs/mode46.json` | Conditional 동적 Fib 모드 |
| `scripts/test_dynfib_modes.py` | 4-way 비교 테스트 스크립트 |

### 수정

| 파일 | 변경 내용 |
|------|----------|
| `scripts/backtest_strategy_compare.py` | PR-DYN-FIB Config 추가, import, 앵커 갱신 로직, TP 후보 확장 |

---

## 테스트 결과

### PR-ENTRY-RR2 (3-way 비교)
```
MODE36 (Champion): 48 trades, $-205.84, -$4.29/trade
MODE42 (Log Fib): 23 trades, $-29.01, -$1.26/trade
MODE43 (RR Limit): 28 trades, $+36.38, +$1.30/trade  ★ WINNER

VERDICT: MODE43 is $5.59/trade better than MODE36
```

### PR-DYN-FIB (4-way 비교)
```
MODE43 (Champion): 28 trades, $36.38, $1.30/trade
MODE44 (ZigZag): 28 trades, $36.38, $1.30/trade
MODE45 (Rolling): 28 trades, $36.38, $1.30/trade
MODE46 (Conditional): 28 trades, $36.38, $1.30/trade

VERDICT: [NO CHANGE] - 동적 Fib가 TP 선택에 영향 없음
```

---

## 구현 상세

### DynamicFibAnchorState
```python
@dataclass
class DynamicFibAnchorState:
    low: float                    # 앵커 저점
    high: float                   # 앵커 고점
    last_update_ts: pd.Timestamp  # 마지막 갱신 시각
    mode: str                     # "zigzag" | "rolling" | "conditional"
    direction: str                # "up" | "down" | "unknown"
    last_pivot_price: float       # 마지막 확정 pivot
```

### 앵커 갱신 함수
- `update_anchor_zigzag()`: ATR 기반 reversal로 pivot 확정
- `update_anchor_rolling()`: N봉 rolling high/low
- `update_anchor_conditional()`: range >= N*ATR 조건부 갱신

### Config 파라미터
```python
use_dynamic_fib: bool = False
dynamic_fib_mode: str = "rolling"  # "zigzag" | "rolling" | "conditional"
dynfib_lookback_bars: int = 96     # 24h
dynfib_reversal_atr_mult: float = 1.5
dynfib_use_as: str = "tp_candidate"  # "tp_candidate" | "entry_filter"
```

---

## 분석 및 인사이트

### 왜 동적 Fib가 효과 없었나?

1. **Macro Fib 레벨이 이미 충분**
   - 1W Log Fib 레벨이 RR 조건을 만족하는 TP로 선택됨
   - 동적 레벨이 추가되어도 더 가까운 유효 TP가 없음

2. **RR 필터링 영향**
   - min_rr_net = 2.0 조건으로 필터링
   - 동적 레벨이 entry에 너무 가까워서 RR 미달

3. **테스트 구간 특성**
   - 테스트 구간에서 동적 스윙이 Macro 구조와 유사

### 향후 개선 방향

1. **Entry Filter 모드 테스트** (`dynfib_use_as = "entry_filter"`)
2. **Confluence 모드** (`dynfib_confluence_with_macro = true`)
3. **더 짧은 lookback** (96봉 → 48봉)으로 더 밀접한 동적 레벨 생성

---

## 현재 챔피언

**MODE43** = Log Fib + ATR tolerance + RR Limit Entry
- **+$1.30/trade** (MODE36 대비 +$5.59 개선)
- 동적 Fib 추가는 현재 설정에서 효과 없음

---

## 다음 작업 제안

1. **Entry Filter 모드 검증** - 동적 레벨 근처에서만 진입
2. **Confluence Zone 검증** - Macro + Dynamic 겹침 구간 분석
3. **다른 구간 테스트** - 2024-2025 최근 데이터로 검증
4. **파라미터 튜닝** - lookback, reversal_mult 조정

---

## 버전 정보

- **v3.3**: PR-DYN-FIB 구현 완료 (2026-01-19)
- 이전: v3.2 (PR-FIB + PR-ENTRY-RR2)
