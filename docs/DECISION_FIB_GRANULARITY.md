# Fib 좌표계 + Divergence 구조 (확정)

## 핵심 결정

> **옵션 C: Fib = 영역, Divergence = 정밀 타점**

---

## 1. 구조

```
1W Fib (고정 좌표계)
    └─ "어디를 볼지" 가이드
    └─ Zone = Fib level ± ATR width

Divergence (동적 바운더리)
    └─ "어디서 칠지" 정밀 타점
    └─ Regular/Hidden Bullish/Bearish 바운더리

Cluster Zone = Fib ∩ Divergence
    └─ 독립 근거 2개 겹침 = 강한 신호
```

---

## 2. Entry Grades

| Grade | 조건 | Size |
|-------|------|------|
| **A** | Fib Zone + Divergence | 1.0 (base) |
| **B** | Fib Zone only + 대체 트리거 | 0.25 ~ 0.5 |

### 대체 트리거 (Divergence 없을 때)
- Micro-structure reversal
- 스프레드/캔들 반전 패턴
- Z-score revert
- 5m absorption 이벤트

### 핵심 원칙
> **Divergence는 '필수'가 아니라 '강화 신호'**
> "Div는 스나이퍼 조준경이지, 총 자체가 아니다."

---

## 3. 멀티 TF Fib 사용법

### ❌ 잘못된 사용 (금지)
```
1D Fib도 보고, 4H Fib도 보고, 1H Fib도 보고...
→ Confluence 아님, 과최적화 지뢰
→ 진입 기회 감소 + 규칙 충돌
```

### ✅ 올바른 사용 (Veto/필터로만)
```
1D Fib가 현재가보다 훨씬 아래 (강한 상승 추세)
→ "롱 스카웃 금지" (Veto)
→ 추가 진입 조건이 아니라 거부권
```

---

## 4. 룩어헤드 방지 (ZigZag 사용 시)

멀티 TF Fib 쓸 경우 반드시:

### 옵션 1: Confirmed Pivot만 사용 (정석)
```python
# pivot_formed_ts: 실제 고점/저점 발생 시점
# pivot_confirm_ts: 확정 시점 (N봉 후)

# 앵커로 사용 가능한 pivot
valid_anchors = [p for p in pivots if p.confirm_ts <= current_ts]
```

### 옵션 2: 시간 지연 앵커 (보수적)
```
1D 앵커: 최소 2~3일 지연
4H 앵커: 최소 3~6봉 지연
```

### 옵션 3: 완전 고정 앵커 (현재 방식)
```
1W Fib: 역사적 스윙 한 번 박아두고 안 바꿈
→ 룩어헤드 걱정 0
→ Zone width + Trigger로 커버
```

---

## 5. 백테스트 검증 포인트

1. **Div 필수 vs Div 강화**
   - Div 필수: 트레이드 수 ↓, 승률 ↑?
   - Div 강화: 트레이드 수 ↑, 전체 수익?

2. **Grade B 대체 트리거 성과**
   - Div 없는 구간에서 알파가 생기나?
   - 아니면 노이즈인가?

3. **멀티 TF Fib Veto 효과**
   - MDD 개선?
   - Tail risk 감소?

4. **룩어헤드 방지 전후 비교**
   - confirmed pivot 적용 전후 성과 차이
   - 크게 변하면 = 사기였던 것

---

## 6. 왜 다른 옵션은 탈락인가

| 옵션 | 탈락 이유 |
|------|----------|
| A (L2, L3 세분화) | 레벨 폭발 + 의미 희석. 그리드 촘촘하다고 MR이 강해지지 않음 |
| B (15m ZigZag Fib) | 룩어헤드/불안정성. "백테 좋고 실전 개판" 1순위 |
| D (TF별 다른 앵커) | 진입 기회 감소 + 규칙 충돌. Veto로만 써야 함 |

---

## 버전

- v1.0 (2026-01-15): GPT 답변 기반 확정
