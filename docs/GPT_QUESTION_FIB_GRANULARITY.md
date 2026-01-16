# GPT 질문: 1W Fib 좌표계 vs 15m 매매 간격 문제

## 배경

Mean-Reversion 비트코인 트레이딩 봇 개발 중.

### 현재 구조

**1W 피보나치 앵커 (고정)**:
```
Fib 0 = $3,120
Fib 1 = $20,650
Range = $17,530

현재가 $95,000 → Fib 5.24
```

**Zone 생성 방식**:
1. Fib Zone: 1W Fib 레벨 ± ATR Zone Width
2. Divergence Zone: RSI 다이버전스 바운더리 (이분탐색으로 계산)
3. **Cluster Zone = Fib Zone ∩ Divergence Zone** (겹치는 곳이 진입 후보)

**TF별 구조**:
| Role | TF | 역할 |
|------|-----|------|
| Context | 1W, 1D, 4H, 1H | 프랙탈 좌표 + 레짐 |
| Anchor | 15m | Zone 생성 |
| Trigger | 5m | 진입 타이밍 |

---

## 문제

### 1W Fib 레벨 간격이 15m 매매에 너무 넓음

**현재 레벨 간격** (현재가 $95k 근처):
```
L0 레벨 간격:
  Fib 5.0   = $90,770
  Fib 5.236 = $94,908  (차이: $4,138)
  Fib 5.382 = $97,469  (차이: $2,561)
  Fib 5.5   = $99,535  (차이: $2,066)

L1 레벨 간격: ~$1,000 ~ $1,500

15m ATR: ~$400 ~ $600
```

**문제점**:
- L0 레벨 사이에 ATR 8~10개 → 15m 스캘핑에 너무 성긴 그리드
- 레벨이 없는 구간에서 기회를 놓칠 수 있음

---

## 선택지

### 옵션 A: 프랙탈 깊이 증가 (L2, L3까지)

```
L0: Fib 5.0, 5.236, 5.382, 5.5, 5.618, 5.786, 6.0
L1: L0 셀 내부를 다시 Fib 분할
L2: L1 셀 내부를 다시 Fib 분할
...
```

**장점**: 순수 수학적 좌표계 유지
**단점**: 레벨 수가 기하급수적 증가, 의미 희석

### 옵션 B: 15m ZigZag 기반 동적 Fib 추가

```
1W Fib: 전체 좌표계 (고정)
15m Fib: 최근 스윙 기반 Fib (동적)
→ 둘이 겹치는 곳 = 강한 Zone
```

**장점**: 현재 시장 구조 반영
**단점**:
- ZigZag는 미래값 문제 (confirm_ts 필요)
- 동적이라 백테스트 시 룩어헤드 위험

### 옵션 C: Fib = 영역, Divergence = 정밀 바운더리 (현재 방향)

```
1W Fib: "이 근처 $90k~$95k 영역이 의미있다" (대략적 좌표)
Divergence: "정확히 $92,150 이하에서 다이버전스 성립" (정밀 진입가)

→ Fib는 "어디를 볼지" 가이드
→ Divergence가 "정확히 어디서" 결정
```

**장점**:
- Fib 간격 문제 해결 (Divergence가 정밀 타점 제공)
- 두 가지 독립 근거가 겹침 = 강한 신호

**단점**:
- Fib Zone이 넓으면 "의미있는 영역" 판단이 모호
- Divergence 없는 구간은 매매 불가?

### 옵션 D: TF별 다른 Fib 앵커

```
1W Fib: 전체 사이클 좌표 ($3,120 ~ $20,650)
1D Fib: 최근 큰 스윙 (예: $73k ~ $108k)
4H Fib: 중간 스윙
1H Fib: 단기 스윙
```

**장점**: 멀티 TF Confluence
**단점**: 동적 Fib = 룩어헤드 위험, 복잡도 증가

---

## 질문

1. **15m 매매에서 1W Fib 좌표계만으로 충분한가?**
   - 옵션 C처럼 Divergence가 정밀 타점을 제공하면 괜찮을까?
   - 아니면 더 세분화된 Fib가 필요한가?

2. **멀티 TF Fib가 필요하다면 어떻게 구현해야 룩어헤드 없이 가능한가?**
   - confirm_ts 기반 ZigZag로 가능?
   - 아니면 고정된 historical swing만 사용?

3. **권장하는 구조는?**
   - 옵션 A, B, C, D 중 어떤 게 Mean-Reversion에 적합한가?

---

## 추가 맥락

### Divergence Zone 계산 방식
```python
# 과매도 세그먼트에서 참조점 설정
ref_price = segment_min_close
ref_rsi = rsi_at(ref_price)

# Regular Bullish 바운더리 (이분탐색)
# "현재가가 여기 이하면 RSI가 ref_rsi보다 높음 → Regular Bullish"
regular_boundary = bisect_find(price < ref_price AND rsi > ref_rsi)

# Hidden Bullish 바운더리 (이분탐색)
# "현재가가 이 범위면 RSI가 ref_rsi보다 낮음 → Hidden Bullish"
hidden_range = bisect_find(price > ref_price AND rsi < ref_rsi)
```

### Cluster Zone = Fib ∩ Divergence
```
Fib Zone: $91,200 ~ $92,500
Regular Boundary: ≤ $92,100
→ Cluster Zone: $91,200 ~ $92,100 (Long 진입 후보)
```

---

## 원하는 답변

1. 15m 매매에 적합한 Fib 구조 추천
2. 멀티 TF Fib 사용 시 주의점
3. Divergence만으로 정밀 타점을 잡는 게 충분한지
4. 백테스트에서 검증할 포인트
