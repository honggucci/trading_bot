# GPT 질문: HMM Gate + Zone 연동 설계

## 배경

Mean-Reversion 기반 비트코인 트레이딩 봇을 만들고 있음.

### 핵심 철학
- 시장은 본질적으로 **횡보(레인지)**
- 레인지 극단에서 반대 방향 진입 (Mean-Reversion)
- Zone 상단 → Short, Zone 하단 → Long

---

## 현재 구조

### 1W Fib 좌표계 (고정)
```
Fib 0 = $3,120
Fib 1 = $20,650
Range = $17,530

현재가 $95,000 → Fib 5.24
```

### TF별 트레이딩 구조
| Anchor TF | Trigger TF | Zone Width |
|-----------|------------|------------|
| 15m | 5m | 15m ATR * 2.75 |
| 1H | 15m | 1H ATR * 2.4 |
| 4H | 1H | 4H ATR * 1.65 |
| 1D | 4H | 1D ATR * 1.0 |

### Zone 구조
- Fib 레벨 (고정 좌표계) + ATR Zone Width
- 프랙탈 구조: L0 (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0), L1 (L0 셀 내부 세분화)
- Zone = Fib Level ± (Zone Width / 2)

---

## 현재 HMM Gate

### 상태 (6개)
1. accumulation - 매집
2. re_accumulation - 재매집
3. distribution - 분배
4. re_distribution - 재분배
5. markup - 상승 추세
6. markdown - 하락 추세

### 현재 로직
```python
# Long Permit
long_allowed = state in ('markup', 'accumulation', 're_accumulation')

# Short Permit
short_allowed = (
    markdown_prob > 0.60 and
    trend_strength < -0.10
)
```

### 문제점
1. **Zone 위치 무시** - HMM 상태만으로 Long/Short 결정
2. **TF별 HMM 없음** - 15m HMM만 있음

---

## 문제 상황

### 충돌 케이스
```
Zone 하단 + markdown 상태 → ???
- Zone 기준: Long 해야 함 (하단 = 지지)
- HMM 기준: Short 해야 함 (하락 추세)

Zone 상단 + markup 상태 → ???
- Zone 기준: Short 해야 함 (상단 = 저항)
- HMM 기준: Long 해야 함 (상승 추세)
```

### 필요한 것
1. **Zone + HMM 조합 로직** - 둘이 충돌할 때 어떻게 할지
2. **TF별 독립 HMM** - 15m, 1H, 4H, 1D 각각 HMM

---

## 질문

### Q1. Zone과 HMM 조합 방법

Mean-Reversion 전략에서 Zone 위치와 HMM 상태를 어떻게 조합해야 할까?

옵션:
1. **Zone 우선** - Zone 위치가 진입 방향 결정, HMM은 필터만
   - Zone 하단 → Long만 허용, Short 금지
   - HMM이 markdown이어도 Long 시도 (단, 사이즈 축소)

2. **HMM 우선** - HMM이 방향 결정, Zone은 타이밍만
   - markup/accumulation → Long 찾기
   - markdown/distribution → Short 찾기
   - Zone 도달 시 진입

3. **둘 다 일치 필수** - Zone과 HMM이 같은 방향일 때만 진입
   - Zone 하단 + accumulation/markup → Long
   - Zone 상단 + distribution/markdown → Short
   - 불일치 시 → 대기 (No Trade)

4. **다른 방법?**

### Q2. TF별 HMM 역할

15m, 1H, 4H, 1D 각각 HMM이 있을 때, 어떻게 조합해야 할까?

현재 생각:
```
1D HMM: 전체 방향성 (macro bias)
4H HMM: Context 필터
1H HMM: Context 필터
15m HMM: Zone 진입 Gate
```

질문:
- 상위 TF HMM이 하위 TF를 override 해야 하나?
- 모든 TF가 일치해야 진입?
- 불일치 시 어떻게?

### Q3. Mean-Reversion vs Trend Following

Zone 하단 + markdown 상황:
- Mean-Reversion: "하단이니까 반등 기대" → Long
- Trend Following: "하락 추세니까 계속 하락" → Short 또는 대기

이 충돌을 어떻게 해결해야 할까?

---

## 추가 맥락

### Wyckoff 관점
- accumulation/re_accumulation = 매집 구간 (바닥 형성)
- distribution/re_distribution = 분배 구간 (천장 형성)
- markup = 상승 추세
- markdown = 하락 추세

### 내 생각
Mean-Reversion이 기본 전략이니까:
- Zone이 1차 필터 (위치)
- HMM이 2차 필터 (상태 확인)
- 둘이 충돌하면 대기?

근데 이러면 트레이드 기회가 너무 적을 수도...

---

## 원하는 답변

1. Zone + HMM 조합 로직 추천
2. TF별 HMM 조합 방법
3. 충돌 상황 처리 방법
4. 백테스트로 검증할 포인트
