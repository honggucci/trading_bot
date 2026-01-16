# HMM + Zone 연동 설계 (확정)

## 핵심 원칙

> **Zone은 "어디서", HMM은 "지금 해도 되냐"**
> 역할 뒤집으면 다른 전략이 된다.

---

## 1. Zone 우선 + HMM Veto

### 구조

```
Zone → 방향 결정 (절대 규칙)
HMM → 허용 여부 / 강도 조절 (Gate)
```

- Zone 하단 → Long 후보만
- Zone 상단 → Short 후보만
- **HMM에게 방향 결정권 주면 Trend Following 됨**

### 충돌 케이스 처리

| Zone | HMM | 행동 |
|------|-----|------|
| 하단 | accumulation / re_acc | **Strong Long** |
| 하단 | markup | **Weak Long (빠른 TP)** |
| 하단 | markdown | **Wait or Micro-Long (1/4 size)** |
| 상단 | distribution / re_dist | **Strong Short** |
| 상단 | markdown | **Weak Short** |
| 상단 | markup | **No Trade** |

---

## 2. TF별 HMM 역할 (Veto 구조)

### Override ❌ / Veto ⭕

```
1D HMM  : Regime Gate (시장 유형) - hostile이면 NO_TRADE
4H HMM  : Structural Context - hostile이면 size 50%
1H HMM  : Trade Quality Filter - conflict이면 size 50%
15m HMM : Execution Gate - confirm 안 되면 WAIT
```

### 로직

```python
direction = zone_direction  # immutable (Zone이 결정)

if hmm_1d in hostile_states:
    return NO_TRADE

size = base_size
if hmm_4h in conflict:
    size *= 0.5

if hmm_1h in conflict:
    size *= 0.5

if not hmm_15m_confirm:
    WAIT

ENTER(direction, size)
```

### 중요

- **모든 TF 일치 필수 ❌** → 트레이드 안 나옴
- **상위 TF는 방향을 안 바꿈** → 거부권만
- **하위 TF만 실제 트리거**

---

## 3. Zone 하단 + markdown 처리

### 본질

- markdown = 추세 진행 중
- Zone 하단 = 통계적 과매도

**둘 다 동시에 참일 수 있음**

### 처리

| 조건 | 행동 |
|------|------|
| markdown 강함 + momentum 유지 | **No Trade** |
| markdown 약화 + volatility 감소 | **Scout Long (1/4 size)** |
| markdown → re_acc 전환 | **Full Long** |

### 핵심

> Mean-Reversion은 '떨어질 때'가 아니라 **'떨어지다 멈출 때'** 먹는다.
> markdown에서 바로 롱 = 칼날 잡기

---

## 4. Decision Flow

```
1. Zone Check → Direction 결정
2. 1D HMM → Regime Veto
3. 4H HMM → Structure Filter
4. 1H HMM → Size Modulator
5. 15m HMM → Entry Gate
6. Trigger 확인 (Layer 4)
7. Execution (ATR 기반)
```

---

## 5. 백테스트 검증 포인트

1. **Zone 하단 + markdown**: 즉시 진입 vs 지연 진입 성과
2. **HMM veto 횟수 vs 손실 회피량**
3. **TF별 HMM 제거 실험**: 1D/4H/1H 각각 제거 시 성과
4. **Size modulation 효과**: full vs scaled entry

---

---

## 6. hostile / conflict / favorable 정의 (확정)

### 1D Regime Veto

| 상태 | 분류 | 이유 |
|------|------|------|
| accumulation | favorable | 레인지 |
| re_accumulation | favorable | 레인지 |
| distribution | favorable | 레인지 |
| re_distribution | favorable | 레인지 |
| markup (강) + trend 강 | **hostile** | MR Short 위험 |
| markdown (강) + trend 강 | **hostile** | MR Long 위험 |

**hostile이면 NO_TRADE**

### 4H / 1H conflict 정의

| Zone | HMM 상태 | 분류 |
|------|----------|------|
| 하단 | markdown | **conflict** |
| 하단 | 그 외 | favorable |
| 상단 | markup | **conflict** |
| 상단 | 그 외 | favorable |

**conflict면 size *= 0.5**

### 15m confirm 정의

아래 중 **하나만** 선택 (여러 개 섞으면 대기봇):

1. `state in (accumulation, re_accumulation)` AND `trend_strength >= -0.05`
2. `markdown_prob` 감소 + `volatility` 피크아웃
3. 2-bar reversal 캔들 패턴

**confirm 없으면 WAIT**

---

## 7. Size Modulation (확정)

```python
base_size = 1.0

if hmm_4h == conflict:
    size *= 0.5

if hmm_1h == conflict:
    size *= 0.5

# Size floor
if size < 0.25:
    return NO_TRADE  # 수수료/슬리피지에 죽음
```

| 상황 | 최종 size |
|------|-----------|
| 4H OK, 1H OK | 1.0 |
| 4H conflict, 1H OK | 0.5 |
| 4H OK, 1H conflict | 0.5 |
| 4H conflict, 1H conflict | 0.25 |

---

## 8. Zone 상단/하단 판단 (확정)

```python
# Zone 내에서 상대 위치
zone_mid = fib_level_price

if price <= zone_mid:
    position = "하단"  # Long 후보
else:
    position = "상단"  # Short 후보
```

여러 Zone 겹칠 때: **가장 가까운 level** 우선

---

## 9. 구현 순서 (확정)

```
1. Zone 상단/하단 판단 로직 구현
2. TF별 HMM 학습 파이프라인 (피처 통일)
3. 상태 → (hostile/conflict/confirm) 매핑
4. Size modulation + floor 구현
5. 백테스트: Zone 하단 + markdown 케이스 리포트
```

---

## 10. HMM 학습 피처 (통일)

모든 TF에 동일 피처 사용:

```python
features = [
    'log_return',           # 수익률
    'realized_vol',         # ATR or rolling std
    'trend_strength',       # EMA slope or z-score
    'range_proxy',          # Bollinger bandwidth or (H-L)/ATR
    'volume_proxy',         # (optional)
]
```

---

## 버전

- v1.0 (2026-01-15): GPT 답변 기반 설계 확정
- v1.1 (2026-01-15): 구현 가이드라인 추가 (hostile/conflict 정의, size modulation, 구현 순서)
