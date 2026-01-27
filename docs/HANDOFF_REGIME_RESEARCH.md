# 핸드오프: 레짐 분류기 연구 결과 (2026-01-18)

## TL;DR

**1H Hilbert (causal)가 최고 성능 (IC=+0.027)**. 레짐은 단독 신호가 아닌 **필터**로 사용.

---

## 테스트 결과 요약

| TF | 방식 | OOS 결과 | IC | 권장 |
|----|------|----------|-----|------|
| 15m | Boltzmann | ❌ FAIL | -0.008 | 사용 안함 |
| **1H** | **Hilbert (causal)** | **✅ BULL/BEAR OK** | **+0.027** | **Gate로 사용** |
| 1D | Boltzmann | ⚠️ 2W만 OK | +0.038 | 바이어스 필터 |

---

## 핵심 파일

```
src/regime/
├── boltzmann_regime.py    # Boltzmann 분포 레짐 (신규)
└── wave_regime.py         # Hilbert 레짐 (causal 추가됨)
    └── classify_series_causal()  # 룩어헤드 없는 버전
```

---

## 사용법

```python
# 1H Hilbert Gate (권장)
from src.regime.wave_regime import WaveRegimeClassifier

classifier = WaveRegimeClassifier(detrend_period=48, hilbert_window=32)
result = classifier.classify_series_causal(prices_1h)
# result['regime'] = 'BULL' | 'BEAR' | 'RANGE'

# 진입 필터
if result.iloc[-1]['regime'] == 'BEAR':
    block_long_entry()
```

---

## 다음 작업

1. `src/gate/hmm_entry_gate.py`에 Hilbert 필터 통합
2. 백테스트로 필터 효과 검증
3. 롱/숏 비대칭 적용 (롱 엄격, 숏 느슨)

---

## 상세 백업

`docs/session_backups/session_backup_20260118_regime_research.md` 참조

---

*RAG Embedding: trading_bot*
