# Session Backup - 2026-01-21 MODE61 RR 2:1 강제 진입가

## 완료된 작업

### 1. MODE61: RR 2:1 강제 진입가 로직 구현
- **목표**: 다이버전스 감지 시 Fib SL/TP 기반으로 정확히 RR 2:1이 되는 지점에 지정가 매수
- **수식**: `Entry = (TP + 2*SL) / 3`

### 2. 코드 수정 사항

#### `scripts/backtest_strategy_compare.py`
1. **새 파라미터 추가** (라인 351):
   ```python
   rr_entry_mode: str = "fixed_rr"  # "fixed_rr" | "offset_ratio"
   ```

2. **RR 2:1 강제 진입가 계산** (라인 3848-3925):
   - SL = 이전 Fib 레벨 (signal_price 아래, 최소 0.5 ATR 거리)
   - TP = 다음 Fib 레벨 (signal_price 위)
   - Entry = (TP + 2*SL) / 3

3. **pre_sl 사용으로 INVARIANT REJECT 수정** (라인 4118-4122):
   ```python
   if self.config.rr_entry_mode == "fixed_rr" and 'pre_sl' in pending_long_signal:
       sl = pending_long_signal['pre_sl']  # 재계산 금지
   ```

#### `configs/mode61.json` (신규 생성)
```json
{
  "_description": "MODE 61: MODE49 + rr_entry_mode=fixed_rr (RR 2:1 강제)",
  "rr_entry_mode": "fixed_rr",
  "use_macro_fib": true,
  "use_dynamic_fib": true
}
```

## 백테스트 결과 (2021-01-01 ~ 2024-12-31)

| 지표 | MODE61 결과 |
|------|-------------|
| Total Trades | 339 |
| Win Rate | 33.3% |
| Total PnL | -$2,010.95 (-20.1%) |
| RR Ratio | 0.96 |
| SL Hit | 272건 (80.2%) |
| 15분 내 SL | 52.2% (142건) |

### 핵심 문제점
1. **SL이 너무 타이트**: p50 = 0.47 ATR
2. **빠른 손절**: 52.2%가 15분 내 SL hit
3. **RR 2:1 미달성**: 설계는 2:1이지만 실현 RR = 0.96

### 결론
`rr_entry_mode: fixed_rr` 로직 자체는 정상 작동하나, Fib 레벨 간격이 좁아서 SL 거리가 불충분함. 최소 1 ATR 이상의 SL 거리 확보 필요.

## 생성/수정된 파일
- `scripts/backtest_strategy_compare.py` - RR 2:1 강제 진입가 로직 추가
- `configs/mode61.json` - 새 모드 설정 파일

## 다음 단계 제안
1. `min_sl_distance_atr_mult` 파라미터 증가 (0.1 → 1.0)
2. SL 거리가 1 ATR 미만인 신호 필터링
3. Dynamic Fib 레벨 간격 확대 검토

## 참고: Step 1-3 수정 (이전 세션)
- Step 2: `d[-1]` → `d[-2]` (다이버전스 룩어헤드 제거)
- Step 3: `extended_ratios` (0.0, 1.0 포함), `fib_zone` 모드

---
*Generated: 2026-01-21*
