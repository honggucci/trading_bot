"""
1W Fib 레벨 지지/저항 백테스트

목적: 1W Fib 앵커 레벨에서 실제로 가격 반응(반전/반등)이 일어났는지 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 1W Fib 앵커 설정
FIB_0 = 3120
FIB_1 = 20650
RANGE = FIB_1 - FIB_0  # 17530

def fib_to_price(fib_level: float) -> float:
    """Fib 레벨을 가격으로 변환"""
    return FIB_0 + (fib_level * RANGE)

def price_to_fib(price: float) -> float:
    """가격을 Fib 레벨로 변환"""
    return (price - FIB_0) / RANGE

# 주요 Fib 레벨들 (0부터 6까지, 주요 비율 포함)
FIB_LEVELS = {}
for i in range(7):  # 0 to 6
    FIB_LEVELS[f"{i}.000"] = fib_to_price(i)
    FIB_LEVELS[f"{i}.236"] = fib_to_price(i + 0.236)
    FIB_LEVELS[f"{i}.382"] = fib_to_price(i + 0.382)
    FIB_LEVELS[f"{i}.500"] = fib_to_price(i + 0.5)
    FIB_LEVELS[f"{i}.618"] = fib_to_price(i + 0.618)
    FIB_LEVELS[f"{i}.786"] = fib_to_price(i + 0.786)

# 특수 레벨 추가
FIB_LEVELS["0.702"] = fib_to_price(0.702)  # 2022 저점
FIB_LEVELS["3.786"] = fib_to_price(3.786)  # 2021 ATH

def load_all_1h_data() -> pd.DataFrame:
    """모든 1시간봉 데이터 로드"""
    data_path = Path(r"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\bronze\binance\futures\BTC-USDT\1h")

    all_dfs = []
    for year_dir in sorted(data_path.iterdir()):
        if year_dir.is_dir():
            for parquet_file in sorted(year_dir.glob("*.parquet")):
                df = pd.read_parquet(parquet_file)
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    combined = combined.drop_duplicates(subset='timestamp', keep='first')

    return combined

def detect_level_reactions(df: pd.DataFrame, level_price: float, tolerance_pct: float = 0.5) -> List[Dict]:
    """
    특정 Fib 레벨에서의 가격 반응 감지

    반응 조건:
    1. 가격이 레벨 ± tolerance 범위에 진입
    2. 진입 후 반대 방향으로 최소 1% 이상 이동
    """
    reactions = []
    tolerance = level_price * (tolerance_pct / 100)
    level_low = level_price - tolerance
    level_high = level_price + tolerance

    in_zone = False
    zone_entry_idx = None
    zone_entry_price = None
    zone_direction = None  # 'from_above' or 'from_below'

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        high = row['high']
        low = row['low']
        close = row['close']
        prev_close = prev_row['close']

        # 존 진입 감지
        if not in_zone:
            # 위에서 아래로 진입 (저항 테스트)
            if prev_close > level_high and low <= level_high:
                in_zone = True
                zone_entry_idx = i
                zone_entry_price = close
                zone_direction = 'from_above'
            # 아래에서 위로 진입 (지지 테스트)
            elif prev_close < level_low and high >= level_low:
                in_zone = True
                zone_entry_idx = i
                zone_entry_price = close
                zone_direction = 'from_below'

        # 존 내에서 반응 감지
        if in_zone:
            price_change_pct = (close - zone_entry_price) / zone_entry_price * 100

            # 반응 판단 (최소 1% 이동)
            if zone_direction == 'from_above' and price_change_pct >= 1.0:
                # 저항에서 반등 (실패한 저항)
                reactions.append({
                    'timestamp': row['timestamp'],
                    'level_price': level_price,
                    'reaction_type': 'resistance_bounce',  # 저항 돌파 후 상승
                    'entry_price': zone_entry_price,
                    'exit_price': close,
                    'change_pct': price_change_pct,
                    'bars_in_zone': i - zone_entry_idx,
                })
                in_zone = False
            elif zone_direction == 'from_above' and price_change_pct <= -1.0:
                # 저항에서 반전 (성공한 저항)
                reactions.append({
                    'timestamp': row['timestamp'],
                    'level_price': level_price,
                    'reaction_type': 'resistance_reject',  # 저항에서 하락
                    'entry_price': zone_entry_price,
                    'exit_price': close,
                    'change_pct': price_change_pct,
                    'bars_in_zone': i - zone_entry_idx,
                })
                in_zone = False
            elif zone_direction == 'from_below' and price_change_pct <= -1.0:
                # 지지에서 반락 (실패한 지지)
                reactions.append({
                    'timestamp': row['timestamp'],
                    'level_price': level_price,
                    'reaction_type': 'support_breakdown',  # 지지 붕괴
                    'entry_price': zone_entry_price,
                    'exit_price': close,
                    'change_pct': price_change_pct,
                    'bars_in_zone': i - zone_entry_idx,
                })
                in_zone = False
            elif zone_direction == 'from_below' and price_change_pct >= 1.0:
                # 지지에서 반등 (성공한 지지)
                reactions.append({
                    'timestamp': row['timestamp'],
                    'level_price': level_price,
                    'reaction_type': 'support_bounce',  # 지지 반등
                    'entry_price': zone_entry_price,
                    'exit_price': close,
                    'change_pct': price_change_pct,
                    'bars_in_zone': i - zone_entry_idx,
                })
                in_zone = False

            # 너무 오래 존에 머무르면 리셋 (24시간 = 24봉)
            if in_zone and (i - zone_entry_idx) > 24:
                in_zone = False

    return reactions

def analyze_fib_level(df: pd.DataFrame, fib_label: str, level_price: float) -> Dict:
    """단일 Fib 레벨 분석"""

    # 가격 범위 체크 (데이터에서 이 레벨이 실제로 테스트되었는지)
    min_price = df['low'].min()
    max_price = df['high'].max()

    if level_price < min_price or level_price > max_price:
        return {
            'fib_label': fib_label,
            'price': level_price,
            'status': 'out_of_range',
            'reactions': [],
            'total_touches': 0,
        }

    reactions = detect_level_reactions(df, level_price, tolerance_pct=0.5)

    # 반응 통계
    support_bounces = len([r for r in reactions if r['reaction_type'] == 'support_bounce'])
    support_breakdowns = len([r for r in reactions if r['reaction_type'] == 'support_breakdown'])
    resistance_rejects = len([r for r in reactions if r['reaction_type'] == 'resistance_reject'])
    resistance_bounces = len([r for r in reactions if r['reaction_type'] == 'resistance_bounce'])

    total_support_tests = support_bounces + support_breakdowns
    total_resistance_tests = resistance_rejects + resistance_bounces

    support_success_rate = support_bounces / total_support_tests * 100 if total_support_tests > 0 else 0
    resistance_success_rate = resistance_rejects / total_resistance_tests * 100 if total_resistance_tests > 0 else 0

    return {
        'fib_label': fib_label,
        'price': round(level_price, 0),
        'status': 'analyzed',
        'total_touches': len(reactions),
        'support_tests': total_support_tests,
        'support_bounces': support_bounces,
        'support_breakdowns': support_breakdowns,
        'support_success_rate': round(support_success_rate, 1),
        'resistance_tests': total_resistance_tests,
        'resistance_rejects': resistance_rejects,
        'resistance_bounces': resistance_bounces,
        'resistance_success_rate': round(resistance_success_rate, 1),
        'avg_change_pct': round(np.mean([abs(r['change_pct']) for r in reactions]), 2) if reactions else 0,
    }

def run_backtest():
    """전체 백테스트 실행"""
    print("=" * 60)
    print("1W Fib 레벨 지지/저항 백테스트")
    print("=" * 60)
    print(f"\n앵커: Fib 0 = ${FIB_0:,}, Fib 1 = ${FIB_1:,}")
    print(f"Range: ${RANGE:,}")
    print("\n데이터 로딩 중...")

    df = load_all_1h_data()
    print(f"로드 완료: {len(df):,}개 1시간봉")
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"가격 범위: ${df['low'].min():,.0f} ~ ${df['high'].max():,.0f}")

    # 실제 데이터 범위 내 Fib 레벨만 필터링
    min_price = df['low'].min()
    max_price = df['high'].max()

    relevant_levels = {k: v for k, v in FIB_LEVELS.items()
                       if min_price * 0.9 <= v <= max_price * 1.1}

    print(f"\n분석 대상 Fib 레벨: {len(relevant_levels)}개")
    print("-" * 60)

    results = []
    for fib_label, level_price in sorted(relevant_levels.items(), key=lambda x: x[1]):
        result = analyze_fib_level(df, fib_label, level_price)
        if result['status'] == 'analyzed' and result['total_touches'] > 0:
            results.append(result)
            print(f"Fib {fib_label:>6} (${level_price:>8,.0f}): "
                  f"터치 {result['total_touches']:>3}회 | "
                  f"지지 {result['support_success_rate']:>5.1f}% ({result['support_bounces']}/{result['support_tests']}) | "
                  f"저항 {result['resistance_success_rate']:>5.1f}% ({result['resistance_rejects']}/{result['resistance_tests']})")

    # 요약 통계
    print("\n" + "=" * 60)
    print("요약 통계")
    print("=" * 60)

    if results:
        total_touches = sum(r['total_touches'] for r in results)
        avg_support_rate = np.mean([r['support_success_rate'] for r in results if r['support_tests'] > 0])
        avg_resistance_rate = np.mean([r['resistance_success_rate'] for r in results if r['resistance_tests'] > 0])

        print(f"총 터치 횟수: {total_touches}")
        print(f"평균 지지 성공률: {avg_support_rate:.1f}%")
        print(f"평균 저항 성공률: {avg_resistance_rate:.1f}%")

        # 가장 신뢰도 높은 레벨
        print("\n가장 신뢰도 높은 지지 레벨 (최소 3회 테스트):")
        support_ranked = sorted([r for r in results if r['support_tests'] >= 3],
                                key=lambda x: x['support_success_rate'], reverse=True)[:5]
        for r in support_ranked:
            print(f"  Fib {r['fib_label']:>6} (${r['price']:>8,.0f}): {r['support_success_rate']:.1f}% ({r['support_bounces']}/{r['support_tests']})")

        print("\n가장 신뢰도 높은 저항 레벨 (최소 3회 테스트):")
        resistance_ranked = sorted([r for r in results if r['resistance_tests'] >= 3],
                                   key=lambda x: x['resistance_success_rate'], reverse=True)[:5]
        for r in resistance_ranked:
            print(f"  Fib {r['fib_label']:>6} (${r['price']:>8,.0f}): {r['resistance_success_rate']:.1f}% ({r['resistance_rejects']}/{r['resistance_tests']})")

        # Fib 타입별 분석 (정수, 황금비, 기타)
        print("\n" + "=" * 60)
        print("Fib 타입별 평균 성공률")
        print("=" * 60)

        integer_fibs = [r for r in results if r['fib_label'].endswith('.000')]
        golden_fibs = [r for r in results if '.618' in r['fib_label'] or '.382' in r['fib_label']]
        half_fibs = [r for r in results if '.500' in r['fib_label']]

        def calc_avg_success(level_list):
            support = [r['support_success_rate'] for r in level_list if r['support_tests'] > 0]
            resistance = [r['resistance_success_rate'] for r in level_list if r['resistance_tests'] > 0]
            return (np.mean(support) if support else 0, np.mean(resistance) if resistance else 0)

        int_s, int_r = calc_avg_success(integer_fibs)
        gold_s, gold_r = calc_avg_success(golden_fibs)
        half_s, half_r = calc_avg_success(half_fibs)

        print(f"정수 Fib (0, 1, 2...):    지지 {int_s:>5.1f}% | 저항 {int_r:>5.1f}%")
        print(f"황금비 (0.382, 0.618):    지지 {gold_s:>5.1f}% | 저항 {gold_r:>5.1f}%")
        print(f"반정수 (0.5):             지지 {half_s:>5.1f}% | 저항 {half_r:>5.1f}%")

    # 결과 저장
    output_path = Path(r"c:\Users\hahonggu\Desktop\coin_master\projects\trading_bot\data\fib_backtest_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n결과 저장: {output_path}")

    return results

if __name__ == "__main__":
    run_backtest()
