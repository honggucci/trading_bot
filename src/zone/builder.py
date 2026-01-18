"""
Zone Builder
============

Fib 레벨 + ATR Zone Width → 트레이딩 Zone 생성

구조:
- Fib 레벨: 순수 수학적 좌표 (고정)
- Zone Width: ATR(w) * k (TF별 설정)
- Zone: Fib Level ± (Zone Width / 2)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path


# =============================================================================
# 1W 피보나치 앵커 (고정)
# =============================================================================
FIB_0 = 3120
FIB_1 = 20650
RANGE = FIB_1 - FIB_0  # 17530

# 표준 피보나치 비율
STANDARD_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

# 확장 Fib (현재가 $90k+ 대응)
# Fib 5.0 = $90,770, Fib 6.0 = $108,300
EXTENDED_FIB_MAX = 8  # Fib 8.0 = $143,360


@dataclass
class FibZone:
    """단일 Fib Zone"""
    fib_ratio: float      # Fib 비율 (예: 0.618)
    fib_price: float      # Fib 가격 (예: $13,954)
    zone_low: float       # Zone 하단
    zone_high: float      # Zone 상단
    zone_width: float     # Zone 폭
    timeframe: str        # 적용된 TF
    level_depth: int      # 프랙탈 깊이 (0=L0, 1=L1, ...)
    parent_range: Tuple[float, float]  # 부모 셀 (fib_low, fib_high)

    @property
    def zone_center(self) -> float:
        return self.fib_price

    def contains(self, price: float) -> bool:
        """가격이 Zone 내에 있는지"""
        return self.zone_low <= price <= self.zone_high

    def distance_to(self, price: float) -> float:
        """가격까지의 거리 (%)"""
        return abs(price - self.fib_price) / self.fib_price * 100


def fib_to_price(fib_ratio: float, cell_low: float = FIB_0, cell_high: float = FIB_1) -> float:
    """Fib 비율을 가격으로 변환"""
    cell_range = cell_high - cell_low
    return cell_low + (fib_ratio * cell_range)


def price_to_fib(price: float, cell_low: float = FIB_0, cell_high: float = FIB_1) -> float:
    """가격을 Fib 비율로 변환"""
    cell_range = cell_high - cell_low
    if cell_range == 0:
        return 0.5
    return (price - cell_low) / cell_range


def load_zone_config() -> dict:
    """Zone 파라미터 로드 (우선순위: DB → JSON)"""

    # 1순위: DB에서 로드
    try:
        from ..db.fib_anchors import ZoneParamsDB
        params = ZoneParamsDB.get_all()
        if params:
            # DB 데이터를 JSON 형식으로 변환
            tf_params = {}
            clamp = {}
            for p in params:
                tf_params[p.timeframe] = {
                    'atr_window': p.atr_window,
                    'k': p.k,
                    'role': p.role,
                }
                if p.min_pct is not None and p.max_pct is not None:
                    clamp[p.timeframe] = {
                        'min_pct': p.min_pct,
                        'max_pct': p.max_pct,
                    }
            return {'tf_params': tf_params, 'clamp': clamp}
    except Exception:
        pass  # DB 실패 시 JSON fallback

    # 2순위: JSON 파일
    config_path = Path(__file__).parent.parent.parent / "config" / "zone_width.json"
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """ATR 계산"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    n = len(tr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.mean(tr[i - window + 1:i + 1])
    return out


# =============================================================================
# Zone Builder
# =============================================================================

class ZoneBuilder:
    """
    Fib Zone Builder

    사용법:
    ```python
    builder = ZoneBuilder()

    # 현재 ATR로 Zone 생성
    zones = builder.build_zones(
        current_atr={'15m': 450, '1h': 800, '4h': 1500, '1d': 2500},
        max_depth=2,
        price_range=(80000, 120000),
    )

    # 특정 가격의 Zone 찾기
    active_zones = builder.find_zones_at_price(95000, zones)

    # Zone 겹침 분석
    overlaps = builder.find_overlapping_zones(zones)
    ```
    """

    def __init__(self):
        self.config = load_zone_config()
        self.fib_0 = FIB_0
        self.fib_1 = FIB_1
        self.range = RANGE

    def _get_zone_width(self, tf: str, atr_value: float, fib_price: float) -> float:
        """
        TF별 Zone Width 계산

        GPT 피드백 반영:
        - ATR은 달러 단위, 가격에 따라 변동
        - Clamp는 Fib 레벨 가격 기준 퍼센트로 적용
        - 고가에서 존이 과도하게 커지는 것 방지

        Formula:
            width = k * ATR
            width = clamp(width, fib_price * min_pct, fib_price * max_pct)
        """
        tf_config = self.config['tf_params'].get(tf, {})
        k = tf_config.get('k')

        if k is None:
            return 0  # 1W는 Zone Width 없음

        # Zone Width = ATR * k
        zone_width = atr_value * k

        # Clamp 적용 (Fib 레벨 가격 기준!)
        clamp = self.config['clamp'].get(tf, {})
        min_pct = clamp.get('min_pct', 0)
        max_pct = clamp.get('max_pct', 1)

        # Fib 레벨 가격 기준으로 min/max 계산
        min_width = fib_price * min_pct
        max_width = fib_price * max_pct

        return np.clip(zone_width, min_width, max_width)

    def _generate_fib_levels_in_cell(
        self,
        cell_low: float,
        cell_high: float,
        depth: int,
        max_depth: int,
        price_range: Tuple[float, float],
    ) -> List[Tuple[float, float, int, Tuple[float, float]]]:
        """
        단일 셀 내 프랙탈 Fib 레벨 생성 (재귀)

        Returns:
            [(fib_ratio, price, depth, parent_range), ...]
        """
        levels = []

        for ratio in STANDARD_RATIOS:
            price = fib_to_price(ratio, cell_low, cell_high)

            # 가격 범위 필터
            if price_range[0] <= price <= price_range[1]:
                levels.append((ratio, price, depth, (cell_low, cell_high)))

        # 더 깊은 프랙탈 생성
        if depth < max_depth:
            for i in range(len(STANDARD_RATIOS) - 1):
                sub_low = fib_to_price(STANDARD_RATIOS[i], cell_low, cell_high)
                sub_high = fib_to_price(STANDARD_RATIOS[i + 1], cell_low, cell_high)

                # 가격 범위 내의 셀만 처리
                if sub_high >= price_range[0] and sub_low <= price_range[1]:
                    sub_levels = self._generate_fib_levels_in_cell(
                        sub_low, sub_high, depth + 1, max_depth, price_range
                    )
                    levels.extend(sub_levels)

        return levels

    def _generate_fib_levels(
        self,
        price_range: Tuple[float, float],
        max_depth: int,
    ) -> List[Tuple[float, float, int, Tuple[float, float]]]:
        """
        확장 Fib 레벨 생성 (Fib 0 ~ 8)

        Returns:
            [(fib_ratio, price, depth, parent_range), ...]
        """
        all_levels = []

        # 정수 Fib 간격마다 (0~1, 1~2, 2~3, ..., 7~8)
        for fib_int in range(EXTENDED_FIB_MAX):
            cell_low = FIB_0 + (fib_int * RANGE)
            cell_high = FIB_0 + ((fib_int + 1) * RANGE)

            # 가격 범위와 겹치는 셀만 처리
            if cell_high < price_range[0] or cell_low > price_range[1]:
                continue

            # 이 셀 내의 프랙탈 레벨 생성
            cell_levels = self._generate_fib_levels_in_cell(
                cell_low, cell_high, 0, max_depth, price_range
            )
            all_levels.extend(cell_levels)

        return all_levels

    def build_zones(
        self,
        current_atr: Dict[str, float],
        max_depth: int = 2,
        price_range: Tuple[float, float] = (50000, 150000),
        timeframe: str = '15m',
    ) -> List[FibZone]:
        """
        Fib Zone 생성

        Args:
            current_atr: TF별 현재 ATR 값 {'15m': 450, '1h': 800, ...}
            max_depth: 프랙탈 깊이 (0=L0만, 1=L1까지, 2=L2까지)
            price_range: 생성할 가격 범위
            timeframe: Zone Width 적용할 TF

        Returns:
            List[FibZone]
        """
        # Fib 레벨 생성 (확장 Fib 포함)
        fib_levels = self._generate_fib_levels(price_range, max_depth)

        # ATR 값
        atr_value = current_atr.get(timeframe, 0)

        # Zone 생성 (각 Fib 레벨별로 Zone Width 계산)
        zones = []
        for fib_ratio, price, depth, parent_range in fib_levels:
            # 각 Fib 레벨 가격 기준으로 Zone Width 계산 (clamp 적용)
            zone_width = self._get_zone_width(timeframe, atr_value, price)

            zone = FibZone(
                fib_ratio=fib_ratio,
                fib_price=price,
                zone_low=price - (zone_width / 2),
                zone_high=price + (zone_width / 2),
                zone_width=zone_width,
                timeframe=timeframe,
                level_depth=depth,
                parent_range=parent_range,
            )
            zones.append(zone)

        # 가격순 정렬
        zones.sort(key=lambda z: z.fib_price)

        return zones

    def find_zones_at_price(self, price: float, zones: List[FibZone]) -> List[FibZone]:
        """특정 가격이 속한 Zone 찾기"""
        return [z for z in zones if z.contains(price)]

    def find_overlapping_zones(self, zones: List[FibZone]) -> List[Tuple[FibZone, FibZone, float]]:
        """
        겹치는 Zone 쌍 찾기

        Returns:
            [(zone1, zone2, overlap_amount), ...]
        """
        overlaps = []
        sorted_zones = sorted(zones, key=lambda z: z.fib_price)

        for i in range(len(sorted_zones) - 1):
            z1 = sorted_zones[i]

            for j in range(i + 1, len(sorted_zones)):
                z2 = sorted_zones[j]

                # z2가 z1보다 높은 가격
                # 겹침: z1.high > z2.low
                if z1.zone_high > z2.zone_low:
                    overlap = z1.zone_high - z2.zone_low
                    overlaps.append((z1, z2, overlap))
                else:
                    # 더 높은 Zone은 겹칠 가능성 없음
                    break

        return overlaps

    def merge_to_confluence(self, zones: List[FibZone]) -> List[Dict]:
        """
        겹치는 Zone을 Confluence Zone으로 병합

        GPT 피드백:
        - 겹침 = 합류(confluence) = 강한 신호
        - 여러 레벨이 같은 가격대를 가리킴 = Liquidity Magnet

        Returns:
            [{
                'zone_low': float,
                'zone_high': float,
                'center': float,
                'width': float,
                'level_count': int,
                'levels': List[FibZone],
                'score': float,
                'grade': str,  # 'A', 'B', 'C'
            }, ...]
        """
        if not zones:
            return []

        sorted_zones = sorted(zones, key=lambda z: z.zone_low)
        confluences = []
        current_group = [sorted_zones[0]]
        current_high = sorted_zones[0].zone_high

        for zone in sorted_zones[1:]:
            # 겹침 체크: 현재 그룹의 상단 > 다음 Zone의 하단
            if current_high >= zone.zone_low:
                current_group.append(zone)
                current_high = max(current_high, zone.zone_high)
            else:
                # 그룹 종료, 새 그룹 시작
                confluences.append(self._create_confluence(current_group))
                current_group = [zone]
                current_high = zone.zone_high

        # 마지막 그룹
        if current_group:
            confluences.append(self._create_confluence(current_group))

        # 점수순 정렬
        confluences.sort(key=lambda c: c['score'], reverse=True)

        return confluences

    def _create_confluence(self, zones: List[FibZone]) -> Dict:
        """Confluence Zone 생성 + 점수 계산"""
        zone_low = min(z.zone_low for z in zones)
        zone_high = max(z.zone_high for z in zones)
        center = (zone_low + zone_high) / 2
        width = zone_high - zone_low

        # 점수 계산
        # 1. 레벨 개수 (많을수록 +)
        level_count = len(zones)
        score_count = min(level_count * 2, 20)  # cap at 20

        # 2. 레벨 등급 (L0 > L1 > L2)
        depth_score = sum(3 - z.level_depth for z in zones)  # L0=3, L1=2, L2=1

        # 3. 좁을수록 + (정밀하게 모임)
        # 기준: $500 이하면 최고점
        width_score = max(0, 20 - (width / 100))

        total_score = score_count + depth_score + width_score

        # 등급
        if total_score >= 30 and level_count >= 3:
            grade = 'A'
        elif total_score >= 15 and level_count >= 2:
            grade = 'B'
        else:
            grade = 'C'

        return {
            'zone_low': zone_low,
            'zone_high': zone_high,
            'center': center,
            'width': width,
            'level_count': level_count,
            'levels': zones,
            'score': total_score,
            'grade': grade,
        }

    def get_confluence_zones(
        self,
        current_atr: Dict[str, float],
        price_range: Tuple[float, float],
        timeframe: str = '15m',
        max_depth: int = 2,
        min_grade: str = 'B',
    ) -> List[Dict]:
        """
        Confluence Zone 생성 (메인 API)

        Args:
            current_atr: TF별 ATR
            price_range: 가격 범위
            timeframe: Zone Width TF
            max_depth: 프랙탈 깊이
            min_grade: 최소 등급 필터 ('A', 'B', 'C')

        Returns:
            Confluence Zone 리스트 (점수순)
        """
        zones = self.build_zones(current_atr, max_depth, price_range, timeframe)
        confluences = self.merge_to_confluence(zones)

        # 등급 필터
        grade_order = {'A': 3, 'B': 2, 'C': 1}
        min_grade_value = grade_order.get(min_grade, 2)

        filtered = [
            c for c in confluences
            if grade_order.get(c['grade'], 0) >= min_grade_value
        ]

        return filtered

    def analyze_zone_density(self, zones: List[FibZone]) -> Dict:
        """Zone 밀도 분석"""
        if not zones:
            return {
                'total_zones': 0,
                'price_range': (0, 0),
                'avg_gap': 0,
                'min_gap': 0,
                'max_gap': 0,
                'avg_width': 0,
                'overlap_ratio': 0,
            }

        prices = [z.fib_price for z in zones]
        gaps = np.diff(prices)
        widths = [z.zone_width for z in zones]

        return {
            'total_zones': len(zones),
            'price_range': (min(prices), max(prices)),
            'avg_gap': np.mean(gaps) if len(gaps) > 0 else 0,
            'min_gap': np.min(gaps) if len(gaps) > 0 else 0,
            'max_gap': np.max(gaps) if len(gaps) > 0 else 0,
            'avg_width': np.mean(widths),
            'overlap_ratio': np.mean(widths) / np.mean(gaps) if len(gaps) > 0 and np.mean(gaps) > 0 else 0,
        }


# =============================================================================
# 분석 함수
# =============================================================================

def analyze_zone_overlaps(
    price_range: Tuple[float, float] = (90000, 110000),
    max_depth: int = 2,
) -> Dict:
    """
    Zone 겹침 분석

    Returns:
        {
            'zones': [...],
            'overlaps': [...],
            'density': {...},
        }
    """
    builder = ZoneBuilder()

    # 샘플 ATR 값 (실제 데이터 필요)
    # 현재가 $100,000 기준 대략적인 값
    sample_atr = {
        '15m': 500,    # ~0.5%
        '1h': 900,     # ~0.9%
        '4h': 1500,    # ~1.5%
        '1d': 2500,    # ~2.5%
    }

    results = {}

    for tf in ['15m', '1h', '4h', '1d']:
        zones = builder.build_zones(
            current_atr=sample_atr,
            max_depth=max_depth,
            price_range=price_range,
            timeframe=tf,
        )

        overlaps = builder.find_overlapping_zones(zones)
        density = builder.analyze_zone_density(zones)

        results[tf] = {
            'zones': zones,
            'overlaps': overlaps,
            'density': density,
            'overlap_count': len(overlaps),
        }

    return results


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Zone Builder - 겹침 분석 (Clamp 수정 후)")
    print("=" * 70)

    builder = ZoneBuilder()

    # 샘플 ATR (15m 기준 ~0.5%)
    sample_atr = {
        '15m': 500,
        '1h': 900,
        '4h': 1500,
        '1d': 2500,
    }

    # 다양한 가격대에서 Zone Width 확인
    print("\n[1] 가격대별 Zone Width (15m TF)")
    print("-" * 50)
    for price in [10000, 50000, 100000]:
        width = builder._get_zone_width('15m', sample_atr['15m'], price)
        pct = width / price * 100
        print(f"  ${price:>7,}: Zone Width = ${width:,.0f} ({pct:.2f}%)")

    # $90k~$110k 범위 겹침 분석
    price_range = (90000, 110000)

    print(f"\n[2] 겹침 분석 ($90k ~ $110k)")
    print("-" * 50)

    for max_depth in [0, 1]:
        print(f"\n  프랙탈 깊이: L{max_depth}")

        zones = builder.build_zones(
            current_atr=sample_atr,
            max_depth=max_depth,
            price_range=price_range,
            timeframe='15m',
        )

        overlaps = builder.find_overlapping_zones(zones)
        density = builder.analyze_zone_density(zones)

        print(f"    Zone 수: {density['total_zones']}")
        print(f"    평균 간격: ${density['avg_gap']:,.0f}")
        print(f"    평균 폭: ${density['avg_width']:,.0f}")
        print(f"    겹침 비율: {density['overlap_ratio']:.2f}")
        print(f"    겹치는 쌍: {len(overlaps)}개")

        # Confluence 결과
        confluences = builder.merge_to_confluence(zones)
        print(f"    → Confluence Zone: {len(confluences)}개")

    # L0 상세
    print(f"\n[3] L0 Zone 상세 목록")
    print("-" * 50)

    zones = builder.build_zones(
        current_atr=sample_atr,
        max_depth=0,
        price_range=price_range,
        timeframe='15m',
    )

    for z in zones:
        print(f"  Fib {z.fib_ratio:.3f}: ${z.fib_price:,.0f} "
              f"→ Zone ${z.zone_low:,.0f} ~ ${z.zone_high:,.0f} "
              f"(폭: ${z.zone_width:,.0f})")
