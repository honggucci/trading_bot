"""
Cycle Anchor - Bitcoin Halving Cycle Based Fibonacci
=====================================================

비트코인 반감기 사이클 기반 Fibonacci 앵커 시스템.

핵심 사상:
- "전 사이클 고점 = 현 사이클 저점 (지지선)"
- 폭락 시 전 사이클 ATH가 지지선 역할
- 사이클 고점/저점으로 Fib 0/1 고정

1W Fib 앵커 (v2.0):
- Fib 0 = $3,120 (2018년 저점)
- Fib 1 = $20,650 (2017/18년 고점)
- 모든 TF의 기준점 (절대 불변)

사용법:
```python
from src.context.cycle_anchor import get_btc_cycle_anchor, get_fib_levels, get_1w_fib_level

# 현재 사이클 앵커 가져오기
anchor = get_btc_cycle_anchor()

# 1W Fib 레벨 계산 (절대 기준)
fib_level = get_1w_fib_level(95000)  # 5.24

# Fib 레벨 계산
levels = get_fib_levels(anchor.cycle_low, anchor.cycle_high)

# 현재가 위치
position = anchor.get_position(92000)  # 0.69 (69%)

# 폭락 시 지지선
support = anchor.get_crash_support()  # $69,000
```
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
import numpy as np


# ============================================================================
# Database Integration (Optional)
# ============================================================================

# DB 모듈 로드 시도 (실패 시 하드코딩 fallback)
_DB_AVAILABLE = False
_FibAnchorDB = None
_FibAnchorClass = None
_FibLevelsDB = None
_FibLevelClass = None

try:
    from ..db.fib_anchors import FibAnchorDB as _FibAnchorDB
    from ..db.fib_anchors import FibAnchor as _FibAnchorClass
    from ..db.fib_anchors import FibLevelsDB as _FibLevelsDB
    from ..db.fib_anchors import FibLevel as _FibLevelClass
    _DB_AVAILABLE = True
except ImportError:
    pass


def _get_anchor_from_db(symbol: str = "BTC") -> Optional[Dict[str, Any]]:
    """DB에서 Fib 앵커 조회 (fallback: None)"""
    if not _DB_AVAILABLE or _FibAnchorDB is None:
        return None

    try:
        anchor = _FibAnchorDB.get(symbol)
        if anchor:
            return {
                "fib_0": anchor.fib_0,
                "fib_1": anchor.fib_1,
                "range": anchor.fib_range,
            }
    except Exception:
        pass
    return None


# ============================================================================
# 1W Fib Anchor (절대 불변 기준점)
# ============================================================================

# 1W Fib 앵커 기본값 (JSON/DB 로드 실패 시 사용)
_DEFAULT_1W_ANCHOR = {
    "fib_0": 3120,
    "fib_1": 20650,
    "range": 17530,
}

# 캐시된 1W 앵커
_1W_ANCHOR_CACHE: Optional[Dict[str, Any]] = None


def load_1w_fib_anchor(force_reload: bool = False, symbol: str = "BTC") -> Dict[str, Any]:
    """
    1W Fib 앵커 로드 (우선순위: DB → JSON → 기본값)

    Args:
        force_reload: 캐시 무시하고 다시 로드
        symbol: 에셋 심볼 (BTC, ETH, XRP 등)

    Returns:
        {
            "anchor": {"fib_0": 3120, "fib_1": 20650, "range": 17530},
            "levels": {"0.618": 13952, "3.618": 66549, ...},
            "key_levels": [...]
        }
    """
    global _1W_ANCHOR_CACHE

    # BTC 캐시만 사용 (다른 심볼은 매번 조회)
    if symbol == "BTC" and _1W_ANCHOR_CACHE is not None and not force_reload:
        return _1W_ANCHOR_CACHE

    # 1순위: DB에서 로드 (멀티 에셋 지원)
    db_anchor = _get_anchor_from_db(symbol)
    if db_anchor is not None:
        result = {"anchor": db_anchor, "levels": {}, "key_levels": []}
        if symbol == "BTC":
            _1W_ANCHOR_CACHE = result
        return result

    # 2순위: JSON 파일 로드 (BTC만)
    if symbol == "BTC":
        config_path = Path(__file__).parent.parent.parent / "config" / "fib_1w_anchor.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _1W_ANCHOR_CACHE = data
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # 3순위: 기본값 (BTC만, 다른 심볼은 에러)
    if symbol == "BTC":
        return {"anchor": _DEFAULT_1W_ANCHOR, "levels": {}, "key_levels": []}

    raise ValueError(f"No Fib anchor found for {symbol}. Add to DB first: python scripts/setup_fib_db.py")


def get_1w_fib_level(price: float, symbol: str = "BTC") -> float:
    """
    현재가의 1W Fib 레벨 계산 (절대 기준)

    Formula: fib_level = (price - fib_0) / range

    Args:
        price: 현재가 (USD)
        symbol: 에셋 심볼 (BTC, ETH, XRP)

    Returns:
        Fib 레벨 (예: 5.24)

    Example:
        >>> get_1w_fib_level(95000)
        5.24
        >>> get_1w_fib_level(3500, symbol="ETH")
        2.56
    """
    data = load_1w_fib_anchor(symbol=symbol)
    anchor = data.get("anchor", _DEFAULT_1W_ANCHOR)
    fib_0 = anchor["fib_0"]
    fib_range = anchor["range"]
    return (price - fib_0) / fib_range


def get_1w_fib_price(fib_level: float, symbol: str = "BTC") -> float:
    """
    1W Fib 레벨에 해당하는 가격 계산

    Formula: price = fib_0 + (fib_level * range)

    Args:
        fib_level: Fib 레벨 (예: 3.618)
        symbol: 에셋 심볼 (BTC, ETH, XRP)

    Returns:
        가격 (USD)

    Example:
        >>> get_1w_fib_price(3.618)
        66549
        >>> get_1w_fib_price(3.618, symbol="ETH")
        4909
    """
    data = load_1w_fib_anchor(symbol=symbol)
    anchor = data.get("anchor", _DEFAULT_1W_ANCHOR)
    fib_0 = anchor["fib_0"]
    fib_range = anchor["range"]
    return fib_0 + (fib_level * fib_range)


def get_1w_key_levels() -> Dict[str, float]:
    """
    1W Fib 주요 레벨 가격 반환

    Returns:
        {"0.702": 15430, "3.618": 66549, "3.786": 69493, ...}
    """
    data = load_1w_fib_anchor()
    return data.get("levels", {})


def get_1w_nearest_level(price: float, tolerance_pct: float = 0.03) -> Optional[str]:
    """
    현재가에 가장 가까운 1W Fib 레벨 찾기

    Args:
        price: 현재가 (USD)
        tolerance_pct: 허용 오차 (3%)

    Returns:
        Fib 레벨 문자열 (예: "3.618") 또는 None
    """
    levels = get_1w_key_levels()
    current_fib = get_1w_fib_level(price)

    for level_str, level_price in levels.items():
        try:
            level_fib = float(level_str)
            if abs(current_fib - level_fib) / max(level_fib, 0.001) <= tolerance_pct:
                return level_str
        except ValueError:
            continue

    return None


# ============================================================================
# Bitcoin Cycle Data (Hardcoded - 사이클 이론 기반)
# ============================================================================

@dataclass
class CycleData:
    """단일 사이클 데이터"""
    cycle_num: int
    halving_date: str
    cycle_low: float
    cycle_low_date: str
    cycle_high: float
    cycle_high_date: str

    @property
    def range(self) -> float:
        return self.cycle_high - self.cycle_low

    @property
    def multiplier(self) -> float:
        """저점 대비 고점 배수"""
        return self.cycle_high / self.cycle_low if self.cycle_low > 0 else 0


# 비트코인 역대 사이클 데이터
BTC_CYCLES: Dict[int, CycleData] = {
    1: CycleData(
        cycle_num=1,
        halving_date="2012-11-28",
        cycle_low=2,
        cycle_low_date="2011-11-18",
        cycle_high=1200,
        cycle_high_date="2013-11-29",
    ),
    2: CycleData(
        cycle_num=2,
        halving_date="2016-07-09",
        cycle_low=200,
        cycle_low_date="2015-01-14",
        cycle_high=20000,
        cycle_high_date="2017-12-17",
    ),
    3: CycleData(
        cycle_num=3,
        halving_date="2020-05-11",
        cycle_low=3200,
        cycle_low_date="2018-12-15",
        cycle_high=69000,
        cycle_high_date="2021-11-10",
    ),
    4: CycleData(
        cycle_num=4,
        halving_date="2024-04-20",
        cycle_low=15500,
        cycle_low_date="2022-11-21",
        cycle_high=126296,
        cycle_high_date="2025-10-06",
    ),
}

# 현재 사이클 번호
CURRENT_CYCLE = 4


# ============================================================================
# Cycle Anchor
# ============================================================================

@dataclass
class CycleAnchor:
    """
    사이클 기반 Fibonacci 앵커

    사이클 이론:
    - Fib 0 = 현 사이클 저점 (≈ 전 사이클 고점)
    - Fib 1 = 현 사이클 고점
    - 폭락 지지선 = 전 사이클 고점
    """
    # 현 사이클
    cycle_num: int
    cycle_low: float
    cycle_high: float
    cycle_low_date: str
    cycle_high_date: str

    # 전 사이클 (폭락 지지선용)
    prev_cycle_high: float
    prev_cycle_high_date: str

    # 메타
    halving_date: str
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_position(self, price: float) -> float:
        """
        현재가의 사이클 내 위치 (0~1)

        0 = 사이클 저점
        1 = 사이클 고점
        >1 = 신고점 돌파
        """
        if self.cycle_high == self.cycle_low:
            return 0.5
        return (price - self.cycle_low) / (self.cycle_high - self.cycle_low)

    def get_price_at_fib(self, fib_level: float) -> float:
        """
        Fib 레벨에 해당하는 가격

        Args:
            fib_level: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0 등
        """
        return self.cycle_low + (self.cycle_high - self.cycle_low) * fib_level

    def get_crash_support(self) -> float:
        """
        폭락 시 예상 지지선 (전 사이클 고점)

        사이클 이론: "전 사이클 ATH = 현 사이클 바닥"
        """
        return self.prev_cycle_high

    def get_expected_crash_low(self, crash_pct: float = 0.75) -> float:
        """
        사이클 종료 후 예상 저점

        역사적으로 사이클 고점에서 70-80% 하락
        기본값: 75% 하락 (0.25 유지)

        Args:
            crash_pct: 하락률 (0.7 = 70%, 0.8 = 80%)

        Returns:
            예상 저점 가격
        """
        return self.cycle_high * (1 - crash_pct)

    def get_wyckoff_zones(self) -> Dict[str, Tuple[float, float]]:
        """
        Wyckoff 패턴 기반 존 (사이클 종료 후)

        사이클 고점에서 70-80% 하락 → Wyckoff Accumulation 시작
        """
        high = self.cycle_high
        return {
            "markdown_zone": (high * 0.5, high * 1.0),      # 하락 추세
            "accumulation_zone": (high * 0.2, high * 0.35), # 매집 구간 (70-80% 하락)
            "spring_zone": (high * 0.15, high * 0.25),      # Spring (추가 하락 후 반등)
            "markup_zone": (high * 0.3, high * 0.5),        # 상승 시작
        }

    def get_all_fib_levels(
        self,
        ratios: Tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0),
    ) -> Dict[str, float]:
        """모든 Fib 레벨 가격 반환"""
        return {
            f"{r:.3f}": self.get_price_at_fib(r)
            for r in ratios
        }

    def get_zone(self, price: float, tolerance: float = 0.02) -> Optional[str]:
        """
        현재가가 위치한 Fib 존 반환

        Args:
            price: 현재가
            tolerance: 존 판정 허용 오차 (2%)

        Returns:
            "0.382", "0.5", "0.618" 등 또는 None
        """
        position = self.get_position(price)
        key_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        for level in key_levels:
            if abs(position - level) <= tolerance:
                return f"{level:.3f}"
        return None

    def is_near_crash_support(self, price: float, tolerance_pct: float = 0.05) -> bool:
        """
        현재가가 폭락 지지선(전 사이클 고점) 근처인지

        Args:
            price: 현재가
            tolerance_pct: 허용 오차 (5%)
        """
        support = self.get_crash_support()
        return abs(price - support) / support <= tolerance_pct


# ============================================================================
# Factory Functions
# ============================================================================

def get_btc_cycle_anchor(cycle_num: Optional[int] = None) -> CycleAnchor:
    """
    비트코인 사이클 앵커 가져오기

    Args:
        cycle_num: 사이클 번호 (None이면 현재 사이클)

    Returns:
        CycleAnchor 객체
    """
    if cycle_num is None:
        cycle_num = CURRENT_CYCLE

    current = BTC_CYCLES[cycle_num]
    prev = BTC_CYCLES.get(cycle_num - 1)

    return CycleAnchor(
        cycle_num=current.cycle_num,
        cycle_low=current.cycle_low,
        cycle_high=current.cycle_high,
        cycle_low_date=current.cycle_low_date,
        cycle_high_date=current.cycle_high_date,
        prev_cycle_high=prev.cycle_high if prev else 0,
        prev_cycle_high_date=prev.cycle_high_date if prev else "",
        halving_date=current.halving_date,
    )


def get_fib_levels(
    low: float,
    high: float,
    ratios: Tuple[float, ...] = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0),
    include_extensions: bool = True,
) -> Dict[str, float]:
    """
    Fibonacci 레벨 계산

    Args:
        low: Fib 0 (저점)
        high: Fib 1 (고점)
        ratios: 되돌림 비율
        include_extensions: 확장 레벨 포함 여부

    Returns:
        {레벨: 가격} 딕셔너리
    """
    range_size = high - low
    levels = {}

    for r in ratios:
        levels[f"{r:.3f}"] = low + range_size * r

    if include_extensions:
        extensions = (1.272, 1.618, 2.0, 2.618)
        for ext in extensions:
            levels[f"{ext:.3f}"] = low + range_size * ext

    return levels


def get_current_cycle_position(price: float, symbol: str = "BTC") -> Dict[str, any]:
    """
    현재가의 사이클 위치 정보 반환

    Args:
        price: 현재가
        symbol: 심볼 (현재 BTC만 지원)

    Returns:
        {
            "position": 0.69,
            "zone": "0.618",
            "crash_support": 69000,
            "near_support": False,
            ...
        }
    """
    if symbol.upper() not in ("BTC", "BTC-USDT", "BTCUSDT"):
        raise ValueError(f"Unsupported symbol: {symbol}. Only BTC supported.")

    anchor = get_btc_cycle_anchor()

    return {
        "symbol": "BTC",
        "price": price,
        "cycle_num": anchor.cycle_num,
        "position": anchor.get_position(price),
        "zone": anchor.get_zone(price),
        "cycle_low": anchor.cycle_low,
        "cycle_high": anchor.cycle_high,
        "crash_support": anchor.get_crash_support(),
        "near_support": anchor.is_near_crash_support(price),
        "fib_levels": anchor.get_all_fib_levels(),
    }


# ============================================================================
# Integration with Spectral Analysis
# ============================================================================

def normalize_price_to_cycle(
    prices: np.ndarray,
    anchor: Optional[CycleAnchor] = None,
) -> np.ndarray:
    """
    가격을 사이클 범위로 정규화 (0~1)

    스펙트럴 분석 전처리용:
    - 사이클 저점 = 0
    - 사이클 고점 = 1

    Args:
        prices: 가격 배열
        anchor: CycleAnchor (None이면 현재 BTC 사이클)

    Returns:
        정규화된 가격 (0~1 범위, 범위 초과 가능)
    """
    if anchor is None:
        anchor = get_btc_cycle_anchor()

    return (prices - anchor.cycle_low) / (anchor.cycle_high - anchor.cycle_low)


def denormalize_price_from_cycle(
    normalized: np.ndarray,
    anchor: Optional[CycleAnchor] = None,
) -> np.ndarray:
    """
    정규화된 가격을 실제 가격으로 복원

    Args:
        normalized: 정규화된 가격 (0~1)
        anchor: CycleAnchor

    Returns:
        실제 가격
    """
    if anchor is None:
        anchor = get_btc_cycle_anchor()

    return normalized * (anchor.cycle_high - anchor.cycle_low) + anchor.cycle_low


# ============================================================================
# ETH Cycle Data (참고용 - 아직 미검증)
# ============================================================================

ETH_CYCLES: Dict[int, CycleData] = {
    1: CycleData(
        cycle_num=1,
        halving_date="N/A",  # ETH는 반감기 없음
        cycle_low=80,
        cycle_low_date="2018-12-15",
        cycle_high=4800,
        cycle_high_date="2021-11-10",
    ),
    2: CycleData(
        cycle_num=2,
        halving_date="N/A",
        cycle_low=880,
        cycle_low_date="2022-06-18",
        cycle_high=4500,  # 추정 (미확정)
        cycle_high_date="2025-??-??",
    ),
}


def get_eth_cycle_anchor(cycle_num: int = 2) -> CycleAnchor:
    """ETH 사이클 앵커 (미검증, 참고용)"""
    current = ETH_CYCLES[cycle_num]
    prev = ETH_CYCLES.get(cycle_num - 1)

    return CycleAnchor(
        cycle_num=current.cycle_num,
        cycle_low=current.cycle_low,
        cycle_high=current.cycle_high,
        cycle_low_date=current.cycle_low_date,
        cycle_high_date=current.cycle_high_date,
        prev_cycle_high=prev.cycle_high if prev else 0,
        prev_cycle_high_date=prev.cycle_high_date if prev else "",
        halving_date=current.halving_date,
    )


# ============================================================================
# Fractal Fib Levels (L0, L1)
# ============================================================================

# 1W Fib 앵커 (고정)
FIB_0 = 3120
FIB_1 = 20650
RANGE = FIB_1 - FIB_0  # 17530

# 표준 피보나치 비율
STANDARD_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

# 확장 Fib 최대값 (Fib 8.0 = $143,360)
EXTENDED_FIB_MAX = 8


@dataclass
class FibLevel:
    """순수 Fib 레벨 (Zone Width 없음)"""
    fib_ratio: float      # 전체 Fib 비율 (예: 5.236)
    price: float          # 가격
    depth: int            # 프랙탈 깊이 (0=L0, 1=L1)
    cell: Tuple[int, int]  # 소속 셀 (fib_int_low, fib_int_high)


def fib_to_price(fib_ratio: float, cell_low: float = FIB_0, cell_high: float = None) -> float:
    """
    Fib 비율 → 가격 변환

    Args:
        fib_ratio: 0~1 사이 비율
        cell_low: 셀 하단 가격
        cell_high: 셀 상단 가격 (None이면 1W 기준)
    """
    if cell_high is None:
        cell_high = cell_low + RANGE
    cell_range = cell_high - cell_low
    return cell_low + (fib_ratio * cell_range)


def price_to_fib(price: float) -> float:
    """
    가격 → 전체 Fib 레벨 변환

    Example:
        >>> price_to_fib(95000)
        5.24
    """
    return (price - FIB_0) / RANGE


def get_fractal_fib_levels(
    price_range: Tuple[float, float],
    max_depth: int = 2,
) -> List[FibLevel]:
    """
    프랙탈 Fib 레벨 생성 (순수 가격, Zone Width 없음)

    Args:
        price_range: (min_price, max_price) - 생성할 가격 범위
        max_depth: 프랙탈 깊이 (0=L0만, 1=L0+L1, 2=L0+L1+L2)

    Returns:
        List[FibLevel] - 가격순 정렬

    Example:
        >>> levels = get_fractal_fib_levels((90000, 100000), max_depth=2)
        >>> for lvl in levels:
        ...     print(f"Fib {lvl.fib_ratio:.3f}: ${lvl.price:,.0f} (L{lvl.depth})")
    """
    levels = []

    # 정수 Fib 간격마다 (0~1, 1~2, ..., 7~8)
    for fib_int in range(EXTENDED_FIB_MAX):
        cell_low_price = FIB_0 + (fib_int * RANGE)
        cell_high_price = FIB_0 + ((fib_int + 1) * RANGE)

        # 가격 범위와 겹치지 않으면 스킵
        if cell_high_price < price_range[0] or cell_low_price > price_range[1]:
            continue

        # L0: 이 셀 내의 표준 Fib 레벨
        for ratio in STANDARD_RATIOS:
            price = fib_to_price(ratio, cell_low_price, cell_high_price)

            if price_range[0] <= price <= price_range[1]:
                full_fib = fib_int + ratio
                levels.append(FibLevel(
                    fib_ratio=full_fib,
                    price=price,
                    depth=0,
                    cell=(fib_int, fib_int + 1),
                ))

        # L1: 각 L0 셀 내부를 다시 분할
        if max_depth >= 1:
            for i in range(len(STANDARD_RATIOS) - 1):
                sub_low_ratio = STANDARD_RATIOS[i]
                sub_high_ratio = STANDARD_RATIOS[i + 1]

                sub_low_price = fib_to_price(sub_low_ratio, cell_low_price, cell_high_price)
                sub_high_price = fib_to_price(sub_high_ratio, cell_low_price, cell_high_price)

                # 가격 범위와 겹치지 않으면 스킵
                if sub_high_price < price_range[0] or sub_low_price > price_range[1]:
                    continue

                # L1 레벨 생성 (0.0, 1.0 제외 - L0과 중복)
                for sub_ratio in STANDARD_RATIOS[1:-1]:
                    price = fib_to_price(sub_ratio, sub_low_price, sub_high_price)

                    if price_range[0] <= price <= price_range[1]:
                        # 전체 Fib 비율 계산
                        cell_ratio = sub_low_ratio + sub_ratio * (sub_high_ratio - sub_low_ratio)
                        full_fib = fib_int + cell_ratio

                        levels.append(FibLevel(
                            fib_ratio=full_fib,
                            price=price,
                            depth=1,
                            cell=(fib_int, fib_int + 1),
                        ))

                # L2: L1 셀 내부를 다시 분할
                if max_depth >= 2:
                    for j in range(len(STANDARD_RATIOS) - 1):
                        l1_low_ratio = STANDARD_RATIOS[j]
                        l1_high_ratio = STANDARD_RATIOS[j + 1]

                        l2_low_price = fib_to_price(l1_low_ratio, sub_low_price, sub_high_price)
                        l2_high_price = fib_to_price(l1_high_ratio, sub_low_price, sub_high_price)

                        # 가격 범위와 겹치지 않으면 스킵
                        if l2_high_price < price_range[0] or l2_low_price > price_range[1]:
                            continue

                        # L2 레벨 생성 (0.0, 1.0 제외)
                        for l2_ratio in STANDARD_RATIOS[1:-1]:
                            price = fib_to_price(l2_ratio, l2_low_price, l2_high_price)

                            if price_range[0] <= price <= price_range[1]:
                                # 전체 Fib 비율 계산
                                l1_cell_ratio = l1_low_ratio + l2_ratio * (l1_high_ratio - l1_low_ratio)
                                cell_ratio = sub_low_ratio + l1_cell_ratio * (sub_high_ratio - sub_low_ratio)
                                full_fib = fib_int + cell_ratio

                                levels.append(FibLevel(
                                    fib_ratio=full_fib,
                                    price=price,
                                    depth=2,
                                    cell=(fib_int, fib_int + 1),
                                ))

    # 가격순 정렬, 중복 제거
    levels.sort(key=lambda x: x.price)

    # 중복 제거 (가격 기준 0.01% 이내)
    unique_levels = []
    for lvl in levels:
        if not unique_levels or abs(lvl.price - unique_levels[-1].price) / lvl.price > 0.0001:
            unique_levels.append(lvl)

    return unique_levels


def get_nearby_fib_levels(
    price: float,
    count: int = 5,
    max_depth: int = 1,
    symbol: str = "BTC",
    use_db: bool = True,
) -> Dict[str, List[FibLevel]]:
    """
    현재가 근처의 Fib 레벨 반환

    Args:
        price: 현재가
        count: 위/아래 각각 몇 개씩
        max_depth: 프랙탈 깊이 (0=L0, 1=L0+L1)
        symbol: 에셋 심볼 (DB 사용 시)
        use_db: True면 DB 우선 조회

    Returns:
        {"above": [...], "below": [...]}
    """
    # 1순위: DB에서 조회 (미리 계산된 레벨)
    if use_db and _DB_AVAILABLE and _FibLevelsDB is not None:
        try:
            # max_depth=0 → L0만, max_depth>=1 → 전체 (L0+L1)
            depth_filter = 0 if max_depth == 0 else None
            db_result = _FibLevelsDB.get_nearby(price, count=count, symbol=symbol, depth=depth_filter)

            # DB FibLevel → 로컬 FibLevel 변환
            def convert(db_lvl):
                return FibLevel(
                    fib_ratio=db_lvl.fib_ratio,
                    price=db_lvl.price,
                    depth=db_lvl.depth,
                    cell=(db_lvl.cell_low, db_lvl.cell_high),
                )

            return {
                "above": [convert(lvl) for lvl in db_result["above"]],
                "below": [convert(lvl) for lvl in db_result["below"]],
            }
        except Exception:
            pass  # DB 실패 시 계산 fallback

    # 2순위: 실시간 계산 (fallback)
    margin = price * 0.3
    levels = get_fractal_fib_levels(
        price_range=(price - margin, price + margin),
        max_depth=max_depth,
    )

    above = [lvl for lvl in levels if lvl.price > price][:count]
    below = [lvl for lvl in levels if lvl.price < price][-count:]

    return {"above": above, "below": below}
