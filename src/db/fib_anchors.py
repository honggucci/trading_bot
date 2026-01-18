"""
Multi-Asset Fib Anchor Database Module

피보나치 앵커 포인트를 MSSQL에 저장/조회
- BTC, ETH, XRP 등 다중 에셋 지원
- 에셋별 Fib 0 (이전 사이클 ATH), Fib 1 (현 사이클 지지) 저장
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

from .connection import execute_query, execute_non_query, get_cursor


@dataclass
class FibAnchor:
    """피보나치 앵커 데이터"""
    symbol: str          # 'BTC', 'ETH', 'XRP'
    fib_0: float         # 앵커 기준점 (이전 사이클 ATH)
    fib_1: float         # 두 번째 기준점 (현 사이클 지지)
    notes: str = ""      # 앵커 설정 근거
    updated_at: Optional[datetime] = None

    @property
    def fib_range(self) -> float:
        """Fib 레벨 간 가격 범위"""
        return self.fib_1 - self.fib_0

    def get_price(self, fib_level: float) -> float:
        """Fib 레벨 → 가격

        Args:
            fib_level: 피보나치 레벨 (0, 1, 2, ..., 12, ...)

        Returns:
            해당 레벨의 가격
        """
        return self.fib_0 + fib_level * self.fib_range

    def get_level(self, price: float) -> float:
        """가격 → Fib 레벨

        Args:
            price: 현재 가격

        Returns:
            해당 가격의 Fib 레벨 (소수점 포함)
        """
        if self.fib_range == 0:
            return 0.0
        return (price - self.fib_0) / self.fib_range

    def get_fib_extension_prices(self, max_level: int = 12) -> Dict[int, float]:
        """Fib 확장 레벨별 가격 반환

        Args:
            max_level: 최대 Fib 레벨 (기본 12)

        Returns:
            {0: 3120, 1: 20650, 2: 38180, ..., 12: 213480}
        """
        return {i: self.get_price(i) for i in range(max_level + 1)}


class FibAnchorDB:
    """Fib Anchor CRUD 클래스"""

    TABLE_NAME = "fib_anchors"

    @classmethod
    def init_table(cls) -> bool:
        """테이블 생성 (없으면)"""
        query = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{cls.TABLE_NAME}' AND xtype='U')
        CREATE TABLE {cls.TABLE_NAME} (
            id INT IDENTITY(1,1) PRIMARY KEY,
            symbol NVARCHAR(10) NOT NULL UNIQUE,
            fib_0 DECIMAL(18,8) NOT NULL,
            fib_1 DECIMAL(18,8) NOT NULL,
            notes NVARCHAR(500),
            created_at DATETIME DEFAULT GETDATE(),
            updated_at DATETIME DEFAULT GETDATE()
        )
        """
        try:
            execute_non_query(query)
            return True
        except Exception as e:
            print(f"테이블 생성 실패: {e}")
            return False

    @classmethod
    def get(cls, symbol: str) -> Optional[FibAnchor]:
        """에셋별 앵커 조회

        Args:
            symbol: 에셋 심볼 ('BTC', 'ETH', 'XRP')

        Returns:
            FibAnchor 또는 None
        """
        query = f"""
        SELECT symbol, fib_0, fib_1, notes, updated_at
        FROM {cls.TABLE_NAME}
        WHERE symbol = ?
        """
        rows = execute_query(query, (symbol.upper(),))

        if rows:
            row = rows[0]
            return FibAnchor(
                symbol=row['symbol'],
                fib_0=float(row['fib_0']),
                fib_1=float(row['fib_1']),
                notes=row['notes'] or "",
                updated_at=row['updated_at'],
            )
        return None

    @classmethod
    def get_all(cls) -> List[FibAnchor]:
        """모든 앵커 조회"""
        query = f"""
        SELECT symbol, fib_0, fib_1, notes, updated_at
        FROM {cls.TABLE_NAME}
        ORDER BY symbol
        """
        rows = execute_query(query)

        return [
            FibAnchor(
                symbol=row['symbol'],
                fib_0=float(row['fib_0']),
                fib_1=float(row['fib_1']),
                notes=row['notes'] or "",
                updated_at=row['updated_at'],
            )
            for row in rows
        ]

    @classmethod
    def upsert(cls, anchor: FibAnchor) -> bool:
        """앵커 생성 또는 업데이트 (UPSERT)

        Args:
            anchor: FibAnchor 객체

        Returns:
            성공 여부
        """
        query = f"""
        MERGE {cls.TABLE_NAME} AS target
        USING (SELECT ? AS symbol, ? AS fib_0, ? AS fib_1, ? AS notes) AS source
        ON target.symbol = source.symbol
        WHEN MATCHED THEN
            UPDATE SET
                fib_0 = source.fib_0,
                fib_1 = source.fib_1,
                notes = source.notes,
                updated_at = GETDATE()
        WHEN NOT MATCHED THEN
            INSERT (symbol, fib_0, fib_1, notes)
            VALUES (source.symbol, source.fib_0, source.fib_1, source.notes);
        """
        try:
            execute_non_query(query, (
                anchor.symbol.upper(),
                anchor.fib_0,
                anchor.fib_1,
                anchor.notes,
            ))
            return True
        except Exception as e:
            print(f"UPSERT 실패: {e}")
            return False

    @classmethod
    def delete(cls, symbol: str) -> bool:
        """앵커 삭제

        Args:
            symbol: 에셋 심볼

        Returns:
            성공 여부
        """
        query = f"DELETE FROM {cls.TABLE_NAME} WHERE symbol = ?"
        try:
            affected = execute_non_query(query, (symbol.upper(),))
            return affected > 0
        except Exception as e:
            print(f"삭제 실패: {e}")
            return False

    @classmethod
    def seed_defaults(cls) -> int:
        """기본 앵커 데이터 시딩 (없는 것만)

        Returns:
            추가된 행 수
        """
        defaults = [
            FibAnchor(
                symbol="BTC",
                fib_0=3120.0,
                fib_1=20650.0,
                notes="Cycle 3 ATH ($3120) → Cycle 4 support ($20650). Range=$17530",
            ),
            FibAnchor(
                symbol="ETH",
                fib_0=88.0,
                fib_1=1420.0,
                notes="2018 ATH (~$88) → 2022 support (~$1420). Range=$1332",
            ),
            FibAnchor(
                symbol="XRP",
                fib_0=0.10,
                fib_1=1.96,
                notes="Historical low (~$0.10) → 2018 ATH ($1.96). Range=$1.86",
            ),
        ]

        count = 0
        for anchor in defaults:
            existing = cls.get(anchor.symbol)
            if not existing:
                if cls.upsert(anchor):
                    count += 1
                    print(f"  [SEED] {anchor.symbol}: Fib0=${anchor.fib_0}, Fib1=${anchor.fib_1}")

        return count


# ============================================================================
# Utility Functions (cycle_anchor.py 호환)
# ============================================================================

def get_fib_anchor(symbol: str = "BTC") -> Optional[FibAnchor]:
    """심볼별 Fib 앵커 조회 (단축 함수)

    cycle_anchor.py에서 DB 연동 시 사용

    Args:
        symbol: 에셋 심볼 (기본 BTC)

    Returns:
        FibAnchor 또는 None (DB 연결 실패 시)
    """
    try:
        return FibAnchorDB.get(symbol)
    except Exception:
        return None


def get_1w_fib_level_db(price: float, symbol: str = "BTC") -> Optional[float]:
    """현재가의 1W Fib 레벨 계산 (DB 버전)

    Args:
        price: 현재 가격
        symbol: 에셋 심볼

    Returns:
        Fib 레벨 (예: 5.7 = Fib 5.7 레벨) 또는 None
    """
    anchor = get_fib_anchor(symbol)
    if anchor:
        return anchor.get_level(price)
    return None


def get_1w_fib_price_db(fib_level: float, symbol: str = "BTC") -> Optional[float]:
    """Fib 레벨에 해당하는 가격 계산 (DB 버전)

    Args:
        fib_level: 피보나치 레벨
        symbol: 에셋 심볼

    Returns:
        가격 또는 None
    """
    anchor = get_fib_anchor(symbol)
    if anchor:
        return anchor.get_price(fib_level)
    return None


# ============================================================================
# FibLevel (fib_levels 테이블용)
# ============================================================================

@dataclass
class FibLevel:
    """미리 계산된 Fib 레벨"""
    symbol: str
    fib_ratio: float    # 예: 5.618
    price: float        # 예: 101,612
    depth: int          # 0=L0, 1=L1
    cell_low: int       # 소속 셀 하단
    cell_high: int      # 소속 셀 상단


class FibLevelsDB:
    """fib_levels 테이블 조회 클래스"""

    TABLE_NAME = "fib_levels"

    @classmethod
    def get_all(cls, symbol: str = "BTC", depth: Optional[int] = None) -> List[FibLevel]:
        """모든 레벨 조회

        Args:
            symbol: 에셋 심볼
            depth: None=전체, 0=L0만, 1=L1만

        Returns:
            List[FibLevel]
        """
        if depth is not None:
            query = f"""
            SELECT symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? AND depth = ?
            ORDER BY fib_ratio
            """
            rows = execute_query(query, (symbol.upper(), depth))
        else:
            query = f"""
            SELECT symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ?
            ORDER BY fib_ratio
            """
            rows = execute_query(query, (symbol.upper(),))

        return [
            FibLevel(
                symbol=row['symbol'],
                fib_ratio=float(row['fib_ratio']),
                price=float(row['price']),
                depth=row['depth'],
                cell_low=row['cell_low'],
                cell_high=row['cell_high'],
            )
            for row in rows
        ]

    @classmethod
    def get_in_range(
        cls,
        price_min: float,
        price_max: float,
        symbol: str = "BTC",
        depth: Optional[int] = None,
    ) -> List[FibLevel]:
        """가격 범위 내 레벨 조회

        Args:
            price_min: 최소 가격
            price_max: 최대 가격
            symbol: 에셋 심볼
            depth: None=전체, 0=L0만, 1=L1만

        Returns:
            List[FibLevel]
        """
        if depth is not None:
            query = f"""
            SELECT symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? AND price BETWEEN ? AND ? AND depth = ?
            ORDER BY price
            """
            rows = execute_query(query, (symbol.upper(), price_min, price_max, depth))
        else:
            query = f"""
            SELECT symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? AND price BETWEEN ? AND ?
            ORDER BY price
            """
            rows = execute_query(query, (symbol.upper(), price_min, price_max))

        return [
            FibLevel(
                symbol=row['symbol'],
                fib_ratio=float(row['fib_ratio']),
                price=float(row['price']),
                depth=row['depth'],
                cell_low=row['cell_low'],
                cell_high=row['cell_high'],
            )
            for row in rows
        ]

    @classmethod
    def get_nearby(
        cls,
        price: float,
        count: int = 5,
        symbol: str = "BTC",
        depth: Optional[int] = None,
    ) -> Dict[str, List[FibLevel]]:
        """현재가 근처 레벨 조회

        Args:
            price: 현재가
            count: 위/아래 각각 몇 개씩
            symbol: 에셋 심볼
            depth: None=전체, 0=L0만, 1=L1만

        Returns:
            {"above": [...], "below": [...]}
        """
        depth_clause = "AND depth = ?" if depth is not None else ""
        params_above = (symbol.upper(), price, depth, count) if depth is not None else (symbol.upper(), price, count)
        params_below = (symbol.upper(), price, depth, count) if depth is not None else (symbol.upper(), price, count)

        # Above
        query_above = f"""
        SELECT TOP (?) symbol, fib_ratio, price, depth, cell_low, cell_high
        FROM {cls.TABLE_NAME}
        WHERE symbol = ? AND price > ? {depth_clause}
        ORDER BY price ASC
        """
        if depth is not None:
            rows_above = execute_query(query_above, (count, symbol.upper(), price, depth))
        else:
            query_above = f"""
            SELECT TOP (?) symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? AND price > ?
            ORDER BY price ASC
            """
            rows_above = execute_query(query_above, (count, symbol.upper(), price))

        # Below
        if depth is not None:
            query_below = f"""
            SELECT TOP (?) symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? AND price < ? {depth_clause}
            ORDER BY price DESC
            """
            rows_below = execute_query(query_below, (count, symbol.upper(), price, depth))
        else:
            query_below = f"""
            SELECT TOP (?) symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? AND price < ?
            ORDER BY price DESC
            """
            rows_below = execute_query(query_below, (count, symbol.upper(), price))

        def to_level(row):
            return FibLevel(
                symbol=row['symbol'],
                fib_ratio=float(row['fib_ratio']),
                price=float(row['price']),
                depth=row['depth'],
                cell_low=row['cell_low'],
                cell_high=row['cell_high'],
            )

        return {
            "above": [to_level(r) for r in rows_above],
            "below": [to_level(r) for r in reversed(rows_below)],  # 가격순 정렬
        }

    @classmethod
    def get_nearest(cls, price: float, symbol: str = "BTC", depth: Optional[int] = None) -> Optional[FibLevel]:
        """현재가에 가장 가까운 레벨

        Args:
            price: 현재가
            symbol: 에셋 심볼
            depth: None=전체, 0=L0만, 1=L1만

        Returns:
            가장 가까운 FibLevel
        """
        depth_clause = "AND depth = ?" if depth is not None else ""

        if depth is not None:
            query = f"""
            SELECT TOP 1 symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ? {depth_clause}
            ORDER BY ABS(price - ?)
            """
            rows = execute_query(query, (symbol.upper(), depth, price))
        else:
            query = f"""
            SELECT TOP 1 symbol, fib_ratio, price, depth, cell_low, cell_high
            FROM {cls.TABLE_NAME}
            WHERE symbol = ?
            ORDER BY ABS(price - ?)
            """
            rows = execute_query(query, (symbol.upper(), price))

        if rows:
            row = rows[0]
            return FibLevel(
                symbol=row['symbol'],
                fib_ratio=float(row['fib_ratio']),
                price=float(row['price']),
                depth=row['depth'],
                cell_low=row['cell_low'],
                cell_high=row['cell_high'],
            )
        return None


# ============================================================================
# ZoneParam (zone_params 테이블용)
# ============================================================================

@dataclass
class ZoneParam:
    """Zone 폭 계산 파라미터"""
    timeframe: str       # '15m', '1h', '4h', '1d'
    atr_window: Optional[int]   # ATR 계산 윈도우
    k: Optional[float]          # ATR 곱수
    min_pct: Optional[float]    # 최소 폭 (%)
    max_pct: Optional[float]    # 최대 폭 (%)
    role: str = ""              # 'zone_generator', 'context_filter', 'htf_filter'
    notes: str = ""
    updated_at: Optional[datetime] = None


class ZoneParamsDB:
    """zone_params 테이블 CRUD 클래스"""

    TABLE_NAME = "zone_params"

    @classmethod
    def init_table(cls) -> bool:
        """테이블 생성 (없으면)"""
        query = f"""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{cls.TABLE_NAME}' AND xtype='U')
        CREATE TABLE {cls.TABLE_NAME} (
            id INT IDENTITY(1,1) PRIMARY KEY,
            timeframe NVARCHAR(10) NOT NULL UNIQUE,
            atr_window INT,
            k DECIMAL(6,4),
            min_pct DECIMAL(8,6),
            max_pct DECIMAL(8,6),
            role NVARCHAR(30),
            notes NVARCHAR(500),
            created_at DATETIME DEFAULT GETDATE(),
            updated_at DATETIME DEFAULT GETDATE()
        )
        """
        try:
            execute_non_query(query)
            return True
        except Exception as e:
            print(f"테이블 생성 실패: {e}")
            return False

    @classmethod
    def get(cls, timeframe: str) -> Optional[ZoneParam]:
        """TF별 파라미터 조회"""
        query = f"""
        SELECT timeframe, atr_window, k, min_pct, max_pct, role, notes, updated_at
        FROM {cls.TABLE_NAME}
        WHERE timeframe = ?
        """
        rows = execute_query(query, (timeframe.lower(),))

        if rows:
            row = rows[0]
            return ZoneParam(
                timeframe=row['timeframe'],
                atr_window=row['atr_window'],
                k=float(row['k']) if row['k'] else None,
                min_pct=float(row['min_pct']) if row['min_pct'] else None,
                max_pct=float(row['max_pct']) if row['max_pct'] else None,
                role=row['role'] or "",
                notes=row['notes'] or "",
                updated_at=row['updated_at'],
            )
        return None

    @classmethod
    def get_all(cls) -> List[ZoneParam]:
        """모든 파라미터 조회"""
        query = f"""
        SELECT timeframe, atr_window, k, min_pct, max_pct, role, notes, updated_at
        FROM {cls.TABLE_NAME}
        ORDER BY
            CASE timeframe
                WHEN '1w' THEN 1
                WHEN '1d' THEN 2
                WHEN '4h' THEN 3
                WHEN '1h' THEN 4
                WHEN '15m' THEN 5
                ELSE 6
            END
        """
        rows = execute_query(query)

        return [
            ZoneParam(
                timeframe=row['timeframe'],
                atr_window=row['atr_window'],
                k=float(row['k']) if row['k'] else None,
                min_pct=float(row['min_pct']) if row['min_pct'] else None,
                max_pct=float(row['max_pct']) if row['max_pct'] else None,
                role=row['role'] or "",
                notes=row['notes'] or "",
                updated_at=row['updated_at'],
            )
            for row in rows
        ]

    @classmethod
    def upsert(cls, param: ZoneParam) -> bool:
        """파라미터 생성 또는 업데이트"""
        query = f"""
        MERGE {cls.TABLE_NAME} AS target
        USING (SELECT ? AS timeframe, ? AS atr_window, ? AS k, ? AS min_pct, ? AS max_pct, ? AS role, ? AS notes) AS source
        ON target.timeframe = source.timeframe
        WHEN MATCHED THEN
            UPDATE SET
                atr_window = source.atr_window,
                k = source.k,
                min_pct = source.min_pct,
                max_pct = source.max_pct,
                role = source.role,
                notes = source.notes,
                updated_at = GETDATE()
        WHEN NOT MATCHED THEN
            INSERT (timeframe, atr_window, k, min_pct, max_pct, role, notes)
            VALUES (source.timeframe, source.atr_window, source.k, source.min_pct, source.max_pct, source.role, source.notes);
        """
        try:
            execute_non_query(query, (
                param.timeframe.lower(),
                param.atr_window,
                param.k,
                param.min_pct,
                param.max_pct,
                param.role,
                param.notes,
            ))
            return True
        except Exception as e:
            print(f"UPSERT 실패: {e}")
            return False

    @classmethod
    def seed_defaults(cls) -> int:
        """기본 파라미터 시딩 (zone_width.json 기준)"""
        defaults = [
            ZoneParam(
                timeframe="1w",
                atr_window=None,
                k=None,
                min_pct=None,
                max_pct=None,
                role="fib_coordinate_only",
                notes="Fib 좌표계만, Zone Width 계산 제외",
            ),
            ZoneParam(
                timeframe="1d",
                atr_window=89,
                k=1.0,
                min_pct=0.005,
                max_pct=0.03,
                role="htf_filter",
                notes="HTF 필터, 긴 윈도우로 안정성 확보",
            ),
            ZoneParam(
                timeframe="4h",
                atr_window=21,
                k=1.65,
                min_pct=0.005,
                max_pct=0.03,
                role="context_filter",
                notes="Context 필터만, 트리거로 사용 금지",
            ),
            ZoneParam(
                timeframe="1h",
                atr_window=21,
                k=2.4,
                min_pct=0.002,
                max_pct=0.015,
                role="context_filter",
                notes="Context 필터만, 트리거로 사용 금지",
            ),
            ZoneParam(
                timeframe="15m",
                atr_window=21,
                k=2.75,
                min_pct=0.002,
                max_pct=0.015,
                role="zone_generator",
                notes="Zone 생성 TF",
            ),
        ]

        count = 0
        for param in defaults:
            existing = cls.get(param.timeframe)
            if not existing:
                if cls.upsert(param):
                    count += 1
                    print(f"  [SEED] {param.timeframe}: ATR={param.atr_window}, k={param.k}")

        return count
