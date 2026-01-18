"""
Trading Bot DB 초기화 스크립트

Usage:
    python scripts/setup_fib_db.py

테이블:
    - fib_anchors: Fib 앵커 (BTC, ETH, XRP)
    - zone_params: Zone 폭 파라미터 (TF별)

환경변수 필요:
    - MSSQL_SERVER
    - MSSQL_DATABASE
    - MSSQL_USER
    - MSSQL_PASSWORD

또는 hattz_empire/.env 파일이 있으면 자동 로드
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import test_connection
from src.db.fib_anchors import FibAnchorDB, FibAnchor, ZoneParamsDB, ZoneParam


def main():
    print("=" * 60)
    print("Trading Bot DB Setup")
    print("=" * 60)

    # 1. 연결 테스트
    print("\n[1/4] Testing DB connection...")
    if not test_connection():
        print("  FAILED - Check MSSQL environment variables")
        return 1
    print("  OK")

    # 2. fib_anchors 테이블
    print("\n[2/4] Creating fib_anchors table...")
    if FibAnchorDB.init_table():
        print("  OK")
    else:
        print("  FAILED")
        return 1

    # 3. zone_params 테이블
    print("\n[3/4] Creating zone_params table...")
    if ZoneParamsDB.init_table():
        print("  OK")
    else:
        print("  FAILED")
        return 1

    # 4. 기본 데이터 시딩
    print("\n[4/4] Seeding default data...")
    anchor_count = FibAnchorDB.seed_defaults()
    zone_count = ZoneParamsDB.seed_defaults()
    print(f"  Added {anchor_count} anchor(s), {zone_count} zone param(s)")

    # ========== 결과 출력 ==========
    print("\n" + "=" * 60)
    print("Fib Anchors:")
    print("=" * 60)
    anchors = FibAnchorDB.get_all()
    if not anchors:
        print("  (no data)")
    else:
        print(f"{'Symbol':<8} {'Fib 0':>12} {'Fib 1':>12} {'Range':>12} {'Fib 12':>14}")
        print("-" * 60)
        for a in anchors:
            fib12 = a.get_price(12)
            print(f"{a.symbol:<8} {a.fib_0:>12,.2f} {a.fib_1:>12,.2f} {a.fib_range:>12,.2f} {fib12:>14,.2f}")

    print("\n" + "=" * 60)
    print("Zone Params:")
    print("=" * 60)
    params = ZoneParamsDB.get_all()
    if not params:
        print("  (no data)")
    else:
        print(f"{'TF':<6} {'ATR':>6} {'k':>8} {'min%':>10} {'max%':>10} {'Role':<20}")
        print("-" * 60)
        for p in params:
            atr = str(p.atr_window) if p.atr_window else "-"
            k = f"{p.k:.2f}" if p.k else "-"
            min_pct = f"{p.min_pct:.4f}" if p.min_pct else "-"
            max_pct = f"{p.max_pct:.4f}" if p.max_pct else "-"
            print(f"{p.timeframe:<6} {atr:>6} {k:>8} {min_pct:>10} {max_pct:>10} {p.role:<20}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
