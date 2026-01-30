"""
MSSQL 데이터베이스 연결 모듈

환경변수:
    MSSQL_SERVER: 서버 주소 (기본: localhost,1433)
    MSSQL_DATABASE: 데이터베이스명 (기본: HattzEmpire)
    MSSQL_USER: 사용자명
    MSSQL_PASSWORD: 비밀번호
"""

import os
from typing import Optional, List, Any, Dict
from contextlib import contextmanager
from pathlib import Path

# .env 파일 로드 (trading_bot/.env 우선, hattz_empire/.env에서 credentials 상속)
try:
    from dotenv import load_dotenv

    # 1순위: hattz_empire/.env (SERVER/USER/PASSWORD)
    hattz_env = Path(__file__).parent.parent.parent.parent.parent / "hattz_empire" / ".env"
    if hattz_env.exists():
        load_dotenv(hattz_env)

    # 2순위: trading_bot/.env (DATABASE 오버라이드)
    local_env = Path(__file__).parent.parent.parent / ".env"
    if local_env.exists():
        load_dotenv(local_env, override=True)  # 로컬 설정이 우선
except ImportError:
    pass  # dotenv 없으면 시스템 환경변수만 사용


def get_connection():
    """MSSQL 연결 객체 반환"""
    import pyodbc

    server = os.getenv("MSSQL_SERVER", "localhost,1433")
    database = os.getenv("MSSQL_DATABASE", "trading_bot")  # trading_bot DB 사용
    user = os.getenv("MSSQL_USER")
    password = os.getenv("MSSQL_PASSWORD")

    if not user or not password:
        raise ValueError("MSSQL_USER and MSSQL_PASSWORD environment variables required")

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )

    return pyodbc.connect(conn_str)


@contextmanager
def get_cursor():
    """커서를 자동으로 닫는 컨텍스트 매니저"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def execute_query(query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """SELECT 쿼리 실행 후 결과 반환 (dict list)"""
    with get_cursor() as cursor:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        columns = [column[0] for column in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]


def execute_non_query(query: str, params: Optional[tuple] = None) -> int:
    """INSERT/UPDATE/DELETE 실행 후 영향받은 행 수 반환"""
    with get_cursor() as cursor:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.rowcount


def test_connection() -> bool:
    """연결 테스트"""
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
            return True
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return False
