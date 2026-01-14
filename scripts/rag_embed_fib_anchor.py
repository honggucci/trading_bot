# -*- coding: utf-8 -*-
"""
RAG Embedding - 1W Fibonacci Anchor Session
============================================

2026-01-15 세션: 1W 피보나치 앵커 확정
"""
import os
import sys
import json

HATTZ_EMPIRE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\hattz_empire"
sys.path.insert(0, HATTZ_EMPIRE_PATH)

from src.services.rag import index_document

PROJECT = "trading_bot"
BASE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\projects\trading_bot"

# 이번 세션에서 생성된 파일
SESSION_FILES = [
    # 1W 피보나치 앵커 (핵심!)
    "config/fib_1w_anchor.json",

    # 세션 백업 문서
    "docs/session_backup_20260115_fib_anchor.md",
]


def embed_file(rel_path: str) -> bool:
    """파일 임베딩"""
    full_path = os.path.join(BASE_PATH, rel_path)

    if not os.path.exists(full_path):
        print(f"  [SKIP] {rel_path} - not found")
        return False

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        source_id = f"{PROJECT}/{rel_path}"

        # JSON 파일은 설명 추가
        if rel_path.endswith('.json'):
            metadata = {
                "project": PROJECT,
                "file_path": rel_path,
                "session": "fib_anchor_20260115",
                "type": "config",
                "description": "1W 피보나치 고정 앵커 - 매매 철학의 기준점",
            }
        else:
            metadata = {
                "project": PROJECT,
                "file_path": rel_path,
                "session": "fib_anchor_20260115",
                "type": "documentation",
            }

        result = index_document(
            source_type="code" if rel_path.endswith('.py') else "document",
            source_id=source_id,
            content=content,
            metadata=metadata,
            project=PROJECT,
            source="file",
        )

        print(f"  [OK] {rel_path}")
        return True

    except Exception as e:
        print(f"  [ERR] {rel_path}: {e}")
        return False


def embed_summary() -> bool:
    """핵심 내용 요약 임베딩"""
    summary = """
# 1W 피보나치 앵커 (2026-01-15 확정)

## 앵커 값 (불변)
- Fib 0 = $3,120 (2018년 저점)
- Fib 1 = $20,650 (2017/18년 고점)
- Range = $17,530

## 검증된 가격
- $15,476 (22년 저점) = Fib 0.705 ≈ 0.702 (0.4% 오차)
- $28,800 (22년 반등) = Fib 1.465 ≈ 1.500 (2.3% 오차)
- $64,863 (21년 4월) = Fib 3.522 ≈ 3.618 (2.7% 오차)
- $69,000 (21년 11월) = Fib 3.759 ≈ 3.786 (0.7% 오차)

## 4차 사이클 고점 예측
- 보수적: $126K (Fib 7.0)
- 중립: $200K (Fib 11.236)
- 낙관: $254K (Fib 14.326)
- 공격: $266K (Fib 15.0)

## 핵심 철학
"전 사이클 고점 = 현 사이클 저점"
$69,000은 폭락 시 최종 지지선.

## 파일
- config/fib_1w_anchor.json: 전체 Fib 레벨 테이블 (0~15)
"""

    try:
        result = index_document(
            source_type="summary",
            source_id=f"{PROJECT}/summary/fib_anchor_20260115",
            content=summary,
            metadata={
                "project": PROJECT,
                "session": "fib_anchor_20260115",
                "type": "summary",
                "importance": "critical",
                "description": "1W 피보나치 앵커 확정 세션 요약",
            },
            project=PROJECT,
            source="session",
        )

        print(f"  [OK] 세션 요약")
        return True

    except Exception as e:
        print(f"  [ERR] 세션 요약: {e}")
        return False


def main():
    print("=" * 60)
    print("RAG Embedding - 1W Fibonacci Anchor Session")
    print(f"Project: {PROJECT}")
    print("Date: 2026-01-15")
    print("=" * 60)

    success = 0
    failed = 0

    # 파일 임베딩
    print("\n[Files]")
    for rel_path in SESSION_FILES:
        if embed_file(rel_path):
            success += 1
        else:
            failed += 1

    # 요약 임베딩
    print("\n[Summary]")
    if embed_summary():
        success += 1
    else:
        failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {success}/{success + failed} items embedded")
    print("=" * 60)


if __name__ == "__main__":
    main()