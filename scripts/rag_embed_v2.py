# -*- coding: utf-8 -*-
"""
RAG Embedding V2 - New Files Only
=================================

신규 생성된 파일들만 RAG 임베딩.
"""
import os
import sys

HATTZ_EMPIRE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\hattz_empire"
sys.path.insert(0, HATTZ_EMPIRE_PATH)

from src.services.rag import index_document

PROJECT = "trading_bot"
BASE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\projects\trading_bot"

# 신규 파일들 (이번 세션에서 생성됨)
NEW_FILES = [
    # Exit Logic
    "src/anchor/exit_logic.py",

    # Unified Signal V3
    "src/anchor/unified_signal_v3.py",

    # Backtest Engine
    "src/backtest/__init__.py",
    "src/backtest/engine.py",

    # Tests
    "tests/test_exit_logic.py",
    "tests/test_backtest_engine.py",
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

        result = index_document(
            source_type="code",
            source_id=source_id,
            content=content,
            metadata={
                "project": PROJECT,
                "file_path": rel_path,
                "session": "v3_integration",
            },
            project=PROJECT,
            source="file",
        )

        print(f"  [OK] {rel_path}")
        return True

    except Exception as e:
        print(f"  [ERR] {rel_path}: {e}")
        return False


def main():
    print("=" * 60)
    print("RAG Embedding V2 - New Files")
    print(f"Project: {PROJECT}")
    print("=" * 60)

    success = 0
    failed = 0

    for rel_path in NEW_FILES:
        if embed_file(rel_path):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {success}/{success + failed} files embedded")
    print("=" * 60)


if __name__ == "__main__":
    main()