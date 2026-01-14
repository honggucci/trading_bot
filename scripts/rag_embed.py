# -*- coding: utf-8 -*-
"""
Trading Bot RAG Embedding Script
================================

trading_bot 프로젝트의 핵심 파일들을 hattz_empire RAG 시스템에 임베딩.
"""
import os
import sys

# hattz_empire 경로 추가
HATTZ_EMPIRE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\hattz_empire"
sys.path.insert(0, HATTZ_EMPIRE_PATH)

from src.services.rag import index_document

# 프로젝트 설정
PROJECT_NAME = "trading_bot"
TRADING_BOT_PATH = r"C:\Users\hahonggu\Desktop\coin_master\projects\trading_bot"

# 임베딩할 핵심 파일들
CORE_FILES = [
    # Context (새로 만든 것들)
    "src/context/cycle_anchor.py",
    "src/context/multi_tf_fib.py",
    "src/context/tf_predictor.py",
    "src/context/zigzag.py",
    "src/context/fibonacci.py",
    "src/context/__init__.py",

    # Wyckoff + Spectral
    "src/wyckoff/spectral.py",
    "src/wyckoff/phases.py",
    "src/wyckoff/box.py",
    "src/wyckoff/indicators.py",
    "src/wyckoff/__init__.py",

    # Anchor (Legacy Confluence)
    "src/anchor/unified_signal.py",
    "src/anchor/legacy_pipeline.py",
    "src/anchor/confluence.py",
    "src/anchor/divergence.py",
    "src/anchor/stochrsi.py",
    "src/anchor/__init__.py",

    # HMM
    "src/hmm/train.py",
    "src/hmm/features.py",
    "src/hmm/states.py",
    "src/hmm/__init__.py",

    # Gate
    "src/gate/hmm_entry_gate.py",
    "src/gate/__init__.py",

    # Tests
    "tests/test_multi_tf_fib.py",
    "tests/test_real_btc_data.py",
]


def embed_file(filepath: str, project: str) -> bool:
    """단일 파일 임베딩"""
    full_path = os.path.join(TRADING_BOT_PATH, filepath)

    if not os.path.exists(full_path):
        print(f"  [SKIP] {filepath} - not found")
        return False

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) < 50:
            print(f"  [SKIP] {filepath} - too short")
            return False

        # 임베딩
        doc_id = index_document(
            source_type="code",
            source_id=filepath,
            content=content,
            metadata={
                "filepath": filepath,
                "project": project,
                "type": "python",
            },
            project=project,
            source="cli",
        )

        print(f"  [OK] {filepath} -> {doc_id[:8]}...")
        return True

    except Exception as e:
        print(f"  [ERROR] {filepath} - {e}")
        return False


def main():
    print("=" * 60)
    print(f"Trading Bot RAG Embedding")
    print(f"Project: {PROJECT_NAME}")
    print("=" * 60)

    # 파일 임베딩
    print(f"\nEmbedding {len(CORE_FILES)} files...")

    success = 0
    failed = 0

    for filepath in CORE_FILES:
        if embed_file(filepath, PROJECT_NAME):
            success += 1
        else:
            failed += 1

    # 요약
    print("\n" + "=" * 60)
    print(f"Embedding Complete")
    print(f"  Success: {success}")
    print(f"  Failed/Skipped: {failed}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
