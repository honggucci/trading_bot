# -*- coding: utf-8 -*-
"""
RAG Embedding - Zone Width Session (2026-01-15)
===============================================

Zone Width 파라미터 확정 세션 파일 임베딩.
"""
import os
import sys

HATTZ_EMPIRE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\hattz_empire"
sys.path.insert(0, HATTZ_EMPIRE_PATH)

from src.services.rag import index_document

PROJECT = "trading_bot"
BASE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\projects\trading_bot"

# 이번 세션 파일들
SESSION_FILES = [
    # Config
    "config/zone_width.json",

    # Volatility Module
    "src/context/volatility.py",

    # Experiment Scripts
    "scripts/experiment_atr_window_multi_tf.py",
    "scripts/experiment_atr_window_critical.py",
    "scripts/experiment_atr_all_tf_oos.py",
    "scripts/monitor_zone_coverage.py",

    # Session Backup
    "docs/session_backups/session_backup_20260115_1600.md",
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
            source_type="code" if rel_path.endswith('.py') else "document",
            source_id=source_id,
            content=content,
            metadata={
                "project": PROJECT,
                "file_path": rel_path,
                "session": "zone_width_20260115",
                "tags": ["zone_width", "atr", "volatility", "parameter"],
            },
            project=PROJECT,
            source="file",
        )

        print(f"  [OK] {rel_path}")
        return True

    except Exception as e:
        print(f"  [ERR] {rel_path}: {e}")
        return False


def embed_session_summary():
    """세션 요약 임베딩"""
    summary = """
# Zone Width Parameter Optimization Session Summary

## Key Decisions
- 1W: Fib coordinate only (no Zone Width calculation, data too sparse)
- 1D: ATR(89), k=1.0 (HTF filter, exception due to -29% OOS loss with ATR(21))
- 4H: ATR(21), k=1.65 (Context filter only)
- 1H: ATR(21), k=2.4 (Context filter only)
- 15m: ATR(21), k=2.75 (Zone generator)

## 1W Fib Anchor (Fixed)
- Fib 0: $3,120
- Fib 1: $20,650
- Range: $17,530
- Data Source: Binance Futures

## Rejected Approaches
- Hilbert amplitude for volatility prediction (OOS Pearson 0.318, last place)
- EMA detrend (replaced with Fib detrend)
- 1W Zone Width calculation

## Recalibration Protocol
- Trigger: 3-month rolling Coverage < 40% or > 60%
- Change condition: OOS Pearson improvement >= 10%
- Do NOT change for: losses, intuition, new papers

## Trade Setup Grades
- A-grade: 15m Zone + 1H Zone both inside
- B-grade: 15m Zone inside, 1H outside

## Next Steps
- 5m trigger quality improvement
- Accumulate 50-200 trade samples
"""
    try:
        result = index_document(
            source_type="session_summary",
            source_id=f"{PROJECT}/sessions/zone_width_20260115",
            content=summary,
            metadata={
                "project": PROJECT,
                "session": "zone_width_20260115",
                "tags": ["zone_width", "atr", "fib", "parameter_optimization", "summary"],
            },
            project=PROJECT,
            source="session",
        )
        print(f"  [OK] Session summary")
        return True
    except Exception as e:
        print(f"  [ERR] Session summary: {e}")
        return False


def main():
    print("=" * 60)
    print("RAG Embedding - Zone Width Session")
    print(f"Project: {PROJECT}")
    print(f"Session: zone_width_20260115")
    print("=" * 60)

    success = 0
    failed = 0

    # Embed files
    print("\n[Files]")
    for rel_path in SESSION_FILES:
        if embed_file(rel_path):
            success += 1
        else:
            failed += 1

    # Embed session summary
    print("\n[Session Summary]")
    if embed_session_summary():
        success += 1
    else:
        failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {success}/{success + failed} items embedded")
    print("=" * 60)


if __name__ == "__main__":
    main()
