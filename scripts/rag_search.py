# -*- coding: utf-8 -*-
"""
RAG Search - Trading Bot Context
"""
import os
import sys

HATTZ_EMPIRE_PATH = r"C:\Users\hahonggu\Desktop\coin_master\hattz_empire"
sys.path.insert(0, HATTZ_EMPIRE_PATH)

from src.services.rag import search, build_context

PROJECT = "trading_bot"

# 핵심 검색 쿼리들
QUERIES = [
    "Legacy Confluence 분석 파이프라인 구조",
    "HMM Entry Gate 통합 방법",
    "unified_signal.py의 핵심 로직",
    "cycle_anchor와 multi_tf_fib 연동",
    "TFPredictor 트레이딩 신호 생성",
    "Spectral 분석 FFT Wavelet 활용",
    "ZigZag 피벗 감지 최적화",
]

def main():
    print("=" * 70)
    print("RAG Search - Trading Bot Integration Context")
    print("=" * 70)

    all_contexts = []

    for query in QUERIES:
        print(f"\n>>> Query: {query}")
        print("-" * 50)

        try:
            result = search(query, project=PROJECT, top_k=3)

            if result.documents:
                for i, doc in enumerate(result.documents[:2]):
                    source_id = doc.get('source_id', 'unknown')[:50]
                    content_preview = doc.get('content', '')[:200].replace('\n', ' ')
                    score = doc.get('score', 0)
                    print(f"  [{i+1}] {source_id} (score: {score:.2f})")
                    print(f"      {content_preview}...")
                    all_contexts.append(doc.get('content', ''))
            else:
                print("  No results")

        except Exception as e:
            print(f"  Error: {e}")

    # 종합 요약
    print("\n" + "=" * 70)
    print("Unique Files Found:")
    print("=" * 70)

    # 파일 목록 추출
    files = set()
    for ctx in all_contexts:
        if 'def ' in ctx or 'class ' in ctx:
            # Python 파일로 추정
            lines = ctx.split('\n')[:5]
            for line in lines:
                if '"""' in line or "'''" in line:
                    print(f"  - {line[:60]}...")
                    break


if __name__ == "__main__":
    main()