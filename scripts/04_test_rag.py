#!/usr/bin/env python3
"""
Phase 4 — Test the RAG Pipeline
Ask a question and see the Gurbani answer before integrating into the web server.
Run: python scripts/04_test_rag.py
"""

import sys
import os

# Add parent dir so we can import rag_engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rag_engine import GurbaniRAG

CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')

TEST_QUESTIONS = [
    "What does Gurbani say about inner peace?",
    "How should we remember God according to Gurbani?",
    "What is the importance of Naam (God's name)?",
    "What does Gurbani say about life and death?",
    "How to overcome ego according to Gurbani?",
]

def main():
    print()
    print("=" * 60)
    print("   🙏  Gurbani GPT — RAG Test Console")
    print("=" * 60)
    print()

    rag = GurbaniRAG(chroma_dir=CHROMA_DIR)

    if not rag.ready():
        print("  ❌ ChromaDB is empty. Run 03_embed_store.py first.")
        return

    print(f"  ✅ Gurbani DB ready — {rag.count()} Shabads indexed")
    print()

    # Run preset tests
    print("  📋 Running preset test questions...\n")
    for q in TEST_QUESTIONS:
        print(f"  ❓ Q: {q}")
        results = rag.retrieve(q, n=3)
        print(f"  📖 Top match: Ang {results[0]['ang']} | {results[0]['raag']} | {results[0]['author']}")
        print(f"  📄 Preview: {results[0]['text'][:120]}...")
        print()

    # Interactive mode
    print("  " + "─" * 56)
    print("  💬 Interactive mode — type your question (or 'exit')")
    print("  " + "─" * 56)
    print()

    while True:
        try:
            q = input("  ❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  👋 Exiting test.\n")
            break

        if not q or q.lower() in ('exit', 'quit', 'bye'):
            print("\n  👋 Exiting test.\n")
            break

        results = rag.retrieve(q, n=5)
        answer  = rag.answer(q, results)

        print()
        print("  " + "─" * 56)
        print("  🙏 Gurbani Answer:")
        print("  " + "─" * 56)
        print(answer)
        print("  " + "─" * 56)
        print()

if __name__ == '__main__':
    main()
