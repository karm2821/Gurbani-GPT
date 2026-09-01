#!/usr/bin/env python3
"""
Phase 3 — Embed Chunks & Store in ChromaDB
Uses Ollama's nomic-embed-text model to embed each Shabad chunk,
then stores them in a persistent ChromaDB vector database on D: drive.
Run: python scripts/03_embed_store.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import sys
import time
import requests
import chromadb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
CHROMA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
CHUNKS_FILE   = os.path.join(PROCESSED_DIR, 'gurbani_chunks.json')

OLLAMA_HOST   = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"     # fast, good quality — pull with: ollama pull nomic-embed-text
COLLECTION    = "gurbani"
BATCH_SIZE    = 50                     # embed N chunks at a time

# ── Helper ─────────────────────────────────────────────────────────────────────
def embed_text(text: str) -> list:
    """Get embedding vector from Ollama."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_embed_model():
    """Check if the embedding model is available."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(EMBED_MODEL in m for m in models)
    except Exception:
        return False

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 55)
    print("   🧮  Embedding Gurbani into ChromaDB...")
    print("=" * 55)
    print()

    # Pre-flight checks
    if not check_ollama():
        print("  ❌ Ollama is not running! Start it with: ollama serve")
        sys.exit(1)

    if not check_embed_model():
        print(f"  ❌ Embedding model '{EMBED_MODEL}' not found.")
        print(f"  ➡️  Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    if not os.path.exists(CHUNKS_FILE):
        print("  ❌ Chunks file not found! Run 02_clean_chunk.py first.")
        sys.exit(1)

    print(f"  ✅ Ollama running | Model: {EMBED_MODEL}")

    # Load chunks
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"  📦 Loaded {len(chunks)} Shabad chunks")

    # Connect to ChromaDB (persistent on D: drive)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Find already-embedded IDs to allow resuming
    existing = set(collection.get()['ids'])
    print(f"  🗄️  ChromaDB: {len(existing)} chunks already indexed")

    to_embed = [c for c in chunks if str(c['id']) not in existing]
    print(f"  🔄 Chunks to embed: {len(to_embed)}")
    print()

    if not to_embed:
        print("  ✅ All chunks already embedded! Nothing to do.")
        print(f"  📊 Total in DB: {collection.count()}")
        return

    # Embed in batches
    total    = len(to_embed)
    embedded = 0
    start    = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = to_embed[i : i + BATCH_SIZE]

        ids        = []
        documents  = []
        embeddings = []
        metadatas  = []

        for chunk in batch:
            try:
                vec = embed_text(chunk['text'])
                ids.append(str(chunk['id']))
                documents.append(chunk['text'])
                embeddings.append(vec)
                metadatas.append({
                    'ang':    chunk['ang'],
                    'raag':   chunk.get('raag_english', ''),
                    'author': chunk.get('writer_english', ''),
                    'lines':  chunk.get('line_count', 0),
                })
            except Exception as e:
                print(f"    ⚠️  Skipping chunk {chunk['id']}: {e}")

        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        embedded += len(batch)
        elapsed  = time.time() - start
        rate     = embedded / elapsed if elapsed > 0 else 1
        eta      = (total - embedded) / rate

        pct = (embedded / total) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.1f}% | {embedded}/{total} | ETA: {eta:.0f}s", end="", flush=True)

    print(f"\n\n  ✅ Done! Total chunks in ChromaDB: {collection.count()}")
    print(f"  📁 Saved to: {os.path.abspath(CHROMA_DIR)}")
    print()

if __name__ == '__main__':
    main()
