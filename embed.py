"""
embed.py — Embedding and ChromaDB vector store
UST Unofficial Housing Guide

Run once to build the index:  python embed.py
Then use retrieve() from query.py.
"""

import pip_system_certs.wrapt_requests  # noqa: F401 — use Windows cert store for HTTPS

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

CHUNKS_FILE = "chunks.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ust_housing"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5                    # number of chunks to retrieve per query
BATCH_SIZE = 64              # embed this many chunks at once to avoid OOM


# ── Load model and DB ─────────────────────────────────────────────────────────

def get_model() -> SentenceTransformer:
    """Load the embedding model (downloads on first run, cached after)."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def get_collection(reset: bool = False) -> chromadb.Collection:
    """
    Return a ChromaDB collection.
    If reset=True, drop and recreate (use when re-indexing from scratch).
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


# ── Build index ───────────────────────────────────────────────────────────────

def build_index(reset: bool = True) -> None:
    """
    Load chunks.json, embed all chunks, and upsert into ChromaDB.
    Call this once after running ingest.py.
    """
    # Load chunks
    print(f"Loading chunks from '{CHUNKS_FILE}'...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded.")

    # Load model
    model = get_model()

    # Get or create collection
    collection = get_collection(reset=reset)

    # Embed and upsert in batches
    print(f"Embedding and indexing in batches of {BATCH_SIZE}...")
    total = len(chunks)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        progress = min(batch_start + BATCH_SIZE, total)
        print(f"  Indexed {progress}/{total} chunks...")

    print(f"\nIndexing complete. Collection '{COLLECTION_NAME}' has {collection.count()} entries.")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed a query and return the top-k most similar chunks.

    Returns a list of dicts, each containing:
        text     — the chunk text
        source   — filename of the source document
        distance — cosine distance (lower = more similar; 0 = identical)
    """
    model = get_model()
    collection = get_collection(reset=False)

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),
        })

    return retrieved


# ── Retrieval test ────────────────────────────────────────────────────────────

def test_retrieval() -> None:
    """Run 3 sample queries and print results for inspection."""
    test_queries = [
        "What do students say about parking near UST?",
        "Are there mold problems in any apartments near campus?",
        "Which property management companies are responsive to maintenance?",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST — inspect chunk relevance and distance scores")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)
        results = retrieve(query)
        for rank, r in enumerate(results, 1):
            score_flag = "✓" if r["distance"] < 0.5 else "⚠"
            print(f"  [{rank}] {score_flag} distance={r['distance']} | {r['source']}")
            # Print first 200 chars of chunk
            snippet = r["text"][:200].replace("\n", " ")
            print(f"       {snippet}...")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UST Unofficial Housing Guide — Embedding Pipeline")
    print("=" * 60)

    build_index(reset=True)
    test_retrieval()

    print("\nDone. Run app.py to start the query interface.")
