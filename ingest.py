"""
ingest.py — Document loading and chunking pipeline
UST Unofficial Housing Guide

Run:  python ingest.py
Output: prints chunk count and 5 sample chunks for inspection
"""

import os
import re
import json


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = "data"          # folder containing .txt source files
CHUNK_SIZE = 400           # characters per chunk
OVERLAP = 80               # character overlap between consecutive chunks
OUTPUT_FILE = "chunks.json"  # save chunks here for embed.py to consume


# ── Step 1: Load documents ────────────────────────────────────────────────────

def load_documents(data_dir: str) -> list[dict]:
    """Load all .txt files from data_dir. Returns list of {source, text} dicts."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        documents.append({"source": filename, "text": raw_text})
        print(f"  Loaded: {filename} ({len(raw_text):,} chars)")
    return documents


# ── Step 2: Clean documents ───────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove boilerplate, normalize whitespace.
    Our source files are plain text so cleaning is light.
    """
    # Remove lines that are purely separator markers
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)

    # Collapse runs of blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from the whole document
    text = text.strip()

    return text


# ── Step 3: Chunk documents ───────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Sliding-window chunking over cleaned text.

    Strategy: advance by (chunk_size - overlap) each step so consecutive
    chunks share 'overlap' characters. This ensures facts near a chunk
    boundary appear in at least one chunk with full surrounding context.

    We try to break at sentence/paragraph boundaries within a small
    tolerance window so chunks don't cut mid-word or mid-sentence.
    """
    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to find a natural break point (newline or period) near end
            # Look back up to 60 chars for a paragraph/sentence boundary
            look_back = text[max(start, end - 60): end]
            para_break = look_back.rfind("\n\n")
            sent_break = look_back.rfind(". ")

            if para_break != -1:
                end = end - 60 + para_break + 2   # include the newlines
            elif sent_break != -1:
                end = end - 60 + sent_break + 2   # include ". "

        chunk = text[start:end].strip()
        if len(chunk) > 50:   # skip tiny fragments
            chunks.append(chunk)

        start += step

    return chunks


# ── Step 4: Build chunk records ───────────────────────────────────────────────

def build_chunks(documents: list[dict]) -> list[dict]:
    """
    For each document, clean text and split into chunks.
    Returns list of {id, source, chunk_index, text} dicts.
    """
    all_chunks = []
    for doc in documents:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned)
        for i, chunk_text_content in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['source']}::chunk_{i}",
                "source": doc["source"],
                "chunk_index": i,
                "text": chunk_text_content,
            })
    return all_chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UST Unofficial Housing Guide — Ingestion Pipeline")
    print("=" * 60)

    # Load
    print(f"\n[1/3] Loading documents from '{DATA_DIR}/'...")
    documents = load_documents(DATA_DIR)
    print(f"      Total documents: {len(documents)}")

    # Chunk
    print(f"\n[2/3] Chunking (size={CHUNK_SIZE}, overlap={OVERLAP})...")
    chunks = build_chunks(documents)
    print(f"      Total chunks: {len(chunks)}")

    if len(chunks) < 50:
        print("  ⚠  Warning: fewer than 50 chunks — consider reducing chunk size")
    if len(chunks) > 2000:
        print("  ⚠  Warning: more than 2000 chunks — consider increasing chunk size")

    # Save
    print(f"\n[3/3] Saving chunks to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"      Saved {len(chunks)} chunks.")

    # ── Inspection: print 5 sample chunks ────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS (inspect for quality)")
    print("=" * 60)

    import random
    random.seed(42)
    sample_indices = random.sample(range(len(chunks)), min(5, len(chunks)))

    for rank, idx in enumerate(sample_indices, 1):
        c = chunks[idx]
        print(f"\n── Sample {rank} ──────────────────────────────────────")
        print(f"   Source:      {c['source']}")
        print(f"   Chunk index: {c['chunk_index']}")
        print(f"   Length:      {len(c['text'])} chars")
        print(f"   Text:\n{c['text']}")

    print("\n" + "=" * 60)
    print("Ingestion complete. Run embed.py next.")


if __name__ == "__main__":
    main()
