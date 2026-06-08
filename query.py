"""
query.py — Grounded response generation using Groq + retrieved context
UST Unofficial Housing Guide

The key design constraint: the LLM is explicitly forbidden from using
knowledge outside the retrieved documents. Every response cites sources.
"""

import pip_system_certs.wrapt_requests  # noqa: F401 — use Windows cert store for HTTPS

import os
from groq import Groq, AuthenticationError, APIConnectionError
from dotenv import load_dotenv
from embed import retrieve

load_dotenv()

API_KEY_HELP = (
    "Invalid or missing Groq API key. "
    "Get a free key at https://console.groq.com, then set GROQ_API_KEY in your .env file."
)
CONNECTION_HELP = (
    "Could not reach the Groq API due to a network or SSL certificate error. "
    "On Windows, run: pip install pip-system-certs"
)

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
MAX_CONTEXT_CHARS = 3000   # cap total context sent to LLM to stay within token limits


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Unofficial Guide assistant for students at the University of St. Thomas (St. Paul, MN). You answer questions about off-campus housing using ONLY the student-written documents provided to you.

RULES — follow these without exception:
1. Answer using only information that appears in the provided documents. Do not use any other knowledge.
2. If the documents do not contain enough information to answer the question, say exactly: "I don't have enough information in the current documents to answer that question."
3. Never speculate, guess, or fill gaps with general knowledge.
4. Always cite which document(s) your answer draws from, using the document name (e.g., "According to reddit_ust_housing_thread_1.txt...").
5. Keep answers concise and useful — 2-4 short paragraphs at most.
6. If documents contain conflicting information, note the disagreement honestly.
"""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Format retrieved chunks as a numbered context block for the LLM.
    Truncates to max_chars to avoid overrunning token limits.
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        entry = f"[Document {i}: {chunk['source']}]\n{chunk['text']}"
        if total_chars + len(entry) > max_chars:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(context_parts)


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(question: str) -> dict:
    """
    Full RAG pipeline: retrieve → format context → generate grounded answer.

    Returns:
        {
            "answer":  str,           # LLM-generated answer (grounded)
            "sources": list[str],     # unique source filenames cited
            "chunks":  list[dict],    # raw retrieved chunks (for debugging)
        }
    """
    # 1. Retrieve relevant chunks
    chunks = retrieve(question, top_k=TOP_K)

    if not chunks:
        return {
            "answer": "I couldn't retrieve any relevant documents for that question.",
            "sources": [],
            "chunks": [],
        }

    # 2. Build context string
    context = build_context(chunks)

    # 3. Format user message
    user_message = f"""Here are the relevant student-written documents:

{context}

---

Question: {question}

Answer using only the documents above. Cite which document(s) your answer draws from."""

    # 4. Call Groq
    api_key = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
    if not api_key or api_key == "your_key_here":
        return {
            "answer": API_KEY_HELP,
            "sources": [],
            "chunks": chunks,
        }

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,      # low temperature = more faithful to context
            max_tokens=600,
        )
    except AuthenticationError:
        return {
            "answer": API_KEY_HELP,
            "sources": [],
            "chunks": chunks,
        }
    except APIConnectionError:
        return {
            "answer": CONNECTION_HELP,
            "sources": [],
            "chunks": chunks,
        }

    answer = response.choices[0].message.content.strip()

    # 5. Collect unique sources (programmatic attribution, not LLM-inferred)
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UST Unofficial Housing Guide — Query Test")
    print("=" * 60)

    test_questions = [
        "What do students say about parking near UST?",
        "Are there mold problems in apartments near campus?",
        "Is it safe to walk at night in Mac-Groveland?",
        "What is the best bus route for students living near UST?",
        "What should I look for when I tour an apartment?",   # in-scope
        "What is the population of St. Paul, Minnesota?",     # out-of-scope
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        result = ask(question)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for src in result["sources"]:
            print(f"  • {src}")
        print()
