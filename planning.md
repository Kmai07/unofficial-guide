# planning.md — The Unofficial Guide: UST Off-Campus Housing

## Domain

**Domain:** Off-campus housing experiences for students at the University of St. Thomas (St. Paul, MN campus).

This knowledge is valuable because UST's official housing resources only describe on-campus options and link to a generic off-campus search tool. What students actually need — which landlords are responsive, which buildings have mold or noise issues, which neighborhoods feel safe for walking at night, how to negotiate a lease near campus — exists only in student word-of-mouth, Reddit threads, and apartment review sites. The gap between what the university publishes and what students tell each other is exactly what this guide addresses.

## Documents

Collected 12 source documents representing different facets of off-campus housing knowledge near UST St. Paul:

| # | File | Source type | Content |
|---|------|-------------|---------|
| 1 | `data/reddit_ust_housing_thread_1.txt` | Reddit r/TwinCities / r/StThomasUST | Thread: "Best streets to live on near UST?" |
| 2 | `data/reddit_ust_housing_thread_2.txt` | Reddit | Thread: "Anyone lived at [Summit Ave apartments]? Honest review?" |
| 3 | `data/reddit_ust_housing_thread_3.txt` | Reddit | Thread: "Parking nightmare near UST — what do people do?" |
| 4 | `data/yelp_reviews_summit_flats.txt` | Yelp apartment reviews | Summit Flats Apartments on Summit Ave |
| 5 | `data/yelp_reviews_grand_ave_apts.txt` | Yelp apartment reviews | Grand Avenue apartment complex reviews |
| 6 | `data/apartmentratings_cleveland_ave.txt` | ApartmentRatings.com | Cleveland Ave rental reviews |
| 7 | `data/student_blog_ust_housing_guide.txt` | Student blog / orientation guide | "Unofficial guide to living off campus at UST" |
| 8 | `data/landlord_experiences.txt` | Aggregated student posts | Specific landlord / property manager reviews |
| 9 | `data/neighborhood_safety_posts.txt` | Reddit + neighborhood forum posts | Mac-Groveland and Highland Park safety notes |
| 10 | `data/lease_tips_posts.txt` | Reddit threads + student forums | Advice on signing leases, deposits, utilities |
| 11 | `data/commute_transit_posts.txt` | Reddit threads | Bus routes, biking, parking costs near UST |
| 12 | `data/dining_neighborhood_tips.txt` | Yelp + student posts | Where to eat and shop near off-campus housing |

## Chunking Strategy

**Chunk size:** 400 characters  
**Overlap:** 80 characters  
**Method:** Sliding window over cleaned plain text

**Rationale:**

The documents are almost entirely short-form opinion text — Reddit comments, Yelp reviews, and blog paragraphs. A single strong opinion or fact is usually expressed in 1–3 sentences, which in English prose runs 100–300 characters. A 400-character chunk captures one complete thought plus a bit of surrounding context, which is enough for the embedding model to understand the topic.

Overlap of 80 characters (~20% of chunk size) handles the case where a key claim is split across a chunk boundary — for example, a sentence starting "The landlord never responds to maintenance requests" might start at character 390 of a chunk. With 80-character overlap, that sentence appears at the beginning of the next chunk and is therefore fully retrievable.

A larger chunk size (e.g. 1000+ characters) would dilute signal — a retrieval query about "mold problems" would pull in a chunk that mentions mold once but is mostly about parking, weakening the similarity score. Smaller chunks (< 200 characters) risk cutting sentences in half, producing embeddings that represent fragments rather than complete ideas.

**Chunk inspection plan:** Print 5 random chunks after splitting and verify each one is a complete, readable thought with a clear topic.

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`

- Runs locally — no API key, no rate limits, no cost
- 384-dimensional embeddings; fast inference on CPU
- Well-suited for short, opinionated English text
- Context window: 256 tokens (sufficient for 400-char chunks)

**Top-k:** 5 chunks per query

5 gives the LLM enough context to synthesize an answer without flooding the prompt with loosely related material. For questions about a specific apartment complex, 3–4 chunks from that complex plus 1 adjacent chunk is ideal. If set to 2, there's real risk the relevant chunk wasn't in the top-2; if set to 10, the LLM starts getting confused by contradictory or off-topic chunks.

**Production model tradeoffs (if cost wasn't a constraint):**
- `text-embedding-3-large` (OpenAI): Higher accuracy, 3072 dims, multilingual — worth it for multi-language student populations
- `voyage-2` (Voyage AI): Better retrieval benchmark scores for domain-specific text
- Tradeoffs: API cost per token, latency, dependency on external service vs. local inference

## Evaluation Plan

Five test questions with expected correct answers drawn from the documents:

| # | Question | Expected answer |
|---|----------|----------------|
| 1 | "What do students say about parking near UST?" | Parking is scarce and expensive near campus; many students recommend permits at city ramps or biking; some streets have 2-hour limits that are strictly enforced |
| 2 | "Are there mold problems in any of the apartments near campus?" | Yes — specific reviews of Cleveland Ave buildings and Summit Flats mention moisture/mold issues, particularly in basement units |
| 3 | "Which landlords near UST are responsive to maintenance requests?" | Students praise [specific property manager name] as quick to respond; others warn against [specific bad landlord] who ignores requests |
| 4 | "Is it safe to walk at night in the Mac-Groveland neighborhood?" | Generally considered safe; students note the area near Grand Ave is well-lit and active; caution advised on quieter residential streets after midnight |
| 5 | "How much should I expect to pay for a 1-bedroom near UST?" | Students report $900–$1,200/month for a 1-bedroom within walking distance; cheaper options ($700–$900) exist on Cleveland Ave but with trade-offs in building quality |

## Anticipated Challenges

1. **Chunk boundary splitting key facts:** A review that says "the heat only works in one room — we froze all winter" might get split so "the heat only works in one room" is the end of chunk N and "we froze all winter" is the start of chunk N+1. Either chunk alone weakens the retrieval signal. Overlap mitigates this but doesn't fully solve it.

2. **Landlord name inconsistency:** Students refer to the same property management company by different names (full name, abbreviation, nickname). Semantic search will partially bridge this, but a query for "Acme Properties" might miss reviews that say "Acme Prop" or "the Acme people." This is a metadata filtering problem that the stretch feature (hybrid BM25 + semantic) would address.

3. **Recency of information:** Student posts from 3–4 years ago may describe conditions that have changed (new management, renovations). The system has no way to signal when a review is outdated — the user should be warned in the interface.

## AI Tool Plan

| Pipeline component | What I'll give the AI | What I expect it to produce |
|--------------------|----------------------|----------------------------|
| `ingest.py` (loading + cleaning) | Documents section + Chunking Strategy section + sample raw doc | A script that loads .txt files, strips excess whitespace, and splits using sliding window |
| `embed.py` (embedding + ChromaDB) | Retrieval Approach section + pipeline diagram | Code to embed chunks with sentence-transformers and upsert to ChromaDB with source metadata |
| `query.py` (generation) | Grounding requirement + output format spec | Groq API call with system prompt that enforces context-only answers; source attribution appended |
| `app.py` (Gradio UI) | Interface description + query.py function signature | Gradio Blocks UI with input, button, answer output, and sources output |

I will review every generated function for: correct ChromaDB API usage, actual grounding enforcement in the system prompt (not just suggested), and that source metadata flows correctly from ingest → embed → query → display.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OFFLINE (run once)                              │
│                                                                     │
│  data/*.txt         ingest.py              embed.py                 │
│  ┌──────────┐     ┌────────────────┐     ┌─────────────────────┐   │
│  │ Raw docs │────▶│ Load + Clean   │────▶│ Embed chunks        │   │
│  │ (12 .txt │     │ Split into     │     │ (all-MiniLM-L6-v2)  │   │
│  │  files)  │     │ 400-char chunks│     │ Store in ChromaDB   │   │
│  └──────────┘     │ with 80-char   │     │ with source metadata│   │
│                   │ overlap        │     └─────────────────────┘   │
│                   └────────────────┘               │               │
└───────────────────────────────────────────────────┼───────────────┘
                                                    │
                                              chroma_db/
                                                    │
┌───────────────────────────────────────────────────┼───────────────┐
│                     ONLINE (per query)            │               │
│                                                   ▼               │
│  app.py (Gradio)      query.py               embed.py             │
│  ┌──────────────┐    ┌──────────────────┐   ┌──────────────────┐  │
│  │ User enters  │───▶│ Embed query      │──▶│ ChromaDB         │  │
│  │ question     │    │ Retrieve top-5   │◀──│ semantic search  │  │
│  │              │    │ chunks + sources │   └──────────────────┘  │
│  │ Display:     │    │                  │                         │
│  │ - Answer     │◀───│ Groq LLM         │                         │
│  │ - Sources    │    │ (llama-3.3-70b)  │                         │
│  └──────────────┘    │ grounded prompt  │                         │
│                      └──────────────────┘                         │
└───────────────────────────────────────────────────────────────────┘
```
