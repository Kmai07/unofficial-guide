# The Unofficial Guide — UST Off-Campus Housing RAG System

A Retrieval-Augmented Generation (RAG) system that makes student-generated off-campus housing knowledge searchable and answerable. Ask plain-language questions and get grounded, cited answers drawn from real student reviews, Reddit threads, and forum posts about living near the University of St. Thomas (St. Paul, MN).

---

## Setup

```bash
# 1. Clone the repo and navigate into it
git clone <your-fork-url>
cd unofficial-guide

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
source .venv/Scripts/activate      # Windows (Git Bash)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env and replace your_key_here with your actual key

# 5. Build the pipeline (run in order)
python ingest.py    # load + chunk documents → chunks.json
python embed.py     # embed chunks → chroma_db/

# 6. Launch the app
python app.py       # opens at http://localhost:7860
```

---

## Domain

**Off-campus housing knowledge for students at the University of St. Thomas (St. Paul, MN).**

UST's official resources only describe on-campus housing and link to a generic apartment search tool. The knowledge students actually need — which landlords are responsive, which buildings have mold problems, which neighborhoods are safe to walk at night, how to get a security deposit back — lives entirely in student word-of-mouth, Reddit threads, and apartment review sites. There is no official channel for this knowledge, which makes it exactly the right target for a retrieval system.

---

## Document Sources

| File | Source type | Content |
|------|-------------|---------|
| `reddit_ust_housing_thread_1.txt` | Reddit r/TwinCities | Thread: best streets/neighborhoods near UST |
| `reddit_ust_housing_thread_2.txt` | Reddit r/StThomasUST | Thread: Summit Flats apartment reviews |
| `reddit_ust_housing_thread_3.txt` | Reddit r/TwinCities | Thread: parking near UST |
| `yelp_reviews_summit_flats.txt` | Yelp reviews | Summit Flats Apartments on Summit Ave |
| `yelp_reviews_grand_ave_apts.txt` | Yelp reviews | Grand Avenue Apartments reviews |
| `apartmentratings_cleveland_ave.txt` | ApartmentRatings.com | Cleveland Ave rental reviews |
| `student_blog_ust_housing_guide.txt` | Student blog | "The Real Guide to Living Off Campus at UST" |
| `landlord_experiences.txt` | Aggregated posts | Sievert Properties, Harbor Bay, individual landlords |
| `neighborhood_safety_posts.txt` | Reddit + neighborhood forums | Mac-Groveland and Highland Park safety |
| `lease_tips_posts.txt` | Reddit + student forums | Lease signing, deposits, Minnesota tenant rights |
| `commute_transit_posts.txt` | Reddit threads | Bus routes, biking, parking costs |
| `dining_neighborhood_tips.txt` | Yelp + student posts | Grocery, restaurants, neighborhood amenities |

**12 total documents** covering apartments, landlords, safety, transit, leases, and neighborhood life.

---

## Chunking Strategy

**Chunk size:** 400 characters  
**Overlap:** 80 characters  
**Method:** Sliding window with sentence/paragraph boundary snapping

All source documents consist of short-form opinion text (Reddit comments, Yelp reviews, blog paragraphs). A complete thought in this kind of text typically runs 100–300 characters. A 400-character chunk captures one full idea with a bit of surrounding context.

An 80-character overlap (20% of chunk size) handles boundary splits — if a key claim begins at character 380 of a chunk, it will appear fully in the next chunk's starting overlap.

Chunks are snapped to sentence or paragraph breaks within a 60-character tolerance window, so they avoid cutting mid-sentence when possible.

### Sample Chunks (5 examples)

**Chunk 1** — `reddit_ust_housing_thread_1.txt` (chunk 3)
```
Avoid anything on Cleveland Ave south of Ford Pkwy if you can. Some decent deals there but a few of the older buildings have had persistent maintenance issues — one of my friends dealt with a leaky ceiling for two months before the landlord bothered to fix it.
```

**Chunk 2** — `yelp_reviews_summit_flats.txt` (chunk 4)
```
Lived here for 14 months. The building itself is stunning — exposed brick, hardwood floors, really nice common areas. But maintenance requests go into a black hole. My oven burner stopped working and I waited 3 weeks for a repair.
```

**Chunk 3** — `apartmentratings_cleveland_ave.txt` (chunk 6)
```
The landlord at 2050 Cleveland is unreachable by phone and inconsistent by email. I had a mouse infestation — plural mice — starting in October. I reported it immediately with photos. First response came 10 days later.
```

**Chunk 4** — `lease_tips_posts.txt` (chunk 2)
```
Minnesota tenant rights basics every UST student should know: Security deposit must be returned within 21 days of lease end with a written itemization of any deductions. Landlords must maintain heating above 68°F when outdoor temps fall below 60°F.
```

**Chunk 5** — `commute_transit_posts.txt` (chunk 1)
```
Metro Transit routes near UST that actually matter: Route 21 (Grand Ave) runs the length of Grand Ave, very frequent on weekdays. Route 84 connects Highland Park to the UST area. The A Line (Snelling Ave rapid bus) is the fastest way to reach the Green Line light rail.
```

Each chunk is a complete, readable thought with a clear topic — no fragments, no HTML artifacts, no empty strings.

---

## Embedding Model

**Model:** `all-MiniLM-L6-v2` (via `sentence-transformers`)

This model runs entirely locally with no API key or rate limits. It produces 384-dimensional embeddings optimized for semantic similarity on short English text, which matches the short-form opinion text in this corpus. Context window is 256 tokens — sufficient for 400-character chunks.

**Production tradeoffs (if I were choosing for real deployment):**

- **OpenAI `text-embedding-3-large`**: Higher benchmark accuracy, 3072-dimensional output. Worth the API cost for a production system with tens of thousands of queries per day. Requires network dependency.
- **Voyage AI `voyage-2`**: Strong performance on domain-specific retrieval, reportedly better than OpenAI on niche corpora. Relevant if this system expanded to specialized academic content.
- **Multilingual models** (`paraphrase-multilingual-mpnet-base-v2`): Necessary if the user base includes non-English speakers or content in other languages. Not needed for this English-only corpus.
- **Local vs. API**: Local (`all-MiniLM-L6-v2`) is free, private, and fast but lower accuracy. API-based models are more accurate but add cost, latency, and a single point of failure. For a student project, local is the right call.

---

## Retrieval Test Results

**Top-k:** 5 chunks per query

### Query 1: "What do students say about parking near UST?"

| Rank | Source | Distance | Relevant? |
|------|--------|----------|-----------|
| 1 | `reddit_ust_housing_thread_3.txt` | 0.18 | ✅ Yes — full thread about parking options |
| 2 | `commute_transit_posts.txt` | 0.21 | ✅ Yes — lists parking costs, ramp prices |
| 3 | `student_blog_ust_housing_guide.txt` | 0.27 | ✅ Yes — section on parking per building |
| 4 | `reddit_ust_housing_thread_1.txt` | 0.34 | ✅ Partial — mentions parking as one of many topics |
| 5 | `yelp_reviews_summit_flats.txt` | 0.41 | ✅ Partial — mentions $75/month parking at Summit Flats |

**Why these chunks are relevant:** The query "parking near UST" has strong semantic overlap with chunks containing words like "parking permit," "ramp," "car," "park," and "spot." The model correctly prioritizes the dedicated parking thread and the student blog's parking section.

---

### Query 2: "Are there mold problems in apartments near campus?"

| Rank | Source | Distance | Relevant? |
|------|--------|----------|-----------|
| 1 | `yelp_reviews_summit_flats.txt` | 0.14 | ✅ Yes — explicit mold review (black mold, bathroom ceiling) |
| 2 | `apartmentratings_cleveland_ave.txt` | 0.19 | ✅ Yes — mold under sink, moisture staining |
| 3 | `student_blog_ust_housing_guide.txt` | 0.31 | ✅ Yes — tour checklist includes checking for mold |
| 4 | `reddit_ust_housing_thread_2.txt` | 0.38 | ✅ Yes — Summit Flats moisture staining in bathroom |
| 5 | `landlord_experiences.txt` | 0.52 | ⚠️ Weak — mentions maintenance issues but not specifically mold |

**Why these chunks are relevant:** "Mold problems in apartments" semantically matches chunks containing "mold," "moisture," "black mold," "mildew," "staining," and "habitability." Chunks 1–4 all directly describe mold incidents. Chunk 5 is weaker but returned because its maintenance-failure context is adjacent to the query's domain.

---

### Query 3: "What property managers are responsive to maintenance requests?"

| Rank | Source | Distance | Relevant? |
|------|--------|----------|-----------|
| 1 | `landlord_experiences.txt` | 0.12 | ✅ Yes — detailed Sievert vs. Harbor Bay comparison |
| 2 | `yelp_reviews_grand_ave_apts.txt` | 0.22 | ✅ Yes — praises specific management responsiveness |
| 3 | `student_blog_ust_housing_guide.txt` | 0.29 | ✅ Yes — section naming property managers |
| 4 | `reddit_ust_housing_thread_2.txt` | 0.33 | ✅ Yes — Summit Flats maintenance slow response |
| 5 | `apartmentratings_cleveland_ave.txt` | 0.44 | ✅ Partial — on-site landlord response vs. absentee |

---

## Grounded Generation

Grounding is enforced in two ways:

**1. System prompt (hard constraint):** The LLM receives an explicit system prompt that says:
> *"Answer using only information that appears in the provided documents. Do not use any other knowledge. If the documents do not contain enough information to answer, say: 'I don't have enough information in the current documents to answer that question.'"*

The word "only" and the fallback response are deliberate. Without a specific fallback instruction, models tend to generate plausible-sounding answers even when context is sparse.

**2. Low temperature (0.2):** This reduces the model's tendency to fill gaps creatively. Higher temperatures produce more varied, less faithful responses.

**3. Programmatic source attribution:** Source filenames are appended to every response from the `chunks` metadata — they are not inferred by the LLM, so attribution is always accurate even if the model's in-text citations are imprecise.

---

## Example Responses

### Example 1 — In-scope question with strong retrieval

**Q:** "What do students say about mold problems at Summit Flats?"

**A:**
> According to `yelp_reviews_summit_flats.txt`, one tenant experienced visible black mold growing in the shower grout and on the bathroom ceiling within three months of moving in. Maintenance came and painted over it, but the mold returned twice. The tenant eventually broke their lease citing habitability concerns. Another reviewer in `reddit_ust_housing_thread_2.txt` reported moisture staining in the bathroom ceiling and documented everything in writing for deposit protection purposes. A student blogger in `student_blog_ust_housing_guide.txt` specifically advises testing the bathroom exhaust fan on tours, noting that weak fans lead to moisture buildup and mold risk.

**Sources retrieved from:** `yelp_reviews_summit_flats.txt`, `reddit_ust_housing_thread_2.txt`, `student_blog_ust_housing_guide.txt`

---

### Example 2 — In-scope question, good retrieval

**Q:** "Is it safe to walk at night in Mac-Groveland?"

**A:**
> According to `neighborhood_safety_posts.txt`, Mac-Groveland is described by multiple students as one of the safer neighborhoods in St. Paul, with low rates of violent crime. The main corridors — Grand Ave and Summit Ave — are well-lit and have regular foot traffic in the evenings. One reviewer noted walking home from Grand Ave at midnight without incident. However, the quieter residential side streets (Fairmount, Goodrich, Portland) are darker and more isolated at night. Several students recommend using a buddy system or rideshare after midnight on those quieter blocks as a precaution, particularly for those walking alone.

**Sources retrieved from:** `neighborhood_safety_posts.txt`, `reddit_ust_housing_thread_1.txt`

---

### Example 3 — Out-of-scope question (refusal response)

**Q:** "What is the population of St. Paul, Minnesota?"

**A:**
> I don't have enough information in the current documents to answer that question. The documents in this system focus on student housing experiences near the University of St. Thomas — apartment reviews, landlord experiences, neighborhood safety, and transit. For demographic information about St. Paul, I'd suggest checking the U.S. Census Bureau or the City of St. Paul's official website.

*(Note: The system correctly declines. The documents contain no population data, and the model does not supply its training-data knowledge.)*

---

## Query Interface

The interface is a Gradio web app (`app.py`). Run `python app.py` and open `http://localhost:7860`.

**Input:** A text box labeled "Your question" where users type or paste a plain-language housing question. Pressing Enter or clicking "Ask" submits it.

**Outputs:**
- **Answer box** — the LLM-generated response, grounded in retrieved documents
- **Retrieved from** — a bulleted list of source document filenames the answer draws from
- **Retrieved chunks (debug view, collapsed)** — the raw top-5 chunks with cosine distance scores, for inspection

**Sample interaction:**

```
Input:  "How much should I expect to pay for a 1-bedroom near UST?"

Answer: According to student posts in reddit_ust_housing_thread_1.txt and
        student_blog_ust_housing_guide.txt, 1-bedroom apartments within
        walking distance of UST typically run $900–$1,200/month. On Summit
        Ave, expect $1,100–$1,300 for nicer converted buildings. Grand Ave
        corridor apartments run $950–$1,100. Cleveland Ave offers cheaper
        options ($775–$850/month) with trade-offs in building quality and
        maintenance. Note that heat is not always included — the student blog
        advises adding $120–$200/month for heat in winter if the unit is
        separately metered.

Retrieved from:
  • reddit_ust_housing_thread_1.txt
  • student_blog_ust_housing_guide.txt
  • apartmentratings_cleveland_ave.txt
  • yelp_reviews_summit_flats.txt
```

---

## Evaluation Report

### Test Question 1

**Q:** "What do students say about parking near UST?"  
**Expected:** Parking is scarce; costs $85–95/month at city ramps; some buildings include parking; on-street parking is contested especially in winter; biking is a popular alternative.  
**System response:** Correctly described the Snelling Ave ramp cost ($85–95/month), on-street competition, building-included parking as the best option, and included the transit alternative. Cited `reddit_ust_housing_thread_3.txt` and `commute_transit_posts.txt` accurately.  
**Accuracy:** ✅ Accurate

---

### Test Question 2

**Q:** "Are there mold problems in any of the apartments near campus?"  
**Expected:** Yes — Summit Flats and Cleveland Ave buildings have documented mold issues, especially in bathrooms and basement units.  
**System response:** Correctly identified Summit Flats (bathroom ceiling mold) and Cleveland Ave (under-sink mold) with specific details. Cited Yelp reviews and ApartmentRatings sources.  
**Accuracy:** ✅ Accurate

---

### Test Question 3

**Q:** "Which landlords or property managers near UST are responsive to maintenance requests?"  
**Expected:** Sievert Properties rated positively; Harbor Bay Management rated negatively; on-site landlord at 1920 Cleveland Ave rated highly; on-site Fairmount landlord praised.  
**System response:** Correctly named Sievert Properties as solid (3–7 days response), Harbor Bay as problematic (2-week heating outage example), and the on-site Cleveland Ave landlord as the most responsive. Cited `landlord_experiences.txt` accurately.  
**Accuracy:** ✅ Accurate

---

### Test Question 4

**Q:** "Is it safe to walk at night in the Mac-Groveland neighborhood?"  
**Expected:** Generally safe, especially on main streets; quieter side streets are darker; property crime exists but violent crime is low; several students walk at midnight without issues.  
**System response:** Accurately reflected the nuance — main corridors safe, side streets darker, recommended caution after midnight on quieter blocks. Mentioned bicycle theft as the more relevant safety concern.  
**Accuracy:** ✅ Accurate

---

### Test Question 5 — Failure case

**Q:** "Which specific apartment complexes on Grand Avenue accept pets?"  
**Expected:** Documents should mention pet policies for specific buildings.  
**System response:** "I don't have enough information in the current documents to answer that question. The documents mention that pet policies should be confirmed in writing before signing, but do not specify which buildings near Grand Avenue accept pets or on what terms."  
**Accuracy:** ❌ Inaccurate (refusal when the question is partially answerable)

**Failure analysis:** The documents mention pets only once, in `student_blog_ust_housing_guide.txt`: *"If you have or plan to get a pet, get this in writing before signing."* This chunk is about the general advice to confirm pet policies, not about which buildings allow them. The query "which complexes accept pets" contains no keywords that overlap with this chunk — "pets" appears once but the surrounding context is about lease-signing procedure, not building policies.

The system correctly declined to hallucinate an answer, which is the right behavior. But a more useful response would have said: "The documents don't list which specific buildings accept pets. According to the student housing guide, you should confirm pet policies with the landlord in writing before signing any lease." This is a **retrieval gap + response design failure**: the retrieved chunk contains marginally relevant information that the system correctly identified as insufficient, but the response could have cited and extended it rather than refusing entirely.

**What would fix it:** Either (a) add documents that explicitly discuss pet policies for named buildings, or (b) tune the system prompt to encourage partial answers: "If the documents contain partial information, share what you found and note the gap."

---

## Spec Reflection

**How the spec helped:** Writing the chunking strategy section before touching any code forced a concrete decision about chunk size. I initially planned 600-character chunks, but thinking through the document structure — mostly short Reddit comments and Yelp reviews — made it obvious that 600 characters would frequently span 3–4 different opinions in a thread, diluting retrieval signal. The spec pushed me to 400, which I verified by printing sample chunks before moving to embedding.

**Where implementation diverged:** The spec's evaluation plan listed "which landlords are responsive" as one of the five test questions and expected named specific managers as the answer. When I actually ran the pipeline, the system correctly named Sievert and Harbor Bay — but it also retrieved a Fairmount Ave landlord not mentioned in the evaluation plan's expected answer. The retrieved documents contained more relevant named entities than the spec anticipated. This was a positive divergence, but it meant the expected answer needed to be updated to match what the documents actually contained. Lesson: write evaluation expected answers after reviewing the documents, not before.

---

## AI Usage

**Instance 1: Chunking function with boundary snapping**

I prompted Claude with my Chunking Strategy section from planning.md and asked it to implement a sliding window chunker that attempts to break at sentence or paragraph boundaries within a tolerance window. It produced a version that checked for `\n\n` and `. ` near the chunk end, which matched my intent. I revised one part: the original returned an error if no boundary was found, but I changed it to fall back silently to the hard character boundary instead, since review text sometimes runs long without sentence breaks.

**Instance 2: System prompt for grounded generation**

I asked Claude to write a system prompt that enforces context-only answers. It produced a good starting point but the fallback instruction was phrased as "try to only use the provided documents." I hardened it to "Answer using only information that appears in the provided documents. Do not use any other knowledge." — the word "only" twice, plus a specific fallback response to say when documents don't cover the question. Testing showed this substantially reduced hallucination compared to the softer prompt.

---

*Built for the "Show What You Know" project — UST Unofficial Housing Guide*
