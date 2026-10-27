# 🎬 IMDB Agentic-RAG Voice Agent (Streamlit · ChromaDB · Redis · Guardrails · free LLMs)

Ask questions about the IMDB Top-1000 movies by **voice or text**. An LLM agent decides which retrieval tools to
call, verifies the results, and answers with reasoning, suggestions and similar-movie recommendations.

## Models & components (all free / open source)
| Purpose | Component |
|---|---|
| LLM (agent controller + answers) | **Gemini `gemini-3.8-flash`** (free tier) *or* **Groq `llama-3.3-70b-versatile`** (free tier) |
| Speech-to-text | Gemini audio understanding *or* Groq `whisper-large-v3-turbo` |
| Text-to-speech | `gTTS` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) — TF-IDF/LSA fallback |
| **Vector DB** | **ChromaDB** (embedded, persisted in `data/chroma_db/`, cosine HNSW) — numpy fallback if not installed |
| **Cache** | **Redis** semantic Q&A cache — in-memory fallback if Redis is unreachable |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
docker compose up -d redis                                # optional but recommended (or any Redis on :6379)
cp .env.example .env      # add a free GEMINI_API_KEY (aistudio.google.com/apikey) or GROQ_API_KEY (console.groq.com/keys)
streamlit run app.py
```
First launch downloads the ~90 MB embedding model and indexes all 1000 movies into ChromaDB (`data/chroma_db/`,
this folder is the shareable local vector store). The sidebar → *Infrastructure* shows what is active.

Offline tests (no key/network/Redis needed): `python tests/test_guardrails.py`, `python tests/test_cache_and_store.py`,
`python tests/test_agent_mock.py`, `python tests/test_engine.py`, and `python tests/test_llm.py`.

## Architecture
```
 voice/text ─► INPUT GUARDRAILS ─► Redis cache lookup ──hit──► answer (0 LLM calls)
                 (injection, PII,        │ miss
                  self-harm, abuse,      ▼
                  length, rate)     AGENT LOOP (max 5 steps, LLM picks 1 tool per step)
                                      ├─ search_movies  → arg sanitiser → pandas filters + ChromaDB similarity
                                      ├─ lookup_movie   → multi-hop first hop
                                      ├─ similar_movies → ChromaDB nearest plots (+ IMDB / Meta scores)
                                      ├─ ask_user       → clarification (e.g. Al Pacino lead vs any role)
                                      ├─ refuse / chat  → off-topic / greetings
                                      └─ final_answer ─► GROUNDING GUARDRAIL ─► answer + table + reasoning
                                                          (retry once, else safe fallback)   + suggestions
                                                                         └─► store in Redis
```
**Why it is "agentic":** the LLM inspects each observation and can *refine* (empty result → relax a filter or rewrite
the semantic query with synonyms), do *multi-hop* lookups, grade plot relevance from overviews (only truly matching
titles are kept), ask the user a clarifying question, or refuse. Every step is shown in the **🤖 Agent trace**.

## Guardrails (`imdb_agent/guardrails.py`)
| Layer | What it does |
|---|---|
| Input | empty / >500 chars, **prompt-injection** patterns, harmful how-to, abusive language, **self-harm → supportive reply**, **PII redaction** (email, phone, SSN, Luhn-valid cards) before anything reaches the LLM, cache or chat history |
| Scope | agent has a `refuse` tool; system prompt restricts it to the movie dataset; dataset text is declared *data, not instructions* (indirect-injection defence) |
| Tool args | LLM output is whitelisted, type-coerced and clamped (genres, sort keys, year/rating ranges, limit ≤ 30, string lengths) — no arbitrary execution |
| Output | **grounding check**: answer may only name movies actually retrieved this turn; on violation the agent must rewrite, then falls back to a deterministic answer built from the rows. API keys are redacted and system-prompt leakage is blocked |
| Abuse | per-session rate limiter (10 questions / minute) |

## Redis cache (`imdb_agent/cache.py`)
* Stores final answers (+ tables) with TTL (default 24 h); **exact** match on normalised text, then **semantic**
  match (cosine ≥ 0.92 using the same embedder).
* A semantic hit is accepted only if the questions' *signatures* match (numbers, genres, direction words like
  before/above/lower, proper nouns) — so "top 5 comedies" never answers "top 7 horror".
* Only successful, standalone answers are cached (not clarifications, refusals or errors). Follow-up turns are
  cached under the agent's self-contained `standalone_question`.
* A Redis sorted set counts hits → the sidebar shows **🔥 Popular questions**.

## Data cleaning
`Gross` "28,341,469" → number; `Runtime` → minutes; `Genre` → list; Apollo 13's bad year ("PG") → 1995; missing
Meta_score/Gross stay NaN.

## Layout
```
app.py                    Streamlit UI (chat, mic, TTS, trace, badges, popular questions)
imdb_agent/agent.py       agentic RAG loop + tools
imdb_agent/guardrails.py  input / args / output guardrails, rate limiter
imdb_agent/cache.py       Redis semantic Q&A cache (+ in-memory fallback)
imdb_agent/vectorstore.py ChromaDB vector store (+ fallbacks)
imdb_agent/engine.py      structured + semantic query executor, recommender
imdb_agent/llm.py         Gemini / Groq wrapper (+ speech-to-text)
docker-compose.yml        Redis
tests/                    offline tests (fake Chroma/Redis APIs, scripted LLM)
```

## Notes
* The agent uses 2–4 LLM calls per uncached question; free tiers are rate-limited (auto-retry; switch provider in the
  sidebar). Cached questions use **zero** calls.
* Guardrails are pattern-based and deliberately conservative; extend the lists in `guardrails.py` for your policy.
* No API key is bundled — create a free one from the links above.
