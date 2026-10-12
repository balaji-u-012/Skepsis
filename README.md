# Skepsis — Multi-Agent AI for Research Discovery

Skepsis takes a research topic, searches **arXiv, Semantic Scholar, OpenAlex
and Crossref**, deduplicates and ranks the results, picks the top 5 papers,
and produces a structured per-paper analysis plus a side-by-side comparison
and research-gap summary.

It is built to run on a restricted, CPU-only office laptop:

- **Backend:** Python, `http.server` (stdlib) + `asyncio` for the web layer, httpx, BeautifulSoup, feedparser, `dataclasses` (stdlib) instead of Pydantic, SQLite (stdlib `sqlite3`)
- **Frontend:** vanilla HTML/CSS/JavaScript, no build step, served with `python -m http.server`
- **No** Node/npm, React, Docker, GPU/CUDA, PyTorch/TensorFlow, local LLMs, or model training anywhere in the pipeline

### Why FastAPI/Uvicorn/Pydantic aren't used

The original build used FastAPI + Pydantic, but Pydantic v2 ships a compiled
Rust extension (`pydantic-core`) with no prebuilt wheel for every Python
version/platform combination — on a locked-down laptop with no C/Rust build
tools, `pip install` for it fails outright. `httpx`, `beautifulsoup4`,
`feedparser` and `python-dotenv` are pure Python and always install cleanly,
so the web layer and data models were rewritten on things that ship with
Python itself: `http.server` for routing/serving and `dataclasses` for the
data models. Same API, same pipeline, zero compiled dependencies.

## Why there's no "pretrained model" in the ranking step

A genuine pretrained neural embedding model (e.g. a sentence-transformer)
needs PyTorch or TensorFlow to run, which this project's constraints
explicitly exclude. So core **discovery, deduplication and ranking run on a
small hand-written TF-IDF + cosine-similarity scorer in plain Python**
(`backend/utils/text_utils.py`) blended with a recency score — no model
file, no training step, no GPU, deterministic and fast on a laptop CPU.

If an `ANTHROPIC_API_KEY` is set in `.env`, the **Paper Analysis, Comparison
and Research Gap agents** additionally call Claude over `httpx` to write
sharper natural-language summaries — but the whole pipeline, including
those three agents, has a non-LLM heuristic fallback, so it works with zero
keys and zero external model calls if you prefer.

## Multi-agent pipeline

```
User Query
  → Query Agent               (backend/agents/query_agent.py)
  → Research Discovery Agents (backend/agents/discovery_agent.py, backend/services/*_service.py)
  → Verification/Dedup Agent  (backend/agents/verification_agent.py)
  → Ranking Agent              (backend/agents/ranking_agent.py)
  → Top 5 Papers
  → Paper Analysis Agent      (backend/agents/analysis_agent.py)
  → Comparison Agent          (backend/agents/comparison_agent.py)
  → Research Gap Agent        (backend/agents/gap_agent.py)
  → Final Report (JSON, persisted to SQLite)
```

Every field Skepsis can't find in the source abstract/metadata is shown as
**"Not reported"** — nothing is invented.

## Project structure

```
skepsis/
├── backend/
│   ├── main.py              stdlib HTTP app, wires the pipeline, defines the API
│   ├── agents/               the 6 agents listed above
│   ├── services/              arxiv / semantic scholar / openalex / crossref
│   │                          clients, the optional LLM client, SQLite layer
│   ├── models/schemas.py     dataclass request/response/domain models
│   └── utils/                config loader, tiny http.server router, TF-IDF/similarity helpers
├── frontend-static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/                     skepsis.db (created automatically on first run)
├── requirements.txt
├── .env.example
└── README.md
```

## API

| Method | Path                       | Purpose                                             |
|--------|-----------------------------|------------------------------------------------------|
| POST   | `/api/search`                | Run the full pipeline for a topic, return top 5 + comparison + gaps |
| GET    | `/api/search/{search_id}`    | Re-fetch a previous search's saved result           |
| GET    | `/api/papers/{paper_id}`     | Fetch one paper's stored analysis                    |
| POST   | `/api/compare`               | Re-run the comparison/gap agents over a chosen set of stored paper ids |
| GET    | `/api/health`                | Backend status + whether LLM enrichment is active    |

There are no auto-generated `/docs` (that's a FastAPI feature) — use the
curl examples further down, or a tool like Postman, to try the API directly.

## Setup (Windows PowerShell)

From the `skepsis/` folder:

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies (all pure-Python — no compiler needed)
pip install -r requirements.txt

# 3. (Optional) configure an LLM key for richer summaries
Copy-Item .env.example backend\.env
notepad backend\.env   # paste ANTHROPIC_API_KEY if you have one, then save

# 4. Run the backend (from the skepsis/ root, so `backend.main` resolves)
python -m backend.main
```

There's no `--reload` flag (that's a Uvicorn feature) — after editing
backend code, stop the server with `Ctrl+C` and re-run `python -m
backend.main`.

Leave that terminal running, then in a **second** PowerShell window:

```powershell
cd skepsis\frontend-static
python -m http.server 5500
```

Open **http://localhost:5500** in your browser. The page checks
`http://127.0.0.1:8000/api/health` automatically; if the dot next to
"Skepsis" turns green, the backend is reachable.

> If your office network blocks outbound HTTPS entirely, the discovery
> agents (which call arXiv/Semantic Scholar/OpenAlex/Crossref over the
> internet) won't return results — that's a network policy issue, not a
> code issue. `/api/health` will still report `ok` even then, since it
> doesn't call out to any external source itself.

## Smoke-testing the backend alone

```powershell
# From a third terminal, with the backend running:
curl http://127.0.0.1:8000/api/health

curl -X POST http://127.0.0.1:8000/api/search `
  -H "Content-Type: application/json" `
  -d '{"topic": "retrieval augmented generation", "max_results": 5}'
```

## Notes on the external APIs used

- **arXiv** — public Atom feed API, no key needed.
- **Semantic Scholar** — public Graph API, no key needed (unauthenticated
  requests are rate-limited; the discovery agent logs and skips gracefully
  on a 429 rather than failing the whole search).
- **OpenAlex** — public Works API, no key needed. Set `CONTACT_EMAIL` in
  `.env` to use OpenAlex's faster "polite pool".
- **Crossref** — public Works API, no key needed. Same `CONTACT_EMAIL`
  courtesy applies.

## Extending it later

- `backend/agents/` and `backend/services/` are separated by responsibility
  on purpose — adding a fifth source means one new file in `services/` and
  one new line in `discovery_agent.py`.
- The ranking weights (`RELEVANCE_WEIGHT` / `RECENCY_WEIGHT` in
  `ranking_agent.py`) and the dedup thresholds
  (`verification_agent.py`) are simple module-level constants, tuned to be
  reasonable defaults rather than fitted — adjust them directly.
- `data/skepsis.db` is a normal SQLite file; inspect it with `sqlite3
  data\skepsis.db` or any SQLite GUI if you want to see stored searches.
