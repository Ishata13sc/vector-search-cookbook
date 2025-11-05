# Vector Search Cookbook

An educational, end-to-end **vector search** demo that covers index variants (naive / IVF / graph), **BM25 lexical search**, **hybrid fusion**, **re-ranking**, lightweight **answering with citations**, dataset upload & schema mapping, evaluation scripts, and a clean responsive web UI.

---

## Highlights

- **Index Variants**
  - **Naive**: exact, brute-force cosine similarity (for small/medium corpora).
  - **IVF**: inverted file index with tunable `nlist` / `nprobe`.
  - **Graph**: HNSW-like graph with tunable degree `M` (demo implementation).
- **Lexical + Hybrid**
  - BM25 (via `rank-bm25`) and vector search.
  - Hybrid fusion (`alpha`) via weighted sum with normalization.
- **Reranker**
  - Take a candidate pool (top-k from vector/lexical/hybrid), then re-rank for better precision@k.
- **Answering Panel**
  - Concise answer with **citations** (document id + snippet).
- **Dataset Panel**
  - Upload `.csv` / `.jsonl`, auto-inspect columns, preview, map `id`/`text`, optional lowercase & deduplicate, import.
- **Index I/O**
  - Save / load / list indexes on disk.
- **Admin Panel**
  - Switch **embedding model** safely (with a toggle to re-embed), cache info & cache clear, optional API key header.
- **Export**
  - Export last run (e.g., hybrid or answer) to a Markdown file.
- **UI/UX**
  - Pure HTML/CSS/JS, dark theme, responsive layout (desktop → mobile).

---

## Architecture

- **Backend**: FastAPI + Uvicorn (Python 3.12+)
- **Embeddings**: `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- **Lexical**: `rank-bm25`
- **ANN**: IVF and graph utilities (minimal, CPU-friendly)
- **Frontend**: Plain HTML + CSS + vanilla JS

Directory sketch:

vector-search-cookbook/
├─ web_demo/
│ ├─ backend/ # FastAPI app (routes, dataset/index orchestration)
│ │ └─ app.py
│ └─ frontend/ # Static web (HTML/CSS/JS)
│ ├─ index.html
│ └─ static/
│ ├─ styles.css
│ └─ app.js
├─ src/vscookbook/ # Small library (index, bm25, hybrid, io, ingest, metrics, etc.)
├─ scripts/ # CLI evaluation & utilities
├─ notebooks/ # Optional notebooks (.ipynb) with .py mirrors
├─ data/
│ ├─ datasets/ # sample / uploaded
│ └─ indexes/ # saved indexes
├─ tests/
├─ requirements.txt
└─ README.md

---

## Requirements

- Python **3.12+**
- Linux/macOS/WSL recommended (CPU-only is fine for small demos)

> **Tip:** For low-spec machines, keep datasets small and prefer MiniLM models.

---

## Quickstart

```bash
git clone <YOUR_REPO_URL>.git
cd vector-search-cookbook

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

uvicorn web_demo.backend.app:app --reload --host 0.0.0.0 --port 8000

# Open http://127.0.0.1:8000
