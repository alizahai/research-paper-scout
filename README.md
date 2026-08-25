# Research Paper Scout

Search academic literature, group it into themes, and synthesize what the papers collectively say.

## Structure

```
research-paper-scout/
├── backend/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── semantic_scholar.py     # Phase 1: fetching papers
│   ├── embeddings.py           # Phase 2: embedding + ChromaDB
│   ├── clustering.py           # Phase 3: clustering + labeling
│   ├── synthesis.py            # Phase 4: RAG + Gemini calls
│   ├── models.py               # Pydantic schemas for requests/responses
│   ├── requirements.txt
│   └── .env                    # API keys (never commit this)
├── frontend/
│   └── app.py                  # Streamlit UI
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv backend/venv
backend\venv\Scripts\activate
pip install -r backend/requirements.txt
```

Build the venv from a standard CPython (python.org or the Microsoft Store), not
MSYS2/UCRT64 Python. PyPI publishes no wheels for the MSYS2 ABI, so `chromadb`
falls back to a Rust source build and `torch` has nothing to install at all.

Fill in `backend/.env` with your keys:

```
GEMINI_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...
```

`SEMANTIC_SCHOLAR_API_KEY` is optional — every endpoint used works without it —
but the unauthenticated rate limit is aggressive enough that 429s are routine
even from an idle client, so a [free key](https://www.semanticscholar.org/product/api#api-key-form)
is worth requesting.

## Running

Backend — run from inside `backend/`, since the modules import each other by
plain name (`from synthesis import generate_text`):

```bash
cd backend
uvicorn main:app --reload
```

Frontend:

```bash
streamlit run frontend/app.py
```

## Attribution

Paper data provided by [Semantic Scholar](https://www.semanticscholar.org/).
