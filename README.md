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
│   ├── synthesis.py            # Phase 4: RAG + Claude calls
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
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r backend/requirements.txt
```

Fill in `backend/.env` with your keys:

```
ANTHROPIC_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...
```

## Running

Backend:

```bash
uvicorn backend.main:app --reload
```

Frontend:

```bash
streamlit run frontend/app.py
```
