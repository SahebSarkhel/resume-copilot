# Resume Copilot

An agentic resume-review system: parse -> ML score -> LLM critique -> rewrite -> re-score,
served via FastAPI.

## Setup

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
cp .env.example .env           # then fill in your GOOGLE_API_KEY
```

Run the API skeleton:
```bash
uvicorn src.api.main:app --reload
```
Visit http://127.0.0.1:8000/health to confirm it's alive.

## Project Structure
```
data/          raw datasets (gitignored — download via Kaggle, don't commit)
notebooks/     exploration only (EDA, embedding tests) — move working code into src/
src/parsing/   resume & JD extraction (PDF/DOCX -> structured text)
src/scoring/   embeddings, similarity, classifier
src/agent/     orchestration loop (score -> critique -> rewrite -> re-score)
src/api/       FastAPI app
frontend/      minimal UI
```

## Datasets
- Resume Dataset (Kaggle): https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
- LinkedIn Job Postings 2023-2024 (Kaggle): https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

Download both into `data/` (they're gitignored). Optionally use the Kaggle CLI:
```bash
kaggle datasets download -d snehaanbhawal/resume-dataset -p data/ --unzip
kaggle datasets download -d arshkon/linkedin-job-postings -p data/ --unzip
```

## Roadmap
- [ ] Week 1: Parsing + EDA on both datasets
- [ ] Week 2: Embedding similarity scoring (resume <-> JD), evaluate
- [ ] Week 3: LLM critique layer (structured JSON output)
- [ ] Week 4: Agentic loop with stopping criteria
- [ ] Week 5: FastAPI wiring, error handling
- [ ] Week 6: Frontend, deployment, write-up
