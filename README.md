# AI Engineer Assessment

A FastAPI chatbot with one endpoint, `POST /ask`. It answers natural-language questions from a
text corpus (Wikipedia paragraphs from the SQuAD dataset) and from the Superhero API, decides
by itself which source a question needs, sometimes both, and returns every answer with the
sources it came from.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
cp .env.example .env            # then fill in the keys
```

## Run

```bash
uvicorn app.main:app --reload
```

Interactive docs at http://localhost:8000/docs. Health at `/health/live` and `/health/ready`.

## Test

```bash
pytest
ruff check . && ruff format --check .
mypy
```
