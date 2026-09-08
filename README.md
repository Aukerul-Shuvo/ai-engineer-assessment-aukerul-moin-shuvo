# AI Engineer Assessment

A FastAPI chatbot with one endpoint, `POST /ask`. It answers natural-language questions from a
text corpus (Wikipedia paragraphs from the SQuAD dataset) and from the Superhero API, decides
by itself which source a question needs, sometimes both, and returns every answer with the
sources it came from.

## Setup

Python 3.12 or newer.

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

The corpus and indexes in `data/` are committed, so nothing needs building to run. To rebuild
them from the SQuAD source, or to add dense vectors once `GEMINI_API_KEY` is set:

```bash
python -m scripts.build_dataset --skip-embeddings   # corpus + BM25, no key needed
python -m scripts.build_dataset --only-embeddings   # add dense vectors
```

Interactive docs at http://localhost:8000/docs. Health at `/health/live` and `/health/ready`.

```bash
curl -s http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who won Super Bowl 50, and how strong is Batman?"}'
```

Add `-H "Accept: text/event-stream"` to receive progress events (plan, evidence, answer,
grounding) followed by a `done` event carrying the same JSON body.

## Tools over MCP

The same four tools the service uses internally are exposed as an MCP server, for MCP Inspector,
Claude Desktop or another agent:

```bash
python -m mcp_server.server                                  # stdio
python -m mcp_server.server --transport streamable-http      # http://127.0.0.1:3001/mcp
npx @modelcontextprotocol/inspector python -m mcp_server.server
```

## Test

```bash
pytest
ruff check . && ruff format --check .
mypy
```
