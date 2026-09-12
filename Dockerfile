# syntax=docker/dockerfile:1.7
#
# Two stages. The builder installs the package into a virtualenv and downloads the reranker
# weights; the runtime copies only the virtualenv, the corpus and the source, and runs as a
# non-root user. The image defaults to ENVIRONMENT=prod, which refuses to start without the
# required secrets. docker-compose overrides that to dev so a reviewer can start it bare.

FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
COPY mcp_server ./mcp_server
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .

# Corpus, indexes, and the reranker weights, fetched once here so the first request is fast.
COPY data ./data
RUN /opt/venv/bin/python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/build/data/models', max_length=512)"


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=prod \
    LOG_JSON=true \
    DATA_DIR=/app/data \
    RERANKER_CACHE_DIR=/app/data/models

RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /build/data ./data
COPY --chown=app:app app ./app
COPY --chown=app:app mcp_server ./mcp_server
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app pyproject.toml README.md ./

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).status == 200 else 1)"]

# One worker on purpose: the rate limiter, response cache and sessions are in-memory, so extra
# workers would each keep their own copy. Scale horizontally behind a load balancer with Redis
# backing those three, which the ADRs name as the scale-up step.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
