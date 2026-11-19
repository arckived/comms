# syntax=docker/dockerfile:1
# One image for every service (trainer, api, dashboard, slack); compose picks the command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# INSTALL_SBERT=false builds a much smaller image that only supports EMBEDDING_BACKEND=tfidf.
ARG INSTALL_SBERT=true
WORKDIR /app

COPY requirements.txt requirements-sbert.txt ./
RUN pip install -r requirements.txt && \
    if [ "$INSTALL_SBERT" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r requirements-sbert.txt; \
    fi

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps -e .
COPY data/ data/
COPY eval/ eval/
COPY scripts/ scripts/
COPY dashboard/ dashboard/
COPY tests/ tests/

RUN useradd --create-home --uid 1000 app && mkdir -p /app/artifacts /models && \
    chown -R app:app /app /models
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "comms_triage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
