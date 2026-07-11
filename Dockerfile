# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
# --user installs to /root/.local, which we copy verbatim into the final stage —
# keeps the runtime image free of pip's build cache and any compiler toolchains.
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim

# curl is only for the HEALTHCHECK below; removed from apt lists immediately after.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Only what's needed to serve: no data/, notebooks/, tests/, mlruns/, dev tooling.
COPY api/ ./api/
COPY src/ ./src/
COPY models/final_model/ ./models/final_model/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
