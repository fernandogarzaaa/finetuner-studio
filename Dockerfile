# --- web UI ------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- runtime (CUDA 12.1, PyTorch preinstalled) --------------------------------
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 \
    FTS_HOME=/data \
    HF_HOME=/data/hf-cache \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY examples/ examples/
RUN pip install -e ".[train]"
COPY --from=web /web/dist web/dist
VOLUME ["/data"]
EXPOSE 8000
CMD ["fts", "serve", "--host", "0.0.0.0", "--port", "8000"]
