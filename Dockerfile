FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "yfinance>=0.2.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.20.0" \
    "openai>=1.0.0"

# Copy application code
COPY . .

# Pre-download and cache market data at build time
RUN python -c "from simulator import MarketDataSimulator; MarketDataSimulator()" || true

# HuggingFace Spaces uses port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/schema || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
