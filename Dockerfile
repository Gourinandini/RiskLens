# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Builder: compile/install Python dependencies in isolation
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed to compile LightGBM / numpy / scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all dependencies into a dedicated prefix so we can copy it cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Runtime: lean final image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata labels
LABEL org.opencontainers.image.title="RiskLens Credit Risk Intelligence Platform"
LABEL org.opencontainers.image.description="AI-Powered Credit Risk Platform – EDA · ML Scoring · NL-to-SQL"
LABEL org.opencontainers.image.authors="Gourinandini"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Runtime system libraries (libgomp needed by LightGBM at runtime, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source (data / models / sql are bind-mounted at runtime)
COPY app.py           ./app.py
COPY src/             ./src/
COPY .streamlit/      ./.streamlit/

# Create runtime-writable directories; they will be over-mounted by volumes
RUN mkdir -p data models sql logs \
    && chmod 777 data models sql logs

# Non-root user for security best-practice
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Streamlit listens on 8501
EXPOSE 8501

# Healthcheck: poll the Streamlit health endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Environment defaults (overridable via docker-compose env_file / -e flags)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATA_PATH=/app/data \
    DB_PATH=/app/sql/credit_risk.db \
    MODEL_PATH=/app/models/lgbm_model.joblib

# Launch Streamlit in headless server mode
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
