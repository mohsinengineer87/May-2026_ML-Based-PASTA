# PASTA-ML — reproducible benchmark container.
# Build:  docker build -t pasta-ml:latest .
# Run:    docker run --rm -p 8501:8501 pasta-ml:latest
# Open:   http://localhost:8501

FROM python:3.11-slim

# Reproducible Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System deps needed by some scientific wheels (compilers, BLAS runtime).
# Kept minimal for a small image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned dependencies first to maximise Docker layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app
COPY pasta_ml_app_improved.py /app/pasta_ml_app_improved.py

EXPOSE 8501

# Healthcheck so orchestrators know when Streamlit is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -q -O - http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "pasta_ml_app_improved.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
