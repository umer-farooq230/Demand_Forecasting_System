# ── Demand Forecasting Pipeline ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# libgomp1 = OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/  /app/src/
COPY data/ /app/data/

RUN mkdir -p /app/models /app/outputs /app/reports /app/mlflow_data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=file:///app/mlflow_data

CMD ["python", "src/run_pipeline.py"]