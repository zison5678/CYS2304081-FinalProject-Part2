FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m zipfile -e data.zip /app && \
    python -m zipfile -e model.zip /app

RUN test -f /app/data/cleaned_incidents.csv && \
    test -f /app/data/clustered_incidents.csv && \
    test -f /app/data/cluster_profiles.csv && \
    test -d /app/model/resolution_time_rf_pipeline_model/metadata && \
    test -d /app/model/resolution_time_rf_pipeline_model/stages

CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
