FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /seller_workload_opt/v3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY algorithms ./algorithms
COPY api ./api
COPY workers ./workers
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/

EXPOSE 5801
CMD ["gunicorn", "--bind", "0.0.0.0:5801", "--workers", "1", "--threads", "4", "--timeout", "60", "api.app:app"]