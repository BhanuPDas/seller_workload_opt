FROM python:3.11-slim

# coinor-cbc: system CBC solver as a fallback/robustness measure -- PuLP
# also ships a bundled linux-x86_64 CBC binary, but installing the system
# package avoids surprises on other architectures/base images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        coinor-cbc \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY seller_workload_opt ./seller_workload_opt

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/

# Default command runs the API; docker-compose overrides `command` for
# each of the 4 workers and the comparator using this same image.
EXPOSE 5801
CMD ["gunicorn", "--bind", "0.0.0.0:5801", "--workers", "1", "--threads", "4", "--timeout", "60", "seller_workload_opt.v3.api.app:app"]
