FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway provides $PORT. Bind to it.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
