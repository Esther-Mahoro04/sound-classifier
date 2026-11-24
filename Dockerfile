FROM python:3.12-slim

RUN apt-get update && apt-get install -y libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src ./src
COPY models ./models
COPY reports ./reports
COPY data ./data
COPY app.py .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]