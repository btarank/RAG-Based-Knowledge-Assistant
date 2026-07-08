FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by pymupdf, torch, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first — leverages Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and vector store
RUN mkdir -p data/uploads data/processed/vectorstore data/eval

# Render/Railway inject PORT dynamically — default to 8000 for local
ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}