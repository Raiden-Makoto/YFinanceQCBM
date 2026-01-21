FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers logs immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install system deps (if future packages need build tools) and Python deps.
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application code.
COPY . .

# Streamlit listens on $PORT; Hugging Face expects 7860.
EXPOSE 7860

CMD ["bash", "-c", "streamlit run monitor.py --server.port=${PORT} --server.address=0.0.0.0"]
