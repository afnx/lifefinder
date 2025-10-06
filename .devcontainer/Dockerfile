FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  HF_HOME=/app/cache/huggingface \
  TRANSFORMERS_CACHE=/app/cache/transformers

WORKDIR /app

# System deps for common ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git curl ffmpeg libsndfile1 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# Install the lifefinder in development mode
RUN pip install -e .

EXPOSE 8501

CMD ["tail", "-f", "/dev/null"]