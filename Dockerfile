FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/models/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
# Bake the offline fallbacks into the image so degraded mode works with no network.
RUN mkdir -p models/piper \
 && curl -sL -o models/piper/en_US-lessac-low.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx \
 && curl -sL -o models/piper/en_US-lessac-low.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8')"
COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY data/utterances ./data/utterances
RUN useradd -m voice && mkdir -p reports && chown -R voice /app
USER voice
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s CMD curl -fs localhost:8000/health || exit 1
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
