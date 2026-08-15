# OpenPronounce web app (FastAPI + UI). CPU only.
#
#   docker build -t openpronounce .
#   docker run -p 8000:8000 openpronounce
#
# The two Wav2Vec2 models (~1.2 GB each) are downloaded at build time so the
# container starts fast and works offline (except gTTS, which needs the network).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels of torch are much smaller than the default CUDA ones.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY openpronounce ./openpronounce
RUN pip install ".[app]"

RUN python -c "from openpronounce.speech import _load_models; _load_models()" \
    && python -c "from openpronounce.phones import _load_model; _load_model()"

COPY server.py ./
COPY templates ./templates
COPY static ./static

EXPOSE 8000
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
