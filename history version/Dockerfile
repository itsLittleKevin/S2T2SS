# 🐳 S2T2SS Docker Container
# Multi-stage build for optimized deployment

# Stage 1: Base Python environment with CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ffmpeg \
    portaudio19-dev \
    pulseaudio \
    alsa-utils \
    libasound2-dev \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Install Japanese language support
RUN pip install fugashi[unidic-lite] cutlet unidic-lite

# Stage 2: Application layer
FROM base as app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models rvc_models/characters config logs

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create non-root user for security
RUN useradd -m -u 1000 s2t2ss && \
    chown -R s2t2ss:s2t2ss /app
USER s2t2ss

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; import funasr; import TTS; print('✅ All modules loaded')" || exit 1

# Expose ports
EXPOSE 8000 1234

# Default command
CMD ["python3", "core/main_pipeline.py"]

# Labels
LABEL maintainer="S2T2SS Team"
LABEL version="1.0.0"
LABEL description="Real-time Speech-to-Text-to-Speech with Translation & RVC"
