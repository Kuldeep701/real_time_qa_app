# Real-time Q&A Application Dockerfile
# Optimized for Ubuntu/WSL environments with comprehensive audio support

FROM ubuntu:22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PULSE_SERVER=/mnt/wslg/PulseServer

# Set working directory
WORKDIR /app

# Install system dependencies with proper error handling
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    # Audio system dependencies
    pulseaudio \
    pulseaudio-utils \
    alsa-utils \
    alsa-base \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libasound2-dev \
    libsndfile1-dev \
    # Media processing
    ffmpeg \
    # Networking and utilities
    curl \
    wget \
    git \
    # Cleanup to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    usermod -aG audio appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Download required NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Create directories for runtime files
RUN mkdir -p /app/temp /app/logs

# Expose port
EXPOSE 5000

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start command
CMD ["python3", "app.py"]
