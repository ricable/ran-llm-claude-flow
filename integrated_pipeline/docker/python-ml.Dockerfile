# Multi-stage build for MLX-optimized Python ML engine
FROM --platform=linux/arm64 python:3.11-slim as base

# Install system dependencies for MLX and M3 Max
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libnuma-dev \
    libhwloc-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install MLX and dependencies
FROM base as builder

WORKDIR /build

# Install MLX framework for M3 Max
RUN pip install --no-cache-dir \
    mlx==0.15.0 \
    mlx-lm==0.10.0 \
    transformers==4.36.0 \
    torch==2.1.0 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    aiohttp==3.9.1 \
    prometheus-client==0.19.0 \
    psutil==5.9.6 \
    huggingface-hub==0.19.4

# Copy requirements and install additional dependencies
COPY python_ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM base as runtime

# Create runtime user
RUN groupadd -r mlengine && useradd -r -g mlengine mlengine

# Create directories
RUN mkdir -p /app/src /app/models /app/logs /app/config /dev/shm/ml_engine \
    && chown -R mlengine:mlengine /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY python_ml/src/ ./src/
COPY python_ml/config/ ./config/
COPY python_ml/models/ ./models/

# Copy startup scripts
COPY python_ml/scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# Set ownership
RUN chown -R mlengine:mlengine /app

# Configure MLX environment for M3 Max
ENV MLX_UNIFIED_MEMORY=48636764160
ENV MLX_GPU_MEMORY=42949672960
ENV MLX_METAL_DEVICE_WRAPPER_TYPE=1
ENV MLX_ENABLE_FLASH_ATTENTION=1
ENV MLX_MEMORY_POOL_LIMIT=45
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Set up Qwen3 model configuration
ENV QWEN3_MODEL_SIZE=auto
ENV QWEN3_CACHE_DIR=/app/models/cache
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV HF_HOME=/app/models/cache

# Performance optimizations
ENV OMP_NUM_THREADS=12
ENV NUMA_NODE=1
ENV CPU_AFFINITY="16-27"

# IPC configuration
ENV IPC_SHARED_MEMORY_SIZE=16106127360
ENV SHARED_MEMORY_PATH=/dev/shm/ml_engine
ENV RUST_CORE_ENDPOINT=rust-core-service:8081

# Monitoring configuration
ENV PROMETHEUS_PORT=9091
ENV LOG_LEVEL=INFO
ENV PERFORMANCE_TRACKING=true

# Expose ports
EXPOSE 8082 8083 9091

# Health check
HEALTHCHECK --interval=45s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8082/health')" || exit 1

# Switch to non-root user
USER mlengine

# Create model download script
RUN cat > /app/download_models.py << 'EOF'
import os
import mlx_lm
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3-1.7B-Instruct",
    "Qwen/Qwen3-7B-Instruct",
    "Qwen/Qwen3-30B-Instruct"
]

for model in models:
    print(f"Downloading {model}...")
    try:
        snapshot_download(
            repo_id=model,
            cache_dir="/app/models/cache",
            local_files_only=False
        )
        print(f"✅ Downloaded {model}")
    except Exception as e:
        print(f"❌ Failed to download {model}: {e}")
EOF

# Startup command with NUMA binding
CMD ["numactl", "--membind=1", "--cpunodebind=1", "python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]

# Labels for metadata  
LABEL maintainer="RAN LLM Team"
LABEL version="1.0.0"
LABEL description="MLX-optimized Python ML engine for M3 Max"
LABEL architecture="arm64"
LABEL framework="MLX"
LABEL models="Qwen3-1.7B,Qwen3-7B,Qwen3-30B"
LABEL performance.memory="45GB"
LABEL performance.unified_memory="true"
LABEL performance.cores="12"