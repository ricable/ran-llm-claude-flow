# Multi-stage Dockerfile for Hybrid Rust-Python RAN LLM Pipeline
# Optimized for M3 Max and production deployment

# Build arguments
ARG BUILD_VERSION=latest
ARG BUILD_DATE
ARG RUST_VERSION=1.75
ARG PYTHON_VERSION=3.11

#===============================================================================
# Stage 1: Rust Builder
#===============================================================================
FROM rust:${RUST_VERSION}-slim as rust-builder

# Install system dependencies for Rust compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Rust environment for optimal compilation
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1"
ENV CARGO_PROFILE_RELEASE_LTO=true
ENV CARGO_PROFILE_RELEASE_PANIC=abort

# Create app directory
WORKDIR /app

# Copy Rust source files
COPY integrated_pipeline/ ./integrated_pipeline/
COPY Cargo.toml Cargo.lock ./

# Build Rust components
WORKDIR /app/integrated_pipeline
RUN find . -name "Cargo.toml" -exec dirname {} \; | while read project_dir; do \
    echo "Building Rust project in: $project_dir"; \
    cd "/app/integrated_pipeline/$project_dir" && \
    cargo build --release && \
    cargo build --release --bins; \
    done

#===============================================================================
# Stage 2: Python Builder
#===============================================================================
FROM python:${PYTHON_VERSION}-slim as python-builder

# Install system dependencies for Python
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=2
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy Python requirements and source
WORKDIR /app
COPY integrated_pipeline/ ./integrated_pipeline/

# Install Python dependencies
WORKDIR /app/integrated_pipeline
RUN find . -name "requirements*.txt" -exec pip install -r {} \;

# Install MLX for Apple Silicon optimization (if applicable)
RUN if [ "$(uname -m)" = "arm64" ]; then \
    pip install mlx mlx-lm; \
    fi

# Install Python packages in development mode
RUN find . -name "setup.py" -o -name "pyproject.toml" | \
    grep -v __pycache__ | \
    head -10 | \
    while read setup_file; do \
        project_dir=$(dirname "$setup_file"); \
        echo "Installing Python package: $project_dir"; \
        cd "/app/integrated_pipeline/$project_dir" && \
        pip install -e .; \
    done

# Compile Python modules for performance
RUN python -m compileall . -f -q

#===============================================================================
# Stage 3: Production Runtime
#===============================================================================
FROM ubuntu:22.04 as runtime

# Build metadata
LABEL maintainer="RAN LLM Pipeline Team"
LABEL version=${BUILD_VERSION}
LABEL build-date=${BUILD_DATE}
LABEL description="Hybrid Rust-Python RAN LLM Pipeline"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    python3 \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser

# Create application directories
RUN mkdir -p /app /app/data /app/logs /app/config /dev/shm/ran-llm
WORKDIR /app

# Copy Python virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Copy Rust binaries from builder
COPY --from=rust-builder /app/integrated_pipeline/target/release/ ./bin/

# Copy application source
COPY --from=rust-builder /app/integrated_pipeline ./
COPY --from=python-builder /app/integrated_pipeline/build-info.json ./

# Copy deployment configuration
COPY integrated_pipeline/deployment/config/ ./config/

# Create configuration files
RUN cat > config/app.json << EOF
{
    "environment": "\${ENVIRONMENT:-production}",
    "deployment_slot": "\${DEPLOYMENT_SLOT:-blue}",
    "rust_config": {
        "log_level": "\${RUST_LOG:-info}",
        "max_threads": 16,
        "memory_pool_gb": 60,
        "zero_copy_enabled": true
    },
    "python_config": {
        "optimize_level": 2,
        "mlx_enabled": true,
        "memory_limit_gb": 45,
        "qwen_models": ["1.7B", "7B", "30B"]
    },
    "ipc_config": {
        "shared_memory_gb": 15,
        "latency_target_us": 100,
        "lock_free": true
    },
    "monitoring": {
        "prometheus_port": 9090,
        "health_check_port": 8080,
        "metrics_enabled": true
    }
}
EOF

# Create startup script
RUN cat > startup.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "🚀 Starting RAN LLM Pipeline"
echo "Environment: ${ENVIRONMENT:-production}"
echo "Deployment Slot: ${DEPLOYMENT_SLOT:-blue}"

# Initialize shared memory
echo "📝 Initializing shared memory..."
mkdir -p /dev/shm/ran-llm
chmod 755 /dev/shm/ran-llm

# Start monitoring
echo "📊 Starting monitoring..."
python monitoring/src/main.py &
MONITORING_PID=$!

# Start Rust core
echo "🦀 Starting Rust core..."
./bin/rust_core --config ./config/app.json &
RUST_PID=$!

# Start Python ML engine
echo "🐍 Starting Python ML engine..."
python python_ml/src/main.py --config ./config/app.json &
PYTHON_PID=$!

# Wait for services to be ready
sleep 10

# Health check
echo "🔍 Running health check..."
curl -f http://localhost:8080/health || {
    echo "❌ Health check failed"
    exit 1
}

echo "✅ All services started successfully"

# Keep container running and handle signals
trap 'echo "🛑 Shutting down..."; kill $MONITORING_PID $RUST_PID $PYTHON_PID; wait' TERM INT

wait
EOF

# Make startup script executable
RUN chmod +x startup.sh

# Set up health check script
RUN cat > health-check.sh << 'EOF'
#!/bin/bash
# Health check for container orchestrator

# Check if all processes are running
if ! pgrep -f "rust_core" > /dev/null; then
    echo "Rust core not running"
    exit 1
fi

if ! pgrep -f "python.*main.py" > /dev/null; then
    echo "Python ML engine not running"
    exit 1
fi

# Check HTTP health endpoint
if ! curl -f -s http://localhost:8080/health > /dev/null; then
    echo "Health endpoint not responding"
    exit 1
fi

echo "All health checks passed"
exit 0
EOF

RUN chmod +x health-check.sh

# Set ownership and permissions
RUN chown -R appuser:appuser /app /dev/shm/ran-llm
RUN chmod -R 755 /app
RUN chmod 1777 /dev/shm/ran-llm

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 80 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./health-check.sh

# Environment variables
ENV ENVIRONMENT=production
ENV DEPLOYMENT_SLOT=blue
ENV RUST_LOG=info
ENV PYTHONOPTIMIZE=2
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["./startup.sh"]