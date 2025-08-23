# Multi-stage build for M3 Max optimized Rust core
FROM --platform=linux/arm64 rust:1.75-slim as builder

# Install system dependencies for M3 Max optimization
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libnuma-dev \
    libhwloc-dev \
    build-essential \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Set ARM64 M3 Max specific build environment
ENV RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C lto=fat -C panic=abort"
ENV CARGO_PROFILE_RELEASE_LTO=true
ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
ENV CARGO_PROFILE_RELEASE_PANIC="abort"

# Create app directory
WORKDIR /app

# Copy Cargo files for dependency caching
COPY rust_core/Cargo.toml rust_core/Cargo.lock ./

# Create dummy source to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --target aarch64-unknown-linux-gnu
RUN rm -rf src

# Copy actual source code
COPY rust_core/src ./src/
COPY rust_core/benches ./benches/
COPY rust_core/examples ./examples/

# Build the application with M3 Max optimizations
RUN cargo build --release --target aarch64-unknown-linux-gnu

# Runtime stage with minimal footprint
FROM --platform=linux/arm64 debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libnuma1 \
    libhwloc15 \
    ca-certificates \
    numactl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create runtime user for security
RUN groupadd -r rustcore && useradd -r -g rustcore rustcore

# Create directories
RUN mkdir -p /app/logs /app/data /app/config /dev/shm/rustcore \
    && chown -R rustcore:rustcore /app

WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/aarch64-unknown-linux-gnu/release/rust_core /app/rust_core
COPY --from=builder /app/target/aarch64-unknown-linux-gnu/release/rust_core_bench /app/rust_core_bench

# Copy configuration files
COPY rust_core/config/ ./config/

# Set executable permissions
RUN chmod +x /app/rust_core /app/rust_core_bench

# Configure huge pages and memory
RUN echo "vm.nr_hugepages = 2048" >> /etc/sysctl.conf
RUN echo "vm.hugetlb_shm_group = $(getent group rustcore | cut -d: -f3)" >> /etc/sysctl.conf

# Set environment variables for M3 Max optimization
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV M3_MAX_CORES=16
ENV MEMORY_POOL_SIZE=60737418240
ENV IPC_SHARED_MEMORY_SIZE=16106127360
ENV ZERO_COPY_ENABLED=true
ENV NUMA_AFFINITY=true
ENV HUGEPAGES_ENABLED=true
ENV PERFORMANCE_MODE=production

# Configure NUMA and CPU affinity
ENV NUMA_NODE=0
ENV CPU_AFFINITY="0-15"

# Expose ports
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER rustcore

# Set memory limits and NUMA policy
CMD ["numactl", "--membind=0", "--cpunodebind=0", "/app/rust_core"]

# Labels for metadata
LABEL maintainer="RAN LLM Team"
LABEL version="1.0.0"
LABEL description="M3 Max optimized Rust core for hybrid pipeline"
LABEL architecture="arm64"
LABEL performance.memory="60GB"
LABEL performance.cores="16"
LABEL performance.ipc_latency="<100μs"