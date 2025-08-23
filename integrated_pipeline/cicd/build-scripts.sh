#!/bin/bash
# Build Scripts for Hybrid Rust-Python RAN LLM Pipeline
# Supports both Rust and Python component building

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_MODE="${BUILD_MODE:-release}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Rust
    if ! command -v rustc &> /dev/null; then
        log_error "Rust compiler not found. Please install Rust."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.11+."
        exit 1
    fi
    
    # Check system dependencies
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! dpkg -l | grep -q build-essential; then
            log_warning "build-essential not found. Installing..."
            sudo apt-get update && sudo apt-get install -y build-essential pkg-config libssl-dev
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew &> /dev/null; then
            log_warning "Homebrew not found. Some dependencies might be missing."
        fi
    fi
    
    log_success "System requirements check completed"
}

# Build Rust components
build_rust() {
    log_info "Building Rust components..."
    
    cd "$PROJECT_ROOT/integrated_pipeline"
    
    # Check if Cargo.toml exists in main directory or subdirectories
    RUST_DIRS=()
    if [[ -f "Cargo.toml" ]]; then
        RUST_DIRS+=(".")
    fi
    
    # Find all subdirectories with Cargo.toml
    while IFS= read -r -d '' dir; do
        RUST_DIRS+=("$dir")
    done < <(find . -name "Cargo.toml" -not -path "./target/*" -not -path "./.git/*" -exec dirname {} \; -print0)
    
    if [[ ${#RUST_DIRS[@]} -eq 0 ]]; then
        log_warning "No Rust projects found (no Cargo.toml files)"
        return 0
    fi
    
    for rust_dir in "${RUST_DIRS[@]}"; do
        log_info "Building Rust project in: $rust_dir"
        cd "$PROJECT_ROOT/integrated_pipeline/$rust_dir"
        
        # Clean previous builds
        cargo clean
        
        # Update dependencies
        cargo update
        
        # Build with optimizations
        if [[ "$BUILD_MODE" == "release" ]]; then
            cargo build --release --jobs "$PARALLEL_JOBS"
            
            # Build with specific optimizations for M3 Max
            export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1"
            cargo build --release --jobs "$PARALLEL_JOBS"
        else
            cargo build --jobs "$PARALLEL_JOBS"
        fi
        
        # Build benchmarks
        cargo build --benches --release
        
        log_success "Rust build completed for: $rust_dir"
    done
    
    log_success "All Rust components built successfully"
}

# Build Python components
build_python() {
    log_info "Building Python components..."
    
    cd "$PROJECT_ROOT/integrated_pipeline"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip and install build tools
    pip install --upgrade pip setuptools wheel
    
    # Install requirements if they exist
    REQUIREMENTS_FILES=()
    while IFS= read -r -d '' req_file; do
        REQUIREMENTS_FILES+=("$req_file")
    done < <(find . -name "requirements*.txt" -not -path "./venv/*" -print0)
    
    for req_file in "${REQUIREMENTS_FILES[@]}"; do
        log_info "Installing requirements from: $req_file"
        pip install -r "$req_file"
    done
    
    # Install MLX if on Apple Silicon
    if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
        log_info "Installing MLX for Apple Silicon optimization..."
        pip install mlx mlx-lm
    fi
    
    # Install development dependencies
    pip install pytest pytest-cov pytest-xdist black isort flake8 mypy
    
    # Find and install Python packages in development mode
    PYTHON_PACKAGES=()
    while IFS= read -r -d '' setup_file; do
        PYTHON_PACKAGES+=("$(dirname "$setup_file")")
    done < <(find . -name "setup.py" -o -name "pyproject.toml" | grep -v venv | head -10 | tr '\n' '\0')
    
    for package_dir in "${PYTHON_PACKAGES[@]}"; do
        log_info "Installing Python package in development mode: $package_dir"
        cd "$PROJECT_ROOT/integrated_pipeline/$package_dir"
        pip install -e .
    done
    
    # Compile Python modules for performance
    log_info "Compiling Python modules..."
    cd "$PROJECT_ROOT/integrated_pipeline"
    python -m compileall . -f -q
    
    log_success "Python components built successfully"
}

# Build shared memory components
build_shared_memory() {
    log_info "Building shared memory IPC components..."
    
    cd "$PROJECT_ROOT/integrated_pipeline"
    
    # Check if shared memory directory exists
    if [[ -d "shared_memory" ]]; then
        cd shared_memory
        
        # Build Rust shared memory components
        if [[ -f "Cargo.toml" ]]; then
            cargo build --release --features="zero-copy,mmap,lock-free"
        fi
        
        # Build Python bindings if they exist
        if [[ -f "setup.py" || -f "pyproject.toml" ]]; then
            pip install -e .
        fi
        
        log_success "Shared memory components built"
    else
        log_warning "Shared memory directory not found, skipping..."
    fi
}

# Build monitoring components
build_monitoring() {
    log_info "Building monitoring components..."
    
    cd "$PROJECT_ROOT/integrated_pipeline"
    
    if [[ -d "monitoring" ]]; then
        cd monitoring
        
        # Build Rust monitoring components
        if [[ -f "Cargo.toml" ]]; then
            cargo build --release --features="metrics,tracing,prometheus"
        fi
        
        # Install Python monitoring dependencies
        if [[ -f "requirements.txt" ]]; then
            pip install -r requirements.txt
        fi
        
        log_success "Monitoring components built"
    else
        log_warning "Monitoring directory not found, skipping..."
    fi
}

# Performance optimizations
apply_performance_optimizations() {
    log_info "Applying performance optimizations..."
    
    # Set environment variables for optimal performance
    export RUST_LOG="info"
    export PYTHONOPTIMIZE="2"
    export MALLOC_ARENA_MAX="4"
    
    # Apple Silicon specific optimizations
    if [[ "$OSTYPE" == "darwin"* ]] && [[ $(uname -m) == "arm64" ]]; then
        export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
        export MLX_NUM_THREADS="16"
    fi
    
    # Linux specific optimizations
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        export OMP_NUM_THREADS="$PARALLEL_JOBS"
        export MKL_NUM_THREADS="$PARALLEL_JOBS"
    fi
    
    log_success "Performance optimizations applied"
}

# Generate build info
generate_build_info() {
    log_info "Generating build information..."
    
    BUILD_INFO_FILE="$PROJECT_ROOT/integrated_pipeline/build-info.json"
    
    cat > "$BUILD_INFO_FILE" <<EOF
{
    "build_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "build_mode": "$BUILD_MODE",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "rust_version": "$(rustc --version)",
    "python_version": "$(python3 --version)",
    "system_info": {
        "os": "$OSTYPE",
        "arch": "$(uname -m)",
        "cores": "$PARALLEL_JOBS"
    }
}
EOF
    
    log_success "Build information saved to: $BUILD_INFO_FILE"
}

# Main build function
main() {
    local component="${1:-all}"
    
    log_info "Starting build process for component: $component"
    log_info "Build mode: $BUILD_MODE"
    log_info "Parallel jobs: $PARALLEL_JOBS"
    
    check_system_requirements
    apply_performance_optimizations
    
    case "$component" in
        "rust")
            build_rust
            ;;
        "python")
            build_python
            ;;
        "shared-memory")
            build_shared_memory
            ;;
        "monitoring")
            build_monitoring
            ;;
        "all")
            build_rust
            build_python
            build_shared_memory
            build_monitoring
            ;;
        *)
            log_error "Unknown component: $component"
            echo "Usage: $0 [rust|python|shared-memory|monitoring|all]"
            exit 1
            ;;
    esac
    
    generate_build_info
    log_success "Build process completed successfully!"
}

# Run main function with all arguments
main "$@"