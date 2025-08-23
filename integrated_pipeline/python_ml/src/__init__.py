#!/usr/bin/env python3
"""
Python ML Engine for Hybrid Rust-Python Pipeline

A high-performance ML engine optimized for Apple Silicon M3 Max with MLX acceleration.
Designed for document processing and QA generation in RAN feature documentation.

Key Components:
- ModelManager: Qwen3 model management with dynamic selection
- SemanticProcessor: Document understanding and QA generation
- MLXAccelerator: M3 Max GPU optimization with unified memory
- IPCClient: High-performance communication with Rust core

Author: Claude Code
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code"
__email__ = "noreply@anthropic.com"
__license__ = "MIT"

# Core imports
from .model_manager import (
    Qwen3ModelManager,
    ModelSize,
    InferenceBackend,
    ModelConfig,
    ProcessingHints,
    ModelPerformance
)

from .semantic_processor import (
    SemanticProcessor,
    QAPair,
    QuestionType,
    DifficultyLevel,
    QualityMetrics,
    DocumentAnalysis
)

from .mlx_accelerator import (
    MLXAccelerator,
    MemoryPool,
    MLXKernelOptimizer,
    BatchProcessor,
    BatchRequest,
    InferenceResult,
    PerformanceMetrics
)

from .ipc_client import (
    IPCClient,
    IPCMessage,
    MessageType,
    Priority,
    DocumentProcessingRequest,
    ProcessingResponse,
    SharedMemoryManager
)

# Convenience imports
__all__ = [
    # Model Management
    "Qwen3ModelManager",
    "ModelSize",
    "InferenceBackend",
    "ModelConfig",
    "ProcessingHints",
    "ModelPerformance",
    
    # Semantic Processing
    "SemanticProcessor",
    "QAPair",
    "QuestionType",
    "DifficultyLevel",
    "QualityMetrics",
    "DocumentAnalysis",
    
    # MLX Acceleration
    "MLXAccelerator",
    "MemoryPool",
    "MLXKernelOptimizer",
    "BatchProcessor",
    "BatchRequest",
    "InferenceResult",
    "PerformanceMetrics",
    
    # IPC Communication
    "IPCClient",
    "IPCMessage",
    "MessageType",
    "Priority",
    "DocumentProcessingRequest",
    "ProcessingResponse",
    "SharedMemoryManager"
]

# Version info
__version_info__ = tuple(map(int, __version__.split('.')))

# Package metadata
__title__ = "Python ML Engine"
__description__ = "High-performance ML engine for Rust-Python hybrid pipeline"
__url__ = "https://github.com/ruvnet/claude-flow"

# Feature flags
TRY_MLX_IMPORT = True
MLX_AVAILABLE = False

if TRY_MLX_IMPORT:
    try:
        import mlx.core as mx
        MLX_AVAILABLE = True
    except ImportError:
        MLX_AVAILABLE = False

# System info
def get_system_info() -> dict:
    """
    Get system information for optimization.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'mlx_available': MLX_AVAILABLE
    }
    
    # Apple Silicon specific info
    if platform.system() == 'Darwin' and 'arm64' in platform.machine():
        info['apple_silicon'] = True
        info['unified_memory'] = True
    else:
        info['apple_silicon'] = False
        info['unified_memory'] = False
        
    return info


# Performance recommendations
def get_performance_recommendations() -> list:
    """
    Get performance optimization recommendations based on system.
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    system_info = get_system_info()
    
    if system_info['apple_silicon']:
        if system_info['mlx_available']:
            recommendations.append("✅ MLX available - optimal performance expected")
            recommendations.append("📈 Enable unified memory optimization")
        else:
            recommendations.append("⚠️ MLX not available - install for better performance")
            recommendations.append("💡 pip install mlx mlx-lm")
            
    if system_info['memory_gb'] > 32:
        recommendations.append("🚀 High memory system - enable large model support")
    elif system_info['memory_gb'] < 16:
        recommendations.append("⚠️ Low memory system - use smaller models and quantization")
        
    if system_info['cpu_count'] > 8:
        recommendations.append("⚡ Multi-core system - enable parallel processing")
        
    return recommendations


# Initialization message
def print_initialization_info():
    """
    Print initialization information and recommendations.
    """
    import sys
    
    print(f"🤖 Python ML Engine v{__version__}")
    print(f"📦 Initialized on {get_system_info()['platform']} {get_system_info()['architecture']}")
    
    if MLX_AVAILABLE:
        print("⚡ MLX acceleration enabled")
    else:
        print("⚠️  MLX not available - using CPU/GPU fallback")
        
    recommendations = get_performance_recommendations()
    if recommendations:
        print("\n💡 Performance Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
            
    print("\n🔗 Integration Components:")
    print("   📊 Model Manager: Qwen3 dynamic selection")
    print("   🧠 Semantic Processor: Document QA generation")
    print("   ⚡ MLX Accelerator: M3 Max optimization")
    print("   🔄 IPC Client: Rust communication")
    print()


# Optional auto-initialization
if __name__ != "__main__":
    # Only show info in interactive environments
    import sys
    if hasattr(sys, 'ps1') or 'jupyter' in sys.modules:
        print_initialization_info()
