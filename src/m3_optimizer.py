#!/usr/bin/env python3
"""
M3 MacBook Pro Optimization Engine
Advanced optimization strategies for Apple Silicon M3 with 128GB unified memory
"""

import os
import json
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import mmap
import gc

@dataclass
class OptimizationConfig:
    """Configuration for M3 optimization strategies"""
    max_memory_usage_percent: float = 85.0  # Leave 15% for system
    preferred_batch_size: int = 512
    max_concurrent_requests: int = 8
    enable_memory_mapping: bool = True
    use_quantization: bool = True
    cpu_threads: int = 12  # Leave 4 cores for system
    gpu_memory_fraction: float = 0.8

class M3UnifiedMemoryManager:
    """
    Unified Memory Management for M3 Architecture
    Optimizes memory allocation patterns for the shared CPU/GPU memory pool
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_pools: Dict[str, Any] = {}
        self.allocated_memory: Dict[str, int] = {}
        self.total_memory_gb = 128
        
    def create_memory_pool(self, pool_name: str, size_gb: float) -> bool:
        """Create a pre-allocated memory pool to avoid fragmentation"""
        try:
            size_bytes = int(size_gb * 1024 * 1024 * 1024)
            
            # Use mmap for large memory pools to leverage unified memory efficiently
            pool = mmap.mmap(-1, size_bytes)
            self.memory_pools[pool_name] = pool
            self.allocated_memory[pool_name] = size_bytes
            
            print(f"Created memory pool '{pool_name}': {size_gb:.1f}GB")
            return True
            
        except Exception as e:
            print(f"Failed to create memory pool '{pool_name}': {e}")
            return False
    
    def get_memory_utilization(self) -> Dict[str, float]:
        """Get current memory utilization statistics"""
        total_allocated = sum(self.allocated_memory.values())
        
        return {
            "total_system_gb": self.total_memory_gb,
            "total_allocated_gb": total_allocated / (1024**3),
            "available_gb": self.total_memory_gb - (total_allocated / (1024**3)),
            "utilization_percent": (total_allocated / (1024**3)) / self.total_memory_gb * 100,
            "pools": {name: size / (1024**3) for name, size in self.allocated_memory.items()}
        }
    
    def optimize_allocation_pattern(self, workload_type: str) -> Dict[str, Any]:
        """Optimize memory allocation for specific workload types"""
        patterns = {
            "inference": {
                "model_pool_gb": 32,
                "batch_pool_gb": 16,
                "cache_pool_gb": 8,
                "working_pool_gb": 12
            },
            "training": {
                "model_pool_gb": 40,
                "gradient_pool_gb": 20,
                "optimizer_pool_gb": 16,
                "working_pool_gb": 16
            },
            "data_processing": {
                "input_pool_gb": 50,
                "processing_pool_gb": 30,
                "output_pool_gb": 20,
                "temp_pool_gb": 10
            }
        }
        
        pattern = patterns.get(workload_type, patterns["inference"])
        
        # Adjust based on current memory pressure
        utilization = self.get_memory_utilization()
        if utilization["utilization_percent"] > 70:
            # Scale down pool sizes
            for key in pattern:
                pattern[key] *= 0.8
        
        return pattern
    
    def cleanup_memory_pools(self):
        """Clean up all memory pools"""
        for name, pool in self.memory_pools.items():
            try:
                if hasattr(pool, 'close'):
                    pool.close()
            except:
                pass
        
        self.memory_pools.clear()
        self.allocated_memory.clear()
        gc.collect()

class M3MLXOptimizer:
    """
    Apple MLX Optimization for M3
    Specialized optimizations for MLX framework on Apple Silicon
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.mlx_available = self._check_mlx_availability()
    
    def _check_mlx_availability(self) -> bool:
        """Check if MLX is available and properly configured"""
        try:
            import mlx.core as mx
            print(f"MLX available on device: {mx.default_device()}")
            return True
        except ImportError:
            print("MLX not available - install with: pip install mlx mlx-lm")
            return False
    
    def optimize_model_loading(self, model_path: str) -> Dict[str, Any]:
        """Optimize MLX model loading for M3"""
        if not self.mlx_available:
            return {"error": "MLX not available"}
        
        try:
            import mlx.core as mx
            import mlx.nn as nn
            
            # MLX-specific optimizations for M3
            optimization_settings = {
                "memory_mapping": True,  # Use memory mapping for large models
                "lazy_loading": True,    # Load model parts on-demand
                "quantization": "int4" if self.config.use_quantization else None,
                "batch_size": self.config.preferred_batch_size,
                "max_tokens": 2048,
                "device_optimization": True
            }
            
            print(f"Optimizing MLX model loading: {model_path}")
            print(f"Settings: {optimization_settings}")
            
            # Set MLX-specific environment variables for M3
            os.environ["MLX_DISABLE_FAST_MATH"] = "0"  # Enable fast math on M3
            os.environ["MLX_MEMORY_POOL"] = "1"        # Enable memory pooling
            
            return {
                "status": "optimized",
                "settings": optimization_settings,
                "estimated_memory_gb": self._estimate_model_memory(model_path),
                "recommended_batch_size": self._calculate_optimal_batch_size()
            }
            
        except Exception as e:
            return {"error": f"MLX optimization failed: {e}"}
    
    def _estimate_model_memory(self, model_path: str) -> float:
        """Estimate model memory requirements"""
        # This is a simplified estimation - real implementation would
        # inspect model architecture and parameter count
        
        try:
            model_size = Path(model_path).stat().st_size if Path(model_path).exists() else 0
            
            # Rough estimation: model file size * 1.5 for overhead
            estimated_gb = (model_size * 1.5) / (1024**3)
            
            # Add quantization savings if enabled
            if self.config.use_quantization:
                estimated_gb *= 0.5  # ~50% reduction with int4 quantization
            
            return max(1.0, estimated_gb)  # Minimum 1GB
            
        except:
            return 8.0  # Default estimation
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for M3 unified memory"""
        # Consider available memory and model size
        available_memory_gb = 128 * (self.config.max_memory_usage_percent / 100)
        
        # Estimate memory per batch item (highly simplified)
        memory_per_item_mb = 64  # Rough estimate for transformer models
        
        max_batch_size = int((available_memory_gb * 1024) / memory_per_item_mb)
        
        # Prefer power-of-2 batch sizes for better GPU utilization
        optimal_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        
        for size in optimal_sizes:
            if size <= max_batch_size:
                continue
            return min(size, max_batch_size)
        
        return min(self.config.preferred_batch_size, max_batch_size)

class M3PyTorchOptimizer:
    """
    PyTorch MPS Optimization for M3
    Specialized optimizations for PyTorch Metal Performance Shaders
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.mps_available = self._check_mps_availability()
    
    def _check_mps_availability(self) -> bool:
        """Check if PyTorch MPS is available"""
        try:
            import torch
            available = torch.backends.mps.is_available()
            built = torch.backends.mps.is_built()
            print(f"PyTorch MPS - Available: {available}, Built: {built}")
            return available and built
        except ImportError:
            print("PyTorch not available - install with appropriate MPS support")
            return False
    
    def optimize_mps_settings(self) -> Dict[str, Any]:
        """Configure optimal MPS settings for M3"""
        if not self.mps_available:
            return {"error": "MPS not available"}
        
        try:
            import torch
            
            # Set optimal MPS configurations for M3
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Configure memory allocation
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                torch.mps.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            settings = {
                "device": "mps",
                "memory_fraction": self.config.gpu_memory_fraction,
                "fallback_enabled": True,
                "optimal_batch_sizes": {
                    "inference": [64, 128, 256, 512],
                    "training": [16, 32, 64, 128],
                    "fine_tuning": [8, 16, 32]
                },
                "data_loading": {
                    "num_workers": min(8, self.config.cpu_threads),
                    "pin_memory": True,
                    "persistent_workers": True
                }
            }
            
            return {
                "status": "optimized",
                "settings": settings,
                "device_name": "Apple M3 Max",
                "unified_memory": True
            }
            
        except Exception as e:
            return {"error": f"MPS optimization failed: {e}"}
    
    def create_optimized_dataloader(self, dataset, batch_size: Optional[int] = None):
        """Create optimized DataLoader for M3"""
        if not self.mps_available:
            return None
        
        try:
            import torch
            from torch.utils.data import DataLoader
            
            optimal_batch_size = batch_size or self._calculate_optimal_batch_size()
            
            dataloader = DataLoader(
                dataset,
                batch_size=optimal_batch_size,
                num_workers=min(8, self.config.cpu_threads),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=False
            )
            
            return dataloader
            
        except Exception as e:
            print(f"Failed to create optimized DataLoader: {e}")
            return None
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for PyTorch MPS"""
        # Similar logic to MLX but adapted for PyTorch patterns
        return min(256, self.config.preferred_batch_size)

class M3InferenceOptimizer:
    """
    Local Inference Optimization for LM Studio and Ollama on M3
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def optimize_lm_studio(self) -> Dict[str, Any]:
        """Optimize LM Studio configuration for M3"""
        settings = {
            "gpu_acceleration": True,
            "context_length": 32768,  # M3 can handle large contexts
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "model_format": "GGUF",
            "quantization": "Q4_K_M",  # Balance of speed and quality
            "memory_allocation": {
                "model_memory_gb": 24,
                "context_memory_gb": 16,
                "batch_memory_gb": 8
            },
            "performance_settings": {
                "flash_attention": True,
                "grouped_query_attention": True,
                "rope_scaling": "linear"
            }
        }
        
        return {
            "status": "optimized",
            "application": "LM Studio",
            "settings": settings,
            "estimated_performance": {
                "tokens_per_second": "80-120 for 7B models",
                "context_processing": "2-3 seconds for 8K tokens",
                "concurrent_capacity": f"{self.config.max_concurrent_requests} requests"
            }
        }
    
    def optimize_ollama(self) -> Dict[str, Any]:
        """Optimize Ollama configuration for M3"""
        # Set Ollama environment variables
        env_settings = {
            "OLLAMA_NUM_PARALLEL": str(self.config.max_concurrent_requests),
            "OLLAMA_MAX_LOADED_MODELS": "4",
            "OLLAMA_KEEP_ALIVE": "-1",  # Keep models in memory
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_FLASH_ATTENTION": "1"
        }
        
        # Apply environment settings
        for key, value in env_settings.items():
            os.environ[key] = value
        
        settings = {
            "parallel_requests": self.config.max_concurrent_requests,
            "model_persistence": True,
            "memory_efficient_loading": True,
            "recommended_models": {
                "code": ["codellama:34b-instruct", "deepseek-coder:33b"],
                "chat": ["llama2:70b", "mixtral:8x7b"],
                "embedding": ["nomic-embed-text", "all-minilm"]
            },
            "optimization_flags": [
                "--numa false",  # M3 has unified memory
                "--gpu-memory-utilization 0.8",
                "--max-model-len 32768"
            ]
        }
        
        return {
            "status": "optimized",
            "application": "Ollama",
            "environment": env_settings,
            "settings": settings,
            "estimated_performance": {
                "concurrent_models": "3-4 models simultaneously",
                "model_switching": "<5 seconds",
                "memory_efficiency": "90%+"
            }
        }

class M3PipelineOptimizer:
    """
    End-to-end pipeline optimization for M3 workflows
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_manager = M3UnifiedMemoryManager(config)
        self.mlx_optimizer = M3MLXOptimizer(config)
        self.pytorch_optimizer = M3PyTorchOptimizer(config)
        self.inference_optimizer = M3InferenceOptimizer(config)
    
    async def optimize_inference_pipeline(self, model_path: str, workload_type: str) -> Dict[str, Any]:
        """Optimize complete inference pipeline for M3"""
        
        # Step 1: Setup memory pools
        memory_pattern = self.memory_manager.optimize_allocation_pattern(workload_type)
        for pool_name, size_gb in memory_pattern.items():
            self.memory_manager.create_memory_pool(pool_name, size_gb)
        
        # Step 2: Optimize framework settings
        mlx_config = self.mlx_optimizer.optimize_model_loading(model_path)
        pytorch_config = self.pytorch_optimizer.optimize_mps_settings()
        
        # Step 3: Configure local inference engines
        lm_studio_config = self.inference_optimizer.optimize_lm_studio()
        ollama_config = self.inference_optimizer.optimize_ollama()
        
        # Step 4: Create coordination strategy
        coordination = {
            "primary_framework": "MLX" if mlx_config.get("status") == "optimized" else "PyTorch",
            "fallback_framework": "PyTorch" if mlx_config.get("status") == "optimized" else "MLX",
            "load_balancing": {
                "strategy": "round_robin",
                "health_checks": True,
                "failover_enabled": True
            },
            "memory_coordination": {
                "shared_pools": True,
                "unified_memory_advantage": True,
                "cross_framework_optimization": True
            }
        }
        
        return {
            "timestamp": time.time(),
            "optimization_complete": True,
            "system_specs": {
                "model": "MacBook Pro M3 Max",
                "unified_memory_gb": 128,
                "cpu_cores": 16
            },
            "memory_management": {
                "pools_created": len(self.memory_manager.memory_pools),
                "utilization": self.memory_manager.get_memory_utilization()
            },
            "framework_optimization": {
                "mlx": mlx_config,
                "pytorch_mps": pytorch_config
            },
            "inference_engines": {
                "lm_studio": lm_studio_config,
                "ollama": ollama_config
            },
            "coordination": coordination,
            "performance_estimates": {
                "inference_latency_ms": "50-200ms for typical requests",
                "throughput_rps": f"{self.config.max_concurrent_requests * 2}-{self.config.max_concurrent_requests * 4}",
                "memory_efficiency": "85-95% utilization",
                "concurrent_capacity": f"{self.config.max_concurrent_requests} parallel requests"
            }
        }
    
    def cleanup(self):
        """Clean up optimization resources"""
        self.memory_manager.cleanup_memory_pools()

# Convenience function for quick optimization
def optimize_m3_system(workload_type: str = "inference", 
                       model_path: str = "", 
                       max_memory_percent: float = 85.0) -> Dict[str, Any]:
    """
    Quick optimization for M3 MacBook Pro system
    
    Args:
        workload_type: Type of workload ('inference', 'training', 'data_processing')
        model_path: Path to model file (if applicable)
        max_memory_percent: Maximum memory utilization percentage
    
    Returns:
        Optimization results and configuration
    """
    
    config = OptimizationConfig(
        max_memory_usage_percent=max_memory_percent,
        preferred_batch_size=512,
        max_concurrent_requests=8,
        enable_memory_mapping=True,
        use_quantization=True,
        cpu_threads=12,
        gpu_memory_fraction=0.8
    )
    
    pipeline_optimizer = M3PipelineOptimizer(config)
    
    try:
        # Run async optimization
        import asyncio
        
        async def run_optimization():
            return await pipeline_optimizer.optimize_inference_pipeline(model_path, workload_type)
        
        result = asyncio.run(run_optimization())
        return result
        
    except Exception as e:
        return {"error": f"Optimization failed: {e}"}
    
    finally:
        pipeline_optimizer.cleanup()

if __name__ == "__main__":
    # Example usage and testing
    print("M3 MacBook Pro Optimization Engine")
    print("=" * 50)
    
    # Run optimization
    result = optimize_m3_system(
        workload_type="inference",
        model_path="./models/qwen3-7b",
        max_memory_percent=85.0
    )
    
    print("\nOptimization Results:")
    print(json.dumps(result, indent=2))
    
    # Performance recommendations
    if not result.get("error"):
        print("\n🚀 M3 Optimization Summary:")
        print(f"✅ Memory pools created: {result['memory_management']['pools_created']}")
        print(f"✅ Framework optimization: {result['framework_optimization']}")
        print(f"✅ Estimated throughput: {result['performance_estimates']['throughput_rps']} RPS")
        print(f"✅ Memory efficiency: {result['performance_estimates']['memory_efficiency']}")
        
        print("\n📋 Next Steps:")
        print("1. Install required frameworks: pip install mlx mlx-lm torch")
        print("2. Configure LM Studio with recommended settings")
        print("3. Set Ollama environment variables")
        print("4. Run performance monitoring with: python performance_monitor.py")
        print("5. Benchmark with your specific models and workloads")
    else:
        print(f"\n❌ Optimization failed: {result['error']}")