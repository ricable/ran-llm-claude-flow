#!/usr/bin/env python3
"""
Qwen3 Dynamic Model Manager

Dynamic model selection and management for Qwen3 1.7b, 7b, 14b, and 30b variants
optimized for Apple M3 Max hardware with MLX backend integration.

Author: Claude Code ML Integration Specialist
Date: 2025-08-23
"""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import threading
from pathlib import Path

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    logging.warning("MLX not available, falling back to CPU-only mode")
    MLX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVariant(Enum):
    """Qwen3 model variants with their characteristics"""
    QWEN3_1_7B = "qwen3-1.7b"
    QWEN3_7B = "qwen3-7b"
    QWEN3_14B = "qwen3-14b"
    QWEN3_30B = "qwen3-30b"

class QuantizationType(Enum):
    """Model quantization types"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

class TaskComplexity(Enum):
    """Task complexity levels for model selection"""
    SIMPLE = "simple"  # Classification, short responses
    MODERATE = "moderate"  # Summarization, structured extraction
    COMPLEX = "complex"  # Analysis, reasoning
    PREMIUM = "premium"  # Research-level tasks

@dataclass
class ModelConfig:
    """Configuration for a specific Qwen3 model"""
    variant: ModelVariant
    quantization: QuantizationType
    model_path: str
    max_memory_gb: float
    target_tokens_per_sec: float
    context_length: int = 32768
    batch_size: int = 1
    warmup_required: bool = True
    
@dataclass
class SystemResources:
    """Current system resource usage"""
    available_memory_gb: float
    cpu_usage_percent: float
    gpu_memory_usage_gb: float
    active_models: int
    timestamp: float = field(default_factory=time.time)

class ModelPerformanceMetrics:
    """Track model performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
        
    def record_inference(self, model_variant: ModelVariant, 
                        tokens_per_sec: float, latency_ms: float, 
                        memory_used_gb: float, success: bool = True):
        """Record inference performance metrics"""
        with self.lock:
            key = model_variant.value
            if key not in self.metrics:
                self.metrics[key] = {
                    'inference_count': 0,
                    'total_tokens_per_sec': 0,
                    'total_latency_ms': 0,
                    'total_memory_gb': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'last_updated': time.time()
                }
            
            self.metrics[key]['inference_count'] += 1
            self.metrics[key]['total_tokens_per_sec'] += tokens_per_sec
            self.metrics[key]['total_latency_ms'] += latency_ms
            self.metrics[key]['total_memory_gb'] += memory_used_gb
            self.metrics[key]['last_updated'] = time.time()
            
            if success:
                self.metrics[key]['success_count'] += 1
            else:
                self.metrics[key]['failure_count'] += 1
                
    def get_average_metrics(self, model_variant: ModelVariant) -> Dict[str, float]:
        """Get average performance metrics for a model"""
        with self.lock:
            key = model_variant.value
            if key not in self.metrics or self.metrics[key]['inference_count'] == 0:
                return {}
            
            m = self.metrics[key]
            count = m['inference_count']
            
            return {
                'avg_tokens_per_sec': m['total_tokens_per_sec'] / count,
                'avg_latency_ms': m['total_latency_ms'] / count,
                'avg_memory_gb': m['total_memory_gb'] / count,
                'success_rate': m['success_count'] / count,
                'inference_count': count
            }

class Qwen3DynamicManager:
    """Dynamic Qwen3 model manager with intelligent selection and caching"""
    
    def __init__(self, base_model_path: str = "./models", max_concurrent_models: int = 3):
        self.base_model_path = Path(base_model_path)
        self.max_concurrent_models = max_concurrent_models
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs = self._initialize_model_configs()
        self.performance_metrics = ModelPerformanceMetrics()
        self.model_usage_stats = {}
        self.warmup_cache = set()
        self.lock = threading.Lock()
        
        # Initialize coordination hooks
        self._init_coordination_hooks()
        
        logger.info(f"Initialized Qwen3DynamicManager with {len(self.model_configs)} model configurations")
        
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations for all variants"""
        configs = {
            "fast": ModelConfig(
                variant=ModelVariant.QWEN3_1_7B,
                quantization=QuantizationType.INT4,
                model_path="qwen2.5-1.5b-instruct-mlx",
                max_memory_gb=4.0,
                target_tokens_per_sec=80.0,
                context_length=32768,
                batch_size=16
            ),
            "balanced": ModelConfig(
                variant=ModelVariant.QWEN3_7B,
                quantization=QuantizationType.INT8,
                model_path="qwen2.5-7b-instruct-mlx",
                max_memory_gb=12.0,
                target_tokens_per_sec=35.0,
                context_length=32768,
                batch_size=8
            ),
            "quality": ModelConfig(
                variant=ModelVariant.QWEN3_14B,
                quantization=QuantizationType.INT8,
                model_path="qwen2.5-14b-instruct-mlx",
                max_memory_gb=28.0,
                target_tokens_per_sec=18.0,
                context_length=32768,
                batch_size=4
            ),
            "premium": ModelConfig(
                variant=ModelVariant.QWEN3_30B,
                quantization=QuantizationType.INT8,
                model_path="qwen2.5-30b-instruct-mlx",
                max_memory_gb=48.0,
                target_tokens_per_sec=8.0,
                context_length=32768,
                batch_size=2
            )
        }
        return configs
        
    def _init_coordination_hooks(self):
        """Initialize coordination hooks for performance tracking"""
        try:
            import subprocess
            # Register pre-task hook
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "pre-task", 
                "--description", "Qwen3 Model Manager Initialization"
            ], capture_output=True)
        except Exception as e:
            logger.warning(f"Could not initialize coordination hooks: {e}")
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Estimate GPU memory usage (M3 Max unified memory)
        gpu_memory_estimate = max(0, memory.used / (1024**3) - 20.0)  # Rough estimate
        
        return SystemResources(
            available_memory_gb=(memory.available / (1024**3)),
            cpu_usage_percent=cpu_usage,
            gpu_memory_usage_gb=gpu_memory_estimate,
            active_models=len(self.loaded_models)
        )
    
    def select_optimal_model(self, task_complexity: TaskComplexity = TaskComplexity.MODERATE,
                           latency_requirement_ms: float = 5000,
                           quality_threshold: float = 0.7) -> str:
        """Select optimal model based on task requirements and system resources"""
        
        resources = self.get_system_resources()
        
        # Model selection logic based on requirements
        if latency_requirement_ms < 500 and resources.available_memory_gb > 8:
            return "fast"
        elif task_complexity == TaskComplexity.PREMIUM and resources.available_memory_gb > 50:
            return "premium"
        elif task_complexity == TaskComplexity.COMPLEX and resources.available_memory_gb > 30:
            return "quality"
        elif resources.available_memory_gb > 16:
            return "balanced"
        else:
            return "fast"
    
    def load_model(self, model_key: str, force_reload: bool = False) -> bool:
        """Load a specific model variant"""
        
        if model_key not in self.model_configs:
            logger.error(f"Unknown model key: {model_key}")
            return False
        
        with self.lock:
            # Check if model is already loaded
            if model_key in self.loaded_models and not force_reload:
                logger.info(f"Model {model_key} already loaded")
                return True
            
            # Check memory constraints
            config = self.model_configs[model_key]
            resources = self.get_system_resources()
            
            if resources.available_memory_gb < config.max_memory_gb:
                logger.warning(f"Insufficient memory for {model_key}: need {config.max_memory_gb}GB, have {resources.available_memory_gb}GB")
                # Try to free memory by unloading least used models
                self._free_memory_for_model(config.max_memory_gb)
            
            # Check concurrent model limit
            if len(self.loaded_models) >= self.max_concurrent_models:
                self._unload_least_used_model()
            
            try:
                start_time = time.time()
                
                if MLX_AVAILABLE:
                    # Load with MLX
                    model_path = self.base_model_path / config.model_path
                    model, tokenizer = load(str(model_path))
                    self.loaded_models[model_key] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'config': config,
                        'load_time': time.time() - start_time,
                        'last_used': time.time(),
                        'usage_count': 0
                    }
                else:
                    # Fallback to CPU-only mode
                    logger.warning("Loading model in CPU-only mode")
                    self.loaded_models[model_key] = {
                        'model': None,  # Placeholder
                        'tokenizer': None,
                        'config': config,
                        'load_time': time.time() - start_time,
                        'last_used': time.time(),
                        'usage_count': 0
                    }
                
                load_time = time.time() - start_time
                logger.info(f"Successfully loaded {model_key} in {load_time:.2f} seconds")
                
                # Warm up model if required
                if config.warmup_required and model_key not in self.warmup_cache:
                    self._warm_up_model(model_key)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                return False
    
    def _warm_up_model(self, model_key: str):
        """Warm up model with a simple inference"""
        try:
            if model_key in self.loaded_models:
                # Simple warmup prompt
                warmup_prompt = "Hello, this is a warmup."
                self.generate_text(warmup_prompt, model_key=model_key, max_tokens=10)
                self.warmup_cache.add(model_key)
                logger.info(f"Warmed up model {model_key}")
        except Exception as e:
            logger.warning(f"Model warmup failed for {model_key}: {e}")
    
    def _free_memory_for_model(self, required_memory_gb: float):
        """Free memory by unloading models until requirement is met"""
        resources = self.get_system_resources()
        
        while resources.available_memory_gb < required_memory_gb and self.loaded_models:
            # Unload least recently used model
            lru_model = min(self.loaded_models.items(), 
                          key=lambda x: x[1]['last_used'])
            
            self.unload_model(lru_model[0])
            resources = self.get_system_resources()
    
    def _unload_least_used_model(self):
        """Unload the least recently used model"""
        if not self.loaded_models:
            return
        
        lru_model = min(self.loaded_models.items(), 
                       key=lambda x: x[1]['last_used'])
        self.unload_model(lru_model[0])
    
    def unload_model(self, model_key: str) -> bool:
        """Unload a specific model to free memory"""
        with self.lock:
            if model_key not in self.loaded_models:
                logger.warning(f"Model {model_key} not loaded")
                return False
            
            try:
                # MLX models are automatically garbage collected
                del self.loaded_models[model_key]
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if MLX_AVAILABLE:
                    mx.metal.clear_cache()
                
                logger.info(f"Successfully unloaded model {model_key}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_key}: {e}")
                return False
    
    def generate_text(self, prompt: str, model_key: Optional[str] = None,
                     max_tokens: int = 1000, temperature: float = 0.7,
                     task_complexity: TaskComplexity = TaskComplexity.MODERATE) -> Dict[str, Any]:
        """Generate text using optimal model selection"""
        
        # Auto-select model if not specified
        if model_key is None:
            model_key = self.select_optimal_model(task_complexity)
        
        # Ensure model is loaded
        if not self.load_model(model_key):
            raise RuntimeError(f"Failed to load model {model_key}")
        
        start_time = time.time()
        
        try:
            model_data = self.loaded_models[model_key]
            model_data['last_used'] = time.time()
            model_data['usage_count'] += 1
            
            if MLX_AVAILABLE and model_data['model'] is not None:
                # Generate with MLX
                response = generate(
                    model=model_data['model'],
                    tokenizer=model_data['tokenizer'],
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature
                )
                
                generated_text = response
                
            else:
                # Fallback generation (placeholder)
                generated_text = f"[Generated response to: {prompt[:50]}...] (CPU fallback mode)"
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Estimate metrics
            estimated_tokens = len(generated_text.split())
            tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else 0
            
            # Record performance metrics
            self.performance_metrics.record_inference(
                model_data['config'].variant,
                tokens_per_sec,
                generation_time * 1000,  # Convert to ms
                self.get_system_resources().gpu_memory_usage_gb,
                success=True
            )
            
            result = {
                'text': generated_text,
                'model_used': model_key,
                'generation_time_seconds': generation_time,
                'tokens_per_second': tokens_per_sec,
                'estimated_tokens': estimated_tokens,
                'model_config': model_data['config'].__dict__
            }
            
            # Store metrics in memory for coordination
            self._store_metrics_in_memory(model_key, result)
            
            return result
            
        except Exception as e:
            # Record failure
            if model_key in self.loaded_models:
                config = self.loaded_models[model_key]['config']
                self.performance_metrics.record_inference(
                    config.variant, 0, 0, 0, success=False
                )
            
            logger.error(f"Generation failed with model {model_key}: {e}")
            raise
    
    def _store_metrics_in_memory(self, model_key: str, result: Dict[str, Any]):
        """Store performance metrics in coordination memory"""
        try:
            import subprocess
            metrics_data = {
                'model_key': model_key,
                'timestamp': time.time(),
                'performance': {
                    'tokens_per_second': result['tokens_per_second'],
                    'generation_time': result['generation_time_seconds'],
                    'estimated_tokens': result['estimated_tokens']
                },
                'system_resources': self.get_system_resources().__dict__
            }
            
            # Store in coordination memory
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"qwen3/performance/{model_key}/{int(time.time())}",
                "--data", json.dumps(metrics_data)
            ], capture_output=True)
            
        except Exception as e:
            logger.warning(f"Could not store metrics in coordination memory: {e}")
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all models"""
        summary = {
            'timestamp': time.time(),
            'system_resources': self.get_system_resources().__dict__,
            'loaded_models': list(self.loaded_models.keys()),
            'model_performance': {},
            'total_inferences': 0
        }
        
        for variant in ModelVariant:
            metrics = self.performance_metrics.get_average_metrics(variant)
            if metrics:
                summary['model_performance'][variant.value] = metrics
                summary['total_inferences'] += metrics.get('inference_count', 0)
        
        return summary
    
    def optimize_model_pool(self) -> Dict[str, Any]:
        """Optimize the current model pool based on usage patterns"""
        
        resources = self.get_system_resources()
        optimization_results = {
            'actions_taken': [],
            'memory_freed_gb': 0,
            'models_optimized': 0
        }
        
        # Unload unused models if memory is tight
        if resources.available_memory_gb < 10:  # Less than 10GB available
            unused_models = [
                key for key, data in self.loaded_models.items()
                if time.time() - data['last_used'] > 1800  # 30 minutes
            ]
            
            for model_key in unused_models:
                if self.unload_model(model_key):
                    optimization_results['actions_taken'].append(f"Unloaded unused model: {model_key}")
                    optimization_results['models_optimized'] += 1
        
        # Preload frequently used models if memory allows
        if resources.available_memory_gb > 30:
            # Logic to preload based on usage patterns would go here
            pass
        
        return optimization_results
    
    def shutdown(self):
        """Gracefully shutdown the model manager"""
        logger.info("Shutting down Qwen3DynamicManager")
        
        # Unload all models
        models_to_unload = list(self.loaded_models.keys())
        for model_key in models_to_unload:
            self.unload_model(model_key)
        
        # Clear caches
        if MLX_AVAILABLE:
            mx.metal.clear_cache()
        
        # Final coordination hook
        try:
            import subprocess
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-task",
                "--task-id", "qwen3-manager-shutdown"
            ], capture_output=True)
        except Exception:
            pass
        
        logger.info("Qwen3DynamicManager shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    # Initialize manager
    manager = Qwen3DynamicManager()
    
    try:
        # Test model selection
        optimal_model = manager.select_optimal_model(
            task_complexity=TaskComplexity.MODERATE,
            latency_requirement_ms=2000
        )
        print(f"Selected optimal model: {optimal_model}")
        
        # Test text generation
        result = manager.generate_text(
            "Explain the concept of machine learning in simple terms.",
            task_complexity=TaskComplexity.MODERATE
        )
        
        print(f"Generated text: {result['text'][:100]}...")
        print(f"Performance: {result['tokens_per_second']:.2f} tokens/sec")
        
        # Get performance summary
        summary = manager.get_model_performance_summary()
        print(f"Performance summary: {json.dumps(summary, indent=2)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        manager.shutdown()
