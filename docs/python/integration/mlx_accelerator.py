#!/usr/bin/env python3
"""
Apple MLX Direct Model Integration for Maximum Performance
Optimized Qwen3 deployment with MLX framework for M3 Max hardware.

Features:
- Direct MLX model loading and inference
- Memory-efficient model management
- Quantization optimization for M3 Max
- Concurrent model serving
- GPU acceleration with unified memory
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
from pathlib import Path
import psutil
from threading import RLock, Event
import numpy as np

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate, load_model, load_tokenizer, convert
    from mlx_lm.utils import get_model_path
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class MLXModelConfig:
    """MLX model configuration"""
    name: str
    path: str
    variant: str  # 1.7b, 7b, 14b, 30b
    quantization: str  # 4bit, 8bit, 16bit, fp32
    memory_gb: float
    context_length: int
    use_cases: List[str]
    precision: str = "float16"  # float16, float32, int8, int4
    batch_size: int = 1
    expected_tokens_per_second: float = 50.0
    
    @property
    def display_name(self) -> str:
        return f"qwen3-{self.variant}-{self.quantization}"


@dataclass
class MLXInferenceRequest:
    """MLX inference request"""
    id: str
    prompt: str
    model_variant: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stream: bool = False
    priority: str = "normal"
    timeout: float = 60.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            content_hash = hashlib.sha256(f"{self.prompt}{time.time()}".encode()).hexdigest()[:12]
            self.id = f"mlx_{content_hash}"


@dataclass
class MLXInferenceResponse:
    """MLX inference response"""
    request_id: str
    text: str
    model_used: str
    processing_time: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage_gb: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MLXModelRegistry:
    """Registry of optimized MLX model configurations for M3 Max"""
    
    def __init__(self):
        self.models = {
            # Fast inference models
            "qwen3-1.7b-4bit": MLXModelConfig(
                name="Qwen3-1.7B-4bit",
                path="/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-4bit",
                variant="1.7b",
                quantization="4bit",
                memory_gb=2.0,
                context_length=32768,
                use_cases=["classification", "quick_qa", "real_time"],
                precision="int4",
                batch_size=4,
                expected_tokens_per_second=120.0
            ),
            
            # Balanced performance models
            "qwen3-7b-8bit": MLXModelConfig(
                name="Qwen3-7B-8bit",
                path="/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-7B-MLX-8bit",
                variant="7b",
                quantization="8bit",
                memory_gb=8.0,
                context_length=32768,
                use_cases=["general_qa", "summarization", "analysis"],
                precision="int8",
                batch_size=2,
                expected_tokens_per_second=45.0
            ),
            
            # High quality models
            "qwen3-14b-8bit": MLXModelConfig(
                name="Qwen3-14B-8bit", 
                path="/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-14B-MLX-8bit",
                variant="14b",
                quantization="8bit",
                memory_gb=16.0,
                context_length=32768,
                use_cases=["complex_reasoning", "technical_writing", "code_generation"],
                precision="int8",
                batch_size=1,
                expected_tokens_per_second=25.0
            ),
            
            # Maximum quality model (if available memory allows)
            "qwen3-30b-16bit": MLXModelConfig(
                name="Qwen3-30B-16bit",
                path="/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-30B-MLX-16bit",
                variant="30b", 
                quantization="16bit",
                memory_gb=32.0,
                context_length=32768,
                use_cases=["research", "complex_analysis", "expert_reasoning"],
                precision="float16",
                batch_size=1,
                expected_tokens_per_second=12.0
            )
        }
    
    def get_model_by_variant(self, variant: str) -> Optional[MLXModelConfig]:
        """Get model configuration by variant"""
        for config in self.models.values():
            if config.variant == variant:
                return config
        return None
    
    def get_model_by_use_case(self, use_case: str) -> Optional[MLXModelConfig]:
        """Get optimal model for use case"""
        for config in self.models.values():
            if use_case in config.use_cases:
                return config
        return None
    
    def get_models_by_memory_constraint(self, max_memory_gb: float) -> List[MLXModelConfig]:
        """Get models that fit within memory constraint"""
        return [
            config for config in self.models.values() 
            if config.memory_gb <= max_memory_gb
        ]
    
    def get_all_models(self) -> List[MLXModelConfig]:
        """Get all model configurations"""
        return list(self.models.values())


class MLXModelManager:
    """Manage MLX model lifecycle and optimization"""
    
    def __init__(self):
        self.registry = MLXModelRegistry()
        self.loaded_models = {}  # model_name -> model_info
        self.model_stats = defaultdict(lambda: deque(maxlen=100))
        self.memory_tracker = {}
        self.lock = RLock()
        self.preload_queue = asyncio.Queue()
        self.preloader_task = None
        self.preloader_active = False
        
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - MLX acceleration disabled")
    
    async def initialize(self) -> bool:
        """Initialize MLX model manager"""
        if not MLX_AVAILABLE:
            logger.error("MLX not available")
            return False
        
        try:
            # Check MLX device availability
            logger.info(f"MLX device: {mx.default_device()}")
            logger.info(f"MLX memory limit: {mx.metal.get_memory_limit() / (1024**3):.1f} GB")
            
            # Start model preloader
            self.preloader_active = True
            self.preloader_task = asyncio.create_task(self._preloader_loop())
            
            # Determine which models to preload based on available memory
            await self._auto_preload_models()
            
            logger.info("MLX Model Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MLX Model Manager initialization failed: {e}")
            return False
    
    async def _auto_preload_models(self):
        """Automatically preload models based on available memory"""
        # Get available system memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Reserve memory for system (20GB) and processing buffer (10GB)
        usable_memory_gb = max(0, available_memory_gb - 30)
        
        # Get models that fit in available memory
        suitable_models = self.registry.get_models_by_memory_constraint(usable_memory_gb)
        
        # Sort by expected performance (tokens per second)
        suitable_models.sort(key=lambda x: x.expected_tokens_per_second, reverse=True)
        
        # Preload models starting with fastest
        models_to_load = []
        total_memory = 0
        
        for model_config in suitable_models:
            if total_memory + model_config.memory_gb <= usable_memory_gb:
                models_to_load.append(model_config)
                total_memory += model_config.memory_gb
        
        logger.info(f"Auto-preloading {len(models_to_load)} models ({total_memory:.1f}GB)")
        
        # Queue for preloading
        for model_config in models_to_load:
            await self.preload_queue.put(model_config)
    
    async def _preloader_loop(self):
        """Background model preloader"""
        while self.preloader_active:
            try:
                # Get model to preload
                model_config = await asyncio.wait_for(
                    self.preload_queue.get(),
                    timeout=1.0
                )
                
                # Load model in thread executor to avoid blocking
                logger.info(f"Preloading MLX model: {model_config.display_name}")
                
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    None,
                    self._load_model_sync,
                    model_config
                )
                
                if success:
                    logger.info(f"Successfully preloaded: {model_config.display_name}")
                else:
                    logger.error(f"Failed to preload: {model_config.display_name}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Preloader error: {e}")
                await asyncio.sleep(5)
    
    def _load_model_sync(self, model_config: MLXModelConfig) -> bool:
        """Synchronously load MLX model"""
        try:
            start_time = time.time()
            
            # Check if path exists
            if not Path(model_config.path).exists():
                logger.warning(f"Model path does not exist: {model_config.path}")
                return False
            
            # Load model and tokenizer
            model, tokenizer = load(model_config.path)
            
            # Apply optimizations for M3 Max
            model = self._optimize_model_for_m3_max(model, model_config)
            
            load_time = time.time() - start_time
            
            # Calculate memory usage
            memory_usage = self._estimate_model_memory_usage(model)
            
            with self.lock:
                self.loaded_models[model_config.display_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": model_config,
                    "load_time": load_time,
                    "memory_usage_gb": memory_usage,
                    "loaded_at": time.time(),
                    "usage_count": 0
                }
                
                self.memory_tracker[model_config.display_name] = memory_usage
            
            logger.info(f"Loaded {model_config.display_name} in {load_time:.1f}s ({memory_usage:.1f}GB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_config.display_name}: {e}")
            return False
    
    def _optimize_model_for_m3_max(self, model, config: MLXModelConfig):
        """Apply M3 Max specific optimizations"""
        try:
            # Enable unified memory optimizations
            if hasattr(mx, 'set_memory_limit'):
                # Set appropriate memory limit
                memory_limit = int(config.memory_gb * 1.2 * (1024**3))  # 20% buffer
                mx.set_memory_limit(memory_limit)
            
            # Apply quantization if specified
            if config.precision == "int4" and hasattr(mx, 'quantize'):
                model = mx.quantize(model, group_size=64, bits=4)
            elif config.precision == "int8" and hasattr(mx, 'quantize'):
                model = mx.quantize(model, group_size=32, bits=8)
            
            # Enable graph optimization
            if hasattr(model, 'update'):
                mx.eval(model.parameters())
            
            return model
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            return model
    
    def _estimate_model_memory_usage(self, model) -> float:
        """Estimate model memory usage in GB"""
        try:
            # Count parameters
            total_params = sum(p.size for p in mx.tree_flatten(model.parameters())[0])
            
            # Estimate memory (rough calculation)
            # Assume 2 bytes per parameter for int8, 1 byte for int4, 4 bytes for fp32
            bytes_per_param = 2  # Default assumption
            total_memory_bytes = total_params * bytes_per_param
            
            return total_memory_bytes / (1024**3)  # Convert to GB
            
        except Exception:
            return 4.0  # Default estimate
    
    async def get_model(self, model_variant: Optional[str] = None, use_case: Optional[str] = None) -> Optional[Tuple[Any, Any, MLXModelConfig]]:
        """Get model, tokenizer, and config"""
        
        # Find best matching model
        model_config = None
        
        if model_variant:
            model_config = self.registry.get_model_by_variant(model_variant)
        elif use_case:
            model_config = self.registry.get_model_by_use_case(use_case)
        
        # Fallback to any available model
        if not model_config:
            with self.lock:
                if self.loaded_models:
                    model_info = list(self.loaded_models.values())[0]
                    model_config = model_info["config"]
        
        if not model_config:
            return None
        
        # Load model if not already loaded
        if model_config.display_name not in self.loaded_models:
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_model_sync,
                model_config
            )
            if not success:
                return None
        
        with self.lock:
            if model_config.display_name in self.loaded_models:
                model_info = self.loaded_models[model_config.display_name]
                model_info["usage_count"] += 1
                
                return (
                    model_info["model"],
                    model_info["tokenizer"],
                    model_info["config"]
                )
        
        return None
    
    def record_performance(self, model_name: str, processing_time: float, tokens: int, success: bool, memory_usage: float = 0):
        """Record model performance metrics"""
        with self.lock:
            self.model_stats[model_name].append({
                "processing_time": processing_time,
                "tokens": tokens,
                "tokens_per_second": tokens / processing_time if processing_time > 0 else 0,
                "success": success,
                "memory_usage_gb": memory_usage,
                "timestamp": time.time()
            })
    
    def get_model_stats(self, model_name: str) -> Dict[str, float]:
        """Get performance statistics for model"""
        with self.lock:
            if model_name not in self.model_stats:
                return {}
            
            stats = list(self.model_stats[model_name])
            if not stats:
                return {}
            
            processing_times = [s["processing_time"] for s in stats]
            tokens_per_sec = [s["tokens_per_second"] for s in stats if s["tokens_per_second"] > 0]
            successes = [s["success"] for s in stats]
            memory_usage = [s["memory_usage_gb"] for s in stats if s["memory_usage_gb"] > 0]
            
            return {
                "avg_processing_time": np.mean(processing_times),
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "success_rate": np.mean(successes),
                "avg_memory_usage_gb": np.mean(memory_usage) if memory_usage else 0,
                "total_requests": len(stats)
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        total_memory = sum(self.memory_tracker.values())
        
        return {
            "total_model_memory_gb": total_memory,
            "individual_models": self.memory_tracker.copy(),
            "system_memory_gb": psutil.virtual_memory().used / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
    
    async def cleanup(self):
        """Cleanup model manager"""
        logger.info("Cleaning up MLX Model Manager...")
        
        self.preloader_active = False
        if self.preloader_task:
            self.preloader_task.cancel()
            try:
                await self.preloader_task
            except asyncio.CancelledError:
                pass
        
        with self.lock:
            self.loaded_models.clear()
            self.memory_tracker.clear()
        
        # Clear MLX memory cache
        if MLX_AVAILABLE and hasattr(mx, 'metal'):
            mx.metal.clear_cache()
        
        logger.info("MLX Model Manager cleanup completed")


class MLXInferenceEngine:
    """High-performance inference engine using MLX"""
    
    def __init__(self, model_manager: MLXModelManager, max_concurrent: int = 4):
        self.model_manager = model_manager
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.request_semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = {}
        self.request_history = deque(maxlen=1000)
        self.lock = RLock()
    
    async def generate(self, request: MLXInferenceRequest) -> MLXInferenceResponse:
        """Generate response using MLX"""
        
        # Acquire semaphore for concurrency control
        async with self.request_semaphore:
            start_time = time.time()
            
            try:
                # Get model, tokenizer, and config
                model_info = await self.model_manager.get_model(
                    model_variant=request.model_variant,
                    use_case=self._infer_use_case(request.prompt)
                )
                
                if not model_info:
                    raise Exception("No suitable MLX model available")
                
                model, tokenizer, config = model_info
                
                # Run inference in thread executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._generate_sync,
                    model,
                    tokenizer,
                    config,
                    request
                )
                
                processing_time = time.time() - start_time
                
                # Create response
                response = MLXInferenceResponse(
                    request_id=request.id,
                    text=result.get("text", ""),
                    model_used=config.display_name,
                    processing_time=processing_time,
                    tokens_generated=result.get("tokens", 0),
                    tokens_per_second=result.get("tokens", 0) / processing_time if processing_time > 0 else 0,
                    memory_usage_gb=result.get("memory_usage", 0),
                    metadata={
                        "model_config": asdict(config),
                        "prompt_length": len(request.prompt),
                        "generation_params": {
                            "temperature": request.temperature,
                            "top_p": request.top_p,
                            "max_tokens": request.max_tokens
                        }
                    }
                )
                
                # Record performance
                self.model_manager.record_performance(
                    config.display_name,
                    processing_time,
                    response.tokens_generated,
                    True,
                    response.memory_usage_gb
                )
                
                # Store in history
                with self.lock:
                    self.request_history.append(response)
                
                return response
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"MLX generation failed: {e}")
                
                response = MLXInferenceResponse(
                    request_id=request.id,
                    text="",
                    model_used="unknown",
                    processing_time=processing_time,
                    tokens_generated=0,
                    tokens_per_second=0,
                    memory_usage_gb=0,
                    success=False,
                    error=str(e)
                )
                
                # Record failure
                if 'config' in locals():
                    self.model_manager.record_performance(
                        config.display_name,
                        processing_time,
                        0,
                        False
                    )
                
                return response
    
    def _generate_sync(self, model, tokenizer, config: MLXModelConfig, request: MLXInferenceRequest) -> Dict[str, Any]:
        """Synchronous generation using MLX"""
        try:
            start_memory = self._get_memory_usage_gb()
            
            # Generate response
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temp=request.temperature,
                repetition_penalty=request.repetition_penalty,
                verbose=False
            )
            
            end_memory = self._get_memory_usage_gb()
            memory_delta = max(0, end_memory - start_memory)
            
            # Count tokens (rough approximation)
            tokens_generated = len(response.split())
            
            return {
                "text": response,
                "tokens": tokens_generated,
                "memory_usage": memory_delta
            }
            
        except Exception as e:
            logger.error(f"Sync generation failed: {e}")
            return {
                "text": "",
                "tokens": 0,
                "memory_usage": 0
            }
    
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            if MLX_AVAILABLE and hasattr(mx, 'metal'):
                return mx.metal.get_active_memory() / (1024**3)
            else:
                return psutil.Process().memory_info().rss / (1024**3)
        except:
            return 0.0
    
    def _infer_use_case(self, prompt: str) -> str:
        """Infer use case from prompt"""
        prompt_lower = prompt.lower()
        
        # Technical/complex prompts
        if any(word in prompt_lower for word in [
            "analyze", "explain complex", "technical", "detailed analysis",
            "comprehensive", "research", "algorithm", "architecture"
        ]):
            return "complex_reasoning"
        
        # Quick Q&A
        if any(word in prompt_lower for word in [
            "what is", "define", "quick", "brief", "summary"
        ]) and len(prompt) < 200:
            return "quick_qa"
        
        # Code generation
        if any(word in prompt_lower for word in [
            "code", "function", "class", "implement", "programming"
        ]):
            return "code_generation"
        
        # Default to general Q&A
        return "general_qa"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference engine performance statistics"""
        with self.lock:
            if not self.request_history:
                return {}
            
            recent_requests = [r for r in self.request_history if time.time() - r.metadata.get("timestamp", time.time()) < 300]
            
            if not recent_requests:
                return {}
            
            processing_times = [r.processing_time for r in recent_requests]
            tokens_per_sec = [r.tokens_per_second for r in recent_requests if r.tokens_per_second > 0]
            successes = [r.success for r in recent_requests]
            memory_usage = [r.memory_usage_gb for r in recent_requests if r.memory_usage_gb > 0]
            
            return {
                "requests_per_minute": len(recent_requests) / 5.0,  # Last 5 minutes
                "avg_processing_time": np.mean(processing_times),
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "success_rate": np.mean(successes),
                "avg_memory_usage_gb": np.mean(memory_usage) if memory_usage else 0,
                "concurrent_capacity": self.max_concurrent,
                "active_requests": len(self.active_requests)
            }
    
    async def cleanup(self):
        """Cleanup inference engine"""
        logger.info("Cleaning up MLX Inference Engine...")
        
        # Wait for active requests to complete (with timeout)
        try:
            if self.executor:
                self.executor.shutdown(wait=True, timeout=30)
        except Exception as e:
            logger.warning(f"Executor shutdown warning: {e}")
        
        with self.lock:
            self.active_requests.clear()
            self.request_history.clear()
        
        logger.info("MLX Inference Engine cleanup completed")


class MLXAccelerator:
    """Main MLX accelerator with coordinated components"""
    
    def __init__(self, max_concurrent_requests: int = 4):
        self.model_manager = MLXModelManager()
        self.inference_engine = MLXInferenceEngine(self.model_manager, max_concurrent_requests)
        self.initialized = False
        self.performance_monitor_task = None
        self.monitoring_active = False
    
    async def initialize(self) -> bool:
        """Initialize MLX accelerator"""
        if not MLX_AVAILABLE:
            logger.error("MLX not available - cannot initialize accelerator")
            return False
        
        try:
            # Initialize model manager
            if not await self.model_manager.initialize():
                return False
            
            # Start performance monitoring
            self.monitoring_active = True
            self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
            
            self.initialized = True
            logger.info("MLX Accelerator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"MLX Accelerator initialization failed: {e}")
            return False
    
    async def generate(self, request: MLXInferenceRequest) -> MLXInferenceResponse:
        """Generate response via MLX inference engine"""
        if not self.initialized:
            raise Exception("MLX Accelerator not initialized")
        
        return await self.inference_engine.generate(request)
    
    async def _performance_monitor_loop(self):
        """Background performance monitoring"""
        while self.monitoring_active:
            try:
                # Log performance statistics periodically
                memory_stats = self.model_manager.get_memory_usage()
                performance_stats = self.inference_engine.get_performance_stats()
                
                if performance_stats:
                    logger.info(
                        f"MLX Performance: "
                        f"{performance_stats['avg_tokens_per_second']:.1f} tok/s, "
                        f"{performance_stats['success_rate']:.1%} success, "
                        f"{memory_stats['total_model_memory_gb']:.1f}GB models"
                    )
                
                await asyncio.sleep(60)  # Log every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get accelerator status"""
        return {
            "initialized": self.initialized,
            "mlx_available": MLX_AVAILABLE,
            "loaded_models": list(self.model_manager.loaded_models.keys()),
            "memory_usage": self.model_manager.get_memory_usage(),
            "performance": self.inference_engine.get_performance_stats(),
            "model_stats": {
                model: self.model_manager.get_model_stats(model)
                for model in self.model_manager.loaded_models.keys()
            }
        }
    
    async def cleanup(self):
        """Cleanup MLX accelerator"""
        logger.info("Cleaning up MLX Accelerator...")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.performance_monitor_task:
            self.performance_monitor_task.cancel()
            try:
                await self.performance_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup components
        await self.inference_engine.cleanup()
        await self.model_manager.cleanup()
        
        self.initialized = False
        logger.info("MLX Accelerator cleanup completed")


# Example usage and testing
async def test_mlx_accelerator():
    """Test MLX accelerator functionality"""
    accelerator = MLXAccelerator(max_concurrent_requests=2)
    
    try:
        # Initialize
        if not await accelerator.initialize():
            print("Failed to initialize MLX accelerator")
            return
        
        # Test requests
        test_requests = [
            MLXInferenceRequest(
                id="test_1",
                prompt="What is 5G NR technology?",
                model_variant="1.7b",
                max_tokens=200,
                temperature=0.7
            ),
            MLXInferenceRequest(
                id="test_2", 
                prompt="Explain the technical details of MIMO technology in telecommunications",
                model_variant="7b",
                max_tokens=400,
                temperature=0.5
            ),
            MLXInferenceRequest(
                id="test_3",
                prompt="Write a Python function to calculate signal-to-noise ratio",
                model_variant="14b", 
                max_tokens=300,
                temperature=0.3
            )
        ]
        
        # Process requests concurrently
        start_time = time.time()
        tasks = [accelerator.generate(request) for request in test_requests]
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Display results
        for i, response in enumerate(responses):
            print(f"\nRequest {i+1}: {response.success}")
            print(f"Model: {response.model_used}")
            print(f"Processing time: {response.processing_time:.2f}s")
            print(f"Tokens/sec: {response.tokens_per_second:.1f}")
            print(f"Memory usage: {response.memory_usage_gb:.2f}GB")
            print(f"Response: {response.text[:150]}...")
        
        print(f"\nTotal concurrent processing time: {total_time:.2f}s")
        
        # Get status
        status = accelerator.get_status()
        print(f"\nAccelerator Status: {json.dumps(status, indent=2, default=str)}")
        
    finally:
        await accelerator.cleanup()


if __name__ == "__main__":
    if not MLX_AVAILABLE:
        print("MLX not available - please install mlx-lm package")
    else:
        asyncio.run(test_mlx_accelerator())