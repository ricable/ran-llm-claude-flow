#!/usr/bin/env python3
"""
Qwen3 Model Variants Manager - M3 Max Optimized MLX Integration
Supports 1.7B, 4B, 8B variants with dynamic selection and MLX acceleration
"""

import json
import asyncio
import logging
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate, load_model, load_tokenizer
    from mlx_lm.utils import get_model_path
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️ MLX not available - falling back to CPU inference")

import requests
import torch
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSize(Enum):
    """Qwen3 model size variants"""
    SMALL = "1.7B"    # Fast inference, basic tasks
    MEDIUM = "4B"     # Balanced performance 
    LARGE = "8B"      # High quality, complex tasks

class InferenceBackend(Enum):
    """Supported inference backends"""
    MLX = "mlx"           # Apple Silicon optimized
    LMSTUDIO = "lmstudio" # LM Studio API
    OLLAMA = "ollama"     # Ollama GGUF
    TRANSFORMERS = "transformers"  # HuggingFace fallback

@dataclass
class ModelConfig:
    """Configuration for a specific model variant"""
    size: ModelSize
    backend: InferenceBackend
    model_path: str
    memory_gb: float
    max_tokens: int = 2048
    batch_size: int = 4
    quantization: Optional[str] = "4bit"
    temperature: float = 0.7

@dataclass
class ModelPerformance:
    """Performance metrics for model variants"""
    tokens_per_second: float
    memory_usage_gb: float
    latency_ms: float
    quality_score: float
    last_updated: float

class M3MaxMemoryManager:
    """Memory management optimized for M3 Max 128GB unified memory"""
    
    def __init__(self, total_memory_gb: int = 128):
        self.total_memory_gb = total_memory_gb
        self.memory_pools = {
            'models': int(total_memory_gb * 0.35),      # 45GB for model storage
            'processing': int(total_memory_gb * 0.25),   # 32GB for processing
            'cache': int(total_memory_gb * 0.15),       # 19GB for caching
            'system': int(total_memory_gb * 0.25)       # 32GB for system
        }
        self.allocated_memory = {}
        
    def allocate_model_memory(self, model_id: str, size_gb: float) -> bool:
        """Allocate memory for a model"""
        current_allocated = sum(self.allocated_memory.values())
        if current_allocated + size_gb <= self.memory_pools['models']:
            self.allocated_memory[model_id] = size_gb
            logger.info(f"🧠 Allocated {size_gb}GB for {model_id} (Total: {current_allocated + size_gb}GB)")
            return True
        return False
    
    def deallocate_model_memory(self, model_id: str) -> None:
        """Deallocate memory for a model"""
        if model_id in self.allocated_memory:
            freed = self.allocated_memory.pop(model_id)
            logger.info(f"🗑️ Freed {freed}GB from {model_id}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory allocation status"""
        allocated = sum(self.allocated_memory.values())
        available = self.memory_pools['models'] - allocated
        system_memory = psutil.virtual_memory()
        
        return {
            'total_system_gb': system_memory.total / (1024**3),
            'available_system_gb': system_memory.available / (1024**3),
            'allocated_models_gb': allocated,
            'available_models_gb': available,
            'utilization_percent': (allocated / self.memory_pools['models']) * 100,
            'loaded_models': list(self.allocated_memory.keys())
        }

class Qwen3ModelManager:
    """Advanced model manager for Qwen3 variants with MLX optimization"""
    
    def __init__(self):
        self.memory_manager = M3MaxMemoryManager()
        self.loaded_models = {}
        self.model_configs = self._initialize_model_configs()
        self.performance_metrics = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_mlx()
        
    def _initialize_mlx(self):
        """Initialize MLX for M3 Max optimization"""
        if MLX_AVAILABLE:
            # Configure MLX for M3 Max
            mx.metal.clear_cache()
            mx.set_default_device(mx.gpu)
            logger.info("🚀 MLX initialized for M3 Max GPU acceleration")
        else:
            logger.warning("⚠️ MLX not available, using CPU fallback")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize configurations for all model variants"""
        configs = {}
        
        # Qwen3-1.7B - Fast inference
        configs['qwen3-1.7b-mlx'] = ModelConfig(
            size=ModelSize.SMALL,
            backend=InferenceBackend.MLX,
            model_path="models/Qwen3-1.7B-Instruct-MLX",
            memory_gb=4.0,
            max_tokens=2048,
            batch_size=8,
            quantization="4bit"
        )
        
        configs['qwen3-1.7b-lmstudio'] = ModelConfig(
            size=ModelSize.SMALL,
            backend=InferenceBackend.LMSTUDIO,
            model_path="http://localhost:1234/v1/chat/completions",
            memory_gb=4.0,
            max_tokens=2048,
            batch_size=4
        )
        
        # Qwen3-4B - Balanced performance
        configs['qwen3-4b-mlx'] = ModelConfig(
            size=ModelSize.MEDIUM,
            backend=InferenceBackend.MLX,
            model_path="models/Qwen3-4B-Instruct-MLX",
            memory_gb=8.0,
            max_tokens=4096,
            batch_size=4,
            quantization="4bit"
        )
        
        configs['qwen3-4b-ollama'] = ModelConfig(
            size=ModelSize.MEDIUM,
            backend=InferenceBackend.OLLAMA,
            model_path="qwen3:4b",
            memory_gb=8.0,
            max_tokens=4096,
            batch_size=2
        )
        
        # Qwen3-8B - High quality
        configs['qwen3-8b-mlx'] = ModelConfig(
            size=ModelSize.LARGE,
            backend=InferenceBackend.MLX,
            model_path="models/Qwen3-8B-Instruct-MLX",
            memory_gb=16.0,
            max_tokens=8192,
            batch_size=2,
            quantization="4bit"
        )
        
        configs['qwen3-8b-lmstudio'] = ModelConfig(
            size=ModelSize.LARGE,
            backend=InferenceBackend.LMSTUDIO,
            model_path="http://localhost:1234/v1/chat/completions",
            memory_gb=16.0,
            max_tokens=8192,
            batch_size=1
        )
        
        return configs
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """Load a specific model variant"""
        if model_id in self.loaded_models and not force_reload:
            logger.info(f"✅ Model {model_id} already loaded")
            return True
            
        config = self.model_configs.get(model_id)
        if not config:
            logger.error(f"❌ Unknown model ID: {model_id}")
            return False
        
        # Check memory availability
        if not self.memory_manager.allocate_model_memory(model_id, config.memory_gb):
            logger.error(f"❌ Insufficient memory for {model_id} ({config.memory_gb}GB required)")
            return False
        
        try:
            start_time = time.time()
            
            if config.backend == InferenceBackend.MLX and MLX_AVAILABLE:
                model, tokenizer = await self._load_mlx_model(config)
            elif config.backend == InferenceBackend.LMSTUDIO:
                model, tokenizer = await self._load_lmstudio_model(config)
            elif config.backend == InferenceBackend.OLLAMA:
                model, tokenizer = await self._load_ollama_model(config)
            else:
                model, tokenizer = await self._load_transformers_model(config)
            
            load_time = time.time() - start_time
            
            self.loaded_models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'config': config,
                'load_time': load_time,
                'last_used': time.time()
            }
            
            # Initialize performance metrics
            self.performance_metrics[model_id] = ModelPerformance(
                tokens_per_second=0.0,
                memory_usage_gb=config.memory_gb,
                latency_ms=load_time * 1000,
                quality_score=0.0,
                last_updated=time.time()
            )
            
            logger.info(f"✅ Loaded {model_id} in {load_time:.2f}s using {config.backend.value}")
            return True
            
        except Exception as e:
            self.memory_manager.deallocate_model_memory(model_id)
            logger.error(f"❌ Failed to load {model_id}: {str(e)}")
            return False
    
    async def _load_mlx_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model using MLX framework"""
        try:
            model_path = Path(config.model_path)
            if not model_path.exists():
                # Try to download from HuggingFace
                model_path, _ = get_model_path(config.model_path)
            
            model, tokenizer = load_model(model_path)
            
            # Apply quantization if specified
            if config.quantization == "4bit":
                # MLX handles quantization automatically for 4-bit models
                logger.info(f"🔧 Applied {config.quantization} quantization")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ MLX model loading failed: {str(e)}")
            raise
    
    async def _load_lmstudio_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Configure LM Studio model connection"""
        # LM Studio uses API calls, no actual model loading
        # Return configuration for API calls
        return {
            'type': 'lmstudio_api',
            'base_url': config.model_path,
            'model_name': f"qwen3-{config.size.value.lower()}"
        }, None
    
    async def _load_ollama_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Configure Ollama model connection"""
        # Ollama uses API calls, verify model is available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if config.model_path in model_names:
                    logger.info(f"✅ Ollama model {config.model_path} available")
                    return {
                        'type': 'ollama_api',
                        'model_name': config.model_path,
                        'base_url': 'http://localhost:11434'
                    }, None
                else:
                    logger.warning(f"⚠️ Model {config.model_path} not found in Ollama")
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {str(e)}")
        
        return None, None
    
    async def _load_transformers_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model using HuggingFace Transformers (CPU fallback)"""
        try:
            model_name = f"Qwen/Qwen{config.size.value}-Chat"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
            
            logger.info(f"✅ Loaded {model_name} with Transformers")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Transformers model loading failed: {str(e)}")
            raise
    
    async def generate_text(self, model_id: str, prompt: str, **kwargs) -> str:
        """Generate text using the specified model"""
        if model_id not in self.loaded_models:
            logger.error(f"❌ Model {model_id} not loaded")
            return ""
        
        model_info = self.loaded_models[model_id]
        config = model_info['config']
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        start_time = time.time()
        
        try:
            if config.backend == InferenceBackend.MLX and MLX_AVAILABLE:
                result = await self._generate_mlx(model, tokenizer, prompt, config, **kwargs)
            elif config.backend == InferenceBackend.LMSTUDIO:
                result = await self._generate_lmstudio(model, prompt, config, **kwargs)
            elif config.backend == InferenceBackend.OLLAMA:
                result = await self._generate_ollama(model, prompt, config, **kwargs)
            else:
                result = await self._generate_transformers(model, tokenizer, prompt, config, **kwargs)
            
            # Update performance metrics
            generation_time = time.time() - start_time
            tokens = len(result.split()) * 1.3  # Rough token estimation
            
            self.performance_metrics[model_id].tokens_per_second = tokens / generation_time
            self.performance_metrics[model_id].latency_ms = generation_time * 1000
            self.performance_metrics[model_id].last_updated = time.time()
            
            # Update last used time
            model_info['last_used'] = time.time()
            
            logger.info(f"🎯 Generated {len(result)} chars in {generation_time:.2f}s with {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Text generation failed with {model_id}: {str(e)}")
            return ""
    
    async def _generate_mlx(self, model: Any, tokenizer: Any, prompt: str, config: ModelConfig, **kwargs) -> str:
        """Generate text using MLX"""
        max_tokens = kwargs.get('max_tokens', config.max_tokens)
        temperature = kwargs.get('temperature', config.temperature)
        
        # Format prompt for Qwen3
        formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False
        )
        
        # Extract just the assistant response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
            
        return response.strip()
    
    async def _generate_lmstudio(self, model_config: Dict, prompt: str, config: ModelConfig, **kwargs) -> str:
        """Generate text using LM Studio API"""
        import aiohttp
        
        payload = {
            "model": model_config['model_name'],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get('temperature', config.temperature),
            "max_tokens": kwargs.get('max_tokens', config.max_tokens),
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model_config['base_url'], json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"❌ LM Studio API error: {response.status}")
                    return ""
    
    async def _generate_ollama(self, model_config: Dict, prompt: str, config: ModelConfig, **kwargs) -> str:
        """Generate text using Ollama API"""
        import aiohttp
        
        payload = {
            "model": model_config['model_name'],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', config.temperature),
                "num_predict": kwargs.get('max_tokens', config.max_tokens)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{model_config['base_url']}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '')
                else:
                    logger.error(f"❌ Ollama API error: {response.status}")
                    return ""
    
    async def _generate_transformers(self, model: Any, tokenizer: Any, prompt: str, config: ModelConfig, **kwargs) -> str:
        """Generate text using Transformers (CPU fallback)"""
        inputs = tokenizer(prompt, return_tensors="pt")
        max_tokens = kwargs.get('max_tokens', config.max_tokens)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_tokens,
                temperature=kwargs.get('temperature', config.temperature),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from response
        response = response[len(prompt):].strip()
        
        return response
    
    def get_optimal_model(self, task_complexity: float = 0.5, max_latency_ms: float = 5000) -> Optional[str]:
        """Select optimal model based on task complexity and latency requirements"""
        candidates = []
        
        for model_id, config in self.model_configs.items():
            # Skip if model requires too much memory
            memory_status = self.memory_manager.get_memory_status()
            if config.memory_gb > memory_status['available_models_gb'] and model_id not in self.loaded_models:
                continue
            
            # Get performance metrics if available
            if model_id in self.performance_metrics:
                perf = self.performance_metrics[model_id]
                if perf.latency_ms > max_latency_ms:
                    continue
                
                # Score based on complexity matching and performance
                complexity_match = 1.0 - abs(self._get_model_complexity(config.size) - task_complexity)
                performance_score = min(perf.tokens_per_second / 100.0, 1.0)  # Normalize to 0-1
                latency_score = max(0, 1.0 - (perf.latency_ms / max_latency_ms))
                
                total_score = (complexity_match * 0.4 + performance_score * 0.3 + latency_score * 0.3)
                candidates.append((model_id, total_score))
            else:
                # No performance metrics yet, use model size as proxy
                complexity_match = 1.0 - abs(self._get_model_complexity(config.size) - task_complexity)
                candidates.append((model_id, complexity_match))
        
        if not candidates:
            logger.warning("⚠️ No suitable models available")
            return None
        
        # Return the best scoring model
        best_model = max(candidates, key=lambda x: x[1])
        logger.info(f"🎯 Selected optimal model: {best_model[0]} (score: {best_model[1]:.3f})")
        return best_model[0]
    
    def _get_model_complexity(self, size: ModelSize) -> float:
        """Convert model size to complexity score (0-1)"""
        complexity_map = {
            ModelSize.SMALL: 0.3,   # Simple tasks
            ModelSize.MEDIUM: 0.6,  # Balanced tasks  
            ModelSize.LARGE: 0.9    # Complex tasks
        }
        return complexity_map.get(size, 0.5)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'memory_status': self.memory_manager.get_memory_status(),
            'loaded_models': list(self.loaded_models.keys()),
            'available_configs': list(self.model_configs.keys()),
            'performance_metrics': {
                model_id: {
                    'tokens_per_second': perf.tokens_per_second,
                    'memory_usage_gb': perf.memory_usage_gb,
                    'latency_ms': perf.latency_ms,
                    'quality_score': perf.quality_score
                }
                for model_id, perf in self.performance_metrics.items()
            },
            'mlx_available': MLX_AVAILABLE,
            'system_health': self._check_system_health()
        }
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_usage_percent': memory.percent,
            'cpu_usage_percent': cpu_percent,
            'healthy': memory.percent < 90 and cpu_percent < 80,
            'warnings': self._get_health_warnings(memory.percent, cpu_percent)
        }
    
    def _get_health_warnings(self, memory_percent: float, cpu_percent: float) -> List[str]:
        """Generate health warnings"""
        warnings = []
        if memory_percent > 90:
            warnings.append("High memory usage - consider unloading unused models")
        if cpu_percent > 80:
            warnings.append("High CPU usage - system may be under stress")
        return warnings
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model to free memory"""
        if model_id not in self.loaded_models:
            logger.warning(f"⚠️ Model {model_id} not loaded")
            return False
        
        try:
            # Clean up model resources
            del self.loaded_models[model_id]
            self.memory_manager.deallocate_model_memory(model_id)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if MLX_AVAILABLE:
                mx.metal.clear_cache()
            
            logger.info(f"🗑️ Unloaded model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload {model_id}: {str(e)}")
            return False
    
    async def cleanup_unused_models(self, max_idle_minutes: int = 30) -> List[str]:
        """Cleanup models that haven't been used recently"""
        current_time = time.time()
        unloaded = []
        
        for model_id, model_info in list(self.loaded_models.items()):
            idle_time = (current_time - model_info['last_used']) / 60  # Convert to minutes
            
            if idle_time > max_idle_minutes:
                if await self.unload_model(model_id):
                    unloaded.append(model_id)
        
        if unloaded:
            logger.info(f"🧹 Cleaned up {len(unloaded)} unused models: {unloaded}")
        
        return unloaded

# Example usage and testing
async def main():
    """Example usage of Qwen3ModelManager"""
    manager = Qwen3ModelManager()
    
    # Print system status
    status = manager.get_system_status()
    print("🖥️ System Status:")
    print(json.dumps(status, indent=2))
    
    # Load a model based on task complexity
    optimal_model = manager.get_optimal_model(task_complexity=0.3, max_latency_ms=3000)
    if optimal_model:
        success = await manager.load_model(optimal_model)
        if success:
            # Test generation
            prompt = "What are the key features of 5G NR technology?"
            response = await manager.generate_text(optimal_model, prompt, max_tokens=512)
            print(f"🎯 Response from {optimal_model}:")
            print(response)
    
    # Clean up
    await manager.cleanup_unused_models(max_idle_minutes=1)

if __name__ == "__main__":
    asyncio.run(main())