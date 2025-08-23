# Local Inference Setup: LM Studio & Ollama Integration

## Executive Summary

Comprehensive setup guide for local Qwen3 inference using LM Studio and Ollama, optimized for M3 Max hardware with MLX backend integration and high-performance API endpoints.

## LM Studio Configuration

### Installation and Initial Setup
```bash
# Install LM Studio (if not already installed)
brew install --cask lm-studio

# Download optimized Qwen3 models
# Via LM Studio GUI or CLI
lms download huggingface/Qwen/Qwen2.5-1.5B-Instruct-MLX
lms download huggingface/Qwen/Qwen2.5-7B-Instruct-MLX
lms download huggingface/Qwen/Qwen2.5-14B-Instruct-MLX
lms download huggingface/Qwen/Qwen2.5-32B-Instruct-MLX
```

### LM Studio Configuration File
```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 1234,
    "cors_allow_origin": "*",
    "request_timeout": 300,
    "max_concurrent_requests": 8
  },
  "models": {
    "default_model": "qwen2.5-7b-instruct-mlx",
    "model_library": {
      "qwen3-fast": {
        "path": "models/Qwen2.5-1.5B-Instruct-MLX",
        "context_length": 32768,
        "batch_size": 16,
        "gpu_layers": -1,
        "quantization": "q4_0"
      },
      "qwen3-balanced": {
        "path": "models/Qwen2.5-7B-Instruct-MLX",
        "context_length": 32768,
        "batch_size": 8,
        "gpu_layers": -1,
        "quantization": "q8_0"
      },
      "qwen3-quality": {
        "path": "models/Qwen2.5-14B-Instruct-MLX",
        "context_length": 32768,
        "batch_size": 4,
        "gpu_layers": -1,
        "quantization": "q8_0"
      },
      "qwen3-premium": {
        "path": "models/Qwen2.5-32B-Instruct-MLX",
        "context_length": 32768,
        "batch_size": 2,
        "gpu_layers": -1,
        "quantization": "q8_0"
      }
    }
  },
  "performance": {
    "mlx_backend": true,
    "unified_memory": true,
    "metal_performance_shaders": true,
    "neural_engine": true,
    "memory_pool_size": "64GB",
    "cache_enabled": true,
    "cache_size": "8GB"
  },
  "api": {
    "openai_compatible": true,
    "streaming": true,
    "embeddings_enabled": true,
    "function_calling": true
  }
}
```

### LM Studio Python Client
```python
import openai
import asyncio
from typing import List, Dict, Any
import time

class LMStudioClient:
    def __init__(self, base_url="http://127.0.0.1:1234/v1", api_key="lm-studio"):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.models = {
            "fast": "qwen3-fast",
            "balanced": "qwen3-balanced", 
            "quality": "qwen3-quality",
            "premium": "qwen3-premium"
        }
        
    def switch_model(self, model_type: str):
        """Switch active model based on performance requirements"""
        if model_type in self.models:
            # LM Studio model switching via API
            response = self.client.models.list()
            available_models = [m.id for m in response.data]
            
            target_model = self.models[model_type]
            if target_model in available_models:
                return target_model
            else:
                print(f"Model {target_model} not available, using default")
                return available_models[0] if available_models else None
    
    def generate_text(self, prompt: str, model_type: str = "balanced", 
                     max_tokens: int = 1000, temperature: float = 0.7):
        """Generate text with specified model"""
        model = self.switch_model(model_type)
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        end_time = time.time()
        
        return {
            "text": response.choices[0].message.content,
            "model_used": model,
            "generation_time": end_time - start_time,
            "tokens_generated": len(response.choices[0].message.content.split())
        }
    
    async def generate_embeddings(self, texts: List[str], model_type: str = "fast"):
        """Generate embeddings for multiple texts"""
        model = self.switch_model(model_type)
        
        # Batch processing for embeddings
        embeddings = []
        batch_size = 32 if model_type == "fast" else 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            response = self.client.embeddings.create(
                model=model,
                input=batch
            )
            
            batch_embeddings = [emb.embedding for emb in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def stream_generation(self, prompt: str, model_type: str = "balanced"):
        """Stream text generation for real-time responses"""
        model = self.switch_model(model_type)
        
        stream = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=1000
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

## Ollama Configuration

### Installation and Model Setup
```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull optimized Qwen3 models
ollama pull qwen2.5:1.5b-q4_0
ollama pull qwen2.5:7b-q8_0  
ollama pull qwen2.5:14b-q8_0
ollama pull qwen2.5:32b-q8_0

# Create model aliases for easier management
ollama tag qwen2.5:1.5b-q4_0 qwen3-fast
ollama tag qwen2.5:7b-q8_0 qwen3-balanced
ollama tag qwen2.5:14b-q8_0 qwen3-quality
ollama tag qwen2.5:32b-q8_0 qwen3-premium

# Verify installations
ollama list
```

### Ollama Custom Modelfile
```dockerfile
# Create custom Qwen3 configurations
# qwen3-fast.Modelfile
FROM qwen2.5:1.5b-q4_0

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# Performance optimizations for M3 Max
PARAMETER num_gpu 38  # Use all GPU cores
PARAMETER num_thread 12  # Use all CPU cores
PARAMETER use_mlock true
PARAMETER use_mmap true
```

### Ollama Python Client
```python
import ollama
import asyncio
from typing import List, Dict, Generator
import json

class OllamaClient:
    def __init__(self, host="http://localhost:11434"):
        self.client = ollama.Client(host=host)
        self.models = {
            "fast": "qwen3-fast",
            "balanced": "qwen3-balanced",
            "quality": "qwen3-quality", 
            "premium": "qwen3-premium"
        }
        
    def ensure_model_loaded(self, model_name: str):
        """Preload model for faster inference"""
        try:
            # Warm up the model with a simple prompt
            self.client.generate(
                model=model_name,
                prompt="Hello",
                options={"num_predict": 1}
            )
            return True
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return False
    
    def generate_text(self, prompt: str, model_type: str = "balanced", 
                     **options) -> Dict:
        """Generate text with Ollama"""
        model = self.models.get(model_type, "qwen3-balanced")
        
        # Ensure model is loaded
        if not self.ensure_model_loaded(model):
            model = "qwen3-fast"  # Fallback to fastest model
        
        default_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1000,
            "num_ctx": 32768
        }
        default_options.update(options)
        
        start_time = time.time()
        response = self.client.generate(
            model=model,
            prompt=prompt,
            options=default_options
        )
        end_time = time.time()
        
        return {
            "text": response["response"],
            "model_used": model,
            "generation_time": end_time - start_time,
            "context_tokens": response.get("prompt_eval_count", 0),
            "generated_tokens": response.get("eval_count", 0)
        }
    
    def stream_text(self, prompt: str, model_type: str = "balanced") -> Generator:
        """Stream text generation"""
        model = self.models.get(model_type, "qwen3-balanced")
        
        stream = self.client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": 0.7,
                "num_ctx": 32768
            }
        )
        
        for chunk in stream:
            if "response" in chunk:
                yield chunk["response"]
    
    def generate_embeddings(self, texts: List[str], model_type: str = "fast"):
        """Generate embeddings using Ollama"""
        model = self.models.get(model_type, "qwen3-fast")
        embeddings = []
        
        for text in texts:
            response = self.client.embeddings(
                model=model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        
        return embeddings
    
    async def batch_generate(self, prompts: List[str], model_type: str = "balanced"):
        """Process multiple prompts in parallel"""
        model = self.models.get(model_type, "qwen3-balanced")
        
        async def process_prompt(prompt):
            return self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.7}
            )
        
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(4)  # Limit concurrent requests
        
        async def bounded_process(prompt):
            async with semaphore:
                return await process_prompt(prompt)
        
        tasks = [bounded_process(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        return results
```

## Unified Client Interface

### Multi-Backend Client
```python
from enum import Enum
from typing import Union, Optional

class InferenceBackend(Enum):
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"

class UnifiedQwen3Client:
    def __init__(self, preferred_backend: InferenceBackend = InferenceBackend.LM_STUDIO):
        self.preferred_backend = preferred_backend
        self.lm_studio = LMStudioClient()
        self.ollama = OllamaClient()
        
        # Health check backends
        self.backend_status = self._check_backends()
        
    def _check_backends(self) -> Dict[str, bool]:
        """Check which backends are available"""
        status = {}
        
        # Check LM Studio
        try:
            self.lm_studio.client.models.list()
            status["lm_studio"] = True
        except:
            status["lm_studio"] = False
        
        # Check Ollama
        try:
            self.ollama.client.list()
            status["ollama"] = True
        except:
            status["ollama"] = False
        
        return status
    
    def get_active_backend(self) -> Union[LMStudioClient, OllamaClient]:
        """Get the best available backend"""
        if self.preferred_backend == InferenceBackend.LM_STUDIO:
            if self.backend_status.get("lm_studio", False):
                return self.lm_studio
            elif self.backend_status.get("ollama", False):
                return self.ollama
        else:
            if self.backend_status.get("ollama", False):
                return self.ollama
            elif self.backend_status.get("lm_studio", False):
                return self.lm_studio
        
        raise RuntimeError("No inference backends available")
    
    def generate(self, prompt: str, model_type: str = "balanced", **kwargs):
        """Generate text using the best available backend"""
        backend = self.get_active_backend()
        return backend.generate_text(prompt, model_type, **kwargs)
    
    def embed(self, texts: List[str], model_type: str = "fast"):
        """Generate embeddings using the best available backend"""
        backend = self.get_active_backend()
        return backend.generate_embeddings(texts, model_type)
```

## Performance Optimization

### Connection Pooling and Caching
```python
import redis
from functools import lru_cache
import hashlib

class CachedInferenceClient:
    def __init__(self, client: UnifiedQwen3Client, redis_host="localhost", redis_port=6379):
        self.client = client
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_ttl = 3600  # 1 hour cache
        
    def _generate_cache_key(self, prompt: str, model_type: str, **kwargs) -> str:
        """Generate cache key for prompt"""
        content = f"{prompt}:{model_type}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def generate_cached(self, prompt: str, model_type: str = "balanced", **kwargs):
        """Generate with caching"""
        cache_key = self._generate_cache_key(prompt, model_type, **kwargs)
        
        # Check cache first
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Generate and cache result
        result = self.client.generate(prompt, model_type, **kwargs)
        
        # Cache the result
        self.redis_client.setex(
            cache_key, 
            self.cache_ttl, 
            json.dumps(result)
        )
        
        return result
```

### Load Balancing and Failover
```python
class LoadBalancedClient:
    def __init__(self):
        self.backends = [
            LMStudioClient(),
            OllamaClient()
        ]
        self.current_backend_idx = 0
        self.max_retries = 3
        
    def get_next_backend(self):
        """Round-robin backend selection"""
        backend = self.backends[self.current_backend_idx]
        self.current_backend_idx = (self.current_backend_idx + 1) % len(self.backends)
        return backend
    
    def generate_with_failover(self, prompt: str, model_type: str = "balanced"):
        """Generate with automatic failover"""
        for attempt in range(self.max_retries):
            backend = self.get_next_backend()
            
            try:
                return backend.generate_text(prompt, model_type)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with {type(backend).__name__}: {e}")
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError("All backends failed")
                
                continue
```

## Configuration Templates

### Development Environment
```yaml
# dev-config.yaml
inference:
  primary_backend: ollama
  model_selection:
    fast_tasks: qwen3-fast
    development: qwen3-balanced
    testing: qwen3-quality
  
  performance:
    concurrent_requests: 4
    timeout: 60
    cache_enabled: true
    
  ollama:
    host: "http://localhost:11434"
    models:
      - qwen3-fast
      - qwen3-balanced
      - qwen3-quality
```

### Production Environment
```yaml
# prod-config.yaml
inference:
  primary_backend: lm_studio
  fallback_backend: ollama
  
  model_selection:
    api_responses: qwen3-quality
    batch_processing: qwen3-balanced
    real_time: qwen3-fast
  
  performance:
    concurrent_requests: 8
    timeout: 120
    cache_enabled: true
    cache_ttl: 3600
    
  lm_studio:
    host: "http://127.0.0.1:1234"
    api_key: "production-key"
    models:
      - qwen3-balanced
      - qwen3-quality
      - qwen3-premium
```

## Monitoring and Health Checks

### Service Health Monitor
```python
import psutil
import time
from dataclasses import dataclass

@dataclass
class SystemHealth:
    cpu_usage: float
    memory_usage: float
    gpu_memory: float
    model_loaded: bool
    response_time: float

class HealthMonitor:
    def __init__(self, client: UnifiedQwen3Client):
        self.client = client
        
    def check_system_health(self) -> SystemHealth:
        """Check overall system health"""
        
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU memory (M3 Max unified memory)
        gpu_memory = self._get_gpu_memory_usage()
        
        # Model health check
        start_time = time.time()
        try:
            result = self.client.generate("test", "fast")
            model_loaded = True
            response_time = time.time() - start_time
        except:
            model_loaded = False
            response_time = float('inf')
        
        return SystemHealth(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            model_loaded=model_loaded,
            response_time=response_time
        )
    
    def _get_gpu_memory_usage(self) -> float:
        """Estimate GPU memory usage on M3 Max"""
        # Since M3 Max uses unified memory, estimate based on total usage
        memory = psutil.virtual_memory()
        return min(memory.percent * 1.2, 100.0)  # Rough estimation
```

## Integration Examples

### Flask API Server
```python
from flask import Flask, request, jsonify
import asyncio

app = Flask(__name__)
client = UnifiedQwen3Client()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    model_type = data.get('model_type', 'balanced')
    
    try:
        result = client.generate(prompt, model_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    texts = data.get('texts', [])
    model_type = data.get('model_type', 'fast')
    
    try:
        embeddings = client.embed(texts, model_type)
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

## Conclusion

This setup provides a robust, high-performance local inference infrastructure using LM Studio and Ollama with Qwen3 models, optimized for M3 Max hardware and integrated with MLX backend for maximum performance.