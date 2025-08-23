#!/usr/bin/env python3
"""
Local Inference Integration for LM Studio and Ollama

High-performance local inference client supporting LM Studio and Ollama
with automatic failover, load balancing, and performance optimization.

Author: Claude Code ML Integration Specialist
Date: 2025-08-23
"""

import asyncio
import aiohttp
import json
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, AsyncGenerator, Tuple
import subprocess
from pathlib import Path

# Third-party imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI client not available for LM Studio integration")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama client not available")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available for caching")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceBackend(Enum):
    """Available inference backends"""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    MLX_LOCAL = "mlx_local"

@dataclass
class InferenceRequest:
    """Standardized inference request"""
    prompt: str
    model_type: str = "balanced"
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False
    timeout: int = 120
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceResponse:
    """Standardized inference response"""
    text: str
    model_used: str
    backend_used: str
    generation_time_seconds: float
    tokens_per_second: float
    total_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class InferenceBackendClient(ABC):
    """Abstract base class for inference backends"""
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy and available"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models"""
        pass
    
    @abstractmethod
    async def generate_text(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str], model_type: str = "fast") -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    async def stream_text(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        pass

class LMStudioClient(InferenceBackendClient):
    """LM Studio inference client with OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", api_key: str = "lm-studio"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = None
        self.models = {
            "fast": "qwen3-fast",
            "balanced": "qwen3-balanced",
            "quality": "qwen3-quality",
            "premium": "qwen3-premium"
        }
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client for LM Studio"""
        if OPENAI_AVAILABLE:
            self.client = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0
            )
        else:
            logger.warning("OpenAI client not available, LM Studio integration disabled")
    
    async def health_check(self) -> bool:
        """Check LM Studio health"""
        try:
            if not self.client:
                return False
            
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"LM Studio health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in LM Studio"""
        try:
            if not self.client:
                return []
            
            response = await self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            return []
    
    async def generate_text(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using LM Studio"""
        if not self.client:
            raise RuntimeError("LM Studio client not available")
        
        model = self.models.get(request.model_type, "qwen3-balanced")
        
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
                timeout=request.timeout
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            text = response.choices[0].message.content
            total_tokens = response.usage.total_tokens if response.usage else len(text.split())
            completion_tokens = response.usage.completion_tokens if response.usage else len(text.split())
            prompt_tokens = response.usage.prompt_tokens if response.usage else len(request.prompt.split())
            
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
            
            return InferenceResponse(
                text=text,
                model_used=model,
                backend_used="lm_studio",
                generation_time_seconds=generation_time,
                tokens_per_second=tokens_per_second,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={"response_id": response.id if hasattr(response, 'id') else None}
            )
            
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str], model_type: str = "fast") -> List[List[float]]:
        """Generate embeddings using LM Studio"""
        if not self.client:
            raise RuntimeError("LM Studio client not available")
        
        model = self.models.get(model_type, "qwen3-fast")
        embeddings = []
        
        # Process in batches for better performance
        batch_size = 32 if model_type == "fast" else 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                batch_embeddings = [emb.embedding for emb in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Embedding generation failed for batch {i}: {e}")
                # Return zeros for failed batch
                embeddings.extend([[0.0] * 768 for _ in batch])  # Assume 768-dim embeddings
        
        return embeddings
    
    async def stream_text(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream text generation from LM Studio"""
        if not self.client:
            raise RuntimeError("LM Studio client not available")
        
        model = self.models.get(request.model_type, "qwen3-balanced")
        
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
                timeout=request.timeout
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LM Studio streaming failed: {e}")
            raise

class OllamaClient(InferenceBackendClient):
    """Ollama inference client"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.client = None
        self.models = {
            "fast": "qwen3-fast",
            "balanced": "qwen3-balanced",
            "quality": "qwen3-quality",
            "premium": "qwen3-premium"
        }
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client"""
        if OLLAMA_AVAILABLE:
            self.client = ollama.AsyncClient(host=self.host)
        else:
            logger.warning("Ollama client not available")
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            if not self.client:
                return False
            
            models = await self.client.list()
            return len(models.get('models', [])) > 0
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            if not self.client:
                return []
            
            response = await self.client.list()
            return [model['name'] for model in response.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure model is loaded in Ollama"""
        try:
            # Warm up the model with a simple prompt
            await self.client.generate(
                model=model_name,
                prompt="Hello",
                options={"num_predict": 1}
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            return False
    
    async def generate_text(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using Ollama"""
        if not self.client:
            raise RuntimeError("Ollama client not available")
        
        model = self.models.get(request.model_type, "qwen3-balanced")
        
        # Ensure model is loaded
        await self.ensure_model_loaded(model)
        
        start_time = time.time()
        
        try:
            response = await self.client.generate(
                model=model,
                prompt=request.prompt,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "num_ctx": 32768
                }
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            text = response["response"]
            prompt_tokens = response.get("prompt_eval_count", len(request.prompt.split()))
            completion_tokens = response.get("eval_count", len(text.split()))
            total_tokens = prompt_tokens + completion_tokens
            
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
            
            return InferenceResponse(
                text=text,
                model_used=model,
                backend_used="ollama",
                generation_time_seconds=generation_time,
                tokens_per_second=tokens_per_second,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={
                    "eval_duration": response.get("eval_duration", 0),
                    "load_duration": response.get("load_duration", 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str], model_type: str = "fast") -> List[List[float]]:
        """Generate embeddings using Ollama"""
        if not self.client:
            raise RuntimeError("Ollama client not available")
        
        model = self.models.get(model_type, "qwen3-fast")
        embeddings = []
        
        for text in texts:
            try:
                response = await self.client.embeddings(
                    model=model,
                    prompt=text
                )
                embeddings.append(response["embedding"])
                
            except Exception as e:
                logger.error(f"Embedding generation failed for text: {e}")
                embeddings.append([0.0] * 768)  # Fallback embedding
        
        return embeddings
    
    async def stream_text(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream text generation from Ollama"""
        if not self.client:
            raise RuntimeError("Ollama client not available")
        
        model = self.models.get(request.model_type, "qwen3-balanced")
        
        try:
            async for chunk in await self.client.generate(
                model=model,
                prompt=request.prompt,
                stream=True,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            ):
                if "response" in chunk:
                    yield chunk["response"]
                    
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

class CachedInferenceLayer:
    """Caching layer for inference requests"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, ttl: int = 3600):
        self.redis_client = None
        self.ttl = ttl
        self.memory_cache = {}  # Fallback in-memory cache
        
        if REDIS_AVAILABLE:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    decode_responses=True,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed, using memory cache: {e}")
                self.redis_client = None
        else:
            logger.info("Using in-memory cache (Redis not available)")
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        content = f"{request.prompt}:{request.model_type}:{request.max_tokens}:{request.temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_cached_response(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Get cached response if available"""
        cache_key = self._generate_cache_key(request)
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return InferenceResponse(**data)
            else:
                # Use memory cache
                if cache_key in self.memory_cache:
                    cache_entry = self.memory_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < self.ttl:
                        return InferenceResponse(**cache_entry['data'])
                    else:
                        del self.memory_cache[cache_key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_response(self, request: InferenceRequest, response: InferenceResponse):
        """Cache the response"""
        cache_key = self._generate_cache_key(request)
        
        try:
            response_data = {
                'text': response.text,
                'model_used': response.model_used,
                'backend_used': response.backend_used,
                'generation_time_seconds': response.generation_time_seconds,
                'tokens_per_second': response.tokens_per_second,
                'total_tokens': response.total_tokens,
                'prompt_tokens': response.prompt_tokens,
                'completion_tokens': response.completion_tokens,
                'metadata': response.metadata
            }
            
            if self.redis_client:
                self.redis_client.setex(
                    cache_key, 
                    self.ttl, 
                    json.dumps(response_data)
                )
            else:
                # Use memory cache
                self.memory_cache[cache_key] = {
                    'data': response_data,
                    'timestamp': time.time()
                }
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

class UnifiedLocalInferenceClient:
    """Unified client for multiple local inference backends"""
    
    def __init__(self, 
                 preferred_backend: InferenceBackend = InferenceBackend.LM_STUDIO,
                 enable_caching: bool = True,
                 max_retries: int = 3):
        
        self.preferred_backend = preferred_backend
        self.max_retries = max_retries
        
        # Initialize backends
        self.backends = {
            InferenceBackend.LM_STUDIO: LMStudioClient(),
            InferenceBackend.OLLAMA: OllamaClient()
        }
        
        # Initialize caching
        self.cache = CachedInferenceLayer() if enable_caching else None
        
        # Backend health status
        self.backend_health = {}
        
        logger.info(f"Initialized UnifiedLocalInferenceClient with preferred backend: {preferred_backend.value}")
    
    async def health_check_all_backends(self) -> Dict[InferenceBackend, bool]:
        """Check health of all backends"""
        health_results = {}
        
        for backend_type, client in self.backends.items():
            try:
                is_healthy = await client.health_check()
                health_results[backend_type] = is_healthy
                self.backend_health[backend_type] = {
                    'healthy': is_healthy,
                    'last_checked': time.time()
                }
            except Exception as e:
                logger.error(f"Health check failed for {backend_type.value}: {e}")
                health_results[backend_type] = False
                self.backend_health[backend_type] = {
                    'healthy': False,
                    'last_checked': time.time(),
                    'error': str(e)
                }
        
        return health_results
    
    async def get_best_available_backend(self) -> Optional[InferenceBackendClient]:
        """Get the best available backend based on health and preference"""
        # Check health of backends
        health_status = await self.health_check_all_backends()
        
        # Try preferred backend first
        if health_status.get(self.preferred_backend, False):
            return self.backends[self.preferred_backend]
        
        # Try other healthy backends
        for backend_type, is_healthy in health_status.items():
            if is_healthy:
                logger.info(f"Falling back to {backend_type.value} backend")
                return self.backends[backend_type]
        
        logger.error("No healthy backends available")
        return None
    
    async def generate_text(self, 
                          prompt: str,
                          model_type: str = "balanced",
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          use_cache: bool = True) -> InferenceResponse:
        """Generate text with automatic backend selection and failover"""
        
        request = InferenceRequest(
            prompt=prompt,
            model_type=model_type,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Check cache first
        if self.cache and use_cache:
            cached_response = await self.cache.get_cached_response(request)
            if cached_response:
                logger.debug("Returning cached response")
                cached_response.metadata["from_cache"] = True
                return cached_response
        
        # Try generation with retries and failover
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                backend = await self.get_best_available_backend()
                if not backend:
                    raise RuntimeError("No backends available")
                
                response = await backend.generate_text(request)
                
                # Cache successful response
                if self.cache and use_cache:
                    await self.cache.cache_response(request, response)
                
                # Store performance metrics
                await self._store_performance_metrics(response)
                
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"All generation attempts failed. Last error: {last_exception}")
    
    async def stream_text(self,
                         prompt: str,
                         model_type: str = "balanced",
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """Stream text generation with automatic backend selection"""
        
        request = InferenceRequest(
            prompt=prompt,
            model_type=model_type,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        backend = await self.get_best_available_backend()
        if not backend:
            raise RuntimeError("No backends available for streaming")
        
        async for chunk in backend.stream_text(request):
            yield chunk
    
    async def generate_embeddings(self, 
                                texts: List[str], 
                                model_type: str = "fast") -> List[List[float]]:
        """Generate embeddings with automatic backend selection"""
        
        backend = await self.get_best_available_backend()
        if not backend:
            raise RuntimeError("No backends available for embeddings")
        
        return await backend.generate_embeddings(texts, model_type)
    
    async def batch_generate(self, 
                           requests: List[InferenceRequest],
                           max_concurrent: int = 4) -> List[InferenceResponse]:
        """Process multiple requests in parallel"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request):
            async with semaphore:
                return await self.generate_text(
                    request.prompt,
                    request.model_type,
                    request.max_tokens,
                    request.temperature
                )
        
        tasks = [process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append(InferenceResponse(
                    text=f"Error: {str(response)}",
                    model_used="error",
                    backend_used="error",
                    generation_time_seconds=0,
                    tokens_per_second=0,
                    total_tokens=0,
                    metadata={"error": str(response)}
                ))
            else:
                results.append(response)
        
        return results
    
    async def _store_performance_metrics(self, response: InferenceResponse):
        """Store performance metrics in coordination memory"""
        try:
            metrics_data = {
                'backend_used': response.backend_used,
                'model_used': response.model_used,
                'timestamp': time.time(),
                'performance': {
                    'tokens_per_second': response.tokens_per_second,
                    'generation_time': response.generation_time_seconds,
                    'total_tokens': response.total_tokens
                }
            }
            
            # Store in coordination memory
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"local_inference/performance/{response.backend_used}/{int(time.time())}",
                "--data", json.dumps(metrics_data)
            ], capture_output=True)
            
        except Exception as e:
            logger.warning(f"Could not store performance metrics: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        health_status = await self.health_check_all_backends()
        
        return {
            'timestamp': time.time(),
            'backend_health': health_status,
            'preferred_backend': self.preferred_backend.value,
            'cache_enabled': self.cache is not None,
            'available_backends': [backend.value for backend, healthy in health_status.items() if healthy]
        }

# Example usage and testing
async def main():
    """Example usage of the unified local inference client"""
    
    # Initialize client
    client = UnifiedLocalInferenceClient()
    
    try:
        # Test health checks
        health = await client.health_check_all_backends()
        print(f"Backend health: {health}")
        
        # Test text generation
        response = await client.generate_text(
            "Explain machine learning in simple terms.",
            model_type="balanced"
        )
        
        print(f"Generated text: {response.text[:100]}...")
        print(f"Backend used: {response.backend_used}")
        print(f"Performance: {response.tokens_per_second:.2f} tokens/sec")
        
        # Test streaming
        print("\nStreaming example:")
        async for chunk in client.stream_text("Write a short poem about AI"):
            print(chunk, end='', flush=True)
        print("\n")
        
        # Test embeddings
        embeddings = await client.generate_embeddings(
            ["Hello world", "Machine learning is fascinating"]
        )
        print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}")
        
        # Performance summary
        summary = await client.get_performance_summary()
        print(f"Performance summary: {json.dumps(summary, indent=2)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
