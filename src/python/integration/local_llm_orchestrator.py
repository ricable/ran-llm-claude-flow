#!/usr/bin/env python3
"""
Local LLM Integration Orchestrator for Qwen3 Multi-Framework Deployment
Optimal integration patterns for LM Studio, Ollama, and Apple MLX frameworks.

This orchestrator provides:
1. Intelligent model selection and routing
2. Multi-framework coordination (LM Studio, Ollama, MLX)
3. Request load balancing and fallback mechanisms
4. Caching strategies for repeated inference
5. Performance monitoring and optimization
6. Error handling and resilience patterns

Key Features:
- Qwen3 model optimization for Apple M3 Max hardware
- Dynamic framework selection based on workload requirements
- Connection pooling and resource management
- Real-time performance monitoring and adaptation
- Comprehensive error handling with circuit breaker patterns
"""

import asyncio
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from urllib.parse import urljoin
import hashlib

import aiohttp
import requests
import psutil
import numpy as np
from threading import RLock
from collections import deque, defaultdict
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# MLX imports for direct model integration
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate, load_model, load_tokenizer
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX not available - MLX integration disabled")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Model specification with performance characteristics"""
    name: str
    variant: str  # 1.7b, 7b, 14b, 30b
    quantization: str  # 4bit, 8bit, 16bit, fp32
    memory_gb: float
    tokens_per_second: float
    context_length: int
    frameworks: List[str]  # ['mlx', 'ollama', 'lmstudio']
    use_cases: List[str]


@dataclass
class InferenceRequest:
    """Standardized inference request"""
    id: str
    prompt: str
    model_variant: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False
    priority: str = "normal"  # low, normal, high, critical
    timeout: float = 30.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            # Generate unique ID
            content_hash = hashlib.sha256(f"{self.prompt}{time.time()}".encode()).hexdigest()[:12]
            self.id = f"req_{content_hash}"


@dataclass
class InferenceResponse:
    """Standardized inference response"""
    request_id: str
    text: str
    model_used: str
    framework_used: str
    processing_time: float
    tokens_per_second: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FrameworkInterface(ABC):
    """Abstract base class for LLM framework interfaces"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize framework connection"""
        pass
    
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response for inference request"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check framework health"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass


class LMStudioInterface(FrameworkInterface):
    """LM Studio API interface with connection pooling"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234", max_connections: int = 10):
        self.base_url = base_url
        self.max_connections = max_connections
        self.session = None
        self.available_models = []
        self.connection_pool_lock = RLock()
        
    async def initialize(self) -> bool:
        """Initialize LM Studio connection with optimized session"""
        try:
            # Create session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Test connection and get models
            models_response = await self.session.get(f"{self.base_url}/v1/models")
            if models_response.status == 200:
                models_data = await models_response.json()
                self.available_models = [model["id"] for model in models_data.get("data", [])]
                logger.info(f"LM Studio connected. Available models: {self.available_models}")
                return True
            else:
                logger.error(f"LM Studio connection failed: {models_response.status}")
                return False
                
        except Exception as e:
            logger.error(f"LM Studio initialization failed: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response via LM Studio API"""
        start_time = time.time()
        
        try:
            # Select appropriate model
            model_name = self._select_model(request.model_variant)
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": request.prompt}
                ],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": request.stream
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LM Studio API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Extract response text
                content = result["choices"][0]["message"]["content"]
                
                # Calculate metrics
                processing_time = time.time() - start_time
                tokens_generated = len(content.split())  # Rough token count
                tokens_per_second = tokens_generated / processing_time if processing_time > 0 else 0
                
                return InferenceResponse(
                    request_id=request.id,
                    text=content,
                    model_used=model_name,
                    framework_used="lmstudio",
                    processing_time=processing_time,
                    tokens_per_second=tokens_per_second,
                    metadata={
                        "usage": result.get("usage", {}),
                        "finish_reason": result["choices"][0].get("finish_reason")
                    }
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"LM Studio generation failed: {e}")
            
            return InferenceResponse(
                request_id=request.id,
                text="",
                model_used="unknown",
                framework_used="lmstudio",
                processing_time=processing_time,
                tokens_per_second=0,
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check LM Studio health"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available models in LM Studio"""
        return self.available_models.copy()
    
    def _select_model(self, variant: Optional[str] = None) -> str:
        """Select appropriate model based on variant preference"""
        if not self.available_models:
            return "default"
        
        if variant:
            # Try to find model matching variant
            for model in self.available_models:
                if variant.lower() in model.lower():
                    return model
        
        # Default to first available model
        return self.available_models[0]
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()


class OllamaInterface(FrameworkInterface):
    """Ollama interface optimized for embeddings and inference"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", max_connections: int = 8):
        self.base_url = base_url
        self.max_connections = max_connections
        self.session = None
        self.available_models = []
        self.model_cache = {}  # Cache for loaded models
        
    async def initialize(self) -> bool:
        """Initialize Ollama connection"""
        try:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                keepalive_timeout=60
            )
            
            self.session = aiohttp.ClientSession(connector=connector)
            
            # Get available models
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
                    logger.info(f"Ollama connected. Available models: {self.available_models}")
                    return True
                else:
                    logger.error(f"Ollama connection failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response via Ollama"""
        start_time = time.time()
        
        try:
            model_name = self._select_ollama_model(request.model_variant)
            
            # Ensure model is loaded
            await self._ensure_model_loaded(model_name)
            
            payload = {
                "model": model_name,
                "prompt": request.prompt,
                "stream": request.stream,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                # Handle streaming vs non-streaming
                if request.stream:
                    full_response = await self._handle_streaming_response(response)
                else:
                    result = await response.json()
                    full_response = result.get("response", "")
                
                processing_time = time.time() - start_time
                tokens_generated = len(full_response.split())
                tokens_per_second = tokens_generated / processing_time if processing_time > 0 else 0
                
                return InferenceResponse(
                    request_id=request.id,
                    text=full_response,
                    model_used=model_name,
                    framework_used="ollama",
                    processing_time=processing_time,
                    tokens_per_second=tokens_per_second
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Ollama generation failed: {e}")
            
            return InferenceResponse(
                request_id=request.id,
                text="",
                model_used="unknown",
                framework_used="ollama",
                processing_time=processing_time,
                tokens_per_second=0,
                success=False,
                error=str(e)
            )
    
    async def _ensure_model_loaded(self, model_name: str):
        """Ensure model is loaded in Ollama"""
        if model_name in self.model_cache:
            return
        
        try:
            # Pull model if not available
            if model_name not in self.available_models:
                logger.info(f"Pulling Ollama model: {model_name}")
                async with self.session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                ) as response:
                    if response.status == 200:
                        self.available_models.append(model_name)
            
            self.model_cache[model_name] = True
            
        except Exception as e:
            logger.warning(f"Model loading failed for {model_name}: {e}")
    
    async def _handle_streaming_response(self, response) -> str:
        """Handle streaming response from Ollama"""
        full_text = ""
        async for line in response.content:
            if line:
                try:
                    chunk = json.loads(line.decode())
                    if "response" in chunk:
                        full_text += chunk["response"]
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        return full_text
    
    def _select_ollama_model(self, variant: Optional[str] = None) -> str:
        """Select appropriate Ollama model"""
        # Map variants to Ollama model names
        variant_mapping = {
            "1.7b": "qwen2.5:1.5b",
            "7b": "qwen2.5:7b",
            "14b": "qwen2.5:14b",
            "fast": "qwen2.5:1.5b-q4_0",
            "balanced": "qwen2.5:7b-q8_0",
            "quality": "qwen2.5:14b-q8_0"
        }
        
        if variant and variant in variant_mapping:
            return variant_mapping[variant]
        
        # Default to fastest model
        return "qwen2.5:1.5b-q4_0"
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available models in Ollama"""
        return self.available_models.copy()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()


class MLXInterface(FrameworkInterface):
    """Apple MLX direct interface for maximum performance"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.models = {}  # Cache for loaded models
        self.model_lock = RLock()
        self.initialized = False
        
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - MLX interface will be disabled")
    
    async def initialize(self) -> bool:
        """Initialize MLX models"""
        if not MLX_AVAILABLE:
            return False
        
        try:
            # Pre-load common Qwen3 variants
            model_variants = {
                "qwen3-1.7b": "/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-1.7B-MLX-4bit",
                "qwen3-7b": "/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-7B-MLX-8bit",
                "qwen3-14b": "/Users/cedric/.lmstudio/models/lmstudio-community/Qwen3-14B-MLX-8bit"
            }
            
            # Load models in parallel (simulate async with ThreadPoolExecutor)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for variant, path in model_variants.items():
                    if Path(path).exists():
                        future = executor.submit(self._load_model_sync, variant, path)
                        futures[future] = variant
                
                # Wait for models to load
                for future in as_completed(futures, timeout=120):
                    variant = futures[future]
                    try:
                        success = future.result()
                        if success:
                            logger.info(f"MLX model loaded: {variant}")
                    except Exception as e:
                        logger.error(f"Failed to load MLX model {variant}: {e}")
            
            self.initialized = True
            logger.info(f"MLX interface initialized with {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"MLX initialization failed: {e}")
            return False
    
    def _load_model_sync(self, variant: str, path: str) -> bool:
        """Synchronously load MLX model (for use in thread executor)"""
        try:
            model, tokenizer = load(path)
            with self.model_lock:
                self.models[variant] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "path": path,
                    "loaded_at": time.time()
                }
            return True
        except Exception as e:
            logger.error(f"Failed to load MLX model {variant}: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response using MLX"""
        if not MLX_AVAILABLE or not self.initialized:
            return InferenceResponse(
                request_id=request.id,
                text="",
                model_used="mlx",
                framework_used="mlx",
                processing_time=0,
                tokens_per_second=0,
                success=False,
                error="MLX not available or not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Select model
            model_variant = self._select_mlx_model(request.model_variant)
            
            if model_variant not in self.models:
                raise Exception(f"MLX model not loaded: {model_variant}")
            
            model_info = self.models[model_variant]
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Generate response (run in thread to avoid blocking)
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                self._generate_sync,
                model,
                tokenizer,
                request.prompt,
                request.max_tokens,
                request.temperature
            )
            
            processing_time = time.time() - start_time
            tokens_generated = len(response_text.split())
            tokens_per_second = tokens_generated / processing_time if processing_time > 0 else 0
            
            return InferenceResponse(
                request_id=request.id,
                text=response_text,
                model_used=model_variant,
                framework_used="mlx",
                processing_time=processing_time,
                tokens_per_second=tokens_per_second,
                metadata={"model_path": model_info["path"]}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"MLX generation failed: {e}")
            
            return InferenceResponse(
                request_id=request.id,
                text="",
                model_used="unknown",
                framework_used="mlx",
                processing_time=processing_time,
                tokens_per_second=0,
                success=False,
                error=str(e)
            )
    
    def _generate_sync(self, model, tokenizer, prompt: str, max_tokens: int, temperature: float) -> str:
        """Synchronous MLX generation"""
        try:
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
            return response
        except Exception as e:
            logger.error(f"MLX sync generation failed: {e}")
            return ""
    
    def _select_mlx_model(self, variant: Optional[str] = None) -> str:
        """Select appropriate MLX model"""
        if variant:
            # Try exact match first
            for model_name in self.models.keys():
                if variant.lower() in model_name.lower():
                    return model_name
        
        # Priority order: 1.7b (fast) -> 7b (balanced) -> 14b (quality)
        priority_order = ["qwen3-1.7b", "qwen3-7b", "qwen3-14b"]
        for preferred in priority_order:
            if preferred in self.models:
                return preferred
        
        # Return any available model
        if self.models:
            return list(self.models.keys())[0]
        
        raise Exception("No MLX models loaded")
    
    async def health_check(self) -> bool:
        """Check MLX health"""
        return MLX_AVAILABLE and self.initialized and len(self.models) > 0
    
    def get_available_models(self) -> List[str]:
        """Get available MLX models"""
        return list(self.models.keys())
    
    async def cleanup(self):
        """Cleanup MLX resources"""
        with self.model_lock:
            self.models.clear()
        self.initialized = False


class CircuitBreaker:
    """Circuit breaker for framework health management"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = RLock()
    
    def can_proceed(self) -> bool:
        """Check if request can proceed through circuit breaker"""
        with self.lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful operation"""
        with self.lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")


class InferenceCache:
    """LRU cache for inference results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = deque()
        self.lock = RLock()
    
    def _generate_key(self, request: InferenceRequest) -> str:
        """Generate cache key from request"""
        key_components = [
            request.prompt,
            str(request.max_tokens),
            str(request.temperature),
            request.model_variant or "default"
        ]
        return hashlib.sha256(":".join(key_components).encode()).hexdigest()
    
    def get(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Get cached response if available and valid"""
        key = self._generate_key(request)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Return cached response with updated request_id
            cached_response = entry["response"]
            cached_response.request_id = request.id
            return cached_response
    
    def put(self, request: InferenceRequest, response: InferenceResponse):
        """Cache inference response"""
        if not response.success:
            return  # Don't cache failures
        
        key = self._generate_key(request)
        
        with self.lock:
            # Evict LRU if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                if self.access_order:
                    lru_key = self.access_order.popleft()
                    if lru_key in self.cache:
                        del self.cache[lru_key]
            
            # Store in cache
            self.cache[key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size
            }


class PerformanceMonitor:
    """Monitor and track framework performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = RLock()
    
    def record_request(self, framework: str, response: InferenceResponse):
        """Record request metrics"""
        with self.lock:
            timestamp = time.time()
            
            self.metrics[f"{framework}_response_time"].append(response.processing_time)
            self.metrics[f"{framework}_tokens_per_second"].append(response.tokens_per_second)
            self.metrics[f"{framework}_success"].append(1 if response.success else 0)
            self.metrics[f"{framework}_timestamp"].append(timestamp)
    
    def get_framework_stats(self, framework: str) -> Dict[str, float]:
        """Get aggregated stats for framework"""
        with self.lock:
            response_times = list(self.metrics[f"{framework}_response_time"])
            tokens_per_sec = list(self.metrics[f"{framework}_tokens_per_second"])
            successes = list(self.metrics[f"{framework}_success"])
            
            if not response_times:
                return {
                    "avg_response_time": 0,
                    "tokens_per_second": 0,
                    "success_rate": 0,
                    "request_count": 0
                }
            
            return {
                "avg_response_time": np.mean(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "success_rate": np.mean(successes),
                "request_count": len(response_times),
                "recent_throughput": len([t for t in self.metrics[f"{framework}_timestamp"] 
                                        if time.time() - t < 60]) / 60.0  # Requests per second in last minute
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all frameworks"""
        frameworks = set()
        for key in self.metrics.keys():
            framework = key.split('_')[0]
            frameworks.add(framework)
        
        return {framework: self.get_framework_stats(framework) for framework in frameworks}
    
    def select_best_framework(self, frameworks: List[str]) -> str:
        """Select best performing framework based on recent metrics"""
        best_framework = frameworks[0]
        best_score = 0
        
        for framework in frameworks:
            stats = self.get_framework_stats(framework)
            
            # Calculate composite score (higher is better)
            score = (
                stats["success_rate"] * 0.4 +  # 40% weight on success rate
                min(stats["avg_tokens_per_second"] / 100, 1.0) * 0.3 +  # 30% weight on throughput
                max(0, 1 - stats["avg_response_time"] / 10) * 0.3  # 30% weight on low latency
            )
            
            if score > best_score:
                best_score = score
                best_framework = framework
        
        return best_framework


class LocalLLMOrchestrator:
    """Main orchestrator for local LLM frameworks"""
    
    def __init__(self, 
                 lmstudio_url: str = "http://127.0.0.1:1234",
                 ollama_url: str = "http://127.0.0.1:11434",
                 mlx_model_path: str = None,
                 cache_size: int = 1000,
                 enable_cache: bool = True):
        
        # Initialize frameworks
        self.frameworks = {
            "lmstudio": LMStudioInterface(lmstudio_url),
            "ollama": OllamaInterface(ollama_url),
            "mlx": MLXInterface(mlx_model_path)
        }
        
        # Circuit breakers for each framework
        self.circuit_breakers = {
            name: CircuitBreaker() for name in self.frameworks.keys()
        }
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Caching
        self.cache = InferenceCache(max_size=cache_size) if enable_cache else None
        
        # Framework availability
        self.framework_health = {name: False for name in self.frameworks.keys()}
        
        # Configuration
        self.model_routing_config = self._create_model_routing_config()
        
        # Background tasks
        self._health_check_task = None
        self._monitoring_active = False
        
        logger.info("Local LLM Orchestrator initialized")
    
    def _create_model_routing_config(self) -> Dict[str, Dict]:
        """Create model routing configuration"""
        return {
            "speed_priority": {
                "primary": "mlx",
                "fallback": ["ollama", "lmstudio"],
                "model_variant": "1.7b"
            },
            "balanced": {
                "primary": "lmstudio", 
                "fallback": ["mlx", "ollama"],
                "model_variant": "7b"
            },
            "quality_priority": {
                "primary": "lmstudio",
                "fallback": ["mlx", "ollama"],
                "model_variant": "14b"
            },
            "embedding": {
                "primary": "ollama",
                "fallback": ["mlx", "lmstudio"],
                "model_variant": "7b"
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize all frameworks"""
        logger.info("Initializing local LLM frameworks...")
        
        initialization_tasks = []
        for name, framework in self.frameworks.items():
            task = asyncio.create_task(self._initialize_framework(name, framework))
            initialization_tasks.append(task)
        
        # Wait for all initializations
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Update health status
        for (name, _), result in zip(self.frameworks.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Framework {name} initialization failed: {result}")
                self.framework_health[name] = False
            else:
                self.framework_health[name] = result
        
        # Start health monitoring
        await self._start_health_monitoring()
        
        available_frameworks = [name for name, healthy in self.framework_health.items() if healthy]
        logger.info(f"Frameworks initialized. Available: {available_frameworks}")
        
        return len(available_frameworks) > 0
    
    async def _initialize_framework(self, name: str, framework: FrameworkInterface) -> bool:
        """Initialize individual framework"""
        try:
            success = await framework.initialize()
            logger.info(f"Framework {name}: {'✓ Ready' if success else '✗ Failed'}")
            return success
        except Exception as e:
            logger.error(f"Framework {name} initialization error: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Main generation method with intelligent routing"""
        
        # Check cache first
        if self.cache:
            cached_response = self.cache.get(request)
            if cached_response:
                logger.debug(f"Cache hit for request {request.id}")
                return cached_response
        
        # Select framework based on request characteristics
        selected_framework = self._select_optimal_framework(request)
        
        # Try primary framework
        response = await self._try_framework_with_fallback(request, selected_framework)
        
        # Cache successful responses
        if self.cache and response.success:
            self.cache.put(request, response)
        
        # Record metrics
        self.performance_monitor.record_request(response.framework_used, response)
        
        return response
    
    async def _try_framework_with_fallback(self, request: InferenceRequest, primary_framework: str) -> InferenceResponse:
        """Try framework with fallback logic"""
        
        # Get routing config
        routing_config = self._get_routing_config(request)
        frameworks_to_try = [primary_framework] + routing_config.get("fallback", [])
        
        # Remove duplicates while preserving order
        frameworks_to_try = list(dict.fromkeys(frameworks_to_try))
        
        for framework_name in frameworks_to_try:
            # Check circuit breaker
            if not self.circuit_breakers[framework_name].can_proceed():
                logger.debug(f"Circuit breaker open for {framework_name}")
                continue
            
            # Check framework health
            if not self.framework_health.get(framework_name, False):
                logger.debug(f"Framework {framework_name} not healthy")
                continue
            
            try:
                # Set model variant in request
                if "model_variant" in routing_config:
                    request.model_variant = routing_config["model_variant"]
                
                framework = self.frameworks[framework_name]
                response = await framework.generate(request)
                
                if response.success:
                    self.circuit_breakers[framework_name].record_success()
                    logger.debug(f"Request {request.id} handled by {framework_name}")
                    return response
                else:
                    self.circuit_breakers[framework_name].record_failure()
                    logger.warning(f"Framework {framework_name} failed: {response.error}")
                    
            except Exception as e:
                self.circuit_breakers[framework_name].record_failure()
                logger.error(f"Framework {framework_name} exception: {e}")
                continue
        
        # All frameworks failed
        return InferenceResponse(
            request_id=request.id,
            text="",
            model_used="none",
            framework_used="none",
            processing_time=0,
            tokens_per_second=0,
            success=False,
            error="All frameworks unavailable"
        )
    
    def _select_optimal_framework(self, request: InferenceRequest) -> str:
        """Select optimal framework based on request characteristics"""
        
        # Analyze request to determine requirements
        request_type = self._analyze_request_type(request)
        
        # Get routing configuration
        routing_config = self.model_routing_config.get(request_type, self.model_routing_config["balanced"])
        
        # Consider performance metrics for final selection
        available_frameworks = [name for name, healthy in self.framework_health.items() 
                              if healthy and self.circuit_breakers[name].can_proceed()]
        
        if not available_frameworks:
            return "lmstudio"  # Default fallback
        
        # Use performance monitor to select among available frameworks
        best_framework = self.performance_monitor.select_best_framework(available_frameworks)
        
        # Override with routing config primary if it's available
        primary_framework = routing_config.get("primary")
        if primary_framework in available_frameworks:
            return primary_framework
        
        return best_framework
    
    def _analyze_request_type(self, request: InferenceRequest) -> str:
        """Analyze request to determine optimal routing strategy"""
        
        # Priority-based routing
        if request.priority == "critical" or request.timeout < 5:
            return "speed_priority"
        
        # Length-based routing
        prompt_length = len(request.prompt)
        if prompt_length > 4000:
            return "quality_priority"
        elif prompt_length < 500:
            return "speed_priority"
        
        # Content-based routing
        prompt_lower = request.prompt.lower()
        if any(keyword in prompt_lower for keyword in ["embed", "vector", "similarity"]):
            return "embedding"
        
        # Temperature-based routing
        if request.temperature < 0.3:
            return "quality_priority"
        elif request.temperature > 0.8:
            return "balanced"
        
        return "balanced"
    
    def _get_routing_config(self, request: InferenceRequest) -> Dict:
        """Get routing configuration for request"""
        request_type = self._analyze_request_type(request)
        return self.model_routing_config.get(request_type, self.model_routing_config["balanced"])
    
    async def _start_health_monitoring(self):
        """Start background health monitoring"""
        self._monitoring_active = True
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self._monitoring_active:
            try:
                health_tasks = []
                for name, framework in self.frameworks.items():
                    if self.framework_health.get(name, False):  # Only check if previously healthy
                        task = asyncio.create_task(framework.health_check())
                        health_tasks.append((name, task))
                
                # Check health in parallel
                for name, task in health_tasks:
                    try:
                        healthy = await asyncio.wait_for(task, timeout=5)
                        old_status = self.framework_health[name]
                        self.framework_health[name] = healthy
                        
                        if old_status != healthy:
                            logger.info(f"Framework {name} health changed: {old_status} -> {healthy}")
                            
                    except asyncio.TimeoutError:
                        self.framework_health[name] = False
                        logger.warning(f"Framework {name} health check timeout")
                    except Exception as e:
                        self.framework_health[name] = False
                        logger.error(f"Framework {name} health check error: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "frameworks": {
                name: {
                    "healthy": self.framework_health.get(name, False),
                    "circuit_breaker_state": self.circuit_breakers[name].state,
                    "available_models": framework.get_available_models() if hasattr(framework, 'get_available_models') else []
                }
                for name, framework in self.frameworks.items()
            },
            "performance": self.performance_monitor.get_all_stats(),
            "cache": self.cache.stats() if self.cache else {"enabled": False},
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        logger.info("Cleaning up Local LLM Orchestrator...")
        
        # Stop monitoring
        self._monitoring_active = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup frameworks
        cleanup_tasks = []
        for framework in self.frameworks.values():
            if hasattr(framework, 'cleanup'):
                cleanup_tasks.append(asyncio.create_task(framework.cleanup()))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear cache
        if self.cache:
            self.cache.clear()
        
        logger.info("Local LLM Orchestrator cleanup completed")


# Utility functions for integration testing and setup

async def test_orchestrator():
    """Test orchestrator functionality"""
    orchestrator = LocalLLMOrchestrator()
    
    try:
        # Initialize
        await orchestrator.initialize()
        
        # Test requests
        test_requests = [
            InferenceRequest(
                id="test_1",
                prompt="What is 5G NR?",
                model_variant="7b",
                priority="normal"
            ),
            InferenceRequest(
                id="test_2", 
                prompt="Explain MIMO technology in telecommunications.",
                model_variant="14b",
                priority="high",
                temperature=0.3
            ),
            InferenceRequest(
                id="test_3",
                prompt="Quick summary of LTE.",
                model_variant="1.7b",
                priority="critical",
                timeout=3.0
            )
        ]
        
        # Process requests
        for request in test_requests:
            response = await orchestrator.generate(request)
            print(f"Request {request.id}: {response.success} via {response.framework_used}")
            print(f"Response: {response.text[:100]}...")
            print(f"Performance: {response.tokens_per_second:.1f} tokens/sec\n")
        
        # Get status
        status = await orchestrator.get_status()
        print(f"Orchestrator status: {json.dumps(status, indent=2)}")
        
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    # Example usage
    asyncio.run(test_orchestrator())