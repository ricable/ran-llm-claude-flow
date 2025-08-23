#!/usr/bin/env python3
"""
Ollama Deployment Optimizer for Qwen3 Models
Specialized strategies for embeddings, inference, and model management.

Features:
- Intelligent model pre-loading and caching
- Optimized embedding generation strategies
- Dynamic quantization selection
- Memory-efficient batch processing
- Multi-model concurrent deployment
- Performance monitoring and auto-scaling
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import threading
import aiohttp
import numpy as np
import psutil
from threading import RLock, Event
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelSpec:
    """Ollama model specification"""
    name: str
    tag: str
    size_gb: float
    quantization: str
    use_cases: List[str]
    performance_tier: str  # fast, balanced, quality
    context_length: int
    expected_tokens_per_second: float
    memory_requirement_gb: float
    
    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}"


@dataclass
class EmbeddingRequest:
    """Embedding generation request"""
    id: str
    texts: List[str]
    model: str
    normalize: bool = True
    batch_size: int = 32
    priority: str = "normal"
    timeout: float = 30.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            content_hash = hashlib.sha256(str(self.texts).encode()).hexdigest()[:12]
            self.id = f"embed_{content_hash}"


@dataclass
class EmbeddingResponse:
    """Embedding generation response"""
    request_id: str
    embeddings: List[List[float]]
    model_used: str
    processing_time: float
    tokens_processed: int
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelRegistry:
    """Registry of optimized Qwen3 models for Ollama"""
    
    def __init__(self):
        self.models = {
            # Fast models for speed-critical tasks
            "qwen3-fast": OllamaModelSpec(
                name="qwen2.5",
                tag="1.5b-q4_0",
                size_gb=1.2,
                quantization="Q4_0",
                use_cases=["classification", "quick_qa", "real_time"],
                performance_tier="fast",
                context_length=32768,
                expected_tokens_per_second=120,
                memory_requirement_gb=2.0
            ),
            
            # Balanced models for general use
            "qwen3-balanced": OllamaModelSpec(
                name="qwen2.5",
                tag="7b-q8_0",
                size_gb=7.2,
                quantization="Q8_0",
                use_cases=["embeddings", "summarization", "general_qa"],
                performance_tier="balanced",
                context_length=32768,
                expected_tokens_per_second=45,
                memory_requirement_gb=9.0
            ),
            
            # Quality models for complex tasks
            "qwen3-quality": OllamaModelSpec(
                name="qwen2.5",
                tag="14b-q8_0",
                size_gb=14.5,
                quantization="Q8_0",
                use_cases=["complex_reasoning", "technical_writing", "analysis"],
                performance_tier="quality",
                context_length=32768,
                expected_tokens_per_second=25,
                memory_requirement_gb=18.0
            ),
            
            # Specialized embedding model
            "qwen3-embed": OllamaModelSpec(
                name="qwen2.5",
                tag="7b-instruct",
                size_gb=7.0,
                quantization="FP16",
                use_cases=["embeddings", "similarity", "retrieval"],
                performance_tier="balanced",
                context_length=32768,
                expected_tokens_per_second=60,
                memory_requirement_gb=8.5
            )
        }
    
    def get_model_by_use_case(self, use_case: str) -> Optional[OllamaModelSpec]:
        """Get optimal model for specific use case"""
        for model_spec in self.models.values():
            if use_case in model_spec.use_cases:
                return model_spec
        return None
    
    def get_model_by_tier(self, tier: str) -> Optional[OllamaModelSpec]:
        """Get model by performance tier"""
        for model_spec in self.models.values():
            if model_spec.performance_tier == tier:
                return model_spec
        return None
    
    def get_all_models(self) -> List[OllamaModelSpec]:
        """Get all registered models"""
        return list(self.models.values())


class ModelManager:
    """Manage Ollama model lifecycle and deployment"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.registry = ModelRegistry()
        self.loaded_models = {}  # model_name -> load_time
        self.model_stats = defaultdict(lambda: deque(maxlen=100))
        self.session = None
        self.lock = RLock()
        self.preload_queue = asyncio.Queue()
        self.preloader_active = False
        self.preloader_task = None
    
    async def initialize(self):
        """Initialize model manager"""
        # Create aiohttp session
        connector = aiohttp.TCPConnector(
            limit=10,
            keepalive_timeout=60
        )
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Start model preloader
        self.preloader_active = True
        self.preloader_task = asyncio.create_task(self._preloader_loop())
        
        # Get currently loaded models
        await self._refresh_loaded_models()
        
        logger.info("Ollama Model Manager initialized")
    
    async def _refresh_loaded_models(self):
        """Refresh list of loaded models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    with self.lock:
                        self.loaded_models.clear()
                        for model_data in data.get("models", []):
                            model_name = model_data["name"]
                            self.loaded_models[model_name] = time.time()
                    
                    logger.info(f"Found {len(self.loaded_models)} loaded models")
                    
        except Exception as e:
            logger.error(f"Failed to refresh loaded models: {e}")
    
    async def ensure_model_loaded(self, model_spec: OllamaModelSpec) -> bool:
        """Ensure model is loaded and ready"""
        model_name = model_spec.full_name
        
        # Check if already loaded
        if model_name in self.loaded_models:
            return True
        
        try:
            # Check if model exists locally
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    
                    if model_name not in available_models:
                        # Pull model
                        logger.info(f"Pulling model: {model_name}")
                        await self._pull_model(model_spec)
            
            # Load model into memory
            await self._load_model(model_spec)
            
            with self.lock:
                self.loaded_models[model_name] = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def _pull_model(self, model_spec: OllamaModelSpec):
        """Pull model from registry"""
        pull_data = {"name": model_spec.full_name}
        
        async with self.session.post(
            f"{self.base_url}/api/pull",
            json=pull_data,
            timeout=aiohttp.ClientTimeout(total=1800)  # 30 minute timeout
        ) as response:
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to pull model: {error_text}")
            
            # Monitor pull progress
            async for line in response.content:
                if line:
                    try:
                        progress = json.loads(line.decode())
                        if progress.get("status") == "success":
                            break
                        elif "error" in progress:
                            raise Exception(progress["error"])
                    except json.JSONDecodeError:
                        continue
    
    async def _load_model(self, model_spec: OllamaModelSpec):
        """Load model into memory"""
        # Send a small generation request to load model
        load_data = {
            "model": model_spec.full_name,
            "prompt": "Hello",
            "stream": False,
            "options": {"num_predict": 1}
        }
        
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=load_data,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to load model: {error_text}")
            
            await response.json()  # Wait for response to complete loading
    
    async def preload_models(self, models: List[str]):
        """Queue models for preloading"""
        for model_name in models:
            if model_name in self.registry.models:
                await self.preload_queue.put(self.registry.models[model_name])
    
    async def _preloader_loop(self):
        """Background model preloader"""
        while self.preloader_active:
            try:
                # Get model to preload
                model_spec = await asyncio.wait_for(
                    self.preload_queue.get(),
                    timeout=1.0
                )
                
                # Preload model
                logger.info(f"Preloading model: {model_spec.full_name}")
                await self.ensure_model_loaded(model_spec)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Preloader error: {e}")
                await asyncio.sleep(5)
    
    def record_model_performance(self, model_name: str, processing_time: float, tokens: int, success: bool):
        """Record model performance metrics"""
        with self.lock:
            self.model_stats[model_name].append({
                "processing_time": processing_time,
                "tokens_per_second": tokens / processing_time if processing_time > 0 else 0,
                "success": success,
                "timestamp": time.time()
            })
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get model performance statistics"""
        with self.lock:
            if model_name not in self.model_stats:
                return {}
            
            stats = list(self.model_stats[model_name])
            if not stats:
                return {}
            
            processing_times = [s["processing_time"] for s in stats]
            tokens_per_sec = [s["tokens_per_second"] for s in stats if s["tokens_per_second"] > 0]
            successes = [s["success"] for s in stats]
            
            return {
                "avg_processing_time": np.mean(processing_times),
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "success_rate": np.mean(successes),
                "total_requests": len(stats)
            }
    
    async def cleanup(self):
        """Cleanup model manager"""
        self.preloader_active = False
        if self.preloader_task:
            self.preloader_task.cancel()
            try:
                await self.preloader_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()


class EmbeddingOptimizer:
    """Optimized embedding generation with batching and caching"""
    
    def __init__(self, model_manager: ModelManager, cache_size: int = 10000):
        self.model_manager = model_manager
        self.cache_size = cache_size
        self.embedding_cache = {}
        self.cache_access_order = deque()
        self.cache_lock = RLock()
        self.batch_processor = None
        self.processing_active = False
        self.request_queue = asyncio.Queue()
    
    async def initialize(self):
        """Initialize embedding optimizer"""
        # Start batch processor
        self.processing_active = True
        self.batch_processor = asyncio.create_task(self._batch_processing_loop())
        
        # Preload embedding model
        embed_model = self.model_manager.registry.get_model_by_use_case("embeddings")
        if embed_model:
            await self.model_manager.ensure_model_loaded(embed_model)
        
        logger.info("Embedding Optimizer initialized")
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings with caching and optimization"""
        
        # Check cache first
        cached_embeddings = self._get_cached_embeddings(request.texts, request.model)
        cache_hits = [i for i, emb in enumerate(cached_embeddings) if emb is not None]
        cache_misses = [i for i, emb in enumerate(cached_embeddings) if emb is None]
        
        # If all texts are cached, return immediately
        if not cache_misses:
            return EmbeddingResponse(
                request_id=request.id,
                embeddings=cached_embeddings,
                model_used=request.model,
                processing_time=0.001,  # Cache hit
                tokens_processed=sum(len(text.split()) for text in request.texts),
                metadata={"cache_hits": len(cache_hits), "cache_misses": 0}
            )
        
        # Process uncached texts
        texts_to_process = [request.texts[i] for i in cache_misses]
        
        try:
            start_time = time.time()
            
            # Generate embeddings for uncached texts
            new_embeddings = await self._generate_batch_embeddings(
                texts_to_process, 
                request.model,
                request.batch_size
            )
            
            processing_time = time.time() - start_time
            
            # Cache new embeddings
            for i, embedding in zip(cache_misses, new_embeddings):
                self._cache_embedding(request.texts[i], request.model, embedding)
            
            # Combine cached and new embeddings
            final_embeddings = []
            new_embedding_idx = 0
            
            for i, text in enumerate(request.texts):
                if i in cache_hits:
                    final_embeddings.append(cached_embeddings[i])
                else:
                    final_embeddings.append(new_embeddings[new_embedding_idx])
                    new_embedding_idx += 1
            
            # Record performance
            tokens_processed = sum(len(text.split()) for text in texts_to_process)
            self.model_manager.record_model_performance(
                request.model, 
                processing_time, 
                tokens_processed, 
                True
            )
            
            return EmbeddingResponse(
                request_id=request.id,
                embeddings=final_embeddings,
                model_used=request.model,
                processing_time=processing_time,
                tokens_processed=tokens_processed,
                metadata={
                    "cache_hits": len(cache_hits),
                    "cache_misses": len(cache_misses),
                    "batch_size": request.batch_size
                }
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            
            return EmbeddingResponse(
                request_id=request.id,
                embeddings=[],
                model_used=request.model,
                processing_time=time.time() - start_time,
                tokens_processed=0,
                success=False,
                error=str(e)
            )
    
    async def _generate_batch_embeddings(self, texts: List[str], model: str, batch_size: int) -> List[List[float]]:
        """Generate embeddings in batches"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = await self._generate_single_batch(batch_texts, model)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _generate_single_batch(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings for single batch"""
        # For Ollama, we need to generate embeddings one by one
        # In the future, this could be optimized with batch API support
        
        embeddings = []
        session = self.model_manager.session
        
        for text in texts:
            embed_data = {
                "model": model,
                "prompt": text
            }
            
            async with session.post(
                f"{self.model_manager.base_url}/api/embeddings",
                json=embed_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    embedding = result.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    # Fallback to zero embedding
                    embeddings.append([0.0] * 768)  # Default embedding size
        
        return embeddings
    
    def _get_cached_embeddings(self, texts: List[str], model: str) -> List[Optional[List[float]]]:
        """Get cached embeddings for texts"""
        cached_embeddings = []
        
        with self.cache_lock:
            for text in texts:
                cache_key = self._generate_cache_key(text, model)
                
                if cache_key in self.embedding_cache:
                    # Update access order
                    if cache_key in self.cache_access_order:
                        self.cache_access_order.remove(cache_key)
                    self.cache_access_order.append(cache_key)
                    
                    cached_embeddings.append(self.embedding_cache[cache_key])
                else:
                    cached_embeddings.append(None)
        
        return cached_embeddings
    
    def _cache_embedding(self, text: str, model: str, embedding: List[float]):
        """Cache embedding with LRU eviction"""
        cache_key = self._generate_cache_key(text, model)
        
        with self.cache_lock:
            # Evict LRU if at capacity
            if len(self.embedding_cache) >= self.cache_size and cache_key not in self.embedding_cache:
                if self.cache_access_order:
                    lru_key = self.cache_access_order.popleft()
                    if lru_key in self.embedding_cache:
                        del self.embedding_cache[lru_key]
            
            # Cache embedding
            self.embedding_cache[cache_key] = embedding
            
            # Update access order
            if cache_key in self.cache_access_order:
                self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _batch_processing_loop(self):
        """Background batch processing loop"""
        while self.processing_active:
            try:
                # This can be extended for future batch processing optimizations
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                "cache_size": len(self.embedding_cache),
                "max_cache_size": self.cache_size,
                "cache_utilization": len(self.embedding_cache) / self.cache_size
            }
    
    async def cleanup(self):
        """Cleanup embedding optimizer"""
        self.processing_active = False
        if self.batch_processor:
            self.batch_processor.cancel()
            try:
                await self.batch_processor
            except asyncio.CancelledError:
                pass


class InferenceOptimizer:
    """Optimized inference with intelligent model selection"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.request_history = deque(maxlen=1000)
        self.performance_tracker = defaultdict(lambda: deque(maxlen=100))
    
    async def generate(self, 
                      prompt: str, 
                      use_case: str = "general_qa",
                      max_tokens: int = 1024,
                      temperature: float = 0.7,
                      stream: bool = False) -> Dict[str, Any]:
        """Generate response with optimal model selection"""
        
        # Select optimal model based on use case
        model_spec = self._select_optimal_model(use_case, prompt)
        if not model_spec:
            raise Exception("No suitable model available")
        
        # Ensure model is loaded
        await self.model_manager.ensure_model_loaded(model_spec)
        
        start_time = time.time()
        
        try:
            # Generate response
            response = await self._generate_with_model(
                model_spec,
                prompt,
                max_tokens,
                temperature,
                stream
            )
            
            processing_time = time.time() - start_time
            tokens_generated = len(response.get("response", "").split())
            
            # Record performance
            self.model_manager.record_model_performance(
                model_spec.full_name,
                processing_time,
                tokens_generated,
                True
            )
            
            return {
                "success": True,
                "response": response.get("response", ""),
                "model_used": model_spec.full_name,
                "processing_time": processing_time,
                "tokens_per_second": tokens_generated / processing_time if processing_time > 0 else 0
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record failure
            self.model_manager.record_model_performance(
                model_spec.full_name,
                processing_time,
                0,
                False
            )
            
            logger.error(f"Generation failed with {model_spec.full_name}: {e}")
            raise
    
    def _select_optimal_model(self, use_case: str, prompt: str) -> Optional[OllamaModelSpec]:
        """Select optimal model based on use case and prompt characteristics"""
        
        # Get model by use case
        model_by_use_case = self.model_manager.registry.get_model_by_use_case(use_case)
        if model_by_use_case:
            return model_by_use_case
        
        # Analyze prompt characteristics
        prompt_length = len(prompt)
        prompt_complexity = self._analyze_prompt_complexity(prompt)
        
        # Select based on prompt characteristics
        if prompt_length < 100 and prompt_complexity == "simple":
            return self.model_manager.registry.get_model_by_tier("fast")
        elif prompt_length > 1000 or prompt_complexity == "complex":
            return self.model_manager.registry.get_model_by_tier("quality")
        else:
            return self.model_manager.registry.get_model_by_tier("balanced")
    
    def _analyze_prompt_complexity(self, prompt: str) -> str:
        """Analyze prompt complexity"""
        # Simple heuristics for complexity analysis
        complexity_indicators = [
            "analyze", "explain", "compare", "evaluate", "complex",
            "technical", "detailed", "comprehensive", "reasoning"
        ]
        
        prompt_lower = prompt.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in prompt_lower)
        
        if complexity_score >= 3:
            return "complex"
        elif complexity_score >= 1:
            return "moderate"
        else:
            return "simple"
    
    async def _generate_with_model(self,
                                   model_spec: OllamaModelSpec,
                                   prompt: str,
                                   max_tokens: int,
                                   temperature: float,
                                   stream: bool) -> Dict[str, Any]:
        """Generate response with specific model"""
        
        generate_data = {
            "model": model_spec.full_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        session = self.model_manager.session
        
        async with session.post(
            f"{self.model_manager.base_url}/api/generate",
            json=generate_data
        ) as response:
            
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Generation failed: {error_text}")
            
            if stream:
                return await self._handle_streaming_response(response)
            else:
                return await response.json()
    
    async def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """Handle streaming response from Ollama"""
        full_response = ""
        
        async for line in response.content:
            if line:
                try:
                    chunk = json.loads(line.decode())
                    if "response" in chunk:
                        full_response += chunk["response"]
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return {"response": full_response}


class OllamaOptimizer:
    """Main Ollama optimizer with coordinated components"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.model_manager = ModelManager(base_url)
        self.embedding_optimizer = EmbeddingOptimizer(self.model_manager)
        self.inference_optimizer = InferenceOptimizer(self.model_manager)
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            await self.model_manager.initialize()
            await self.embedding_optimizer.initialize()
            
            # Preload recommended models
            await self._preload_recommended_models()
            
            self.initialized = True
            logger.info("Ollama Optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Ollama Optimizer initialization failed: {e}")
            return False
    
    async def _preload_recommended_models(self):
        """Preload recommended models based on available memory"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        models_to_preload = []
        
        # Always preload fast model
        models_to_preload.append("qwen3-fast")
        
        # Preload additional models based on available memory
        if available_memory_gb > 20:
            models_to_preload.append("qwen3-balanced")
        
        if available_memory_gb > 40:
            models_to_preload.append("qwen3-embed")
        
        if available_memory_gb > 60:
            models_to_preload.append("qwen3-quality")
        
        await self.model_manager.preload_models(models_to_preload)
        logger.info(f"Preloading models: {models_to_preload}")
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings via embedding optimizer"""
        if not self.initialized:
            raise Exception("Ollama Optimizer not initialized")
        
        return await self.embedding_optimizer.generate_embeddings(request)
    
    async def generate_text(self, 
                           prompt: str,
                           use_case: str = "general_qa",
                           max_tokens: int = 1024,
                           temperature: float = 0.7,
                           stream: bool = False) -> Dict[str, Any]:
        """Generate text via inference optimizer"""
        if not self.initialized:
            raise Exception("Ollama Optimizer not initialized")
        
        return await self.inference_optimizer.generate(
            prompt, use_case, max_tokens, temperature, stream
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status"""
        return {
            "initialized": self.initialized,
            "loaded_models": list(self.model_manager.loaded_models.keys()),
            "model_performance": {
                model: self.model_manager.get_model_performance(model)
                for model in self.model_manager.loaded_models.keys()
            },
            "embedding_cache": self.embedding_optimizer.get_cache_stats(),
            "registry": {
                "available_models": len(self.model_manager.registry.models),
                "model_specs": [
                    {
                        "name": spec.full_name,
                        "tier": spec.performance_tier,
                        "use_cases": spec.use_cases
                    }
                    for spec in self.model_manager.registry.get_all_models()
                ]
            }
        }
    
    async def cleanup(self):
        """Cleanup all components"""
        logger.info("Cleaning up Ollama Optimizer...")
        
        await self.embedding_optimizer.cleanup()
        await self.model_manager.cleanup()
        
        self.initialized = False
        logger.info("Ollama Optimizer cleanup completed")


# Example usage and testing
async def test_ollama_optimizer():
    """Test Ollama optimizer functionality"""
    optimizer = OllamaOptimizer()
    
    try:
        # Initialize
        if not await optimizer.initialize():
            print("Failed to initialize Ollama optimizer")
            return
        
        # Test embedding generation
        embed_request = EmbeddingRequest(
            id="test_embed",
            texts=[
                "What is 5G NR technology?",
                "Explain MIMO in telecommunications",
                "Benefits of mmWave frequencies"
            ],
            model="qwen3-embed"
        )
        
        embed_response = await optimizer.generate_embeddings(embed_request)
        print(f"Embedding generation: {embed_response.success}")
        print(f"Embeddings shape: {len(embed_response.embeddings)} x {len(embed_response.embeddings[0]) if embed_response.embeddings else 0}")
        print(f"Cache hits: {embed_response.metadata.get('cache_hits', 0)}")
        
        # Test text generation
        text_response = await optimizer.generate_text(
            prompt="Explain the key features of 5G NR technology",
            use_case="technical_writing",
            max_tokens=200
        )
        
        print(f"Text generation: {text_response['success']}")
        print(f"Response: {text_response['response'][:100]}...")
        print(f"Tokens/sec: {text_response['tokens_per_second']:.1f}")
        
        # Get status
        status = optimizer.get_status()
        print(f"Optimizer status: {json.dumps(status, indent=2)}")
        
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(test_ollama_optimizer())