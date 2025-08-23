#!/usr/bin/env python3
"""
Comprehensive unit tests for Ollama Optimizer
Tests model management, embedding optimization, inference routing, and performance monitoring.
"""

import pytest
import asyncio
import json
import time
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import asdict
from collections import deque

# Import modules to test
import sys
sys.path.append('/Users/cedric/orange/ran-llm-claude-flow')

from docs.python.integration.ollama_optimizer import (
    OllamaModelSpec, EmbeddingRequest, EmbeddingResponse,
    ModelRegistry, ModelManager, EmbeddingOptimizer, 
    InferenceOptimizer, OllamaOptimizer
)

class TestOllamaModelSpec:
    """Test OllamaModelSpec dataclass"""
    
    def test_model_spec_creation(self):
        """Test model specification creation"""
        spec = OllamaModelSpec(
            name="qwen2.5",
            tag="7b-q8_0",
            size_gb=7.2,
            quantization="Q8_0",
            use_cases=["embeddings", "general_qa"],
            performance_tier="balanced",
            context_length=32768,
            expected_tokens_per_second=45.0,
            memory_requirement_gb=9.0
        )
        
        assert spec.name == "qwen2.5"
        assert spec.tag == "7b-q8_0"
        assert spec.full_name == "qwen2.5:7b-q8_0"
        assert spec.performance_tier == "balanced"
        assert "embeddings" in spec.use_cases
    
    def test_full_name_property(self):
        """Test full name property generation"""
        spec = OllamaModelSpec(
            name="test-model",
            tag="1.5b-q4",
            size_gb=1.5,
            quantization="Q4_0",
            use_cases=["test"],
            performance_tier="fast",
            context_length=8192,
            expected_tokens_per_second=100.0,
            memory_requirement_gb=2.0
        )
        
        assert spec.full_name == "test-model:1.5b-q4"

class TestEmbeddingRequest:
    """Test EmbeddingRequest dataclass"""
    
    def test_embedding_request_creation(self):
        """Test embedding request creation"""
        request = EmbeddingRequest(
            id="embed_test",
            texts=["Text 1", "Text 2", "Text 3"],
            model="qwen3-embed",
            normalize=True,
            batch_size=16,
            priority="high"
        )
        
        assert request.id == "embed_test"
        assert len(request.texts) == 3
        assert request.normalize is True
        assert request.batch_size == 16
        assert request.metadata is not None
    
    def test_auto_id_generation(self):
        """Test automatic ID generation for embeddings"""
        request = EmbeddingRequest(
            id=None,
            texts=["Test text"],
            model="test-model"
        )
        
        assert request.id is not None
        assert request.id.startswith("embed_")

class TestEmbeddingResponse:
    """Test EmbeddingResponse dataclass"""
    
    def test_successful_embedding_response(self):
        """Test successful embedding response"""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        response = EmbeddingResponse(
            request_id="embed_test",
            embeddings=embeddings,
            model_used="qwen3-embed",
            processing_time=1.5,
            tokens_processed=50,
            success=True
        )
        
        assert response.success is True
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 3
        assert response.tokens_processed == 50
    
    def test_error_embedding_response(self):
        """Test error embedding response"""
        response = EmbeddingResponse(
            request_id="embed_error",
            embeddings=[],
            model_used="qwen3-embed",
            processing_time=0.1,
            tokens_processed=0,
            success=False,
            error="Model not available"
        )
        
        assert response.success is False
        assert response.error == "Model not available"
        assert len(response.embeddings) == 0

class TestModelRegistry:
    """Test ModelRegistry functionality"""
    
    @pytest.fixture
    def registry(self):
        """Create model registry for testing"""
        return ModelRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry contains expected models"""
        models = registry.get_all_models()
        assert len(models) >= 4  # Should have fast, balanced, quality, embed models
        
        # Check for expected tiers
        tiers = [model.performance_tier for model in models]
        assert "fast" in tiers
        assert "balanced" in tiers
        assert "quality" in tiers
    
    def test_get_model_by_use_case(self, registry):
        """Test retrieving model by use case"""
        embedding_model = registry.get_model_by_use_case("embeddings")
        assert embedding_model is not None
        assert "embeddings" in embedding_model.use_cases
        
        qa_model = registry.get_model_by_use_case("general_qa")
        assert qa_model is not None
        assert "general_qa" in qa_model.use_cases
        
        # Test non-existent use case
        unknown_model = registry.get_model_by_use_case("unknown_task")
        assert unknown_model is None
    
    def test_get_model_by_tier(self, registry):
        """Test retrieving model by performance tier"""
        fast_model = registry.get_model_by_tier("fast")
        assert fast_model is not None
        assert fast_model.performance_tier == "fast"
        
        quality_model = registry.get_model_by_tier("quality")
        assert quality_model is not None
        assert quality_model.performance_tier == "quality"
        
        # Test non-existent tier
        unknown_tier = registry.get_model_by_tier("unknown_tier")
        assert unknown_tier is None

class TestModelManager:
    """Test ModelManager functionality"""
    
    @pytest.fixture
    def mock_ollama_session(self):
        """Mock Ollama HTTP session for testing"""
        session = AsyncMock()
        
        # Mock tags response (list models)
        mock_tags_response = AsyncMock()
        mock_tags_response.status = 200
        mock_tags_response.json = AsyncMock(return_value={
            "models": [
                {"name": "qwen2.5:1.5b-q4_0", "size": 1200000000},
                {"name": "qwen2.5:7b-q8_0", "size": 7200000000},
                {"name": "qwen2.5:14b-q8_0", "size": 14500000000}
            ]
        })
        
        # Mock generate response (for model loading)
        mock_generate_response = AsyncMock()
        mock_generate_response.status = 200
        mock_generate_response.json = AsyncMock(return_value={
            "response": "Hello",
            "done": True
        })
        
        # Mock pull response (for model pulling)
        mock_pull_response = AsyncMock()
        mock_pull_response.status = 200
        mock_pull_response.content = AsyncMock()
        mock_pull_response.content.__aiter__ = AsyncMock(return_value=[
            b'{"status": "downloading"}',
            b'{"status": "success"}'
        ])
        
        # Configure session responses
        session.get.return_value.__aenter__.return_value = mock_tags_response
        session.post.return_value.__aenter__.return_value = mock_generate_response
        
        return session
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager for testing"""
        return ModelManager()
    
    @pytest.mark.asyncio
    async def test_initialization(self, model_manager, mock_ollama_session):
        """Test model manager initialization"""
        with patch('aiohttp.ClientSession', return_value=mock_ollama_session):
            await model_manager.initialize()
            
            assert model_manager.session is not None
            assert model_manager.preloader_active is True
            assert len(model_manager.loaded_models) == 3  # From mock response
    
    @pytest.mark.asyncio
    async def test_refresh_loaded_models(self, model_manager, mock_ollama_session):
        """Test refreshing loaded models"""
        model_manager.session = mock_ollama_session
        
        await model_manager._refresh_loaded_models()
        
        assert len(model_manager.loaded_models) == 3
        assert "qwen2.5:1.5b-q4_0" in model_manager.loaded_models
        assert "qwen2.5:7b-q8_0" in model_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_ensure_model_loaded_already_loaded(self, model_manager):
        """Test ensuring model is loaded when already loaded"""
        spec = OllamaModelSpec(
            name="qwen2.5",
            tag="7b-q8_0",
            size_gb=7.2,
            quantization="Q8_0",
            use_cases=["test"],
            performance_tier="balanced",
            context_length=32768,
            expected_tokens_per_second=45.0,
            memory_requirement_gb=9.0
        )
        
        # Model already loaded
        model_manager.loaded_models[spec.full_name] = time.time()
        
        result = await model_manager.ensure_model_loaded(spec)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_model_loaded_new_model(self, model_manager, mock_ollama_session):
        """Test ensuring model is loaded for new model"""
        model_manager.session = mock_ollama_session
        
        spec = OllamaModelSpec(
            name="qwen2.5",
            tag="new-model",
            size_gb=5.0,
            quantization="Q8_0",
            use_cases=["test"],
            performance_tier="balanced",
            context_length=32768,
            expected_tokens_per_second=45.0,
            memory_requirement_gb=6.0
        )
        
        # Mock available models check
        mock_ollama_session.get.return_value.__aenter__.return_value.json.return_value = {
            "models": []  # Model not available locally
        }
        
        # Mock successful pull and load
        mock_pull_response = AsyncMock()
        mock_pull_response.status = 200
        mock_pull_response.content.__aiter__.return_value = [b'{"status": "success"}']
        
        with patch.object(model_manager.session, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_pull_response
            
            result = await model_manager.ensure_model_loaded(spec)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_preload_models(self, model_manager):
        """Test model preloading queue"""
        models_to_preload = ["qwen3-fast", "qwen3-balanced"]
        
        await model_manager.preload_models(models_to_preload)
        
        # Should have queued models
        assert not model_manager.preload_queue.empty()
    
    def test_record_model_performance(self, model_manager):
        """Test performance recording"""
        model_manager.record_model_performance(
            model_name="qwen2.5:7b-q8_0",
            processing_time=2.0,
            tokens=40,
            success=True
        )
        
        stats = model_manager.get_model_performance("qwen2.5:7b-q8_0")
        assert stats["total_requests"] == 1
        assert stats["avg_processing_time"] == 2.0
        assert stats["success_rate"] == 1.0

class TestEmbeddingOptimizer:
    """Test EmbeddingOptimizer functionality"""
    
    @pytest.fixture
    def mock_model_manager(self, mock_ollama_session):
        """Create mock model manager for testing"""
        manager = MagicMock()
        manager.session = mock_ollama_session
        manager.base_url = "http://127.0.0.1:11434"
        manager.registry = ModelRegistry()
        manager.ensure_model_loaded = AsyncMock(return_value=True)
        manager.record_model_performance = MagicMock()
        return manager
    
    @pytest.fixture
    def embedding_optimizer(self, mock_model_manager):
        """Create embedding optimizer for testing"""
        return EmbeddingOptimizer(mock_model_manager, cache_size=100)
    
    @pytest.mark.asyncio
    async def test_initialization(self, embedding_optimizer):
        """Test embedding optimizer initialization"""
        await embedding_optimizer.initialize()
        
        assert embedding_optimizer.processing_active is True
        assert embedding_optimizer.batch_processor is not None
    
    def test_cache_key_generation(self, embedding_optimizer):
        """Test cache key generation"""
        key = embedding_optimizer._generate_cache_key("test text", "qwen3-embed")
        
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hash length
        
        # Same input should generate same key
        key2 = embedding_optimizer._generate_cache_key("test text", "qwen3-embed")
        assert key == key2
        
        # Different input should generate different key
        key3 = embedding_optimizer._generate_cache_key("different text", "qwen3-embed")
        assert key != key3
    
    def test_cache_embedding(self, embedding_optimizer):
        """Test embedding caching"""
        text = "test text"
        model = "qwen3-embed"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        embedding_optimizer._cache_embedding(text, model, embedding)
        
        cached_embeddings = embedding_optimizer._get_cached_embeddings([text], model)
        assert cached_embeddings[0] == embedding
    
    def test_cache_lru_eviction(self, embedding_optimizer):
        """Test LRU cache eviction"""
        # Set small cache size
        embedding_optimizer.cache_size = 2
        
        # Add embeddings to fill cache
        texts = ["text1", "text2", "text3"]
        model = "qwen3-embed"
        
        for i, text in enumerate(texts):
            embedding = [float(i)] * 5
            embedding_optimizer._cache_embedding(text, model, embedding)
        
        # First text should be evicted (LRU)
        cached_embeddings = embedding_optimizer._get_cached_embeddings(texts, model)
        assert cached_embeddings[0] is None  # text1 evicted
        assert cached_embeddings[1] is not None  # text2 still cached
        assert cached_embeddings[2] is not None  # text3 still cached
    
    @pytest.mark.asyncio
    async def test_generate_single_batch(self, embedding_optimizer, mock_ollama_session):
        """Test single batch embedding generation"""
        texts = ["text1", "text2"]
        model = "qwen3-embed"
        
        # Mock embeddings response
        mock_embeddings_response = AsyncMock()
        mock_embeddings_response.status = 200
        mock_embeddings_response.json = AsyncMock(return_value={
            "embedding": [0.1, 0.2, 0.3] * 256  # 768-dim embedding
        })
        
        mock_ollama_session.post.return_value.__aenter__.return_value = mock_embeddings_response
        
        embeddings = await embedding_optimizer._generate_single_batch(texts, model)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 768
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_all_cached(self, embedding_optimizer):
        """Test embedding generation with all texts cached"""
        texts = ["cached text 1", "cached text 2"]
        model = "qwen3-embed"
        
        # Pre-cache embeddings
        for i, text in enumerate(texts):
            embedding = [float(i)] * 768
            embedding_optimizer._cache_embedding(text, model, embedding)
        
        request = EmbeddingRequest(
            id="test_cached",
            texts=texts,
            model=model
        )
        
        response = await embedding_optimizer.generate_embeddings(request)
        
        assert response.success is True
        assert len(response.embeddings) == 2
        assert response.processing_time < 0.01  # Should be very fast (cache hit)
        assert response.metadata["cache_hits"] == 2
        assert response.metadata["cache_misses"] == 0
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_mixed_cache(self, embedding_optimizer, mock_ollama_session):
        """Test embedding generation with partial cache hits"""
        texts = ["cached text", "new text"]
        model = "qwen3-embed"
        
        # Pre-cache one embedding
        embedding_optimizer._cache_embedding(texts[0], model, [0.1] * 768)
        
        # Mock API response for new text
        mock_embeddings_response = AsyncMock()
        mock_embeddings_response.status = 200
        mock_embeddings_response.json = AsyncMock(return_value={
            "embedding": [0.2] * 768
        })
        
        mock_ollama_session.post.return_value.__aenter__.return_value = mock_embeddings_response
        
        request = EmbeddingRequest(
            id="test_mixed",
            texts=texts,
            model=model
        )
        
        response = await embedding_optimizer.generate_embeddings(request)
        
        assert response.success is True
        assert len(response.embeddings) == 2
        assert response.metadata["cache_hits"] == 1
        assert response.metadata["cache_misses"] == 1
    
    def test_get_cache_stats(self, embedding_optimizer):
        """Test cache statistics"""
        # Add some cached embeddings
        for i in range(5):
            embedding_optimizer._cache_embedding(f"text_{i}", "model", [float(i)] * 768)
        
        stats = embedding_optimizer.get_cache_stats()
        
        assert "cache_size" in stats
        assert "max_cache_size" in stats
        assert "cache_utilization" in stats
        assert stats["cache_size"] == 5

class TestInferenceOptimizer:
    """Test InferenceOptimizer functionality"""
    
    @pytest.fixture
    def mock_model_manager(self, mock_ollama_session):
        """Create mock model manager for inference testing"""
        manager = MagicMock()
        manager.session = mock_ollama_session
        manager.base_url = "http://127.0.0.1:11434"
        manager.registry = ModelRegistry()
        manager.ensure_model_loaded = AsyncMock(return_value=True)
        manager.record_model_performance = MagicMock()
        return manager
    
    @pytest.fixture
    def inference_optimizer(self, mock_model_manager):
        """Create inference optimizer for testing"""
        return InferenceOptimizer(mock_model_manager)
    
    def test_analyze_prompt_complexity(self, inference_optimizer):
        """Test prompt complexity analysis"""
        # Simple prompt
        simple_prompt = "What is AI?"
        complexity = inference_optimizer._analyze_prompt_complexity(simple_prompt)
        assert complexity == "simple"
        
        # Moderate complexity
        moderate_prompt = "Explain the concept of artificial intelligence"
        complexity = inference_optimizer._analyze_prompt_complexity(moderate_prompt)
        assert complexity == "moderate"
        
        # Complex prompt
        complex_prompt = "Analyze and compare the technical architecture of different neural network approaches, evaluate their performance characteristics, and provide detailed reasoning about their implementation complexity"
        complexity = inference_optimizer._analyze_prompt_complexity(complex_prompt)
        assert complexity == "complex"
    
    def test_select_optimal_model(self, inference_optimizer):
        """Test optimal model selection"""
        # Test use case based selection
        model = inference_optimizer._select_optimal_model("embeddings", "Generate embeddings")
        assert model is not None
        assert "embeddings" in model.use_cases
        
        # Test prompt characteristic based selection
        # Short simple prompt
        model = inference_optimizer._select_optimal_model("general_qa", "Hi")
        assert model.performance_tier == "fast"
        
        # Long complex prompt  
        long_prompt = "a" * 1500 + " analyze and explain in detail"
        model = inference_optimizer._select_optimal_model("general_qa", long_prompt)
        assert model.performance_tier == "quality"
    
    @pytest.mark.asyncio
    async def test_generate_with_model(self, inference_optimizer, mock_ollama_session):
        """Test generation with specific model"""
        spec = OllamaModelSpec(
            name="qwen2.5",
            tag="7b-q8_0",
            size_gb=7.2,
            quantization="Q8_0",
            use_cases=["general_qa"],
            performance_tier="balanced",
            context_length=32768,
            expected_tokens_per_second=45.0,
            memory_requirement_gb=9.0
        )
        
        # Mock successful generation response
        mock_generate_response = AsyncMock()
        mock_generate_response.status = 200
        mock_generate_response.json = AsyncMock(return_value={
            "response": "This is a test response from Ollama",
            "done": True
        })
        
        mock_ollama_session.post.return_value.__aenter__.return_value = mock_generate_response
        
        result = await inference_optimizer._generate_with_model(
            spec, 
            "Test prompt", 
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        assert result["response"] == "This is a test response from Ollama"
    
    @pytest.mark.asyncio
    async def test_generate_full_flow(self, inference_optimizer, mock_ollama_session):
        """Test full generation flow with model selection"""
        # Mock successful generation
        mock_generate_response = AsyncMock()
        mock_generate_response.status = 200
        mock_generate_response.json = AsyncMock(return_value={
            "response": "Full flow test response",
            "done": True
        })
        
        mock_ollama_session.post.return_value.__aenter__.return_value = mock_generate_response
        
        result = await inference_optimizer.generate(
            prompt="What is machine learning?",
            use_case="general_qa",
            max_tokens=200,
            temperature=0.7
        )
        
        assert result["success"] is True
        assert result["response"] == "Full flow test response"
        assert "model_used" in result
        assert "tokens_per_second" in result

class TestOllamaOptimizer:
    """Test main OllamaOptimizer functionality"""
    
    @pytest.fixture
    def mock_components(self, mock_ollama_session):
        """Mock optimizer components"""
        with patch('docs.python.integration.ollama_optimizer.ModelManager') as MockModelManager:
            with patch('docs.python.integration.ollama_optimizer.EmbeddingOptimizer') as MockEmbeddingOptimizer:
                with patch('docs.python.integration.ollama_optimizer.InferenceOptimizer') as MockInferenceOptimizer:
                    
                    # Configure mocks
                    mock_model_manager = MockModelManager.return_value
                    mock_model_manager.initialize = AsyncMock()
                    mock_model_manager.preload_models = AsyncMock()
                    mock_model_manager.loaded_models = {"qwen2.5:7b-q8_0": time.time()}
                    mock_model_manager.get_model_performance = MagicMock(return_value={"avg_tokens_per_second": 45.0})
                    
                    mock_embedding_optimizer = MockEmbeddingOptimizer.return_value
                    mock_embedding_optimizer.initialize = AsyncMock()
                    mock_embedding_optimizer.generate_embeddings = AsyncMock()
                    mock_embedding_optimizer.get_cache_stats = MagicMock(return_value={"cache_size": 10})
                    
                    mock_inference_optimizer = MockInferenceOptimizer.return_value
                    mock_inference_optimizer.generate = AsyncMock()
                    
                    yield {
                        'model_manager': mock_model_manager,
                        'embedding_optimizer': mock_embedding_optimizer,
                        'inference_optimizer': mock_inference_optimizer
                    }
    
    @pytest.fixture
    def optimizer(self, mock_components):
        """Create optimizer for testing"""
        return OllamaOptimizer()
    
    @pytest.mark.asyncio
    async def test_initialization(self, optimizer, mock_components):
        """Test optimizer initialization"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 64 * 1024**3  # 64GB available
            
            success = await optimizer.initialize()
            assert success is True
            assert optimizer.initialized is True
    
    @pytest.mark.asyncio
    async def test_preload_recommended_models(self, optimizer, mock_components):
        """Test recommended model preloading based on memory"""
        with patch('psutil.virtual_memory') as mock_memory:
            # Test with high memory (should preload all models)
            mock_memory.return_value.available = 80 * 1024**3  # 80GB
            
            await optimizer._preload_recommended_models()
            
            # Should have called preload_models with multiple models
            optimizer.model_manager.preload_models.assert_called_once()
            called_args = optimizer.model_manager.preload_models.call_args[0][0]
            assert len(called_args) >= 3  # Should preload multiple models
    
    @pytest.mark.asyncio
    async def test_preload_with_limited_memory(self, optimizer, mock_components):
        """Test model preloading with limited memory"""
        with patch('psutil.virtual_memory') as mock_memory:
            # Test with limited memory (should only preload fast model)
            mock_memory.return_value.available = 10 * 1024**3  # 10GB
            
            await optimizer._preload_recommended_models()
            
            optimizer.model_manager.preload_models.assert_called_once()
            called_args = optimizer.model_manager.preload_models.call_args[0][0]
            assert "qwen3-fast" in called_args
            assert len(called_args) == 1  # Only fast model
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, optimizer, mock_components):
        """Test embedding generation through optimizer"""
        optimizer.initialized = True
        
        request = EmbeddingRequest(
            id="test_optimizer_embed",
            texts=["test text"],
            model="qwen3-embed"
        )
        
        mock_response = EmbeddingResponse(
            request_id="test_optimizer_embed",
            embeddings=[[0.1, 0.2, 0.3]],
            model_used="qwen3-embed",
            processing_time=1.0,
            tokens_processed=10
        )
        
        optimizer.embedding_optimizer.generate_embeddings.return_value = mock_response
        
        response = await optimizer.generate_embeddings(request)
        
        assert response.success is True
        optimizer.embedding_optimizer.generate_embeddings.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_generate_text(self, optimizer, mock_components):
        """Test text generation through optimizer"""
        optimizer.initialized = True
        
        mock_result = {
            "success": True,
            "response": "Generated text response",
            "model_used": "qwen2.5:7b-q8_0",
            "tokens_per_second": 45.0
        }
        
        optimizer.inference_optimizer.generate.return_value = mock_result
        
        result = await optimizer.generate_text(
            prompt="Test prompt",
            use_case="general_qa",
            max_tokens=200
        )
        
        assert result["success"] is True
        assert result["response"] == "Generated text response"
        optimizer.inference_optimizer.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_not_initialized_errors(self, optimizer):
        """Test error handling when not initialized"""
        optimizer.initialized = False
        
        # Test embedding generation
        request = EmbeddingRequest(id="test", texts=["test"], model="test")
        with pytest.raises(Exception, match="not initialized"):
            await optimizer.generate_embeddings(request)
        
        # Test text generation
        with pytest.raises(Exception, match="not initialized"):
            await optimizer.generate_text("test prompt")
    
    def test_get_status(self, optimizer, mock_components):
        """Test status reporting"""
        optimizer.initialized = True
        
        status = optimizer.get_status()
        
        expected_keys = ["initialized", "loaded_models", "model_performance", "embedding_cache", "registry"]
        assert all(key in status for key in expected_keys)
        assert status["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer, mock_components):
        """Test optimizer cleanup"""
        optimizer.initialized = True
        
        await optimizer.cleanup()
        
        assert optimizer.initialized is False
        optimizer.embedding_optimizer.cleanup.assert_called_once()
        optimizer.model_manager.cleanup.assert_called_once()

class TestPerformanceAndConcurrency:
    """Test performance optimization and concurrent operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_requests(self, mock_ollama_session):
        """Test concurrent embedding generation"""
        optimizer = OllamaOptimizer()
        
        with patch.object(optimizer.model_manager, 'initialize'):
            with patch.object(optimizer.embedding_optimizer, 'initialize'):
                await optimizer.initialize()
                optimizer.initialized = True
                
                # Mock embedding responses
                mock_response = EmbeddingResponse(
                    request_id="test",
                    embeddings=[[0.1] * 768],
                    model_used="qwen3-embed",
                    processing_time=0.5,
                    tokens_processed=10
                )
                
                with patch.object(optimizer.embedding_optimizer, 'generate_embeddings', return_value=mock_response):
                    # Create concurrent requests
                    requests = [
                        EmbeddingRequest(id=f"concurrent_{i}", texts=[f"text {i}"], model="qwen3-embed")
                        for i in range(5)
                    ]
                    
                    # Process concurrently
                    start_time = time.time()
                    tasks = [optimizer.generate_embeddings(req) for req in requests]
                    responses = await asyncio.gather(*tasks)
                    total_time = time.time() - start_time
                    
                    # Verify all completed
                    assert len(responses) == 5
                    assert all(r.success for r in responses)
                    
                    # Should be faster than sequential
                    assert total_time < len(requests) * 0.5

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test handling of initialization failures"""
        optimizer = OllamaOptimizer()
        
        with patch.object(optimizer.model_manager, 'initialize', side_effect=Exception("Init failed")):
            success = await optimizer.initialize()
            assert success is False
            assert optimizer.initialized is False
    
    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, mock_ollama_session):
        """Test handling of embedding generation failures"""
        manager = ModelManager()
        manager.session = mock_ollama_session
        
        # Mock API error response
        mock_error_response = AsyncMock()
        mock_error_response.status = 500
        mock_error_response.text = AsyncMock(return_value="Internal Server Error")
        
        mock_ollama_session.post.return_value.__aenter__.return_value = mock_error_response
        
        optimizer = EmbeddingOptimizer(manager)
        
        request = EmbeddingRequest(
            id="error_test",
            texts=["test text"],
            model="qwen3-embed"
        )
        
        response = await optimizer.generate_embeddings(request)
        
        assert response.success is False
        assert "error" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self, mock_ollama_session):
        """Test handling of model loading failures"""
        manager = ModelManager()
        manager.session = mock_ollama_session
        
        spec = OllamaModelSpec(
            name="failing-model",
            tag="1b",
            size_gb=1.0,
            quantization="Q4_0",
            use_cases=["test"],
            performance_tier="fast",
            context_length=8192,
            expected_tokens_per_second=100.0,
            memory_requirement_gb=2.0
        )
        
        # Mock API failure
        mock_ollama_session.get.return_value.__aenter__.return_value.status = 500
        
        result = await manager.ensure_model_loaded(spec)
        assert result is False

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])