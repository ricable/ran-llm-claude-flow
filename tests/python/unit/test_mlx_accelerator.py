#!/usr/bin/env python3
"""
Comprehensive unit tests for MLX Accelerator
Tests direct MLX model integration, memory management, performance optimization, and M3 Max specific features.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import modules to test  
import sys
sys.path.append('/Users/cedric/orange/ran-llm-claude-flow')

from docs.python.integration.mlx_accelerator import (
    MLXModelConfig, MLXInferenceRequest, MLXInferenceResponse,
    MLXModelRegistry, MLXModelManager, MLXInferenceEngine, MLXAccelerator
)

class TestMLXModelConfig:
    """Test MLXModelConfig dataclass"""
    
    def test_model_config_creation(self):
        """Test model configuration creation"""
        config = MLXModelConfig(
            name="Qwen3-7B",
            path="/path/to/model", 
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["general_qa", "analysis"],
            expected_tokens_per_second=45.0
        )
        
        assert config.name == "Qwen3-7B"
        assert config.variant == "7b"
        assert config.display_name == "qwen3-7b-8bit"
        assert config.memory_gb == 8.0
        assert "general_qa" in config.use_cases
    
    def test_display_name_property(self):
        """Test display name property generation"""
        config = MLXModelConfig(
            name="Test Model",
            path="/test",
            variant="14b",
            quantization="4bit",
            memory_gb=4.0,
            context_length=8192,
            use_cases=["test"]
        )
        
        assert config.display_name == "qwen3-14b-4bit"

class TestMLXInferenceRequest:
    """Test MLXInferenceRequest dataclass"""
    
    def test_request_creation(self):
        """Test inference request creation"""
        request = MLXInferenceRequest(
            id="test_123",
            prompt="What is machine learning?",
            model_variant="7b",
            max_tokens=200,
            temperature=0.8,
            stream=True
        )
        
        assert request.id == "test_123"
        assert request.model_variant == "7b"
        assert request.stream is True
        assert request.metadata is not None
    
    def test_auto_id_generation(self):
        """Test automatic ID generation"""
        request = MLXInferenceRequest(
            id=None,
            prompt="Test prompt"
        )
        
        assert request.id is not None
        assert request.id.startswith("mlx_")

class TestMLXInferenceResponse:
    """Test MLXInferenceResponse dataclass"""
    
    def test_successful_response(self):
        """Test successful response creation"""
        response = MLXInferenceResponse(
            request_id="test_123",
            text="Machine learning is a subset of AI...",
            model_used="qwen3-7b-8bit",
            processing_time=2.5,
            tokens_generated=50,
            tokens_per_second=20.0,
            memory_usage_gb=8.2,
            success=True
        )
        
        assert response.success is True
        assert response.tokens_per_second == 20.0
        assert response.memory_usage_gb == 8.2
        assert response.error is None
    
    def test_error_response(self):
        """Test error response creation"""
        response = MLXInferenceResponse(
            request_id="test_123",
            text="",
            model_used="qwen3-7b-8bit",
            processing_time=0.1,
            tokens_generated=0,
            tokens_per_second=0.0,
            memory_usage_gb=0.0,
            success=False,
            error="Model loading failed"
        )
        
        assert response.success is False
        assert response.error == "Model loading failed"
        assert response.tokens_generated == 0

class TestMLXModelRegistry:
    """Test MLXModelRegistry functionality"""
    
    @pytest.fixture
    def registry(self):
        """Create model registry for testing"""
        return MLXModelRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry contains expected models"""
        models = registry.get_all_models()
        assert len(models) > 0
        
        # Check for expected model variants
        variants = [model.variant for model in models]
        assert "1.7b" in variants
        assert "7b" in variants
        assert "14b" in variants
    
    def test_get_model_by_variant(self, registry):
        """Test retrieving model by variant"""
        model_7b = registry.get_model_by_variant("7b")
        assert model_7b is not None
        assert model_7b.variant == "7b"
        
        # Test non-existent variant
        model_unknown = registry.get_model_by_variant("unknown")
        assert model_unknown is None
    
    def test_get_model_by_use_case(self, registry):
        """Test retrieving model by use case"""
        model_qa = registry.get_model_by_use_case("general_qa")
        assert model_qa is not None
        assert "general_qa" in model_qa.use_cases
        
        # Test non-existent use case
        model_unknown = registry.get_model_by_use_case("unknown_use_case")
        assert model_unknown is None
    
    def test_get_models_by_memory_constraint(self, registry):
        """Test filtering models by memory constraint"""
        # Get models that fit in 10GB
        small_models = registry.get_models_by_memory_constraint(10.0)
        assert all(model.memory_gb <= 10.0 for model in small_models)
        
        # Get models that fit in 1GB (should be empty or very few)
        tiny_models = registry.get_models_by_memory_constraint(1.0)
        assert len(tiny_models) == 0 or all(model.memory_gb <= 1.0 for model in tiny_models)

class TestMLXModelManager:
    """Test MLXModelManager functionality"""
    
    @pytest.fixture
    def mock_mlx_available(self):
        """Mock MLX availability and functions"""
        with patch('docs.python.integration.mlx_accelerator.MLX_AVAILABLE', True):
            # Mock MLX core functions
            mock_mx = MagicMock()
            mock_mx.default_device.return_value = "gpu"
            mock_mx.metal.get_memory_limit.return_value = 128 * 1024**3
            mock_mx.metal.clear_cache = MagicMock()
            
            with patch('docs.python.integration.mlx_accelerator.mx', mock_mx):
                # Mock MLX-LM load function
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                
                with patch('docs.python.integration.mlx_accelerator.load', return_value=(mock_model, mock_tokenizer)):
                    yield {
                        'mx': mock_mx,
                        'model': mock_model,
                        'tokenizer': mock_tokenizer
                    }
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager for testing"""
        return MLXModelManager()
    
    @pytest.mark.asyncio
    async def test_initialization_without_mlx(self, model_manager):
        """Test initialization when MLX is not available"""
        with patch('docs.python.integration.mlx_accelerator.MLX_AVAILABLE', False):
            success = await model_manager.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_initialization_with_mlx(self, model_manager, mock_mlx_available):
        """Test successful initialization with MLX"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 64 * 1024**3  # 64GB available
            
            success = await model_manager.initialize()
            assert success is True
            assert model_manager.preloader_active is True
    
    def test_load_model_sync(self, model_manager, mock_mlx_available):
        """Test synchronous model loading"""
        config = MLXModelConfig(
            name="Test Model",
            path="/fake/path",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["test"]
        )
        
        # Mock path existence
        with patch('pathlib.Path.exists', return_value=True):
            success = model_manager._load_model_sync(config)
            assert success is True
            assert config.display_name in model_manager.loaded_models
    
    def test_load_model_sync_missing_path(self, model_manager, mock_mlx_available):
        """Test model loading with missing path"""
        config = MLXModelConfig(
            name="Missing Model",
            path="/missing/path", 
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["test"]
        )
        
        # Mock path not existing
        with patch('pathlib.Path.exists', return_value=False):
            success = model_manager._load_model_sync(config)
            assert success is False
    
    def test_optimize_model_for_m3_max(self, model_manager, mock_mlx_available):
        """Test M3 Max specific optimizations"""
        mock_model = MagicMock()
        config = MLXModelConfig(
            name="Test Model",
            path="/test",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["test"],
            precision="int8"
        )
        
        optimized_model = model_manager._optimize_model_for_m3_max(mock_model, config)
        assert optimized_model is not None
    
    def test_estimate_model_memory_usage(self, model_manager, mock_mlx_available):
        """Test memory usage estimation"""
        mock_model = MagicMock()
        
        # Mock model parameters
        with patch('docs.python.integration.mlx_accelerator.mx.tree_flatten') as mock_flatten:
            mock_param = MagicMock()
            mock_param.size = 1000000  # 1M parameters
            mock_flatten.return_value = ([mock_param], None)
            
            memory_gb = model_manager._estimate_model_memory_usage(mock_model)
            assert memory_gb > 0
    
    @pytest.mark.asyncio
    async def test_get_model(self, model_manager, mock_mlx_available):
        """Test model retrieval"""
        # Mock a loaded model
        config = MLXModelConfig(
            name="Test Model",
            path="/test",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["general_qa"]
        )
        
        model_manager.loaded_models[config.display_name] = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "config": config,
            "usage_count": 0
        }
        
        result = await model_manager.get_model(model_variant="7b")
        assert result is not None
        model, tokenizer, returned_config = result
        assert returned_config.variant == "7b"
    
    def test_record_performance(self, model_manager):
        """Test performance recording"""
        model_manager.record_performance(
            model_name="qwen3-7b-8bit",
            processing_time=2.5,
            tokens=50,
            success=True,
            memory_usage=8.2
        )
        
        stats = model_manager.get_model_stats("qwen3-7b-8bit")
        assert stats["total_requests"] == 1
        assert stats["avg_processing_time"] == 2.5
        assert stats["success_rate"] == 1.0
    
    def test_get_memory_usage(self, model_manager):
        """Test memory usage reporting"""
        # Mock some memory usage
        model_manager.memory_tracker["test_model"] = 8.0
        
        usage = model_manager.get_memory_usage()
        
        assert "total_model_memory_gb" in usage
        assert "individual_models" in usage
        assert "system_memory_gb" in usage
        assert usage["total_model_memory_gb"] == 8.0

class TestMLXInferenceEngine:
    """Test MLXInferenceEngine functionality"""
    
    @pytest.fixture
    def mock_model_manager(self, mock_mlx_available):
        """Create mock model manager"""
        manager = MagicMock()
        
        # Mock model config
        config = MLXModelConfig(
            name="Test Model",
            path="/test",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["general_qa"]
        )
        
        # Mock get_model return
        manager.get_model = AsyncMock(return_value=(
            mock_mlx_available['model'],
            mock_mlx_available['tokenizer'],
            config
        ))
        
        manager.record_performance = MagicMock()
        
        return manager
    
    @pytest.fixture
    def inference_engine(self, mock_model_manager):
        """Create inference engine for testing"""
        return MLXInferenceEngine(mock_model_manager, max_concurrent=2)
    
    def test_initialization(self, inference_engine):
        """Test inference engine initialization"""
        assert inference_engine.max_concurrent == 2
        assert inference_engine.executor is not None
        assert inference_engine.request_semaphore._value == 2
    
    def test_infer_use_case(self, inference_engine):
        """Test use case inference from prompt"""
        # Technical prompt
        technical_prompt = "Analyze the complex algorithm architecture for distributed systems"
        use_case = inference_engine._infer_use_case(technical_prompt)
        assert use_case == "complex_reasoning"
        
        # Quick Q&A
        quick_prompt = "What is AI?"
        use_case = inference_engine._infer_use_case(quick_prompt)
        assert use_case == "quick_qa"
        
        # Code generation
        code_prompt = "Write a Python function to sort a list"
        use_case = inference_engine._infer_use_case(code_prompt)
        assert use_case == "code_generation"
        
        # Default case
        general_prompt = "Tell me about the weather"
        use_case = inference_engine._infer_use_case(general_prompt)
        assert use_case == "general_qa"
    
    @pytest.mark.asyncio
    async def test_generate(self, inference_engine, mock_mlx_available):
        """Test inference generation"""
        request = MLXInferenceRequest(
            id="test_generate",
            prompt="What is machine learning?",
            model_variant="7b",
            max_tokens=200,
            temperature=0.7
        )
        
        # Mock the sync generation
        with patch.object(inference_engine, '_generate_sync', return_value={
            "text": "Machine learning is a subset of AI",
            "tokens": 25,
            "memory_usage": 8.2
        }):
            response = await inference_engine.generate(request)
            
            assert response.success is True
            assert response.text == "Machine learning is a subset of AI"
            assert response.tokens_generated == 25
            assert response.model_used.startswith("qwen3")
    
    @pytest.mark.asyncio
    async def test_generate_error_handling(self, inference_engine):
        """Test error handling in generation"""
        request = MLXInferenceRequest(
            id="test_error",
            prompt="Test prompt"
        )
        
        # Mock model manager to return None (no model available)
        inference_engine.model_manager.get_model = AsyncMock(return_value=None)
        
        response = await inference_engine.generate(request)
        
        assert response.success is False
        assert "No suitable MLX model available" in response.error
    
    def test_generate_sync(self, inference_engine, mock_mlx_available):
        """Test synchronous generation"""
        config = MLXModelConfig(
            name="Test Model",
            path="/test",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["test"]
        )
        
        request = MLXInferenceRequest(
            id="test_sync",
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        # Mock generate function
        with patch('docs.python.integration.mlx_accelerator.generate', return_value="Test response"):
            with patch.object(inference_engine, '_get_memory_usage_gb', return_value=8.0):
                result = inference_engine._generate_sync(
                    mock_mlx_available['model'],
                    mock_mlx_available['tokenizer'],
                    config,
                    request
                )
                
                assert result["text"] == "Test response"
                assert result["tokens"] > 0
    
    def test_get_memory_usage_gb(self, inference_engine, mock_mlx_available):
        """Test memory usage measurement"""
        memory_gb = inference_engine._get_memory_usage_gb()
        assert memory_gb >= 0.0
    
    def test_get_performance_stats(self, inference_engine):
        """Test performance statistics"""
        # Add mock request history
        mock_response = MLXInferenceResponse(
            request_id="test",
            text="Response",
            model_used="qwen3-7b-8bit",
            processing_time=1.5,
            tokens_generated=20,
            tokens_per_second=13.3,
            memory_usage_gb=8.0,
            metadata={"timestamp": time.time()}
        )
        
        inference_engine.request_history.append(mock_response)
        
        stats = inference_engine.get_performance_stats()
        
        assert "requests_per_minute" in stats
        assert "avg_processing_time" in stats
        assert "success_rate" in stats

class TestMLXAccelerator:
    """Test main MLXAccelerator functionality"""
    
    @pytest.fixture
    def accelerator(self):
        """Create MLX accelerator for testing"""
        return MLXAccelerator(max_concurrent_requests=2)
    
    @pytest.mark.asyncio
    async def test_initialization_without_mlx(self, accelerator):
        """Test initialization when MLX is not available"""
        with patch('docs.python.integration.mlx_accelerator.MLX_AVAILABLE', False):
            success = await accelerator.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_initialization_with_mlx(self, accelerator, mock_mlx_available):
        """Test successful initialization with MLX"""
        with patch.object(accelerator.model_manager, 'initialize', return_value=True):
            success = await accelerator.initialize()
            assert success is True
            assert accelerator.initialized is True
            assert accelerator.monitoring_active is True
    
    @pytest.mark.asyncio
    async def test_generate_not_initialized(self, accelerator):
        """Test generate when not initialized"""
        request = MLXInferenceRequest(id="test", prompt="Test")
        
        with pytest.raises(Exception, match="not initialized"):
            await accelerator.generate(request)
    
    @pytest.mark.asyncio
    async def test_generate_initialized(self, accelerator, mock_mlx_available):
        """Test generate when initialized"""
        accelerator.initialized = True
        
        request = MLXInferenceRequest(
            id="test_initialized",
            prompt="Test prompt"
        )
        
        # Mock inference engine
        mock_response = MLXInferenceResponse(
            request_id="test_initialized",
            text="Test response",
            model_used="qwen3-7b-8bit",
            processing_time=1.0,
            tokens_generated=10,
            tokens_per_second=10.0,
            memory_usage_gb=8.0
        )
        
        with patch.object(accelerator.inference_engine, 'generate', return_value=mock_response):
            response = await accelerator.generate(request)
            
            assert response.success is True
            assert response.text == "Test response"
    
    def test_get_status(self, accelerator, mock_mlx_available):
        """Test status reporting"""
        accelerator.initialized = True
        
        # Mock model manager data
        accelerator.model_manager.loaded_models = {"test_model": {}}
        accelerator.model_manager.get_memory_usage = MagicMock(return_value={"total_model_memory_gb": 8.0})
        accelerator.model_manager.get_model_stats = MagicMock(return_value={"avg_tokens_per_second": 20.0})
        accelerator.inference_engine.get_performance_stats = MagicMock(return_value={"requests_per_minute": 10.0})
        
        status = accelerator.get_status()
        
        expected_keys = ["initialized", "mlx_available", "loaded_models", "memory_usage", "performance", "model_stats"]
        assert all(key in status for key in expected_keys)
        assert status["initialized"] is True

class TestConcurrencyAndPerformance:
    """Test concurrency handling and performance optimization"""
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_requests(self, mock_mlx_available):
        """Test handling of concurrent inference requests"""
        accelerator = MLXAccelerator(max_concurrent_requests=3)
        
        with patch.object(accelerator.model_manager, 'initialize', return_value=True):
            await accelerator.initialize()
            
            # Create multiple requests
            requests = [
                MLXInferenceRequest(id=f"concurrent_{i}", prompt=f"Request {i}")
                for i in range(5)
            ]
            
            # Mock responses
            mock_response = MLXInferenceResponse(
                request_id="test",
                text="Concurrent response",
                model_used="qwen3-7b-8bit",
                processing_time=0.5,
                tokens_generated=10,
                tokens_per_second=20.0,
                memory_usage_gb=8.0
            )
            
            with patch.object(accelerator.inference_engine, 'generate', return_value=mock_response):
                # Process requests concurrently
                start_time = time.time()
                tasks = [accelerator.generate(request) for request in requests]
                responses = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                # Verify all requests completed
                assert len(responses) == 5
                assert all(r.success for r in responses)
                
                # Should be faster than sequential processing due to concurrency
                assert total_time < len(requests) * 0.5  # Less than sequential time

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self, mock_mlx_available):
        """Test handling of model loading failures"""
        manager = MLXModelManager()
        
        config = MLXModelConfig(
            name="Failing Model",
            path="/nonexistent/path",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            context_length=32768,
            use_cases=["test"]
        )
        
        # Mock path not existing
        with patch('pathlib.Path.exists', return_value=False):
            success = manager._load_model_sync(config)
            assert success is False
    
    @pytest.mark.asyncio
    async def test_inference_timeout_handling(self):
        """Test handling of inference timeouts"""
        manager = MagicMock()
        engine = MLXInferenceEngine(manager, max_concurrent=1)
        
        request = MLXInferenceRequest(
            id="timeout_test",
            prompt="Test",
            timeout=0.1  # Very short timeout
        )
        
        # Mock slow model retrieval
        manager.get_model = AsyncMock(side_effect=asyncio.sleep(1))  # Sleep longer than timeout
        
        response = await engine.generate(request)
        assert response.success is False
    
    def test_memory_estimation_fallback(self):
        """Test memory estimation fallback when calculation fails"""
        manager = MLXModelManager()
        
        # Mock model that raises exception during parameter counting
        mock_model = MagicMock()
        with patch('docs.python.integration.mlx_accelerator.mx.tree_flatten', side_effect=Exception("Calculation failed")):
            memory_gb = manager._estimate_model_memory_usage(mock_model)
            assert memory_gb == 4.0  # Should return default estimate

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])