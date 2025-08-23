#!/usr/bin/env python3
"""
Comprehensive unit tests for Local LLM Orchestrator
Tests framework coordination, intelligent routing, circuit breakers, and performance optimization.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import asdict
from collections import deque

# Import modules to test
import sys
sys.path.append('/Users/cedric/orange/ran-llm-claude-flow')

from docs.python.integration.local_llm_orchestrator import (
    ModelSpec, InferenceRequest, InferenceResponse, FrameworkInterface,
    LMStudioInterface, OllamaInterface, MLXInterface, CircuitBreaker,
    InferenceCache, PerformanceMonitor, LocalLLMOrchestrator
)

class TestModelSpec:
    """Test ModelSpec dataclass"""
    
    def test_model_spec_creation(self):
        """Test ModelSpec creation and attributes"""
        spec = ModelSpec(
            name="qwen3-7b",
            variant="7b",
            quantization="8bit",
            memory_gb=8.0,
            tokens_per_second=45.0,
            context_length=32768,
            frameworks=["mlx", "ollama", "lmstudio"],
            use_cases=["general_qa", "summarization"]
        )
        
        assert spec.name == "qwen3-7b"
        assert spec.variant == "7b"
        assert spec.memory_gb == 8.0
        assert "mlx" in spec.frameworks
        assert "general_qa" in spec.use_cases

class TestInferenceRequest:
    """Test InferenceRequest dataclass"""
    
    def test_request_creation_with_defaults(self):
        """Test request creation with default values"""
        request = InferenceRequest(
            id="test_123",
            prompt="What is 5G NR technology?"
        )
        
        assert request.id == "test_123"
        assert request.max_tokens == 1024
        assert request.temperature == 0.7
        assert request.priority == "normal"
        assert request.metadata is not None
    
    def test_request_auto_id_generation(self):
        """Test automatic ID generation"""
        request = InferenceRequest(
            id=None,
            prompt="Test prompt"
        )
        
        assert request.id is not None
        assert request.id.startswith("req_")

class TestInferenceResponse:
    """Test InferenceResponse dataclass"""
    
    def test_successful_response(self):
        """Test successful response creation"""
        response = InferenceResponse(
            request_id="test_123",
            text="This is a test response",
            model_used="qwen3-7b",
            framework_used="mlx",
            processing_time=1.5,
            tokens_per_second=30.0,
            success=True
        )
        
        assert response.success is True
        assert response.error is None
        assert response.tokens_per_second == 30.0
    
    def test_error_response(self):
        """Test error response creation"""
        response = InferenceResponse(
            request_id="test_123",
            text="",
            model_used="unknown",
            framework_used="mlx",
            processing_time=0.1,
            tokens_per_second=0.0,
            success=False,
            error="Model loading failed"
        )
        
        assert response.success is False
        assert response.error == "Model loading failed"

class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing"""
        return CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    
    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state"""
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.can_proceed() is True
    
    def test_failure_tracking(self, circuit_breaker):
        """Test failure counting and state transitions"""
        # Record failures
        for i in range(2):
            circuit_breaker.record_failure()
            assert circuit_breaker.state == "CLOSED"  # Still closed
            assert circuit_breaker.can_proceed() is True
        
        # Third failure should open circuit
        circuit_breaker.record_failure()
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.can_proceed() is False
    
    def test_recovery_timeout(self, circuit_breaker):
        """Test recovery after timeout"""
        # Open circuit
        for _ in range(3):
            circuit_breaker.record_failure()
        
        assert circuit_breaker.state == "OPEN"
        
        # Simulate time passage
        circuit_breaker.last_failure_time = time.time() - 15  # 15 seconds ago
        
        # Should transition to HALF_OPEN
        assert circuit_breaker.can_proceed() is True
        assert circuit_breaker.state == "HALF_OPEN"
    
    def test_success_after_half_open(self, circuit_breaker):
        """Test success recovery from HALF_OPEN"""
        # Set to HALF_OPEN
        circuit_breaker.state = "HALF_OPEN"
        
        # Record success
        circuit_breaker.record_success()
        
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0

class TestInferenceCache:
    """Test InferenceCache LRU caching"""
    
    @pytest.fixture
    def cache(self):
        """Create inference cache for testing"""
        return InferenceCache(max_size=3, ttl_seconds=60)
    
    def test_cache_miss(self, cache):
        """Test cache miss scenario"""
        request = InferenceRequest(id="test", prompt="What is AI?")
        result = cache.get(request)
        assert result is None
    
    def test_cache_hit(self, cache):
        """Test cache hit scenario"""
        request = InferenceRequest(id="test", prompt="What is AI?")
        response = InferenceResponse(
            request_id="test",
            text="AI is artificial intelligence",
            model_used="qwen3-7b",
            framework_used="mlx",
            processing_time=1.0,
            tokens_per_second=20.0
        )
        
        # Cache the response
        cache.put(request, response)
        
        # Retrieve from cache
        cached = cache.get(request)
        assert cached is not None
        assert cached.text == response.text
        assert cached.request_id == request.id  # Should be updated
    
    def test_cache_eviction(self, cache):
        """Test LRU eviction when cache is full"""
        responses = []
        
        # Fill cache to capacity
        for i in range(3):
            request = InferenceRequest(id=f"test_{i}", prompt=f"Prompt {i}")
            response = InferenceResponse(
                request_id=f"test_{i}",
                text=f"Response {i}",
                model_used="qwen3-7b",
                framework_used="mlx", 
                processing_time=1.0,
                tokens_per_second=20.0
            )
            cache.put(request, response)
            responses.append((request, response))
        
        # Add one more (should evict LRU)
        new_request = InferenceRequest(id="test_new", prompt="New prompt")
        new_response = InferenceResponse(
            request_id="test_new",
            text="New response",
            model_used="qwen3-7b",
            framework_used="mlx",
            processing_time=1.0,
            tokens_per_second=20.0
        )
        cache.put(new_request, new_response)
        
        # First item should be evicted
        assert cache.get(responses[0][0]) is None
        # Last item should be present
        assert cache.get(new_request) is not None
    
    def test_ttl_expiration(self, cache):
        """Test TTL-based cache expiration"""
        request = InferenceRequest(id="test", prompt="Test")
        response = InferenceResponse(
            request_id="test",
            text="Response",
            model_used="qwen3-7b",
            framework_used="mlx",
            processing_time=1.0,
            tokens_per_second=20.0
        )
        
        # Cache with short TTL
        cache.ttl_seconds = 0.1
        cache.put(request, response)
        
        # Should be available immediately
        assert cache.get(request) is not None
        
        # Wait for TTL expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get(request) is None

class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor(window_size=5)
    
    def test_record_request(self, monitor):
        """Test request recording"""
        response = InferenceResponse(
            request_id="test",
            text="Response",
            model_used="qwen3-7b",
            framework_used="mlx",
            processing_time=1.5,
            tokens_per_second=30.0,
            success=True
        )
        
        monitor.record_request("mlx", response)
        
        # Check metrics recorded
        assert len(monitor.metrics["mlx_response_time"]) == 1
        assert len(monitor.metrics["mlx_tokens_per_second"]) == 1
        assert len(monitor.metrics["mlx_success"]) == 1
    
    def test_framework_stats(self, monitor):
        """Test framework statistics calculation"""
        # Record multiple requests
        for i in range(3):
            response = InferenceResponse(
                request_id=f"test_{i}",
                text=f"Response {i}",
                model_used="qwen3-7b",
                framework_used="mlx",
                processing_time=1.0 + i * 0.5,
                tokens_per_second=30.0 - i * 5,
                success=True
            )
            monitor.record_request("mlx", response)
        
        stats = monitor.get_framework_stats("mlx")
        
        assert stats["request_count"] == 3
        assert stats["success_rate"] == 1.0
        assert stats["avg_response_time"] > 0
        assert stats["avg_tokens_per_second"] > 0
    
    def test_best_framework_selection(self, monitor):
        """Test best framework selection logic"""
        # Record performance for different frameworks
        frameworks = ["mlx", "ollama", "lmstudio"]
        
        for i, framework in enumerate(frameworks):
            for j in range(2):
                response = InferenceResponse(
                    request_id=f"{framework}_{j}",
                    text=f"Response from {framework}",
                    model_used="qwen3-7b",
                    framework_used=framework,
                    processing_time=1.0 + i * 0.5,  # MLX fastest
                    tokens_per_second=40.0 - i * 10,  # MLX highest throughput
                    success=True
                )
                monitor.record_request(framework, response)
        
        best = monitor.select_best_framework(frameworks)
        assert best == "mlx"  # Should select MLX due to better performance

class MockFrameworkInterface:
    """Mock framework interface for testing"""
    
    def __init__(self, name, success_rate=1.0, response_time=1.0):
        self.name = name
        self.success_rate = success_rate
        self.response_time = response_time
        self.initialized = False
        self.available_models = ["qwen3-7b", "qwen3-14b"]
    
    async def initialize(self):
        self.initialized = True
        return True
    
    async def generate(self, request):
        await asyncio.sleep(self.response_time)  # Simulate processing time
        
        success = __import__('random').random() < self.success_rate
        
        if success:
            return InferenceResponse(
                request_id=request.id,
                text=f"Response from {self.name}",
                model_used="qwen3-7b",
                framework_used=self.name,
                processing_time=self.response_time,
                tokens_per_second=20.0,
                success=True
            )
        else:
            return InferenceResponse(
                request_id=request.id,
                text="",
                model_used="qwen3-7b",
                framework_used=self.name,
                processing_time=self.response_time,
                tokens_per_second=0.0,
                success=False,
                error="Simulated failure"
            )
    
    async def health_check(self):
        return self.success_rate > 0.5
    
    def get_available_models(self):
        return self.available_models

class TestLocalLLMOrchestrator:
    """Test main LocalLLMOrchestrator functionality"""
    
    @pytest.fixture
    def mock_frameworks(self):
        """Create mock frameworks for testing"""
        return {
            "mlx": MockFrameworkInterface("mlx", success_rate=0.9, response_time=0.5),
            "ollama": MockFrameworkInterface("ollama", success_rate=0.8, response_time=1.0),
            "lmstudio": MockFrameworkInterface("lmstudio", success_rate=0.95, response_time=0.8)
        }
    
    @pytest.fixture
    def orchestrator(self, mock_frameworks):
        """Create orchestrator with mock frameworks"""
        orchestrator = LocalLLMOrchestrator()
        orchestrator.frameworks = mock_frameworks
        orchestrator.circuit_breakers = {
            name: CircuitBreaker() for name in mock_frameworks.keys()
        }
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        success = await orchestrator.initialize()
        assert success is True
        
        # Check all frameworks initialized
        for framework in orchestrator.frameworks.values():
            assert framework.initialized is True
    
    def test_request_type_analysis(self, orchestrator):
        """Test request type analysis logic"""
        # Critical priority
        request = InferenceRequest(id="test", prompt="Quick response", priority="critical")
        request_type = orchestrator._analyze_request_type(request)
        assert request_type == "speed_priority"
        
        # Long prompt
        request = InferenceRequest(id="test", prompt="a" * 5000)
        request_type = orchestrator._analyze_request_type(request)
        assert request_type == "quality_priority"
        
        # Embedding request
        request = InferenceRequest(id="test", prompt="Generate embeddings for similarity search")
        request_type = orchestrator._analyze_request_type(request)
        assert request_type == "embedding"
        
        # Low temperature (factual)
        request = InferenceRequest(id="test", prompt="What is the capital?", temperature=0.1)
        request_type = orchestrator._analyze_request_type(request)
        assert request_type == "quality_priority"
    
    def test_optimal_framework_selection(self, orchestrator):
        """Test optimal framework selection"""
        # Mock framework health
        orchestrator.framework_health = {"mlx": True, "ollama": True, "lmstudio": True}
        
        # Test speed priority
        request = InferenceRequest(id="test", prompt="Quick", priority="critical")
        framework = orchestrator._select_optimal_framework(request)
        assert framework in orchestrator.frameworks
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback(self, orchestrator):
        """Test generation with fallback logic"""
        await orchestrator.initialize()
        orchestrator.framework_health = {"mlx": True, "ollama": True, "lmstudio": True}
        
        request = InferenceRequest(
            id="test_fallback",
            prompt="Test prompt for fallback",
            model_variant="7b"
        )
        
        response = await orchestrator.generate(request)
        
        assert response is not None
        assert response.request_id == "test_fallback"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, orchestrator):
        """Test circuit breaker integration"""
        await orchestrator.initialize()
        
        # Make one framework fail consistently
        orchestrator.frameworks["mlx"].success_rate = 0.0
        orchestrator.framework_health = {"mlx": True, "ollama": True, "lmstudio": True}
        
        # Make several requests to trigger circuit breaker
        for i in range(5):
            request = InferenceRequest(id=f"test_{i}", prompt="Test")
            response = await orchestrator._try_framework_with_fallback(request, "mlx")
            
            # Should eventually fallback to other frameworks
            if i >= 3:  # After circuit breaker opens
                assert response.framework_used in ["ollama", "lmstudio"]
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, orchestrator):
        """Test caching integration"""
        await orchestrator.initialize()
        orchestrator.framework_health = {"mlx": True, "ollama": True, "lmstudio": True}
        
        # Enable caching
        orchestrator.cache = InferenceCache(max_size=10)
        
        request = InferenceRequest(
            id="test_cache",
            prompt="Cached prompt test",
            temperature=0.7,
            max_tokens=100
        )
        
        # First request
        start_time = time.time()
        response1 = await orchestrator.generate(request)
        first_request_time = time.time() - start_time
        
        # Second identical request (should hit cache)
        start_time = time.time()
        response2 = await orchestrator.generate(request)
        second_request_time = time.time() - start_time
        
        assert response1.success
        assert response2.success
        assert second_request_time < first_request_time  # Cache should be faster
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, orchestrator):
        """Test handling of concurrent requests"""
        await orchestrator.initialize()
        orchestrator.framework_health = {"mlx": True, "ollama": True, "lmstudio": True}
        
        # Create multiple concurrent requests
        requests = [
            InferenceRequest(id=f"concurrent_{i}", prompt=f"Concurrent test {i}")
            for i in range(5)
        ]
        
        # Process concurrently
        start_time = time.time()
        tasks = [orchestrator.generate(request) for request in requests]
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all requests completed
        assert len(responses) == 5
        successful_responses = [r for r in responses if r.success]
        assert len(successful_responses) > 0
        
        # Concurrent processing should be faster than sequential
        assert total_time < sum(r.processing_time for r in successful_responses)
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, orchestrator):
        """Test status reporting"""
        await orchestrator.initialize()
        
        status = await orchestrator.get_status()
        
        expected_keys = ["frameworks", "performance", "cache", "timestamp"]
        assert all(key in status for key in expected_keys)
        
        # Check framework status
        for framework_name in orchestrator.frameworks.keys():
            assert framework_name in status["frameworks"]
            framework_status = status["frameworks"][framework_name]
            assert "healthy" in framework_status
            assert "circuit_breaker_state" in framework_status

class TestFrameworkInterfaces:
    """Test individual framework interface implementations"""
    
    @pytest.mark.asyncio
    async def test_lmstudio_interface(self, mock_aiohttp_session):
        """Test LMStudioInterface"""
        interface = LMStudioInterface()
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            with patch.object(interface, 'session', mock_aiohttp_session):
                success = await interface.initialize()
                assert success is True
                assert len(interface.available_models) == 3
    
    @pytest.mark.asyncio
    async def test_ollama_interface(self, mock_ollama_session):
        """Test OllamaInterface"""
        interface = OllamaInterface()
        
        with patch('aiohttp.ClientSession', return_value=mock_ollama_session):
            with patch.object(interface, 'session', mock_ollama_session):
                success = await interface.initialize()
                assert success is True
                assert len(interface.available_models) == 3
    
    @pytest.mark.asyncio
    async def test_mlx_interface_without_mlx(self):
        """Test MLXInterface when MLX is not available"""
        with patch('docs.python.integration.local_llm_orchestrator.MLX_AVAILABLE', False):
            interface = MLXInterface()
            success = await interface.initialize()
            assert success is False

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_all_frameworks_unavailable(self, orchestrator):
        """Test behavior when all frameworks are unavailable"""
        await orchestrator.initialize()
        
        # Make all frameworks unhealthy
        orchestrator.framework_health = {"mlx": False, "ollama": False, "lmstudio": False}
        
        request = InferenceRequest(id="test_no_frameworks", prompt="Test")
        response = await orchestrator.generate(request)
        
        assert response.success is False
        assert "unavailable" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_framework_exception_handling(self, orchestrator):
        """Test handling of framework exceptions"""
        await orchestrator.initialize()
        orchestrator.framework_health = {"mlx": True, "ollama": True, "lmstudio": True}
        
        # Make framework raise exception
        async def failing_generate(request):
            raise Exception("Framework failure")
        
        orchestrator.frameworks["mlx"].generate = failing_generate
        
        request = InferenceRequest(id="test_exception", prompt="Test")
        response = await orchestrator.generate(request)
        
        # Should fallback to other frameworks
        assert response.framework_used in ["ollama", "lmstudio"]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])