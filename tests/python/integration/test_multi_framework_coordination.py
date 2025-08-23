#!/usr/bin/env python3
"""
Integration tests for multi-framework coordination
Tests the interaction between LM Studio, Ollama, and MLX frameworks through the orchestrator.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import modules to test
import sys
sys.path.append('/Users/cedric/orange/ran-llm-claude-flow')

from docs.python.integration.local_llm_orchestrator import (
    LocalLLMOrchestrator, InferenceRequest, InferenceResponse
)
from docs.python.integration.lmstudio_connector import LMStudioConnector, LMStudioConfig
from docs.python.integration.ollama_optimizer import OllamaOptimizer, EmbeddingRequest
from docs.python.integration.mlx_accelerator import MLXAccelerator, MLXInferenceRequest

class TestMultiFrameworkCoordination:
    """Test coordination between multiple LLM frameworks"""
    
    @pytest.fixture
    def mock_all_frameworks(self):
        """Mock all framework implementations"""
        mocks = {}
        
        # Mock LM Studio
        with patch('docs.python.integration.local_llm_orchestrator.LMStudioInterface') as MockLMStudio:
            mock_lmstudio = MockLMStudio.return_value
            mock_lmstudio.initialize = AsyncMock(return_value=True)
            mock_lmstudio.health_check = AsyncMock(return_value=True)
            mock_lmstudio.get_available_models = MagicMock(return_value=["qwen3-7b", "qwen3-14b"])
            mock_lmstudio.generate = AsyncMock(return_value=InferenceResponse(
                request_id="test",
                text="LM Studio response",
                model_used="qwen3-7b",
                framework_used="lmstudio",
                processing_time=0.8,
                tokens_per_second=25.0,
                success=True
            ))
            mocks['lmstudio'] = mock_lmstudio
        
        # Mock Ollama
        with patch('docs.python.integration.local_llm_orchestrator.OllamaInterface') as MockOllama:
            mock_ollama = MockOllama.return_value
            mock_ollama.initialize = AsyncMock(return_value=True)
            mock_ollama.health_check = AsyncMock(return_value=True)
            mock_ollama.get_available_models = MagicMock(return_value=["qwen2.5:7b", "qwen2.5:14b"])
            mock_ollama.generate = AsyncMock(return_value=InferenceResponse(
                request_id="test",
                text="Ollama response",
                model_used="qwen2.5:7b",
                framework_used="ollama",
                processing_time=1.2,
                tokens_per_second=20.0,
                success=True
            ))
            mocks['ollama'] = mock_ollama
        
        # Mock MLX
        with patch('docs.python.integration.local_llm_orchestrator.MLXInterface') as MockMLX:
            mock_mlx = MockMLX.return_value
            mock_mlx.initialize = AsyncMock(return_value=True)
            mock_mlx.health_check = AsyncMock(return_value=True)
            mock_mlx.get_available_models = MagicMock(return_value=["qwen3-1.7b", "qwen3-7b"])
            mock_mlx.generate = AsyncMock(return_value=InferenceResponse(
                request_id="test",
                text="MLX response",
                model_used="qwen3-1.7b",
                framework_used="mlx",
                processing_time=0.5,
                tokens_per_second=40.0,
                success=True
            ))
            mocks['mlx'] = mock_mlx
        
        return mocks
    
    @pytest.fixture
    def orchestrator(self, mock_all_frameworks):
        """Create orchestrator with mocked frameworks"""
        return LocalLLMOrchestrator()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator can initialize all frameworks"""
        success = await orchestrator.initialize()
        assert success is True
        
        # Verify all frameworks initialized
        for framework in orchestrator.frameworks.values():
            assert framework.initialized is True
        
        # Verify health monitoring started
        assert orchestrator._monitoring_active is True
    
    @pytest.mark.asyncio
    async def test_framework_selection_by_priority(self, orchestrator):
        """Test framework selection based on priority"""
        await orchestrator.initialize()
        
        # Test critical priority (should prefer fastest)
        critical_request = InferenceRequest(
            id="critical_test",
            prompt="Quick response needed",
            priority="critical",
            timeout=2.0
        )
        
        response = await orchestrator.generate(critical_request)
        assert response.success is True
        # MLX should be selected for speed
        assert response.framework_used in ["mlx", "lmstudio"]
    
    @pytest.mark.asyncio
    async def test_framework_selection_by_content_length(self, orchestrator):
        """Test framework selection based on content complexity"""
        await orchestrator.initialize()
        
        # Long complex prompt should use quality-focused framework
        complex_request = InferenceRequest(
            id="complex_test",
            prompt="a" * 5000 + " Please provide detailed analysis with comprehensive explanations",
            temperature=0.3
        )
        
        response = await orchestrator.generate(complex_request)
        assert response.success is True
        # Should prefer quality frameworks for complex tasks
        assert response.framework_used in ["lmstudio", "mlx"]
    
    @pytest.mark.asyncio
    async def test_framework_fallback_mechanism(self, orchestrator, mock_all_frameworks):
        """Test fallback when primary framework fails"""
        await orchestrator.initialize()
        
        # Make MLX fail
        mock_all_frameworks['mlx'].generate = AsyncMock(return_value=InferenceResponse(
            request_id="test",
            text="",
            model_used="qwen3-1.7b",
            framework_used="mlx",
            processing_time=0.1,
            tokens_per_second=0.0,
            success=False,
            error="MLX model failed"
        ))
        
        request = InferenceRequest(
            id="fallback_test",
            prompt="Test fallback mechanism",
            priority="critical"  # Would normally choose MLX
        )
        
        response = await orchestrator.generate(request)
        assert response.success is True
        # Should fallback to working framework
        assert response.framework_used in ["lmstudio", "ollama"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_coordination(self, orchestrator, mock_all_frameworks):
        """Test circuit breaker prevents using failing framework"""
        await orchestrator.initialize()
        
        # Make MLX consistently fail to trigger circuit breaker
        mock_all_frameworks['mlx'].generate = AsyncMock(return_value=InferenceResponse(
            request_id="test",
            text="",
            model_used="qwen3-1.7b",
            framework_used="mlx",
            processing_time=0.1,
            tokens_per_second=0.0,
            success=False,
            error="Consistent failure"
        ))
        
        # Make multiple requests to trigger circuit breaker
        for i in range(6):  # More than failure threshold
            request = InferenceRequest(
                id=f"circuit_test_{i}",
                prompt="Test circuit breaker",
                priority="critical"
            )
            response = await orchestrator.generate(request)
            
            if i >= 4:  # After circuit breaker opens
                assert response.framework_used in ["lmstudio", "ollama"]
    
    @pytest.mark.asyncio
    async def test_concurrent_multi_framework_requests(self, orchestrator):
        """Test handling concurrent requests across multiple frameworks"""
        await orchestrator.initialize()
        
        # Create diverse requests that would use different frameworks
        requests = [
            InferenceRequest(id="concurrent_1", prompt="Quick", priority="critical"),
            InferenceRequest(id="concurrent_2", prompt="a" * 1000 + " complex analysis"),
            InferenceRequest(id="concurrent_3", prompt="Generate embeddings for similarity", temperature=0.1),
            InferenceRequest(id="concurrent_4", prompt="Standard question"),
            InferenceRequest(id="concurrent_5", prompt="Another quick one", priority="high")
        ]
        
        # Process all concurrently
        start_time = time.time()
        tasks = [orchestrator.generate(request) for request in requests]
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all completed successfully
        assert len(responses) == 5
        successful_responses = [r for r in responses if r.success]
        assert len(successful_responses) >= 4  # At least 4 should succeed
        
        # Verify different frameworks were used
        frameworks_used = set(r.framework_used for r in successful_responses)
        assert len(frameworks_used) >= 2  # Should use multiple frameworks
        
        # Concurrent execution should be faster than sequential
        estimated_sequential_time = sum(r.processing_time for r in successful_responses)
        assert total_time < estimated_sequential_time * 0.8  # At least 20% faster
    
    @pytest.mark.asyncio
    async def test_caching_across_frameworks(self, orchestrator):
        """Test caching works across different frameworks"""
        await orchestrator.initialize()
        
        # Enable caching
        from docs.python.integration.local_llm_orchestrator import InferenceCache
        orchestrator.cache = InferenceCache(max_size=100)
        
        request = InferenceRequest(
            id="cache_test",
            prompt="Cached prompt test",
            model_variant="7b",
            temperature=0.7,
            max_tokens=100
        )
        
        # First request
        response1 = await orchestrator.generate(request)
        assert response1.success is True
        first_framework = response1.framework_used
        
        # Second identical request
        response2 = await orchestrator.generate(request)
        assert response2.success is True
        
        # Response should be the same regardless of which framework was used
        assert response1.text == response2.text
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, orchestrator):
        """Test performance monitoring across all frameworks"""
        await orchestrator.initialize()
        
        # Make requests to different frameworks
        requests = [
            InferenceRequest(id="perf_1", prompt="Quick", priority="critical"),
            InferenceRequest(id="perf_2", prompt="Medium complexity analysis"),
            InferenceRequest(id="perf_3", prompt="a" * 2000 + " detailed explanation")
        ]
        
        for request in requests:
            await orchestrator.generate(request)
        
        # Check performance monitor has data for multiple frameworks
        performance_stats = orchestrator.performance_monitor.get_all_stats()
        assert len(performance_stats) >= 2  # Should have stats for multiple frameworks
        
        # Verify each framework has meaningful stats
        for framework_name, stats in performance_stats.items():
            if stats["request_count"] > 0:
                assert "avg_response_time" in stats
                assert "success_rate" in stats
                assert stats["success_rate"] >= 0.8  # Should be mostly successful
    
    @pytest.mark.asyncio
    async def test_health_monitoring_coordination(self, orchestrator, mock_all_frameworks):
        """Test health monitoring affects framework selection"""
        await orchestrator.initialize()
        
        # Let health monitoring run briefly
        await asyncio.sleep(0.1)
        
        # Make one framework unhealthy
        mock_all_frameworks['ollama'].health_check = AsyncMock(return_value=False)
        
        # Wait for health check to detect issue
        await asyncio.sleep(0.5)
        
        # Requests should avoid unhealthy framework
        for _ in range(3):
            request = InferenceRequest(
                id="health_test",
                prompt="Test with unhealthy framework"
            )
            response = await orchestrator.generate(request)
            assert response.success is True
            assert response.framework_used != "ollama"  # Should avoid unhealthy framework

class TestFrameworkSpecificIntegration:
    """Test integration with specific framework implementations"""
    
    @pytest.mark.asyncio
    async def test_lmstudio_connector_integration(self):
        """Test LM Studio connector integration"""
        config = LMStudioConfig(max_connections=2, batch_size=1)
        connector = LMStudioConnector(config)
        
        with patch('aiohttp.ClientSession') as MockSession:
            mock_session = AsyncMock()
            
            # Mock models response
            mock_models_response = AsyncMock()
            mock_models_response.status = 200
            mock_models_response.json = AsyncMock(return_value={
                "data": [{"id": "qwen3-7b", "name": "Qwen3 7B"}]
            })
            
            # Mock chat response
            mock_chat_response = AsyncMock()
            mock_chat_response.status = 200
            mock_chat_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "LM Studio integration test"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
            })
            
            mock_session.get.return_value.__aenter__.return_value = mock_models_response
            mock_session.post.return_value.__aenter__.return_value = mock_chat_response
            MockSession.return_value = mock_session
            
            # Test initialization and generation
            success = await connector.initialize()
            assert success is True
            
            messages = [{"role": "user", "content": "Test message"}]
            result = await connector.generate(messages=messages)
            
            assert result["success"] is True
            assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_ollama_optimizer_integration(self):
        """Test Ollama optimizer integration"""
        with patch('aiohttp.ClientSession') as MockSession:
            mock_session = AsyncMock()
            
            # Mock Ollama responses
            mock_tags_response = AsyncMock()
            mock_tags_response.status = 200
            mock_tags_response.json = AsyncMock(return_value={
                "models": [{"name": "qwen2.5:7b-q8_0", "size": 7200000000}]
            })
            
            mock_generate_response = AsyncMock()
            mock_generate_response.status = 200
            mock_generate_response.json = AsyncMock(return_value={
                "response": "Ollama integration test response"
            })
            
            mock_session.get.return_value.__aenter__.return_value = mock_tags_response
            mock_session.post.return_value.__aenter__.return_value = mock_generate_response
            MockSession.return_value = mock_session
            
            optimizer = OllamaOptimizer()
            
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.available = 32 * 1024**3  # 32GB
                
                success = await optimizer.initialize()
                assert success is True
                
                # Test text generation
                result = await optimizer.generate_text(
                    prompt="Integration test prompt",
                    use_case="general_qa"
                )
                
                assert result["success"] is True
                assert "response" in result
    
    @pytest.mark.asyncio
    async def test_mlx_accelerator_integration(self, mock_mlx_available):
        """Test MLX accelerator integration"""
        accelerator = MLXAccelerator(max_concurrent_requests=1)
        
        with patch.object(accelerator.model_manager, 'initialize', return_value=True):
            success = await accelerator.initialize()
            assert success is True
            
            request = MLXInferenceRequest(
                id="integration_test",
                prompt="MLX integration test",
                model_variant="7b"
            )
            
            # Mock successful response
            from docs.python.integration.mlx_accelerator import MLXInferenceResponse
            mock_response = MLXInferenceResponse(
                request_id="integration_test",
                text="MLX integration response",
                model_used="qwen3-7b-8bit",
                processing_time=0.5,
                tokens_generated=15,
                tokens_per_second=30.0,
                memory_usage_gb=8.0
            )
            
            with patch.object(accelerator.inference_engine, 'generate', return_value=mock_response):
                response = await accelerator.generate(request)
                
                assert response.success is True
                assert response.text == "MLX integration response"

class TestEndToEndScenarios:
    """End-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_fallbacks(self, mock_all_frameworks):
        """Test complete pipeline with realistic failure scenarios"""
        orchestrator = LocalLLMOrchestrator()
        await orchestrator.initialize()
        
        # Scenario 1: Normal operation
        normal_request = InferenceRequest(
            id="e2e_normal",
            prompt="What is artificial intelligence?",
            max_tokens=200
        )
        
        response = await orchestrator.generate(normal_request)
        assert response.success is True
        assert len(response.text) > 0
        
        # Scenario 2: Framework failure with fallback
        mock_all_frameworks['mlx'].generate = AsyncMock(side_effect=Exception("MLX failure"))
        
        fallback_request = InferenceRequest(
            id="e2e_fallback",
            prompt="Handle framework failure gracefully",
            priority="critical"  # Would normally prefer MLX
        )
        
        response = await orchestrator.generate(fallback_request)
        assert response.success is True
        assert response.framework_used in ["lmstudio", "ollama"]
        
        # Scenario 3: Load balancing across frameworks
        load_requests = [
            InferenceRequest(id=f"e2e_load_{i}", prompt=f"Load test request {i}")
            for i in range(10)
        ]
        
        responses = await asyncio.gather(*[
            orchestrator.generate(req) for req in load_requests
        ])
        
        successful_responses = [r for r in responses if r.success]
        assert len(successful_responses) >= 8  # At least 80% success
        
        # Should distribute across multiple frameworks
        frameworks_used = set(r.framework_used for r in successful_responses)
        assert len(frameworks_used) >= 2
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_all_frameworks):
        """Test system performance under concurrent load"""
        orchestrator = LocalLLMOrchestrator()
        await orchestrator.initialize()
        
        # Create varied workload
        request_types = [
            ("critical", "Quick response", "critical"),
            ("normal", "Standard processing", "normal"),
            ("complex", "a" * 1000 + " detailed analysis", "normal"),
            ("embedding", "similarity search embeddings", "normal"),
        ]
        
        all_requests = []
        for i in range(20):  # 80 total requests (20 * 4 types)
            req_type, prompt_base, priority = request_types[i % len(request_types)]
            request = InferenceRequest(
                id=f"load_{req_type}_{i}",
                prompt=f"{prompt_base} {i}",
                priority=priority
            )
            all_requests.append(request)
        
        # Measure performance
        start_time = time.time()
        
        # Process in batches to simulate realistic load
        batch_size = 10
        all_responses = []
        
        for i in range(0, len(all_requests), batch_size):
            batch = all_requests[i:i + batch_size]
            batch_responses = await asyncio.gather(*[
                orchestrator.generate(req) for req in batch
            ], return_exceptions=True)
            all_responses.extend(batch_responses)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_responses = [r for r in all_responses 
                              if hasattr(r, 'success') and r.success]
        
        success_rate = len(successful_responses) / len(all_requests)
        avg_response_time = sum(r.processing_time for r in successful_responses) / len(successful_responses) if successful_responses else 0
        
        # Performance assertions
        assert success_rate >= 0.85  # At least 85% success rate
        assert avg_response_time < 5.0  # Average response under 5 seconds
        assert total_time < len(all_requests) * 2.0  # Concurrent processing efficiency
        
        print(f"Load test results: {success_rate:.1%} success, {avg_response_time:.2f}s avg response, {total_time:.2f}s total")

class TestErrorRecovery:
    """Test error recovery and resilience patterns"""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_all_frameworks):
        """Test graceful degradation when frameworks fail"""
        orchestrator = LocalLLMOrchestrator()
        await orchestrator.initialize()
        
        # Progressively disable frameworks
        scenarios = [
            ("All healthy", {}),
            ("MLX down", {"mlx": False}),
            ("MLX and Ollama down", {"mlx": False, "ollama": False}),
            ("Only LMStudio", {"mlx": False, "ollama": False, "lmstudio": True})
        ]
        
        for scenario_name, framework_health in scenarios:
            # Update framework health
            for framework_name, healthy in framework_health.items():
                if not healthy:
                    mock_all_frameworks[framework_name].health_check = AsyncMock(return_value=False)
                    mock_all_frameworks[framework_name].generate = AsyncMock(return_value=InferenceResponse(
                        request_id="test",
                        text="",
                        model_used="unknown",
                        framework_used=framework_name,
                        processing_time=0.1,
                        tokens_per_second=0.0,
                        success=False,
                        error=f"{framework_name} unavailable"
                    ))
            
            # Test request
            request = InferenceRequest(
                id=f"degradation_{scenario_name}",
                prompt="Test graceful degradation"
            )
            
            response = await orchestrator.generate(request)
            
            if any(framework_health.values()) or not framework_health:
                # Should succeed if any framework is healthy or all are healthy
                assert response.success is True
            else:
                # Should fail gracefully if all frameworks are down
                assert response.success is False
                assert "unavailable" in response.error.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])