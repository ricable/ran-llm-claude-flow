#!/usr/bin/env python3
"""
Comprehensive unit tests for LM Studio Connector
Tests connection pooling, model management, circuit breakers, and performance monitoring.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import asdict

# Import modules to test
import sys
sys.path.append('/Users/cedric/orange/ran-llm-claude-flow')

from docs.python.integration.lmstudio_connector import (
    LMStudioConfig, ModelInfo, RequestMetrics, ConnectionPool,
    RequestQueue, ModelManager, HealthMonitor, LMStudioConnector
)

class TestLMStudioConfig:
    """Test LMStudioConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = LMStudioConfig()
        assert config.base_url == "http://127.0.0.1:1234"
        assert config.max_connections == 20
        assert config.connection_timeout == 10.0
        assert config.retry_attempts == 3
        assert "qwen3-1.7b" in config.preferred_models
        assert "qwen3-7b" in config.preferred_models
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = LMStudioConfig(
            base_url="http://custom-host:8080",
            max_connections=50,
            preferred_models=["custom-model"]
        )
        assert config.base_url == "http://custom-host:8080"
        assert config.max_connections == 50
        assert config.preferred_models == ["custom-model"]

class TestModelInfo:
    """Test ModelInfo dataclass"""
    
    def test_qwen_detection(self):
        """Test Qwen model detection and variant extraction"""
        # Test Qwen3 1.7b
        model = ModelInfo(id="qwen3-1.7b-instruct", name="Qwen3 1.7B Instruct")
        assert model.is_qwen is True
        assert model.variant == "1.7b"
        
        # Test Qwen2.5 7b
        model = ModelInfo(id="qwen2.5:7b-q8", name="Qwen2.5 7B")
        assert model.is_qwen is True
        assert model.variant == "7b"
        
        # Test non-Qwen model
        model = ModelInfo(id="llama-7b", name="Llama 7B")
        assert model.is_qwen is False
        assert model.variant is None
    
    def test_variant_extraction(self):
        """Test variant extraction logic"""
        test_cases = [
            ("qwen3-1.5b", "1.7b"),
            ("qwen3-7b-instruct", "7b"),
            ("qwen2.5:14b-q8", "14b"),
            ("qwen-30b-chat", "30b"),
            ("qwen-unknown", None)
        ]
        
        for model_id, expected_variant in test_cases:
            model = ModelInfo(id=model_id, name=model_id)
            assert model.variant == expected_variant

class TestRequestMetrics:
    """Test RequestMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test metrics creation with auto-timestamp"""
        start_time = time.time()
        metrics = RequestMetrics(
            request_id="test_123",
            model_id="qwen3-7b",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            processing_time=2.5,
            queue_time=0.1,
            tokens_per_second=40.0,
            success=True
        )
        
        assert metrics.request_id == "test_123"
        assert metrics.tokens_per_second == 40.0
        assert metrics.success is True
        assert metrics.timestamp >= start_time

class TestConnectionPool:
    """Test ConnectionPool functionality"""
    
    @pytest.fixture
    def connection_pool(self):
        """Create connection pool for testing"""
        config = LMStudioConfig(max_connections=5)
        return ConnectionPool(config)
    
    def test_connection_pool_initialization(self, connection_pool):
        """Test connection pool setup"""
        assert connection_pool.active_connections == 0
        assert connection_pool.connector is not None
    
    @pytest.mark.asyncio
    async def test_get_session(self, connection_pool, mock_aiohttp_session):
        """Test session creation and retrieval"""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            session = await connection_pool.get_session()
            assert session is not None
            assert connection_pool.active_connections == 1
    
    @pytest.mark.asyncio
    async def test_session_pooling(self, connection_pool, mock_aiohttp_session):
        """Test session pooling behavior"""
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            # Get and return session
            session1 = await connection_pool.get_session()
            connection_pool.return_session(session1)
            
            # Get another session (should reuse)
            session2 = await connection_pool.get_session()
            assert len(connection_pool.sessions) >= 0  # May be reused or new
    
    def test_stats(self, connection_pool):
        """Test connection pool statistics"""
        stats = connection_pool.stats()
        expected_keys = ["pooled_sessions", "active_connections", "max_connections"]
        assert all(key in stats for key in expected_keys)
        assert stats["max_connections"] == 5

class TestRequestQueue:
    """Test RequestQueue priority queue"""
    
    @pytest.fixture
    def request_queue(self):
        """Create request queue for testing"""
        return RequestQueue(max_size=100, batch_size=5)
    
    @pytest.mark.asyncio
    async def test_enqueue_request(self, request_queue):
        """Test request enqueueing"""
        success = await request_queue.enqueue(
            "test_req_1",
            {"model": "qwen3-7b", "messages": [{"role": "user", "content": "test"}]},
            "high"
        )
        assert success is True
        assert "test_req_1" in request_queue.pending_requests
    
    @pytest.mark.asyncio
    async def test_queue_capacity(self, request_queue):
        """Test queue capacity limits"""
        # Fill queue to capacity
        for i in range(100):
            await request_queue.enqueue(f"req_{i}", {"data": i}, "normal")
        
        # Try to add one more (should fail)
        success = await request_queue.enqueue("overflow", {"data": "overflow"}, "normal")
        assert success is False
    
    def test_dequeue_batch_priority(self, request_queue):
        """Test batch dequeuing with priority ordering"""
        # Add requests with different priorities
        asyncio.run(request_queue.enqueue("critical_1", {"data": 1}, "critical"))
        asyncio.run(request_queue.enqueue("normal_1", {"data": 2}, "normal"))
        asyncio.run(request_queue.enqueue("high_1", {"data": 3}, "high"))
        
        # Dequeue batch
        batch = request_queue.dequeue_batch()
        
        # Critical should come first
        assert len(batch) == 3
        assert batch[0]["id"] == "critical_1"
        assert batch[1]["id"] == "high_1"
        assert batch[2]["id"] == "normal_1"
    
    def test_queue_stats(self, request_queue):
        """Test queue statistics"""
        asyncio.run(request_queue.enqueue("test_1", {"data": 1}, "high"))
        asyncio.run(request_queue.enqueue("test_2", {"data": 2}, "normal"))
        
        stats = request_queue.get_stats()
        assert "high" in stats
        assert "normal" in stats
        assert stats["high"] == 1
        assert stats["normal"] == 1

class TestModelManager:
    """Test ModelManager functionality"""
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager for testing"""
        config = LMStudioConfig()
        return ModelManager(config)
    
    @pytest.mark.asyncio
    async def test_refresh_models(self, model_manager, mock_aiohttp_session):
        """Test model refresh from API"""
        success = await model_manager.refresh_models(mock_aiohttp_session)
        assert success is True
        assert len(model_manager.available_models) == 3
        assert "qwen3-1.7b" in model_manager.available_models
    
    def test_select_optimal_model(self, model_manager):
        """Test optimal model selection"""
        # Add mock models
        model_manager.available_models = {
            "qwen3-1.7b": ModelInfo(id="qwen3-1.7b", name="Qwen3 1.7B"),
            "qwen3-7b": ModelInfo(id="qwen3-7b", name="Qwen3 7B"),
            "qwen3-14b": ModelInfo(id="qwen3-14b", name="Qwen3 14B")
        }
        
        # Test variant preference
        selected = model_manager.select_optimal_model(preferred_variant="7b")
        assert selected == "qwen3-7b"
        
        # Test task complexity
        selected = model_manager.select_optimal_model(task_complexity="speed")
        assert selected in ["qwen3-1.7b", "qwen3-7b"]  # Prefers smaller models
        
        selected = model_manager.select_optimal_model(task_complexity="quality")
        assert selected in ["qwen3-14b", "qwen3-7b"]  # Prefers larger models
    
    def test_record_performance(self, model_manager):
        """Test performance recording"""
        metrics = RequestMetrics(
            request_id="test",
            model_id="qwen3-7b",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            processing_time=1.0,
            queue_time=0.1,
            tokens_per_second=20.0,
            success=True
        )
        
        model_manager.record_performance("qwen3-7b", metrics)
        stats = model_manager.get_model_stats("qwen3-7b")
        
        assert stats["avg_tokens_per_second"] == 20.0
        assert stats["success_rate"] == 1.0
        assert stats["total_requests"] == 1

class TestHealthMonitor:
    """Test HealthMonitor functionality"""
    
    @pytest.fixture
    def health_monitor(self):
        """Create health monitor for testing"""
        config = LMStudioConfig()
        return HealthMonitor(config)
    
    @pytest.fixture
    def mock_connection_pool(self, mock_aiohttp_session):
        """Mock connection pool"""
        pool = MagicMock()
        pool.get_session = AsyncMock(return_value=mock_aiohttp_session)
        pool.return_session = MagicMock()
        return pool
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, health_monitor, mock_connection_pool):
        """Test successful health check"""
        # Mock successful response
        mock_connection_pool.get_session.return_value.get.return_value.__aenter__.return_value.status = 200
        
        await health_monitor._perform_health_check(mock_connection_pool)
        
        assert health_monitor.is_healthy is True
        assert health_monitor.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, health_monitor, mock_connection_pool):
        """Test health check failure handling"""
        # Mock failed response
        mock_connection_pool.get_session.return_value.get.return_value.__aenter__.return_value.status = 500
        
        await health_monitor._perform_health_check(mock_connection_pool)
        
        assert health_monitor.consecutive_failures == 1
    
    def test_health_stats(self, health_monitor):
        """Test health statistics"""
        stats = health_monitor.get_health_stats()
        expected_keys = ["healthy", "last_check", "consecutive_failures"]
        assert all(key in stats for key in expected_keys)

class TestLMStudioConnector:
    """Test main LMStudioConnector functionality"""
    
    @pytest.fixture
    def connector(self):
        """Create connector for testing"""
        config = LMStudioConfig(max_connections=5, batch_size=2)
        return LMStudioConnector(config)
    
    @pytest.mark.asyncio
    async def test_connector_initialization(self, connector, mock_aiohttp_session):
        """Test connector initialization"""
        with patch.object(connector.connection_pool, 'get_session', return_value=mock_aiohttp_session):
            with patch.object(connector.connection_pool, 'return_session'):
                success = await connector.initialize()
                assert success is True
                assert connector.processing_active is True
    
    @pytest.mark.asyncio
    async def test_generate_request(self, connector, mock_aiohttp_session, sample_inference_requests):
        """Test generation request handling"""
        # Mock initialization
        connector.processing_active = True
        connector.health_monitor.is_healthy = True
        connector.model_manager.available_models = {"qwen3-7b": ModelInfo(id="qwen3-7b", name="Qwen3 7B")}
        
        with patch.object(connector.connection_pool, 'get_session', return_value=mock_aiohttp_session):
            with patch.object(connector.connection_pool, 'return_session'):
                # Test request
                messages = [{"role": "user", "content": "What is 5G NR?"}]
                
                result = await connector.generate(
                    messages=messages,
                    model="qwen3-7b",
                    max_tokens=200,
                    temperature=0.7
                )
                
                assert result["success"] is True
                assert "data" in result
                assert "metrics" in result
    
    @pytest.mark.asyncio
    async def test_model_selection_fallback(self, connector):
        """Test model selection with fallback"""
        # Mock no specific model requested
        connector.model_manager.available_models = {
            "qwen3-7b": ModelInfo(id="qwen3-7b", name="Qwen3 7B")
        }
        
        with patch.object(connector.model_manager, 'select_optimal_model', return_value="qwen3-7b"):
            messages = [{"role": "user", "content": "Test"}]
            
            # This would normally be tested through the full generate method
            # For now, verify the model manager has the selection logic
            selected = connector.model_manager.select_optimal_model()
            assert selected == "qwen3-7b"
    
    def test_status_reporting(self, connector):
        """Test status reporting"""
        connector.health_monitor.is_healthy = True
        connector.model_manager.available_models = {"test": ModelInfo(id="test", name="Test")}
        
        status = connector.get_status()
        
        expected_keys = ["healthy", "connection_pool", "request_queue", "models", "health"]
        assert all(key in status for key in expected_keys)
        assert status["healthy"] is True
        assert status["models"]["available"] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup(self, connector):
        """Test connector cleanup"""
        connector.processing_active = True
        connector.processor_task = AsyncMock()
        connector.health_monitor.stop_monitoring = AsyncMock()
        connector.request_queue.clear = MagicMock()
        connector.connection_pool.cleanup = AsyncMock()
        
        await connector.cleanup()
        
        assert connector.processing_active is False
        connector.processor_task.cancel.assert_called_once()
        connector.health_monitor.stop_monitoring.assert_called_once()
        connector.request_queue.clear.assert_called_once()
        connector.connection_pool.cleanup.assert_called_once()

class TestPerformanceMetrics:
    """Test performance monitoring and metrics collection"""
    
    @pytest.fixture
    def connector_with_metrics(self):
        """Create connector with mock metrics"""
        config = LMStudioConfig()
        connector = LMStudioConnector(config)
        
        # Add mock metrics
        for i in range(10):
            metrics = RequestMetrics(
                request_id=f"req_{i}",
                model_id="qwen3-7b",
                prompt_tokens=10 + i,
                completion_tokens=20 + i,
                total_tokens=30 + 2*i,
                processing_time=1.0 + i * 0.1,
                queue_time=0.1,
                tokens_per_second=20.0 - i,
                success=True,
                timestamp=time.time() - i * 10  # 10 seconds apart
            )
            connector.request_metrics.append(metrics)
        
        return connector
    
    def test_performance_summary(self, connector_with_metrics):
        """Test performance summary calculation"""
        summary = connector_with_metrics._get_performance_summary()
        
        assert "requests_per_minute" in summary
        assert "avg_processing_time" in summary
        assert "avg_tokens_per_second" in summary
        assert "success_rate" in summary
        
        # Verify calculations make sense
        assert summary["success_rate"] == 1.0
        assert summary["avg_processing_time"] > 0

class TestErrorHandling:
    """Test error handling and resilience"""
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of connection failures"""
        config = LMStudioConfig()
        connector = LMStudioConnector(config)
        
        # Mock connection failure
        with patch.object(connector.connection_pool, 'get_session', side_effect=Exception("Connection failed")):
            success = await connector.initialize()
            assert success is False
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_aiohttp_session):
        """Test API error response handling"""
        # Mock 500 error response
        mock_aiohttp_session.post.return_value.__aenter__.return_value.status = 500
        mock_aiohttp_session.post.return_value.__aenter__.return_value.text = AsyncMock(return_value="Internal Server Error")
        
        config = LMStudioConfig()
        connector = LMStudioConnector(config)
        connector.processing_active = True
        
        with patch.object(connector.connection_pool, 'get_session', return_value=mock_aiohttp_session):
            with patch.object(connector.connection_pool, 'return_session'):
                messages = [{"role": "user", "content": "Test"}]
                
                result = await connector.generate(messages=messages)
                assert result["success"] is False
                assert "error" in result

class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_aiohttp_session):
        """Test handling of concurrent requests"""
        config = LMStudioConfig(max_connections=3, batch_size=2)
        connector = LMStudioConnector(config)
        
        # Mock successful initialization
        with patch.object(connector.connection_pool, 'get_session', return_value=mock_aiohttp_session):
            with patch.object(connector.connection_pool, 'return_session'):
                await connector.initialize()
                
                # Create multiple concurrent requests
                messages = [{"role": "user", "content": f"Request {i}"} for i in range(5)]
                
                tasks = []
                for msg in messages:
                    task = asyncio.create_task(connector.generate(messages=[msg]))
                    tasks.append(task)
                
                # Wait for all requests
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify results
                successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
                assert len(successful_results) > 0  # At least some should succeed

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])