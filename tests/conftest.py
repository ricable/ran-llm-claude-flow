"""
Main conftest.py for all Python pipeline tests
Comprehensive test configuration with fixtures and shared setup
"""
import pytest
import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional
import asyncio

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Global test configuration
def pytest_configure(config):
    """Configure pytest with custom options and markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "mcp: marks tests as MCP protocol tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "mcp" in str(item.fspath):
            item.add_marker(pytest.mark.mcp)

# Shared fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Get path to test data directory"""
    return Path(__file__).parent / "fixtures" / "data"

@pytest.fixture(scope="session")
def temp_test_dir():
    """Create temporary directory for test outputs"""
    with tempfile.TemporaryDirectory(prefix="pipeline_test_") as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_m3_optimizer():
    """Mock M3 optimizer for testing"""
    mock_optimizer = Mock()
    mock_optimizer.optimize_m3_system.return_value = {
        "status": "optimized",
        "memory_pools_created": 4,
        "estimated_throughput": "200-400 RPS",
        "memory_efficiency": "90%+",
        "performance_estimates": {
            "inference_latency_ms": "50-200ms",
            "concurrent_capacity": "8 parallel requests"
        }
    }
    return mock_optimizer

@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor for testing"""
    mock_monitor = Mock()
    mock_monitor.start_monitoring.return_value = None
    mock_monitor.stop_monitoring.return_value = None
    mock_monitor.get_memory_usage_mb.return_value = 512.5
    mock_monitor.get_average_memory_mb.return_value = 384.2
    mock_monitor.peak_memory_mb = 768.8
    return mock_monitor

@pytest.fixture
def mock_quality_controller():
    """Mock quality control framework for testing"""
    from quality_control_framework import QualityMetrics
    
    mock_controller = Mock()
    mock_controller.validate_record.return_value = (
        True, 
        QualityMetrics(
            content_coherence=0.9,
            technical_accuracy=0.85,
            metadata_completeness=0.95,
            conversation_flow=0.88,
            terminology_consistency=0.92,
            overall_score=8.9
        ),
        []
    )
    return mock_controller

@pytest.fixture
def sample_pipeline_record():
    """Sample pipeline record for testing"""
    return {
        "messages": [
            {
                "role": "user",
                "content": "How do I configure the eNodeB for LTE handover optimization with RSRP thresholds?"
            },
            {
                "role": "assistant",
                "content": "To configure LTE handover optimization, you need to set the RSRP threshold in the RRC configuration. Use the following parameters: RSRP threshold to -110 dBm, time-to-trigger to 320ms, and hysteresis to 2 dB for optimal performance."
            }
        ],
        "metadata": {
            "feature_name": "LTE Handover Optimization",
            "quality_score": 9.2,
            "technical_content": True,
            "technical_terms": ["eNodeB", "LTE", "RRC", "RSRP", "handover"],
            "processing_time": 1.45,
            "model_used": "qwen3:7b"
        }
    }

@pytest.fixture
def sample_pipeline_batch():
    """Sample batch of pipeline records for testing"""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is MIMO in 5G?"},
                {"role": "assistant", "content": "MIMO (Multiple-Input Multiple-Output) is a key 5G technology that uses multiple antennas to improve data throughput and reliability."}
            ],
            "metadata": {"quality_score": 8.5, "technical_terms": ["MIMO", "5G"]}
        },
        {
            "messages": [
                {"role": "user", "content": "How does carrier aggregation work?"},
                {"role": "assistant", "content": "Carrier aggregation combines multiple frequency bands to increase bandwidth and data rates in LTE and 5G networks."}
            ],
            "metadata": {"quality_score": 9.1, "technical_terms": ["carrier aggregation", "LTE", "5G"]}
        },
        {
            "messages": [
                {"role": "user", "content": "Explain beamforming benefits"},
                {"role": "assistant", "content": "Beamforming directs radio signals toward specific users, improving signal quality and reducing interference in wireless networks."}
            ],
            "metadata": {"quality_score": 8.8, "technical_terms": ["beamforming"]}
        }
    ]

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    mock_client = Mock()
    mock_client.generate.return_value = {
        "response": json.dumps({
            "main_features": ["Feature A", "Feature B"],
            "parameters": ["param1", "param2"],
            "quality_assessment": {"score": 8.5, "confidence": 0.9}
        })
    }
    mock_client.list.return_value = {
        "models": [
            {"name": "qwen3:7b", "size": "4.1GB"},
            {"name": "gemma2:9b", "size": "5.4GB"}
        ]
    }
    return mock_client

@pytest.fixture
def mock_lm_studio_client():
    """Mock LM Studio client for testing"""
    mock_client = AsyncMock()
    mock_client.completions.create = AsyncMock(return_value={
        "choices": [{
            "text": json.dumps({
                "extracted_features": ["Feature X", "Feature Y"],
                "quality_score": 9.0
            })
        }]
    })
    return mock_client

@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker for testing"""
    mock_breaker = Mock()
    mock_breaker.state = "closed"
    mock_breaker.failure_count = 0
    mock_breaker.call = Mock(return_value="success")
    mock_breaker.open = Mock()
    mock_breaker.close = Mock()
    mock_breaker.half_open = Mock()
    return mock_breaker

@pytest.fixture
def mock_memory_manager():
    """Mock unified memory manager for M3 testing"""
    mock_manager = Mock()
    mock_manager.create_memory_pool.return_value = True
    mock_manager.get_memory_utilization.return_value = {
        "total_system_gb": 128,
        "total_allocated_gb": 64.5,
        "available_gb": 63.5,
        "utilization_percent": 50.4,
        "pools": {"model_pool": 32.0, "batch_pool": 16.0, "cache_pool": 8.0, "working_pool": 8.5}
    }
    mock_manager.optimize_allocation_pattern.return_value = {
        "model_pool_gb": 32,
        "batch_pool_gb": 16,
        "cache_pool_gb": 8,
        "working_pool_gb": 12
    }
    return mock_manager

@pytest.fixture
def performance_baseline():
    """Performance baseline metrics for regression testing"""
    return {
        "processing_speed_rps": 150.0,  # records per second
        "memory_efficiency_mb": 2048.0,  # max memory usage
        "quality_score_threshold": 7.42,  # minimum quality score
        "model_switch_time_ms": 4500.0,  # model switching time
        "circuit_breaker_threshold": 5,  # failure threshold
        "inference_latency_p99_ms": 250.0,  # 99th percentile latency
        "concurrent_request_limit": 8  # maximum concurrent requests
    }

@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for protocol testing"""
    mock_server = Mock()
    mock_server.list_resources.return_value = [
        {
            "uri": "memory://test/resource1",
            "name": "Test Resource 1",
            "mimeType": "application/json"
        },
        {
            "uri": "memory://test/resource2", 
            "name": "Test Resource 2",
            "mimeType": "text/plain"
        }
    ]
    mock_server.read_resource.return_value = {
        "contents": [{"text": "Test resource content"}]
    }
    return mock_server

@pytest.fixture
def disable_external_dependencies():
    """Mock external dependencies to avoid network calls in tests"""
    with patch('requests.Session') as mock_session, \
         patch('aiohttp.ClientSession') as mock_async_session, \
         patch('ollama.Client') as mock_ollama, \
         patch('psutil.Process') as mock_process:
        
        # Mock requests session
        mock_session.return_value.post.return_value.json.return_value = {"response": "mocked"}
        mock_session.return_value.post.return_value.status_code = 200
        
        # Mock async session
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"result": "mocked"})
        mock_response.status = 200
        mock_async_session.return_value.post = AsyncMock(return_value=mock_response)
        
        # Mock ollama
        mock_ollama.return_value.generate.return_value = {"response": "mocked"}
        mock_ollama.return_value.list.return_value = {"models": []}
        
        # Mock psutil
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100
        mock_process.return_value.cpu_percent.return_value = 25.0
        
        yield {
            "requests": mock_session,
            "aiohttp": mock_async_session,
            "ollama": mock_ollama,
            "psutil": mock_process
        }

@pytest.fixture
def event_loop():
    """Create event loop for async testing"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Performance testing utilities
@pytest.fixture
def performance_test_config():
    """Configuration for performance tests"""
    return {
        "test_duration_seconds": 10,
        "warmup_duration_seconds": 2,
        "max_concurrent_requests": 8,
        "target_rps": 100,
        "memory_limit_mb": 2048,
        "quality_threshold": 7.0,
        "latency_threshold_ms": 500
    }

@pytest.fixture
def benchmark_dataset(test_data_dir):
    """Create benchmark dataset for performance testing"""
    benchmark_data = []
    
    # Generate test records
    for i in range(100):
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Test question about 5G feature {i}?"
                },
                {
                    "role": "assistant",
                    "content": f"Test answer about 5G feature {i} with technical details."
                }
            ],
            "metadata": {
                "quality_score": 8.0 + (i % 20) * 0.1,
                "technical_terms": ["5G", "NR", "gNodeB"],
                "test_id": f"benchmark_{i}"
            }
        }
        benchmark_data.append(record)
    
    # Save to file
    benchmark_file = test_data_dir / "benchmark_dataset.jsonl"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(benchmark_file, 'w') as f:
        for record in benchmark_data:
            f.write(json.dumps(record) + '\n')
    
    return benchmark_file

# Regression testing fixtures
@pytest.fixture
def regression_baseline_metrics():
    """Baseline metrics for regression testing"""
    return {
        "version": "1.0.0",
        "test_timestamp": "2024-01-01T00:00:00Z",
        "metrics": {
            "processing_speed": {
                "records_per_second": 150.0,
                "batch_processing_time_ms": 2000.0
            },
            "memory_usage": {
                "peak_memory_mb": 1024.0,
                "average_memory_mb": 512.0
            },
            "quality_scores": {
                "average_quality": 8.2,
                "min_quality_threshold": 7.0
            },
            "model_performance": {
                "switch_time_ms": 4000.0,
                "inference_latency_p95_ms": 200.0
            },
            "reliability": {
                "success_rate": 0.99,
                "circuit_breaker_triggers": 0
            }
        }
    }

# Test data factories
class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_conversation_record(quality_score: float = 8.0, 
                                 technical_terms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a conversation record with specified quality"""
        if technical_terms is None:
            technical_terms = ["LTE", "eNodeB", "5G"]
            
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I configure handover parameters?"
                },
                {
                    "role": "assistant",
                    "content": "Configure handover by setting RSRP threshold to -110 dBm and time-to-trigger to 320ms."
                }
            ],
            "metadata": {
                "quality_score": quality_score,
                "technical_terms": technical_terms,
                "processing_time": 1.2,
                "model_used": "qwen3:7b"
            }
        }
    
    @staticmethod
    def create_batch_records(count: int = 10, 
                           quality_range: tuple = (7.0, 9.5)) -> List[Dict[str, Any]]:
        """Create a batch of test records"""
        import random
        records = []
        
        for i in range(count):
            quality = random.uniform(quality_range[0], quality_range[1])
            record = TestDataFactory.create_conversation_record(quality)
            record["metadata"]["test_id"] = f"batch_record_{i}"
            records.append(record)
            
        return records

@pytest.fixture
def test_data_factory():
    """Test data factory fixture"""
    return TestDataFactory