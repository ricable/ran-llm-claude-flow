#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Python testing
"""

import pytest
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Pytest configuration
pytest_plugins = []

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for HTTP requests"""
    session = AsyncMock()
    
    # Mock successful model list response
    mock_models_response = AsyncMock()
    mock_models_response.status = 200
    mock_models_response.json = AsyncMock(return_value={
        "data": [
            {"id": "qwen3-1.7b", "name": "Qwen3 1.7B", "owned_by": "lmstudio"},
            {"id": "qwen3-7b", "name": "Qwen3 7B", "owned_by": "lmstudio"},
            {"id": "qwen3-14b", "name": "Qwen3 14B", "owned_by": "lmstudio"}
        ]
    })
    
    # Mock successful chat completion response
    mock_chat_response = AsyncMock()
    mock_chat_response.status = 200
    mock_chat_response.json = AsyncMock(return_value={
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the mocked LLM."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    })
    
    # Configure session context managers
    session.get.return_value.__aenter__.return_value = mock_models_response
    session.post.return_value.__aenter__.return_value = mock_chat_response
    
    return session

@pytest.fixture
def mock_ollama_session():
    """Mock Ollama HTTP session"""
    session = AsyncMock()
    
    # Mock Ollama tags response
    mock_tags_response = AsyncMock()
    mock_tags_response.status = 200
    mock_tags_response.json = AsyncMock(return_value={
        "models": [
            {"name": "qwen2.5:1.5b-q4_0", "size": 1200000000},
            {"name": "qwen2.5:7b-q8_0", "size": 7200000000},
            {"name": "qwen2.5:14b-q8_0", "size": 14500000000}
        ]
    })
    
    # Mock Ollama generate response
    mock_generate_response = AsyncMock()
    mock_generate_response.status = 200
    mock_generate_response.json = AsyncMock(return_value={
        "response": "This is a test response from mocked Ollama.",
        "done": True
    })
    
    # Mock embeddings response
    mock_embeddings_response = AsyncMock()
    mock_embeddings_response.status = 200
    mock_embeddings_response.json = AsyncMock(return_value={
        "embedding": [0.1] * 768  # Mock 768-dim embedding
    })
    
    session.get.return_value.__aenter__.return_value = mock_tags_response
    session.post.return_value.__aenter__.return_value = mock_generate_response
    
    return session

@pytest.fixture
def mock_mlx_available():
    """Mock MLX availability"""
    with patch('docs.python.integration.mlx_accelerator.MLX_AVAILABLE', True):
        # Mock MLX modules
        mock_mx = MagicMock()
        mock_mx.default_device.return_value = "gpu"
        mock_mx.metal.get_memory_limit.return_value = 128 * 1024**3  # 128GB
        mock_mx.metal.get_active_memory.return_value = 8 * 1024**3   # 8GB
        mock_mx.metal.clear_cache = MagicMock()
        
        with patch('docs.python.integration.mlx_accelerator.mx', mock_mx):
            # Mock MLX-LM functions
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            
            with patch('docs.python.integration.mlx_accelerator.load', return_value=(mock_model, mock_tokenizer)):
                with patch('docs.python.integration.mlx_accelerator.generate', return_value="Test MLX response"):
                    yield {
                        'mx': mock_mx,
                        'model': mock_model,
                        'tokenizer': mock_tokenizer
                    }

@pytest.fixture
def sample_inference_requests():
    """Sample inference requests for testing"""
    return [
        {
            "id": "test_1",
            "prompt": "What is 5G NR technology?",
            "model_variant": "7b",
            "max_tokens": 200,
            "temperature": 0.7,
            "priority": "normal"
        },
        {
            "id": "test_2",
            "prompt": "Explain MIMO technology in detail with technical specifications.",
            "model_variant": "14b", 
            "max_tokens": 500,
            "temperature": 0.5,
            "priority": "high"
        },
        {
            "id": "test_3",
            "prompt": "Quick summary",
            "model_variant": "1.7b",
            "max_tokens": 100,
            "temperature": 0.9,
            "priority": "critical"
        }
    ]

@pytest.fixture
def sample_embedding_requests():
    """Sample embedding requests for testing"""
    return {
        "texts": [
            "5G NR technology overview",
            "MIMO antenna systems", 
            "mmWave frequency benefits",
            "Network slicing concepts",
            "Edge computing in 5G"
        ],
        "model": "qwen3-embed",
        "batch_size": 16,
        "normalize": True
    }

@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics for testing"""
    return {
        "processing_times": [0.5, 0.7, 0.3, 0.9, 0.4],
        "tokens_per_second": [45.2, 38.7, 67.1, 29.3, 52.8],
        "success_rates": [1.0, 1.0, 1.0, 0.0, 1.0],
        "memory_usage_gb": [2.1, 2.3, 1.8, 2.5, 2.0]
    }

@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Auto-mock external dependencies that aren't core to testing"""
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value.available = 64 * 1024**3  # 64GB available
        mock_memory.return_value.used = 32 * 1024**3      # 32GB used
        mock_memory.return_value.total = 128 * 1024**3    # 128GB total
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1024**3  # 1GB
            yield

# Async test helper functions
async def async_test_wrapper(async_func, *args, **kwargs):
    """Wrapper for async test functions"""
    return await async_func(*args, **kwargs)

def pytest_collection_modifyitems(config, items):
    """Add async marker to async test functions"""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

# Performance test configuration
@pytest.fixture
def performance_test_config():
    """Configuration for performance tests"""
    return {
        "max_response_time": 5.0,  # seconds
        "min_tokens_per_second": 10.0,
        "max_memory_usage_gb": 50.0,
        "success_rate_threshold": 0.95,
        "concurrent_requests": 5
    }

# Coverage report helper
def generate_coverage_report():
    """Generate coverage report after tests"""
    import coverage
    cov = coverage.Coverage()
    cov.start()
    # Tests run here
    cov.stop()
    cov.save()
    
    # Generate HTML report
    cov.html_report(directory='tests/python/htmlcov')
    cov.report()