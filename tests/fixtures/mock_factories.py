"""
Mock factories for creating test doubles and fixtures
Comprehensive mocking utilities for the Python pipeline testing
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from unittest.mock import Mock, MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass
import random


@dataclass
class MockResponse:
    """Mock response object for HTTP clients"""
    status_code: int = 200
    text: str = ""
    json_data: Dict[str, Any] = None
    
    def json(self):
        return self.json_data or {}


class OllamaMockFactory:
    """Factory for creating Ollama client mocks"""
    
    @staticmethod
    def create_ollama_client_mock():
        """Create comprehensive Ollama client mock"""
        mock_client = Mock()
        
        # Mock generate method
        def mock_generate(model, prompt, **kwargs):
            return {
                "response": json.dumps({
                    "main_features": ["Feature A", "Feature B"],
                    "parameters": ["RadioConfig.maxPower", "NetworkConfig.timeout"],
                    "counters": ["pmRadioUtilization", "pmNetworkLatency"],
                    "events": ["RadioFailure", "NetworkTimeout"],
                    "quality_assessment": {
                        "score": 8.5,
                        "confidence": 0.9,
                        "technical_terms_count": 5
                    }
                }),
                "model": model,
                "created_at": "2024-01-01T12:00:00Z",
                "done": True,
                "total_duration": 1500000000,  # nanoseconds
                "load_duration": 500000000,
                "prompt_eval_count": 50,
                "prompt_eval_duration": 200000000,
                "eval_count": 100,
                "eval_duration": 800000000
            }
        
        mock_client.generate = Mock(side_effect=mock_generate)
        
        # Mock list method
        mock_client.list = Mock(return_value={
            "models": [
                {
                    "name": "qwen3:7b",
                    "size": 4365492224,
                    "digest": "abc123",
                    "modified_at": "2024-01-01T12:00:00Z"
                },
                {
                    "name": "gemma2:9b",
                    "size": 5698547712,
                    "digest": "def456",
                    "modified_at": "2024-01-01T12:00:00Z"
                }
            ]
        })
        
        # Mock pull method
        mock_client.pull = Mock(return_value={"status": "success"})
        
        # Mock show method
        mock_client.show = Mock(return_value={
            "license": "MIT",
            "modelfile": "FROM qwen3:7b",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9
            },
            "template": "{{ .System }}\n{{ .Prompt }}",
            "system": "You are a helpful assistant."
        })
        
        return mock_client


class LMStudioMockFactory:
    """Factory for creating LM Studio client mocks"""
    
    @staticmethod
    def create_lm_studio_mock():
        """Create LM Studio API mock"""
        mock_client = AsyncMock()
        
        async def mock_completions_create(**kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "id": "cmpl-123456",
                "object": "text_completion",
                "created": int(time.time()),
                "model": kwargs.get("model", "qwen3-7b"),
                "choices": [{
                    "text": json.dumps({
                        "extracted_features": ["Advanced Radio Management", "Network Optimization"],
                        "technical_parameters": ["RSRP", "RSRQ", "CQI"],
                        "quality_score": 9.2,
                        "confidence": 0.95
                    }),
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 150,
                    "total_tokens": 250
                }
            }
        
        mock_client.completions.create = mock_completions_create
        
        # Mock models list
        mock_client.models.list = AsyncMock(return_value={
            "object": "list",
            "data": [
                {
                    "id": "qwen3-7b",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "lm-studio"
                },
                {
                    "id": "gemma2-9b", 
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "lm-studio"
                }
            ]
        })
        
        return mock_client


class CircuitBreakerMockFactory:
    """Factory for creating circuit breaker mocks"""
    
    @staticmethod
    def create_circuit_breaker_mock(initial_state: str = "closed"):
        """Create circuit breaker mock with configurable behavior"""
        mock_breaker = Mock()
        
        # State management
        mock_breaker.state = initial_state
        mock_breaker.failure_count = 0
        mock_breaker.failure_threshold = 5
        mock_breaker.recovery_timeout = 30
        mock_breaker.half_open_max_calls = 3
        
        # Behavior simulation
        def mock_call(func, *args, **kwargs):
            if mock_breaker.state == "open":
                raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs) if callable(func) else func
                mock_breaker.failure_count = max(0, mock_breaker.failure_count - 1)
                return result
            except Exception as e:
                mock_breaker.failure_count += 1
                if mock_breaker.failure_count >= mock_breaker.failure_threshold:
                    mock_breaker.state = "open"
                raise e
        
        def mock_open():
            mock_breaker.state = "open"
        
        def mock_close():
            mock_breaker.state = "closed"
            mock_breaker.failure_count = 0
        
        def mock_half_open():
            mock_breaker.state = "half_open"
        
        mock_breaker.call = Mock(side_effect=mock_call)
        mock_breaker.open = Mock(side_effect=mock_open)
        mock_breaker.close = Mock(side_effect=mock_close) 
        mock_breaker.half_open = Mock(side_effect=mock_half_open)
        
        return mock_breaker


class PerformanceMonitorMockFactory:
    """Factory for creating performance monitor mocks"""
    
    @staticmethod
    def create_performance_monitor_mock():
        """Create performance monitor mock with realistic metrics"""
        mock_monitor = Mock()
        
        # Simulate memory usage over time
        base_memory = 512.0
        memory_samples = []
        
        def get_memory_usage():
            # Simulate memory fluctuation
            current = base_memory + random.uniform(-50, 100)
            memory_samples.append(current)
            return current
        
        def get_average_memory():
            return sum(memory_samples) / len(memory_samples) if memory_samples else base_memory
        
        mock_monitor.get_memory_usage_mb = Mock(side_effect=get_memory_usage)
        mock_monitor.get_average_memory_mb = Mock(side_effect=get_average_memory)
        mock_monitor.peak_memory_mb = 768.5
        mock_monitor.start_monitoring = Mock()
        mock_monitor.stop_monitoring = Mock()
        
        # CPU monitoring
        mock_monitor.get_cpu_percent = Mock(return_value=45.2)
        
        # Memory samples for testing
        mock_monitor.memory_samples = memory_samples
        
        return mock_monitor


class QualityControllerMockFactory:
    """Factory for creating quality controller mocks"""
    
    @staticmethod
    def create_quality_controller_mock(default_quality_score: float = 8.5):
        """Create quality controller mock with configurable scoring"""
        from quality_control_framework import QualityMetrics
        
        mock_controller = Mock()
        
        def mock_validate_record(record):
            # Simulate validation logic
            messages = record.get("messages", [])
            metadata = record.get("metadata", {})
            
            base_quality = metadata.get("quality_score", default_quality_score)
            
            # Adjust based on content
            content_length = sum(len(msg.get("content", "")) for msg in messages)
            if content_length < 50:
                base_quality *= 0.7
            elif content_length > 1000:
                base_quality *= 1.1
            
            # Create metrics
            metrics = QualityMetrics(
                content_coherence=min(1.0, base_quality / 10.0),
                technical_accuracy=min(1.0, (base_quality + 1) / 10.0),
                metadata_completeness=0.95 if metadata else 0.5,
                conversation_flow=min(1.0, base_quality / 9.0),
                terminology_consistency=0.9,
                overall_score=min(10.0, base_quality)
            )
            
            is_valid = base_quality >= 7.0 and len(messages) >= 2
            errors = [] if is_valid else ["Quality score too low"]
            
            return is_valid, metrics, errors
        
        mock_controller.validate_record = Mock(side_effect=mock_validate_record)
        
        return mock_controller


class M3OptimizerMockFactory:
    """Factory for creating M3 optimizer mocks"""
    
    @staticmethod
    def create_m3_optimizer_mock():
        """Create M3 optimizer mock with realistic responses"""
        mock_optimizer = Mock()
        
        # Memory manager mock
        memory_manager = Mock()
        memory_manager.create_memory_pool = Mock(return_value=True)
        memory_manager.get_memory_utilization = Mock(return_value={
            "total_system_gb": 128,
            "total_allocated_gb": 64.5,
            "available_gb": 63.5,
            "utilization_percent": 50.4,
            "pools": {
                "model_pool": 32.0,
                "batch_pool": 16.0,
                "cache_pool": 8.0,
                "working_pool": 8.5
            }
        })
        memory_manager.optimize_allocation_pattern = Mock(return_value={
            "model_pool_gb": 32,
            "batch_pool_gb": 16,
            "cache_pool_gb": 8,
            "working_pool_gb": 12
        })
        
        mock_optimizer.memory_manager = memory_manager
        
        # MLX optimizer mock
        mlx_optimizer = Mock()
        mlx_optimizer.optimize_model_loading = Mock(return_value={
            "status": "optimized",
            "settings": {
                "memory_mapping": True,
                "lazy_loading": True,
                "quantization": "int4",
                "batch_size": 512
            },
            "estimated_memory_gb": 8.5,
            "recommended_batch_size": 512
        })
        
        mock_optimizer.mlx_optimizer = mlx_optimizer
        
        # Main optimization method
        async def mock_optimize_inference_pipeline(model_path, workload_type):
            await asyncio.sleep(0.1)  # Simulate optimization time
            return {
                "timestamp": time.time(),
                "optimization_complete": True,
                "system_specs": {
                    "model": "MacBook Pro M3 Max",
                    "unified_memory_gb": 128,
                    "cpu_cores": 16
                },
                "memory_management": {
                    "pools_created": 4,
                    "utilization": memory_manager.get_memory_utilization()
                },
                "performance_estimates": {
                    "inference_latency_ms": "50-200ms for typical requests",
                    "throughput_rps": "16-32",
                    "memory_efficiency": "85-95% utilization",
                    "concurrent_capacity": "8 parallel requests"
                }
            }
        
        mock_optimizer.optimize_inference_pipeline = AsyncMock(
            side_effect=mock_optimize_inference_pipeline
        )
        
        return mock_optimizer


class MCPServerMockFactory:
    """Factory for creating MCP server mocks"""
    
    @staticmethod
    def create_mcp_server_mock():
        """Create MCP server mock for protocol testing"""
        mock_server = Mock()
        
        # Resource management
        mock_server.list_resources = Mock(return_value=[
            {
                "uri": "memory://pipeline/processing_stats",
                "name": "Processing Statistics",
                "mimeType": "application/json",
                "description": "Pipeline processing statistics"
            },
            {
                "uri": "memory://pipeline/quality_metrics",
                "name": "Quality Metrics",
                "mimeType": "application/json",
                "description": "Quality assessment metrics"
            },
            {
                "uri": "memory://pipeline/performance_data",
                "name": "Performance Data",
                "mimeType": "application/json",
                "description": "Performance monitoring data"
            }
        ])
        
        def mock_read_resource(uri):
            if "processing_stats" in uri:
                return {
                    "contents": [{
                        "text": json.dumps({
                            "total_processed": 1000,
                            "success_rate": 0.98,
                            "average_quality": 8.2,
                            "processing_time_ms": 15000
                        })
                    }]
                }
            elif "quality_metrics" in uri:
                return {
                    "contents": [{
                        "text": json.dumps({
                            "content_coherence": 0.9,
                            "technical_accuracy": 0.85,
                            "overall_score": 8.7
                        })
                    }]
                }
            elif "performance_data" in uri:
                return {
                    "contents": [{
                        "text": json.dumps({
                            "memory_usage_mb": 1024,
                            "cpu_percent": 65,
                            "records_per_second": 67
                        })
                    }]
                }
            else:
                return {"contents": [{"text": "Resource not found"}]}
        
        mock_server.read_resource = Mock(side_effect=mock_read_resource)
        
        # Tool execution
        def mock_call_tool(name, arguments):
            if name == "memory_store":
                return {"success": True, "stored": True}
            elif name == "memory_retrieve":
                return {
                    "success": True,
                    "data": {"test_key": "test_value"}
                }
            elif name == "performance_report":
                return {
                    "success": True,
                    "report": {
                        "processing_speed": "150 rps",
                        "memory_efficiency": "90%",
                        "quality_score": 8.5
                    }
                }
            else:
                return {"error": f"Unknown tool: {name}"}
        
        mock_server.call_tool = Mock(side_effect=mock_call_tool)
        
        return mock_server


class FilesystemMockFactory:
    """Factory for creating filesystem mocks"""
    
    @staticmethod
    def create_file_mock(content: str = "", exists: bool = True):
        """Create mock file object"""
        mock_file = Mock()
        mock_file.read.return_value = content
        mock_file.write = Mock()
        mock_file.exists.return_value = exists
        mock_file.stat.return_value.st_size = len(content)
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        return mock_file
    
    @staticmethod
    def create_pathlib_mock(files_content: Dict[str, str]):
        """Create comprehensive pathlib mock"""
        
        def mock_path_factory(path_str):
            mock_path = Mock()
            mock_path.name = path_str.split('/')[-1]
            mock_path.suffix = '.' + path_str.split('.')[-1] if '.' in path_str else ''
            mock_path.parent = Mock()
            mock_path.parent.mkdir = Mock()
            
            # File operations
            def mock_exists():
                return path_str in files_content
            
            def mock_read_text():
                return files_content.get(path_str, "")
            
            def mock_write_text(content):
                files_content[path_str] = content
            
            def mock_open(mode='r'):
                content = files_content.get(path_str, "")
                return FilesystemMockFactory.create_file_mock(content)
            
            mock_path.exists = Mock(side_effect=mock_exists)
            mock_path.read_text = Mock(side_effect=mock_read_text)
            mock_path.write_text = Mock(side_effect=mock_write_text)
            mock_path.open = Mock(side_effect=mock_open)
            
            # Path operations
            mock_path.__str__ = Mock(return_value=path_str)
            mock_path.__truediv__ = Mock(side_effect=lambda x: mock_path_factory(f"{path_str}/{x}"))
            
            return mock_path
        
        return mock_path_factory


class DatabaseMockFactory:
    """Factory for creating database mocks"""
    
    @staticmethod
    def create_memory_db_mock():
        """Create in-memory database mock"""
        storage = {}
        
        mock_db = Mock()
        
        def mock_store(key, value, namespace="default", ttl=None):
            storage[f"{namespace}:{key}"] = {
                "value": value,
                "ttl": ttl,
                "created_at": time.time()
            }
            return True
        
        def mock_retrieve(key, namespace="default"):
            full_key = f"{namespace}:{key}"
            if full_key in storage:
                return storage[full_key]["value"]
            return None
        
        def mock_search(pattern, namespace="default", limit=10):
            results = []
            prefix = f"{namespace}:"
            for key, data in storage.items():
                if key.startswith(prefix) and pattern in key:
                    results.append({
                        "key": key[len(prefix):],
                        "value": data["value"]
                    })
                    if len(results) >= limit:
                        break
            return results
        
        mock_db.store = Mock(side_effect=mock_store)
        mock_db.retrieve = Mock(side_effect=mock_retrieve)
        mock_db.search = Mock(side_effect=mock_search)
        mock_db.delete = Mock(return_value=True)
        
        # Expose storage for testing
        mock_db._storage = storage
        
        return mock_db


def create_comprehensive_mock_environment():
    """Create a comprehensive mock environment for testing"""
    return {
        "ollama_client": OllamaMockFactory.create_ollama_client_mock(),
        "lm_studio_client": LMStudioMockFactory.create_lm_studio_mock(),
        "circuit_breaker": CircuitBreakerMockFactory.create_circuit_breaker_mock(),
        "performance_monitor": PerformanceMonitorMockFactory.create_performance_monitor_mock(),
        "quality_controller": QualityControllerMockFactory.create_quality_controller_mock(),
        "m3_optimizer": M3OptimizerMockFactory.create_m3_optimizer_mock(),
        "mcp_server": MCPServerMockFactory.create_mcp_server_mock(),
        "memory_db": DatabaseMockFactory.create_memory_db_mock()
    }


# Context managers for mock environments
class MockEnvironmentManager:
    """Context manager for setting up mock environments"""
    
    def __init__(self, mock_config: Dict[str, Any]):
        self.mock_config = mock_config
        self.patches = []
    
    def __enter__(self):
        # Apply patches based on configuration
        for mock_name, mock_obj in self.mock_config.items():
            if mock_name == "ollama_client":
                patcher = patch('ollama.Client', return_value=mock_obj)
            elif mock_name == "psutil_process":
                patcher = patch('psutil.Process', return_value=mock_obj)
            elif mock_name == "requests_session":
                patcher = patch('requests.Session', return_value=mock_obj)
            else:
                continue
            
            self.patches.append(patcher.start())
        
        return self.mock_config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patcher in reversed(self.patches):
            try:
                patcher.stop()
            except:
                pass
class MockDocumentFactory:
    """Factory for creating mock documents for testing"""
    
    def __init__(self):
        self.document_counter = 0
    
    def generate_single_document(self, doc_id: str = None) -> Dict[str, Any]:
        """Generate a single mock document"""
        if doc_id is None:
            doc_id = f"doc_{self.document_counter}"
            self.document_counter += 1
        
        return {
            "id": doc_id,
            "content": f"This is test document content for {doc_id}. " * random.randint(10, 50),
            "metadata": {
                "title": f"Test Document {doc_id}",
                "type": "technical",
                "size": random.randint(1000, 10000),
                "created_at": time.time(),
                "technical_terms": ["LTE", "5G", "eNodeB", "gNodeB"],
                "quality_score": random.uniform(7.0, 9.5)
            }
        }
    
    def generate_mixed_documents(self, count: int) -> List[Dict[str, Any]]:
        """Generate a mix of different document types"""
        documents = []
        
        for i in range(count):
            doc_type = random.choice(["technical", "configuration", "troubleshooting"])
            size_category = random.choice(["small", "medium", "large"])
            
            if size_category == "small":
                content_multiplier = random.randint(5, 15)
            elif size_category == "medium":
                content_multiplier = random.randint(20, 50)
            else:  # large
                content_multiplier = random.randint(60, 100)
            
            document = {
                "id": f"mixed_doc_{i}",
                "content": f"This is {doc_type} document content. " * content_multiplier,
                "metadata": {
                    "title": f"Mixed Document {i}",
                    "type": doc_type,
                    "size_category": size_category,
                    "size": len(f"This is {doc_type} document content. " * content_multiplier),
                    "created_at": time.time(),
                    "technical_terms": self._get_technical_terms_by_type(doc_type),
                    "quality_score": random.uniform(7.0, 9.5)
                }
            }
            documents.append(document)
        
        return documents
    
    def _get_technical_terms_by_type(self, doc_type: str) -> List[str]:
        """Get technical terms based on document type"""
        terms_map = {
            "technical": ["LTE", "5G", "NR", "eNodeB", "gNodeB", "MIMO", "beamforming"],
            "configuration": ["RRC", "PDCP", "MAC", "PHY", "parameters", "settings"],
            "troubleshooting": ["alarms", "KPIs", "performance", "optimization", "debugging"]
        }
        available_terms = terms_map.get(doc_type, ["generic", "term"])
        sample_size = min(random.randint(3, 6), len(available_terms))
        return random.sample(available_terms, k=sample_size)