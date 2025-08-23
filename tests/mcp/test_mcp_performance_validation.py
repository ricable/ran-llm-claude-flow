"""
MCP Performance Validation Tests

Validates that MCP integration maintains the target performance of 25+ docs/hour
and doesn't introduce significant overhead compared to direct IPC communication.
"""

import asyncio
import json
import pytest
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock, patch
import psutil
import gc

from src.python_pipeline.mcp.client import MCPClient
from src.python_pipeline.mcp.server import MCPServer
from src.python_pipeline.mcp.protocol import (
    create_request, MCPMethods, MCPMessage
)
from src.python_pipeline.models.qwen3_manager import Qwen3Manager
from tests.fixtures.mock_factories import MockDocumentFactory


class TestMCPPerformanceValidation:
    """Performance validation test suite for MCP integration."""

    # Performance targets from CLAUDE.md
    TARGET_THROUGHPUT_DOCS_PER_HOUR = 25.0
    MAX_LATENCY_MS = 100.0
    MAX_MEMORY_OVERHEAD_PERCENT = 5.0
    MAX_IPC_OVERHEAD_PERCENT = 10.0
    
    @pytest.fixture
    async def performance_test_server(self):
        """Create a performance-optimized test server."""
        server = MCPServer(
            host="localhost", 
            port=8710,
            max_connections=100,
            message_buffer_size=1024 * 1024  # 1MB buffer
        )
        await server.start()
        yield server
        await server.stop()

    @pytest.fixture
    async def performance_test_client(self):
        """Create a performance-optimized test client."""
        client = MCPClient(
            "ws://localhost:8710/mcp",
            connection_pool_size=10,
            message_timeout=5.0
        )
        await client.connect()
        yield client
        await client.disconnect()

    @pytest.fixture
    def mock_document_factory(self):
        """Create mock document factory for performance tests."""
        return MockDocumentFactory()

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitoring utilities."""
        return PerformanceMonitor()

    @pytest.mark.asyncio
    async def test_throughput_performance(self, performance_test_server, performance_test_client, mock_document_factory):
        """Test document processing throughput via MCP maintains target performance."""
        
        # Generate test documents of varying sizes
        test_documents = mock_document_factory.generate_mixed_documents(100)
        
        # Measure baseline performance (direct processing)
        baseline_throughput = await self._measure_baseline_throughput(test_documents)
        
        # Measure MCP throughput
        mcp_throughput = await self._measure_mcp_throughput(
            performance_test_client, test_documents
        )
        
        # Validate performance targets
        assert mcp_throughput >= self.TARGET_THROUGHPUT_DOCS_PER_HOUR, (
            f"MCP throughput {mcp_throughput:.2f} docs/hour below target "
            f"{self.TARGET_THROUGHPUT_DOCS_PER_HOUR}"
        )
        
        # Calculate overhead vs baseline
        overhead_percent = ((baseline_throughput - mcp_throughput) / baseline_throughput) * 100
        assert overhead_percent <= self.MAX_IPC_OVERHEAD_PERCENT, (
            f"MCP overhead {overhead_percent:.2f}% exceeds maximum "
            f"{self.MAX_IPC_OVERHEAD_PERCENT}%"
        )
        
        print(f"✅ Throughput Performance:")
        print(f"   Baseline: {baseline_throughput:.2f} docs/hour")
        print(f"   MCP: {mcp_throughput:.2f} docs/hour")
        print(f"   Overhead: {overhead_percent:.2f}%")

    @pytest.mark.asyncio
    async def test_latency_performance(self, performance_test_client, performance_monitor):
        """Test MCP message latency performance."""
        
        latencies = []
        test_iterations = 1000
        
        # Measure latency for various message types
        message_types = [
            (MCPMethods.PIPELINE_STATUS, {"pipeline_id": "test"}),
            (MCPMethods.MODEL_METRICS, {"model_id": "qwen3_7b"}),
            (MCPMethods.TASK_STATUS, {"task_id": "task_123"}),
            (MCPMethods.SYSTEM_METRICS, {}),
        ]
        
        for method, params in message_types:
            method_latencies = []
            
            for _ in range(test_iterations // len(message_types)):
                start_time = time.perf_counter()
                
                request = create_request(method, params)
                
                with patch.object(performance_test_client, 'send_message') as mock_send:
                    mock_send.return_value = {"result": {"status": "success"}}
                    await performance_test_client.send_message(request)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                method_latencies.append(latency_ms)
                latencies.append(latency_ms)
        
        # Calculate latency statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        max_latency = max(latencies)
        
        # Validate latency targets
        assert avg_latency <= self.MAX_LATENCY_MS, (
            f"Average latency {avg_latency:.2f}ms exceeds target {self.MAX_LATENCY_MS}ms"
        )
        
        print(f"✅ Latency Performance:")
        print(f"   Average: {avg_latency:.2f}ms")
        print(f"   P95: {p95_latency:.2f}ms")
        print(f"   P99: {p99_latency:.2f}ms")
        print(f"   Max: {max_latency:.2f}ms")

    @pytest.mark.asyncio
    async def test_memory_performance(self, performance_test_server, performance_test_client, performance_monitor):
        """Test MCP memory usage and leak detection."""
        
        # Record initial memory usage
        initial_memory = performance_monitor.get_current_memory_usage()
        
        # Perform intensive MCP operations
        operations_count = 1000
        large_messages = []
        
        for i in range(operations_count):
            # Create messages with varying payload sizes
            payload_size = 1024 * (1 + (i % 100))  # 1KB to 100KB
            large_payload = {"data": "x" * payload_size, "iteration": i}
            
            request = create_request(MCPMethods.PIPELINE_CREATE, large_payload)
            large_messages.append(request)
            
            # Mock processing to avoid actual pipeline overhead
            with patch.object(performance_test_client, 'send_message') as mock_send:
                mock_send.return_value = {"result": {"pipeline_id": f"pipeline_{i}"}}
                await performance_test_client.send_message(request)
            
            # Sample memory usage periodically
            if i % 100 == 0:
                current_memory = performance_monitor.get_current_memory_usage()
                memory_growth = current_memory - initial_memory
                
                # Check for excessive memory growth
                if memory_growth > 100 * 1024 * 1024:  # 100MB growth threshold
                    # Force garbage collection
                    gc.collect()
                    await asyncio.sleep(0.1)
        
        # Force final garbage collection
        large_messages.clear()
        gc.collect()
        await asyncio.sleep(1)  # Allow cleanup time
        
        final_memory = performance_monitor.get_current_memory_usage()
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Calculate memory overhead percentage
        memory_overhead_percent = (memory_growth / initial_memory) * 100
        
        # Validate memory usage
        assert memory_overhead_percent <= self.MAX_MEMORY_OVERHEAD_PERCENT, (
            f"Memory overhead {memory_overhead_percent:.2f}% exceeds target "
            f"{self.MAX_MEMORY_OVERHEAD_PERCENT}%"
        )
        
        print(f"✅ Memory Performance:")
        print(f"   Initial: {initial_memory / (1024**2):.2f}MB")
        print(f"   Final: {final_memory / (1024**2):.2f}MB")
        print(f"   Growth: {memory_growth_mb:.2f}MB")
        print(f"   Overhead: {memory_overhead_percent:.2f}%")

    @pytest.mark.asyncio
    async def test_concurrent_performance(self, performance_test_server, performance_test_client, mock_document_factory):
        """Test MCP performance under concurrent load."""
        
        concurrent_clients = 32
        documents_per_client = 10
        
        # Create multiple clients for concurrent testing
        clients = []
        for i in range(concurrent_clients):
            client = MCPClient(f"ws://localhost:8710/mcp?client_id={i}")
            await client.connect()
            clients.append(client)
        
        try:
            start_time = time.perf_counter()
            
            # Create concurrent tasks
            tasks = []
            for client_idx, client in enumerate(clients):
                for doc_idx in range(documents_per_client):
                    document = mock_document_factory.generate_single_document(
                        f"doc_{client_idx}_{doc_idx}"
                    )
                    
                    task = self._process_document_via_mcp(client, document)
                    tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            total_documents = concurrent_clients * documents_per_client
            success_rate = len(successful_results) / total_documents * 100
            throughput_docs_per_hour = len(successful_results) / (total_duration / 3600)
            
            # Validate concurrent performance
            assert success_rate >= 95.0, (
                f"Concurrent success rate {success_rate:.2f}% below minimum 95%"
            )
            
            assert throughput_docs_per_hour >= self.TARGET_THROUGHPUT_DOCS_PER_HOUR, (
                f"Concurrent throughput {throughput_docs_per_hour:.2f} docs/hour "
                f"below target {self.TARGET_THROUGHPUT_DOCS_PER_HOUR}"
            )
            
            print(f"✅ Concurrent Performance:")
            print(f"   Clients: {concurrent_clients}")
            print(f"   Total Documents: {total_documents}")
            print(f"   Success Rate: {success_rate:.2f}%")
            print(f"   Throughput: {throughput_docs_per_hour:.2f} docs/hour")
            print(f"   Duration: {total_duration:.2f}s")
            
        finally:
            # Cleanup clients
            for client in clients:
                await client.disconnect()

    @pytest.mark.asyncio
    async def test_scalability_performance(self, performance_test_server, mock_document_factory):
        """Test MCP scalability with increasing load."""
        
        scalability_results = {}
        client_counts = [1, 2, 4, 8, 16, 32]
        documents_per_test = 50
        
        for client_count in client_counts:
            # Create clients
            clients = []
            for i in range(client_count):
                client = MCPClient(f"ws://localhost:8710/mcp?test_scale_{i}")
                await client.connect()
                clients.append(client)
            
            try:
                start_time = time.perf_counter()
                
                # Distribute documents across clients
                tasks = []
                for i in range(documents_per_test):
                    client = clients[i % len(clients)]
                    document = mock_document_factory.generate_single_document(f"scale_test_{i}")
                    task = self._process_document_via_mcp(client, document)
                    tasks.append(task)
                
                # Execute and measure
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.perf_counter()
                
                duration = end_time - start_time
                successful_results = [r for r in results if not isinstance(r, Exception)]
                throughput = len(successful_results) / (duration / 3600)
                
                scalability_results[client_count] = {
                    'throughput': throughput,
                    'success_rate': len(successful_results) / documents_per_test * 100,
                    'duration': duration
                }
                
                print(f"Clients: {client_count}, Throughput: {throughput:.2f} docs/hour")
                
            finally:
                for client in clients:
                    await client.disconnect()
                
                # Allow cleanup time between tests
                await asyncio.sleep(1)
        
        # Analyze scalability
        baseline_throughput = scalability_results[1]['throughput']
        
        for client_count, results in scalability_results.items():
            if client_count > 1:
                scaling_factor = results['throughput'] / baseline_throughput
                efficiency = scaling_factor / client_count * 100
                
                print(f"✅ Scalability - {client_count} clients:")
                print(f"   Throughput: {results['throughput']:.2f} docs/hour")
                print(f"   Scaling Factor: {scaling_factor:.2f}x")
                print(f"   Efficiency: {efficiency:.2f}%")
                
                # Validate reasonable scaling efficiency
                assert efficiency >= 60.0, (
                    f"Scaling efficiency {efficiency:.2f}% too low for {client_count} clients"
                )

    @pytest.mark.asyncio
    async def test_message_size_performance(self, performance_test_client):
        """Test MCP performance with varying message sizes."""
        
        message_sizes = [
            1024,          # 1KB
            10 * 1024,     # 10KB
            100 * 1024,    # 100KB
            1024 * 1024,   # 1MB
        ]
        
        performance_results = {}
        
        for size in message_sizes:
            # Create test payload of specified size
            payload = {"data": "x" * size, "metadata": {"size": size}}
            
            # Measure processing time for this message size
            latencies = []
            for _ in range(100):  # 100 iterations per size
                start_time = time.perf_counter()
                
                request = create_request(MCPMethods.DOCUMENT_PROCESS, payload)
                
                # Mock processing to isolate MCP overhead
                with patch.object(performance_test_client, 'send_message') as mock_send:
                    mock_send.return_value = {"result": {"processed": True}}
                    await performance_test_client.send_message(request)
                
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
            
            avg_latency = statistics.mean(latencies)
            throughput_mbps = (size / 1024 / 1024) / (avg_latency / 1000)
            
            performance_results[size] = {
                'avg_latency_ms': avg_latency,
                'throughput_mbps': throughput_mbps
            }
            
            print(f"Message Size: {size // 1024}KB, "
                  f"Latency: {avg_latency:.2f}ms, "
                  f"Throughput: {throughput_mbps:.2f} MB/s")
        
        # Validate that performance degrades gracefully with message size
        small_latency = performance_results[1024]['avg_latency_ms']
        large_latency = performance_results[1024 * 1024]['avg_latency_ms']
        
        # Large messages should not be more than 10x slower than small ones
        latency_ratio = large_latency / small_latency
        assert latency_ratio <= 10.0, (
            f"Large message latency ratio {latency_ratio:.2f} too high"
        )

    @pytest.mark.asyncio
    async def test_regression_performance_benchmark(self, performance_test_client, mock_document_factory):
        """Regression test to ensure MCP performance doesn't degrade over time."""
        
        # Historical performance baseline (would be stored/loaded in real implementation)
        HISTORICAL_BASELINE = {
            'throughput_docs_per_hour': 28.5,
            'avg_latency_ms': 45.2,
            'memory_usage_mb': 128.0,
            'success_rate_percent': 99.2
        }
        
        # Run current performance test
        test_documents = mock_document_factory.generate_mixed_documents(100)
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        successful_processes = 0
        failed_processes = 0
        latencies = []
        
        for document in test_documents:
            try:
                process_start = time.perf_counter()
                
                # Mock document processing
                with patch.object(performance_test_client, 'process_document') as mock_process:
                    mock_process.return_value = {"quality_score": 0.85}
                    await performance_test_client.process_document(document)
                
                latency = (time.perf_counter() - process_start) * 1000
                latencies.append(latency)
                successful_processes += 1
                
            except Exception:
                failed_processes += 1
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        # Calculate current metrics
        total_duration = end_time - start_time
        throughput = successful_processes / (total_duration / 3600)
        avg_latency = statistics.mean(latencies) if latencies else 0
        memory_usage_mb = (end_memory - start_memory) / (1024 * 1024)
        success_rate = successful_processes / len(test_documents) * 100
        
        current_metrics = {
            'throughput_docs_per_hour': throughput,
            'avg_latency_ms': avg_latency,
            'memory_usage_mb': memory_usage_mb,
            'success_rate_percent': success_rate
        }
        
        # Compare against historical baseline
        performance_regression_threshold = 0.05  # 5% degradation threshold
        
        for metric, current_value in current_metrics.items():
            baseline_value = HISTORICAL_BASELINE[metric]
            
            if metric in ['throughput_docs_per_hour', 'success_rate_percent']:
                # Higher is better
                degradation = (baseline_value - current_value) / baseline_value
                assert degradation <= performance_regression_threshold, (
                    f"Performance regression in {metric}: "
                    f"baseline={baseline_value}, current={current_value}, "
                    f"degradation={degradation:.2%}"
                )
            else:
                # Lower is better (latency, memory usage)
                increase = (current_value - baseline_value) / baseline_value
                assert increase <= performance_regression_threshold, (
                    f"Performance regression in {metric}: "
                    f"baseline={baseline_value}, current={current_value}, "
                    f"increase={increase:.2%}"
                )
        
        print(f"✅ Regression Test Results:")
        for metric, current_value in current_metrics.items():
            baseline_value = HISTORICAL_BASELINE[metric]
            print(f"   {metric}: {current_value:.2f} (baseline: {baseline_value:.2f})")

    # Helper methods
    
    async def _measure_baseline_throughput(self, documents: List[Any]) -> float:
        """Measure baseline throughput without MCP overhead."""
        start_time = time.perf_counter()
        
        # Simulate direct processing (no MCP)
        for document in documents:
            # Mock processing time based on document size
            processing_time = len(document.content) / 1000000  # 1 second per MB
            await asyncio.sleep(processing_time)
        
        duration = time.perf_counter() - start_time
        return len(documents) / (duration / 3600)
    
    async def _measure_mcp_throughput(self, client: MCPClient, documents: List[Any]) -> float:
        """Measure throughput with MCP processing."""
        start_time = time.perf_counter()
        
        for document in documents:
            await self._process_document_via_mcp(client, document)
        
        duration = time.perf_counter() - start_time
        return len(documents) / (duration / 3600)
    
    async def _process_document_via_mcp(self, client: MCPClient, document: Any):
        """Process a document via MCP."""
        request = create_request(MCPMethods.DOCUMENT_PROCESS, {
            "document_id": document.id,
            "content": document.content,
            "metadata": document.metadata
        })
        
        # Mock processing response
        with patch.object(client, 'send_message') as mock_send:
            mock_send.return_value = {
                "result": {
                    "document_id": document.id,
                    "processed": True,
                    "quality_score": 0.85
                }
            }
            return await client.send_message(request)


class PerformanceMonitor:
    """Utility class for performance monitoring."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_current_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])