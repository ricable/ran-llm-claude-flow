#!/usr/bin/env python3
"""
Performance benchmarks for Python integration modules
Tests throughput, latency, and resource usage of LLM framework components.
"""

import asyncio
import time
import psutil
import statistics
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules to benchmark
import sys
sys.path.append('/Users/cedric/orange/ran-llm-claude-flow')

from docs.python.integration.local_llm_orchestrator import LocalLLMOrchestrator, InferenceRequest
from docs.python.integration.lmstudio_connector import LMStudioConnector, LMStudioConfig
from docs.python.integration.ollama_optimizer import OllamaOptimizer, EmbeddingRequest

class PerformanceBenchmarks:
    """Performance benchmarking suite"""
    
    def __init__(self):
        self.results = {}
    
    async def benchmark_lmstudio_connector(self) -> Dict[str, Any]:
        """Benchmark LM Studio connector performance"""
        print("📊 Benchmarking LM Studio connector...")
        
        config = LMStudioConfig(max_connections=10, batch_size=5)
        connector = LMStudioConnector(config)
        
        # Mock successful responses
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
                "choices": [{"message": {"content": "Benchmark response " * 20}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 40, "total_tokens": 50}
            })
            
            mock_session.get.return_value.__aenter__.return_value = mock_models_response
            mock_session.post.return_value.__aenter__.return_value = mock_chat_response
            MockSession.return_value = mock_session
            
            await connector.initialize()
            
            # Benchmark single request latency
            single_request_times = []
            for _ in range(10):
                start_time = time.time()
                
                messages = [{"role": "user", "content": "Benchmark test message"}]
                result = await connector.generate(messages=messages)
                
                end_time = time.time()
                
                if result["success"]:
                    single_request_times.append(end_time - start_time)
            
            # Benchmark concurrent requests throughput
            concurrent_requests = 20
            start_time = time.time()
            
            tasks = []
            for i in range(concurrent_requests):
                messages = [{"role": "user", "content": f"Concurrent test {i}"}]
                task = asyncio.create_task(connector.generate(messages=messages))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            
            # Calculate metrics
            avg_latency = statistics.mean(single_request_times) if single_request_times else 0
            throughput = len(successful_results) / (end_time - start_time)
            success_rate = len(successful_results) / concurrent_requests
            
            await connector.cleanup()
            
            return {
                "component": "lmstudio_connector",
                "avg_latency_ms": avg_latency * 1000,
                "throughput_rps": throughput,
                "success_rate": success_rate,
                "concurrent_requests": concurrent_requests,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
            }
    
    async def benchmark_ollama_optimizer(self) -> Dict[str, Any]:
        """Benchmark Ollama optimizer performance"""
        print("📊 Benchmarking Ollama optimizer...")
        
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
                "response": "Benchmark response from Ollama " * 15
            })
            
            mock_embeddings_response = AsyncMock()
            mock_embeddings_response.status = 200
            mock_embeddings_response.json = AsyncMock(return_value={
                "embedding": [0.1] * 768  # 768-dim embedding
            })
            
            mock_session.get.return_value.__aenter__.return_value = mock_tags_response
            mock_session.post.return_value.__aenter__.return_value = mock_generate_response
            MockSession.return_value = mock_session
            
            optimizer = OllamaOptimizer()
            
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.available = 32 * 1024**3
                
                await optimizer.initialize()
                
                # Benchmark text generation
                text_gen_times = []
                for _ in range(10):
                    start_time = time.time()
                    
                    result = await optimizer.generate_text(
                        prompt="Benchmark text generation test",
                        use_case="general_qa"
                    )
                    
                    end_time = time.time()
                    
                    if result["success"]:
                        text_gen_times.append(end_time - start_time)
                
                # Benchmark embedding generation with caching
                embedding_times = []
                cache_hit_times = []
                
                # First request (cache miss)
                embed_request = EmbeddingRequest(
                    id="benchmark_embed",
                    texts=["Embedding benchmark text " + str(i) for i in range(5)],
                    model="qwen3-embed"
                )
                
                start_time = time.time()
                response1 = await optimizer.generate_embeddings(embed_request)
                embedding_times.append(time.time() - start_time)
                
                # Second identical request (cache hit)
                start_time = time.time() 
                response2 = await optimizer.generate_embeddings(embed_request)
                cache_hit_times.append(time.time() - start_time)
                
                await optimizer.cleanup()
                
                return {
                    "component": "ollama_optimizer",
                    "text_gen_avg_latency_ms": statistics.mean(text_gen_times) * 1000 if text_gen_times else 0,
                    "embedding_avg_latency_ms": statistics.mean(embedding_times) * 1000 if embedding_times else 0,
                    "cache_hit_latency_ms": statistics.mean(cache_hit_times) * 1000 if cache_hit_times else 0,
                    "cache_speedup_factor": (statistics.mean(embedding_times) / statistics.mean(cache_hit_times)) if cache_hit_times else 1,
                    "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
                }
    
    async def benchmark_orchestrator_coordination(self) -> Dict[str, Any]:
        """Benchmark multi-framework orchestrator coordination"""
        print("📊 Benchmarking orchestrator coordination...")
        
        # Mock all frameworks
        with patch('docs.python.integration.local_llm_orchestrator.LMStudioInterface') as MockLMStudio:
            with patch('docs.python.integration.local_llm_orchestrator.OllamaInterface') as MockOllama:
                with patch('docs.python.integration.local_llm_orchestrator.MLXInterface') as MockMLX:
                    
                    # Configure mock frameworks with different performance characteristics
                    frameworks = {
                        'lmstudio': MockLMStudio.return_value,
                        'ollama': MockOllama.return_value,
                        'mlx': MockMLX.return_value
                    }
                    
                    response_times = {'lmstudio': 0.8, 'ollama': 1.2, 'mlx': 0.5}
                    
                    for name, mock_framework in frameworks.items():
                        mock_framework.initialize = AsyncMock(return_value=True)
                        mock_framework.health_check = AsyncMock(return_value=True)
                        mock_framework.get_available_models = MagicMock(return_value=["qwen3-7b"])
                        
                        async def create_generate_func(framework_name, response_time):
                            async def generate_func(request):
                                await asyncio.sleep(response_time)  # Simulate processing time
                                from docs.python.integration.local_llm_orchestrator import InferenceResponse
                                return InferenceResponse(
                                    request_id=request.id,
                                    text=f"Response from {framework_name}",
                                    model_used="qwen3-7b",
                                    framework_used=framework_name,
                                    processing_time=response_time,
                                    tokens_per_second=20.0,
                                    success=True
                                )
                            return generate_func
                        
                        mock_framework.generate = await create_generate_func(name, response_times[name])
                    
                    orchestrator = LocalLLMOrchestrator()
                    await orchestrator.initialize()
                    
                    # Benchmark framework selection efficiency
                    selection_times = []
                    request_types = [
                        ("critical", "Quick response", "critical"),
                        ("normal", "Standard processing", "normal"), 
                        ("complex", "a" * 1000 + " detailed analysis", "normal"),
                    ]
                    
                    for req_type, prompt, priority in request_types:
                        for _ in range(5):
                            start_time = time.time()
                            
                            request = InferenceRequest(
                                id=f"benchmark_{req_type}",
                                prompt=prompt,
                                priority=priority
                            )
                            
                            response = await orchestrator.generate(request)
                            end_time = time.time()
                            
                            if response.success:
                                selection_times.append(end_time - start_time)
                    
                    # Benchmark concurrent coordination
                    concurrent_requests = 15
                    mixed_requests = []
                    
                    for i in range(concurrent_requests):
                        req_type, prompt, priority = request_types[i % len(request_types)]
                        request = InferenceRequest(
                            id=f"concurrent_bench_{i}",
                            prompt=f"{prompt} {i}",
                            priority=priority
                        )
                        mixed_requests.append(request)
                    
                    start_time = time.time()
                    tasks = [orchestrator.generate(req) for req in mixed_requests]
                    responses = await asyncio.gather(*tasks)
                    end_time = time.time()
                    
                    successful_responses = [r for r in responses if r.success]
                    
                    # Analyze framework distribution
                    framework_usage = {}
                    for response in successful_responses:
                        framework = response.framework_used
                        framework_usage[framework] = framework_usage.get(framework, 0) + 1
                    
                    await orchestrator.cleanup()
                    
                    return {
                        "component": "orchestrator_coordination", 
                        "avg_selection_latency_ms": statistics.mean(selection_times) * 1000 if selection_times else 0,
                        "concurrent_throughput_rps": len(successful_responses) / (end_time - start_time),
                        "success_rate": len(successful_responses) / concurrent_requests,
                        "framework_distribution": framework_usage,
                        "load_balancing_efficiency": 1.0 - (max(framework_usage.values()) - min(framework_usage.values())) / sum(framework_usage.values()) if framework_usage else 0,
                        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
                    }
    
    async def benchmark_memory_usage_patterns(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns under load"""
        print("📊 Benchmarking memory usage patterns...")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples = [initial_memory]
        
        # Simulate sustained load
        with patch('docs.python.integration.local_llm_orchestrator.LMStudioInterface') as MockLMStudio:
            mock_lmstudio = MockLMStudio.return_value
            mock_lmstudio.initialize = AsyncMock(return_value=True)
            mock_lmstudio.health_check = AsyncMock(return_value=True)
            mock_lmstudio.get_available_models = MagicMock(return_value=["qwen3-7b"])
            
            async def memory_efficient_generate(request):
                # Simulate memory allocation and cleanup
                dummy_data = [0] * 1000  # Small allocation
                await asyncio.sleep(0.01)
                del dummy_data
                
                from docs.python.integration.local_llm_orchestrator import InferenceResponse
                return InferenceResponse(
                    request_id=request.id,
                    text="Memory benchmark response",
                    model_used="qwen3-7b",
                    framework_used="lmstudio",
                    processing_time=0.01,
                    tokens_per_second=100.0,
                    success=True
                )
            
            mock_lmstudio.generate = memory_efficient_generate
            
            orchestrator = LocalLLMOrchestrator()
            await orchestrator.initialize()
            
            # Run sustained load test
            for batch in range(10):
                batch_requests = []
                for i in range(5):
                    request = InferenceRequest(
                        id=f"memory_bench_{batch}_{i}",
                        prompt=f"Memory test batch {batch} request {i}"
                    )
                    batch_requests.append(orchestrator.generate(request))
                
                await asyncio.gather(*batch_requests)
                
                # Sample memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            await orchestrator.cleanup()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            return {
                "component": "memory_usage_patterns",
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "peak_memory_mb": max(memory_samples),
                "avg_memory_mb": statistics.mean(memory_samples),
                "memory_growth_mb": final_memory - initial_memory,
                "memory_stability": statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
            }
    
    async def benchmark_error_handling_overhead(self) -> Dict[str, Any]:
        """Benchmark error handling and recovery overhead"""
        print("📊 Benchmarking error handling overhead...")
        
        with patch('docs.python.integration.local_llm_orchestrator.LMStudioInterface') as MockLMStudio:
            mock_lmstudio = MockLMStudio.return_value
            mock_lmstudio.initialize = AsyncMock(return_value=True)
            mock_lmstudio.health_check = AsyncMock(return_value=True)
            mock_lmstudio.get_available_models = MagicMock(return_value=["qwen3-7b"])
            
            # Test successful requests
            successful_times = []
            
            async def successful_generate(request):
                from docs.python.integration.local_llm_orchestrator import InferenceResponse
                return InferenceResponse(
                    request_id=request.id,
                    text="Successful response",
                    model_used="qwen3-7b",
                    framework_used="lmstudio", 
                    processing_time=0.1,
                    tokens_per_second=50.0,
                    success=True
                )
            
            mock_lmstudio.generate = successful_generate
            
            orchestrator = LocalLLMOrchestrator()
            await orchestrator.initialize()
            
            # Benchmark successful requests
            for _ in range(10):
                start_time = time.time()
                
                request = InferenceRequest(
                    id="success_bench",
                    prompt="Success benchmark test"
                )
                
                response = await orchestrator.generate(request)
                end_time = time.time()
                
                if response.success:
                    successful_times.append(end_time - start_time)
            
            # Test error handling overhead
            error_handling_times = []
            
            async def failing_generate(request):
                await asyncio.sleep(0.05)  # Simulate some processing before failure
                from docs.python.integration.local_llm_orchestrator import InferenceResponse
                return InferenceResponse(
                    request_id=request.id,
                    text="",
                    model_used="qwen3-7b", 
                    framework_used="lmstudio",
                    processing_time=0.05,
                    tokens_per_second=0.0,
                    success=False,
                    error="Simulated failure for benchmark"
                )
            
            mock_lmstudio.generate = failing_generate
            
            # Benchmark error handling
            for _ in range(10):
                start_time = time.time()
                
                request = InferenceRequest(
                    id="error_bench",
                    prompt="Error benchmark test"
                )
                
                response = await orchestrator.generate(request)
                end_time = time.time()
                
                error_handling_times.append(end_time - start_time)
            
            await orchestrator.cleanup()
            
            avg_success_time = statistics.mean(successful_times) if successful_times else 0
            avg_error_time = statistics.mean(error_handling_times) if error_handling_times else 0
            
            return {
                "component": "error_handling_overhead",
                "avg_success_latency_ms": avg_success_time * 1000,
                "avg_error_handling_latency_ms": avg_error_time * 1000,
                "error_handling_overhead_ms": (avg_error_time - avg_success_time) * 1000,
                "error_handling_overhead_percent": ((avg_error_time - avg_success_time) / avg_success_time * 100) if avg_success_time > 0 else 0
            }

async def run_performance_tests() -> Dict[str, Any]:
    """Run all performance benchmarks"""
    print("🚀 Starting performance benchmarks...")
    
    benchmarks = PerformanceBenchmarks()
    results = {
        "timestamp": time.time(),
        "benchmarks": {}
    }
    
    # Run all benchmarks
    benchmark_functions = [
        benchmarks.benchmark_lmstudio_connector,
        benchmarks.benchmark_ollama_optimizer,
        benchmarks.benchmark_orchestrator_coordination,
        benchmarks.benchmark_memory_usage_patterns,
        benchmarks.benchmark_error_handling_overhead
    ]
    
    for benchmark_func in benchmark_functions:
        try:
            result = await benchmark_func()
            component_name = result["component"]
            results["benchmarks"][component_name] = result
            print(f"✅ Completed benchmark: {component_name}")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {benchmark_func.__name__}: {e}")
            results["benchmarks"][benchmark_func.__name__] = {
                "error": str(e),
                "success": False
            }
    
    # Calculate summary metrics
    successful_benchmarks = [b for b in results["benchmarks"].values() if not b.get("error")]
    
    results["summary"] = {
        "total_benchmarks": len(benchmark_functions),
        "successful_benchmarks": len(successful_benchmarks),
        "success_rate": len(successful_benchmarks) / len(benchmark_functions),
        "avg_memory_usage_mb": statistics.mean([b.get("memory_usage_mb", 0) for b in successful_benchmarks if b.get("memory_usage_mb")]) if successful_benchmarks else 0
    }
    
    print(f"✅ Performance benchmarks completed: {len(successful_benchmarks)}/{len(benchmark_functions)} successful")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(run_performance_tests())
    print(json.dumps(results, indent=2, default=str))