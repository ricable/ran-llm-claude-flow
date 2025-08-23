# Local LLM Integration Deployment Guide

## Overview

This guide provides comprehensive deployment strategies for integrating Qwen3 models across LM Studio, Ollama, and Apple MLX frameworks on M3 Max hardware.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 Local LLM Orchestrator                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ LM Studio   │  │   Ollama    │  │  MLX Direct │             │
│  │ Connector   │  │ Optimizer   │  │ Accelerator │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│        Request Router │ Circuit Breaker │ Cache Layer           │
├─────────────────────────────────────────────────────────────────┤
│    Performance Monitor │ Error Handling │ Memory Management     │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: Apple M3 Pro or higher
- **RAM**: 32GB unified memory
- **Storage**: 100GB free space for models
- **macOS**: 14.0 or later

### Recommended Configuration (M3 Max)
- **CPU**: Apple M3 Max
- **RAM**: 128GB unified memory
- **Storage**: 1TB SSD with 500GB free
- **Network**: Stable internet for initial model downloads

## Framework-Specific Setup

### 1. LM Studio Setup

#### Installation
```bash
# Download LM Studio from https://lmstudio.ai
# Install and launch the application

# Verify API server is running
curl http://127.0.0.1:1234/v1/models
```

#### Model Installation
1. **Qwen3-1.7B-4bit** (Fast inference)
   - Search: `lmstudio-community/Qwen3-1.7B-MLX-4bit`
   - Size: ~1.2GB
   - Use case: Real-time responses, classification

2. **Qwen3-7B-8bit** (Balanced performance)
   - Search: `lmstudio-community/Qwen3-7B-MLX-8bit`
   - Size: ~7.2GB
   - Use case: General Q&A, summarization

3. **Qwen3-14B-8bit** (High quality)
   - Search: `lmstudio-community/Qwen3-14B-MLX-8bit`
   - Size: ~14.5GB
   - Use case: Complex reasoning, technical writing

#### Configuration
```json
{
  "server": {
    "port": 1234,
    "host": "127.0.0.1",
    "cors": true,
    "max_concurrent_requests": 4
  },
  "model_settings": {
    "context_length": 32768,
    "batch_size": 512,
    "gpu_acceleration": true,
    "unified_memory": true
  }
}
```

### 2. Ollama Setup

#### Installation
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

#### Model Installation
```bash
# Fast model for quick responses
ollama pull qwen2.5:1.5b-q4_0

# Balanced model for general use
ollama pull qwen2.5:7b-q8_0

# Quality model for complex tasks
ollama pull qwen2.5:14b-q8_0

# Create aliases for easier reference
ollama tag qwen2.5:1.5b-q4_0 qwen3-fast
ollama tag qwen2.5:7b-q8_0 qwen3-balanced
ollama tag qwen2.5:14b-q8_0 qwen3-quality
```

#### Ollama Configuration
```json
{
  "origins": ["*"],
  "host": "0.0.0.0:11434",
  "keep_alive": "24h",
  "parallel_requests": 4,
  "max_loaded_models": 3,
  "gpu_memory_fraction": 0.8
}
```

### 3. MLX Setup

#### Installation
```bash
# Install MLX and MLX-LM
pip install mlx
pip install mlx-lm

# Verify MLX installation
python -c "import mlx.core as mx; print(f'MLX device: {mx.default_device()}')"
```

#### Model Conversion (if needed)
```bash
# Convert models to MLX format if not already available
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-1.5B-Instruct \
  --mlx-path ~/.mlx_models/qwen3-1.7b-mlx \
  --quantize

python -m mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-7B-Instruct \
  --mlx-path ~/.mlx_models/qwen3-7b-mlx \
  --quantize
```

## Deployment Strategies

### Strategy 1: Speed-Optimized Setup

**Use Case**: Real-time applications, interactive chatbots
**Primary Framework**: MLX Direct
**Fallback Order**: Ollama → LM Studio

```python
# Configuration
config = {
    "primary_framework": "mlx",
    "preferred_models": ["qwen3-1.7b-4bit"],
    "max_response_time": 2.0,
    "cache_enabled": True,
    "cache_size": 5000
}
```

### Strategy 2: Quality-Optimized Setup

**Use Case**: Technical documentation, complex analysis
**Primary Framework**: LM Studio
**Fallback Order**: MLX → Ollama

```python
# Configuration
config = {
    "primary_framework": "lmstudio",
    "preferred_models": ["qwen3-14b-8bit", "qwen3-30b-16bit"],
    "max_response_time": 30.0,
    "quality_threshold": 0.9
}
```

### Strategy 3: Balanced Setup

**Use Case**: General-purpose applications
**Primary Framework**: Dynamic selection
**Fallback Order**: All frameworks with intelligent routing

```python
# Configuration
config = {
    "framework_selection": "adaptive",
    "model_selection": "dynamic",
    "performance_weight": 0.4,
    "quality_weight": 0.4,
    "availability_weight": 0.2
}
```

## Integration Code Examples

### Basic Integration
```python
import asyncio
from local_llm_orchestrator import LocalLLMOrchestrator, InferenceRequest

async def main():
    # Initialize orchestrator
    orchestrator = LocalLLMOrchestrator(
        lmstudio_url="http://127.0.0.1:1234",
        ollama_url="http://127.0.0.1:11434",
        enable_cache=True
    )
    
    # Initialize all frameworks
    await orchestrator.initialize()
    
    # Create request
    request = InferenceRequest(
        id="example_1",
        prompt="Explain 5G NR technology",
        model_variant="7b",
        priority="normal"
    )
    
    # Generate response
    response = await orchestrator.generate(request)
    
    print(f"Response: {response.text}")
    print(f"Framework: {response.framework_used}")
    print(f"Performance: {response.tokens_per_second:.1f} tok/s")
    
    # Cleanup
    await orchestrator.cleanup()

asyncio.run(main())
```

### Advanced Integration with Error Handling
```python
import asyncio
import logging
from local_llm_orchestrator import LocalLLMOrchestrator, InferenceRequest

async def robust_inference(prompt: str, max_retries: int = 3):
    orchestrator = LocalLLMOrchestrator()
    
    try:
        await orchestrator.initialize()
        
        for attempt in range(max_retries):
            try:
                request = InferenceRequest(
                    prompt=prompt,
                    timeout=30.0,
                    priority="high" if attempt > 0 else "normal"
                )
                
                response = await orchestrator.generate(request)
                
                if response.success:
                    return response
                else:
                    logging.warning(f"Attempt {attempt + 1} failed: {response.error}")
                    
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} error: {e}")
                
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("All attempts failed")
        
    finally:
        await orchestrator.cleanup()
```

### Batch Processing
```python
async def batch_inference(prompts: List[str], batch_size: int = 4):
    orchestrator = LocalLLMOrchestrator()
    await orchestrator.initialize()
    
    results = []
    
    try:
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Create requests
            requests = [
                InferenceRequest(prompt=prompt, id=f"batch_{i}_{j}")
                for j, prompt in enumerate(batch)
            ]
            
            # Process batch concurrently
            tasks = [orchestrator.generate(req) for req in requests]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            results.extend(batch_results)
    
    finally:
        await orchestrator.cleanup()
    
    return results
```

## Performance Tuning

### Memory Optimization

```python
# Memory-constrained deployment
class MemoryOptimizedOrchestrator(LocalLLMOrchestrator):
    def __init__(self):
        super().__init__(
            cache_size=1000,  # Reduced cache
            enable_cache=True
        )
        
        # Monitor memory usage
        self.memory_threshold = 0.85
        
    async def _check_memory_before_request(self):
        memory_usage = psutil.virtual_memory().percent / 100
        if memory_usage > self.memory_threshold:
            # Clear cache or switch to lighter model
            self.cache.clear()
            # Force garbage collection
            import gc
            gc.collect()
```

### Concurrent Request Management

```python
# High-throughput deployment
class HighThroughputOrchestrator(LocalLLMOrchestrator):
    def __init__(self):
        super().__init__()
        self.request_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent
        
    async def generate(self, request):
        async with self.request_semaphore:
            return await super().generate(request)
```

## Monitoring and Debugging

### Health Check Script
```python
#!/usr/bin/env python3
"""Health check script for local LLM deployment"""

import asyncio
import time
from local_llm_orchestrator import LocalLLMOrchestrator, InferenceRequest

async def health_check():
    orchestrator = LocalLLMOrchestrator()
    
    try:
        # Initialize
        init_success = await orchestrator.initialize()
        print(f"Initialization: {'✓' if init_success else '✗'}")
        
        # Test each framework
        frameworks = ["lmstudio", "ollama", "mlx"]
        test_prompt = "Hello, world!"
        
        for framework in frameworks:
            try:
                start_time = time.time()
                
                request = InferenceRequest(
                    prompt=test_prompt,
                    timeout=10.0
                )
                
                response = await orchestrator.generate(request)
                response_time = time.time() - start_time
                
                if response.success and response.framework_used == framework:
                    print(f"{framework}: ✓ ({response_time:.2f}s)")
                else:
                    print(f"{framework}: ✗ (fallback to {response.framework_used})")
                    
            except Exception as e:
                print(f"{framework}: ✗ ({e})")
        
        # Get overall status
        status = await orchestrator.get_status()
        print(f"\nOverall Status:")
        print(f"  Healthy frameworks: {sum(1 for fw in status['frameworks'].values() if fw['healthy'])}/3")
        print(f"  Cache hit rate: {status['cache'].get('hit_rate', 0):.1%}")
        print(f"  Average response time: {status['performance'].get('avg_response_time', 0):.2f}s")
        
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(health_check())
```

### Performance Monitoring
```python
#!/usr/bin/env python3
"""Continuous performance monitoring"""

import asyncio
import time
import json
from datetime import datetime
from local_llm_orchestrator import LocalLLMOrchestrator

async def monitor_performance(duration_minutes: int = 60):
    orchestrator = LocalLLMOrchestrator()
    await orchestrator.initialize()
    
    end_time = time.time() + (duration_minutes * 60)
    metrics_history = []
    
    try:
        while time.time() < end_time:
            status = await orchestrator.get_status()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "frameworks": {
                    name: fw_status["healthy"]
                    for name, fw_status in status["frameworks"].items()
                },
                "performance": status["performance"],
                "memory_usage": orchestrator.get_memory_usage()
            }
            
            metrics_history.append(metrics)
            
            # Log current metrics
            print(f"[{metrics['timestamp']}] "
                  f"Healthy: {sum(metrics['frameworks'].values())}/3, "
                  f"Avg Response: {status['performance'].get('avg_response_time', 0):.2f}s")
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Save metrics to file
        with open(f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(metrics_history, f, indent=2)
            
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(monitor_performance(60))  # Monitor for 1 hour
```

## Troubleshooting

### Common Issues

#### 1. LM Studio Connection Failed
```bash
# Check if LM Studio is running
curl http://127.0.0.1:1234/v1/models

# Check firewall settings
sudo lsof -i :1234

# Restart LM Studio server
# In LM Studio: Server tab → Stop → Start
```

#### 2. Ollama Model Not Found
```bash
# List available models
ollama list

# Pull missing models
ollama pull qwen2.5:7b-q8_0

# Check Ollama service
ollama ps
```

#### 3. MLX Import Error
```bash
# Verify MLX installation
python -c "import mlx.core as mx; print('MLX OK')"

# Reinstall if needed
pip uninstall mlx mlx-lm
pip install mlx mlx-lm --upgrade
```

#### 4. Memory Issues
```python
# Memory diagnostic
import psutil

memory = psutil.virtual_memory()
print(f"Total: {memory.total / (1024**3):.1f}GB")
print(f"Available: {memory.available / (1024**3):.1f}GB")
print(f"Used: {memory.percent:.1f}%")

# If memory usage > 85%, consider:
# 1. Using smaller models
# 2. Reducing cache size
# 3. Limiting concurrent requests
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create orchestrator with debug settings
orchestrator = LocalLLMOrchestrator(
    cache_size=100,  # Smaller cache for debugging
    enable_cache=False  # Disable cache to see all requests
)

# Add request tracking
async def debug_generate(request):
    print(f"[DEBUG] Request: {request.id}")
    print(f"[DEBUG] Prompt length: {len(request.prompt)}")
    
    response = await orchestrator.generate(request)
    
    print(f"[DEBUG] Response: {response.success}")
    print(f"[DEBUG] Framework: {response.framework_used}")
    print(f"[DEBUG] Time: {response.processing_time:.2f}s")
    
    return response
```

## Production Deployment Checklist

- [ ] All frameworks installed and tested individually
- [ ] Models downloaded and verified
- [ ] Health check script passes
- [ ] Performance monitoring configured
- [ ] Error handling and fallback mechanisms tested
- [ ] Memory usage optimized for workload
- [ ] Logging and debugging configured
- [ ] Backup/recovery procedures documented
- [ ] Load testing completed
- [ ] Security considerations reviewed

## Security Considerations

1. **Local-Only Deployment**: All models run locally, no data leaves the machine
2. **API Security**: If exposing APIs, use authentication and rate limiting
3. **Model Integrity**: Verify model checksums after download
4. **Resource Limits**: Set appropriate memory and CPU limits
5. **Access Control**: Restrict file system access for model directories

## Support and Resources

- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **Ollama Documentation**: https://github.com/ollama/ollama
- **LM Studio Support**: https://lmstudio.ai/docs
- **Qwen3 Models**: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

## Future Enhancements

1. **Model Fine-tuning Pipeline**: Integration with existing Qwen3 fine-tuning scripts
2. **Distributed Inference**: Multi-machine deployment capabilities  
3. **GPU Clustering**: Support for multiple GPU devices
4. **Advanced Caching**: Semantic caching based on embedding similarity
5. **Model Serving**: RESTful API server for external applications