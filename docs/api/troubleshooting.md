# RAN-LLM Hybrid Pipeline Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps diagnose and resolve common issues in the hybrid Rust-Python RAN document processing pipeline optimized for M3 Max hardware.

## System Requirements Validation

### Hardware Verification
```bash
# Check system information
system_profiler SPHardwareDataType

# Verify M3 Max chip and memory
sysctl hw.memsize hw.ncpu machdep.cpu.brand_string

# Expected output for M3 Max:
# hw.memsize: 137438953472 (128GB)
# hw.ncpu: 12 (8 Performance + 4 Efficiency cores)
# machdep.cpu.brand_string: Apple M3 Max
```

### Software Dependencies
```bash
# Verify Rust installation
rustc --version
cargo --version

# Should be Rust 1.75.0 or later

# Check Python and uv
python3 --version
uv --version

# Verify Ollama/LM Studio
ollama version
# or check LM Studio installation
ls -la /Applications/LM\ Studio.app
```

## Common Issues and Solutions

### 1. Pipeline Startup Issues

#### Issue: "Failed to bind to port 8700"
**Symptoms:**
- Server fails to start
- Port already in use error

**Diagnosis:**
```bash
# Check if port is in use
lsof -i :8700
netstat -an | grep 8700

# Check for other pipeline instances
ps aux | grep -i pipeline
```

**Solution:**
```bash
# Kill existing processes
sudo kill -9 $(lsof -t -i:8700)

# Or use different port
cargo run --bin pipeline-server -- --port 8701

# Update configuration
export PIPELINE_PORT=8701
```

#### Issue: "Model server not responding"
**Symptoms:**
- Pipeline starts but model requests fail
- Connection refused errors

**Diagnosis:**
```bash
# Test Ollama connection
ollama list
curl http://localhost:11434/api/version

# Test LM Studio connection
curl http://localhost:1234/v1/models
```

**Solution:**
```bash
# Restart Ollama
brew services restart ollama
# Or manually start
ollama serve

# For LM Studio, restart the application
# Check loaded models in LM Studio UI
```

### 2. Memory Issues

#### Issue: "Out of Memory (OOM) during processing"
**Symptoms:**
- Pipeline crashes during large document processing
- System becomes unresponsive
- Memory pressure warnings

**Diagnosis:**
```bash
# Check current memory usage
memory_pressure

# Monitor memory in real-time
sudo memory_pressure -s

# Check pipeline memory usage
ps aux | grep -E "(pipeline|python)" | awk '{sum+=$6} END {print "Total Memory (KB):", sum}'
```

**Solution:**
```bash
# Reduce batch size in configuration
# Edit config/pipeline.yaml
pipeline:
  batch_size: 25  # Reduce from 100
  max_parallel_stages: 4  # Reduce parallelism

# Reduce model memory allocation
memory:
  pools:
    models:
      size_gb: 30.0  # Reduce from 50.0
    processing:
      size_gb: 40.0  # Reduce from 60.0
```

#### Issue: "Model loading fails with insufficient memory"
**Symptoms:**
- Qwen3-30B model fails to load
- Memory allocation errors

**Solution:**
```bash
# Use smaller model variant
curl -X POST http://localhost:8700/api/v1/models/qwen3_7b/load

# Or configure adaptive model selection
{
  "model_strategy": "adaptive",
  "fallback_model": "qwen3_1_7b"
}

# Free up memory by unloading unused models
curl -X POST http://localhost:8700/api/v1/models/qwen3_30b/unload
```

### 3. Performance Issues

#### Issue: "Processing speed slower than expected"
**Expected:** 20-30 documents/hour
**Actual:** <10 documents/hour

**Diagnosis:**
```bash
# Check current throughput
curl http://localhost:8700/api/v1/metrics | jq '.pipeline.documents_per_hour'

# Monitor CPU/GPU utilization
sudo powermetrics --samplers cpu_power,gpu_power -n 1

# Check for thermal throttling
pmset -g thermlog
```

**Solution:**
```bash
# Enable all M3 Max optimizations
hardware:
  enable_apple_silicon_acceleration: true
  enable_metal_performance_shaders: true
  enable_neural_engine: true
  enable_amx_coprocessor: true

# Increase parallelism if memory allows
pipeline:
  max_parallel_stages: 8
  batch_size: 150

# Use faster model for simple tasks
model_strategy:
  qwen3_1_7b:
    use_cases: ["simple_extraction", "embeddings", "preprocessing"]
```

#### Issue: "High CPU usage but low throughput"
**Diagnosis:**
```bash
# Profile the application
cd src/rust-pipeline
cargo install flamegraph
sudo cargo flamegraph --bin pipeline-server

# Check I/O bottlenecks
iostat 1

# Monitor context switches
sar -w 1
```

**Solution:**
```bash
# Optimize I/O operations
stages:
  - stage_id: "stage_1_raw_input"
    max_workers: 2  # Reduce I/O contention
    batch_size: 25  # Smaller I/O batches

# Enable caching
cache:
  size_gb: 30.0
  intelligent_warming: true
  ttl_seconds: 7200
```

### 4. Quality Score Issues

#### Issue: "Quality scores below target threshold"
**Target:** >0.742
**Actual:** <0.700

**Diagnosis:**
```bash
# Check quality distribution
curl http://localhost:8700/api/v1/metrics | jq '.pipeline.quality_score_avg'

# Examine failed documents
curl http://localhost:8700/api/v1/pipeline/pl_abc123 | jq '.stages[] | select(.error_count > 0)'
```

**Solution:**
```bash
# Use higher-quality model
{
  "model_strategy": "qwen3_7b",  # Instead of qwen3_1_7b
  "quality_threshold": 0.720     # Lower threshold temporarily
}

# Improve preprocessing
stages:
  - stage_id: "stage_3_preprocessing"
    processors:
      - type: "quality_assessor"
        config:
          min_quality_score: 0.5  # Pre-filter low-quality content
          assessment_criteria: ["length", "structure", "technical_content"]

# Enable quality feedback loop
quality:
  enable_quality_feedback: true
  quality_improvement: true
```

### 5. Model Inference Issues

#### Issue: "Model inference timeouts"
**Symptoms:**
- Requests hang indefinitely
- Timeout errors after 30+ seconds

**Diagnosis:**
```bash
# Check model status
curl http://localhost:8700/api/v1/models

# Test direct model access
ollama run qwen3:7b "Test prompt"
curl http://localhost:11434/api/generate -d '{"model": "qwen3:7b", "prompt": "test"}'
```

**Solution:**
```bash
# Increase timeout values
models:
  model_timeout: 600  # 10 minutes
  context_length: 16384  # Reduce if using very long prompts

# Enable circuit breaker
optimization:
  circuit_breaker:
    failure_threshold: 3
    timeout_seconds: 120
    success_threshold: 2

# Use async processing
stages:
  - stage_id: "stage_4_langextract"
    execution_mode: "async"
    timeout_seconds: 1800
```

### 6. IPC Communication Issues

#### Issue: "Rust-Python communication failures"
**Symptoms:**
- Messages not being received
- Serialization/deserialization errors

**Diagnosis:**
```bash
# Check MCP protocol logs
tail -f pipeline.log | grep -i "mcp\|ipc"

# Test WebSocket connection
wscat -c ws://localhost:8700/mcp

# Monitor message queue
curl http://localhost:8700/api/v1/system/ipc-status
```

**Solution:**
```bash
# Restart both components
# Terminal 1
cargo run --bin pipeline-server

# Terminal 2 (in python-pipeline directory)
python -m src.main

# Enable message validation
mcp:
  enable_message_validation: true
  max_message_size: 10485760  # 10MB
  compression: true

# Check firewall settings
sudo pfctl -s nat
```

## Advanced Diagnostics

### System Health Check Script
```bash
#!/bin/bash
# health-check.sh

echo "=== RAN-LLM Pipeline Health Check ==="

# System resources
echo "1. System Resources:"
echo "Memory: $(system_profiler SPHardwareDataType | grep Memory | awk '{print $2}')"
echo "Available: $(vm_stat | grep free | awk '{print $3 * 4 / 1024}')MB"

# Pipeline status
echo -e "\n2. Pipeline Status:"
PIPELINE_STATUS=$(curl -s http://localhost:8700/api/v1/health || echo "UNREACHABLE")
echo "Health: $PIPELINE_STATUS"

# Model status
echo -e "\n3. Model Status:"
ollama ps || echo "Ollama not running"

# Disk space
echo -e "\n4. Disk Space:"
df -h | grep -E "(/$|/tmp)"

# Network
echo -e "\n5. Network:"
netstat -an | grep :8700 | head -5

echo "=== End Health Check ==="
```

### Performance Monitoring Script
```bash
#!/bin/bash
# performance-monitor.sh

echo "Starting performance monitoring..."

while true; do
  TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
  
  # Get metrics
  METRICS=$(curl -s http://localhost:8700/api/v1/metrics)
  
  # Extract key values
  THROUGHPUT=$(echo $METRICS | jq -r '.pipeline.documents_per_hour // "N/A"')
  QUALITY=$(echo $METRICS | jq -r '.pipeline.quality_score_avg // "N/A"')
  MEMORY=$(echo $METRICS | jq -r '.system.memory_used_gb // "N/A"')
  CPU=$(echo $METRICS | jq -r '.system.cpu_utilization // "N/A"')
  
  echo "$TIMESTAMP | Throughput: ${THROUGHPUT}/hr | Quality: $QUALITY | Memory: ${MEMORY}GB | CPU: ${CPU}%"
  
  sleep 60
done
```

## Environment-Specific Issues

### Development Environment
```bash
# Enable debug mode
export RUST_LOG=debug
export PIPELINE_ENV=development

# Use smaller models for faster iteration
{
  "model_strategy": "qwen3_1_7b",
  "batch_size": 10,
  "enable_quality_gates": false
}
```

### Production Environment
```bash
# Optimize for production
export RUST_LOG=info
export PIPELINE_ENV=production

# Use production configuration
{
  "model_strategy": "adaptive",
  "enable_monitoring": true,
  "enable_quality_gates": true,
  "retry_attempts": 3
}
```

## Error Code Reference

### HTTP Status Codes
- **400**: Bad Request - Invalid parameters
- **404**: Not Found - Pipeline/document not found
- **429**: Too Many Requests - Rate limited
- **500**: Internal Server Error - Pipeline failure
- **503**: Service Unavailable - Models not loaded

### Custom Error Codes
- **PE001**: Pipeline configuration error
- **PE002**: Model loading failure
- **PE003**: Memory allocation error
- **PE004**: Quality threshold not met
- **PE005**: Processing timeout
- **PE006**: IPC communication failure

### Example Error Response
```json
{
  "error": "PE003",
  "message": "Insufficient memory for model loading",
  "details": {
    "model_id": "qwen3_30b",
    "required_gb": 40.0,
    "available_gb": 12.5,
    "suggestions": [
      "Unload other models",
      "Use qwen3_7b instead",
      "Reduce batch size"
    ]
  },
  "timestamp": "2024-08-23T10:30:00Z",
  "trace_id": "tr_abc123def456"
}
```

## Support and Escalation

### Self-Service Resources
1. Check [GitHub Issues](https://github.com/ricable/ran-llm-claude-flow/issues)
2. Review [API Documentation](./openapi.yaml)
3. Run health check script
4. Check system logs

### Information to Gather
When reporting issues, include:
- System specifications (chip, memory, OS version)
- Pipeline configuration file
- Error messages and logs
- Performance metrics
- Steps to reproduce

### Log Locations
```bash
# Pipeline logs
tail -f pipeline.log

# System logs
log show --predicate 'process == "pipeline-server"' --last 1h

# Ollama logs
tail -f ~/.ollama/logs/server.log

# Python worker logs
tail -f python-pipeline.log
```

## Prevention and Best Practices

### Regular Maintenance
```bash
# Weekly health check
./scripts/health-check.sh

# Monitor disk space
df -h | awk '$5 > 80 {print $0}'

# Update models
ollama pull qwen3:7b
ollama pull qwen3:30b

# Clean up old logs
find . -name "*.log" -mtime +7 -delete
```

### Performance Optimization
```bash
# Regular performance testing
./scripts/benchmark.sh

# Profile memory usage
./scripts/memory-profile.sh

# Test different configurations
./scripts/config-test.sh
```

By following this troubleshooting guide, you should be able to diagnose and resolve most common issues with the RAN-LLM hybrid pipeline system.