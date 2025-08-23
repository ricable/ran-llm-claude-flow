# RAN-LLM Hybrid Pipeline - Developer Quickstart Guide

## Overview

This quickstart guide helps you get the hybrid Rust-Python RAN document processing pipeline running on your MacBook Pro M3 Max system in under 30 minutes.

## Prerequisites

### Hardware Requirements
- **MacBook Pro M3 Max** with 128GB unified memory (recommended)
- **Minimum**: MacBook Pro M3 with 64GB unified memory
- **Storage**: 100GB+ free space for models and processing data

### Software Requirements
- **macOS Sonoma 14.0+** or **macOS Sequoia 15.0+**
- **Rust 1.75+** with Cargo
- **Python 3.11+** with uv package manager
- **LM Studio** or **Ollama** for local model serving
- **Git** for repository management

## Quick Installation (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/ricable/ran-llm-claude-flow.git
cd ran-llm-claude-flow
```

### 2. Install Rust Dependencies
```bash
# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install required components
rustup component add clippy rustfmt

# Build Rust pipeline with M3 Max optimizations
cd src/rust-pipeline
cargo build --release --features m3-max
```

### 3. Setup Python Environment
```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate Python environment
cd ../python-pipeline
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 4. Install Local Models
```bash
# Option 1: Using Ollama (Recommended)
brew install ollama
ollama serve &

# Pull Qwen3 models
ollama pull qwen3:1.7b
ollama pull qwen3:7b
ollama pull qwen3:30b

# Option 2: Using LM Studio
# Download and install LM Studio from https://lmstudio.ai/
# Load Qwen3 models through the UI
```

### 5. Configure Pipeline
```bash
# Copy optimized configuration
cp src/python-pipeline/config/examples/m3_max_optimized.yaml config/pipeline.yaml

# Update paths in configuration file
sed -i '' 's|/path/to/input|./data/input|g' config/pipeline.yaml
sed -i '' 's|/path/to/output|./data/output|g' config/pipeline.yaml
```

## First Run (10 minutes)

### 1. Prepare Test Data
```bash
# Create directories
mkdir -p data/{input,output}

# Download sample RAN documents (optional)
curl -O https://example.com/sample-ran-docs.zip
unzip sample-ran-docs.zip -d data/input/

# Or create a simple test document
echo "# RAN Feature: Carrier Aggregation

Carrier Aggregation is a key feature that allows combining multiple carriers to increase bandwidth.

## Parameters
- CA_ENABLE: Boolean parameter to enable/disable CA
- MAX_CARRIERS: Maximum number of carriers (1-32)

## Procedures
1. Configure primary carrier
2. Add secondary carriers
3. Activate aggregation
4. Monitor performance" > data/input/test-document.md
```

### 2. Start the Pipeline Server
```bash
# Terminal 1: Start Rust core server
cd src/rust-pipeline
cargo run --release --bin pipeline-server

# Should see output:
# [INFO] Initializing Rust-Python hybrid pipeline for M3 Max
# [INFO] MCP server started on port 8700
# [INFO] Pipeline ready for requests
```

### 3. Process Your First Document
```bash
# Terminal 2: Send processing request
curl -X POST http://localhost:8700/api/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "name": "First RAN Processing",
    "input_path": "./data/input",
    "output_path": "./data/output",
    "quality_threshold": 0.742,
    "model_strategy": "adaptive",
    "batch_size": 10
  }'

# Response:
# {
#   "pipeline_id": "pl_7d4c9b2a1e8f",
#   "status": "created",
#   "estimated_duration": 600,
#   "estimated_throughput": 15
# }
```

### 4. Monitor Progress
```bash
# Check pipeline status
curl http://localhost:8700/api/v1/pipeline/pl_7d4c9b2a1e8f

# View real-time metrics
curl http://localhost:8700/api/v1/metrics

# Access web dashboard (if enabled)
open http://localhost:8700/dashboard
```

## API Usage Examples (10 minutes)

### Python Client Example
```python
import asyncio
import aiohttp
import json

async def process_documents():
    # Create pipeline
    async with aiohttp.ClientSession() as session:
        pipeline_data = {
            "name": "Batch RAN Processing",
            "input_path": "./data/input",
            "output_path": "./data/output",
            "quality_threshold": 0.742,
            "model_strategy": "qwen3_7b",
            "stages": ["conversion", "preprocessing", "langextract", "conversation"]
        }
        
        async with session.post(
            "http://localhost:8700/api/v1/pipeline",
            json=pipeline_data
        ) as resp:
            result = await resp.json()
            pipeline_id = result["pipeline_id"]
            print(f"Created pipeline: {pipeline_id}")
        
        # Monitor until completion
        while True:
            async with session.get(
                f"http://localhost:8700/api/v1/pipeline/{pipeline_id}"
            ) as resp:
                status = await resp.json()
                
                print(f"Progress: {status['progress']:.1f}% - {status['status']}")
                print(f"Documents: {status['documents_processed']}/{status['documents_total']}")
                print(f"Quality: {status['quality_score_avg']:.3f}")
                print(f"Throughput: {status['throughput_current']:.1f} docs/hour")
                
                if status["status"] in ["completed", "failed"]:
                    break
                
                await asyncio.sleep(5)
        
        print(f"Pipeline {status['status']}!")

# Run the example
asyncio.run(process_documents())
```

### Rust Client Example
```rust
use reqwest;
use serde_json::json;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    
    // Create pipeline
    let pipeline_request = json!({
        "name": "Rust API Test",
        "input_path": "./data/input",
        "output_path": "./data/output",
        "quality_threshold": 0.742,
        "model_strategy": "adaptive"
    });
    
    let response = client
        .post("http://localhost:8700/api/v1/pipeline")
        .json(&pipeline_request)
        .send()
        .await?;
    
    let pipeline_result: serde_json::Value = response.json().await?;
    let pipeline_id = pipeline_result["pipeline_id"].as_str().unwrap();
    
    println!("Created pipeline: {}", pipeline_id);
    
    // Monitor progress
    loop {
        let status_response = client
            .get(&format!("http://localhost:8700/api/v1/pipeline/{}", pipeline_id))
            .send()
            .await?;
        
        let status: serde_json::Value = status_response.json().await?;
        
        println!(
            "Progress: {:.1}% - {}",
            status["progress"].as_f64().unwrap_or(0.0),
            status["status"].as_str().unwrap_or("unknown")
        );
        
        if status["status"] == "completed" || status["status"] == "failed" {
            break;
        }
        
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    }
    
    Ok(())
}
```

### JavaScript/Node.js Client Example
```javascript
const axios = require('axios');

async function processDocuments() {
  try {
    // Create pipeline
    const pipelineResponse = await axios.post('http://localhost:8700/api/v1/pipeline', {
      name: 'JavaScript Client Test',
      input_path: './data/input',
      output_path: './data/output',
      quality_threshold: 0.742,
      model_strategy: 'adaptive',
      batch_size: 50
    });
    
    const pipelineId = pipelineResponse.data.pipeline_id;
    console.log(`Created pipeline: ${pipelineId}`);
    
    // Monitor progress
    while (true) {
      const statusResponse = await axios.get(`http://localhost:8700/api/v1/pipeline/${pipelineId}`);
      const status = statusResponse.data;
      
      console.log(`Progress: ${status.progress.toFixed(1)}% - ${status.status}`);
      console.log(`Documents: ${status.documents_processed}/${status.documents_total}`);
      console.log(`Quality: ${status.quality_score_avg.toFixed(3)}`);
      console.log(`Memory: ${status.memory_usage_gb.toFixed(1)}GB`);
      console.log('---');
      
      if (status.status === 'completed' || status.status === 'failed') {
        break;
      }
      
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    console.log('Pipeline completed!');
    
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

processDocuments();
```

## Advanced Configuration (5 minutes)

### Custom Model Configuration
```yaml
# config/custom-models.yaml
models:
  local_models:
    qwen3_1_7b:
      path: "qwen3:1.7b"
      api_base: "http://localhost:11434"
      context_length: 32768
      memory_gb: 8
      use_cases: ["simple_extraction", "embeddings"]
    qwen3_7b_custom:
      path: "custom-qwen3-7b-finetuned"
      api_base: "http://localhost:1234"
      context_length: 32768
      memory_gb: 28
      use_cases: ["ran_specific_processing"]
```

### Performance Tuning
```yaml
# config/performance.yaml
hardware:
  target_platform: "m3_max"
  enable_apple_silicon_acceleration: true
  enable_metal_performance_shaders: true
  enable_neural_engine: true
  
memory:
  pools:
    models: 
      size_gb: 50.0
      strategy: "adaptive"
    processing:
      size_gb: 60.0
      strategy: "eager"
    cache:
      size_gb: 18.0
      ttl_seconds: 3600

pipeline:
  max_parallel_stages: 8
  batch_size: 200
  enable_circuit_breaker: true
  circuit_breaker_threshold: 3
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```bash
# Check Ollama service
ollama list
ollama ps

# Test model loading
ollama run qwen3:7b "Hello, how are you?"

# Check memory usage
htop
# Or use Activity Monitor
```

#### 2. Memory Issues
```bash
# Check available memory
system_profiler SPHardwareDataType | grep Memory

# Monitor memory usage during processing
while true; do
  memory_pressure
  sleep 5
done
```

#### 3. Performance Issues
```bash
# Check CPU and GPU utilization
sudo powermetrics --samplers cpu_power,gpu_power -n 1

# Profile the Rust application
cd src/rust-pipeline
cargo install flamegraph
sudo cargo flamegraph --bin pipeline-server
```

### Debug Mode
```bash
# Enable verbose logging
export RUST_LOG=debug
export PYTHONDONTWRITEBYTECODE=1

# Start with debug configuration
cargo run --bin pipeline-server -- --config config/debug.yaml --verbose
```

### Health Checks
```bash
# Verify system health
curl http://localhost:8700/api/v1/health

# Check model availability
curl http://localhost:8700/api/v1/models

# View detailed metrics
curl http://localhost:8700/api/v1/metrics?timeframe=15m
```

## Performance Benchmarks

Expected performance on M3 Max with 128GB:

| Document Type | Processing Speed | Quality Score | Memory Usage |
|---------------|------------------|---------------|--------------|
| PDF (Large)   | 12-15 sec       | 0.745-0.760   | 45-60GB      |
| HTML (Complex)| 8-12 sec        | 0.740-0.755   | 35-50GB      |
| Markdown      | 6-10 sec        | 0.750-0.770   | 30-45GB      |
| CSV (Structured)| 4-8 sec       | 0.760-0.780   | 25-40GB      |

**Target Throughput**: 20-30 documents/hour
**Quality Consistency**: ±0.05 variance from target score

## Next Steps

### 1. Production Deployment
- Review [Production Deployment Guide](./production-deployment.md)
- Configure monitoring and alerting
- Set up automated backups
- Implement security measures

### 2. Custom Models
- Fine-tune Qwen3 models for your specific RAN documents
- Implement custom extraction categories
- Optimize quality scoring for your use case

### 3. Integration
- Implement custom pipeline stages
- Add external data sources
- Build custom monitoring dashboards
- Integrate with existing RAN management systems

### 4. Scaling
- Implement distributed processing
- Add additional model servers
- Optimize for different hardware configurations
- Implement caching strategies

## Support

- **Documentation**: [Full API Documentation](./openapi.yaml)
- **Issues**: [GitHub Issues](https://github.com/ricable/ran-llm-claude-flow/issues)
- **Performance**: [Benchmarking Guide](./benchmarks.md)
- **Examples**: [Code Examples Repository](../examples/)

---

**You're now ready to process RAN documents with the hybrid pipeline!**

Start with small batches, monitor performance metrics, and gradually scale up as you optimize the configuration for your specific use case.