# External System Integration Points

## Overview

The Python pipeline integrates with multiple external systems to create a comprehensive telecommunications document processing and fine-tuning workflow. These integrations span from local ML serving to cloud-based model repositories.

## Primary Integration Systems

### 1. Ollama Integration

#### Purpose
Local LLM serving for document feature extraction and classification

#### Integration Pattern
```python
class OllamaConnection:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.mount("http://", HTTPAdapter(max_retries=3))
    
    async def extract_features(self, content: str, model: str = "qwen3:1.7b") -> Dict:
        prompt = self.build_extraction_prompt(content)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9}
        }
        response = await self.session.post(f"{self.base_url}/api/generate", json=payload)
        return response.json()
```

#### Key Features
- **Dynamic Model Selection**: Automatic model routing based on document complexity
- **Connection Management**: Retry logic with exponential backoff
- **Health Monitoring**: Continuous connection health checks
- **Performance Optimization**: Request batching and connection pooling

#### Models Supported
- `qwen3:1.7b` - Primary extraction model
- `gemma3:4b` - Alternative model for complex documents
- `llama3.2:1b` - Fallback for resource-constrained scenarios

#### Usage Locations
- `ollama_langextract.py` - Core extraction engine
- `hybrid_processor.py` - Multi-model processing
- Various CLI tools for testing and validation

### 2. MLX Framework Integration

#### Purpose
Apple Silicon optimized machine learning framework for Qwen3 fine-tuning

#### Integration Architecture
```python
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, train
from mlx_lm.utils import get_model_path, load_model, load_tokenizer
from mlx_lm.tuner.lora import LoRALinear

class MLXFineTuner:
    def __init__(self):
        # M3 Max specific optimizations
        mx.set_default_device(mx.gpu)
        self.model_config = {
            "lora_rank": 64,
            "batch_size": 32,
            "gradient_accumulation": 4,
            "mixed_precision": True
        }
    
    def optimize_for_hardware(self):
        """Dynamic optimization for M3 Max 128GB"""
        available_memory = psutil.virtual_memory().available
        if available_memory > 100 * 1024**3:
            self.model_config["batch_size"] = 64
            self.model_config["lora_rank"] = 128
```

#### Key Capabilities
- **LoRA Fine-tuning**: Memory-efficient parameter adaptation
- **Mixed Precision Training**: FP16 optimization for speed
- **Dynamic Memory Management**: Automatic resource allocation
- **Model Sharding**: Support for large model distribution

#### Performance Optimizations
- **M3 Max Specific**: 128GB RAM utilization strategies
- **Batch Size Scaling**: Dynamic adjustment based on available memory
- **Gradient Accumulation**: Memory efficiency optimization
- **Parallel Data Loading**: Multi-threaded data preparation

### 3. LM Studio Integration

#### Purpose
Alternative local model serving with enhanced UI and management features

#### Integration Pattern
```python
class LMStudioConnection:
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="lm-studio")
    
    async def process_with_model(self, content: str, model: str) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=4000
        )
        return response.choices[0].message.content
```

#### Features
- **OpenAI-Compatible API**: Standard integration pattern
- **Model Management**: Easy model switching and configuration
- **Performance Monitoring**: Built-in metrics and monitoring
- **Resource Management**: Automatic GPU/CPU allocation

### 4. Docling Integration

#### Purpose
Advanced PDF processing with layout understanding and table extraction

#### Integration Implementation
```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

class DoclingProcessor:
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: self.pipeline_options}
        )
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        result = self.converter.convert(pdf_path)
        return {
            "markdown": result.document.export_to_markdown(),
            "tables": self.extract_tables(result),
            "metadata": result.document.meta
        }
```

#### Advanced Features
- **OCR Capabilities**: Text extraction from images and scanned PDFs
- **Table Structure Recognition**: Intelligent table parsing
- **Layout Understanding**: Semantic document structure analysis
- **Multi-format Export**: Markdown, JSON, HTML output formats

### 5. Rust Pipeline Coordination

#### Purpose
High-performance document processing coordination with Python workflows

#### Integration Strategy
```python
# Python delegates to Rust for performance-critical operations
def coordinate_with_rust(self, config_path: Path) -> ProcessingResult:
    """Coordinate with Rust pipeline for high-performance processing"""
    rust_command = [
        "cargo", "run", "--release", 
        "--bin", "ericsson-dataset-pipeline",
        "--", str(config_path)
    ]
    
    result = subprocess.run(
        rust_command, 
        capture_output=True, 
        text=True,
        timeout=3600  # 1 hour timeout
    )
    
    if result.returncode == 0:
        return ProcessingResult.from_json(result.stdout)
    else:
        raise RustProcessingError(result.stderr)
```

#### Coordination Patterns
- **Configuration Sharing**: YAML configs used by both Python and Rust
- **Data Exchange**: JSON-based structured data exchange
- **Process Orchestration**: Python orchestrates Rust execution
- **Error Handling**: Cross-language error propagation

### 6. External APIs and Services

#### HuggingFace Integration
```python
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

class HuggingFaceIntegration:
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "huggingface"
    
    def download_model(self, model_id: str) -> Path:
        """Download model for local MLX conversion"""
        return snapshot_download(
            repo_id=model_id,
            cache_dir=self.cache_dir,
            local_files_only=False
        )
```

#### 3GPP Standards Download
```python
class ThreeGPPDownloader:
    def __init__(self):
        self.base_url = "https://www.3gpp.org/ftp/Specs/archive/"
        self.session = requests.Session()
    
    async def download_specifications(self, spec_numbers: List[str]) -> List[Path]:
        """Download 3GPP specifications for processing"""
        tasks = [self.download_single_spec(spec) for spec in spec_numbers]
        return await asyncio.gather(*tasks)
```

## Integration Challenges and Solutions

### 1. Connection Reliability

#### Challenge
External services (Ollama, LM Studio) may be unavailable or unstable

#### Solution: Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def call_with_breaker(self, func: Callable, *args, **kwargs):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerOpenError()
            self.state = CircuitBreakerState.HALF_OPEN
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

### 2. Version Compatibility

#### Challenge
Different external systems may have incompatible versions or APIs

#### Solution: Adapter Pattern
```python
class ModelServiceAdapter:
    def __init__(self):
        self.adapters = {
            'ollama': OllamaAdapter(),
            'lm_studio': LMStudioAdapter(),
            'openai': OpenAIAdapter()
        }
    
    async def generate_completion(self, service: str, prompt: str) -> str:
        adapter = self.adapters.get(service)
        if not adapter:
            raise UnsupportedServiceError(f"No adapter for {service}")
        
        return await adapter.generate_completion(prompt)
```

### 3. Resource Contention

#### Challenge
Multiple integrations competing for system resources

#### Solution: Resource Pool Management
```python
class ResourcePool:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_connections = {}
    
    async def acquire_resource(self, service_name: str):
        async with self.semaphore:
            if service_name not in self.active_connections:
                self.active_connections[service_name] = await self.create_connection(service_name)
            return self.active_connections[service_name]
```

## Performance Considerations

### 1. Connection Pooling
- **HTTP Session Reuse**: Persistent connections for API calls
- **Connection Limits**: Respect rate limits and concurrent connection limits
- **Timeout Management**: Appropriate timeouts for different service types

### 2. Caching Strategies
- **Response Caching**: Cache expensive API responses with TTL
- **Model Caching**: Local caching of downloaded models
- **Configuration Caching**: Cache parsed configurations

### 3. Error Recovery
- **Graceful Degradation**: Fallback to alternative services
- **Retry Logic**: Exponential backoff for transient failures
- **Health Monitoring**: Continuous service health assessment

## Security Considerations

### 1. API Key Management
```python
import os
from pathlib import Path

def load_api_credentials() -> Dict[str, str]:
    """Load API credentials from secure sources"""
    credentials = {}
    
    # Try environment variables first
    for service in ['OPENAI_API_KEY', 'HUGGINGFACE_TOKEN']:
        if os.getenv(service):
            credentials[service] = os.getenv(service)
    
    # Fallback to secure credential file
    cred_file = Path.home() / '.config' / 'ericsson-pipeline' / 'credentials.json'
    if cred_file.exists():
        with open(cred_file, 'r') as f:
            file_creds = json.load(f)
            credentials.update(file_creds)
    
    return credentials
```

### 2. Network Security
- **TLS/SSL Verification**: Enforce secure connections
- **Certificate Validation**: Proper certificate handling
- **Request Sanitization**: Input validation for all external requests

### 3. Data Privacy
- **Local Processing**: Prefer local models for sensitive data
- **Data Scrubbing**: Remove sensitive information before external API calls
- **Audit Logging**: Track all external service interactions

## Monitoring and Observability

### 1. Integration Health Monitoring
```python
class IntegrationMonitor:
    def __init__(self):
        self.health_checks = {
            'ollama': self.check_ollama_health,
            'mlx': self.check_mlx_health,
            'docling': self.check_docling_health
        }
    
    async def monitor_all_integrations(self) -> Dict[str, bool]:
        results = {}
        for name, check_func in self.health_checks.items():
            try:
                results[name] = await check_func()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
        return results
```

### 2. Performance Metrics
- **Response Times**: Track API response latencies
- **Success Rates**: Monitor success/failure rates
- **Resource Usage**: Monitor memory and CPU usage per integration
- **Throughput**: Track processing rates and bottlenecks

### 3. Alerting and Notifications
- **Service Failures**: Alert when critical services are down
- **Performance Degradation**: Alert on performance threshold violations
- **Resource Exhaustion**: Alert on resource usage limits