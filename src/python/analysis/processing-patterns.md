# Key Processing Patterns and Implementations

## Overview

The codebase implements sophisticated processing patterns optimized for M3 Max hardware with emphasis on concurrent processing, memory efficiency, and fault tolerance.

## Core Processing Patterns

### 1. Concurrent Processing Architecture

#### ProcessPoolExecutor Pattern
```python
# Pattern: CPU-intensive document processing
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for batch in file_batches:
        future = executor.submit(process_batch, batch)
        futures.append(future)
    
    for future in as_completed(futures):
        result = future.result()
        process_result(result)
```

**Usage Locations:**
- `unified_document_processor.py:2640` - Main document processing
- `docling_converter.py:1291` - PDF processing
- `preprocessing_pipeline.py:1027` - Batch markdown processing

**Optimizations:**
- Dynamic worker count: `min(max_workers, len(batches), cpu_count())`
- Batch size optimization for memory efficiency
- Progress tracking with concurrent futures

#### ThreadPoolExecutor Pattern
```python
# Pattern: I/O-bound operations
with ThreadPoolExecutor(max_workers=min(self.max_workers, len(specs))) as executor:
    download_futures = {
        executor.submit(self.download_spec, spec): spec 
        for spec in key_specs
    }
```

**Usage Locations:**
- `3gpp_downloader.py:376` - Concurrent downloads
- Various network request operations
- File I/O operations

### 2. Memory Management Patterns

#### Stream Processing Pattern
```python
def process_large_files(self, file_path: Path) -> Iterator[ProcessedChunk]:
    """Process files in chunks to manage memory usage"""
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(self.chunk_size)
            if not chunk:
                break
            yield self.process_chunk(chunk)
```

#### Dynamic Memory Allocation
- **M3 Max Optimization**: Aggressive batch sizing up to 128GB capacity
- **Memory Monitoring**: Real-time RAM usage tracking with psutil
- **Adaptive Sizing**: Dynamic adjustment based on available memory

### 3. Document Processing Patterns

#### Multi-Format Processing Chain
```python
class BaseProcessor:
    """Abstract base for all document processors"""
    
    def process_document(self, input_path: Path) -> ProcessingResult:
        # 1. Format detection
        format_type = self.detect_format(input_path)
        
        # 2. Specialized processing
        processor = self.get_processor(format_type)
        raw_content = processor.extract_content(input_path)
        
        # 3. Normalization
        normalized = self.normalize_content(raw_content)
        
        # 4. Quality assessment
        quality_score = self.assess_quality(normalized)
        
        return ProcessingResult(content=normalized, quality=quality_score)
```

**Supported Formats:**
- HTML → BeautifulSoup processing
- PDF → pypdf/pdfplumber dual strategy
- CSV/Excel → pandas processing
- 3GPP ZIP → specialized extraction
- TXT/Markdown → encoding detection + normalization

#### LangExtract 6-Category System
```python
@dataclass
class DocumentExtraction:
    main_features: List[str]
    parameters: List[str]      # Never contain "pm"
    counters: List[str]        # Always contain "pm"
    events: List[str]
    related_features: List[str]
    activation_procedures: List[str]
```

**Quality Progression:**
- Basic Mode: 0.371 baseline quality
- CMEDIT Enhanced: 0.430 quality (+15.6%)
- Feature Grouped: 0.657 quality (+76.8%)
- Hybrid Target: 0.742+ quality (+100%+)

### 4. Error Handling and Resilience Patterns

#### Circuit Breaker Pattern
```python
class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure < self.timeout:
                    raise CircuitBreakerOpenError()
                self.state = CircuitBreakerState.HALF_OPEN
            
            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except Exception as e:
                self.on_failure()
                raise
        return wrapper
```

#### Retry with Exponential Backoff
```python
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, ConnectionError),
    max_tries=3,
    max_time=30
)
def robust_api_call(self, endpoint: str) -> Dict[str, Any]:
    response = requests.get(endpoint, timeout=self.timeout)
    response.raise_for_status()
    return response.json()
```

### 5. MLX Optimization Patterns

#### M3 Max Fine-tuning Configuration
```python
class OptimizedQwen3TelecomFineTuner:
    def __init__(self):
        # M3 Max Optimizations
        self.lora_rank = 64              # Increased for better adaptation
        self.batch_size = 32             # Aggressive for 128GB
        self.gradient_accumulation = 4    # Memory efficiency
        self.mixed_precision = True       # FP16 training
        self.parallel_loading = True      # Multi-threaded data loading
```

#### Dynamic Resource Management
```python
def optimize_for_m3_max(self):
    """Optimize training configuration for M3 Max hardware"""
    available_memory = psutil.virtual_memory().available
    
    if available_memory > 100 * 1024**3:  # > 100GB
        self.batch_size = 64
        self.max_workers = 16
    elif available_memory > 50 * 1024**3:  # > 50GB  
        self.batch_size = 32
        self.max_workers = 12
    else:
        self.batch_size = 16
        self.max_workers = 8
```

## Advanced Processing Techniques

### 1. Hybrid Processing Strategy
- **Route Classification**: Documents classified by complexity and type
- **Multi-Mode Processing**: Basic, Enhanced, Feature Grouped, Hybrid
- **Quality-Based Routing**: Automatic mode selection based on quality targets

### 2. Intelligent Chunking
```python
def smart_chunk_document(self, content: str, target_size: int = 4000) -> List[str]:
    """Intelligent chunking preserving semantic boundaries"""
    sentences = self.sentence_splitter.split(content)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### 3. Quality-Driven Processing
- **Progressive Enhancement**: Start with basic processing, enhance based on quality
- **Quality Metrics**: Automated assessment of extraction completeness
- **Adaptive Routing**: Route complex documents to more sophisticated processors

## Performance Optimization Strategies

### 1. Memory Pooling
- **Object Reuse**: Processor instance pooling
- **Buffer Management**: Reusable buffers for large documents
- **Garbage Collection**: Strategic memory cleanup

### 2. I/O Optimization  
- **Bulk Operations**: Batch file operations
- **Streaming**: Process large files without full loading
- **Caching**: Intelligent result caching with TTL

### 3. CPU Utilization
- **Multi-core Processing**: Full CPU utilization with ProcessPoolExecutor
- **Affinity Optimization**: CPU core pinning for consistent performance
- **NUMA Awareness**: Memory locality optimization

## Integration Patterns

### 1. Ollama Integration
```python
class OllamaConnection:
    def __init__(self):
        self.session = requests.Session()
        self.session.mount("http://", HTTPAdapter(max_retries=3))
        
    async def extract_features(self, content: str) -> DocumentExtraction:
        prompt = self.build_extraction_prompt(content)
        response = await self.call_ollama(prompt)
        return self.parse_response(response)
```

### 2. MLX Framework Integration
- **Model Loading**: Efficient model loading and caching
- **Batch Processing**: Optimized batch inference
- **Memory Management**: MLX-specific memory optimization

### 3. Cross-Language Coordination
- **Configuration Sharing**: YAML configs shared between Python and Rust
- **Data Exchange**: JSON-based data exchange formats
- **Process Orchestration**: Coordinated execution patterns