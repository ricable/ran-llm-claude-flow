# Architecture Changes and Improvements

## Overview

This document details the structural changes required to transform the current fragmented RAG-LLM pipeline into a unified, maintainable, and high-performance system optimized for M3 Max. The architecture changes focus on consolidation, performance optimization, and developer experience improvements.

## Current Architecture Analysis

### Existing Structure
```
Current Complex Structure:
┌─ code-samples/ran-llm/
│  ├─ packages/
│  │  ├─ core/                    # Basic utilities (minimal)
│  │  ├─ processors/              # 95+ Python files, complex hierarchy  
│  │  └─ finetuning/             # Training utilities
│  ├─ rust/                      # Parallel Rust implementation
│  │  ├─ src/                    # 30+ Rust files
│  │  └─ benches/                # Performance benchmarks
│  ├─ config/config.yaml         # 594-line monolithic config
│  └─ analysis/                  # Standalone analysis tools
```

### Problems with Current Architecture
1. **Fragmented Processing Logic** - Similar functionality spread across Python/Rust
2. **Deep Module Hierarchies** - Complex import paths (`processors.document.langextract.analysis.advanced_quality_assessment`)
3. **Configuration Sprawl** - Single massive config file with duplication
4. **Inconsistent Interfaces** - Different patterns across processors
5. **Resource Management Chaos** - Multiple optimization strategies working against each other

## Target Architecture

### New Unified Structure
```
Simplified Unified Structure:
┌─ src/
│  ├─ core/                      # Foundation layer
│  │  ├─ interfaces/             # Abstract interfaces
│  │  ├─ config/                 # Configuration management
│  │  ├─ models/                 # Data models
│  │  └─ utils/                  # Shared utilities
│  ├─ llm/                       # LLM abstraction layer
│  │  ├─ clients/                # LLM client implementations
│  │  ├─ profiles/               # Model profiles (thinking, fast, instruct)
│  │  └─ optimization/           # MLX/M3 optimizations
│  ├─ processing/                # Document processing engine
│  │  ├─ strategies/             # Processing strategy implementations
│  │  ├─ pipelines/              # Processing pipelines
│  │  └─ quality/                # Quality assessment
│  ├─ output/                    # Output formatting
│  │  ├─ formats/                # Output format handlers
│  │  └─ compression/            # Compression utilities
│  ├─ cli/                       # Command line interface
│  └─ performance/               # M3 Max optimization
├─ config/                       # Hierarchical configuration
│  ├─ base.yaml                  # System defaults
│  ├─ profiles/                  # Environment-specific configs
│  └─ schemas/                   # Configuration validation
├─ tests/                        # Comprehensive test suite
└─ docs/                         # Documentation
```

## Core Architecture Components

### 1. Foundation Layer (`src/core/`)

#### 1.1 Interface Design
```python
# src/core/interfaces/processor.py
from abc import ABC, abstractmethod
from typing import Protocol

class DocumentProcessor(Protocol):
    async def process(self, document: Document) -> ProcessingResult: ...
    def validate_config(self, config: ProcessingConfig) -> bool: ...
    def estimate_resources(self, workload: Workload) -> ResourceEstimate: ...

class QualityAssessor(Protocol):
    def assess(self, content: str, metadata: dict) -> QualityScore: ...
    def batch_assess(self, contents: list[str]) -> list[QualityScore]: ...
```

#### 1.2 Configuration Management
```python
# src/core/config/manager.py
class ConfigurationManager:
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.schema_validator = ConfigSchemaValidator()
        
    def load_profile(self, profile: str) -> PipelineConfig:
        """Load hierarchical configuration with inheritance"""
        base_config = self._load_yaml("base.yaml")
        profile_config = self._load_yaml(f"profiles/{profile}.yaml")
        return self._merge_configs(base_config, profile_config)
        
    def validate(self, config: PipelineConfig) -> ValidationResult:
        """Validate configuration against schema"""
        return self.schema_validator.validate(config)
```

#### 1.3 Data Models
```python
# src/core/models/document.py
@dataclass
class Document:
    id: str
    content: str
    metadata: DocumentMetadata
    source_path: Path
    processing_hints: dict[str, Any] = field(default_factory=dict)

@dataclass  
class ProcessingResult:
    document_id: str
    qa_pairs: list[QAPair]
    quality_scores: list[QualityScore]
    processing_metadata: ProcessingMetadata
    errors: list[ProcessingError] = field(default_factory=list)
```

### 2. LLM Abstraction Layer (`src/llm/`)

#### 2.1 Unified Client Interface
```python
# src/llm/clients/unified_client.py
class UnifiedLLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.profile_manager = ProfileManager(config.profiles)
        self.connection_pool = ConnectionPool(config.connection_pool)
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker)
        
    async def generate(self, 
                      prompt: str, 
                      profile: str = "default",
                      **kwargs) -> LLMResponse:
        """Unified generation interface"""
        model_config = self.profile_manager.get_profile(profile)
        
        async with self.connection_pool.get_connection() as conn:
            return await self._generate_with_retries(
                conn, prompt, model_config, **kwargs
            )
            
    async def batch_generate(self, 
                           prompts: list[str], 
                           profile: str = "default") -> list[LLMResponse]:
        """Optimized batch processing"""
        # Implement intelligent batching with M3 optimization
```

#### 2.2 Profile Management
```python
# src/llm/profiles/profile_manager.py
class ProfileManager:
    def __init__(self, profiles_config: dict):
        self.profiles = {
            'thinking': ThinkingProfile(profiles_config['thinking']),
            'fast': FastProfile(profiles_config['fast']),
            'instruct': InstructProfile(profiles_config['instruct'])
        }
        
    def get_profile(self, name: str) -> ModelProfile:
        return self.profiles.get(name, self.profiles['instruct'])
```

#### 2.3 M3 Max Optimization
```python
# src/llm/optimization/m3_optimizer.py
class M3MaxOptimizer:
    def __init__(self):
        self.unified_memory_gb = 128
        self.performance_cores = 12
        self.efficiency_cores = 4
        
    def optimize_for_workload(self, workload: LLMWorkload) -> OptimizationConfig:
        """Dynamic optimization based on workload characteristics"""
        if workload.is_thinking_intensive():
            return self._optimize_for_thinking(workload)
        elif workload.is_batch_heavy():
            return self._optimize_for_batch(workload)
        else:
            return self._optimize_balanced(workload)
            
    def _optimize_for_thinking(self, workload: LLMWorkload) -> OptimizationConfig:
        return OptimizationConfig(
            max_concurrent_requests=1,  # Single request for thinking
            memory_allocation=self.unified_memory_gb * 0.8,
            use_performance_cores=True,
            enable_mlx_acceleration=True
        )
```

### 3. Processing Engine (`src/processing/`)

#### 3.1 Strategy Pattern Implementation
```python
# src/processing/strategies/base.py
class ProcessingStrategy(ABC):
    def __init__(self, llm_client: UnifiedLLMClient, quality_assessor: QualityAssessor):
        self.llm_client = llm_client
        self.quality_assessor = quality_assessor
        
    @abstractmethod
    async def process(self, document: Document) -> ProcessingResult:
        pass
        
    @abstractmethod
    def estimate_processing_time(self, document: Document) -> timedelta:
        pass

# src/processing/strategies/langextract.py
class LangExtractStrategy(ProcessingStrategy):
    async def process(self, document: Document) -> ProcessingResult:
        # Consolidates current langextract logic
        chunks = await self._chunk_document(document)
        qa_pairs = []
        
        for chunk in chunks:
            chunk_qa = await self._generate_qa_pairs(chunk)
            qa_pairs.extend(chunk_qa)
            
        quality_scores = await self.quality_assessor.batch_assess(
            [pair.answer for pair in qa_pairs]
        )
        
        return ProcessingResult(
            document_id=document.id,
            qa_pairs=qa_pairs,
            quality_scores=quality_scores,
            processing_metadata=self._create_metadata()
        )
```

#### 3.2 Pipeline Orchestration
```python
# src/processing/pipelines/unified_pipeline.py
class UnifiedProcessingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm_client = UnifiedLLMClient(config.llm)
        self.quality_assessor = UnifiedQualityAssessor(config.quality)
        self.performance_monitor = PerformanceMonitor()
        
    async def process_documents(self, documents: list[Document]) -> PipelineResult:
        """Main processing orchestration"""
        strategy = self._select_strategy(documents)
        processor = DocumentProcessor(strategy)
        
        # M3 Max optimized batch processing
        batch_size = self._calculate_optimal_batch_size()
        batches = self._create_batches(documents, batch_size)
        
        results = []
        async for batch in batches:
            batch_results = await processor.process_batch(batch)
            results.extend(batch_results)
            
            # Adaptive performance adjustment
            self._adjust_batch_size_if_needed()
            
        return PipelineResult(results=results, metadata=self._create_metadata())
```

### 4. Quality Assessment Unification

#### 4.1 Unified Quality Engine
```python
# src/processing/quality/unified_assessor.py
class UnifiedQualityAssessor:
    def __init__(self, config: QualityConfig):
        self.config = config
        self.technical_analyzer = TechnicalAnalyzer(config.technical_terms)
        self.complexity_calculator = ComplexityCalculator()
        self.content_validator = ContentValidator()
        
    async def assess(self, content: str, metadata: dict = None) -> QualityScore:
        """Single assessment interface for all content types"""
        
        # Parallel assessment components
        technical_score = await self.technical_analyzer.analyze(content)
        complexity_score = self.complexity_calculator.calculate(content)  
        validation_score = self.content_validator.validate(content)
        
        # Weighted final score
        final_score = self._calculate_weighted_score(
            technical_score, complexity_score, validation_score
        )
        
        return QualityScore(
            overall=final_score,
            technical=technical_score,
            complexity=complexity_score,
            validation=validation_score,
            metadata=metadata or {}
        )
```

## Configuration Architecture

### Hierarchical Configuration System

#### Base Configuration
```yaml
# config/base.yaml
system:
  name: "ericsson-dataset-pipeline"
  version: "2.0.0"
  
  # M3 Max hardware optimization
  hardware:
    unified_memory_gb: 128
    performance_cores: 12
    efficiency_cores: 4
    enable_mlx: true
    enable_simd: true

llm:
  provider: "LMStudio"  
  base_url: "http://localhost:1234"
  
  # Connection optimization
  connection_pool:
    max_connections: 20
    idle_timeout_seconds: 1800
    
  # Unified profiles
  profiles:
    thinking:
      model_name: "qwen3-30b-a3b-thinking-2507-mlx"
      timeout_seconds: 1800
      max_tokens: 12000
      temperature: 0.5
      max_concurrent: 1
      
    fast:
      model_name: "qwen/qwen3-1.7b:2" 
      timeout_seconds: 300
      max_tokens: 2000
      temperature: 0.3
      max_concurrent: 4
      
    instruct:
      model_name: "qwen3-coder-30b-a3b-instruct-dwq"
      timeout_seconds: 900
      max_tokens: 8000
      temperature: 0.4
      max_concurrent: 2

processing:
  # Intelligent defaults
  batch_size: 32
  quality_threshold: 3.8
  enable_adaptive_scaling: true
  
  # M3 Max optimization
  workers: 8  # 75% of performance cores
  memory_limit_gb: 96  # 75% of unified memory
  
quality:
  # Unified quality thresholds
  min_score: 3.8
  technical_weight: 0.4
  complexity_weight: 0.3
  validation_weight: 0.3

output:
  formats: ["jsonl", "parquet"]
  compression:
    algorithm: "zstd"
    level: 3
```

#### Environment-Specific Profiles
```yaml
# config/profiles/development.yaml
extends: base

processing:
  max_documents: 10  # Limit for development
  workers: 4         # Reduced for development
  
llm:
  profiles:
    thinking:
      timeout_seconds: 600  # Shorter timeout for dev

# config/profiles/production.yaml  
extends: base

processing:
  max_documents: 0  # Unlimited
  enable_monitoring: true
  enable_metrics: true
  
performance:
  enable_profiling: true
  memory_monitoring: true
```

## Performance Architecture

### M3 Max Specific Optimizations

#### 1. Unified Memory Management
```python
# src/performance/memory_manager.py
class UnifiedMemoryManager:
    def __init__(self, total_memory_gb: int = 128):
        self.total_memory = total_memory_gb * 1024 * 1024 * 1024  # bytes
        self.reserved_system = self.total_memory * 0.15  # 15% for system
        self.available = self.total_memory - self.reserved_system
        
    def allocate_for_pipeline(self, worker_count: int) -> MemoryAllocation:
        """Smart memory allocation across pipeline components"""
        
        # Memory distribution strategy
        llm_memory = self.available * 0.6    # 60% for LLM operations
        processing_memory = self.available * 0.25  # 25% for processing
        buffer_memory = self.available * 0.15      # 15% for buffers
        
        return MemoryAllocation(
            llm_memory_per_worker=llm_memory / worker_count,
            processing_memory=processing_memory,
            buffer_memory=buffer_memory
        )
```

#### 2. CPU Core Optimization
```python  
# src/performance/cpu_optimizer.py
class M3MaxCPUOptimizer:
    def __init__(self):
        self.performance_cores = 12
        self.efficiency_cores = 4
        
    def optimize_for_workload(self, workload_type: str) -> CPUConfig:
        """Intelligent core allocation based on workload"""
        
        if workload_type == "thinking_intensive":
            # Use all performance cores for single thinking task
            return CPUConfig(
                performance_cores=self.performance_cores,
                efficiency_cores=0,
                thread_affinity=True,
                enable_turbo=True
            )
        elif workload_type == "batch_processing":
            # Distribute across both core types
            return CPUConfig(
                performance_cores=8,  # Main processing
                efficiency_cores=4,   # Background tasks
                thread_affinity=False,
                enable_parallel=True
            )
```

## API Design Changes

### Unified API Interface

#### 1. Main Processing API
```python
# src/api/processor.py
class ProcessorAPI:
    def __init__(self, config: PipelineConfig):
        self.pipeline = UnifiedProcessingPipeline(config)
        
    async def process_documents(self, 
                              documents: list[Document],
                              options: ProcessingOptions = None) -> ProcessingResult:
        """Main processing entry point"""
        return await self.pipeline.process_documents(documents)
        
    async def process_single(self, 
                           document: Document,
                           strategy: str = "auto") -> ProcessingResult:
        """Single document processing"""
        return await self.pipeline.process_single(document, strategy)
```

#### 2. CLI Unification  
```python
# src/cli/main.py
@click.group()
@click.option('--config', default='production', help='Configuration profile')
@click.option('--verbose', is_flag=True, help='Verbose output')
def cli(config, verbose):
    """Unified Ericsson Dataset Pipeline CLI"""

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--strategy', default='auto', help='Processing strategy')
def process(input_path, output_path, strategy):
    """Process documents into training datasets"""
    # Unified processing command

@cli.command()
@click.argument('csv_path', type=click.Path(exists=True))
def cmedit(csv_path):
    """Generate CMEDIT workflows from CSV parameters"""
    # CMEDIT workflow generation

@cli.command()
@click.argument('dataset_path', type=click.Path(exists=True))
def analyze(dataset_path):
    """Analyze dataset quality and diversity"""
    # Dataset analysis
```

## Migration Path

### Phase 1: Foundation (Week 1-2)
1. Create new `src/` structure
2. Implement core interfaces and models
3. Build configuration management system
4. Create unified LLM client

### Phase 2: Processing Unification (Week 3-4)
1. Implement processing strategies
2. Build unified pipeline orchestration
3. Consolidate quality assessment
4. Create performance optimization layer

### Phase 3: Integration (Week 5-6)
1. Migrate existing functionality
2. Update CLI and API interfaces
3. Comprehensive testing
4. Performance validation

### Phase 4: Cleanup (Week 7)
1. Remove deprecated code
2. Update documentation
3. Final performance tuning
4. Production deployment

## Benefits of New Architecture

### Maintainability
- **Clear separation of concerns** - Each layer has distinct responsibilities
- **Consistent interfaces** - Unified patterns across all components  
- **Reduced complexity** - Simple, hierarchical structure
- **Better testability** - Clear dependencies and interfaces

### Performance  
- **M3 Max optimization** - Native support for unified memory and MLX
- **Intelligent resource management** - Dynamic allocation based on workload
- **Reduced overhead** - Elimination of duplicate processing
- **Better scalability** - Adaptive performance tuning

### Developer Experience
- **Simple configuration** - Hierarchical, validated configuration system
- **Unified CLI** - Single command interface for all operations
- **Clear documentation** - Architecture matches mental model
- **Easy extension** - Plugin-based strategy pattern

This architecture provides a solid foundation for the unified RAG-LLM pipeline while maintaining flexibility for future enhancements and optimizations.