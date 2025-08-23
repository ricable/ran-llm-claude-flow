# Migration Phases - Step-by-Step Implementation Plan

## Executive Summary

This document provides a detailed, week-by-week migration plan to transform the current fragmented RAG-LLM pipeline into a unified, optimized system. The plan is designed to maintain 100% functionality throughout the migration with minimal disruption to ongoing operations.

## Migration Overview

### Timeline: 7 Weeks Total
- **Phase 1:** Foundation Layer (Weeks 1-2)
- **Phase 2:** Core Systems Migration (Weeks 3-4) 
- **Phase 3:** Feature Integration (Weeks 5-6)
- **Phase 4:** Optimization & Cleanup (Week 7)

### Key Principles
1. **Zero Downtime** - Maintain working system throughout migration
2. **Incremental Progress** - Each week delivers working functionality
3. **Risk Mitigation** - Rollback capability at every phase
4. **Quality Assurance** - Comprehensive testing at each milestone

---

## Phase 1: Foundation Layer (Weeks 1-2)

### Week 1: Core Infrastructure

#### Day 1-2: Project Structure Setup
```bash
# Create new unified structure
mkdir -p src/{core,llm,processing,output,cli,performance}
mkdir -p src/core/{interfaces,config,models,utils}
mkdir -p src/llm/{clients,profiles,optimization}
mkdir -p config/{profiles,schemas}

# Initialize new pyproject.toml with unified dependencies
```

**Deliverables:**
- New project structure created
- Updated pyproject.toml with consolidated dependencies
- Base configuration schema defined

**Validation:**
- [ ] Directory structure matches architecture specification
- [ ] Dependencies install successfully with uv
- [ ] Basic import structure works

#### Day 3-4: Interface Design
Create core interfaces that will govern the unified system:

```python
# src/core/interfaces/processor.py
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class DocumentProcessor(Protocol):
    async def process(self, document: Document) -> ProcessingResult: ...
    def validate_config(self, config: ProcessingConfig) -> bool: ...
    def estimate_resources(self, workload: Workload) -> ResourceEstimate: ...

# src/core/interfaces/llm_client.py  
@runtime_checkable
class LLMClient(Protocol):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...
    async def batch_generate(self, prompts: list[str], **kwargs) -> list[LLMResponse]: ...
```

**Deliverables:**
- Core interface definitions
- Base data models (Document, ProcessingResult, etc.)
- Type annotations for all interfaces

**Validation:**
- [ ] All interfaces are properly typed
- [ ] Interface design reviewed and approved
- [ ] Basic usage examples work

#### Day 5-7: Configuration System
Implement hierarchical configuration management:

```python
# src/core/config/manager.py
class ConfigurationManager:
    def load_profile(self, profile: str) -> PipelineConfig:
        """Load hierarchical configuration with validation"""
        base_config = self._load_yaml("config/base.yaml")
        profile_config = self._load_yaml(f"config/profiles/{profile}.yaml")
        merged = self._merge_configs(base_config, profile_config)
        return self._validate_and_parse(merged)
```

**Deliverables:**
- Hierarchical configuration system
- Configuration validation with Pydantic
- Migration utility for existing config.yaml

**Validation:**
- [ ] Can load and validate configurations
- [ ] Existing config.yaml successfully migrated
- [ ] Configuration profiles work correctly

### Week 2: LLM Abstraction Layer

#### Day 1-3: Unified LLM Client
Build the core LLM client that consolidates all model interactions:

```python
# src/llm/clients/unified_client.py
class UnifiedLLMClient:
    def __init__(self, config: LLMConfig):
        self.profiles = self._load_profiles(config.profiles)
        self.connection_pool = ConnectionPool(config.connection_pool)
        self.m3_optimizer = M3MaxOptimizer() if self._is_m3_max() else None
        
    async def generate(self, prompt: str, profile: str = "instruct", **kwargs) -> LLMResponse:
        model_config = self.profiles[profile]
        
        # M3 Max optimized request handling
        if self.m3_optimizer and profile == "thinking":
            return await self._generate_thinking_optimized(prompt, model_config)
        else:
            return await self._generate_standard(prompt, model_config)
```

**Deliverables:**
- Unified LLM client implementation
- Profile management system (thinking, fast, instruct)
- Connection pooling and circuit breaker
- Basic M3 Max optimizations

**Validation:**
- [ ] Can successfully connect to LMStudio
- [ ] All three profiles (thinking, fast, instruct) work
- [ ] Connection pooling reduces latency
- [ ] Circuit breaker prevents cascade failures

#### Day 4-5: Profile System
Implement intelligent model selection:

```python
# src/llm/profiles/profile_manager.py
class ProfileManager:
    def select_profile(self, task_context: TaskContext) -> str:
        """Intelligent profile selection based on task requirements"""
        if task_context.requires_deep_thinking():
            return "thinking"
        elif task_context.is_simple_extraction():
            return "fast"  
        else:
            return "instruct"
```

**Deliverables:**
- Automatic profile selection logic
- Task classification system
- Profile performance monitoring

**Validation:**
- [ ] Profile selection works correctly for different task types
- [ ] Performance metrics are collected per profile
- [ ] Can override automatic selection when needed

#### Day 6-7: Integration Testing
Test the foundation components together:

**Deliverables:**
- Integration tests for configuration + LLM client
- Performance benchmarks for M3 Max optimizations
- Documentation for foundation layer APIs

**Validation:**
- [ ] Integration tests pass
- [ ] Performance meets expectations
- [ ] APIs are documented and stable

---

## Phase 2: Core Systems Migration (Weeks 3-4)

### Week 3: Processing Engine Migration

#### Day 1-2: Quality Assessment Unification
Consolidate the scattered quality assessment logic:

```python
# src/processing/quality/unified_assessor.py
class UnifiedQualityAssessor:
    def __init__(self, config: QualityConfig):
        # Consolidate logic from:
        # - processors/document/langextract/analysis/advanced_quality_assessment.py
        # - quality_assessment section of config
        # - Various quality scoring throughout codebase
        pass
        
    async def assess(self, content: str, metadata: dict = None) -> QualityScore:
        """Single quality assessment interface"""
        # Parallel execution of quality components
        results = await asyncio.gather(
            self.technical_analyzer.analyze(content),
            self.complexity_calculator.calculate(content),
            self.content_validator.validate(content)
        )
        return self._combine_scores(results)
```

**Deliverables:**
- Unified quality assessment engine
- Migration of existing quality logic
- Performance optimized assessment pipeline

**Validation:**
- [ ] Quality scores match existing implementation
- [ ] Assessment performance improved by 20-30%
- [ ] All quality features preserved

#### Day 3-4: Processing Strategy Implementation
Implement the strategy pattern for different processing types:

```python
# src/processing/strategies/langextract_strategy.py
class LangExtractStrategy(ProcessingStrategy):
    async def process(self, document: Document) -> ProcessingResult:
        # Migrate logic from processors/document/langextract/
        chunks = await self.chunker.chunk_document(document)
        
        qa_pairs = []
        async for chunk in chunks:
            chunk_qa = await self.llm_client.generate(
                self._build_qa_prompt(chunk), 
                profile="thinking"
            )
            qa_pairs.extend(self._parse_qa_response(chunk_qa))
            
        return ProcessingResult(
            document_id=document.id,
            qa_pairs=qa_pairs,
            quality_scores=await self.quality_assessor.batch_assess(qa_pairs)
        )
```

**Deliverables:**
- LangExtract strategy implementation
- CSV processing strategy  
- 3GPP processing strategy
- Strategy factory and selection logic

**Validation:**
- [ ] Each strategy produces equivalent results to original
- [ ] Strategy selection works automatically
- [ ] Performance is maintained or improved

#### Day 5-7: Pipeline Orchestration
Build the main pipeline that coordinates all processing:

```python
# src/processing/pipelines/unified_pipeline.py
class UnifiedProcessingPipeline:
    async def process_documents(self, documents: list[Document]) -> PipelineResult:
        # Intelligent batch processing with M3 Max optimization
        strategy = self.strategy_factory.create_strategy(documents[0])
        processor = DocumentProcessor(strategy)
        
        optimal_batch_size = self.m3_optimizer.calculate_batch_size(
            documents, self.available_memory
        )
        
        results = []
        async for batch in self.batch_processor.process_batches(documents, optimal_batch_size):
            batch_results = await processor.process_batch(batch)
            results.extend(batch_results)
            
        return PipelineResult(results)
```

**Deliverables:**
- Unified pipeline orchestration
- M3 Max optimized batch processing
- Progress monitoring and adaptive scaling
- Error handling and recovery

**Validation:**
- [ ] Pipeline processes documents end-to-end
- [ ] Batch optimization improves throughput
- [ ] Error handling works correctly
- [ ] Progress monitoring is accurate

### Week 4: Output System & CLI Migration

#### Day 1-3: Output Format Unification
Consolidate output formatting across all processors:

```python
# src/output/formats/jsonl_formatter.py
class JSONLFormatter(OutputFormatter):
    def format(self, results: list[ProcessingResult]) -> str:
        # Unified formatting logic for all processors
        return "\n".join(
            json.dumps(self._format_result(result))
            for result in results
        )

# src/output/manager.py        
class OutputManager:
    def __init__(self, config: OutputConfig):
        self.formatters = {
            'jsonl': JSONLFormatter(config.jsonl),
            'parquet': ParquetFormatter(config.parquet),
            'csv': CSVFormatter(config.csv)
        }
        
    async def write_results(self, results: list[ProcessingResult], output_path: Path):
        for format_name in self.config.enabled_formats:
            formatter = self.formatters[format_name]
            formatted = formatter.format(results)
            await self._write_with_compression(formatted, output_path, format_name)
```

**Deliverables:**
- Unified output formatting system
- Support for JSONL, Parquet, CSV formats
- Compression handling (Zstd, Gzip)
- Metadata inclusion options

**Validation:**
- [ ] All output formats work correctly
- [ ] Compression reduces file sizes as expected  
- [ ] Metadata is included when configured
- [ ] Output quality matches existing system

#### Day 4-5: CLI Unification
Create single CLI that replaces multiple existing CLIs:

```python
# src/cli/main.py
@click.group()
@click.option('--config', default='production')
@click.option('--verbose', is_flag=True)
def cli(config, verbose):
    """Unified Ericsson Dataset Pipeline CLI"""

@cli.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('--strategy', help='Processing strategy (auto, langextract, csv, 3gpp)')
@click.option('--max-documents', type=int, help='Limit number of documents')
@click.option('--workers', type=int, help='Number of worker threads')
def process(input_path, output_path, strategy, max_documents, workers):
    """Process documents into training datasets"""
    # Unified processing command that handles all document types
```

**Deliverables:**
- Single unified CLI replacing multiple CLIs
- Backwards compatibility with existing commands
- Rich progress display and monitoring
- Configuration override capabilities

**Validation:**  
- [ ] All existing CLI functionality preserved
- [ ] New CLI is more intuitive than originals
- [ ] Progress display works correctly
- [ ] Configuration overrides work

#### Day 6-7: Integration & Testing
Test all Phase 2 components together:

**Deliverables:**
- End-to-end integration tests
- Performance benchmarks vs original system
- Memory usage profiling
- CLI user acceptance testing

**Validation:**
- [ ] End-to-end processing works correctly
- [ ] Performance is equal or better than original
- [ ] Memory usage is optimized
- [ ] CLI usability is improved

---

## Phase 3: Feature Integration (Weeks 5-6)

### Week 5: Advanced Features Migration

#### Day 1-2: CMEDIT Workflow Integration
Migrate CMEDIT workflow generation capabilities:

```python
# src/processing/strategies/cmedit_strategy.py  
class CMEditStrategy(ProcessingStrategy):
    async def process(self, document: Document) -> ProcessingResult:
        # Migrate from packages/processors/src/processors/document/cmedit/
        parameters = self.parameter_extractor.extract(document)
        workflows = await self.workflow_generator.generate_workflows(parameters)
        
        return ProcessingResult(
            document_id=document.id,
            workflows=workflows,
            metadata={"type": "cmedit", "parameter_count": len(parameters)}
        )
```

**Deliverables:**
- CMEDIT workflow generation integration
- Feature grouping and parameter extraction
- Workflow validation and quality checks

**Validation:**
- [ ] CMEDIT workflows generate correctly
- [ ] Feature grouping works as before
- [ ] Workflow quality is maintained

#### Day 3-4: Advanced Analytics Integration
Migrate analysis and monitoring capabilities:

```python
# src/processing/analytics/analytics_engine.py
class AnalyticsEngine:
    def __init__(self, config: AnalyticsConfig):
        # Consolidate from:
        # - packages/processors/src/processors/document/langextract/analysis/
        # - analysis/performance_optimization_engine.py
        pass
        
    async def analyze_dataset(self, dataset_path: Path) -> AnalysisReport:
        """Comprehensive dataset analysis"""
        diversity_metrics = await self.diversity_analyzer.analyze(dataset_path)
        quality_metrics = await self.quality_analyzer.analyze(dataset_path) 
        performance_metrics = await self.performance_analyzer.analyze(dataset_path)
        
        return AnalysisReport(
            diversity=diversity_metrics,
            quality=quality_metrics,
            performance=performance_metrics
        )
```

**Deliverables:**
- Unified analytics engine
- Dataset diversity analysis
- Performance monitoring and optimization
- Comprehensive reporting system

**Validation:**
- [ ] Analytics produce accurate metrics
- [ ] Performance monitoring works in real-time
- [ ] Reports are comprehensive and useful

#### Day 5-7: Special Processing Modes
Migrate specialized processing for different document types:

**Deliverables:**
- PDF processing optimization
- 3GPP specification handling
- HTML document processing
- Path-based routing system

**Validation:**
- [ ] All document types process correctly
- [ ] Routing selects appropriate processors
- [ ] Special features are preserved

### Week 6: Performance Optimization & Monitoring

#### Day 1-3: M3 Max Performance Tuning
Implement comprehensive M3 Max optimizations:

```python
# src/performance/m3_optimizer.py
class M3MaxOptimizer:
    def __init__(self):
        self.unified_memory = UnifiedMemoryManager(128)  # GB
        self.cpu_manager = CPUManager(performance_cores=12, efficiency_cores=4)
        self.mlx_acceleration = MLXAccelerator() if self._mlx_available() else None
        
    def optimize_pipeline(self, pipeline_config: PipelineConfig) -> OptimizedConfig:
        """Comprehensive M3 Max optimization"""
        
        # Memory optimization
        memory_allocation = self.unified_memory.allocate_for_pipeline(
            pipeline_config.workers
        )
        
        # CPU optimization  
        cpu_config = self.cpu_manager.optimize_for_workload(
            pipeline_config.workload_type
        )
        
        # MLX acceleration
        if self.mlx_acceleration:
            mlx_config = self.mlx_acceleration.configure_for_qwen3()
        
        return OptimizedConfig(
            memory=memory_allocation,
            cpu=cpu_config,
            mlx=mlx_config
        )
```

**Deliverables:**
- Comprehensive M3 Max performance optimization
- MLX acceleration for Qwen3 models
- Unified memory management
- CPU core optimization

**Validation:**
- [ ] Performance improvements of 15-25%
- [ ] Memory usage optimized for M3 Max
- [ ] MLX acceleration working correctly
- [ ] CPU utilization optimized

#### Day 4-5: Monitoring & Observability
Implement comprehensive monitoring:

```python
# src/monitoring/monitor.py
class PerformanceMonitor:
    def __init__(self, config: MonitoringConfig):
        self.metrics_collector = MetricsCollector()
        self.dashboard = RealTimeDashboard() if config.enable_dashboard else None
        
    async def monitor_pipeline(self, pipeline: UnifiedProcessingPipeline):
        """Real-time pipeline monitoring"""
        async for metrics in self.metrics_collector.collect_metrics(pipeline):
            await self._analyze_performance(metrics)
            if self.dashboard:
                await self.dashboard.update(metrics)
```

**Deliverables:**
- Real-time performance monitoring
- Bottleneck detection and alerting
- Resource utilization tracking
- Performance trend analysis

**Validation:**
- [ ] Monitoring captures all key metrics
- [ ] Bottlenecks are detected accurately
- [ ] Performance trends are useful
- [ ] Alerts work correctly

#### Day 6-7: Integration Testing
Comprehensive testing of all Phase 3 features:

**Deliverables:**
- Full system integration tests
- Performance regression testing
- Feature compatibility validation
- User acceptance testing

**Validation:**
- [ ] All features work together correctly
- [ ] No performance regressions
- [ ] Feature compatibility maintained
- [ ] System ready for production use

---

## Phase 4: Optimization & Cleanup (Week 7)

### Day 1-2: Code Cleanup & Deprecation
Remove old code and clean up the system:

```bash
# Remove old implementations
rm -rf code-samples/ran-llm/packages/
rm -rf code-samples/ran-llm/rust/  # If migrated functionality

# Clean up configuration
mv code-samples/ran-llm/config/config.yaml config/legacy/config.yaml.backup
```

**Activities:**
- Remove deprecated code paths
- Clean up unused dependencies  
- Update import statements throughout codebase
- Remove temporary compatibility layers

**Deliverables:**
- Clean, minimal codebase
- Updated dependency list
- Deprecation warnings removed
- Import structure simplified

### Day 3-4: Documentation Update
Update all documentation for the new unified system:

**Deliverables:**
- Updated README with new structure
- API documentation for all public interfaces
- Migration guide for users
- Performance optimization guide

### Day 5-6: Final Performance Tuning
Optimize the system based on real-world usage:

**Activities:**
- Profile actual usage patterns
- Fine-tune M3 Max optimizations
- Optimize configuration defaults
- Performance regression testing

**Deliverables:**
- Optimized default configurations
- Performance tuning recommendations  
- Benchmark results vs original system
- Regression test suite

### Day 7: Production Deployment
Prepare for production deployment:

**Activities:**
- Final integration testing
- Production configuration validation
- Deployment documentation
- Rollback procedures testing

**Deliverables:**
- Production-ready system
- Deployment documentation
- Rollback procedures
- Monitoring alerts configured

---

## Risk Management & Contingencies

### Risk Mitigation Strategies

#### 1. Functionality Preservation
- **Risk:** Loss of existing functionality during migration
- **Mitigation:** 
  - Comprehensive test suite comparing old vs new
  - Feature parity checklist at each phase
  - Staged rollout with rollback capability

#### 2. Performance Regression  
- **Risk:** New system performs worse than original
- **Mitigation:**
  - Continuous performance benchmarking
  - M3 Max specific optimization validation
  - Performance acceptance criteria

#### 3. Configuration Complexity
- **Risk:** New configuration system is harder to use
- **Mitigation:**
  - Configuration migration utilities
  - Comprehensive validation and error messages
  - Backwards compatibility layer

#### 4. Integration Issues
- **Risk:** Components don't work well together
- **Mitigation:**
  - Integration testing at each phase
  - Interface design review process
  - Gradual integration approach

### Rollback Procedures

#### Git Branch Strategy
```bash
# Development branches
git checkout -b migration/phase-1-foundation
git checkout -b migration/phase-2-core-systems  
git checkout -b migration/phase-3-features
git checkout -b migration/phase-4-optimization

# Main integration branch
git checkout -b migration/unified-system

# Production branch (only after full validation)
git checkout main
```

#### Component-Level Rollback
- Each phase maintains compatibility with previous phase
- Feature flags enable/disable new functionality
- Configuration allows fallback to original behavior

#### Emergency Rollback
- Complete system rollback to last known good state
- Automated backup of working configurations
- Documentation of rollback procedures

### Success Criteria

#### Phase 1 Success Criteria
- [ ] Foundation layer compiles and imports correctly
- [ ] Configuration system loads existing config successfully
- [ ] LLM client connects and generates responses
- [ ] Performance meets baseline requirements

#### Phase 2 Success Criteria  
- [ ] Processing pipeline produces equivalent results
- [ ] Quality assessment scores match original system
- [ ] CLI functionality preserved and improved
- [ ] Memory usage optimized for M3 Max

#### Phase 3 Success Criteria
- [ ] All advanced features migrated successfully
- [ ] Performance improvements of 15-25% achieved
- [ ] Monitoring and analytics working correctly
- [ ] System handles production workloads

#### Phase 4 Success Criteria
- [ ] Codebase reduced by 40-50% in size
- [ ] Configuration complexity reduced by 75%
- [ ] Documentation complete and accurate
- [ ] Production deployment successful

### Communication Plan

#### Weekly Milestones
- **Monday:** Phase kickoff and goal setting
- **Wednesday:** Mid-week progress review
- **Friday:** Phase completion and validation

#### Stakeholder Updates
- Daily progress updates in development channel
- Weekly summary reports with metrics
- Phase completion demos and reviews

#### Issue Escalation
- **Minor issues:** Resolve within development team
- **Major issues:** Escalate with rollback options
- **Critical issues:** Execute emergency rollback procedures

This migration plan provides a systematic approach to transforming the RAG-LLM pipeline while maintaining functionality and minimizing risk. Each phase builds upon the previous one, ensuring steady progress toward the unified, optimized system.