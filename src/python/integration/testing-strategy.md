# Testing and Validation Strategy

## Executive Summary

This document outlines the comprehensive testing strategy to ensure the unified RAG-LLM pipeline maintains 100% feature parity while improving performance and maintainability. The strategy covers unit testing, integration testing, performance validation, and production readiness verification.

## Testing Philosophy

### Core Principles
1. **Quality Preservation** - Zero regression in output quality or functionality
2. **Performance Validation** - Verify M3 Max optimizations deliver expected improvements
3. **Behavioral Compatibility** - Maintain existing API contracts and behaviors
4. **Risk Mitigation** - Comprehensive coverage of failure scenarios
5. **Continuous Validation** - Testing integrated into development workflow

### Testing Pyramid
```
                    E2E Tests (5%)
                ─────────────────────
           Integration Tests (20%)
        ───────────────────────────────
       Unit Tests (75%)
    ─────────────────────────────────────
```

## Testing Architecture

### Test Structure
```
tests/
├── unit/                        # Unit tests (75% of test suite)
│   ├── core/                   # Core component tests
│   ├── llm/                    # LLM client tests
│   ├── processing/             # Processing strategy tests
│   ├── quality/                # Quality assessment tests
│   └── performance/            # Performance component tests
├── integration/                # Integration tests (20% of test suite)
│   ├── pipeline/               # End-to-end pipeline tests
│   ├── cli/                    # CLI integration tests
│   ├── configuration/          # Configuration loading tests
│   └── compatibility/          # Backwards compatibility tests
├── e2e/                        # End-to-end tests (5% of test suite)
│   ├── scenarios/              # Real-world usage scenarios
│   ├── performance/            # Performance validation
│   └── production/             # Production readiness tests
├── fixtures/                   # Test data and fixtures
│   ├── documents/              # Sample documents for testing
│   ├── configurations/         # Test configurations
│   └── expected_outputs/       # Expected results for validation
└── utils/                      # Testing utilities
    ├── helpers.py              # Test helper functions
    ├── comparisons.py          # Result comparison utilities
    └── generators.py           # Test data generators
```

## Unit Testing Strategy

### 1. Core Components Testing

#### Configuration System Tests
```python
# tests/unit/core/test_configuration_manager.py
class TestConfigurationManager:
    def test_load_base_configuration(self):
        """Test loading base configuration"""
        config_manager = ConfigurationManager()
        config = config_manager.load_profile("base")
        
        assert config.system.memory_limit_gb == 128
        assert config.llm.provider == "LMStudio"
        assert len(config.llm.profiles) == 3

    def test_profile_inheritance(self):
        """Test configuration profile inheritance"""
        config_manager = ConfigurationManager()
        dev_config = config_manager.load_profile("development")
        
        # Should inherit from base but override specific values
        assert dev_config.processing.max_documents == 10  # Override
        assert dev_config.system.memory_limit_gb == 128    # Inherited

    def test_configuration_validation(self):
        """Test configuration validation"""
        invalid_config = {"llm": {"provider": "InvalidProvider"}}
        
        with pytest.raises(ValidationError):
            ConfigurationManager()._validate_config(invalid_config)
```

#### LLM Client Tests
```python
# tests/unit/llm/test_unified_client.py
class TestUnifiedLLMClient:
    @pytest.fixture
    def mock_llm_config(self):
        return LLMConfig(
            provider="LMStudio",
            base_url="http://localhost:1234",
            profiles={
                "thinking": ProfileConfig(model_name="qwen3-thinking", timeout=1800),
                "fast": ProfileConfig(model_name="qwen3-fast", timeout=300)
            }
        )

    @pytest.mark.asyncio
    async def test_generate_with_thinking_profile(self, mock_llm_config):
        """Test generation with thinking profile"""
        client = UnifiedLLMClient(mock_llm_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.generate("Test prompt", profile="thinking")
            
            assert result.content == "Test response"
            assert mock_post.called
            # Verify thinking profile parameters were used
            call_args = mock_post.call_args
            assert call_args[1]['json']['model'] == "qwen3-thinking"

    def test_profile_selection(self, mock_llm_config):
        """Test automatic profile selection"""
        client = UnifiedLLMClient(mock_llm_config)
        
        # Thinking-intensive task
        task_context = TaskContext(complexity="high", requires_reasoning=True)
        profile = client.profile_manager.select_profile(task_context)
        assert profile == "thinking"
        
        # Simple extraction task
        task_context = TaskContext(complexity="low", task_type="extraction")
        profile = client.profile_manager.select_profile(task_context)
        assert profile == "fast"
```

### 2. Processing Strategy Tests

#### Quality Assessment Tests
```python
# tests/unit/processing/test_unified_quality_assessor.py
class TestUnifiedQualityAssessor:
    @pytest.fixture
    def sample_technical_content(self):
        return """
        The eNodeB supports MIMO configurations with up to 4x4 antenna arrays.
        Key parameters include:
        - CellReselectionPriorityforEUTRA: 7
        - SIntraSearchThreshold: 24
        - pmCellDowntimeAuto: 150
        """

    @pytest.fixture
    def sample_low_quality_content(self):
        return "Short text with no technical content."

    @pytest.mark.asyncio
    async def test_technical_content_assessment(self, sample_technical_content):
        """Test quality assessment of technical content"""
        assessor = UnifiedQualityAssessor(QualityConfig())
        
        score = await assessor.assess(sample_technical_content)
        
        assert score.overall >= 7.0  # High quality expected
        assert score.technical >= 0.8  # High technical content
        assert score.complexity >= 0.6  # Good complexity
        
    @pytest.mark.asyncio
    async def test_low_quality_content_assessment(self, sample_low_quality_content):
        """Test quality assessment of low-quality content"""
        assessor = UnifiedQualityAssessor(QualityConfig())
        
        score = await assessor.assess(sample_low_quality_content)
        
        assert score.overall < 4.0  # Low quality expected
        assert score.technical < 0.2  # Low technical content

    @pytest.mark.asyncio
    async def test_batch_assessment_performance(self):
        """Test batch assessment performance"""
        assessor = UnifiedQualityAssessor(QualityConfig())
        contents = [f"Sample content {i} with technical terms like LTE and MIMO" 
                   for i in range(100)]
        
        start_time = time.time()
        scores = await assessor.batch_assess(contents)
        end_time = time.time()
        
        assert len(scores) == 100
        assert end_time - start_time < 5.0  # Should complete in under 5 seconds
```

#### Processing Strategy Tests
```python
# tests/unit/processing/test_langextract_strategy.py
class TestLangExtractStrategy:
    @pytest.fixture
    def mock_dependencies(self):
        llm_client = Mock(spec=UnifiedLLMClient)
        quality_assessor = Mock(spec=UnifiedQualityAssessor)
        return llm_client, quality_assessor

    @pytest.fixture
    def sample_document(self):
        return Document(
            id="test-doc-1",
            content="Sample Ericsson documentation with technical parameters...",
            metadata=DocumentMetadata(category="feature", source="html"),
            source_path=Path("test.md")
        )

    @pytest.mark.asyncio
    async def test_document_processing(self, mock_dependencies, sample_document):
        """Test complete document processing"""
        llm_client, quality_assessor = mock_dependencies
        strategy = LangExtractStrategy(llm_client, quality_assessor)
        
        # Mock LLM responses
        llm_client.generate.return_value = LLMResponse(
            content='{"qa_pairs": [{"question": "What is LTE?", "answer": "Long Term Evolution"}]}'
        )
        
        # Mock quality assessment
        quality_assessor.batch_assess.return_value = [
            QualityScore(overall=8.5, technical=0.9, complexity=0.7)
        ]
        
        result = await strategy.process(sample_document)
        
        assert result.document_id == "test-doc-1"
        assert len(result.qa_pairs) > 0
        assert len(result.quality_scores) > 0
        assert result.processing_metadata.strategy == "langextract"

    def test_resource_estimation(self, mock_dependencies, sample_document):
        """Test processing resource estimation"""
        llm_client, quality_assessor = mock_dependencies
        strategy = LangExtractStrategy(llm_client, quality_assessor)
        
        workload = Workload(documents=[sample_document])
        estimate = strategy.estimate_resources(workload)
        
        assert estimate.estimated_time > timedelta(0)
        assert estimate.memory_required_mb > 0
        assert estimate.cpu_cores_recommended > 0
```

## Integration Testing Strategy

### 1. Pipeline Integration Tests

#### End-to-End Pipeline Tests
```python
# tests/integration/pipeline/test_unified_pipeline.py
class TestUnifiedProcessingPipeline:
    @pytest.fixture
    async def pipeline(self):
        """Create a test pipeline with real configuration"""
        config = ConfigurationManager().load_profile("testing")
        pipeline = UnifiedProcessingPipeline(config)
        await pipeline.initialize()
        return pipeline

    @pytest.mark.asyncio
    async def test_single_document_processing(self, pipeline):
        """Test processing a single document end-to-end"""
        document = Document(
            id="integration-test-1",
            content=self._load_test_content("ericsson_feature_doc.md"),
            metadata=DocumentMetadata(category="feature"),
            source_path=Path("test.md")
        )
        
        result = await pipeline.process_single(document)
        
        assert result.success
        assert len(result.qa_pairs) >= 5  # Minimum expected QA pairs
        assert all(qa.quality_score >= 3.8 for qa in result.qa_pairs)
        assert result.processing_time < timedelta(minutes=10)

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, pipeline):
        """Test batch processing performance"""
        documents = [
            self._create_test_document(f"doc-{i}")
            for i in range(10)
        ]
        
        start_time = time.time()
        results = await pipeline.process_documents(documents)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert len(results.results) == 10
        assert all(result.success for result in results.results)
        assert processing_time < 300  # Should complete in under 5 minutes
        
        # Verify M3 Max optimization effectiveness
        throughput = len(documents) / processing_time
        assert throughput > 0.5  # At least 0.5 docs/second

    @pytest.mark.asyncio
    async def test_adaptive_scaling(self, pipeline):
        """Test adaptive scaling under load"""
        # Start with small batch
        small_batch = [self._create_test_document(f"small-{i}") for i in range(3)]
        await pipeline.process_documents(small_batch)
        
        initial_workers = pipeline.adaptive_manager.current_workers
        
        # Process larger batch to trigger scaling
        large_batch = [self._create_test_document(f"large-{i}") for i in range(20)]
        await pipeline.process_documents(large_batch)
        
        scaled_workers = pipeline.adaptive_manager.current_workers
        
        # Should have scaled up
        assert scaled_workers > initial_workers
        assert pipeline.adaptive_manager.success_rate > 95.0
```

### 2. Configuration Integration Tests

#### Configuration Loading Tests
```python
# tests/integration/configuration/test_configuration_loading.py
class TestConfigurationIntegration:
    def test_all_profiles_load_successfully(self):
        """Test that all configuration profiles load without errors"""
        config_manager = ConfigurationManager()
        
        profiles = ["base", "development", "production", "testing"]
        for profile in profiles:
            config = config_manager.load_profile(profile)
            
            # Validate core configuration structure
            assert hasattr(config, 'system')
            assert hasattr(config, 'llm')
            assert hasattr(config, 'processing')
            assert hasattr(config, 'quality')
            
            # Validate configuration values are reasonable
            assert config.system.memory_limit_gb > 0
            assert config.processing.workers > 0
            assert config.llm.base_url.startswith('http')

    def test_legacy_config_migration(self):
        """Test migration from legacy config.yaml"""
        config_manager = ConfigurationManager()
        
        # Load legacy configuration
        legacy_config_path = Path("config/legacy/config.yaml")
        if legacy_config_path.exists():
            migrated_config = config_manager.migrate_legacy_config(legacy_config_path)
            
            # Verify key values were preserved
            assert migrated_config.processing.max_documents >= 0
            assert migrated_config.llm.profiles['thinking'].timeout_seconds == 1200
            assert migrated_config.quality.min_score == 3.8
```

### 3. CLI Integration Tests

#### CLI Command Tests
```python
# tests/integration/cli/test_cli_commands.py
class TestCLIIntegration:
    def test_process_command_basic(self, tmp_path):
        """Test basic process command functionality"""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test input files
        (input_dir / "test_doc.md").write_text("Sample Ericsson content with LTE parameters")
        
        # Run CLI command
        runner = CliRunner()
        result = runner.invoke(cli, [
            'process',
            str(input_dir),
            str(output_dir),
            '--max-documents', '1',
            '--config', 'testing'
        ])
        
        assert result.exit_code == 0
        assert (output_dir / "ericsson_dataset.jsonl").exists()
        
        # Verify output quality
        output_content = (output_dir / "ericsson_dataset.jsonl").read_text()
        qa_pairs = [json.loads(line) for line in output_content.strip().split('\n')]
        assert len(qa_pairs) > 0
        assert all('question' in qa and 'answer' in qa for qa in qa_pairs)

    def test_cmedit_command(self, tmp_path):
        """Test CMEDIT workflow generation command"""
        csv_file = tmp_path / "parameters.csv"
        csv_file.write_text("""
MO Class,Attribute,Description,Type
EUtranCellFDD,dlChannelBandwidth,Downlink channel bandwidth,Integer
EUtranCellFDD,ulChannelBandwidth,Uplink channel bandwidth,Integer
        """.strip())
        
        output_dir = tmp_path / "cmedit_output"
        output_dir.mkdir()
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'cmedit',
            str(csv_file),
            '--output', str(output_dir),
            '--config', 'testing'
        ])
        
        assert result.exit_code == 0
        assert any(f.name.endswith('.jsonl') for f in output_dir.iterdir())
```

## Performance Validation Strategy

### 1. M3 Max Optimization Validation

#### Performance Benchmark Tests
```python
# tests/e2e/performance/test_m3_max_performance.py
class TestM3MaxPerformance:
    @pytest.mark.performance
    def test_memory_optimization(self):
        """Test unified memory optimization effectiveness"""
        config = ConfigurationManager().load_profile("production")
        pipeline = UnifiedProcessingPipeline(config)
        
        # Monitor memory usage during processing
        memory_monitor = MemoryMonitor()
        
        documents = [self._create_large_document() for _ in range(50)]
        
        with memory_monitor:
            asyncio.run(pipeline.process_documents(documents))
        
        # Verify memory efficiency
        peak_memory_gb = memory_monitor.peak_usage_gb
        assert peak_memory_gb < 96  # Should use < 75% of available memory
        
        # Verify no memory leaks
        assert memory_monitor.final_usage_gb < memory_monitor.initial_usage_gb + 5

    @pytest.mark.performance  
    def test_cpu_optimization(self):
        """Test CPU core utilization optimization"""
        config = ConfigurationManager().load_profile("production")
        pipeline = UnifiedProcessingPipeline(config)
        
        cpu_monitor = CPUMonitor()
        
        documents = [self._create_test_document() for _ in range(100)]
        
        with cpu_monitor:
            asyncio.run(pipeline.process_documents(documents))
        
        # Verify efficient CPU utilization
        assert cpu_monitor.average_cpu_percent > 70  # Good utilization
        assert cpu_monitor.max_cpu_percent < 95     # Not overloaded
        
        # Verify performance core preference
        performance_core_usage = cpu_monitor.performance_core_usage_percent
        efficiency_core_usage = cpu_monitor.efficiency_core_usage_percent
        assert performance_core_usage > efficiency_core_usage

    @pytest.mark.performance
    def test_mlx_acceleration(self):
        """Test MLX acceleration effectiveness"""
        if not self._is_mlx_available():
            pytest.skip("MLX not available")
            
        # Test with MLX enabled
        config_mlx = ConfigurationManager().load_profile("production")
        config_mlx.performance.enable_mlx = True
        
        # Test with MLX disabled
        config_no_mlx = ConfigurationManager().load_profile("production") 
        config_no_mlx.performance.enable_mlx = False
        
        documents = [self._create_test_document() for _ in range(10)]
        
        # Measure performance with MLX
        start_time = time.time()
        pipeline_mlx = UnifiedProcessingPipeline(config_mlx)
        asyncio.run(pipeline_mlx.process_documents(documents))
        mlx_time = time.time() - start_time
        
        # Measure performance without MLX
        start_time = time.time()
        pipeline_no_mlx = UnifiedProcessingPipeline(config_no_mlx)
        asyncio.run(pipeline_no_mlx.process_documents(documents))
        no_mlx_time = time.time() - start_time
        
        # MLX should provide significant speedup
        speedup_ratio = no_mlx_time / mlx_time
        assert speedup_ratio > 1.2  # At least 20% improvement
```

### 2. Throughput and Latency Tests

#### Throughput Validation
```python
# tests/e2e/performance/test_throughput.py
class TestThroughputValidation:
    @pytest.mark.performance
    @pytest.mark.parametrize("document_count", [10, 50, 100, 500])
    def test_throughput_scaling(self, document_count):
        """Test throughput scaling with document count"""
        config = ConfigurationManager().load_profile("production")
        pipeline = UnifiedProcessingPipeline(config)
        
        documents = [self._create_test_document(f"doc-{i}") 
                    for i in range(document_count)]
        
        start_time = time.time()
        results = asyncio.run(pipeline.process_documents(documents))
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = document_count / processing_time
        
        # Verify throughput meets expectations
        if document_count <= 50:
            assert throughput >= 1.0  # At least 1 doc/second for small batches
        elif document_count <= 100:
            assert throughput >= 0.8  # Slight decrease for medium batches
        else:
            assert throughput >= 0.5  # Minimum acceptable for large batches
        
        # Verify quality is maintained
        assert all(result.success for result in results.results)
        avg_quality = sum(
            sum(qa.quality_score for qa in result.qa_pairs) / len(result.qa_pairs)
            for result in results.results
        ) / len(results.results)
        assert avg_quality >= 6.0
```

## Quality Assurance Testing

### 1. Output Quality Validation

#### Quality Regression Tests
```python
# tests/integration/quality/test_output_quality.py
class TestOutputQuality:
    @pytest.fixture
    def baseline_results(self):
        """Load baseline results from original system"""
        return self._load_baseline_results("tests/fixtures/baseline_outputs/")

    def test_qa_pair_quality_regression(self, baseline_results):
        """Test that QA pair quality matches or exceeds baseline"""
        config = ConfigurationManager().load_profile("testing")
        pipeline = UnifiedProcessingPipeline(config)
        
        # Process same documents as baseline
        test_documents = self._load_test_documents("tests/fixtures/test_documents/")
        results = asyncio.run(pipeline.process_documents(test_documents))
        
        # Compare quality metrics
        for i, result in enumerate(results.results):
            baseline_result = baseline_results[i]
            
            # Overall quality should be maintained or improved
            current_avg_quality = sum(qa.quality_score for qa in result.qa_pairs) / len(result.qa_pairs)
            baseline_avg_quality = baseline_result.average_quality_score
            
            assert current_avg_quality >= baseline_avg_quality - 0.1  # Allow small variance
            
            # Diversity should be maintained
            current_diversity = self._calculate_diversity(result.qa_pairs)
            baseline_diversity = baseline_result.diversity_score
            
            assert current_diversity >= baseline_diversity - 0.05

    def test_technical_accuracy_preservation(self):
        """Test that technical accuracy is preserved"""
        technical_document = Document(
            id="technical-test",
            content=self._load_technical_content(),
            metadata=DocumentMetadata(category="technical"),
            source_path=Path("technical.md")
        )
        
        config = ConfigurationManager().load_profile("testing")
        pipeline = UnifiedProcessingPipeline(config)
        
        result = asyncio.run(pipeline.process_single(technical_document))
        
        # Verify technical terms are preserved accurately
        technical_terms = self._extract_technical_terms(technical_document.content)
        generated_content = " ".join(qa.answer for qa in result.qa_pairs)
        
        preserved_terms = sum(1 for term in technical_terms 
                            if term in generated_content)
        preservation_rate = preserved_terms / len(technical_terms)
        
        assert preservation_rate >= 0.9  # 90% of technical terms should be preserved
```

### 2. Compatibility Testing

#### Backwards Compatibility Tests
```python
# tests/integration/compatibility/test_backwards_compatibility.py
class TestBackwardsCompatibility:
    def test_existing_configurations_work(self):
        """Test that existing configurations continue to work"""
        legacy_configs = [
            "tests/fixtures/legacy_configs/config_v1.yaml",
            "tests/fixtures/legacy_configs/config_v2.yaml"
        ]
        
        for legacy_config_path in legacy_configs:
            if Path(legacy_config_path).exists():
                config_manager = ConfigurationManager()
                migrated_config = config_manager.migrate_legacy_config(
                    Path(legacy_config_path)
                )
                
                # Should be able to create pipeline with migrated config
                pipeline = UnifiedProcessingPipeline(migrated_config)
                assert pipeline is not None
                
                # Basic functionality should work
                test_doc = self._create_simple_test_document()
                result = asyncio.run(pipeline.process_single(test_doc))
                assert result.success

    def test_api_compatibility(self):
        """Test that existing API contracts are maintained"""
        # Test that old import patterns still work (with deprecation warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # These imports should work but generate deprecation warnings
            from ericsson_dataset_pipeline.processors.langextract import process_document
            from ericsson_dataset_pipeline.quality import assess_quality
            
            assert len(w) >= 2  # Should have deprecation warnings
            assert all("deprecated" in str(warning.message).lower() for warning in w)
```

## Test Data Management

### 1. Test Fixtures

#### Document Fixtures
```python
# tests/fixtures/documents.py
class DocumentFixtures:
    @staticmethod
    def create_ericsson_feature_document() -> Document:
        """Create a representative Ericsson feature document"""
        content = """
        # Carrier Aggregation Feature
        
        Carrier Aggregation (CA) allows UEs to simultaneously use multiple 
        component carriers to increase bandwidth and improve performance.
        
        ## Key Parameters
        - CarrierAggregationFunction.caCapability: ENABLED
        - CarrierAggregationFunction.scellActDeactTimer: 1000
        - EUtranCellFDD.additionalSpectrumEmission: 1
        
        ## Performance Counters
        - pmCaScellActivationSucc: Number of successful SCell activations
        - pmCaScellDeactivationSucc: Number of successful SCell deactivations
        """
        
        return Document(
            id="feature-ca-001",
            content=content,
            metadata=DocumentMetadata(
                category="feature",
                technical_density=8.5,
                parameter_count=3,
                counter_count=2
            ),
            source_path=Path("features/carrier_aggregation.md")
        )

    @staticmethod
    def create_csv_parameter_document() -> Document:
        """Create a CSV parameter document"""
        content = """
MO Class,Attribute,Description,Type,Unit
EUtranCellFDD,dlChannelBandwidth,Downlink channel bandwidth,Integer,MHz
EUtranCellFDD,ulChannelBandwidth,Uplink channel bandwidth,Integer,MHz
CarrierAggregationFunction,caCapability,CA capability status,Enum,None
        """
        
        return Document(
            id="csv-params-001",
            content=content,
            metadata=DocumentMetadata(
                category="csv_parameters",
                parameter_count=3
            ),
            source_path=Path("parameters/eutrancell.csv")
        )
```

#### Expected Output Fixtures
```python
# tests/fixtures/expected_outputs.py
class ExpectedOutputs:
    @staticmethod
    def get_expected_qa_pairs_for_ca_feature():
        """Expected QA pairs for Carrier Aggregation feature document"""
        return [
            {
                "question": "What is Carrier Aggregation (CA)?",
                "answer": "Carrier Aggregation (CA) allows UEs to simultaneously use multiple component carriers to increase bandwidth and improve performance.",
                "quality_score": 8.5,
                "confidence": 0.95,
                "source": "feature-ca-001"
            },
            {
                "question": "What does the caCapability parameter control?",
                "answer": "The CarrierAggregationFunction.caCapability parameter controls the CA capability status and can be set to ENABLED or DISABLED.",
                "quality_score": 7.8,
                "confidence": 0.92,
                "source": "feature-ca-001"
            }
        ]
```

### 2. Test Data Generation

#### Synthetic Data Generation
```python
# tests/utils/generators.py
class TestDataGenerator:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.faker = Faker()
        Faker.seed(seed)

    def generate_documents(self, count: int, category: str = "feature") -> list[Document]:
        """Generate synthetic documents for testing"""
        documents = []
        
        for i in range(count):
            content = self._generate_technical_content(category)
            document = Document(
                id=f"generated-{category}-{i:03d}",
                content=content,
                metadata=DocumentMetadata(
                    category=category,
                    technical_density=self.random.uniform(3.0, 9.0),
                    parameter_count=self.random.randint(1, 20)
                ),
                source_path=Path(f"generated/{category}_{i}.md")
            )
            documents.append(document)
            
        return documents
    
    def _generate_technical_content(self, category: str) -> str:
        """Generate realistic technical content"""
        technical_terms = [
            "LTE", "5G", "eNodeB", "gNodeB", "MIMO", "Carrier Aggregation",
            "Handover", "ANR", "SON", "QoS", "RSRP", "RSRQ", "CQI"
        ]
        
        parameters = [
            f"{self.faker.word().title()}.{self.faker.word()}",
            f"pm{self.faker.word().title()}{self.faker.word().title()}"
        ]
        
        # Generate structured content with technical terms and parameters
        content_sections = []
        
        # Title
        content_sections.append(f"# {self.faker.catch_phrase()} Feature")
        
        # Overview
        content_sections.append(f"""
        ## Overview
        This feature implements {self.random.choice(technical_terms)} 
        functionality to improve {self.faker.word()} performance and efficiency.
        """)
        
        # Parameters section
        content_sections.append("## Key Parameters")
        for param in self.random.sample(parameters, k=min(3, len(parameters))):
            value = self.random.choice([
                str(self.random.randint(1, 1000)),
                self.random.choice(["ENABLED", "DISABLED", "AUTO"]),
                f"{self.random.uniform(0.1, 10.0):.1f}"
            ])
            content_sections.append(f"- {param}: {value}")
        
        return "\n".join(content_sections)
```

## Continuous Integration Testing

### 1. CI Pipeline Configuration

#### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, migration/*]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install UV
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: uv install --dev
    
    - name: Run unit tests
      run: uv run pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: macos-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install dependencies
      run: uv install --dev
    
    - name: Run integration tests
      run: uv run pytest tests/integration/ -v --timeout=300
    
    - name: Run CLI tests
      run: uv run pytest tests/integration/cli/ -v

  performance-tests:
    runs-on: macos-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main' || contains(github.event.pull_request.labels.*.name, 'performance')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.12"
    
    - name: Install dependencies
      run: uv install --dev
    
    - name: Run performance tests
      run: uv run pytest tests/e2e/performance/ -v -m performance
    
    - name: Generate performance report
      run: uv run python tests/utils/generate_performance_report.py
```

### 2. Quality Gates

#### Coverage Requirements
```python
# tests/conftest.py
import pytest

def pytest_configure(config):
    """Configure pytest with quality gates"""
    # Minimum coverage requirements
    config.addinivalue_line(
        "markers", 
        "coverage: marks tests that affect coverage requirements"
    )

def pytest_sessionfinish(session, exitstatus):
    """Enforce quality gates after test session"""
    if hasattr(session.config, '_coverage'):
        coverage = session.config._coverage
        total_coverage = coverage.report()
        
        # Enforce minimum coverage
        if total_coverage < 90.0:
            pytest.exit(f"Coverage {total_coverage:.1f}% below minimum 90%")
```

#### Performance Benchmarks
```python
# tests/utils/performance_benchmarks.py
class PerformanceBenchmarks:
    @staticmethod
    def enforce_performance_requirements(test_results: dict):
        """Enforce performance requirements"""
        requirements = {
            "throughput_docs_per_second": 0.5,
            "memory_usage_gb": 96,
            "response_time_p95_seconds": 30,
            "error_rate_percent": 2.0
        }
        
        failures = []
        for metric, threshold in requirements.items():
            actual = test_results.get(metric)
            if actual is None:
                failures.append(f"Missing metric: {metric}")
                continue
                
            if metric.endswith("_percent") or metric.endswith("_gb"):
                if actual > threshold:
                    failures.append(f"{metric}: {actual} > {threshold}")
            else:
                if actual < threshold:
                    failures.append(f"{metric}: {actual} < {threshold}")
        
        if failures:
            raise AssertionError(f"Performance requirements failed: {failures}")
```

## Test Execution Strategy

### 1. Local Development Testing

#### Pre-commit Hooks
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run unit tests
echo "Running unit tests..."
uv run pytest tests/unit/ --timeout=60 -q
if [ $? -ne 0 ]; then
    echo "Unit tests failed. Commit aborted."
    exit 1
fi

# Run code quality checks
echo "Running code quality checks..."
uv run black --check src/ tests/
uv run isort --check-only src/ tests/
uv run mypy src/

if [ $? -ne 0 ]; then
    echo "Code quality checks failed. Commit aborted."
    exit 1
fi

echo "All checks passed. Proceeding with commit."
```

#### Development Test Commands
```bash
# Quick unit test run
make test-unit

# Integration tests
make test-integration

# Full test suite
make test-all

# Performance validation
make test-performance

# Coverage report
make coverage
```

### 2. Release Testing

#### Release Validation Checklist
```python
# tests/release/test_release_validation.py
class TestReleaseValidation:
    def test_all_features_functional(self):
        """Comprehensive feature functionality test"""
        test_cases = [
            ("langextract", "Process Ericsson feature documents"),
            ("csv", "Process CSV parameter files"),
            ("3gpp", "Process 3GPP specification documents"),
            ("cmedit", "Generate CMEDIT workflows"),
            ("analysis", "Analyze dataset quality and diversity")
        ]
        
        for feature, description in test_cases:
            with self.subTest(feature=feature):
                self._test_feature_functionality(feature, description)
    
    def test_performance_requirements_met(self):
        """Verify all performance requirements are met"""
        performance_tests = [
            self._test_throughput_requirements,
            self._test_memory_requirements,
            self._test_latency_requirements,
            self._test_quality_requirements
        ]
        
        for test_func in performance_tests:
            test_func()
    
    def test_production_readiness(self):
        """Verify production readiness"""
        readiness_checks = [
            self._check_configuration_completeness,
            self._check_error_handling_robustness,
            self._check_monitoring_capabilities,
            self._check_documentation_completeness
        ]
        
        for check_func in readiness_checks:
            check_func()
```

This comprehensive testing strategy ensures that the unified RAG-LLM pipeline maintains quality, performance, and functionality throughout the migration process while providing confidence in the production-ready system.