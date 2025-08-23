"""
Python RAN LLM Pipeline Architecture

A modular, factory-based pipeline system optimized for M3 Max hardware that transforms
the monolithic 2640+ line processor into a clean 6-stage processing architecture.

## Architecture Overview

The pipeline follows a 6-stage processing flow:
1. **Raw Input Processing**: ZIP extraction, file detection, batching
2. **Document Conversion**: HTML/PDF/CSV/TXT → Markdown conversion  
3. **Preprocessing**: Legal content removal, image processing, quality filtering
4. **LangExtract Processing**: 6-category content extraction with model selection
5. **Conversation Generation**: Conversational Q&A format with CMEDIT integration
6. **Dataset Finalization**: Multi-format output, deduplication, validation

## Key Design Principles

- **Factory Patterns**: Centralized component creation with dependency injection
- **Interface Segregation**: Clean contracts for TDD London School testing
- **M3 Max Optimization**: Unified memory management for 128GB Apple Silicon
- **Modular Architecture**: Transform monolith into testable components

## Quick Start

```python
from python_pipeline import PipelineFactory

# Create M3 Max optimized pipeline
factory = PipelineFactory()
pipeline = await factory.create_optimized_m3_max_pipeline(
    input_path="/path/to/data",
    output_path="/path/to/output",
    quality_threshold=0.742
)

# Execute pipeline
result = await pipeline.execute(input_data, context)
```

## Memory Management

The system uses intelligent memory pools for optimal M3 Max utilization:
- **Models Pool**: 40GB for Qwen3 model variants
- **Processing Pool**: 50GB for document processing
- **Cache Pool**: 30GB for intelligent caching
- **System Pool**: 8GB for overhead

## Performance Targets

- **Throughput**: 15-30 documents/hour (4-5x improvement)
- **Memory Efficiency**: 95% utilization without swapping
- **Model Switching**: <5 second latency
- **Quality Consistency**: >0.742 score target
"""

# Core components
from .core import (
    PipelineFactory,
    ComponentRegistry,
    M3MaxMemoryManager,
    M3MaxResourceAllocator,
    StageCoordinatorImpl,
    PipelineCoordinatorImpl
)

# Configuration management
from .config import (
    PipelineConfigurationManager,
    PipelineConfigurationSchema,
    StageConfigImpl
)

# Interfaces (for type hints and testing)
from .interfaces import (
    IPipelineCoordinator,
    IStageCoordinator, 
    IMemoryManager,
    IConfigurationManager,
    IStage,
    IProcessor,
    ProcessingResult,
    ProcessingContext,
    ProcessingStatus
)

__version__ = "1.0.0"
__author__ = "Python Architecture Specialist"
__description__ = "Modular Python pipeline architecture for RAN LLM processing"

__all__ = [
    # Core factory and orchestration
    'PipelineFactory',
    'ComponentRegistry',
    
    # Memory management
    'M3MaxMemoryManager',
    'M3MaxResourceAllocator',
    
    # Coordination
    'StageCoordinatorImpl',
    'PipelineCoordinatorImpl',
    
    # Configuration
    'PipelineConfigurationManager',
    'PipelineConfigurationSchema',
    'StageConfigImpl',
    
    # Interfaces
    'IPipelineCoordinator',
    'IStageCoordinator',
    'IMemoryManager', 
    'IConfigurationManager',
    'IStage',
    'IProcessor',
    
    # Common types
    'ProcessingResult',
    'ProcessingContext',
    'ProcessingStatus',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]


def create_m3_max_pipeline(input_path: str, output_path: str, **kwargs):
    """
    Convenience function to create M3 Max optimized pipeline.
    
    Args:
        input_path: Path to input data
        output_path: Path for output files
        **kwargs: Additional configuration options
        
    Returns:
        Configured pipeline ready for execution
    """
    import asyncio
    
    async def _create():
        factory = PipelineFactory()
        return await factory.create_optimized_m3_max_pipeline(
            input_path=input_path,
            output_path=output_path,
            **kwargs
        )
    
    return asyncio.run(_create())


# Performance optimization hints for M3 Max
OPTIMIZATION_HINTS = {
    'memory': {
        'unified_memory_gb': 128,
        'recommended_pools': {
            'models': 40,
            'processing': 50, 
            'cache': 30,
            'system': 8
        }
    },
    'compute': {
        'performance_cores': 8,
        'efficiency_cores': 4,
        'gpu_cores': 40,
        'neural_engine_tops': 15.8
    },
    'models': {
        'qwen3_1_7b': {'memory_gb': 8, 'throughput': '2000+ chunks/min'},
        'qwen3_7b': {'memory_gb': 25, 'throughput': '150-300 items/min'},
        'qwen3_30b': {'memory_gb': 40, 'throughput': '50-100 items/min'}
    }
}