"""
Main Pipeline Factory - Central component creation and orchestration.

This factory implements the Abstract Factory pattern to create and configure
all pipeline components with proper dependency injection for TDD London School testing.
"""

from typing import Any, Dict, List, Optional, Type
import asyncio
import logging
from dataclasses import dataclass

from ..interfaces import (
    IFactory, IPipelineCoordinator, IStageCoordinator, IStage,
    IMemoryManager, IConfigurationManager, IProcessor,
    PipelineConfig, StageConfig
)
from ..interfaces.processors import (
    IRawInputProcessor, IDocumentConverter, IPreprocessor,
    ILangExtractor, IConversationGenerator, IDatasetFinalizer
)
from .memory_manager import M3MaxMemoryManager
from .stage_coordinator import StageCoordinatorImpl
from ..config.pipeline_config import PipelineConfigurationManager


@dataclass
class FactoryDependencies:
    """Dependencies required by pipeline factory."""
    memory_manager: IMemoryManager
    config_manager: IConfigurationManager
    logger: logging.Logger


class ComponentRegistry:
    """Registry for component types and their implementations."""
    
    def __init__(self):
        self._stage_types: Dict[str, Type[IStage]] = {}
        self._processor_types: Dict[str, Type[IProcessor]] = {}
        self._coordinator_types: Dict[str, Type[IStageCoordinator]] = {}
    
    def register_stage(self, stage_type: str, implementation: Type[IStage]) -> None:
        """Register stage implementation."""
        self._stage_types[stage_type] = implementation
    
    def register_processor(self, processor_type: str, implementation: Type[IProcessor]) -> None:
        """Register processor implementation."""
        self._processor_types[processor_type] = implementation
    
    def register_coordinator(self, coordinator_type: str, implementation: Type[IStageCoordinator]) -> None:
        """Register coordinator implementation."""
        self._coordinator_types[coordinator_type] = implementation
    
    def get_stage_type(self, stage_type: str) -> Optional[Type[IStage]]:
        """Get stage implementation by type."""
        return self._stage_types.get(stage_type)
    
    def get_processor_type(self, processor_type: str) -> Optional[Type[IProcessor]]:
        """Get processor implementation by type."""
        return self._processor_types.get(processor_type)
    
    def get_coordinator_type(self, coordinator_type: str) -> Optional[Type[IStageCoordinator]]:
        """Get coordinator implementation by type."""
        return self._coordinator_types.get(coordinator_type)
    
    def list_available_stages(self) -> List[str]:
        """List all available stage types."""
        return list(self._stage_types.keys())
    
    def list_available_processors(self) -> List[str]:
        """List all available processor types."""
        return list(self._processor_types.keys())


class ProcessorFactory(IFactory[IProcessor]):
    """Factory for creating processors with proper dependency injection."""
    
    def __init__(self, dependencies: FactoryDependencies, registry: ComponentRegistry):
        self._dependencies = dependencies
        self._registry = registry
        self._logger = dependencies.logger
    
    def create(self, config: Dict[str, Any]) -> IProcessor:
        """Create processor instance based on configuration."""
        processor_type = config.get('type')
        if not processor_type:
            raise ValueError("Processor type must be specified in configuration")
        
        implementation_class = self._registry.get_processor_type(processor_type)
        if not implementation_class:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        # Create processor with dependency injection
        processor = implementation_class(
            config=config,
            memory_manager=self._dependencies.memory_manager,
            config_manager=self._dependencies.config_manager,
            logger=self._logger.getChild(processor_type)
        )
        
        self._logger.info(f"Created processor: {processor_type} with ID: {processor.processor_id}")
        return processor
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported processor types."""
        return self._registry.list_available_processors()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate processor configuration."""
        required_fields = ['type']
        return all(field in config for field in required_fields)


class StageFactory(IFactory[IStage]):
    """Factory for creating pipeline stages."""
    
    def __init__(self, dependencies: FactoryDependencies, registry: ComponentRegistry):
        self._dependencies = dependencies
        self._registry = registry
        self._processor_factory = ProcessorFactory(dependencies, registry)
        self._logger = dependencies.logger
    
    def create(self, config: Dict[str, Any]) -> IStage:
        """Create stage instance with configured processors."""
        stage_type = config.get('type')
        if not stage_type:
            raise ValueError("Stage type must be specified in configuration")
        
        implementation_class = self._registry.get_stage_type(stage_type)
        if not implementation_class:
            raise ValueError(f"Unknown stage type: {stage_type}")
        
        # Create processors for this stage
        processors = []
        processor_configs = config.get('processors', [])
        
        for processor_config in processor_configs:
            if self._processor_factory.validate_config(processor_config):
                processor = self._processor_factory.create(processor_config)
                processors.append(processor)
            else:
                self._logger.warning(f"Invalid processor config: {processor_config}")
        
        # Create stage with processors
        stage = implementation_class(
            stage_config=config,
            processors=processors,
            memory_manager=self._dependencies.memory_manager,
            config_manager=self._dependencies.config_manager,
            logger=self._logger.getChild(stage_type)
        )
        
        self._logger.info(f"Created stage: {stage_type} with {len(processors)} processors")
        return stage
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported stage types."""
        return self._registry.list_available_stages()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate stage configuration."""
        required_fields = ['type', 'stage_id']
        return all(field in config for field in required_fields)


class CoordinatorFactory(IFactory[IStageCoordinator]):
    """Factory for creating stage coordinators."""
    
    def __init__(self, dependencies: FactoryDependencies, registry: ComponentRegistry):
        self._dependencies = dependencies
        self._registry = registry
        self._logger = dependencies.logger
    
    def create(self, config: Dict[str, Any]) -> IStageCoordinator:
        """Create stage coordinator instance."""
        coordinator_type = config.get('coordinator_type', 'default')
        stage = config.get('stage')
        stage_config = StageConfig(**config.get('stage_config', {}))
        
        if not stage:
            raise ValueError("Stage instance must be provided for coordinator")
        
        implementation_class = self._registry.get_coordinator_type(coordinator_type)
        if not implementation_class:
            # Use default coordinator implementation
            implementation_class = StageCoordinatorImpl
        
        coordinator = implementation_class(
            stage=stage,
            config=stage_config,
            memory_manager=self._dependencies.memory_manager,
            logger=self._logger.getChild(f"coordinator-{stage.stage_id}")
        )
        
        self._logger.info(f"Created coordinator for stage: {stage.stage_id}")
        return coordinator
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported coordinator types."""
        return ['default'] + list(self._registry._coordinator_types.keys())
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate coordinator configuration."""
        return 'stage' in config


class PipelineFactory(IFactory[IPipelineCoordinator]):
    """Main factory for creating complete pipeline coordinators."""
    
    def __init__(self, 
                 memory_manager: Optional[IMemoryManager] = None,
                 config_manager: Optional[IConfigurationManager] = None,
                 logger: Optional[logging.Logger] = None):
        
        # Initialize default dependencies if not provided
        self._logger = logger or logging.getLogger(__name__)
        self._memory_manager = memory_manager or M3MaxMemoryManager()
        self._config_manager = config_manager or PipelineConfigurationManager()
        
        self._dependencies = FactoryDependencies(
            memory_manager=self._memory_manager,
            config_manager=self._config_manager,
            logger=self._logger
        )
        
        # Initialize component registry and factories
        self._registry = ComponentRegistry()
        self._stage_factory = StageFactory(self._dependencies, self._registry)
        self._coordinator_factory = CoordinatorFactory(self._dependencies, self._registry)
        
        # Register default components
        self._register_default_components()
    
    def _register_default_components(self) -> None:
        """Register default component implementations."""
        # TODO: Register actual implementations when they're created
        # This is where we'd register:
        # - RawInputProcessorImpl
        # - DocumentConverterImpl  
        # - PreprocessorImpl
        # - LangExtractorImpl
        # - ConversationGeneratorImpl
        # - DatasetFinalizerImpl
        # - Various stage implementations
        
        self._logger.info("Default components registered")
    
    async def create(self, config: Dict[str, Any]) -> IPipelineCoordinator:
        """Create complete pipeline coordinator from configuration."""
        pipeline_config = PipelineConfig(**config.get('pipeline', {}))
        stage_configs = config.get('stages', [])
        
        if not stage_configs:
            raise ValueError("At least one stage configuration must be provided")
        
        # Create stages
        stages = []
        coordinators = []
        
        for stage_config in stage_configs:
            if self._stage_factory.validate_config(stage_config):
                # Create stage
                stage = self._stage_factory.create(stage_config)
                stages.append(stage)
                
                # Create coordinator for stage
                coordinator_config = {
                    'stage': stage,
                    'stage_config': stage_config.get('coordinator', {}),
                    'coordinator_type': stage_config.get('coordinator_type', 'default')
                }
                coordinator = self._coordinator_factory.create(coordinator_config)
                coordinators.append(coordinator)
            else:
                self._logger.error(f"Invalid stage configuration: {stage_config}")
                raise ValueError(f"Invalid stage configuration for stage: {stage_config.get('stage_id', 'unknown')}")
        
        # Create pipeline coordinator (implementation needed)
        from .pipeline_coordinator import PipelineCoordinatorImpl
        
        pipeline_coordinator = PipelineCoordinatorImpl(
            stages=stages,
            coordinators=coordinators,
            config=pipeline_config,
            memory_manager=self._memory_manager,
            logger=self._logger.getChild('pipeline')
        )
        
        # Initialize pipeline
        await pipeline_coordinator.initialize()
        
        self._logger.info(f"Created pipeline with {len(stages)} stages")
        return pipeline_coordinator
    
    def register_stage_type(self, stage_type: str, implementation: Type[IStage]) -> None:
        """Register custom stage implementation."""
        self._registry.register_stage(stage_type, implementation)
        self._logger.info(f"Registered stage type: {stage_type}")
    
    def register_processor_type(self, processor_type: str, implementation: Type[IProcessor]) -> None:
        """Register custom processor implementation."""
        self._registry.register_processor(processor_type, implementation)
        self._logger.info(f"Registered processor type: {processor_type}")
    
    def register_coordinator_type(self, coordinator_type: str, implementation: Type[IStageCoordinator]) -> None:
        """Register custom coordinator implementation."""
        self._registry.register_coordinator(coordinator_type, implementation)
        self._logger.info(f"Registered coordinator type: {coordinator_type}")
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported pipeline types."""
        return ['standard', 'm3_max_optimized', 'high_throughput', 'quality_focused']
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate complete pipeline configuration."""
        required_sections = ['pipeline', 'stages']
        if not all(section in config for section in required_sections):
            return False
        
        # Validate each stage configuration
        for stage_config in config.get('stages', []):
            if not self._stage_factory.validate_config(stage_config):
                return False
        
        return True
    
    async def create_optimized_m3_max_pipeline(self, 
                                             input_path: str,
                                             output_path: str,
                                             quality_threshold: float = 0.742) -> IPipelineCoordinator:
        """Create M3 Max optimized pipeline with default configuration."""
        config = await self._create_m3_max_config(input_path, output_path, quality_threshold)
        return await self.create(config)
    
    async def _create_m3_max_config(self, 
                                   input_path: str, 
                                   output_path: str,
                                   quality_threshold: float) -> Dict[str, Any]:
        """Create M3 Max optimized configuration."""
        return {
            'pipeline': {
                'mode': 'adaptive',
                'max_parallel_stages': 6,
                'batch_size': 100,
                'memory_limit_gb': 100.0,
                'enable_checkpointing': True
            },
            'stages': [
                {
                    'type': 'raw_input',
                    'stage_id': 'stage_1_raw_input',
                    'processors': [
                        {'type': 'zip_extractor'},
                        {'type': 'file_detector'},
                        {'type': 'batch_organizer'}
                    ]
                },
                {
                    'type': 'document_conversion',
                    'stage_id': 'stage_2_conversion',
                    'processors': [
                        {'type': 'html_converter'},
                        {'type': 'pdf_converter'},
                        {'type': 'csv_converter'},
                        {'type': 'txt_converter'}
                    ]
                },
                {
                    'type': 'preprocessing',
                    'stage_id': 'stage_3_preprocessing',
                    'processors': [
                        {'type': 'legal_content_remover'},
                        {'type': 'image_processor'},
                        {'type': 'table_preserver'},
                        {'type': 'quality_assessor'}
                    ]
                },
                {
                    'type': 'langextract',
                    'stage_id': 'stage_4_langextract',
                    'processors': [
                        {'type': 'intelligent_chunker'},
                        {'type': 'category_extractor'},
                        {'type': 'model_selector'}
                    ]
                },
                {
                    'type': 'conversation_generation',
                    'stage_id': 'stage_5_conversation',
                    'processors': [
                        {'type': 'qa_generator'},
                        {'type': 'cmedit_integrator'},
                        {'type': 'quality_scorer'}
                    ]
                },
                {
                    'type': 'dataset_finalization',
                    'stage_id': 'stage_6_finalization',
                    'processors': [
                        {'type': 'multi_format_exporter'},
                        {'type': 'data_splitter'},
                        {'type': 'deduplicator'},
                        {'type': 'final_validator'}
                    ]
                }
            ]
        }