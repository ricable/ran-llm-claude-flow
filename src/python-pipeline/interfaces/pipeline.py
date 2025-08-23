"""
Pipeline coordination interfaces.
Defines contracts for pipeline orchestration and stage coordination.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum

from .base import ProcessingResult, ProcessingContext, IStage


class PipelineMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    STREAMING = "streaming"


class StageExecutionMode(Enum):
    """Stage execution modes."""
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"
    BATCH = "batch"
    STREAM = "stream"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    mode: PipelineMode = PipelineMode.ADAPTIVE
    max_parallel_stages: int = 4
    batch_size: int = 100
    timeout_seconds: int = 3600
    memory_limit_gb: float = 100.0
    enable_checkpointing: bool = True
    checkpoint_interval: int = 1000
    retry_attempts: int = 3
    error_handling: str = "stop_on_error"  # "stop_on_error", "skip_errors", "log_and_continue"


@dataclass
class StageConfig:
    """Configuration for individual stage execution."""
    execution_mode: StageExecutionMode = StageExecutionMode.BLOCKING
    max_workers: int = 8
    batch_size: int = 50
    memory_limit_gb: float = 20.0
    timeout_seconds: int = 600
    retry_attempts: int = 2


@dataclass 
class PipelineMetrics:
    """Pipeline execution metrics."""
    total_processed: int = 0
    successful_processed: int = 0
    failed_processed: int = 0
    skipped_processed: int = 0
    processing_rate: float = 0.0
    memory_usage_gb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    stage_metrics: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.stage_metrics is None:
            self.stage_metrics = {}


class IPipelineCoordinator(ABC):
    """Interface for overall pipeline coordination."""
    
    @property
    @abstractmethod
    def pipeline_id(self) -> str:
        """Unique identifier for this pipeline."""
        pass
    
    @property
    @abstractmethod
    def stages(self) -> List[IStage]:
        """List of stages in execution order."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> PipelineConfig:
        """Pipeline configuration."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute the complete pipeline."""
        pass
    
    @abstractmethod
    async def execute_streaming(
        self, 
        input_stream: AsyncIterator[Any], 
        context: ProcessingContext
    ) -> AsyncIterator[ProcessingResult]:
        """Execute pipeline in streaming mode."""
        pass
    
    @abstractmethod
    async def execute_batch(
        self, 
        input_batch: List[Any], 
        context: ProcessingContext
    ) -> List[ProcessingResult]:
        """Execute pipeline in batch mode."""
        pass
    
    @abstractmethod
    async def validate_pipeline(self) -> bool:
        """Validate pipeline configuration and stage compatibility."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        pass
    
    @abstractmethod
    async def create_checkpoint(self, checkpoint_id: str) -> bool:
        """Create pipeline state checkpoint."""
        pass
    
    @abstractmethod
    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore pipeline from checkpoint."""
        pass


class IStageCoordinator(ABC):
    """Interface for individual stage coordination."""
    
    @property
    @abstractmethod
    def stage(self) -> IStage:
        """Associated stage instance."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> StageConfig:
        """Stage configuration."""
        pass
    
    @abstractmethod
    async def execute_stage(
        self, 
        input_data: Any, 
        context: ProcessingContext
    ) -> ProcessingResult:
        """Execute stage with coordination logic."""
        pass
    
    @abstractmethod
    async def execute_batch_stage(
        self, 
        input_batch: List[Any], 
        context: ProcessingContext
    ) -> List[ProcessingResult]:
        """Execute stage in batch mode."""
        pass
    
    @abstractmethod
    async def monitor_execution(self) -> Dict[str, Any]:
        """Monitor stage execution status."""
        pass
    
    @abstractmethod
    async def handle_stage_failure(
        self, 
        error: Exception, 
        context: ProcessingContext
    ) -> ProcessingResult:
        """Handle stage execution failures."""
        pass
    
    @abstractmethod
    async def cleanup_stage(self, context: ProcessingContext) -> None:
        """Cleanup stage resources."""
        pass


class IPipelineOrchestrator(ABC):
    """High-level pipeline orchestration interface."""
    
    @abstractmethod
    async def create_pipeline(
        self, 
        pipeline_config: PipelineConfig,
        stage_configs: Dict[str, StageConfig]
    ) -> IPipelineCoordinator:
        """Create configured pipeline instance."""
        pass
    
    @abstractmethod
    async def register_stage(self, stage: IStage, config: StageConfig) -> None:
        """Register stage with orchestrator."""
        pass
    
    @abstractmethod
    async def get_available_stages(self) -> List[str]:
        """Get list of available stage types."""
        pass
    
    @abstractmethod
    async def optimize_pipeline(
        self, 
        coordinator: IPipelineCoordinator,
        historical_metrics: List[PipelineMetrics]
    ) -> PipelineConfig:
        """Optimize pipeline configuration based on metrics."""
        pass