"""
Base interfaces for the Python pipeline architecture.
Provides fundamental contracts for processors, stages, and factories.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')
U = TypeVar('U')


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed" 
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """Standard processing result container."""
    status: ProcessingStatus
    data: Any = None
    metadata: Dict[str, Any] = None
    errors: List[str] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []
        if self.metrics is None:
            self.metrics = {}


@dataclass
class ProcessingContext:
    """Context information for processing operations."""
    stage_id: str
    processor_id: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    config: Dict[str, Any] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class IProcessor(ABC, Generic[T, U]):
    """Base interface for all processors in the pipeline."""
    
    @property
    @abstractmethod
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        pass
    
    @property
    @abstractmethod
    def supported_input_types(self) -> List[str]:
        """List of supported input types/formats."""
        pass
    
    @property
    @abstractmethod
    def output_type(self) -> str:
        """Output type/format produced by this processor."""
        pass
    
    @abstractmethod
    async def process(self, input_data: T, context: ProcessingContext) -> ProcessingResult:
        """Process input data and return result."""
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: T) -> bool:
        """Validate input data compatibility."""
        pass
    
    @abstractmethod
    async def cleanup(self, context: ProcessingContext) -> None:
        """Cleanup resources after processing."""
        pass


class IStage(ABC):
    """Interface for pipeline stages."""
    
    @property
    @abstractmethod
    def stage_id(self) -> str:
        """Unique identifier for this stage."""
        pass
    
    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Human-readable name for this stage."""
        pass
    
    @property
    @abstractmethod
    def processors(self) -> List[IProcessor]:
        """List of processors in this stage."""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute the stage processing."""
        pass
    
    @abstractmethod
    async def can_process(self, input_data: Any) -> bool:
        """Check if stage can process the input data."""
        pass
    
    @abstractmethod
    async def prepare(self, context: ProcessingContext) -> None:
        """Prepare stage for execution."""
        pass
    
    @abstractmethod
    async def finalize(self, context: ProcessingContext) -> None:
        """Finalize stage execution."""
        pass


class IFactory(ABC, Generic[T]):
    """Base interface for factories."""
    
    @abstractmethod
    def create(self, config: Dict[str, Any]) -> T:
        """Create instance with configuration."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported creation types."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration for creation."""
        pass