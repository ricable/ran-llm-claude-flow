"""
Processor-specific interfaces for the 6-stage pipeline.
Defines contracts for each stage's processing capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .base import IProcessor, ProcessingResult, ProcessingContext


class DocumentType(Enum):
    """Supported document types."""
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    TXT = "txt"
    ZIP = "zip"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"


class ExtractionCategory(Enum):
    """LangExtract extraction categories."""
    FEATURES = "features"
    PARAMETERS = "parameters"
    PROCEDURES = "procedures"
    TROUBLESHOOTING = "troubleshooting"
    SPECIFICATIONS = "specifications"
    EXAMPLES = "examples"


@dataclass
class DocumentMetadata:
    """Document metadata container."""
    file_path: Path
    file_size: int
    document_type: DocumentType
    encoding: Optional[str] = None
    language: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    checksum: Optional[str] = None
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metadata is None:
            self.custom_metadata = {}


@dataclass
class ConversionResult:
    """Result of document conversion."""
    converted_content: str
    metadata: DocumentMetadata
    conversion_quality: float  # 0.0 to 1.0
    extracted_images: List[Path] = None
    extracted_tables: List[Dict[str, Any]] = None
    conversion_warnings: List[str] = None
    
    def __post_init__(self):
        if self.extracted_images is None:
            self.extracted_images = []
        if self.extracted_tables is None:
            self.extracted_tables = []
        if self.conversion_warnings is None:
            self.conversion_warnings = []


@dataclass
class ExtractionResult:
    """Result of LangExtract processing."""
    extracted_content: Dict[ExtractionCategory, str]
    confidence_scores: Dict[ExtractionCategory, float]
    model_used: str
    processing_time: float
    chunk_count: int
    quality_score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IDocumentConverter(IProcessor[Any, ConversionResult]):
    """Interface for Stage 2: Document Conversion."""
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[DocumentType]:
        """List of supported input document formats."""
        pass
    
    @abstractmethod
    async def convert_html_to_markdown(
        self,
        html_content: str,
        metadata: DocumentMetadata,
        context: ProcessingContext
    ) -> ConversionResult:
        """Convert HTML to Markdown using Docling + BeautifulSoup."""
        pass
    
    @abstractmethod
    async def convert_pdf_to_markdown(
        self,
        pdf_path: Path,
        metadata: DocumentMetadata,
        context: ProcessingContext
    ) -> ConversionResult:
        """Convert PDF to Markdown with OCR and table extraction."""
        pass
    
    @abstractmethod
    async def convert_csv_to_structured(
        self,
        csv_path: Path,
        metadata: DocumentMetadata,
        context: ProcessingContext
    ) -> ConversionResult:
        """Convert CSV to structured data with format detection."""
        pass
    
    @abstractmethod
    async def convert_txt_to_markdown(
        self,
        txt_content: str,
        metadata: DocumentMetadata,
        context: ProcessingContext
    ) -> ConversionResult:
        """Convert TXT to preprocessed Markdown."""
        pass
    
    @abstractmethod
    async def extract_images(
        self,
        document_path: Path,
        output_dir: Path
    ) -> List[Path]:
        """Extract images from document."""
        pass


class IPreprocessor(IProcessor[ConversionResult, str]):
    """Interface for Stage 3: Intelligent Preprocessing."""
    
    @abstractmethod
    async def remove_legal_content(
        self,
        content: str,
        context: ProcessingContext
    ) -> str:
        """Remove legal and copyright content."""
        pass
    
    @abstractmethod
    async def extract_and_process_images(
        self,
        content: str,
        image_paths: List[Path],
        context: ProcessingContext
    ) -> str:
        """Extract and process embedded images."""
        pass
    
    @abstractmethod
    async def preserve_table_structure(
        self,
        content: str,
        tables: List[Dict[str, Any]],
        context: ProcessingContext
    ) -> str:
        """Preserve table structure in markdown."""
        pass
    
    @abstractmethod
    async def assess_content_quality(
        self,
        content: str,
        metadata: DocumentMetadata,
        context: ProcessingContext
    ) -> float:
        """Assess and return content quality score (0.0-1.0)."""
        pass
    
    @abstractmethod
    async def filter_by_quality(
        self,
        content: str,
        quality_threshold: float,
        context: ProcessingContext
    ) -> bool:
        """Filter content based on quality threshold."""
        pass


class ILangExtractor(IProcessor[str, ExtractionResult]):
    """Interface for Stage 4: LangExtract Processing."""
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported language models."""
        pass
    
    @abstractmethod
    async def intelligent_chunk_document(
        self,
        content: str,
        chunk_size: int,
        overlap: int,
        context: ProcessingContext
    ) -> List[str]:
        """Intelligently chunk document for processing."""
        pass
    
    @abstractmethod
    async def extract_categories(
        self,
        chunks: List[str],
        categories: List[ExtractionCategory],
        model_name: str,
        context: ProcessingContext
    ) -> ExtractionResult:
        """Extract content by categories using specified model."""
        pass
    
    @abstractmethod
    async def select_optimal_model(
        self,
        content_length: int,
        complexity_score: float,
        context: ProcessingContext
    ) -> str:
        """Select optimal model for processing (qwen3:4b vs qwen3:1.7b)."""
        pass
    
    @abstractmethod
    async def apply_circuit_breaker(
        self,
        model_name: str,
        timeout_seconds: int,
        context: ProcessingContext
    ) -> bool:
        """Apply circuit breaker protection for model calls."""
        pass


class IConversationGenerator(IProcessor[ExtractionResult, Dict[str, Any]]):
    """Interface for Stage 5: Conversation Generation."""
    
    @abstractmethod
    async def create_conversational_format(
        self,
        extraction_result: ExtractionResult,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Create conversational Q&A format from extracted content."""
        pass
    
    @abstractmethod
    async def integrate_cmedit_commands(
        self,
        conversation: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Integrate CMEDIT command examples."""
        pass
    
    @abstractmethod
    async def score_conversation_quality(
        self,
        conversation: Dict[str, Any],
        context: ProcessingContext
    ) -> float:
        """Score conversation quality (0.0-1.0)."""
        pass
    
    @abstractmethod
    async def validate_conversation_format(
        self,
        conversation: Dict[str, Any],
        context: ProcessingContext
    ) -> bool:
        """Validate conversation format compliance."""
        pass
    
    @abstractmethod
    async def enrich_with_metadata(
        self,
        conversation: Dict[str, Any],
        metadata: DocumentMetadata,
        extraction_metadata: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Enrich conversation with metadata."""
        pass


class IDatasetFinalizer(IProcessor[List[Dict[str, Any]], Dict[str, Any]]):
    """Interface for Stage 6: Dataset Finalization."""
    
    @property
    @abstractmethod
    def supported_output_formats(self) -> List[str]:
        """Supported output formats (JSONL, Parquet, CSV)."""
        pass
    
    @abstractmethod
    async def create_multi_format_output(
        self,
        conversations: List[Dict[str, Any]],
        output_dir: Path,
        formats: List[str],
        context: ProcessingContext
    ) -> Dict[str, Path]:
        """Create multi-format dataset outputs."""
        pass
    
    @abstractmethod
    async def split_train_val_test(
        self,
        conversations: List[Dict[str, Any]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        context: ProcessingContext
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Split dataset into train/validation/test sets."""
        pass
    
    @abstractmethod
    async def deduplicate_conversations(
        self,
        conversations: List[Dict[str, Any]],
        similarity_threshold: float,
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Remove duplicate conversations."""
        pass
    
    @abstractmethod
    async def apply_quality_filtering(
        self,
        conversations: List[Dict[str, Any]],
        quality_threshold: float,
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Filter conversations by quality score."""
        pass
    
    @abstractmethod
    async def generate_dataset_metrics(
        self,
        conversations: List[Dict[str, Any]],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Generate comprehensive dataset metrics."""
        pass
    
    @abstractmethod
    async def validate_final_dataset(
        self,
        dataset_path: Path,
        expected_format: str,
        context: ProcessingContext
    ) -> bool:
        """Validate final dataset format and integrity."""
        pass


class IRawInputProcessor(IProcessor[Any, List[DocumentMetadata]]):
    """Interface for Stage 1: Raw Input Processing."""
    
    @abstractmethod
    async def extract_zip_files(
        self,
        zip_path: Path,
        extraction_dir: Path,
        context: ProcessingContext
    ) -> List[Path]:
        """Extract and organize ZIP files."""
        pass
    
    @abstractmethod
    async def detect_file_types(
        self,
        file_paths: List[Path],
        context: ProcessingContext
    ) -> Dict[DocumentType, List[Path]]:
        """Detect and categorize file types."""
        pass
    
    @abstractmethod
    async def analyze_volume(
        self,
        file_paths: List[Path],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze processing volume and create batching strategy."""
        pass
    
    @abstractmethod
    async def preserve_metadata(
        self,
        file_paths: List[Path],
        context: ProcessingContext
    ) -> List[DocumentMetadata]:
        """Extract and preserve file metadata."""
        pass
    
    @abstractmethod
    async def create_processing_batches(
        self,
        metadata_list: List[DocumentMetadata],
        batch_size: int,
        context: ProcessingContext
    ) -> List[List[DocumentMetadata]]:
        """Create optimized processing batches."""
        pass