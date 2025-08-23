#!/usr/bin/env python3
"""
Enhanced Dataset Generation Pipeline - Zip to Fine-Tuned Models
Complete pipeline from raw zip files to production-ready training datasets
"""

import asyncio
import logging
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import pandas as pd
import numpy as np

# Import our custom modules
from ..model_management.qwen3_variants import Qwen3ModelManager, ModelSize
from ..embeddings.sentence_transformer_manager import SentenceTransformerManager, EmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileType(Enum):
    """Supported file types for processing"""
    MARKDOWN = "md"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    TXT = "txt"
    PDF = "pdf"

class QualityLevel(Enum):
    """Quality levels for dataset generation"""
    BASIC = "basic"      # Fast processing, lower quality
    STANDARD = "standard" # Balanced quality and speed
    PREMIUM = "premium"   # High quality, slower processing
    ULTRA = "ultra"      # Maximum quality, slowest

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    files_processed: int = 0
    files_failed: int = 0
    conversations_generated: int = 0
    processing_time: float = 0.0
    quality_scores: List[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []
        if self.errors is None:
            self.errors = []

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Input configuration
    input_paths: List[str]
    output_dir: str
    
    # Processing configuration
    quality_level: QualityLevel = QualityLevel.STANDARD
    max_files_per_type: Optional[int] = None
    batch_size: int = 32
    workers: int = 8
    
    # Model configuration
    primary_model: str = "qwen3-7b-mlx"
    fallback_model: str = "qwen3-1.7b-mlx"
    embedding_model: EmbeddingModel = EmbeddingModel.BGE_BASE_EN
    
    # Quality configuration
    min_quality_score: float = 0.6
    deduplication_threshold: float = 0.85
    diversity_target: float = 0.7
    
    # Output configuration
    output_formats: List[str] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["jsonl", "parquet"]

class EnhancedDatasetPipeline:
    """Complete dataset generation pipeline from zip to fine-tuned models"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.model_manager = Qwen3ModelManager()
        self.embedding_manager = SentenceTransformerManager()
        self.executor = ThreadPoolExecutor(max_workers=config.workers)
        
        # Processing statistics
        self.stats = ProcessingStats()
        self.file_registry = {}  # Track processed files
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Enhanced Dataset Pipeline initialized")
        logger.info(f"   Output: {config.output_dir}")
        logger.info(f"   Quality: {config.quality_level.value}")
        logger.info(f"   Workers: {config.workers}")
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🔧 Initializing pipeline components...")
        
        # Load primary model
        success = await self.model_manager.load_model(self.config.primary_model)
        if not success:
            logger.warning(f"⚠️ Failed to load primary model, trying fallback...")
            success = await self.model_manager.load_model(self.config.fallback_model)
            if not success:
                raise RuntimeError("❌ Failed to load any models")
        
        # Load embedding model
        from ..embeddings.sentence_transformer_manager import EmbeddingConfig
        embed_config = EmbeddingConfig(
            model=self.config.embedding_model,
            batch_size=self.config.batch_size,
            max_seq_length=512
        )
        await self.embedding_manager.load_model(embed_config)
        
        logger.info("✅ Pipeline components initialized")
    
    async def process_zip_files(self) -> ProcessingStats:
        """Process all zip files in input paths"""
        logger.info("📦 Processing zip files...")
        start_time = time.time()
        
        all_files = []
        
        # Extract and catalog all files
        for input_path in self.config.input_paths:
            input_path = Path(input_path)
            
            if input_path.is_file() and input_path.suffix.lower() == '.zip':
                extracted_files = await self._extract_zip_file(input_path)
                all_files.extend(extracted_files)
            elif input_path.is_dir():
                # Process all zip files in directory
                for zip_file in input_path.glob("*.zip"):
                    extracted_files = await self._extract_zip_file(zip_file)
                    all_files.extend(extracted_files)
            else:
                logger.warning(f"⚠️ Skipping non-zip input: {input_path}")
        
        if not all_files:
            logger.error("❌ No files found to process")
            return self.stats
        
        # Organize files by type
        files_by_type = self._organize_files_by_type(all_files)
        logger.info(f"📊 Found {len(all_files)} total files across {len(files_by_type)} types")
        
        # Process each file type
        all_conversations = []
        
        for file_type, files in files_by_type.items():
            logger.info(f"🔄 Processing {len(files)} {file_type.value.upper()} files...")
            
            # Apply file limits if configured
            if self.config.max_files_per_type:
                files = files[:self.config.max_files_per_type]
            
            conversations = await self._process_files_by_type(file_type, files)
            all_conversations.extend(conversations)
        
        # Post-process conversations
        if all_conversations:
            final_conversations = await self._post_process_conversations(all_conversations)
            
            # Generate output in requested formats
            await self._generate_outputs(final_conversations)
        
        # Update final statistics
        self.stats.processing_time = time.time() - start_time
        logger.info(f"✅ Pipeline completed in {self.stats.processing_time:.2f}s")
        logger.info(f"   📈 Generated {self.stats.conversations_generated} conversations")
        logger.info(f"   📊 Average quality: {np.mean(self.stats.quality_scores):.3f}")
        
        return self.stats
    
    async def _extract_zip_file(self, zip_path: Path) -> List[Dict[str, Any]]:
        """Extract zip file and return file information"""
        logger.info(f"📦 Extracting {zip_path.name}...")
        
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create temporary extraction directory
                extract_dir = Path(tempfile.mkdtemp(prefix="dataset_extract_"))
                
                # Extract all files
                zip_ref.extractall(extract_dir)
                
                # Catalog extracted files
                for extracted_path in extract_dir.rglob("*"):
                    if extracted_path.is_file():
                        file_type = self._detect_file_type(extracted_path)
                        if file_type:
                            file_info = {
                                'path': extracted_path,
                                'original_zip': zip_path.name,
                                'relative_path': extracted_path.relative_to(extract_dir),
                                'file_type': file_type,
                                'size_bytes': extracted_path.stat().st_size,
                                'extract_dir': extract_dir  # Keep reference for cleanup
                            }
                            extracted_files.append(file_info)
                
        except Exception as e:
            logger.error(f"❌ Failed to extract {zip_path}: {str(e)}")
            self.stats.errors.append(f"Zip extraction failed: {zip_path} - {str(e)}")
        
        logger.info(f"   📁 Extracted {len(extracted_files)} files")
        return extracted_files
    
    def _detect_file_type(self, file_path: Path) -> Optional[FileType]:
        """Detect file type from extension and content"""
        suffix = file_path.suffix.lower()
        
        type_map = {
            '.md': FileType.MARKDOWN,
            '.html': FileType.HTML,
            '.htm': FileType.HTML,
            '.csv': FileType.CSV,
            '.json': FileType.JSON,
            '.xml': FileType.XML,
            '.txt': FileType.TXT,
            '.pdf': FileType.PDF
        }
        
        return type_map.get(suffix)
    
    def _organize_files_by_type(self, files: List[Dict[str, Any]]) -> Dict[FileType, List[Dict[str, Any]]]:
        """Organize files by type for processing"""
        files_by_type = {}
        
        for file_info in files:
            file_type = file_info['file_type']
            
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            
            files_by_type[file_type].append(file_info)
        
        # Sort files within each type by size (larger first for better batching)
        for file_type in files_by_type:
            files_by_type[file_type].sort(key=lambda x: x['size_bytes'], reverse=True)
        
        return files_by_type
    
    async def _process_files_by_type(self, file_type: FileType, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process files of a specific type"""
        conversations = []
        
        # Process files in batches
        batch_size = min(self.config.batch_size, len(files))
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            logger.info(f"   📋 Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            batch_conversations = await self._process_file_batch(file_type, batch_files)
            conversations.extend(batch_conversations)
            
            # Update statistics
            self.stats.files_processed += len(batch_files)
        
        return conversations
    
    async def _process_file_batch(self, file_type: FileType, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of files concurrently"""
        
        # Create tasks for concurrent processing
        tasks = []
        for file_info in files:
            task = asyncio.create_task(self._process_single_file(file_type, file_info))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        conversations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ File processing failed: {files[i]['path']} - {str(result)}")
                self.stats.files_failed += 1
                self.stats.errors.append(f"File processing failed: {files[i]['path']} - {str(result)}")
            elif result:
                conversations.extend(result)
        
        return conversations
    
    async def _process_single_file(self, file_type: FileType, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single file to generate conversations"""
        file_path = file_info['path']
        
        try:
            # Read file content
            content = await self._read_file_content(file_path, file_type)
            if not content or len(content.strip()) < 100:
                logger.warning(f"⚠️ Skipping file with insufficient content: {file_path}")
                return []
            
            # Determine processing complexity
            complexity = self._assess_content_complexity(content)
            
            # Select appropriate model
            model_id = await self._select_model_for_complexity(complexity)
            
            # Generate conversation based on file type
            conversation = await self._generate_conversation(content, file_type, model_id, file_info)
            
            if conversation:
                # Add metadata
                conversation['metadata'] = {
                    **conversation.get('metadata', {}),
                    'source_file': str(file_info['relative_path']),
                    'source_zip': file_info['original_zip'],
                    'file_type': file_type.value,
                    'content_complexity': complexity,
                    'model_used': model_id,
                    'processing_timestamp': time.time()
                }
                
                return [conversation]
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {str(e)}")
            raise
        
        return []
    
    async def _read_file_content(self, file_path: Path, file_type: FileType) -> str:
        """Read and preprocess file content"""
        try:
            if file_type == FileType.PDF:
                # For PDFs, we'd need additional processing
                # For now, skip or use a PDF processing library
                logger.warning(f"⚠️ PDF processing not implemented: {file_path}")
                return ""
            
            # Read text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic preprocessing based on file type
            if file_type == FileType.HTML:
                # Strip HTML tags (basic implementation)
                import re
                content = re.sub(r'<[^>]+>', '', content)
                content = re.sub(r'\s+', ' ', content)
            
            elif file_type == FileType.CSV:
                # Convert CSV to structured text
                try:
                    df = pd.read_csv(file_path)
                    content = self._csv_to_text(df, file_path.name)
                except Exception as e:
                    logger.warning(f"⚠️ CSV processing failed: {e}")
                    return ""
            
            elif file_type == FileType.JSON:
                # Format JSON content
                try:
                    data = json.loads(content)
                    content = self._json_to_text(data, file_path.name)
                except Exception as e:
                    logger.warning(f"⚠️ JSON processing failed: {e}")
                    return ""
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to read {file_path}: {str(e)}")
            return ""
    
    def _csv_to_text(self, df: pd.DataFrame, filename: str) -> str:
        """Convert CSV DataFrame to readable text"""
        text_parts = [f"# Data from {filename}\n"]
        
        # Add basic statistics
        text_parts.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.\n")
        text_parts.append(f"Columns: {', '.join(df.columns)}\n")
        
        # Add sample data
        if len(df) > 0:
            text_parts.append("\n## Sample Data:")
            
            # Show first few rows
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                text_parts.append(f"\nRecord {i+1}:")
                for col in df.columns:
                    value = str(row[col])[:100]  # Limit value length
                    text_parts.append(f"  - {col}: {value}")
        
        return '\n'.join(text_parts)
    
    def _json_to_text(self, data: Any, filename: str) -> str:
        """Convert JSON data to readable text"""
        text_parts = [f"# Data from {filename}\n"]
        
        if isinstance(data, dict):
            text_parts.append("Structure: Dictionary/Object")
            text_parts.append(f"Keys: {list(data.keys())}")
            
            # Add key-value pairs
            for key, value in data.items():
                text_parts.append(f"\n## {key}")
                if isinstance(value, (dict, list)):
                    text_parts.append(f"Type: {type(value).__name__}")
                    text_parts.append(f"Content: {str(value)[:200]}...")
                else:
                    text_parts.append(f"Value: {str(value)}")
        
        elif isinstance(data, list):
            text_parts.append(f"Structure: Array/List with {len(data)} items")
            
            # Show sample items
            for i, item in enumerate(data[:3]):
                text_parts.append(f"\nItem {i+1}: {str(item)[:200]}...")
        
        else:
            text_parts.append(f"Simple value: {str(data)}")
        
        return '\n'.join(text_parts)
    
    def _assess_content_complexity(self, content: str) -> float:
        """Assess content complexity for model selection"""
        # Basic complexity scoring (0.0 to 1.0)
        factors = []
        
        # Length factor
        length_score = min(len(content) / 10000, 1.0)
        factors.append(length_score * 0.3)
        
        # Technical terminology factor
        technical_terms = [
            'parameter', 'configuration', 'algorithm', 'optimization',
            'protocol', 'interface', 'specification', 'implementation',
            'architecture', 'framework', 'methodology', 'analysis'
        ]
        
        tech_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        tech_score = min(tech_count / 10, 1.0)
        factors.append(tech_score * 0.4)
        
        # Structure complexity (lists, tables, code)
        structure_indicators = ['|', '```', '- ', '* ', '1.', '2.', '3.']
        struct_count = sum(content.count(indicator) for indicator in structure_indicators)
        struct_score = min(struct_count / 20, 1.0)
        factors.append(struct_score * 0.3)
        
        return sum(factors)
    
    async def _select_model_for_complexity(self, complexity: float) -> str:
        """Select appropriate model based on content complexity"""
        if complexity < 0.3:
            # Simple content - use fast model
            model_id = self.config.fallback_model
        elif complexity < 0.7:
            # Medium complexity - use standard model
            model_id = self.config.primary_model
        else:
            # High complexity - use best available model
            model_id = self.config.primary_model
        
        # Ensure model is loaded
        if model_id not in self.model_manager.loaded_models:
            success = await self.model_manager.load_model(model_id)
            if not success:
                # Fallback to any available model
                available_models = list(self.model_manager.loaded_models.keys())
                if available_models:
                    model_id = available_models[0]
                    logger.warning(f"⚠️ Using fallback model: {model_id}")
                else:
                    raise RuntimeError("❌ No models available")
        
        return model_id
    
    async def _generate_conversation(
        self,
        content: str,
        file_type: FileType,
        model_id: str,
        file_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate conversation from content"""
        
        # Create context-aware prompt based on file type
        prompt = self._create_prompt_for_type(content, file_type, file_info)
        
        # Generate response using selected model
        response = await self.model_manager.generate_text(
            model_id,
            prompt,
            max_tokens=1024,
            temperature=0.7
        )
        
        if not response or len(response.strip()) < 50:
            logger.warning(f"⚠️ Insufficient response from model for {file_info['path']}")
            return None
        
        # Create conversation structure
        conversation = {
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant specializing in technical documentation and data analysis.'
                },
                {
                    'role': 'user', 
                    'content': self._extract_question_from_prompt(prompt)
                },
                {
                    'role': 'assistant',
                    'content': response
                }
            ],
            'quality_score': self._assess_conversation_quality(prompt, response, content),
            'generation_timestamp': time.time()
        }
        
        # Update statistics
        self.stats.conversations_generated += 1
        self.stats.quality_scores.append(conversation['quality_score'])
        
        return conversation
    
    def _create_prompt_for_type(self, content: str, file_type: FileType, file_info: Dict[str, Any]) -> str:
        """Create type-specific prompts for content"""
        
        base_context = f"Based on the following {file_type.value.upper()} content from {file_info['relative_path']}:"
        
        if file_type == FileType.MARKDOWN:
            question = "What are the key technical concepts and information presented in this documentation?"
            
        elif file_type == FileType.HTML:
            question = "What is the main content and technical information from this web page?"
            
        elif file_type == FileType.CSV:
            question = "What insights can you provide about this dataset? Include key statistics, patterns, and potential use cases."
            
        elif file_type == FileType.JSON:
            question = "Explain the structure and content of this data. What are the key components and their purposes?"
            
        elif file_type == FileType.XML:
            question = "What is the structure and meaning of this XML data? Explain the key elements and their relationships."
            
        elif file_type == FileType.TXT:
            question = "Summarize the main points and technical information from this text."
            
        else:
            question = "What are the key points and technical information in this content?"
        
        # Truncate content if too long
        max_content_length = 6000  # Leave room for prompt structure
        if len(content) > max_content_length:
            content = content[:max_content_length] + "...\n[Content truncated for processing]"
        
        prompt = f"{base_context}\n\n{content}\n\nQuestion: {question}"
        return prompt
    
    def _extract_question_from_prompt(self, prompt: str) -> str:
        """Extract the question portion from the full prompt"""
        if "Question: " in prompt:
            return prompt.split("Question: ")[-1]
        return "Please analyze and explain the provided content."
    
    def _assess_conversation_quality(self, prompt: str, response: str, original_content: str) -> float:
        """Assess the quality of generated conversation"""
        quality_factors = []
        
        # Response length factor
        length_score = min(len(response) / 500, 1.0)
        quality_factors.append(length_score * 0.2)
        
        # Content relevance (basic keyword overlap)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        relevance_score = min(overlap / 50, 1.0)
        quality_factors.append(relevance_score * 0.3)
        
        # Technical depth (presence of technical terms)
        technical_terms = ['parameter', 'configuration', 'system', 'data', 'analysis', 'structure']
        tech_count = sum(1 for term in technical_terms if term in response.lower())
        tech_score = min(tech_count / 5, 1.0)
        quality_factors.append(tech_score * 0.3)
        
        # Coherence (basic sentence structure check)
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        coherence_score = 1.0 if 8 <= avg_sentence_length <= 25 else 0.5
        quality_factors.append(coherence_score * 0.2)
        
        return min(sum(quality_factors), 1.0)
    
    async def _post_process_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process conversations for quality and diversity"""
        logger.info(f"🔧 Post-processing {len(conversations)} conversations...")
        
        # Filter by quality threshold
        high_quality = [
            conv for conv in conversations 
            if conv.get('quality_score', 0) >= self.config.min_quality_score
        ]
        
        logger.info(f"   📊 {len(high_quality)}/{len(conversations)} conversations meet quality threshold")
        
        # Remove duplicates using embeddings
        if len(high_quality) > 1:
            deduplicated = await self._remove_duplicate_conversations(high_quality)
            logger.info(f"   🗂️ {len(deduplicated)}/{len(high_quality)} conversations after deduplication")
        else:
            deduplicated = high_quality
        
        # Enhance diversity if needed
        if len(deduplicated) > 10:
            diversified = await self._enhance_conversation_diversity(deduplicated)
            logger.info(f"   🎯 {len(diversified)}/{len(deduplicated)} conversations after diversity enhancement")
        else:
            diversified = deduplicated
        
        return diversified
    
    async def _remove_duplicate_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate conversations using embeddings"""
        
        # Extract user questions for similarity comparison
        questions = []
        for conv in conversations:
            user_msg = None
            for msg in conv.get('messages', []):
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                    break
            questions.append(user_msg or "")
        
        # Generate embeddings for questions
        result = await self.embedding_manager.generate_embeddings(
            questions,
            self.config.embedding_model.value,
            show_progress=False
        )
        
        if not result:
            logger.warning("⚠️ Failed to generate embeddings for deduplication")
            return conversations
        
        embeddings = result.embeddings
        
        # Find similar conversations
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Mark duplicates
        to_remove = set()
        for i in range(len(conversations)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(conversations)):
                if j in to_remove:
                    continue
                    
                if similarities[i][j] >= self.config.deduplication_threshold:
                    # Keep the higher quality conversation
                    quality_i = conversations[i].get('quality_score', 0)
                    quality_j = conversations[j].get('quality_score', 0)
                    
                    if quality_i >= quality_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        # Return conversations without duplicates
        return [conv for i, conv in enumerate(conversations) if i not in to_remove]
    
    async def _enhance_conversation_diversity(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance conversation diversity through clustering and selection"""
        
        # Extract questions for clustering
        questions = []
        for conv in conversations:
            user_msg = None
            for msg in conv.get('messages', []):
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                    break
            questions.append(user_msg or "")
        
        # Cluster conversations
        cluster_result = await self.embedding_manager.cluster_texts(
            questions,
            self.config.embedding_model.value,
            method="kmeans",
            n_clusters=min(10, len(conversations) // 3)
        )
        
        if not cluster_result or 'clusters' not in cluster_result:
            logger.warning("⚠️ Failed to cluster conversations for diversity")
            return conversations
        
        # Select diverse representatives from clusters
        diverse_conversations = []
        clusters = cluster_result['clusters']
        
        for cluster_id, items in clusters.items():
            if cluster_id == -1:  # Skip noise cluster from DBSCAN
                continue
                
            # Sort items in cluster by quality
            cluster_conversations = [
                (conversations[item['index']], conversations[item['index']].get('quality_score', 0))
                for item in items
            ]
            cluster_conversations.sort(key=lambda x: x[1], reverse=True)
            
            # Take best representatives from each cluster
            num_to_take = max(1, len(cluster_conversations) // 3)
            for i in range(min(num_to_take, len(cluster_conversations))):
                diverse_conversations.append(cluster_conversations[i][0])
        
        return diverse_conversations
    
    async def _generate_outputs(self, conversations: List[Dict[str, Any]]):
        """Generate output files in requested formats"""
        logger.info(f"📁 Generating outputs in {len(self.config.output_formats)} formats...")
        
        output_base = Path(self.config.output_dir) / "final_dataset"
        
        for format_name in self.config.output_formats:
            try:
                if format_name == "jsonl":
                    await self._write_jsonl(conversations, output_base.with_suffix(".jsonl"))
                
                elif format_name == "parquet":
                    await self._write_parquet(conversations, output_base.with_suffix(".parquet"))
                
                elif format_name == "json":
                    await self._write_json(conversations, output_base.with_suffix(".json"))
                
                elif format_name == "csv":
                    await self._write_csv(conversations, output_base.with_suffix(".csv"))
                
                else:
                    logger.warning(f"⚠️ Unknown output format: {format_name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate {format_name} output: {str(e)}")
                self.stats.errors.append(f"Output generation failed: {format_name} - {str(e)}")
        
        # Generate metadata file
        await self._write_metadata(conversations, output_base.with_suffix(".metadata.json"))
    
    async def _write_jsonl(self, conversations: List[Dict[str, Any]], output_path: Path):
        """Write conversations to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        
        logger.info(f"✅ JSONL output: {output_path} ({len(conversations)} conversations)")
    
    async def _write_parquet(self, conversations: List[Dict[str, Any]], output_path: Path):
        """Write conversations to Parquet format"""
        
        # Flatten conversations for tabular format
        rows = []
        for i, conv in enumerate(conversations):
            messages = conv.get('messages', [])
            
            # Extract system, user, and assistant messages
            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_msg = content
                elif role == 'user':
                    user_msg = content
                elif role == 'assistant':
                    assistant_msg = content
            
            # Create row
            row = {
                'id': i,
                'system': system_msg,
                'user': user_msg,
                'assistant': assistant_msg,
                'quality_score': conv.get('quality_score', 0.0),
                'generation_timestamp': conv.get('generation_timestamp', 0),
            }
            
            # Add metadata fields
            metadata = conv.get('metadata', {})
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    row[f'metadata_{key}'] = value
            
            rows.append(row)
        
        # Write to parquet
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
        
        logger.info(f"✅ Parquet output: {output_path} ({len(rows)} rows)")
    
    async def _write_json(self, conversations: List[Dict[str, Any]], output_path: Path):
        """Write conversations to JSON format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON output: {output_path}")
    
    async def _write_csv(self, conversations: List[Dict[str, Any]], output_path: Path):
        """Write conversations to CSV format"""
        
        # Flatten to CSV format
        rows = []
        for i, conv in enumerate(conversations):
            messages = conv.get('messages', [])
            
            # Find user and assistant messages
            user_content = ""
            assistant_content = ""
            
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_content = msg.get('content', '')
            
            rows.append({
                'id': i,
                'question': user_content,
                'answer': assistant_content,
                'quality_score': conv.get('quality_score', 0.0)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ CSV output: {output_path} ({len(rows)} rows)")
    
    async def _write_metadata(self, conversations: List[Dict[str, Any]], output_path: Path):
        """Write dataset metadata"""
        metadata = {
            'dataset_info': {
                'total_conversations': len(conversations),
                'generation_timestamp': time.time(),
                'pipeline_version': "1.0.0",
                'config': asdict(self.config)
            },
            'statistics': asdict(self.stats),
            'quality_distribution': {
                'min_quality': float(np.min(self.stats.quality_scores)) if self.stats.quality_scores else 0,
                'max_quality': float(np.max(self.stats.quality_scores)) if self.stats.quality_scores else 0,
                'mean_quality': float(np.mean(self.stats.quality_scores)) if self.stats.quality_scores else 0,
                'std_quality': float(np.std(self.stats.quality_scores)) if self.stats.quality_scores else 0
            },
            'model_info': self.model_manager.get_system_status(),
            'embedding_info': self.embedding_manager.get_system_status()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Metadata: {output_path}")
    
    async def cleanup(self):
        """Cleanup resources and temporary files"""
        logger.info("🧹 Cleaning up resources...")
        
        # Cleanup models
        await self.model_manager.cleanup_unused_models(max_idle_minutes=0)
        await self.embedding_manager.cleanup_cache()
        
        # Cleanup temporary extraction directories
        for file_info in self.file_registry.values():
            if 'extract_dir' in file_info:
                extract_dir = file_info['extract_dir']
                try:
                    shutil.rmtree(extract_dir)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cleanup {extract_dir}: {e}")
        
        logger.info("✅ Cleanup completed")

# Example usage
async def main():
    """Example usage of EnhancedDatasetPipeline"""
    
    # Configuration
    config = DatasetConfig(
        input_paths=[
            "/path/to/documents.zip",
            "/path/to/data_directory/"
        ],
        output_dir="enhanced_dataset_output",
        quality_level=QualityLevel.STANDARD,
        max_files_per_type=100,
        batch_size=16,
        workers=4,
        primary_model="qwen3-7b-mlx",
        embedding_model=EmbeddingModel.BGE_BASE_EN,
        min_quality_score=0.7,
        output_formats=["jsonl", "parquet", "json"]
    )
    
    # Run pipeline
    pipeline = EnhancedDatasetPipeline(config)
    
    try:
        await pipeline.initialize()
        stats = await pipeline.process_zip_files()
        
        print("📊 Pipeline Results:")
        print(f"   Files processed: {stats.files_processed}")
        print(f"   Conversations generated: {stats.conversations_generated}")
        print(f"   Average quality: {np.mean(stats.quality_scores):.3f}")
        print(f"   Processing time: {stats.processing_time:.2f}s")
        
    finally:
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())