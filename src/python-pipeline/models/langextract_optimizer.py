#!/usr/bin/env python3
"""
LangExtract Optimizer

Optimized LangExtract processing pipeline integrating Qwen3 models with
performance optimizations and quality assurance for maximum throughput.

Author: Claude Code ML Integration Specialist
Date: 2025-08-23
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import multiprocessing as mp
import threading
import subprocess

# Import local modules
try:
    from .qwen3_manager import Qwen3DynamicManager, TaskComplexity, ModelVariant
    from .local_inference import UnifiedLocalInferenceClient, InferenceRequest
    from .model_pool import AdvancedModelPool, create_optimized_model_pool
    from .quality_scorer import AdvancedQualityScorer, create_quality_scorer
except ImportError:
    # Fallback for testing
    import sys
    sys.path.append('.')
    from qwen3_manager import Qwen3DynamicManager, TaskComplexity, ModelVariant
    from local_inference import UnifiedLocalInferenceClient, InferenceRequest
    from model_pool import AdvancedModelPool, create_optimized_model_pool
    from quality_scorer import AdvancedQualityScorer, create_quality_scorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """LangExtract processing modes"""
    SPEED = "speed"  # Maximize throughput
    QUALITY = "quality"  # Maximize quality
    BALANCED = "balanced"  # Balance speed and quality
    ADAPTIVE = "adaptive"  # Adapt based on content

class ContentType(Enum):
    """Types of content for processing"""
    TEXT = "text"
    DOCUMENT = "document"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    STRUCTURED = "structured"  # JSON, YAML, etc.

@dataclass
class ProcessingTask:
    """Individual processing task"""
    task_id: str
    content: str
    content_type: ContentType = ContentType.TEXT
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    priority: int = 1  # 1-5, higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ProcessingResult:
    """Processing result with performance metrics"""
    task_id: str
    original_content: str
    processed_content: str
    extraction_data: Dict[str, Any]
    quality_score: float
    processing_time_seconds: float
    model_used: str
    tokens_processed: int
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_processing_time: float
    average_quality_score: float
    throughput_items_per_minute: float
    model_usage_stats: Dict[str, int]
    error_summary: List[str] = field(default_factory=list)

class ContentPreprocessor:
    """Preprocess content for optimal extraction"""
    
    def __init__(self):
        self.preprocessing_stats = {
            'texts_processed': 0,
            'average_processing_time': 0.0,
            'optimizations_applied': 0
        }
    
    def preprocess_content(self, content: str, content_type: ContentType) -> Tuple[str, Dict[str, Any]]:
        """Preprocess content based on type"""
        start_time = time.time()
        
        metadata = {
            'original_length': len(content),
            'preprocessing_applied': []
        }
        
        processed_content = content
        
        # Common preprocessing
        if content_type in [ContentType.TEXT, ContentType.DOCUMENT]:
            processed_content = self._preprocess_text(processed_content, metadata)
        elif content_type == ContentType.MARKDOWN:
            processed_content = self._preprocess_markdown(processed_content, metadata)
        elif content_type == ContentType.CODE:
            processed_content = self._preprocess_code(processed_content, metadata)
        elif content_type == ContentType.STRUCTURED:
            processed_content = self._preprocess_structured(processed_content, metadata)
        
        # Update stats
        processing_time = time.time() - start_time
        self.preprocessing_stats['texts_processed'] += 1
        n = self.preprocessing_stats['texts_processed']
        old_avg = self.preprocessing_stats['average_processing_time']
        self.preprocessing_stats['average_processing_time'] = (
            old_avg * (n - 1) + processing_time
        ) / n
        
        metadata['preprocessing_time'] = processing_time
        metadata['final_length'] = len(processed_content)
        metadata['compression_ratio'] = len(processed_content) / len(content) if content else 1.0
        
        return processed_content, metadata
    
    def _preprocess_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """Preprocess plain text"""
        # Remove excessive whitespace
        if '\n\n\n' in text:
            text = '\n\n'.join(paragraph.strip() for paragraph in text.split('\n\n') if paragraph.strip())
            metadata['preprocessing_applied'].append('whitespace_normalization')
        
        # Fix common encoding issues
        replacements = {
            '\u201c': '"', '\u201d': '"',  # Smart quotes
            '\u2018': "'", '\u2019': "'",
            '\u2013': '-', '\u2014': '--',  # Dashes
            '\xa0': ' '  # Non-breaking space
        }
        
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                metadata['preprocessing_applied'].append('encoding_fix')
        
        return text
    
    def _preprocess_markdown(self, text: str, metadata: Dict[str, Any]) -> str:
        """Preprocess Markdown content"""
        # Preserve structure but clean content
        text = self._preprocess_text(text, metadata)
        
        # Normalize headers
        import re
        if re.search(r'^#{1,6}\s', text, re.MULTILINE):
            # Headers are properly formatted
            metadata['preprocessing_applied'].append('markdown_structure_preserved')
        
        return text
    
    def _preprocess_code(self, text: str, metadata: Dict[str, Any]) -> str:
        """Preprocess code content"""
        # Minimal preprocessing for code to preserve structure
        metadata['preprocessing_applied'].append('code_structure_preserved')
        return text
    
    def _preprocess_structured(self, text: str, metadata: Dict[str, Any]) -> str:
        """Preprocess structured data (JSON, YAML, etc.)"""
        # Validate and potentially reformat structured data
        try:
            # Try to parse as JSON first
            data = json.loads(text)
            text = json.dumps(data, indent=2)  # Reformat nicely
            metadata['preprocessing_applied'].append('json_reformatted')
        except json.JSONDecodeError:
            # Not JSON, keep as is
            metadata['preprocessing_applied'].append('structured_preserved')
        
        return text

class ExtractionEngine:
    """Core extraction engine using Qwen3 models"""
    
    def __init__(self, model_pool: AdvancedModelPool):
        self.model_pool = model_pool
        self.extraction_templates = {
            ContentType.TEXT: {
                'speed': "Extract key information from this text: {content}",
                'quality': "Perform comprehensive analysis and extract all important information, entities, themes, and insights from this text: {content}",
                'balanced': "Extract key information, main themes, and important details from this text: {content}"
            },
            ContentType.DOCUMENT: {
                'speed': "Summarize this document and extract key points: {content}",
                'quality': "Provide detailed analysis of this document including summary, key points, themes, conclusions, and actionable insights: {content}",
                'balanced': "Analyze this document and extract summary, key points, and main insights: {content}"
            },
            ContentType.CODE: {
                'speed': "Extract function names and brief descriptions from this code: {content}",
                'quality': "Analyze this code thoroughly: identify functions, classes, dependencies, design patterns, potential issues, and provide comprehensive documentation: {content}",
                'balanced': "Analyze this code and extract functions, classes, main logic, and key insights: {content}"
            },
            ContentType.STRUCTURED: {
                'speed': "Extract key fields and values from this structured data: {content}",
                'quality': "Thoroughly analyze this structured data, identify all fields, relationships, patterns, and provide comprehensive insights: {content}",
                'balanced': "Analyze this structured data and extract key fields, values, and patterns: {content}"
            }
        }
    
    async def extract_content(self, task: ProcessingTask) -> Dict[str, Any]:
        """Extract content using appropriate model and template"""
        
        # Select extraction template
        template_category = task.processing_mode.value
        content_templates = self.extraction_templates.get(task.content_type, self.extraction_templates[ContentType.TEXT])
        template = content_templates.get(template_category, content_templates['balanced'])
        
        # Format prompt
        prompt = template.format(content=task.content[:4000])  # Limit content length
        
        # Select model complexity based on mode
        if task.processing_mode == ProcessingMode.SPEED:
            complexity = TaskComplexity.SIMPLE
        elif task.processing_mode == ProcessingMode.QUALITY:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        
        try:
            # Generate extraction
            response = await self.model_pool.generate_with_pool(
                prompt=prompt,
                task_complexity=complexity,
                max_tokens=1000 if task.processing_mode == ProcessingMode.SPEED else 2000,
                temperature=0.3  # Lower temperature for more consistent extraction
            )
            
            # Parse extraction result
            extraction_data = self._parse_extraction_result(
                response.text if hasattr(response, 'text') else str(response),
                task.content_type
            )
            
            extraction_data['model_used'] = getattr(response, 'model_used', 'unknown')
            extraction_data['tokens_per_second'] = getattr(response, 'tokens_per_second', 0)
            
            return extraction_data
            
        except Exception as e:
            logger.error(f"Extraction failed for task {task.task_id}: {e}")
            return {
                'error': str(e),
                'success': False,
                'extracted_text': '',
                'model_used': 'error'
            }
    
    def _parse_extraction_result(self, result: str, content_type: ContentType) -> Dict[str, Any]:
        """Parse extraction result based on content type"""
        
        extraction = {
            'extracted_text': result,
            'success': True
        }
        
        # Basic parsing - in production, this would be more sophisticated
        if content_type == ContentType.CODE:
            # Try to extract function/class information
            import re
            functions = re.findall(r'def\s+(\w+)', result)
            classes = re.findall(r'class\s+(\w+)', result)
            extraction.update({
                'functions': functions,
                'classes': classes,
                'code_elements': len(functions) + len(classes)
            })
        
        elif content_type == ContentType.STRUCTURED:
            # Try to extract key-value pairs
            lines = result.split('\n')
            key_values = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key_values[key.strip()] = value.strip()
            extraction['key_values'] = key_values
        
        # Common extraction metrics
        extraction.update({
            'word_count': len(result.split()),
            'char_count': len(result),
            'line_count': len(result.split('\n'))
        })
        
        return extraction

class LangExtractOptimizer:
    """Main LangExtract optimization system"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 8,
                 quality_threshold: float = 0.742,
                 enable_multiprocessing: bool = True):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.quality_threshold = quality_threshold
        self.enable_multiprocessing = enable_multiprocessing
        
        # Initialize components
        self.model_pool = create_optimized_model_pool({
            'max_models': 3,
            'max_memory_gb': 50.0
        })
        self.quality_scorer = create_quality_scorer(quality_threshold)
        self.preprocessor = ContentPreprocessor()
        self.extraction_engine = ExtractionEngine(self.model_pool)
        
        # Processing queues
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.result_queue = asyncio.Queue(maxsize=1000)
        
        # Performance tracking
        self.performance_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'total_processing_time': 0.0,
            'average_quality_score': 0.0,
            'throughput_per_minute': 0.0,
            'model_usage': {},
            'content_type_stats': {}
        }
        
        # Worker management
        self.workers_active = False
        self.worker_tasks = set()
        
        logger.info(f"Initialized LangExtractOptimizer with {max_concurrent_tasks} max concurrent tasks")
    
    async def start_workers(self):
        """Start processing workers"""
        self.workers_active = True
        
        # Start processing workers
        for i in range(self.max_concurrent_tasks):
            task = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.worker_tasks.add(task)
            task.add_done_callback(self.worker_tasks.discard)
        
        logger.info(f"Started {self.max_concurrent_tasks} processing workers")
    
    async def stop_workers(self):
        """Stop processing workers"""
        self.workers_active = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Stopped all processing workers")
    
    async def _processing_worker(self, worker_id: str):
        """Individual processing worker"""
        logger.info(f"Started processing worker: {worker_id}")
        
        while self.workers_active:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process the task
                result = await self._process_single_task(task)
                
                # Put result in queue
                await self.result_queue.put(result)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    async def _process_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task"""
        start_time = time.time()
        
        try:
            # Preprocess content
            processed_content, preprocessing_metadata = self.preprocessor.preprocess_content(
                task.content, task.content_type
            )
            
            # Update task with preprocessed content
            processing_task = ProcessingTask(
                task_id=task.task_id,
                content=processed_content,
                content_type=task.content_type,
                processing_mode=task.processing_mode,
                priority=task.priority,
                metadata={**task.metadata, **preprocessing_metadata}
            )
            
            # Extract content
            extraction_data = await self.extraction_engine.extract_content(processing_task)
            
            # Quality assessment
            quality_assessment = self.quality_scorer.assess_quality(
                extraction_data.get('extracted_text', ''),
                context={'query': f"Extract information from {task.content_type.value}"}
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                task_id=task.task_id,
                original_content=task.content,
                processed_content=processed_content,
                extraction_data=extraction_data,
                quality_score=quality_assessment.overall_score,
                processing_time_seconds=processing_time,
                model_used=extraction_data.get('model_used', 'unknown'),
                tokens_processed=extraction_data.get('word_count', 0),
                success=extraction_data.get('success', True),
                metadata={
                    'quality_assessment': {
                        'overall_score': quality_assessment.overall_score,
                        'meets_threshold': quality_assessment.meets_threshold,
                        'consistency_score': quality_assessment.consistency_score
                    },
                    'preprocessing': preprocessing_metadata,
                    'extraction_stats': extraction_data
                }
            )
            
            # Update performance stats
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Task {task.task_id} processing failed: {e}")
            
            return ProcessingResult(
                task_id=task.task_id,
                original_content=task.content,
                processed_content=task.content,
                extraction_data={'error': str(e)},
                quality_score=0.0,
                processing_time_seconds=processing_time,
                model_used='error',
                tokens_processed=0,
                success=False,
                error_message=str(e)
            )
    
    def _update_performance_stats(self, result: ProcessingResult):
        """Update performance statistics"""
        self.performance_stats['total_processed'] += 1
        
        if result.success:
            self.performance_stats['successful_processed'] += 1
        
        self.performance_stats['total_processing_time'] += result.processing_time_seconds
        
        # Update average quality score
        n = self.performance_stats['total_processed']
        old_avg = self.performance_stats['average_quality_score']
        self.performance_stats['average_quality_score'] = (
            old_avg * (n - 1) + result.quality_score
        ) / n
        
        # Update model usage
        model_used = result.model_used
        self.performance_stats['model_usage'][model_used] = \
            self.performance_stats['model_usage'].get(model_used, 0) + 1
        
        # Update throughput
        if self.performance_stats['total_processing_time'] > 0:
            self.performance_stats['throughput_per_minute'] = (
                self.performance_stats['successful_processed'] * 60 / 
                self.performance_stats['total_processing_time']
            )
    
    async def process_single(self, 
                           content: str,
                           content_type: ContentType = ContentType.TEXT,
                           processing_mode: ProcessingMode = ProcessingMode.BALANCED,
                           priority: int = 1) -> ProcessingResult:
        """Process a single item"""
        
        task = ProcessingTask(
            task_id=f"single-{int(time.time() * 1000)}",
            content=content,
            content_type=content_type,
            processing_mode=processing_mode,
            priority=priority
        )
        
        return await self._process_single_task(task)
    
    async def process_batch(self, 
                          contents: List[str],
                          content_types: Optional[List[ContentType]] = None,
                          processing_modes: Optional[List[ProcessingMode]] = None,
                          priorities: Optional[List[int]] = None) -> BatchResult:
        """Process multiple items in batch"""
        
        start_time = time.time()
        batch_id = f"batch-{int(start_time * 1000)}"
        
        # Prepare tasks
        tasks = []
        for i, content in enumerate(contents):
            content_type = content_types[i] if content_types and i < len(content_types) else ContentType.TEXT
            processing_mode = processing_modes[i] if processing_modes and i < len(processing_modes) else ProcessingMode.BALANCED
            priority = priorities[i] if priorities and i < len(priorities) else 1
            
            task = ProcessingTask(
                task_id=f"{batch_id}-{i}",
                content=content,
                content_type=content_type,
                processing_mode=processing_mode,
                priority=priority
            )
            tasks.append(task)
        
        # Add tasks to queue
        for task in tasks:
            await self.processing_queue.put(task)
        
        # Start workers if not already running
        if not self.workers_active:
            await self.start_workers()
        
        # Collect results
        results = []
        successful = 0
        failed = 0
        quality_scores = []
        model_usage = {}
        errors = []
        
        for _ in range(len(tasks)):
            result = await self.result_queue.get()
            results.append(result)
            
            if result.success:
                successful += 1
                quality_scores.append(result.quality_score)
            else:
                failed += 1
                errors.append(result.error_message)
            
            model_usage[result.model_used] = model_usage.get(result.model_used, 0) + 1
        
        total_time = time.time() - start_time
        
        batch_result = BatchResult(
            batch_id=batch_id,
            total_tasks=len(tasks),
            successful_tasks=successful,
            failed_tasks=failed,
            total_processing_time=total_time,
            average_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            throughput_items_per_minute=len(tasks) * 60 / total_time if total_time > 0 else 0,
            model_usage_stats=model_usage,
            error_summary=list(set(errors))  # Unique errors
        )
        
        return batch_result
    
    async def process_file(self, 
                         file_path: Union[str, Path],
                         content_type: ContentType = ContentType.DOCUMENT,
                         processing_mode: ProcessingMode = ProcessingMode.BALANCED) -> ProcessingResult:
        """Process content from file"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Auto-detect content type if not specified
        if content_type == ContentType.DOCUMENT:
            if file_path.suffix.lower() == '.md':
                content_type = ContentType.MARKDOWN
            elif file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
                content_type = ContentType.CODE
            elif file_path.suffix.lower() in ['.json', '.yaml', '.yml']:
                content_type = ContentType.STRUCTURED
        
        return await self.process_single(
            content=content,
            content_type=content_type,
            processing_mode=processing_mode
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            'timestamp': time.time(),
            'processing_stats': self.performance_stats,
            'model_pool_stats': self.model_pool.get_pool_statistics(),
            'quality_scorer_stats': self.quality_scorer.get_performance_summary(),
            'preprocessor_stats': self.preprocessor.preprocessing_stats,
            'system_info': {
                'workers_active': self.workers_active,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'queue_sizes': {
                    'processing_queue': self.processing_queue.qsize(),
                    'result_queue': self.result_queue.qsize()
                }
            }
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize performance based on current statistics"""
        
        optimization_results = {
            'timestamp': time.time(),
            'optimizations_applied': []
        }
        
        # Optimize model pool
        pool_optimization = await self.model_pool.optimize_pool()
        optimization_results['model_pool'] = pool_optimization
        optimization_results['optimizations_applied'].extend(
            pool_optimization.get('actions_taken', [])
        )
        
        # Optimize quality scorer
        quality_optimization = self.quality_scorer.optimize_for_threshold()
        optimization_results['quality_scorer'] = quality_optimization
        if quality_optimization.get('adjustments_made'):
            optimization_results['optimizations_applied'].extend(
                quality_optimization['adjustments_made']
            )
        
        # Adjust worker count based on performance
        current_throughput = self.performance_stats.get('throughput_per_minute', 0)
        if current_throughput > 0:
            if current_throughput < 50 and self.max_concurrent_tasks < 12:  # Low throughput
                self.max_concurrent_tasks += 1
                optimization_results['optimizations_applied'].append(
                    f"Increased max concurrent tasks to {self.max_concurrent_tasks}"
                )
            elif current_throughput > 200 and self.max_concurrent_tasks > 4:  # High throughput, might be memory bound
                self.max_concurrent_tasks -= 1
                optimization_results['optimizations_applied'].append(
                    f"Decreased max concurrent tasks to {self.max_concurrent_tasks}"
                )
        
        return optimization_results
    
    async def shutdown(self):
        """Gracefully shutdown the optimizer"""
        logger.info("Shutting down LangExtractOptimizer")
        
        # Stop workers
        await self.stop_workers()
        
        # Shutdown components
        await self.model_pool.shutdown()
        
        # Final performance summary
        final_summary = self.get_performance_summary()
        logger.info(f"Final performance: {final_summary['processing_stats']}")
        
        # Store final metrics in coordination memory
        try:
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-task",
                "--task-id", "langextract-optimizer-shutdown",
                "--data", json.dumps(final_summary)
            ], capture_output=True)
        except Exception:
            pass
        
        logger.info("LangExtractOptimizer shutdown complete")

# Factory function for easy initialization
def create_langextract_optimizer(**kwargs) -> LangExtractOptimizer:
    """Create an optimized LangExtract processor"""
    
    default_config = {
        'max_concurrent_tasks': 6,  # Conservative for M3 Max
        'quality_threshold': 0.742,
        'enable_multiprocessing': True
    }
    
    default_config.update(kwargs)
    
    return LangExtractOptimizer(**default_config)

# Example usage and testing
async def main():
    """Example usage of the LangExtract optimizer"""
    
    optimizer = create_langextract_optimizer(max_concurrent_tasks=4)
    
    try:
        await optimizer.start_workers()
        
        # Test single processing
        result = await optimizer.process_single(
            "Machine learning is transforming industries by enabling computers to learn from data without explicit programming. This technology powers recommendation systems, autonomous vehicles, and medical diagnosis tools.",
            content_type=ContentType.TEXT,
            processing_mode=ProcessingMode.QUALITY
        )
        
        print(f"Single processing result:")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Processing Time: {result.processing_time_seconds:.2f}s")
        print(f"Model Used: {result.model_used}")
        print(f"Success: {result.success}")
        print(f"Extracted: {result.extraction_data.get('extracted_text', '')[:100]}...")
        
        # Test batch processing
        test_contents = [
            "Artificial intelligence represents a paradigm shift in computing.",
            "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
            "# Machine Learning Tutorial\n\nThis guide covers the basics of ML algorithms.",
            '{"name": "John", "age": 30, "skills": ["Python", "ML", "AI"]}'
        ]
        
        test_types = [
            ContentType.TEXT,
            ContentType.CODE,
            ContentType.MARKDOWN,
            ContentType.STRUCTURED
        ]
        
        batch_result = await optimizer.process_batch(
            contents=test_contents,
            content_types=test_types,
            processing_modes=[ProcessingMode.BALANCED] * len(test_contents)
        )
        
        print(f"\nBatch processing result:")
        print(f"Total Tasks: {batch_result.total_tasks}")
        print(f"Successful: {batch_result.successful_tasks}")
        print(f"Failed: {batch_result.failed_tasks}")
        print(f"Average Quality: {batch_result.average_quality_score:.3f}")
        print(f"Throughput: {batch_result.throughput_items_per_minute:.1f} items/min")
        print(f"Model Usage: {batch_result.model_usage_stats}")
        
        # Performance summary
        summary = optimizer.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"Total Processed: {summary['processing_stats']['total_processed']}")
        print(f"Success Rate: {summary['processing_stats']['successful_processed'] / summary['processing_stats']['total_processed']:.1%}")
        print(f"Average Quality: {summary['processing_stats']['average_quality_score']:.3f}")
        print(f"Throughput: {summary['processing_stats']['throughput_per_minute']:.1f} items/min")
        
        # Test optimization
        optimization = await optimizer.optimize_performance()
        print(f"\nOptimization Results:")
        print(f"Optimizations Applied: {optimization['optimizations_applied']}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        await optimizer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
