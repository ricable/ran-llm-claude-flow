#!/usr/bin/env python3
"""
Qwen3 Model Integration Package

Comprehensive Qwen3 model optimization and local inference integration
for Apple M3 Max hardware with advanced quality assurance.

Author: Claude Code ML Integration Specialist
Date: 2025-08-23
"""

__version__ = "1.0.0"
__author__ = "Claude Code ML Integration Specialist"

# Core imports
from .qwen3_manager import (
    Qwen3DynamicManager,
    ModelVariant,
    TaskComplexity,
    QuantizationType,
    ModelConfig,
    SystemResources,
    ModelPerformanceMetrics
)

from .local_inference import (
    UnifiedLocalInferenceClient,
    LMStudioClient,
    OllamaClient,
    InferenceBackend,
    InferenceRequest,
    InferenceResponse,
    CachedInferenceLayer
)

from .model_pool import (
    AdvancedModelPool,
    ModelPoolEntry,
    PoolingStrategy,
    WarmupStrategy,
    ModelPredictor,
    ModelWarmer,
    create_optimized_model_pool
)

from .quality_scorer import (
    AdvancedQualityScorer,
    QualityDimension,
    QualityLevel,
    QualityScore,
    QualityAssessment,
    create_quality_scorer
)

from .langextract_optimizer import (
    LangExtractOptimizer,
    ProcessingMode,
    ContentType,
    ProcessingTask,
    ProcessingResult,
    BatchResult,
    create_langextract_optimizer
)

# Package-level utilities
import logging
import time
import asyncio
from typing import Dict, Any, Optional

# Configure package logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

class Qwen3IntegrationSuite:
    """Integrated suite for Qwen3 model optimization and processing"""
    
    def __init__(self, 
                 quality_threshold: float = 0.742,
                 max_concurrent_models: int = 3,
                 max_memory_gb: float = 60.0):
        
        self.quality_threshold = quality_threshold
        self.max_concurrent_models = max_concurrent_models
        self.max_memory_gb = max_memory_gb
        
        # Initialize components
        self.model_manager = None
        self.model_pool = None
        self.inference_client = None
        self.quality_scorer = None
        self.langextract_optimizer = None
        
        self.initialized = False
        self.performance_metrics = {
            'initialization_time': 0,
            'total_inferences': 0,
            'average_quality_score': 0,
            'system_utilization': {}
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all components"""
        start_time = time.time()
        
        initialization_results = {
            'status': 'success',
            'components_initialized': [],
            'initialization_time': 0,
            'errors': []
        }
        
        try:
            # Initialize model manager
            self.model_manager = Qwen3DynamicManager(
                max_concurrent_models=self.max_concurrent_models
            )
            initialization_results['components_initialized'].append('model_manager')
            
            # Initialize model pool
            self.model_pool = create_optimized_model_pool({
                'max_models': self.max_concurrent_models,
                'max_memory_gb': self.max_memory_gb
            })
            initialization_results['components_initialized'].append('model_pool')
            
            # Initialize inference client
            self.inference_client = UnifiedLocalInferenceClient()
            initialization_results['components_initialized'].append('inference_client')
            
            # Initialize quality scorer
            self.quality_scorer = create_quality_scorer(self.quality_threshold)
            initialization_results['components_initialized'].append('quality_scorer')
            
            # Initialize LangExtract optimizer
            self.langextract_optimizer = create_langextract_optimizer(
                quality_threshold=self.quality_threshold,
                max_concurrent_tasks=self.max_concurrent_models * 2
            )
            initialization_results['components_initialized'].append('langextract_optimizer')
            
            # Start background services
            await self.model_pool.start_background_optimization()
            await self.langextract_optimizer.start_workers()
            
            self.initialized = True
            
            initialization_time = time.time() - start_time
            self.performance_metrics['initialization_time'] = initialization_time
            initialization_results['initialization_time'] = initialization_time
            
            logging.info(f"Qwen3IntegrationSuite initialized successfully in {initialization_time:.2f}s")
            
        except Exception as e:
            initialization_results['status'] = 'error'
            initialization_results['errors'].append(str(e))
            logging.error(f"Failed to initialize Qwen3IntegrationSuite: {e}")
        
        return initialization_results
    
    async def process_content(self, 
                            content: str,
                            content_type: ContentType = ContentType.TEXT,
                            processing_mode: ProcessingMode = ProcessingMode.BALANCED,
                            assess_quality: bool = True) -> Dict[str, Any]:
        """Process content using the integrated pipeline"""
        
        if not self.initialized:
            raise RuntimeError("Suite not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Process with LangExtract optimizer
        result = await self.langextract_optimizer.process_single(
            content=content,
            content_type=content_type,
            processing_mode=processing_mode
        )
        
        # Additional quality assessment if requested
        quality_details = None
        if assess_quality and result.success:
            quality_assessment = self.quality_scorer.assess_quality(
                result.extraction_data.get('extracted_text', ''),
                context={'query': f'Process {content_type.value} content'}
            )
            quality_details = {
                'overall_score': quality_assessment.overall_score,
                'meets_threshold': quality_assessment.meets_threshold,
                'quality_level': quality_assessment.quality_level.value,
                'dimension_scores': {
                    dim.value: score.score 
                    for dim, score in quality_assessment.dimension_scores.items()
                }
            }
        
        # Update performance metrics
        self.performance_metrics['total_inferences'] += 1
        if result.quality_score > 0:
            n = self.performance_metrics['total_inferences']
            old_avg = self.performance_metrics['average_quality_score']
            self.performance_metrics['average_quality_score'] = (
                old_avg * (n - 1) + result.quality_score
            ) / n
        
        processing_time = time.time() - start_time
        
        return {
            'success': result.success,
            'processed_content': result.processed_content,
            'extracted_data': result.extraction_data,
            'quality_score': result.quality_score,
            'quality_details': quality_details,
            'model_used': result.model_used,
            'processing_time': processing_time,
            'tokens_processed': result.tokens_processed,
            'metadata': result.metadata
        }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all components"""
        
        if not self.initialized:
            return {'status': 'not_initialized'}
        
        status = {
            'timestamp': time.time(),
            'initialized': self.initialized,
            'performance_metrics': self.performance_metrics
        }
        
        # Get component status
        try:
            status['model_pool'] = self.model_pool.get_pool_statistics()
            status['quality_scorer'] = self.quality_scorer.get_performance_summary()
            status['langextract_optimizer'] = self.langextract_optimizer.get_performance_summary()
            status['inference_client'] = asyncio.create_task(
                self.inference_client.get_performance_summary()
            ) if hasattr(self.inference_client, 'get_performance_summary') else 'not_available'
        except Exception as e:
            status['component_status_error'] = str(e)
        
        return status
    
    async def optimize_all_components(self) -> Dict[str, Any]:
        """Optimize all components for better performance"""
        
        if not self.initialized:
            raise RuntimeError("Suite not initialized")
        
        optimization_results = {
            'timestamp': time.time(),
            'components_optimized': {},
            'overall_improvement': 0
        }
        
        # Optimize model pool
        try:
            pool_optimization = await self.model_pool.optimize_pool()
            optimization_results['components_optimized']['model_pool'] = pool_optimization
        except Exception as e:
            optimization_results['components_optimized']['model_pool'] = {'error': str(e)}
        
        # Optimize quality scorer
        try:
            quality_optimization = self.quality_scorer.optimize_for_threshold()
            optimization_results['components_optimized']['quality_scorer'] = quality_optimization
        except Exception as e:
            optimization_results['components_optimized']['quality_scorer'] = {'error': str(e)}
        
        # Optimize LangExtract optimizer
        try:
            langextract_optimization = await self.langextract_optimizer.optimize_performance()
            optimization_results['components_optimized']['langextract_optimizer'] = langextract_optimization
        except Exception as e:
            optimization_results['components_optimized']['langextract_optimizer'] = {'error': str(e)}
        
        return optimization_results
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        
        if not self.initialized:
            return
        
        logging.info("Shutting down Qwen3IntegrationSuite")
        
        try:
            # Shutdown components in reverse order
            if self.langextract_optimizer:
                await self.langextract_optimizer.shutdown()
            
            if self.model_pool:
                await self.model_pool.shutdown()
            
            if self.model_manager:
                self.model_manager.shutdown()
            
            self.initialized = False
            logging.info("Qwen3IntegrationSuite shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

# Convenience functions
def create_integrated_suite(**kwargs) -> Qwen3IntegrationSuite:
    """Create a fully integrated Qwen3 processing suite"""
    return Qwen3IntegrationSuite(**kwargs)

async def quick_process(content: str, 
                       content_type: ContentType = ContentType.TEXT,
                       quality_threshold: float = 0.742) -> Dict[str, Any]:
    """Quick processing with automatic setup and teardown"""
    
    suite = create_integrated_suite(quality_threshold=quality_threshold)
    
    try:
        await suite.initialize()
        result = await suite.process_content(content, content_type)
        return result
    finally:
        await suite.shutdown()

# Package information
__all__ = [
    # Core classes
    'Qwen3DynamicManager',
    'UnifiedLocalInferenceClient', 
    'AdvancedModelPool',
    'AdvancedQualityScorer',
    'LangExtractOptimizer',
    
    # Enums
    'ModelVariant',
    'TaskComplexity',
    'QuantizationType',
    'InferenceBackend',
    'PoolingStrategy',
    'WarmupStrategy',
    'QualityDimension',
    'QualityLevel',
    'ProcessingMode',
    'ContentType',
    
    # Data classes
    'ModelConfig',
    'SystemResources',
    'InferenceRequest',
    'InferenceResponse',
    'QualityScore',
    'QualityAssessment',
    'ProcessingTask',
    'ProcessingResult',
    
    # Factory functions
    'create_optimized_model_pool',
    'create_quality_scorer',
    'create_langextract_optimizer',
    'create_integrated_suite',
    
    # Utilities
    'Qwen3IntegrationSuite',
    'quick_process'
]
