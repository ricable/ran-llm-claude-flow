#!/usr/bin/env python3
"""
Python ML Engine Main Integration Script

Main entry point for the Python ML engine that coordinates all components:
- Model management with Qwen3 variants
- Semantic processing for QA generation
- MLX acceleration for M3 Max optimization
- IPC communication with Rust core

Usage:
    python -m src.main --config config/models_config.yaml
    python -m src.main --test  # Run built-in tests
    python -m src.main --benchmark  # Run performance benchmarks

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import logging
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import yaml

# Core ML Engine imports
from .model_manager import Qwen3ModelManager, ModelSize, ProcessingHints
from .semantic_processor import SemanticProcessor, DocumentAnalysis, QAPair
from .mlx_accelerator import MLXAccelerator
from .ipc_client import IPCClient, DocumentProcessingRequest, ProcessingResponse

# System utilities
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from . import get_system_info, get_performance_recommendations, print_initialization_info


class PythonMLEngine:
    """
    Main Python ML Engine coordinator.
    
    Integrates all components and provides a unified interface for
    document processing and QA generation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/models_config.yaml")
        self.config = {}
        
        # Components
        self.model_manager: Optional[Qwen3ModelManager] = None
        self.semantic_processor: Optional[SemanticProcessor] = None
        self.mlx_accelerator: Optional[MLXAccelerator] = None
        self.ipc_client: Optional[IPCClient] = None
        
        # State
        self.initialized = False
        self.performance_stats = {
            'documents_processed': 0,
            'qa_pairs_generated': 0,
            'total_processing_time': 0.0,
            'start_time': time.time()
        }
        
    def load_config(self) -> bool:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Use default configuration
                self.config = self._get_default_config()
                self.logger.warning("Using default configuration - config file not found")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'memory': {'total_budget_gb': 45.0},
            'models': {
                'qwen3_1_5b': {
                    'size': '1.5B',
                    'backend': 'mlx',
                    'model_path': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'memory_budget_gb': 4.0
                }
            },
            'performance': {
                'batching': {'max_batch_size': 8},
                'mlx': {'enable_metal_acceleration': True}
            },
            'quality': {
                'thresholds': {'target': 0.75}
            },
            'integration': {
                'targets': {
                    'documents_per_hour': 25,
                    'qa_pairs_per_document': 5
                }
            }
        }
        
    async def initialize(self) -> bool:
        """Initialize all ML engine components"""
        self.logger.info("Initializing Python ML Engine")
        
        try:
            # Load configuration
            if not self.load_config():
                return False
                
            # Initialize MLX accelerator
            memory_budget = self.config.get('memory', {}).get('total_budget_gb', 45.0)
            self.mlx_accelerator = MLXAccelerator(memory_budget_gb=memory_budget)
            await self.mlx_accelerator.initialize()
            
            # Initialize model manager
            self.model_manager = Qwen3ModelManager(self.config_path)
            await self.model_manager.initialize()
            
            # Initialize semantic processor
            self.semantic_processor = SemanticProcessor(self.model_manager)
            await self.semantic_processor.initialize()
            
            # Initialize IPC client
            ipc_config = self.config.get('integration', {}).get('ipc', {})
            pipe_path = ipc_config.get('pipe_path', '/tmp/rust_python_ipc')
            self.ipc_client = IPCClient(pipe_path)
            await self.ipc_client.initialize()
            
            self.initialized = True
            self.logger.info("Python ML Engine initialized successfully")
            
            # Print system information
            await self._log_system_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML engine: {e}")
            return False
            
    async def _log_system_info(self):
        """Log system information and performance expectations"""
        system_info = get_system_info()
        
        self.logger.info("System Information:")
        self.logger.info(f"  Platform: {system_info['platform']} {system_info['architecture']}")
        self.logger.info(f"  Memory: {system_info['memory_gb']}GB")
        self.logger.info(f"  CPU Cores: {system_info['cpu_count']}")
        self.logger.info(f"  MLX Available: {system_info['mlx_available']}")
        self.logger.info(f"  Apple Silicon: {system_info['apple_silicon']}")
        
        # Performance expectations
        targets = self.config.get('integration', {}).get('targets', {})
        self.logger.info("Performance Targets:")
        self.logger.info(f"  Documents/hour: {targets.get('documents_per_hour', 25)}")
        self.logger.info(f"  QA pairs/document: {targets.get('qa_pairs_per_document', 5)}")
        self.logger.info(f"  Quality threshold: {targets.get('quality_threshold', 0.75)}")
        
    async def process_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        qa_target_count: int = 5,
        quality_threshold: float = 0.75
    ) -> ProcessingResponse:
        """
        Process a single document through the complete ML pipeline.
        
        Args:
            content: Document content
            metadata: Document metadata
            qa_target_count: Target number of QA pairs
            quality_threshold: Minimum quality threshold
            
        Returns:
            Processing response with QA pairs and metrics
        """
        if not self.initialized:
            raise RuntimeError("ML engine not initialized")
            
        start_time = time.time()
        document_id = metadata.get('document_id', f"doc_{int(start_time)}")
        
        self.logger.info(f"Processing document {document_id}")
        
        try:
            # 1. Analyze document semantically
            analysis = self.semantic_processor.analyze_document(content, metadata)
            
            self.logger.debug(
                f"Document analysis: complexity={analysis.complexity:.2f}, "
                f"technical_terms={len(analysis.technical_terms)}, "
                f"parameters={len(analysis.parameters)}"
            )
            
            # 2. Generate QA pairs
            qa_pairs = await self.semantic_processor.generate_qa_pairs(
                analysis, content, qa_target_count, quality_threshold
            )
            
            # 3. Calculate processing metrics
            processing_time = time.time() - start_time
            avg_quality = sum(qa.quality_metrics.overall_score for qa in qa_pairs) / max(len(qa_pairs), 1)
            
            # 4. Update performance statistics
            self._update_performance_stats(processing_time, len(qa_pairs))
            
            # 5. Create response
            response = ProcessingResponse(
                document_id=document_id,
                success=True,
                qa_pairs=[qa.to_dict() for qa in qa_pairs],
                quality_metrics={
                    'average_quality': avg_quality,
                    'quality_distribution': self._calculate_quality_distribution(qa_pairs),
                    'document_complexity': analysis.complexity,
                    'technical_density': analysis.technical_density
                },
                processing_time=processing_time,
                model_used=getattr(self.model_manager, 'last_used_model', 'unknown')
            )
            
            self.logger.info(
                f"Document {document_id} processed successfully: "
                f"{len(qa_pairs)} QA pairs, avg_quality={avg_quality:.2f}, "
                f"time={processing_time:.1f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process document {document_id}: {e}")
            return ProcessingResponse(
                document_id=document_id,
                success=False,
                error_message=str(e)
            )
            
    def _calculate_quality_distribution(self, qa_pairs: List[QAPair]) -> Dict[str, int]:
        """Calculate quality score distribution"""
        distribution = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
        
        for qa in qa_pairs:
            score = qa.quality_metrics.overall_score
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.8:
                distribution['good'] += 1
            elif score >= 0.6:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1
                
        return distribution
        
    def _update_performance_stats(self, processing_time: float, qa_count: int):
        """Update performance statistics"""
        self.performance_stats['documents_processed'] += 1
        self.performance_stats['qa_pairs_generated'] += qa_count
        self.performance_stats['total_processing_time'] += processing_time
        
    async def process_batch(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        qa_per_document: int = 5
    ) -> List[ProcessingResponse]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of (content, metadata) tuples
            qa_per_document: Target QA pairs per document
            
        Returns:
            List of processing responses
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        # Use semantic processor's batch processing
        batch_results = await self.semantic_processor.process_batch(
            documents, qa_per_document
        )
        
        # Convert to ProcessingResponse format
        responses = []
        for result in batch_results:
            response = ProcessingResponse(
                document_id=result['document_id'],
                success=True,
                qa_pairs=result['qa_pairs'],
                quality_metrics=result.get('document_analysis', {}),
                processing_time=result['processing_metrics']['processing_time'],
                model_used='batch_processed'
            )
            responses.append(response)
            
        return responses
        
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        uptime = time.time() - self.performance_stats['start_time']
        
        report = {
            'engine_stats': dict(self.performance_stats),
            'system_info': get_system_info(),
            'uptime_seconds': uptime,
            'performance_metrics': {}
        }
        
        # Add component-specific metrics
        if self.model_manager:
            report['performance_metrics']['model_manager'] = self.model_manager.get_model_status()
            report['performance_metrics']['memory_usage'] = self.model_manager.get_memory_usage()
            
        if self.mlx_accelerator:
            report['performance_metrics']['mlx_accelerator'] = self.mlx_accelerator.get_performance_report()
            
        if self.ipc_client:
            report['performance_metrics']['ipc_client'] = self.ipc_client.get_local_metrics()
            
        # Calculate derived metrics
        if self.performance_stats['documents_processed'] > 0:
            avg_time = self.performance_stats['total_processing_time'] / self.performance_stats['documents_processed']
            docs_per_hour = 3600 / max(avg_time, 0.001)
            
            report['derived_metrics'] = {
                'avg_processing_time': avg_time,
                'documents_per_hour': docs_per_hour,
                'qa_pairs_per_document': self.performance_stats['qa_pairs_generated'] / self.performance_stats['documents_processed']
            }
            
        return report
        
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            'healthy': True,
            'issues': [],
            'warnings': [],
            'component_status': {}
        }
        
        # Check initialization
        if not self.initialized:
            health['healthy'] = False
            health['issues'].append('Engine not initialized')
            return health
            
        # Check components
        try:
            if self.model_manager:
                model_health = await self.model_manager.health_check()
                health['component_status']['model_manager'] = model_health
                if not model_health.get('healthy', True):
                    health['warnings'].append('Model manager issues detected')
                    
            if self.mlx_accelerator:
                mlx_health = await self.mlx_accelerator.health_check()
                health['component_status']['mlx_accelerator'] = mlx_health
                if not mlx_health.get('healthy', True):
                    health['warnings'].append('MLX accelerator issues detected')
                    
            if self.ipc_client:
                ipc_health = await self.ipc_client.health_check()
                health['component_status']['ipc_client'] = ipc_health
                if not ipc_health.get('healthy', True):
                    health['warnings'].append('IPC client issues detected')
                    
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f'Health check failed: {str(e)}')
            
        # System-level checks
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            health['warnings'].append('High system memory usage')
            
        health['timestamp'] = time.time()
        return health
        
    async def cleanup(self):
        """Clean up all resources"""
        self.logger.info("Cleaning up Python ML Engine")
        
        cleanup_tasks = []
        
        if self.ipc_client:
            cleanup_tasks.append(self.ipc_client.cleanup())
        if self.mlx_accelerator:
            cleanup_tasks.append(self.mlx_accelerator.cleanup())
        if self.model_manager:
            cleanup_tasks.append(self.model_manager.cleanup())
            
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        self.initialized = False
        self.logger.info("Python ML Engine cleanup complete")


async def run_tests(engine: PythonMLEngine):
    """Run built-in tests"""
    print("🧪 Running Python ML Engine tests...")
    
    # Test document processing
    test_content = """
    # LTE Handover Procedures
    
    ## Overview
    LTE handover procedures enable seamless mobility for UE devices across eNodeB coverage areas.
    
    ## Parameters
    - **EUtranCellFDD.a3Offset**: Offset value for A3 event reporting
      - Range: -30 to 30 dB
      - Default: 0 dB
    - **EUtranCellFDD.hysteresisA3**: Hysteresis for A3 measurements
      - Range: 0 to 15 dB
      - Default: 2 dB
    
    ## Process
    1. UE performs measurements on serving and neighbor cells
    2. A3 event is triggered when neighbor becomes stronger
    3. eNodeB initiates handover preparation
    4. UE switches to target cell
    """
    
    metadata = {
        'document_id': 'test_handover_001',
        'feature_name': 'LTE Handover',
        'source': 'test'
    }
    
    try:
        response = await engine.process_document(content=test_content, metadata=metadata, qa_target_count=3)
        
        if response.success:
            print(f"✅ Test passed: Generated {len(response.qa_pairs)} QA pairs")
            print(f"   Average quality: {response.quality_metrics.get('average_quality', 0):.2f}")
            print(f"   Processing time: {response.processing_time:.2f}s")
            
            # Show sample QA pair
            if response.qa_pairs:
                sample = response.qa_pairs[0]
                print(f"   Sample Q: {sample['question'][:80]}...")
                print(f"   Sample A: {sample['answer'][:80]}...")
        else:
            print(f"❌ Test failed: {response.error_message}")
            
    except Exception as e:
        print(f"❌ Test exception: {e}")
        
    # Health check test
    health = await engine.health_check()
    print(f"🏥 Health check: {'✅ Healthy' if health['healthy'] else '❌ Issues detected'}")
    
    # Performance report
    report = await engine.get_performance_report()
    print("📊 Performance Report:")
    print(f"   Documents processed: {report['engine_stats']['documents_processed']}")
    print(f"   QA pairs generated: {report['engine_stats']['qa_pairs_generated']}")
    

async def run_benchmark(engine: PythonMLEngine):
    """Run performance benchmark"""
    print("🏃 Running performance benchmark...")
    
    # Generate test documents
    test_docs = []
    for i in range(5):
        content = f"""# Test Feature {i+1}
        
This is a test document for benchmarking purposes. It contains technical content 
about telecommunications features and parameters.
        
## Parameters
- Parameter{i}A: Test parameter A with range 0-100
- Parameter{i}B: Test parameter B with boolean values
        
## Description
Detailed description of feature {i+1} with technical specifications and use cases.
"""
        metadata = {'document_id': f'benchmark_{i+1}', 'feature_name': f'Test Feature {i+1}'}
        test_docs.append((content, metadata))
        
    start_time = time.time()
    
    try:
        responses = await engine.process_batch(test_docs, qa_per_document=3)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in responses if r.success)
        total_qa = sum(len(r.qa_pairs) for r in responses)
        
        print(f"📈 Benchmark Results:")
        print(f"   Documents: {len(test_docs)} ({successful} successful)")
        print(f"   Total QA pairs: {total_qa}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Documents/hour: {len(test_docs) / total_time * 3600:.1f}")
        print(f"   Avg time per doc: {total_time / len(test_docs):.1f}s")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('python_ml_engine.log')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Python ML Engine')
    parser.add_argument('--config', type=Path, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run built-in tests')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    # Print initialization info
    print_initialization_info()
    
    # Initialize engine
    engine = PythonMLEngine(args.config)
    
    try:
        success = await engine.initialize()
        if not success:
            print("❌ Failed to initialize ML engine")
            sys.exit(1)
            
        if args.test:
            await run_tests(engine)
        elif args.benchmark:
            await run_benchmark(engine)
        elif args.interactive:
            print("🔄 Interactive mode - engine ready for requests")
            # Keep running for external requests
            while True:
                await asyncio.sleep(1)
        else:
            # Default: show status and exit
            health = await engine.health_check()
            report = await engine.get_performance_report()
            
            print("📊 Engine Status:")
            print(f"   Healthy: {'✅' if health['healthy'] else '❌'}")
            print(f"   Components: {len(health['component_status'])}")
            print(f"   Memory: {report['system_info']['memory_gb']}GB")
            print(f"   MLX Available: {'✅' if report['system_info']['mlx_available'] else '❌'}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await engine.cleanup()
        

if __name__ == "__main__":
    asyncio.run(main())
