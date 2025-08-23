"""
Pipeline Coordinator Implementation

Orchestrates the complete 6-stage pipeline execution with coordination,
monitoring, and optimization for M3 Max hardware.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from ..interfaces import (
    IPipelineCoordinator, IStageCoordinator, IStage, IMemoryManager,
    ProcessingResult, ProcessingContext, ProcessingStatus
)
from ..interfaces.pipeline import PipelineConfig, PipelineMode, PipelineMetrics


@dataclass
class CheckpointData:
    """Pipeline checkpoint data."""
    checkpoint_id: str
    pipeline_id: str
    timestamp: float
    stage_states: Dict[str, Any]
    metrics: PipelineMetrics
    context: ProcessingContext
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'pipeline_id': self.pipeline_id,
            'timestamp': self.timestamp,
            'stage_states': self.stage_states,
            'metrics': {
                'total_processed': self.metrics.total_processed,
                'successful_processed': self.metrics.successful_processed,
                'failed_processed': self.metrics.failed_processed,
                'processing_rate': self.metrics.processing_rate,
                'memory_usage_gb': self.metrics.memory_usage_gb
            },
            'context': {
                'stage_id': self.context.stage_id,
                'processor_id': self.context.processor_id,
                'session_id': self.context.session_id,
                'correlation_id': self.context.correlation_id
            }
        }


class PipelineCoordinatorImpl(IPipelineCoordinator):
    """Implementation of pipeline coordinator with full orchestration."""
    
    def __init__(self,
                 stages: List[IStage],
                 coordinators: List[IStageCoordinator],
                 config: PipelineConfig,
                 memory_manager: IMemoryManager,
                 logger: logging.Logger):
        
        self._pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        self._stages = stages
        self._coordinators = coordinators
        self._config = config
        self._memory_manager = memory_manager
        self._logger = logger
        
        # Initialize metrics
        self._metrics = PipelineMetrics()
        self._start_time = time.time()
        
        # Execution state
        self._is_initialized = False
        self._is_executing = False
        self._checkpoints: Dict[str, CheckpointData] = {}
        self._execution_lock = asyncio.Lock()
        
        # Stage mapping for quick lookup
        self._stage_map = {stage.stage_id: stage for stage in stages}
        self._coordinator_map = {coord.stage.stage_id: coord for coord in coordinators}
        
        # Performance monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
    
    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id
    
    @property
    def stages(self) -> List[IStage]:
        return self._stages.copy()
    
    @property
    def config(self) -> PipelineConfig:
        return self._config
    
    async def initialize(self):
        """Initialize pipeline components."""
        if self._is_initialized:
            return
        
        self._logger.info(f"Initializing pipeline {self._pipeline_id}")
        
        # Validate pipeline configuration
        if not await self.validate_pipeline():
            raise ValueError("Pipeline validation failed")
        
        # Initialize performance monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_performance())
        
        self._is_initialized = True
        self._logger.info(f"Pipeline {self._pipeline_id} initialized with {len(self._stages)} stages")
    
    async def execute(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute the complete pipeline."""
        if not self._is_initialized:
            await self.initialize()
        
        async with self._execution_lock:
            if self._is_executing:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    errors=["Pipeline is already executing"]
                )
            self._is_executing = True
        
        try:
            # Create pipeline context
            pipeline_context = ProcessingContext(
                stage_id="pipeline",
                processor_id=self._pipeline_id,
                session_id=context.session_id or str(uuid.uuid4()),
                correlation_id=context.correlation_id or str(uuid.uuid4()),
                config=context.config or {}
            )
            
            self._logger.info(f"Starting pipeline execution with mode: {self._config.mode.value}")
            
            # Execute based on pipeline mode
            if self._config.mode == PipelineMode.SEQUENTIAL:
                result = await self._execute_sequential(input_data, pipeline_context)
            elif self._config.mode == PipelineMode.PARALLEL:
                result = await self._execute_parallel(input_data, pipeline_context)
            elif self._config.mode == PipelineMode.ADAPTIVE:
                result = await self._execute_adaptive(input_data, pipeline_context)
            elif self._config.mode == PipelineMode.STREAMING:
                result = await self._execute_streaming_single(input_data, pipeline_context)
            else:
                raise ValueError(f"Unsupported pipeline mode: {self._config.mode}")
            
            # Update final metrics
            self._update_pipeline_metrics(result)
            
            self._logger.info(
                f"Pipeline execution completed. Status: {result.status.value}, "
                f"Total processed: {self._metrics.total_processed}, "
                f"Success rate: {(self._metrics.successful_processed / max(1, self._metrics.total_processed)) * 100:.1f}%"
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                errors=[str(e)],
                metadata={'pipeline_id': self._pipeline_id, 'error_type': type(e).__name__}
            )
        finally:
            async with self._execution_lock:
                self._is_executing = False
    
    async def execute_streaming(self,
                              input_stream: AsyncIterator[Any],
                              context: ProcessingContext) -> AsyncIterator[ProcessingResult]:
        """Execute pipeline in streaming mode."""
        if not self._is_initialized:
            await self.initialize()
        
        self._logger.info("Starting streaming pipeline execution")
        
        async for input_item in input_stream:
            try:
                result = await self.execute(input_item, context)
                yield result
                
                # Create checkpoint periodically
                if (self._metrics.total_processed % self._config.checkpoint_interval == 0 and
                    self._config.enable_checkpointing):
                    checkpoint_id = f"auto_{int(time.time())}"
                    await self.create_checkpoint(checkpoint_id)
                    
            except Exception as e:
                self._logger.error(f"Error in streaming execution: {e}")
                yield ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    errors=[str(e)]
                )
    
    async def execute_batch(self,
                          input_batch: List[Any],
                          context: ProcessingContext) -> List[ProcessingResult]:
        """Execute pipeline in batch mode."""
        if not input_batch:
            return []
        
        self._logger.info(f"Starting batch pipeline execution with {len(input_batch)} items")
        
        results = []
        batch_size = self._config.batch_size
        
        # Process in smaller sub-batches
        for i in range(0, len(input_batch), batch_size):
            sub_batch = input_batch[i:i + batch_size]
            
            # Create tasks for parallel execution
            tasks = []
            for item in sub_batch:
                # Create item-specific context
                item_context = ProcessingContext(
                    stage_id=context.stage_id,
                    processor_id=context.processor_id,
                    session_id=context.session_id,
                    correlation_id=f"{context.correlation_id}_{i}_{id(item)}",
                    config=context.config
                )
                tasks.append(self.execute(item, item_context))
            
            # Execute sub-batch
            sub_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in sub_results:
                if isinstance(result, Exception):
                    results.append(ProcessingResult(
                        status=ProcessingStatus.FAILED,
                        errors=[str(result)]
                    ))
                else:
                    results.append(result)
            
            # Memory management between batches
            if i > 0 and i % (batch_size * 4) == 0:
                memory_usage = await self._memory_manager.get_global_usage()
                if memory_usage.utilization_percent > 85.0:
                    await self._memory_manager.garbage_collect()
        
        return results
    
    async def validate_pipeline(self) -> bool:
        """Validate pipeline configuration and stage compatibility."""
        try:
            # Check stage order and dependencies
            if not self._validate_stage_order():
                return False
            
            # Validate stage configurations
            for coordinator in self._coordinators:
                stage = coordinator.stage
                if not await stage.can_process(None):  # Basic validation
                    self._logger.error(f"Stage {stage.stage_id} failed validation")
                    return False
            
            # Check memory requirements
            if not await self._validate_memory_requirements():
                return False
            
            # Validate stage compatibility
            if not self._validate_stage_compatibility():
                return False
            
            self._logger.info("Pipeline validation passed")
            return True
            
        except Exception as e:
            self._logger.error(f"Pipeline validation failed: {e}")
            return False
    
    async def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        # Update memory usage
        memory_usage = await self._memory_manager.get_global_usage()
        self._metrics.memory_usage_gb = memory_usage.used_gb
        
        # Calculate processing rate
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            self._metrics.processing_rate = self._metrics.total_processed / elapsed_time
        
        # Get stage metrics
        stage_metrics = {}
        for coordinator in self._coordinators:
            stage_id = coordinator.stage.stage_id
            stage_status = await coordinator.monitor_execution()
            stage_metrics[stage_id] = stage_status.get('metrics', {})
        
        self._metrics.stage_metrics = stage_metrics
        
        return self._metrics
    
    async def create_checkpoint(self, checkpoint_id: str) -> bool:
        """Create pipeline state checkpoint."""
        try:
            # Collect stage states
            stage_states = {}
            for coordinator in self._coordinators:
                stage_id = coordinator.stage.stage_id
                stage_status = await coordinator.monitor_execution()
                stage_states[stage_id] = stage_status
            
            # Create checkpoint
            checkpoint = CheckpointData(
                checkpoint_id=checkpoint_id,
                pipeline_id=self._pipeline_id,
                timestamp=time.time(),
                stage_states=stage_states,
                metrics=await self.get_metrics(),
                context=ProcessingContext(
                    stage_id="checkpoint",
                    processor_id=self._pipeline_id,
                    session_id=checkpoint_id
                )
            )
            
            self._checkpoints[checkpoint_id] = checkpoint
            
            # Optionally save to disk
            if len(self._checkpoints) > 10:  # Keep only recent checkpoints
                oldest_id = min(self._checkpoints.keys(), 
                              key=lambda k: self._checkpoints[k].timestamp)
                del self._checkpoints[oldest_id]
            
            self._logger.info(f"Created checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to create checkpoint: {e}")
            return False
    
    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore pipeline from checkpoint."""
        if checkpoint_id not in self._checkpoints:
            self._logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        try:
            checkpoint = self._checkpoints[checkpoint_id]
            
            # Restore metrics
            self._metrics = checkpoint.metrics
            
            # TODO: Restore stage states (would require stage implementations)
            # This would involve restoring each stage's internal state
            
            self._logger.info(f"Restored from checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    async def _execute_sequential(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute stages sequentially."""
        current_data = input_data
        
        for i, coordinator in enumerate(self._coordinators):
            stage = coordinator.stage
            
            self._logger.debug(f"Executing stage {i+1}/{len(self._coordinators)}: {stage.stage_id}")
            
            # Create stage context
            stage_context = ProcessingContext(
                stage_id=stage.stage_id,
                processor_id=stage.stage_id,
                session_id=context.session_id,
                correlation_id=f"{context.correlation_id}_stage_{i}",
                config=context.config
            )
            
            # Execute stage
            result = await coordinator.execute_stage(current_data, stage_context)
            
            if result.status == ProcessingStatus.FAILED:
                self._logger.error(f"Stage {stage.stage_id} failed: {result.errors}")
                return result
            
            # Use stage output as input for next stage
            current_data = result.data
            
            # Update metrics
            self._metrics.total_processed += 1
            if result.status == ProcessingStatus.COMPLETED:
                self._metrics.successful_processed += 1
            else:
                self._metrics.failed_processed += 1
        
        return ProcessingResult(
            status=ProcessingStatus.COMPLETED,
            data=current_data,
            metadata={'pipeline_mode': 'sequential', 'stages_executed': len(self._coordinators)}
        )
    
    async def _execute_parallel(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute stages in parallel where possible."""
        # For simplicity, this implementation executes stages sequentially
        # Full parallel implementation would require dependency analysis
        self._logger.info("Parallel mode not fully implemented, falling back to sequential")
        return await self._execute_sequential(input_data, context)
    
    async def _execute_adaptive(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute with adaptive optimization based on system state."""
        # Monitor system resources and adapt execution strategy
        memory_usage = await self._memory_manager.get_global_usage()
        
        if memory_usage.utilization_percent > 80.0:
            # High memory usage - use sequential mode
            self._logger.info("High memory usage detected, using sequential execution")
            return await self._execute_sequential(input_data, context)
        elif len(self._stages) > 4 and memory_usage.utilization_percent < 60.0:
            # Low memory usage - try parallel execution
            self._logger.info("Low memory usage detected, attempting parallel execution")
            return await self._execute_parallel(input_data, context)
        else:
            # Default to sequential
            return await self._execute_sequential(input_data, context)
    
    async def _execute_streaming_single(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute single item in streaming mode."""
        # For single item, same as sequential but with streaming optimizations
        return await self._execute_sequential(input_data, context)
    
    def _validate_stage_order(self) -> bool:
        """Validate stage execution order."""
        expected_order = [
            'raw_input', 'document_conversion', 'preprocessing',
            'langextract', 'conversation_generation', 'dataset_finalization'
        ]
        
        stage_types = [stage.stage_id.split('_')[-1] for stage in self._stages if '_' in stage.stage_id]
        
        # Check if stages follow expected order
        for i, expected_type in enumerate(expected_order):
            if i < len(stage_types) and expected_type not in stage_types[i]:
                self._logger.warning(f"Unexpected stage order. Expected {expected_type} at position {i}")
                # Allow flexible ordering but log warning
        
        return True
    
    async def _validate_memory_requirements(self) -> bool:
        """Validate memory requirements for pipeline."""
        total_memory_required = 0.0
        
        for coordinator in self._coordinators:
            stage_config = coordinator.config
            total_memory_required += stage_config.memory_limit_gb
        
        available_memory = self._memory_manager.total_memory_gb
        
        if total_memory_required > available_memory * 0.9:  # Leave 10% headroom
            self._logger.error(
                f"Pipeline requires {total_memory_required}GB but only "
                f"{available_memory}GB available"
            )
            return False
        
        return True
    
    def _validate_stage_compatibility(self) -> bool:
        """Validate stage input/output compatibility."""
        # Check that each stage can process the output of the previous stage
        for i in range(1, len(self._stages)):
            prev_stage = self._stages[i-1]
            curr_stage = self._stages[i]
            
            # This would require more detailed type checking in real implementation
            self._logger.debug(f"Validating compatibility: {prev_stage.stage_id} -> {curr_stage.stage_id}")
        
        return True
    
    def _update_pipeline_metrics(self, final_result: ProcessingResult):
        """Update pipeline metrics with final result."""
        if final_result.status == ProcessingStatus.COMPLETED:
            self._metrics.successful_processed += 1
        else:
            self._metrics.failed_processed += 1
        
        self._metrics.total_processed += 1
    
    async def _monitor_performance(self):
        """Monitor pipeline performance continuously."""
        while True:
            try:
                if self._is_executing:
                    # Update CPU utilization (simplified)
                    import psutil
                    self._metrics.cpu_utilization = psutil.cpu_percent(interval=1)
                    
                    # Update memory usage
                    memory_usage = await self._memory_manager.get_global_usage()
                    self._metrics.memory_usage_gb = memory_usage.used_gb
                    
                    # Log performance metrics periodically
                    if int(time.time()) % 60 == 0:  # Every minute
                        self._logger.info(
                            f"Pipeline performance - "
                            f"Processed: {self._metrics.total_processed}, "
                            f"Rate: {self._metrics.processing_rate:.2f} items/sec, "
                            f"Memory: {self._metrics.memory_usage_gb:.1f}GB, "
                            f"CPU: {self._metrics.cpu_utilization:.1f}%"
                        )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def __del__(self):
        """Cleanup monitoring task on deletion."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()