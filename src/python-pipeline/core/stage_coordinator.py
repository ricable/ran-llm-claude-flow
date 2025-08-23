"""
Stage Coordinator Implementation

Manages execution of individual pipeline stages with coordination logic,
error handling, and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..interfaces import (
    IStageCoordinator, IStage, IMemoryManager,
    ProcessingResult, ProcessingContext, ProcessingStatus
)
from ..interfaces.pipeline import StageConfig, StageExecutionMode


@dataclass
class StageExecutionMetrics:
    """Metrics for stage execution."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    processing_time: float = 0.0
    items_processed: int = 0
    items_successful: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    memory_usage_gb: float = 0.0
    cpu_utilization: float = 0.0
    error_count: int = 0
    warnings_count: int = 0
    
    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.items_processed == 0:
            return 0.0
        return (self.items_successful / self.items_processed) * 100.0
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (items/second)."""
        if self.processing_time == 0:
            return 0.0
        return self.items_processed / self.processing_time


class StageCoordinatorImpl(IStageCoordinator):
    """Default implementation of stage coordinator."""
    
    def __init__(self,
                 stage: IStage,
                 config: StageConfig,
                 memory_manager: IMemoryManager,
                 logger: logging.Logger):
        self._stage = stage
        self._config = config
        self._memory_manager = memory_manager
        self._logger = logger
        self._metrics = StageExecutionMetrics()
        self._execution_lock = asyncio.Lock()
        self._is_executing = False
        self._semaphore = asyncio.Semaphore(config.max_workers)
        
    @property
    def stage(self) -> IStage:
        return self._stage
    
    @property
    def config(self) -> StageConfig:
        return self._config
    
    async def execute_stage(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute stage with coordination logic."""
        async with self._execution_lock:
            if self._is_executing and self._config.execution_mode == StageExecutionMode.BLOCKING:
                self._logger.warning(f"Stage {self._stage.stage_id} is already executing in blocking mode")
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    errors=["Stage already executing in blocking mode"]
                )
            
            self._is_executing = True
        
        try:
            # Prepare stage for execution
            await self._prepare_stage_execution(context)
            
            # Execute based on execution mode
            if self._config.execution_mode == StageExecutionMode.BATCH:
                result = await self._execute_batch_mode(input_data, context)
            elif self._config.execution_mode == StageExecutionMode.STREAM:
                result = await self._execute_stream_mode(input_data, context)
            else:
                result = await self._execute_standard_mode(input_data, context)
            
            # Update metrics
            self._update_execution_metrics(result)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Stage execution failed: {e}", exc_info=True)
            return await self.handle_stage_failure(e, context)
        finally:
            async with self._execution_lock:
                self._is_executing = False
            await self.cleanup_stage(context)
    
    async def execute_batch_stage(self, 
                                input_batch: List[Any], 
                                context: ProcessingContext) -> List[ProcessingResult]:
        """Execute stage in batch mode."""
        if not input_batch:
            return []
        
        self._logger.info(f"Executing batch of {len(input_batch)} items in stage {self._stage.stage_id}")
        
        # Prepare for batch execution
        await self._prepare_stage_execution(context)
        
        results = []
        batch_size = self._config.batch_size
        
        # Process in smaller sub-batches to manage memory
        for i in range(0, len(input_batch), batch_size):
            sub_batch = input_batch[i:i + batch_size]
            sub_results = await self._process_batch_with_semaphore(sub_batch, context)
            results.extend(sub_results)
            
            # Check memory pressure between batches
            memory_usage = await self._memory_manager.get_global_usage()
            if memory_usage.utilization_percent > 85.0:
                self._logger.warning(f"High memory usage: {memory_usage.utilization_percent}%. Triggering optimization.")
                await self._memory_manager.optimize_allocation()
        
        self._logger.info(f"Batch execution completed. Processed {len(results)} items.")
        return results
    
    async def _process_batch_with_semaphore(self, 
                                          batch: List[Any], 
                                          context: ProcessingContext) -> List[ProcessingResult]:
        """Process batch with semaphore to limit concurrency."""
        async def process_item(item):
            async with self._semaphore:
                try:
                    # Create item-specific context
                    item_context = ProcessingContext(
                        stage_id=context.stage_id,
                        processor_id=context.processor_id,
                        config=context.config,
                        session_id=context.session_id,
                        correlation_id=f"{context.correlation_id}_{id(item)}"
                    )
                    
                    # Execute stage for single item
                    result = await self._stage.execute(item, item_context)
                    self._metrics.items_processed += 1
                    
                    if result.status == ProcessingStatus.COMPLETED:
                        self._metrics.items_successful += 1
                    elif result.status == ProcessingStatus.FAILED:
                        self._metrics.items_failed += 1
                    elif result.status == ProcessingStatus.SKIPPED:
                        self._metrics.items_skipped += 1
                    
                    return result
                    
                except Exception as e:
                    self._metrics.items_failed += 1
                    self._metrics.error_count += 1
                    self._logger.error(f"Error processing item: {e}")
                    return ProcessingResult(
                        status=ProcessingStatus.FAILED,
                        errors=[str(e)]
                    )
        
        # Process all items in batch concurrently
        tasks = [process_item(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self._logger.error(f"Task failed with exception: {result}")
                processed_results.append(ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def monitor_execution(self) -> Dict[str, Any]:
        """Monitor stage execution status."""
        memory_usage = await self._memory_manager.get_global_usage()
        
        return {
            'stage_id': self._stage.stage_id,
            'is_executing': self._is_executing,
            'execution_mode': self._config.execution_mode.value,
            'max_workers': self._config.max_workers,
            'current_semaphore_value': self._semaphore._value,
            'metrics': {
                'items_processed': self._metrics.items_processed,
                'items_successful': self._metrics.items_successful,
                'items_failed': self._metrics.items_failed,
                'error_count': self._metrics.error_count,
                'success_rate': self._metrics.success_rate,
                'throughput': self._metrics.throughput,
                'processing_time': self._metrics.processing_time
            },
            'memory': {
                'usage_gb': memory_usage.used_gb,
                'utilization_percent': memory_usage.utilization_percent
            },
            'config': {
                'batch_size': self._config.batch_size,
                'memory_limit_gb': self._config.memory_limit_gb,
                'timeout_seconds': self._config.timeout_seconds
            }
        }
    
    async def handle_stage_failure(self, error: Exception, context: ProcessingContext) -> ProcessingResult:
        """Handle stage execution failures."""
        self._metrics.error_count += 1
        self._logger.error(f"Stage {self._stage.stage_id} failed: {error}", exc_info=True)
        
        # Attempt recovery based on error type
        recovery_attempted = await self._attempt_recovery(error, context)
        
        return ProcessingResult(
            status=ProcessingStatus.FAILED,
            errors=[str(error)],
            metadata={
                'stage_id': self._stage.stage_id,
                'error_type': type(error).__name__,
                'recovery_attempted': recovery_attempted,
                'context': {
                    'correlation_id': context.correlation_id,
                    'session_id': context.session_id
                }
            }
        )
    
    async def cleanup_stage(self, context: ProcessingContext) -> None:
        """Cleanup stage resources."""
        try:
            # Finalize stage execution
            await self._stage.finalize(context)
            
            # Finalize metrics
            self._metrics.finalize()
            
            # Log final metrics
            self._logger.info(
                f"Stage {self._stage.stage_id} completed. "
                f"Processed: {self._metrics.items_processed}, "
                f"Success rate: {self._metrics.success_rate:.1f}%, "
                f"Throughput: {self._metrics.throughput:.2f} items/sec"
            )
            
            # Optional garbage collection for large stages
            if self._metrics.items_processed > 1000:
                gc_stats = await self._memory_manager.garbage_collect()
                self._logger.debug(f"Post-stage GC freed {gc_stats.get('freed_gb', 0)}GB")
                
        except Exception as e:
            self._logger.error(f"Error during stage cleanup: {e}")
    
    async def _prepare_stage_execution(self, context: ProcessingContext):
        """Prepare stage for execution."""
        self._metrics = StageExecutionMetrics()  # Reset metrics
        
        # Check memory availability
        memory_usage = await self._memory_manager.get_global_usage()
        if memory_usage.utilization_percent > 90.0:
            self._logger.warning("High memory usage detected. Optimizing before stage execution.")
            await self._memory_manager.optimize_allocation()
        
        # Prepare stage
        await self._stage.prepare(context)
        
        self._logger.info(f"Stage {self._stage.stage_id} prepared for execution")
    
    async def _execute_standard_mode(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute stage in standard mode."""
        # Apply timeout
        timeout = self._config.timeout_seconds
        
        try:
            result = await asyncio.wait_for(
                self._stage.execute(input_data, context),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            self._logger.error(f"Stage {self._stage.stage_id} timed out after {timeout} seconds")
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                errors=[f"Stage execution timed out after {timeout} seconds"]
            )
    
    async def _execute_batch_mode(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute stage in batch mode."""
        if isinstance(input_data, list):
            batch_results = await self.execute_batch_stage(input_data, context)
            
            # Aggregate batch results
            successful_count = sum(1 for r in batch_results if r.status == ProcessingStatus.COMPLETED)
            failed_count = sum(1 for r in batch_results if r.status == ProcessingStatus.FAILED)
            
            if failed_count == 0:
                status = ProcessingStatus.COMPLETED
            elif successful_count == 0:
                status = ProcessingStatus.FAILED
            else:
                status = ProcessingStatus.COMPLETED  # Partial success
            
            return ProcessingResult(
                status=status,
                data=[r.data for r in batch_results],
                metadata={
                    'batch_size': len(input_data),
                    'successful_count': successful_count,
                    'failed_count': failed_count,
                    'individual_results': batch_results
                }
            )
        else:
            # Single item - execute normally
            return await self._execute_standard_mode(input_data, context)
    
    async def _execute_stream_mode(self, input_data: Any, context: ProcessingContext) -> ProcessingResult:
        """Execute stage in streaming mode."""
        # For now, treat as standard mode
        # Full streaming implementation would require AsyncIterator support
        self._logger.info(f"Stream mode not fully implemented, falling back to standard mode")
        return await self._execute_standard_mode(input_data, context)
    
    async def _attempt_recovery(self, error: Exception, context: ProcessingContext) -> bool:
        """Attempt recovery from stage failure."""
        recovery_strategies = {
            MemoryError: self._recover_from_memory_error,
            TimeoutError: self._recover_from_timeout,
            ConnectionError: self._recover_from_connection_error
        }
        
        strategy = recovery_strategies.get(type(error))
        if strategy:
            try:
                await strategy(error, context)
                return True
            except Exception as recovery_error:
                self._logger.error(f"Recovery attempt failed: {recovery_error}")
        
        return False
    
    async def _recover_from_memory_error(self, error: MemoryError, context: ProcessingContext):
        """Recover from memory errors."""
        self._logger.info("Attempting memory error recovery")
        
        # Force garbage collection and memory optimization
        await self._memory_manager.garbage_collect(force=True)
        
        # Reduce batch size if applicable
        if self._config.batch_size > 10:
            self._config.batch_size = max(10, self._config.batch_size // 2)
            self._logger.info(f"Reduced batch size to {self._config.batch_size}")
    
    async def _recover_from_timeout(self, error: TimeoutError, context: ProcessingContext):
        """Recover from timeout errors."""
        self._logger.info("Attempting timeout error recovery")
        
        # Increase timeout for retry
        self._config.timeout_seconds = int(self._config.timeout_seconds * 1.5)
        self._logger.info(f"Increased timeout to {self._config.timeout_seconds} seconds")
    
    async def _recover_from_connection_error(self, error: ConnectionError, context: ProcessingContext):
        """Recover from connection errors."""
        self._logger.info("Attempting connection error recovery")
        
        # Wait and retry
        await asyncio.sleep(5)
    
    def _update_execution_metrics(self, result: ProcessingResult):
        """Update execution metrics based on result."""
        if result.status == ProcessingStatus.COMPLETED:
            self._metrics.items_successful += 1
        elif result.status == ProcessingStatus.FAILED:
            self._metrics.items_failed += 1
        elif result.status == ProcessingStatus.SKIPPED:
            self._metrics.items_skipped += 1
        
        self._metrics.items_processed += 1
        
        # Update error and warning counts
        if result.errors:
            self._metrics.error_count += len(result.errors)
        
        # Extract metrics from result if available
        if result.metrics:
            self._metrics.memory_usage_gb = result.metrics.get('memory_usage_gb', 0)
            self._metrics.cpu_utilization = result.metrics.get('cpu_utilization', 0)