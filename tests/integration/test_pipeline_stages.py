"""
Integration tests for pipeline stages
Testing complete pipeline workflows with real data flow
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import modules under test
import sys
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.append(str(src_path))

from m3_optimizer import M3PipelineOptimizer, OptimizationConfig
from performance_optimization import OptimizedDatasetProcessor
from quality_control_framework import QualityController, validate_dataset_batch


@pytest.mark.integration
class TestPipelineStagesIntegration:
    """Integration tests for complete pipeline stages"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.optimization_config = OptimizationConfig(
            max_memory_usage_percent=75.0,
            preferred_batch_size=256,
            max_concurrent_requests=4
        )
        
        # Create test dataset
        self.create_test_dataset()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def create_test_dataset(self):
        """Create test dataset for integration testing"""
        test_records = []
        
        # Create diverse test records
        for i in range(50):
            quality_score = 7.0 + (i % 3) * 0.5  # Varies between 7.0-8.0
            
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"How do I configure the eNodeB parameter {i} for LTE optimization?"
                    },
                    {
                        "role": "assistant",
                        "content": f"To configure eNodeB parameter {i}, set the RSRP threshold to -{100+i} dBm and time-to-trigger to {300+i*10}ms for optimal LTE performance."
                    }
                ],
                "metadata": {
                    "feature_name": f"LTE Feature {i}",
                    "quality_score": quality_score,
                    "technical_content": True,
                    "technical_terms": ["eNodeB", "LTE", "RSRP"],
                    "test_id": f"integration_test_{i}"
                }
            }
            test_records.append(record)
        
        # Save test dataset
        test_file = self.test_data_dir / "test_dataset.jsonl"
        with open(test_file, 'w') as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
        
        return test_file
    
    @pytest.mark.slow
    def test_quality_validation_pipeline_stage(self):
        """Test quality validation as a complete pipeline stage"""
        # Setup quality controller
        quality_controller = QualityController({
            "min_quality_score": 7.5,
            "technical_term_threshold": 2
        })
        
        # Load test data
        test_file = self.test_data_dir / "test_dataset.jsonl"
        records = []
        
        with open(test_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        
        # Process through quality validation
        valid_records, stats = validate_dataset_batch(records, quality_controller)
        
        # Verify pipeline stage results
        assert len(valid_records) > 0
        assert len(valid_records) <= len(records)  # Some may be filtered
        assert stats["total_processed"] == len(records)
        assert stats["valid_records"] + stats["invalid_records"] == len(records)
        assert stats["average_quality"] > 0.0
        
        # Verify quality filtering worked
        for record in valid_records:
            is_valid, metrics, _ = quality_controller.validate_record(record)
            assert is_valid is True
            assert metrics.overall_score >= 7.5
    
    @pytest.mark.slow
    def test_performance_optimization_pipeline_stage(self):
        """Test performance optimization as a complete pipeline stage"""
        # Setup optimized processor
        processor = OptimizedDatasetProcessor(
            max_memory_mb=512,
            max_workers=2,
            processing_strategy="thread",
            cache_enabled=True,
            batch_size=10
        )
        
        def test_processing_function(batch):
            """Simulate processing work"""
            import time
            time.sleep(0.01)  # Simulate processing time
            
            processed_batch = []
            for record in batch:
                processed_record = record.copy()
                processed_record["processed"] = True
                processed_record["processing_timestamp"] = time.time()
                processed_batch.append(processed_record)
            
            return processed_batch
        
        # Process test dataset
        input_files = [self.test_data_dir / "test_dataset.jsonl"]
        output_file = self.test_data_dir / "processed_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
            mock_process_instance.cpu_percent.return_value = 45.0
            mock_process.return_value = mock_process_instance
            
            metrics = processor.process_large_dataset(
                input_files,
                test_processing_function,
                output_file
            )
        
        # Verify pipeline stage results
        assert metrics.total_records_processed > 0
        assert metrics.records_per_second > 0
        assert metrics.peak_memory_usage_mb > 0
        assert metrics.processing_duration > 0
        assert metrics.error_count == 0
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify processed records
        processed_records = []
        with open(output_file, 'r') as f:
            for line in f:
                processed_records.append(json.loads(line))
        
        assert len(processed_records) == 50  # All test records processed
        assert all(record.get("processed") is True for record in processed_records)
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_m3_optimization_pipeline_stage(self):
        """Test M3 optimization as a complete pipeline stage"""
        # Setup M3 pipeline optimizer
        pipeline_optimizer = M3PipelineOptimizer(self.optimization_config)
        
        # Mock external dependencies
        with patch.object(pipeline_optimizer.memory_manager, 'create_memory_pool', return_value=True), \
             patch.object(pipeline_optimizer.memory_manager, 'optimize_allocation_pattern') as mock_pattern, \
             patch.object(pipeline_optimizer.memory_manager, 'get_memory_utilization') as mock_utilization, \
             patch.object(pipeline_optimizer.mlx_optimizer, 'optimize_model_loading') as mock_mlx, \
             patch.object(pipeline_optimizer.pytorch_optimizer, 'optimize_mps_settings') as mock_pytorch, \
             patch.object(pipeline_optimizer.inference_optimizer, 'optimize_lm_studio') as mock_lm_studio, \
             patch.object(pipeline_optimizer.inference_optimizer, 'optimize_ollama') as mock_ollama:
            
            # Setup mock responses
            mock_pattern.return_value = {
                "model_pool_gb": 16, "batch_pool_gb": 8,
                "cache_pool_gb": 4, "working_pool_gb": 6
            }
            mock_utilization.return_value = {
                "utilization_percent": 45.0,
                "available_gb": 70.4
            }
            mock_mlx.return_value = {"status": "optimized", "estimated_memory_gb": 12.0}
            mock_pytorch.return_value = {"status": "optimized", "device": "mps"}
            mock_lm_studio.return_value = {
                "status": "optimized", "application": "LM Studio",
                "estimated_performance": {"tokens_per_second": "80-120"}
            }
            mock_ollama.return_value = {
                "status": "optimized", "application": "Ollama",
                "estimated_performance": {"concurrent_models": "3-4"}
            }
            
            # Run optimization pipeline stage
            result = await pipeline_optimizer.optimize_inference_pipeline(
                "test_model_path", "inference"
            )
            
            # Verify pipeline stage results
            assert result["optimization_complete"] is True
            assert result["system_specs"]["unified_memory_gb"] == 128
            assert "memory_management" in result
            assert "framework_optimization" in result
            assert "inference_engines" in result
            assert "coordination" in result
            assert "performance_estimates" in result
            
            # Verify coordination strategy
            coordination = result["coordination"]
            assert coordination["primary_framework"] in ["MLX", "PyTorch"]
            assert coordination["fallback_framework"] in ["MLX", "PyTorch"]
            assert coordination["load_balancing"]["strategy"] == "round_robin"
            
            # Cleanup
            pipeline_optimizer.cleanup()
    
    @pytest.mark.slow
    def test_complete_pipeline_integration(self):
        """Test complete pipeline with all stages integrated"""
        # Stage 1: Quality Validation
        quality_controller = QualityController({
            "min_quality_score": 7.0,
            "technical_term_threshold": 2
        })
        
        # Load raw data
        test_file = self.test_data_dir / "test_dataset.jsonl"
        raw_records = []
        with open(test_file, 'r') as f:
            for line in f:
                raw_records.append(json.loads(line))
        
        # Stage 1: Quality validation
        validated_records, validation_stats = validate_dataset_batch(
            raw_records, quality_controller
        )
        
        assert len(validated_records) > 0
        assert validation_stats["valid_records"] > 0
        
        # Stage 2: Save validated records for processing
        validated_file = self.test_data_dir / "validated_dataset.jsonl"
        with open(validated_file, 'w') as f:
            for record in validated_records:
                f.write(json.dumps(record) + '\n')
        
        # Stage 2: Performance optimization processing
        processor = OptimizedDatasetProcessor(
            max_memory_mb=512,
            max_workers=2,
            batch_size=5
        )
        
        def enhancement_processing_function(batch):
            """Simulate enhancement processing"""
            enhanced_batch = []
            for record in batch:
                enhanced_record = record.copy()
                enhanced_record["metadata"]["enhancement_applied"] = True
                enhanced_record["metadata"]["processing_stage"] = "enhanced"
                
                # Simulate quality improvement
                original_score = enhanced_record["metadata"]["quality_score"]
                enhanced_record["metadata"]["quality_score"] = min(10.0, original_score + 0.3)
                
                enhanced_batch.append(enhanced_record)
            
            return enhanced_batch
        
        # Process validated records
        enhanced_output = self.test_data_dir / "enhanced_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 256 * 1024 * 1024
            mock_process_instance.cpu_percent.return_value = 35.0
            mock_process.return_value = mock_process_instance
            
            processing_metrics = processor.process_large_dataset(
                [validated_file],
                enhancement_processing_function,
                enhanced_output
            )
        
        # Verify processing results
        assert processing_metrics.total_records_processed > 0
        assert processing_metrics.error_count == 0
        assert enhanced_output.exists()
        
        # Stage 3: Final quality check on enhanced records
        enhanced_records = []
        with open(enhanced_output, 'r') as f:
            for line in f:
                enhanced_records.append(json.loads(line))
        
        final_validated, final_stats = validate_dataset_batch(
            enhanced_records, quality_controller
        )
        
        # Verify final pipeline results
        assert len(final_validated) == len(enhanced_records)  # All should pass
        assert final_stats["average_quality"] > validation_stats["average_quality"]
        assert all(record["metadata"]["enhancement_applied"] for record in final_validated)
        
        # Verify complete pipeline metrics
        pipeline_results = {
            "input_records": len(raw_records),
            "validated_records": len(validated_records),
            "processed_records": processing_metrics.total_records_processed,
            "final_records": len(final_validated),
            "validation_rate": len(validated_records) / len(raw_records),
            "processing_speed_rps": processing_metrics.records_per_second,
            "quality_improvement": final_stats["average_quality"] - validation_stats["average_quality"],
            "memory_efficiency_mb": processing_metrics.peak_memory_usage_mb
        }
        
        assert pipeline_results["validation_rate"] > 0.8  # At least 80% pass validation
        assert pipeline_results["processing_speed_rps"] > 10.0  # Reasonable speed
        assert pipeline_results["quality_improvement"] >= 0.0  # Quality maintained or improved
        assert pipeline_results["memory_efficiency_mb"] < 1024.0  # Memory usage reasonable
    
    def test_pipeline_error_handling_and_recovery(self):
        """Test pipeline error handling and recovery mechanisms"""
        # Create dataset with some problematic records
        problematic_records = [
            {  # Valid record
                "messages": [
                    {"role": "user", "content": "Valid question about eNodeB configuration?"},
                    {"role": "assistant", "content": "Valid answer about LTE optimization parameters."}
                ],
                "metadata": {"feature_name": "Valid Feature", "quality_score": 8.5, "technical_content": True}
            },
            {  # Invalid record - missing messages
                "metadata": {"feature_name": "Invalid Feature", "quality_score": 8.0}
            },
            {  # Valid record
                "messages": [
                    {"role": "user", "content": "Another valid question about 5G NR?"},
                    {"role": "assistant", "content": "Answer about 5G beamforming and MIMO techniques."}
                ],
                "metadata": {"feature_name": "Another Valid", "quality_score": 9.0, "technical_content": True}
            },
            {  # Record that will cause processing failure
                "messages": [
                    {"role": "user", "content": "TRIGGER_PROCESSING_ERROR"},
                    {"role": "assistant", "content": "This will cause an error in processing."}
                ],
                "metadata": {"feature_name": "Error Trigger", "quality_score": 8.0, "technical_content": True}
            }
        ]
        
        # Save problematic dataset
        problem_file = self.test_data_dir / "problematic_dataset.jsonl"
        with open(problem_file, 'w') as f:
            for record in problematic_records:
                f.write(json.dumps(record) + '\n')
        
        # Test quality validation error handling
        quality_controller = QualityController()
        validated_records, validation_stats = validate_dataset_batch(
            problematic_records, quality_controller
        )
        
        # Should handle invalid records gracefully
        assert len(validated_records) == 2  # Only valid records
        assert validation_stats["invalid_records"] == 2  # Two invalid records
        assert len(validation_stats["validation_errors"]) > 0
        
        # Test processing error handling
        processor = OptimizedDatasetProcessor(
            max_memory_mb=256,
            max_workers=2,
            batch_size=2
        )
        
        def error_prone_processing_function(batch):
            """Processing function that fails on specific content"""
            processed_batch = []
            for record in batch:
                # Simulate processing error on trigger content
                if "TRIGGER_PROCESSING_ERROR" in record["messages"][0]["content"]:
                    raise ValueError("Simulated processing error")
                
                processed_record = record.copy()
                processed_record["processed"] = True
                processed_batch.append(processed_record)
            
            return processed_batch
        
        error_output = self.test_data_dir / "error_handling_output.jsonl"
        
        # Should not crash on processing errors
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 128 * 1024 * 1024
            mock_process_instance.cpu_percent.return_value = 25.0
            mock_process.return_value = mock_process_instance
            
            # Process should complete despite errors
            metrics = processor.process_large_dataset(
                [problem_file],
                error_prone_processing_function,
                error_output
            )
        
        # Verify error handling
        assert metrics.error_count >= 1  # At least one error expected
        assert error_output.exists()  # Output file should still be created
        
        # Verify some records were still processed successfully
        if error_output.stat().st_size > 0:
            processed_records = []
            with open(error_output, 'r') as f:
                for line in f:
                    if line.strip():
                        processed_records.append(json.loads(line))
            
            # Should have successfully processed some records
            assert len(processed_records) >= 2
            assert all(record.get("processed") is True for record in processed_records)


@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformanceIntegration:
    """Integration tests focused on performance characteristics"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.create_large_test_dataset(1000)  # 1000 records for performance testing
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def create_large_test_dataset(self, num_records):
        """Create large dataset for performance testing"""
        test_file = self.test_data_dir / "large_dataset.jsonl"
        
        with open(test_file, 'w') as f:
            for i in range(num_records):
                record = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"How do I configure parameter {i} for optimal LTE performance in urban environments?"
                        },
                        {
                            "role": "assistant",
                            "content": f"To configure parameter {i} for LTE optimization, set the RSRP threshold to -{90+i%20} dBm, adjust time-to-trigger to {240+i%100}ms, and enable carrier aggregation for enhanced throughput."
                        }
                    ],
                    "metadata": {
                        "feature_name": f"LTE Parameter {i}",
                        "quality_score": 7.0 + (i % 30) * 0.1,
                        "technical_content": True,
                        "technical_terms": ["LTE", "RSRP", "carrier aggregation"],
                        "performance_test_id": i
                    }
                }
                f.write(json.dumps(record) + '\n')
        
        return test_file
    
    def test_large_dataset_processing_performance(self):
        """Test processing performance with large datasets"""
        processor = OptimizedDatasetProcessor(
            max_memory_mb=1024,
            max_workers=4,
            batch_size=50,
            cache_enabled=True
        )
        
        def performance_test_processing(batch):
            """Optimized processing function for performance testing"""
            import time
            start_time = time.time()
            
            processed_batch = []
            for record in batch:
                # Simulate lightweight processing
                processed_record = record.copy()
                processed_record["metadata"]["processed_at"] = time.time()
                processed_batch.append(processed_record)
            
            processing_time = time.time() - start_time
            
            # Add processing metadata
            for record in processed_batch:
                record["metadata"]["batch_processing_time"] = processing_time
            
            return processed_batch
        
        # Process large dataset
        large_file = self.test_data_dir / "large_dataset.jsonl"
        output_file = self.test_data_dir / "performance_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
            mock_process_instance.cpu_percent.return_value = 70.0
            mock_process.return_value = mock_process_instance
            
            start_time = time.time()
            metrics = processor.process_large_dataset(
                [large_file],
                performance_test_processing,
                output_file
            )
            total_time = time.time() - start_time
        
        # Performance assertions
        assert metrics.total_records_processed == 1000
        assert metrics.records_per_second > 100  # At least 100 RPS
        assert metrics.peak_memory_usage_mb < 1536  # Under 1.5GB
        assert total_time < 30.0  # Complete in under 30 seconds
        assert metrics.error_count == 0
        
        # Verify throughput consistency
        assert metrics.processing_duration > 0
        calculated_rps = metrics.total_records_processed / metrics.processing_duration
        assert abs(calculated_rps - metrics.records_per_second) < 10.0  # Within 10 RPS
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency during high-load processing"""
        processor = OptimizedDatasetProcessor(
            max_memory_mb=512,  # Constrained memory
            max_workers=2,
            batch_size=25,
            cache_enabled=False  # Disable cache to test raw processing
        )
        
        def memory_intensive_processing(batch):
            """Processing that simulates memory usage"""
            # Simulate memory-intensive operations
            large_temp_data = [{"temp": list(range(100))} for _ in range(len(batch))]
            
            processed_batch = []
            for i, record in enumerate(batch):
                processed_record = record.copy()
                processed_record["temp_data_size"] = len(large_temp_data[i]["temp"])
                processed_batch.append(processed_record)
            
            # Cleanup temp data
            del large_temp_data
            
            return processed_batch
        
        large_file = self.test_data_dir / "large_dataset.jsonl"
        memory_output = self.test_data_dir / "memory_test_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            # Simulate varying memory usage
            memory_values = [256, 384, 512, 480, 320, 256] * 200  # Simulate fluctuation
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = Mock(side_effect=lambda: (memory_values.pop(0) if memory_values else 256) * 1024 * 1024)
            mock_process_instance.cpu_percent.return_value = 60.0
            mock_process.return_value = mock_process_instance
            
            metrics = processor.process_large_dataset(
                [large_file],
                memory_intensive_processing,
                memory_output
            )
        
        # Memory efficiency assertions
        assert metrics.peak_memory_usage_mb <= 600  # Within reasonable bounds
        assert metrics.total_records_processed == 1000
        assert metrics.error_count == 0
        
        # Verify processing completed despite memory constraints
        assert memory_output.exists()
        
        # Verify efficiency score
        efficiency = metrics.efficiency_score
        assert 0.3 <= efficiency <= 1.0  # Reasonable efficiency range
    
    def test_concurrent_pipeline_stage_performance(self):
        """Test performance of concurrent pipeline stages"""
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Setup multiple processors for concurrent testing
        processors = [
            OptimizedDatasetProcessor(
                max_memory_mb=256,
                max_workers=2,
                batch_size=20,
                cache_enabled=True
            ) for _ in range(3)
        ]
        
        def concurrent_processing_task(processor_id, processor):
            """Task for concurrent processing"""
            def task_processing_function(batch):
                # Simulate processor-specific work
                processed_batch = []
                for record in batch:
                    processed_record = record.copy()
                    processed_record["metadata"]["processor_id"] = processor_id
                    processed_record["metadata"]["concurrent_processed"] = True
                    processed_batch.append(processed_record)
                return processed_batch
            
            # Create subset of data for this processor
            subset_file = self.test_data_dir / f"subset_{processor_id}.jsonl"
            output_file = self.test_data_dir / f"concurrent_output_{processor_id}.jsonl"
            
            # Create subset (every 3rd record starting from processor_id)
            with open(self.test_data_dir / "large_dataset.jsonl", 'r') as input_f, \
                 open(subset_file, 'w') as subset_f:
                for i, line in enumerate(input_f):
                    if i % 3 == processor_id:
                        subset_f.write(line)
            
            # Process subset
            with patch('psutil.Process') as mock_process:
                mock_process_instance = Mock()
                mock_process_instance.memory_info.return_value.rss = (200 + processor_id * 50) * 1024 * 1024
                mock_process_instance.cpu_percent.return_value = 40.0 + processor_id * 10
                mock_process.return_value = mock_process_instance
                
                metrics = processor.process_large_dataset(
                    [subset_file],
                    task_processing_function,
                    output_file
                )
            
            return processor_id, metrics
        
        # Run concurrent processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(concurrent_processing_task, i, processor): i 
                for i, processor in enumerate(processors)
            }
            
            results = {}
            for future in as_completed(futures):
                processor_id, metrics = future.result()
                results[processor_id] = metrics
        
        total_concurrent_time = time.time() - start_time
        
        # Verify concurrent processing results
        assert len(results) == 3
        
        total_records = sum(metrics.total_records_processed for metrics in results.values())
        assert total_records >= 900  # Allow for some subset variation
        
        # Verify all processors completed without errors
        assert all(metrics.error_count == 0 for metrics in results.values())
        
        # Verify concurrent processing was faster than sequential would be
        avg_processing_time = sum(metrics.processing_duration for metrics in results.values()) / 3
        assert total_concurrent_time < avg_processing_time * 2  # Significant speedup
        
        # Verify outputs were created
        for i in range(3):
            output_file = self.test_data_dir / f"concurrent_output_{i}.jsonl"
            assert output_file.exists()
            
            # Verify processor IDs in outputs
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        assert record["metadata"]["processor_id"] == i
                        assert record["metadata"]["concurrent_processed"] is True


@pytest.mark.integration
class TestPipelineBoundaryConditions:
    """Test pipeline behavior under boundary conditions and edge cases"""
    
    def setup_method(self):
        """Setup boundary condition tests"""
        self.test_data_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    def test_empty_dataset_handling(self):
        """Test pipeline behavior with empty datasets"""
        # Create empty dataset file
        empty_file = self.test_data_dir / "empty_dataset.jsonl"
        empty_file.touch()
        
        processor = OptimizedDatasetProcessor(
            max_memory_mb=256,
            max_workers=2,
            batch_size=10
        )
        
        def empty_processing_function(batch):
            return batch
        
        output_file = self.test_data_dir / "empty_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 128 * 1024 * 1024
            mock_process_instance.cpu_percent.return_value = 5.0
            mock_process.return_value = mock_process_instance
            
            metrics = processor.process_large_dataset(
                [empty_file],
                empty_processing_function,
                output_file
            )
        
        # Verify empty dataset handling
        assert metrics.total_records_processed == 0
        assert metrics.records_per_second == 0.0
        assert metrics.error_count == 0
        assert output_file.exists()
        assert output_file.stat().st_size == 0  # Empty output
    
    def test_single_record_processing(self):
        """Test pipeline with single record"""
        single_record_file = self.test_data_dir / "single_record.jsonl"
        
        single_record = {
            "messages": [
                {"role": "user", "content": "Single test question about eNodeB?"},
                {"role": "assistant", "content": "Single test answer about LTE configuration parameters."}
            ],
            "metadata": {
                "feature_name": "Single Test",
                "quality_score": 8.5,
                "technical_content": True
            }
        }
        
        with open(single_record_file, 'w') as f:
            f.write(json.dumps(single_record) + '\n')
        
        processor = OptimizedDatasetProcessor(
            max_memory_mb=256,
            max_workers=2,
            batch_size=10  # Batch size larger than dataset
        )
        
        def single_processing_function(batch):
            assert len(batch) == 1  # Should receive single record in batch
            processed_batch = []
            for record in batch:
                processed_record = record.copy()
                processed_record["single_processed"] = True
                processed_batch.append(processed_record)
            return processed_batch
        
        output_file = self.test_data_dir / "single_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 64 * 1024 * 1024
            mock_process_instance.cpu_percent.return_value = 10.0
            mock_process.return_value = mock_process_instance
            
            metrics = processor.process_large_dataset(
                [single_record_file],
                single_processing_function,
                output_file
            )
        
        # Verify single record processing
        assert metrics.total_records_processed == 1
        assert metrics.records_per_second > 0
        assert metrics.error_count == 0
        
        # Verify output
        with open(output_file, 'r') as f:
            processed_record = json.loads(f.read().strip())
            assert processed_record["single_processed"] is True
            assert processed_record["metadata"]["feature_name"] == "Single Test"
    
    def test_extremely_large_batch_size(self):
        """Test pipeline with batch size larger than dataset"""
        # Create small dataset
        small_dataset = self.test_data_dir / "small_dataset.jsonl"
        
        records = []
        for i in range(5):
            record = {
                "messages": [
                    {"role": "user", "content": f"Question {i}?"},
                    {"role": "assistant", "content": f"Answer {i}."}
                ],
                "metadata": {"feature_name": f"Feature {i}", "quality_score": 8.0, "technical_content": True}
            }
            records.append(record)
        
        with open(small_dataset, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        processor = OptimizedDatasetProcessor(
            max_memory_mb=256,
            max_workers=2,
            batch_size=100  # Much larger than dataset size
        )
        
        def large_batch_processing_function(batch):
            assert len(batch) <= 5  # Should not exceed dataset size
            return [{"processed": True, **record} for record in batch]
        
        output_file = self.test_data_dir / "large_batch_output.jsonl"
        
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 128 * 1024 * 1024
            mock_process_instance.cpu_percent.return_value = 20.0
            mock_process.return_value = mock_process_instance
            
            metrics = processor.process_large_dataset(
                [small_dataset],
                large_batch_processing_function,
                output_file
            )
        
        # Verify large batch handling
        assert metrics.total_records_processed == 5
        assert metrics.error_count == 0
        
        # Verify all records processed
        processed_records = []
        with open(output_file, 'r') as f:
            for line in f:
                processed_records.append(json.loads(line))
        
        assert len(processed_records) == 5
        assert all(record["processed"] is True for record in processed_records)