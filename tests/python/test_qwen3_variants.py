#!/usr/bin/env python3
"""
Comprehensive test suite for Qwen3 variants manager
Tests model selection, MLX optimization, and M3 Max integration
"""

import asyncio
import json
import pytest
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir / "src" / "python-ml"))

try:
    from model_management.qwen3_variants import Qwen3VariantsManager
except ImportError as e:
    print(f"Import error: {e}")
    pytest.skip("Qwen3 variants module not available", allow_module_level=True)

class TestQwen3VariantsManager:
    """Test suite for Qwen3 variants manager"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_config = {
            "mlx_available": True,
            "model_cache_dir": str(Path(cls.temp_dir) / "models"),
            "max_concurrent_models": 2,
            "unified_memory_gb": 45,
            "model_configs": {
                "qwen3-1.7b": {
                    "memory_requirement_gb": 2,
                    "context_length": 4096,
                    "optimal_batch_size": 8
                },
                "qwen3-7b": {
                    "memory_requirement_gb": 8,
                    "context_length": 8192,
                    "optimal_batch_size": 4
                },
                "qwen3-30b": {
                    "memory_requirement_gb": 30,
                    "context_length": 16384,
                    "optimal_batch_size": 1
                }
            }
        }
    
    @pytest.fixture
    def qwen3_manager(self):
        """Create Qwen3 manager instance for testing"""
        return Qwen3VariantsManager(self.test_config)
    
    def test_manager_initialization(self, qwen3_manager):
        """Test manager initialization"""
        assert qwen3_manager.config == self.test_config
        assert qwen3_manager.mlx_available == self.test_config["mlx_available"]
        assert qwen3_manager.max_concurrent_models == self.test_config["max_concurrent_models"]
        assert qwen3_manager.unified_memory_gb == self.test_config["unified_memory_gb"]
        
        # Test model configurations loaded
        assert "qwen3-1.7b" in qwen3_manager.model_configs
        assert "qwen3-7b" in qwen3_manager.model_configs
        assert "qwen3-30b" in qwen3_manager.model_configs
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, qwen3_manager):
        """Test getting available models"""
        with patch.object(qwen3_manager, '_check_model_availability', return_value=True):
            available_models = await qwen3_manager.get_available_models()
            
            assert isinstance(available_models, list)
            assert len(available_models) >= 1
            
            # Should include at least the base models
            expected_models = {"qwen3-1.7b", "qwen3-7b", "qwen3-30b"}
            available_set = set(available_models)
            assert expected_models.intersection(available_set), f"Expected models {expected_models} not in {available_set}"
    
    @pytest.mark.asyncio
    async def test_model_selection_fast_complexity(self, qwen3_manager):
        """Test model selection for fast/simple documents"""
        complexity_hints = {
            "parameter_count": 2,
            "counter_count": 1,
            "technical_density": 0.5,
            "content_length": 500
        }
        
        selected_model = await qwen3_manager.select_optimal_model(complexity_hints)
        
        # Fast complexity should select the smallest model
        assert selected_model in ["qwen3-1.7b", "qwen3-1.7b-mlx"]
    
    @pytest.mark.asyncio
    async def test_model_selection_balanced_complexity(self, qwen3_manager):
        """Test model selection for balanced complexity documents"""
        complexity_hints = {
            "parameter_count": 8,
            "counter_count": 4,
            "technical_density": 3.0,
            "content_length": 3000
        }
        
        selected_model = await qwen3_manager.select_optimal_model(complexity_hints)
        
        # Balanced complexity should select the medium model
        assert selected_model in ["qwen3-7b", "qwen3-7b-mlx"]
    
    @pytest.mark.asyncio
    async def test_model_selection_quality_complexity(self, qwen3_manager):
        """Test model selection for high complexity documents"""
        complexity_hints = {
            "parameter_count": 20,
            "counter_count": 12,
            "technical_density": 8.0,
            "content_length": 15000
        }
        
        selected_model = await qwen3_manager.select_optimal_model(complexity_hints)
        
        # High complexity should select the largest model
        assert selected_model in ["qwen3-30b", "qwen3-30b-mlx"]
    
    @pytest.mark.asyncio
    async def test_model_loading_simulation(self, qwen3_manager):
        """Test model loading and unloading"""
        model_name = "qwen3-7b"
        
        # Mock the model loading process
        with patch.object(qwen3_manager, '_load_model_mlx') as mock_load:
            mock_model = Mock()
            mock_model.config = {"hidden_size": 4096, "num_attention_heads": 32}
            mock_load.return_value = mock_model
            
            # Test loading
            success = await qwen3_manager.load_model(model_name)
            assert success
            mock_load.assert_called_once_with(model_name)
            
            # Verify model is tracked as loaded
            assert model_name in qwen3_manager.loaded_models
            
            # Test unloading
            await qwen3_manager.unload_model(model_name)
            assert model_name not in qwen3_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_memory_management(self, qwen3_manager):
        """Test memory management and limits"""
        # Test memory availability check
        assert qwen3_manager.unified_memory_gb == 45
        
        # Test memory requirement calculation
        memory_req_1_7b = qwen3_manager.get_model_memory_requirement("qwen3-1.7b")
        memory_req_7b = qwen3_manager.get_model_memory_requirement("qwen3-7b")
        memory_req_30b = qwen3_manager.get_model_memory_requirement("qwen3-30b")
        
        assert memory_req_1_7b < memory_req_7b < memory_req_30b
        assert memory_req_30b <= qwen3_manager.unified_memory_gb  # Should fit in M3 Max memory
        
        # Test concurrent model limits
        max_concurrent = qwen3_manager.get_max_concurrent_models()
        assert max_concurrent == self.test_config["max_concurrent_models"]
    
    @pytest.mark.asyncio
    async def test_model_switching_optimization(self, qwen3_manager):
        """Test model switching and caching optimization"""
        # Simulate switching between models
        models_to_test = ["qwen3-1.7b", "qwen3-7b", "qwen3-1.7b"]  # Switch back to test caching
        
        with patch.object(qwen3_manager, '_load_model_mlx') as mock_load, \
             patch.object(qwen3_manager, '_unload_model') as mock_unload:
            
            mock_load.return_value = Mock()
            
            switch_times = []
            for model in models_to_test:
                start_time = time.time()
                await qwen3_manager.switch_to_model(model)
                switch_time = time.time() - start_time
                switch_times.append(switch_time)
            
            # Model switching should be tracked
            assert len(switch_times) == len(models_to_test)
            
            # Verify optimization metrics
            avg_switch_time = sum(switch_times) / len(switch_times)
            assert avg_switch_time < 3.0  # Should switch in under 3 seconds
    
    @pytest.mark.asyncio
    async def test_qa_generation_simulation(self, qwen3_manager):
        """Test QA pair generation simulation"""
        document_content = """# Carrier Aggregation Feature

## Overview
Carrier Aggregation (CA) enables the aggregation of multiple carriers to increase bandwidth.

## Parameters
- **carrierAggregationEnabled**: Boolean parameter to enable/disable CA
  - Valid Values: true, false
  - Default: false

## Counters
- pmCarrierAggregationAttempts: Number of CA attempts
- pmCarrierAggregationSuccesses: Number of successful CA setups"""
        
        model_name = "qwen3-7b"
        expected_qa_pairs = 5
        
        # Mock the QA generation process
        with patch.object(qwen3_manager, '_generate_qa_pairs_mlx') as mock_generate:
            mock_qa_pairs = [
                {
                    "question": "What is Carrier Aggregation?",
                    "answer": "Carrier Aggregation (CA) enables the aggregation of multiple carriers to increase bandwidth.",
                    "confidence": 0.92,
                    "metadata": {
                        "question_type": "factual",
                        "complexity": "medium"
                    }
                },
                {
                    "question": "How do you enable Carrier Aggregation?",
                    "answer": "Set the carrierAggregationEnabled parameter to true.",
                    "confidence": 0.88,
                    "metadata": {
                        "question_type": "procedural",
                        "complexity": "low"
                    }
                }
            ]
            mock_generate.return_value = mock_qa_pairs
            
            # Test QA generation
            qa_pairs = await qwen3_manager.generate_qa_pairs(document_content, model_name, expected_qa_pairs)
            
            assert isinstance(qa_pairs, list)
            assert len(qa_pairs) == len(mock_qa_pairs)
            
            # Verify QA pair structure
            for qa_pair in qa_pairs:
                assert "question" in qa_pair
                assert "answer" in qa_pair
                assert "confidence" in qa_pair
                assert "metadata" in qa_pair
                assert qa_pair["confidence"] >= 0.0 and qa_pair["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, qwen3_manager):
        """Test batch processing optimization"""
        documents = [
            "Document 1: Simple feature description",
            "Document 2: Another basic feature",
            "Document 3: Third simple document"
        ]
        model_name = "qwen3-1.7b"
        batch_size = 3
        
        with patch.object(qwen3_manager, '_process_batch_mlx') as mock_batch:
            mock_results = [
                {"qa_pairs": [{"question": f"Q{i}", "answer": f"A{i}", "confidence": 0.8}]}
                for i in range(len(documents))
            ]
            mock_batch.return_value = mock_results
            
            # Test batch processing
            results = await qwen3_manager.process_documents_batch(documents, model_name, batch_size)
            
            assert isinstance(results, list)
            assert len(results) == len(documents)
            mock_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, qwen3_manager):
        """Test performance monitoring and metrics"""
        # Simulate processing multiple documents
        processing_times = []
        quality_scores = []
        
        for i in range(5):
            start_time = time.time()
            
            # Simulate document processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            processing_time = time.time() - start_time
            quality_score = 0.8 + (i * 0.02)  # Gradually increasing quality
            
            processing_times.append(processing_time)
            quality_scores.append(quality_score)
            
            # Record metrics
            await qwen3_manager.record_processing_metrics(
                model_name="qwen3-7b",
                processing_time=processing_time,
                quality_score=quality_score,
                tokens_processed=150 + i * 10,
                memory_used_mb=4096
            )
        
        # Get performance statistics
        stats = await qwen3_manager.get_performance_statistics()
        
        assert "models" in stats
        assert "qwen3-7b" in stats["models"]
        
        model_stats = stats["models"]["qwen3-7b"]
        assert model_stats["requests_processed"] == 5
        assert model_stats["average_processing_time"] > 0
        assert model_stats["average_quality_score"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_mlx_optimization_features(self, qwen3_manager):
        """Test MLX-specific optimization features"""
        if not qwen3_manager.mlx_available:
            pytest.skip("MLX not available for testing")
        
        # Test MLX memory optimization
        memory_usage_before = await qwen3_manager.get_memory_usage()
        await qwen3_manager.optimize_mlx_memory()
        memory_usage_after = await qwen3_manager.get_memory_usage()
        
        # Memory optimization should not increase usage
        assert memory_usage_after["used_gb"] <= memory_usage_before["used_gb"] + 1.0  # Allow small tolerance
        
        # Test MLX GPU utilization
        gpu_utilization = await qwen3_manager.get_gpu_utilization()
        assert isinstance(gpu_utilization, (int, float))
        assert gpu_utilization >= 0.0 and gpu_utilization <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, qwen3_manager):
        """Test error handling and recovery"""
        # Test handling of invalid model name
        with pytest.raises((ValueError, KeyError)):
            await qwen3_manager.load_model("invalid-model-name")
        
        # Test handling of insufficient memory
        with patch.object(qwen3_manager, 'get_available_memory', return_value=1.0):  # Only 1GB available
            result = await qwen3_manager.load_model("qwen3-30b")  # Requires 30GB
            assert not result  # Should fail gracefully
        
        # Test recovery from model loading failure
        with patch.object(qwen3_manager, '_load_model_mlx', side_effect=Exception("Model loading failed")):
            result = await qwen3_manager.load_model("qwen3-7b")
            assert not result  # Should return False instead of raising exception
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, qwen3_manager):
        """Test configuration validation"""
        # Test valid configuration
        assert qwen3_manager.validate_configuration()
        
        # Test invalid configuration
        invalid_config = self.test_config.copy()
        invalid_config["unified_memory_gb"] = -5  # Invalid negative memory
        
        with pytest.raises(ValueError):
            Qwen3VariantsManager(invalid_config)
    
    @pytest.mark.asyncio
    async def test_concurrent_model_management(self, qwen3_manager):
        """Test concurrent model loading and management"""
        models_to_load = ["qwen3-1.7b", "qwen3-7b"]
        
        with patch.object(qwen3_manager, '_load_model_mlx', return_value=Mock()):
            # Load models concurrently
            tasks = [qwen3_manager.load_model(model) for model in models_to_load]
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(results)
            
            # Check that models are loaded
            for model in models_to_load:
                assert model in qwen3_manager.loaded_models
            
            # Test concurrent model limits
            assert len(qwen3_manager.loaded_models) <= qwen3_manager.max_concurrent_models
    
    @pytest.mark.asyncio
    async def test_model_fallback_strategy(self, qwen3_manager):
        """Test model fallback when preferred model is not available"""
        complexity_hints = {
            "parameter_count": 25,  # Would normally select qwen3-30b
            "counter_count": 15,
            "technical_density": 10.0,
            "content_length": 20000
        }
        
        # Simulate qwen3-30b not being available
        with patch.object(qwen3_manager, 'is_model_available', side_effect=lambda x: x != "qwen3-30b"):
            selected_model = await qwen3_manager.select_optimal_model(complexity_hints)
            
            # Should fallback to qwen3-7b
            assert selected_model in ["qwen3-7b", "qwen3-7b-mlx"]
    
    def test_cleanup(self, qwen3_manager):
        """Test proper cleanup of resources"""
        # Simulate cleanup
        qwen3_manager.cleanup()
        
        # Verify cleanup state
        assert len(qwen3_manager.loaded_models) == 0
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])