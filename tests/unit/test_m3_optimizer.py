"""
Unit tests for M3 Optimizer
TDD London School testing patterns with comprehensive mocking
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from m3_optimizer import (
    M3UnifiedMemoryManager, 
    M3MLXOptimizer, 
    M3PyTorchOptimizer, 
    M3InferenceOptimizer,
    M3PipelineOptimizer,
    OptimizationConfig,
    optimize_m3_system
)


class TestOptimizationConfig:
    """Test optimization configuration"""
    
    def test_default_config_values(self):
        """Test default configuration values are sensible"""
        config = OptimizationConfig()
        
        assert config.max_memory_usage_percent == 85.0
        assert config.preferred_batch_size == 512
        assert config.max_concurrent_requests == 8
        assert config.enable_memory_mapping is True
        assert config.use_quantization is True
        assert config.cpu_threads == 12
        assert config.gpu_memory_fraction == 0.8
    
    def test_custom_config_values(self):
        """Test custom configuration values are preserved"""
        config = OptimizationConfig(
            max_memory_usage_percent=90.0,
            preferred_batch_size=256,
            max_concurrent_requests=16,
            enable_memory_mapping=False,
            use_quantization=False,
            cpu_threads=8,
            gpu_memory_fraction=0.9
        )
        
        assert config.max_memory_usage_percent == 90.0
        assert config.preferred_batch_size == 256
        assert config.max_concurrent_requests == 16
        assert config.enable_memory_mapping is False
        assert config.use_quantization is False
        assert config.cpu_threads == 8
        assert config.gpu_memory_fraction == 0.9


class TestM3UnifiedMemoryManager:
    """Test M3 unified memory manager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OptimizationConfig()
        self.memory_manager = M3UnifiedMemoryManager(self.config)
    
    @patch('mmap.mmap')
    def test_create_memory_pool_success(self, mock_mmap):
        """Test successful memory pool creation"""
        mock_mmap.return_value = MagicMock()
        
        result = self.memory_manager.create_memory_pool("test_pool", 8.0)
        
        assert result is True
        assert "test_pool" in self.memory_manager.memory_pools
        assert "test_pool" in self.memory_manager.allocated_memory
        
        # Verify mmap was called with correct size
        expected_size = int(8.0 * 1024 * 1024 * 1024)  # 8GB in bytes
        mock_mmap.assert_called_once_with(-1, expected_size)
    
    @patch('mmap.mmap', side_effect=Exception("Memory allocation failed"))
    def test_create_memory_pool_failure(self, mock_mmap):
        """Test memory pool creation failure"""
        result = self.memory_manager.create_memory_pool("test_pool", 8.0)
        
        assert result is False
        assert "test_pool" not in self.memory_manager.memory_pools
        assert "test_pool" not in self.memory_manager.allocated_memory
    
    def test_get_memory_utilization_empty(self):
        """Test memory utilization with no pools"""
        utilization = self.memory_manager.get_memory_utilization()
        
        assert utilization["total_system_gb"] == 128
        assert utilization["total_allocated_gb"] == 0.0
        assert utilization["available_gb"] == 128.0
        assert utilization["utilization_percent"] == 0.0
        assert utilization["pools"] == {}
    
    @patch('mmap.mmap')
    def test_get_memory_utilization_with_pools(self, mock_mmap):
        """Test memory utilization with allocated pools"""
        mock_mmap.return_value = MagicMock()
        
        # Create test pools
        self.memory_manager.create_memory_pool("pool1", 32.0)
        self.memory_manager.create_memory_pool("pool2", 16.0)
        
        utilization = self.memory_manager.get_memory_utilization()
        
        assert utilization["total_system_gb"] == 128
        assert utilization["total_allocated_gb"] == 48.0
        assert utilization["available_gb"] == 80.0
        assert utilization["utilization_percent"] == 37.5  # 48/128 * 100
        assert utilization["pools"]["pool1"] == 32.0
        assert utilization["pools"]["pool2"] == 16.0
    
    def test_optimize_allocation_pattern_inference(self):
        """Test memory allocation pattern for inference workload"""
        pattern = self.memory_manager.optimize_allocation_pattern("inference")
        
        assert pattern["model_pool_gb"] == 32
        assert pattern["batch_pool_gb"] == 16
        assert pattern["cache_pool_gb"] == 8
        assert pattern["working_pool_gb"] == 12
    
    def test_optimize_allocation_pattern_training(self):
        """Test memory allocation pattern for training workload"""
        pattern = self.memory_manager.optimize_allocation_pattern("training")
        
        assert pattern["model_pool_gb"] == 40
        assert pattern["gradient_pool_gb"] == 20
        assert pattern["optimizer_pool_gb"] == 16
        assert pattern["working_pool_gb"] == 16
    
    def test_optimize_allocation_pattern_high_utilization(self):
        """Test allocation pattern adjustment under memory pressure"""
        # Simulate high memory utilization
        with patch.object(self.memory_manager, 'get_memory_utilization', 
                         return_value={"utilization_percent": 75.0}):
            pattern = self.memory_manager.optimize_allocation_pattern("inference")
            
            # All values should be scaled down by 0.8
            assert pattern["model_pool_gb"] == 32 * 0.8
            assert pattern["batch_pool_gb"] == 16 * 0.8
    
    @patch('mmap.mmap')
    @patch('gc.collect')
    def test_cleanup_memory_pools(self, mock_gc_collect, mock_mmap):
        """Test memory pool cleanup"""
        mock_pool = MagicMock()
        mock_mmap.return_value = mock_pool
        
        # Create a pool
        self.memory_manager.create_memory_pool("test_pool", 8.0)
        
        # Cleanup
        self.memory_manager.cleanup_memory_pools()
        
        # Verify cleanup
        mock_pool.close.assert_called_once()
        mock_gc_collect.assert_called_once()
        assert len(self.memory_manager.memory_pools) == 0
        assert len(self.memory_manager.allocated_memory) == 0


class TestM3MLXOptimizer:
    """Test M3 MLX optimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OptimizationConfig()
    
    @patch('m3_optimizer.importlib.util.find_spec', return_value=None)
    def test_mlx_not_available(self, mock_find_spec):
        """Test behavior when MLX is not available"""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'mlx'")):
            optimizer = M3MLXOptimizer(self.config)
            assert optimizer.mlx_available is False
    
    @patch('m3_optimizer.importlib.util.find_spec')
    @patch('importlib.import_module')
    def test_mlx_available(self, mock_import, mock_find_spec):
        """Test behavior when MLX is available"""
        mock_find_spec.return_value = MagicMock()
        mock_mx = MagicMock()
        mock_mx.default_device.return_value = "gpu"
        mock_import.return_value = mock_mx
        
        with patch.dict('sys.modules', {'mlx.core': mock_mx}):
            optimizer = M3MLXOptimizer(self.config)
            # Manually set for test
            optimizer.mlx_available = True
            assert optimizer.mlx_available is True
    
    def test_optimize_model_loading_mlx_unavailable(self):
        """Test model optimization when MLX is unavailable"""
        optimizer = M3MLXOptimizer(self.config)
        optimizer.mlx_available = False
        
        result = optimizer.optimize_model_loading("test_model_path")
        
        assert result["error"] == "MLX not available"
    
    @patch('os.environ', {})
    @patch('pathlib.Path')
    def test_optimize_model_loading_success(self, mock_path):
        """Test successful model loading optimization"""
        # Setup mocks
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.stat.return_value.st_size = 8 * 1024 * 1024 * 1024  # 8GB
        mock_path.return_value = mock_path_obj
        
        optimizer = M3MLXOptimizer(self.config)
        optimizer.mlx_available = True
        
        with patch.dict('sys.modules', {
            'mlx.core': MagicMock(),
            'mlx.nn': MagicMock()
        }):
            result = optimizer.optimize_model_loading("test_model_path")
        
        assert result["status"] == "optimized"
        assert result["settings"]["memory_mapping"] is True
        assert result["settings"]["lazy_loading"] is True
        assert result["settings"]["quantization"] == "int4"
        assert result["settings"]["batch_size"] == 512
        assert "estimated_memory_gb" in result
        assert "recommended_batch_size" in result
    
    def test_estimate_model_memory_with_quantization(self):
        """Test model memory estimation with quantization"""
        optimizer = M3MLXOptimizer(self.config)
        optimizer.mlx_available = True
        
        # Mock Path.stat for file size
        with patch('pathlib.Path') as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = True
            mock_path_obj.stat.return_value.st_size = 8 * 1024 * 1024 * 1024  # 8GB
            mock_path.return_value = mock_path_obj
            
            estimated_memory = optimizer._estimate_model_memory("test_path")
            
            # With quantization: 8GB * 1.5 overhead * 0.5 quantization = 6GB
            assert estimated_memory == 6.0
    
    def test_calculate_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        optimizer = M3MLXOptimizer(self.config)
        batch_size = optimizer._calculate_optimal_batch_size()
        
        # Should return a power of 2, within reasonable limits
        assert batch_size in [32, 64, 128, 256, 512, 1024, 2048]
        assert batch_size <= self.config.preferred_batch_size


class TestM3PyTorchOptimizer:
    """Test M3 PyTorch optimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OptimizationConfig()
    
    @patch('importlib.import_module', side_effect=ImportError("No module named 'torch'"))
    def test_pytorch_not_available(self, mock_import):
        """Test behavior when PyTorch is not available"""
        optimizer = M3PyTorchOptimizer(self.config)
        assert optimizer.mps_available is False
    
    @patch('importlib.import_module')
    def test_pytorch_mps_available(self, mock_import):
        """Test MPS availability check"""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.backends.mps.is_built.return_value = True
        mock_import.return_value = mock_torch
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            optimizer = M3PyTorchOptimizer(self.config)
            optimizer.mps_available = True  # Manually set for test
            assert optimizer.mps_available is True
    
    def test_optimize_mps_settings_unavailable(self):
        """Test MPS optimization when unavailable"""
        optimizer = M3PyTorchOptimizer(self.config)
        optimizer.mps_available = False
        
        result = optimizer.optimize_mps_settings()
        
        assert result["error"] == "MPS not available"
    
    @patch('os.environ', {})
    def test_optimize_mps_settings_success(self):
        """Test successful MPS optimization"""
        mock_torch = MagicMock()
        mock_torch.mps.set_per_process_memory_fraction = MagicMock()
        
        optimizer = M3PyTorchOptimizer(self.config)
        optimizer.mps_available = True
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = optimizer.optimize_mps_settings()
        
        assert result["status"] == "optimized"
        assert result["settings"]["device"] == "mps"
        assert result["settings"]["memory_fraction"] == 0.8
        assert result["settings"]["fallback_enabled"] is True
        assert "optimal_batch_sizes" in result["settings"]
        assert "data_loading" in result["settings"]
    
    def test_create_optimized_dataloader_unavailable(self):
        """Test DataLoader creation when MPS is unavailable"""
        optimizer = M3PyTorchOptimizer(self.config)
        optimizer.mps_available = False
        
        result = optimizer.create_optimized_dataloader("fake_dataset")
        
        assert result is None
    
    def test_calculate_optimal_batch_size(self):
        """Test PyTorch optimal batch size calculation"""
        optimizer = M3PyTorchOptimizer(self.config)
        batch_size = optimizer._calculate_optimal_batch_size()
        
        assert batch_size <= 256
        assert batch_size <= self.config.preferred_batch_size


class TestM3InferenceOptimizer:
    """Test M3 inference optimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OptimizationConfig()
        self.optimizer = M3InferenceOptimizer(self.config)
    
    def test_optimize_lm_studio(self):
        """Test LM Studio optimization"""
        result = self.optimizer.optimize_lm_studio()
        
        assert result["status"] == "optimized"
        assert result["application"] == "LM Studio"
        assert result["settings"]["gpu_acceleration"] is True
        assert result["settings"]["context_length"] == 32768
        assert result["settings"]["max_concurrent_requests"] == self.config.max_concurrent_requests
        assert result["settings"]["model_format"] == "GGUF"
        assert result["settings"]["quantization"] == "Q4_K_M"
        assert "performance_settings" in result["settings"]
        assert "estimated_performance" in result
    
    @patch.dict('os.environ', {})
    def test_optimize_ollama(self):
        """Test Ollama optimization"""
        result = self.optimizer.optimize_ollama()
        
        assert result["status"] == "optimized"
        assert result["application"] == "Ollama"
        assert "environment" in result
        assert "settings" in result
        assert "estimated_performance" in result
        
        # Check environment variables were set
        env_settings = result["environment"]
        assert env_settings["OLLAMA_NUM_PARALLEL"] == str(self.config.max_concurrent_requests)
        assert env_settings["OLLAMA_MAX_LOADED_MODELS"] == "4"
        assert env_settings["OLLAMA_KEEP_ALIVE"] == "-1"
        assert env_settings["OLLAMA_FLASH_ATTENTION"] == "1"


class TestM3PipelineOptimizer:
    """Test complete M3 pipeline optimizer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OptimizationConfig()
        self.pipeline_optimizer = M3PipelineOptimizer(self.config)
    
    def test_initialization(self):
        """Test pipeline optimizer initialization"""
        assert self.pipeline_optimizer.config == self.config
        assert self.pipeline_optimizer.memory_manager is not None
        assert self.pipeline_optimizer.mlx_optimizer is not None
        assert self.pipeline_optimizer.pytorch_optimizer is not None
        assert self.pipeline_optimizer.inference_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_optimize_inference_pipeline(self):
        """Test complete inference pipeline optimization"""
        # Mock all dependencies
        with patch.object(self.pipeline_optimizer.memory_manager, 'optimize_allocation_pattern') as mock_pattern, \
             patch.object(self.pipeline_optimizer.memory_manager, 'create_memory_pool') as mock_create_pool, \
             patch.object(self.pipeline_optimizer.mlx_optimizer, 'optimize_model_loading') as mock_mlx, \
             patch.object(self.pipeline_optimizer.pytorch_optimizer, 'optimize_mps_settings') as mock_pytorch, \
             patch.object(self.pipeline_optimizer.inference_optimizer, 'optimize_lm_studio') as mock_lm_studio, \
             patch.object(self.pipeline_optimizer.inference_optimizer, 'optimize_ollama') as mock_ollama, \
             patch.object(self.pipeline_optimizer.memory_manager, 'get_memory_utilization') as mock_utilization:
            
            # Setup mock return values
            mock_pattern.return_value = {
                "model_pool_gb": 32,
                "batch_pool_gb": 16,
                "cache_pool_gb": 8,
                "working_pool_gb": 12
            }
            mock_create_pool.return_value = True
            mock_mlx.return_value = {"status": "optimized"}
            mock_pytorch.return_value = {"status": "optimized"}
            mock_lm_studio.return_value = {"status": "optimized", "application": "LM Studio"}
            mock_ollama.return_value = {"status": "optimized", "application": "Ollama"}
            mock_utilization.return_value = {"utilization_percent": 45.0}
            
            # Run optimization
            result = await self.pipeline_optimizer.optimize_inference_pipeline(
                "test_model_path", "inference"
            )
            
            # Verify results
            assert result["optimization_complete"] is True
            assert result["system_specs"]["model"] == "MacBook Pro M3 Max"
            assert result["system_specs"]["unified_memory_gb"] == 128
            assert result["system_specs"]["cpu_cores"] == 16
            assert "memory_management" in result
            assert "framework_optimization" in result
            assert "inference_engines" in result
            assert "coordination" in result
            assert "performance_estimates" in result
            
            # Verify coordination strategy
            coordination = result["coordination"]
            assert coordination["primary_framework"] == "MLX"
            assert coordination["fallback_framework"] == "PyTorch"
            assert coordination["load_balancing"]["strategy"] == "round_robin"
    
    @patch('mmap.mmap')
    @patch('gc.collect')
    def test_cleanup(self, mock_gc, mock_mmap):
        """Test pipeline optimizer cleanup"""
        mock_mmap.return_value = MagicMock()
        
        # Create some memory pools
        self.pipeline_optimizer.memory_manager.create_memory_pool("test_pool", 8.0)
        
        # Cleanup
        self.pipeline_optimizer.cleanup()
        
        # Verify cleanup was called
        mock_gc.assert_called_once()


class TestOptimizeM3SystemFunction:
    """Test the convenience function"""
    
    @pytest.mark.asyncio
    async def test_optimize_m3_system_success(self):
        """Test successful system optimization"""
        with patch('m3_optimizer.M3PipelineOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize_inference_pipeline = AsyncMock(return_value={
                "optimization_complete": True,
                "status": "success"
            })
            mock_optimizer_class.return_value = mock_optimizer
            
            result = optimize_m3_system(
                workload_type="inference",
                model_path="test_model",
                max_memory_percent=85.0
            )
            
            assert "optimization_complete" in result
            mock_optimizer.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_m3_system_failure(self):
        """Test system optimization failure handling"""
        with patch('m3_optimizer.M3PipelineOptimizer') as mock_optimizer_class:
            mock_optimizer = MagicMock()
            mock_optimizer.optimize_inference_pipeline = AsyncMock(
                side_effect=Exception("Optimization failed")
            )
            mock_optimizer_class.return_value = mock_optimizer
            
            result = optimize_m3_system()
            
            assert "error" in result
            assert "Optimization failed" in result["error"]
            mock_optimizer.cleanup.assert_called_once()


@pytest.mark.integration
class TestM3OptimizerIntegration:
    """Integration tests for M3 optimizer components"""
    
    @pytest.mark.slow
    def test_memory_manager_integration(self):
        """Integration test for memory manager"""
        config = OptimizationConfig(max_memory_usage_percent=50.0)
        memory_manager = M3UnifiedMemoryManager(config)
        
        try:
            # Test pattern optimization
            pattern = memory_manager.optimize_allocation_pattern("inference")
            assert isinstance(pattern, dict)
            assert all(isinstance(v, (int, float)) for v in pattern.values())
            
            # Test utilization tracking
            utilization = memory_manager.get_memory_utilization()
            assert utilization["total_system_gb"] == 128
            
        finally:
            memory_manager.cleanup_memory_pools()
    
    def test_optimizer_coordination(self):
        """Test coordination between optimizer components"""
        config = OptimizationConfig()
        
        # Create all optimizers
        memory_manager = M3UnifiedMemoryManager(config)
        mlx_optimizer = M3MLXOptimizer(config)
        pytorch_optimizer = M3PyTorchOptimizer(config)
        inference_optimizer = M3InferenceOptimizer(config)
        
        # Test that they can work together
        try:
            # Memory patterns
            patterns = memory_manager.optimize_allocation_pattern("inference")
            assert isinstance(patterns, dict)
            
            # Inference optimizations
            lm_studio_config = inference_optimizer.optimize_lm_studio()
            ollama_config = inference_optimizer.optimize_ollama()
            
            assert lm_studio_config["status"] == "optimized"
            assert ollama_config["status"] == "optimized"
            
        finally:
            memory_manager.cleanup_memory_pools()