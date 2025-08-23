#!/usr/bin/env python3
"""
Model Pooling and Caching System

Advanced model pooling and caching strategies for Qwen3 models with
intelligent resource management, warming strategies, and performance optimization.

Author: Claude Code ML Integration Specialist
Date: 2025-08-23
"""

import asyncio
import logging
import psutil
import time
import threading
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import weakref
import gc
import subprocess

# Import model manager
try:
    from .qwen3_manager import Qwen3DynamicManager, ModelVariant, TaskComplexity
    from .local_inference import UnifiedLocalInferenceClient, InferenceRequest, InferenceResponse
except ImportError:
    # Fallback imports for testing
    import sys
    sys.path.append('.')
    from qwen3_manager import Qwen3DynamicManager, ModelVariant, TaskComplexity
    from local_inference import UnifiedLocalInferenceClient, InferenceRequest, InferenceResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoolingStrategy(Enum):
    """Model pooling strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    PREDICTIVE = "predictive"  # Based on usage patterns
    MEMORY_AWARE = "memory_aware"  # Based on memory constraints
    PERFORMANCE_BASED = "performance_based"  # Based on performance metrics

class WarmupStrategy(Enum):
    """Model warmup strategies"""
    IMMEDIATE = "immediate"  # Warmup immediately after loading
    LAZY = "lazy"  # Warmup on first use
    SCHEDULED = "scheduled"  # Warmup on schedule
    PREDICTIVE = "predictive"  # Warmup based on predicted usage

@dataclass
class ModelPoolEntry:
    """Entry in the model pool"""
    model_key: str
    model_variant: ModelVariant
    model_instance: Any
    load_time: float
    last_used: float
    usage_count: int = 0
    memory_usage_gb: float = 0.0
    warmup_completed: bool = False
    performance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PoolingMetrics:
    """Metrics for the model pool"""
    total_models_loaded: int = 0
    active_models: int = 0
    total_memory_used_gb: float = 0.0
    cache_hit_rate: float = 0.0
    average_load_time: float = 0.0
    warmup_success_rate: float = 0.0
    eviction_count: int = 0

class ModelPredictor:
    """Predict model usage patterns for proactive loading"""
    
    def __init__(self, history_window: int = 1000):
        self.usage_history = deque(maxlen=history_window)
        self.model_patterns = defaultdict(list)
        self.time_patterns = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_usage(self, model_key: str, timestamp: float = None):
        """Record model usage for pattern learning"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.usage_history.append((model_key, timestamp))
            
            # Extract hour of day for time-based patterns
            hour = int((timestamp % 86400) / 3600)  # Hour of day (0-23)
            self.time_patterns[hour].append(model_key)
            
            # Keep recent patterns only
            if len(self.time_patterns[hour]) > 100:
                self.time_patterns[hour] = self.time_patterns[hour][-50:]
    
    def predict_next_models(self, current_time: float = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict which models are likely to be used next"""
        if current_time is None:
            current_time = time.time()
        
        with self.lock:
            # Time-based prediction
            hour = int((current_time % 86400) / 3600)
            recent_models = self.time_patterns.get(hour, [])
            
            if not recent_models:
                return []
            
            # Count frequency in this hour
            model_counts = defaultdict(int)
            for model in recent_models[-50:]:  # Recent history for this hour
                model_counts[model] += 1
            
            # Calculate probabilities
            total_count = sum(model_counts.values())
            predictions = []
            
            for model, count in model_counts.items():
                probability = count / total_count
                predictions.append((model, probability))
            
            # Sort by probability and return top k
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:top_k]
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for analysis"""
        with self.lock:
            if not self.usage_history:
                return {}
            
            model_usage = defaultdict(int)
            for model, _ in self.usage_history:
                model_usage[model] += 1
            
            total_usage = len(self.usage_history)
            
            return {
                'total_requests': total_usage,
                'unique_models': len(model_usage),
                'model_distribution': dict(model_usage),
                'most_used_model': max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None,
                'history_span_hours': (self.usage_history[-1][1] - self.usage_history[0][1]) / 3600 if len(self.usage_history) > 1 else 0
            }

class ModelWarmer:
    """Intelligent model warming system"""
    
    def __init__(self, warmup_prompts: List[str] = None):
        self.warmup_prompts = warmup_prompts or [
            "Hello, this is a test.",
            "Explain artificial intelligence.",
            "Write a short summary.",
            "Generate some code.",
            "Analyze this data."
        ]
        self.warmup_tasks = {}
        self.warmup_results = {}
    
    async def warm_up_model(self, model_pool_entry: ModelPoolEntry, 
                           inference_client: Any) -> bool:
        """Warm up a model with test prompts"""
        model_key = model_pool_entry.model_key
        
        if model_pool_entry.warmup_completed:
            return True
        
        logger.info(f"Starting warmup for model {model_key}")
        start_time = time.time()
        
        try:
            # Use a simple prompt for warmup
            warmup_prompt = self.warmup_prompts[0]
            
            if hasattr(inference_client, 'generate_text'):
                response = await inference_client.generate_text(
                    warmup_prompt,
                    model_type=model_key,
                    max_tokens=50,
                    temperature=0.5
                )
                
                warmup_time = time.time() - start_time
                model_pool_entry.warmup_completed = True
                
                # Store warmup results
                self.warmup_results[model_key] = {
                    'success': True,
                    'warmup_time': warmup_time,
                    'tokens_per_second': response.tokens_per_second if hasattr(response, 'tokens_per_second') else 0,
                    'timestamp': time.time()
                }
                
                logger.info(f"Warmup completed for {model_key} in {warmup_time:.2f}s")
                return True
            else:
                # Generic warmup for other client types
                await asyncio.sleep(0.1)  # Simulate warmup time
                model_pool_entry.warmup_completed = True
                logger.info(f"Generic warmup completed for {model_key}")
                return True
        
        except Exception as e:
            logger.error(f"Warmup failed for {model_key}: {e}")
            self.warmup_results[model_key] = {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
            return False
    
    async def scheduled_warmup(self, model_keys: List[str], 
                              inference_client: Any,
                              model_pool: 'AdvancedModelPool'):
        """Perform scheduled warmup for multiple models"""
        for model_key in model_keys:
            try:
                if model_key in model_pool.pool:
                    entry = model_pool.pool[model_key]
                    await self.warm_up_model(entry, inference_client)
                    await asyncio.sleep(1)  # Small delay between warmups
            except Exception as e:
                logger.error(f"Scheduled warmup failed for {model_key}: {e}")
    
    def get_warmup_statistics(self) -> Dict[str, Any]:
        """Get warmup performance statistics"""
        if not self.warmup_results:
            return {}
        
        successful = sum(1 for r in self.warmup_results.values() if r.get('success', False))
        total = len(self.warmup_results)
        
        successful_times = [r['warmup_time'] for r in self.warmup_results.values() 
                           if r.get('success', False) and 'warmup_time' in r]
        
        return {
            'total_warmups': total,
            'successful_warmups': successful,
            'success_rate': successful / total if total > 0 else 0,
            'average_warmup_time': sum(successful_times) / len(successful_times) if successful_times else 0,
            'max_warmup_time': max(successful_times) if successful_times else 0,
            'min_warmup_time': min(successful_times) if successful_times else 0
        }

class AdvancedModelPool:
    """Advanced model pool with intelligent caching and resource management"""
    
    def __init__(self,
                 max_models: int = 4,
                 max_memory_gb: float = 80.0,
                 pooling_strategy: PoolingStrategy = PoolingStrategy.MEMORY_AWARE,
                 warmup_strategy: WarmupStrategy = WarmupStrategy.PREDICTIVE,
                 enable_prediction: bool = True):
        
        self.max_models = max_models
        self.max_memory_gb = max_memory_gb
        self.pooling_strategy = pooling_strategy
        self.warmup_strategy = warmup_strategy
        
        # Core components
        self.pool: Dict[str, ModelPoolEntry] = {}
        self.model_manager = Qwen3DynamicManager(max_concurrent_models=max_models)
        self.inference_client = UnifiedLocalInferenceClient()
        
        # Advanced features
        self.predictor = ModelPredictor() if enable_prediction else None
        self.warmer = ModelWarmer()
        self.metrics = PoolingMetrics()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.background_tasks = set()
        self.optimization_interval = 300  # 5 minutes
        
        logger.info(f"Initialized AdvancedModelPool with {max_models} max models, {max_memory_gb}GB max memory")
    
    async def get_model(self, model_key: str, 
                       task_complexity: TaskComplexity = TaskComplexity.MODERATE,
                       auto_warmup: bool = True) -> ModelPoolEntry:
        """Get a model from the pool, loading if necessary"""
        
        with self.lock:
            # Record usage for prediction
            if self.predictor:
                self.predictor.record_usage(model_key)
            
            # Check if model is already in pool
            if model_key in self.pool:
                entry = self.pool[model_key]
                entry.last_used = time.time()
                entry.usage_count += 1
                
                # Warmup if needed
                if auto_warmup and not entry.warmup_completed:
                    asyncio.create_task(self.warmer.warm_up_model(entry, self.inference_client))
                
                return entry
            
            # Need to load model
            await self._ensure_capacity_for_new_model()
            
            # Load the model
            success = self.model_manager.load_model(model_key)
            if not success:
                raise RuntimeError(f"Failed to load model {model_key}")
            
            # Create pool entry
            model_data = self.model_manager.loaded_models.get(model_key)
            if not model_data:
                raise RuntimeError(f"Model {model_key} not found after loading")
            
            entry = ModelPoolEntry(
                model_key=model_key,
                model_variant=model_data['config'].variant,
                model_instance=model_data.get('model'),
                load_time=model_data['load_time'],
                last_used=time.time(),
                usage_count=1,
                memory_usage_gb=model_data['config'].max_memory_gb,
                warmup_completed=False,
                performance_score=0.0
            )
            
            self.pool[model_key] = entry
            self.metrics.total_models_loaded += 1
            self.metrics.active_models += 1
            
            # Warmup if needed
            if auto_warmup:
                asyncio.create_task(self.warmer.warm_up_model(entry, self.inference_client))
            
            logger.info(f"Loaded model {model_key} into pool")
            return entry
    
    async def _ensure_capacity_for_new_model(self):
        """Ensure there's capacity for a new model"""
        current_memory = sum(entry.memory_usage_gb for entry in self.pool.values())
        
        # Check model count limit
        while len(self.pool) >= self.max_models:
            await self._evict_model()
        
        # Check memory limit (estimate for new model - use balanced as default)
        estimated_new_memory = 12.0  # GB for balanced model
        while current_memory + estimated_new_memory > self.max_memory_gb:
            await self._evict_model()
            current_memory = sum(entry.memory_usage_gb for entry in self.pool.values())
    
    async def _evict_model(self):
        """Evict a model based on the pooling strategy"""
        if not self.pool:
            return
        
        if self.pooling_strategy == PoolingStrategy.LRU:
            # Evict least recently used
            lru_entry = min(self.pool.items(), key=lambda x: x[1].last_used)
        elif self.pooling_strategy == PoolingStrategy.LFU:
            # Evict least frequently used
            lru_entry = min(self.pool.items(), key=lambda x: x[1].usage_count)
        elif self.pooling_strategy == PoolingStrategy.MEMORY_AWARE:
            # Evict largest memory user with low usage
            def memory_usage_score(item):
                entry = item[1]
                # Higher score = higher priority for eviction
                memory_weight = entry.memory_usage_gb
                usage_weight = 1.0 / (entry.usage_count + 1)
                recency_weight = 1.0 / (time.time() - entry.last_used + 1)
                return memory_weight * usage_weight * recency_weight
            
            lru_entry = max(self.pool.items(), key=memory_usage_score)
        elif self.pooling_strategy == PoolingStrategy.PERFORMANCE_BASED:
            # Evict lowest performing model
            lru_entry = min(self.pool.items(), key=lambda x: x[1].performance_score)
        else:
            # Default to LRU
            lru_entry = min(self.pool.items(), key=lambda x: x[1].last_used)
        
        model_key = lru_entry[0]
        await self.remove_model(model_key)
        self.metrics.eviction_count += 1
        
        logger.info(f"Evicted model {model_key} using {self.pooling_strategy.value} strategy")
    
    async def remove_model(self, model_key: str) -> bool:
        """Remove a model from the pool"""
        with self.lock:
            if model_key not in self.pool:
                return False
            
            # Remove from pool
            del self.pool[model_key]
            self.metrics.active_models -= 1
            
            # Unload from model manager
            success = self.model_manager.unload_model(model_key)
            
            logger.info(f"Removed model {model_key} from pool")
            return success
    
    async def preload_predicted_models(self):
        """Preload models based on usage predictions"""
        if not self.predictor:
            return
        
        predictions = self.predictor.predict_next_models(top_k=2)
        current_memory = sum(entry.memory_usage_gb for entry in self.pool.values())
        
        for model_key, probability in predictions:
            # Only preload if model is not already loaded and we have capacity
            if (model_key not in self.pool and 
                len(self.pool) < self.max_models and 
                current_memory < self.max_memory_gb * 0.8 and  # Keep 20% buffer
                probability > 0.3):  # Only if reasonably likely
                
                try:
                    logger.info(f"Preloading predicted model {model_key} (probability: {probability:.2f})")
                    await self.get_model(model_key, auto_warmup=True)
                except Exception as e:
                    logger.warning(f"Failed to preload model {model_key}: {e}")
    
    async def optimize_pool(self) -> Dict[str, Any]:
        """Optimize the model pool based on usage patterns and performance"""
        optimization_results = {
            'timestamp': time.time(),
            'actions_taken': [],
            'models_evicted': 0,
            'models_preloaded': 0,
            'memory_optimized': False
        }
        
        # Check memory usage
        current_memory = sum(entry.memory_usage_gb for entry in self.pool.values())
        memory_utilization = current_memory / self.max_memory_gb
        
        if memory_utilization > 0.9:  # Over 90% memory usage
            # Evict low-usage models
            low_usage_models = [
                key for key, entry in self.pool.items()
                if entry.usage_count < 5 and time.time() - entry.last_used > 3600  # 1 hour
            ]
            
            for model_key in low_usage_models[:2]:  # Evict up to 2 models
                await self.remove_model(model_key)
                optimization_results['actions_taken'].append(f"Evicted low-usage model: {model_key}")
                optimization_results['models_evicted'] += 1
            
            optimization_results['memory_optimized'] = True
        
        # Preload predicted models if we have capacity
        if len(self.pool) < self.max_models * 0.8:  # Less than 80% capacity
            await self.preload_predicted_models()
            optimization_results['actions_taken'].append("Preloaded predicted models")
        
        # Update performance scores
        await self._update_performance_scores()
        
        return optimization_results
    
    async def _update_performance_scores(self):
        """Update performance scores for all models in the pool"""
        for model_key, entry in self.pool.items():
            # Get performance metrics from model manager
            metrics = self.model_manager.performance_metrics.get_average_metrics(entry.model_variant)
            
            if metrics:
                # Calculate composite performance score
                tokens_per_sec = metrics.get('avg_tokens_per_sec', 0)
                success_rate = metrics.get('success_rate', 0)
                memory_efficiency = 1.0 / (entry.memory_usage_gb + 1)  # Lower memory = higher score
                
                # Weighted composite score
                entry.performance_score = (
                    tokens_per_sec * 0.4 +
                    success_rate * 100 * 0.3 +  # Scale success rate to similar range
                    memory_efficiency * 50 * 0.3  # Scale memory efficiency
                )
    
    async def start_background_optimization(self):
        """Start background optimization task"""
        async def optimization_loop():
            while True:
                try:
                    await asyncio.sleep(self.optimization_interval)
                    await self.optimize_pool()
                except Exception as e:
                    logger.error(f"Background optimization failed: {e}")
        
        task = asyncio.create_task(optimization_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Started background optimization task")
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        current_memory = sum(entry.memory_usage_gb for entry in self.pool.values())
        
        pool_stats = {
            'timestamp': time.time(),
            'active_models': len(self.pool),
            'max_models': self.max_models,
            'memory_used_gb': current_memory,
            'max_memory_gb': self.max_memory_gb,
            'memory_utilization': current_memory / self.max_memory_gb,
            'pooling_strategy': self.pooling_strategy.value,
            'warmup_strategy': self.warmup_strategy.value,
            'models_in_pool': list(self.pool.keys())
        }
        
        # Add model-specific stats
        model_stats = {}
        for model_key, entry in self.pool.items():
            model_stats[model_key] = {
                'usage_count': entry.usage_count,
                'last_used_seconds_ago': time.time() - entry.last_used,
                'memory_usage_gb': entry.memory_usage_gb,
                'warmup_completed': entry.warmup_completed,
                'performance_score': entry.performance_score,
                'load_time': entry.load_time
            }
        
        pool_stats['model_details'] = model_stats
        
        # Add predictor stats if available
        if self.predictor:
            pool_stats['prediction_stats'] = self.predictor.get_usage_statistics()
        
        # Add warmup stats
        pool_stats['warmup_stats'] = self.warmer.get_warmup_statistics()
        
        # Add overall metrics
        pool_stats['metrics'] = {
            'total_models_loaded': self.metrics.total_models_loaded,
            'eviction_count': self.metrics.eviction_count,
            'cache_hit_rate': len(self.pool) / max(self.metrics.total_models_loaded, 1)
        }
        
        return pool_stats
    
    async def generate_with_pool(self, 
                               prompt: str,
                               model_key: Optional[str] = None,
                               task_complexity: TaskComplexity = TaskComplexity.MODERATE,
                               **kwargs) -> InferenceResponse:
        """Generate text using the model pool"""
        
        # Auto-select model if not specified
        if model_key is None:
            model_key = self.model_manager.select_optimal_model(task_complexity)
        
        # Get model from pool
        pool_entry = await self.get_model(model_key, task_complexity)
        
        # Generate using inference client
        response = await self.inference_client.generate_text(
            prompt=prompt,
            model_type=model_key,
            **kwargs
        )
        
        # Update performance score
        if hasattr(response, 'tokens_per_second'):
            pool_entry.performance_score = (
                pool_entry.performance_score * 0.9 + 
                response.tokens_per_second * 0.1  # Exponential moving average
            )
        
        return response
    
    async def shutdown(self):
        """Gracefully shutdown the model pool"""
        logger.info("Shutting down AdvancedModelPool")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Remove all models
        model_keys = list(self.pool.keys())
        for model_key in model_keys:
            await self.remove_model(model_key)
        
        # Shutdown components
        self.model_manager.shutdown()
        
        logger.info("AdvancedModelPool shutdown complete")

# Factory function for easy initialization
def create_optimized_model_pool(config: Dict[str, Any] = None) -> AdvancedModelPool:
    """Create an optimized model pool with sensible defaults"""
    
    default_config = {
        'max_models': 3,
        'max_memory_gb': 60.0,  # Conservative for M3 Max
        'pooling_strategy': PoolingStrategy.MEMORY_AWARE,
        'warmup_strategy': WarmupStrategy.PREDICTIVE,
        'enable_prediction': True
    }
    
    if config:
        default_config.update(config)
    
    pool = AdvancedModelPool(
        max_models=default_config['max_models'],
        max_memory_gb=default_config['max_memory_gb'],
        pooling_strategy=default_config['pooling_strategy'],
        warmup_strategy=default_config['warmup_strategy'],
        enable_prediction=default_config['enable_prediction']
    )
    
    return pool

# Example usage and testing
async def main():
    """Example usage of the advanced model pool"""
    
    # Create optimized pool
    pool = create_optimized_model_pool({
        'max_models': 2,
        'max_memory_gb': 30.0
    })
    
    try:
        # Start background optimization
        await pool.start_background_optimization()
        
        # Test model loading and usage
        response1 = await pool.generate_with_pool(
            "Explain machine learning concepts",
            task_complexity=TaskComplexity.MODERATE
        )
        
        print(f"Response 1: {response1.text[:100] if hasattr(response1, 'text') else 'Generated'}...")
        
        # Test with different model
        response2 = await pool.generate_with_pool(
            "Write a complex analysis",
            task_complexity=TaskComplexity.COMPLEX
        )
        
        print(f"Response 2: {response2.text[:100] if hasattr(response2, 'text') else 'Generated'}...")
        
        # Get pool statistics
        stats = pool.get_pool_statistics()
        print(f"Pool statistics: {json.dumps(stats, indent=2)}")
        
        # Test optimization
        optimization_results = await pool.optimize_pool()
        print(f"Optimization results: {json.dumps(optimization_results, indent=2)}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        await pool.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
