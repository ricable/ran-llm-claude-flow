#!/usr/bin/env python3
"""
Auto-Adjustment Algorithms for Dynamic Scaling
Phase 2 MCP Advanced Features - Intelligent Parameter Tuning
Automatically adjusts ML model parameters, batch sizes, and processing configurations
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdjustmentType(Enum):
    """Types of automatic adjustments"""
    BATCH_SIZE = "batch_size"
    MODEL_SIZE = "model_size" 
    CONCURRENCY = "concurrency"
    MEMORY_ALLOCATION = "memory_allocation"
    CACHE_SIZE = "cache_size"
    IPC_BUFFER_SIZE = "ipc_buffer_size"
    PROCESSING_THRESHOLD = "processing_threshold"
    QUALITY_THRESHOLD = "quality_threshold"

class OptimizationStrategy(Enum):
    """Optimization strategies for parameter tuning"""
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HILL_CLIMBING = "hill_climbing"
    ADAPTIVE_LEARNING = "adaptive_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization"""
    throughput_docs_per_hour: float
    average_latency_ms: float
    memory_utilization_percent: float
    cpu_utilization_percent: float
    quality_score: float
    error_rate_percent: float
    ipc_latency_ms: float
    cache_hit_rate: float
    timestamp: float
    
    def composite_score(self) -> float:
        """Calculate composite performance score (0-1, higher is better)"""
        # Normalize and weight different metrics
        throughput_norm = min(self.throughput_docs_per_hour / 50.0, 1.0)  # Target 50+ docs/hour
        latency_norm = max(1.0 - self.average_latency_ms / 1000.0, 0.0)   # Target < 1s
        memory_norm = max(1.0 - abs(self.memory_utilization_percent - 85.0) / 15.0, 0.0)  # Target 85%
        cpu_norm = max(1.0 - abs(self.cpu_utilization_percent - 80.0) / 20.0, 0.0)       # Target 80%
        quality_norm = self.quality_score  # Already 0-1
        error_norm = max(1.0 - self.error_rate_percent / 5.0, 0.0)       # Target < 5%
        ipc_norm = max(1.0 - self.ipc_latency_ms / 10.0, 0.0)           # Target < 10ms
        cache_norm = self.cache_hit_rate  # Already 0-1
        
        # Weighted composite score
        weights = {
            'throughput': 0.25, 'latency': 0.20, 'memory': 0.15, 'cpu': 0.15,
            'quality': 0.15, 'error': 0.05, 'ipc': 0.03, 'cache': 0.02
        }
        
        return (throughput_norm * weights['throughput'] +
                latency_norm * weights['latency'] +
                memory_norm * weights['memory'] +
                cpu_norm * weights['cpu'] +
                quality_norm * weights['quality'] +
                error_norm * weights['error'] +
                ipc_norm * weights['ipc'] +
                cache_norm * weights['cache'])

@dataclass
class AdjustmentCandidate:
    """Candidate parameter adjustment"""
    parameter_name: str
    adjustment_type: AdjustmentType
    current_value: float
    suggested_value: float
    expected_improvement: float
    confidence: float
    risk_level: str  # "low", "medium", "high"
    rollback_plan: Optional[Dict] = None

@dataclass
class AdjustmentResult:
    """Result of applying an adjustment"""
    parameter_name: str
    old_value: float
    new_value: float
    performance_before: PerformanceMetrics
    performance_after: Optional[PerformanceMetrics]
    actual_improvement: Optional[float]
    success: bool
    timestamp: float
    notes: str

class AdaptiveLearningOptimizer:
    """Adaptive learning optimizer for parameter tuning"""
    
    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
        self.performance_history = deque(maxlen=100)
        
    def update_parameter(self, parameter_name: str, current_value: float, 
                        gradient: float) -> float:
        """Update parameter using momentum-based gradient descent"""
        if parameter_name not in self.velocity:
            self.velocity[parameter_name] = 0.0
            
        # Update velocity with momentum
        self.velocity[parameter_name] = (self.momentum * self.velocity[parameter_name] - 
                                       self.learning_rate * gradient)
        
        # Update parameter
        new_value = current_value + self.velocity[parameter_name]
        return max(0.0, new_value)  # Ensure non-negative values
    
    def calculate_gradient(self, parameter_name: str, current_value: float,
                          current_performance: float) -> float:
        """Calculate approximate gradient for parameter"""
        if len(self.performance_history) < 2:
            return 0.0
            
        # Simple finite difference approximation
        prev_performance = self.performance_history[-2]
        performance_change = current_performance - prev_performance
        
        # Assume small parameter change (this would be improved with actual history)
        parameter_change = 0.1  # This is a simplification
        
        return performance_change / parameter_change if parameter_change != 0 else 0.0

class BayesianOptimizer:
    """Bayesian optimization for parameter tuning"""
    
    def __init__(self):
        self.parameter_history = defaultdict(list)
        self.performance_history = []
        
    def suggest_parameter(self, parameter_name: str, current_value: float,
                         bounds: Tuple[float, float]) -> float:
        """Suggest next parameter value using Bayesian optimization"""
        if len(self.parameter_history[parameter_name]) < 3:
            # Random exploration for initial samples
            lower, upper = bounds
            return np.random.uniform(lower, upper)
        
        # Simplified Gaussian Process approximation
        values = np.array(self.parameter_history[parameter_name])
        performances = np.array(self.performance_history[-len(values):])
        
        # Find the value that gave best performance
        best_idx = np.argmax(performances)
        best_value = values[best_idx]
        
        # Add some exploration around the best value
        exploration_factor = 0.1 * (bounds[1] - bounds[0])
        suggested = best_value + np.random.normal(0, exploration_factor)
        
        # Ensure within bounds
        return max(bounds[0], min(bounds[1], suggested))
    
    def update_observation(self, parameter_name: str, value: float, 
                          performance: float):
        """Update observation for parameter"""
        self.parameter_history[parameter_name].append(value)
        self.performance_history.append(performance)

class AutoAdjuster:
    """Main auto-adjustment system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.performance_history = deque(maxlen=1000)
        self.adjustment_history = []
        self.active_experiments = {}
        
        # Optimizers
        self.adaptive_optimizer = AdaptiveLearningOptimizer()
        self.bayesian_optimizer = BayesianOptimizer()
        
        # Parameter bounds and constraints
        self.parameter_bounds = {
            "batch_size": (1, 256),
            "memory_allocation_gb": (10.0, 80.0),
            "concurrency_level": (1, 16),
            "cache_size_gb": (1.0, 20.0),
            "ipc_buffer_size_mb": (64, 1024),
            "quality_threshold": (0.5, 0.95),
        }
        
        # Current parameter values
        self.current_parameters = {
            "batch_size": 32,
            "memory_allocation_gb": 60.0,
            "concurrency_level": 4,
            "cache_size_gb": 5.0,
            "ipc_buffer_size_mb": 256,
            "quality_threshold": 0.75,
        }
        
        # Adjustment constraints
        self.max_adjustments_per_hour = 10
        self.adjustment_cooldown_minutes = 5
        self.last_adjustment_time = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file"""
        default_config = {
            "optimization_strategy": "adaptive_learning",
            "enable_auto_adjustment": True,
            "conservative_mode": False,
            "min_confidence_threshold": 0.7,
            "max_performance_regression": 0.05,
            "learning_rate": 0.1,
            "exploration_rate": 0.1,
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
    
    async def analyze_and_adjust(self, current_metrics: PerformanceMetrics) -> List[AdjustmentResult]:
        """Analyze current performance and make adjustments"""
        if not self.config["enable_auto_adjustment"]:
            return []
            
        self.performance_history.append(current_metrics)
        
        # Get adjustment candidates
        candidates = await self._generate_adjustment_candidates(current_metrics)
        
        # Filter candidates based on constraints
        viable_candidates = self._filter_candidates(candidates)
        
        # Apply adjustments
        results = []
        for candidate in viable_candidates:
            if await self._should_apply_adjustment(candidate):
                result = await self._apply_adjustment(candidate, current_metrics)
                results.append(result)
                
        return results
    
    async def _generate_adjustment_candidates(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Generate adjustment candidates based on current performance"""
        candidates = []
        
        # Analyze each potential adjustment type
        if metrics.throughput_docs_per_hour < 25.0:
            # Low throughput - consider increasing batch size or concurrency
            candidates.extend(self._suggest_throughput_improvements(metrics))
            
        if metrics.memory_utilization_percent > 90.0:
            # High memory usage - consider reducing allocations
            candidates.extend(self._suggest_memory_optimizations(metrics))
            
        if metrics.average_latency_ms > 500.0:
            # High latency - consider optimizations
            candidates.extend(self._suggest_latency_optimizations(metrics))
            
        if metrics.ipc_latency_ms > 5.0:
            # IPC bottleneck - optimize communication
            candidates.extend(self._suggest_ipc_optimizations(metrics))
            
        if metrics.cache_hit_rate < 0.8:
            # Low cache efficiency - optimize caching
            candidates.extend(self._suggest_cache_optimizations(metrics))
            
        if metrics.quality_score < 0.8:
            # Quality issues - adjust quality thresholds
            candidates.extend(self._suggest_quality_optimizations(metrics))
            
        return candidates
    
    def _suggest_throughput_improvements(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Suggest improvements for throughput"""
        candidates = []
        
        current_batch_size = self.current_parameters["batch_size"]
        current_concurrency = self.current_parameters["concurrency_level"]
        
        # Increase batch size if memory allows
        if metrics.memory_utilization_percent < 80.0 and current_batch_size < 128:
            new_batch_size = min(current_batch_size * 1.5, 128)
            candidates.append(AdjustmentCandidate(
                parameter_name="batch_size",
                adjustment_type=AdjustmentType.BATCH_SIZE,
                current_value=current_batch_size,
                suggested_value=new_batch_size,
                expected_improvement=0.15,
                confidence=0.8,
                risk_level="low"
            ))
        
        # Increase concurrency if CPU allows
        if metrics.cpu_utilization_percent < 70.0 and current_concurrency < 8:
            new_concurrency = min(current_concurrency + 1, 8)
            candidates.append(AdjustmentCandidate(
                parameter_name="concurrency_level",
                adjustment_type=AdjustmentType.CONCURRENCY,
                current_value=current_concurrency,
                suggested_value=new_concurrency,
                expected_improvement=0.12,
                confidence=0.75,
                risk_level="medium"
            ))
            
        return candidates
    
    def _suggest_memory_optimizations(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Suggest memory optimizations"""
        candidates = []
        
        current_allocation = self.current_parameters["memory_allocation_gb"]
        current_cache_size = self.current_parameters["cache_size_gb"]
        
        # Reduce memory allocation if heavily utilized
        if metrics.memory_utilization_percent > 90.0:
            new_allocation = current_allocation * 0.9
            candidates.append(AdjustmentCandidate(
                parameter_name="memory_allocation_gb",
                adjustment_type=AdjustmentType.MEMORY_ALLOCATION,
                current_value=current_allocation,
                suggested_value=new_allocation,
                expected_improvement=0.08,
                confidence=0.7,
                risk_level="medium"
            ))
        
        # Adjust cache size
        if metrics.cache_hit_rate < 0.8 and current_cache_size < 15.0:
            new_cache_size = min(current_cache_size * 1.2, 15.0)
            candidates.append(AdjustmentCandidate(
                parameter_name="cache_size_gb",
                adjustment_type=AdjustmentType.CACHE_SIZE,
                current_value=current_cache_size,
                suggested_value=new_cache_size,
                expected_improvement=0.05,
                confidence=0.8,
                risk_level="low"
            ))
            
        return candidates
    
    def _suggest_latency_optimizations(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Suggest latency optimizations"""
        candidates = []
        
        current_batch_size = self.current_parameters["batch_size"]
        current_buffer_size = self.current_parameters["ipc_buffer_size_mb"]
        
        # Reduce batch size for lower latency
        if metrics.average_latency_ms > 500.0 and current_batch_size > 8:
            new_batch_size = max(current_batch_size * 0.8, 8)
            candidates.append(AdjustmentCandidate(
                parameter_name="batch_size",
                adjustment_type=AdjustmentType.BATCH_SIZE,
                current_value=current_batch_size,
                suggested_value=new_batch_size,
                expected_improvement=0.10,
                confidence=0.75,
                risk_level="low"
            ))
        
        # Increase buffer size for smoother processing
        if current_buffer_size < 512:
            new_buffer_size = min(current_buffer_size * 1.5, 512)
            candidates.append(AdjustmentCandidate(
                parameter_name="ipc_buffer_size_mb",
                adjustment_type=AdjustmentType.IPC_BUFFER_SIZE,
                current_value=current_buffer_size,
                suggested_value=new_buffer_size,
                expected_improvement=0.06,
                confidence=0.6,
                risk_level="low"
            ))
            
        return candidates
    
    def _suggest_ipc_optimizations(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Suggest IPC optimizations"""
        candidates = []
        
        current_buffer_size = self.current_parameters["ipc_buffer_size_mb"]
        
        if metrics.ipc_latency_ms > 5.0:
            # Increase buffer size to reduce IPC overhead
            new_buffer_size = min(current_buffer_size * 1.3, 1024)
            candidates.append(AdjustmentCandidate(
                parameter_name="ipc_buffer_size_mb",
                adjustment_type=AdjustmentType.IPC_BUFFER_SIZE,
                current_value=current_buffer_size,
                suggested_value=new_buffer_size,
                expected_improvement=0.08,
                confidence=0.7,
                risk_level="low"
            ))
            
        return candidates
    
    def _suggest_cache_optimizations(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Suggest cache optimizations"""
        candidates = []
        
        current_cache_size = self.current_parameters["cache_size_gb"]
        
        if metrics.cache_hit_rate < 0.8 and current_cache_size < 20.0:
            new_cache_size = min(current_cache_size * 1.4, 20.0)
            candidates.append(AdjustmentCandidate(
                parameter_name="cache_size_gb",
                adjustment_type=AdjustmentType.CACHE_SIZE,
                current_value=current_cache_size,
                suggested_value=new_cache_size,
                expected_improvement=0.12,
                confidence=0.8,
                risk_level="low"
            ))
            
        return candidates
    
    def _suggest_quality_optimizations(self, metrics: PerformanceMetrics) -> List[AdjustmentCandidate]:
        """Suggest quality optimizations"""
        candidates = []
        
        current_threshold = self.current_parameters["quality_threshold"]
        
        if metrics.quality_score < 0.8 and current_threshold > 0.6:
            # Lower quality threshold to improve throughput
            new_threshold = max(current_threshold - 0.05, 0.6)
            candidates.append(AdjustmentCandidate(
                parameter_name="quality_threshold",
                adjustment_type=AdjustmentType.QUALITY_THRESHOLD,
                current_value=current_threshold,
                suggested_value=new_threshold,
                expected_improvement=0.07,
                confidence=0.6,
                risk_level="medium"
            ))
            
        return candidates
    
    def _filter_candidates(self, candidates: List[AdjustmentCandidate]) -> List[AdjustmentCandidate]:
        """Filter candidates based on constraints and policies"""
        filtered = []
        
        for candidate in candidates:
            # Check confidence threshold
            if candidate.confidence < self.config["min_confidence_threshold"]:
                continue
                
            # Check cooldown period
            if candidate.parameter_name in self.last_adjustment_time:
                time_since_last = time.time() - self.last_adjustment_time[candidate.parameter_name]
                if time_since_last < self.adjustment_cooldown_minutes * 60:
                    continue
            
            # Check parameter bounds
            param_name = candidate.parameter_name
            if param_name in self.parameter_bounds:
                lower, upper = self.parameter_bounds[param_name]
                if not (lower <= candidate.suggested_value <= upper):
                    continue
            
            # Conservative mode checks
            if self.config["conservative_mode"] and candidate.risk_level == "high":
                continue
                
            filtered.append(candidate)
            
        return filtered
    
    async def _should_apply_adjustment(self, candidate: AdjustmentCandidate) -> bool:
        """Determine if adjustment should be applied"""
        # Check hourly adjustment limit
        current_hour = int(time.time() // 3600)
        recent_adjustments = [
            adj for adj in self.adjustment_history 
            if int(adj.timestamp // 3600) == current_hour
        ]
        
        if len(recent_adjustments) >= self.max_adjustments_per_hour:
            return False
            
        # Check if there's already an active experiment for this parameter
        if candidate.parameter_name in self.active_experiments:
            return False
            
        return True
    
    async def _apply_adjustment(self, candidate: AdjustmentCandidate, 
                              baseline_metrics: PerformanceMetrics) -> AdjustmentResult:
        """Apply the adjustment and measure results"""
        old_value = self.current_parameters[candidate.parameter_name]
        
        try:
            # Apply the adjustment
            self.current_parameters[candidate.parameter_name] = candidate.suggested_value
            self.last_adjustment_time[candidate.parameter_name] = time.time()
            
            # Wait for system to stabilize
            await asyncio.sleep(30)  # 30 second stabilization period
            
            # This would measure actual performance - simplified for now
            success = True
            notes = f"Successfully adjusted {candidate.parameter_name}"
            
            result = AdjustmentResult(
                parameter_name=candidate.parameter_name,
                old_value=old_value,
                new_value=candidate.suggested_value,
                performance_before=baseline_metrics,
                performance_after=None,  # Would be measured in real implementation
                actual_improvement=None,  # Would be calculated in real implementation
                success=success,
                timestamp=time.time(),
                notes=notes
            )
            
            self.adjustment_history.append(result)
            logger.info(f"Applied adjustment: {candidate.parameter_name} "
                       f"{old_value} -> {candidate.suggested_value}")
            
            return result
            
        except Exception as e:
            # Rollback on failure
            self.current_parameters[candidate.parameter_name] = old_value
            
            result = AdjustmentResult(
                parameter_name=candidate.parameter_name,
                old_value=old_value,
                new_value=old_value,  # Rolled back
                performance_before=baseline_metrics,
                performance_after=None,
                actual_improvement=None,
                success=False,
                timestamp=time.time(),
                notes=f"Adjustment failed: {str(e)}"
            )
            
            self.adjustment_history.append(result)
            logger.error(f"Failed to apply adjustment for {candidate.parameter_name}: {e}")
            
            return result
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values"""
        return self.current_parameters.copy()
    
    def get_adjustment_history(self) -> List[AdjustmentResult]:
        """Get history of adjustments"""
        return self.adjustment_history.copy()
    
    def get_performance_trend(self, metric_name: str, window_size: int = 20) -> Dict[str, float]:
        """Calculate performance trend for a specific metric"""
        if len(self.performance_history) < window_size:
            return {"trend": 0.0, "confidence": 0.0}
        
        recent_metrics = list(self.performance_history)[-window_size:]
        values = []
        
        for metrics in recent_metrics:
            if metric_name == "composite_score":
                values.append(metrics.composite_score())
            else:
                values.append(getattr(metrics, metric_name, 0.0))
        
        if len(values) < 2:
            return {"trend": 0.0, "confidence": 0.0}
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "trend": float(slope),
            "confidence": float(r_squared),
            "current_value": float(values[-1]),
            "predicted_next": float(slope * len(values) + intercept)
        }
    
    async def emergency_rollback(self, parameter_name: str) -> bool:
        """Emergency rollback of a parameter"""
        if not self.adjustment_history:
            return False
        
        # Find the last adjustment for this parameter
        for adjustment in reversed(self.adjustment_history):
            if adjustment.parameter_name == parameter_name and adjustment.success:
                self.current_parameters[parameter_name] = adjustment.old_value
                logger.warning(f"Emergency rollback: {parameter_name} -> {adjustment.old_value}")
                return True
                
        return False
    
    def save_state(self, filepath: str):
        """Save current state to file"""
        state = {
            "current_parameters": self.current_parameters,
            "adjustment_history": [asdict(adj) for adj in self.adjustment_history[-50:]],  # Last 50 adjustments
            "timestamp": time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"State saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self, filepath: str):
        """Load state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_parameters.update(state.get("current_parameters", {}))
            
            # Reload adjustment history
            history_data = state.get("adjustment_history", [])
            for adj_data in history_data:
                # Convert back to AdjustmentResult objects
                adj_data["performance_before"] = PerformanceMetrics(**adj_data["performance_before"])
                if adj_data["performance_after"]:
                    adj_data["performance_after"] = PerformanceMetrics(**adj_data["performance_after"])
                self.adjustment_history.append(AdjustmentResult(**adj_data))
            
            logger.info(f"State loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

# Example usage and testing
async def main():
    """Example usage of the auto-adjuster"""
    adjuster = AutoAdjuster()
    
    # Simulate some performance metrics
    test_metrics = PerformanceMetrics(
        throughput_docs_per_hour=20.0,  # Below target
        average_latency_ms=600.0,       # High latency
        memory_utilization_percent=85.0,
        cpu_utilization_percent=60.0,   # Low CPU usage
        quality_score=0.82,
        error_rate_percent=1.5,
        ipc_latency_ms=8.0,             # High IPC latency
        cache_hit_rate=0.75,            # Low cache hit rate
        timestamp=time.time()
    )
    
    print("Current performance metrics:")
    print(f"  Composite score: {test_metrics.composite_score():.3f}")
    print(f"  Throughput: {test_metrics.throughput_docs_per_hour} docs/hour")
    print(f"  Latency: {test_metrics.average_latency_ms}ms")
    
    # Analyze and get adjustment suggestions
    adjustments = await adjuster.analyze_and_adjust(test_metrics)
    
    print(f"\nApplied {len(adjustments)} adjustments:")
    for adj in adjustments:
        print(f"  {adj.parameter_name}: {adj.old_value} -> {adj.new_value} ({'✓' if adj.success else '✗'})")
    
    print(f"\nCurrent parameters:")
    for param, value in adjuster.get_current_parameters().items():
        print(f"  {param}: {value}")

if __name__ == "__main__":
    asyncio.run(main())