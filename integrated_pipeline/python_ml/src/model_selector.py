#!/usr/bin/env python3
"""
Intelligent Model Selector for Multi-Qwen3 Pipeline

Advanced model selection system optimized for M3 Max with dynamic switching
between Qwen3-1.7B (fast), Qwen3-7B (balanced), and Qwen3-30B (quality).

Performance Targets:
- Model switching latency: <3 seconds
- Memory efficiency: 90%+ utilization
- Throughput optimization: 25-30 docs/hour
- Quality consistency: >0.75 average

Author: Claude Code
Version: 2.0.0
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import asynccontextmanager
import numpy as np
from collections import deque, defaultdict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, memory monitoring will be limited")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class ModelSize(Enum):
    """Qwen3 model variants with target characteristics"""
    FAST = "1.7B"      # Speed: <2s, Memory: 4GB, Use: Simple tasks
    BALANCED = "7B"    # Speed: <5s, Memory: 12GB, Use: Standard processing
    QUALITY = "30B"    # Speed: <15s, Memory: 30GB, Use: Complex analysis


class SelectionStrategy(Enum):
    """Model selection strategies"""
    PERFORMANCE_FIRST = "performance"    # Prioritize speed
    QUALITY_FIRST = "quality"           # Prioritize accuracy
    BALANCED = "balanced"               # Balance speed and quality
    ADAPTIVE = "adaptive"               # Learn from usage patterns
    MEMORY_AWARE = "memory"             # Optimize for memory usage


class ModelBackend(Enum):
    """Available model backends"""
    MLX = "mlx"                # Apple Silicon optimized
    LMSTUDIO = "lmstudio"      # Local API server
    OLLAMA = "ollama"          # Ollama service
    TRANSFORMERS = "transformers" # HuggingFace transformers


@dataclass
class DocumentComplexity:
    """Document complexity analysis for model selection"""
    content_length: int
    parameter_count: int
    counter_count: int
    technical_term_count: int
    structural_depth: int
    readability_score: float
    domain_specificity: float
    processing_urgency: float = 0.5
    quality_requirement: float = 0.7
    
    def complexity_score(self) -> float:
        """Calculate normalized complexity score (0-1)"""
        # Length factor (normalized to 10k chars)
        length_factor = min(self.content_length / 10000.0, 1.0)
        
        # Parameter density factor
        param_factor = min(self.parameter_count / 50.0, 1.0)
        
        # Counter density factor
        counter_factor = min(self.counter_count / 20.0, 1.0)
        
        # Technical density factor
        tech_factor = min(self.technical_term_count / 100.0, 1.0)
        
        # Structural complexity factor
        structure_factor = min(self.structural_depth / 5.0, 1.0)
        
        # Domain specificity bonus
        domain_factor = self.domain_specificity
        
        # Weighted combination
        complexity = (
            length_factor * 0.20 +
            param_factor * 0.25 +
            counter_factor * 0.20 +
            tech_factor * 0.15 +
            structure_factor * 0.10 +
            domain_factor * 0.10
        )
        
        return min(complexity, 1.0)


@dataclass
class ModelCapabilities:
    """Model capabilities and performance characteristics"""
    model_size: ModelSize
    backend: ModelBackend
    max_memory_gb: float
    typical_inference_time: float
    max_context_length: int
    quality_score: float
    reliability_score: float
    batch_efficiency: float
    specializations: List[str]
    last_updated: float


@dataclass
class SelectionContext:
    """Context for model selection decision"""
    document_complexity: DocumentComplexity
    current_memory_usage: float
    available_models: List[ModelSize]
    recent_performance: Dict[ModelSize, float]
    batch_context: Optional[Dict] = None
    user_preferences: Optional[Dict] = None
    time_constraints: Optional[float] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model evaluation"""
    avg_inference_time: float
    avg_quality_score: float
    success_rate: float
    memory_efficiency: float
    throughput_docs_hour: float
    cost_per_document: float
    user_satisfaction: float
    last_measurement: float
    sample_count: int


@dataclass
class SelectionResult:
    """Result of model selection with rationale"""
    selected_model: ModelSize
    confidence: float
    rationale: str
    expected_performance: ModelPerformanceMetrics
    fallback_models: List[ModelSize]
    selection_time: float


class AdaptiveModelSelector:
    """
    Intelligent model selector with adaptive learning and performance optimization.
    
    Features:
    - Multi-criteria decision making (speed, quality, memory)
    - Adaptive learning from usage patterns
    - Performance prediction and optimization
    - Dynamic model switching with minimal latency
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self, config_path: Optional[Path] = None, strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/models_config.yaml")
        self.strategy = strategy
        
        # Model capabilities (loaded from config or defaults)
        self.model_capabilities: Dict[ModelSize, ModelCapabilities] = {}
        
        # Performance tracking
        self.performance_history: Dict[ModelSize, deque] = {
            model: deque(maxlen=100) for model in ModelSize
        }
        self.current_metrics: Dict[ModelSize, ModelPerformanceMetrics] = {}
        
        # Adaptive learning components
        self.usage_patterns: Dict[str, Any] = {}
        self.model_preferences: Dict[Tuple[float, float], ModelSize] = {}  # (complexity, quality) -> model
        self.learning_rate = 0.1
        self.exploration_factor = 0.1
        
        # System state
        self.last_selection: Optional[SelectionResult] = None
        self.selection_history: deque = deque(maxlen=1000)
        self.model_switch_count = 0
        self.total_selections = 0
        
        # Performance prediction model (simple regression)
        self.prediction_weights: Dict[ModelSize, np.ndarray] = {}
        self.prediction_bias: Dict[ModelSize, float] = {}
        
        # Initialize components
        self._initialize_model_capabilities()
        self._load_historical_data()
        
    def _initialize_model_capabilities(self):
        """Initialize model capabilities with default values"""
        self.model_capabilities = {
            ModelSize.FAST: ModelCapabilities(
                model_size=ModelSize.FAST,
                backend=ModelBackend.MLX if MLX_AVAILABLE else ModelBackend.TRANSFORMERS,
                max_memory_gb=4.0,
                typical_inference_time=1.5,
                max_context_length=4096,
                quality_score=0.72,
                reliability_score=0.95,
                batch_efficiency=0.8,
                specializations=["simple_qa", "parameter_extraction", "fast_processing"],
                last_updated=time.time()
            ),
            ModelSize.BALANCED: ModelCapabilities(
                model_size=ModelSize.BALANCED,
                backend=ModelBackend.MLX if MLX_AVAILABLE else ModelBackend.TRANSFORMERS,
                max_memory_gb=12.0,
                typical_inference_time=4.0,
                max_context_length=8192,
                quality_score=0.82,
                reliability_score=0.92,
                batch_efficiency=0.85,
                specializations=["technical_qa", "complex_reasoning", "balanced_processing"],
                last_updated=time.time()
            ),
            ModelSize.QUALITY: ModelCapabilities(
                model_size=ModelSize.QUALITY,
                backend=ModelBackend.LMSTUDIO,
                max_memory_gb=30.0,
                typical_inference_time=12.0,
                max_context_length=32768,
                quality_score=0.91,
                reliability_score=0.88,
                batch_efficiency=0.75,
                specializations=["expert_analysis", "complex_documentation", "high_quality"],
                last_updated=time.time()
            )
        }
        
        # Initialize default performance metrics
        for model_size in ModelSize:
            capabilities = self.model_capabilities[model_size]
            self.current_metrics[model_size] = ModelPerformanceMetrics(
                avg_inference_time=capabilities.typical_inference_time,
                avg_quality_score=capabilities.quality_score,
                success_rate=capabilities.reliability_score,
                memory_efficiency=0.8,
                throughput_docs_hour=3600.0 / capabilities.typical_inference_time,
                cost_per_document=capabilities.typical_inference_time / 10.0,  # Simplified cost model
                user_satisfaction=0.8,
                last_measurement=time.time(),
                sample_count=0
            )
        
        self.logger.info("Model capabilities initialized for 3 Qwen3 variants")
    
    def _load_historical_data(self):
        """Load historical performance data and learned patterns"""
        try:
            history_file = Path("model_selection_history.json")
            if history_file.exists():
                with open(history_file) as f:
                    data = json.load(f)
                    
                # Load usage patterns
                self.usage_patterns = data.get("usage_patterns", {})
                
                # Load model preferences
                preferences_data = data.get("model_preferences", {})
                for key_str, model_str in preferences_data.items():
                    try:
                        complexity, quality = map(float, key_str.strip("()").split(", "))
                        self.model_preferences[(complexity, quality)] = ModelSize(model_str)
                    except (ValueError, KeyError):
                        continue
                
                # Load performance metrics
                metrics_data = data.get("performance_metrics", {})
                for model_str, metrics_dict in metrics_data.items():
                    try:
                        model_size = ModelSize(model_str)
                        self.current_metrics[model_size] = ModelPerformanceMetrics(**metrics_dict)
                    except (ValueError, KeyError, TypeError):
                        continue
                        
                self.logger.info(f"Loaded historical data: {len(self.model_preferences)} preferences, "
                               f"{len(self.usage_patterns)} patterns")
        except Exception as e:
            self.logger.warning(f"Failed to load historical data: {e}")
    
    async def select_optimal_model(self, context: SelectionContext) -> SelectionResult:
        """
        Select optimal model based on context and learned patterns.
        
        Args:
            context: Selection context with document complexity and constraints
            
        Returns:
            SelectionResult with model choice and rationale
        """
        start_time = time.time()
        self.total_selections += 1
        
        self.logger.debug(f"Selecting model for document complexity: {context.document_complexity.complexity_score():.3f}")
        
        # Apply selection strategy
        if self.strategy == SelectionStrategy.PERFORMANCE_FIRST:
            result = await self._select_performance_first(context)
        elif self.strategy == SelectionStrategy.QUALITY_FIRST:
            result = await self._select_quality_first(context)
        elif self.strategy == SelectionStrategy.MEMORY_AWARE:
            result = await self._select_memory_aware(context)
        elif self.strategy == SelectionStrategy.ADAPTIVE:
            result = await self._select_adaptive(context)
        else:  # BALANCED
            result = await self._select_balanced(context)
        
        result.selection_time = time.time() - start_time
        
        # Update tracking
        self.last_selection = result
        self.selection_history.append(result)
        
        if (self.last_selection and 
            self.last_selection.selected_model != result.selected_model):
            self.model_switch_count += 1
        
        self.logger.info(
            f"Selected {result.selected_model.value} model "
            f"(confidence: {result.confidence:.2f}, time: {result.selection_time:.3f}s) - {result.rationale}"
        )
        
        return result
    
    async def _select_performance_first(self, context: SelectionContext) -> SelectionResult:
        """Select model prioritizing processing speed"""
        # Calculate performance scores for available models
        scores = {}
        
        for model in context.available_models:
            if not self._check_memory_feasible(model, context.current_memory_usage):
                continue
                
            capabilities = self.model_capabilities[model]
            metrics = self.current_metrics[model]
            
            # Performance score (inverse of inference time)
            time_score = 1.0 / (metrics.avg_inference_time + 0.1)
            
            # Success rate bonus
            reliability_bonus = metrics.success_rate
            
            # Memory efficiency bonus
            memory_bonus = metrics.memory_efficiency * 0.5
            
            # Throughput bonus
            throughput_bonus = min(metrics.throughput_docs_hour / 100.0, 1.0) * 0.3
            
            scores[model] = time_score + reliability_bonus + memory_bonus + throughput_bonus
        
        if not scores:
            # Fallback to fastest available model
            selected = min(context.available_models, 
                         key=lambda m: self.model_capabilities[m].typical_inference_time)
            confidence = 0.5
            rationale = "Fallback to fastest available model"
        else:
            selected = max(scores, key=scores.get)
            confidence = min(scores[selected] / max(scores.values()) * 0.9, 1.0)
            rationale = f"Performance-first selection (score: {scores[selected]:.2f})"
        
        return self._create_selection_result(selected, confidence, rationale, context)
    
    async def _select_quality_first(self, context: SelectionContext) -> SelectionResult:
        """Select model prioritizing output quality"""
        quality_req = context.document_complexity.quality_requirement
        
        # Find models that can meet quality requirement
        suitable_models = []
        for model in context.available_models:
            if not self._check_memory_feasible(model, context.current_memory_usage):
                continue
                
            metrics = self.current_metrics[model]
            if metrics.avg_quality_score >= quality_req * 0.9:  # 10% tolerance
                suitable_models.append(model)
        
        if not suitable_models:
            # Fallback to highest quality available
            selected = max(context.available_models, 
                         key=lambda m: self.current_metrics[m].avg_quality_score)
            confidence = 0.6
            rationale = "Fallback to highest quality available model"
        else:
            # Select highest quality among suitable models
            selected = max(suitable_models, 
                         key=lambda m: self.current_metrics[m].avg_quality_score)
            confidence = min(self.current_metrics[selected].avg_quality_score / quality_req, 1.0)
            rationale = f"Quality-first selection (quality: {self.current_metrics[selected].avg_quality_score:.2f})"
        
        return self._create_selection_result(selected, confidence, rationale, context)
    
    async def _select_memory_aware(self, context: SelectionContext) -> SelectionResult:
        """Select model with memory constraints as primary consideration"""
        available_memory = 45.0 - context.current_memory_usage  # M3 Max 45GB for Python
        
        # Sort models by memory usage
        memory_sorted = sorted(context.available_models, 
                             key=lambda m: self.model_capabilities[m].max_memory_gb)
        
        # Find largest model that fits in available memory
        selected = None
        for model in reversed(memory_sorted):  # Largest first
            if self.model_capabilities[model].max_memory_gb <= available_memory * 0.9:
                selected = model
                break
        
        if selected is None:
            # Emergency fallback to smallest model
            selected = memory_sorted[0]
            confidence = 0.3
            rationale = "Emergency fallback - insufficient memory"
        else:
            memory_utilization = self.model_capabilities[selected].max_memory_gb / available_memory
            confidence = 0.8 if memory_utilization < 0.8 else 0.6
            rationale = f"Memory-aware selection ({memory_utilization:.1%} utilization)"
        
        return self._create_selection_result(selected, confidence, rationale, context)
    
    async def _select_adaptive(self, context: SelectionContext) -> SelectionResult:
        """Select model using adaptive learning from historical patterns"""
        complexity = context.document_complexity
        complexity_score = complexity.complexity_score()
        quality_req = complexity.quality_requirement
        
        # Check learned preferences for similar contexts
        learned_model = self._query_learned_preferences(complexity_score, quality_req)
        
        if learned_model and learned_model in context.available_models:
            # Use learned preference
            confidence = 0.85
            rationale = f"Adaptive learning match (complexity: {complexity_score:.2f}, quality: {quality_req:.2f})"
            selected = learned_model
        else:
            # Use prediction model or heuristic
            predicted_models = self._predict_model_performance(context)
            
            if predicted_models:
                selected = max(predicted_models, key=predicted_models.get)
                confidence = min(predicted_models[selected], 1.0)
                rationale = f"Predictive model selection (score: {predicted_models[selected]:.2f})"
            else:
                # Fallback to heuristic selection
                selected = self._heuristic_selection(context)
                confidence = 0.7
                rationale = "Heuristic fallback selection"
        
        # Add exploration factor for learning
        if np.random.random() < self.exploration_factor:
            # Occasionally try different models for learning
            alternative_models = [m for m in context.available_models if m != selected]
            if alternative_models:
                selected = np.random.choice(alternative_models)
                confidence *= 0.8  # Reduce confidence for exploration
                rationale += " (exploration variant)"
        
        return self._create_selection_result(selected, confidence, rationale, context)
    
    async def _select_balanced(self, context: SelectionContext) -> SelectionResult:
        """Select model balancing speed, quality, and resource usage"""
        complexity = context.document_complexity
        complexity_score = complexity.complexity_score()
        
        # Multi-criteria scoring
        scores = {}
        
        for model in context.available_models:
            if not self._check_memory_feasible(model, context.current_memory_usage):
                continue
            
            metrics = self.current_metrics[model]
            capabilities = self.model_capabilities[model]
            
            # Quality score (match to requirements)
            quality_match = 1.0 - abs(metrics.avg_quality_score - complexity.quality_requirement)
            quality_score = quality_match * 0.3
            
            # Performance score (inverse time, normalized)
            perf_score = (1.0 / (metrics.avg_inference_time + 0.1)) / 10.0 * 0.25
            
            # Complexity appropriateness
            if complexity_score < 0.3 and model == ModelSize.FAST:
                complexity_score_val = 0.25
            elif 0.3 <= complexity_score < 0.7 and model == ModelSize.BALANCED:
                complexity_score_val = 0.25
            elif complexity_score >= 0.7 and model == ModelSize.QUALITY:
                complexity_score_val = 0.25
            else:
                complexity_score_val = 0.1
            
            # Resource efficiency
            resource_score = metrics.memory_efficiency * 0.1
            
            # Reliability bonus
            reliability_score = metrics.success_rate * 0.1
            
            total_score = (quality_score + perf_score + complexity_score_val + 
                          resource_score + reliability_score)
            
            scores[model] = total_score
        
        if not scores:
            # Fallback based on complexity
            if complexity_score < 0.4:
                selected = ModelSize.FAST
            elif complexity_score < 0.75:
                selected = ModelSize.BALANCED  
            else:
                selected = ModelSize.QUALITY
            confidence = 0.6
            rationale = "Fallback complexity-based selection"
        else:
            selected = max(scores, key=scores.get)
            max_score = max(scores.values())
            confidence = min(scores[selected] / max_score * 0.9, 1.0)
            rationale = f"Balanced multi-criteria selection (score: {scores[selected]:.3f})"
        
        return self._create_selection_result(selected, confidence, rationale, context)
    
    def _query_learned_preferences(self, complexity: float, quality_req: float) -> Optional[ModelSize]:
        """Query learned model preferences for similar contexts"""
        # Find closest learned preference
        min_distance = float('inf')
        best_match = None
        
        for (learned_complexity, learned_quality), model in self.model_preferences.items():
            distance = np.sqrt((complexity - learned_complexity)**2 + 
                              (quality_req - learned_quality)**2)
            
            if distance < min_distance and distance < 0.2:  # Similarity threshold
                min_distance = distance
                best_match = model
        
        return best_match
    
    def _predict_model_performance(self, context: SelectionContext) -> Dict[ModelSize, float]:
        """Predict model performance scores using learned patterns"""
        predictions = {}
        complexity = context.document_complexity
        
        # Simple feature vector for prediction
        features = np.array([
            complexity.complexity_score(),
            complexity.quality_requirement,
            complexity.processing_urgency,
            complexity.content_length / 10000.0,  # Normalized
            complexity.parameter_count / 50.0,    # Normalized
            context.current_memory_usage / 45.0   # Memory pressure
        ])
        
        for model in context.available_models:
            if model in self.prediction_weights:
                weights = self.prediction_weights[model]
                bias = self.prediction_bias[model]
                
                # Simple linear prediction
                if len(weights) == len(features):
                    score = np.dot(weights, features) + bias
                    predictions[model] = max(0.0, min(1.0, score))
        
        return predictions
    
    def _heuristic_selection(self, context: SelectionContext) -> ModelSize:
        """Heuristic model selection based on complexity thresholds"""
        complexity_score = context.document_complexity.complexity_score()
        quality_req = context.document_complexity.quality_requirement
        
        # Simple threshold-based selection with quality adjustment
        if complexity_score < 0.3 and quality_req < 0.8:
            return ModelSize.FAST
        elif complexity_score < 0.7 and quality_req < 0.9:
            return ModelSize.BALANCED
        else:
            return ModelSize.QUALITY
    
    def _check_memory_feasible(self, model: ModelSize, current_usage: float) -> bool:
        """Check if model can fit in available memory"""
        required_memory = self.model_capabilities[model].max_memory_gb
        total_memory = 45.0  # Python ML allocation on M3 Max
        
        return (current_usage + required_memory) <= total_memory * 0.95
    
    def _create_selection_result(
        self, 
        selected: ModelSize, 
        confidence: float, 
        rationale: str, 
        context: SelectionContext
    ) -> SelectionResult:
        """Create detailed selection result"""
        # Generate fallback models
        fallback_models = [m for m in context.available_models if m != selected]
        fallback_models.sort(key=lambda m: (
            abs(self.current_metrics[m].avg_quality_score - 
                context.document_complexity.quality_requirement),
            self.model_capabilities[m].typical_inference_time
        ))
        
        # Get expected performance
        expected_performance = self.current_metrics[selected]
        
        return SelectionResult(
            selected_model=selected,
            confidence=confidence,
            rationale=rationale,
            expected_performance=expected_performance,
            fallback_models=fallback_models[:2],  # Top 2 fallbacks
            selection_time=0.0  # Will be set by caller
        )
    
    async def update_performance_feedback(
        self, 
        model: ModelSize, 
        actual_performance: Dict[str, float]
    ):
        """Update model performance metrics with actual feedback"""
        if model not in self.current_metrics:
            return
        
        metrics = self.current_metrics[model]
        alpha = self.learning_rate  # Learning rate for exponential moving average
        
        # Update metrics with feedback
        if 'inference_time' in actual_performance:
            metrics.avg_inference_time = (
                alpha * actual_performance['inference_time'] + 
                (1 - alpha) * metrics.avg_inference_time
            )
        
        if 'quality_score' in actual_performance:
            metrics.avg_quality_score = (
                alpha * actual_performance['quality_score'] + 
                (1 - alpha) * metrics.avg_quality_score
            )
        
        if 'success' in actual_performance:
            success = 1.0 if actual_performance['success'] else 0.0
            metrics.success_rate = (
                alpha * success + (1 - alpha) * metrics.success_rate
            )
        
        if 'memory_usage' in actual_performance:
            memory_eff = min(1.0, actual_performance['memory_usage'] / 
                           self.model_capabilities[model].max_memory_gb)
            metrics.memory_efficiency = (
                alpha * memory_eff + (1 - alpha) * metrics.memory_efficiency
            )
        
        metrics.sample_count += 1
        metrics.last_measurement = time.time()
        
        # Update throughput
        if metrics.avg_inference_time > 0:
            metrics.throughput_docs_hour = 3600.0 / metrics.avg_inference_time
        
        # Add to performance history
        self.performance_history[model].append({
            'timestamp': time.time(),
            'metrics': asdict(metrics),
            'feedback': actual_performance
        })
        
        self.logger.debug(f"Updated performance metrics for {model.value}: "
                         f"quality={metrics.avg_quality_score:.3f}, "
                         f"time={metrics.avg_inference_time:.2f}s")
    
    async def learn_from_selection(
        self, 
        context: SelectionContext, 
        result: SelectionResult, 
        outcome: Dict[str, Any]
    ):
        """Learn from selection outcome to improve future decisions"""
        complexity_key = (
            round(context.document_complexity.complexity_score(), 1),
            round(context.document_complexity.quality_requirement, 1)
        )
        
        # Update learned preferences if outcome was successful
        if outcome.get('success', False) and outcome.get('user_satisfaction', 0.5) > 0.7:
            self.model_preferences[complexity_key] = result.selected_model
            
            # Update usage patterns
            pattern_key = f"{result.selected_model.value}_{complexity_key[0]}_{complexity_key[1]}"
            if pattern_key not in self.usage_patterns:
                self.usage_patterns[pattern_key] = {
                    'count': 0,
                    'success_rate': 0.0,
                    'avg_satisfaction': 0.0
                }
            
            pattern = self.usage_patterns[pattern_key]
            pattern['count'] += 1
            
            # Update success rate
            success = 1.0 if outcome.get('success', False) else 0.0
            pattern['success_rate'] = (
                (pattern['success_rate'] * (pattern['count'] - 1) + success) / 
                pattern['count']
            )
            
            # Update satisfaction
            satisfaction = outcome.get('user_satisfaction', 0.5)
            pattern['avg_satisfaction'] = (
                (pattern['avg_satisfaction'] * (pattern['count'] - 1) + satisfaction) / 
                pattern['count']
            )
        
        # Update prediction model weights (simple gradient update)
        if result.selected_model in self.prediction_weights:
            self._update_prediction_weights(context, result, outcome)
        
        self.logger.debug(f"Learning updated for {result.selected_model.value} selection")
    
    def _update_prediction_weights(
        self, 
        context: SelectionContext, 
        result: SelectionResult, 
        outcome: Dict[str, Any]
    ):
        """Update prediction model weights based on outcome"""
        model = result.selected_model
        
        if model not in self.prediction_weights:
            # Initialize weights if not present
            feature_count = 6  # Number of features in prediction
            self.prediction_weights[model] = np.random.normal(0, 0.1, feature_count)
            self.prediction_bias[model] = 0.0
        
        # Calculate prediction error
        actual_score = outcome.get('overall_score', result.confidence)
        predicted_score = result.confidence
        error = actual_score - predicted_score
        
        # Simple gradient update
        complexity = context.document_complexity
        features = np.array([
            complexity.complexity_score(),
            complexity.quality_requirement,
            complexity.processing_urgency,
            complexity.content_length / 10000.0,
            complexity.parameter_count / 50.0,
            context.current_memory_usage / 45.0
        ])
        
        learning_rate = 0.01
        self.prediction_weights[model] += learning_rate * error * features
        self.prediction_bias[model] += learning_rate * error
        
        # Clip weights to prevent instability
        self.prediction_weights[model] = np.clip(self.prediction_weights[model], -1.0, 1.0)
        self.prediction_bias[model] = np.clip(self.prediction_bias[model], -0.5, 0.5)
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive selection statistics"""
        stats = {
            'total_selections': self.total_selections,
            'model_switch_count': self.model_switch_count,
            'switch_rate': self.model_switch_count / max(1, self.total_selections),
            'model_usage_distribution': {},
            'average_confidence': 0.0,
            'learned_preferences_count': len(self.model_preferences),
            'usage_patterns_count': len(self.usage_patterns),
            'current_strategy': self.strategy.value
        }
        
        # Calculate model usage distribution
        model_counts = defaultdict(int)
        total_confidence = 0.0
        
        for result in self.selection_history:
            model_counts[result.selected_model.value] += 1
            total_confidence += result.confidence
        
        if self.selection_history:
            stats['average_confidence'] = total_confidence / len(self.selection_history)
        
        # Normalize usage distribution
        total_selections = sum(model_counts.values())
        if total_selections > 0:
            for model, count in model_counts.items():
                stats['model_usage_distribution'][model] = count / total_selections
        
        # Add performance metrics summary
        stats['model_performance'] = {}
        for model, metrics in self.current_metrics.items():
            stats['model_performance'][model.value] = {
                'avg_quality_score': metrics.avg_quality_score,
                'avg_inference_time': metrics.avg_inference_time,
                'success_rate': metrics.success_rate,
                'throughput_docs_hour': metrics.throughput_docs_hour,
                'sample_count': metrics.sample_count
            }
        
        return stats
    
    async def save_state(self, path: Optional[Path] = None):
        """Save current state and learned patterns"""
        save_path = path or Path("model_selection_history.json")
        
        # Prepare data for serialization
        data = {
            'usage_patterns': self.usage_patterns,
            'model_preferences': {
                f"{k[0]}, {k[1]}": v.value for k, v in self.model_preferences.items()
            },
            'performance_metrics': {
                model.value: asdict(metrics) for model, metrics in self.current_metrics.items()
            },
            'selection_statistics': self.get_selection_statistics(),
            'last_updated': time.time()
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Model selector state saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model selector state: {e}")
    
    async def optimize_selection_strategy(self):
        """Analyze performance and optimize selection strategy"""
        if len(self.selection_history) < 10:
            return  # Not enough data
        
        # Analyze recent performance by strategy
        recent_selections = list(self.selection_history)[-50:]  # Last 50 selections
        
        strategy_performance = {}
        for strategy in SelectionStrategy:
            # Simulate what each strategy would have selected
            strategy_confidence = []
            for result in recent_selections:
                # This is simplified - in practice would need to replay decisions
                strategy_confidence.append(result.confidence)
            
            if strategy_confidence:
                strategy_performance[strategy] = np.mean(strategy_confidence)
        
        # Find best performing strategy
        if strategy_performance:
            best_strategy = max(strategy_performance, key=strategy_performance.get)
            
            if best_strategy != self.strategy and strategy_performance[best_strategy] > strategy_performance.get(self.strategy, 0):
                self.logger.info(f"Optimizing strategy from {self.strategy.value} to {best_strategy.value} "
                               f"(performance improvement: {strategy_performance[best_strategy] - strategy_performance.get(self.strategy, 0):.3f})")
                self.strategy = best_strategy
    
    @asynccontextmanager
    async def selection_session(self):
        """Context manager for selection session with cleanup"""
        try:
            yield self
        finally:
            await self.save_state()
            await self.optimize_selection_strategy()


# Factory function for easy instantiation
def create_model_selector(
    strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
    config_path: Optional[Path] = None
) -> AdaptiveModelSelector:
    """Create and initialize model selector"""
    return AdaptiveModelSelector(config_path=config_path, strategy=strategy)


# Example usage and testing
if __name__ == "__main__":
    async def test_model_selector():
        """Test the model selector"""
        logging.basicConfig(level=logging.INFO)
        
        selector = create_model_selector(SelectionStrategy.ADAPTIVE)
        
        # Create test context
        test_complexity = DocumentComplexity(
            content_length=2500,
            parameter_count=8,
            counter_count=3,
            technical_term_count=15,
            structural_depth=3,
            readability_score=0.7,
            domain_specificity=0.8,
            quality_requirement=0.8
        )
        
        test_context = SelectionContext(
            document_complexity=test_complexity,
            current_memory_usage=5.0,
            available_models=list(ModelSize),
            recent_performance={}
        )
        
        # Test selection
        async with selector.selection_session():
            result = await selector.select_optimal_model(test_context)
            
            print(f"Selected: {result.selected_model.value}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Rationale: {result.rationale}")
            print(f"Fallbacks: {[m.value for m in result.fallback_models]}")
            
            # Simulate feedback
            feedback = {
                'success': True,
                'inference_time': 3.5,
                'quality_score': 0.85,
                'user_satisfaction': 0.9
            }
            
            await selector.update_performance_feedback(result.selected_model, feedback)
            await selector.learn_from_selection(test_context, result, feedback)
            
            # Get statistics
            stats = selector.get_selection_statistics()
            print("\nSelection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    asyncio.run(test_model_selector())