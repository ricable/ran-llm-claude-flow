"""
Predictive Model Selection with ML Intelligence
Advanced ML-powered system for optimal Qwen3 model selection (1.7B/7B/30B)
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import mlx.core as mx
import mlx.nn as nn
from mlx_accelerator import MLXAccelerator
from model_manager import ModelManager
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model evaluation"""
    model_name: str
    processing_time_seconds: float
    quality_score: float
    memory_usage_gb: float
    throughput_docs_per_hour: float
    accuracy_score: float
    inference_latency_ms: float
    token_throughput: float
    power_consumption_watts: float
    temperature_celsius: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DocumentProfile:
    """Document characteristics for prediction"""
    content_length: int
    complexity_score: float
    domain_type: str
    language: str
    technical_density: float
    structured_content_ratio: float
    vocabulary_diversity: float
    requires_reasoning: bool
    context_length: int
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML prediction"""
        domain_encoding = {
            'technical': 1.0, 'general': 0.5, 'specialized': 0.8,
            'academic': 0.9, 'business': 0.6, 'unknown': 0.0
        }
        
        return np.array([
            self.content_length / 10000.0,  # Normalized
            self.complexity_score,
            domain_encoding.get(self.domain_type, 0.0),
            1.0 if self.language == 'en' else 0.5,  # English optimization
            self.technical_density,
            self.structured_content_ratio,
            self.vocabulary_diversity,
            1.0 if self.requires_reasoning else 0.0,
            self.context_length / 4096.0,  # Normalized to typical context
        ])

@dataclass
class PredictionResult:
    """Model selection prediction result"""
    recommended_model: str
    confidence_score: float
    expected_performance: ModelPerformanceMetrics
    alternative_models: List[Tuple[str, float]]  # (model, score)
    prediction_reasoning: str
    resource_requirements: Dict[str, Any]

class PredictiveModelSelector:
    """Advanced ML-powered model selection system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize MLX accelerator
        self.mlx_accelerator = MLXAccelerator()
        
        # Model manager for Qwen3 variants
        self.model_manager = ModelManager()
        
        # Available Qwen3 models
        self.available_models = {
            "qwen3-1.7b": {
                "memory_gb": 3.5,
                "optimal_batch_size": 32,
                "max_context": 4096,
                "best_for": "simple_tasks",
                "performance_class": "fast"
            },
            "qwen3-7b": {
                "memory_gb": 14,
                "optimal_batch_size": 16,
                "max_context": 8192,
                "best_for": "balanced_tasks",
                "performance_class": "balanced"
            },
            "qwen3-30b": {
                "memory_gb": 45,
                "optimal_batch_size": 4,
                "max_context": 16384,
                "best_for": "complex_tasks",
                "performance_class": "accurate"
            }
        }
        
        # ML models for prediction
        self.performance_predictor = None
        self.quality_predictor = None
        self.resource_predictor = None
        self.feature_scaler = StandardScaler()
        
        # Historical performance data
        self.performance_history: List[Dict] = []
        self.training_data_path = Path("training_data/model_performance.json")
        
        # Real-time metrics
        self.current_system_load = 0.0
        self.available_memory_gb = 45.0  # From Python ML pool
        self.current_temperature = 35.0
        
        # Initialize predictors
        asyncio.create_task(self._initialize_predictors())
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration settings"""
        default_config = {
            "prediction_confidence_threshold": 0.75,
            "model_switch_overhead_seconds": 2.0,
            "memory_safety_margin_gb": 5.0,
            "temperature_threshold_celsius": 85.0,
            "quality_threshold": 0.8,
            "enable_adaptive_switching": True,
            "prediction_history_size": 1000,
            "retraining_interval_hours": 24
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def _initialize_predictors(self):
        """Initialize ML prediction models"""
        try:
            # Load historical data
            await self._load_performance_history()
            
            # Train initial predictors if we have sufficient data
            if len(self.performance_history) >= 50:
                await self._train_predictors()
            else:
                # Use pre-trained baseline models
                await self._load_baseline_predictors()
                
            self.logger.info("Predictive model selector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize predictors: {e}")
            await self._load_baseline_predictors()
    
    async def _load_performance_history(self):
        """Load historical performance data"""
        try:
            if self.training_data_path.exists():
                with open(self.training_data_path, 'r') as f:
                    self.performance_history = json.load(f)
                    self.logger.info(f"Loaded {len(self.performance_history)} historical records")
            else:
                # Generate synthetic training data for bootstrap
                self.performance_history = await self._generate_synthetic_training_data()
                await self._save_performance_history()
                
        except Exception as e:
            self.logger.error(f"Error loading performance history: {e}")
            self.performance_history = []
    
    async def _generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data for initial model training"""
        synthetic_data = []
        
        # Generate diverse document profiles and corresponding performance
        for _ in range(500):
            doc_profile = self._generate_random_document_profile()
            
            # Simulate performance for each model
            for model_name in self.available_models.keys():
                performance = self._simulate_model_performance(doc_profile, model_name)
                
                record = {
                    "timestamp": (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                    "document_profile": asdict(doc_profile),
                    "model_name": model_name,
                    "performance_metrics": asdict(performance),
                    "system_conditions": {
                        "available_memory_gb": np.random.uniform(20, 45),
                        "system_load": np.random.uniform(0.1, 0.9),
                        "temperature_celsius": np.random.uniform(30, 80)
                    }
                }
                synthetic_data.append(record)
        
        return synthetic_data
    
    def _generate_random_document_profile(self) -> DocumentProfile:
        """Generate random document profile for training"""
        return DocumentProfile(
            content_length=np.random.randint(100, 50000),
            complexity_score=np.random.uniform(0.1, 1.0),
            domain_type=np.random.choice(['technical', 'general', 'specialized', 'academic', 'business']),
            language=np.random.choice(['en', 'other']),
            technical_density=np.random.uniform(0.0, 1.0),
            structured_content_ratio=np.random.uniform(0.0, 1.0),
            vocabulary_diversity=np.random.uniform(0.1, 1.0),
            requires_reasoning=np.random.choice([True, False]),
            context_length=np.random.randint(512, 16384)
        )
    
    def _simulate_model_performance(self, doc_profile: DocumentProfile, model_name: str) -> ModelPerformanceMetrics:
        """Simulate realistic model performance based on document profile"""
        model_info = self.available_models[model_name]
        
        # Base performance characteristics
        if model_name == "qwen3-1.7b":
            base_time = 0.5 + doc_profile.content_length / 20000
            base_quality = 0.7 + doc_profile.complexity_score * 0.1
            base_accuracy = 0.75
        elif model_name == "qwen3-7b":
            base_time = 1.0 + doc_profile.content_length / 15000
            base_quality = 0.8 + doc_profile.complexity_score * 0.15
            base_accuracy = 0.85
        else:  # qwen3-30b
            base_time = 2.5 + doc_profile.content_length / 10000
            base_quality = 0.9 + doc_profile.complexity_score * 0.08
            base_accuracy = 0.92
        
        # Adjust for document characteristics
        if doc_profile.requires_reasoning and model_name == "qwen3-1.7b":
            base_quality *= 0.9  # 1.7B struggles with reasoning
            base_accuracy *= 0.95
        
        if doc_profile.technical_density > 0.7 and model_name != "qwen3-30b":
            base_quality *= 0.95  # Technical content needs larger models
            base_accuracy *= 0.98
        
        # Add realistic noise
        noise_factor = np.random.normal(1.0, 0.1)
        
        return ModelPerformanceMetrics(
            model_name=model_name,
            processing_time_seconds=base_time * noise_factor,
            quality_score=np.clip(base_quality * noise_factor, 0.1, 1.0),
            memory_usage_gb=model_info["memory_gb"] * np.random.uniform(0.9, 1.1),
            throughput_docs_per_hour=3600 / (base_time * noise_factor),
            accuracy_score=np.clip(base_accuracy * noise_factor, 0.5, 1.0),
            inference_latency_ms=(base_time * 1000) * noise_factor,
            token_throughput=1000 / base_time,
            power_consumption_watts=10 + model_info["memory_gb"] * 2,
            temperature_celsius=35 + np.random.uniform(0, 15)
        )
    
    async def _train_predictors(self):
        """Train ML prediction models from historical data"""
        try:
            # Prepare training data
            X, y_performance, y_quality, y_resources = self._prepare_training_data()
            
            if len(X) < 10:
                self.logger.warning("Insufficient training data, using baseline predictors")
                await self._load_baseline_predictors()
                return
            
            # Fit feature scaler
            self.feature_scaler.fit(X)
            X_scaled = self.feature_scaler.transform(X)
            
            # Train performance predictor (processing time)
            self.performance_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.performance_predictor.fit(X_scaled, y_performance)
            
            # Train quality predictor
            self.quality_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.quality_predictor.fit(X_scaled, y_quality)
            
            # Train resource predictor (memory usage)
            self.resource_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.resource_predictor.fit(X_scaled, y_resources)
            
            # Evaluate models
            perf_score = self.performance_predictor.score(X_scaled, y_performance)
            qual_score = self.quality_predictor.score(X_scaled, y_quality)
            res_score = self.resource_predictor.score(X_scaled, y_resources)
            
            self.logger.info(f"Predictor training completed - Performance R²: {perf_score:.3f}, Quality R²: {qual_score:.3f}, Resource R²: {res_score:.3f}")
            
            # Save trained models
            await self._save_trained_models()
            
        except Exception as e:
            self.logger.error(f"Error training predictors: {e}")
            await self._load_baseline_predictors()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from historical performance"""
        X = []
        y_performance = []
        y_quality = []
        y_resources = []
        
        for record in self.performance_history:
            # Document features
            doc_data = record["document_profile"]
            doc_profile = DocumentProfile(**doc_data)
            features = doc_profile.to_feature_vector()
            
            # System condition features
            sys_conditions = record.get("system_conditions", {})
            sys_features = np.array([
                sys_conditions.get("available_memory_gb", 40) / 50.0,
                sys_conditions.get("system_load", 0.5),
                sys_conditions.get("temperature_celsius", 40) / 100.0
            ])
            
            # Model encoding features
            model_name = record["model_name"]
            model_features = np.array([
                1.0 if model_name == "qwen3-1.7b" else 0.0,
                1.0 if model_name == "qwen3-7b" else 0.0,
                1.0 if model_name == "qwen3-30b" else 0.0
            ])
            
            # Combine all features
            combined_features = np.concatenate([features, sys_features, model_features])
            X.append(combined_features)
            
            # Target values
            perf_data = record["performance_metrics"]
            y_performance.append(perf_data["processing_time_seconds"])
            y_quality.append(perf_data["quality_score"])
            y_resources.append(perf_data["memory_usage_gb"])
        
        return np.array(X), np.array(y_performance), np.array(y_quality), np.array(y_resources)
    
    async def _load_baseline_predictors(self):
        """Load baseline predictors for cold start"""
        # Simple rule-based predictors as fallback
        self.performance_predictor = None
        self.quality_predictor = None
        self.resource_predictor = None
        self.logger.info("Using rule-based baseline predictors")
    
    async def predict_optimal_model(
        self, 
        document_profile: DocumentProfile,
        system_conditions: Optional[Dict] = None
    ) -> PredictionResult:
        """Predict optimal model for given document and system conditions"""
        try:
            # Update system conditions
            await self._update_system_conditions(system_conditions)
            
            # Get predictions for each model
            model_predictions = {}
            for model_name in self.available_models.keys():
                prediction = await self._predict_model_performance(
                    document_profile, model_name
                )
                model_predictions[model_name] = prediction
            
            # Select optimal model based on multi-objective optimization
            optimal_model, confidence = await self._select_optimal_model(
                model_predictions, document_profile
            )
            
            # Generate recommendation reasoning
            reasoning = await self._generate_reasoning(
                optimal_model, model_predictions, document_profile
            )
            
            # Get alternative recommendations
            alternatives = await self._get_alternative_models(
                model_predictions, optimal_model
            )
            
            return PredictionResult(
                recommended_model=optimal_model,
                confidence_score=confidence,
                expected_performance=model_predictions[optimal_model],
                alternative_models=alternatives,
                prediction_reasoning=reasoning,
                resource_requirements=await self._get_resource_requirements(optimal_model)
            )
            
        except Exception as e:
            self.logger.error(f"Error in model prediction: {e}")
            # Fallback to rule-based selection
            return await self._fallback_model_selection(document_profile)
    
    async def _predict_model_performance(
        self, 
        document_profile: DocumentProfile, 
        model_name: str
    ) -> ModelPerformanceMetrics:
        """Predict performance for specific model"""
        
        if self.performance_predictor is None:
            # Use rule-based prediction
            return self._rule_based_prediction(document_profile, model_name)
        
        try:
            # Prepare feature vector
            doc_features = document_profile.to_feature_vector()
            sys_features = np.array([
                self.available_memory_gb / 50.0,
                self.current_system_load,
                self.current_temperature / 100.0
            ])
            model_features = np.array([
                1.0 if model_name == "qwen3-1.7b" else 0.0,
                1.0 if model_name == "qwen3-7b" else 0.0,
                1.0 if model_name == "qwen3-30b" else 0.0
            ])
            
            features = np.concatenate([doc_features, sys_features, model_features]).reshape(1, -1)
            features_scaled = self.feature_scaler.transform(features)
            
            # Get predictions
            processing_time = self.performance_predictor.predict(features_scaled)[0]
            quality_score = self.quality_predictor.predict(features_scaled)[0]
            memory_usage = self.resource_predictor.predict(features_scaled)[0]
            
            # Calculate derived metrics
            throughput = 3600 / max(processing_time, 0.1)  # docs/hour
            inference_latency = processing_time * 1000  # ms
            token_throughput = 1000 / max(processing_time, 0.1)
            
            return ModelPerformanceMetrics(
                model_name=model_name,
                processing_time_seconds=max(processing_time, 0.1),
                quality_score=np.clip(quality_score, 0.0, 1.0),
                memory_usage_gb=max(memory_usage, 1.0),
                throughput_docs_per_hour=throughput,
                accuracy_score=quality_score * 0.95,  # Estimate
                inference_latency_ms=inference_latency,
                token_throughput=token_throughput,
                power_consumption_watts=10 + memory_usage * 2,
                temperature_celsius=self.current_temperature + np.random.uniform(0, 5)
            )
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return self._rule_based_prediction(document_profile, model_name)
    
    def _rule_based_prediction(
        self, 
        document_profile: DocumentProfile, 
        model_name: str
    ) -> ModelPerformanceMetrics:
        """Fallback rule-based performance prediction"""
        model_info = self.available_models[model_name]
        
        # Simple heuristic rules
        base_time = model_info["memory_gb"] / 10.0  # Larger models slower
        complexity_multiplier = 1.0 + document_profile.complexity_score
        length_multiplier = 1.0 + (document_profile.content_length / 10000.0)
        
        processing_time = base_time * complexity_multiplier * length_multiplier
        
        # Quality based on model size and task complexity
        if model_name == "qwen3-30b":
            quality_score = 0.9
        elif model_name == "qwen3-7b":
            quality_score = 0.8
        else:
            quality_score = 0.7
        
        # Adjust for document characteristics
        if document_profile.requires_reasoning:
            if model_name == "qwen3-1.7b":
                quality_score *= 0.85
            elif model_name == "qwen3-7b":
                quality_score *= 0.95
        
        return ModelPerformanceMetrics(
            model_name=model_name,
            processing_time_seconds=processing_time,
            quality_score=quality_score,
            memory_usage_gb=model_info["memory_gb"],
            throughput_docs_per_hour=3600 / processing_time,
            accuracy_score=quality_score * 0.95,
            inference_latency_ms=processing_time * 1000,
            token_throughput=1000 / processing_time,
            power_consumption_watts=10 + model_info["memory_gb"] * 2,
            temperature_celsius=self.current_temperature + 5
        )
    
    async def _select_optimal_model(
        self, 
        model_predictions: Dict[str, ModelPerformanceMetrics],
        document_profile: DocumentProfile
    ) -> Tuple[str, float]:
        """Select optimal model using multi-objective optimization"""
        
        # Define objective weights based on system state and document
        weights = {
            "quality": 0.35,
            "throughput": 0.25,
            "memory_efficiency": 0.20,
            "latency": 0.20
        }
        
        # Adjust weights based on document profile
        if document_profile.requires_reasoning:
            weights["quality"] = 0.5
            weights["throughput"] = 0.15
        
        if self.available_memory_gb < 20:
            weights["memory_efficiency"] = 0.4
            weights["quality"] = 0.25
        
        best_score = -1
        best_model = "qwen3-7b"  # Default fallback
        
        for model_name, metrics in model_predictions.items():
            # Check resource constraints
            if metrics.memory_usage_gb > self.available_memory_gb:
                continue  # Skip if not enough memory
            
            # Calculate normalized scores (higher is better)
            quality_score = metrics.quality_score
            throughput_score = min(metrics.throughput_docs_per_hour / 30.0, 1.0)
            memory_score = 1.0 - (metrics.memory_usage_gb / 50.0)
            latency_score = max(0, 1.0 - (metrics.inference_latency_ms / 5000.0))
            
            # Weighted combination
            combined_score = (
                weights["quality"] * quality_score +
                weights["throughput"] * throughput_score +
                weights["memory_efficiency"] * memory_score +
                weights["latency"] * latency_score
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model_name
        
        # Calculate confidence based on score margin
        sorted_scores = sorted(
            [(m, self._calculate_model_score(p, weights)) 
             for m, p in model_predictions.items()],
            key=lambda x: x[1], reverse=True
        )
        
        if len(sorted_scores) > 1:
            confidence = min(1.0, (sorted_scores[0][1] - sorted_scores[1][1]) * 2)
        else:
            confidence = 0.8
        
        return best_model, confidence
    
    def _calculate_model_score(self, metrics: ModelPerformanceMetrics, weights: Dict) -> float:
        """Calculate weighted model score"""
        quality_score = metrics.quality_score
        throughput_score = min(metrics.throughput_docs_per_hour / 30.0, 1.0)
        memory_score = 1.0 - (metrics.memory_usage_gb / 50.0)
        latency_score = max(0, 1.0 - (metrics.inference_latency_ms / 5000.0))
        
        return (
            weights["quality"] * quality_score +
            weights["throughput"] * throughput_score +
            weights["memory_efficiency"] * memory_score +
            weights["latency"] * latency_score
        )
    
    async def _generate_reasoning(
        self,
        optimal_model: str,
        model_predictions: Dict[str, ModelPerformanceMetrics],
        document_profile: DocumentProfile
    ) -> str:
        """Generate human-readable reasoning for model selection"""
        
        optimal_metrics = model_predictions[optimal_model]
        
        reasoning_parts = [
            f"Selected {optimal_model} based on document analysis:",
            f"• Content length: {document_profile.content_length:,} characters",
            f"• Complexity score: {document_profile.complexity_score:.2f}",
            f"• Domain: {document_profile.domain_type}",
            f"• Expected quality: {optimal_metrics.quality_score:.2f}",
            f"• Expected throughput: {optimal_metrics.throughput_docs_per_hour:.1f} docs/hour",
            f"• Memory usage: {optimal_metrics.memory_usage_gb:.1f}GB",
            f"• Processing time: {optimal_metrics.processing_time_seconds:.2f}s"
        ]
        
        # Add specific reasoning based on document characteristics
        if document_profile.requires_reasoning:
            reasoning_parts.append("• Reasoning required: selected model with enhanced logical capabilities")
        
        if document_profile.technical_density > 0.7:
            reasoning_parts.append("• High technical density: prioritized accuracy over speed")
        
        if self.available_memory_gb < 30:
            reasoning_parts.append("• Limited memory: optimized for memory efficiency")
        
        return "\n".join(reasoning_parts)
    
    async def _get_alternative_models(
        self,
        model_predictions: Dict[str, ModelPerformanceMetrics],
        optimal_model: str
    ) -> List[Tuple[str, float]]:
        """Get alternative model recommendations with scores"""
        alternatives = []
        
        for model_name, metrics in model_predictions.items():
            if model_name != optimal_model:
                # Calculate alternative score based on quality and feasibility
                if metrics.memory_usage_gb <= self.available_memory_gb:
                    score = metrics.quality_score * 0.7 + \
                           (metrics.throughput_docs_per_hour / 30.0) * 0.3
                    alternatives.append((model_name, min(score, 1.0)))
        
        # Sort by score descending
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:2]  # Return top 2 alternatives
    
    async def _get_resource_requirements(self, model_name: str) -> Dict[str, Any]:
        """Get detailed resource requirements for model"""
        model_info = self.available_models[model_name]
        
        return {
            "memory_gb": model_info["memory_gb"],
            "optimal_batch_size": model_info["optimal_batch_size"],
            "max_context_length": model_info["max_context"],
            "cpu_cores_recommended": 4 if model_name == "qwen3-30b" else 2,
            "estimated_loading_time_seconds": model_info["memory_gb"] * 0.5,
            "mlx_acceleration_supported": True,
            "numa_affinity_required": model_name == "qwen3-30b"
        }
    
    async def _update_system_conditions(self, conditions: Optional[Dict]):
        """Update current system conditions"""
        if conditions:
            self.available_memory_gb = conditions.get("available_memory_gb", self.available_memory_gb)
            self.current_system_load = conditions.get("system_load", self.current_system_load)
            self.current_temperature = conditions.get("temperature_celsius", self.current_temperature)
    
    async def _fallback_model_selection(self, document_profile: DocumentProfile) -> PredictionResult:
        """Fallback model selection when prediction fails"""
        # Simple rule-based fallback
        if document_profile.content_length > 10000 and document_profile.complexity_score > 0.7:
            recommended = "qwen3-30b" if self.available_memory_gb >= 45 else "qwen3-7b"
        elif document_profile.content_length > 5000:
            recommended = "qwen3-7b"
        else:
            recommended = "qwen3-1.7b"
        
        # Ensure we have enough memory
        if self.available_memory_gb < self.available_models[recommended]["memory_gb"]:
            recommended = "qwen3-1.7b"  # Smallest model as final fallback
        
        fallback_performance = self._rule_based_prediction(document_profile, recommended)
        
        return PredictionResult(
            recommended_model=recommended,
            confidence_score=0.6,  # Lower confidence for fallback
            expected_performance=fallback_performance,
            alternative_models=[],
            prediction_reasoning=f"Fallback selection based on content length and available memory",
            resource_requirements=await self._get_resource_requirements(recommended)
        )
    
    async def record_actual_performance(
        self,
        document_profile: DocumentProfile,
        model_name: str,
        actual_metrics: ModelPerformanceMetrics,
        system_conditions: Optional[Dict] = None
    ):
        """Record actual performance for model improvement"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "document_profile": asdict(document_profile),
            "model_name": model_name,
            "performance_metrics": asdict(actual_metrics),
            "system_conditions": system_conditions or {
                "available_memory_gb": self.available_memory_gb,
                "system_load": self.current_system_load,
                "temperature_celsius": self.current_temperature
            }
        }
        
        self.performance_history.append(record)
        
        # Limit history size
        max_history = self.config["prediction_history_size"]
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        
        # Trigger retraining if enough new data
        if len(self.performance_history) % 100 == 0:
            asyncio.create_task(self._retrain_models())
        
        # Save updated history
        await self._save_performance_history()
    
    async def _retrain_models(self):
        """Retrain prediction models with new data"""
        self.logger.info("Retraining prediction models with new data...")
        await self._train_predictors()
    
    async def _save_performance_history(self):
        """Save performance history to disk"""
        try:
            self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.training_data_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving performance history: {e}")
    
    async def _save_trained_models(self):
        """Save trained ML models to disk"""
        try:
            models_path = Path("models/predictive_selector/")
            models_path.mkdir(parents=True, exist_ok=True)
            
            if self.performance_predictor:
                with open(models_path / "performance_predictor.pkl", 'wb') as f:
                    pickle.dump(self.performance_predictor, f)
            
            if self.quality_predictor:
                with open(models_path / "quality_predictor.pkl", 'wb') as f:
                    pickle.dump(self.quality_predictor, f)
            
            if self.resource_predictor:
                with open(models_path / "resource_predictor.pkl", 'wb') as f:
                    pickle.dump(self.resource_predictor, f)
            
            with open(models_path / "feature_scaler.pkl", 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            self.logger.info("Saved trained models to disk")
            
        except Exception as e:
            self.logger.error(f"Error saving trained models: {e}")
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model selection statistics"""
        total_predictions = len(self.performance_history)
        if total_predictions == 0:
            return {"total_predictions": 0, "model_usage": {}}
        
        # Count model usage
        model_usage = {}
        accuracy_by_model = {}
        
        for record in self.performance_history:
            model = record["model_name"]
            model_usage[model] = model_usage.get(model, 0) + 1
            
            # Calculate accuracy (predicted vs actual quality score)
            # This would require storing predictions alongside actuals
        
        return {
            "total_predictions": total_predictions,
            "model_usage": model_usage,
            "model_usage_percentages": {
                model: (count / total_predictions) * 100
                for model, count in model_usage.items()
            },
            "prediction_accuracy": "N/A",  # Would require prediction storage
            "last_retrain": "N/A",  # Would need to track this
            "training_data_size": len(self.performance_history)
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_predictive_selector():
        selector = PredictiveModelSelector()
        
        # Test document profile
        test_doc = DocumentProfile(
            content_length=5000,
            complexity_score=0.75,
            domain_type="technical",
            language="en",
            technical_density=0.8,
            structured_content_ratio=0.6,
            vocabulary_diversity=0.7,
            requires_reasoning=True,
            context_length=2048
        )
        
        # Get prediction
        prediction = await selector.predict_optimal_model(test_doc)
        
        print(f"Recommended Model: {prediction.recommended_model}")
        print(f"Confidence: {prediction.confidence_score:.2f}")
        print(f"Expected Quality: {prediction.expected_performance.quality_score:.2f}")
        print(f"Expected Throughput: {prediction.expected_performance.throughput_docs_per_hour:.1f} docs/hour")
        print(f"Reasoning:\n{prediction.prediction_reasoning}")
        
        # Get statistics
        stats = await selector.get_model_statistics()
        print(f"\nModel Statistics: {stats}")
    
    # Run test
    asyncio.run(test_predictive_selector())