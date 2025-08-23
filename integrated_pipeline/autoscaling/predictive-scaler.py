#!/usr/bin/env python3
"""
Predictive Auto-Scaler with ML Workload Pattern Analysis
Implements intelligent scaling based on historical patterns and real-time metrics
"""

import asyncio
import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkloadMetrics:
    """Represents workload metrics at a point in time"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    queue_depth: int
    error_rate: float
    active_connections: int

@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    timestamp: float
    action: str  # 'scale_up', 'scale_down', 'no_action'
    target_instances: int
    current_instances: int
    confidence: float
    reasoning: str
    cost_impact: float

@dataclass
class ScalingConfig:
    """Scaling configuration parameters"""
    min_instances: int = 1
    max_instances: int = 50
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 40.0
    cooldown_period: int = 300  # seconds
    prediction_window: int = 900  # 15 minutes
    ml_prediction_weight: float = 0.7
    metrics_window: int = 60  # seconds

class MLWorkloadPredictor:
    """ML model for predicting future workload patterns"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("../intelligence/scaling-ml-model.pkl")
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = [
            'hour_of_day', 'day_of_week', 'cpu_usage', 'memory_usage',
            'request_rate', 'response_time', 'queue_depth', 'error_rate',
            'active_connections', 'trend_cpu', 'trend_memory', 'trend_requests'
        ]
        
    def extract_features(self, metrics_history: List[WorkloadMetrics]) -> np.ndarray:
        """Extract features from metrics history for ML prediction"""
        if len(metrics_history) < 3:
            return np.array([])
            
        latest = metrics_history[-1]
        dt = datetime.fromtimestamp(latest.timestamp)
        
        # Time-based features
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        
        # Current metrics
        current_metrics = [
            latest.cpu_usage, latest.memory_usage, latest.request_rate,
            latest.response_time, latest.queue_depth, latest.error_rate,
            latest.active_connections
        ]
        
        # Trend features (last 3 data points)
        recent_metrics = metrics_history[-3:]
        cpu_trend = np.mean([m.cpu_usage for m in recent_metrics[-2:]]) - recent_metrics[0].cpu_usage
        memory_trend = np.mean([m.memory_usage for m in recent_metrics[-2:]]) - recent_metrics[0].memory_usage
        request_trend = np.mean([m.request_rate for m in recent_metrics[-2:]]) - recent_metrics[0].request_rate
        
        features = [hour_of_day, day_of_week] + current_metrics + [cpu_trend, memory_trend, request_trend]
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Tuple[List[WorkloadMetrics], int]]):
        """Train the ML model with historical data"""
        logger.info(f"Training ML model with {len(training_data)} data points")
        
        X, y = [], []
        for metrics_history, target_instances in training_data:
            features = self.extract_features(metrics_history)
            if features.size > 0:
                X.append(features.flatten())
                y.append(target_instances)
        
        if len(X) < 10:
            logger.warning("Insufficient training data, using default scaling")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model training complete - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        # Save model
        self.save_model()
        self.is_trained = True
        return test_score > 0.7
    
    def predict_workload(self, metrics_history: List[WorkloadMetrics]) -> Tuple[int, float]:
        """Predict required instances for future workload"""
        if not self.is_trained:
            # Fallback to simple heuristic
            latest = metrics_history[-1]
            if latest.cpu_usage > 80 or latest.memory_usage > 85:
                return min(len(metrics_history) + 2, 50), 0.5
            elif latest.cpu_usage < 30 and latest.memory_usage < 40:
                return max(len(metrics_history) - 1, 1), 0.5
            return len(metrics_history), 0.3
        
        features = self.extract_features(metrics_history)
        if features.size == 0:
            return 1, 0.0
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on feature importance and prediction stability
        confidence = min(0.95, max(0.1, 1.0 / (1.0 + abs(prediction - len(metrics_history)))))
        
        return max(1, min(50, int(round(prediction)))), confidence
    
    def save_model(self):
        """Save trained model to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        if not self.model_path.exists():
            return False
        
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

class PredictiveScaler:
    """Main predictive scaling engine"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.predictor = MLWorkloadPredictor()
        self.metrics_history: List[WorkloadMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        self.last_scaling_action = 0
        self.current_instances = config.min_instances
        
        # Load existing model
        self.predictor.load_model()
        
        logger.info("Predictive Scaler initialized")
    
    async def add_metrics(self, metrics: WorkloadMetrics):
        """Add new metrics to the history"""
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of data
        cutoff_time = time.time() - 86400
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        logger.debug(f"Added metrics: CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}%")
    
    def calculate_reactive_scaling(self, current_metrics: WorkloadMetrics) -> Tuple[int, float, str]:
        """Calculate scaling decision based on current metrics (reactive)"""
        target_instances = self.current_instances
        confidence = 0.8
        reasoning = "Reactive scaling: "
        
        # CPU-based scaling
        if current_metrics.cpu_usage > self.config.scale_up_threshold:
            additional_instances = max(1, int((current_metrics.cpu_usage - self.config.target_cpu_utilization) / 20))
            target_instances = min(self.config.max_instances, self.current_instances + additional_instances)
            reasoning += f"CPU usage high ({current_metrics.cpu_usage:.1f}%)"
        elif current_metrics.cpu_usage < self.config.scale_down_threshold:
            target_instances = max(self.config.min_instances, self.current_instances - 1)
            reasoning += f"CPU usage low ({current_metrics.cpu_usage:.1f}%)"
        
        # Memory-based scaling
        elif current_metrics.memory_usage > self.config.scale_up_threshold:
            target_instances = min(self.config.max_instances, self.current_instances + 1)
            reasoning += f"Memory usage high ({current_metrics.memory_usage:.1f}%)"
        
        # Request rate scaling
        elif current_metrics.queue_depth > 100:
            additional_instances = max(1, current_metrics.queue_depth // 50)
            target_instances = min(self.config.max_instances, self.current_instances + additional_instances)
            reasoning += f"Queue depth high ({current_metrics.queue_depth})"
        
        else:
            reasoning += "No scaling needed"
        
        return target_instances, confidence, reasoning
    
    async def make_scaling_decision(self) -> Optional[ScalingDecision]:
        """Make intelligent scaling decision combining reactive and predictive approaches"""
        if len(self.metrics_history) < 1:
            return None
        
        current_time = time.time()
        current_metrics = self.metrics_history[-1]
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.config.cooldown_period:
            return None
        
        # Get reactive scaling decision
        reactive_instances, reactive_confidence, reactive_reasoning = self.calculate_reactive_scaling(current_metrics)
        
        # Get predictive scaling decision
        predictive_instances, predictive_confidence = self.predictor.predict_workload(self.metrics_history)
        
        # Combine reactive and predictive decisions
        ml_weight = self.config.ml_prediction_weight
        reactive_weight = 1.0 - ml_weight
        
        # Weighted average for target instances
        combined_instances = int(round(
            ml_weight * predictive_instances + reactive_weight * reactive_instances
        ))
        combined_instances = max(self.config.min_instances, min(self.config.max_instances, combined_instances))
        
        # Combined confidence
        combined_confidence = ml_weight * predictive_confidence + reactive_weight * reactive_confidence
        
        # Determine action
        action = "no_action"
        if combined_instances > self.current_instances:
            action = "scale_up"
        elif combined_instances < self.current_instances:
            action = "scale_down"
        
        # Calculate cost impact
        instance_cost = 0.10  # $0.10 per hour per instance (example)
        cost_impact = (combined_instances - self.current_instances) * instance_cost
        
        reasoning = f"Combined decision: Reactive={reactive_instances} (conf={reactive_confidence:.2f}), "
        reasoning += f"Predictive={predictive_instances} (conf={predictive_confidence:.2f}), "
        reasoning += f"Final={combined_instances}. {reactive_reasoning}"
        
        decision = ScalingDecision(
            timestamp=current_time,
            action=action,
            target_instances=combined_instances,
            current_instances=self.current_instances,
            confidence=combined_confidence,
            reasoning=reasoning,
            cost_impact=cost_impact
        )
        
        # Only return decision if action is needed and confidence is sufficient
        if action != "no_action" and combined_confidence > 0.6:
            return decision
        
        return None
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision"""
        logger.info(f"Executing scaling decision: {decision.action} to {decision.target_instances} instances "
                   f"(confidence={decision.confidence:.2f}, cost_impact=${decision.cost_impact:.2f}/hour)")
        
        try:
            # Here you would integrate with your container orchestrator (K8s, Docker Swarm, etc.)
            # For now, we'll simulate the scaling action
            await self.simulate_scaling(decision.target_instances)
            
            self.current_instances = decision.target_instances
            self.last_scaling_action = decision.timestamp
            self.scaling_history.append(decision)
            
            logger.info(f"Scaling completed successfully: {self.current_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
    
    async def simulate_scaling(self, target_instances: int):
        """Simulate scaling action (replace with real orchestrator integration)"""
        await asyncio.sleep(0.1)  # Simulate scaling delay
        logger.debug(f"Simulated scaling to {target_instances} instances")
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling performance metrics"""
        if not self.scaling_history:
            return {}
        
        recent_decisions = [d for d in self.scaling_history if d.timestamp > time.time() - 3600]
        
        return {
            "total_scaling_actions": len(self.scaling_history),
            "recent_scaling_actions": len(recent_decisions),
            "current_instances": self.current_instances,
            "average_confidence": np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0,
            "cost_savings_last_hour": -sum([d.cost_impact for d in recent_decisions]),
            "prediction_accuracy": self.predictor.is_trained,
            "last_scaling_action": self.last_scaling_action
        }
    
    async def train_from_history(self):
        """Train ML model from historical scaling decisions"""
        if len(self.scaling_history) < 20:
            logger.info("Insufficient scaling history for training")
            return
        
        training_data = []
        for decision in self.scaling_history:
            # Find metrics that led to this decision
            decision_metrics = [m for m in self.metrics_history 
                             if abs(m.timestamp - decision.timestamp) < 300]
            if len(decision_metrics) >= 3:
                training_data.append((decision_metrics[-3:], decision.target_instances))
        
        if training_data:
            success = self.predictor.train(training_data)
            if success:
                logger.info("ML model training completed successfully")
            else:
                logger.warning("ML model training failed or insufficient data")

async def main():
    """Example usage of the Predictive Scaler"""
    config = ScalingConfig(
        min_instances=2,
        max_instances=20,
        target_cpu_utilization=70.0,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        cooldown_period=180
    )
    
    scaler = PredictiveScaler(config)
    
    logger.info("Starting predictive scaling demo")
    
    # Simulate workload pattern
    for i in range(100):
        # Generate realistic metrics with daily patterns
        hour = (time.time() / 3600) % 24
        base_load = 30 + 40 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
        noise = np.random.normal(0, 10)
        
        metrics = WorkloadMetrics(
            timestamp=time.time(),
            cpu_usage=max(0, min(100, base_load + noise)),
            memory_usage=max(0, min(100, base_load * 0.8 + noise * 0.5)),
            request_rate=max(0, base_load * 2 + noise),
            response_time=max(50, 200 - base_load + abs(noise)),
            queue_depth=max(0, int((base_load - 50) * 2 + noise)),
            error_rate=max(0, min(10, (base_load - 80) * 0.1 + noise * 0.1)),
            active_connections=max(0, int(base_load * 5 + noise * 2))
        )
        
        await scaler.add_metrics(metrics)
        
        # Make scaling decision every 10 iterations
        if i % 10 == 0:
            decision = await scaler.make_scaling_decision()
            if decision:
                await scaler.execute_scaling_decision(decision)
        
        await asyncio.sleep(0.1)  # Simulate time passage
    
    # Train model from collected data
    await scaler.train_from_history()
    
    # Print final metrics
    final_metrics = scaler.get_scaling_metrics()
    logger.info(f"Scaling demo completed: {json.dumps(final_metrics, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())