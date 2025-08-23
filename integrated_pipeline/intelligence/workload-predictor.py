#!/usr/bin/env python3
"""
AI-Powered Workload Predictor for Intelligent Auto-Scaling
Implements advanced ML models for predicting workload patterns and resource needs
"""

import asyncio
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Time series analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available, some advanced time series features disabled")

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, deep learning features disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkloadDataPoint:
    """Single workload measurement point"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    concurrent_users: int
    error_rate: float
    network_io: float
    disk_io: float
    queue_depth: int
    
    # Derived features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_business_hours: bool = False
    
    def __post_init__(self):
        dt = datetime.fromtimestamp(self.timestamp)
        self.hour_of_day = dt.hour
        self.day_of_week = dt.weekday()
        self.is_weekend = dt.weekday() >= 5
        self.is_business_hours = 9 <= dt.hour <= 17 and not self.is_weekend

@dataclass
class WorkloadPrediction:
    """Predicted workload metrics"""
    timestamp: float
    horizon_minutes: int
    predicted_cpu: float
    predicted_memory: float
    predicted_requests: float
    predicted_response_time: float
    confidence_interval: Tuple[float, float]
    model_confidence: float
    anomaly_score: float

@dataclass
class ScalingRecommendation:
    """AI-generated scaling recommendation"""
    timestamp: float
    current_instances: int
    recommended_instances: int
    reasoning: str
    confidence: float
    expected_cost_change: float
    expected_performance_impact: float
    urgency_level: str  # 'low', 'medium', 'high', 'critical'

class FeatureEngineer:
    """Advanced feature engineering for workload prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_history = []
        
    def engineer_features(self, data_points: List[WorkloadDataPoint]) -> pd.DataFrame:
        """Create advanced features from raw workload data"""
        if not data_points:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(dp) for dp in data_points])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_business_hours'] = ((df.index.hour >= 9) & (df.index.hour <= 17) & (df['is_weekend'] == 0)).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Rolling statistics (if enough data)
        if len(df) >= 10:
            window_sizes = [5, 10, 20] if len(df) >= 20 else [5, min(len(df)//2, 10)]
            
            for window in window_sizes:
                df[f'cpu_rolling_mean_{window}'] = df['cpu_usage'].rolling(window=window, min_periods=1).mean()
                df[f'cpu_rolling_std_{window}'] = df['cpu_usage'].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'memory_rolling_mean_{window}'] = df['memory_usage'].rolling(window=window, min_periods=1).mean()
                df[f'requests_rolling_mean_{window}'] = df['request_rate'].rolling(window=window, min_periods=1).mean()
                df[f'response_time_rolling_mean_{window}'] = df['response_time'].rolling(window=window, min_periods=1).mean()
        
        # Lag features
        if len(df) >= 3:
            lags = [1, 2, 3] if len(df) >= 5 else [1, min(2, len(df)-1)]
            for lag in lags:
                df[f'cpu_lag_{lag}'] = df['cpu_usage'].shift(lag).fillna(df['cpu_usage'].iloc[0])
                df[f'memory_lag_{lag}'] = df['memory_usage'].shift(lag).fillna(df['memory_usage'].iloc[0])
                df[f'requests_lag_{lag}'] = df['request_rate'].shift(lag).fillna(df['request_rate'].iloc[0])
        
        # Rate of change features
        if len(df) >= 2:
            df['cpu_rate_of_change'] = df['cpu_usage'].diff().fillna(0)
            df['memory_rate_of_change'] = df['memory_usage'].diff().fillna(0)
            df['requests_rate_of_change'] = df['request_rate'].diff().fillna(0)
        
        # Interaction features
        df['cpu_memory_interaction'] = df['cpu_usage'] * df['memory_usage']
        df['load_density'] = df['request_rate'] / (df['cpu_usage'] + 1e-6)
        df['efficiency_score'] = df['request_rate'] / (df['response_time'] + 1e-6)
        
        # Anomaly indicators
        if len(df) >= 10:
            for col in ['cpu_usage', 'memory_usage', 'request_rate']:
                q75, q25 = np.percentile(df[col], [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                df[f'{col}_is_anomaly'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
        
        # Business logic features
        df['peak_hours'] = ((df.index.hour >= 9) & (df.index.hour <= 11) | 
                           (df.index.hour >= 14) & (df.index.hour <= 16)).astype(int)
        df['lunch_hours'] = ((df.index.hour >= 12) & (df.index.hour <= 13)).astype(int)
        df['night_hours'] = ((df.index.hour <= 6) | (df.index.hour >= 22)).astype(int)
        
        # Resource utilization ratios
        df['memory_to_cpu_ratio'] = df['memory_usage'] / (df['cpu_usage'] + 1e-6)
        df['utilization_balance'] = np.abs(df['cpu_usage'] - df['memory_usage'])
        
        # Fill any remaining NaNs
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Select most important features for prediction"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and timestamp columns
        feature_columns = [col for col in numeric_columns 
                          if col not in [target_column, 'timestamp'] and not col.endswith('_target')]
        
        return feature_columns[:50]  # Limit to top 50 features

class MultiModelPredictor:
    """Ensemble of multiple ML models for robust workload prediction"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("../intelligence/models/")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42
            ),
            'linear_ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        self.scalers = {}
        self.feature_columns = {}
        self.is_trained = {}
        self.model_weights = {}
        
        # Initialize for each target variable
        self.target_variables = ['cpu_usage', 'memory_usage', 'request_rate', 'response_time']
        
        for target in self.target_variables:
            self.scalers[target] = StandardScaler()
            self.is_trained[target] = False
            self.model_weights[target] = {name: 1.0 for name in self.models.keys()}
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str, 
                            horizon_minutes: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with proper time series structure"""
        if len(df) < 10:
            raise ValueError("Insufficient data for training")
        
        # Create target with future horizon
        horizon_steps = max(1, horizon_minutes // 5)  # Assuming 5-minute intervals
        df[f'{target_column}_target'] = df[target_column].shift(-horizon_steps)
        
        # Remove rows without targets
        df_clean = df.dropna(subset=[f'{target_column}_target'])
        
        if len(df_clean) < 5:
            raise ValueError("Insufficient data after creating targets")
        
        feature_columns = self.feature_columns.get(target_column, [])
        if not feature_columns:
            feature_engineer = FeatureEngineer()
            feature_columns = feature_engineer.select_features(df_clean, target_column)
            self.feature_columns[target_column] = feature_columns
        
        X = df_clean[feature_columns].values
        y = df_clean[f'{target_column}_target'].values
        
        return X, y
    
    def train_models(self, df: pd.DataFrame, target_column: str, horizon_minutes: int = 15):
        """Train all models for a specific target variable"""
        logger.info(f"Training models for {target_column} with {horizon_minutes}-minute horizon")
        
        try:
            X, y = self.prepare_training_data(df, target_column, horizon_minutes)
            
            # Split data temporally
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Scale features
            X_train_scaled = self.scalers[target_column].fit_transform(X_train)
            X_test_scaled = self.scalers[target_column].transform(X_test)
            
            model_scores = {}
            
            # Train each model
            for name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    train_score = r2_score(y_train, y_pred_train)
                    test_score = r2_score(y_test, y_pred_test)
                    mae = mean_absolute_error(y_test, y_pred_test)
                    
                    model_scores[name] = {
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'mae': mae,
                        'overall_score': test_score - abs(train_score - test_score) * 0.1  # Penalize overfitting
                    }
                    
                    logger.info(f"{name} - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}, MAE: {mae:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    model_scores[name] = {'overall_score': -1}
            
            # Calculate model weights based on performance
            max_score = max([scores['overall_score'] for scores in model_scores.values() if scores['overall_score'] > 0])
            if max_score > 0:
                for name, scores in model_scores.items():
                    if scores['overall_score'] > 0:
                        self.model_weights[target_column][name] = max(0.1, scores['overall_score'] / max_score)
                    else:
                        self.model_weights[target_column][name] = 0.1
            
            self.is_trained[target_column] = True
            self.save_models(target_column)
            
            logger.info(f"Model training completed for {target_column}. Weights: {self.model_weights[target_column]}")
            
        except Exception as e:
            logger.error(f"Failed to train models for {target_column}: {e}")
            self.is_trained[target_column] = False
    
    def predict(self, features: np.ndarray, target_column: str) -> Tuple[float, Tuple[float, float], float]:
        """Make ensemble prediction with confidence intervals"""
        if not self.is_trained.get(target_column, False):
            # Fallback prediction
            if target_column == 'cpu_usage':
                return 50.0, (20.0, 80.0), 0.3
            elif target_column == 'memory_usage':
                return 60.0, (30.0, 90.0), 0.3
            elif target_column == 'request_rate':
                return 100.0, (50.0, 200.0), 0.3
            else:  # response_time
                return 200.0, (100.0, 400.0), 0.3
        
        # Scale features
        features_scaled = self.scalers[target_column].transform(features.reshape(1, -1))
        
        predictions = []
        weights = []
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                pred = model.predict(features_scaled)[0]
                weight = self.model_weights[target_column].get(name, 0.1)
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Failed to get prediction from {name}: {e}")
        
        if not predictions:
            return 0.0, (0.0, 0.0), 0.0
        
        # Weighted ensemble prediction
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_pred = np.average(predictions, weights=weights)
        
        # Calculate confidence interval based on prediction variance
        variance = np.average((predictions - ensemble_pred) ** 2, weights=weights)
        std = np.sqrt(variance)
        confidence_interval = (
            ensemble_pred - 1.96 * std,
            ensemble_pred + 1.96 * std
        )
        
        # Calculate model confidence based on agreement
        confidence = max(0.1, 1.0 - (std / (abs(ensemble_pred) + 1e-6)))
        confidence = min(0.95, confidence)
        
        return float(ensemble_pred), confidence_interval, float(confidence)
    
    def save_models(self, target_column: str):
        """Save trained models for persistence"""
        model_file = self.model_dir / f"models_{target_column}.pkl"
        
        model_data = {
            'models': self.models,
            'scaler': self.scalers[target_column],
            'feature_columns': self.feature_columns[target_column],
            'weights': self.model_weights[target_column],
            'is_trained': self.is_trained[target_column]
        }
        
        joblib.dump(model_data, model_file)
        logger.info(f"Models saved for {target_column}")
    
    def load_models(self, target_column: str) -> bool:
        """Load pre-trained models"""
        model_file = self.model_dir / f"models_{target_column}.pkl"
        
        if not model_file.exists():
            return False
        
        try:
            model_data = joblib.load(model_file)
            self.models = model_data['models']
            self.scalers[target_column] = model_data['scaler']
            self.feature_columns[target_column] = model_data['feature_columns']
            self.model_weights[target_column] = model_data['weights']
            self.is_trained[target_column] = model_data['is_trained']
            
            logger.info(f"Models loaded for {target_column}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models for {target_column}: {e}")
            return False

class WorkloadPredictor:
    """Main workload prediction engine with advanced AI capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'prediction_horizons': [5, 15, 30, 60],  # minutes
            'min_training_points': 50,
            'retrain_interval_hours': 24,
            'anomaly_threshold': 2.0,
            'confidence_threshold': 0.6
        }
        
        self.feature_engineer = FeatureEngineer()
        self.predictor = MultiModelPredictor()
        self.workload_history: List[WorkloadDataPoint] = []
        self.prediction_history: List[WorkloadPrediction] = []
        self.last_training_time = 0
        
        # Load existing models
        for target in self.predictor.target_variables:
            self.predictor.load_models(target)
        
        logger.info("Workload Predictor initialized")
    
    async def add_workload_data(self, data_point: WorkloadDataPoint):
        """Add new workload data point"""
        self.workload_history.append(data_point)
        
        # Keep only recent data (last 7 days)
        cutoff_time = data_point.timestamp - (7 * 24 * 3600)
        self.workload_history = [dp for dp in self.workload_history if dp.timestamp > cutoff_time]
        
        # Auto-retrain if needed
        if (data_point.timestamp - self.last_training_time > self.config['retrain_interval_hours'] * 3600 and
            len(self.workload_history) >= self.config['min_training_points']):
            await self.train_models()
    
    async def train_models(self):
        """Train prediction models on historical data"""
        if len(self.workload_history) < self.config['min_training_points']:
            logger.warning(f"Insufficient data for training: {len(self.workload_history)} points")
            return False
        
        logger.info(f"Training models with {len(self.workload_history)} data points")
        
        # Convert to DataFrame with engineered features
        df = self.feature_engineer.engineer_features(self.workload_history)
        
        if df.empty:
            logger.error("Failed to engineer features")
            return False
        
        # Train models for each target variable
        for target in self.predictor.target_variables:
            if target in df.columns:
                self.predictor.train_models(df, target, horizon_minutes=15)
        
        self.last_training_time = self.workload_history[-1].timestamp
        logger.info("Model training completed")
        return True
    
    async def predict_workload(self, horizon_minutes: int = 15) -> Optional[WorkloadPrediction]:
        """Predict future workload metrics"""
        if len(self.workload_history) < 5:
            logger.warning("Insufficient data for prediction")
            return None
        
        # Engineer features for recent data
        recent_data = self.workload_history[-20:]  # Use last 20 points for context
        df = self.feature_engineer.engineer_features(recent_data)
        
        if df.empty:
            return None
        
        # Get latest feature vector
        latest_features = df.iloc[-1]
        
        predictions = {}
        confidence_scores = []
        
        # Predict each target variable
        for target in self.predictor.target_variables:
            if target in self.predictor.feature_columns:
                try:
                    feature_cols = self.predictor.feature_columns[target]
                    features = latest_features[feature_cols].values
                    
                    pred, conf_interval, confidence = self.predictor.predict(features, target)
                    predictions[target] = pred
                    confidence_scores.append(confidence)
                    
                except Exception as e:
                    logger.error(f"Failed to predict {target}: {e}")
                    predictions[target] = latest_features[target]
                    confidence_scores.append(0.3)
        
        # Calculate anomaly score
        anomaly_score = self.calculate_anomaly_score(predictions)
        
        # Overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.3
        
        # Create prediction
        future_timestamp = self.workload_history[-1].timestamp + (horizon_minutes * 60)
        
        prediction = WorkloadPrediction(
            timestamp=future_timestamp,
            horizon_minutes=horizon_minutes,
            predicted_cpu=predictions.get('cpu_usage', 50.0),
            predicted_memory=predictions.get('memory_usage', 60.0),
            predicted_requests=predictions.get('request_rate', 100.0),
            predicted_response_time=predictions.get('response_time', 200.0),
            confidence_interval=(
                predictions.get('cpu_usage', 50.0) * 0.8,
                predictions.get('cpu_usage', 50.0) * 1.2
            ),
            model_confidence=overall_confidence,
            anomaly_score=anomaly_score
        )
        
        self.prediction_history.append(prediction)
        
        # Keep prediction history limited
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
        
        logger.debug(f"Generated prediction for {horizon_minutes}min horizon: "
                    f"CPU={prediction.predicted_cpu:.1f}%, "
                    f"Memory={prediction.predicted_memory:.1f}%, "
                    f"Confidence={overall_confidence:.3f}")
        
        return prediction
    
    def calculate_anomaly_score(self, predictions: Dict[str, float]) -> float:
        """Calculate anomaly score for predictions"""
        if len(self.workload_history) < 10:
            return 0.0
        
        recent_data = self.workload_history[-10:]
        scores = []
        
        for target, pred_value in predictions.items():
            if hasattr(recent_data[0], target):
                historical_values = [getattr(dp, target) for dp in recent_data]
                mean_val = np.mean(historical_values)
                std_val = np.std(historical_values)
                
                if std_val > 0:
                    z_score = abs(pred_value - mean_val) / std_val
                    scores.append(z_score)
        
        return np.mean(scores) if scores else 0.0
    
    async def generate_scaling_recommendation(self, current_instances: int) -> Optional[ScalingRecommendation]:
        """Generate AI-powered scaling recommendation"""
        prediction = await self.predict_workload(horizon_minutes=15)
        
        if not prediction or prediction.model_confidence < self.config['confidence_threshold']:
            return None
        
        # Scaling logic based on predictions
        cpu_threshold_up = 75.0
        cpu_threshold_down = 30.0
        memory_threshold_up = 80.0
        
        scaling_factor = 0
        reasoning_parts = []
        
        # CPU-based scaling
        if prediction.predicted_cpu > cpu_threshold_up:
            cpu_scaling = max(1, int((prediction.predicted_cpu - cpu_threshold_up) / 20))
            scaling_factor += cpu_scaling
            reasoning_parts.append(f"CPU predicted at {prediction.predicted_cpu:.1f}% (+{cpu_scaling} instances)")
        elif prediction.predicted_cpu < cpu_threshold_down:
            scaling_factor -= 1
            reasoning_parts.append(f"CPU predicted at {prediction.predicted_cpu:.1f}% (-1 instance)")
        
        # Memory-based scaling
        if prediction.predicted_memory > memory_threshold_up:
            scaling_factor += 1
            reasoning_parts.append(f"Memory predicted at {prediction.predicted_memory:.1f}% (+1 instance)")
        
        # Request rate scaling
        if prediction.predicted_requests > 200:  # High request rate
            request_scaling = max(1, int(prediction.predicted_requests / 200))
            scaling_factor += request_scaling
            reasoning_parts.append(f"Request rate predicted at {prediction.predicted_requests:.1f}/s (+{request_scaling} instances)")
        
        # Anomaly-based scaling
        if prediction.anomaly_score > self.config['anomaly_threshold']:
            scaling_factor += 1
            reasoning_parts.append(f"Anomaly detected (score: {prediction.anomaly_score:.2f})")
        
        recommended_instances = max(1, min(50, current_instances + scaling_factor))
        
        if recommended_instances == current_instances:
            return None  # No change needed
        
        # Calculate urgency
        urgency = 'low'
        if abs(scaling_factor) >= 3 or prediction.anomaly_score > 3.0:
            urgency = 'critical'
        elif abs(scaling_factor) >= 2 or prediction.anomaly_score > 2.0:
            urgency = 'high'
        elif abs(scaling_factor) >= 1:
            urgency = 'medium'
        
        # Estimate cost and performance impact
        instance_cost = 0.10  # $0.10 per hour per instance
        cost_change = (recommended_instances - current_instances) * instance_cost
        performance_impact = scaling_factor * 0.2  # 20% per instance
        
        reasoning = f"AI prediction ({prediction.horizon_minutes}min): " + "; ".join(reasoning_parts)
        
        recommendation = ScalingRecommendation(
            timestamp=prediction.timestamp,
            current_instances=current_instances,
            recommended_instances=recommended_instances,
            reasoning=reasoning,
            confidence=prediction.model_confidence,
            expected_cost_change=cost_change,
            expected_performance_impact=performance_impact,
            urgency_level=urgency
        )
        
        logger.info(f"Generated scaling recommendation: {recommendation.reasoning}")
        return recommendation
    
    def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get prediction performance metrics"""
        if not self.prediction_history:
            return {}
        
        recent_predictions = self.prediction_history[-24:]  # Last 24 predictions
        
        return {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent_predictions),
            'average_confidence': np.mean([p.model_confidence for p in recent_predictions]),
            'average_anomaly_score': np.mean([p.anomaly_score for p in recent_predictions]),
            'models_trained': all(self.predictor.is_trained.values()),
            'data_points': len(self.workload_history),
            'last_training_time': self.last_training_time,
            'training_data_days': (self.workload_history[-1].timestamp - self.workload_history[0].timestamp) / 86400 if self.workload_history else 0
        }

async def main():
    """Example usage of the Workload Predictor"""
    predictor = WorkloadPredictor()
    
    logger.info("Starting workload prediction demo")
    
    # Generate realistic workload data
    import time
    base_time = time.time()
    
    for i in range(100):
        # Simulate daily and weekly patterns
        hour_of_day = ((base_time + i * 300) % 86400) / 3600  # 5-minute intervals
        day_of_week = ((base_time + i * 300) % (7 * 86400)) / 86400
        
        # Daily pattern (higher during business hours)
        daily_pattern = 30 + 40 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = 1.0 if day_of_week < 5 else 0.6
        
        # Add some noise and trends
        noise = np.random.normal(0, 10)
        trend = i * 0.1  # Slight upward trend
        
        base_cpu = max(0, min(100, daily_pattern * weekly_pattern + noise + trend))
        
        data_point = WorkloadDataPoint(
            timestamp=base_time + i * 300,  # 5-minute intervals
            cpu_usage=base_cpu,
            memory_usage=base_cpu * 0.8 + np.random.normal(0, 5),
            request_rate=max(0, base_cpu * 2 + np.random.normal(0, 20)),
            response_time=max(50, 300 - base_cpu + np.random.normal(0, 30)),
            concurrent_users=max(1, int(base_cpu * 0.5 + np.random.normal(0, 10))),
            error_rate=max(0, min(10, (base_cpu - 80) * 0.1 + np.random.normal(0, 0.5))),
            network_io=max(0, base_cpu * 1.5 + np.random.normal(0, 15)),
            disk_io=max(0, base_cpu * 1.2 + np.random.normal(0, 12)),
            queue_depth=max(0, int((base_cpu - 50) * 0.8 + np.random.normal(0, 5)))
        )
        
        await predictor.add_workload_data(data_point)
        
        # Generate predictions and recommendations periodically
        if i % 20 == 19 and i > 50:  # After collecting enough data
            prediction = await predictor.predict_workload(horizon_minutes=15)
            if prediction:
                logger.info(f"Prediction: CPU={prediction.predicted_cpu:.1f}%, "
                           f"Memory={prediction.predicted_memory:.1f}%, "
                           f"Confidence={prediction.model_confidence:.3f}")
            
            recommendation = await predictor.generate_scaling_recommendation(current_instances=5)
            if recommendation:
                logger.info(f"Scaling Recommendation: {recommendation.current_instances} → "
                           f"{recommendation.recommended_instances} instances "
                           f"(Urgency: {recommendation.urgency_level})")
    
    # Final metrics
    metrics = predictor.get_prediction_metrics()
    logger.info(f"Prediction metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())