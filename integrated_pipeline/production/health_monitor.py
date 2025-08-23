"""
Enterprise-Grade Production Health Monitor
Comprehensive health monitoring system with predictive analytics and real-time alerts
"""

import asyncio
import json
import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import requests
import yaml
from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry, push_to_gateway
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

@dataclass
class HealthMetrics:
    """Comprehensive health metrics"""
    timestamp: datetime
    
    # System metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    
    # Application metrics
    throughput_docs_per_hour: float
    average_quality_score: float
    error_rate_percent: float
    response_time_ms: float
    
    # M3 Max specific metrics
    temperature_celsius: float
    power_consumption_watts: float
    numa_efficiency: float
    cache_hit_ratio: float
    
    # Process metrics
    rust_core_memory_gb: float
    python_ml_memory_gb: float
    ipc_memory_gb: float
    
    # Pipeline metrics
    queue_depth: int
    active_connections: int
    model_switch_count: int
    prediction_accuracy: float
    
    # Health status
    overall_health_score: float
    status: str  # "healthy", "warning", "critical", "degraded"
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class HealthAlert:
    """Health alert definition"""
    alert_id: str
    severity: str  # "info", "warning", "critical"
    component: str
    metric: str
    threshold: float
    current_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class PredictiveInsight:
    """Predictive analytics insight"""
    insight_id: str
    prediction_type: str  # "bottleneck", "failure", "optimization"
    confidence: float
    time_horizon_minutes: int
    predicted_metric: str
    predicted_value: float
    current_trend: str
    recommendation: str
    timestamp: datetime

class ProductionHealthMonitor:
    """Enterprise-grade health monitoring with predictive analytics"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Monitoring state
        self.running = False
        self.metrics_history: List[HealthMetrics] = []
        self.alerts: List[HealthAlert] = []
        self.predictions: List[PredictiveInsight] = []
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Predictive models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trend_predictors = {}
        
        # Alert channels
        self.alert_channels = {
            'slack': self._send_slack_alert,
            'email': self._send_email_alert,
            'pagerduty': self._send_pagerduty_alert
        }
        
        # Component health checkers
        self.health_checkers = {
            'rust_core': self._check_rust_core_health,
            'python_ml': self._check_python_ml_health,
            'ipc_layer': self._check_ipc_health,
            'models': self._check_models_health,
            'storage': self._check_storage_health,
            'network': self._check_network_health
        }
        
        # Performance baselines
        self.performance_baselines = {}
        self._initialize_baselines()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "interval_seconds": 5,
                "history_retention_hours": 24,
                "enable_predictive_analytics": True,
                "prometheus_gateway": "http://localhost:9091",
                "health_check_endpoints": {
                    "rust_core": "http://localhost:8080/health",
                    "python_ml": "http://localhost:8081/health",
                    "ipc": "http://localhost:8082/health"
                }
            },
            "thresholds": {
                "cpu_warning": 70,
                "cpu_critical": 85,
                "memory_warning": 80,
                "memory_critical": 90,
                "disk_warning": 80,
                "disk_critical": 90,
                "error_rate_warning": 1,
                "error_rate_critical": 5,
                "response_time_warning": 1000,
                "response_time_critical": 2000,
                "temperature_warning": 80,
                "temperature_critical": 90,
                "quality_score_warning": 0.75,
                "quality_score_critical": 0.70
            },
            "alerts": {
                "enable_smart_alerts": True,
                "cooldown_minutes": 15,
                "channels": ["slack", "email"]
            },
            "predictive": {
                "enable": True,
                "prediction_window_minutes": 30,
                "anomaly_sensitivity": 0.1,
                "trend_analysis_points": 100
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                if config_path.suffix == '.yaml':
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # System metrics
        self.cpu_usage_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_usage_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        
        # Application metrics
        self.throughput_gauge = Gauge('app_throughput_docs_per_hour', 'Document processing throughput', registry=self.registry)
        self.quality_gauge = Gauge('app_quality_score', 'Average quality score', registry=self.registry)
        self.error_rate_gauge = Gauge('app_error_rate_percent', 'Error rate percentage', registry=self.registry)
        self.response_time_gauge = Gauge('app_response_time_ms', 'Average response time', registry=self.registry)
        
        # M3 Max specific
        self.temperature_gauge = Gauge('m3_max_temperature_celsius', 'M3 Max temperature', registry=self.registry)
        self.power_gauge = Gauge('m3_max_power_watts', 'M3 Max power consumption', registry=self.registry)
        
        # Health score
        self.health_score_gauge = Gauge('system_health_score', 'Overall health score', registry=self.registry)
        
        # Counters
        self.alert_counter = Counter('alerts_total', 'Total alerts generated', ['severity', 'component'], registry=self.registry)
        self.prediction_counter = Counter('predictions_total', 'Total predictions made', ['type'], registry=self.registry)
        
        # Histograms
        self.monitoring_duration = Histogram('monitoring_cycle_duration_seconds', 'Monitoring cycle duration', registry=self.registry)
    
    def _initialize_baselines(self):
        """Initialize performance baselines"""
        self.performance_baselines = {
            'throughput_docs_per_hour': 25.0,
            'quality_score': 0.80,
            'response_time_ms': 1000.0,
            'cpu_usage_percent': 50.0,
            'memory_usage_percent': 70.0,
            'error_rate_percent': 0.5
        }
    
    async def start_monitoring(self):
        """Start the health monitoring system"""
        self.running = True
        self.logger.info("Starting production health monitoring...")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self._predictive_analytics_loop()),
            asyncio.create_task(self._metrics_cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Health monitoring error: {e}")
            self.running = False
    
    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        self.running = False
        self.logger.info("Stopping health monitoring...")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            start_time = time.time()
            
            try:
                # Collect metrics
                metrics = await self._collect_comprehensive_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Check thresholds and generate alerts
                await self._check_thresholds(metrics)
                
                # Log health status
                self.logger.info(f"Health check: {metrics.status} (score: {metrics.overall_health_score:.2f})")
                
                # Record monitoring duration
                duration = time.time() - start_time
                self.monitoring_duration.observe(duration)
                
                # Ensure sub-1% overhead
                if duration > 0.05:  # 50ms max for 5s interval = 1% overhead
                    self.logger.warning(f"Monitoring overhead: {duration:.3f}s exceeds target")
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next interval
            await asyncio.sleep(self.config['monitoring']['interval_seconds'])
    
    async def _collect_comprehensive_metrics(self) -> HealthMetrics:
        """Collect comprehensive system and application metrics"""
        timestamp = datetime.now()
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Temperature (M3 Max specific - simulated)
        temperature = await self._get_system_temperature()
        
        # Application metrics from endpoints
        app_metrics = await self._collect_application_metrics()
        
        # Process-specific metrics
        process_metrics = await self._collect_process_metrics()
        
        # Calculate health score
        health_score = self._calculate_health_score({
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent / disk.total * 100,
            'temperature': temperature,
            **app_metrics,
            **process_metrics
        })
        
        # Determine status
        status = self._determine_health_status(health_score, {
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'error_rate': app_metrics.get('error_rate_percent', 0),
            'temperature': temperature
        })
        
        return HealthMetrics(
            timestamp=timestamp,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent / disk.total * 100,
            temperature_celsius=temperature,
            power_consumption_watts=app_metrics.get('power_consumption_watts', 45.0),
            numa_efficiency=process_metrics.get('numa_efficiency', 0.95),
            cache_hit_ratio=process_metrics.get('cache_hit_ratio', 0.92),
            throughput_docs_per_hour=app_metrics.get('throughput_docs_per_hour', 0.0),
            average_quality_score=app_metrics.get('average_quality_score', 0.0),
            error_rate_percent=app_metrics.get('error_rate_percent', 0.0),
            response_time_ms=app_metrics.get('response_time_ms', 0.0),
            rust_core_memory_gb=process_metrics.get('rust_core_memory_gb', 0.0),
            python_ml_memory_gb=process_metrics.get('python_ml_memory_gb', 0.0),
            ipc_memory_gb=process_metrics.get('ipc_memory_gb', 0.0),
            queue_depth=app_metrics.get('queue_depth', 0),
            active_connections=app_metrics.get('active_connections', 0),
            model_switch_count=app_metrics.get('model_switch_count', 0),
            prediction_accuracy=app_metrics.get('prediction_accuracy', 0.0),
            overall_health_score=health_score,
            status=status
        )
    
    async def _get_system_temperature(self) -> float:
        """Get M3 Max system temperature (simulated)"""
        try:
            # On macOS, we might use powermetrics or other tools
            # For now, simulate realistic temperature based on load
            cpu_usage = psutil.cpu_percent()
            base_temp = 35.0  # Base temperature
            load_temp = cpu_usage * 0.5  # Temperature increase with load
            return min(base_temp + load_temp, 95.0)  # Max 95°C
        except:
            return 40.0  # Default temperature
    
    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect metrics from application endpoints"""
        metrics = {}
        
        for component, endpoint in self.config['monitoring']['health_check_endpoints'].items():
            try:
                response = requests.get(endpoint, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract metrics based on component
                    if component == 'rust_core':
                        metrics.update({
                            'throughput_docs_per_hour': data.get('throughput', 0.0),
                            'error_rate_percent': data.get('error_rate', 0.0),
                            'response_time_ms': data.get('response_time_ms', 0.0)
                        })
                    elif component == 'python_ml':
                        metrics.update({
                            'average_quality_score': data.get('quality_score', 0.0),
                            'model_switch_count': data.get('model_switches', 0),
                            'prediction_accuracy': data.get('prediction_accuracy', 0.0)
                        })
                    elif component == 'ipc':
                        metrics.update({
                            'queue_depth': data.get('queue_depth', 0),
                            'active_connections': data.get('connections', 0)
                        })
                        
            except Exception as e:
                self.logger.warning(f"Failed to collect metrics from {component}: {e}")
                # Use default values
                if component == 'rust_core':
                    metrics.update({
                        'throughput_docs_per_hour': 0.0,
                        'error_rate_percent': 100.0,  # High error rate if unreachable
                        'response_time_ms': 10000.0
                    })
        
        return metrics
    
    async def _collect_process_metrics(self) -> Dict[str, float]:
        """Collect process-specific metrics"""
        metrics = {}
        
        try:
            # Find processes by name (simplified)
            rust_memory = 0.0
            python_memory = 0.0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'rust' in proc.info['name'].lower():
                        rust_memory += proc.info['memory_info'].rss / (1024**3)  # GB
                    elif 'python' in proc.info['name'].lower():
                        python_memory += proc.info['memory_info'].rss / (1024**3)  # GB
                except:
                    continue
            
            metrics.update({
                'rust_core_memory_gb': rust_memory,
                'python_ml_memory_gb': python_memory,
                'ipc_memory_gb': 5.0,  # Simulated IPC memory usage
                'numa_efficiency': 0.95,  # M3 Max unified memory efficiency
                'cache_hit_ratio': 0.92  # Simulated cache performance
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting process metrics: {e}")
            # Default values
            metrics.update({
                'rust_core_memory_gb': 30.0,
                'python_ml_memory_gb': 20.0,
                'ipc_memory_gb': 5.0,
                'numa_efficiency': 0.90,
                'cache_hit_ratio': 0.85
            })
        
        return metrics
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall health score (0.0 - 1.0)"""
        scores = []
        
        # System health (30% weight)
        cpu_score = max(0, 1.0 - (metrics.get('cpu_usage', 0) / 100.0))
        memory_score = max(0, 1.0 - (metrics.get('memory_usage', 0) / 100.0))
        disk_score = max(0, 1.0 - (metrics.get('disk_usage', 0) / 100.0))
        system_score = (cpu_score + memory_score + disk_score) / 3
        scores.append(('system', system_score, 0.30))
        
        # Application performance (40% weight)
        throughput_score = min(1.0, metrics.get('throughput_docs_per_hour', 0) / 35.0)  # Target 35 docs/hour
        quality_score = metrics.get('average_quality_score', 0.8)
        error_score = max(0, 1.0 - (metrics.get('error_rate_percent', 0) / 5.0))  # 5% max error rate
        response_score = max(0, 1.0 - (metrics.get('response_time_ms', 1000) / 2000.0))  # 2s max response
        app_score = (throughput_score + quality_score + error_score + response_score) / 4
        scores.append(('application', app_score, 0.40))
        
        # Hardware health (20% weight)
        temp_score = max(0, 1.0 - (metrics.get('temperature', 40) - 35) / 55.0)  # 35-90°C range
        power_score = max(0, 1.0 - (metrics.get('power_consumption_watts', 45) - 30) / 100.0)
        hardware_score = (temp_score + power_score) / 2
        scores.append(('hardware', hardware_score, 0.20))
        
        # Efficiency metrics (10% weight)
        numa_score = metrics.get('numa_efficiency', 0.95)
        cache_score = metrics.get('cache_hit_ratio', 0.92)
        efficiency_score = (numa_score + cache_score) / 2
        scores.append(('efficiency', efficiency_score, 0.10))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        return max(0.0, min(1.0, total_score))
    
    def _determine_health_status(self, health_score: float, metrics: Dict[str, float]) -> str:
        """Determine overall health status"""
        # Critical conditions (immediate attention required)
        if (metrics.get('cpu_usage', 0) > 95 or 
            metrics.get('memory_usage', 0) > 95 or 
            metrics.get('error_rate', 0) > 10 or
            metrics.get('temperature', 40) > 90):
            return "critical"
        
        # Warning conditions
        if (health_score < 0.7 or
            metrics.get('cpu_usage', 0) > 80 or
            metrics.get('memory_usage', 0) > 85 or
            metrics.get('error_rate', 0) > 3):
            return "warning"
        
        # Degraded performance
        if health_score < 0.8:
            return "degraded"
        
        # Healthy
        return "healthy"
    
    def _update_prometheus_metrics(self, metrics: HealthMetrics):
        """Update Prometheus metrics"""
        try:
            # System metrics
            self.cpu_usage_gauge.set(metrics.cpu_usage_percent)
            self.memory_usage_gauge.set(metrics.memory_usage_percent)
            self.disk_usage_gauge.set(metrics.disk_usage_percent)
            
            # Application metrics
            self.throughput_gauge.set(metrics.throughput_docs_per_hour)
            self.quality_gauge.set(metrics.average_quality_score)
            self.error_rate_gauge.set(metrics.error_rate_percent)
            self.response_time_gauge.set(metrics.response_time_ms)
            
            # M3 Max metrics
            self.temperature_gauge.set(metrics.temperature_celsius)
            self.power_gauge.set(metrics.power_consumption_watts)
            
            # Health score
            self.health_score_gauge.set(metrics.overall_health_score)
            
            # Push to gateway if configured
            if self.config['monitoring'].get('prometheus_gateway'):
                try:
                    push_to_gateway(
                        self.config['monitoring']['prometheus_gateway'],
                        job='hybrid-pipeline-health',
                        registry=self.registry
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to push to Prometheus gateway: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error updating Prometheus metrics: {e}")
    
    async def _check_thresholds(self, metrics: HealthMetrics):
        """Check thresholds and generate alerts"""
        thresholds = self.config['thresholds']
        
        # Check each threshold
        checks = [
            ('cpu_usage', metrics.cpu_usage_percent, thresholds['cpu_warning'], thresholds['cpu_critical']),
            ('memory_usage', metrics.memory_usage_percent, thresholds['memory_warning'], thresholds['memory_critical']),
            ('disk_usage', metrics.disk_usage_percent, thresholds['disk_warning'], thresholds['disk_critical']),
            ('error_rate', metrics.error_rate_percent, thresholds['error_rate_warning'], thresholds['error_rate_critical']),
            ('response_time', metrics.response_time_ms, thresholds['response_time_warning'], thresholds['response_time_critical']),
            ('temperature', metrics.temperature_celsius, thresholds['temperature_warning'], thresholds['temperature_critical']),
            ('quality_score', metrics.average_quality_score, thresholds['quality_score_warning'], thresholds['quality_score_critical'], True)  # Lower is worse
        ]
        
        for check in checks:
            metric_name = check[0]
            current_value = check[1]
            warning_threshold = check[2]
            critical_threshold = check[3]
            lower_is_worse = len(check) > 4 and check[4]
            
            # Skip if no data
            if current_value == 0 and metric_name in ['throughput_docs_per_hour', 'average_quality_score']:
                continue
            
            severity = None
            if lower_is_worse:
                if current_value < critical_threshold:
                    severity = 'critical'
                elif current_value < warning_threshold:
                    severity = 'warning'
            else:
                if current_value > critical_threshold:
                    severity = 'critical'
                elif current_value > warning_threshold:
                    severity = 'warning'
            
            if severity:
                await self._create_alert(
                    component='system',
                    metric=metric_name,
                    threshold=critical_threshold if severity == 'critical' else warning_threshold,
                    current_value=current_value,
                    severity=severity,
                    message=f"{metric_name} {severity}: {current_value:.2f} (threshold: {critical_threshold if severity == 'critical' else warning_threshold})"
                )
    
    async def _create_alert(self, component: str, metric: str, threshold: float, 
                          current_value: float, severity: str, message: str):
        """Create a new alert"""
        alert_id = f"{component}_{metric}_{severity}_{int(time.time())}"
        
        alert = HealthAlert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            metric=metric,
            threshold=threshold,
            current_value=current_value,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.alert_counter.labels(severity=severity, component=component).inc()
        
        self.logger.error(f"ALERT [{severity.upper()}]: {message}")
    
    async def _alert_processing_loop(self):
        """Process and send alerts"""
        while self.running:
            try:
                # Process unacknowledged alerts
                unack_alerts = [alert for alert in self.alerts if not alert.acknowledged]
                
                for alert in unack_alerts:
                    # Check cooldown
                    if self._is_alert_in_cooldown(alert):
                        continue
                    
                    # Send alert through configured channels
                    for channel in self.config['alerts']['channels']:
                        if channel in self.alert_channels:
                            await self.alert_channels[channel](alert)
                    
                    # Mark as processed (not acknowledged - that's manual)
                    alert.acknowledged = True
                    
            except Exception as e:
                self.logger.error(f"Error in alert processing: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    def _is_alert_in_cooldown(self, alert: HealthAlert) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_minutes = self.config['alerts']['cooldown_minutes']
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        # Check if we have a recent similar alert
        for existing_alert in self.alerts:
            if (existing_alert.component == alert.component and
                existing_alert.metric == alert.metric and
                existing_alert.severity == alert.severity and
                existing_alert.timestamp > cutoff_time):
                return True
        
        return False
    
    async def _send_slack_alert(self, alert: HealthAlert):
        """Send alert to Slack (placeholder)"""
        # Implementation would use Slack API
        self.logger.info(f"Would send Slack alert: {alert.message}")
    
    async def _send_email_alert(self, alert: HealthAlert):
        """Send alert via email (placeholder)"""
        # Implementation would use SMTP
        self.logger.info(f"Would send email alert: {alert.message}")
    
    async def _send_pagerduty_alert(self, alert: HealthAlert):
        """Send alert to PagerDuty (placeholder)"""
        # Implementation would use PagerDuty API
        self.logger.info(f"Would send PagerDuty alert: {alert.message}")
    
    async def _predictive_analytics_loop(self):
        """Predictive analytics and anomaly detection loop"""
        while self.running:
            try:
                if len(self.metrics_history) >= 20:  # Need minimum data points
                    await self._run_predictive_analysis()
                    
            except Exception as e:
                self.logger.error(f"Error in predictive analytics: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    async def _run_predictive_analysis(self):
        """Run predictive analysis on collected metrics"""
        if not self.config['predictive']['enable']:
            return
        
        # Prepare data for analysis
        recent_metrics = self.metrics_history[-self.config['predictive']['trend_analysis_points']:]
        
        # Convert to feature matrix
        features = []
        targets = {}
        
        for metric in recent_metrics:
            feature_vector = [
                metric.cpu_usage_percent,
                metric.memory_usage_percent,
                metric.throughput_docs_per_hour,
                metric.response_time_ms,
                metric.error_rate_percent,
                metric.temperature_celsius
            ]
            features.append(feature_vector)
            
            # Track individual metrics for trend prediction
            timestamp = metric.timestamp.timestamp()
            for attr in ['cpu_usage_percent', 'memory_usage_percent', 'throughput_docs_per_hour']:
                if attr not in targets:
                    targets[attr] = []
                targets[attr].append((timestamp, getattr(metric, attr)))
        
        features_array = np.array(features)
        
        # Anomaly detection
        await self._detect_anomalies(features_array, recent_metrics[-1])
        
        # Trend prediction
        await self._predict_trends(targets)
        
        # Bottleneck analysis
        await self._analyze_bottlenecks(recent_metrics)
    
    async def _detect_anomalies(self, features: np.ndarray, latest_metric: HealthMetrics):
        """Detect anomalies in current metrics"""
        try:
            # Fit anomaly detector
            self.anomaly_detector.fit(features[:-1])  # Train on historical data
            
            # Check latest metric
            current_features = features[-1:] 
            anomaly_score = self.anomaly_detector.decision_function(current_features)[0]
            is_anomaly = self.anomaly_detector.predict(current_features)[0] == -1
            
            if is_anomaly:
                insight = PredictiveInsight(
                    insight_id=f"anomaly_{int(time.time())}",
                    prediction_type="anomaly",
                    confidence=abs(anomaly_score),
                    time_horizon_minutes=0,  # Current anomaly
                    predicted_metric="overall_system_behavior",
                    predicted_value=anomaly_score,
                    current_trend="anomalous",
                    recommendation="Investigate unusual system behavior patterns",
                    timestamp=datetime.now()
                )
                
                self.predictions.append(insight)
                self.prediction_counter.labels(type='anomaly').inc()
                self.logger.warning(f"Anomaly detected: {anomaly_score:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
    
    async def _predict_trends(self, targets: Dict[str, List[Tuple[float, float]]]):
        """Predict future trends for key metrics"""
        for metric_name, data_points in targets.items():
            if len(data_points) < 10:
                continue
                
            try:
                # Prepare data
                X = np.array([point[0] for point in data_points]).reshape(-1, 1)
                y = np.array([point[1] for point in data_points])
                
                # Fit trend model
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict future value (30 minutes ahead)
                future_time = time.time() + (self.config['predictive']['prediction_window_minutes'] * 60)
                predicted_value = model.predict([[future_time]])[0]
                
                # Calculate confidence based on recent trend consistency
                recent_predictions = model.predict(X[-10:])
                recent_actuals = y[-10:]
                confidence = 1.0 - np.mean(np.abs(recent_predictions - recent_actuals) / (recent_actuals + 1e-6))
                confidence = max(0.0, min(1.0, confidence))
                
                # Determine trend direction
                slope = model.coef_[0]
                if slope > 0.1:
                    trend = "increasing"
                elif slope < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                # Generate recommendation
                recommendation = self._generate_trend_recommendation(metric_name, predicted_value, trend, slope)
                
                if confidence > 0.6:  # Only high-confidence predictions
                    insight = PredictiveInsight(
                        insight_id=f"trend_{metric_name}_{int(time.time())}",
                        prediction_type="trend",
                        confidence=confidence,
                        time_horizon_minutes=self.config['predictive']['prediction_window_minutes'],
                        predicted_metric=metric_name,
                        predicted_value=predicted_value,
                        current_trend=trend,
                        recommendation=recommendation,
                        timestamp=datetime.now()
                    )
                    
                    self.predictions.append(insight)
                    self.prediction_counter.labels(type='trend').inc()
                    
            except Exception as e:
                self.logger.error(f"Error predicting trend for {metric_name}: {e}")
    
    def _generate_trend_recommendation(self, metric: str, predicted_value: float, trend: str, slope: float) -> str:
        """Generate recommendation based on trend analysis"""
        if metric == 'cpu_usage_percent' and trend == 'increasing' and predicted_value > 80:
            return "CPU usage trending upward. Consider scaling or optimizing workload."
        elif metric == 'memory_usage_percent' and trend == 'increasing' and predicted_value > 85:
            return "Memory usage growing. Monitor for memory leaks or increase allocation."
        elif metric == 'throughput_docs_per_hour' and trend == 'decreasing' and predicted_value < 20:
            return "Throughput declining. Investigate performance bottlenecks."
        else:
            return f"{metric} trending {trend} (slope: {slope:.3f}). Monitor closely."
    
    async def _analyze_bottlenecks(self, recent_metrics: List[HealthMetrics]):
        """Analyze potential system bottlenecks"""
        if len(recent_metrics) < 5:
            return
        
        # Calculate metric correlations to identify bottlenecks
        metrics_data = {
            'cpu': [m.cpu_usage_percent for m in recent_metrics],
            'memory': [m.memory_usage_percent for m in recent_metrics],
            'throughput': [m.throughput_docs_per_hour for m in recent_metrics],
            'response_time': [m.response_time_ms for m in recent_metrics],
            'error_rate': [m.error_rate_percent for m in recent_metrics]
        }
        
        # Identify bottlenecks based on patterns
        bottlenecks = []
        
        # CPU bottleneck: High CPU with decreasing throughput
        if (np.mean(metrics_data['cpu'][-3:]) > 80 and 
            np.mean(metrics_data['throughput'][-3:]) < np.mean(metrics_data['throughput'][:-3])):
            bottlenecks.append("CPU processing capacity limiting throughput")
        
        # Memory bottleneck: High memory with increasing response times
        if (np.mean(metrics_data['memory'][-3:]) > 85 and 
            np.mean(metrics_data['response_time'][-3:]) > np.mean(metrics_data['response_time'][:-3]) * 1.5):
            bottlenecks.append("Memory pressure causing response time degradation")
        
        # Quality vs Speed tradeoff
        latest_quality = recent_metrics[-1].average_quality_score
        latest_throughput = recent_metrics[-1].throughput_docs_per_hour
        if latest_quality < 0.75 and latest_throughput > 30:
            bottlenecks.append("Quality declining due to high throughput demands")
        
        # Generate bottleneck insights
        for bottleneck in bottlenecks:
            insight = PredictiveInsight(
                insight_id=f"bottleneck_{int(time.time())}",
                prediction_type="bottleneck",
                confidence=0.8,
                time_horizon_minutes=15,
                predicted_metric="system_performance",
                predicted_value=0.0,
                current_trend="degrading",
                recommendation=f"Bottleneck identified: {bottleneck}",
                timestamp=datetime.now()
            )
            
            self.predictions.append(insight)
            self.prediction_counter.labels(type='bottleneck').inc()
            self.logger.warning(f"Bottleneck detected: {bottleneck}")
    
    async def _metrics_cleanup_loop(self):
        """Clean up old metrics to prevent memory bloat"""
        while self.running:
            try:
                # Keep only recent metrics
                retention_hours = self.config['monitoring']['history_retention_hours']
                cutoff_time = datetime.now() - timedelta(hours=retention_hours)
                
                # Clean metrics
                self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
                
                # Clean alerts (keep last 100)
                self.alerts = self.alerts[-100:]
                
                # Clean predictions (keep last 50)
                self.predictions = self.predictions[-50:]
                
            except Exception as e:
                self.logger.error(f"Error in metrics cleanup: {e}")
            
            await asyncio.sleep(3600)  # Run every hour
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}
        
        latest = self.metrics_history[-1]
        
        # Calculate trends
        trends = {}
        if len(self.metrics_history) >= 10:
            recent = self.metrics_history[-10:]
            trends = {
                'cpu_trend': np.mean([m.cpu_usage_percent for m in recent[-5:]]) - np.mean([m.cpu_usage_percent for m in recent[:5]]),
                'memory_trend': np.mean([m.memory_usage_percent for m in recent[-5:]]) - np.mean([m.memory_usage_percent for m in recent[:5]]),
                'throughput_trend': np.mean([m.throughput_docs_per_hour for m in recent[-5:]]) - np.mean([m.throughput_docs_per_hour for m in recent[:5]])
            }
        
        # Recent alerts
        recent_alerts = [alert.to_dict() for alert in self.alerts[-10:]]
        
        # Recent predictions
        recent_predictions = [pred.__dict__ for pred in self.predictions[-5:]]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": latest.status,
            "health_score": latest.overall_health_score,
            "current_metrics": latest.to_dict(),
            "trends": trends,
            "recent_alerts": recent_alerts,
            "predictions": recent_predictions,
            "system_info": {
                "total_metrics_collected": len(self.metrics_history),
                "total_alerts_generated": len(self.alerts),
                "total_predictions_made": len(self.predictions),
                "monitoring_uptime_hours": (datetime.now() - self.metrics_history[0].timestamp).total_seconds() / 3600 if self.metrics_history else 0
            }
        }
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard"""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}
        
        # Time series data for charts
        time_series = {
            'timestamps': [m.timestamp.isoformat() for m in self.metrics_history[-100:]],
            'cpu_usage': [m.cpu_usage_percent for m in self.metrics_history[-100:]],
            'memory_usage': [m.memory_usage_percent for m in self.metrics_history[-100:]],
            'throughput': [m.throughput_docs_per_hour for m in self.metrics_history[-100:]],
            'quality_score': [m.average_quality_score for m in self.metrics_history[-100:]],
            'response_time': [m.response_time_ms for m in self.metrics_history[-100:]],
            'health_score': [m.overall_health_score for m in self.metrics_history[-100:]]
        }
        
        # Current status
        latest = self.metrics_history[-1]
        
        return {
            "current_status": {
                "health_score": latest.overall_health_score,
                "status": latest.status,
                "throughput": latest.throughput_docs_per_hour,
                "quality": latest.average_quality_score,
                "cpu_usage": latest.cpu_usage_percent,
                "memory_usage": latest.memory_usage_percent,
                "temperature": latest.temperature_celsius
            },
            "time_series": time_series,
            "alerts": {
                "active_count": len([a for a in self.alerts if not a.acknowledged]),
                "recent": [a.to_dict() for a in self.alerts[-5:]]
            },
            "predictions": {
                "count": len(self.predictions),
                "recent": [p.__dict__ for p in self.predictions[-3:]]
            }
        }

# Example usage
if __name__ == "__main__":
    async def run_monitor():
        monitor = ProductionHealthMonitor()
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # Let it run for a bit, then get report
        await asyncio.sleep(30)
        
        report = await monitor.get_health_report()
        print("Health Report:")
        print(json.dumps(report, indent=2, default=str))
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    asyncio.run(run_monitor())