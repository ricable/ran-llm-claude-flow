"""
Real-time Performance Monitoring via MCP

This module implements comprehensive real-time performance monitoring
for Python pipelines using the Model Context Protocol.
"""

import asyncio
import json
import logging
import psutil
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import threading
import weakref

from .protocol import (
    MCPMessage, MCPMessageType, MCPMethods,
    ModelMetrics, create_notification
)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int


@dataclass
class PipelineMetrics:
    """Pipeline-specific performance metrics"""
    pipeline_id: str
    timestamp: str
    tasks_completed: int
    tasks_failed: int
    tasks_running: int
    tasks_queued: int
    average_task_duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_tasks_per_minute: float
    error_rate: float


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    timestamp: str
    tasks_completed: int
    tasks_failed: int
    current_task_duration: Optional[float]
    average_task_duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    performance_score: float


@dataclass
class ModelPerformanceMetrics:
    """ML Model performance metrics"""
    model_id: str
    timestamp: str
    inference_count: int
    average_inference_time_ms: float
    memory_usage_mb: float
    gpu_usage_percent: float
    gpu_memory_mb: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    batch_size: int
    throughput_inferences_per_second: float


class MetricsCollector:
    """Base class for metrics collection"""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.running = False
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for metrics updates"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove metrics callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def _notify_callbacks(self, metrics: Dict[str, Any]):
        """Notify all callbacks of new metrics"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logging.error(f"Metrics callback error: {e}")
    
    async def start(self):
        """Start metrics collection"""
        self.running = True
        while self.running:
            try:
                metrics = await self.collect_metrics()
                await self._notify_callbacks(metrics)
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics (to be implemented by subclasses)"""
        raise NotImplementedError


class SystemMetricsCollector(MetricsCollector):
    """System resource metrics collector"""
    
    def __init__(self, collection_interval: float = 5.0):
        super().__init__(collection_interval)
        self.network_counters = psutil.net_io_counters()
        self.last_network_check = time.time()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        current_network = psutil.net_io_counters()
        current_time = time.time()
        time_diff = current_time - self.last_network_check
        
        network_sent_mb = (current_network.bytes_sent - self.network_counters.bytes_sent) / (1024 * 1024) / time_diff
        network_recv_mb = (current_network.bytes_recv - self.network_counters.bytes_recv) / (1024 * 1024) / time_diff
        
        self.network_counters = current_network
        self.last_network_check = current_time
        
        # Load average
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            # Windows doesn't have load average
            load_average = [0.0, 0.0, 0.0]
        
        # Process count
        process_count = len(psutil.pids())
        
        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            load_average=load_average,
            process_count=process_count
        )
        
        return {"type": "system", "metrics": asdict(metrics)}


class PipelineMetricsCollector(MetricsCollector):
    """Pipeline performance metrics collector"""
    
    def __init__(self, pipeline_manager, collection_interval: float = 10.0):
        super().__init__(collection_interval)
        self.pipeline_manager = weakref.ref(pipeline_manager)
        self.task_history = deque(maxlen=1000)  # Keep last 1000 tasks for calculations
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect pipeline metrics"""
        manager = self.pipeline_manager()
        if not manager:
            return {"type": "pipeline", "metrics": {}}
        
        all_metrics = {}
        
        for pipeline_id, pipeline in manager.pipelines.items():
            # Get tasks for this pipeline
            pipeline_tasks = [t for t in manager.tasks.values() if t.pipeline_id == pipeline_id]
            
            # Calculate metrics
            tasks_completed = len([t for t in pipeline_tasks if t.status.value == "completed"])
            tasks_failed = len([t for t in pipeline_tasks if t.status.value == "failed"])
            tasks_running = len([t for t in pipeline_tasks if t.status.value == "running"])
            tasks_queued = len([t for t in pipeline_tasks if t.status.value == "pending"])
            
            # Calculate average task duration
            completed_tasks = [t for t in pipeline_tasks if t.status.value == "completed" and t.started_at and t.completed_at]
            avg_duration = 0.0
            if completed_tasks:
                durations = []
                for task in completed_tasks:
                    started = datetime.fromisoformat(task.started_at)
                    completed = datetime.fromisoformat(task.completed_at)
                    durations.append((completed - started).total_seconds())
                avg_duration = sum(durations) / len(durations)
            
            # Calculate throughput (tasks per minute in last 10 minutes)
            now = datetime.now()
            recent_completed = [
                t for t in completed_tasks
                if t.completed_at and (now - datetime.fromisoformat(t.completed_at)).total_seconds() <= 600
            ]
            throughput = len(recent_completed) / 10.0  # tasks per minute
            
            # Calculate error rate
            total_finished = tasks_completed + tasks_failed
            error_rate = (tasks_failed / total_finished) if total_finished > 0 else 0.0
            
            # Estimate resource usage (simplified)
            memory_usage = 50.0 * tasks_running  # MB per running task
            cpu_usage = 10.0 * tasks_running  # % per running task
            
            metrics = PipelineMetrics(
                pipeline_id=pipeline_id,
                timestamp=datetime.now().isoformat(),
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
                tasks_running=tasks_running,
                tasks_queued=tasks_queued,
                average_task_duration=avg_duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=min(100.0, cpu_usage),
                throughput_tasks_per_minute=throughput,
                error_rate=error_rate
            )
            
            all_metrics[pipeline_id] = asdict(metrics)
        
        return {"type": "pipeline", "metrics": all_metrics}


class AgentMetricsCollector(MetricsCollector):
    """Agent performance metrics collector"""
    
    def __init__(self, agent_manager, collection_interval: float = 10.0):
        super().__init__(collection_interval)
        self.agent_manager = weakref.ref(agent_manager)
        self.agent_history = {}  # Track task history per agent
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect agent metrics"""
        manager = self.agent_manager()
        if not manager:
            return {"type": "agent", "metrics": {}}
        
        all_metrics = {}
        
        for agent_id, agent in manager.agents.items():
            # Initialize history if needed
            if agent_id not in self.agent_history:
                self.agent_history[agent_id] = {
                    "completed_tasks": [],
                    "failed_tasks": [],
                    "task_durations": deque(maxlen=100)
                }
            
            history = self.agent_history[agent_id]
            
            # Get tasks for this agent
            agent_tasks = [t for t in manager.tasks.values() if t.assigned_agent == agent_id]
            
            # Update history
            for task in agent_tasks:
                if task.status.value == "completed" and task.id not in history["completed_tasks"]:
                    history["completed_tasks"].append(task.id)
                    if task.started_at and task.completed_at:
                        started = datetime.fromisoformat(task.started_at)
                        completed = datetime.fromisoformat(task.completed_at)
                        duration = (completed - started).total_seconds()
                        history["task_durations"].append(duration)
                
                elif task.status.value == "failed" and task.id not in history["failed_tasks"]:
                    history["failed_tasks"].append(task.id)
            
            # Calculate metrics
            tasks_completed = len(history["completed_tasks"])
            tasks_failed = len(history["failed_tasks"])
            
            # Current task duration
            current_task_duration = None
            if agent.current_task:
                current_task = next((t for t in agent_tasks if t.id == agent.current_task), None)
                if current_task and current_task.started_at:
                    started = datetime.fromisoformat(current_task.started_at)
                    current_task_duration = (datetime.now() - started).total_seconds()
            
            # Average task duration
            avg_duration = 0.0
            if history["task_durations"]:
                avg_duration = sum(history["task_durations"]) / len(history["task_durations"])
            
            # Success rate
            total_tasks = tasks_completed + tasks_failed
            success_rate = (tasks_completed / total_tasks) if total_tasks > 0 else 1.0
            
            # Estimate resource usage
            memory_usage = 30.0 if agent.current_task else 10.0  # MB
            cpu_usage = 15.0 if agent.current_task else 2.0  # %
            
            # Performance score (based on success rate and efficiency)
            base_score = success_rate
            if avg_duration > 0:
                # Adjust based on task completion speed (lower is better)
                efficiency_factor = max(0.1, 1.0 - (avg_duration / 300.0))  # 5 minutes baseline
                performance_score = base_score * efficiency_factor
            else:
                performance_score = base_score
            
            metrics = AgentMetrics(
                agent_id=agent_id,
                timestamp=datetime.now().isoformat(),
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
                current_task_duration=current_task_duration,
                average_task_duration=avg_duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                success_rate=success_rate,
                performance_score=performance_score
            )
            
            all_metrics[agent_id] = asdict(metrics)
            
            # Update agent performance score
            agent.performance_score = performance_score
        
        return {"type": "agent", "metrics": all_metrics}


class ModelMetricsCollector(MetricsCollector):
    """ML Model performance metrics collector"""
    
    def __init__(self, model_registry, collection_interval: float = 15.0):
        super().__init__(collection_interval)
        self.model_registry = weakref.ref(model_registry)
        self.inference_history = {}
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect model performance metrics"""
        registry = self.model_registry()
        if not registry:
            return {"type": "model", "metrics": {}}
        
        all_metrics = {}
        
        for model_id, model_info in registry.models.items():
            # Initialize history if needed
            if model_id not in self.inference_history:
                self.inference_history[model_id] = {
                    "inference_times": deque(maxlen=1000),
                    "inference_count": 0,
                    "last_update": datetime.now()
                }
            
            history = self.inference_history[model_id]
            
            # Simulate collecting model metrics (in real implementation, 
            # this would integrate with actual model monitoring)
            import random
            
            # Simulate inference metrics
            inference_count = history["inference_count"] + random.randint(0, 50)
            history["inference_count"] = inference_count
            
            # Simulate inference times
            new_inference_time = random.uniform(10, 100)  # ms
            history["inference_times"].append(new_inference_time)
            
            avg_inference_time = sum(history["inference_times"]) / len(history["inference_times"]) if history["inference_times"] else 0
            
            # Simulate resource usage
            memory_usage = random.uniform(100, 500)  # MB
            gpu_usage = random.uniform(20, 80)  # %
            gpu_memory = random.uniform(1000, 4000)  # MB
            
            # Simulate model quality metrics
            accuracy = random.uniform(0.85, 0.98)
            precision = random.uniform(0.80, 0.95)
            recall = random.uniform(0.78, 0.92)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            # Calculate throughput
            time_diff = (datetime.now() - history["last_update"]).total_seconds()
            throughput = len(history["inference_times"]) / max(time_diff, 1.0)
            
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                timestamp=datetime.now().isoformat(),
                inference_count=inference_count,
                average_inference_time_ms=avg_inference_time,
                memory_usage_mb=memory_usage,
                gpu_usage_percent=gpu_usage,
                gpu_memory_mb=gpu_memory,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                batch_size=random.randint(1, 32),
                throughput_inferences_per_second=throughput
            )
            
            all_metrics[model_id] = asdict(metrics)
            history["last_update"] = datetime.now()
        
        return {"type": "model", "metrics": all_metrics}


class AlertSystem:
    """Alert system for monitoring thresholds"""
    
    def __init__(self):
        self.thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "disk_usage": 80.0,
            "error_rate": 0.1,
            "task_failure_rate": 0.2,
            "average_task_duration": 300.0,  # 5 minutes
            "model_inference_time": 200.0,  # 200ms
            "agent_success_rate": 0.8
        }
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.alert_cooldown = 300  # 5 minutes cooldown
        self.last_alerts = {}
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for alerts"""
        self.callbacks.append(callback)
    
    def set_threshold(self, metric: str, value: float):
        """Set alert threshold for a metric"""
        self.thresholds[metric] = value
    
    async def check_system_metrics(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_usage"]:
            alerts.append({
                "type": "system",
                "metric": "cpu_usage",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_usage"],
                "severity": "high" if metrics.cpu_percent > 95 else "medium",
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%"
            })
        
        if metrics.memory_percent > self.thresholds["memory_usage"]:
            alerts.append({
                "type": "system",
                "metric": "memory_usage",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_usage"],
                "severity": "high" if metrics.memory_percent > 95 else "medium",
                "message": f"High memory usage: {metrics.memory_percent:.1f}%"
            })
        
        if metrics.disk_usage_percent > self.thresholds["disk_usage"]:
            alerts.append({
                "type": "system",
                "metric": "disk_usage",
                "value": metrics.disk_usage_percent,
                "threshold": self.thresholds["disk_usage"],
                "severity": "high" if metrics.disk_usage_percent > 95 else "medium",
                "message": f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            })
        
        await self._send_alerts(alerts)
    
    async def check_pipeline_metrics(self, metrics: PipelineMetrics):
        """Check pipeline metrics against thresholds"""
        alerts = []
        
        if metrics.error_rate > self.thresholds["error_rate"]:
            alerts.append({
                "type": "pipeline",
                "pipeline_id": metrics.pipeline_id,
                "metric": "error_rate",
                "value": metrics.error_rate,
                "threshold": self.thresholds["error_rate"],
                "severity": "high" if metrics.error_rate > 0.5 else "medium",
                "message": f"High error rate in pipeline {metrics.pipeline_id}: {metrics.error_rate:.1%}"
            })
        
        if metrics.average_task_duration > self.thresholds["average_task_duration"]:
            alerts.append({
                "type": "pipeline",
                "pipeline_id": metrics.pipeline_id,
                "metric": "average_task_duration",
                "value": metrics.average_task_duration,
                "threshold": self.thresholds["average_task_duration"],
                "severity": "medium",
                "message": f"Slow task execution in pipeline {metrics.pipeline_id}: {metrics.average_task_duration:.1f}s"
            })
        
        await self._send_alerts(alerts)
    
    async def check_agent_metrics(self, metrics: AgentMetrics):
        """Check agent metrics against thresholds"""
        alerts = []
        
        if metrics.success_rate < self.thresholds["agent_success_rate"]:
            alerts.append({
                "type": "agent",
                "agent_id": metrics.agent_id,
                "metric": "success_rate",
                "value": metrics.success_rate,
                "threshold": self.thresholds["agent_success_rate"],
                "severity": "high" if metrics.success_rate < 0.5 else "medium",
                "message": f"Low success rate for agent {metrics.agent_id}: {metrics.success_rate:.1%}"
            })
        
        await self._send_alerts(alerts)
    
    async def check_model_metrics(self, metrics: ModelPerformanceMetrics):
        """Check model metrics against thresholds"""
        alerts = []
        
        if metrics.average_inference_time_ms > self.thresholds["model_inference_time"]:
            alerts.append({
                "type": "model",
                "model_id": metrics.model_id,
                "metric": "inference_time",
                "value": metrics.average_inference_time_ms,
                "threshold": self.thresholds["model_inference_time"],
                "severity": "medium",
                "message": f"Slow inference for model {metrics.model_id}: {metrics.average_inference_time_ms:.1f}ms"
            })
        
        if metrics.accuracy and metrics.accuracy < 0.8:
            alerts.append({
                "type": "model",
                "model_id": metrics.model_id,
                "metric": "accuracy",
                "value": metrics.accuracy,
                "threshold": 0.8,
                "severity": "high",
                "message": f"Low accuracy for model {metrics.model_id}: {metrics.accuracy:.3f}"
            })
        
        await self._send_alerts(alerts)
    
    async def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts to registered callbacks"""
        now = datetime.now()
        
        for alert in alerts:
            alert_key = f"{alert['type']}:{alert['metric']}"
            
            # Check cooldown
            if alert_key in self.last_alerts:
                time_diff = (now - self.last_alerts[alert_key]).total_seconds()
                if time_diff < self.alert_cooldown:
                    continue
            
            # Send alert
            alert["timestamp"] = now.isoformat()
            alert["alert_id"] = f"alert_{int(time.time())}_{hash(alert_key) % 10000}"
            
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logging.error(f"Alert callback error: {e}")
            
            # Update cooldown
            self.last_alerts[alert_key] = now


class MCPMonitoringService:
    """Main monitoring service integrating with MCP"""
    
    def __init__(self, mcp_server=None):
        self.mcp_server = weakref.ref(mcp_server) if mcp_server else None
        
        # Collectors
        self.system_collector = SystemMetricsCollector()
        self.pipeline_collector = None
        self.agent_collector = None
        self.model_collector = None
        
        # Alert system
        self.alert_system = AlertSystem()
        self.alert_system.add_alert_callback(self._handle_alert)
        
        # Storage
        self.metrics_history = {
            "system": deque(maxlen=1000),
            "pipeline": deque(maxlen=1000),
            "agent": deque(maxlen=1000),
            "model": deque(maxlen=1000)
        }
        
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def initialize_collectors(self, pipeline_manager=None, agent_manager=None, model_registry=None):
        """Initialize collectors with managers"""
        if pipeline_manager:
            self.pipeline_collector = PipelineMetricsCollector(pipeline_manager)
            self.pipeline_collector.add_callback(self._handle_pipeline_metrics)
        
        if agent_manager:
            self.agent_collector = AgentMetricsCollector(agent_manager)
            self.agent_collector.add_callback(self._handle_agent_metrics)
        
        if model_registry:
            self.model_collector = ModelMetricsCollector(model_registry)
            self.model_collector.add_callback(self._handle_model_metrics)
        
        # Add system metrics callback
        self.system_collector.add_callback(self._handle_system_metrics)
    
    async def start(self):
        """Start monitoring service"""
        self.running = True
        self.logger.info("Starting MCP monitoring service")
        
        # Start collectors
        tasks = [self.system_collector.start()]
        
        if self.pipeline_collector:
            tasks.append(self.pipeline_collector.start())
        
        if self.agent_collector:
            tasks.append(self.agent_collector.start())
        
        if self.model_collector:
            tasks.append(self.model_collector.start())
        
        # Start cleanup task
        tasks.append(self._cleanup_task())
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop(self):
        """Stop monitoring service"""
        self.running = False
        
        # Stop collectors
        self.system_collector.stop()
        if self.pipeline_collector:
            self.pipeline_collector.stop()
        if self.agent_collector:
            self.agent_collector.stop()
        if self.model_collector:
            self.model_collector.stop()
        
        self.logger.info("MCP monitoring service stopped")
    
    async def _handle_system_metrics(self, data: Dict[str, Any]):
        """Handle system metrics update"""
        if data["type"] == "system":
            metrics = SystemMetrics(**data["metrics"])
            self.metrics_history["system"].append(metrics)
            
            # Check alerts
            await self.alert_system.check_system_metrics(metrics)
            
            # Broadcast via MCP if available
            if self.mcp_server:
                server = self.mcp_server()
                if server:
                    await server._broadcast_notification("monitoring/system_metrics", asdict(metrics))
    
    async def _handle_pipeline_metrics(self, data: Dict[str, Any]):
        """Handle pipeline metrics update"""
        if data["type"] == "pipeline":
            for pipeline_id, metrics_data in data["metrics"].items():
                metrics = PipelineMetrics(**metrics_data)
                self.metrics_history["pipeline"].append(metrics)
                
                # Check alerts
                await self.alert_system.check_pipeline_metrics(metrics)
            
            # Broadcast via MCP
            if self.mcp_server:
                server = self.mcp_server()
                if server:
                    await server._broadcast_notification("monitoring/pipeline_metrics", data["metrics"])
    
    async def _handle_agent_metrics(self, data: Dict[str, Any]):
        """Handle agent metrics update"""
        if data["type"] == "agent":
            for agent_id, metrics_data in data["metrics"].items():
                metrics = AgentMetrics(**metrics_data)
                self.metrics_history["agent"].append(metrics)
                
                # Check alerts
                await self.alert_system.check_agent_metrics(metrics)
            
            # Broadcast via MCP
            if self.mcp_server:
                server = self.mcp_server()
                if server:
                    await server._broadcast_notification("monitoring/agent_metrics", data["metrics"])
    
    async def _handle_model_metrics(self, data: Dict[str, Any]):
        """Handle model metrics update"""
        if data["type"] == "model":
            for model_id, metrics_data in data["metrics"].items():
                metrics = ModelPerformanceMetrics(**metrics_data)
                self.metrics_history["model"].append(metrics)
                
                # Check alerts
                await self.alert_system.check_model_metrics(metrics)
            
            # Broadcast via MCP
            if self.mcp_server:
                server = self.mcp_server()
                if server:
                    await server._broadcast_notification("monitoring/model_metrics", data["metrics"])
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle alert"""
        self.logger.warning(f"Alert: {alert['message']}")
        
        # Broadcast alert via MCP
        if self.mcp_server:
            server = self.mcp_server()
            if server:
                await server._broadcast_notification("monitoring/alert", alert)
    
    async def _cleanup_task(self):
        """Periodic cleanup of old metrics"""
        while self.running:
            try:
                # Cleanup is handled by deque maxlen, but we could add
                # additional cleanup logic here if needed
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        result = {}
        
        if self.metrics_history["system"]:
            result["system"] = asdict(self.metrics_history["system"][-1])
        
        if self.metrics_history["pipeline"]:
            result["pipeline"] = asdict(self.metrics_history["pipeline"][-1])
        
        if self.metrics_history["agent"]:
            result["agent"] = asdict(self.metrics_history["agent"][-1])
        
        if self.metrics_history["model"]:
            result["model"] = asdict(self.metrics_history["model"][-1])
        
        return result
    
    def get_metrics_history(self, metric_type: str, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for specified duration"""
        if metric_type not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        history = self.metrics_history[metric_type]
        
        return [
            asdict(m) for m in history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
    
    def set_alert_threshold(self, metric: str, value: float):
        """Set alert threshold"""
        self.alert_system.set_threshold(metric, value)


# Example usage
async def main():
    """Example usage of MCP monitoring service"""
    logging.basicConfig(level=logging.INFO)
    
    # Create a mock pipeline manager for testing
    class MockPipelineManager:
        def __init__(self):
            self.pipelines = {}
            self.tasks = {}
    
    class MockAgentManager:
        def __init__(self):
            self.agents = {}
            self.tasks = {}
    
    class MockModelRegistry:
        def __init__(self):
            self.models = {"model-1": {"name": "Test Model"}}
    
    # Initialize monitoring service
    monitoring = MCPMonitoringService()
    
    # Initialize with mock managers
    pipeline_mgr = MockPipelineManager()
    agent_mgr = MockAgentManager()
    model_registry = MockModelRegistry()
    
    monitoring.initialize_collectors(pipeline_mgr, agent_mgr, model_registry)
    
    try:
        # Start monitoring
        await asyncio.wait_for(monitoring.start(), timeout=10)
    except asyncio.TimeoutError:
        print("Monitoring service running...")
    except KeyboardInterrupt:
        print("Stopping monitoring service...")
    finally:
        monitoring.stop()


if __name__ == "__main__":
    asyncio.run(main())