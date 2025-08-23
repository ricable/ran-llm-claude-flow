#!/usr/bin/env python3
"""Metrics Collection System for RAN LLM Claude Flow Python ML Engine

Comprehensive metrics collection for the 45GB MLX-optimized Python ML component.
Targets: MLX GPU utilization, neural model performance, quality scoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, start_http_server
)
import psutil
import threading
from contextlib import contextmanager

# MLX imports for GPU monitoring
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX not available - GPU metrics will be disabled")


@dataclass
class MetricsConfig:
    """Configuration for metrics collection system"""
    collection_interval: float = 1.0  # seconds
    http_port: int = 8081
    enable_gpu_metrics: bool = True
    enable_quality_metrics: bool = True
    enable_neural_metrics: bool = True
    max_metric_age: float = 300.0  # 5 minutes
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)


class MLMetricsCollector:
    """Comprehensive metrics collector for Python ML Engine"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = config.registry
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # Performance tracking
        self._start_time = time.time()
        self._document_count = 0
        self._quality_scores: List[float] = []
        
        logging.info(f"Metrics collector initialized on port {config.http_port}")
    
    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics"""
        
        # Document processing metrics
        self.documents_processed = Counter(
            'documents_processed_total',
            'Total number of documents processed',
            ['component', 'stage', 'model'],
            registry=self.registry
        )
        
        self.document_processing_duration = Histogram(
            'document_processing_duration_seconds',
            'Time spent processing documents',
            ['stage', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Quality metrics
        self.document_quality_score = Gauge(
            'document_quality_score',
            'Current document quality score (0-1)',
            ['model', 'document_type'],
            registry=self.registry
        )
        
        self.quality_score_histogram = Histogram(
            'quality_score_distribution',
            'Distribution of quality scores',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=self.registry
        )
        
        # MLX GPU metrics (if available)
        if MLX_AVAILABLE and self.config.enable_gpu_metrics:
            self.mlx_gpu_utilization = Gauge(
                'mlx_gpu_utilization',
                'MLX GPU utilization percentage (0-1)',
                registry=self.registry
            )
            
            self.mlx_gpu_memory_used = Gauge(
                'mlx_gpu_memory_used_bytes',
                'MLX GPU memory usage in bytes',
                registry=self.registry
            )
            
            self.mlx_gpu_memory_total = Gauge(
                'mlx_gpu_memory_total_bytes',
                'Total MLX GPU memory in bytes',
                registry=self.registry
            )
        
        # Neural model metrics
        self.neural_model_confidence = Gauge(
            'neural_model_confidence',
            'Current neural model confidence score',
            ['model'],
            registry=self.registry
        )
        
        self.neural_predictions = Counter(
            'neural_predictions_total',
            'Total neural model predictions',
            ['model', 'confidence_level'],
            registry=self.registry
        )
        
        self.neural_inference_duration = Histogram(
            'neural_inference_duration_seconds',
            'Neural model inference time',
            ['model'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # System resource metrics
        self.python_ml_memory_bytes = Gauge(
            'python_ml_memory_bytes',
            'Python ML engine memory usage in bytes',
            registry=self.registry
        )
        
        self.python_ml_cpu_utilization = Gauge(
            'python_ml_cpu_utilization',
            'Python ML engine CPU utilization',
            registry=self.registry
        )
        
        # IPC communication metrics
        self.ipc_messages_sent = Counter(
            'ipc_messages_sent_total',
            'Total IPC messages sent to Rust core',
            ['message_type'],
            registry=self.registry
        )
        
        self.ipc_messages_received = Counter(
            'ipc_messages_received_total', 
            'Total IPC messages received from Rust core',
            ['message_type'],
            registry=self.registry
        )
        
        self.ipc_latency = Histogram(
            'ipc_latency_seconds',
            'IPC communication latency',
            ['direction'],
            buckets=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
            registry=self.registry
        )
        
        # Model loading and switching metrics
        self.model_switches = Counter(
            'model_switches_total',
            'Total number of model switches',
            ['from_model', 'to_model'],
            registry=self.registry
        )
        
        self.model_load_duration = Histogram(
            'model_load_duration_seconds',
            'Time to load neural models',
            ['model', 'size'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )
    
    @contextmanager
    def track_processing_time(self, stage: str, model: str):
        """Context manager for tracking document processing time"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.document_processing_duration.labels(
                stage=stage, model=model
            ).observe(duration)
    
    @contextmanager
    def track_neural_inference(self, model: str):
        """Context manager for tracking neural inference time"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.neural_inference_duration.labels(model=model).observe(duration)
    
    @contextmanager
    def track_ipc_latency(self, direction: str):
        """Context manager for tracking IPC latency"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.ipc_latency.labels(direction=direction).observe(duration)
    
    def record_document_processed(self, component: str, stage: str, model: str) -> None:
        """Record a processed document"""
        self.documents_processed.labels(
            component=component, stage=stage, model=model
        ).inc()
        self._document_count += 1
    
    def record_quality_score(self, score: float, model: str, document_type: str) -> None:
        """Record document quality score"""
        self.document_quality_score.labels(
            model=model, document_type=document_type
        ).set(score)
        self.quality_score_histogram.observe(score)
        self._quality_scores.append(score)
        
        # Keep only recent scores for moving average
        if len(self._quality_scores) > 100:
            self._quality_scores = self._quality_scores[-100:]
    
    def record_neural_prediction(self, model: str, confidence: float) -> None:
        """Record neural model prediction"""
        self.neural_model_confidence.labels(model=model).set(confidence)
        
        # Categorize confidence levels
        if confidence >= 0.9:
            confidence_level = "high"
        elif confidence >= 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        self.neural_predictions.labels(
            model=model, confidence_level=confidence_level
        ).inc()
    
    def record_model_switch(self, from_model: str, to_model: str, load_time: float) -> None:
        """Record model switch event"""
        self.model_switches.labels(
            from_model=from_model, to_model=to_model
        ).inc()
        
        # Estimate model size category
        if "1.7b" in to_model.lower():
            size_category = "small"
        elif "7b" in to_model.lower():
            size_category = "medium"
        elif "30b" in to_model.lower():
            size_category = "large"
        else:
            size_category = "unknown"
        
        self.model_load_duration.labels(
            model=to_model, size=size_category
        ).observe(load_time)
    
    def record_ipc_message(self, direction: str, message_type: str) -> None:
        """Record IPC message sent/received"""
        if direction == "sent":
            self.ipc_messages_sent.labels(message_type=message_type).inc()
        elif direction == "received":
            self.ipc_messages_received.labels(message_type=message_type).inc()
    
    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics"""
        process = psutil.Process()
        
        # Memory usage
        memory_info = process.memory_info()
        self.python_ml_memory_bytes.set(memory_info.rss)
        
        # CPU utilization
        cpu_percent = process.cpu_percent()
        self.python_ml_cpu_utilization.set(cpu_percent / 100.0)
    
    def _collect_mlx_metrics(self) -> None:
        """Collect MLX GPU metrics if available"""
        if not MLX_AVAILABLE or not self.config.enable_gpu_metrics:
            return
        
        try:
            # Get GPU memory usage
            gpu_memory = mx.metal.get_memory_info()
            if gpu_memory:
                used_memory = gpu_memory.get('current', 0)
                peak_memory = gpu_memory.get('peak', 0)
                
                self.mlx_gpu_memory_used.set(used_memory)
                # Estimate total as peak + some headroom
                estimated_total = max(peak_memory * 1.2, used_memory * 2)
                self.mlx_gpu_memory_total.set(estimated_total)
                
                # Calculate utilization
                utilization = used_memory / estimated_total if estimated_total > 0 else 0
                self.mlx_gpu_utilization.set(utilization)
            
        except Exception as e:
            logging.warning(f"Failed to collect MLX metrics: {e}")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop"""
        logging.info("Starting metrics collection loop")
        
        while self._running:
            try:
                self._collect_system_metrics()
                self._collect_mlx_metrics()
                
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")
                time.sleep(1.0)
        
        logging.info("Metrics collection loop stopped")
    
    def start(self) -> None:
        """Start the metrics collection system"""
        if self._running:
            logging.warning("Metrics collector already running")
            return
        
        self._running = True
        
        # Start HTTP server for Prometheus scraping
        start_http_server(self.config.http_port, registry=self.registry)
        logging.info(f"Prometheus HTTP server started on port {self.config.http_port}")
        
        # Start metrics collection thread
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        logging.info("Metrics collector started")
    
    def stop(self) -> None:
        """Stop the metrics collection system"""
        if not self._running:
            return
        
        self._running = False
        
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        
        logging.info("Metrics collector stopped")
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        runtime = time.time() - self._start_time
        docs_per_hour = (self._document_count / runtime) * 3600 if runtime > 0 else 0
        avg_quality = sum(self._quality_scores) / len(self._quality_scores) if self._quality_scores else 0
        
        return {
            "runtime_seconds": runtime,
            "documents_processed": self._document_count,
            "documents_per_hour": docs_per_hour,
            "average_quality_score": avg_quality,
            "recent_quality_scores": self._quality_scores[-10:],
            "performance_target_met": docs_per_hour >= 25.0,
            "quality_target_met": avg_quality >= 0.75
        }


# Global metrics collector instance
_metrics_collector: Optional[MLMetricsCollector] = None


def init_metrics(config: Optional[MetricsConfig] = None) -> MLMetricsCollector:
    """Initialize global metrics collector"""
    global _metrics_collector
    
    if _metrics_collector is not None:
        logging.warning("Metrics collector already initialized")
        return _metrics_collector
    
    if config is None:
        config = MetricsConfig()
    
    _metrics_collector = MLMetricsCollector(config)
    _metrics_collector.start()
    
    return _metrics_collector


def get_metrics_collector() -> Optional[MLMetricsCollector]:
    """Get the global metrics collector instance"""
    return _metrics_collector


def shutdown_metrics() -> None:
    """Shutdown the global metrics collector"""
    global _metrics_collector
    
    if _metrics_collector:
        _metrics_collector.stop()
        _metrics_collector = None


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    config = MetricsConfig(
        collection_interval=2.0,
        http_port=8081
    )
    
    collector = init_metrics(config)
    
    # Simulate some metrics
    import random
    
    async def simulate_processing():
        """Simulate document processing for testing"""
        models = ["qwen3-1.7b", "qwen3-7b", "qwen3-30b"]
        
        for i in range(100):
            model = random.choice(models)
            
            # Simulate processing time
            with collector.track_processing_time("semantic_analysis", model):
                await asyncio.sleep(random.uniform(0.1, 2.0))
            
            # Record completion
            collector.record_document_processed("python_ml", "semantic_analysis", model)
            
            # Record quality score
            quality = random.uniform(0.6, 0.95)
            collector.record_quality_score(quality, model, "technical_doc")
            
            # Record neural prediction
            confidence = random.uniform(0.5, 0.98)
            collector.record_neural_prediction(model, confidence)
            
            await asyncio.sleep(0.5)
    
    try:
        # Run simulation
        asyncio.run(simulate_processing())
        
        # Print performance summary
        summary = collector.get_performance_summary()
        print("\nPerformance Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Keep running for metrics collection
        logging.info("Metrics server running. Check http://localhost:8081/metrics")
        input("Press Enter to stop...")
        
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_metrics()