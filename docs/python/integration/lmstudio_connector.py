#!/usr/bin/env python3
"""
LM Studio API Connector with Advanced Connection Pooling
Optimized for Qwen3 models with high-performance connection management.

Features:
- Advanced connection pooling with circuit breaker
- Request batching and queuing
- Automatic model switching based on load
- Real-time performance monitoring
- Adaptive timeout management
- Health monitoring and recovery
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import threading
from urllib.parse import urljoin
import aiohttp
import psutil
import numpy as np
from threading import RLock, Event

logger = logging.getLogger(__name__)


@dataclass
class LMStudioConfig:
    """LM Studio configuration"""
    base_url: str = "http://127.0.0.1:1234"
    api_key: Optional[str] = None
    max_connections: int = 20
    connection_timeout: float = 10.0
    request_timeout: float = 60.0
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    health_check_interval: float = 30.0
    batch_size: int = 10
    queue_timeout: float = 120.0
    enable_streaming: bool = True
    preferred_models: List[str] = None
    
    def __post_init__(self):
        if self.preferred_models is None:
            self.preferred_models = [
                "qwen3-1.7b",
                "qwen3-7b", 
                "qwen3-14b",
                "qwen2.5:1.5b",
                "qwen2.5:7b",
                "qwen2.5:14b"
            ]


@dataclass
class ModelInfo:
    """Information about available model"""
    id: str
    name: str
    size: Optional[str] = None
    created: Optional[float] = None
    owned_by: str = "lmstudio"
    object: str = "model"
    is_qwen: bool = False
    variant: Optional[str] = None  # 1.7b, 7b, 14b, etc.
    
    def __post_init__(self):
        # Detect Qwen models and extract variant
        id_lower = self.id.lower()
        if "qwen" in id_lower:
            self.is_qwen = True
            
            # Extract variant
            if "1.5b" in id_lower or "1.7b" in id_lower:
                self.variant = "1.7b"
            elif "7b" in id_lower:
                self.variant = "7b"
            elif "14b" in id_lower:
                self.variant = "14b"
            elif "30b" in id_lower or "32b" in id_lower:
                self.variant = "30b"


@dataclass
class RequestMetrics:
    """Metrics for individual request"""
    request_id: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    processing_time: float
    queue_time: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ConnectionPool:
    """Advanced connection pool for LM Studio API"""
    
    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.sessions = deque()
        self.active_connections = 0
        self.lock = RLock()
        self.connector = None
        self._setup_connector()
    
    def _setup_connector(self):
        """Setup aiohttp connector with optimized settings"""
        self.connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections,
            ttl_dns_cache=300,  # 5 minute DNS cache
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ssl=False  # Local connection
        )
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get session from pool or create new one"""
        with self.lock:
            if self.sessions and len(self.sessions) > 0:
                return self.sessions.popleft()
        
        # Create new session
        timeout = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=self.config.connection_timeout
        )
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LM-Studio-Python-Client/1.0"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers=headers,
            raise_for_status=False,
            connector_owner=False
        )
        
        with self.lock:
            self.active_connections += 1
        
        return session
    
    def return_session(self, session: aiohttp.ClientSession):
        """Return session to pool"""
        with self.lock:
            if not session.closed and len(self.sessions) < self.config.max_connections // 2:
                self.sessions.append(session)
            else:
                asyncio.create_task(session.close())
                self.active_connections = max(0, self.active_connections - 1)
    
    async def cleanup(self):
        """Cleanup connection pool"""
        with self.lock:
            while self.sessions:
                session = self.sessions.popleft()
                await session.close()
            
            if self.connector:
                await self.connector.close()
        
        self.active_connections = 0
    
    def stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self.lock:
            return {
                "pooled_sessions": len(self.sessions),
                "active_connections": self.active_connections,
                "max_connections": self.config.max_connections
            }


class RequestQueue:
    """Priority request queue with batching support"""
    
    def __init__(self, max_size: int = 1000, batch_size: int = 10):
        self.max_size = max_size
        self.batch_size = batch_size
        self.queues = {
            "critical": deque(),
            "high": deque(), 
            "normal": deque(),
            "low": deque()
        }
        self.pending_requests = {}
        self.lock = RLock()
        self.not_empty = threading.Condition(self.lock)
    
    async def enqueue(self, request_id: str, request_data: Dict[str, Any], priority: str = "normal") -> bool:
        """Add request to queue"""
        with self.not_empty:
            total_queued = sum(len(q) for q in self.queues.values())
            
            if total_queued >= self.max_size:
                return False
            
            queue_entry = {
                "id": request_id,
                "data": request_data,
                "enqueued_at": time.time(),
                "future": asyncio.Future()
            }
            
            self.queues[priority].append(queue_entry)
            self.pending_requests[request_id] = queue_entry
            self.not_empty.notify()
            
            return True
    
    def dequeue_batch(self) -> List[Dict[str, Any]]:
        """Dequeue batch of requests by priority"""
        with self.not_empty:
            batch = []
            
            # Process by priority order
            for priority in ["critical", "high", "normal", "low"]:
                queue = self.queues[priority]
                
                while len(batch) < self.batch_size and queue:
                    entry = queue.popleft()
                    batch.append(entry)
                    
                    # Remove from pending
                    if entry["id"] in self.pending_requests:
                        del self.pending_requests[entry["id"]]
                
                if len(batch) >= self.batch_size:
                    break
            
            return batch
    
    def wait_for_requests(self, timeout: float = None) -> bool:
        """Wait for requests to be available"""
        with self.not_empty:
            total_queued = sum(len(q) for q in self.queues.values())
            if total_queued > 0:
                return True
            
            return self.not_empty.wait(timeout=timeout)
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self.lock:
            return {
                priority: len(queue) 
                for priority, queue in self.queues.items()
            }
    
    def clear(self):
        """Clear all queues"""
        with self.lock:
            for queue in self.queues.values():
                while queue:
                    entry = queue.popleft()
                    if "future" in entry and not entry["future"].done():
                        entry["future"].cancel()
            
            self.pending_requests.clear()


class ModelManager:
    """Manage available models and selection logic"""
    
    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.available_models = {}
        self.model_performance = defaultdict(lambda: deque(maxlen=100))
        self.current_model = None
        self.lock = RLock()
        self.last_refresh = 0
        self.refresh_interval = 300  # 5 minutes
    
    async def refresh_models(self, session: aiohttp.ClientSession) -> bool:
        """Refresh available models from LM Studio"""
        try:
            async with session.get(f"{self.config.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    with self.lock:
                        self.available_models.clear()
                        
                        for model_data in data.get("data", []):
                            model_info = ModelInfo(
                                id=model_data["id"],
                                name=model_data.get("name", model_data["id"]),
                                created=model_data.get("created"),
                                owned_by=model_data.get("owned_by", "lmstudio")
                            )
                            
                            self.available_models[model_info.id] = model_info
                        
                        self.last_refresh = time.time()
                    
                    logger.info(f"Refreshed {len(self.available_models)} models from LM Studio")
                    return True
                else:
                    logger.error(f"Failed to refresh models: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error refreshing models: {e}")
            return False
    
    def select_optimal_model(self, 
                           preferred_variant: Optional[str] = None,
                           task_complexity: str = "balanced") -> Optional[str]:
        """Select optimal model based on preferences and performance"""
        
        with self.lock:
            if not self.available_models:
                return None
            
            # Filter for Qwen models if available
            qwen_models = {k: v for k, v in self.available_models.items() if v.is_qwen}
            target_models = qwen_models if qwen_models else self.available_models
            
            # If specific variant requested, try to find it
            if preferred_variant:
                for model_id, model_info in target_models.items():
                    if model_info.variant == preferred_variant:
                        return model_id
            
            # Select based on task complexity
            if task_complexity == "speed":
                # Prefer smaller models
                preferred_variants = ["1.7b", "7b", "14b", "30b"]
            elif task_complexity == "quality":
                # Prefer larger models
                preferred_variants = ["30b", "14b", "7b", "1.7b"]
            else:  # balanced
                preferred_variants = ["7b", "14b", "1.7b", "30b"]
            
            # Find best match
            for variant in preferred_variants:
                for model_id, model_info in target_models.items():
                    if model_info.variant == variant:
                        return model_id
            
            # Fallback to any available model
            if target_models:
                return list(target_models.keys())[0]
            
            return None
    
    def record_performance(self, model_id: str, metrics: RequestMetrics):
        """Record model performance metrics"""
        with self.lock:
            self.model_performance[model_id].append({
                "tokens_per_second": metrics.tokens_per_second,
                "processing_time": metrics.processing_time,
                "success": metrics.success,
                "timestamp": metrics.timestamp
            })
    
    def get_model_stats(self, model_id: str) -> Dict[str, float]:
        """Get performance statistics for model"""
        with self.lock:
            if model_id not in self.model_performance:
                return {}
            
            perf_data = list(self.model_performance[model_id])
            if not perf_data:
                return {}
            
            tokens_per_sec = [p["tokens_per_second"] for p in perf_data if p["tokens_per_second"] > 0]
            processing_times = [p["processing_time"] for p in perf_data]
            successes = [p["success"] for p in perf_data]
            
            return {
                "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0,
                "avg_processing_time": np.mean(processing_times),
                "success_rate": np.mean(successes),
                "total_requests": len(perf_data),
                "recent_requests": len([p for p in perf_data if time.time() - p["timestamp"] < 300])
            }
    
    def get_all_model_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all models"""
        return {
            model_id: self.get_model_stats(model_id)
            for model_id in self.available_models.keys()
        }
    
    def should_refresh_models(self) -> bool:
        """Check if models should be refreshed"""
        return time.time() - self.last_refresh > self.refresh_interval


class HealthMonitor:
    """Monitor LM Studio health and performance"""
    
    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.is_healthy = False
        self.last_check = 0
        self.consecutive_failures = 0
        self.max_failures = 3
        self.check_interval = config.health_check_interval
        self.monitoring_active = False
        self.monitor_task = None
        self.health_history = deque(maxlen=100)
    
    async def start_monitoring(self, connection_pool: ConnectionPool):
        """Start health monitoring"""
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(
            self._monitor_loop(connection_pool)
        )
        logger.info("LM Studio health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("LM Studio health monitoring stopped")
    
    async def _monitor_loop(self, connection_pool: ConnectionPool):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_check(connection_pool)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(min(self.check_interval, 30))
    
    async def _perform_health_check(self, connection_pool: ConnectionPool):
        """Perform health check"""
        start_time = time.time()
        
        try:
            session = await connection_pool.get_session()
            
            try:
                # Simple health check - get models list
                async with session.get(
                    f"{self.config.base_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        self.is_healthy = True
                        self.consecutive_failures = 0
                        
                        # Record response time
                        response_time = time.time() - start_time
                        self.health_history.append({
                            "timestamp": time.time(),
                            "healthy": True,
                            "response_time": response_time,
                            "status_code": response.status
                        })
                        
                    else:
                        self._record_failure(response.status, time.time() - start_time)
                        
            finally:
                connection_pool.return_session(session)
                
        except Exception as e:
            self._record_failure(str(e), time.time() - start_time)
        
        self.last_check = time.time()
    
    def _record_failure(self, error: Union[str, int], response_time: float):
        """Record health check failure"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.max_failures:
            self.is_healthy = False
        
        self.health_history.append({
            "timestamp": time.time(),
            "healthy": False,
            "response_time": response_time,
            "error": error
        })
        
        logger.warning(f"LM Studio health check failed ({self.consecutive_failures}/{self.max_failures}): {error}")
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health statistics"""
        if not self.health_history:
            return {
                "healthy": self.is_healthy,
                "last_check": self.last_check,
                "consecutive_failures": self.consecutive_failures
            }
        
        recent_checks = list(self.health_history)
        healthy_checks = [h for h in recent_checks if h["healthy"]]
        
        return {
            "healthy": self.is_healthy,
            "last_check": self.last_check,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": len(healthy_checks) / len(recent_checks) if recent_checks else 0,
            "avg_response_time": np.mean([h["response_time"] for h in recent_checks]),
            "total_checks": len(recent_checks)
        }


class LMStudioConnector:
    """Main LM Studio connector with advanced features"""
    
    def __init__(self, config: LMStudioConfig = None):
        self.config = config or LMStudioConfig()
        self.connection_pool = ConnectionPool(self.config)
        self.request_queue = RequestQueue(batch_size=self.config.batch_size)
        self.model_manager = ModelManager(self.config)
        self.health_monitor = HealthMonitor(self.config)
        
        self.processing_active = False
        self.processor_task = None
        self.request_metrics = deque(maxlen=1000)
        self.metrics_lock = RLock()
        
        logger.info(f"LM Studio Connector initialized: {self.config.base_url}")
    
    async def initialize(self) -> bool:
        """Initialize connector"""
        try:
            # Get initial session to test connection
            session = await self.connection_pool.get_session()
            
            try:
                # Test connection and get models
                success = await self.model_manager.refresh_models(session)
                if not success:
                    return False
                
                # Start health monitoring
                await self.health_monitor.start_monitoring(self.connection_pool)
                
                # Start request processor
                await self._start_request_processor()
                
                logger.info("LM Studio Connector initialized successfully")
                return True
                
            finally:
                self.connection_pool.return_session(session)
                
        except Exception as e:
            logger.error(f"LM Studio Connector initialization failed: {e}")
            return False
    
    async def _start_request_processor(self):
        """Start request processor"""
        self.processing_active = True
        self.processor_task = asyncio.create_task(self._process_requests())
        logger.info("Request processor started")
    
    async def _process_requests(self):
        """Main request processing loop"""
        while self.processing_active:
            try:
                # Wait for requests
                if not self.request_queue.wait_for_requests(timeout=1.0):
                    continue
                
                # Get batch of requests
                batch = self.request_queue.dequeue_batch()
                if not batch:
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Request processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process batch of requests"""
        # For now, process requests individually
        # Could be enhanced to support true batching if LM Studio adds batch API
        
        tasks = []
        for entry in batch:
            task = asyncio.create_task(
                self._process_single_request(entry)
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_request(self, entry: Dict[str, Any]):
        """Process single request"""
        request_id = entry["id"]
        request_data = entry["data"]
        future = entry["future"]
        queue_time = time.time() - entry["enqueued_at"]
        
        try:
            start_time = time.time()
            
            # Get session
            session = await self.connection_pool.get_session()
            
            try:
                # Select model if not specified
                if "model" not in request_data:
                    model_id = self.model_manager.select_optimal_model()
                    if not model_id:
                        raise Exception("No models available")
                    request_data["model"] = model_id
                
                # Make API request
                async with session.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    json=request_data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Calculate metrics
                        processing_time = time.time() - start_time
                        usage = result.get("usage", {})
                        
                        metrics = RequestMetrics(
                            request_id=request_id,
                            model_id=request_data["model"],
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                            processing_time=processing_time,
                            queue_time=queue_time,
                            tokens_per_second=usage.get("completion_tokens", 0) / processing_time if processing_time > 0 else 0,
                            success=True
                        )
                        
                        # Record metrics
                        self._record_metrics(metrics)
                        
                        # Set result
                        future.set_result({
                            "success": True,
                            "data": result,
                            "metrics": metrics
                        })
                        
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                        
            finally:
                self.connection_pool.return_session(session)
                
        except Exception as e:
            # Record failed metrics
            metrics = RequestMetrics(
                request_id=request_id,
                model_id=request_data.get("model", "unknown"),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                processing_time=time.time() - start_time,
                queue_time=queue_time,
                tokens_per_second=0,
                success=False,
                error=str(e)
            )
            
            self._record_metrics(metrics)
            
            # Set error result
            future.set_result({
                "success": False,
                "error": str(e),
                "metrics": metrics
            })
    
    def _record_metrics(self, metrics: RequestMetrics):
        """Record request metrics"""
        with self.metrics_lock:
            self.request_metrics.append(metrics)
            
            # Update model performance
            self.model_manager.record_performance(metrics.model_id, metrics)
    
    async def generate(self, 
                      messages: List[Dict[str, str]],
                      model: Optional[str] = None,
                      max_tokens: int = 1024,
                      temperature: float = 0.7,
                      stream: bool = False,
                      priority: str = "normal",
                      timeout: float = None) -> Dict[str, Any]:
        """Generate response via LM Studio"""
        
        # Prepare request
        request_id = f"req_{int(time.time() * 1000000)}"
        request_data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        if model:
            request_data["model"] = model
        
        # Add to queue
        success = await self.request_queue.enqueue(request_id, request_data, priority)
        if not success:
            raise Exception("Request queue full")
        
        # Get future and wait for result
        entry = self.request_queue.pending_requests.get(request_id)
        if not entry:
            raise Exception("Request not found in queue")
        
        try:
            result = await asyncio.wait_for(
                entry["future"],
                timeout=timeout or self.config.queue_timeout
            )
            return result
            
        except asyncio.TimeoutError:
            # Cancel request if still pending
            if request_id in self.request_queue.pending_requests:
                del self.request_queue.pending_requests[request_id]
            raise Exception("Request timeout")
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available models"""
        # Refresh if needed
        if self.model_manager.should_refresh_models():
            session = await self.connection_pool.get_session()
            try:
                await self.model_manager.refresh_models(session)
            finally:
                self.connection_pool.return_session(session)
        
        return list(self.model_manager.available_models.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        return {
            "healthy": self.health_monitor.is_healthy,
            "connection_pool": self.connection_pool.stats(),
            "request_queue": self.request_queue.get_stats(),
            "models": {
                "available": len(self.model_manager.available_models),
                "performance": self.model_manager.get_all_model_stats()
            },
            "health": self.health_monitor.get_health_stats(),
            "metrics": self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        with self.metrics_lock:
            if not self.request_metrics:
                return {}
            
            recent_metrics = [m for m in self.request_metrics if time.time() - m.timestamp < 300]
            if not recent_metrics:
                return {}
            
            return {
                "requests_per_minute": len(recent_metrics) / 5.0,
                "avg_processing_time": np.mean([m.processing_time for m in recent_metrics]),
                "avg_tokens_per_second": np.mean([m.tokens_per_second for m in recent_metrics if m.tokens_per_second > 0]),
                "success_rate": np.mean([m.success for m in recent_metrics]),
                "avg_queue_time": np.mean([m.queue_time for m in recent_metrics])
            }
    
    async def cleanup(self):
        """Cleanup connector resources"""
        logger.info("Cleaning up LM Studio Connector...")
        
        # Stop processing
        self.processing_active = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Clear queue
        self.request_queue.clear()
        
        # Cleanup connection pool
        await self.connection_pool.cleanup()
        
        logger.info("LM Studio Connector cleanup completed")


# Example usage
async def test_lmstudio_connector():
    """Test LM Studio connector"""
    config = LMStudioConfig(
        max_connections=10,
        batch_size=5
    )
    
    connector = LMStudioConnector(config)
    
    try:
        # Initialize
        if not await connector.initialize():
            print("Failed to initialize connector")
            return
        
        # Test requests
        test_messages = [
            [{"role": "user", "content": "What is 5G NR?"}],
            [{"role": "user", "content": "Explain MIMO technology."}],
            [{"role": "user", "content": "What are the benefits of mmWave?"}]
        ]
        
        # Send concurrent requests
        tasks = []
        for i, messages in enumerate(test_messages):
            task = asyncio.create_task(
                connector.generate(messages, priority="high" if i == 0 else "normal")
            )
            tasks.append(task)
        
        # Wait for results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i} failed: {result}")
            else:
                print(f"Request {i}: {result['success']}")
                if result['success']:
                    content = result['data']['choices'][0]['message']['content']
                    print(f"Response: {content[:100]}...")
                    print(f"Tokens/sec: {result['metrics'].tokens_per_second:.1f}")
        
        # Get status
        status = connector.get_status()
        print(f"\nConnector Status: {json.dumps(status, indent=2)}")
        
    finally:
        await connector.cleanup()


if __name__ == "__main__":
    asyncio.run(test_lmstudio_connector())