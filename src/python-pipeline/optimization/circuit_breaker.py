"""
Circuit Breaker Pattern Implementation for Python Pipeline
Provides reliability, fault tolerance, and graceful degradation
Optimized for ML model operations and resource-intensive processing
"""

import asyncio
import time
import threading
import logging
import json
import subprocess
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import statistics
from collections import deque

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"             # Circuit is open, blocking requests
    HALF_OPEN = "half_open"   # Testing if service has recovered

class FailureType(Enum):
    """Types of failures to track"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_ERROR = "model_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5           # Failures before opening circuit
    success_threshold: int = 3           # Successes needed to close circuit
    timeout_seconds: float = 30.0        # Request timeout
    recovery_timeout_seconds: float = 60.0  # Time to wait before trying half-open
    window_size_seconds: float = 300.0   # Sliding window for failure tracking
    max_half_open_requests: int = 3      # Max requests in half-open state
    exponential_backoff: bool = True     # Use exponential backoff for recovery
    max_backoff_seconds: float = 300.0   # Maximum backoff time

@dataclass
class FailureRecord:
    """Record of a failure event"""
    timestamp: float
    failure_type: FailureType
    error_message: str
    duration_seconds: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitMetrics:
    """Metrics tracking for circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    avg_response_time_ms: float = 0.0
    current_failure_rate: float = 0.0
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

class CircuitBreaker:
    """
    Circuit breaker implementation for reliable service protection
    Monitors failures, timeouts, and resource exhaustion
    """
    
    def __init__(self, 
                 name: str, 
                 config: CircuitBreakerConfig = None,
                 fallback_function: Callable = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_function = fallback_function
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # State management
        self.state = CircuitState.CLOSED
        self.state_changed_at = time.time()
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        
        # Failure tracking
        self.failure_history = deque(maxlen=1000)
        self.response_times = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        self.metrics = CircuitMetrics()
        
        # Exponential backoff
        self.backoff_attempts = 0
        self.last_failure_time = 0
        
        self.logger.info(f"CircuitBreaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        Returns result or raises exception based on circuit state
        """
        
        with self.lock:
            self.metrics.total_requests += 1
            
            # Check if circuit allows request
            if not self._should_allow_request():
                if self.fallback_function:
                    self.logger.warning(f"Circuit {self.name} is open, using fallback")
                    return await self._execute_fallback(*args, **kwargs)
                else:
                    raise CircuitOpenException(f"Circuit {self.name} is open")
        
        # Execute the request with monitoring
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Record success
            duration = time.time() - start_time
            self._record_success(duration)
            
            return result
            
        except asyncio.TimeoutError as e:
            # Record timeout failure
            duration = time.time() - start_time
            self._record_failure(FailureType.TIMEOUT, str(e), duration, {"args": args, "kwargs": kwargs})
            raise CircuitBreakerTimeoutException(f"Circuit {self.name} timeout after {duration:.2f}s")
            
        except MemoryError as e:
            # Record memory error
            duration = time.time() - start_time
            self._record_failure(FailureType.MEMORY_ERROR, str(e), duration, {"memory_critical": True})
            raise
            
        except Exception as e:
            # Record general failure
            duration = time.time() - start_time
            failure_type = self._classify_error(e)
            self._record_failure(failure_type, str(e), duration, {"exception_type": type(e).__name__})
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback function"""
        if asyncio.iscoroutinefunction(self.fallback_function):
            return await self.fallback_function(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.fallback_function(*args, **kwargs))
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on circuit state"""
        
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.state_changed_at >= self._get_recovery_timeout():
                self._transition_to_half_open()
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self.half_open_requests < self.config.max_half_open_requests
    
    def _get_recovery_timeout(self) -> float:
        """Get recovery timeout with exponential backoff if enabled"""
        
        if not self.config.exponential_backoff:
            return self.config.recovery_timeout_seconds
        
        # Exponential backoff: 2^attempts * base_timeout
        backoff_multiplier = min(2 ** self.backoff_attempts, self.config.max_backoff_seconds / self.config.recovery_timeout_seconds)
        return min(self.config.recovery_timeout_seconds * backoff_multiplier, self.config.max_backoff_seconds)
    
    def _record_success(self, duration: float):
        """Record successful request"""
        
        with self.lock:
            self.metrics.successful_requests += 1
            self.success_count += 1
            self.response_times.append(duration * 1000)  # Convert to ms
            
            # Update average response time
            if self.response_times:
                self.metrics.avg_response_time_ms = statistics.mean(self.response_times)
            
            # Check if circuit should close
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            # Store metrics
            asyncio.create_task(self._store_metrics())
    
    def _record_failure(self, failure_type: FailureType, error_message: str, duration: float, context: Dict[str, Any] = None):
        """Record failed request"""
        
        with self.lock:
            self.metrics.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if failure_type == FailureType.TIMEOUT:
                self.metrics.timeouts += 1
            
            # Record failure in history
            failure_record = FailureRecord(
                timestamp=time.time(),
                failure_type=failure_type,
                error_message=error_message,
                duration_seconds=duration,
                context=context or {}
            )
            self.failure_history.append(failure_record)
            
            # Update failure rate
            self._update_failure_rate()
            
            # Check if circuit should open
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
                    
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._transition_to_open()
            
            # Store metrics
            asyncio.create_task(self._store_metrics())
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for better tracking"""
        
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureType.TIMEOUT
        elif "memory" in error_str or "out of memory" in error_str:
            return FailureType.MEMORY_ERROR
        elif "model" in error_str or "inference" in error_str:
            return FailureType.MODEL_ERROR
        elif "connection" in error_str or "network" in error_str:
            return FailureType.NETWORK_ERROR
        elif "resource" in error_str or "limit" in error_str:
            return FailureType.RESOURCE_EXHAUSTION
        else:
            return FailureType.EXCEPTION
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state"""
        
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_changed_at = time.time()
        self.metrics.circuit_opens += 1
        self.backoff_attempts += 1
        
        state_change = {
            "from_state": old_state.value,
            "to_state": self.state.value,
            "timestamp": self.state_changed_at,
            "failure_count": self.failure_count,
            "reason": "failure_threshold_reached"
        }
        self.metrics.state_changes.append(state_change)
        
        self.logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state"""
        
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = time.time()
        self.half_open_requests = 0
        self.success_count = 0
        
        state_change = {
            "from_state": old_state.value,
            "to_state": self.state.value,
            "timestamp": self.state_changed_at,
            "recovery_timeout": self._get_recovery_timeout(),
            "reason": "recovery_timeout_reached"
        }
        self.metrics.state_changes.append(state_change)
        
        self.logger.info(f"Circuit {self.name} half-opened for testing")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state"""
        
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_changed_at = time.time()
        self.failure_count = 0
        self.success_count = 0
        self.backoff_attempts = 0  # Reset backoff on successful recovery
        
        state_change = {
            "from_state": old_state.value,
            "to_state": self.state.value,
            "timestamp": self.state_changed_at,
            "success_count": self.success_count,
            "reason": "success_threshold_reached"
        }
        self.metrics.state_changes.append(state_change)
        self.metrics.circuit_closes += 1
        
        self.logger.info(f"Circuit {self.name} closed after {self.success_count} successful requests")
    
    def _update_failure_rate(self):
        """Update current failure rate based on recent history"""
        
        current_time = time.time()
        window_start = current_time - self.config.window_size_seconds
        
        # Count failures in current window
        recent_failures = sum(1 for failure in self.failure_history 
                            if failure.timestamp >= window_start)
        
        # Calculate failure rate
        total_recent_requests = self.metrics.total_requests  # Simplified calculation
        if total_recent_requests > 0:
            self.metrics.current_failure_rate = recent_failures / total_recent_requests
        else:
            self.metrics.current_failure_rate = 0.0
    
    async def _store_metrics(self):
        """Store circuit breaker metrics in coordination memory"""
        try:
            metrics_data = {
                'name': self.name,
                'state': self.state.value,
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_rate': (self.metrics.successful_requests / max(1, self.metrics.total_requests)) * 100,
                'current_failure_rate': self.metrics.current_failure_rate,
                'avg_response_time_ms': self.metrics.avg_response_time_ms,
                'circuit_opens': self.metrics.circuit_opens,
                'circuit_closes': self.metrics.circuit_closes,
                'state_changed_at': self.state_changed_at,
                'failure_count': self.failure_count,
                'timestamp': time.time()
            }
            
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "post-edit",
                "--memory-key", f"python-pipeline/performance/circuit-breaker-{self.name}",
                "--data", json.dumps(metrics_data)
            ], capture_output=True, text=True, check=True)
            
        except Exception as e:
            self.logger.debug(f"Failed to store metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and metrics"""
        
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'state_changed_at': self.state_changed_at,
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests,
                    'success_rate_percent': (self.metrics.successful_requests / max(1, self.metrics.total_requests)) * 100,
                    'current_failure_rate': self.metrics.current_failure_rate,
                    'avg_response_time_ms': self.metrics.avg_response_time_ms,
                    'timeouts': self.metrics.timeouts,
                    'circuit_opens': self.metrics.circuit_opens,
                    'circuit_closes': self.metrics.circuit_closes
                },
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout_seconds': self.config.timeout_seconds,
                    'recovery_timeout_seconds': self.config.recovery_timeout_seconds
                }
            }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        
        with self.lock:
            self.state = CircuitState.CLOSED
            self.state_changed_at = time.time()
            self.failure_count = 0
            self.success_count = 0
            self.half_open_requests = 0
            self.backoff_attempts = 0
            
            self.failure_history.clear()
            self.response_times.clear()
            
            # Reset metrics but keep historical data
            previous_opens = self.metrics.circuit_opens
            previous_closes = self.metrics.circuit_closes
            
            self.metrics = CircuitMetrics()
            self.metrics.circuit_opens = previous_opens
            self.metrics.circuit_closes = previous_closes
            
            self.logger.info(f"Circuit {self.name} reset to initial state")

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services/operations
    Provides centralized monitoring and coordination
    """
    
    def __init__(self):
        self.circuit_breakers = {}
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
    
    def create_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        fallback_function: Callable = None
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker"""
        
        with self.lock:
            if name in self.circuit_breakers:
                raise ValueError(f"Circuit breaker '{name}' already exists")
            
            circuit_breaker = CircuitBreaker(name, config, fallback_function)
            self.circuit_breakers[name] = circuit_breaker
            
            self.logger.info(f"Created circuit breaker '{name}'")
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        
        with self.lock:
            return {name: cb.get_status() for name, cb in self.circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        
        with self.lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker.reset()
            
            self.logger.info("Reset all circuit breakers")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all circuit breakers"""
        
        health_status = {
            'overall_health': 'healthy',
            'circuit_breakers': {},
            'summary': {
                'total_circuits': len(self.circuit_breakers),
                'open_circuits': 0,
                'half_open_circuits': 0,
                'closed_circuits': 0
            }
        }
        
        with self.lock:
            for name, cb in self.circuit_breakers.items():
                status = cb.get_status()
                health_status['circuit_breakers'][name] = status
                
                # Count states
                if status['state'] == 'open':
                    health_status['summary']['open_circuits'] += 1
                elif status['state'] == 'half_open':
                    health_status['summary']['half_open_circuits'] += 1
                else:
                    health_status['summary']['closed_circuits'] += 1
        
        # Determine overall health
        if health_status['summary']['open_circuits'] > 0:
            health_status['overall_health'] = 'degraded'
        
        return health_status

# Custom exceptions

class CircuitOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutException(Exception):
    """Raised when circuit breaker times out"""
    pass

# Global manager instance
circuit_breaker_manager = CircuitBreakerManager()

# Utility functions and decorators

def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig = None,
    fallback: Callable = None
):
    """Decorator to add circuit breaker protection to a function"""
    
    def decorator(func: Callable):
        cb = circuit_breaker_manager.create_circuit_breaker(name, config, fallback)
        
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

def create_model_circuit_breaker(model_name: str) -> CircuitBreaker:
    """Create circuit breaker optimized for ML model operations"""
    
    config = CircuitBreakerConfig(
        failure_threshold=3,           # Models fail faster
        success_threshold=2,           # Recover faster
        timeout_seconds=60.0,          # Longer timeout for model operations
        recovery_timeout_seconds=120.0, # Longer recovery for models
        window_size_seconds=600.0,     # 10-minute window
        exponential_backoff=True,
        max_backoff_seconds=600.0      # Max 10 minutes
    )
    
    return circuit_breaker_manager.create_circuit_breaker(
        f"model_{model_name}",
        config
    )

def create_processing_circuit_breaker(operation_name: str) -> CircuitBreaker:
    """Create circuit breaker optimized for document processing operations"""
    
    config = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=30.0,
        recovery_timeout_seconds=60.0,
        window_size_seconds=300.0,
        exponential_backoff=True
    )
    
    return circuit_breaker_manager.create_circuit_breaker(
        f"processing_{operation_name}",
        config
    )

# Context manager for circuit breaker protection

@asynccontextmanager
async def protected_operation(circuit_name: str, operation: Callable, *args, **kwargs):
    """Context manager for protected operation execution"""
    
    cb = circuit_breaker_manager.get_circuit_breaker(circuit_name)
    if not cb:
        raise ValueError(f"Circuit breaker '{circuit_name}' not found")
    
    try:
        result = await cb.call(operation, *args, **kwargs)
        yield result
    except Exception as e:
        raise e