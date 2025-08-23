/*!
# MCP Error Handling and Fault Tolerance

Comprehensive error handling, circuit breaker patterns, and fault tolerance mechanisms
for the Model Context Protocol implementation.
*/

use crate::{PipelineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Failing, requests rejected
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub success_threshold: u32,
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 3,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    name: String,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure_time: Arc::new(RwLock::new(None)),
            name,
        }
    }

    /// Execute operation through circuit breaker
    pub async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        // Check circuit breaker state
        if !self.can_execute().await? {
            return Err(PipelineError::Mcp(format!(
                "Circuit breaker '{}' is OPEN - rejecting request",
                self.name
            )));
        }

        // Execute operation with timeout
        match tokio::time::timeout(self.config.timeout, operation).await {
            Ok(Ok(result)) => {
                self.record_success().await;
                Ok(result)
            }
            Ok(Err(error)) => {
                self.record_failure().await;
                Err(error)
            }
            Err(_) => {
                let timeout_error = PipelineError::Mcp(format!(
                    "Operation timeout in circuit breaker '{}'",
                    self.name
                ));
                self.record_failure().await;
                Err(timeout_error)
            }
        }
    }

    /// Check if circuit breaker allows execution
    async fn can_execute(&self) -> Result<bool> {
        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitBreakerState::Closed => Ok(true),
            CircuitBreakerState::Open => {
                // Check if recovery timeout has elapsed
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() >= self.config.recovery_timeout {
                        // Transition to half-open
                        *self.state.write().await = CircuitBreakerState::HalfOpen;
                        self.success_count.store(0, Ordering::Relaxed);
                        info!("Circuit breaker '{}' transitioning to HALF_OPEN", self.name);
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            CircuitBreakerState::HalfOpen => Ok(true),
        }
    }

    /// Record successful operation
    async fn record_success(&self) {
        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitBreakerState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if success_count >= self.config.success_threshold.into() {
                    *self.state.write().await = CircuitBreakerState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    info!("Circuit breaker '{}' recovered to CLOSED", self.name);
                }
            }
            _ => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Record failed operation
    async fn record_failure(&self) {
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure_time.write().await = Some(Instant::now());

        let current_state = self.state.read().await.clone();

        if matches!(
            current_state,
            CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen
        ) && failure_count >= self.config.failure_threshold.into()
        {
            *self.state.write().await = CircuitBreakerState::Open;
            warn!(
                "Circuit breaker '{}' opened after {} failures",
                self.name, failure_count
            );
        }
    }

    /// Get current state
    pub async fn get_state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            name: self.name.clone(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            last_failure_time: self.last_failure_time.clone(),
        }
    }
}

/// Circuit breaker metrics
#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    pub name: String,
    pub failure_count: u64,
    pub success_count: u64,
    pub last_failure_time: Arc<RwLock<Option<Instant>>>,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
            jitter: true,
        }
    }
}

/// Retry mechanism with exponential backoff
pub struct RetryMechanism {
    config: RetryConfig,
}

impl RetryMechanism {
    /// Create new retry mechanism
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute operation with retry
    pub async fn execute<F, T, E>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> std::pin::Pin<
            Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send>,
        >,
        E: std::fmt::Display + Send + 'static,
    {
        let mut attempt = 0;
        let mut delay = self.config.initial_delay;

        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if attempt >= self.config.max_attempts {
                        return Err(PipelineError::Mcp(format!(
                            "Operation failed after {} attempts: {}",
                            attempt, error
                        )));
                    }

                    warn!(
                        "Attempt {} failed: {}. Retrying in {:?}",
                        attempt, error, delay
                    );

                    // Sleep with optional jitter
                    if self.config.jitter {
                        let jitter_delay = self.add_jitter(delay);
                        sleep(jitter_delay).await;
                    } else {
                        sleep(delay).await;
                    }

                    // Calculate next delay with exponential backoff
                    delay = std::cmp::min(
                        Duration::from_millis(
                            (delay.as_millis() as f64 * self.config.backoff_factor) as u64,
                        ),
                        self.config.max_delay,
                    );
                }
            }
        }
    }

    /// Add random jitter to delay
    fn add_jitter(&self, delay: Duration) -> Duration {
        let jitter_range = delay.as_millis() as f64 * 0.1; // 10% jitter
        let jitter = (rand::random::<f64>() - 0.5) * jitter_range;
        let jittered_millis = delay.as_millis() as f64 + jitter;
        Duration::from_millis(jittered_millis.max(0.0) as u64)
    }
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub healthy_threshold: u32,
    pub unhealthy_threshold: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            healthy_threshold: 2,
            unhealthy_threshold: 3,
        }
    }
}

/// Health status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub component: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub timestamp: u64,
    pub response_time: Duration,
}

/// Health monitor for component monitoring
pub struct HealthMonitor {
    config: HealthCheckConfig,
    components: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    is_running: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct ComponentHealth {
    status: HealthStatus,
    consecutive_successes: u32,
    consecutive_failures: u32,
    last_check: u64,
    total_checks: u64,
    success_rate: f64,
}

impl HealthMonitor {
    /// Create new health monitor
    pub fn new(config: HealthCheckConfig) -> Self {
        Self {
            config,
            components: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start health monitoring
    pub async fn start(&self) -> Result<()> {
        self.is_running.store(true, Ordering::Relaxed);
        info!("Health monitor started");
        Ok(())
    }

    /// Stop health monitoring
    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, Ordering::Relaxed);
        info!("Health monitor stopped");
        Ok(())
    }

    /// Register component for monitoring
    pub async fn register_component(&self, component_name: String) {
        let component_health = ComponentHealth {
            status: HealthStatus::Unknown,
            consecutive_successes: 0,
            consecutive_failures: 0,
            last_check: current_timestamp(),
            total_checks: 0,
            success_rate: 0.0,
        };

        self.components
            .write()
            .await
            .insert(component_name.clone(), component_health);
        info!(
            "Registered component for health monitoring: {}",
            component_name
        );
    }

    /// Record health check result
    pub async fn record_health_check(&self, result: HealthCheckResult) {
        let mut components = self.components.write().await;

        if let Some(component_health) = components.get_mut(&result.component) {
            component_health.last_check = result.timestamp;
            component_health.total_checks += 1;

            match result.status {
                HealthStatus::Healthy => {
                    component_health.consecutive_successes += 1;
                    component_health.consecutive_failures = 0;

                    // Update status if healthy threshold is met
                    if component_health.consecutive_successes >= self.config.healthy_threshold {
                        component_health.status = HealthStatus::Healthy;
                    }
                }
                HealthStatus::Unhealthy => {
                    component_health.consecutive_failures += 1;
                    component_health.consecutive_successes = 0;

                    // Update status if unhealthy threshold is met
                    if component_health.consecutive_failures >= self.config.unhealthy_threshold {
                        component_health.status = HealthStatus::Unhealthy;
                        warn!(
                            "Component {} marked as unhealthy after {} consecutive failures",
                            result.component, component_health.consecutive_failures
                        );
                    }
                }
                _ => {}
            }

            // Update success rate
            let success_count = component_health.total_checks
                - u64::from(
                    component_health
                        .consecutive_failures
                        .min(component_health.total_checks as u32),
                );
            component_health.success_rate =
                success_count as f64 / component_health.total_checks as f64;

            debug!(
                "Health check recorded for {}: {:?} (success rate: {:.2}%)",
                result.component,
                result.status,
                component_health.success_rate * 100.0
            );
        }
    }

    /// Get component health status
    pub async fn get_component_health(&self, component_name: &str) -> Option<HealthStatus> {
        self.components
            .read()
            .await
            .get(component_name)
            .map(|h| h.status.clone())
    }

    /// Get all component health statuses
    pub async fn get_all_health_statuses(&self) -> HashMap<String, HealthStatus> {
        self.components
            .read()
            .await
            .iter()
            .map(|(name, health)| (name.clone(), health.status.clone()))
            .collect()
    }

    /// Get system health summary
    pub async fn get_system_health(&self) -> SystemHealthSummary {
        let components = self.components.read().await;
        let total_components = components.len();
        let healthy_components = components
            .values()
            .filter(|h| h.status == HealthStatus::Healthy)
            .count();
        let unhealthy_components = components
            .values()
            .filter(|h| h.status == HealthStatus::Unhealthy)
            .count();

        let overall_status = if unhealthy_components > 0 {
            HealthStatus::Unhealthy
        } else if healthy_components == total_components && total_components > 0 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        SystemHealthSummary {
            overall_status,
            total_components,
            healthy_components,
            unhealthy_components,
            timestamp: current_timestamp(),
        }
    }
}

/// System health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthSummary {
    pub overall_status: HealthStatus,
    pub total_components: usize,
    pub healthy_components: usize,
    pub unhealthy_components: usize,
    pub timestamp: u64,
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    Retry(RetryConfig),
    CircuitBreaker(CircuitBreakerConfig),
    Fallback(String), // Fallback service name
    Ignore,
    Escalate,
}

/// Fault tolerance manager
pub struct FaultToleranceManager {
    circuit_breakers: Arc<RwLock<HashMap<String, Arc<CircuitBreaker>>>>,
    retry_mechanisms: Arc<RwLock<HashMap<String, Arc<RetryMechanism>>>>,
    health_monitor: Arc<HealthMonitor>,
    recovery_strategies: Arc<RwLock<HashMap<String, RecoveryStrategy>>>,
}

impl FaultToleranceManager {
    /// Create new fault tolerance manager
    pub fn new() -> Self {
        Self {
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            retry_mechanisms: Arc::new(RwLock::new(HashMap::new())),
            health_monitor: Arc::new(HealthMonitor::new(HealthCheckConfig::default())),
            recovery_strategies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register circuit breaker for a service
    pub async fn register_circuit_breaker(
        &self,
        service_name: String,
        config: CircuitBreakerConfig,
    ) {
        let circuit_breaker = Arc::new(CircuitBreaker::new(service_name.clone(), config));
        self.circuit_breakers
            .write()
            .await
            .insert(service_name, circuit_breaker);
    }

    /// Register retry mechanism for a service
    pub async fn register_retry_mechanism(&self, service_name: String, config: RetryConfig) {
        let retry_mechanism = Arc::new(RetryMechanism::new(config));
        self.retry_mechanisms
            .write()
            .await
            .insert(service_name, retry_mechanism);
    }

    /// Register recovery strategy for a service
    pub async fn register_recovery_strategy(
        &self,
        service_name: String,
        strategy: RecoveryStrategy,
    ) {
        self.recovery_strategies
            .write()
            .await
            .insert(service_name, strategy);
    }

    /// Execute operation with fault tolerance
    pub async fn execute_with_fault_tolerance<F, T>(
        &self,
        service_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send,
    {
        // Check if circuit breaker exists for this service
        if let Some(circuit_breaker) = self.circuit_breakers.read().await.get(service_name) {
            let cb = circuit_breaker.clone();
            return cb.execute(operation).await;
        }

        // Fallback to direct execution
        operation.await
    }

    /// Get fault tolerance metrics
    pub async fn get_metrics(&self) -> FaultToleranceMetrics {
        let circuit_breakers = self.circuit_breakers.read().await;
        let cb_metrics: Vec<_> = circuit_breakers
            .values()
            .map(|cb| cb.get_metrics())
            .collect();

        let health_summary = self.health_monitor.get_system_health().await;

        FaultToleranceMetrics {
            circuit_breaker_metrics: cb_metrics,
            system_health: health_summary,
            timestamp: current_timestamp(),
        }
    }

    /// Start all fault tolerance mechanisms
    pub async fn start(&self) -> Result<()> {
        self.health_monitor.start().await?;
        info!("Fault tolerance manager started");
        Ok(())
    }

    /// Stop all fault tolerance mechanisms
    pub async fn stop(&self) -> Result<()> {
        self.health_monitor.stop().await?;
        info!("Fault tolerance manager stopped");
        Ok(())
    }
}

/// Fault tolerance metrics
#[derive(Debug, Clone)]
pub struct FaultToleranceMetrics {
    pub circuit_breaker_metrics: Vec<CircuitBreakerMetrics>,
    pub system_health: SystemHealthSummary,
    pub timestamp: u64,
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            success_threshold: 1,
            timeout: Duration::from_secs(1),
        };

        let cb = CircuitBreaker::new("test".to_string(), config);

        // Test successful operation
        let result = cb
            .execute(async { Ok::<_, PipelineError>("success") })
            .await;
        assert!(result.is_ok());

        // Test failing operations to open circuit breaker
        for _ in 0..3 {
            let _ = cb
                .execute(async { Err::<String, _>(PipelineError::Mcp("test error".to_string())) })
                .await;
        }

        assert_eq!(cb.get_state().await, CircuitBreakerState::Open);

        // Test that requests are rejected when circuit is open
        let result = cb
            .execute(async { Ok::<_, PipelineError>("should fail") })
            .await;
        assert!(result.is_err());

        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;

        // Test recovery to half-open
        let result = cb
            .execute(async { Ok::<_, PipelineError>("recovery") })
            .await;
        assert!(result.is_ok());
        assert_eq!(cb.get_state().await, CircuitBreakerState::Closed);
    }

    #[tokio::test]
    async fn test_retry_mechanism() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            backoff_factor: 2.0,
            jitter: false,
        };

        let retry = RetryMechanism::new(config);
        let mut attempt_count = 0;

        let result = retry
            .execute(|| {
                attempt_count += 1;
                Box::pin(async move {
                    if attempt_count < 3 {
                        Err("temporary failure")
                    } else {
                        Ok("success")
                    }
                })
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(attempt_count, 3);
    }

    #[tokio::test]
    async fn test_health_monitor() {
        let monitor = HealthMonitor::new(HealthCheckConfig::default());
        monitor.start().await.unwrap();

        monitor
            .register_component("test_component".to_string())
            .await;

        // Record healthy check
        let healthy_result = HealthCheckResult {
            component: "test_component".to_string(),
            status: HealthStatus::Healthy,
            message: None,
            timestamp: current_timestamp(),
            response_time: Duration::from_millis(10),
        };

        monitor.record_health_check(healthy_result).await;

        let health_status = monitor.get_component_health("test_component").await;
        assert_eq!(health_status, Some(HealthStatus::Unknown)); // Still needs threshold

        // Record more healthy checks to meet threshold
        for _ in 0..2 {
            let result = HealthCheckResult {
                component: "test_component".to_string(),
                status: HealthStatus::Healthy,
                message: None,
                timestamp: current_timestamp(),
                response_time: Duration::from_millis(10),
            };
            monitor.record_health_check(result).await;
        }

        let health_status = monitor.get_component_health("test_component").await;
        assert_eq!(health_status, Some(HealthStatus::Healthy));

        monitor.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_fault_tolerance_manager() {
        let manager = FaultToleranceManager::new();
        manager.start().await.unwrap();

        // Register circuit breaker
        let cb_config = CircuitBreakerConfig::default();
        manager
            .register_circuit_breaker("test_service".to_string(), cb_config)
            .await;

        // Test execution with fault tolerance
        let result = manager
            .execute_with_fault_tolerance("test_service", async {
                Ok::<_, PipelineError>("success")
            })
            .await;

        assert!(result.is_ok());

        // Get metrics
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.circuit_breaker_metrics.len(), 1);

        manager.stop().await.unwrap();
    }
}
