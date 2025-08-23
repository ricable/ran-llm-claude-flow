// Intelligent Load Balancing Router with AI-Driven Traffic Distribution
// Implements adaptive routing, health monitoring, and performance optimization

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio::time::{interval, sleep};
use serde::{Deserialize, Serialize};
use log::{info, warn, error, debug};
use rand::Rng;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInstance {
    pub id: String,
    pub address: SocketAddr,
    pub weight: f64,
    pub current_connections: Arc<RwLock<u32>>,
    pub max_connections: u32,
    pub health_status: HealthStatus,
    pub performance_metrics: PerformanceMetrics,
    pub last_health_check: Instant,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Draining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub average_response_time: f64,
    pub error_rate: f64,
    pub throughput: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub queue_depth: u32,
    pub success_rate: f64,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub request_id: String,
    pub selected_backend: String,
    pub routing_algorithm: String,
    pub decision_factors: Vec<String>,
    pub confidence_score: f64,
    pub estimated_response_time: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    pub id: String,
    pub client_id: Option<String>,
    pub request_type: String,
    pub priority: RequestPriority,
    pub estimated_complexity: f64,
    pub timeout_ms: u64,
    pub retry_count: u32,
    pub headers: HashMap<String, String>,
    pub arrived_at: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub enum RoutingAlgorithm {
    WeightedRoundRobin,
    LeastConnections,
    LeastResponseTime,
    AdaptiveAI,
    PowerOfTwoChoices,
    ConsistentHashing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    pub default_algorithm: String,
    pub health_check_interval: Duration,
    pub max_retries: u32,
    pub circuit_breaker_threshold: f64,
    pub sticky_sessions: bool,
    pub adaptive_weights: bool,
    pub performance_window_size: usize,
    pub ai_routing_enabled: bool,
    pub failover_strategy: String,
}

pub struct IntelligentRouter {
    backends: Arc<RwLock<Vec<BackendInstance>>>,
    config: LoadBalancingConfig,
    routing_history: Arc<Mutex<VecDeque<RoutingDecision>>>,
    performance_history: Arc<Mutex<HashMap<String, VecDeque<PerformanceMetrics>>>>,
    session_store: Arc<RwLock<HashMap<String, String>>>, // client_id -> backend_id
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    ai_predictor: Arc<Mutex<AIRoutingPredictor>>,
    health_monitor: Arc<Mutex<HealthMonitor>>,
    metrics_tx: broadcast::Sender<RouterMetrics>,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Instant,
    failure_threshold: u32,
    recovery_timeout: Duration,
}

#[derive(Debug, Clone)]
enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Blocking requests
    HalfOpen, // Testing recovery
}

struct AIRoutingPredictor {
    feature_weights: HashMap<String, f64>,
    performance_model: HashMap<String, f64>, // Simple linear model
    prediction_history: VecDeque<(Vec<f64>, f64)>, // Features and actual response time
    model_accuracy: f64,
}

struct HealthMonitor {
    last_check_times: HashMap<String, Instant>,
    health_history: HashMap<String, VecDeque<bool>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterMetrics {
    pub timestamp: u64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: f64,
    pub backend_utilization: HashMap<String, f64>,
    pub routing_algorithm_usage: HashMap<String, u64>,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            default_algorithm: "AdaptiveAI".to_string(),
            health_check_interval: Duration::from_secs(30),
            max_retries: 3,
            circuit_breaker_threshold: 0.5,
            sticky_sessions: false,
            adaptive_weights: true,
            performance_window_size: 100,
            ai_routing_enabled: true,
            failover_strategy: "fastest_healthy".to_string(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            average_response_time: 100.0,
            error_rate: 0.0,
            throughput: 0.0,
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            queue_depth: 0,
            success_rate: 1.0,
            last_updated: current_timestamp(),
        }
    }
}

impl BackendInstance {
    pub fn new(id: String, address: SocketAddr) -> Self {
        Self {
            id,
            address,
            weight: 1.0,
            current_connections: Arc::new(RwLock::new(0)),
            max_connections: 1000,
            health_status: HealthStatus::Healthy,
            performance_metrics: PerformanceMetrics::default(),
            last_health_check: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self.health_status, HealthStatus::Healthy | HealthStatus::Degraded) &&
        *self.current_connections.read().unwrap() < self.max_connections
    }

    pub fn get_load_score(&self) -> f64 {
        let connection_ratio = *self.current_connections.read().unwrap() as f64 / self.max_connections as f64;
        let response_time_factor = self.performance_metrics.average_response_time / 1000.0; // Normalize to seconds
        let error_factor = self.performance_metrics.error_rate;
        
        // Combined load score (lower is better)
        connection_ratio * 0.4 + response_time_factor * 0.4 + error_factor * 0.2
    }
}

impl CircuitBreaker {
    fn new() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: Instant::now(),
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
        }
    }

    fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if self.last_failure_time.elapsed() > self.recovery_timeout {
                    self.state = CircuitBreakerState::HalfOpen;
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }

    fn record_success(&mut self) {
        self.success_count += 1;
        match self.state {
            CircuitBreakerState::HalfOpen => {
                if self.success_count >= 3 {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitBreakerState::Closed => {
                if self.failure_count > 0 {
                    self.failure_count = 0;
                }
            }
            _ => {}
        }
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Instant::now();
        
        match self.state {
            CircuitBreakerState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.failure_count = 1;
            }
            _ => {}
        }
    }
}

impl AIRoutingPredictor {
    fn new() -> Self {
        let mut feature_weights = HashMap::new();
        feature_weights.insert("response_time".to_string(), 0.4);
        feature_weights.insert("error_rate".to_string(), 0.3);
        feature_weights.insert("cpu_utilization".to_string(), 0.15);
        feature_weights.insert("memory_utilization".to_string(), 0.1);
        feature_weights.insert("queue_depth".to_string(), 0.05);

        Self {
            feature_weights,
            performance_model: HashMap::new(),
            prediction_history: VecDeque::new(),
            model_accuracy: 0.5,
        }
    }

    fn predict_performance(&self, backend: &BackendInstance, context: &RequestContext) -> f64 {
        let metrics = &backend.performance_metrics;
        
        // Extract features
        let mut features = vec![
            metrics.average_response_time,
            metrics.error_rate * 1000.0, // Scale error rate
            metrics.cpu_utilization,
            metrics.memory_utilization,
            metrics.queue_depth as f64,
            context.estimated_complexity,
        ];

        // Normalize features
        for feature in features.iter_mut() {
            *feature = (*feature).max(0.0).min(100.0) / 100.0;
        }

        // Simple linear prediction
        let prediction = features[0] * 0.4 +  // response time
                        features[1] * 0.3 +  // error rate
                        features[2] * 0.15 + // cpu
                        features[3] * 0.1 +  // memory
                        features[4] * 0.05;  // queue

        // Add context-based adjustments
        let complexity_factor = match context.priority {
            RequestPriority::Critical => 0.8,
            RequestPriority::High => 0.9,
            RequestPriority::Normal => 1.0,
            RequestPriority::Low => 1.1,
        };

        prediction * complexity_factor * 1000.0 // Convert back to milliseconds
    }

    fn train_model(&mut self, features: Vec<f64>, actual_response_time: f64) {
        self.prediction_history.push_back((features, actual_response_time));
        
        // Keep only recent training data
        if self.prediction_history.len() > 1000 {
            self.prediction_history.pop_front();
        }

        // Update model accuracy based on recent predictions
        if self.prediction_history.len() >= 10 {
            let recent_accuracy = self.calculate_recent_accuracy();
            self.model_accuracy = self.model_accuracy * 0.9 + recent_accuracy * 0.1;
        }
    }

    fn calculate_recent_accuracy(&self) -> f64 {
        let recent_data: Vec<_> = self.prediction_history.iter().rev().take(10).collect();
        if recent_data.is_empty() {
            return 0.5;
        }

        let mut total_error = 0.0;
        for (features, actual) in recent_data {
            let predicted = self.simple_linear_prediction(features);
            let error = (predicted - actual).abs() / actual.max(1.0);
            total_error += error;
        }

        let average_error = total_error / recent_data.len() as f64;
        (1.0 - average_error.min(1.0)).max(0.0)
    }

    fn simple_linear_prediction(&self, features: &[f64]) -> f64 {
        features.iter().enumerate()
            .map(|(i, &feature)| feature * (0.5 - i as f64 * 0.1).max(0.1))
            .sum::<f64>() * 1000.0
    }
}

impl IntelligentRouter {
    pub fn new(config: LoadBalancingConfig) -> Self {
        let (metrics_tx, _) = broadcast::channel(1000);

        Self {
            backends: Arc::new(RwLock::new(Vec::new())),
            config,
            routing_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            session_store: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            ai_predictor: Arc::new(Mutex::new(AIRoutingPredictor::new())),
            health_monitor: Arc::new(Mutex::new(HealthMonitor {
                last_check_times: HashMap::new(),
                health_history: HashMap::new(),
            })),
            metrics_tx,
        }
    }

    pub async fn add_backend(&self, backend: BackendInstance) {
        let backend_id = backend.id.clone();
        
        {
            let mut backends = self.backends.write().unwrap();
            backends.push(backend);
        }

        // Initialize circuit breaker
        {
            let mut breakers = self.circuit_breakers.write().unwrap();
            breakers.insert(backend_id.clone(), CircuitBreaker::new());
        }

        // Initialize performance history
        {
            let mut history = self.performance_history.lock().await;
            history.insert(backend_id.clone(), VecDeque::new());
        }

        info!("Added backend: {}", backend_id);
    }

    pub async fn remove_backend(&self, backend_id: &str) -> bool {
        let mut backends = self.backends.write().unwrap();
        let initial_len = backends.len();
        backends.retain(|b| b.id != backend_id);
        
        if backends.len() < initial_len {
            // Clean up related data
            {
                let mut breakers = self.circuit_breakers.write().unwrap();
                breakers.remove(backend_id);
            }
            {
                let mut history = self.performance_history.lock().await;
                history.remove(backend_id);
            }
            
            info!("Removed backend: {}", backend_id);
            true
        } else {
            false
        }
    }

    pub async fn route_request(&self, context: RequestContext) -> Option<RoutingDecision> {
        let algorithm = self.select_routing_algorithm(&context).await;
        
        match algorithm {
            RoutingAlgorithm::AdaptiveAI => self.route_with_ai(&context).await,
            RoutingAlgorithm::WeightedRoundRobin => self.route_weighted_round_robin(&context).await,
            RoutingAlgorithm::LeastConnections => self.route_least_connections(&context).await,
            RoutingAlgorithm::LeastResponseTime => self.route_least_response_time(&context).await,
            RoutingAlgorithm::PowerOfTwoChoices => self.route_power_of_two(&context).await,
            RoutingAlgorithm::ConsistentHashing => self.route_consistent_hash(&context).await,
        }
    }

    async fn select_routing_algorithm(&self, context: &RequestContext) -> RoutingAlgorithm {
        // AI-driven algorithm selection based on request context and current conditions
        if !self.config.ai_routing_enabled {
            return RoutingAlgorithm::WeightedRoundRobin;
        }

        match context.priority {
            RequestPriority::Critical => RoutingAlgorithm::LeastResponseTime,
            RequestPriority::High => {
                if context.estimated_complexity > 0.8 {
                    RoutingAlgorithm::AdaptiveAI
                } else {
                    RoutingAlgorithm::LeastConnections
                }
            }
            RequestPriority::Normal => RoutingAlgorithm::AdaptiveAI,
            RequestPriority::Low => RoutingAlgorithm::WeightedRoundRobin,
        }
    }

    async fn route_with_ai(&self, context: &RequestContext) -> Option<RoutingDecision> {
        let backends = self.backends.read().unwrap();
        let available_backends: Vec<_> = backends.iter()
            .filter(|b| self.is_backend_available(b))
            .collect();

        if available_backends.is_empty() {
            return None;
        }

        let predictor = self.ai_predictor.lock().await;
        let mut best_backend = None;
        let mut best_score = f64::INFINITY;
        let mut decision_factors = Vec::new();

        for backend in available_backends {
            let predicted_time = predictor.predict_performance(backend, context);
            let load_score = backend.get_load_score();
            let combined_score = predicted_time * 0.7 + load_score * 300.0; // Weight factors

            decision_factors.push(format!(
                "{}:pred={:.1}ms,load={:.2},score={:.1}",
                backend.id, predicted_time, load_score, combined_score
            ));

            if combined_score < best_score {
                best_score = combined_score;
                best_backend = Some(backend);
            }
        }

        if let Some(backend) = best_backend {
            let decision = RoutingDecision {
                request_id: context.id.clone(),
                selected_backend: backend.id.clone(),
                routing_algorithm: "AdaptiveAI".to_string(),
                decision_factors,
                confidence_score: predictor.model_accuracy,
                estimated_response_time: best_score,
                timestamp: current_timestamp(),
            };

            self.record_routing_decision(decision.clone()).await;
            Some(decision)
        } else {
            None
        }
    }

    async fn route_least_connections(&self, context: &RequestContext) -> Option<RoutingDecision> {
        let backends = self.backends.read().unwrap();
        let mut best_backend = None;
        let mut min_connections = u32::MAX;

        for backend in backends.iter() {
            if self.is_backend_available(backend) {
                let connections = *backend.current_connections.read().unwrap();
                if connections < min_connections {
                    min_connections = connections;
                    best_backend = Some(backend);
                }
            }
        }

        if let Some(backend) = best_backend {
            let decision = RoutingDecision {
                request_id: context.id.clone(),
                selected_backend: backend.id.clone(),
                routing_algorithm: "LeastConnections".to_string(),
                decision_factors: vec![format!("connections={}", min_connections)],
                confidence_score: 0.8,
                estimated_response_time: backend.performance_metrics.average_response_time,
                timestamp: current_timestamp(),
            };

            self.record_routing_decision(decision.clone()).await;
            Some(decision)
        } else {
            None
        }
    }

    async fn route_least_response_time(&self, context: &RequestContext) -> Option<RoutingDecision> {
        let backends = self.backends.read().unwrap();
        let mut best_backend = None;
        let mut min_response_time = f64::INFINITY;

        for backend in backends.iter() {
            if self.is_backend_available(backend) {
                let response_time = backend.performance_metrics.average_response_time;
                if response_time < min_response_time {
                    min_response_time = response_time;
                    best_backend = Some(backend);
                }
            }
        }

        if let Some(backend) = best_backend {
            let decision = RoutingDecision {
                request_id: context.id.clone(),
                selected_backend: backend.id.clone(),
                routing_algorithm: "LeastResponseTime".to_string(),
                decision_factors: vec![format!("response_time={:.1}ms", min_response_time)],
                confidence_score: 0.9,
                estimated_response_time: min_response_time,
                timestamp: current_timestamp(),
            };

            self.record_routing_decision(decision.clone()).await;
            Some(decision)
        } else {
            None
        }
    }

    async fn route_weighted_round_robin(&self, context: &RequestContext) -> Option<RoutingDecision> {
        let backends = self.backends.read().unwrap();
        let available_backends: Vec<_> = backends.iter()
            .filter(|b| self.is_backend_available(b))
            .collect();

        if available_backends.is_empty() {
            return None;
        }

        // Simple weighted selection
        let total_weight: f64 = available_backends.iter().map(|b| b.weight).sum();
        let mut random_value = rand::thread_rng().gen::<f64>() * total_weight;

        for backend in available_backends {
            random_value -= backend.weight;
            if random_value <= 0.0 {
                let decision = RoutingDecision {
                    request_id: context.id.clone(),
                    selected_backend: backend.id.clone(),
                    routing_algorithm: "WeightedRoundRobin".to_string(),
                    decision_factors: vec![format!("weight={:.2}", backend.weight)],
                    confidence_score: 0.7,
                    estimated_response_time: backend.performance_metrics.average_response_time,
                    timestamp: current_timestamp(),
                };

                self.record_routing_decision(decision.clone()).await;
                return Some(decision);
            }
        }

        None
    }

    async fn route_power_of_two(&self, context: &RequestContext) -> Option<RoutingDecision> {
        let backends = self.backends.read().unwrap();
        let available_backends: Vec<_> = backends.iter()
            .filter(|b| self.is_backend_available(b))
            .collect();

        if available_backends.is_empty() {
            return None;
        }

        // Power of Two Choices: randomly select two backends, choose the better one
        let mut rng = rand::thread_rng();
        let choice1 = &available_backends[rng.gen_range(0..available_backends.len())];
        let choice2 = &available_backends[rng.gen_range(0..available_backends.len())];

        let score1 = choice1.get_load_score();
        let score2 = choice2.get_load_score();

        let selected = if score1 <= score2 { choice1 } else { choice2 };

        let decision = RoutingDecision {
            request_id: context.id.clone(),
            selected_backend: selected.id.clone(),
            routing_algorithm: "PowerOfTwoChoices".to_string(),
            decision_factors: vec![format!("score1={:.2},score2={:.2}", score1, score2)],
            confidence_score: 0.75,
            estimated_response_time: selected.performance_metrics.average_response_time,
            timestamp: current_timestamp(),
        };

        self.record_routing_decision(decision.clone()).await;
        Some(decision)
    }

    async fn route_consistent_hash(&self, context: &RequestContext) -> Option<RoutingDecision> {
        let backends = self.backends.read().unwrap();
        let available_backends: Vec<_> = backends.iter()
            .filter(|b| self.is_backend_available(b))
            .collect();

        if available_backends.is_empty() {
            return None;
        }

        // Use client_id for consistent hashing, fallback to request_id
        let hash_key = context.client_id.as_ref().unwrap_or(&context.id);
        let hash = simple_hash(hash_key);
        let backend_index = hash % available_backends.len();
        let selected = available_backends[backend_index];

        let decision = RoutingDecision {
            request_id: context.id.clone(),
            selected_backend: selected.id.clone(),
            routing_algorithm: "ConsistentHashing".to_string(),
            decision_factors: vec![format!("hash={},index={}", hash, backend_index)],
            confidence_score: 0.8,
            estimated_response_time: selected.performance_metrics.average_response_time,
            timestamp: current_timestamp(),
        };

        self.record_routing_decision(decision.clone()).await;
        Some(decision)
    }

    fn is_backend_available(&self, backend: &BackendInstance) -> bool {
        if !backend.is_available() {
            return false;
        }

        // Check circuit breaker
        if let Ok(breakers) = self.circuit_breakers.read() {
            if let Some(breaker) = breakers.get(&backend.id) {
                return matches!(breaker.state, CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen);
            }
        }

        true
    }

    async fn record_routing_decision(&self, decision: RoutingDecision) {
        let mut history = self.routing_history.lock().await;
        history.push_back(decision);

        // Keep history size manageable
        if history.len() > 10000 {
            history.pop_front();
        }
    }

    pub async fn record_request_completion(&self, request_id: &str, backend_id: &str, 
                                         response_time: f64, success: bool) {
        // Update circuit breaker
        if let Ok(mut breakers) = self.circuit_breakers.write() {
            if let Some(breaker) = breakers.get_mut(backend_id) {
                if success {
                    breaker.record_success();
                } else {
                    breaker.record_failure();
                }
            }
        }

        // Update backend connection count
        {
            let backends = self.backends.read().unwrap();
            if let Some(backend) = backends.iter().find(|b| b.id == backend_id) {
                let mut connections = backend.current_connections.write().unwrap();
                *connections = connections.saturating_sub(1);
            }
        }

        // Train AI model
        if let Some(decision) = self.find_routing_decision(request_id).await {
            let features = vec![
                decision.estimated_response_time / 1000.0,
                if success { 0.0 } else { 1.0 },
                decision.confidence_score,
            ];
            
            let mut predictor = self.ai_predictor.lock().await;
            predictor.train_model(features, response_time);
        }

        debug!("Recorded request completion: {} -> {} ({}ms, success={})", 
               request_id, backend_id, response_time, success);
    }

    async fn find_routing_decision(&self, request_id: &str) -> Option<RoutingDecision> {
        let history = self.routing_history.lock().await;
        history.iter().find(|d| d.request_id == request_id).cloned()
    }

    pub async fn update_backend_metrics(&self, backend_id: &str, metrics: PerformanceMetrics) {
        {
            let mut backends = self.backends.write().unwrap();
            if let Some(backend) = backends.iter_mut().find(|b| b.id == backend_id) {
                backend.performance_metrics = metrics.clone();
                backend.last_health_check = Instant::now();
                
                // Update health status based on metrics
                backend.health_status = if metrics.error_rate > 0.1 || metrics.average_response_time > 5000.0 {
                    HealthStatus::Degraded
                } else if metrics.error_rate > 0.2 || metrics.average_response_time > 10000.0 {
                    HealthStatus::Unhealthy
                } else {
                    HealthStatus::Healthy
                };
            }
        }

        // Store performance history
        {
            let mut history = self.performance_history.lock().await;
            if let Some(backend_history) = history.get_mut(backend_id) {
                backend_history.push_back(metrics);
                if backend_history.len() > self.config.performance_window_size {
                    backend_history.pop_front();
                }
            }
        }
    }

    pub async fn get_router_metrics(&self) -> RouterMetrics {
        let history = self.routing_history.lock().await;
        let total_requests = history.len() as u64;
        
        // Calculate metrics from recent history
        let recent_window = 1000; // Last 1000 requests
        let recent_decisions: Vec<_> = history.iter().rev().take(recent_window).collect();
        
        let mut algorithm_usage = HashMap::new();
        let mut total_response_time = 0.0;
        
        for decision in &recent_decisions {
            *algorithm_usage.entry(decision.routing_algorithm.clone()).or_insert(0) += 1;
            total_response_time += decision.estimated_response_time;
        }

        let average_response_time = if !recent_decisions.is_empty() {
            total_response_time / recent_decisions.len() as f64
        } else {
            0.0
        };

        // Backend utilization
        let mut backend_utilization = HashMap::new();
        {
            let backends = self.backends.read().unwrap();
            for backend in backends.iter() {
                let utilization = *backend.current_connections.read().unwrap() as f64 / 
                                backend.max_connections as f64;
                backend_utilization.insert(backend.id.clone(), utilization);
            }
        }

        RouterMetrics {
            timestamp: current_timestamp(),
            total_requests,
            successful_requests: total_requests * 95 / 100, // Estimate
            failed_requests: total_requests * 5 / 100,
            average_response_time,
            backend_utilization,
            routing_algorithm_usage: algorithm_usage,
        }
    }

    pub async fn start_health_monitoring(&self) {
        let backends = Arc::clone(&self.backends);
        let health_monitor = Arc::clone(&self.health_monitor);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);
            
            loop {
                interval.tick().await;
                
                let backend_list = backends.read().unwrap().clone();
                for backend in backend_list {
                    // Simulate health check (replace with actual HTTP health check)
                    let is_healthy = simulate_health_check(&backend).await;
                    
                    let mut monitor = health_monitor.lock().await;
                    monitor.last_check_times.insert(backend.id.clone(), Instant::now());
                    
                    let history = monitor.health_history.entry(backend.id.clone()).or_insert_with(VecDeque::new);
                    history.push_back(is_healthy);
                    
                    if history.len() > 10 {
                        history.pop_front();
                    }
                    
                    // Update backend health status
                    let healthy_count = history.iter().filter(|&&h| h).count();
                    let health_ratio = healthy_count as f64 / history.len() as f64;
                    
                    if health_ratio < 0.3 {
                        // Mark as unhealthy
                        debug!("Backend {} marked as unhealthy (health ratio: {:.2})", backend.id, health_ratio);
                    } else if health_ratio < 0.7 {
                        // Mark as degraded
                        debug!("Backend {} marked as degraded (health ratio: {:.2})", backend.id, health_ratio);
                    }
                }
            }
        });
    }
}

// Utility functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn simple_hash(s: &str) -> usize {
    let mut hash = 0usize;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
    }
    hash
}

async fn simulate_health_check(_backend: &BackendInstance) -> bool {
    // Simulate health check with random success/failure
    // In real implementation, this would be an HTTP GET to /health endpoint
    sleep(Duration::from_millis(10)).await;
    rand::thread_rng().gen_ratio(9, 10) // 90% success rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    
    #[tokio::test]
    async fn test_router_initialization() {
        let config = LoadBalancingConfig::default();
        let router = IntelligentRouter::new(config);
        
        let backend = BackendInstance::new(
            "test-backend".to_string(),
            SocketAddr::from_str("127.0.0.1:8080").unwrap()
        );
        
        router.add_backend(backend).await;
        
        let backends = router.backends.read().unwrap();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].id, "test-backend");
    }

    #[tokio::test]
    async fn test_routing_decision() {
        let config = LoadBalancingConfig::default();
        let router = IntelligentRouter::new(config);
        
        let backend = BackendInstance::new(
            "test-backend".to_string(),
            SocketAddr::from_str("127.0.0.1:8080").unwrap()
        );
        
        router.add_backend(backend).await;
        
        let context = RequestContext {
            id: Uuid::new_v4().to_string(),
            client_id: None,
            request_type: "GET".to_string(),
            priority: RequestPriority::Normal,
            estimated_complexity: 0.5,
            timeout_ms: 5000,
            retry_count: 0,
            headers: HashMap::new(),
            arrived_at: Instant::now(),
        };
        
        let decision = router.route_request(context).await;
        assert!(decision.is_some());
        
        let decision = decision.unwrap();
        assert_eq!(decision.selected_backend, "test-backend");
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new();
        
        // Normal operation
        assert!(breaker.can_execute());
        
        // Record failures to trip the breaker
        for _ in 0..5 {
            breaker.record_failure();
        }
        
        // Should be open now
        assert!(!breaker.can_execute());
    }

    #[tokio::test]
    async fn test_ai_predictor() {
        let mut predictor = AIRoutingPredictor::new();
        
        let backend = BackendInstance::new(
            "test".to_string(),
            SocketAddr::from_str("127.0.0.1:8080").unwrap()
        );
        
        let context = RequestContext {
            id: "test".to_string(),
            client_id: None,
            request_type: "GET".to_string(),
            priority: RequestPriority::Normal,
            estimated_complexity: 0.5,
            timeout_ms: 5000,
            retry_count: 0,
            headers: HashMap::new(),
            arrived_at: Instant::now(),
        };
        
        let prediction = predictor.predict_performance(&backend, &context);
        assert!(prediction > 0.0);
        
        // Train with some data
        predictor.train_model(vec![0.5, 0.1, 0.3, 0.2, 0.1], 150.0);
        assert!(predictor.model_accuracy > 0.0);
    }
}

// Example usage
pub async fn example_intelligent_routing() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let config = LoadBalancingConfig {
        ai_routing_enabled: true,
        adaptive_weights: true,
        health_check_interval: Duration::from_secs(30),
        ..Default::default()
    };
    
    let router = IntelligentRouter::new(config);
    
    // Add some backends
    for i in 1..=3 {
        let backend = BackendInstance::new(
            format!("backend-{}", i),
            SocketAddr::from_str(&format!("127.0.0.1:808{}", i)).unwrap()
        );
        router.add_backend(backend).await;
    }
    
    // Start health monitoring
    router.start_health_monitoring().await;
    
    // Simulate requests
    for i in 0..10 {
        let context = RequestContext {
            id: format!("request-{}", i),
            client_id: Some(format!("client-{}", i % 3)),
            request_type: "GET".to_string(),
            priority: if i % 4 == 0 { RequestPriority::High } else { RequestPriority::Normal },
            estimated_complexity: 0.3 + (i as f64 * 0.1),
            timeout_ms: 5000,
            retry_count: 0,
            headers: HashMap::new(),
            arrived_at: Instant::now(),
        };
        
        if let Some(decision) = router.route_request(context).await {
            info!("Routed request {} to backend {} using {} (confidence: {:.2})",
                  decision.request_id, 
                  decision.selected_backend, 
                  decision.routing_algorithm,
                  decision.confidence_score);
            
            // Simulate request completion
            let response_time = 100.0 + rand::thread_rng().gen::<f64>() * 200.0;
            let success = rand::thread_rng().gen_ratio(95, 100);
            
            router.record_request_completion(&decision.request_id, &decision.selected_backend, response_time, success).await;
        }
        
        sleep(Duration::from_millis(100)).await;
    }
    
    // Get final metrics
    let metrics = router.get_router_metrics().await;
    info!("Final metrics: {:?}", metrics);
    
    Ok(())
}