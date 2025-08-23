use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::time::interval;
use warp::{Filter, Reply};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{WebSocketStream, tungstenite::Message};

/// M3 Max optimized performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M3MaxMetrics {
    pub timestamp: u64,
    pub unified_memory_total: u64,
    pub unified_memory_used: u64,
    pub unified_memory_free: u64,
    pub memory_bandwidth_utilization: f64,
    pub neural_engine_utilization: f64,
    pub gpu_core_utilization: f64,
    pub cpu_performance_cores: u8,
    pub cpu_efficiency_cores: u8,
    pub cpu_load: f64,
    pub thermal_state: String,
    pub power_consumption: f64,
}

/// Real-time system performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub memory_usage_percent: f64,
    pub memory_efficiency: f64,
    pub cpu_load: f64,
    pub disk_io_read_mb: f64,
    pub disk_io_write_mb: f64,
    pub network_rx_mb: f64,
    pub network_tx_mb: f64,
    pub active_processes: u32,
    pub thread_count: u32,
    pub temperature: f64,
}

/// Pipeline throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub timestamp: u64,
    pub documents_processed: u64,
    pub documents_per_minute: f64,
    pub average_processing_time_ms: f64,
    pub queue_depth: u32,
    pub active_workers: u32,
    pub success_rate: f64,
    pub error_rate: f64,
    pub model_inference_time_ms: f64,
    pub data_pipeline_latency_ms: f64,
}

/// Performance alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThreshold {
    pub memory_usage_percent: f64,
    pub cpu_load_percent: f64,
    pub temperature_celsius: f64,
    pub error_rate_percent: f64,
    pub response_time_ms: f64,
    pub queue_depth: u32,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub value: f64,
    pub threshold: f64,
    pub resolution_suggestions: Vec<String>,
}

/// Performance regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub percentage_change: f64,
    pub is_regression: bool,
    pub confidence_score: f64,
    pub trend_direction: String,
    pub recommended_actions: Vec<String>,
}

/// Real-time dashboard state
#[derive(Debug, Clone)]
pub struct DashboardState {
    pub m3_max_metrics: Arc<Mutex<Vec<M3MaxMetrics>>>,
    pub system_metrics: Arc<Mutex<Vec<SystemMetrics>>>,
    pub throughput_metrics: Arc<Mutex<Vec<ThroughputMetrics>>>,
    pub alerts: Arc<Mutex<Vec<PerformanceAlert>>>,
    pub regression_analysis: Arc<Mutex<HashMap<String, RegressionAnalysis>>>,
    pub alert_thresholds: Arc<Mutex<AlertThreshold>>,
    pub connected_clients: Arc<Mutex<Vec<Arc<Mutex<warp::ws::WebSocket>>>>>,
}

/// Performance monitoring dashboard
pub struct PerformanceDashboard {
    pub state: DashboardState,
    metrics_collector: MetricsCollector,
    alert_system: AlertSystem,
    regression_detector: RegressionDetector,
}

impl PerformanceDashboard {
    pub fn new() -> Self {
        let state = DashboardState {
            m3_max_metrics: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            system_metrics: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            throughput_metrics: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            alerts: Arc::new(Mutex::new(Vec::new())),
            regression_analysis: Arc::new(Mutex::new(HashMap::new())),
            alert_thresholds: Arc::new(Mutex::new(AlertThreshold {
                memory_usage_percent: 85.0,
                cpu_load_percent: 80.0,
                temperature_celsius: 80.0,
                error_rate_percent: 5.0,
                response_time_ms: 1000.0,
                queue_depth: 100,
            })),
            connected_clients: Arc::new(Mutex::new(Vec::new())),
        };
        
        Self {
            metrics_collector: MetricsCollector::new(state.clone()),
            alert_system: AlertSystem::new(state.clone()),
            regression_detector: RegressionDetector::new(state.clone()),
            state,
        }
    }
    
    /// Start the performance monitoring dashboard
    pub async fn start(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        println!("🚀 Starting Performance Monitoring Dashboard on port {}", port);
        
        // Start metrics collection
        self.start_metrics_collection().await;
        
        // Start alert monitoring
        self.start_alert_monitoring().await;
        
        // Start regression detection
        self.start_regression_detection().await;
        
        // Start web server
        self.start_web_server(port).await
    }
    
    async fn start_metrics_collection(&self) {
        let state = self.state.clone();
        let collector = self.metrics_collector.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                collector.collect_all_metrics().await;
            }
        });
    }
    
    async fn start_alert_monitoring(&self) {
        let state = self.state.clone();
        let alert_system = self.alert_system.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                alert_system.check_alerts().await;
            }
        });
    }
    
    async fn start_regression_detection(&self) {
        let state = self.state.clone();
        let detector = self.regression_detector.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                detector.analyze_regressions().await;
            }
        });
    }
    
    async fn start_web_server(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.state.clone();
        
        // WebSocket route for real-time updates
        let state_for_ws = state.clone();
        let ws_route = warp::path("ws")
            .and(warp::ws())
            .and(warp::any().map(move || state_for_ws.clone()))
            .map(|ws: warp::ws::Ws, state: DashboardState| {
                ws.on_upgrade(move |websocket| handle_websocket(websocket, state))
            });
        
        // Static files and API routes
        let state_for_metrics = state.clone();
        let state_for_alerts = state.clone();
        let state_for_regression = state.clone();
        
        let api_routes = warp::path("api")
            .and(
                warp::path("metrics")
                .and(warp::get())
                .and(warp::any().map(move || state_for_metrics.clone()))
                .and_then(get_metrics)
                .or(
                    warp::path("alerts")
                    .and(warp::get())
                    .and(warp::any().map(move || state_for_alerts.clone()))
                    .and_then(get_alerts)
                )
                .or(
                    warp::path("regression")
                    .and(warp::get())
                    .and(warp::any().map(move || state_for_regression.clone()))
                    .and_then(get_regression_analysis)
                )
            );
        
        let static_files = warp::path("static")
            .and(warp::fs::dir("web/static"));
        
        let index = warp::path::end()
            .and(warp::fs::file("web/templates/dashboard.html"));
        
        let routes = ws_route
            .or(api_routes)
            .or(static_files)
            .or(index);
        
        println!("📊 Dashboard available at http://localhost:{}", port);
        warp::serve(routes)
            .run(([0, 0, 0, 0], port))
            .await;
        
        Ok(())
    }
}

/// Metrics collector for M3 Max optimized collection
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    state: DashboardState,
}

impl MetricsCollector {
    pub fn new(state: DashboardState) -> Self {
        Self { state }
    }
    
    pub async fn collect_all_metrics(&self) {
        tokio::join!(
            self.collect_m3_max_metrics(),
            self.collect_system_metrics(),
            self.collect_throughput_metrics()
        );
    }
    
    async fn collect_m3_max_metrics(&self) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        // M3 Max specific metrics collection
        let metrics = M3MaxMetrics {
            timestamp,
            unified_memory_total: self.get_unified_memory_total().await,
            unified_memory_used: self.get_unified_memory_used().await,
            unified_memory_free: self.get_unified_memory_free().await,
            memory_bandwidth_utilization: self.get_memory_bandwidth_utilization().await,
            neural_engine_utilization: self.get_neural_engine_utilization().await,
            gpu_core_utilization: self.get_gpu_utilization().await,
            cpu_performance_cores: 8,
            cpu_efficiency_cores: 4,
            cpu_load: self.get_cpu_load().await,
            thermal_state: self.get_thermal_state().await,
            power_consumption: self.get_power_consumption().await,
        };
        
        let mut m3_metrics = self.state.m3_max_metrics.lock().unwrap();
        m3_metrics.push(metrics);
        
        // Keep only last 1000 samples
        if m3_metrics.len() > 1000 {
            m3_metrics.remove(0);
        }
    }
    
    async fn collect_system_metrics(&self) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let metrics = SystemMetrics {
            timestamp,
            memory_usage_percent: self.get_memory_usage_percent().await,
            memory_efficiency: self.get_memory_efficiency().await,
            cpu_load: self.get_cpu_load().await,
            disk_io_read_mb: self.get_disk_io_read().await,
            disk_io_write_mb: self.get_disk_io_write().await,
            network_rx_mb: self.get_network_rx().await,
            network_tx_mb: self.get_network_tx().await,
            active_processes: self.get_active_processes().await,
            thread_count: self.get_thread_count().await,
            temperature: self.get_cpu_temperature().await,
        };
        
        let mut sys_metrics = self.state.system_metrics.lock().unwrap();
        sys_metrics.push(metrics);
        
        if sys_metrics.len() > 1000 {
            sys_metrics.remove(0);
        }
    }
    
    async fn collect_throughput_metrics(&self) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let metrics = ThroughputMetrics {
            timestamp,
            documents_processed: self.get_documents_processed().await,
            documents_per_minute: self.get_documents_per_minute().await,
            average_processing_time_ms: self.get_avg_processing_time().await,
            queue_depth: self.get_queue_depth().await,
            active_workers: self.get_active_workers().await,
            success_rate: self.get_success_rate().await,
            error_rate: self.get_error_rate().await,
            model_inference_time_ms: self.get_model_inference_time().await,
            data_pipeline_latency_ms: self.get_pipeline_latency().await,
        };
        
        let mut throughput_metrics = self.state.throughput_metrics.lock().unwrap();
        throughput_metrics.push(metrics);
        
        if throughput_metrics.len() > 1000 {
            throughput_metrics.remove(0);
        }
    }
    
    // M3 Max specific metric collection methods
    async fn get_unified_memory_total(&self) -> u64 {
        // Use system_profiler to get M3 Max unified memory info
        128 * 1024 * 1024 * 1024 // 128GB typical M3 Max config
    }
    
    async fn get_unified_memory_used(&self) -> u64 {
        // Read from system APIs
        self.read_memory_pressure().await.unwrap_or(0)
    }
    
    async fn get_unified_memory_free(&self) -> u64 {
        let total = self.get_unified_memory_total().await;
        let used = self.get_unified_memory_used().await;
        total.saturating_sub(used)
    }
    
    async fn get_memory_bandwidth_utilization(&self) -> f64 {
        // M3 Max specific memory bandwidth monitoring
        self.read_memory_bandwidth().await.unwrap_or(0.0)
    }
    
    async fn get_neural_engine_utilization(&self) -> f64 {
        // Monitor Neural Engine usage for ML inference
        self.read_neural_engine_usage().await.unwrap_or(0.0)
    }
    
    async fn get_gpu_utilization(&self) -> f64 {
        // M3 Max GPU core utilization
        self.read_gpu_utilization().await.unwrap_or(0.0)
    }
    
    async fn get_cpu_load(&self) -> f64 {
        // CPU load average
        self.read_cpu_load().await.unwrap_or(0.0)
    }
    
    async fn get_thermal_state(&self) -> String {
        // M3 Max thermal monitoring
        self.read_thermal_state().await.unwrap_or_else(|| "Normal".to_string())
    }
    
    async fn get_power_consumption(&self) -> f64 {
        // Power consumption monitoring
        self.read_power_consumption().await.unwrap_or(0.0)
    }
    
    // System metric collection methods (implementation details)
    async fn get_memory_usage_percent(&self) -> f64 { 0.0 }
    async fn get_memory_efficiency(&self) -> f64 { 0.0 }
    async fn get_disk_io_read(&self) -> f64 { 0.0 }
    async fn get_disk_io_write(&self) -> f64 { 0.0 }
    async fn get_network_rx(&self) -> f64 { 0.0 }
    async fn get_network_tx(&self) -> f64 { 0.0 }
    async fn get_active_processes(&self) -> u32 { 0 }
    async fn get_thread_count(&self) -> u32 { 0 }
    async fn get_cpu_temperature(&self) -> f64 { 0.0 }
    
    // Throughput metric collection methods
    async fn get_documents_processed(&self) -> u64 { 0 }
    async fn get_documents_per_minute(&self) -> f64 { 0.0 }
    async fn get_avg_processing_time(&self) -> f64 { 0.0 }
    async fn get_queue_depth(&self) -> u32 { 0 }
    async fn get_active_workers(&self) -> u32 { 0 }
    async fn get_success_rate(&self) -> f64 { 0.0 }
    async fn get_error_rate(&self) -> f64 { 0.0 }
    async fn get_model_inference_time(&self) -> f64 { 0.0 }
    async fn get_pipeline_latency(&self) -> f64 { 0.0 }
    
    // Low-level system readers (platform specific implementations)
    async fn read_memory_pressure(&self) -> Option<u64> { None }
    async fn read_memory_bandwidth(&self) -> Option<f64> { None }
    async fn read_neural_engine_usage(&self) -> Option<f64> { None }
    async fn read_gpu_utilization(&self) -> Option<f64> { None }
    async fn read_cpu_load(&self) -> Option<f64> { None }
    async fn read_thermal_state(&self) -> Option<String> { None }
    async fn read_power_consumption(&self) -> Option<f64> { None }
}

/// Alert system for performance monitoring
#[derive(Debug, Clone)]
pub struct AlertSystem {
    state: DashboardState,
}

impl AlertSystem {
    pub fn new(state: DashboardState) -> Self {
        Self { state }
    }
    
    pub async fn check_alerts(&self) {
        self.check_memory_alerts().await;
        self.check_cpu_alerts().await;
        self.check_temperature_alerts().await;
        self.check_throughput_alerts().await;
        self.check_error_rate_alerts().await;
    }
    
    async fn check_memory_alerts(&self) {
        let new_alert = {
            let m3_metrics = self.state.m3_max_metrics.lock().unwrap();
            let thresholds = self.state.alert_thresholds.lock().unwrap();
            
            if let Some(latest) = m3_metrics.last() {
                let usage_percent = (latest.unified_memory_used as f64 / latest.unified_memory_total as f64) * 100.0;
                
                if usage_percent > thresholds.memory_usage_percent {
                    let alert = PerformanceAlert {
                        timestamp: latest.timestamp,
                        severity: if usage_percent > 95.0 { AlertSeverity::Critical } else { AlertSeverity::Warning },
                        component: "Memory".to_string(),
                        message: format!("High memory usage: {:.1}%", usage_percent),
                        value: usage_percent,
                        threshold: thresholds.memory_usage_percent,
                        resolution_suggestions: vec![
                            "Consider reducing batch size".to_string(),
                            "Enable memory pooling".to_string(),
                            "Scale horizontally with more workers".to_string(),
                        ],
                    };
                    
                    let new_alert = {
                        let mut alerts = self.state.alerts.lock().unwrap();
                        alerts.push(alert);
                        alerts.last().unwrap().clone()
                    };
                    
                    Some(new_alert)
                } else {
                    None
                }
            } else {
                None
            }
        };
        
        if let Some(alert) = new_alert {
            self.broadcast_alert(&alert).await;
        }
    }
    
    async fn check_cpu_alerts(&self) {
        // CPU alerting logic
    }
    
    async fn check_temperature_alerts(&self) {
        // Temperature alerting logic
    }
    
    async fn check_throughput_alerts(&self) {
        // Throughput alerting logic
    }
    
    async fn check_error_rate_alerts(&self) {
        // Error rate alerting logic
    }

    async fn broadcast_alert(&self, alert: &PerformanceAlert) {
        let alert_msg = serde_json::to_string(&alert).unwrap();
        println!("📢 Broadcasting alert: {}", alert_msg);
        
        // Simplified alert broadcasting - in a real implementation,
        // this would use a proper WebSocket broadcasting mechanism
        // that doesn't hold locks across await points
    }
}

/// Regression detector for performance analysis
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    state: DashboardState,
}

impl RegressionDetector {
    pub fn new(state: DashboardState) -> Self {
        Self { state }
    }
    
    pub async fn analyze_regressions(&self) {
        self.analyze_throughput_regression().await;
        // Add other regression analyses here
    }
    
    async fn analyze_throughput_regression(&self) {
        let throughput_metrics = self.state.throughput_metrics.lock().unwrap();
        let mut analysis_map = self.state.regression_analysis.lock().unwrap();
        
        if throughput_metrics.len() < 100 {
            return; // Not enough data for analysis
        }
        
        // Simple baseline: last 50 vs. previous 50
        let recent_metrics = &throughput_metrics[throughput_metrics.len()-50..];
        let baseline_metrics = &throughput_metrics[throughput_metrics.len()-100..throughput_metrics.len()-50];
        
        let recent_avg = recent_metrics.iter().map(|m| m.average_processing_time_ms).sum::<f64>() / 50.0;
        let baseline_avg = baseline_metrics.iter().map(|m| m.average_processing_time_ms).sum::<f64>() / 50.0;
        
        let percentage_change = ((recent_avg - baseline_avg) / baseline_avg) * 100.0;
        
        let analysis = RegressionAnalysis {
            metric_name: "Average Processing Time".to_string(),
            baseline_value: baseline_avg,
            current_value: recent_avg,
            percentage_change,
            is_regression: percentage_change > 10.0, // 10% increase is a regression
            confidence_score: self.calculate_confidence_score(recent_metrics, baseline_metrics),
            trend_direction: if percentage_change > 0.0 { "Increasing".to_string() } else { "Decreasing".to_string() },
            recommended_actions: if percentage_change > 10.0 {
                vec![
                    "Review recent code changes for performance impact.".to_string(),
                    "Analyze query performance in database.".to_string(),
                    "Check for resource contention (CPU, memory).".to_string(),
                ]
            } else {
                Vec::new()
            },
        };
        
        analysis_map.insert("avg_processing_time".to_string(), analysis);
    }
    
    fn calculate_confidence_score(&self, recent: &[ThroughputMetrics], baseline: &[ThroughputMetrics]) -> f64 {
        // Simplified confidence score
        let recent_std_dev = self.calculate_std_dev(recent);
        let baseline_std_dev = self.calculate_std_dev(baseline);
        
        if (recent_std_dev + baseline_std_dev) > 0.0 {
            1.0 - (recent_std_dev / (recent_std_dev + baseline_std_dev))
        } else {
            0.5
        }
    }

    fn calculate_std_dev(&self, metrics: &[ThroughputMetrics]) -> f64 {
        let mean = metrics.iter().map(|m| m.average_processing_time_ms).sum::<f64>() / metrics.len() as f64;
        let variance = metrics.iter().map(|m| {
            let diff = m.average_processing_time_ms - mean;
            diff * diff
        }).sum::<f64>() / metrics.len() as f64;
        variance.sqrt()
    }
}


async fn handle_websocket(websocket: warp::ws::WebSocket, state: DashboardState) {
    println!("🔌 WebSocket client connected");

    let client_ws = Arc::new(Mutex::new(websocket));
    state.connected_clients.lock().unwrap().push(client_ws.clone());

    let client_rx_fut = async {
        // Simplified WebSocket handling to avoid split() issues
        println!("🔌 WebSocket client ready for messages");
    };
    client_rx_fut.await;

    println!("🔌 WebSocket client disconnected");
}


async fn get_metrics(state: DashboardState) -> Result<impl Reply, warp::Rejection> {
    let m3_metrics = state.m3_max_metrics.lock().unwrap();
    let sys_metrics = state.system_metrics.lock().unwrap();
    let throughput_metrics = state.throughput_metrics.lock().unwrap();
    
    let response = serde_json::json!({
        "m3_max_metrics": m3_metrics.clone(),
        "system_metrics": sys_metrics.clone(),
        "throughput_metrics": throughput_metrics.clone(),
    });
    
    Ok(warp::reply::json(&response))
}

async fn get_alerts(state: DashboardState) -> Result<impl Reply, warp::Rejection> {
    let alerts = state.alerts.lock().unwrap();
    Ok(warp::reply::json(&alerts.clone()))
}

async fn get_regression_analysis(state: DashboardState) -> Result<impl Reply, warp::Rejection> {
    let analysis = state.regression_analysis.lock().unwrap();
    Ok(warp::reply::json(&analysis.clone()))
}