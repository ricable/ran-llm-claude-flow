//! Real-time performance dashboard
//! 
//! Provides a web-based interface for monitoring system performance,
//! bottlenecks, and optimizations in real-time.

use anyhow::Result;
use chrono::{DateTime, Utc};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};
use tokio::sync::{broadcast, RwLock};
use warp::{
    filters::ws::{Message, WebSocket, Ws},
    http::Response,
    hyper::body::Bytes,
    Filter,
};

use crate::{
    bottleneck_analyzer::{Bottleneck, BottleneckAnalyzer},
    config::MonitoringConfig,
    metrics::{MetricsCollector, SystemMetrics},
    optimizer::{AdaptiveOptimizer, Optimization},
};

/// High-performance web dashboard server
pub struct DashboardServer {
    config: MonitoringConfig,
    metrics_collector: Arc<MetricsCollector>,
    bottleneck_analyzer: Arc<BottleneckAnalyzer>,
    adaptive_optimizer: Arc<AdaptiveOptimizer>,
    websocket_clients: Arc<RwLock<Vec<WebSocketClient>>>,
    dashboard_state: Arc<RwLock<DashboardState>>,
}

impl DashboardServer {
    pub fn new(
        config: MonitoringConfig,
        metrics_collector: Arc<MetricsCollector>,
        bottleneck_analyzer: Arc<BottleneckAnalyzer>,
        adaptive_optimizer: Arc<AdaptiveOptimizer>,
    ) -> Self {
        Self {
            config,
            metrics_collector,
            bottleneck_analyzer,
            adaptive_optimizer,
            websocket_clients: Arc::new(RwLock::new(Vec::new())),
            dashboard_state: Arc::new(RwLock::new(DashboardState::new())),
        }
    }

    pub async fn start(&self) -> Result<()> {
        if !self.config.dashboard.enabled {
            tracing::info!("Dashboard disabled in configuration");
            return Ok(());
        }

        let bind_addr: SocketAddr = format!("{}:{}", 
            self.config.dashboard.bind_address, 
            self.config.dashboard.port
        ).parse()?;

        tracing::info!("Starting dashboard server on {}", bind_addr);

        // Start background data updater
        let updater = self.clone_for_task();
        tokio::spawn(async move {
            updater.data_update_loop().await;
        });

        // WebSocket clients cleanup
        let cleanup = self.clone_for_task();
        tokio::spawn(async move {
            cleanup.websocket_cleanup_loop().await;
        });

        // Build warp routes
        let routes = self.build_routes();

        // Start server
        warp::serve(routes)
            .run(bind_addr)
            .await;

        Ok(())
    }

    fn build_routes(&self) -> impl Filter<Extract = impl warp::Reply> + Clone {
        let static_files = warp::path("static")
            .and(warp::fs::dir("integrated_pipeline/monitoring/web/static"));

        let api_routes = self.build_api_routes();
        let websocket_route = self.build_websocket_route();

        // Main dashboard page
        let dashboard_page = warp::path::end()
            .map(|| {
                warp::reply::html(include_str!("../web/templates/dashboard.html"))
            });

        dashboard_page
            .or(static_files)
            .or(api_routes)
            .or(websocket_route)
            .with(warp::cors().allow_any_origin())
    }

    fn build_api_routes(&self) -> impl Filter<Extract = impl warp::Reply> + Clone {
        let metrics_collector = self.metrics_collector.clone();
        let bottleneck_analyzer = self.bottleneck_analyzer.clone();
        let adaptive_optimizer = self.adaptive_optimizer.clone();
        let dashboard_state = self.dashboard_state.clone();

        // GET /api/metrics - Current metrics
        let get_metrics = warp::path!("api" / "metrics")
            .and(warp::get())
            .and_then(move || {
                let collector = metrics_collector.clone();
                async move {
                    match collector.get_current_metrics().await {
                        Ok(metrics) => Ok::<_, warp::Rejection>(warp::reply::json(&ApiResponse::success(metrics))),
                        Err(e) => Ok::<_, warp::Rejection>(warp::reply::json(&ApiResponse::<()>::error(&e.to_string()))),
                    }
                }
            });

        // GET /api/bottlenecks - Current bottlenecks
        let bottleneck_analyzer_clone = bottleneck_analyzer.clone();
        let get_bottlenecks = warp::path!("api" / "bottlenecks")
            .and(warp::get())
            .and_then(move || {
                let analyzer = bottleneck_analyzer_clone.clone();
                async move {
                    match analyzer.detect_current_bottlenecks().await {
                        Ok(bottlenecks) => Ok::<_, warp::Rejection>(warp::reply::json(&ApiResponse::success(bottlenecks))),
                        Err(e) => Ok::<_, warp::Rejection>(warp::reply::json(&ApiResponse::<()>::error(&e.to_string()))),
                    }
                }
            });

        // GET /api/optimizations - Active optimizations
        let adaptive_optimizer_clone = adaptive_optimizer.clone();
        let get_optimizations = warp::path!("api" / "optimizations")
            .and(warp::get())
            .and_then(move || {
                let optimizer = adaptive_optimizer_clone.clone();
                async move {
                    match optimizer.get_active_optimizations().await {
                        Ok(optimizations) => Ok::<_, warp::Rejection>(warp::reply::json(&ApiResponse::success(optimizations))),
                        Err(e) => Ok::<_, warp::Rejection>(warp::reply::json(&ApiResponse::<()>::error(&e.to_string()))),
                    }
                }
            });

        // GET /api/dashboard - Dashboard state
        let get_dashboard_state = warp::path!("api" / "dashboard")
            .and(warp::get())
            .and_then(move || {
                let state = dashboard_state.clone();
                async move {
                    let dashboard_data = state.read().await;
                    Result::<_, warp::Rejection>::Ok(warp::reply::json(&ApiResponse::success(&*dashboard_data)))
                }
            });

        // GET /api/health - Health check
        let get_health = warp::path!("api" / "health")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&ApiResponse::success(serde_json::json!({
                    "status": "healthy",
                    "timestamp": Utc::now()
                })))
            });

        get_metrics
            .or(get_bottlenecks)
            .or(get_optimizations)
            .or(get_dashboard_state)
            .or(get_health)
    }

    fn build_websocket_route(&self) -> impl Filter<Extract = impl warp::Reply> + Clone {
        let clients = self.websocket_clients.clone();
        let dashboard_state = self.dashboard_state.clone();

        warp::path("ws")
            .and(warp::ws())
            .and_then(move |ws: Ws| {
                let clients_clone = clients.clone();
                let state_clone = dashboard_state.clone();
                
                async move {
                    Ok::<_, warp::Rejection>(ws.on_upgrade(move |websocket| {
                        handle_websocket_connection(websocket, clients_clone, state_clone)
                    }))
                }
            })
    }

    async fn data_update_loop(&self) {
        let mut interval = tokio::time::interval(
            Duration::from_secs(self.config.dashboard.update_interval_seconds)
        );

        while interval.tick().await.elapsed().is_ok() {
            if let Err(e) = self.update_dashboard_data().await {
                tracing::error!("Dashboard data update failed: {}", e);
            }
        }
    }

    async fn update_dashboard_data(&self) -> Result<()> {
        // Collect current data
        let metrics = self.metrics_collector.get_current_metrics().await?;
        let bottlenecks = self.bottleneck_analyzer.detect_current_bottlenecks().await?;
        let optimizations = self.adaptive_optimizer.get_active_optimizations().await?;

        // Calculate dashboard metrics
        let dashboard_metrics = DashboardMetrics::from_system_metrics(&metrics, &bottlenecks, &optimizations);

        // Update dashboard state
        {
            let mut state = self.dashboard_state.write().await;
            state.update(dashboard_metrics, bottlenecks, optimizations);
        }

        // Send updates to WebSocket clients
        self.broadcast_to_clients().await?;

        Ok(())
    }

    async fn broadcast_to_clients(&self) -> Result<()> {
        let state = self.dashboard_state.read().await;
        let update_message = WebSocketMessage::DashboardUpdate {
            timestamp: Utc::now(),
            metrics: state.current_metrics.clone(),
            bottlenecks: state.current_bottlenecks.clone(),
            optimizations: state.current_optimizations.clone(),
        };

        let message_json = serde_json::to_string(&update_message)?;
        let clients = self.websocket_clients.read().await;
        
        for client in clients.iter() {
            if let Err(e) = client.sender.send(message_json.clone()) {
                tracing::debug!("Failed to send WebSocket message to client: {}", e);
            }
        }

        Ok(())
    }

    async fn websocket_cleanup_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;
            
            let mut clients = self.websocket_clients.write().await;
            clients.retain(|client| !client.sender.is_closed());
            
            if clients.len() > 100 { // Prevent memory leaks
                clients.truncate(50);
                tracing::warn!("WebSocket client limit exceeded, disconnected oldest clients");
            }
        }
    }

    pub fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_collector: self.metrics_collector.clone(),
            bottleneck_analyzer: self.bottleneck_analyzer.clone(),
            adaptive_optimizer: self.adaptive_optimizer.clone(),
            websocket_clients: self.websocket_clients.clone(),
            dashboard_state: self.dashboard_state.clone(),
        }
    }
}

async fn handle_websocket_connection(
    websocket: WebSocket,
    clients: Arc<RwLock<Vec<WebSocketClient>>>,
    _dashboard_state: Arc<RwLock<DashboardState>>,
) {
    let (mut ws_sender, mut ws_receiver) = websocket.split();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    // Add client to list
    {
        let mut client_list = clients.write().await;
        client_list.push(WebSocketClient {
            id: uuid::Uuid::new_v4(),
            connected_at: Utc::now(),
            sender: tx,
        });
    }

    // Handle outgoing messages
    let send_task = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if ws_sender.send(Message::text(message)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    let receive_task = tokio::spawn(async move {
        while let Some(result) = ws_receiver.next().await {
            match result {
                Ok(message) => {
                    if message.is_text() {
                        // Handle client messages (e.g., subscribe to specific metrics)
                        tracing::debug!("Received WebSocket message: {:?}", message);
                    }
                },
                Err(e) => {
                    tracing::debug!("WebSocket error: {}", e);
                    break;
                }
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {},
        _ = receive_task => {},
    }
}

/// Dashboard state management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    pub current_metrics: DashboardMetrics,
    pub current_bottlenecks: Vec<Bottleneck>,
    pub current_optimizations: Vec<Optimization>,
    pub historical_data: Vec<DashboardSnapshot>,
    pub last_updated: DateTime<Utc>,
}

impl DashboardState {
    pub fn new() -> Self {
        Self {
            current_metrics: DashboardMetrics::default(),
            current_bottlenecks: Vec::new(),
            current_optimizations: Vec::new(),
            historical_data: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    pub fn update(
        &mut self,
        metrics: DashboardMetrics,
        bottlenecks: Vec<Bottleneck>,
        optimizations: Vec<Optimization>,
    ) {
        // Store historical snapshot
        if self.historical_data.len() >= 1000 {
            self.historical_data.remove(0);
        }
        
        self.historical_data.push(DashboardSnapshot {
            timestamp: self.last_updated,
            metrics: self.current_metrics.clone(),
            bottleneck_count: self.current_bottlenecks.len(),
            optimization_count: self.current_optimizations.len(),
        });

        // Update current state
        self.current_metrics = metrics;
        self.current_bottlenecks = bottlenecks;
        self.current_optimizations = optimizations;
        self.last_updated = Utc::now();
    }
}

/// Simplified metrics for dashboard display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub cpu_utilization: f32,
    pub memory_utilization: f64,
    pub document_processing_rate: f64,
    pub ipc_latency_p99: f64,
    pub error_rate: f64,
    pub throughput_trend: f64, // -1 to +1 (negative = declining, positive = improving)
    pub bottleneck_severity_score: f64, // 0-1 scale
    pub optimization_effectiveness: f64, // 0-1 scale
    pub system_health_score: f64, // 0-1 scale (1 = excellent, 0 = critical)
}

impl Default for DashboardMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            document_processing_rate: 0.0,
            ipc_latency_p99: 0.0,
            error_rate: 0.0,
            throughput_trend: 0.0,
            bottleneck_severity_score: 0.0,
            optimization_effectiveness: 0.0,
            system_health_score: 1.0,
        }
    }
}

impl DashboardMetrics {
    pub fn from_system_metrics(
        metrics: &SystemMetrics,
        bottlenecks: &[Bottleneck],
        optimizations: &[Optimization],
    ) -> Self {
        let memory_utilization = if metrics.memory.total_bytes > 0 {
            (metrics.memory.used_bytes as f64 / metrics.memory.total_bytes as f64) * 100.0
        } else {
            0.0
        };

        let error_rate = if metrics.documents_processed > 0 {
            (metrics.errors_total as f64 / metrics.documents_processed as f64) * 100.0
        } else {
            0.0
        };

        // Calculate bottleneck severity score
        let bottleneck_severity_score = if bottlenecks.is_empty() {
            0.0
        } else {
            bottlenecks.iter().map(|b| b.impact_score).sum::<f64>() / bottlenecks.len() as f64
        };

        // Calculate optimization effectiveness
        let optimization_effectiveness = if optimizations.is_empty() {
            1.0
        } else {
            optimizations.iter().map(|o| o.expected_improvement).sum::<f64>() / optimizations.len() as f64
        };

        // Calculate overall system health score
        let health_factors = vec![
            (100.0 - metrics.cpu_usage.utilization_percent as f64) / 100.0,
            (100.0 - memory_utilization) / 100.0,
            (metrics.document_processing_rate / 30.0).min(1.0),
            (10.0 - metrics.ipc_latency_p99).max(0.0) / 10.0,
            (1.0 - error_rate / 100.0).max(0.0),
        ];
        let system_health_score = health_factors.iter().sum::<f64>() / health_factors.len() as f64;

        Self {
            cpu_utilization: metrics.cpu_usage.utilization_percent,
            memory_utilization,
            document_processing_rate: metrics.document_processing_rate,
            ipc_latency_p99: metrics.ipc_latency_p99,
            error_rate,
            throughput_trend: 0.0, // TODO: Calculate from historical data
            bottleneck_severity_score,
            optimization_effectiveness,
            system_health_score,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub timestamp: DateTime<Utc>,
    pub metrics: DashboardMetrics,
    pub bottleneck_count: usize,
    pub optimization_count: usize,
}

#[derive(Debug)]
struct WebSocketClient {
    id: uuid::Uuid,
    connected_at: DateTime<Utc>,
    sender: tokio::sync::mpsc::UnboundedSender<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum WebSocketMessage {
    DashboardUpdate {
        timestamp: DateTime<Utc>,
        metrics: DashboardMetrics,
        bottlenecks: Vec<Bottleneck>,
        optimizations: Vec<Optimization>,
    },
    Alert {
        timestamp: DateTime<Utc>,
        severity: String,
        message: String,
        details: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
    timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: Utc::now(),
        }
    }
}

impl ApiResponse<()> {
    fn error(message: &str) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.to_string()),
            timestamp: Utc::now(),
        }
    }
}