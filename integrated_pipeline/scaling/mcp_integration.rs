// MCP Server Integration for Dynamic Scaling
// Phase 2 MCP Advanced Features - External Communication and Control
// Provides MCP server integration for scaling system monitoring and control

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, mpsc};
use tokio::time::{interval, sleep};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use tokio_tungstenite::{connect_async, WebSocketStream, MaybeTlsStream};
use tokio_tungstenite::tungstenite::Message;
use futures_util::{SinkExt, StreamExt};

use crate::scaling::{ScalingDecision, ScalingStatistics, WorkloadPattern, ResourceStatistics};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpScalingMessage {
    pub message_type: McpMessageType,
    pub timestamp: u64,
    pub data: serde_json::Value,
    pub correlation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpMessageType {
    ScalingDecision,
    WorkloadPattern,
    ResourceUpdate,
    PerformanceMetrics,
    Alert,
    StatusUpdate,
    Command,
    Response,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub metrics: HashMap<String, f64>,
    pub recommended_actions: Vec<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighMemoryUsage,
    HighCpuUsage,
    IpcLatencySpike,
    ThroughputDrop,
    QualityDegradation,
    EmergencyScaling,
    ResourceExhaustion,
    PatternAnomaly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCommand {
    pub command_type: CommandType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub correlation_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandType {
    ForceScaling,
    AdjustParameters,
    EmergencyStop,
    ResetCounters,
    GetStatistics,
    SetConfiguration,
    TriggerAnalysis,
    ExportMetrics,
}

pub struct McpScalingIntegration {
    config: McpIntegrationConfig,
    client: Client,
    websocket_tx: Option<mpsc::Sender<McpScalingMessage>>,
    command_rx: Arc<RwLock<Option<mpsc::Receiver<McpCommand>>>>,
    statistics_cache: Arc<RwLock<ScalingStatistics>>,
    alert_history: Arc<RwLock<Vec<ScalingAlert>>>,
    is_connected: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone)]
pub struct McpIntegrationConfig {
    pub websocket_url: String,
    pub http_base_url: String,
    pub reporting_interval_seconds: u64,
    pub reconnect_attempts: usize,
    pub reconnect_delay_seconds: u64,
    pub enable_websocket: bool,
    pub enable_http_fallback: bool,
    pub max_message_queue_size: usize,
    pub alert_debounce_seconds: u64,
}

impl Default for McpIntegrationConfig {
    fn default() -> Self {
        Self {
            websocket_url: "ws://127.0.0.1:8000/scaling".to_string(),
            http_base_url: "http://127.0.0.1:8001".to_string(),
            reporting_interval_seconds: 30,
            reconnect_attempts: 5,
            reconnect_delay_seconds: 10,
            enable_websocket: true,
            enable_http_fallback: true,
            max_message_queue_size: 1000,
            alert_debounce_seconds: 60,
        }
    }
}

impl McpScalingIntegration {
    pub fn new(config: McpIntegrationConfig) -> Self {
        Self {
            client: Client::new(),
            config,
            websocket_tx: None,
            command_rx: Arc::new(RwLock::new(None)),
            statistics_cache: Arc::new(RwLock::new(ScalingStatistics {
                current_agents: 2,
                scaling_effectiveness: 0.0,
                total_scaling_decisions: 0,
                memory_utilization_percent: 0.0,
                cpu_utilization_percent: 0.0,
                active_processes: 0,
                uptime_seconds: 0,
            })),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            is_connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the MCP integration system
    pub async fn start(&mut self) -> Result<(), McpIntegrationError> {
        // Start WebSocket connection if enabled
        if self.config.enable_websocket {
            self.start_websocket_connection().await?;
        }

        // Start periodic reporting
        let reporting_task = self.start_reporting_task();

        // Start command processing
        let command_task = self.start_command_processing();

        // Run all tasks concurrently
        tokio::select! {
            result = reporting_task => {
                log::error!("Reporting task ended: {:?}", result);
            }
            result = command_task => {
                log::error!("Command processing task ended: {:?}", result);
            }
        }

        Ok(())
    }

    async fn start_websocket_connection(&mut self) -> Result<(), McpIntegrationError> {
        let (tx, mut rx) = mpsc::channel(self.config.max_message_queue_size);
        self.websocket_tx = Some(tx);

        let ws_url = self.config.websocket_url.clone();
        let reconnect_attempts = self.config.reconnect_attempts;
        let reconnect_delay = Duration::from_secs(self.config.reconnect_delay_seconds);
        let is_connected = self.is_connected.clone();

        tokio::spawn(async move {
            let mut attempts = 0;
            
            while attempts < reconnect_attempts {
                match Self::connect_websocket(&ws_url).await {
                    Ok(mut ws_stream) => {
                        *is_connected.write().await = true;
                        log::info!("WebSocket connected to MCP server");
                        
                        // Handle WebSocket communication
                        loop {
                            tokio::select! {
                                // Send messages from queue
                                msg = rx.recv() => {
                                    match msg {
                                        Some(message) => {
                                            let json_msg = match serde_json::to_string(&message) {
                                                Ok(json) => json,
                                                Err(e) => {
                                                    log::error!("Failed to serialize message: {}", e);
                                                    continue;
                                                }
                                            };
                                            
                                            if let Err(e) = ws_stream.send(Message::Text(json_msg)).await {
                                                log::error!("Failed to send WebSocket message: {}", e);
                                                break;
                                            }
                                        }
                                        None => break, // Channel closed
                                    }
                                }
                                
                                // Receive messages (commands)
                                ws_msg = ws_stream.next() => {
                                    match ws_msg {
                                        Some(Ok(Message::Text(text))) => {
                                            if let Ok(command) = serde_json::from_str::<McpCommand>(&text) {
                                                // Process command (would be implemented)
                                                log::debug!("Received MCP command: {:?}", command.command_type);
                                            }
                                        }
                                        Some(Ok(Message::Close(_))) => {
                                            log::info!("WebSocket connection closed by server");
                                            break;
                                        }
                                        Some(Err(e)) => {
                                            log::error!("WebSocket error: {}", e);
                                            break;
                                        }
                                        None => break,
                                    }
                                }
                            }
                        }
                        
                        *is_connected.write().await = false;
                        log::warn!("WebSocket connection lost, attempting to reconnect...");
                    }
                    Err(e) => {
                        log::error!("Failed to connect to WebSocket: {}", e);
                        attempts += 1;
                        if attempts < reconnect_attempts {
                            sleep(reconnect_delay).await;
                        }
                    }
                }
            }
            
            log::error!("Exhausted WebSocket reconnection attempts");
        });

        Ok(())
    }

    async fn connect_websocket(url: &str) -> Result<WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>, McpIntegrationError> {
        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| McpIntegrationError::ConnectionFailed(e.to_string()))?;
        Ok(ws_stream)
    }

    async fn start_reporting_task(&self) -> Result<(), McpIntegrationError> {
        let mut interval = interval(Duration::from_secs(self.config.reporting_interval_seconds));
        let statistics_cache = self.statistics_cache.clone();
        let websocket_tx = self.websocket_tx.clone();
        let client = self.client.clone();
        let http_url = format!("{}/scaling/metrics", self.config.http_base_url);
        let enable_websocket = self.config.enable_websocket;
        let enable_http_fallback = self.config.enable_http_fallback;
        let is_connected = self.is_connected.clone();

        tokio::spawn(async move {
            loop {
                interval.tick().await;
                
                let stats = statistics_cache.read().await.clone();
                let message = McpScalingMessage {
                    message_type: McpMessageType::PerformanceMetrics,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    data: serde_json::to_value(&stats).unwrap(),
                    correlation_id: None,
                };

                let mut sent = false;

                // Try WebSocket first if enabled and connected
                if enable_websocket && *is_connected.read().await {
                    if let Some(ref tx) = websocket_tx {
                        if tx.send(message.clone()).await.is_ok() {
                            sent = true;
                        }
                    }
                }

                // Fallback to HTTP if WebSocket failed or not available
                if !sent && enable_http_fallback {
                    match client.post(&http_url)
                        .json(&message)
                        .timeout(Duration::from_secs(10))
                        .send()
                        .await
                    {
                        Ok(response) => {
                            if response.status().is_success() {
                                sent = true;
                            } else {
                                log::warn!("HTTP reporting failed with status: {}", response.status());
                            }
                        }
                        Err(e) => {
                            log::warn!("HTTP reporting failed: {}", e);
                        }
                    }
                }

                if !sent {
                    log::warn!("Failed to send scaling metrics via all channels");
                }
            }
        });

        Ok(())
    }

    async fn start_command_processing(&self) -> Result<(), McpIntegrationError> {
        // This would process incoming commands from the MCP server
        // For now, just a placeholder that logs commands
        let command_rx = self.command_rx.clone();
        
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(1)).await;
                
                // Process any queued commands
                if let Some(mut rx) = command_rx.write().await.take() {
                    while let Ok(command) = rx.try_recv() {
                        log::info!("Processing MCP command: {:?}", command.command_type);
                        // Command processing would be implemented here
                    }
                    *command_rx.write().await = Some(rx);
                }
            }
        });

        Ok(())
    }

    /// Send a scaling decision notification to MCP server
    pub async fn notify_scaling_decision(&self, decision: &ScalingDecision) {
        let message = McpScalingMessage {
            message_type: McpMessageType::ScalingDecision,
            timestamp: decision.timestamp,
            data: serde_json::to_value(decision).unwrap(),
            correlation_id: None,
        };

        self.send_message(message).await;
    }

    /// Send a workload pattern notification to MCP server
    pub async fn notify_workload_pattern(&self, pattern: &WorkloadPattern) {
        let message = McpScalingMessage {
            message_type: McpMessageType::WorkloadPattern,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            data: serde_json::to_value(pattern).unwrap(),
            correlation_id: None,
        };

        self.send_message(message).await;
    }

    /// Send a resource update notification to MCP server
    pub async fn notify_resource_update(&self, resources: &ResourceStatistics) {
        let message = McpScalingMessage {
            message_type: McpMessageType::ResourceUpdate,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            data: serde_json::to_value(resources).unwrap(),
            correlation_id: None,
        };

        self.send_message(message).await;
    }

    /// Send an alert to MCP server
    pub async fn send_alert(&self, alert: ScalingAlert) {
        // Check for alert debouncing
        if self.should_debounce_alert(&alert).await {
            return;
        }

        let message = McpScalingMessage {
            message_type: McpMessageType::Alert,
            timestamp: alert.timestamp,
            data: serde_json::to_value(&alert).unwrap(),
            correlation_id: None,
        };

        // Add to alert history
        let mut history = self.alert_history.write().await;
        history.push(alert);
        if history.len() > 100 {
            history.remove(0);
        }

        self.send_message(message).await;
    }

    async fn should_debounce_alert(&self, alert: &ScalingAlert) -> bool {
        let history = self.alert_history.read().await;
        let debounce_threshold = alert.timestamp - (self.config.alert_debounce_seconds * 1000);
        
        history.iter().any(|prev_alert| {
            prev_alert.timestamp > debounce_threshold && 
            std::mem::discriminant(&prev_alert.alert_type) == std::mem::discriminant(&alert.alert_type)
        })
    }

    async fn send_message(&self, message: McpScalingMessage) {
        if let Some(ref tx) = self.websocket_tx {
            if let Err(e) = tx.send(message.clone()).await {
                log::warn!("Failed to queue WebSocket message: {}", e);
                
                // Fallback to HTTP if enabled
                if self.config.enable_http_fallback {
                    self.send_http_message(&message).await;
                }
            }
        } else if self.config.enable_http_fallback {
            self.send_http_message(&message).await;
        }
    }

    async fn send_http_message(&self, message: &McpScalingMessage) {
        let endpoint = match message.message_type {
            McpMessageType::ScalingDecision => "scaling/decisions",
            McpMessageType::WorkloadPattern => "scaling/patterns",
            McpMessageType::ResourceUpdate => "scaling/resources",
            McpMessageType::Alert => "scaling/alerts",
            McpMessageType::PerformanceMetrics => "scaling/metrics",
            _ => "scaling/events",
        };
        
        let url = format!("{}/{}", self.config.http_base_url, endpoint);
        
        match self.client.post(&url)
            .json(message)
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    log::warn!("HTTP message failed with status: {}", response.status());
                }
            }
            Err(e) => {
                log::warn!("HTTP message send failed: {}", e);
            }
        }
    }

    /// Update cached statistics
    pub async fn update_statistics(&self, stats: ScalingStatistics) {
        *self.statistics_cache.write().await = stats;
    }

    /// Get connection status
    pub async fn is_connected(&self) -> bool {
        *self.is_connected.read().await
    }

    /// Get alert history
    pub async fn get_alert_history(&self) -> Vec<ScalingAlert> {
        self.alert_history.read().await.clone()
    }

    /// Create emergency alert
    pub fn create_emergency_alert(
        alert_type: AlertType,
        message: String,
        metrics: HashMap<String, f64>,
    ) -> ScalingAlert {
        ScalingAlert {
            alert_type,
            severity: AlertSeverity::Emergency,
            message,
            metrics,
            recommended_actions: match alert_type {
                AlertType::HighMemoryUsage => vec![
                    "Scale down non-critical processes".to_string(),
                    "Trigger garbage collection".to_string(),
                    "Reduce cache sizes".to_string(),
                ],
                AlertType::HighCpuUsage => vec![
                    "Reduce concurrency".to_string(),
                    "Scale down processing agents".to_string(),
                    "Throttle incoming requests".to_string(),
                ],
                AlertType::ResourceExhaustion => vec![
                    "Emergency scale down".to_string(),
                    "Stop non-essential processes".to_string(),
                    "Clear processing queues".to_string(),
                ],
                _ => vec!["Review system configuration".to_string()],
            },
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum McpIntegrationError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("WebSocket error: {0}")]
    WebSocketError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_integration_creation() {
        let config = McpIntegrationConfig::default();
        let integration = McpScalingIntegration::new(config);
        assert!(!integration.is_connected().await);
    }

    #[tokio::test]
    async fn test_alert_debouncing() {
        let config = McpIntegrationConfig {
            alert_debounce_seconds: 60,
            ..Default::default()
        };
        let integration = McpScalingIntegration::new(config);
        
        let alert = McpScalingIntegration::create_emergency_alert(
            AlertType::HighMemoryUsage,
            "Test alert".to_string(),
            HashMap::new(),
        );
        
        // First alert should not be debounced
        assert!(!integration.should_debounce_alert(&alert).await);
        
        // Add alert to history
        integration.send_alert(alert.clone()).await;
        
        // Second alert within debounce period should be debounced
        let second_alert = McpScalingIntegration::create_emergency_alert(
            AlertType::HighMemoryUsage,
            "Second test alert".to_string(),
            HashMap::new(),
        );
        assert!(integration.should_debounce_alert(&second_alert).await);
    }
}