/*!
# MCP Server Implementation

Model Context Protocol server for coordinating Rust-Python pipeline operations.
Provides WebSocket-based communication with load balancing and fault tolerance.
*/

use crate::mcp::{ClientType, McpConnection, McpError, McpMessage};
use crate::{PipelineError, Result};
use futures::future::join_all;
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, Duration};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// MCP server configuration
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub heartbeat_interval: Duration,
    pub connection_timeout: Duration,
    pub enable_compression: bool,
    pub max_message_size: usize,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8700,
            max_connections: 1000,
            heartbeat_interval: Duration::from_secs(30),
            connection_timeout: Duration::from_secs(300),
            enable_compression: false,
            max_message_size: 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Connection state for a WebSocket client
#[derive(Debug)]
pub struct ConnectionState {
    pub connection: McpConnection,
    pub sender: mpsc::UnboundedSender<Message>,
    pub last_heartbeat: AtomicU64,
    pub message_count: AtomicU64,
    pub is_active: AtomicBool,
}

/// Server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    pub total_connections: u64,
    pub active_connections: usize,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub uptime_seconds: u64,
    pub average_message_latency_ms: f64,
    pub connections_by_type: HashMap<String, usize>,
}

/// MCP Server for coordinating pipeline operations
#[derive(Clone)]
pub struct McpServer {
    config: McpServerConfig,
    connections: Arc<RwLock<HashMap<Uuid, ConnectionState>>>,
    broadcast_sender: broadcast::Sender<McpMessage>,
    stats: Arc<ServerStats>,
    is_running: Arc<AtomicBool>,
    start_time: u64,
    message_handler: Arc<dyn MessageHandler + Send + Sync>,
}

/// Message handler trait for processing MCP messages
#[async_trait::async_trait]
pub trait MessageHandler {
    async fn handle_message(
        &self,
        connection_id: Uuid,
        message: McpMessage,
    ) -> Result<Option<McpMessage>>;
    async fn handle_connection_established(&self, connection: &McpConnection) -> Result<()>;
    async fn handle_connection_closed(&self, connection_id: Uuid) -> Result<()>;
}

/// Default message handler implementation
pub struct DefaultMessageHandler {
    pipeline_state: Arc<RwLock<PipelineState>>,
}

#[derive(Debug, Default)]
struct PipelineState {
    active_tasks: HashMap<Uuid, TaskInfo>,
    worker_status: HashMap<String, WorkerStatus>,
    performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct TaskInfo {
    task_id: Uuid,
    task_type: String,
    assigned_worker: Option<String>,
    start_time: u64,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct WorkerStatus {
    worker_id: String,
    client_type: ClientType,
    status: String,
    last_seen: u64,
    tasks_completed: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
struct PerformanceMetrics {
    total_tasks_processed: u64,
    average_task_duration_ms: f64,
    throughput_tasks_per_minute: f64,
    error_rate_percent: f64,
}

impl McpServer {
    /// Create new MCP server
    pub fn new() -> Self {
        Self::with_config(McpServerConfig::default())
    }

    /// Create MCP server with custom configuration
    pub fn with_config(config: McpServerConfig) -> Self {
        let (broadcast_sender, _) = broadcast::channel(1000);
        let stats = ServerStats {
            total_connections: 0,
            active_connections: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            uptime_seconds: 0,
            average_message_latency_ms: 0.0,
            connections_by_type: HashMap::new(),
        };

        let message_handler = Arc::new(DefaultMessageHandler {
            pipeline_state: Arc::new(RwLock::new(PipelineState::default())),
        });

        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            broadcast_sender,
            stats: Arc::new(stats),
            is_running: Arc::new(AtomicBool::new(false)),
            start_time: current_timestamp(),
            message_handler,
        }
    }

    /// Start the MCP server
    pub async fn start(&self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| PipelineError::Mcp(format!("Failed to bind to {}: {}", addr, e)))?;

        info!("MCP Server started on {}", addr);
        self.is_running.store(true, Ordering::Relaxed);

        // Start background tasks
        self.start_heartbeat_task().await;
        self.start_stats_task().await;

        // Main server loop
        while self.is_running.load(Ordering::Relaxed) {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    if self.connections.read().len() >= self.config.max_connections {
                        warn!(
                            "Connection limit reached, rejecting connection from {}",
                            peer_addr
                        );
                        continue;
                    }

                    let connections = self.connections.clone();
                    let broadcast_sender = self.broadcast_sender.clone();
                    let config = self.config.clone();
                    let message_handler = self.message_handler.clone();
                    let is_running = self.is_running.clone();

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(
                            stream,
                            peer_addr,
                            connections,
                            broadcast_sender,
                            config,
                            message_handler,
                            is_running,
                        )
                        .await
                        {
                            error!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }

        Ok(())
    }

    /// Stop the MCP server
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping MCP server");
        self.is_running.store(false, Ordering::Relaxed);

        // Close all active connections
        let connection_ids: Vec<Uuid> = { self.connections.read().keys().copied().collect() };

        for connection_id in connection_ids {
            self.close_connection(connection_id).await?;
        }

        info!("MCP server stopped");
        Ok(())
    }

    /// Send message to specific connection
    pub async fn send_to_connection(&self, connection_id: Uuid, message: McpMessage) -> Result<()> {
        let connection_state = {
            self.connections
                .read()
                .get(&connection_id)
                .map(|cs| cs.sender.clone())
        };

        if let Some(sender) = connection_state {
            let message_json = serde_json::to_string(&message)
                .map_err(|e| PipelineError::Mcp(format!("Failed to serialize message: {}", e)))?;

            sender
                .send(Message::Text(message_json))
                .map_err(|e| PipelineError::Mcp(format!("Failed to send message: {}", e)))?;

            debug!("Sent message to connection {}", connection_id);
            Ok(())
        } else {
            Err(PipelineError::Mcp(format!(
                "Connection {} not found",
                connection_id
            )))
        }
    }

    /// Broadcast message to all connections
    pub async fn broadcast(&self, message: McpMessage) -> Result<()> {
        let _ = self.broadcast_sender.send(message);
        debug!("Broadcast message to all connections");
        Ok(())
    }

    /// Broadcast message to connections of specific type
    pub async fn broadcast_to_type(
        &self,
        client_type: ClientType,
        message: McpMessage,
    ) -> Result<()> {
        let connections = self.connections.read();
        let target_connections: Vec<mpsc::UnboundedSender<Message>> = connections
            .values()
            .filter(|cs| cs.connection.client_type == client_type)
            .map(|cs| cs.sender.clone())
            .collect();

        drop(connections);

        let message_json = serde_json::to_string(&message)
            .map_err(|e| PipelineError::Mcp(format!("Failed to serialize message: {}", e)))?;

        let send_tasks = target_connections.into_iter().map(|sender| {
            let msg = Message::Text(message_json.clone());
            async move {
                if let Err(e) = sender.send(msg) {
                    warn!("Failed to send broadcast message: {}", e);
                }
            }
        });

        join_all(send_tasks).await;
        debug!("Broadcast message to {:?} connections", client_type);
        Ok(())
    }

    /// Get active connections
    pub async fn get_connections(&self) -> Vec<McpConnection> {
        self.connections
            .read()
            .values()
            .filter(|cs| cs.is_active.load(Ordering::Relaxed))
            .map(|cs| cs.connection.clone())
            .collect()
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStats {
        let connections = self.connections.read();
        let active_connections = connections
            .values()
            .filter(|cs| cs.is_active.load(Ordering::Relaxed))
            .count();

        let mut connections_by_type = HashMap::new();
        for cs in connections.values() {
            let client_type = format!("{:?}", cs.connection.client_type);
            *connections_by_type.entry(client_type).or_insert(0) += 1;
        }

        let uptime_seconds = current_timestamp() - self.start_time;

        ServerStats {
            total_connections: self.stats.total_connections,
            active_connections,
            total_messages_sent: self.stats.total_messages_sent,
            total_messages_received: self.stats.total_messages_received,
            uptime_seconds,
            average_message_latency_ms: self.stats.average_message_latency_ms,
            connections_by_type,
        }
    }

    /// Close a specific connection
    async fn close_connection(&self, connection_id: Uuid) -> Result<()> {
        let connection_state = self.connections.write().remove(&connection_id);

        if let Some(cs) = connection_state {
            cs.is_active.store(false, Ordering::Relaxed);
            let _ = self
                .message_handler
                .handle_connection_closed(connection_id)
                .await;
            info!("Closed connection: {}", connection_id);
        }

        Ok(())
    }

    /// Handle individual WebSocket connection
    async fn handle_connection(
        stream: TcpStream,
        peer_addr: SocketAddr,
        connections: Arc<RwLock<HashMap<Uuid, ConnectionState>>>,
        broadcast_receiver: broadcast::Sender<McpMessage>,
        config: McpServerConfig,
        message_handler: Arc<dyn MessageHandler + Send + Sync>,
        is_running: Arc<AtomicBool>,
    ) -> Result<()> {
        let ws_stream = accept_async(stream)
            .await
            .map_err(|e| PipelineError::Mcp(format!("WebSocket handshake failed: {}", e)))?;

        let connection_id = Uuid::new_v4();
        let connection = McpConnection {
            connection_id,
            client_type: ClientType::ExternalTool, // Will be updated during handshake
            capabilities: Vec::new(),
            last_heartbeat: chrono::Utc::now(),
        };

        info!(
            "New WebSocket connection: {} from {}",
            connection_id, peer_addr
        );

        let (mut ws_sender, mut ws_receiver) = ws_stream.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

        let connection_state = ConnectionState {
            connection: connection.clone(),
            sender: tx,
            last_heartbeat: AtomicU64::new(current_timestamp()),
            message_count: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        };

        // Store connection
        connections.write().insert(connection_id, connection_state);

        // Notify handler of new connection
        let _ = message_handler
            .handle_connection_established(&connection)
            .await;

        // Spawn task to handle outgoing messages
        let connections_clone = connections.clone();
        let outgoing_task = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if let Err(e) = ws_sender.send(message).await {
                    error!("Failed to send WebSocket message: {}", e);
                    break;
                }
            }
        });

        // Spawn task to handle broadcast messages
        let tx_clone = {
            connections
                .read()
                .get(&connection_id)
                .map(|cs| cs.sender.clone())
        };

        let broadcast_task = if let Some(tx) = tx_clone {
            let mut broadcast_rx = broadcast_receiver.subscribe();
            Some(tokio::spawn(async move {
                while let Ok(message) = broadcast_rx.recv().await {
                    if let Ok(message_json) = serde_json::to_string(&message) {
                        let _ = tx.send(Message::Text(message_json));
                    }
                }
            }))
        } else {
            None
        };

        // Main message processing loop
        while is_running.load(Ordering::Relaxed) {
            match tokio::time::timeout(config.connection_timeout, ws_receiver.next()).await {
                Ok(Some(Ok(message))) => {
                    if let Err(e) = Self::process_message(
                        connection_id,
                        message,
                        &connections,
                        &message_handler,
                    )
                    .await
                    {
                        error!("Error processing message: {}", e);
                    }
                }
                Ok(Some(Err(e))) => {
                    warn!("WebSocket error: {}", e);
                    break;
                }
                Ok(None) => {
                    info!("WebSocket connection closed by client: {}", connection_id);
                    break;
                }
                Err(_) => {
                    warn!("Connection timeout: {}", connection_id);
                    break;
                }
            }
        }

        // Cleanup
        outgoing_task.abort();
        if let Some(task) = broadcast_task {
            task.abort();
        }

        connections.write().remove(&connection_id);
        let _ = message_handler
            .handle_connection_closed(connection_id)
            .await;

        info!("Connection {} closed", connection_id);
        Ok(())
    }

    /// Process incoming WebSocket message
    async fn process_message(
        connection_id: Uuid,
        message: Message,
        connections: &Arc<RwLock<HashMap<Uuid, ConnectionState>>>,
        message_handler: &Arc<dyn MessageHandler + Send + Sync>,
    ) -> Result<()> {
        let text = match message {
            Message::Text(text) => text,
            Message::Binary(data) => String::from_utf8(data).map_err(|e| {
                PipelineError::Mcp(format!("Invalid UTF-8 in binary message: {}", e))
            })?,
            Message::Ping(data) => {
                // Send pong response
                if let Some(cs) = connections.read().get(&connection_id) {
                    let _ = cs.sender.send(Message::Pong(data));
                }
                return Ok(());
            }
            Message::Pong(_) => {
                // Update heartbeat
                if let Some(cs) = connections.read().get(&connection_id) {
                    cs.last_heartbeat
                        .store(current_timestamp(), Ordering::Relaxed);
                }
                return Ok(());
            }
            Message::Close(_) => {
                return Ok(()); // Connection will be closed by the main loop
            }
            Message::Frame(_) => {
                return Ok(()); // Raw frames are handled internally
            }
        };

        // Parse MCP message
        let mcp_message: McpMessage = serde_json::from_str(&text)
            .map_err(|e| PipelineError::Mcp(format!("Failed to parse MCP message: {}", e)))?;

        debug!("Received message from {}: {:?}", connection_id, mcp_message);

        // Update connection statistics
        if let Some(cs) = connections.read().get(&connection_id) {
            cs.message_count.fetch_add(1, Ordering::Relaxed);
            cs.last_heartbeat
                .store(current_timestamp(), Ordering::Relaxed);
        }

        // Handle message
        if let Ok(Some(response)) = message_handler
            .handle_message(connection_id, mcp_message)
            .await
        {
            let response_json = serde_json::to_string(&response)
                .map_err(|e| PipelineError::Mcp(format!("Failed to serialize response: {}", e)))?;

            if let Some(cs) = connections.read().get(&connection_id) {
                let _ = cs.sender.send(Message::Text(response_json));
            }
        }

        Ok(())
    }

    /// Start heartbeat monitoring task
    async fn start_heartbeat_task(&self) {
        let connections = self.connections.clone();
        let heartbeat_interval = self.config.heartbeat_interval;
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(heartbeat_interval);

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                let current_time = current_timestamp();
                let mut connections_to_close = Vec::new();

                {
                    let connections_read = connections.read();
                    for (connection_id, cs) in connections_read.iter() {
                        let last_heartbeat = cs.last_heartbeat.load(Ordering::Relaxed);
                        let age = current_time - last_heartbeat;

                        if age > heartbeat_interval.as_secs() * 3 {
                            connections_to_close.push(*connection_id);
                        } else {
                            // Send ping to check connection
                            let _ = cs.sender.send(Message::Ping(Vec::new()));
                        }
                    }
                }

                // Close stale connections
                for connection_id in connections_to_close {
                    connections.write().remove(&connection_id);
                    warn!("Closed stale connection: {}", connection_id);
                }
            }
        });
    }

    /// Start statistics collection task
    async fn start_stats_task(&self) {
        let stats = self.stats.clone();
        let connections = self.connections.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                let connections_read = connections.read();
                let active_count = connections_read
                    .values()
                    .filter(|cs| cs.is_active.load(Ordering::Relaxed))
                    .count();

                drop(connections_read);

                debug!("Active connections: {}", active_count);
            }
        });
    }
}

#[async_trait::async_trait]
impl MessageHandler for DefaultMessageHandler {
    async fn handle_message(
        &self,
        connection_id: Uuid,
        message: McpMessage,
    ) -> Result<Option<McpMessage>> {
        match message {
            McpMessage::HealthCheck => Ok(Some(McpMessage::Success {
                request_id: Uuid::new_v4(),
                data: Some(serde_json::json!({
                    "status": "healthy",
                    "timestamp": current_timestamp()
                })),
            })),
            McpMessage::PipelineStatus { pipeline_id } => {
                let state = self.pipeline_state.read();
                let status = if state.active_tasks.contains_key(&pipeline_id) {
                    "running"
                } else {
                    "idle"
                };

                Ok(Some(McpMessage::Success {
                    request_id: Uuid::new_v4(),
                    data: Some(serde_json::json!({
                        "pipeline_id": pipeline_id,
                        "status": status,
                        "active_tasks": state.active_tasks.len(),
                        "performance": state.performance_metrics
                    })),
                }))
            }
            McpMessage::TaskSubmit { task } => {
                let task_info = TaskInfo {
                    task_id: task.task_id,
                    task_type: format!("{:?}", task.task_type),
                    assigned_worker: None,
                    start_time: current_timestamp(),
                    status: "submitted".to_string(),
                };

                {
                    let mut state = self.pipeline_state.write();
                    state.active_tasks.insert(task.task_id, task_info);
                }

                Ok(Some(McpMessage::Success {
                    request_id: Uuid::new_v4(),
                    data: Some(serde_json::json!({
                        "task_id": task.task_id,
                        "status": "submitted"
                    })),
                }))
            }
            McpMessage::TaskStatus { task_id } => {
                let state = self.pipeline_state.read();
                if let Some(task_info) = state.active_tasks.get(&task_id) {
                    Ok(Some(McpMessage::Success {
                        request_id: Uuid::new_v4(),
                        data: Some(serde_json::json!(task_info)),
                    }))
                } else {
                    Ok(Some(McpMessage::Error {
                        request_id: Uuid::new_v4(),
                        error: McpError::not_found(&format!("Task {} not found", task_id)),
                    }))
                }
            }
            _ => {
                // Default handler for other message types
                Ok(Some(McpMessage::Error {
                    request_id: Uuid::new_v4(),
                    error: McpError::invalid_request(
                        "Message type not supported by default handler",
                    ),
                }))
            }
        }
    }

    async fn handle_connection_established(&self, connection: &McpConnection) -> Result<()> {
        info!("Connection established: {:?}", connection.client_type);
        Ok(())
    }

    async fn handle_connection_closed(&self, connection_id: Uuid) -> Result<()> {
        info!("Connection closed: {}", connection_id);
        Ok(())
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Start MCP server with default configuration
pub async fn start(port: u16) -> Result<()> {
    let mut config = McpServerConfig::default();
    config.port = port;

    let server = McpServer::with_config(config);
    server.start().await
}

/// Stop MCP server
pub async fn stop() -> Result<()> {
    // This would need to be implemented with a global server instance
    // For now, just return Ok
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let server = McpServer::new();
        assert_eq!(server.config.port, 8700);
        assert_eq!(server.config.host, "0.0.0.0");
    }

    #[tokio::test]
    async fn test_server_stats() {
        let server = McpServer::new();
        let stats = server.get_stats().await;
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_connections, 0);
    }

    #[tokio::test]
    async fn test_message_serialization() {
        let message = McpMessage::HealthCheck;
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("HealthCheck"));
    }
}
