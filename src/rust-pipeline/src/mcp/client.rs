/*!
# MCP Client Implementation

Model Context Protocol client for coordinating with the MCP server and Python workers.
Provides agent coordination interface with reconnection logic and fault tolerance.
*/

use crate::mcp::{McpMessage, McpError, ClientType};
use crate::{Result, PipelineError};
use futures::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, broadcast};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use tracing::{error, info, warn, debug, trace};
use url::Url;
use uuid::Uuid;

/// MCP client configuration
#[derive(Debug, Clone)]
pub struct McpClientConfig {
    pub server_url: String,
    pub client_type: ClientType,
    pub capabilities: Vec<String>,
    pub reconnect_attempts: u32,
    pub reconnect_delay: Duration,
    pub heartbeat_interval: Duration,
    pub request_timeout: Duration,
    pub max_queue_size: usize,
    pub enable_compression: bool,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            server_url: "ws://localhost:8700".to_string(),
            client_type: ClientType::RustCore,
            capabilities: vec![
                "task_execution".to_string(),
                "model_management".to_string(),
                "performance_monitoring".to_string(),
            ],
            reconnect_attempts: 10,
            reconnect_delay: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            max_queue_size: 10000,
            enable_compression: false,
        }
    }
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Queued message with priority and retry information
#[derive(Debug)]
pub struct QueuedMessage {
    pub message: McpMessage,
    pub priority: MessagePriority,
    pub request_id: Uuid,
    pub sender: Option<oneshot::Sender<Result<McpMessage>>>,
    pub retry_count: u32,
    pub created_at: u64,
}

/// Agent coordination state
#[derive(Debug, Clone)]
pub struct AgentState {
    pub agent_id: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
    pub current_task: Option<Uuid>,
    pub performance_metrics: AgentMetrics,
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Working,
    Paused,
    Error(String),
    Disconnected,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub tasks_completed: u64,
    pub average_task_duration_ms: f64,
    pub success_rate_percent: f64,
    pub error_count: u64,
    pub memory_usage_mb: u64,
}

/// Connection state tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
}

/// MCP Client for agent coordination
pub struct McpClient {
    config: McpClientConfig,
    connection_state: Arc<RwLock<ConnectionState>>,
    message_queue: Arc<RwLock<VecDeque<QueuedMessage>>>,
    pending_requests: Arc<RwLock<HashMap<Uuid, oneshot::Sender<Result<McpMessage>>>>>,
    agents: Arc<RwLock<HashMap<String, AgentState>>>,
    event_sender: broadcast::Sender<ClientEvent>,
    is_running: Arc<AtomicBool>,
    message_counter: AtomicU64,
    connection_id: Arc<RwLock<Option<Uuid>>>,
    stats: Arc<RwLock<ClientStats>>,
}

/// Client events for monitoring and coordination
#[derive(Debug, Clone)]
pub enum ClientEvent {
    Connected,
    Disconnected,
    Reconnecting,
    MessageSent(Uuid),
    MessageReceived(Uuid),
    AgentUpdated(String, AgentStatus),
    TaskAssigned(Uuid, String),
    TaskCompleted(Uuid, String),
    Error(String),
}

/// Client statistics
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub reconnection_count: u64,
    pub queue_size: usize,
    pub pending_requests: usize,
    pub active_agents: usize,
    pub average_response_time_ms: f64,
    pub connection_uptime_seconds: u64,
}

impl McpClient {
    /// Create new MCP client
    pub fn new() -> Self {
        Self::with_config(McpClientConfig::default())
    }

    /// Create MCP client with custom configuration
    pub fn with_config(config: McpClientConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        
        Self {
            config,
            connection_state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            message_queue: Arc::new(RwLock::new(VecDeque::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            agents: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            is_running: Arc::new(AtomicBool::new(false)),
            message_counter: AtomicU64::new(0),
            connection_id: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(ClientStats::default())),
        }
    }

    /// Start the MCP client
    pub async fn start(&self) -> Result<()> {
        info!("Starting MCP client for {:?}", self.config.client_type);
        self.is_running.store(true, Ordering::Relaxed);

        // Start background tasks
        self.start_connection_manager().await;
        self.start_queue_processor().await;
        self.start_heartbeat_task().await;
        self.start_stats_collector().await;

        // Send connected event
        let _ = self.event_sender.send(ClientEvent::Connected);
        
        info!("MCP client started successfully");
        Ok(())
    }

    /// Stop the MCP client
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping MCP client");
        self.is_running.store(false, Ordering::Relaxed);

        // Clear pending requests
        let pending_requests = {
            let mut requests = self.pending_requests.write();
            std::mem::take(&mut *requests)
        };

        for (_, sender) in pending_requests {
            let _ = sender.send(Err(PipelineError::Mcp("Client shutting down".to_string())));
        }

        // Update connection state
        *self.connection_state.write() = ConnectionState::Disconnected;
        let _ = self.event_sender.send(ClientEvent::Disconnected);

        info!("MCP client stopped");
        Ok(())
    }

    /// Send message with response waiting
    pub async fn send_message(&self, message: McpMessage) -> Result<McpMessage> {
        self.send_message_with_priority(message, MessagePriority::Normal).await
    }

    /// Send message with specific priority and wait for response
    pub async fn send_message_with_priority(&self, message: McpMessage, priority: MessagePriority) -> Result<McpMessage> {
        let request_id = Uuid::new_v4();
        let (tx, rx) = oneshot::channel();

        let queued_message = QueuedMessage {
            message,
            priority,
            request_id,
            sender: Some(tx),
            retry_count: 0,
            created_at: current_timestamp(),
        };

        // Add to priority queue
        {
            let mut queue = self.message_queue.write();
            if queue.len() >= self.config.max_queue_size {
                return Err(PipelineError::Mcp("Message queue full".to_string()));
            }
            
            // Insert based on priority (higher priority first)
            let mut insert_index = None;
            for (i, existing) in queue.iter().enumerate() {
                if priority > existing.priority {
                    insert_index = Some(i);
                    break;
                }
            }
            
            if let Some(index) = insert_index {
                queue.insert(index, queued_message);
            } else {
                queue.push_back(queued_message);
            }
        }

        // Wait for response with timeout
        match timeout(self.config.request_timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(PipelineError::Mcp("Response channel closed".to_string())),
            Err(_) => Err(PipelineError::Mcp("Request timeout".to_string())),
        }
    }

    /// Send message without waiting for response
    pub async fn send_message_async(&self, message: McpMessage, priority: MessagePriority) -> Result<Uuid> {
        let request_id = Uuid::new_v4();

        let queued_message = QueuedMessage {
            message,
            priority,
            request_id,
            sender: None,
            retry_count: 0,
            created_at: current_timestamp(),
        };

        // Add to priority queue
        {
            let mut queue = self.message_queue.write();
            if queue.len() >= self.config.max_queue_size {
                return Err(PipelineError::Mcp("Message queue full".to_string()));
            }
            
            // Insert based on priority (higher priority first)
            let mut insert_index = None;
            for (i, existing) in queue.iter().enumerate() {
                if priority > existing.priority {
                    insert_index = Some(i);
                    break;
                }
            }
            
            if let Some(index) = insert_index {
                queue.insert(index, queued_message);
            } else {
                queue.push_back(queued_message);
            }
        }

        Ok(request_id)
    }

    /// Register agent with the coordination system
    pub async fn register_agent(&self, agent: AgentState) -> Result<()> {
        let agent_id = agent.agent_id.clone();
        
        // Store agent state
        {
            let mut agents = self.agents.write();
            agents.insert(agent_id.clone(), agent.clone());
        }

        // Send registration message to server
        let message = McpMessage::IpcRegister {
            process_id: agent_id.clone(),
            capabilities: agent.capabilities.clone(),
        };

        self.send_message_async(message, MessagePriority::High).await?;

        // Notify event listeners
        let _ = self.event_sender.send(ClientEvent::AgentUpdated(
            agent_id,
            agent.status
        ));

        info!("Agent registered: {}", agent.agent_id);
        Ok(())
    }

    /// Update agent status
    pub async fn update_agent_status(&self, agent_id: &str, status: AgentStatus) -> Result<()> {
        {
            let mut agents = self.agents.write();
            if let Some(agent) = agents.get_mut(agent_id) {
                agent.status = status.clone();
                agent.last_heartbeat = current_timestamp();
            } else {
                return Err(PipelineError::Mcp(format!("Agent {} not found", agent_id)));
            }
        }

        // Notify event listeners
        let _ = self.event_sender.send(ClientEvent::AgentUpdated(
            agent_id.to_string(),
            status
        ));

        Ok(())
    }

    /// Assign task to agent
    pub async fn assign_task(&self, agent_id: &str, task_id: Uuid) -> Result<()> {
        {
            let mut agents = self.agents.write();
            if let Some(agent) = agents.get_mut(agent_id) {
                agent.current_task = Some(task_id);
                agent.status = AgentStatus::Working;
            } else {
                return Err(PipelineError::Mcp(format!("Agent {} not found", agent_id)));
            }
        }

        // Notify event listeners
        let _ = self.event_sender.send(ClientEvent::TaskAssigned(
            task_id,
            agent_id.to_string()
        ));

        info!("Task {} assigned to agent {}", task_id, agent_id);
        Ok(())
    }

    /// Complete task for agent
    pub async fn complete_task(&self, agent_id: &str, task_id: Uuid, success: bool) -> Result<()> {
        {
            let mut agents = self.agents.write();
            if let Some(agent) = agents.get_mut(agent_id) {
                agent.current_task = None;
                agent.status = AgentStatus::Idle;
                agent.performance_metrics.tasks_completed += 1;
                
                if !success {
                    agent.performance_metrics.error_count += 1;
                }

                // Update success rate
                let total_tasks = agent.performance_metrics.tasks_completed;
                let errors = agent.performance_metrics.error_count;
                agent.performance_metrics.success_rate_percent = 
                    ((total_tasks - errors) as f64 / total_tasks as f64) * 100.0;
            } else {
                return Err(PipelineError::Mcp(format!("Agent {} not found", agent_id)));
            }
        }

        // Notify event listeners
        let _ = self.event_sender.send(ClientEvent::TaskCompleted(
            task_id,
            agent_id.to_string()
        ));

        info!("Task {} completed by agent {} (success: {})", task_id, agent_id, success);
        Ok(())
    }

    /// Get all registered agents
    pub async fn get_agents(&self) -> HashMap<String, AgentState> {
        self.agents.read().clone()
    }

    /// Get client statistics
    pub async fn get_stats(&self) -> ClientStats {
        let mut stats = self.stats.read().clone();
        stats.queue_size = self.message_queue.read().len();
        stats.pending_requests = self.pending_requests.read().len();
        stats.active_agents = self.agents.read().values()
            .filter(|agent| matches!(agent.status, AgentStatus::Working))
            .count();
        stats
    }

    /// Subscribe to client events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ClientEvent> {
        self.event_sender.subscribe()
    }

    /// Get connection state
    pub fn get_connection_state(&self) -> ConnectionState {
        *self.connection_state.read()
    }

    /// Get message queue (for testing)
    #[cfg(test)]
    pub fn get_message_queue(&self) -> Arc<RwLock<VecDeque<QueuedMessage>>> {
        self.message_queue.clone()
    }

    /// Start connection manager task
    async fn start_connection_manager(&self) {
        let config = self.config.clone();
        let connection_state = self.connection_state.clone();
        let event_sender = self.event_sender.clone();
        let is_running = self.is_running.clone();
        let message_queue = self.message_queue.clone();
        let pending_requests = self.pending_requests.clone();
        let stats = self.stats.clone();
        let connection_id = self.connection_id.clone();

        tokio::spawn(async move {
            let mut reconnect_count = 0;
            
            while is_running.load(Ordering::Relaxed) {
                // Update connection state
                *connection_state.write() = ConnectionState::Connecting;
                let _ = event_sender.send(ClientEvent::Reconnecting);

                match Self::establish_connection(&config).await {
                    Ok(ws_stream) => {
                        info!("Connected to MCP server at {}", config.server_url);
                        *connection_state.write() = ConnectionState::Connected;
                        let conn_id = Uuid::new_v4();
                        *connection_id.write() = Some(conn_id);
                        let _ = event_sender.send(ClientEvent::Connected);
                        reconnect_count = 0;

                        // Handle the connection
                        if let Err(e) = Self::handle_connection(
                            ws_stream,
                            connection_state.clone(),
                            event_sender.clone(),
                            is_running.clone(),
                            message_queue.clone(),
                            pending_requests.clone(),
                            stats.clone(),
                        ).await {
                            error!("Connection error: {}", e);
                            let _ = event_sender.send(ClientEvent::Error(e.to_string()));
                        }
                    }
                    Err(e) => {
                        error!("Failed to connect to MCP server: {}", e);
                        let _ = event_sender.send(ClientEvent::Error(e.to_string()));
                    }
                }

                // Update connection state to disconnected
                *connection_state.write() = ConnectionState::Disconnected;
                *connection_id.write() = None;
                let _ = event_sender.send(ClientEvent::Disconnected);

                // Check if we should attempt reconnection
                if is_running.load(Ordering::Relaxed) {
                    reconnect_count += 1;
                    if reconnect_count <= config.reconnect_attempts {
                        warn!("Reconnecting in {} seconds (attempt {}/{})", 
                              config.reconnect_delay.as_secs(), 
                              reconnect_count, 
                              config.reconnect_attempts);
                        
                        tokio::time::sleep(config.reconnect_delay).await;
                    } else {
                        error!("Max reconnection attempts reached, giving up");
                        *connection_state.write() = ConnectionState::Failed;
                        break;
                    }
                }
            }
        });
    }

    /// Establish WebSocket connection
    async fn establish_connection(config: &McpClientConfig) -> Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
        let url = Url::parse(&config.server_url)
            .map_err(|e| PipelineError::Mcp(format!("Invalid server URL: {}", e)))?;

        let (ws_stream, _) = connect_async(url).await
            .map_err(|e| PipelineError::Mcp(format!("WebSocket connection failed: {}", e)))?;

        Ok(ws_stream)
    }

    /// Handle WebSocket connection
    async fn handle_connection(
        ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
        connection_state: Arc<RwLock<ConnectionState>>,
        event_sender: broadcast::Sender<ClientEvent>,
        is_running: Arc<AtomicBool>,
        message_queue: Arc<RwLock<VecDeque<QueuedMessage>>>,
        pending_requests: Arc<RwLock<HashMap<Uuid, oneshot::Sender<Result<McpMessage>>>>>,
        stats: Arc<RwLock<ClientStats>>,
    ) -> Result<()> {
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

        // Spawn outgoing message handler
        let outgoing_task = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if let Err(e) = ws_sender.send(message).await {
                    error!("Failed to send WebSocket message: {}", e);
                    break;
                }
            }
        });

        // Spawn queue processor for this connection
        let tx_clone = tx.clone();
        let message_queue_clone = message_queue.clone();
        let is_running_clone = is_running.clone();
        let stats_clone = stats.clone();
        let event_sender_clone = event_sender.clone();

        let queue_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(10));
            
            while is_running_clone.load(Ordering::Relaxed) {
                interval.tick().await;

                let message_to_send = {
                    let mut queue = message_queue_clone.write();
                    queue.pop_front()
                };

                if let Some(queued_msg) = message_to_send {
                    match serde_json::to_string(&queued_msg.message) {
                        Ok(json) => {
                            if let Err(e) = tx_clone.send(Message::Text(json)) {
                                error!("Failed to queue outgoing message: {}", e);
                                // Put message back in queue
                                message_queue_clone.write().push_front(queued_msg);
                                break;
                            } else {
                                // Update stats
                                {
                                    let mut stats = stats_clone.write();
                                    stats.total_messages_sent += 1;
                                }
                                
                                let _ = event_sender_clone.send(ClientEvent::MessageSent(queued_msg.request_id));
                            }
                        }
                        Err(e) => {
                            error!("Failed to serialize message: {}", e);
                            
                            // Send error response if there's a sender
                            if let Some(sender) = queued_msg.sender {
                                let _ = sender.send(Err(PipelineError::Mcp(format!("Serialization error: {}", e))));
                            }
                        }
                    }
                }
            }
        });

        // Main message receiving loop
        while is_running.load(Ordering::Relaxed) && 
              *connection_state.read() == ConnectionState::Connected {
            
            match timeout(Duration::from_secs(30), ws_receiver.next()).await {
                Ok(Some(Ok(message))) => {
                    if let Err(e) = Self::process_incoming_message(
                        message,
                        &pending_requests,
                        &stats,
                        &event_sender,
                    ).await {
                        error!("Error processing incoming message: {}", e);
                    }
                }
                Ok(Some(Err(e))) => {
                    warn!("WebSocket receive error: {}", e);
                    break;
                }
                Ok(None) => {
                    info!("WebSocket connection closed by server");
                    break;
                }
                Err(_) => {
                    // Timeout - send ping to keep connection alive
                    if let Err(e) = tx.send(Message::Ping(Vec::new())) {
                        error!("Failed to send ping: {}", e);
                        break;
                    }
                }
            }
        }

        // Cleanup
        outgoing_task.abort();
        queue_task.abort();

        Ok(())
    }

    /// Process incoming WebSocket message
    async fn process_incoming_message(
        message: Message,
        pending_requests: &Arc<RwLock<HashMap<Uuid, oneshot::Sender<Result<McpMessage>>>>>,
        stats: &Arc<RwLock<ClientStats>>,
        event_sender: &broadcast::Sender<ClientEvent>,
    ) -> Result<()> {
        let text = match message {
            Message::Text(text) => text,
            Message::Binary(data) => {
                String::from_utf8(data)
                    .map_err(|e| PipelineError::Mcp(format!("Invalid UTF-8 in binary message: {}", e)))?
            }
            Message::Pong(_) => {
                debug!("Received pong from server");
                return Ok(());
            }
            _ => return Ok(()), // Ignore other message types
        };

        // Parse MCP message
        let mcp_message: McpMessage = serde_json::from_str(&text)
            .map_err(|e| PipelineError::Mcp(format!("Failed to parse MCP message: {}", e)))?;

        debug!("Received message: {:?}", mcp_message);

        // Update stats
        {
            let mut stats = stats.write();
            stats.total_messages_received += 1;
        }

        // Handle response messages
        match &mcp_message {
            McpMessage::Success { request_id, .. } | McpMessage::Error { request_id, .. } => {
                if let Some(sender) = pending_requests.write().remove(request_id) {
                    let result = match &mcp_message {
                        McpMessage::Success { .. } => Ok(mcp_message.clone()),
                        McpMessage::Error { error, .. } => Err(PipelineError::Mcp(error.message.clone())),
                        _ => unreachable!(),
                    };
                    
                    let _ = sender.send(result);
                    let _ = event_sender.send(ClientEvent::MessageReceived(*request_id));
                }
            }
            _ => {
                // Handle other message types (broadcasts, notifications, etc.)
                trace!("Received non-response message: {:?}", mcp_message);
            }
        }

        Ok(())
    }

    /// Start message queue processor
    async fn start_queue_processor(&self) {
        // The queue processor is now integrated into the connection handler
        // This is a placeholder for any additional queue management logic
    }

    /// Start heartbeat task
    async fn start_heartbeat_task(&self) {
        let config = self.config.clone();
        let is_running = self.is_running.clone();
        let connection_state = self.connection_state.clone();
        let agents = self.agents.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.heartbeat_interval);

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                if *connection_state.read() == ConnectionState::Connected {
                    // Update agent heartbeats
                    let current_time = current_timestamp();
                    let mut agents_to_update = Vec::new();

                    {
                        let agents = agents.read();
                        for (agent_id, agent) in agents.iter() {
                            if current_time - agent.last_heartbeat > config.heartbeat_interval.as_secs() * 2 {
                                agents_to_update.push(agent_id.clone());
                            }
                        }
                    }

                    // Mark stale agents as disconnected
                    for agent_id in agents_to_update {
                        let mut agents = agents.write();
                        if let Some(agent) = agents.get_mut(&agent_id) {
                            agent.status = AgentStatus::Disconnected;
                        }
                    }
                }
            }
        });
    }

    /// Start statistics collector
    async fn start_stats_collector(&self) {
        let stats = self.stats.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                // Update uptime
                {
                    let mut stats = stats.write();
                    stats.connection_uptime_seconds += 60;
                }

                debug!("Updated client statistics");
            }
        });
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Create MCP client for Rust core coordination
pub async fn create_rust_client(server_url: &str) -> Result<McpClient> {
    let mut config = McpClientConfig::default();
    config.server_url = server_url.to_string();
    config.client_type = ClientType::RustCore;
    config.capabilities = vec![
        "task_orchestration".to_string(),
        "model_management".to_string(),
        "performance_monitoring".to_string(),
        "ipc_coordination".to_string(),
    ];

    let client = McpClient::with_config(config);
    client.start().await?;
    Ok(client)
}

/// Create MCP client for Python worker coordination
pub async fn create_python_client(server_url: &str) -> Result<McpClient> {
    let mut config = McpClientConfig::default();
    config.server_url = server_url.to_string();
    config.client_type = ClientType::PythonWorker;
    config.capabilities = vec![
        "model_inference".to_string(),
        "document_processing".to_string(),
        "quality_validation".to_string(),
    ];

    let client = McpClient::with_config(config);
    client.start().await?;
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = McpClient::new();
        assert_eq!(client.get_connection_state(), ConnectionState::Disconnected);
    }

    #[tokio::test]
    async fn test_message_priority_queue() {
        let client = McpClient::new();
        
        let low_msg = McpMessage::HealthCheck;
        let high_msg = McpMessage::SystemShutdown;
        
        let _ = client.send_message_async(low_msg, MessagePriority::Low).await;
        let _ = client.send_message_async(high_msg, MessagePriority::Critical).await;
        
        // Verify high priority message is first
        let queue = client.message_queue.read();
        assert_eq!(queue.len(), 2);
        assert_eq!(queue[0].priority, MessagePriority::Critical);
        assert_eq!(queue[1].priority, MessagePriority::Low);
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let client = McpClient::new();
        
        let agent = AgentState {
            agent_id: "test_agent".to_string(),
            agent_type: "processor".to_string(),
            capabilities: vec!["processing".to_string()],
            status: AgentStatus::Idle,
            current_task: None,
            performance_metrics: AgentMetrics::default(),
            last_heartbeat: current_timestamp(),
        };

        let result = client.register_agent(agent).await;
        assert!(result.is_ok());

        let agents = client.get_agents().await;
        assert!(agents.contains_key("test_agent"));
    }

    #[tokio::test]
    async fn test_task_assignment() {
        let client = McpClient::new();
        
        let agent = AgentState {
            agent_id: "worker_agent".to_string(),
            agent_type: "worker".to_string(),
            capabilities: vec!["task_execution".to_string()],
            status: AgentStatus::Idle,
            current_task: None,
            performance_metrics: AgentMetrics::default(),
            last_heartbeat: current_timestamp(),
        };

        client.register_agent(agent).await.unwrap();
        
        let task_id = Uuid::new_v4();
        let result = client.assign_task("worker_agent", task_id).await;
        assert!(result.is_ok());

        let agents = client.get_agents().await;
        let agent = &agents["worker_agent"];
        assert_eq!(agent.current_task, Some(task_id));
        assert!(matches!(agent.status, AgentStatus::Working));
    }
}