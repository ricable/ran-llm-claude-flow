/*!
# Message Queue System

High-performance message queue for inter-process communication between Rust core and Python ML workers.
Uses lock-free queues optimized for M3 Max architecture.
*/

use crate::{Result, PipelineError};
use crate::ipc::IpcMessage;
use crossbeam_channel::{self, Receiver, Sender, TryRecvError, TrySendError};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;
use uuid::Uuid;

/// Message queue priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Message wrapper with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMessage {
    pub id: Uuid,
    pub sender: String,
    pub recipient: String,
    pub priority: MessagePriority,
    pub timestamp: u64,
    pub ttl_seconds: Option<u64>,
    pub retry_count: u32,
    pub payload: IpcMessage,
}

/// Message queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub messages_in_flight: u32,
    pub queue_depths: HashMap<String, u32>,
    pub average_latency_ms: f64,
    pub throughput_msgs_per_sec: f64,
    pub error_count: u64,
}

/// High-performance message queue manager
pub struct MessageQueue {
    /// Per-worker message queues (worker_id -> channels)
    worker_queues: Arc<DashMap<String, WorkerQueue>>,
    
    /// Broadcast channel for system-wide messages
    broadcast_sender: Sender<QueueMessage>,
    broadcast_receiver: Arc<RwLock<Receiver<QueueMessage>>>,
    
    /// Message routing table
    routing_table: Arc<DashMap<String, String>>, // message_type -> preferred_worker
    
    /// Performance metrics
    metrics: Arc<QueueMetrics>,
    
    /// Configuration
    config: MessageQueueConfig,
    
    /// Notification for new messages
    message_notify: Arc<Notify>,
}

#[derive(Debug)]
struct WorkerQueue {
    high_priority: (Sender<QueueMessage>, Receiver<QueueMessage>),
    normal_priority: (Sender<QueueMessage>, Receiver<QueueMessage>),
    low_priority: (Sender<QueueMessage>, Receiver<QueueMessage>),
    worker_id: String,
    last_activity: AtomicU64,
    is_active: bool,
}

#[derive(Debug)]
struct QueueMetrics {
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    messages_dropped: AtomicU64,
    total_latency_ms: AtomicU64,
    message_count: AtomicU64,
    last_throughput_check: AtomicU64,
    throughput_counter: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct MessageQueueConfig {
    /// Maximum queue size per worker
    pub max_queue_size: usize,
    /// Message timeout in seconds
    pub message_timeout: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Enable message compression
    pub enable_compression: bool,
    /// Batch size for bulk operations
    pub batch_size: usize,
}

impl Default for MessageQueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            message_timeout: 30,
            max_retries: 3,
            enable_compression: true,
            batch_size: 100,
        }
    }
}

impl MessageQueue {
    /// Create new message queue system
    pub async fn new() -> Result<Self> {
        Self::with_config(MessageQueueConfig::default()).await
    }
    
    /// Create message queue with custom configuration
    pub async fn with_config(config: MessageQueueConfig) -> Result<Self> {
        tracing::info!("Initializing message queue system with max queue size: {}", config.max_queue_size);
        
        let (broadcast_sender, broadcast_receiver) = 
            crossbeam_channel::unbounded();
        
        let metrics = QueueMetrics {
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            messages_dropped: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            message_count: AtomicU64::new(0),
            last_throughput_check: AtomicU64::new(current_timestamp_ms()),
            throughput_counter: AtomicU64::new(0),
        };

        Ok(Self {
            worker_queues: Arc::new(DashMap::new()),
            broadcast_sender,
            broadcast_receiver: Arc::new(RwLock::new(broadcast_receiver)),
            routing_table: Arc::new(DashMap::new()),
            metrics: Arc::new(metrics),
            config,
            message_notify: Arc::new(Notify::new()),
        })
    }

    /// Register a new worker with dedicated queues
    pub async fn register_worker(&self, worker_id: String) -> Result<()> {
        tracing::info!("Registering worker: {}", worker_id);
        
        let high_priority = crossbeam_channel::bounded(self.config.max_queue_size / 4);
        let normal_priority = crossbeam_channel::bounded(self.config.max_queue_size / 2);
        let low_priority = crossbeam_channel::bounded(self.config.max_queue_size / 4);
        
        let worker_queue = WorkerQueue {
            high_priority,
            normal_priority,
            low_priority,
            worker_id: worker_id.clone(),
            last_activity: AtomicU64::new(current_timestamp_ms()),
            is_active: true,
        };
        
        self.worker_queues.insert(worker_id.clone(), worker_queue);
        
        tracing::info!("Worker {} registered successfully", worker_id);
        Ok(())
    }

    /// Unregister a worker and cleanup queues
    pub async fn unregister_worker(&self, worker_id: &str) -> Result<()> {
        tracing::info!("Unregistering worker: {}", worker_id);
        
        if let Some((_, mut worker_queue)) = self.worker_queues.remove(worker_id) {
            worker_queue.is_active = false;
            
            // Drain remaining messages to avoid blocking senders
            self.drain_worker_queues(&worker_queue).await;
        }
        
        // Remove from routing table
        self.routing_table.retain(|_, v| v != worker_id);
        
        tracing::info!("Worker {} unregistered successfully", worker_id);
        Ok(())
    }

    /// Send message to specific worker
    pub async fn send_to_worker(&self, worker_id: &str, message: IpcMessage) -> Result<()> {
        let queue_message = QueueMessage {
            id: Uuid::new_v4(),
            sender: "rust-core".to_string(),
            recipient: worker_id.to_string(),
            priority: self.determine_message_priority(&message),
            timestamp: current_timestamp_ms(),
            ttl_seconds: Some(self.config.message_timeout),
            retry_count: 0,
            payload: message,
        };

        self.send_message_internal(worker_id, queue_message).await
    }

    /// Send high priority message to specific worker
    pub async fn send_priority_message(&self, worker_id: &str, message: IpcMessage, priority: MessagePriority) -> Result<()> {
        let queue_message = QueueMessage {
            id: Uuid::new_v4(),
            sender: "rust-core".to_string(),
            recipient: worker_id.to_string(),
            priority,
            timestamp: current_timestamp_ms(),
            ttl_seconds: Some(self.config.message_timeout),
            retry_count: 0,
            payload: message,
        };

        self.send_message_internal(worker_id, queue_message).await
    }

    /// Broadcast message to all workers
    pub async fn broadcast(&self, message: IpcMessage) -> Result<()> {
        let broadcast_message = QueueMessage {
            id: Uuid::new_v4(),
            sender: "rust-core".to_string(),
            recipient: "broadcast".to_string(),
            priority: MessagePriority::Normal,
            timestamp: current_timestamp_ms(),
            ttl_seconds: Some(self.config.message_timeout),
            retry_count: 0,
            payload: message,
        };

        self.broadcast_sender.send(broadcast_message)
            .map_err(|e| PipelineError::Ipc(format!("Broadcast failed: {}", e)))?;

        self.metrics.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.message_notify.notify_waiters();
        
        tracing::debug!("Broadcast message sent to all workers");
        Ok(())
    }

    /// Receive message from any worker (priority-ordered)
    pub async fn receive(&self) -> Result<(String, IpcMessage)> {
        loop {
            // First check for broadcast messages
            if let Ok(broadcast_msg) = self.try_receive_broadcast() {
                self.metrics.messages_received.fetch_add(1, Ordering::Relaxed);
                return Ok((broadcast_msg.sender, broadcast_msg.payload));
            }

            // Then check worker queues by priority
            if let Some((sender, message)) = self.try_receive_from_workers().await? {
                self.metrics.messages_received.fetch_add(1, Ordering::Relaxed);
                
                // Calculate latency
                let latency_ms = current_timestamp_ms() - message.timestamp;
                self.metrics.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
                self.metrics.message_count.fetch_add(1, Ordering::Relaxed);
                
                return Ok((sender, message.payload));
            }

            // Wait for notification of new messages
            self.message_notify.notified().await;
            
            // Small delay to prevent tight loop
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Try to receive message without blocking
    pub async fn try_receive(&self) -> Result<Option<(String, IpcMessage)>> {
        // Check broadcast messages first
        if let Ok(broadcast_msg) = self.try_receive_broadcast() {
            self.metrics.messages_received.fetch_add(1, Ordering::Relaxed);
            return Ok(Some((broadcast_msg.sender, broadcast_msg.payload)));
        }

        // Check worker queues
        if let Some((sender, message)) = self.try_receive_from_workers().await? {
            self.metrics.messages_received.fetch_add(1, Ordering::Relaxed);
            return Ok(Some((sender, message.payload)));
        }

        Ok(None)
    }

    /// Get current queue depth for a worker
    pub async fn get_queue_depth(&self) -> Result<u32> {
        let mut total_depth = 0u32;
        
        for entry in self.worker_queues.iter() {
            let worker_queue = entry.value();
            total_depth += worker_queue.high_priority.0.len() as u32;
            total_depth += worker_queue.normal_priority.0.len() as u32;
            total_depth += worker_queue.low_priority.0.len() as u32;
        }
        
        Ok(total_depth)
    }

    /// Get total message count processed
    pub async fn get_message_count(&self) -> Result<u64> {
        Ok(self.metrics.messages_sent.load(Ordering::Relaxed))
    }

    /// Get detailed queue statistics
    pub async fn get_stats(&self) -> Result<QueueStats> {
        let messages_sent = self.metrics.messages_sent.load(Ordering::Relaxed);
        let messages_received = self.metrics.messages_received.load(Ordering::Relaxed);
        let total_latency = self.metrics.total_latency_ms.load(Ordering::Relaxed);
        let message_count = self.metrics.message_count.load(Ordering::Relaxed);

        let average_latency_ms = if message_count > 0 {
            total_latency as f64 / message_count as f64
        } else {
            0.0
        };

        // Calculate throughput
        let current_time = current_timestamp_ms();
        let last_check = self.metrics.last_throughput_check.load(Ordering::Relaxed);
        let time_diff_ms = current_time - last_check;
        
        let throughput_msgs_per_sec = if time_diff_ms > 0 {
            let throughput_count = self.metrics.throughput_counter.load(Ordering::Relaxed);
            (throughput_count as f64) / (time_diff_ms as f64 / 1000.0)
        } else {
            0.0
        };

        // Update throughput tracking
        if time_diff_ms > 5000 { // Reset every 5 seconds
            self.metrics.last_throughput_check.store(current_time, Ordering::Relaxed);
            self.metrics.throughput_counter.store(0, Ordering::Relaxed);
        }

        // Get individual queue depths
        let mut queue_depths = HashMap::new();
        for entry in self.worker_queues.iter() {
            let worker_id = entry.key().clone();
            let worker_queue = entry.value();
            let depth = worker_queue.high_priority.0.len() + 
                       worker_queue.normal_priority.0.len() + 
                       worker_queue.low_priority.0.len();
            queue_depths.insert(worker_id, depth as u32);
        }

        let current_depth = self.get_queue_depth().await.unwrap_or(0);

        Ok(QueueStats {
            total_messages_sent: messages_sent,
            total_messages_received: messages_received,
            messages_in_flight: current_depth,
            queue_depths,
            average_latency_ms,
            throughput_msgs_per_sec,
            error_count: self.metrics.messages_dropped.load(Ordering::Relaxed),
        })
    }

    /// Add message type routing preference
    pub async fn add_route(&self, message_type: String, preferred_worker: String) -> Result<()> {
        self.routing_table.insert(message_type, preferred_worker);
        Ok(())
    }

    /// Remove message route
    pub async fn remove_route(&self, message_type: &str) -> Result<()> {
        self.routing_table.remove(message_type);
        Ok(())
    }

    /// Get active worker count
    pub async fn get_active_worker_count(&self) -> usize {
        self.worker_queues.len()
    }

    /// Cleanup expired messages
    pub async fn cleanup_expired_messages(&self) -> Result<u32> {
        let mut cleaned_count = 0u32;
        let current_time = current_timestamp_ms();

        for entry in self.worker_queues.iter() {
            let worker_queue = entry.value();
            
            // This is a simplified cleanup - in production, we'd need more sophisticated expiry handling
            // For now, just update the last activity timestamp
            worker_queue.last_activity.store(current_time, Ordering::Relaxed);
        }

        if cleaned_count > 0 {
            tracing::info!("Cleaned up {} expired messages", cleaned_count);
        }

        Ok(cleaned_count)
    }

    // Private helper methods

    /// Send message to worker queue based on priority
    async fn send_message_internal(&self, worker_id: &str, message: QueueMessage) -> Result<()> {
        let worker_queue = self.worker_queues.get(worker_id)
            .ok_or_else(|| PipelineError::Ipc(format!("Worker {} not found", worker_id)))?;

        let sender = match message.priority {
            MessagePriority::Critical | MessagePriority::High => &worker_queue.high_priority.0,
            MessagePriority::Normal => &worker_queue.normal_priority.0,
            MessagePriority::Low => &worker_queue.low_priority.0,
        };

        match sender.try_send(message) {
            Ok(_) => {
                self.metrics.messages_sent.fetch_add(1, Ordering::Relaxed);
                self.metrics.throughput_counter.fetch_add(1, Ordering::Relaxed);
                worker_queue.last_activity.store(current_timestamp_ms(), Ordering::Relaxed);
                self.message_notify.notify_waiters();
                
                tracing::debug!("Message sent to worker: {}", worker_id);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.metrics.messages_dropped.fetch_add(1, Ordering::Relaxed);
                Err(PipelineError::Ipc(format!("Worker {} queue is full", worker_id)))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.metrics.messages_dropped.fetch_add(1, Ordering::Relaxed);
                Err(PipelineError::Ipc(format!("Worker {} is disconnected", worker_id)))
            }
        }
    }

    /// Try to receive from broadcast channel
    fn try_receive_broadcast(&self) -> Result<QueueMessage, TryRecvError> {
        let receiver = self.broadcast_receiver.read();
        receiver.try_recv()
    }

    /// Try to receive from worker queues (priority order)
    async fn try_receive_from_workers(&self) -> Result<Option<(String, QueueMessage)>> {
        // Check high priority queues first
        for entry in self.worker_queues.iter() {
            let worker_id = entry.key().clone();
            let worker_queue = entry.value();
            
            if let Ok(message) = worker_queue.high_priority.1.try_recv() {
                return Ok(Some((worker_id, message)));
            }
        }

        // Then normal priority
        for entry in self.worker_queues.iter() {
            let worker_id = entry.key().clone();
            let worker_queue = entry.value();
            
            if let Ok(message) = worker_queue.normal_priority.1.try_recv() {
                return Ok(Some((worker_id, message)));
            }
        }

        // Finally low priority
        for entry in self.worker_queues.iter() {
            let worker_id = entry.key().clone();
            let worker_queue = entry.value();
            
            if let Ok(message) = worker_queue.low_priority.1.try_recv() {
                return Ok(Some((worker_id, message)));
            }
        }

        Ok(None)
    }

    /// Determine message priority based on type
    fn determine_message_priority(&self, message: &IpcMessage) -> MessagePriority {
        match message {
            IpcMessage::SystemShutdown => MessagePriority::Critical,
            IpcMessage::Error { .. } => MessagePriority::High,
            IpcMessage::WorkerHeartbeat { .. } => MessagePriority::Low,
            IpcMessage::DocumentProcess { .. } => MessagePriority::Normal,
            IpcMessage::ModelInference { .. } => MessagePriority::Normal,
            IpcMessage::QualityCheck { .. } => MessagePriority::Normal,
            _ => MessagePriority::Normal,
        }
    }

    /// Drain queues for a worker being unregistered
    async fn drain_worker_queues(&self, worker_queue: &WorkerQueue) {
        let mut drained_count = 0u32;
        
        // Drain high priority
        while worker_queue.high_priority.1.try_recv().is_ok() {
            drained_count += 1;
        }
        
        // Drain normal priority
        while worker_queue.normal_priority.1.try_recv().is_ok() {
            drained_count += 1;
        }
        
        // Drain low priority
        while worker_queue.low_priority.1.try_recv().is_ok() {
            drained_count += 1;
        }
        
        if drained_count > 0 {
            tracing::warn!("Drained {} messages from worker {}", drained_count, worker_queue.worker_id);
        }
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::IpcMessage;

    #[tokio::test]
    async fn test_message_queue_basic_operations() {
        let mq = MessageQueue::new().await.unwrap();
        
        // Register worker
        mq.register_worker("test-worker".to_string()).await.unwrap();
        
        // Send message
        let test_message = IpcMessage::WorkerHeartbeat {
            worker_id: "test-worker".to_string(),
            status: crate::ipc::WorkerStatus::Ready,
            metrics: crate::ipc::WorkerMetrics {
                cpu_usage_percent: 50.0,
                memory_usage_mb: 1024,
                tasks_completed: 10,
                average_processing_time_ms: 100.0,
                error_count: 0,
            },
        };
        
        mq.send_to_worker("test-worker", test_message).await.unwrap();
        
        // Receive message
        let (sender, _received_message) = mq.receive().await.unwrap();
        assert_eq!(sender, "test-worker");
        
        // Unregister worker
        mq.unregister_worker("test-worker").await.unwrap();
    }

    #[tokio::test]
    async fn test_priority_message_ordering() {
        let mq = MessageQueue::new().await.unwrap();
        mq.register_worker("priority-worker".to_string()).await.unwrap();
        
        // Send low priority message first
        let low_msg = IpcMessage::WorkerHeartbeat {
            worker_id: "priority-worker".to_string(),
            status: crate::ipc::WorkerStatus::Ready,
            metrics: crate::ipc::WorkerMetrics {
                cpu_usage_percent: 50.0,
                memory_usage_mb: 1024,
                tasks_completed: 10,
                average_processing_time_ms: 100.0,
                error_count: 0,
            },
        };
        mq.send_priority_message("priority-worker", low_msg, MessagePriority::Low).await.unwrap();
        
        // Send high priority message
        let high_msg = IpcMessage::Error {
            error_type: "test".to_string(),
            message: "high priority".to_string(),
            source_component: "test".to_string(),
        };
        mq.send_priority_message("priority-worker", high_msg, MessagePriority::High).await.unwrap();
        
        // High priority should be received first
        let (_, received) = mq.receive().await.unwrap();
        if let IpcMessage::Error { message, .. } = received {
            assert_eq!(message, "high priority");
        } else {
            panic!("Expected high priority message first");
        }
    }

    #[tokio::test]
    async fn test_broadcast_messages() {
        let mq = MessageQueue::new().await.unwrap();
        
        // Register multiple workers
        mq.register_worker("worker1".to_string()).await.unwrap();
        mq.register_worker("worker2".to_string()).await.unwrap();
        
        // Send broadcast
        let broadcast_msg = IpcMessage::SystemShutdown;
        mq.broadcast(broadcast_msg).await.unwrap();
        
        // Should be able to receive broadcast message
        let (sender, received) = mq.receive().await.unwrap();
        assert_eq!(sender, "rust-core");
        assert!(matches!(received, IpcMessage::SystemShutdown));
    }
}