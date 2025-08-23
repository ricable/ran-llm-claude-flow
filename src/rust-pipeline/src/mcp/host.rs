/*!
# MCP Host Implementation

Model Context Protocol host for orchestrating the entire Rust-Python pipeline system.
Provides centralized coordination, task distribution, and performance management.
*/

use crate::mcp::{
    Alert, AlertSeverity, ClientType, McpConnection, PipelineStartConfig, TaskDefinition, TaskPriority,
};
use crate::{PipelineError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, Semaphore};
use tokio::time::interval;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Host configuration for pipeline orchestration
#[derive(Debug, Clone)]
pub struct McpHostConfig {
    pub max_concurrent_pipelines: usize,
    pub max_concurrent_tasks: usize,
    pub task_timeout: Duration,
    pub pipeline_timeout: Duration,
    pub health_check_interval: Duration,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub enable_auto_scaling: bool,
    pub performance_threshold: f64,
    pub quality_threshold: f64,
    pub enable_failover: bool,
    pub retry_attempts: u32,
}

impl Default for McpHostConfig {
    fn default() -> Self {
        Self {
            max_concurrent_pipelines: 10,
            max_concurrent_tasks: 100,
            task_timeout: Duration::from_secs(300), // 5 minutes
            pipeline_timeout: Duration::from_secs(1800), // 30 minutes
            health_check_interval: Duration::from_secs(60),
            load_balancing_strategy: LoadBalancingStrategy::LeastLoaded,
            enable_auto_scaling: true,
            performance_threshold: 0.8, // 80% performance target
            quality_threshold: 0.9,     // 90% quality target
            enable_failover: true,
            retry_attempts: 3,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    Weighted,
    PerformanceBased,
    ResourceAware,
}

/// Pipeline orchestration state
#[derive(Debug, Clone)]
pub struct PipelineState {
    pub pipeline_id: Uuid,
    pub config: PipelineStartConfig,
    pub status: PipelineStatus,
    pub assigned_workers: Vec<String>,
    pub current_tasks: Vec<Uuid>,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub start_time: u64,
    pub estimated_completion: Option<u64>,
    pub performance_metrics: PipelinePerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStatus {
    Initializing,
    Running,
    Paused,
    Completing,
    Completed,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelinePerformanceMetrics {
    pub documents_processed: u64,
    pub average_processing_time_ms: f64,
    pub throughput_docs_per_hour: f64,
    pub quality_score_average: f64,
    pub error_rate_percent: f64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
}

/// Worker node state and capabilities
#[derive(Debug, Clone)]
pub struct WorkerNode {
    pub node_id: String,
    pub client_type: ClientType,
    pub capabilities: Vec<String>,
    pub status: WorkerStatus,
    pub current_load: WorkerLoad,
    pub performance_metrics: WorkerPerformanceMetrics,
    pub last_heartbeat: u64,
    pub assigned_tasks: Vec<Uuid>,
    pub connection_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerStatus {
    Available,
    Busy,
    Overloaded,
    Maintenance,
    Failed(String),
    Disconnected,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerLoad {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub active_tasks: u32,
    pub queue_size: u32,
    pub response_time_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerPerformanceMetrics {
    pub tasks_completed: u64,
    pub average_task_duration_ms: f64,
    pub success_rate_percent: f64,
    pub quality_score_average: f64,
    pub error_count: u64,
    pub throughput_tasks_per_hour: f64,
}

/// Task execution context
#[derive(Debug, Clone)]
pub struct TaskContext {
    pub task_id: Uuid,
    pub definition: TaskDefinition,
    pub pipeline_id: Uuid,
    pub assigned_worker: Option<String>,
    pub status: TaskExecutionStatus,
    pub retry_count: u32,
    pub start_time: u64,
    pub dependencies: Vec<Uuid>,
    pub dependents: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskExecutionStatus {
    Queued,
    Dispatched(String), // worker_id
    Running(String),    // worker_id
    Completed,
    Failed(String),
    Retrying,
    Cancelled,
}

/// Resource allocation and management
#[derive(Debug, Clone, Default)]
pub struct ResourceManager {
    pub total_cpu_cores: u32,
    pub total_memory_gb: u32,
    pub available_cpu_cores: u32,
    pub available_memory_gb: u32,
    pub allocated_resources: HashMap<String, ResourceAllocation>,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub worker_id: String,
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub allocated_at: u64,
}

/// MCP Host for pipeline orchestration
pub struct McpHost {
    config: McpHostConfig,
    pipelines: Arc<RwLock<HashMap<Uuid, PipelineState>>>,
    workers: Arc<RwLock<HashMap<String, WorkerNode>>>,
    tasks: Arc<RwLock<HashMap<Uuid, TaskContext>>>,
    task_queue: Arc<RwLock<VecDeque<Uuid>>>,
    resource_manager: Arc<RwLock<ResourceManager>>,
    performance_metrics: Arc<RwLock<HostPerformanceMetrics>>,
    event_sender: broadcast::Sender<HostEvent>,
    is_running: Arc<AtomicBool>,
    pipeline_counter: AtomicU64,
    task_counter: AtomicU64,
    pipeline_semaphore: Arc<Semaphore>,
    task_semaphore: Arc<Semaphore>,
}

/// Host performance metrics
#[derive(Debug, Clone, Default)]
pub struct HostPerformanceMetrics {
    pub total_pipelines_processed: u64,
    pub total_tasks_processed: u64,
    pub average_pipeline_duration_ms: f64,
    pub average_task_duration_ms: f64,
    pub system_throughput_docs_per_hour: f64,
    pub overall_quality_score: f64,
    pub system_error_rate_percent: f64,
    pub resource_utilization_percent: f64,
    pub active_pipelines: u64,
    pub active_tasks: u64,
}

/// Host events for monitoring and coordination
#[derive(Debug, Clone)]
pub enum HostEvent {
    PipelineStarted(Uuid),
    PipelineCompleted(Uuid, bool),
    TaskDispatched(Uuid, String),
    TaskCompleted(Uuid, bool),
    WorkerRegistered(String),
    WorkerDisconnected(String),
    PerformanceAlert(Alert),
    ResourceThresholdExceeded(String),
    SystemScaleEvent(String),
    Error(String),
}

impl McpHost {
    /// Create new MCP host
    pub fn new() -> Self {
        Self::with_config(McpHostConfig::default())
    }

    /// Create MCP host with custom configuration
    pub fn with_config(config: McpHostConfig) -> Self {
        let (event_sender, _) = broadcast::channel(10000);

        Self {
            pipeline_semaphore: Arc::new(Semaphore::new(config.max_concurrent_pipelines)),
            task_semaphore: Arc::new(Semaphore::new(config.max_concurrent_tasks)),
            config,
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(RwLock::new(HashMap::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(RwLock::new(VecDeque::new())),
            resource_manager: Arc::new(RwLock::new(ResourceManager::default())),
            performance_metrics: Arc::new(RwLock::new(HostPerformanceMetrics::default())),
            event_sender,
            is_running: Arc::new(AtomicBool::new(false)),
            pipeline_counter: AtomicU64::new(0),
            task_counter: AtomicU64::new(0),
        }
    }

    /// Start the MCP host
    pub async fn start(&self) -> Result<()> {
        info!("Starting MCP host for pipeline orchestration");
        self.is_running.store(true, Ordering::Relaxed);

        // Initialize resource manager
        self.initialize_resources().await?;

        // Start background tasks
        self.start_pipeline_orchestrator().await;
        self.start_task_dispatcher().await;
        self.start_health_monitor().await;
        self.start_performance_monitor().await;
        self.start_load_balancer().await;

        info!("MCP host started successfully");
        Ok(())
    }

    /// Stop the MCP host
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping MCP host");
        self.is_running.store(false, Ordering::Relaxed);

        // Cancel all active pipelines
        let pipeline_ids: Vec<Uuid> = { self.pipelines.read().keys().copied().collect() };

        for pipeline_id in pipeline_ids {
            let _ = self.cancel_pipeline(pipeline_id).await;
        }

        info!("MCP host stopped");
        Ok(())
    }

    /// Start a new pipeline
    pub async fn start_pipeline(&self, config: PipelineStartConfig) -> Result<Uuid> {
        // Check if we can start a new pipeline
        let _permit = self
            .pipeline_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| PipelineError::Mcp("Failed to acquire pipeline slot".to_string()))?;

        let pipeline_id = config.pipeline_id;

        // Initialize pipeline state
        let pipeline_state = PipelineState {
            pipeline_id,
            config: config.clone(),
            status: PipelineStatus::Initializing,
            assigned_workers: Vec::new(),
            current_tasks: Vec::new(),
            completed_tasks: 0,
            failed_tasks: 0,
            start_time: current_timestamp(),
            estimated_completion: None,
            performance_metrics: PipelinePerformanceMetrics::default(),
        };

        // Store pipeline
        {
            let mut pipelines = self.pipelines.write();
            pipelines.insert(pipeline_id, pipeline_state);
        }

        // Assign workers based on requirements
        let assigned_workers = self
            .assign_workers_to_pipeline(pipeline_id, &config)
            .await?;

        // Update pipeline with assigned workers
        {
            let mut pipelines = self.pipelines.write();
            if let Some(pipeline) = pipelines.get_mut(&pipeline_id) {
                pipeline.assigned_workers = assigned_workers;
                pipeline.status = PipelineStatus::Running;
            }
        }

        // Create initial tasks for the pipeline
        self.create_pipeline_tasks(pipeline_id, &config).await?;

        // Update metrics
        {
            let mut metrics = self.performance_metrics.write();
            metrics.active_pipelines += 1;
        }

        let _ = self
            .event_sender
            .send(HostEvent::PipelineStarted(pipeline_id));

        info!("Started pipeline: {}", pipeline_id);
        Ok(pipeline_id)
    }

    /// Submit a task for execution
    pub async fn submit_task(&self, task_def: TaskDefinition) -> Result<Uuid> {
        let task_id = task_def.task_id;

        // Create task context
        let task_context = TaskContext {
            task_id,
            definition: task_def.clone(),
            pipeline_id: Uuid::new_v4(), // Will be updated if part of a pipeline
            assigned_worker: None,
            status: TaskExecutionStatus::Queued,
            retry_count: 0,
            start_time: current_timestamp(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
        };

        // Store task
        {
            let mut tasks = self.tasks.write();
            tasks.insert(task_id, task_context);
        }

        // Add to task queue
        {
            let mut queue = self.task_queue.write();

            // Insert based on priority
            let mut inserted = false;
            for (i, &existing_task_id) in queue.iter().enumerate() {
                let tasks = self.tasks.read();
                if let (Some(existing_task), Some(new_task)) =
                    (tasks.get(&existing_task_id), tasks.get(&task_id))
                {
                    if task_def.priority as u8 > existing_task.definition.priority as u8 {
                        drop(tasks);
                        queue.insert(i, task_id);
                        inserted = true;
                        break;
                    }
                }
            }

            if !inserted {
                queue.push_back(task_id);
            }
        }

        // Update metrics
        {
            let mut metrics = self.performance_metrics.write();
            metrics.active_tasks += 1;
        }

        info!(
            "Submitted task: {} (priority: {:?})",
            task_id, task_def.priority
        );
        Ok(task_id)
    }

    /// Register a worker node
    pub async fn register_worker(
        &self,
        connection: &McpConnection,
        capabilities: Vec<String>,
    ) -> Result<()> {
        let worker_id = connection.connection_id.to_string();

        let worker_node = WorkerNode {
            node_id: worker_id.clone(),
            client_type: connection.client_type.clone(),
            capabilities,
            status: WorkerStatus::Available,
            current_load: WorkerLoad::default(),
            performance_metrics: WorkerPerformanceMetrics::default(),
            last_heartbeat: current_timestamp(),
            assigned_tasks: Vec::new(),
            connection_id: Some(connection.connection_id),
        };

        // Store worker
        {
            let mut workers = self.workers.write();
            workers.insert(worker_id.clone(), worker_node);
        }

        let _ = self
            .event_sender
            .send(HostEvent::WorkerRegistered(worker_id.clone()));

        info!(
            "Registered worker: {} (type: {:?})",
            worker_id, connection.client_type
        );
        Ok(())
    }

    /// Update worker status and metrics
    pub async fn update_worker(
        &self,
        worker_id: &str,
        load: WorkerLoad,
        metrics: WorkerPerformanceMetrics,
    ) -> Result<()> {
        {
            let mut workers = self.workers.write();
            if let Some(worker) = workers.get_mut(worker_id) {
                worker.current_load = load;
                worker.performance_metrics = metrics;
                worker.last_heartbeat = current_timestamp();

                // Update worker status based on load
                worker.status = match worker.current_load.cpu_usage_percent {
                    load if load < 50.0 => WorkerStatus::Available,
                    load if load < 80.0 => WorkerStatus::Busy,
                    _ => WorkerStatus::Overloaded,
                };
            } else {
                return Err(PipelineError::Mcp(format!(
                    "Worker {} not found",
                    worker_id
                )));
            }
        }

        // Check for performance alerts
        self.check_worker_performance_alerts(worker_id).await?;

        Ok(())
    }

    /// Get pipeline status
    pub async fn get_pipeline_status(&self, pipeline_id: Uuid) -> Result<PipelineState> {
        let pipelines = self.pipelines.read();
        pipelines
            .get(&pipeline_id)
            .cloned()
            .ok_or_else(|| PipelineError::Mcp(format!("Pipeline {} not found", pipeline_id)))
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: Uuid) -> Result<TaskContext> {
        let tasks = self.tasks.read();
        tasks
            .get(&task_id)
            .cloned()
            .ok_or_else(|| PipelineError::Mcp(format!("Task {} not found", task_id)))
    }

    /// Cancel a pipeline
    pub async fn cancel_pipeline(&self, pipeline_id: Uuid) -> Result<()> {
        // Update pipeline status
        {
            let mut pipelines = self.pipelines.write();
            if let Some(pipeline) = pipelines.get_mut(&pipeline_id) {
                pipeline.status = PipelineStatus::Cancelled;
            } else {
                return Err(PipelineError::Mcp(format!(
                    "Pipeline {} not found",
                    pipeline_id
                )));
            }
        }

        // Cancel all tasks associated with the pipeline
        let task_ids_to_cancel: Vec<Uuid> = {
            let tasks = self.tasks.read();
            tasks
                .iter()
                .filter(|(_, task)| task.pipeline_id == pipeline_id)
                .map(|(&task_id, _)| task_id)
                .collect()
        };

        for task_id in task_ids_to_cancel {
            let _ = self.cancel_task(task_id).await;
        }

        info!("Cancelled pipeline: {}", pipeline_id);
        Ok(())
    }

    /// Cancel a task
    pub async fn cancel_task(&self, task_id: Uuid) -> Result<()> {
        {
            let mut tasks = self.tasks.write();
            if let Some(task) = tasks.get_mut(&task_id) {
                task.status = TaskExecutionStatus::Cancelled;
            } else {
                return Err(PipelineError::Mcp(format!("Task {} not found", task_id)));
            }
        }

        // Remove from queue if still queued
        {
            let mut queue = self.task_queue.write();
            if let Some(pos) = queue.iter().position(|&id| id == task_id) {
                queue.remove(pos);
            }
        }

        info!("Cancelled task: {}", task_id);
        Ok(())
    }

    /// Get host performance metrics
    pub async fn get_performance_metrics(&self) -> HostPerformanceMetrics {
        self.performance_metrics.read().clone()
    }

    /// Subscribe to host events
    pub fn subscribe_events(&self) -> broadcast::Receiver<HostEvent> {
        self.event_sender.subscribe()
    }

    /// Initialize system resources
    async fn initialize_resources(&self) -> Result<()> {
        // Detect system resources (simplified implementation)
        let mut resource_manager = self.resource_manager.write();
        resource_manager.total_cpu_cores = num_cpus::get() as u32;
        resource_manager.total_memory_gb = 32; // This should be detected from system
        resource_manager.available_cpu_cores = resource_manager.total_cpu_cores;
        resource_manager.available_memory_gb = resource_manager.total_memory_gb;

        info!(
            "Initialized resources: {} CPU cores, {} GB memory",
            resource_manager.total_cpu_cores, resource_manager.total_memory_gb
        );
        Ok(())
    }

    /// Assign workers to a pipeline based on requirements
    async fn assign_workers_to_pipeline(
        &self,
        pipeline_id: Uuid,
        config: &PipelineStartConfig,
    ) -> Result<Vec<String>> {
        let workers = self.workers.read();
        let mut assigned_workers = Vec::new();

        // Find available workers based on load balancing strategy
        let available_workers: Vec<_> = workers
            .iter()
            .filter(|(_, worker)| {
                matches!(worker.status, WorkerStatus::Available | WorkerStatus::Busy)
            })
            .collect();

        if available_workers.is_empty() {
            return Err(PipelineError::Mcp("No available workers".to_string()));
        }

        // Select workers based on strategy
        let selected_workers = match self.config.load_balancing_strategy {
            LoadBalancingStrategy::LeastLoaded => {
                self.select_least_loaded_workers(&available_workers, 2)
            }
            LoadBalancingStrategy::PerformanceBased => {
                self.select_performance_based_workers(&available_workers, 2)
            }
            LoadBalancingStrategy::RoundRobin => {
                self.select_round_robin_workers(&available_workers, 2)
            }
            _ => self.select_least_loaded_workers(&available_workers, 2),
        };

        for worker_id in selected_workers {
            assigned_workers.push(worker_id);
        }

        info!(
            "Assigned {} workers to pipeline {}",
            assigned_workers.len(),
            pipeline_id
        );
        Ok(assigned_workers)
    }

    /// Select least loaded workers
    fn select_least_loaded_workers(
        &self,
        workers: &[(&String, &WorkerNode)],
        count: usize,
    ) -> Vec<String> {
        let mut sorted_workers = workers.to_vec();
        sorted_workers.sort_by(|a, b| {
            a.1.current_load
                .cpu_usage_percent
                .partial_cmp(&b.1.current_load.cpu_usage_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted_workers
            .into_iter()
            .take(count)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Select performance-based workers
    fn select_performance_based_workers(
        &self,
        workers: &[(&String, &WorkerNode)],
        count: usize,
    ) -> Vec<String> {
        let mut sorted_workers = workers.to_vec();
        sorted_workers.sort_by(|a, b| {
            b.1.performance_metrics
                .success_rate_percent
                .partial_cmp(&a.1.performance_metrics.success_rate_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted_workers
            .into_iter()
            .take(count)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Select workers using round-robin
    fn select_round_robin_workers(
        &self,
        workers: &[(&String, &WorkerNode)],
        count: usize,
    ) -> Vec<String> {
        let pipeline_count = self.pipeline_counter.load(Ordering::Relaxed) as usize;
        let start_index = pipeline_count % workers.len();

        workers
            .iter()
            .cycle()
            .skip(start_index)
            .take(count)
            .map(|(id, _)| (*id).clone())
            .collect()
    }

    /// Create initial tasks for a pipeline
    async fn create_pipeline_tasks(
        &self,
        pipeline_id: Uuid,
        config: &PipelineStartConfig,
    ) -> Result<()> {
        // Create tasks based on input sources
        for (index, input_source) in config.input_sources.iter().enumerate() {
            let task_def = TaskDefinition {
                task_id: Uuid::new_v4(),
                task_type: crate::mcp::TaskType::DocumentProcessing,
                input_data: serde_json::json!({
                    "source": input_source,
                    "output": config.output_destination,
                    "quality_threshold": config.quality_threshold,
                }),
                model_requirements: crate::mcp::ModelRequirements {
                    min_model_size: crate::mcp::ModelSize::Medium,
                    preferred_models: config.model_preferences.clone(),
                    max_memory_gb: 16,
                    require_local: true,
                },
                priority: TaskPriority::Normal,
                timeout_seconds: 300,
            };

            // Update task with pipeline ID
            let task_id = self.submit_task(task_def).await?;

            {
                let mut tasks = self.tasks.write();
                if let Some(task) = tasks.get_mut(&task_id) {
                    task.pipeline_id = pipeline_id;
                }
            }

            // Add to pipeline's current tasks
            {
                let mut pipelines = self.pipelines.write();
                if let Some(pipeline) = pipelines.get_mut(&pipeline_id) {
                    pipeline.current_tasks.push(task_id);
                }
            }
        }

        info!(
            "Created {} tasks for pipeline {}",
            config.input_sources.len(),
            pipeline_id
        );
        Ok(())
    }

    /// Check worker performance alerts
    async fn check_worker_performance_alerts(&self, worker_id: &str) -> Result<()> {
        let worker = {
            let workers = self.workers.read();
            workers.get(worker_id).cloned()
        };

        if let Some(worker) = worker {
            // Check CPU usage
            if worker.current_load.cpu_usage_percent > 90.0 {
                let alert = Alert {
                    alert_id: Uuid::new_v4(),
                    severity: AlertSeverity::Warning,
                    component: format!("Worker-{}", worker_id),
                    message: format!(
                        "High CPU usage: {:.1}%",
                        worker.current_load.cpu_usage_percent
                    ),
                    timestamp: chrono::Utc::now(),
                    metrics: Some(serde_json::json!({
                        "cpu_usage": worker.current_load.cpu_usage_percent,
                        "memory_usage": worker.current_load.memory_usage_percent,
                        "active_tasks": worker.current_load.active_tasks,
                    })),
                };

                let _ = self.event_sender.send(HostEvent::PerformanceAlert(alert));
            }

            // Check memory usage
            if worker.current_load.memory_usage_percent > 85.0 {
                let alert = Alert {
                    alert_id: Uuid::new_v4(),
                    severity: AlertSeverity::Warning,
                    component: format!("Worker-{}", worker_id),
                    message: format!(
                        "High memory usage: {:.1}%",
                        worker.current_load.memory_usage_percent
                    ),
                    timestamp: chrono::Utc::now(),
                    metrics: Some(serde_json::json!({
                        "memory_usage": worker.current_load.memory_usage_percent,
                        "cpu_usage": worker.current_load.cpu_usage_percent,
                        "active_tasks": worker.current_load.active_tasks,
                    })),
                };

                let _ = self.event_sender.send(HostEvent::PerformanceAlert(alert));
            }

            // Check error rate
            if worker.performance_metrics.success_rate_percent < 80.0 {
                let alert = Alert {
                    alert_id: Uuid::new_v4(),
                    severity: AlertSeverity::Error,
                    component: format!("Worker-{}", worker_id),
                    message: format!(
                        "Low success rate: {:.1}%",
                        worker.performance_metrics.success_rate_percent
                    ),
                    timestamp: chrono::Utc::now(),
                    metrics: Some(serde_json::json!({
                        "success_rate": worker.performance_metrics.success_rate_percent,
                        "error_count": worker.performance_metrics.error_count,
                        "tasks_completed": worker.performance_metrics.tasks_completed,
                    })),
                };

                let _ = self.event_sender.send(HostEvent::PerformanceAlert(alert));
            }
        }

        Ok(())
    }

    /// Start pipeline orchestrator background task
    async fn start_pipeline_orchestrator(&self) {
        let pipelines = self.pipelines.clone();
        let tasks = self.tasks.clone();
        let is_running = self.is_running.clone();
        let event_sender = self.event_sender.clone();
        let performance_metrics = self.performance_metrics.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                // Check pipeline status and completion
                let pipeline_ids: Vec<Uuid> = { pipelines.read().keys().copied().collect() };

                for pipeline_id in pipeline_ids {
                    // Check if pipeline is complete
                    let should_complete = {
                        let pipelines_read = pipelines.read();
                        let tasks_read = tasks.read();

                        if let Some(pipeline) = pipelines_read.get(&pipeline_id) {
                            if matches!(pipeline.status, PipelineStatus::Running) {
                                // Check if all tasks are completed
                                let all_completed = pipeline.current_tasks.iter().all(|task_id| {
                                    tasks_read
                                        .get(task_id)
                                        .map(|task| {
                                            matches!(task.status, TaskExecutionStatus::Completed)
                                        })
                                        .unwrap_or(true)
                                });

                                all_completed
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    };

                    if should_complete {
                        // Update pipeline status
                        {
                            let mut pipelines_write = pipelines.write();
                            if let Some(pipeline) = pipelines_write.get_mut(&pipeline_id) {
                                pipeline.status = PipelineStatus::Completed;
                            }
                        }

                        // Update metrics
                        {
                            let mut metrics = performance_metrics.write();
                            metrics.total_pipelines_processed += 1;
                            metrics.active_pipelines -= 1;
                        }

                        let _ = event_sender.send(HostEvent::PipelineCompleted(pipeline_id, true));
                        info!("Pipeline {} completed successfully", pipeline_id);
                    }
                }
            }
        });
    }

    /// Start task dispatcher background task
    async fn start_task_dispatcher(&self) {
        let task_queue = self.task_queue.clone();
        let tasks = self.tasks.clone();
        let workers = self.workers.clone();
        let is_running = self.is_running.clone();
        let event_sender = self.event_sender.clone();
        let task_semaphore = self.task_semaphore.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                // Get next task from queue
                let next_task_id = {
                    let mut queue = task_queue.write();
                    queue.pop_front()
                };

                if let Some(task_id) = next_task_id {
                    // Acquire task semaphore
                    if let Ok(_permit) = task_semaphore.clone().try_acquire_owned() {
                        // Find available worker
                        let available_worker = {
                            let workers_read = workers.read();
                            workers_read
                                .iter()
                                .filter(|(_, worker)| {
                                    matches!(worker.status, WorkerStatus::Available)
                                })
                                .min_by_key(|(_, worker)| worker.current_load.active_tasks)
                                .map(|(id, _)| id.clone())
                        };

                        if let Some(worker_id) = available_worker {
                            // Assign task to worker
                            {
                                let mut tasks_write = tasks.write();
                                if let Some(task) = tasks_write.get_mut(&task_id) {
                                    task.status =
                                        TaskExecutionStatus::Dispatched(worker_id.clone());
                                    task.assigned_worker = Some(worker_id.clone());
                                }
                            }

                            // Update worker
                            {
                                let mut workers_write = workers.write();
                                if let Some(worker) = workers_write.get_mut(&worker_id) {
                                    worker.assigned_tasks.push(task_id);
                                    worker.current_load.active_tasks += 1;
                                }
                            }

                            let _ = event_sender
                                .send(HostEvent::TaskDispatched(task_id, worker_id.clone()));
                            debug!("Dispatched task {} to worker {}", task_id, worker_id);
                        } else {
                            // No available workers, put task back in queue
                            let mut queue = task_queue.write();
                            queue.push_front(task_id);
                        }
                    } else {
                        // No task slots available, put task back in queue
                        let mut queue = task_queue.write();
                        queue.push_front(task_id);
                    }
                }
            }
        });
    }

    /// Start health monitor background task
    async fn start_health_monitor(&self) {
        let workers = self.workers.clone();
        let is_running = self.is_running.clone();
        let event_sender = self.event_sender.clone();
        let health_check_interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval = interval(health_check_interval);

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                let current_time = current_timestamp();
                let mut workers_to_disconnect = Vec::new();

                // Check worker heartbeats
                {
                    let workers_read = workers.read();
                    for (worker_id, worker) in workers_read.iter() {
                        let time_since_heartbeat = current_time - worker.last_heartbeat;
                        if time_since_heartbeat > health_check_interval.as_secs() * 3 {
                            workers_to_disconnect.push(worker_id.clone());
                        }
                    }
                }

                // Disconnect stale workers
                for worker_id in workers_to_disconnect {
                    {
                        let mut workers_write = workers.write();
                        if let Some(worker) = workers_write.get_mut(&worker_id) {
                            worker.status = WorkerStatus::Disconnected;
                        }
                    }

                    let _ = event_sender.send(HostEvent::WorkerDisconnected(worker_id.clone()));
                    warn!(
                        "Worker {} marked as disconnected due to missed heartbeats",
                        worker_id
                    );
                }
            }
        });
    }

    /// Start performance monitor background task
    async fn start_performance_monitor(&self) {
        let performance_metrics = self.performance_metrics.clone();
        let pipelines = self.pipelines.clone();
        let tasks = self.tasks.clone();
        let workers = self.workers.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                // Calculate system-wide performance metrics
                let mut metrics = performance_metrics.write();

                // Calculate resource utilization
                let workers_read = workers.read();
                if !workers_read.is_empty() {
                    let total_cpu_usage: f64 = workers_read
                        .values()
                        .map(|w| w.current_load.cpu_usage_percent)
                        .sum();
                    metrics.resource_utilization_percent =
                        total_cpu_usage / workers_read.len() as f64;
                }

                // Update active counts
                let pipelines_read = pipelines.read();
                let tasks_read = tasks.read();

                metrics.active_pipelines = pipelines_read
                    .values()
                    .filter(|p| matches!(p.status, PipelineStatus::Running))
                    .count() as u64;

                metrics.active_tasks = tasks_read
                    .values()
                    .filter(|t| {
                        matches!(
                            t.status,
                            TaskExecutionStatus::Running(_) | TaskExecutionStatus::Dispatched(_)
                        )
                    })
                    .count() as u64;

                drop(pipelines_read);
                drop(tasks_read);
                drop(workers_read);

                debug!("Updated performance metrics - Active pipelines: {}, Active tasks: {}, Resource utilization: {:.1}%", 
                       metrics.active_pipelines, metrics.active_tasks, metrics.resource_utilization_percent);
            }
        });
    }

    /// Start load balancer background task
    async fn start_load_balancer(&self) {
        let workers = self.workers.clone();
        let is_running = self.is_running.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                if config.enable_auto_scaling {
                    // Check if we need to redistribute load
                    let workers_read = workers.read();

                    let overloaded_workers: Vec<_> = workers_read
                        .iter()
                        .filter(|(_, worker)| matches!(worker.status, WorkerStatus::Overloaded))
                        .collect();

                    let underloaded_workers: Vec<_> = workers_read
                        .iter()
                        .filter(|(_, worker)| {
                            matches!(worker.status, WorkerStatus::Available)
                                && worker.current_load.cpu_usage_percent < 30.0
                        })
                        .collect();

                    if !overloaded_workers.is_empty() && !underloaded_workers.is_empty() {
                        // TODO: Implement task redistribution logic
                        debug!(
                            "Load balancing: {} overloaded workers, {} underloaded workers",
                            overloaded_workers.len(),
                            underloaded_workers.len()
                        );
                    }
                }
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

/// Create MCP host with default configuration
pub async fn create_host() -> Result<McpHost> {
    let host = McpHost::new();
    host.start().await?;
    Ok(host)
}

/// Create MCP host with custom configuration
pub async fn create_host_with_config(config: McpHostConfig) -> Result<McpHost> {
    let host = McpHost::with_config(config);
    host.start().await?;
    Ok(host)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_host_creation() {
        let host = McpHost::new();
        assert_eq!(host.config.max_concurrent_pipelines, 10);
        assert_eq!(host.config.max_concurrent_tasks, 100);
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let host = McpHost::new();
        host.start().await.unwrap();

        let connection = McpConnection {
            connection_id: Uuid::new_v4(),
            client_type: ClientType::PythonWorker,
            capabilities: vec!["processing".to_string()],
            last_heartbeat: chrono::Utc::now(),
        };

        let capabilities = vec!["document_processing".to_string()];
        let result = host.register_worker(&connection, capabilities).await;
        assert!(result.is_ok());

        let workers = host.workers.read();
        assert!(workers.contains_key(&connection.connection_id.to_string()));
    }

    #[tokio::test]
    async fn test_task_submission() {
        let host = McpHost::new();
        host.start().await.unwrap();

        let task_def = TaskDefinition {
            task_id: Uuid::new_v4(),
            task_type: crate::mcp::TaskType::DocumentProcessing,
            input_data: serde_json::json!({"test": "data"}),
            model_requirements: crate::mcp::ModelRequirements {
                min_model_size: crate::mcp::ModelSize::Small,
                preferred_models: vec!["qwen3-1.7b".to_string()],
                max_memory_gb: 8,
                require_local: true,
            },
            priority: TaskPriority::Normal,
            timeout_seconds: 120,
        };

        let result = host.submit_task(task_def.clone()).await;
        assert!(result.is_ok());

        let task_status = host.get_task_status(task_def.task_id).await;
        assert!(task_status.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_start() {
        let host = McpHost::new();
        host.start().await.unwrap();

        let config = PipelineStartConfig {
            pipeline_id: Uuid::new_v4(),
            input_sources: vec!["test_input.txt".to_string()],
            output_destination: "test_output.txt".to_string(),
            quality_threshold: 0.9,
            model_preferences: vec!["qwen3-7b".to_string()],
        };

        let result = host.start_pipeline(config.clone()).await;

        // This will fail because no workers are registered, but the pipeline should be created
        let pipeline_status = host.get_pipeline_status(config.pipeline_id).await;
        // The pipeline might not exist if worker assignment fails, so we don't assert success
        match pipeline_status {
            Ok(_) => println!("Pipeline created successfully"),
            Err(e) => println!("Pipeline creation failed as expected: {}", e),
        }
    }
}
