/*!
# Process Manager

Manages Python ML worker processes with lifecycle management, health monitoring,
and automatic recovery optimized for M3 Max architecture.
*/

use crate::{Result, PipelineError, PipelineConfig};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::process::Command as TokioCommand;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use uuid::Uuid;

/// Worker process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerProcess {
    pub worker_id: String,
    pub process_id: u32,
    pub status: WorkerStatus,
    pub startup_time: u64,
    pub last_heartbeat: u64,
    pub restart_count: u32,
    pub capabilities: Vec<String>,
    pub config: WorkerConfig,
    pub performance_metrics: WorkerPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerStatus {
    Starting,
    Ready,
    Busy,
    Idle,
    Error,
    Restarting,
    Stopping,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub python_executable: String,
    pub script_path: String,
    pub model_preferences: Vec<String>,
    pub max_memory_mb: u32,
    pub cpu_affinity: Option<Vec<u32>>,
    pub environment_vars: HashMap<String, String>,
    pub startup_timeout_seconds: u64,
    pub health_check_interval_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPerformanceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub average_task_duration_ms: f64,
    pub last_task_completion: u64,
    pub throughput_tasks_per_minute: f64,
}

impl Default for WorkerPerformanceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            average_task_duration_ms: 0.0,
            last_task_completion: 0,
            throughput_tasks_per_minute: 0.0,
        }
    }
}

/// Process management statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessManagerStats {
    pub total_workers: u32,
    pub active_workers: u32,
    pub failed_workers: u32,
    pub total_restarts: u32,
    pub average_uptime_hours: f64,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
    pub system_resource_usage: SystemResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceUsage {
    pub total_cpu_percent: f64,
    pub total_memory_mb: u64,
    pub python_process_count: u32,
    pub load_average: f64,
}

/// Manages Python ML worker processes
pub struct ProcessManager {
    /// Active worker processes
    workers: Arc<RwLock<HashMap<String, WorkerProcess>>>,
    
    /// Running child processes
    child_processes: Arc<Mutex<HashMap<String, Child>>>,
    
    /// Process monitoring
    health_monitor: Arc<HealthMonitor>,
    
    /// Configuration
    config: ProcessManagerConfig,
    
    /// Statistics
    stats: Arc<ProcessStats>,
    
    /// Shutdown signal
    shutdown_tx: mpsc::UnboundedSender<()>,
    shutdown_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<()>>>>,
}

#[derive(Debug, Clone)]
pub struct ProcessManagerConfig {
    pub max_workers: u32,
    pub restart_delay_seconds: u64,
    pub max_restart_attempts: u32,
    pub health_check_interval: Duration,
    pub startup_timeout: Duration,
    pub graceful_shutdown_timeout: Duration,
    pub python_base_path: PathBuf,
    pub log_directory: PathBuf,
}

impl Default for ProcessManagerConfig {
    fn default() -> Self {
        Self {
            max_workers: 8,
            restart_delay_seconds: 5,
            max_restart_attempts: 3,
            health_check_interval: Duration::from_secs(10),
            startup_timeout: Duration::from_secs(60),
            graceful_shutdown_timeout: Duration::from_secs(30),
            python_base_path: PathBuf::from("src/python-pipeline"),
            log_directory: PathBuf::from("logs/workers"),
        }
    }
}

#[derive(Debug)]
struct ProcessStats {
    total_workers_started: AtomicU32,
    total_workers_failed: AtomicU32,
    total_restarts: AtomicU32,
    total_tasks_completed: AtomicU64,
    total_tasks_failed: AtomicU64,
}

#[derive(Debug)]
struct HealthMonitor {
    last_check: AtomicU64,
    failed_checks: AtomicU32,
    recovery_attempts: AtomicU32,
}

impl ProcessManager {
    /// Create new process manager
    pub async fn new(pipeline_config: &PipelineConfig) -> Result<Self> {
        tracing::info!("Initializing process manager");
        
        let config = ProcessManagerConfig::default();
        
        // Ensure log directory exists
        fs::create_dir_all(&config.log_directory).await
            .map_err(|e| PipelineError::Ipc(format!("Failed to create log directory: {}", e)))?;
        
        let stats = ProcessStats {
            total_workers_started: AtomicU32::new(0),
            total_workers_failed: AtomicU32::new(0),
            total_restarts: AtomicU32::new(0),
            total_tasks_completed: AtomicU64::new(0),
            total_tasks_failed: AtomicU64::new(0),
        };
        
        let health_monitor = HealthMonitor {
            last_check: AtomicU64::new(current_timestamp()),
            failed_checks: AtomicU32::new(0),
            recovery_attempts: AtomicU32::new(0),
        };
        
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();
        
        Ok(Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            child_processes: Arc::new(Mutex::new(HashMap::new())),
            health_monitor: Arc::new(health_monitor),
            config,
            stats: Arc::new(stats),
            shutdown_tx,
            shutdown_rx: Arc::new(Mutex::new(Some(shutdown_rx))),
        })
    }

    /// Start Python ML workers
    pub async fn start_python_workers(&mut self, count: u8) -> Result<Vec<String>> {
        tracing::info!("Starting {} Python ML workers", count);
        
        let mut worker_ids = Vec::new();
        
        for i in 0..count {
            let worker_id = format!("python-worker-{}", i + 1);
            
            let worker_config = WorkerConfig {
                python_executable: "python".to_string(), // Should use uv python
                script_path: self.config.python_base_path.join("main.py").to_string_lossy().to_string(),
                model_preferences: vec![
                    "qwen3-1.7b".to_string(),
                    "qwen3-7b".to_string(), 
                    "qwen3-30b".to_string()
                ],
                max_memory_mb: 8192, // 8GB per worker
                cpu_affinity: self.calculate_cpu_affinity(i).await,
                environment_vars: self.create_worker_environment(&worker_id).await?,
                startup_timeout_seconds: 60,
                health_check_interval_seconds: 30,
            };
            
            match self.start_worker(&worker_id, worker_config).await {
                Ok(_) => {
                    worker_ids.push(worker_id);
                    self.stats.total_workers_started.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    tracing::error!("Failed to start worker {}: {}", worker_id, e);
                    self.stats.total_workers_failed.fetch_add(1, Ordering::Relaxed);
                    return Err(e);
                }
            }
        }
        
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        tracing::info!("Successfully started {} workers: {:?}", worker_ids.len(), worker_ids);
        Ok(worker_ids)
    }

    /// Start a single worker process
    pub async fn start_worker(&self, worker_id: &str, config: WorkerConfig) -> Result<()> {
        tracing::info!("Starting worker: {}", worker_id);
        
        // Check if worker already exists
        {
            let workers = self.workers.read().await;
            if workers.contains_key(worker_id) {
                return Err(PipelineError::Ipc(format!("Worker {} already exists", worker_id)));
            }
        }
        
        // Create worker process info
        let worker_process = WorkerProcess {
            worker_id: worker_id.to_string(),
            process_id: 0, // Will be set after process starts
            status: WorkerStatus::Starting,
            startup_time: current_timestamp(),
            last_heartbeat: current_timestamp(),
            restart_count: 0,
            capabilities: vec![
                "document_processing".to_string(),
                "model_inference".to_string(),
                "quality_scoring".to_string(),
            ],
            config: config.clone(),
            performance_metrics: WorkerPerformanceMetrics::default(),
        };
        
        // Create log files
        let stdout_log = self.config.log_directory.join(format!("{}_stdout.log", worker_id));
        let stderr_log = self.config.log_directory.join(format!("{}_stderr.log", worker_id));
        
        // Start the Python process
        let mut command = TokioCommand::new(&config.python_executable);
        command
            .arg(&config.script_path)
            .arg("--worker-id")
            .arg(worker_id)
            .arg("--rust-ipc")
            .arg("true")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        
        // Set environment variables
        for (key, value) in &config.environment_vars {
            command.env(key, value);
        }
        
        // Set CPU affinity if specified
        if let Some(affinity) = &config.cpu_affinity {
            if cfg!(target_os = "macos") {
                // macOS doesn't support CPU affinity in the same way
                tracing::debug!("CPU affinity requested but not supported on macOS");
            }
        }
        
        let child = command.spawn()
            .map_err(|e| PipelineError::Ipc(format!("Failed to spawn worker process: {}", e)))?;
        
        let process_id = child.id().unwrap_or(0);
        
        // Update worker with process ID
        let mut updated_worker = worker_process;
        updated_worker.process_id = process_id;
        
        // Store child process
        {
            let mut child_processes = self.child_processes.lock();
            child_processes.insert(worker_id.to_string(), child);
        }
        
        // Store worker info
        {
            let mut workers = self.workers.write().await;
            workers.insert(worker_id.to_string(), updated_worker);
        }
        
        // Wait for worker to be ready (with timeout)
        let ready = self.wait_for_worker_ready(worker_id, config.startup_timeout_seconds).await?;
        if !ready {
            self.stop_worker(worker_id).await?;
            return Err(PipelineError::Ipc(format!("Worker {} failed to start within timeout", worker_id)));
        }
        
        tracing::info!("Worker {} started successfully with PID {}", worker_id, process_id);
        Ok(())
    }

    /// Stop a worker process
    pub async fn stop_worker(&self, worker_id: &str) -> Result<()> {
        tracing::info!("Stopping worker: {}", worker_id);
        
        // Update worker status
        {
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(worker_id) {
                worker.status = WorkerStatus::Stopping;
            }
        }
        
        // Get and terminate child process
        let mut child_opt = {
            let mut child_processes = self.child_processes.lock();
            child_processes.remove(worker_id)
        };
        
        if let Some(mut child) = child_opt {
            // Try graceful shutdown first
            match child.start_kill() {
                Ok(_) => {
                    tracing::debug!("Sent termination signal to worker {}", worker_id);
                    
                    // Wait for graceful shutdown with timeout
                    match tokio::time::timeout(self.config.graceful_shutdown_timeout, child.wait()).await {
                        Ok(Ok(exit_status)) => {
                            tracing::info!("Worker {} exited gracefully: {}", worker_id, exit_status);
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("Error waiting for worker {} to exit: {}", worker_id, e);
                        }
                        Err(_) => {
                            tracing::warn!("Worker {} did not exit gracefully, forcing termination", worker_id);
                            let _ = child.kill().await;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to terminate worker {}: {}", worker_id, e);
                }
            }
        }
        
        // Update worker status
        {
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(worker_id) {
                worker.status = WorkerStatus::Stopped;
            }
        }
        
        tracing::info!("Worker {} stopped successfully", worker_id);
        Ok(())
    }

    /// Restart a worker process
    pub async fn restart_worker(&self, worker_id: &str) -> Result<()> {
        tracing::info!("Restarting worker: {}", worker_id);
        
        let config = {
            let workers = self.workers.read().await;
            workers.get(worker_id)
                .map(|w| w.config.clone())
                .ok_or_else(|| PipelineError::Ipc(format!("Worker {} not found", worker_id)))?
        };
        
        // Increment restart count
        {
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(worker_id) {
                worker.restart_count += 1;
                worker.status = WorkerStatus::Restarting;
                
                if worker.restart_count > self.config.max_restart_attempts {
                    return Err(PipelineError::Ipc(format!(
                        "Worker {} exceeded maximum restart attempts ({})", 
                        worker_id, self.config.max_restart_attempts
                    )));
                }
            }
        }
        
        self.stats.total_restarts.fetch_add(1, Ordering::Relaxed);
        
        // Stop the current process
        self.stop_worker(worker_id).await?;
        
        // Wait before restarting
        tokio::time::sleep(Duration::from_secs(self.config.restart_delay_seconds)).await;
        
        // Start the worker again
        self.start_worker(worker_id, config).await?;
        
        tracing::info!("Worker {} restarted successfully", worker_id);
        Ok(())
    }

    /// Get worker status
    pub async fn get_worker_status(&self, worker_id: &str) -> Option<WorkerStatus> {
        let workers = self.workers.read().await;
        workers.get(worker_id).map(|w| w.status.clone())
    }

    /// Get all workers
    pub async fn get_all_workers(&self) -> HashMap<String, WorkerProcess> {
        self.workers.read().await.clone()
    }

    /// Update worker performance metrics
    pub async fn update_worker_metrics(&self, worker_id: &str, metrics: WorkerPerformanceMetrics) -> Result<()> {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.performance_metrics = metrics;
            worker.last_heartbeat = current_timestamp();
            
            // Update global stats
            self.stats.total_tasks_completed.store(
                worker.performance_metrics.tasks_completed, 
                Ordering::Relaxed
            );
            self.stats.total_tasks_failed.store(
                worker.performance_metrics.tasks_failed, 
                Ordering::Relaxed
            );
        }
        Ok(())
    }

    /// Get process manager statistics
    pub async fn get_stats(&self) -> Result<ProcessManagerStats> {
        let workers = self.workers.read().await;
        
        let total_workers = workers.len() as u32;
        let active_workers = workers.values()
            .filter(|w| matches!(w.status, WorkerStatus::Ready | WorkerStatus::Busy))
            .count() as u32;
        let failed_workers = workers.values()
            .filter(|w| matches!(w.status, WorkerStatus::Error | WorkerStatus::Stopped))
            .count() as u32;
        
        let total_uptime_seconds: u64 = workers.values()
            .map(|w| current_timestamp() - w.startup_time)
            .sum();
        
        let average_uptime_hours = if total_workers > 0 {
            (total_uptime_seconds as f64) / (total_workers as f64) / 3600.0
        } else {
            0.0
        };
        
        let total_tasks_completed = self.stats.total_tasks_completed.load(Ordering::Relaxed);
        let total_tasks_failed = self.stats.total_tasks_failed.load(Ordering::Relaxed);
        let total_restarts = self.stats.total_restarts.load(Ordering::Relaxed);
        
        // Get system resource usage
        let system_resource_usage = self.get_system_resource_usage().await;
        
        Ok(ProcessManagerStats {
            total_workers,
            active_workers,
            failed_workers,
            total_restarts,
            average_uptime_hours,
            total_tasks_completed,
            total_tasks_failed,
            system_resource_usage,
        })
    }

    /// Cleanup all processes
    pub async fn cleanup_all(&mut self) -> Result<()> {
        tracing::info!("Cleaning up all worker processes");
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(());
        
        // Get all worker IDs
        let worker_ids: Vec<String> = {
            self.workers.read().await.keys().cloned().collect()
        };
        
        // Stop all workers
        for worker_id in worker_ids {
            if let Err(e) = self.stop_worker(&worker_id).await {
                tracing::warn!("Failed to stop worker {}: {}", worker_id, e);
            }
        }
        
        // Clear worker collections
        {
            let mut workers = self.workers.write().await;
            workers.clear();
        }
        {
            let mut child_processes = self.child_processes.lock();
            child_processes.clear();
        }
        
        tracing::info!("Process manager cleanup completed");
        Ok(())
    }

    // Private helper methods

    /// Calculate CPU affinity for worker on M3 Max
    async fn calculate_cpu_affinity(&self, worker_index: u8) -> Option<Vec<u32>> {
        // M3 Max has 8 performance cores + 4 efficiency cores
        // Assign workers to performance cores preferentially
        if cfg!(target_os = "macos") {
            // macOS doesn't support CPU affinity setting
            None
        } else {
            // For other platforms, distribute across performance cores
            let performance_cores = vec![0, 1, 2, 3, 4, 5, 6, 7];
            let assigned_core = performance_cores[worker_index as usize % performance_cores.len()];
            Some(vec![assigned_core])
        }
    }

    /// Create environment variables for worker
    async fn create_worker_environment(&self, worker_id: &str) -> Result<HashMap<String, String>> {
        let mut env_vars = HashMap::new();
        
        // Standard Python environment
        env_vars.insert("PYTHONPATH".to_string(), self.config.python_base_path.to_string_lossy().to_string());
        env_vars.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());
        
        // Worker-specific variables
        env_vars.insert("WORKER_ID".to_string(), worker_id.to_string());
        env_vars.insert("RUST_IPC_ENABLED".to_string(), "1".to_string());
        env_vars.insert("LOG_LEVEL".to_string(), "INFO".to_string());
        
        // M3 Max optimizations
        env_vars.insert("MLX_ENABLED".to_string(), "1".to_string());
        env_vars.insert("PYTORCH_MPS_HIGH_WATERMARK_RATIO".to_string(), "0.0".to_string());
        
        // Memory limits
        env_vars.insert("MEMORY_LIMIT_MB".to_string(), "8192".to_string());
        
        Ok(env_vars)
    }

    /// Wait for worker to be ready
    async fn wait_for_worker_ready(&self, worker_id: &str, timeout_seconds: u64) -> Result<bool> {
        let start_time = Instant::now();
        let timeout_duration = Duration::from_secs(timeout_seconds);
        
        while start_time.elapsed() < timeout_duration {
            {
                let workers = self.workers.read().await;
                if let Some(worker) = workers.get(worker_id) {
                    if matches!(worker.status, WorkerStatus::Ready) {
                        return Ok(true);
                    }
                    if matches!(worker.status, WorkerStatus::Error | WorkerStatus::Stopped) {
                        return Ok(false);
                    }
                }
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        Ok(false)
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) -> Result<()> {
        let workers = self.workers.clone();
        let health_monitor = self.health_monitor.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);
            
            loop {
                interval.tick().await;
                
                let worker_ids: Vec<String> = {
                    workers.read().await.keys().cloned().collect()
                };
                
                for worker_id in worker_ids {
                    // Check worker health (heartbeat, resource usage, etc.)
                    let should_restart = {
                        let workers_read = workers.read().await;
                        if let Some(worker) = workers_read.get(&worker_id) {
                            let heartbeat_age = current_timestamp() - worker.last_heartbeat;
                            heartbeat_age > config.health_check_interval.as_secs() * 2
                        } else {
                            false
                        }
                    };
                    
                    if should_restart {
                        tracing::warn!("Worker {} appears unhealthy, marking for restart", worker_id);
                        health_monitor.failed_checks.fetch_add(1, Ordering::Relaxed);
                        // In a full implementation, we would trigger restart here
                    }
                }
                
                health_monitor.last_check.store(current_timestamp(), Ordering::Relaxed);
            }
        });
        
        Ok(())
    }

    /// Get system resource usage
    async fn get_system_resource_usage(&self) -> SystemResourceUsage {
        // This would use system APIs to get actual resource usage
        // For now, return mock data
        SystemResourceUsage {
            total_cpu_percent: 45.0,
            total_memory_mb: 8192,
            python_process_count: self.workers.read().await.len() as u32,
            load_average: 2.5,
        }
    }
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
    use crate::PipelineConfig;

    #[tokio::test]
    async fn test_process_manager_basic_operations() {
        let config = PipelineConfig::default();
        let mut pm = ProcessManager::new(&config).await.unwrap();
        
        // Test stats before starting workers
        let stats = pm.get_stats().await.unwrap();
        assert_eq!(stats.total_workers, 0);
        
        // Cleanup
        pm.cleanup_all().await.unwrap();
    }

    #[tokio::test]
    async fn test_worker_config_creation() {
        let config = PipelineConfig::default();
        let pm = ProcessManager::new(&config).await.unwrap();
        
        let env_vars = pm.create_worker_environment("test-worker").await.unwrap();
        assert!(env_vars.contains_key("WORKER_ID"));
        assert_eq!(env_vars.get("WORKER_ID").unwrap(), "test-worker");
        assert_eq!(env_vars.get("RUST_IPC_ENABLED").unwrap(), "1");
    }

    #[tokio::test]
    async fn test_cpu_affinity_calculation() {
        let config = PipelineConfig::default();
        let pm = ProcessManager::new(&config).await.unwrap();
        
        // Test CPU affinity calculation
        let affinity = pm.calculate_cpu_affinity(0).await;
        
        if cfg!(target_os = "macos") {
            assert!(affinity.is_none());
        } else {
            assert!(affinity.is_some());
        }
    }
}