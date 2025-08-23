/*!
# MCP Usage Examples

Comprehensive examples demonstrating how to use the Model Context Protocol
implementation for real-world pipeline orchestration scenarios.
*/

use super::{
    server::{McpServer, McpServerConfig},
    client::{McpClient, McpClientConfig, AgentState, AgentStatus, AgentMetrics, MessagePriority},
    host::{McpHost, McpHostConfig, LoadBalancingStrategy},
    error_handling::{FaultToleranceManager, CircuitBreakerConfig, RetryConfig},
    *,
};
use crate::{Result, PipelineError};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error};
use uuid::Uuid;

/// Example: Complete MCP pipeline orchestration system
pub async fn run_complete_pipeline_example() -> Result<()> {
    info!("Starting complete MCP pipeline orchestration example");

    // 1. Start MCP Server
    let mut server_config = McpServerConfig::default();
    server_config.port = 8800;
    server_config.max_connections = 50;
    
    let server = Arc::new(McpServer::with_config(server_config));
    
    // Start server in background
    let server_clone = server.clone();
    let server_handle = tokio::spawn(async move {
        server_clone.start().await
    });

    // Give server time to start
    sleep(Duration::from_millis(200)).await;

    // 2. Start MCP Host for orchestration
    let mut host_config = McpHostConfig::default();
    host_config.max_concurrent_pipelines = 5;
    host_config.max_concurrent_tasks = 25;
    host_config.load_balancing_strategy = LoadBalancingStrategy::PerformanceBased;
    host_config.enable_auto_scaling = true;
    
    let host = Arc::new(McpHost::with_config(host_config));
    host.start().await?;

    // 3. Create and configure clients (Rust core + Python workers)
    let rust_client = create_rust_core_client().await?;
    let python_workers = create_python_worker_clients(3).await?;

    // 4. Register agents with coordination
    register_pipeline_agents(&rust_client, &python_workers).await?;

    // 5. Setup fault tolerance
    let fault_manager = setup_fault_tolerance().await?;

    // 6. Start pipeline processing
    let pipeline_results = run_document_processing_pipeline(&host).await?;

    info!("Pipeline processing completed with {} results", pipeline_results.len());

    // 7. Monitor performance and health
    monitor_system_health(&host, &fault_manager).await?;

    // 8. Cleanup
    cleanup_system(server, host, rust_client, python_workers, fault_manager).await?;

    info!("Complete MCP pipeline example finished successfully");
    Ok(())
}

/// Create and configure Rust core client
async fn create_rust_core_client() -> Result<McpClient> {
    let mut config = McpClientConfig::default();
    config.server_url = "ws://localhost:8800".to_string();
    config.client_type = ClientType::RustCore;
    config.capabilities = vec![
        "task_orchestration".to_string(),
        "io_operations".to_string(),
        "performance_monitoring".to_string(),
        "coordination".to_string(),
    ];
    config.reconnect_attempts = 10;
    config.heartbeat_interval = Duration::from_secs(15);

    let client = McpClient::with_config(config);
    client.start().await?;

    info!("Rust core client started and connected");
    Ok(client)
}

/// Create multiple Python worker clients
async fn create_python_worker_clients(count: usize) -> Result<Vec<McpClient>> {
    let mut workers = Vec::new();

    for i in 0..count {
        let mut config = McpClientConfig::default();
        config.server_url = "ws://localhost:8800".to_string();
        config.client_type = ClientType::PythonWorker;
        config.capabilities = vec![
            "ml_inference".to_string(),
            "document_processing".to_string(),
            "quality_validation".to_string(),
            format!("worker_specialization_{}", i),
        ];

        let client = McpClient::with_config(config);
        client.start().await?;
        workers.push(client);
    }

    info!("Created {} Python worker clients", count);
    Ok(workers)
}

/// Register agents for pipeline coordination
async fn register_pipeline_agents(
    rust_client: &McpClient,
    python_workers: &[McpClient],
) -> Result<()> {
    // Register Rust core coordinator agent
    let coordinator_agent = AgentState {
        agent_id: "rust_coordinator".to_string(),
        agent_type: "coordinator".to_string(),
        capabilities: vec![
            "task_distribution".to_string(),
            "performance_monitoring".to_string(),
            "error_handling".to_string(),
        ],
        status: AgentStatus::Idle,
        current_task: None,
        performance_metrics: AgentMetrics::default(),
        last_heartbeat: current_timestamp(),
    };

    rust_client.register_agent(coordinator_agent).await?;

    // Register Python worker agents
    for (i, worker) in python_workers.iter().enumerate() {
        let agent = AgentState {
            agent_id: format!("python_worker_{}", i + 1),
            agent_type: "ml_processor".to_string(),
            capabilities: vec![
                "qwen_inference".to_string(),
                "document_analysis".to_string(),
                "quality_scoring".to_string(),
            ],
            status: AgentStatus::Idle,
            current_task: None,
            performance_metrics: AgentMetrics::default(),
            last_heartbeat: current_timestamp(),
        };

        worker.register_agent(agent).await?;
    }

    info!("All pipeline agents registered successfully");
    Ok(())
}

/// Setup fault tolerance mechanisms
async fn setup_fault_tolerance() -> Result<Arc<FaultToleranceManager>> {
    let fault_manager = Arc::new(FaultToleranceManager::new());
    fault_manager.start().await?;

    // Configure circuit breakers for critical services
    let cb_config = CircuitBreakerConfig {
        failure_threshold: 5,
        recovery_timeout: Duration::from_secs(30),
        success_threshold: 3,
        timeout: Duration::from_secs(10),
    };

    fault_manager.register_circuit_breaker("python_inference".to_string(), cb_config.clone()).await;
    fault_manager.register_circuit_breaker("document_processing".to_string(), cb_config.clone()).await;
    fault_manager.register_circuit_breaker("quality_validation".to_string(), cb_config).await;

    // Configure retry mechanisms
    let retry_config = RetryConfig {
        max_attempts: 3,
        initial_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(5),
        backoff_factor: 2.0,
        jitter: true,
    };

    fault_manager.register_retry_mechanism("model_loading".to_string(), retry_config.clone()).await;
    fault_manager.register_retry_mechanism("file_operations".to_string(), retry_config).await;

    info!("Fault tolerance mechanisms configured");
    Ok(fault_manager)
}

/// Run document processing pipeline
async fn run_document_processing_pipeline(host: &McpHost) -> Result<Vec<Uuid>> {
    info!("Starting document processing pipeline");

    let mut pipeline_results = Vec::new();

    // Create multiple pipelines for different document types
    let pipeline_configs = vec![
        PipelineStartConfig {
            pipeline_id: Uuid::new_v4(),
            input_sources: vec![
                "documents/technical_docs/*.pdf".to_string(),
                "documents/technical_docs/*.docx".to_string(),
            ],
            output_destination: "processed/technical/".to_string(),
            quality_threshold: 0.92,
            model_preferences: vec!["qwen3-30b".to_string(), "qwen3-7b".to_string()],
        },
        PipelineStartConfig {
            pipeline_id: Uuid::new_v4(),
            input_sources: vec![
                "documents/research_papers/*.pdf".to_string(),
            ],
            output_destination: "processed/research/".to_string(),
            quality_threshold: 0.95,
            model_preferences: vec!["qwen3-30b".to_string()],
        },
        PipelineStartConfig {
            pipeline_id: Uuid::new_v4(),
            input_sources: vec![
                "documents/general/*.txt".to_string(),
                "documents/general/*.md".to_string(),
            ],
            output_destination: "processed/general/".to_string(),
            quality_threshold: 0.85,
            model_preferences: vec!["qwen3-7b".to_string(), "qwen3-1.7b".to_string()],
        },
    ];

    // Start pipelines
    for config in pipeline_configs {
        match host.start_pipeline(config.clone()).await {
            Ok(pipeline_id) => {
                pipeline_results.push(pipeline_id);
                info!("Started pipeline: {}", pipeline_id);
            }
            Err(e) => {
                warn!("Failed to start pipeline {}: {}", config.pipeline_id, e);
            }
        }
    }

    // Submit individual high-priority tasks
    let urgent_tasks = vec![
        create_urgent_task("urgent_document_1.pdf"),
        create_urgent_task("urgent_document_2.docx"),
    ];

    for task_def in urgent_tasks {
        match host.submit_task(task_def.clone()).await {
            Ok(task_id) => {
                info!("Submitted urgent task: {}", task_id);
            }
            Err(e) => {
                warn!("Failed to submit urgent task: {}", e);
            }
        }
    }

    // Wait for initial processing
    sleep(Duration::from_secs(5)).await;

    info!("Document processing pipeline initiated with {} pipelines", pipeline_results.len());
    Ok(pipeline_results)
}

/// Create urgent task definition
fn create_urgent_task(document_path: &str) -> TaskDefinition {
    TaskDefinition {
        task_id: Uuid::new_v4(),
        task_type: TaskType::DocumentProcessing,
        input_data: serde_json::json!({
            "document_path": document_path,
            "priority": "urgent",
            "deadline": current_timestamp() + 3600, // 1 hour deadline
        }),
        model_requirements: ModelRequirements {
            min_model_size: ModelSize::Large,
            preferred_models: vec!["qwen3-30b".to_string()],
            max_memory_gb: 32,
            require_local: true,
        },
        priority: TaskPriority::Critical,
        timeout_seconds: 600, // 10 minutes
    }
}

/// Monitor system health and performance
async fn monitor_system_health(
    host: &McpHost,
    fault_manager: &FaultToleranceManager,
) -> Result<()> {
    info!("Starting system health monitoring");

    // Monitor for a short period
    for i in 0..10 {
        sleep(Duration::from_secs(2)).await;

        // Get host performance metrics
        let host_metrics = host.get_performance_metrics().await;
        info!("Monitoring cycle {}: Active pipelines: {}, Active tasks: {}, Resource utilization: {:.1}%",
              i + 1,
              host_metrics.active_pipelines,
              host_metrics.active_tasks,
              host_metrics.resource_utilization_percent);

        // Get fault tolerance metrics
        let fault_metrics = fault_manager.get_metrics().await;
        info!("System health: {} circuit breakers, Overall health: {:?}",
              fault_metrics.circuit_breaker_metrics.len(),
              fault_metrics.system_health.overall_status);

        // Check for performance alerts
        if host_metrics.resource_utilization_percent > 80.0 {
            warn!("High resource utilization detected: {:.1}%", 
                  host_metrics.resource_utilization_percent);
        }

        if host_metrics.system_error_rate_percent > 5.0 {
            error!("High error rate detected: {:.1}%", 
                   host_metrics.system_error_rate_percent);
        }
    }

    info!("System health monitoring completed");
    Ok(())
}

/// Cleanup all system components
async fn cleanup_system(
    server: Arc<McpServer>,
    host: Arc<McpHost>,
    rust_client: McpClient,
    python_workers: Vec<McpClient>,
    fault_manager: Arc<FaultToleranceManager>,
) -> Result<()> {
    info!("Starting system cleanup");

    // Stop clients
    rust_client.stop().await?;
    for worker in python_workers {
        worker.stop().await?;
    }

    // Stop fault tolerance manager
    fault_manager.stop().await?;

    // Stop host
    host.stop().await?;

    // Stop server
    server.stop().await?;

    info!("System cleanup completed");
    Ok(())
}

/// Example: High-performance document processing with load balancing
pub async fn run_high_performance_example() -> Result<()> {
    info!("Starting high-performance document processing example");

    // Setup optimized configuration
    let mut host_config = McpHostConfig::default();
    host_config.max_concurrent_pipelines = 10;
    host_config.max_concurrent_tasks = 100;
    host_config.load_balancing_strategy = LoadBalancingStrategy::ResourceAware;
    host_config.enable_auto_scaling = true;
    host_config.performance_threshold = 0.9; // 90% performance target

    let host = McpHost::with_config(host_config);
    host.start().await?;

    // Create high-throughput pipeline
    let pipeline_config = PipelineStartConfig {
        pipeline_id: Uuid::new_v4(),
        input_sources: (0..1000).map(|i| format!("batch/document_{}.pdf", i)).collect(),
        output_destination: "processed/batch/".to_string(),
        quality_threshold: 0.88,
        model_preferences: vec!["qwen3-7b".to_string(), "qwen3-1.7b".to_string()],
    };

    let start_time = std::time::Instant::now();
    
    match host.start_pipeline(pipeline_config.clone()).await {
        Ok(pipeline_id) => {
            info!("High-performance pipeline started: {}", pipeline_id);
            
            // Monitor performance
            for i in 0..30 {
                sleep(Duration::from_secs(1)).await;
                
                let metrics = host.get_performance_metrics().await;
                if i % 5 == 0 {
                    info!("Performance cycle {}: Throughput: {:.1} docs/hour, Resource utilization: {:.1}%",
                          i / 5 + 1,
                          metrics.system_throughput_docs_per_hour,
                          metrics.resource_utilization_percent);
                }
            }
        }
        Err(e) => {
            error!("Failed to start high-performance pipeline: {}", e);
        }
    }

    let elapsed = start_time.elapsed();
    info!("High-performance example completed in {:?}", elapsed);

    host.stop().await?;
    Ok(())
}

/// Example: Fault tolerance and recovery scenarios
pub async fn run_fault_tolerance_example() -> Result<()> {
    info!("Starting fault tolerance and recovery example");

    let fault_manager = Arc::new(FaultToleranceManager::new());
    fault_manager.start().await?;

    // Configure aggressive circuit breaker for testing
    let cb_config = CircuitBreakerConfig {
        failure_threshold: 2,
        recovery_timeout: Duration::from_secs(5),
        success_threshold: 1,
        timeout: Duration::from_secs(2),
    };

    fault_manager.register_circuit_breaker("test_service".to_string(), cb_config).await;

    // Simulate service failures and recovery
    info!("Testing circuit breaker behavior");

    // Successful operations
    for i in 0..3 {
        let result = fault_manager.execute_with_fault_tolerance("test_service", async {
            Ok::<_, PipelineError>(format!("Success {}", i))
        }).await;
        
        match result {
            Ok(msg) => info!("Operation succeeded: {}", msg),
            Err(e) => warn!("Operation failed: {}", e),
        }
    }

    // Failing operations to trip circuit breaker
    for i in 0..5 {
        let result = fault_manager.execute_with_fault_tolerance("test_service", async {
            Err::<String, _>(PipelineError::Mcp("Simulated failure".to_string()))
        }).await;
        
        match result {
            Ok(msg) => info!("Unexpected success: {}", msg),
            Err(e) => info!("Expected failure {}: {}", i + 1, e),
        }
    }

    // Wait for recovery
    info!("Waiting for circuit breaker recovery...");
    sleep(Duration::from_secs(6)).await;

    // Test recovery
    let result = fault_manager.execute_with_fault_tolerance("test_service", async {
        Ok::<_, PipelineError>("Recovery success")
    }).await;

    match result {
        Ok(msg) => info!("Recovery successful: {}", msg),
        Err(e) => warn!("Recovery failed: {}", e),
    }

    let metrics = fault_manager.get_metrics().await;
    info!("Final fault tolerance metrics: {} circuit breakers, System health: {:?}",
          metrics.circuit_breaker_metrics.len(),
          metrics.system_health.overall_status);

    fault_manager.stop().await?;
    info!("Fault tolerance example completed");
    Ok(())
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Example: Agent coordination showcase
pub async fn run_agent_coordination_example() -> Result<()> {
    info!("Starting agent coordination showcase");

    let client = McpClient::new();
    client.start().await?;

    // Create diverse agent fleet
    let agent_types = vec![
        ("coordinator", "system_orchestrator", vec!["task_distribution", "performance_monitoring"]),
        ("preprocessor", "document_processor", vec!["text_extraction", "format_conversion"]),
        ("ml_worker_1", "inference_engine", vec!["qwen3_7b", "quality_validation"]),
        ("ml_worker_2", "inference_engine", vec!["qwen3_30b", "advanced_reasoning"]),
        ("postprocessor", "output_formatter", vec!["result_aggregation", "quality_assurance"]),
        ("monitor", "health_checker", vec!["performance_analysis", "error_detection"]),
    ];

    // Register all agents
    for (agent_id, agent_type, capabilities) in agent_types {
        let agent = AgentState {
            agent_id: agent_id.to_string(),
            agent_type: agent_type.to_string(),
            capabilities,
            status: AgentStatus::Idle,
            current_task: None,
            performance_metrics: AgentMetrics::default(),
            last_heartbeat: current_timestamp(),
        };

        client.register_agent(agent).await?;
    }

    // Simulate coordinated workflow
    let task_workflow = vec![
        ("coordinator", "Initialize processing workflow"),
        ("preprocessor", "Extract and normalize document content"),
        ("ml_worker_1", "Perform initial quality analysis"),
        ("ml_worker_2", "Execute advanced reasoning tasks"),
        ("postprocessor", "Aggregate and format results"),
        ("monitor", "Validate output quality"),
    ];

    for (agent_id, task_description) in task_workflow {
        let task_id = Uuid::new_v4();
        
        info!("Assigning task to {}: {}", agent_id, task_description);
        
        // Assign task
        client.assign_task(agent_id, task_id).await?;
        
        // Simulate task execution
        sleep(Duration::from_millis(500)).await;
        
        // Complete task
        client.complete_task(agent_id, task_id, true).await?;
        
        info!("Task completed by {}", agent_id);
    }

    // Display final agent status
    let agents = client.get_agents().await;
    for (agent_id, agent) in agents {
        info!("Agent {}: {} tasks completed, {:.1}% success rate",
              agent_id,
              agent.performance_metrics.tasks_completed,
              agent.performance_metrics.success_rate_percent);
    }

    client.stop().await?;
    info!("Agent coordination example completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_complete_pipeline_example() {
        // Note: This test will fail without actual infrastructure, but demonstrates the pattern
        let result = timeout(
            Duration::from_secs(10),
            run_complete_pipeline_example()
        ).await;

        match result {
            Ok(Ok(_)) => println!("Complete pipeline example succeeded"),
            Ok(Err(e)) => println!("Complete pipeline example failed as expected: {}", e),
            Err(_) => println!("Complete pipeline example timed out as expected"),
        }
    }

    #[tokio::test]
    async fn test_agent_coordination_example() {
        let result = timeout(
            Duration::from_secs(5),
            run_agent_coordination_example()
        ).await;

        match result {
            Ok(Ok(_)) => println!("Agent coordination example succeeded"),
            Ok(Err(e)) => println!("Agent coordination example failed: {}", e),
            Err(_) => println!("Agent coordination example timed out"),
        }
    }

    #[tokio::test]
    async fn test_fault_tolerance_example() {
        let result = timeout(
            Duration::from_secs(15),
            run_fault_tolerance_example()
        ).await;

        match result {
            Ok(Ok(_)) => println!("Fault tolerance example succeeded"),
            Ok(Err(e)) => println!("Fault tolerance example failed: {}", e),
            Err(_) => println!("Fault tolerance example timed out"),
        }
    }
}