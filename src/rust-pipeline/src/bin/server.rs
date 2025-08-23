/*!
# Rust-Python Pipeline Server

Main entry point for the hybrid pipeline server optimized for M3 Max hardware.
*/

use std::sync::Arc;
use tokio::signal;
use tracing::{info, error};
use rust_python_pipeline::{PipelineConfig, initialize_pipeline, shutdown_pipeline};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for M3 Max optimized logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🚀 Starting Rust-Python Pipeline Server for M3 Max");
    info!("Hardware: MacBook Pro M3 Max with 128GB unified memory");

    // Load configuration
    let config = load_config().await?;
    log_config_summary(&config);

    // Initialize pipeline
    match initialize_pipeline(config).await {
        Ok(()) => info!("✅ Pipeline initialized successfully"),
        Err(e) => {
            error!("❌ Failed to initialize pipeline: {}", e);
            return Err(e.into());
        }
    }

    // Setup graceful shutdown handler
    let shutdown_handler = setup_shutdown_handler();

    info!("🎯 Pipeline server is ready for processing");
    info!("Performance targets:");
    info!("  • Throughput: 20-30 documents/hour (4-5x improvement)");
    info!("  • Memory efficiency: 95% of 128GB unified memory");
    info!("  • Model switching: <5 second latency");
    info!("  • Quality consistency: >0.742 score target");

    // Wait for shutdown signal
    shutdown_handler.await;

    info!("📡 Received shutdown signal, gracefully stopping...");

    // Shutdown pipeline
    match shutdown_pipeline().await {
        Ok(()) => info!("✅ Pipeline shutdown completed successfully"),
        Err(e) => error!("❌ Error during shutdown: {}", e),
    }

    info!("👋 Rust-Python Pipeline Server stopped");
    Ok(())
}

async fn load_config() -> Result<PipelineConfig, Box<dyn std::error::Error>> {
    // Try loading from config file first
    if let Ok(config_str) = std::fs::read_to_string("pipeline_config.toml") {
        info!("📄 Loading configuration from pipeline_config.toml");
        return Ok(toml::from_str(&config_str)?);
    }

    // Check for environment-based configuration
    if let Ok(config_str) = std::env::var("PIPELINE_CONFIG") {
        info!("🌍 Loading configuration from environment variable");
        return Ok(serde_json::from_str(&config_str)?);
    }

    // Use default configuration optimized for M3 Max
    info!("⚙️  Using default M3 Max optimized configuration");
    Ok(PipelineConfig::default())
}

fn log_config_summary(config: &PipelineConfig) {
    info!("📊 Configuration Summary:");
    info!("  M3 Max Memory Allocation:");
    info!("    • Total: {}GB unified memory", config.m3_max.total_memory_gb);
    info!("    • Processing: {}GB", config.m3_max.memory_pools.processing);
    info!("    • IPC: {}GB", config.m3_max.memory_pools.ipc);
    info!("    • Cache: {}GB", config.m3_max.memory_pools.cache);
    info!("    • System: {}GB", config.m3_max.memory_pools.system);
    info!("  CPU Cores:");
    info!("    • Performance cores: {}", config.m3_max.cpu_cores.performance_cores);
    info!("    • Efficiency cores: {}", config.m3_max.cpu_cores.efficiency_cores);
    info!("    • GPU cores: {}", config.m3_max.gpu_cores);
    info!("    • Neural Engine: {}", if config.m3_max.neural_engine { "enabled" } else { "disabled" });
    info!("  Python Workers: {}", config.python.worker_count);
    info!("  MCP Server Port: {}", config.mcp.server_port);
    info!("  Quality Target: {}", config.quality.consistency_target);
}

async fn setup_shutdown_handler() {
    let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
        .expect("Failed to create SIGINT handler");
    
    let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
        .expect("Failed to create SIGTERM handler");

    tokio::select! {
        _ = sigint.recv() => {
            info!("📡 Received SIGINT (Ctrl+C)");
        }
        _ = sigterm.recv() => {
            info!("📡 Received SIGTERM");
        }
    }
}

// Performance monitoring task
async fn start_performance_monitor() {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
    
    loop {
        interval.tick().await;
        
        // Collect M3 Max specific metrics
        let metrics = collect_m3_max_metrics().await;
        
        info!("📈 Performance Metrics:");
        info!("  • Memory usage: {:.1}% of 128GB", metrics.memory_usage_percent);
        info!("  • CPU utilization: {:.1}% (P-cores: {:.1}%, E-cores: {:.1}%)", 
              metrics.cpu_total, metrics.cpu_performance, metrics.cpu_efficiency);
        info!("  • GPU utilization: {:.1}% of 40 cores", metrics.gpu_utilization);
        info!("  • Processing throughput: {:.1} docs/hour", metrics.docs_per_hour);
        info!("  • Quality score: {:.3}", metrics.quality_score);
        
        // Alert on performance issues
        if metrics.memory_usage_percent > 95.0 {
            error!("⚠️  Memory usage critical: {:.1}%", metrics.memory_usage_percent);
        }
        
        if metrics.quality_score < 0.742 {
            error!("⚠️  Quality score below target: {:.3}", metrics.quality_score);
        }
        
        if metrics.docs_per_hour < 20.0 {
            error!("⚠️  Throughput below target: {:.1} docs/hour", metrics.docs_per_hour);
        }
    }
}

#[derive(Debug)]
struct M3MaxMetrics {
    memory_usage_percent: f64,
    cpu_total: f64,
    cpu_performance: f64,
    cpu_efficiency: f64,
    gpu_utilization: f64,
    docs_per_hour: f64,
    quality_score: f64,
}

async fn collect_m3_max_metrics() -> M3MaxMetrics {
    // This would integrate with actual system monitoring
    // For now, return mock metrics
    M3MaxMetrics {
        memory_usage_percent: 87.5,
        cpu_total: 65.2,
        cpu_performance: 78.1,
        cpu_efficiency: 45.3,
        gpu_utilization: 42.8,
        docs_per_hour: 24.7,
        quality_score: 0.751,
    }
}