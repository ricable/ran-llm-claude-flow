/*!
# Pipeline Coordinator Binary

Main coordinator binary for the RAN document processing pipeline.
Orchestrates the entire pipeline system with performance monitoring.
*/

use ran_document_pipeline::{
    Result, PipelineError, PipelineConfig, 
    initialize, start_performance_dashboard
};
use std::env;
use tokio::signal;
use tracing::{info, error, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting RAN Document Pipeline Coordinator");
    
    // Load configuration
    let config = load_config().await?;
    info!("📋 Configuration loaded successfully");
    
    // Initialize the pipeline system
    initialize().await?;
    info!("✅ Pipeline system initialized");
    
    // Start performance dashboard
    let dashboard_port = env::var("DASHBOARD_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);
    
    tokio::spawn(async move {
        if let Err(e) = start_performance_dashboard(dashboard_port).await {
            error!("Failed to start dashboard: {}", e);
        }
    });
    
    info!("🌐 Performance dashboard started on port {}", dashboard_port);
    
    // Wait for shutdown signal
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("🛑 Received shutdown signal, stopping pipeline coordinator");
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }
    
    info!("👋 Pipeline coordinator stopped");
    Ok(())
}

async fn load_config() -> Result<PipelineConfig> {
    // Try to load from environment variable first
    if let Ok(config_path) = env::var("PIPELINE_CONFIG_PATH") {
        info!("Loading configuration from: {}", config_path);
        load_config_from_file(&config_path).await
    } else {
        warn!("No config path specified, using default configuration");
        Ok(PipelineConfig::default())
    }
}

async fn load_config_from_file(path: &str) -> Result<PipelineConfig> {
    let config_str = tokio::fs::read_to_string(path).await
        .map_err(|e| PipelineError::ConfigError(format!("Failed to read config file: {}", e)))?;
    
    // Try YAML first, then TOML
    if path.ends_with(".yaml") || path.ends_with(".yml") {
        serde_yaml::from_str(&config_str)
            .map_err(|e| PipelineError::ConfigError(format!("Failed to parse YAML config: {}", e)))
    } else if path.ends_with(".toml") {
        toml::from_str(&config_str)
            .map_err(|e| PipelineError::ConfigError(format!("Failed to parse TOML config: {}", e)))
    } else {
        // Try JSON as fallback
        serde_json::from_str(&config_str)
            .map_err(|e| PipelineError::ConfigError(format!("Failed to parse JSON config: {}", e)))
    }
}