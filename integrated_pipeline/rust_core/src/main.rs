use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn, error};

mod document_processor;
mod ipc_manager;
mod quality_validator;
mod types;
mod config;

use crate::document_processor::DocumentProcessor;
use crate::config::ProcessingConfig;

#[derive(Parser)]
#[command(name = "rust_core")]
#[command(about = "High-performance Rust core for hybrid ML pipeline")]
struct Cli {
    /// Input directory containing documents to process
    #[arg(short, long)]
    input: PathBuf,
    
    /// Output directory for processed results
    #[arg(short, long)]
    output: PathBuf,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Maximum concurrent documents to process
    #[arg(long, default_value = "16")]
    max_concurrent: usize,
    
    /// Memory limit in GB (default: 60GB for M3 Max)
    #[arg(long, default_value = "60")]
    memory_limit_gb: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize tracing
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("rust_core={}", log_level))
        .init();
    
    info!("Starting Rust Core Pipeline");
    info!("M3 Max Optimization: ENABLED");
    info!("Available CPU cores: {}", num_cpus::get());
    info!("Memory limit: {}GB", cli.memory_limit_gb);
    
    // Load configuration
    let config = match cli.config {
        Some(config_path) => ProcessingConfig::from_file(&config_path)?,
        None => ProcessingConfig::default_m3_max(cli.max_concurrent, cli.memory_limit_gb),
    };
    
    info!("Configuration loaded: {:?}", config);
    
    // Initialize document processor
    let processor = DocumentProcessor::new(config).await?;
    
    // Start processing pipeline
    match processor.process_directory(&cli.input, &cli.output).await {
        Ok(stats) => {
            info!("Processing completed successfully");
            info!("Documents processed: {}", stats.documents_processed);
            info!("Average quality score: {:.2}", stats.average_quality);
            info!("Total processing time: {:?}", stats.total_time);
            info!("Throughput: {:.2} docs/hour", stats.throughput_per_hour());
        }
        Err(e) => {
            error!("Processing failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}