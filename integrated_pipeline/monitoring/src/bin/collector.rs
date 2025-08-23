//! Standalone metrics collector binary
//! 
//! High-performance metrics collection daemon that can run independently
//! or be integrated into larger systems.

use anyhow::Result;
use clap::Parser;
use monitoring::{
    config::MonitoringConfig,
    metrics::MetricsCollector,
    MonitoringSystem,
};
use std::path::PathBuf;
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "monitoring-collector")]
#[command(about = "High-performance metrics collection daemon")]
struct Args {
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Override collection interval in milliseconds
    #[arg(long)]
    interval: Option<u64>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Output metrics to file instead of memory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Enable Prometheus metrics endpoint
    #[arg(long)]
    prometheus: bool,

    /// Prometheus endpoint port
    #[arg(long, default_value = "9090")]
    prometheus_port: u16,

    /// Run for specific duration in seconds (for testing)
    #[arg(long)]
    duration: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("monitoring={},monitoring-collector={}", log_level, log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting monitoring collector daemon");

    // Load configuration
    let mut config = if let Some(config_path) = args.config {
        MonitoringConfig::load_from_file(config_path)?
    } else {
        MonitoringConfig::load()
    };

    // Apply command line overrides
    if let Some(interval) = args.interval {
        config.collection_interval_ms = interval;
    }

    // Validate configuration
    config.validate()?;
    info!("Configuration loaded and validated");

    // Initialize monitoring system
    let mut monitoring_system = MonitoringSystem::with_config(config.clone()).await?;
    monitoring_system.start().await?;

    info!(
        "Metrics collection started - interval: {}ms, max_history: {}",
        config.collection_interval_ms,
        config.max_history_size
    );

    // Setup Prometheus endpoint if requested
    if args.prometheus {
        let prometheus_task = setup_prometheus_endpoint(args.prometheus_port);
        tokio::spawn(prometheus_task);
        info!("Prometheus metrics endpoint started on port {}", args.prometheus_port);
    }

    // Run for specified duration or until signal
    match args.duration {
        Some(duration) => {
            info!("Running for {} seconds", duration);
            tokio::time::sleep(tokio::time::Duration::from_secs(duration)).await;
            info!("Duration completed, shutting down");
        },
        None => {
            info!("Press Ctrl+C to stop the collector");
            
            // Wait for shutdown signal
            match signal::ctrl_c().await {
                Ok(()) => {
                    info!("Shutdown signal received");
                },
                Err(err) => {
                    warn!("Unable to listen for shutdown signal: {}", err);
                }
            }
        }
    }

    // Graceful shutdown
    info!("Shutting down monitoring system");
    monitoring_system.shutdown().await?;
    
    info!("Monitoring collector stopped");
    Ok(())
}

async fn setup_prometheus_endpoint(port: u16) -> Result<()> {
    use warp::Filter;
    
    let metrics_route = warp::path("metrics")
        .and(warp::get())
        .map(|| {
            // TODO: Implement Prometheus metrics formatting
            let metrics = format!(
                "# HELP monitoring_uptime_seconds Total uptime in seconds\n\
                 # TYPE monitoring_uptime_seconds counter\n\
                 monitoring_uptime_seconds 42\n\
                 # HELP monitoring_memory_bytes Current memory usage in bytes\n\
                 # TYPE monitoring_memory_bytes gauge\n\
                 monitoring_memory_bytes 1048576\n"
            );
            warp::reply::with_header(metrics, "content-type", "text/plain")
        });

    let health_route = warp::path("health")
        .and(warp::get())
        .map(|| {
            warp::reply::json(&serde_json::json!({
                "status": "healthy",
                "service": "monitoring-collector",
                "timestamp": chrono::Utc::now()
            }))
        });

    let routes = metrics_route.or(health_route);

    warp::serve(routes)
        .run(([0, 0, 0, 0], port))
        .await;

    Ok(())
}