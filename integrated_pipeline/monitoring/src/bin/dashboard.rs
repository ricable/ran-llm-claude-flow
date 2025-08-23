//! Standalone dashboard server binary
//! 
//! Web-based monitoring dashboard that can run independently
//! or be integrated into larger systems.

use anyhow::Result;
use clap::Parser;
use monitoring::{
    bottleneck_analyzer::BottleneckAnalyzer,
    config::MonitoringConfig,
    dashboard::DashboardServer,
    metrics::MetricsCollector,
    optimizer::AdaptiveOptimizer,
    alerts::AlertSystem,
};
use std::{path::PathBuf, sync::Arc};
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "monitoring-dashboard")]
#[command(about = "Real-time performance monitoring dashboard")]
struct Args {
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Override dashboard port
    #[arg(short, long)]
    port: Option<u16>,

    /// Override bind address
    #[arg(short, long)]
    bind: Option<String>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Disable WebSocket real-time updates
    #[arg(long)]
    no_websocket: bool,

    /// Enable demo mode with simulated data
    #[arg(long)]
    demo: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("monitoring={},monitoring-dashboard={}", log_level, log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting monitoring dashboard server");

    // Load configuration
    let mut config = if let Some(config_path) = args.config {
        MonitoringConfig::load_from_file(config_path)?
    } else {
        MonitoringConfig::load()
    };

    // Apply command line overrides
    if let Some(port) = args.port {
        config.dashboard.port = port;
    }
    if let Some(bind_addr) = args.bind {
        config.dashboard.bind_address = bind_addr;
    }
    if args.no_websocket {
        config.dashboard.websocket_enabled = false;
    }

    // Validate configuration
    config.validate()?;
    info!("Configuration loaded and validated");

    // Initialize monitoring components
    let metrics_collector = Arc::new(MetricsCollector::new(&config).await?);
    let bottleneck_analyzer = Arc::new(BottleneckAnalyzer::new(&config)?);
    let adaptive_optimizer = Arc::new(AdaptiveOptimizer::new(&config)?);
    let mut alert_system = AlertSystem::new(config.alerts.clone())?;

    // Start monitoring components
    let mut collector = Arc::try_unwrap(metrics_collector.clone()).unwrap_or_else(|arc| {
        panic!("Multiple references to metrics collector");
    });
    collector.start().await?;
    let metrics_collector = Arc::new(collector);

    let mut analyzer = Arc::try_unwrap(bottleneck_analyzer.clone()).unwrap_or_else(|arc| {
        panic!("Multiple references to bottleneck analyzer");
    });
    analyzer.initialize().await?;
    let bottleneck_analyzer = Arc::new(analyzer);

    let mut optimizer = Arc::try_unwrap(adaptive_optimizer.clone()).unwrap_or_else(|arc| {
        panic!("Multiple references to adaptive optimizer");
    });
    optimizer.enable().await?;
    let adaptive_optimizer = Arc::new(optimizer);

    alert_system.start().await?;

    info!("All monitoring components initialized");

    // Start demo data generator if requested
    if args.demo {
        let demo_generator = start_demo_data_generator(
            metrics_collector.clone(),
            bottleneck_analyzer.clone(),
        );
        tokio::spawn(demo_generator);
        info!("Demo mode enabled - generating simulated data");
    }

    // Initialize dashboard server
    let dashboard = DashboardServer::new(
        config.clone(),
        metrics_collector.clone(),
        bottleneck_analyzer.clone(),
        adaptive_optimizer.clone(),
    );

    info!(
        "Starting dashboard on {}:{}",
        config.dashboard.bind_address,
        config.dashboard.port
    );

    // Start dashboard in background
    let dashboard_task = tokio::spawn(async move {
        if let Err(e) = dashboard.start().await {
            tracing::error!("Dashboard server error: {}", e);
        }
    });

    // Wait for shutdown signal
    info!("Dashboard server running - Press Ctrl+C to stop");
    match signal::ctrl_c().await {
        Ok(()) => {
            info!("Shutdown signal received");
        },
        Err(err) => {
            warn!("Unable to listen for shutdown signal: {}", err);
        }
    }

    // Graceful shutdown
    info!("Shutting down dashboard server");
    dashboard_task.abort();

    // Shutdown monitoring components
    if let Ok(mut collector) = Arc::try_unwrap(metrics_collector) {
        let _ = collector.shutdown().await;
    }
    if let Ok(mut analyzer) = Arc::try_unwrap(bottleneck_analyzer) {
        let _ = analyzer.shutdown().await;
    }
    if let Ok(mut optimizer) = Arc::try_unwrap(adaptive_optimizer) {
        let _ = optimizer.shutdown().await;
    }
    let _ = alert_system.shutdown().await;

    info!("Dashboard server stopped");
    Ok(())
}

async fn start_demo_data_generator(
    metrics_collector: Arc<MetricsCollector>,
    _bottleneck_analyzer: Arc<BottleneckAnalyzer>,
) {
    use rand::Rng;
    use tokio::time::{interval, Duration};
    
    let mut rng = rand::thread_rng();
    let mut interval = interval(Duration::from_secs(2));
    
    loop {
        interval.tick().await;
        
        // Simulate document processing
        let docs_to_process = rng.gen_range(1..=5);
        for _ in 0..docs_to_process {
            metrics_collector.increment_documents_processed();
        }
        
        // Simulate IPC messages
        let ipc_messages = rng.gen_range(5..=20);
        for _ in 0..ipc_messages {
            metrics_collector.increment_ipc_messages_sent();
            metrics_collector.increment_ipc_messages_received();
        }
        
        // Simulate varying IPC latency
        let latency = rng.gen_range(1.0..15.0);
        metrics_collector.record_ipc_latency(latency);
        
        // Occasionally simulate errors
        if rng.gen_bool(0.05) { // 5% chance of error
            // Note: We don't have an error increment method, so we simulate this
            tracing::debug!("Simulated processing error");
        }
    }
}