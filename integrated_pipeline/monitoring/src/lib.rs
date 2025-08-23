//! High-Performance Pipeline Monitoring System
//! 
//! This module provides comprehensive real-time monitoring capabilities for the
//! hybrid Rust-Python document processing pipeline with <1% performance overhead.

pub mod metrics;
pub mod bottleneck_analyzer;
pub mod optimizer; 
pub mod dashboard;
pub mod alerts;
pub mod config;
pub mod utils;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Core monitoring system coordinator
pub struct MonitoringSystem {
    metrics_collector: metrics::MetricsCollector,
    bottleneck_analyzer: bottleneck_analyzer::BottleneckAnalyzer,
    optimizer: optimizer::AdaptiveOptimizer,
    config: config::MonitoringConfig,
    session_id: Uuid,
    start_time: DateTime<Utc>,
}

impl MonitoringSystem {
    /// Initialize the monitoring system with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(config::MonitoringConfig::default()).await
    }

    /// Initialize with custom configuration
    pub async fn with_config(config: config::MonitoringConfig) -> Result<Self> {
        let session_id = Uuid::new_v4();
        tracing::info!("Initializing monitoring system with session {}", session_id);

        Ok(Self {
            metrics_collector: metrics::MetricsCollector::new(&config).await?,
            bottleneck_analyzer: bottleneck_analyzer::BottleneckAnalyzer::new(&config)?,
            optimizer: optimizer::AdaptiveOptimizer::new(&config)?,
            config,
            session_id,
            start_time: Utc::now(),
        })
    }

    /// Start the monitoring system with all components
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting monitoring system session {}", self.session_id);

        // Start metrics collection
        self.metrics_collector.start().await?;
        
        // Initialize bottleneck detection
        self.bottleneck_analyzer.initialize().await?;
        
        // Enable adaptive optimization
        self.optimizer.enable().await?;

        tracing::info!("Monitoring system fully operational");
        Ok(())
    }

    /// Get current system performance snapshot
    pub async fn get_performance_snapshot(&self) -> Result<PerformanceSnapshot> {
        let metrics = self.metrics_collector.get_current_metrics().await?;
        let bottlenecks = self.bottleneck_analyzer.detect_current_bottlenecks().await?;
        let optimizations = self.optimizer.get_active_optimizations().await?;

        Ok(PerformanceSnapshot {
            session_id: self.session_id,
            timestamp: Utc::now(),
            metrics,
            bottlenecks,
            optimizations,
        })
    }

    /// Shutdown monitoring system gracefully  
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down monitoring system");
        
        self.optimizer.shutdown().await?;
        self.bottleneck_analyzer.shutdown().await?;
        self.metrics_collector.shutdown().await?;
        
        Ok(())
    }
}

/// Comprehensive performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub session_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub metrics: metrics::SystemMetrics,
    pub bottlenecks: Vec<bottleneck_analyzer::Bottleneck>,
    pub optimizations: Vec<optimizer::Optimization>,
}

/// Key performance indicators tracked by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceKPIs {
    pub document_processing_rate: f64, // docs/hour
    pub rust_memory_utilization: f64,  // GB
    pub python_memory_utilization: f64, // GB
    pub shared_memory_utilization: f64, // GB  
    pub ipc_latency_p99: f64,          // ms
    pub model_inference_time_avg: f64,  // ms
    pub total_throughput: f64,         // docs/second
    pub error_rate: f64,               // percentage
    pub cpu_utilization_avg: f64,      // percentage
    pub gpu_utilization_avg: f64,      // percentage
}

impl PerformanceKPIs {
    /// Check if KPIs meet performance targets
    pub fn meets_targets(&self) -> bool {
        self.document_processing_rate >= 20.0 &&
        self.rust_memory_utilization <= 60.0 &&
        self.python_memory_utilization <= 45.0 &&
        self.shared_memory_utilization <= 15.0 &&
        self.ipc_latency_p99 <= 10.0 &&
        self.error_rate <= 1.0
    }
}