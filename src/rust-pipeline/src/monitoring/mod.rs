//! Performance Monitoring Module
//!
//! This module provides comprehensive performance monitoring capabilities for the RAN-LLM pipeline,
//! including real-time metrics collection, alerting, regression detection, and web dashboard.

pub mod dashboard;

pub use dashboard::{
    AlertSeverity, AlertSystem, AlertThreshold, DashboardState, M3MaxMetrics, MetricsCollector,
    PerformanceAlert, PerformanceDashboard, RegressionAnalysis, RegressionDetector, SystemMetrics,
    ThroughputMetrics,
};

use std::sync::Arc;
use tokio::sync::RwLock;

/// Global performance monitoring instance
pub static PERFORMANCE_MONITOR: std::sync::LazyLock<Arc<RwLock<Option<PerformanceDashboard>>>> =
    std::sync::LazyLock::new(|| Arc::new(RwLock::new(None)));

/// Initialize the performance monitoring system
pub async fn initialize_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let dashboard = PerformanceDashboard::new();

    // Store the dashboard instance globally
    let mut monitor = PERFORMANCE_MONITOR.write().await;
    *monitor = Some(dashboard);

    println!("📊 Performance monitoring system initialized");
    Ok(())
}

/// Start the performance dashboard server
pub async fn start_dashboard(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let monitor = PERFORMANCE_MONITOR.read().await;

    if let Some(dashboard) = monitor.as_ref() {
        dashboard.start(port).await?;
    } else {
        return Err("Performance monitoring not initialized".into());
    }

    Ok(())
}

/// Get current M3 Max metrics
pub async fn get_current_m3_metrics() -> Option<M3MaxMetrics> {
    let monitor = PERFORMANCE_MONITOR.read().await;

    if let Some(dashboard) = monitor.as_ref() {
        let metrics = dashboard.state.m3_max_metrics.lock().unwrap();
        metrics.last().cloned()
    } else {
        None
    }
}

/// Get current system metrics
pub async fn get_current_system_metrics() -> Option<SystemMetrics> {
    let monitor = PERFORMANCE_MONITOR.read().await;

    if let Some(dashboard) = monitor.as_ref() {
        let metrics = dashboard.state.system_metrics.lock().unwrap();
        metrics.last().cloned()
    } else {
        None
    }
}

/// Get current throughput metrics
pub async fn get_current_throughput_metrics() -> Option<ThroughputMetrics> {
    let monitor = PERFORMANCE_MONITOR.read().await;

    if let Some(dashboard) = monitor.as_ref() {
        let metrics = dashboard.state.throughput_metrics.lock().unwrap();
        metrics.last().cloned()
    } else {
        None
    }
}

/// Get active alerts
pub async fn get_active_alerts() -> Vec<PerformanceAlert> {
    let monitor = PERFORMANCE_MONITOR.read().await;

    if let Some(dashboard) = monitor.as_ref() {
        dashboard.state.alerts.lock().unwrap().clone()
    } else {
        Vec::new()
    }
}

/// Check if system is healthy based on current metrics
pub async fn is_system_healthy() -> bool {
    let m3_metrics = get_current_m3_metrics().await;
    let _system_metrics = get_current_system_metrics().await;
    let throughput_metrics = get_current_throughput_metrics().await;

    // Define health criteria
    let memory_healthy = m3_metrics
        .as_ref()
        .map(|m| (m.unified_memory_used as f64 / m.unified_memory_total as f64) < 0.9)
        .unwrap_or(false);

    let cpu_healthy = m3_metrics
        .as_ref()
        .map(|m| m.cpu_load < 0.8)
        .unwrap_or(false);

    let throughput_healthy = throughput_metrics
        .as_ref()
        .map(|t| t.error_rate < 5.0) // Less than 5% error rate
        .unwrap_or(false);

    memory_healthy && cpu_healthy && throughput_healthy
}

/// Generate a performance summary report
pub async fn generate_performance_report() -> String {
    let m3_metrics = get_current_m3_metrics().await;
    let _system_metrics = get_current_system_metrics().await;
    let throughput_metrics = get_current_throughput_metrics().await;
    let alerts = get_active_alerts().await;
    let health_status = is_system_healthy().await;

    let mut report = String::new();
    report.push_str("🚀 RAN-LLM Performance Report\n");
    report.push_str("=".repeat(50).as_str());
    report.push_str("\n\n");

    // System health
    report.push_str(&format!(
        "System Health: {}\n",
        if health_status {
            "✅ Healthy"
        } else {
            "⚠️ Issues Detected"
        }
    ));
    report.push_str("\n");

    // M3 Max metrics
    if let Some(m3) = m3_metrics {
        let memory_usage_percent =
            (m3.unified_memory_used as f64 / m3.unified_memory_total as f64) * 100.0;
        report.push_str("📊 M3 Max Metrics:\n");
        report.push_str(&format!(
            "  Memory Usage: {:.1}% ({:.1}GB / {:.1}GB)\n",
            memory_usage_percent,
            m3.unified_memory_used as f64 / (1024.0 * 1024.0 * 1024.0),
            m3.unified_memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        report.push_str(&format!("  CPU Load: {:.1}%\n", m3.cpu_load * 100.0));
        report.push_str(&format!(
            "  Neural Engine: {:.1}%\n",
            m3.neural_engine_utilization
        ));
        report.push_str(&format!(
            "  GPU Utilization: {:.1}%\n",
            m3.gpu_core_utilization
        ));
        report.push_str(&format!("  Thermal State: {}\n", m3.thermal_state));
        report.push_str(&format!(
            "  Power Consumption: {:.1}W\n",
            m3.power_consumption
        ));
        report.push_str("\n");
    }

    // Throughput metrics
    if let Some(throughput) = throughput_metrics.as_ref() {
        report.push_str("📈 Pipeline Performance:\n");
        report.push_str(&format!(
            "  Documents Processed: {}\n",
            throughput.documents_processed
        ));
        report.push_str(&format!(
            "  Throughput: {:.1} docs/min\n",
            throughput.documents_per_minute
        ));
        report.push_str(&format!(
            "  Avg Processing Time: {:.1}ms\n",
            throughput.average_processing_time_ms
        ));
        report.push_str(&format!("  Queue Depth: {}\n", throughput.queue_depth));
        report.push_str(&format!(
            "  Active Workers: {}\n",
            throughput.active_workers
        ));
        report.push_str(&format!(
            "  Success Rate: {:.1}%\n",
            throughput.success_rate
        ));
        report.push_str(&format!("  Error Rate: {:.1}%\n", throughput.error_rate));
        report.push_str("\n");
    }

    // Alerts
    if !alerts.is_empty() {
        report.push_str(&format!("🚨 Active Alerts ({}):\n", alerts.len()));
        for alert in alerts.iter().take(5) {
            // Show only first 5 alerts
            let severity_icon = match alert.severity {
                AlertSeverity::Critical => "🔴",
                AlertSeverity::Warning => "🟡",
                AlertSeverity::Info => "🔵",
                AlertSeverity::Emergency => "🟣",
            };
            report.push_str(&format!(
                "  {} {}: {}\n",
                severity_icon, alert.component, alert.message
            ));
        }
        if alerts.len() > 5 {
            report.push_str(&format!("  ... and {} more alerts\n", alerts.len() - 5));
        }
        report.push_str("\n");
    }

    // Performance targets
    let target_throughput = 25.0; // docs/hour
    if let Some(throughput) = throughput_metrics {
        let current_hourly = throughput.documents_per_minute * 60.0;
        let efficiency = (current_hourly / target_throughput * 100.0).min(100.0);

        report.push_str("🎯 Performance Targets:\n");
        report.push_str(&format!("  Target: {:.0} docs/hour\n", target_throughput));
        report.push_str(&format!("  Current: {:.1} docs/hour\n", current_hourly));
        report.push_str(&format!("  Efficiency: {:.1}%\n", efficiency));

        if efficiency >= 80.0 {
            report.push_str("  Status: ✅ Meeting targets\n");
        } else if efficiency >= 60.0 {
            report.push_str("  Status: ⚠️ Below optimal\n");
        } else {
            report.push_str("  Status: 🔴 Needs attention\n");
        }
        report.push_str("\n");
    }

    report.push_str(&format!(
        "Report generated at: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));

    report
}

/// Log current performance metrics
pub async fn log_performance_metrics() {
    if let Some(report) = Some(generate_performance_report().await) {
        println!("{}", report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_initialization() {
        let result = initialize_monitoring().await;
        assert!(result.is_ok());

        // Check that the monitor is initialized
        let monitor = PERFORMANCE_MONITOR.read().await;
        assert!(monitor.is_some());
    }

    #[tokio::test]
    async fn test_health_check_with_no_metrics() {
        // Before initialization, should return false
        let health = is_system_healthy().await;
        assert!(!health);
    }

    #[tokio::test]
    async fn test_performance_report_generation() {
        let report = generate_performance_report().await;
        assert!(report.contains("RAN-LLM Performance Report"));
        assert!(report.contains("System Health"));
    }

    #[tokio::test]
    async fn test_metrics_retrieval_before_initialization() {
        let m3_metrics = get_current_m3_metrics().await;
        let system_metrics = get_current_system_metrics().await;
        let throughput_metrics = get_current_throughput_metrics().await;
        let alerts = get_active_alerts().await;

        assert!(m3_metrics.is_none());
        assert!(system_metrics.is_none());
        assert!(throughput_metrics.is_none());
        assert!(alerts.is_empty());
    }
}
