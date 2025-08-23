//! Utility functions and helpers for the monitoring system

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};
use systemstat::{Duration as StatDuration, Platform, System};

use async_trait::async_trait;
/// Performance measurement utilities
pub struct PerformanceMeasurer {
    start_time: std::time::Instant,
    checkpoints: HashMap<String, std::time::Instant>,
}

impl PerformanceMeasurer {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            checkpoints: HashMap::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.insert(name.to_string(), std::time::Instant::now());
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    pub fn elapsed_since_checkpoint(&self, name: &str) -> Option<std::time::Duration> {
        self.checkpoints.get(name).map(|t| t.elapsed())
    }

    pub fn elapsed_between_checkpoints(&self, start: &str, end: &str) -> Option<std::time::Duration> {
        if let (Some(start_time), Some(end_time)) = (self.checkpoints.get(start), self.checkpoints.get(end)) {
            Some(end_time.duration_since(*start_time))
        } else {
            None
        }
    }
}

/// System information gathering utilities
pub struct SystemInfo {
    pub hostname: String,
    pub platform: String,
    pub cpu_count: usize,
    pub total_memory: u64,
    pub architecture: String,
}

impl SystemInfo {
    pub fn collect() -> Result<Self> {
        let system = System::new();
        
        let hostname = hostname::get()?.to_string_lossy().to_string();
        let platform = std::env::consts::OS.to_string();
        let cpu_count = num_cpus::get();
        let architecture = std::env::consts::ARCH.to_string();
        
        let total_memory = system.memory()
            .map(|mem| mem.total.as_u64())
            .unwrap_or(0);

        Ok(Self {
            hostname,
            platform,
            cpu_count,
            total_memory,
            architecture,
        })
    }
}

/// Time-based utilities for monitoring
pub struct TimeUtils;

impl TimeUtils {
    pub fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub fn now_micros() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    pub fn format_duration(duration: std::time::Duration) -> String {
        let secs = duration.as_secs();
        let millis = duration.subsec_millis();
        let micros = duration.subsec_micros() % 1000;

        if secs > 0 {
            format!("{}.{:03}s", secs, millis)
        } else if millis > 0 {
            format!("{}.{:03}ms", millis, micros)
        } else {
            format!("{}μs", duration.as_micros())
        }
    }

    pub fn humanize_timestamp(timestamp: DateTime<Utc>) -> String {
        let now = Utc::now();
        let diff = now - timestamp;

        if diff < Duration::seconds(60) {
            format!("{}s ago", diff.num_seconds())
        } else if diff < Duration::hours(1) {
            format!("{}m ago", diff.num_minutes())
        } else if diff < Duration::days(1) {
            format!("{}h ago", diff.num_hours())
        } else {
            format!("{}d ago", diff.num_days())
        }
    }
}

/// Statistical utilities for performance analysis
pub struct StatUtils;

impl StatUtils {
    pub fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    pub fn calculate_mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    pub fn calculate_median(values: &[f64]) -> f64 {
        Self::calculate_percentile(values, 50.0)
    }

    pub fn calculate_std_deviation(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = Self::calculate_mean(values);
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || window > values.len() {
            return values.to_vec();
        }

        let mut result = Vec::with_capacity(values.len() - window + 1);
        
        for i in window - 1..values.len() {
            let sum: f64 = values[i - window + 1..=i].iter().sum();
            result.push(sum / window as f64);
        }

        result
    }

    pub fn calculate_trend(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values.iter().enumerate()
            .map(|(i, v)| i as f64 * v)
            .sum::<f64>();
        let x2_sum = (0..values.len())
            .map(|i| (i as f64).powi(2))
            .sum::<f64>();

        // Linear regression slope
        (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2))
    }
}

/// Memory and resource utilities
pub struct ResourceUtils;

impl ResourceUtils {
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.1}{}", size, UNITS[unit_index])
    }

    pub fn format_percentage(value: f64) -> String {
        format!("{:.1}%", value)
    }

    pub fn format_rate(value: f64, unit: &str) -> String {
        if value >= 1000.0 {
            format!("{:.1}K {}/s", value / 1000.0, unit)
        } else {
            format!("{:.0} {}/s", value, unit)
        }
    }

    pub fn get_process_memory_usage() -> Result<u64> {
        let system = System::new();
        if let Ok(memory) = system.memory() {
            // This is a rough estimate - in production, you'd want to use process-specific memory tracking
            Ok(memory.total.as_u64() - memory.free.as_u64())
        } else {
            Ok(0)
        }
    }
}

/// Health check utilities
pub struct HealthChecker {
    checks: Vec<Box<dyn HealthCheck>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
        }
    }

    pub fn add_check<T: HealthCheck + 'static>(&mut self, check: T) {
        self.checks.push(Box::new(check));
    }

    pub async fn run_all_checks(&self) -> HealthReport {
        let mut results = Vec::new();
        let mut overall_healthy = true;

        for check in &self.checks {
            let result = check.check().await;
            if !result.healthy {
                overall_healthy = false;
            }
            results.push(result);
        }

        HealthReport {
            overall_healthy,
            checks: results,
            checked_at: Utc::now(),
        }
    }
}

#[async_trait]
pub trait HealthCheck: Send + Sync {
    async fn check(&self) -> HealthCheckResult;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub name: String,
    pub healthy: bool,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub overall_healthy: bool,
    pub checks: Vec<HealthCheckResult>,
    pub checked_at: DateTime<Utc>,
}

/// Basic system health checks
pub struct CpuHealthCheck;

#[async_trait]
impl HealthCheck for CpuHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        let system = System::new();
        
        match system.load_average() {
            Ok(load) => {
                let cpu_count = num_cpus::get() as f32;
                let load_per_core = load.one / cpu_count;
                let healthy = load_per_core < 2.0; // Less than 2.0 load per core is considered healthy
                
                HealthCheckResult {
                    name: "CPU Load".to_string(),
                    healthy,
                    message: if healthy {
                        format!("CPU load is healthy: {:.2} (per core: {:.2})", load.one, load_per_core)
                    } else {
                        format!("High CPU load: {:.2} (per core: {:.2})", load.one, load_per_core)
                    },
                    details: Some(serde_json::json!({
                        "load_1min": load.one,
                        "load_5min": load.five,
                        "load_15min": load.fifteen,
                        "cpu_cores": cpu_count,
                        "load_per_core": load_per_core
                    })),
                    checked_at: Utc::now(),
                }
            },
            Err(e) => HealthCheckResult {
                name: "CPU Load".to_string(),
                healthy: false,
                message: format!("Failed to check CPU load: {}", e),
                details: None,
                checked_at: Utc::now(),
            }
        }
    }
}

pub struct MemoryHealthCheck;

#[async_trait]
impl HealthCheck for MemoryHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        let system = System::new();
        
        match system.memory() {
            Ok(memory) => {
                let usage_percent = ((memory.total.as_u64() - memory.free.as_u64()) as f64 / memory.total.as_u64() as f64) * 100.0;
                let healthy = usage_percent < 90.0;
                
                HealthCheckResult {
                    name: "Memory Usage".to_string(),
                    healthy,
                    message: if healthy {
                        format!("Memory usage is healthy: {:.1}%", usage_percent)
                    } else {
                        format!("High memory usage: {:.1}%", usage_percent)
                    },
                    details: Some(serde_json::json!({
                        "usage_percent": usage_percent,
                        "total_gb": memory.total.as_u64() as f64 / (1024.0 * 1024.0 * 1024.0),
                        "used_gb": (memory.total.as_u64() - memory.free.as_u64()) as f64 / (1024.0 * 1024.0 * 1024.0),
                        "free_gb": memory.free.as_u64() as f64 / (1024.0 * 1024.0 * 1024.0)
                    })),
                    checked_at: Utc::now(),
                }
            },
            Err(e) => HealthCheckResult {
                name: "Memory Usage".to_string(),
                healthy: false,
                message: format!("Failed to check memory usage: {}", e),
                details: None,
                checked_at: Utc::now(),
            }
        }
    }
}

/// Configuration validation utilities
pub struct ConfigValidator;

impl ConfigValidator {
    pub fn validate_port(port: u16) -> Result<()> {
        if port < 1024 {
            return Err(anyhow::anyhow!("Port {} is reserved (< 1024)", port));
        }
        if port > 65535 {
            return Err(anyhow::anyhow!("Port {} is invalid (> 65535)", port));
        }
        Ok(())
    }

    pub fn validate_percentage(value: f64, name: &str) -> Result<()> {
        if value < 0.0 || value > 100.0 {
            return Err(anyhow::anyhow!("{} must be between 0 and 100, got {}", name, value));
        }
        Ok(())
    }

    pub fn validate_positive_number(value: f64, name: &str) -> Result<()> {
        if value <= 0.0 {
            return Err(anyhow::anyhow!("{} must be positive, got {}", name, value));
        }
        Ok(())
    }

    pub fn validate_interval(interval_ms: u64, name: &str) -> Result<()> {
        if interval_ms == 0 {
            return Err(anyhow::anyhow!("{} interval cannot be zero", name));
        }
        if interval_ms < 10 {
            return Err(anyhow::anyhow!("{} interval {} ms is too low (minimum 10ms)", name, interval_ms));
        }
        if interval_ms > 300_000 { // 5 minutes
            return Err(anyhow::anyhow!("{} interval {} ms is too high (maximum 5 minutes)", name, interval_ms));
        }
        Ok(())
    }
}