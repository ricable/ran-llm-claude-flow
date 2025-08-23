//! Intelligent alerting system for performance degradation
//! 
//! Provides real-time alert generation with configurable channels,
//! severity-based routing, and intelligent alert suppression.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{sync::broadcast, time};
use async_trait::async_trait;

use crate::{
    bottleneck_analyzer::{Bottleneck, BottleneckSeverity, BottleneckType},
    config::{AlertConfig, AlertThresholds},
    metrics::SystemMetrics,
};

/// Advanced alerting system with intelligent suppression
pub struct AlertSystem {
    config: AlertConfig,
    running: AtomicBool,
    alert_sender: broadcast::Sender<Alert>,
    
    // Alert management
    active_alerts: Arc<DashMap<String, Alert>>,
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    suppressed_alerts: Arc<DashMap<String, DateTime<Utc>>>,
    
    // Rate limiting
    alert_rate_limiter: Arc<RwLock<AlertRateLimiter>>,
    
    // Alert channels
    channels: Vec<Arc<dyn AlertChannel>>,
    
    // Performance tracking
    alerts_sent: AtomicU64,
    alerts_suppressed: AtomicU64,
}

impl AlertSystem {
    pub fn new(config: AlertConfig) -> Result<Self> {
        let (alert_sender, _) = broadcast::channel(1000);
        
        // Initialize alert channels
        let mut channels: Vec<Arc<dyn AlertChannel>> = Vec::new();
        for channel_config in &config.channels {
            match channel_config.channel_type.as_str() {
                "log" => channels.push(Arc::new(LogAlertChannel::new(channel_config)?)),
                "webhook" => channels.push(Arc::new(WebhookAlertChannel::new(channel_config)?)),
                "email" => channels.push(Arc::new(EmailAlertChannel::new(channel_config)?)),
                _ => tracing::warn!("Unknown alert channel type: {}", channel_config.channel_type),
            }
        }
        
        Ok(Self {
            config,
            running: AtomicBool::new(false),
            alert_sender,
            active_alerts: Arc::new(DashMap::new()),
            alert_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            suppressed_alerts: Arc::new(DashMap::new()),
            alert_rate_limiter: Arc::new(RwLock::new(AlertRateLimiter::new())),
            channels,
            alerts_sent: AtomicU64::new(0),
            alerts_suppressed: AtomicU64::new(0),
        })
    }

    pub async fn start(&mut self) -> Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        if !self.config.enabled {
            tracing::info!("Alert system disabled in configuration");
            return Ok(());
        }

        self.running.store(true, Ordering::Release);
        tracing::info!("Starting alert system");

        // Start alert processing loop
        let processor = self.clone_for_task();
        tokio::spawn(async move {
            processor.alert_processing_loop().await;
        });

        // Start cleanup loop
        let cleanup = self.clone_for_task();
        tokio::spawn(async move {
            cleanup.cleanup_loop().await;
        });

        Ok(())
    }

    /// Process new metrics and generate alerts
    pub async fn process_metrics(&self, metrics: &SystemMetrics) -> Result<()> {
        let mut alerts = Vec::new();

        // Check CPU utilization
        if metrics.cpu_usage.utilization_percent > self.config.severity_thresholds.cpu_utilization {
            alerts.push(self.create_threshold_alert(
                "cpu_utilization",
                "High CPU Utilization",
                &format!("CPU utilization is {:.1}% (threshold: {:.1}%)", 
                        metrics.cpu_usage.utilization_percent,
                        self.config.severity_thresholds.cpu_utilization),
                self.determine_severity(metrics.cpu_usage.utilization_percent as f64, 
                                      self.config.severity_thresholds.cpu_utilization as f64, 95.0),
                serde_json::json!({
                    "cpu_utilization": metrics.cpu_usage.utilization_percent,
                    "load_1min": metrics.cpu_usage.load_1min,
                    "load_5min": metrics.cpu_usage.load_5min,
                    "temperature": metrics.cpu_usage.temperature_celsius
                })
            ));
        }

        // Check memory utilization
        let memory_utilization = (metrics.memory.used_bytes as f64 / metrics.memory.total_bytes as f64) * 100.0;
        if memory_utilization > self.config.severity_thresholds.memory_utilization {
            alerts.push(self.create_threshold_alert(
                "memory_utilization",
                "High Memory Utilization",
                &format!("Memory utilization is {:.1}% (threshold: {:.1}%)", 
                        memory_utilization,
                        self.config.severity_thresholds.memory_utilization),
                self.determine_severity(memory_utilization, 
                                      self.config.severity_thresholds.memory_utilization, 95.0),
                serde_json::json!({
                    "memory_utilization": memory_utilization,
                    "used_gb": metrics.memory.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    "total_gb": metrics.memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    "rust_heap_mb": metrics.memory.rust_heap_mb,
                    "python_heap_mb": metrics.memory.python_heap_mb
                })
            ));
        }

        // Check IPC latency
        if metrics.ipc_latency_p99 > self.config.severity_thresholds.ipc_latency_ms {
            alerts.push(self.create_threshold_alert(
                "ipc_latency",
                "High IPC Latency",
                &format!("IPC P99 latency is {:.1}ms (threshold: {:.1}ms)", 
                        metrics.ipc_latency_p99,
                        self.config.severity_thresholds.ipc_latency_ms),
                self.determine_severity(metrics.ipc_latency_p99, 
                                      self.config.severity_thresholds.ipc_latency_ms, 50.0),
                serde_json::json!({
                    "ipc_latency_p99": metrics.ipc_latency_p99,
                    "ipc_messages_sent": metrics.ipc_messages_sent,
                    "ipc_messages_received": metrics.ipc_messages_received
                })
            ));
        }

        // Check document processing rate
        if metrics.document_processing_rate > 0.0 && 
           metrics.document_processing_rate < self.config.severity_thresholds.document_rate {
            alerts.push(self.create_threshold_alert(
                "document_rate",
                "Low Document Processing Rate",
                &format!("Document processing rate is {:.1} docs/hour (threshold: {:.1} docs/hour)", 
                        metrics.document_processing_rate,
                        self.config.severity_thresholds.document_rate),
                self.determine_severity(self.config.severity_thresholds.document_rate - metrics.document_processing_rate, 
                                      5.0, 15.0),
                serde_json::json!({
                    "document_rate": metrics.document_processing_rate,
                    "documents_processed": metrics.documents_processed,
                    "errors_total": metrics.errors_total
                })
            ));
        }

        // Check error rate
        let error_rate = if metrics.documents_processed > 0 {
            (metrics.errors_total as f64 / metrics.documents_processed as f64) * 100.0
        } else {
            0.0
        };

        if error_rate > self.config.severity_thresholds.error_rate {
            alerts.push(self.create_threshold_alert(
                "error_rate",
                "High Error Rate",
                &format!("Error rate is {:.2}% (threshold: {:.1}%)", 
                        error_rate,
                        self.config.severity_thresholds.error_rate),
                self.determine_severity(error_rate, 
                                      self.config.severity_thresholds.error_rate, 10.0),
                serde_json::json!({
                    "error_rate": error_rate,
                    "errors_total": metrics.errors_total,
                    "documents_processed": metrics.documents_processed
                })
            ));
        }

        // Process all generated alerts
        for alert in alerts {
            self.process_alert(alert).await?;
        }

        Ok(())
    }

    /// Process bottleneck-based alerts
    pub async fn process_bottlenecks(&self, bottlenecks: &[Bottleneck]) -> Result<()> {
        for bottleneck in bottlenecks {
            let alert = Alert {
                id: format!("bottleneck_{}", bottleneck.id),
                alert_type: AlertType::Bottleneck,
                severity: match bottleneck.severity {
                    BottleneckSeverity::Critical => AlertSeverity::Critical,
                    BottleneckSeverity::High => AlertSeverity::High,
                    BottleneckSeverity::Medium => AlertSeverity::Medium,
                    BottleneckSeverity::Low => AlertSeverity::Low,
                },
                title: format!("Performance Bottleneck: {:?}", bottleneck.bottleneck_type),
                message: bottleneck.description.clone(),
                details: serde_json::json!({
                    "bottleneck_type": format!("{:?}", bottleneck.bottleneck_type),
                    "impact_score": bottleneck.impact_score,
                    "suggested_actions": bottleneck.suggested_actions,
                    "metrics_snapshot": bottleneck.metrics_snapshot
                }),
                created_at: Utc::now(),
                resolved_at: None,
                tags: vec![
                    format!("bottleneck_type:{:?}", bottleneck.bottleneck_type),
                    format!("severity:{:?}", bottleneck.severity),
                ],
            };

            self.process_alert(alert).await?;
        }

        Ok(())
    }

    async fn process_alert(&self, alert: Alert) -> Result<()> {
        // Check if alert should be suppressed
        if self.should_suppress_alert(&alert).await {
            self.alerts_suppressed.fetch_add(1, Ordering::Relaxed);
            tracing::debug!("Alert suppressed: {}", alert.id);
            return Ok(());
        }

        // Check rate limiting
        if !self.check_rate_limit().await {
            self.alerts_suppressed.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("Alert rate limit exceeded, suppressing alert: {}", alert.id);
            return Ok(());
        }

        // Send alert to appropriate channels
        self.send_alert(&alert).await?;

        // Store alert
        self.active_alerts.insert(alert.id.clone(), alert.clone());
        
        {
            let mut history = self.alert_history.write();
            history.push_back(alert.clone());
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        // Publish alert
        if self.alert_sender.receiver_count() > 0 {
            let _ = self.alert_sender.send(alert);
        }

        self.alerts_sent.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    async fn should_suppress_alert(&self, alert: &Alert) -> bool {
        // Check if similar alert was recently sent
        if let Some(last_sent) = self.suppressed_alerts.get(&alert.id) {
            let cooldown_duration = Duration::seconds(self.config.cooldown_seconds as i64);
            if Utc::now() - *last_sent < cooldown_duration {
                return true;
            }
        }

        // Update suppression timestamp
        self.suppressed_alerts.insert(alert.id.clone(), Utc::now());
        false
    }

    async fn check_rate_limit(&self) -> bool {
        let mut rate_limiter = self.alert_rate_limiter.write();
        rate_limiter.check_rate_limit(self.config.max_alerts_per_minute)
    }

    async fn send_alert(&self, alert: &Alert) -> Result<()> {
        for channel in &self.channels {
            if channel.should_send(alert) {
                if let Err(e) = channel.send_alert(alert).await {
                    tracing::error!("Failed to send alert via {}: {}", channel.name(), e);
                }
            }
        }
        Ok(())
    }

    fn create_threshold_alert(
        &self,
        alert_id: &str,
        title: &str,
        message: &str,
        severity: AlertSeverity,
        details: serde_json::Value,
    ) -> Alert {
        Alert {
            id: alert_id.to_string(),
            alert_type: AlertType::Threshold,
            severity,
            title: title.to_string(),
            message: message.to_string(),
            details,
            created_at: Utc::now(),
            resolved_at: None,
            tags: vec![format!("type:threshold"), format!("metric:{}", alert_id)],
        }
    }

    fn determine_severity(&self, value: f64, warning_threshold: f64, critical_threshold: f64) -> AlertSeverity {
        if value >= critical_threshold {
            AlertSeverity::Critical
        } else if value >= warning_threshold {
            AlertSeverity::High
        } else {
            AlertSeverity::Medium
        }
    }

    async fn alert_processing_loop(&self) {
        let mut interval = time::interval(std::time::Duration::from_secs(60));

        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.process_periodic_checks().await {
                tracing::error!("Periodic alert checks failed: {}", e);
            }
        }
    }

    async fn process_periodic_checks(&self) -> Result<()> {
        // Check for auto-resolved alerts
        let mut resolved_alerts = Vec::new();
        
        for entry in self.active_alerts.iter() {
            let (id, alert) = (entry.key(), entry.value());
            
            // Check if alert should be auto-resolved (implement specific logic)
            if self.should_auto_resolve_alert(alert).await? {
                resolved_alerts.push(id.clone());
            }
        }

        for alert_id in resolved_alerts {
            self.resolve_alert(&alert_id).await?;
        }

        Ok(())
    }

    async fn should_auto_resolve_alert(&self, _alert: &Alert) -> Result<bool> {
        // TODO: Implement auto-resolution logic based on current metrics
        Ok(false)
    }

    pub async fn resolve_alert(&self, alert_id: &str) -> Result<()> {
        if let Some((_, mut alert)) = self.active_alerts.remove(alert_id) {
            alert.resolved_at = Some(Utc::now());
            
            // Store resolved alert in history
            let mut history = self.alert_history.write();
            history.push_back(alert);
        }

        Ok(())
    }

    async fn cleanup_loop(&self) {
        let mut interval = time::interval(std::time::Duration::from_secs(3600)); // 1 hour

        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.cleanup_old_data().await {
                tracing::error!("Alert cleanup failed: {}", e);
            }
        }
    }

    async fn cleanup_old_data(&self) -> Result<()> {
        let cutoff = Utc::now() - Duration::hours(24);
        
        // Clean up old suppression entries
        self.suppressed_alerts.retain(|_, timestamp| *timestamp > cutoff);
        
        // Clean up rate limiter
        {
            let mut rate_limiter = self.alert_rate_limiter.write();
            rate_limiter.cleanup();
        }

        tracing::debug!("Alert cleanup completed");
        Ok(())
    }

    pub fn get_alert_stats(&self) -> AlertStats {
        AlertStats {
            alerts_sent: self.alerts_sent.load(Ordering::Relaxed),
            alerts_suppressed: self.alerts_suppressed.load(Ordering::Relaxed),
            active_alerts: self.active_alerts.len(),
            suppressed_entries: self.suppressed_alerts.len(),
        }
    }

    pub fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            running: AtomicBool::new(self.running.load(Ordering::Acquire)),
            alert_sender: self.alert_sender.clone(),
            active_alerts: self.active_alerts.clone(),
            alert_history: self.alert_history.clone(),
            suppressed_alerts: self.suppressed_alerts.clone(),
            alert_rate_limiter: self.alert_rate_limiter.clone(),
            channels: self.channels.clone(),
            alerts_sent: AtomicU64::new(self.alerts_sent.load(Ordering::Relaxed)),
            alerts_suppressed: AtomicU64::new(self.alerts_suppressed.load(Ordering::Relaxed)),
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        tracing::info!("Alert system shutdown complete");
        Ok(())
    }
}

/// Alert data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub details: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertType {
    Threshold,
    Bottleneck,
    System,
    Performance,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertStats {
    pub alerts_sent: u64,
    pub alerts_suppressed: u64,
    pub active_alerts: usize,
    pub suppressed_entries: usize,
}

/// Rate limiting for alerts
struct AlertRateLimiter {
    alerts_this_minute: VecDeque<DateTime<Utc>>,
}

impl AlertRateLimiter {
    fn new() -> Self {
        Self {
            alerts_this_minute: VecDeque::new(),
        }
    }

    fn check_rate_limit(&mut self, max_per_minute: u32) -> bool {
        let now = Utc::now();
        let cutoff = now - Duration::minutes(1);

        // Remove old entries
        while let Some(&front_time) = self.alerts_this_minute.front() {
            if front_time <= cutoff {
                self.alerts_this_minute.pop_front();
            } else {
                break;
            }
        }

        if self.alerts_this_minute.len() < max_per_minute as usize {
            self.alerts_this_minute.push_back(now);
            true
        } else {
            false
        }
    }

    fn cleanup(&mut self) {
        let cutoff = Utc::now() - Duration::minutes(5);
        while let Some(&front_time) = self.alerts_this_minute.front() {
            if front_time <= cutoff {
                self.alerts_this_minute.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Alert channel trait
#[async_trait]
pub trait AlertChannel: Send + Sync {
    fn name(&self) -> &str;
    fn should_send(&self, alert: &Alert) -> bool;
    async fn send_alert(&self, alert: &Alert) -> Result<()>;
}

/// Log-based alert channel
pub struct LogAlertChannel {
    name: String,
    min_severity: AlertSeverity,
}

impl LogAlertChannel {
    pub fn new(config: &crate::config::AlertChannel) -> Result<Self> {
        let min_severity = match config.min_severity.as_str() {
            "low" => AlertSeverity::Low,
            "medium" => AlertSeverity::Medium,
            "high" => AlertSeverity::High,
            "critical" => AlertSeverity::Critical,
            _ => AlertSeverity::Medium,
        };

        Ok(Self {
            name: "log".to_string(),
            min_severity,
        })
    }
}

#[async_trait]
impl AlertChannel for LogAlertChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn should_send(&self, alert: &Alert) -> bool {
        self.severity_level(alert.severity) >= self.severity_level(self.min_severity)
    }

    async fn send_alert(&self, alert: &Alert) -> Result<()> {
        match alert.severity {
            AlertSeverity::Critical => tracing::error!("ALERT [{}]: {} - {}", alert.severity as u8, alert.title, alert.message),
            AlertSeverity::High => tracing::warn!("ALERT [{}]: {} - {}", alert.severity as u8, alert.title, alert.message),
            AlertSeverity::Medium => tracing::warn!("ALERT [{}]: {} - {}", alert.severity as u8, alert.title, alert.message),
            AlertSeverity::Low => tracing::info!("ALERT [{}]: {} - {}", alert.severity as u8, alert.title, alert.message),
        }
        Ok(())
    }
}

impl LogAlertChannel {
    fn severity_level(&self, severity: AlertSeverity) -> u8 {
        match severity {
            AlertSeverity::Low => 1,
            AlertSeverity::Medium => 2,
            AlertSeverity::High => 3,
            AlertSeverity::Critical => 4,
        }
    }
}

/// Webhook alert channel (placeholder implementation)
pub struct WebhookAlertChannel {
    name: String,
    webhook_url: String,
    min_severity: AlertSeverity,
}

impl WebhookAlertChannel {
    pub fn new(config: &crate::config::AlertChannel) -> Result<Self> {
        let webhook_url = config.config.get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let min_severity = match config.min_severity.as_str() {
            "low" => AlertSeverity::Low,
            "medium" => AlertSeverity::Medium,
            "high" => AlertSeverity::High,
            "critical" => AlertSeverity::Critical,
            _ => AlertSeverity::Medium,
        };

        Ok(Self {
            name: "webhook".to_string(),
            webhook_url,
            min_severity,
        })
    }
}

#[async_trait]
impl AlertChannel for WebhookAlertChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn should_send(&self, alert: &Alert) -> bool {
        !self.webhook_url.is_empty() && 
        self.severity_level(alert.severity) >= self.severity_level(self.min_severity)
    }

    async fn send_alert(&self, alert: &Alert) -> Result<()> {
        // TODO: Implement webhook sending
        tracing::debug!("Would send webhook alert to {}: {}", self.webhook_url, alert.title);
        Ok(())
    }
}

impl WebhookAlertChannel {
    fn severity_level(&self, severity: AlertSeverity) -> u8 {
        match severity {
            AlertSeverity::Low => 1,
            AlertSeverity::Medium => 2,
            AlertSeverity::High => 3,
            AlertSeverity::Critical => 4,
        }
    }
}

/// Email alert channel (placeholder implementation)
pub struct EmailAlertChannel {
    name: String,
    recipients: Vec<String>,
    min_severity: AlertSeverity,
}

impl EmailAlertChannel {
    pub fn new(config: &crate::config::AlertChannel) -> Result<Self> {
        let recipients = config.config.get("recipients")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_default();

        let min_severity = match config.min_severity.as_str() {
            "low" => AlertSeverity::Low,
            "medium" => AlertSeverity::Medium,
            "high" => AlertSeverity::High,
            "critical" => AlertSeverity::Critical,
            _ => AlertSeverity::High,
        };

        Ok(Self {
            name: "email".to_string(),
            recipients,
            min_severity,
        })
    }
}

#[async_trait]
impl AlertChannel for EmailAlertChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn should_send(&self, alert: &Alert) -> bool {
        !self.recipients.is_empty() && 
        self.severity_level(alert.severity) >= self.severity_level(self.min_severity)
    }

    async fn send_alert(&self, alert: &Alert) -> Result<()> {
        // TODO: Implement email sending
        tracing::debug!("Would send email alert to {:?}: {}", self.recipients, alert.title);
        Ok(())
    }
}

impl EmailAlertChannel {
    fn severity_level(&self, severity: AlertSeverity) -> u8 {
        match severity {
            AlertSeverity::Low => 1,
            AlertSeverity::Medium => 2,
            AlertSeverity::High => 3,
            AlertSeverity::Critical => 4,
        }
    }
}