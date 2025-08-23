/*!
# Performance Monitor

Real-time performance monitoring for the document processing pipeline.
*/

use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Performance monitor for tracking pipeline metrics
#[derive(Debug)]
pub struct PerformanceMonitor {
    monitor_id: Uuid,
    metrics: Arc<PerformanceMetrics>,
    start_time: SystemTime,
}

#[derive(Debug)]
pub struct PerformanceMetrics {
    pub documents_processed: AtomicU64,
    pub total_processing_time_ms: AtomicU64,
    pub errors_encountered: AtomicU64,
    pub peak_memory_usage_mb: AtomicU64,
    pub current_memory_usage_mb: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub monitor_id: Uuid,
    pub uptime_seconds: u64,
    pub documents_processed: u64,
    pub documents_per_second: f64,
    pub average_processing_time_ms: f64,
    pub error_rate: f64,
    pub memory_usage_mb: u64,
    pub peak_memory_usage_mb: u64,
    pub timestamp: SystemTime,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            monitor_id: Uuid::new_v4(),
            metrics: Arc::new(PerformanceMetrics {
                documents_processed: AtomicU64::new(0),
                total_processing_time_ms: AtomicU64::new(0),
                errors_encountered: AtomicU64::new(0),
                peak_memory_usage_mb: AtomicU64::new(0),
                current_memory_usage_mb: AtomicU64::new(0),
            }),
            start_time: SystemTime::now(),
        }
    }

    /// Record document processing
    pub fn record_document_processed(&self, processing_time_ms: u64) {
        self.metrics
            .documents_processed
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_processing_time_ms
            .fetch_add(processing_time_ms, Ordering::Relaxed);
    }

    /// Record error
    pub fn record_error(&self) {
        self.metrics
            .errors_encountered
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, current_mb: u64) {
        self.metrics
            .current_memory_usage_mb
            .store(current_mb, Ordering::Relaxed);

        // Update peak if necessary
        let current_peak = self.metrics.peak_memory_usage_mb.load(Ordering::Relaxed);
        if current_mb > current_peak {
            self.metrics
                .peak_memory_usage_mb
                .store(current_mb, Ordering::Relaxed);
        }
    }

    /// Get current performance snapshot
    pub fn get_snapshot(&self) -> Result<PerformanceSnapshot> {
        let uptime = self
            .start_time
            .elapsed()
            .unwrap_or(Duration::from_secs(0))
            .as_secs();

        let documents_processed = self.metrics.documents_processed.load(Ordering::Relaxed);
        let total_processing_time = self
            .metrics
            .total_processing_time_ms
            .load(Ordering::Relaxed);
        let errors = self.metrics.errors_encountered.load(Ordering::Relaxed);

        let documents_per_second = if uptime > 0 {
            documents_processed as f64 / uptime as f64
        } else {
            0.0
        };

        let average_processing_time = if documents_processed > 0 {
            total_processing_time as f64 / documents_processed as f64
        } else {
            0.0
        };

        let error_rate = if documents_processed > 0 {
            errors as f64 / documents_processed as f64
        } else {
            0.0
        };

        Ok(PerformanceSnapshot {
            monitor_id: self.monitor_id,
            uptime_seconds: uptime,
            documents_processed,
            documents_per_second,
            average_processing_time_ms: average_processing_time,
            error_rate,
            memory_usage_mb: self.metrics.current_memory_usage_mb.load(Ordering::Relaxed),
            peak_memory_usage_mb: self.metrics.peak_memory_usage_mb.load(Ordering::Relaxed),
            timestamp: SystemTime::now(),
        })
    }

    /// Reset metrics
    pub fn reset(&self) {
        self.metrics.documents_processed.store(0, Ordering::Relaxed);
        self.metrics
            .total_processing_time_ms
            .store(0, Ordering::Relaxed);
        self.metrics.errors_encountered.store(0, Ordering::Relaxed);
        self.metrics
            .peak_memory_usage_mb
            .store(0, Ordering::Relaxed);
        self.metrics
            .current_memory_usage_mb
            .store(0, Ordering::Relaxed);
    }
}

/// Initialize performance monitor
pub async fn initialize() -> Result<()> {
    tracing::info!("Initializing performance monitor");
    Ok(())
}
