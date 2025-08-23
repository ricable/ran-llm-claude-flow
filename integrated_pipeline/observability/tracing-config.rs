//! Distributed Tracing Configuration for RAN LLM Claude Flow
//! Implements OpenTelemetry-based tracing for Rust-Python IPC communication
//! Targets: <100μs IPC latency tracking, zero-copy operations monitoring

use opentelemetry::{
    global, sdk::trace as sdktrace, trace::{TraceError, Tracer},
};
use opentelemetry_jaeger::JaegerTraceRuntime;
use opentelemetry_otlp::WithExportConfig;
use std::collections::HashMap;
use tracing::{info, instrument, span, Level};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Configuration for distributed tracing system
#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub service_name: String,
    pub jaeger_endpoint: String,
    pub sampling_rate: f64,
    pub max_events_per_span: u32,
    pub enable_ipc_tracing: bool,
    pub enable_performance_tracing: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "ran-llm-claude-flow".to_string(),
            jaeger_endpoint: "http://localhost:14268/api/traces".to_string(),
            sampling_rate: 0.1, // 10% sampling to minimize overhead
            max_events_per_span: 128,
            enable_ipc_tracing: true,
            enable_performance_tracing: true,
        }
    }
}

/// Initialize distributed tracing system with OpenTelemetry
pub fn init_tracing(config: TracingConfig) -> Result<(), TraceError> {
    // Configure OpenTelemetry tracer
    let tracer = opentelemetry_jaeger::new_agent_pipeline()
        .with_service_name(&config.service_name)
        .with_endpoint(&config.jaeger_endpoint)
        .with_max_packet_size(65000)
        .with_auto_split_batch(true)
        .install_batch(JaegerTraceRuntime::Tokio)?;

    // Configure tracing subscriber with OpenTelemetry layer
    let telemetry_layer = OpenTelemetryLayer::new(tracer);
    
    tracing_subscriber::registry()
        .with(telemetry_layer)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_level(true)
        )
        .with(
            tracing_subscriber::filter::LevelFilter::from_level(Level::INFO)
        )
        .init();

    info!("Distributed tracing initialized for service: {}", config.service_name);
    Ok(())
}

/// Trace IPC operations with microsecond precision
#[instrument(level = "debug", skip(data))]
pub async fn trace_ipc_operation<T>(
    operation: &str,
    component: &str,
    data_size: usize,
    operation_fn: impl std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();
    
    let span = span!(Level::DEBUG, "ipc_operation",
        operation = operation,
        component = component,
        data_size = data_size,
    );
    
    let _entered = span.enter();
    
    match operation_fn.await {
        Ok(result) => {
            let duration = start_time.elapsed();
            span.record("duration_us", duration.as_micros() as i64);
            span.record("success", true);
            
            // Alert if IPC latency exceeds 100μs threshold
            if duration.as_micros() > 100 {
                tracing::warn!(
                    "IPC operation {} exceeded 100μs threshold: {}μs",
                    operation,
                    duration.as_micros()
                );
            }
            
            Ok(result)
        }
        Err(e) => {
            let duration = start_time.elapsed();
            span.record("duration_us", duration.as_micros() as i64);
            span.record("success", false);
            span.record("error", e.to_string().as_str());
            Err(e)
        }
    }
}

/// Trace document processing pipeline with performance metrics
#[instrument(level = "info", skip(processor))]
pub async fn trace_document_processing<T>(
    document_id: &str,
    processing_stage: &str,
    processor: impl std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();
    
    let span = span!(Level::INFO, "document_processing",
        document_id = document_id,
        stage = processing_stage,
    );
    
    let _entered = span.enter();
    
    match processor.await {
        Ok(result) => {
            let duration = start_time.elapsed();
            span.record("duration_ms", duration.as_millis() as i64);
            span.record("success", true);
            
            // Record performance metrics for throughput calculation
            metrics::counter!("documents_processed_total", 1,
                "stage" => processing_stage.to_string(),
                "component" => "rust_core".to_string()
            );
            metrics::histogram!("document_processing_duration_ms", duration.as_millis() as f64,
                "stage" => processing_stage.to_string()
            );
            
            Ok(result)
        }
        Err(e) => {
            let duration = start_time.elapsed();
            span.record("duration_ms", duration.as_millis() as i64);
            span.record("success", false);
            span.record("error", e.to_string().as_str());
            
            metrics::counter!("document_processing_errors_total", 1,
                "stage" => processing_stage.to_string(),
                "error" => e.to_string()
            );
            
            Err(e)
        }
    }
}

/// Trace memory operations with zero-copy validation
#[instrument(level = "debug", skip(operation))]
pub fn trace_memory_operation<T>(
    operation_type: &str,
    memory_pool: &str,
    size_bytes: usize,
    operation: impl FnOnce() -> Result<T, Box<dyn std::error::Error + Send + Sync>>,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();
    
    let span = span!(Level::DEBUG, "memory_operation",
        operation_type = operation_type,
        memory_pool = memory_pool,
        size_bytes = size_bytes,
    );
    
    let _entered = span.enter();
    
    match operation() {
        Ok(result) => {
            let duration = start_time.elapsed();
            span.record("duration_ns", duration.as_nanos() as i64);
            span.record("success", true);
            
            // Record memory metrics
            metrics::histogram!("memory_operation_duration_ns", duration.as_nanos() as f64,
                "operation" => operation_type.to_string(),
                "pool" => memory_pool.to_string()
            );
            metrics::counter!("memory_operations_total", 1,
                "operation" => operation_type.to_string(),
                "pool" => memory_pool.to_string()
            );
            
            Ok(result)
        }
        Err(e) => {
            let duration = start_time.elapsed();
            span.record("duration_ns", duration.as_nanos() as i64);
            span.record("success", false);
            span.record("error", e.to_string().as_str());
            
            metrics::counter!("memory_operation_errors_total", 1,
                "operation" => operation_type.to_string(),
                "error" => e.to_string()
            );
            
            Err(e)
        }
    }
}

/// Trace neural model inference with performance tracking
#[instrument(level = "info", skip(inference_fn))]
pub async fn trace_neural_inference<T>(
    model_name: &str,
    input_tokens: usize,
    inference_fn: impl std::future::Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync>>>,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = std::time::Instant::now();
    
    let span = span!(Level::INFO, "neural_inference",
        model = model_name,
        input_tokens = input_tokens,
    );
    
    let _entered = span.enter();
    
    match inference_fn.await {
        Ok(result) => {
            let duration = start_time.elapsed();
            span.record("duration_ms", duration.as_millis() as i64);
            span.record("success", true);
            
            // Calculate tokens per second for performance monitoring
            let tokens_per_sec = if duration.as_millis() > 0 {
                (input_tokens as f64) / (duration.as_millis() as f64 / 1000.0)
            } else {
                0.0
            };
            
            span.record("tokens_per_second", tokens_per_sec as i64);
            
            metrics::histogram!("neural_inference_duration_ms", duration.as_millis() as f64,
                "model" => model_name.to_string()
            );
            metrics::histogram!("neural_inference_tokens_per_second", tokens_per_sec,
                "model" => model_name.to_string()
            );
            
            Ok(result)
        }
        Err(e) => {
            let duration = start_time.elapsed();
            span.record("duration_ms", duration.as_millis() as i64);
            span.record("success", false);
            span.record("error", e.to_string().as_str());
            
            metrics::counter!("neural_inference_errors_total", 1,
                "model" => model_name.to_string(),
                "error" => e.to_string()
            );
            
            Err(e)
        }
    }
}

/// Create custom span for bottleneck analysis
pub fn create_bottleneck_span(component: &str, operation: &str) -> tracing::Span {
    span!(Level::WARN, "bottleneck_detection",
        component = component,
        operation = operation,
        severity = tracing::field::Empty,
        recommendation = tracing::field::Empty,
    )
}

/// Shutdown tracing system gracefully
pub fn shutdown_tracing() {
    global::shutdown_tracer_provider();
    info!("Distributed tracing system shut down");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_ipc_tracing() {
        let config = TracingConfig::default();
        let _ = init_tracing(config);

        let result = trace_ipc_operation(
            "test_operation",
            "test_component", 
            1024,
            async {
                sleep(Duration::from_micros(50)).await;
                Ok("success")
            },
        ).await;

        assert!(result.is_ok());
    }

    #[tokio::test] 
    async fn test_document_processing_tracing() {
        let result = trace_document_processing(
            "doc_123",
            "processing",
            async {
                sleep(Duration::from_millis(10)).await;
                Ok("processed")
            },
        ).await;

        assert!(result.is_ok());
    }
}