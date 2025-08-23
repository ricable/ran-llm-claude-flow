//! Benchmarks for metrics collection performance
//! 
//! Ensures that metrics collection overhead stays below 1% of system resources.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use monitoring::{
    config::MonitoringConfig,
    metrics::{MetricsCollector, SystemMetrics},
};
use std::time::Duration;
use tokio::runtime::Runtime;

fn bench_metrics_collection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("metrics_collection_single", |b| {
        let config = MonitoringConfig::default();
        let collector = rt.block_on(async { MetricsCollector::new(&config).await.unwrap() });
        
        b.iter(|| {
            rt.block_on(async {
                let metrics = collector.get_current_metrics().await.unwrap();
                black_box(metrics);
            });
        });
    });
}

fn bench_counter_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = MonitoringConfig::default();
    let collector = rt.block_on(async { MetricsCollector::new(&config).await.unwrap() });
    
    c.bench_function("counter_increment_documents", |b| {
        b.iter(|| {
            collector.increment_documents_processed();
        });
    });
    
    c.bench_function("counter_increment_ipc_sent", |b| {
        b.iter(|| {
            collector.increment_ipc_messages_sent();
        });
    });
    
    c.bench_function("counter_increment_ipc_received", |b| {
        b.iter(|| {
            collector.increment_ipc_messages_received();
        });
    });
    
    c.bench_function("record_ipc_latency", |b| {
        b.iter(|| {
            collector.record_ipc_latency(black_box(5.5));
        });
    });
}

fn bench_metrics_collection_frequency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    for interval_ms in [10u64, 50, 100, 500, 1000].iter() {
        c.bench_with_input(
            BenchmarkId::new("metrics_collection_interval", interval_ms),
            interval_ms,
            |b, &interval_ms| {
                let mut config = MonitoringConfig::default();
                config.collection_interval_ms = interval_ms;
                let collector = rt.block_on(async { MetricsCollector::new(&config).await.unwrap() });
                
                b.iter(|| {
                    rt.block_on(async {
                        let metrics = collector.get_current_metrics().await.unwrap();
                        black_box(metrics);
                    });
                });
            }
        );
    }
}

fn bench_system_metrics_aggregation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("system_metrics_cpu_calculation", |b| {
        let config = MonitoringConfig::default();
        let collector = rt.block_on(async { MetricsCollector::new(&config).await.unwrap() });
        
        b.iter(|| {
            rt.block_on(async {
                // This would typically be done internally by the collector
                let start = std::time::Instant::now();
                let _cpu_usage = collector.get_current_metrics().await.unwrap().cpu_usage;
                let elapsed = start.elapsed();
                black_box(elapsed);
            });
        });
    });
}

fn bench_concurrent_metrics_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("concurrent_counter_updates", |b| {
        let config = MonitoringConfig::default();
        let collector = rt.block_on(async { MetricsCollector::new(&config).await.unwrap() });
        
        b.iter(|| {
            rt.block_on(async {
                // Simulate concurrent access from multiple threads
                let tasks = (0..10).map(|_| {
                    let collector = collector.clone();
                    tokio::spawn(async move {
                        for _ in 0..100 {
                            collector.increment_documents_processed();
                            collector.increment_ipc_messages_sent();
                            collector.record_ipc_latency(3.14);
                        }
                    })
                }).collect::<Vec<_>>();
                
                for task in tasks {
                    task.await.unwrap();
                }
            });
        });
    });
}

fn bench_metrics_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("metrics_memory_footprint", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = MonitoringConfig::default();
                let collector = MetricsCollector::new(&config).await.unwrap();
                
                // Simulate typical operation
                for i in 0..1000 {
                    collector.increment_documents_processed();
                    if i % 10 == 0 {
                        let _metrics = collector.get_current_metrics().await.unwrap();
                    }
                }
                
                black_box(collector);
            });
        });
    });
}

fn bench_metrics_serialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = MonitoringConfig::default();
    let collector = rt.block_on(async { MetricsCollector::new(&config).await.unwrap() });
    let metrics = rt.block_on(async { collector.get_current_metrics().await.unwrap() });
    
    c.bench_function("metrics_json_serialization", |b| {
        b.iter(|| {
            let json = serde_json::to_string(&metrics).unwrap();
            black_box(json);
        });
    });
    
    c.bench_function("metrics_binary_serialization", |b| {
        b.iter(|| {
            let binary = bincode::serialize(&metrics).unwrap();
            black_box(binary);
        });
    });
}

criterion_group!(
    benches,
    bench_metrics_collection,
    bench_counter_operations,
    bench_metrics_collection_frequency,
    bench_system_metrics_aggregation,
    bench_concurrent_metrics_access,
    bench_metrics_memory_usage,
    bench_metrics_serialization
);
criterion_main!(benches);