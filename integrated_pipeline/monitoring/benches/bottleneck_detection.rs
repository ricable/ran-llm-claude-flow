//! Benchmarks for bottleneck detection performance
//! 
//! Ensures bottleneck detection can analyze system state within 5 seconds
//! while maintaining high accuracy.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use monitoring::{
    bottleneck_analyzer::{BottleneckAnalyzer, Bottleneck, BottleneckType, BottleneckSeverity},
    config::MonitoringConfig,
    metrics::SystemMetrics,
};
use chrono::Utc;
use std::collections::VecDeque;
use tokio::runtime::Runtime;

fn bench_bottleneck_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("bottleneck_detection_single_pass", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        let test_metrics = generate_test_metrics(100);
        
        b.iter(|| {
            rt.block_on(async {
                // Add metrics to analyzer
                for metric in &test_metrics {
                    analyzer.add_metrics(metric.clone());
                }
                
                let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                black_box(bottlenecks);
            });
        });
    });
}

fn bench_bottleneck_pattern_matching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    for metric_count in [10, 50, 100, 500, 1000].iter() {
        c.bench_with_input(
            BenchmarkId::new("pattern_matching", metric_count),
            metric_count,
            |b, &metric_count| {
                let config = MonitoringConfig::default();
                let analyzer = rt.block_on(async { 
                    BottleneckAnalyzer::new(&config).unwrap()
                });
                
                let test_metrics = generate_test_metrics(metric_count);
                
                b.iter(|| {
                    rt.block_on(async {
                        // Simulate pattern analysis
                        for metric in &test_metrics {
                            analyzer.add_metrics(metric.clone());
                        }
                        
                        let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                        black_box(bottlenecks);
                    });
                });
            }
        );
    }
}

fn bench_cpu_bottleneck_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("cpu_bottleneck_analysis", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        let high_cpu_metrics = generate_high_cpu_metrics(50);
        
        b.iter(|| {
            rt.block_on(async {
                for metric in &high_cpu_metrics {
                    analyzer.add_metrics(metric.clone());
                }
                
                let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                black_box(bottlenecks);
            });
        });
    });
}

fn bench_memory_bottleneck_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("memory_bottleneck_analysis", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        let high_memory_metrics = generate_high_memory_metrics(50);
        
        b.iter(|| {
            rt.block_on(async {
                for metric in &high_memory_metrics {
                    analyzer.add_metrics(metric.clone());
                }
                
                let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                black_box(bottlenecks);
            });
        });
    });
}

fn bench_ipc_bottleneck_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("ipc_bottleneck_analysis", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        let high_latency_metrics = generate_high_ipc_latency_metrics(50);
        
        b.iter(|| {
            rt.block_on(async {
                for metric in &high_latency_metrics {
                    analyzer.add_metrics(metric.clone());
                }
                
                let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                black_box(bottlenecks);
            });
        });
    });
}

fn bench_bottleneck_threshold_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("threshold_detection_accuracy", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        // Mix of normal and problematic metrics
        let mixed_metrics = generate_mixed_performance_metrics(100);
        
        b.iter(|| {
            rt.block_on(async {
                for metric in &mixed_metrics {
                    analyzer.add_metrics(metric.clone());
                }
                
                let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                
                // Verify we detect the right number of bottlenecks
                let critical_count = bottlenecks.iter()
                    .filter(|b| matches!(b.severity, BottleneckSeverity::Critical))
                    .count();
                
                black_box((bottlenecks, critical_count));
            });
        });
    });
}

fn bench_concurrent_bottleneck_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("concurrent_analysis", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        b.iter(|| {
            rt.block_on(async {
                // Simulate concurrent metric ingestion and analysis
                let tasks = (0..5).map(|i| {
                    let analyzer = analyzer.clone();
                    tokio::spawn(async move {
                        let metrics = generate_test_metrics(20);
                        for metric in metrics {
                            analyzer.add_metrics(metric);
                        }
                        analyzer.detect_current_bottlenecks().await.unwrap()
                    })
                }).collect::<Vec<_>>();
                
                let results = futures::future::join_all(tasks).await;
                let all_bottlenecks: Vec<_> = results.into_iter()
                    .filter_map(|r| r.ok())
                    .flatten()
                    .collect();
                
                black_box(all_bottlenecks);
            });
        });
    });
}

fn bench_bottleneck_history_analysis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("historical_pattern_analysis", |b| {
        let config = MonitoringConfig::default();
        let analyzer = rt.block_on(async { 
            BottleneckAnalyzer::new(&config).unwrap()
        });
        
        // Pre-populate with historical data
        let historical_metrics = generate_time_series_metrics(1000);
        rt.block_on(async {
            for metric in &historical_metrics {
                analyzer.add_metrics(metric.clone());
            }
        });
        
        b.iter(|| {
            rt.block_on(async {
                // Add new metrics and analyze patterns
                let new_metrics = generate_test_metrics(10);
                for metric in new_metrics {
                    analyzer.add_metrics(metric);
                }
                
                let bottlenecks = analyzer.detect_current_bottlenecks().await.unwrap();
                black_box(bottlenecks);
            });
        });
    });
}

// Helper functions to generate test data

fn generate_test_metrics(count: usize) -> Vec<SystemMetrics> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..count).map(|i| {
        let mut metrics = SystemMetrics::default();
        metrics.timestamp = Utc::now();
        metrics.cpu_usage.utilization_percent = rng.gen_range(20.0..80.0);
        metrics.memory.used_bytes = rng.gen_range(1_000_000_000..8_000_000_000);
        metrics.memory.total_bytes = 16_000_000_000;
        metrics.ipc_latency_p99 = rng.gen_range(1.0..15.0);
        metrics.document_processing_rate = rng.gen_range(15.0..35.0);
        metrics.documents_processed = (i * 10) as u64;
        metrics.errors_total = (i / 100) as u64;
        metrics
    }).collect()
}

fn generate_high_cpu_metrics(count: usize) -> Vec<SystemMetrics> {
    (0..count).map(|i| {
        let mut metrics = SystemMetrics::default();
        metrics.timestamp = Utc::now();
        metrics.cpu_usage.utilization_percent = 95.0 + (i as f32 * 0.1); // High CPU
        metrics.memory.used_bytes = 4_000_000_000;
        metrics.memory.total_bytes = 16_000_000_000;
        metrics.ipc_latency_p99 = 5.0;
        metrics.document_processing_rate = 10.0; // Low due to CPU bottleneck
        metrics.documents_processed = (i * 5) as u64;
        metrics
    }).collect()
}

fn generate_high_memory_metrics(count: usize) -> Vec<SystemMetrics> {
    (0..count).map(|i| {
        let mut metrics = SystemMetrics::default();
        metrics.timestamp = Utc::now();
        metrics.cpu_usage.utilization_percent = 50.0;
        metrics.memory.used_bytes = 15_000_000_000; // High memory usage
        metrics.memory.total_bytes = 16_000_000_000;
        metrics.memory.rust_heap_mb = 65_000.0; // Over limit
        metrics.memory.python_heap_mb = 50_000.0; // Over limit
        metrics.ipc_latency_p99 = 8.0;
        metrics.document_processing_rate = 12.0; // Low due to memory pressure
        metrics.documents_processed = (i * 6) as u64;
        metrics
    }).collect()
}

fn generate_high_ipc_latency_metrics(count: usize) -> Vec<SystemMetrics> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..count).map(|i| {
        let mut metrics = SystemMetrics::default();
        metrics.timestamp = Utc::now();
        metrics.cpu_usage.utilization_percent = 60.0;
        metrics.memory.used_bytes = 8_000_000_000;
        metrics.memory.total_bytes = 16_000_000_000;
        metrics.ipc_latency_p99 = rng.gen_range(25.0..100.0); // High latency
        metrics.document_processing_rate = 8.0; // Low due to IPC bottleneck
        metrics.documents_processed = (i * 4) as u64;
        metrics.ipc_messages_sent = (i * 50) as u64;
        metrics.ipc_messages_received = (i * 45) as u64;
        metrics
    }).collect()
}

fn generate_mixed_performance_metrics(count: usize) -> Vec<SystemMetrics> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..count).map(|i| {
        let mut metrics = SystemMetrics::default();
        metrics.timestamp = Utc::now();
        
        // 20% chance of performance issues
        if rng.gen_bool(0.2) {
            match rng.gen_range(0..3) {
                0 => {
                    // CPU bottleneck
                    metrics.cpu_usage.utilization_percent = rng.gen_range(90.0..100.0);
                    metrics.document_processing_rate = rng.gen_range(5.0..15.0);
                },
                1 => {
                    // Memory bottleneck
                    metrics.memory.used_bytes = rng.gen_range(14_000_000_000..16_000_000_000);
                    metrics.memory.rust_heap_mb = rng.gen_range(65_000.0..80_000.0);
                },
                _ => {
                    // IPC bottleneck
                    metrics.ipc_latency_p99 = rng.gen_range(20.0..80.0);
                    metrics.errors_total = (i / 20) as u64;
                }
            }
        } else {
            // Normal performance
            metrics.cpu_usage.utilization_percent = rng.gen_range(30.0..70.0);
            metrics.memory.used_bytes = rng.gen_range(4_000_000_000..12_000_000_000);
            metrics.ipc_latency_p99 = rng.gen_range(2.0..8.0);
            metrics.document_processing_rate = rng.gen_range(20.0..30.0);
        }
        
        metrics.memory.total_bytes = 16_000_000_000;
        metrics.documents_processed = (i * 10) as u64;
        metrics
    }).collect()
}

fn generate_time_series_metrics(count: usize) -> Vec<SystemMetrics> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..count).map(|i| {
        let mut metrics = SystemMetrics::default();
        metrics.timestamp = Utc::now() - chrono::Duration::minutes((count - i) as i64);
        
        // Simulate realistic time series with trends
        let time_factor = i as f32 / count as f32;
        let cpu_trend = 40.0 + (time_factor * 30.0) + rng.gen_range(-10.0..10.0);
        let memory_trend = 6_000_000_000.0 + (time_factor * 4_000_000_000.0) + rng.gen_range(-1_000_000_000.0..1_000_000_000.0);
        
        metrics.cpu_usage.utilization_percent = cpu_trend.max(10.0).min(100.0);
        metrics.memory.used_bytes = memory_trend.max(2_000_000_000.0).min(15_000_000_000.0) as u64;
        metrics.memory.total_bytes = 16_000_000_000;
        metrics.ipc_latency_p99 = 3.0 + rng.gen_range(-1.0..5.0);
        metrics.document_processing_rate = 25.0 - (time_factor * 10.0) + rng.gen_range(-5.0..5.0);
        metrics.documents_processed = (i * 8) as u64;
        metrics.errors_total = (i / 150) as u64;
        
        metrics
    }).collect()
}

criterion_group!(
    benches,
    bench_bottleneck_analysis,
    bench_bottleneck_pattern_matching,
    bench_cpu_bottleneck_analysis,
    bench_memory_bottleneck_analysis,
    bench_ipc_bottleneck_analysis,
    bench_bottleneck_threshold_detection,
    bench_concurrent_bottleneck_detection,
    bench_bottleneck_history_analysis
);
criterion_main!(benches);