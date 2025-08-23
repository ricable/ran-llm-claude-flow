use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;

use integrated_pipeline_tests::fixtures::test_data::TestDataFixtures;
use integrated_pipeline_tests::performance::PerformanceTestSuite;
use integrated_pipeline_tests::e2e::E2EPipelineTestSuite;
use rust_core::types::*;

/// Benchmarks for document processing throughput
fn document_processing_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("document_processing");
    
    // Benchmark different document sizes
    let document_sizes = vec![1, 5, 10, 20, 50];
    
    for size in document_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_processing", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let mut suite = PerformanceTestSuite::new();
                    black_box(suite.measure_batch_throughput(size).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for different document formats
fn document_format_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("document_formats");
    
    let test_documents = vec![
        ("markdown", TestDataFixtures::sample_ericsson_document()),
        ("3gpp", TestDataFixtures::complex_3gpp_document()),
        ("csv", TestDataFixtures::simple_csv_document()),
    ];
    
    for (format_name, document) in test_documents {
        group.throughput(Throughput::Bytes(document.size_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("format_processing", format_name),
            &document,
            |b, doc| {
                b.to_async(&rt).iter(|| async {
                    // Simulate document processing time based on complexity
                    let processing_time = match doc.metadata.complexity_hints.estimated_complexity {
                        ComplexityLevel::Fast => Duration::from_millis(200),
                        ComplexityLevel::Balanced => Duration::from_millis(800),
                        ComplexityLevel::Quality => Duration::from_millis(2000),
                    };
                    
                    tokio::time::sleep(processing_time).await;
                    black_box(doc.clone())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for QA pair generation quality vs speed tradeoffs
fn qa_generation_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("qa_generation");
    
    let complexity_levels = vec![
        ComplexityLevel::Fast,
        ComplexityLevel::Balanced, 
        ComplexityLevel::Quality,
    ];
    
    for complexity in complexity_levels {
        let complexity_name = match complexity {
            ComplexityLevel::Fast => "fast",
            ComplexityLevel::Balanced => "balanced",
            ComplexityLevel::Quality => "quality",
        };
        
        group.bench_with_input(
            BenchmarkId::new("qa_generation", complexity_name),
            &complexity,
            |b, &complexity| {
                b.to_async(&rt).iter(|| async {
                    let qa_pairs = TestDataFixtures::expected_qa_pairs();
                    
                    // Simulate processing time based on complexity
                    let processing_time = match complexity {
                        ComplexityLevel::Fast => Duration::from_millis(500),
                        ComplexityLevel::Balanced => Duration::from_millis(1500),
                        ComplexityLevel::Quality => Duration::from_millis(3000),
                    };
                    
                    tokio::time::sleep(processing_time).await;
                    black_box(qa_pairs)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for concurrent processing scalability
fn concurrent_processing_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_processing");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for concurrent tests
    
    let concurrency_levels = vec![1, 2, 4, 8, 16];
    
    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent_docs", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let documents = (0..concurrency)
                        .map(|_| TestDataFixtures::sample_ericsson_document())
                        .collect::<Vec<_>>();
                    
                    // Process documents concurrently
                    let mut handles = Vec::new();
                    for doc in documents {
                        let handle = tokio::spawn(async move {
                            tokio::time::sleep(Duration::from_millis(1000)).await;
                            doc
                        });
                        handles.push(handle);
                    }
                    
                    let results = futures::future::join_all(handles).await;
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for memory allocation and deallocation patterns
fn memory_allocation_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_allocation");
    
    let allocation_sizes = vec![1024, 10240, 102400, 1048576]; // 1KB to 1MB
    
    for size in allocation_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("memory_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let allocation = vec![0u8; size];
                    black_box(allocation);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for IPC communication latency
fn ipc_communication_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("ipc_communication");
    
    let message_sizes = vec![1024, 10240, 102400, 1048576]; // 1KB to 1MB messages
    
    for size in message_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("ipc_message", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    // Simulate IPC message serialization/deserialization
                    let message_data = vec![0u8; size];
                    let serialized = serde_json::to_vec(&message_data).unwrap();
                    let _deserialized: Vec<u8> = serde_json::from_slice(&serialized).unwrap();
                    
                    // Simulate network/IPC latency
                    tokio::time::sleep(Duration::from_micros(100)).await;
                    black_box(size)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for quality assessment algorithms
fn quality_assessment_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("quality_assessment");
    
    let qa_pairs = TestDataFixtures::expected_qa_pairs();
    let document = TestDataFixtures::sample_ericsson_document();
    
    group.bench_function("quality_scoring", |b| {
        b.iter(|| {
            for qa_pair in &qa_pairs {
                // Simulate quality scoring algorithm
                let question_score = qa_pair.question.len() as f64 / 100.0;
                let answer_score = qa_pair.answer.len() as f64 / 200.0;
                let relevance_score = if document.content.contains(&qa_pair.question[..10.min(qa_pair.question.len())]) { 1.0 } else { 0.5 };
                
                let overall_score = (question_score + answer_score + relevance_score) / 3.0;
                black_box(overall_score);
            }
        });
    });
    
    group.bench_function("diversity_analysis", |b| {
        b.iter(|| {
            // Simulate diversity analysis
            let mut vocabulary = std::collections::HashSet::new();
            let mut total_words = 0;
            
            for qa_pair in &qa_pairs {
                for word in qa_pair.question.split_whitespace() {
                    vocabulary.insert(word.to_lowercase());
                    total_words += 1;
                }
                for word in qa_pair.answer.split_whitespace() {
                    vocabulary.insert(word.to_lowercase());
                    total_words += 1;
                }
            }
            
            let diversity_score = vocabulary.len() as f64 / total_words as f64;
            black_box(diversity_score);
        });
    });
    
    group.finish();
}

/// Benchmarks for end-to-end pipeline performance
fn e2e_pipeline_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("e2e_pipeline");
    group.measurement_time(Duration::from_secs(60)); // Longer measurement for E2E tests
    group.sample_size(10); // Fewer samples for expensive E2E tests
    
    group.bench_function("complete_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            let mut suite = E2EPipelineTestSuite::new().unwrap();
            black_box(suite.test_complete_pipeline_workflow().await.unwrap())
        });
    });
    
    group.bench_function("multi_format_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let mut suite = E2EPipelineTestSuite::new().unwrap();
            black_box(suite.test_multi_format_processing().await.unwrap())
        });
    });
    
    group.finish();
}

/// Regression benchmarks to track performance over time
fn regression_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("regression_tracking");
    
    // Standardized benchmark for tracking performance regression
    group.bench_function("standard_workload", |b| {
        b.to_async(&rt).iter(|| async {
            // Process 10 documents of mixed complexity
            let documents = vec![
                TestDataFixtures::simple_csv_document(),
                TestDataFixtures::sample_ericsson_document(),
                TestDataFixtures::complex_3gpp_document(),
            ];
            
            let mut total_processing_time = Duration::from_secs(0);
            let mut total_qa_pairs = 0;
            let mut quality_scores = Vec::new();
            
            for doc in documents {
                let start = std::time::Instant::now();
                
                // Simulate document processing
                let processing_time = match doc.metadata.complexity_hints.estimated_complexity {
                    ComplexityLevel::Fast => Duration::from_millis(300),
                    ComplexityLevel::Balanced => Duration::from_millis(1000),
                    ComplexityLevel::Quality => Duration::from_millis(2500),
                };
                
                tokio::time::sleep(processing_time).await;
                
                // Simulate QA generation
                let qa_pairs = TestDataFixtures::expected_qa_pairs();
                total_qa_pairs += qa_pairs.len();
                
                // Simulate quality assessment
                let quality_score = qa_pairs.iter()
                    .map(|qa| qa.confidence)
                    .sum::<f64>() / qa_pairs.len() as f64;
                quality_scores.push(quality_score);
                
                total_processing_time += start.elapsed();
            }
            
            let avg_quality = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
            let throughput = (total_qa_pairs as f64) / total_processing_time.as_secs_f64() * 3600.0;
            
            black_box((throughput, avg_quality))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    document_processing_benchmarks,
    document_format_benchmarks,
    qa_generation_benchmarks,
    concurrent_processing_benchmarks,
    memory_allocation_benchmarks,
    ipc_communication_benchmarks,
    quality_assessment_benchmarks,
    e2e_pipeline_benchmarks,
    regression_benchmarks
);

criterion_main!(benches);