/*!
# Model Selection Performance Benchmarks

Benchmarks for dynamic Qwen3 model selection system performance and accuracy.
*/

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ran_document_pipeline::ml::*;
use std::time::Duration;
use tokio::runtime::Runtime;
use uuid::Uuid;

fn create_test_request(doc_type: DocumentType, size_bytes: usize, complexity: f64) -> MLRequest {
    MLRequest {
        request_id: Uuid::new_v4(),
        document_type: doc_type,
        document_size_bytes: size_bytes,
        complexity_score: complexity,
        priority: Priority::Medium,
        quality_requirements: QualityRequirements {
            min_score: 0.742,
            consistency_target: 0.75,
            accuracy_threshold: 0.8,
            enable_validation: true,
        },
        processing_deadline: Some(Duration::from_secs(30)),
    }
}

fn bench_model_selection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Initialize the ML system
    rt.block_on(async {
        model_selector::initialize().await.unwrap();
        workload_analyzer::initialize().await.unwrap();
        memory_predictor::initialize().await.unwrap();
        performance_benchmark::initialize().await.unwrap();
    });

    let mut group = c.benchmark_group("model_selection");

    // Test different document types
    let test_cases = vec![
        ("plain_text_small", DocumentType::PlainText, 1024, 0.2),
        (
            "plain_text_large",
            DocumentType::PlainText,
            1024 * 1024,
            0.3,
        ),
        ("pdf_medium", DocumentType::Pdf, 100 * 1024, 0.7),
        ("pdf_large", DocumentType::Pdf, 10 * 1024 * 1024, 0.9),
        (
            "technical_complex",
            DocumentType::Technical,
            500 * 1024,
            0.8,
        ),
        (
            "standards_3gpp",
            DocumentType::Standards3Gpp,
            2 * 1024 * 1024,
            0.9,
        ),
    ];

    for (name, doc_type, size, complexity) in test_cases {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("workload_analysis", name),
            &(doc_type, size, complexity),
            |b, (doc_type, size, complexity)| {
                b.iter_async(&rt, 1, |b, _| {
                    let doc_type = *doc_type;
                    let size = *size;
                    let complexity = *complexity;
                    async move {
                        let request = create_test_request(doc_type, size, complexity);
                        let workload = workload_analyzer::analyze(black_box(&request))
                            .await
                            .unwrap();
                        black_box(workload)
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("model_selection", name),
            &(doc_type, size, complexity),
            |b, (doc_type, size, complexity)| {
                b.iter_async(&rt, 1, |b, _| {
                    let doc_type = *doc_type;
                    let size = *size;
                    let complexity = *complexity;
                    async move {
                        let request = create_test_request(doc_type, size, complexity);
                        let workload = workload_analyzer::analyze(&request).await.unwrap();
                        let selected_model =
                            model_selector::select_model(black_box(&request), black_box(&workload))
                                .await
                                .unwrap();
                        black_box(selected_model)
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_prediction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        memory_predictor::initialize().await.unwrap();
    });

    let mut group = c.benchmark_group("memory_prediction");

    let models = [
        Qwen3Model::Qwen3_1_7B,
        Qwen3Model::Qwen3_7B,
        Qwen3Model::Qwen3_30B,
    ];
    let test_cases = vec![
        ("small_doc", DocumentType::PlainText, 1024, 0.2),
        ("large_pdf", DocumentType::Pdf, 10 * 1024 * 1024, 0.8),
        ("complex_tech", DocumentType::Technical, 1024 * 1024, 0.9),
    ];

    for model in &models {
        for (case_name, doc_type, size, complexity) in &test_cases {
            let bench_name = format!("{}_{}", model.name(), case_name);

            group.bench_with_input(
                BenchmarkId::new("memory_prediction", bench_name),
                &(*model, *doc_type, *size, *complexity),
                |b, (model, doc_type, size, complexity)| {
                    b.iter_async(&rt, 1, |b, _| {
                        let model = *model;
                        let doc_type = *doc_type;
                        let size = *size;
                        let complexity = *complexity;
                        async move {
                            let request = create_test_request(doc_type, size, complexity);
                            let prediction = memory_predictor::predict_memory_usage(
                                black_box(&model),
                                black_box(&request),
                            )
                            .await
                            .unwrap();
                            black_box(prediction)
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_end_to_end_selection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        initialize_ml_module().await.unwrap();
    });

    let mut group = c.benchmark_group("end_to_end_selection");
    group.throughput(Throughput::Elements(1));

    let test_requests = vec![
        create_test_request(DocumentType::PlainText, 2048, 0.3),
        create_test_request(DocumentType::Pdf, 1024 * 1024, 0.7),
        create_test_request(DocumentType::Technical, 500 * 1024, 0.8),
        create_test_request(DocumentType::Standards3Gpp, 2 * 1024 * 1024, 0.9),
    ];

    group.bench_function("complete_pipeline", |b| {
        b.iter_async(&rt, 1, |b, _| async {
            for request in &test_requests {
                let response = process_request(black_box(request.clone())).await.unwrap();
                black_box(response);
            }
        });
    });

    group.finish();
}

fn bench_model_switching_overhead(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        performance_benchmark::initialize().await.unwrap();
    });

    let mut group = c.benchmark_group("model_switching");

    let switching_scenarios = vec![
        ("1.7b_to_7b", Qwen3Model::Qwen3_1_7B, Qwen3Model::Qwen3_7B),
        ("7b_to_30b", Qwen3Model::Qwen3_7B, Qwen3Model::Qwen3_30B),
        ("30b_to_1.7b", Qwen3Model::Qwen3_30B, Qwen3Model::Qwen3_1_7B),
    ];

    for (name, from_model, to_model) in switching_scenarios {
        group.bench_with_input(
            BenchmarkId::new("switch_overhead", name),
            &(from_model, to_model),
            |b, (from_model, to_model)| {
                b.iter_async(&rt, 1, |b, _| {
                    let from_model = *from_model;
                    let to_model = *to_model;
                    async move {
                        performance_benchmark::record_model_switch(
                            Some(black_box(from_model)),
                            black_box(to_model),
                            "benchmark_switch".to_string(),
                        )
                        .await
                        .unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_selection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        initialize_ml_module().await.unwrap();
    });

    let mut group = c.benchmark_group("concurrent_selection");

    let concurrent_levels = [1, 2, 4, 8, 16];

    for &concurrency in &concurrent_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter_async(&rt, 1, |b, _| async move {
                    let mut handles = Vec::new();

                    for i in 0..concurrency {
                        let request = create_test_request(
                            if i % 2 == 0 {
                                DocumentType::PlainText
                            } else {
                                DocumentType::Technical
                            },
                            1024 * (i + 1),
                            0.5 + (i as f64 * 0.1),
                        );

                        let handle =
                            tokio::spawn(async move { process_request(request).await.unwrap() });

                        handles.push(handle);
                    }

                    let results: Vec<_> = futures::future::join_all(handles)
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, tokio::task::JoinError>>()
                        .unwrap();

                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

fn bench_cache_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        workload_analyzer::initialize().await.unwrap();
        memory_predictor::initialize().await.unwrap();
    });

    let mut group = c.benchmark_group("cache_performance");

    let test_request = create_test_request(DocumentType::Technical, 100 * 1024, 0.7);

    // Warm up the cache
    rt.block_on(async {
        for _ in 0..5 {
            let _ = workload_analyzer::analyze(&test_request).await;
            let _ =
                memory_predictor::predict_memory_usage(&Qwen3Model::Qwen3_7B, &test_request).await;
        }
    });

    group.bench_function("workload_analysis_cached", |b| {
        b.iter_async(&rt, 1, |b, _| async {
            let analysis = workload_analyzer::analyze(black_box(&test_request))
                .await
                .unwrap();
            black_box(analysis)
        });
    });

    group.bench_function("memory_prediction_cached", |b| {
        b.iter_async(&rt, 1, |b, _| async {
            let prediction = memory_predictor::predict_memory_usage(
                black_box(&Qwen3Model::Qwen3_7B),
                black_box(&test_request),
            )
            .await
            .unwrap();
            black_box(prediction)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_model_selection,
    bench_memory_prediction,
    bench_end_to_end_selection,
    bench_model_switching_overhead,
    bench_concurrent_selection,
    bench_cache_performance
);
criterion_main!(benches);
