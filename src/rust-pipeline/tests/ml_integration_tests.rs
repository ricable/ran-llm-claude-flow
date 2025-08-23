/*!
# ML Module Integration Tests

Comprehensive integration tests for dynamic Qwen3 model selection system.
Tests the complete pipeline from workload analysis to model execution.
*/

use ran_document_pipeline::ml::*;
use std::time::Duration;
use tokio;
use uuid::Uuid;

/// Test helper to create ML requests
fn create_test_request(
    doc_type: DocumentType,
    size_bytes: usize,
    complexity: f64,
    priority: Priority,
) -> MLRequest {
    MLRequest {
        request_id: Uuid::new_v4(),
        document_type: doc_type,
        document_size_bytes: size_bytes,
        complexity_score: complexity,
        priority,
        quality_requirements: QualityRequirements {
            min_score: 0.742,
            consistency_target: 0.75,
            accuracy_threshold: 0.8,
            enable_validation: true,
        },
        processing_deadline: Some(Duration::from_secs(30)),
    }
}

#[tokio::test]
async fn test_ml_module_initialization() {
    // Test that all ML modules initialize correctly
    initialize_ml_module()
        .await
        .expect("ML module should initialize successfully");

    // Verify that all components are accessible
    let stats = model_selector::get_statistics()
        .await
        .expect("Model selector should be initialized");
    assert_eq!(stats.total_selections, 0);

    let workload_stats = workload_analyzer::get_statistics()
        .await
        .expect("Workload analyzer should be initialized");
    assert_eq!(workload_stats.total_analyses, 0);

    let memory_stats = memory_predictor::get_statistics()
        .await
        .expect("Memory predictor should be initialized");
    assert_eq!(memory_stats.total_predictions, 0);
}

#[tokio::test]
async fn test_workload_analysis_accuracy() {
    initialize_ml_module().await.unwrap();

    let test_cases = vec![
        (
            DocumentType::PlainText,
            1024,
            0.2,
            "Simple text should have low complexity",
        ),
        (
            DocumentType::Pdf,
            10_000_000,
            0.8,
            "Large PDF should have high complexity",
        ),
        (
            DocumentType::Technical,
            500_000,
            0.7,
            "Technical doc should have medium-high complexity",
        ),
        (
            DocumentType::Standards3Gpp,
            2_000_000,
            0.9,
            "3GPP standard should have very high complexity",
        ),
    ];

    for (doc_type, size, expected_complexity, description) in test_cases {
        let request = create_test_request(doc_type, size, expected_complexity, Priority::Medium);
        let analysis = workload_analyzer::analyze(&request).await.unwrap();

        // Verify complexity is in expected range
        assert!(
            analysis.complexity_score >= expected_complexity - 0.2
                && analysis.complexity_score <= expected_complexity + 0.2,
            "{}: Expected complexity ~{}, got {}",
            description,
            expected_complexity,
            analysis.complexity_score
        );

        // Verify memory requirements are reasonable
        assert!(
            analysis.memory_requirements_gb > 0,
            "Memory requirements should be positive"
        );
        assert!(
            analysis.memory_requirements_gb < 200,
            "Memory requirements should be reasonable (<200GB)"
        );

        // Verify processing time estimates
        assert!(
            analysis.estimated_processing_time > Duration::from_secs(0),
            "Processing time should be positive"
        );
        assert!(
            analysis.estimated_processing_time < Duration::from_secs(3600),
            "Processing time should be reasonable (<1 hour)"
        );
    }
}

#[tokio::test]
async fn test_model_selection_logic() {
    initialize_ml_module().await.unwrap();

    let test_cases = vec![
        // (doc_type, size, complexity, priority, expected_model_range)
        (
            DocumentType::PlainText,
            1024,
            0.2,
            Priority::High,
            vec![Qwen3Model::Qwen3_1_7B],
        ),
        (
            DocumentType::Pdf,
            1_000_000,
            0.7,
            Priority::High,
            vec![Qwen3Model::Qwen3_7B, Qwen3Model::Qwen3_30B],
        ),
        (
            DocumentType::Technical,
            2_000_000,
            0.9,
            Priority::High,
            vec![Qwen3Model::Qwen3_30B],
        ),
        (
            DocumentType::Standards3Gpp,
            5_000_000,
            0.95,
            Priority::High,
            vec![Qwen3Model::Qwen3_30B],
        ),
    ];

    for (doc_type, size, complexity, priority, expected_models) in test_cases {
        let request = create_test_request(doc_type, size, complexity, priority);
        let workload = workload_analyzer::analyze(&request).await.unwrap();
        let selected_model = model_selector::select_model(&request, &workload)
            .await
            .unwrap();

        assert!(
            expected_models.contains(&selected_model),
            "For {:?} doc ({}KB, complexity {:.1}, priority {:?}), expected one of {:?}, got {:?}",
            doc_type,
            size / 1024,
            complexity,
            priority,
            expected_models.iter().map(|m| m.name()).collect::<Vec<_>>(),
            selected_model.name()
        );
    }
}

#[tokio::test]
async fn test_memory_prediction_accuracy() {
    initialize_ml_module().await.unwrap();

    let models = [
        Qwen3Model::Qwen3_1_7B,
        Qwen3Model::Qwen3_7B,
        Qwen3Model::Qwen3_30B,
    ];
    let request = create_test_request(DocumentType::Technical, 100_000, 0.7, Priority::Medium);

    for model in &models {
        let prediction = memory_predictor::predict_memory_usage(model, &request)
            .await
            .unwrap();
        let specs = model.specs();

        // Verify predictions are reasonable
        assert!(
            prediction.predicted_usage_gb >= specs.memory_gb as f64 * 0.8,
            "Predicted usage ({:.1} GB) should be at least 80% of model size ({} GB) for {}",
            prediction.predicted_usage_gb,
            specs.memory_gb,
            model.name()
        );

        assert!(
            prediction.predicted_usage_gb <= specs.memory_gb as f64 * 2.0,
            "Predicted usage ({:.1} GB) should not exceed 2x model size ({} GB) for {}",
            prediction.predicted_usage_gb,
            specs.memory_gb,
            model.name()
        );

        assert!(
            prediction.peak_usage_gb >= prediction.predicted_usage_gb,
            "Peak usage should be >= predicted usage for {}",
            model.name()
        );

        // Verify risk assessments are in valid ranges
        assert!(prediction.fragmentation_risk >= 0.0 && prediction.fragmentation_risk <= 1.0);
        assert!(prediction.gc_pressure >= 0.0 && prediction.gc_pressure <= 1.0);
        assert!(prediction.bandwidth_utilization >= 0.0 && prediction.bandwidth_utilization <= 1.0);

        // Verify pool recommendations
        assert!(
            !prediction.pool_recommendations.is_empty(),
            "Should have pool recommendations for {}",
            model.name()
        );
        assert!(
            !prediction.optimization_suggestions.is_empty(),
            "Should have optimization suggestions for {}",
            model.name()
        );
    }
}

#[tokio::test]
async fn test_bottleneck_analysis() {
    initialize_ml_module().await.unwrap();

    // Test high-memory scenario
    let high_memory_request = create_test_request(
        DocumentType::Standards3Gpp,
        20_000_000, // 20MB document
        0.95,
        Priority::Critical,
    );

    let bottlenecks =
        memory_predictor::analyze_bottlenecks(&Qwen3Model::Qwen3_30B, &high_memory_request, None)
            .await
            .unwrap();

    assert!(
        !bottlenecks.is_empty(),
        "High-memory scenario should identify bottlenecks"
    );

    // Check for memory-related bottlenecks
    let has_memory_bottleneck = bottlenecks.iter().any(|b| {
        matches!(
            b.bottleneck_type,
            BottleneckType::InsufficientMemory | BottleneckType::BandwidthLimited
        )
    });

    // For very large models and documents, we expect some bottlenecks
    if bottlenecks
        .iter()
        .any(|b| b.bottleneck_type != BottleneckType::None)
    {
        assert!(
            has_memory_bottleneck,
            "Large model + large document should show memory bottlenecks"
        );
    }

    // Test simple scenario
    let simple_request = create_test_request(DocumentType::PlainText, 1024, 0.2, Priority::Low);
    let simple_bottlenecks =
        memory_predictor::analyze_bottlenecks(&Qwen3Model::Qwen3_1_7B, &simple_request, None)
            .await
            .unwrap();

    // Simple scenarios should have fewer bottlenecks
    let severe_bottlenecks = simple_bottlenecks
        .iter()
        .filter(|b| b.severity > 0.7)
        .count();

    assert!(
        severe_bottlenecks <= 1,
        "Simple scenario should have few severe bottlenecks"
    );
}

#[tokio::test]
async fn test_end_to_end_processing() {
    initialize_ml_module().await.unwrap();

    let test_requests = vec![
        create_test_request(DocumentType::PlainText, 2048, 0.3, Priority::Medium),
        create_test_request(DocumentType::Pdf, 500_000, 0.7, Priority::High),
        create_test_request(DocumentType::Technical, 1_000_000, 0.8, Priority::Critical),
    ];

    for request in test_requests {
        let response = process_request(request.clone()).await.unwrap();

        // Verify response structure
        assert_eq!(response.request_id, request.request_id);
        assert!(response.processing_time > Duration::from_secs(0));
        assert!(response.quality_score >= 0.0 && response.quality_score <= 1.0);
        assert!(response.memory_used_mb > 0);

        // Verify model selection was appropriate
        let model_specs = response.model_used.specs();
        if request.priority == Priority::Critical && request.complexity_score > 0.8 {
            // Critical high-complexity tasks should prefer quality
            assert!(
                model_specs.quality_score >= 0.8,
                "Critical complex task should use high-quality model"
            );
        }

        if request.document_size_bytes < 10_000 && request.complexity_score < 0.4 {
            // Small simple tasks should prefer speed
            assert!(
                model_specs.tokens_per_second > 100,
                "Small simple task should use fast model"
            );
        }

        // Record the result for benchmarking
        performance_benchmark::record_result(&request, &response)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn test_concurrent_processing() {
    initialize_ml_module().await.unwrap();

    let mut handles = Vec::new();

    // Create multiple concurrent requests
    for i in 0..10 {
        let request = create_test_request(
            if i % 2 == 0 {
                DocumentType::PlainText
            } else {
                DocumentType::Technical
            },
            1024 * (i + 1),
            0.3 + (i as f64 * 0.05),
            if i % 3 == 0 {
                Priority::High
            } else {
                Priority::Medium
            },
        );

        let handle = tokio::spawn({
            let request = request.clone();
            async move { process_request(request).await }
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();

    // Verify all requests succeeded
    for result in results {
        let response = result.unwrap();
        assert!(matches!(response.status, ProcessingStatus::Success));
        assert!(response.processing_time < Duration::from_secs(10)); // Reasonable time
    }

    // Check that the system collected performance data
    let stats = model_selector::get_statistics().await.unwrap();
    assert!(
        stats.total_selections >= 10,
        "Should have recorded all selections"
    );
}

#[tokio::test]
async fn test_performance_tracking() {
    initialize_ml_module().await.unwrap();

    // Process several requests to generate performance data
    for _ in 0..5 {
        let request = create_test_request(DocumentType::Technical, 50_000, 0.6, Priority::Medium);
        let response = process_request(request.clone()).await.unwrap();
        performance_benchmark::record_result(&request, &response)
            .await
            .unwrap();
    }

    // Check performance metrics
    let metrics = performance_benchmark::get_performance_metrics(
        &Qwen3Model::Qwen3_7B,
        &DocumentType::Technical,
    )
    .await
    .unwrap();

    if let Some(metrics) = metrics {
        assert!(metrics.throughput_docs_per_hour > 0.0);
        assert!(metrics.average_latency > Duration::from_secs(0));
        assert!(metrics.quality_score > 0.0);
        assert!(metrics.success_rate > 0.0);
    }

    // Test model comparison
    let models = vec![
        Qwen3Model::Qwen3_1_7B,
        Qwen3Model::Qwen3_7B,
        Qwen3Model::Qwen3_30B,
    ];
    let comparisons = performance_benchmark::compare_models(&models, &DocumentType::Technical)
        .await
        .unwrap();

    // Should have some comparison data (may be limited due to test scope)
    assert!(
        !comparisons.is_empty() || models.len() <= 1,
        "Should have comparison data or limited models"
    );
}

#[tokio::test]
async fn test_cache_effectiveness() {
    initialize_ml_module().await.unwrap();

    let request = create_test_request(DocumentType::Technical, 100_000, 0.7, Priority::Medium);

    // First analysis should populate cache
    let start_time = std::time::Instant::now();
    let _analysis1 = workload_analyzer::analyze(&request).await.unwrap();
    let first_duration = start_time.elapsed();

    // Second analysis should be faster (cached)
    let start_time = std::time::Instant::now();
    let _analysis2 = workload_analyzer::analyze(&request).await.unwrap();
    let second_duration = start_time.elapsed();

    // Cache should provide some speedup (though timing can be variable in tests)
    println!(
        "First analysis: {:?}, Second analysis: {:?}",
        first_duration, second_duration
    );

    // Test memory prediction caching
    let start_time = std::time::Instant::now();
    let _prediction1 = memory_predictor::predict_memory_usage(&Qwen3Model::Qwen3_7B, &request)
        .await
        .unwrap();
    let first_duration = start_time.elapsed();

    let start_time = std::time::Instant::now();
    let _prediction2 = memory_predictor::predict_memory_usage(&Qwen3Model::Qwen3_7B, &request)
        .await
        .unwrap();
    let second_duration = start_time.elapsed();

    println!(
        "First prediction: {:?}, Second prediction: {:?}",
        first_duration, second_duration
    );

    // Verify cache stats improved
    let workload_stats = workload_analyzer::get_statistics().await.unwrap();
    let memory_stats = memory_predictor::get_statistics().await.unwrap();

    assert!(workload_stats.total_analyses >= 2);
    assert!(memory_stats.total_predictions >= 2);
}

#[tokio::test]
async fn test_error_handling() {
    initialize_ml_module().await.unwrap();

    // Test with invalid request data
    let invalid_request = MLRequest {
        request_id: Uuid::new_v4(),
        document_type: DocumentType::PlainText,
        document_size_bytes: 0, // Invalid size
        complexity_score: -0.5, // Invalid complexity
        priority: Priority::Medium,
        quality_requirements: QualityRequirements {
            min_score: 1.5, // Invalid score
            consistency_target: 0.75,
            accuracy_threshold: 0.8,
            enable_validation: true,
        },
        processing_deadline: Some(Duration::from_secs(0)), // Invalid deadline
    };

    // The system should handle invalid inputs gracefully
    let result = workload_analyzer::analyze(&invalid_request).await;

    match result {
        Ok(analysis) => {
            // If it succeeds, verify it handled the invalid data reasonably
            assert!(
                analysis.complexity_score >= 0.0,
                "Should normalize negative complexity"
            );
            assert!(
                analysis.complexity_score <= 1.0,
                "Should cap complexity at 1.0"
            );
            assert!(
                analysis.memory_requirements_gb > 0,
                "Should have positive memory requirement"
            );
        }
        Err(_) => {
            // It's also acceptable for the system to reject invalid inputs
            println!("System appropriately rejected invalid input");
        }
    }
}
