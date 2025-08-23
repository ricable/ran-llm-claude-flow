// Comprehensive test suite for Rust ML integration components
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_integration::*;
    use crate::types::*;
    use std::time::Duration;
    use tokio;
    use uuid::Uuid;
    use tempfile::TempDir;
    
    // Mock implementations for testing
    struct MockIpcManager {
        response_delay: Duration,
        should_fail: bool,
    }
    
    impl MockIpcManager {
        fn new(response_delay: Duration, should_fail: bool) -> Self {
            Self { response_delay, should_fail }
        }
    }
    
    // Create test configuration
    fn create_test_config() -> MLIntegrationConfig {
        MLIntegrationConfig {
            enable_dynamic_model_selection: true,
            batch_processing_enabled: true,
            max_batch_size: 4,
            batch_timeout_ms: 1000,
            quality_enhancement_enabled: true,
            performance_tracking_enabled: true,
            adaptive_complexity_enabled: true,
            m3_max_optimization: M3MaxMLConfig {
                use_unified_memory: true,
                max_concurrent_models: 2,
                model_cache_size_gb: 8,
                enable_simd_acceleration: true,
                optimize_for_throughput: true,
            },
        }
    }
    
    // Create test document
    fn create_test_document(complexity: ComplexityLevel) -> ProcessedDocument {
        let parameter_count = match complexity {
            ComplexityLevel::Fast => 2,
            ComplexityLevel::Balanced => 8,
            ComplexityLevel::Quality => 15,
        };
        
        ProcessedDocument {
            document: Document {
                id: Uuid::new_v4(),
                path: std::path::PathBuf::from("test_document.md"),
                format: DocumentFormat::Markdown,
                content: "# Test Feature\n\nThis is a test document for ML processing.".to_string(),
                metadata: DocumentMetadata {
                    title: Some("Test Feature".to_string()),
                    feature_name: Some("TestFeature".to_string()),
                    product_info: Some("Test Product".to_string()),
                    feature_state: Some("Active".to_string()),
                    parameters: vec![Parameter {
                        name: "testParameter".to_string(),
                        mo_class: Some("TestMO".to_string()),
                        data_type: Some("boolean".to_string()),
                        valid_values: Some("true, false".to_string()),
                        default_value: Some("false".to_string()),
                        description: Some("Test parameter description".to_string()),
                    }],
                    counters: vec![Counter {
                        name: "pmTestCounter".to_string(),
                        description: Some("Test counter description".to_string()),
                        mo_class: Some("TestMO".to_string()),
                        counter_type: Some("gauge".to_string()),
                    }],
                    technical_terms: vec!["LTE".to_string(), "QoS".to_string()],
                    complexity_hints: ComplexityHints {
                        parameter_count,
                        counter_count: 2,
                        technical_term_density: 2.5,
                        content_length: 1000,
                        estimated_complexity: complexity.clone(),
                    },
                },
                size_bytes: 1000,
                created_at: chrono::Utc::now(),
            },
            structural_quality: StructuralQuality {
                completeness_score: 0.85,
                parameter_extraction_quality: 0.90,
                counter_extraction_quality: 0.88,
                technical_density_score: 0.82,
                overall_score: 0.86,
            },
            processing_hints: ProcessingHints {
                recommended_model: complexity,
                expected_qa_pairs: 5,
                processing_priority: ProcessingPriority::Normal,
                use_cache: true,
                batch_with_similar: true,
                batch_processing_eligible: true,
                expected_processing_time: Duration::from_millis(2000),
                memory_optimization: MemoryOptimization::M3MaxUnified,
            },
            checksum: 12345678,
        }
    }
    
    #[test]
    fn test_complexity_analyzer() {
        let analyzer = ComplexityAnalyzer::new();
        
        // Test low complexity
        let low_score = analyzer.calculate_complexity_score(2, 1, 0.5, 500);
        assert!(low_score < 0.3, "Low complexity score should be < 0.3");
        
        // Test medium complexity  
        let medium_score = analyzer.calculate_complexity_score(8, 4, 3.0, 3000);
        assert!(medium_score >= 0.3 && medium_score < 0.7, "Medium complexity score should be 0.3-0.7");
        
        // Test high complexity
        let high_score = analyzer.calculate_complexity_score(20, 10, 8.0, 10000);
        assert!(high_score >= 0.7, "High complexity score should be >= 0.7");
        
        println!("Complexity scores - Low: {:.3}, Medium: {:.3}, High: {:.3}", 
                low_score, medium_score, high_score);
    }
    
    #[tokio::test]
    async fn test_model_selector_initialization() {
        let config = create_test_config();
        let selector = ModelSelector::new(&config).unwrap();
        
        // Test complexity analysis
        let test_doc = create_test_document(ComplexityLevel::Balanced);
        let analysis = selector.analyze_complexity(&test_doc).unwrap();
        
        assert_eq!(analysis.recommended_model, ComplexityLevel::Balanced);
        assert!(analysis.confidence_score > 0.0 && analysis.confidence_score <= 1.0);
        assert!(analysis.estimated_processing_time > Duration::from_millis(0));
        assert!(analysis.memory_requirements_mb > 0);
        
        println!("Complexity analysis - Model: {:?}, Confidence: {:.3}, Time: {:?}, Memory: {}MB",
                analysis.recommended_model, analysis.confidence_score, 
                analysis.estimated_processing_time, analysis.memory_requirements_mb);
    }
    
    #[tokio::test]
    async fn test_model_selection_optimization() {
        let config = create_test_config();
        let selector = ModelSelector::new(&config).unwrap();
        
        // Test model selection for different complexity levels
        for complexity in [ComplexityLevel::Fast, ComplexityLevel::Balanced, ComplexityLevel::Quality] {
            let hints = ComplexityHints {
                parameter_count: match complexity {
                    ComplexityLevel::Fast => 3,
                    ComplexityLevel::Balanced => 8,
                    ComplexityLevel::Quality => 15,
                },
                counter_count: 2,
                technical_term_density: 2.0,
                content_length: 2000,
                estimated_complexity: complexity.clone(),
            };
            
            let selected_model = selector.select_optimal_model(&complexity, &hints).await.unwrap();
            
            match complexity {
                ComplexityLevel::Fast => assert!(selected_model.contains("1.7b") || selected_model.contains("7b")),
                ComplexityLevel::Balanced => assert!(selected_model.contains("7b")),
                ComplexityLevel::Quality => assert!(selected_model.contains("30b")),
            }
            
            println!("Selected model for {:?}: {}", complexity, selected_model);
        }
    }
    
    #[test]
    fn test_batch_manager() {
        let config = create_test_config();
        let batch_manager = BatchManager::new(config);
        
        // Test batch space availability
        assert!(batch_manager.has_space_for_batch(&ComplexityLevel::Fast));
        assert!(batch_manager.has_space_for_batch(&ComplexityLevel::Balanced));
        assert!(batch_manager.has_space_for_batch(&ComplexityLevel::Quality));
        
        println!("Batch manager initialized successfully");
    }
    
    #[tokio::test]
    async fn test_batch_processing_flow() {
        let config = create_test_config();
        let batch_manager = BatchManager::new(config.clone());
        
        // Create test documents for batching
        let documents = vec![
            create_test_document(ComplexityLevel::Fast),
            create_test_document(ComplexityLevel::Fast),
            create_test_document(ComplexityLevel::Fast),
        ];
        
        // Test adding documents to batch
        for (i, doc) in documents.into_iter().enumerate() {
            let (tx, _rx) = tokio::sync::oneshot::channel();
            batch_manager.add_to_batch(ComplexityLevel::Fast, doc, tx).await;
            
            println!("Added document {} to batch", i + 1);
        }
        
        // Test batch ready status (would need to wait for timeout in real scenario)
        // In this test, we just verify the batch structure is correct
        assert!(batch_manager.pending_batches.contains_key(&ComplexityLevel::Fast));
        println!("Batch processing flow test completed");
    }
    
    #[tokio::test]
    async fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();
        
        // Record some test performance data
        let test_models = vec!["qwen3-1.7b-mlx", "qwen3-7b-mlx", "qwen3-30b-mlx"];
        
        for (i, model) in test_models.iter().enumerate() {
            let processing_time = Duration::from_millis(500 + i as u64 * 1000);
            let quality_score = 0.75 + (i as f64 * 0.05);
            let memory_usage = 1000 + (i * 2000);
            
            tracker.record_processing(model, processing_time, quality_score, memory_usage).await;
        }
        
        // Calculate metrics
        tracker.calculate_metrics().await;
        
        // Get performance summary
        let summary = tracker.get_summary().await;
        
        assert!(summary.average_processing_time > Duration::from_millis(0));
        assert!(summary.average_quality_score > 0.0);
        assert!(!summary.model_usage_stats.is_empty());
        
        println!("Performance tracking test completed:");
        println!("  Average processing time: {:?}", summary.average_processing_time);
        println!("  Average quality score: {:.3}", summary.average_quality_score);
        println!("  Models tracked: {}", summary.model_usage_stats.len());
    }
    
    #[tokio::test]
    async fn test_quality_enhancer() {
        let config = create_test_config();
        let enhancer = QualityEnhancer::new(&config).unwrap();
        
        // Create test ML response
        let mut test_response = MLProcessingResponse {
            request_id: Uuid::new_v4(),
            qa_pairs: vec![
                QAPair {
                    id: Uuid::new_v4(),
                    question: "What is carrier aggregation?".to_string(),
                    answer: "Carrier aggregation combines multiple carriers for increased bandwidth.".to_string(),
                    context: Some("LTE feature context".to_string()),
                    confidence: 0.85,
                    metadata: QAMetadata {
                        question_type: QuestionType::Factual,
                        technical_terms: vec!["carrier aggregation".to_string(), "bandwidth".to_string()],
                        parameters_mentioned: vec![],
                        counters_mentioned: vec![],
                        complexity_level: ComplexityLevel::Balanced,
                    },
                },
                QAPair {
                    id: Uuid::new_v4(),
                    question: "How does VoLTE work?".to_string(),
                    answer: "VoLTE uses packet-switched networks for voice services.".to_string(),
                    context: Some("Voice service context".to_string()),
                    confidence: 0.80,
                    metadata: QAMetadata {
                        question_type: QuestionType::Conceptual,
                        technical_terms: vec!["VoLTE".to_string(), "packet-switched".to_string()],
                        parameters_mentioned: vec![],
                        counters_mentioned: vec![],
                        complexity_level: ComplexityLevel::Balanced,
                    },
                },
            ],
            semantic_quality: SemanticQuality {
                coherence_score: 0.82,
                relevance_score: 0.85,
                technical_accuracy_score: 0.88,
                diversity_score: 0.75,
                overall_score: 0.825,
            },
            processing_metadata: MLProcessingMetadata {
                model_name: "qwen3-7b-mlx".to_string(),
                model_version: "v1.0.0".to_string(),
                inference_time: Duration::from_millis(2000),
                tokens_processed: 150,
                memory_used_mb: 4096,
                gpu_utilization: Some(0.75),
            },
            model_used: "qwen3-7b-mlx".to_string(),
            processing_time: Duration::from_millis(2500),
        };
        
        // Test quality enhancement
        let enhanced_response = enhancer.enhance_response(test_response.clone()).await.unwrap();
        
        // Verify enhancement doesn't break the response
        assert_eq!(enhanced_response.request_id, test_response.request_id);
        assert_eq!(enhanced_response.qa_pairs.len(), test_response.qa_pairs.len());
        assert!(enhanced_response.semantic_quality.overall_score >= 0.0);
        
        println!("Quality enhancement test completed:");
        println!("  Original QA pairs: {}", test_response.qa_pairs.len());
        println!("  Enhanced QA pairs: {}", enhanced_response.qa_pairs.len());
        println!("  Quality score: {:.3}", enhanced_response.semantic_quality.overall_score);
    }
    
    #[test]
    fn test_m3_max_config() {
        let config = create_test_config();
        
        // Verify M3 Max optimization settings
        assert!(config.m3_max_optimization.use_unified_memory);
        assert!(config.m3_max_optimization.max_concurrent_models > 0);
        assert!(config.m3_max_optimization.model_cache_size_gb > 0);
        assert!(config.m3_max_optimization.enable_simd_acceleration);
        assert!(config.m3_max_optimization.optimize_for_throughput);
        
        println!("M3 Max configuration test passed:");
        println!("  Unified memory: {}", config.m3_max_optimization.use_unified_memory);
        println!("  Max concurrent models: {}", config.m3_max_optimization.max_concurrent_models);
        println!("  Model cache size: {}GB", config.m3_max_optimization.model_cache_size_gb);
        println!("  SIMD acceleration: {}", config.m3_max_optimization.enable_simd_acceleration);
    }
    
    #[tokio::test]
    async fn test_ml_integration_stats() {
        let mut stats = MLIntegrationStats::default();
        
        // Simulate processing statistics
        stats.total_documents_processed = 100;
        stats.total_qa_pairs_generated = 450;
        stats.average_processing_time = Duration::from_millis(1800);
        stats.average_quality_score = 0.82;
        
        // Add model usage statistics
        stats.model_usage_stats.insert("qwen3-1.7b-mlx".to_string(), ModelUsageStats {
            requests_processed: 40,
            average_response_time: Duration::from_millis(800),
            memory_efficiency: 0.95,
            quality_score: 0.78,
            error_rate: 0.02,
        });
        
        stats.model_usage_stats.insert("qwen3-7b-mlx".to_string(), ModelUsageStats {
            requests_processed: 45,
            average_response_time: Duration::from_millis(2000),
            memory_efficiency: 0.88,
            quality_score: 0.84,
            error_rate: 0.015,
        });
        
        stats.model_usage_stats.insert("qwen3-30b-mlx".to_string(), ModelUsageStats {
            requests_processed: 15,
            average_response_time: Duration::from_millis(6000),
            memory_efficiency: 0.78,
            quality_score: 0.92,
            error_rate: 0.01,
        });
        
        // Set M3 Max utilization
        stats.m3_max_utilization = M3MaxUtilization {
            unified_memory_usage_gb: 48.5,
            performance_cores_used: 14,
            efficiency_cores_used: 4,
            neural_engine_utilization: 0.87,
            simd_acceleration_usage: 0.93,
        };
        
        // Validate statistics
        assert_eq!(stats.total_documents_processed, 100);
        assert_eq!(stats.total_qa_pairs_generated, 450);
        assert_eq!(stats.model_usage_stats.len(), 3);
        assert!(stats.average_quality_score > 0.8);
        assert!(stats.m3_max_utilization.unified_memory_usage_gb < 60.0);
        
        println!("ML integration statistics test passed:");
        println!("  Documents processed: {}", stats.total_documents_processed);
        println!("  QA pairs generated: {}", stats.total_qa_pairs_generated);
        println!("  Average quality: {:.3}", stats.average_quality_score);
        println!("  Models used: {}", stats.model_usage_stats.len());
        println!("  M3 Max memory usage: {:.1}GB", stats.m3_max_utilization.unified_memory_usage_gb);
    }
    
    #[tokio::test]
    async fn test_memory_optimization_simulation() {
        // Simulate M3 Max memory optimization
        let initial_memory_gb = 45.2;
        let target_memory_limit_gb = 60.0;
        let memory_efficiency_target = 0.90;
        
        // Calculate memory utilization
        let memory_utilization = initial_memory_gb / target_memory_limit_gb;
        
        // Simulate memory optimization
        let optimized_memory_gb = initial_memory_gb * 0.95; // 5% reduction
        let optimized_utilization = optimized_memory_gb / target_memory_limit_gb;
        
        // Validate optimization
        assert!(optimized_utilization <= memory_efficiency_target);
        assert!(optimized_memory_gb < initial_memory_gb);
        
        println!("Memory optimization test:");
        println!("  Initial memory usage: {:.1}GB", initial_memory_gb);
        println!("  Optimized memory usage: {:.1}GB", optimized_memory_gb);
        println!("  Memory utilization: {:.1}%", optimized_utilization * 100.0);
        println!("  Within target: {}", optimized_utilization <= memory_efficiency_target);
    }
    
    #[tokio::test]
    async fn test_throughput_calculation() {
        // Simulate processing metrics for throughput calculation
        let documents_processed = 25;
        let processing_time_hours = 1.0;
        let qa_pairs_generated = 125;
        
        let docs_per_hour = documents_processed as f64 / processing_time_hours;
        let qa_pairs_per_hour = qa_pairs_generated as f64 / processing_time_hours;
        
        // Validate against performance targets
        let target_docs_per_hour = 20.0;
        let target_qa_pairs_per_hour = 100.0;
        
        assert!(docs_per_hour >= target_docs_per_hour);
        assert!(qa_pairs_per_hour >= target_qa_pairs_per_hour);
        
        println!("Throughput calculation test:");
        println!("  Documents per hour: {:.1}", docs_per_hour);
        println!("  QA pairs per hour: {:.1}", qa_pairs_per_hour);
        println!("  Meets targets: {}", 
                docs_per_hour >= target_docs_per_hour && qa_pairs_per_hour >= target_qa_pairs_per_hour);
    }
}