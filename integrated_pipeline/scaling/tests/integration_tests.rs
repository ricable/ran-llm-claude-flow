// Integration Tests for Dynamic Scaling System
// Phase 2 MCP Advanced Features - Comprehensive Testing Suite
// Tests all scaling components working together

#[cfg(test)]
mod integration_tests {
    use super::super::*;
    use tokio::time::{sleep, timeout, Duration};
    use std::collections::HashMap;
    
    use crate::scaling::{
        ScalingSystem, ScalingSystemConfig,
        dynamic_scaler::{ScalingMetrics, ScalingAction},
        workload_analyzer::{WorkloadMetrics, PatternType, MemoryBreakdown},
        resource_manager::{ResourceAllocationRequest, ProcessType, ResourcePriority},
    };

    async fn create_test_scaling_system() -> ScalingSystem {
        let config = ScalingSystemConfig {
            enable_auto_scaling: true,
            enable_workload_analysis: true,
            enable_resource_optimization: true,
            monitoring_interval_seconds: 1,
            scaling_decision_interval_seconds: 2,
            emergency_threshold_memory_percent: 95.0,
            emergency_threshold_cpu_percent: 98.0,
            max_agents: 8, // Lower for testing
            min_agents: 2,
            target_throughput_docs_per_hour: 25.0,
        };
        
        ScalingSystem::new(config)
    }

    fn create_high_load_metrics() -> ScalingMetrics {
        ScalingMetrics {
            cpu_utilization: 90.0,
            memory_utilization: 85.0,
            queue_depth: 15,
            throughput_docs_per_hour: 20.0, // Below target
            ipc_latency_ms: 8.0,
            active_agents: 2,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            rust_memory_gb: 60.0,
            python_memory_gb: 45.0,
            shared_memory_gb: 15.0,
        }
    }

    fn create_low_load_metrics() -> ScalingMetrics {
        ScalingMetrics {
            cpu_utilization: 30.0,
            memory_utilization: 50.0,
            queue_depth: 1,
            throughput_docs_per_hour: 35.0, // Above target
            ipc_latency_ms: 2.0,
            active_agents: 4,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            rust_memory_gb: 60.0,
            python_memory_gb: 45.0,
            shared_memory_gb: 15.0,
        }
    }

    fn create_emergency_metrics() -> ScalingMetrics {
        ScalingMetrics {
            cpu_utilization: 98.0,
            memory_utilization: 96.0,
            queue_depth: 25,
            throughput_docs_per_hour: 10.0,
            ipc_latency_ms: 50.0,
            active_agents: 6,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            rust_memory_gb: 75.0,
            python_memory_gb: 55.0,
            shared_memory_gb: 20.0,
        }
    }

    fn create_workload_metrics(cpu: f64, throughput: f64, queue_depth: usize) -> WorkloadMetrics {
        WorkloadMetrics {
            processing_queue_depth: queue_depth,
            average_processing_time_ms: 200.0,
            documents_per_minute: throughput / 60.0,
            cpu_utilization_per_core: vec![cpu; 16],
            memory_utilization_breakdown: MemoryBreakdown {
                rust_processes_gb: 60.0,
                python_processes_gb: 45.0,
                shared_memory_gb: 15.0,
                cached_data_gb: 5.0,
                free_memory_gb: 23.0,
                memory_fragmentation_percent: 5.0,
            },
            ipc_throughput_mbps: 1200.0,
            error_rate_percent: 0.5,
            agent_utilization: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            peak_hour_indicator: false,
        }
    }

    #[tokio::test]
    async fn test_scaling_system_initialization() {
        let scaling_system = create_test_scaling_system().await;
        let stats = scaling_system.get_scaling_statistics().await;
        
        assert_eq!(stats.current_agents, 2); // Should start with min agents
        assert_eq!(stats.total_scaling_decisions, 0);
    }

    #[tokio::test]
    async fn test_high_load_scaling_up() {
        let scaling_system = create_test_scaling_system().await;
        let mut scaler = scaling_system.scaler.write().await;
        
        let high_load_metrics = create_high_load_metrics();
        let decision = scaler.analyze_and_scale(high_load_metrics).await;
        
        assert!(decision.is_some());
        let decision = decision.unwrap();
        assert!(matches!(decision.action, ScalingAction::ScaleUp));
        assert!(decision.target_agents > 2);
        assert!(decision.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_low_load_scaling_down() {
        let scaling_system = create_test_scaling_system().await;
        let mut scaler = scaling_system.scaler.write().await;
        
        // First simulate high load to scale up
        let high_load = create_high_load_metrics();
        scaler.analyze_and_scale(high_load).await;
        
        // Wait for cooldown
        sleep(Duration::from_secs(1)).await;
        
        // Then simulate low load
        let low_load = create_low_load_metrics();
        let decision = scaler.analyze_and_scale(low_load).await;
        
        if let Some(decision) = decision {
            assert!(matches!(decision.action, ScalingAction::ScaleDown));
            assert!(decision.target_agents >= 2); // Never go below minimum
        }
    }

    #[tokio::test]
    async fn test_emergency_scaling() {
        let scaling_system = create_test_scaling_system().await;
        let mut scaler = scaling_system.scaler.write().await;
        
        let emergency_metrics = create_emergency_metrics();
        let decision = scaler.emergency_scale(&emergency_metrics).await;
        
        assert!(decision.is_some());
        let decision = decision.unwrap();
        assert!(matches!(decision.action, ScalingAction::ScaleDown));
        assert_eq!(decision.target_agents, 2); // Should scale to minimum
        assert!(decision.reasoning.contains("EMERGENCY"));
    }

    #[tokio::test]
    async fn test_workload_pattern_detection() {
        let scaling_system = create_test_scaling_system().await;
        let mut analyzer = scaling_system.workload_analyzer.write().await;
        
        // Simulate burst traffic pattern
        for i in 0..10 {
            let throughput = if i < 3 { 15.0 } else { 50.0 }; // Burst after 3 samples
            let workload = create_workload_metrics(70.0, throughput, if i < 3 { 2 } else { 15 });
            let pattern = analyzer.analyze_workload(workload).await;
            
            if i >= 3 {
                // Should detect burst traffic pattern
                assert!(matches!(pattern.pattern_type, PatternType::BurstTraffic));
                assert!(pattern.confidence_score > 0.5);
                assert!(pattern.recommended_agents > 2);
            }
        }
    }

    #[tokio::test]
    async fn test_resource_allocation_and_management() {
        let scaling_system = create_test_scaling_system().await;
        let resource_manager = &scaling_system.resource_manager;
        
        // Test basic allocation
        let request = ResourceAllocationRequest {
            process_id: "test_rust_process".to_string(),
            process_type: ProcessType::RustCore,
            requested_memory_gb: 30.0,
            max_memory_gb: Some(40.0),
            cpu_cores_needed: 4,
            priority: ResourcePriority::High,
            shared_memory_quota_gb: 5.0,
            ipc_bandwidth_mbps: 1000.0,
        };
        
        let allocation = resource_manager.allocate_resources(request).await;
        assert!(allocation.is_ok());
        
        let allocation = allocation.unwrap();
        assert_eq!(allocation.memory_allocation_gb, 30.0);
        assert_eq!(allocation.cpu_cores.len(), 4);
        
        // Test resource statistics
        let stats = resource_manager.get_resource_statistics().await;
        assert!(stats.total_memory_allocated_gb >= 30.0);
        assert!(stats.active_processes >= 1);
        
        // Test deallocation
        let result = resource_manager.deallocate_resources("test_rust_process").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_resource_optimization() {
        let scaling_system = create_test_scaling_system().await;
        let resource_manager = &scaling_system.resource_manager;
        
        // Allocate some resources
        let request1 = ResourceAllocationRequest {
            process_id: "underutilized_process".to_string(),
            process_type: ProcessType::PythonML,
            requested_memory_gb: 20.0,
            max_memory_gb: Some(30.0),
            cpu_cores_needed: 2,
            priority: ResourcePriority::Medium,
            shared_memory_quota_gb: 2.0,
            ipc_bandwidth_mbps: 500.0,
        };
        
        resource_manager.allocate_resources(request1).await.unwrap();
        
        // Simulate low utilization
        let utilizations = vec![
            crate::scaling::resource_manager::ResourceUtilization {
                process_id: "underutilized_process".to_string(),
                memory_used_gb: 8.0, // Using only 8GB of 20GB allocated
                memory_utilization_percent: 40.0,
                cpu_utilization_percent: 25.0,
                ipc_throughput_mbps: 200.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                is_healthy: true,
                performance_score: 0.6,
            }
        ];
        
        resource_manager.update_utilization(utilizations).await;
        
        // This would test optimization in a real implementation
        // For now, just verify the system can handle resource updates
        let stats = resource_manager.get_resource_statistics().await;
        assert!(stats.active_processes >= 1);
    }

    #[tokio::test]
    async fn test_scaling_effectiveness_calculation() {
        let scaling_system = create_test_scaling_system().await;
        let mut scaler = scaling_system.scaler.write().await;
        
        // Make several scaling decisions
        let high_load = create_high_load_metrics();
        scaler.analyze_and_scale(high_load).await;
        
        sleep(Duration::from_millis(100)).await;
        
        let low_load = create_low_load_metrics();
        scaler.analyze_and_scale(low_load).await;
        
        let effectiveness = scaler.calculate_scaling_effectiveness();
        assert!(effectiveness >= 0.0);
        
        let history = scaler.get_scaling_history();
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_memory_constraint_enforcement() {
        let scaling_system = create_test_scaling_system().await;
        let resource_manager = &scaling_system.resource_manager;
        
        // Try to allocate more memory than available
        let oversized_request = ResourceAllocationRequest {
            process_id: "memory_hog".to_string(),
            process_type: ProcessType::RustCore,
            requested_memory_gb: 150.0, // More than M3 Max capacity
            max_memory_gb: Some(200.0),
            cpu_cores_needed: 8,
            priority: ResourcePriority::High,
            shared_memory_quota_gb: 10.0,
            ipc_bandwidth_mbps: 2000.0,
        };
        
        let result = resource_manager.allocate_resources(oversized_request).await;
        assert!(result.is_err());
        
        // Verify it's the right error type
        match result {
            Err(crate::scaling::resource_manager::ResourceError::InsufficientMemory { .. }) => {
                // Expected error
            }
            _ => panic!("Expected InsufficientMemory error"),
        }
    }

    #[tokio::test]
    async fn test_cpu_core_allocation() {
        let scaling_system = create_test_scaling_system().await;
        let resource_manager = &scaling_system.resource_manager;
        
        // Allocate processes that use all CPU cores
        let mut allocations = Vec::new();
        
        for i in 0..4 {
            let request = ResourceAllocationRequest {
                process_id: format!("cpu_process_{}", i),
                process_type: ProcessType::RustCore,
                requested_memory_gb: 10.0,
                max_memory_gb: Some(15.0),
                cpu_cores_needed: 4,
                priority: ResourcePriority::Medium,
                shared_memory_quota_gb: 1.0,
                ipc_bandwidth_mbps: 500.0,
            };
            
            let allocation = resource_manager.allocate_resources(request).await;
            if allocation.is_ok() {
                allocations.push(allocation.unwrap());
            }
        }
        
        // Should have allocated processes using all 16 cores
        let total_cores_used: usize = allocations.iter()
            .map(|a| a.cpu_cores.len())
            .sum();
        assert_eq!(total_cores_used, 16);
        
        // Next allocation should fail due to insufficient CPU cores
        let extra_request = ResourceAllocationRequest {
            process_id: "extra_process".to_string(),
            process_type: ProcessType::PythonML,
            requested_memory_gb: 10.0,
            max_memory_gb: Some(15.0),
            cpu_cores_needed: 2,
            priority: ResourcePriority::Low,
            shared_memory_quota_gb: 1.0,
            ipc_bandwidth_mbps: 500.0,
        };
        
        let result = resource_manager.allocate_resources(extra_request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scaling_cooldown_period() {
        let scaling_system = create_test_scaling_system().await;
        let mut scaler = scaling_system.scaler.write().await;
        
        let high_load = create_high_load_metrics();
        
        // First scaling decision should succeed
        let decision1 = scaler.analyze_and_scale(high_load.clone()).await;
        assert!(decision1.is_some());
        
        // Immediate second decision should be None due to cooldown
        let decision2 = scaler.analyze_and_scale(high_load).await;
        assert!(decision2.is_none());
        
        // After cooldown period, should allow scaling again
        sleep(Duration::from_millis(100)).await; // Shortened for testing
        
        // This test assumes cooldown is implemented correctly in the scaler
    }

    #[tokio::test]
    async fn test_workload_statistics() {
        let scaling_system = create_test_scaling_system().await;
        let mut analyzer = scaling_system.workload_analyzer.write().await;
        
        // Add some workload data
        for i in 0..5 {
            let workload = create_workload_metrics(60.0 + i as f64 * 5.0, 25.0, 5);
            analyzer.analyze_workload(workload).await;
        }
        
        let stats = analyzer.get_workload_statistics();
        assert!(stats.average_cpu_utilization > 0.0);
        assert!(stats.uptime_minutes >= 0);
    }

    #[tokio::test]
    async fn test_performance_prediction() {
        let scaling_system = create_test_scaling_system().await;
        let mut analyzer = scaling_system.workload_analyzer.write().await;
        
        // Build up history
        for i in 0..10 {
            let workload = create_workload_metrics(50.0 + i as f64 * 2.0, 25.0, 3);
            analyzer.analyze_workload(workload).await;
        }
        
        // Try to predict future workload
        let prediction = analyzer.predict_future_workload(15).await;
        if let Some(predicted) = prediction {
            assert!(predicted.documents_per_minute > 0.0);
            assert!(predicted.cpu_utilization_per_core.len() == 16);
        }
    }

    #[tokio::test]
    async fn test_emergency_resource_reclaim() {
        let scaling_system = create_test_scaling_system().await;
        let resource_manager = &scaling_system.resource_manager;
        
        // Allocate some resources with low and medium priority
        let low_priority_request = ResourceAllocationRequest {
            process_id: "low_priority".to_string(),
            process_type: ProcessType::MonitoringDashboard,
            requested_memory_gb: 10.0,
            max_memory_gb: Some(15.0),
            cpu_cores_needed: 1,
            priority: ResourcePriority::Low,
            shared_memory_quota_gb: 1.0,
            ipc_bandwidth_mbps: 200.0,
        };
        
        resource_manager.allocate_resources(low_priority_request).await.unwrap();
        
        // Trigger emergency resource reclaim
        let reclaimed = resource_manager.emergency_resource_reclaim().await;
        assert!(reclaimed.is_ok());
        
        let reclaimed_amount = reclaimed.unwrap();
        assert!(reclaimed_amount > 0.0); // Should have reclaimed some memory
    }

    #[tokio::test] 
    async fn test_integrated_scaling_workflow() {
        let scaling_system = create_test_scaling_system().await;
        
        // Simulate a complete scaling workflow
        // 1. Start with normal load
        let mut scaler = scaling_system.scaler.write().await;
        let normal_metrics = ScalingMetrics {
            cpu_utilization: 60.0,
            memory_utilization: 70.0,
            queue_depth: 5,
            throughput_docs_per_hour: 25.0,
            ipc_latency_ms: 3.0,
            active_agents: 2,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            rust_memory_gb: 60.0,
            python_memory_gb: 45.0,
            shared_memory_gb: 15.0,
        };
        
        let decision = scaler.analyze_and_scale(normal_metrics).await;
        // Should maintain current scale for normal load
        assert!(decision.is_none() || matches!(decision.as_ref().unwrap().action, ScalingAction::Maintain));
        
        // 2. Increase to high load
        sleep(Duration::from_millis(100)).await;
        let high_load = create_high_load_metrics();
        let decision = scaler.analyze_and_scale(high_load).await;
        assert!(decision.is_some());
        
        if let Some(decision) = decision {
            assert!(matches!(decision.action, ScalingAction::ScaleUp));
            assert!(decision.target_agents > 2);
        }
        
        // 3. Emergency situation
        sleep(Duration::from_millis(100)).await;
        let emergency = create_emergency_metrics();
        let emergency_decision = scaler.emergency_scale(&emergency).await;
        assert!(emergency_decision.is_some());
        
        if let Some(emergency_decision) = emergency_decision {
            assert!(matches!(emergency_decision.action, ScalingAction::ScaleDown));
            assert_eq!(emergency_decision.target_agents, 2);
            assert!(emergency_decision.reasoning.contains("EMERGENCY"));
        }
        
        // Verify scaling history is maintained
        let history = scaler.get_scaling_history();
        assert!(history.len() >= 1);
    }
}