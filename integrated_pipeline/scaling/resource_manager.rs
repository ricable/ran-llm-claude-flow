// Resource Manager for Dynamic Scaling
// Phase 2 MCP Advanced Features - Intelligent Resource Allocation
// Manages memory, CPU, and IPC resources within M3 Max constraints

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub process_id: String,
    pub process_type: ProcessType,
    pub memory_allocation_gb: f64,
    pub cpu_cores: Vec<usize>,
    pub priority: ResourcePriority,
    pub max_memory_gb: f64,
    pub shared_memory_quota_gb: f64,
    pub ipc_bandwidth_mbps: f64,
    pub creation_timestamp: u64,
    pub last_adjusted: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessType {
    RustCore,
    PythonML,
    SharedMemory,
    MonitoringDashboard,
    IPCManager,
    MCPServer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourcePriority {
    Critical,   // System-critical processes
    High,       // Primary processing agents
    Medium,     // Secondary processing agents
    Low,        // Background tasks
    Monitoring, // Monitoring and logging
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub process_id: String,
    pub memory_used_gb: f64,
    pub memory_utilization_percent: f64,
    pub cpu_utilization_percent: f64,
    pub ipc_throughput_mbps: f64,
    pub timestamp: u64,
    pub is_healthy: bool,
    pub performance_score: f64, // 0-1, higher is better
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConstraints {
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub cpu_cores: usize,
    pub max_ipc_bandwidth_mbps: f64,
    pub reserved_memory_gb: f64, // OS and other processes
    pub memory_safety_margin_gb: f64,
}

pub struct ResourceManager {
    allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    utilization_history: Arc<Mutex<VecDeque<Vec<ResourceUtilization>>>>,
    system_constraints: Arc<RwLock<SystemConstraints>>,
    resource_semaphore: Arc<Semaphore>,
    allocation_optimizer: AllocationOptimizer,
    rebalancing_config: RebalancingConfig,
}

#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    pub rebalance_interval_seconds: u64,
    pub memory_pressure_threshold: f64,
    pub cpu_imbalance_threshold: f64,
    pub min_rebalance_benefit: f64,
    pub max_rebalance_frequency_per_hour: usize,
    pub enable_proactive_rebalancing: bool,
}

impl Default for RebalancingConfig {
    fn default() -> Self {
        Self {
            rebalance_interval_seconds: 60,
            memory_pressure_threshold: 85.0,
            cpu_imbalance_threshold: 20.0,
            min_rebalance_benefit: 0.05, // 5% improvement minimum
            max_rebalance_frequency_per_hour: 6, // Max 6 rebalances per hour
            enable_proactive_rebalancing: true,
        }
    }
}

pub struct AllocationOptimizer {
    memory_allocation_history: VecDeque<HashMap<String, f64>>,
    performance_correlation_matrix: HashMap<String, Vec<f64>>,
    optimization_learning_rate: f64,
}

impl AllocationOptimizer {
    pub fn new() -> Self {
        Self {
            memory_allocation_history: VecDeque::with_capacity(100),
            performance_correlation_matrix: HashMap::new(),
            optimization_learning_rate: 0.02,
        }
    }

    pub fn optimize_allocation(&mut self, 
                             current_allocations: &HashMap<String, ResourceAllocation>,
                             utilizations: &[ResourceUtilization],
                             constraints: &SystemConstraints) -> OptimizationResult {
        
        let total_allocated = current_allocations.values()
            .map(|a| a.memory_allocation_gb)
            .sum::<f64>();

        if total_allocated > constraints.available_memory_gb {
            return self.handle_memory_pressure(current_allocations, utilizations, constraints);
        }

        // Find underutilized and overutilized processes
        let mut optimization_candidates = Vec::new();
        
        for utilization in utilizations {
            if let Some(allocation) = current_allocations.get(&utilization.process_id) {
                let memory_efficiency = utilization.memory_used_gb / allocation.memory_allocation_gb;
                let performance_per_gb = utilization.performance_score / allocation.memory_allocation_gb;
                
                if memory_efficiency < 0.6 && allocation.memory_allocation_gb > 10.0 {
                    // Underutilized - candidate for reduction
                    optimization_candidates.push(OptimizationCandidate {
                        process_id: utilization.process_id.clone(),
                        current_allocation: allocation.memory_allocation_gb,
                        suggested_allocation: allocation.memory_allocation_gb * 0.85,
                        expected_benefit: performance_per_gb * -0.15, // Negative = giving up resources
                        confidence: 0.7,
                        optimization_type: OptimizationType::ReduceMemory,
                    });
                } else if memory_efficiency > 0.95 && utilization.performance_score < 0.8 {
                    // Overutilized - candidate for increase
                    let additional_memory = (allocation.memory_allocation_gb * 0.2).min(10.0);
                    if total_allocated + additional_memory <= constraints.available_memory_gb {
                        optimization_candidates.push(OptimizationCandidate {
                            process_id: utilization.process_id.clone(),
                            current_allocation: allocation.memory_allocation_gb,
                            suggested_allocation: allocation.memory_allocation_gb + additional_memory,
                            expected_benefit: performance_per_gb * 0.2,
                            confidence: 0.8,
                            optimization_type: OptimizationType::IncreaseMemory,
                        });
                    }
                }
            }
        }

        // Sort by expected benefit
        optimization_candidates.sort_by(|a, b| b.expected_benefit.partial_cmp(&a.expected_benefit).unwrap());

        OptimizationResult {
            candidates: optimization_candidates,
            total_memory_freed: 0.0,
            total_memory_requested: 0.0,
            net_performance_change: 0.0,
            confidence_score: 0.75,
        }
    }

    fn handle_memory_pressure(&self, 
                            current_allocations: &HashMap<String, ResourceAllocation>,
                            utilizations: &[ResourceUtilization],
                            constraints: &SystemConstraints) -> OptimizationResult {
        
        let mut pressure_relief_candidates = Vec::new();
        
        for utilization in utilizations {
            if let Some(allocation) = current_allocations.get(&utilization.process_id) {
                if matches!(allocation.priority, ResourcePriority::Medium | ResourcePriority::Low) {
                    let memory_efficiency = utilization.memory_used_gb / allocation.memory_allocation_gb;
                    if memory_efficiency < 0.8 {
                        let reduction = allocation.memory_allocation_gb * (1.0 - memory_efficiency) * 0.5;
                        pressure_relief_candidates.push(OptimizationCandidate {
                            process_id: utilization.process_id.clone(),
                            current_allocation: allocation.memory_allocation_gb,
                            suggested_allocation: allocation.memory_allocation_gb - reduction,
                            expected_benefit: -0.1, // Small performance cost
                            confidence: 0.9,
                            optimization_type: OptimizationType::PressureRelief,
                        });
                    }
                }
            }
        }

        let total_freed = pressure_relief_candidates.iter()
            .map(|c| c.current_allocation - c.suggested_allocation)
            .sum::<f64>();

        OptimizationResult {
            candidates: pressure_relief_candidates,
            total_memory_freed: total_freed,
            total_memory_requested: 0.0,
            net_performance_change: -0.05, // Small performance cost for stability
            confidence_score: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationCandidate {
    pub process_id: String,
    pub current_allocation: f64,
    pub suggested_allocation: f64,
    pub expected_benefit: f64,
    pub confidence: f64,
    pub optimization_type: OptimizationType,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    IncreaseMemory,
    ReduceMemory,
    PressureRelief,
    LoadBalancing,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub candidates: Vec<OptimizationCandidate>,
    pub total_memory_freed: f64,
    pub total_memory_requested: f64,
    pub net_performance_change: f64,
    pub confidence_score: f64,
}

impl ResourceManager {
    pub fn new() -> Self {
        let m3_max_constraints = SystemConstraints {
            total_memory_gb: 128.0,
            available_memory_gb: 120.0, // Reserve 8GB for macOS
            cpu_cores: 16,
            max_ipc_bandwidth_mbps: 10000.0, // M3 Max theoretical maximum
            reserved_memory_gb: 8.0,
            memory_safety_margin_gb: 5.0,
        };

        Self {
            allocations: Arc::new(RwLock::new(HashMap::new())),
            utilization_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            system_constraints: Arc::new(RwLock::new(m3_max_constraints)),
            resource_semaphore: Arc::new(Semaphore::new(100)), // Limit concurrent operations
            allocation_optimizer: AllocationOptimizer::new(),
            rebalancing_config: RebalancingConfig::default(),
        }
    }

    pub async fn allocate_resources(&self, request: ResourceAllocationRequest) -> Result<ResourceAllocation, ResourceError> {
        let _permit = self.resource_semaphore.acquire().await.unwrap();
        
        let constraints = self.system_constraints.read().await;
        let mut allocations = self.allocations.write().await;
        
        // Check if we have enough resources
        let current_memory_used = allocations.values()
            .map(|a| a.memory_allocation_gb)
            .sum::<f64>();
        
        let available_memory = constraints.available_memory_gb - current_memory_used;
        
        if request.requested_memory_gb > available_memory {
            return Err(ResourceError::InsufficientMemory {
                requested: request.requested_memory_gb,
                available: available_memory,
            });
        }

        // Assign CPU cores based on priority
        let available_cores = self.find_available_cpu_cores(&allocations, request.cpu_cores_needed).await;
        if available_cores.len() < request.cpu_cores_needed {
            return Err(ResourceError::InsufficientCpuCores {
                requested: request.cpu_cores_needed,
                available: available_cores.len(),
            });
        }

        // Create allocation
        let allocation = ResourceAllocation {
            process_id: request.process_id.clone(),
            process_type: request.process_type.clone(),
            memory_allocation_gb: request.requested_memory_gb,
            cpu_cores: available_cores.into_iter().take(request.cpu_cores_needed).collect(),
            priority: request.priority,
            max_memory_gb: request.max_memory_gb.unwrap_or(request.requested_memory_gb * 1.5),
            shared_memory_quota_gb: request.shared_memory_quota_gb,
            ipc_bandwidth_mbps: request.ipc_bandwidth_mbps,
            creation_timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            last_adjusted: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
        };

        allocations.insert(request.process_id.clone(), allocation.clone());
        
        Ok(allocation)
    }

    async fn find_available_cpu_cores(&self, allocations: &HashMap<String, ResourceAllocation>, needed: usize) -> Vec<usize> {
        let mut used_cores = std::collections::HashSet::new();
        
        for allocation in allocations.values() {
            for &core in &allocation.cpu_cores {
                used_cores.insert(core);
            }
        }

        let constraints = self.system_constraints.read().await;
        let mut available_cores = Vec::new();
        
        for core in 0..constraints.cpu_cores {
            if !used_cores.contains(&core) {
                available_cores.push(core);
            }
        }

        available_cores
    }

    pub async fn deallocate_resources(&self, process_id: &str) -> Result<(), ResourceError> {
        let _permit = self.resource_semaphore.acquire().await.unwrap();
        
        let mut allocations = self.allocations.write().await;
        match allocations.remove(process_id) {
            Some(_) => Ok(()),
            None => Err(ResourceError::ProcessNotFound(process_id.to_string())),
        }
    }

    pub async fn update_utilization(&self, utilizations: Vec<ResourceUtilization>) {
        let mut history = self.utilization_history.lock().unwrap();
        
        if history.len() >= 1000 {
            history.pop_front();
        }
        history.push_back(utilizations);
    }

    pub async fn optimize_allocations(&mut self) -> OptimizationResult {
        let allocations = self.allocations.read().await.clone();
        let constraints = self.system_constraints.read().await.clone();
        
        let history = self.utilization_history.lock().unwrap();
        let latest_utilization = history.back().cloned().unwrap_or_default();
        drop(history);

        self.allocation_optimizer.optimize_allocation(&allocations, &latest_utilization, &constraints)
    }

    pub async fn apply_optimization(&self, candidate: OptimizationCandidate) -> Result<(), ResourceError> {
        let _permit = self.resource_semaphore.acquire().await.unwrap();
        
        let mut allocations = self.allocations.write().await;
        
        if let Some(allocation) = allocations.get_mut(&candidate.process_id) {
            let old_allocation = allocation.memory_allocation_gb;
            allocation.memory_allocation_gb = candidate.suggested_allocation;
            allocation.last_adjusted = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            
            // Log the change for monitoring
            eprintln!("Resource optimization applied to {}: {:.1}GB -> {:.1}GB", 
                     candidate.process_id, old_allocation, candidate.suggested_allocation);
            
            Ok(())
        } else {
            Err(ResourceError::ProcessNotFound(candidate.process_id))
        }
    }

    pub async fn get_resource_statistics(&self) -> ResourceStatistics {
        let allocations = self.allocations.read().await;
        let constraints = self.system_constraints.read().await;
        
        let total_allocated_memory = allocations.values()
            .map(|a| a.memory_allocation_gb)
            .sum::<f64>();
        
        let total_allocated_cores = allocations.values()
            .map(|a| a.cpu_cores.len())
            .sum::<usize>();

        let history = self.utilization_history.lock().unwrap();
        let avg_memory_utilization = if let Some(latest) = history.back() {
            latest.iter().map(|u| u.memory_utilization_percent).sum::<f64>() / latest.len() as f64
        } else {
            0.0
        };

        ResourceStatistics {
            total_memory_allocated_gb: total_allocated_memory,
            total_memory_available_gb: constraints.available_memory_gb,
            memory_utilization_percent: (total_allocated_memory / constraints.available_memory_gb) * 100.0,
            total_cpu_cores_allocated: total_allocated_cores,
            total_cpu_cores_available: constraints.cpu_cores,
            active_processes: allocations.len(),
            average_memory_efficiency: avg_memory_utilization,
            memory_fragmentation_gb: self.calculate_memory_fragmentation(&*allocations).await,
        }
    }

    async fn calculate_memory_fragmentation(&self, allocations: &HashMap<String, ResourceAllocation>) -> f64 {
        let history = self.utilization_history.lock().unwrap();
        
        if let Some(latest_utilization) = history.back() {
            let mut fragmentation = 0.0;
            
            for utilization in latest_utilization {
                if let Some(allocation) = allocations.get(&utilization.process_id) {
                    let unused_memory = allocation.memory_allocation_gb - utilization.memory_used_gb;
                    if unused_memory > 1.0 { // Only count significant unused memory
                        fragmentation += unused_memory;
                    }
                }
            }
            
            fragmentation
        } else {
            0.0
        }
    }

    pub async fn emergency_resource_reclaim(&self) -> Result<f64, ResourceError> {
        let _permit = self.resource_semaphore.acquire().await.unwrap();
        
        let mut allocations = self.allocations.write().await;
        let mut reclaimed_memory = 0.0;

        // Force reduce allocations for non-critical processes
        for (process_id, allocation) in allocations.iter_mut() {
            if matches!(allocation.priority, ResourcePriority::Low | ResourcePriority::Medium) {
                let reduction = allocation.memory_allocation_gb * 0.2; // 20% reduction
                allocation.memory_allocation_gb -= reduction;
                allocation.last_adjusted = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
                reclaimed_memory += reduction;
                
                eprintln!("Emergency resource reclaim: {} reduced by {:.1}GB", process_id, reduction);
            }
        }

        Ok(reclaimed_memory)
    }

    pub async fn get_allocation_history(&self, process_id: &str) -> Vec<ResourceAllocation> {
        let allocations = self.allocations.read().await;
        if let Some(allocation) = allocations.get(process_id) {
            vec![allocation.clone()] // In a real implementation, this would return historical data
        } else {
            vec![]
        }
    }

    // Automatic rebalancing based on utilization patterns
    pub async fn auto_rebalance(&mut self) -> RebalanceResult {
        if !self.rebalancing_config.enable_proactive_rebalancing {
            return RebalanceResult::disabled();
        }

        let optimization_result = self.optimize_allocations().await;
        
        if optimization_result.confidence_score < 0.6 {
            return RebalanceResult::skipped("Low confidence in optimization".to_string());
        }

        let mut applied_optimizations = Vec::new();
        let mut total_benefit = 0.0;

        for candidate in optimization_result.candidates {
            if candidate.expected_benefit > self.rebalancing_config.min_rebalance_benefit {
                match self.apply_optimization(candidate.clone()).await {
                    Ok(_) => {
                        applied_optimizations.push(candidate.process_id.clone());
                        total_benefit += candidate.expected_benefit;
                    }
                    Err(e) => {
                        eprintln!("Failed to apply optimization to {}: {:?}", candidate.process_id, e);
                    }
                }
            }
        }

        RebalanceResult {
            success: !applied_optimizations.is_empty(),
            processes_affected: applied_optimizations,
            total_benefit,
            memory_rebalanced_gb: optimization_result.total_memory_freed,
            reason: if total_benefit > 0.0 { 
                "Proactive optimization applied".to_string() 
            } else { 
                "No beneficial optimizations found".to_string() 
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceAllocationRequest {
    pub process_id: String,
    pub process_type: ProcessType,
    pub requested_memory_gb: f64,
    pub max_memory_gb: Option<f64>,
    pub cpu_cores_needed: usize,
    pub priority: ResourcePriority,
    pub shared_memory_quota_gb: f64,
    pub ipc_bandwidth_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatistics {
    pub total_memory_allocated_gb: f64,
    pub total_memory_available_gb: f64,
    pub memory_utilization_percent: f64,
    pub total_cpu_cores_allocated: usize,
    pub total_cpu_cores_available: usize,
    pub active_processes: usize,
    pub average_memory_efficiency: f64,
    pub memory_fragmentation_gb: f64,
}

#[derive(Debug, Clone)]
pub struct RebalanceResult {
    pub success: bool,
    pub processes_affected: Vec<String>,
    pub total_benefit: f64,
    pub memory_rebalanced_gb: f64,
    pub reason: String,
}

impl RebalanceResult {
    pub fn disabled() -> Self {
        Self {
            success: false,
            processes_affected: vec![],
            total_benefit: 0.0,
            memory_rebalanced_gb: 0.0,
            reason: "Auto-rebalancing is disabled".to_string(),
        }
    }

    pub fn skipped(reason: String) -> Self {
        Self {
            success: false,
            processes_affected: vec![],
            total_benefit: 0.0,
            memory_rebalanced_gb: 0.0,
            reason,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    #[error("Insufficient memory: requested {requested}GB, available {available}GB")]
    InsufficientMemory { requested: f64, available: f64 },
    
    #[error("Insufficient CPU cores: requested {requested}, available {available}")]
    InsufficientCpuCores { requested: usize, available: usize },
    
    #[error("Process not found: {0}")]
    ProcessNotFound(String),
    
    #[error("Resource allocation conflict: {0}")]
    AllocationConflict(String),
    
    #[error("System constraints violated: {0}")]
    ConstraintViolation(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_allocation() {
        let manager = ResourceManager::new();
        
        let request = ResourceAllocationRequest {
            process_id: "rust_core_1".to_string(),
            process_type: ProcessType::RustCore,
            requested_memory_gb: 60.0,
            max_memory_gb: Some(80.0),
            cpu_cores_needed: 4,
            priority: ResourcePriority::High,
            shared_memory_quota_gb: 15.0,
            ipc_bandwidth_mbps: 2000.0,
        };

        let allocation = manager.allocate_resources(request).await.unwrap();
        assert_eq!(allocation.memory_allocation_gb, 60.0);
        assert_eq!(allocation.cpu_cores.len(), 4);
    }

    #[tokio::test]
    async fn test_insufficient_memory_error() {
        let manager = ResourceManager::new();
        
        let request = ResourceAllocationRequest {
            process_id: "memory_hog".to_string(),
            process_type: ProcessType::PythonML,
            requested_memory_gb: 150.0, // More than available
            max_memory_gb: None,
            cpu_cores_needed: 2,
            priority: ResourcePriority::Medium,
            shared_memory_quota_gb: 5.0,
            ipc_bandwidth_mbps: 1000.0,
        };

        let result = manager.allocate_resources(request).await;
        assert!(matches!(result, Err(ResourceError::InsufficientMemory { .. })));
    }

    #[tokio::test]
    async fn test_resource_deallocation() {
        let manager = ResourceManager::new();
        
        let request = ResourceAllocationRequest {
            process_id: "test_process".to_string(),
            process_type: ProcessType::RustCore,
            requested_memory_gb: 30.0,
            max_memory_gb: None,
            cpu_cores_needed: 2,
            priority: ResourcePriority::Medium,
            shared_memory_quota_gb: 5.0,
            ipc_bandwidth_mbps: 1000.0,
        };

        manager.allocate_resources(request).await.unwrap();
        let result = manager.deallocate_resources("test_process").await;
        assert!(result.is_ok());

        let result = manager.deallocate_resources("nonexistent_process").await;
        assert!(matches!(result, Err(ResourceError::ProcessNotFound(_))));
    }
}