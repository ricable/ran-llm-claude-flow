//! M3 Max Resource Utilization Tracker
//!
//! Monitors CPU/GPU/memory usage patterns with specialized tracking
//! for Apple M3 Max architecture and 128GB unified memory.

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicF64, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use prometheus::core::{Atomic, AtomicF64};
use tokio::time;

use crate::config::MonitoringConfig;

/// M3 Max specific resource tracker
pub struct ResourceTracker {
    config: MonitoringConfig,
    running: AtomicBool,
    
    // M3 Max specific metrics
    m3_metrics: Arc<RwLock<M3MaxMetrics>>,
    resource_history: Arc<RwLock<VecDeque<ResourceSnapshot>>>,
    
    // Performance counters
    samples_collected: AtomicU64,
    gpu_utilization_avg: AtomicF64,
    memory_bandwidth_mbps: AtomicF64,
    
    // Resource allocation tracking
    rust_allocation: Arc<RwLock<ResourceAllocation>>,
    python_allocation: Arc<RwLock<ResourceAllocation>>,
    shared_allocation: Arc<RwLock<ResourceAllocation>>,
}

impl ResourceTracker {
    pub fn new(config: &MonitoringConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            running: AtomicBool::new(false),
            m3_metrics: Arc::new(RwLock::new(M3MaxMetrics::new())),
            resource_history: Arc::new(RwLock::new(VecDeque::with_capacity(50000))),
            samples_collected: AtomicU64::new(0),
            gpu_utilization_avg: AtomicF64::new(0.0),
            memory_bandwidth_mbps: AtomicF64::new(0.0),
            rust_allocation: Arc::new(RwLock::new(ResourceAllocation::new("rust_core", 60.0))),
            python_allocation: Arc::new(RwLock::new(ResourceAllocation::new("python_ml", 45.0))),
            shared_allocation: Arc::new(RwLock::new(ResourceAllocation::new("shared_memory", 15.0))),
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.running.store(true, Ordering::Release);
        tracing::info!("Initializing M3 Max resource tracker");

        // Detect M3 Max capabilities
        self.detect_m3_max_capabilities().await?;
        
        // Start high-frequency resource monitoring (100ms intervals)
        let tracker = self.clone_for_task();
        tokio::spawn(async move {
            tracker.resource_monitoring_loop().await;
        });

        // Start resource optimization loop
        let optimizer = self.clone_for_task();
        tokio::spawn(async move {
            optimizer.resource_optimization_loop().await;
        });

        // Start bandwidth monitoring
        let bandwidth_monitor = self.clone_for_task();
        tokio::spawn(async move {
            bandwidth_monitor.bandwidth_monitoring_loop().await;
        });

        tracing::info!("M3 Max resource tracker operational");
        Ok(())
    }

    async fn detect_m3_max_capabilities(&self) -> Result<()> {
        let mut metrics = self.m3_metrics.write();
        
        // Detect M3 Max specific features
        metrics.cpu_cores = self.detect_cpu_cores().await?;
        metrics.gpu_cores = self.detect_gpu_cores().await?;
        metrics.unified_memory_gb = self.detect_unified_memory().await?;
        metrics.memory_bandwidth_gbps = self.detect_memory_bandwidth().await?;
        metrics.neural_engine_tops = self.detect_neural_engine().await?;
        
        tracing::info!(
            "Detected M3 Max: {} CPU cores, {} GPU cores, {:.1}GB unified memory, {:.1}GB/s bandwidth",
            metrics.cpu_cores,
            metrics.gpu_cores,
            metrics.unified_memory_gb,
            metrics.memory_bandwidth_gbps
        );
        
        Ok(())
    }

    async fn resource_monitoring_loop(&self) {
        let mut interval = time::interval(Duration::from_millis(100)); // 100ms for high frequency

        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.collect_resource_metrics().await {
                tracing::error!("Resource collection failed: {}", e);
            }
            
            self.samples_collected.fetch_add(1, Ordering::Relaxed);
        }
    }

    async fn resource_optimization_loop(&self) {
        let mut interval = time::interval(Duration::from_secs(30));

        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.optimize_resource_allocation().await {
                tracing::error!("Resource optimization failed: {}", e);
            }
        }
    }

    async fn bandwidth_monitoring_loop(&self) {
        let mut interval = time::interval(Duration::from_millis(500)); // 500ms for bandwidth

        while self.running.load(Ordering::Acquire) {
            interval.tick().await;
            
            if let Err(e) = self.monitor_memory_bandwidth().await {
                tracing::error!("Bandwidth monitoring failed: {}", e);
            }
        }
    }

    async fn collect_resource_metrics(&self) -> Result<()> {
        let start = std::time::Instant::now();
        
        // Collect M3 Max specific metrics
        let cpu_metrics = self.collect_cpu_metrics().await?;
        let gpu_metrics = self.collect_gpu_metrics().await?;
        let memory_metrics = self.collect_memory_metrics().await?;
        let thermal_metrics = self.collect_thermal_metrics().await?;
        let power_metrics = self.collect_power_metrics().await?;
        
        // Create resource snapshot
        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            cpu_metrics,
            gpu_metrics,
            memory_metrics,
            thermal_metrics,
            power_metrics,
            collection_overhead_us: start.elapsed().as_micros() as f64,
            allocation_efficiency: self.calculate_allocation_efficiency().await?,
        };
        
        // Store in history
        {
            let mut history = self.resource_history.write();
            history.push_back(snapshot.clone());
            if history.len() > 50000 {
                history.pop_front();
            }
        }
        
        // Update running averages
        self.update_running_averages(&snapshot).await;
        
        Ok(())
    }

    async fn collect_cpu_metrics(&self) -> Result<CpuResourceMetrics> {
        // M3 Max CPU monitoring
        Ok(CpuResourceMetrics {
            performance_cores_utilization: self.get_performance_cores_utilization().await?,
            efficiency_cores_utilization: self.get_efficiency_cores_utilization().await?,
            total_utilization: self.get_total_cpu_utilization().await?,
            frequency_mhz: self.get_cpu_frequency().await?,
            temperature_celsius: self.get_cpu_temperature().await?,
            power_watts: self.get_cpu_power().await?,
            cache_hit_rate: self.get_cache_hit_rate().await?,
            instructions_per_second: self.get_instructions_per_second().await?,
        })
    }

    async fn collect_gpu_metrics(&self) -> Result<GpuResourceMetrics> {
        // M3 Max GPU monitoring
        Ok(GpuResourceMetrics {
            gpu_utilization: self.get_gpu_utilization().await?,
            gpu_memory_utilization: self.get_gpu_memory_utilization().await?,
            gpu_frequency_mhz: self.get_gpu_frequency().await?,
            gpu_temperature_celsius: self.get_gpu_temperature().await?,
            gpu_power_watts: self.get_gpu_power().await?,
            compute_units_active: self.get_active_compute_units().await?,
            memory_bandwidth_utilization: self.get_gpu_memory_bandwidth().await?,
            shader_core_utilization: self.get_shader_core_utilization().await?,
        })
    }

    async fn collect_memory_metrics(&self) -> Result<MemoryResourceMetrics> {
        // Unified memory system metrics
        Ok(MemoryResourceMetrics {
            unified_memory_total_gb: self.get_unified_memory_total().await?,
            unified_memory_used_gb: self.get_unified_memory_used().await?,
            unified_memory_available_gb: self.get_unified_memory_available().await?,
            memory_bandwidth_gbps: self.get_memory_bandwidth().await?,
            memory_pressure_level: self.get_memory_pressure().await?,
            swap_usage_gb: self.get_swap_usage().await?,
            
            // Component-specific allocations
            rust_allocation_gb: self.get_rust_memory_usage().await?,
            python_allocation_gb: self.get_python_memory_usage().await?,
            shared_memory_allocation_gb: self.get_shared_memory_usage().await?,
            gpu_allocation_gb: self.get_gpu_memory_usage().await?,
            
            // Performance metrics
            memory_latency_ns: self.get_memory_latency().await?,
            cache_efficiency: self.get_memory_cache_efficiency().await?,
        })
    }

    async fn collect_thermal_metrics(&self) -> Result<ThermalMetrics> {
        Ok(ThermalMetrics {
            cpu_die_temperature: self.get_cpu_die_temperature().await?,
            gpu_die_temperature: self.get_gpu_die_temperature().await?,
            system_temperature: self.get_system_temperature().await?,
            thermal_state: self.get_thermal_state().await?,
            fan_speed_rpm: self.get_fan_speed().await?,
            throttling_active: self.is_thermal_throttling().await?,
        })
    }

    async fn collect_power_metrics(&self) -> Result<PowerMetrics> {
        Ok(PowerMetrics {
            total_power_watts: self.get_total_power().await?,
            cpu_power_watts: self.get_cpu_power().await?,
            gpu_power_watts: self.get_gpu_power().await?,
            memory_power_watts: self.get_memory_power().await?,
            system_power_watts: self.get_system_power().await?,
            power_efficiency_score: self.calculate_power_efficiency().await?,
            battery_level_percent: self.get_battery_level().await?,
        })
    }

    async fn monitor_memory_bandwidth(&self) -> Result<()> {
        let bandwidth_gbps = self.measure_memory_bandwidth().await?;
        self.memory_bandwidth_mbps.store(bandwidth_gbps * 1000.0, Ordering::Relaxed);
        
        // Check for memory bandwidth bottlenecks
        let m3_metrics = self.m3_metrics.read();
        let theoretical_max = m3_metrics.memory_bandwidth_gbps;
        let utilization = bandwidth_gbps / theoretical_max;
        
        if utilization > 0.9 {
            tracing::warn!(
                "High memory bandwidth utilization: {:.1}% ({:.1}GB/s of {:.1}GB/s)",
                utilization * 100.0,
                bandwidth_gbps,
                theoretical_max
            );
        }
        
        Ok(())
    }

    async fn optimize_resource_allocation(&self) -> Result<()> {
        let current_usage = self.get_current_resource_usage().await?;
        
        // Optimize Rust allocation
        self.optimize_rust_allocation(&current_usage).await?;
        
        // Optimize Python allocation
        self.optimize_python_allocation(&current_usage).await?;
        
        // Optimize shared memory allocation
        self.optimize_shared_allocation(&current_usage).await?;
        
        // Balance GPU resources
        self.balance_gpu_resources(&current_usage).await?;
        
        Ok(())
    }

    async fn optimize_rust_allocation(&self, usage: &CurrentResourceUsage) -> Result<()> {
        let mut allocation = self.rust_allocation.write();
        
        // Adjust based on current usage patterns
        if usage.rust_cpu_utilization > 80.0 && usage.rust_memory_utilization < 70.0 {
            // CPU bound - consider increasing memory allocation for caching
            let new_limit = (allocation.memory_limit_gb * 1.1).min(70.0);
            if new_limit != allocation.memory_limit_gb {
                allocation.memory_limit_gb = new_limit;
                tracing::info!("Increased Rust memory allocation to {:.1}GB", new_limit);
            }
        } else if usage.rust_memory_utilization > 90.0 {
            // Memory pressure - consider reducing allocation or optimizing usage
            let new_limit = (allocation.memory_limit_gb * 0.95).max(50.0);
            if new_limit != allocation.memory_limit_gb {
                allocation.memory_limit_gb = new_limit;
                tracing::warn!("Reduced Rust memory allocation to {:.1}GB due to pressure", new_limit);
            }
        }
        
        allocation.last_optimized = Utc::now();
        Ok(())
    }

    async fn optimize_python_allocation(&self, usage: &CurrentResourceUsage) -> Result<()> {
        let mut allocation = self.python_allocation.write();
        
        // Python ML workload optimization
        if usage.python_gpu_utilization > 85.0 && usage.python_memory_utilization < 80.0 {
            // GPU intensive - increase memory for model caching
            let new_limit = (allocation.memory_limit_gb * 1.1).min(50.0);
            if new_limit != allocation.memory_limit_gb {
                allocation.memory_limit_gb = new_limit;
                tracing::info!("Increased Python memory allocation to {:.1}GB for ML workload", new_limit);
            }
        }
        
        allocation.last_optimized = Utc::now();
        Ok(())
    }

    async fn optimize_shared_allocation(&self, usage: &CurrentResourceUsage) -> Result<()> {
        let mut allocation = self.shared_allocation.write();
        
        // Optimize shared memory based on IPC patterns
        if usage.ipc_message_rate > 1000.0 && usage.shared_memory_utilization < 60.0 {
            // High IPC traffic - increase shared memory
            let new_limit = (allocation.memory_limit_gb * 1.2).min(20.0);
            if new_limit != allocation.memory_limit_gb {
                allocation.memory_limit_gb = new_limit;
                tracing::info!("Increased shared memory allocation to {:.1}GB for IPC", new_limit);
            }
        }
        
        allocation.last_optimized = Utc::now();
        Ok(())
    }

    async fn balance_gpu_resources(&self, usage: &CurrentResourceUsage) -> Result<()> {
        // Balance GPU resources between Python ML and system
        if usage.python_gpu_utilization > 90.0 && usage.system_gpu_utilization < 20.0 {
            // High ML GPU usage - consider priority adjustment
            tracing::info!("High ML GPU utilization detected: {:.1}%", usage.python_gpu_utilization);
        }
        
        Ok(())
    }

    async fn calculate_allocation_efficiency(&self) -> Result<f64> {
        let rust_alloc = self.rust_allocation.read();
        let python_alloc = self.python_allocation.read();
        let shared_alloc = self.shared_allocation.read();
        
        let total_allocated = rust_alloc.memory_limit_gb + python_alloc.memory_limit_gb + shared_alloc.memory_limit_gb;
        let total_used = self.get_total_memory_used().await?;
        
        // Efficiency is how well we're using our allocated resources
        let efficiency = if total_allocated > 0.0 {
            (total_used / total_allocated).min(1.0)
        } else {
            0.0
        };
        
        Ok(efficiency)
    }

    async fn update_running_averages(&self, snapshot: &ResourceSnapshot) {
        // Update GPU utilization average
        let current_gpu_avg = self.gpu_utilization_avg.load(Ordering::Relaxed);
        let new_gpu_avg = current_gpu_avg * 0.95 + snapshot.gpu_metrics.gpu_utilization * 0.05;
        self.gpu_utilization_avg.store(new_gpu_avg, Ordering::Relaxed);
        
        // Update memory bandwidth average
        let current_bw_avg = self.memory_bandwidth_mbps.load(Ordering::Relaxed);
        let new_bw_avg = current_bw_avg * 0.95 + snapshot.memory_metrics.memory_bandwidth_gbps * 1000.0 * 0.05;
        self.memory_bandwidth_mbps.store(new_bw_avg, Ordering::Relaxed);
    }

    pub async fn get_current_resource_snapshot(&self) -> Result<ResourceSnapshot> {
        let history = self.resource_history.read();
        if let Some(latest) = history.back() {
            Ok(latest.clone())
        } else {
            // Create a default snapshot if no history exists
            Ok(ResourceSnapshot::default())
        }
    }

    pub async fn get_resource_statistics(&self) -> ResourceStatistics {
        let samples = self.samples_collected.load(Ordering::Relaxed);
        let gpu_avg = self.gpu_utilization_avg.load(Ordering::Relaxed);
        let bandwidth_avg = self.memory_bandwidth_mbps.load(Ordering::Relaxed);
        
        let history = self.resource_history.read();
        let recent_snapshots: Vec<_> = history.iter().rev().take(600).cloned().collect(); // Last 1 minute
        
        let avg_cpu_temp = if !recent_snapshots.is_empty() {
            recent_snapshots.iter().map(|s| s.cpu_metrics.temperature_celsius).sum::<f64>() / recent_snapshots.len() as f64
        } else {
            0.0
        };
        
        let avg_gpu_temp = if !recent_snapshots.is_empty() {
            recent_snapshots.iter().map(|s| s.gpu_metrics.gpu_temperature_celsius).sum::<f64>() / recent_snapshots.len() as f64
        } else {
            0.0
        };
        
        ResourceStatistics {
            total_samples_collected: samples,
            average_gpu_utilization: gpu_avg,
            average_memory_bandwidth_mbps: bandwidth_avg,
            average_cpu_temperature: avg_cpu_temp,
            average_gpu_temperature: avg_gpu_temp,
            current_allocation_efficiency: recent_snapshots.last().map(|s| s.allocation_efficiency).unwrap_or(0.0),
            memory_pressure_events: 0, // TODO: track pressure events
            thermal_throttling_events: 0, // TODO: track throttling events
        }
    }

    // Hardware detection methods (simplified - would use actual system APIs)
    async fn detect_cpu_cores(&self) -> Result<u32> {
        Ok(12) // M3 Max has 12 CPU cores (8 performance + 4 efficiency)
    }

    async fn detect_gpu_cores(&self) -> Result<u32> {
        Ok(38) // M3 Max has 38 GPU cores (varies by model)
    }

    async fn detect_unified_memory(&self) -> Result<f64> {
        Ok(128.0) // Assuming 128GB configuration
    }

    async fn detect_memory_bandwidth(&self) -> Result<f64> {
        Ok(400.0) // M3 Max memory bandwidth
    }

    async fn detect_neural_engine(&self) -> Result<f64> {
        Ok(15.8) // Neural Engine TOPS
    }

    // Metric collection methods (would integrate with actual system APIs)
    async fn get_performance_cores_utilization(&self) -> Result<f64> { Ok(45.0) }
    async fn get_efficiency_cores_utilization(&self) -> Result<f64> { Ok(25.0) }
    async fn get_total_cpu_utilization(&self) -> Result<f64> { Ok(40.0) }
    async fn get_cpu_frequency(&self) -> Result<f64> { Ok(3200.0) }
    async fn get_cpu_temperature(&self) -> Result<f64> { Ok(65.0) }
    async fn get_cpu_power(&self) -> Result<f64> { Ok(15.5) }
    async fn get_cache_hit_rate(&self) -> Result<f64> { Ok(0.95) }
    async fn get_instructions_per_second(&self) -> Result<f64> { Ok(1_000_000_000.0) }
    
    async fn get_gpu_utilization(&self) -> Result<f64> { Ok(55.0) }
    async fn get_gpu_memory_utilization(&self) -> Result<f64> { Ok(45.0) }
    async fn get_gpu_frequency(&self) -> Result<f64> { Ok(1200.0) }
    async fn get_gpu_temperature(&self) -> Result<f64> { Ok(70.0) }
    async fn get_gpu_power(&self) -> Result<f64> { Ok(25.0) }
    async fn get_active_compute_units(&self) -> Result<u32> { Ok(32) }
    async fn get_gpu_memory_bandwidth(&self) -> Result<f64> { Ok(0.75) }
    async fn get_shader_core_utilization(&self) -> Result<f64> { Ok(60.0) }
    
    async fn get_unified_memory_total(&self) -> Result<f64> { Ok(128.0) }
    async fn get_unified_memory_used(&self) -> Result<f64> { Ok(75.0) }
    async fn get_unified_memory_available(&self) -> Result<f64> { Ok(53.0) }
    async fn get_memory_bandwidth(&self) -> Result<f64> { Ok(250.0) }
    async fn get_memory_pressure(&self) -> Result<f64> { Ok(0.6) }
    async fn get_swap_usage(&self) -> Result<f64> { Ok(2.1) }
    async fn get_rust_memory_usage(&self) -> Result<f64> { Ok(42.0) }
    async fn get_python_memory_usage(&self) -> Result<f64> { Ok(35.0) }
    async fn get_shared_memory_usage(&self) -> Result<f64> { Ok(12.5) }
    async fn get_gpu_memory_usage(&self) -> Result<f64> { Ok(18.0) }
    async fn get_memory_latency(&self) -> Result<f64> { Ok(85.0) }
    async fn get_memory_cache_efficiency(&self) -> Result<f64> { Ok(0.92) }
    
    async fn get_cpu_die_temperature(&self) -> Result<f64> { Ok(68.0) }
    async fn get_gpu_die_temperature(&self) -> Result<f64> { Ok(72.0) }
    async fn get_system_temperature(&self) -> Result<f64> { Ok(45.0) }
    async fn get_thermal_state(&self) -> Result<ThermalState> { Ok(ThermalState::Normal) }
    async fn get_fan_speed(&self) -> Result<f64> { Ok(1200.0) }
    async fn is_thermal_throttling(&self) -> Result<bool> { Ok(false) }
    
    async fn get_total_power(&self) -> Result<f64> { Ok(45.0) }
    async fn get_memory_power(&self) -> Result<f64> { Ok(8.0) }
    async fn get_system_power(&self) -> Result<f64> { Ok(12.0) }
    async fn calculate_power_efficiency(&self) -> Result<f64> { Ok(0.85) }
    async fn get_battery_level(&self) -> Result<Option<f64>> { Ok(Some(85.0)) }
    
    async fn measure_memory_bandwidth(&self) -> Result<f64> { Ok(245.0) }
    async fn get_current_resource_usage(&self) -> Result<CurrentResourceUsage> {
        Ok(CurrentResourceUsage {
            rust_cpu_utilization: 35.0,
            rust_memory_utilization: 70.0,
            python_cpu_utilization: 25.0,
            python_gpu_utilization: 60.0,
            python_memory_utilization: 78.0,
            system_gpu_utilization: 15.0,
            shared_memory_utilization: 83.0,
            ipc_message_rate: 850.0,
        })
    }
    
    async fn get_total_memory_used(&self) -> Result<f64> { Ok(89.5) }

    pub fn clone_for_task(&self) -> Self {
        Self {
            config: self.config.clone(),
            running: AtomicBool::new(self.running.load(Ordering::Acquire)),
            m3_metrics: self.m3_metrics.clone(),
            resource_history: self.resource_history.clone(),
            samples_collected: AtomicU64::new(self.samples_collected.load(Ordering::Relaxed)),
            gpu_utilization_avg: AtomicF64::new(self.gpu_utilization_avg.load(Ordering::Relaxed)),
            memory_bandwidth_mbps: AtomicF64::new(self.memory_bandwidth_mbps.load(Ordering::Relaxed)),
            rust_allocation: self.rust_allocation.clone(),
            python_allocation: self.python_allocation.clone(),
            shared_allocation: self.shared_allocation.clone(),
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        tracing::info!("Resource tracker shutdown complete");
        Ok(())
    }
}

// Data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
struct M3MaxMetrics {
    cpu_cores: u32,
    gpu_cores: u32,
    unified_memory_gb: f64,
    memory_bandwidth_gbps: f64,
    neural_engine_tops: f64,
}

impl M3MaxMetrics {
    fn new() -> Self {
        Self {
            cpu_cores: 0,
            gpu_cores: 0,
            unified_memory_gb: 0.0,
            memory_bandwidth_gbps: 0.0,
            neural_engine_tops: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu_metrics: CpuResourceMetrics,
    pub gpu_metrics: GpuResourceMetrics,
    pub memory_metrics: MemoryResourceMetrics,
    pub thermal_metrics: ThermalMetrics,
    pub power_metrics: PowerMetrics,
    pub collection_overhead_us: f64,
    pub allocation_efficiency: f64,
}

impl Default for ResourceSnapshot {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            cpu_metrics: CpuResourceMetrics::default(),
            gpu_metrics: GpuResourceMetrics::default(),
            memory_metrics: MemoryResourceMetrics::default(),
            thermal_metrics: ThermalMetrics::default(),
            power_metrics: PowerMetrics::default(),
            collection_overhead_us: 0.0,
            allocation_efficiency: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuResourceMetrics {
    pub performance_cores_utilization: f64,
    pub efficiency_cores_utilization: f64,
    pub total_utilization: f64,
    pub frequency_mhz: f64,
    pub temperature_celsius: f64,
    pub power_watts: f64,
    pub cache_hit_rate: f64,
    pub instructions_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuResourceMetrics {
    pub gpu_utilization: f64,
    pub gpu_memory_utilization: f64,
    pub gpu_frequency_mhz: f64,
    pub gpu_temperature_celsius: f64,
    pub gpu_power_watts: f64,
    pub compute_units_active: u32,
    pub memory_bandwidth_utilization: f64,
    pub shader_core_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryResourceMetrics {
    pub unified_memory_total_gb: f64,
    pub unified_memory_used_gb: f64,
    pub unified_memory_available_gb: f64,
    pub memory_bandwidth_gbps: f64,
    pub memory_pressure_level: f64,
    pub swap_usage_gb: f64,
    pub rust_allocation_gb: f64,
    pub python_allocation_gb: f64,
    pub shared_memory_allocation_gb: f64,
    pub gpu_allocation_gb: f64,
    pub memory_latency_ns: f64,
    pub cache_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThermalMetrics {
    pub cpu_die_temperature: f64,
    pub gpu_die_temperature: f64,
    pub system_temperature: f64,
    pub thermal_state: ThermalState,
    pub fan_speed_rpm: f64,
    pub throttling_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ThermalState {
    #[default]
    Normal,
    Warm,
    Hot,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerMetrics {
    pub total_power_watts: f64,
    pub cpu_power_watts: f64,
    pub gpu_power_watts: f64,
    pub memory_power_watts: f64,
    pub system_power_watts: f64,
    pub power_efficiency_score: f64,
    pub battery_level_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResourceAllocation {
    component_name: String,
    memory_limit_gb: f64,
    cpu_priority: u32,
    gpu_priority: u32,
    last_optimized: DateTime<Utc>,
}

impl ResourceAllocation {
    fn new(component_name: &str, memory_limit_gb: f64) -> Self {
        Self {
            component_name: component_name.to_string(),
            memory_limit_gb,
            cpu_priority: 50, // Default priority
            gpu_priority: 50,
            last_optimized: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CurrentResourceUsage {
    rust_cpu_utilization: f64,
    rust_memory_utilization: f64,
    python_cpu_utilization: f64,
    python_gpu_utilization: f64,
    python_memory_utilization: f64,
    system_gpu_utilization: f64,
    shared_memory_utilization: f64,
    ipc_message_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceStatistics {
    pub total_samples_collected: u64,
    pub average_gpu_utilization: f64,
    pub average_memory_bandwidth_mbps: f64,
    pub average_cpu_temperature: f64,
    pub average_gpu_temperature: f64,
    pub current_allocation_efficiency: f64,
    pub memory_pressure_events: u64,
    pub thermal_throttling_events: u64,
}
