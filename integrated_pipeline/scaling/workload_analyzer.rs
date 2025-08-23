// Workload Analyzer for Dynamic Scaling
// Phase 2 MCP Advanced Features - Intelligent Load Balancing Component
// Analyzes workload patterns to make informed scaling decisions

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMetrics {
    pub processing_queue_depth: usize,
    pub average_processing_time_ms: f64,
    pub documents_per_minute: f64,
    pub cpu_utilization_per_core: Vec<f64>,
    pub memory_utilization_breakdown: MemoryBreakdown,
    pub ipc_throughput_mbps: f64,
    pub error_rate_percent: f64,
    pub agent_utilization: HashMap<String, f64>,
    pub timestamp: u64,
    pub peak_hour_indicator: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBreakdown {
    pub rust_processes_gb: f64,
    pub python_processes_gb: f64,
    pub shared_memory_gb: f64,
    pub cached_data_gb: f64,
    pub free_memory_gb: f64,
    pub memory_fragmentation_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPattern {
    pub pattern_type: PatternType,
    pub confidence_score: f64,
    pub duration_minutes: u64,
    pub characteristics: PatternCharacteristics,
    pub recommended_agents: usize,
    pub memory_recommendation: MemoryAllocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    BurstTraffic,     // High short-term load
    SteadyState,      // Consistent load
    BatchProcessing,  // Periodic high load
    IdlePeriod,       // Low activity
    GrowingLoad,      // Increasing trend
    DecliningLoad,    // Decreasing trend
    CyclicPattern,    // Predictable cycles
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCharacteristics {
    pub avg_cpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub peak_queue_depth: usize,
    pub throughput_variance: f64,
    pub latency_p95_ms: f64,
    pub predictability_score: f64, // 0-1, how predictable the pattern is
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub rust_memory_gb: f64,
    pub python_memory_gb: f64,
    pub shared_memory_gb: f64,
    pub cache_memory_gb: f64,
}

pub struct WorkloadAnalyzer {
    metrics_history: Arc<Mutex<VecDeque<WorkloadMetrics>>>,
    pattern_cache: Arc<Mutex<HashMap<String, WorkloadPattern>>>,
    analysis_config: AnalysisConfig,
    machine_learning_model: SimpleMLModel,
}

#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub history_window_minutes: u64,
    pub pattern_detection_sensitivity: f64,
    pub min_confidence_threshold: f64,
    pub burst_detection_threshold: f64,
    pub steady_state_variance_threshold: f64,
    pub memory_optimization_threshold: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            history_window_minutes: 60,
            pattern_detection_sensitivity: 0.7,
            min_confidence_threshold: 0.6,
            burst_detection_threshold: 2.0,
            steady_state_variance_threshold: 0.2,
            memory_optimization_threshold: 0.85,
        }
    }
}

// Simple machine learning model for workload prediction
pub struct SimpleMLModel {
    cpu_weights: Vec<f64>,
    memory_weights: Vec<f64>,
    throughput_weights: Vec<f64>,
    learning_rate: f64,
    prediction_horizon: usize,
}

impl SimpleMLModel {
    pub fn new() -> Self {
        Self {
            cpu_weights: vec![0.3, 0.4, 0.2, 0.1],
            memory_weights: vec![0.25, 0.35, 0.25, 0.15],
            throughput_weights: vec![0.4, 0.3, 0.2, 0.1],
            learning_rate: 0.01,
            prediction_horizon: 10,
        }
    }

    pub fn predict_workload(&self, historical_metrics: &[WorkloadMetrics]) -> Option<WorkloadMetrics> {
        if historical_metrics.len() < 4 {
            return None;
        }

        let recent_metrics = &historical_metrics[historical_metrics.len()-4..];
        
        let predicted_cpu = self.weighted_prediction(&recent_metrics.iter().map(|m| {
            m.cpu_utilization_per_core.iter().sum::<f64>() / m.cpu_utilization_per_core.len() as f64
        }).collect::<Vec<f64>>(), &self.cpu_weights);

        let predicted_memory = self.weighted_prediction(&recent_metrics.iter().map(|m| {
            (m.memory_utilization_breakdown.rust_processes_gb + 
             m.memory_utilization_breakdown.python_processes_gb + 
             m.memory_utilization_breakdown.shared_memory_gb) / 128.0 * 100.0
        }).collect::<Vec<f64>>(), &self.memory_weights);

        let predicted_throughput = self.weighted_prediction(&recent_metrics.iter().map(|m| {
            m.documents_per_minute
        }).collect::<Vec<f64>>(), &self.throughput_weights);

        let latest = &recent_metrics[recent_metrics.len()-1];
        
        Some(WorkloadMetrics {
            processing_queue_depth: (latest.processing_queue_depth as f64 * 0.9) as usize,
            average_processing_time_ms: latest.average_processing_time_ms,
            documents_per_minute: predicted_throughput,
            cpu_utilization_per_core: vec![predicted_cpu; latest.cpu_utilization_per_core.len()],
            memory_utilization_breakdown: MemoryBreakdown {
                rust_processes_gb: 60.0,
                python_processes_gb: 45.0,
                shared_memory_gb: 15.0,
                cached_data_gb: 5.0,
                free_memory_gb: 128.0 - (predicted_memory * 1.28),
                memory_fragmentation_percent: latest.memory_utilization_breakdown.memory_fragmentation_percent,
            },
            ipc_throughput_mbps: latest.ipc_throughput_mbps,
            error_rate_percent: latest.error_rate_percent * 0.95, // Assume slight improvement
            agent_utilization: latest.agent_utilization.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            peak_hour_indicator: Self::is_peak_hour(),
        })
    }

    fn weighted_prediction(&self, values: &[f64], weights: &[f64]) -> f64 {
        values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
    }

    fn is_peak_hour() -> bool {
        let now = chrono::Local::now();
        let hour = now.hour();
        // Assume peak hours are 9-11 AM and 2-4 PM
        (hour >= 9 && hour <= 11) || (hour >= 14 && hour <= 16)
    }

    pub fn train(&mut self, actual: &WorkloadMetrics, predicted: &WorkloadMetrics) {
        // Simple gradient descent update
        let cpu_actual = actual.cpu_utilization_per_core.iter().sum::<f64>() / actual.cpu_utilization_per_core.len() as f64;
        let cpu_predicted = predicted.cpu_utilization_per_core.iter().sum::<f64>() / predicted.cpu_utilization_per_core.len() as f64;
        let cpu_error = cpu_actual - cpu_predicted;

        for weight in &mut self.cpu_weights {
            *weight += self.learning_rate * cpu_error;
        }
        
        // Normalize weights
        let sum: f64 = self.cpu_weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut self.cpu_weights {
                *weight /= sum;
            }
        }
    }
}

impl WorkloadAnalyzer {
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            metrics_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            pattern_cache: Arc::new(Mutex::new(HashMap::new())),
            analysis_config: config,
            machine_learning_model: SimpleMLModel::new(),
        }
    }

    pub async fn analyze_workload(&mut self, metrics: WorkloadMetrics) -> WorkloadPattern {
        // Add metrics to history
        {
            let mut history = self.metrics_history.lock().unwrap();
            history.push_back(metrics.clone());
            
            // Keep only recent history
            let cutoff_time = metrics.timestamp - (self.analysis_config.history_window_minutes * 60 * 1000);
            while let Some(front) = history.front() {
                if front.timestamp < cutoff_time {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        let pattern = self.detect_pattern(&metrics).await;
        
        // Cache the pattern for future reference
        let pattern_key = format!("{}_{}", pattern.pattern_type.pattern_name(), metrics.timestamp / 60000); // Per-minute key
        {
            let mut cache = self.pattern_cache.lock().unwrap();
            cache.insert(pattern_key, pattern.clone());
            
            // Limit cache size
            if cache.len() > 1000 {
                let oldest_key = cache.keys().next().unwrap().clone();
                cache.remove(&oldest_key);
            }
        }

        pattern
    }

    async fn detect_pattern(&self, current_metrics: &WorkloadMetrics) -> WorkloadPattern {
        let history = self.metrics_history.lock().unwrap();
        let historical_data: Vec<WorkloadMetrics> = history.iter().cloned().collect();
        drop(history);

        if historical_data.len() < 5 {
            return self.default_pattern(current_metrics);
        }

        let recent_window = &historical_data[historical_data.len().saturating_sub(10)..];
        
        // Calculate statistics
        let avg_cpu = self.calculate_average_cpu(&recent_window);
        let cpu_variance = self.calculate_cpu_variance(&recent_window, avg_cpu);
        let avg_memory = self.calculate_average_memory(&recent_window);
        let throughput_trend = self.calculate_throughput_trend(&recent_window);
        let queue_depth_trend = self.calculate_queue_trend(&recent_window);

        // Pattern detection logic
        let (pattern_type, confidence) = if self.is_burst_pattern(&recent_window, current_metrics) {
            (PatternType::BurstTraffic, 0.85)
        } else if self.is_batch_processing_pattern(&recent_window) {
            (PatternType::BatchProcessing, 0.8)
        } else if cpu_variance < self.analysis_config.steady_state_variance_threshold {
            (PatternType::SteadyState, 0.9)
        } else if throughput_trend > 0.2 {
            (PatternType::GrowingLoad, 0.75)
        } else if throughput_trend < -0.2 {
            (PatternType::DecliningLoad, 0.75)
        } else if self.is_cyclic_pattern(&historical_data) {
            (PatternType::CyclicPattern, 0.7)
        } else if avg_cpu < 20.0 && current_metrics.processing_queue_depth < 3 {
            (PatternType::IdlePeriod, 0.8)
        } else {
            (PatternType::SteadyState, 0.6)
        };

        // Calculate recommended resources
        let recommended_agents = self.calculate_recommended_agents(&pattern_type, current_metrics, avg_cpu);
        let memory_recommendation = self.calculate_memory_recommendation(&pattern_type, avg_memory, current_metrics);

        WorkloadPattern {
            pattern_type,
            confidence_score: confidence,
            duration_minutes: self.calculate_pattern_duration(&recent_window),
            characteristics: PatternCharacteristics {
                avg_cpu_utilization: avg_cpu,
                avg_memory_utilization: avg_memory,
                peak_queue_depth: recent_window.iter().map(|m| m.processing_queue_depth).max().unwrap_or(0),
                throughput_variance: self.calculate_throughput_variance(&recent_window),
                latency_p95_ms: self.calculate_latency_p95(&recent_window),
                predictability_score: self.calculate_predictability_score(&recent_window),
            },
            recommended_agents,
            memory_recommendation,
        }
    }

    fn is_burst_pattern(&self, window: &[WorkloadMetrics], current: &WorkloadMetrics) -> bool {
        if window.len() < 3 {
            return false;
        }

        let recent_throughput = current.documents_per_minute;
        let avg_throughput = window.iter().map(|m| m.documents_per_minute).sum::<f64>() / window.len() as f64;
        
        recent_throughput > avg_throughput * self.analysis_config.burst_detection_threshold
    }

    fn is_batch_processing_pattern(&self, window: &[WorkloadMetrics]) -> bool {
        let queue_depths: Vec<usize> = window.iter().map(|m| m.processing_queue_depth).collect();
        
        // Look for periodic spikes in queue depth
        let max_queue = queue_depths.iter().max().unwrap_or(&0);
        let avg_queue = queue_depths.iter().sum::<usize>() as f64 / queue_depths.len() as f64;
        
        *max_queue as f64 > avg_queue * 3.0 && queue_depths.iter().filter(|&&q| q == 0).count() > queue_depths.len() / 3
    }

    fn is_cyclic_pattern(&self, historical_data: &[WorkloadMetrics]) -> bool {
        if historical_data.len() < 20 {
            return false;
        }
        
        // Simple cycle detection using autocorrelation
        let throughputs: Vec<f64> = historical_data.iter().map(|m| m.documents_per_minute).collect();
        let correlation = self.calculate_autocorrelation(&throughputs, 10);
        
        correlation > 0.7
    }

    fn calculate_autocorrelation(&self, data: &[f64], lag: usize) -> f64 {
        if data.len() <= lag {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        
        if variance == 0.0 {
            return 1.0;
        }
        
        let covariance = data[..data.len()-lag].iter().zip(data[lag..].iter())
            .map(|(x, y)| (x - mean) * (y - mean))
            .sum::<f64>() / (data.len() - lag) as f64;
        
        covariance / variance
    }

    fn calculate_average_cpu(&self, window: &[WorkloadMetrics]) -> f64 {
        let total_cpu: f64 = window.iter().map(|m| {
            m.cpu_utilization_per_core.iter().sum::<f64>() / m.cpu_utilization_per_core.len() as f64
        }).sum();
        
        total_cpu / window.len() as f64
    }

    fn calculate_cpu_variance(&self, window: &[WorkloadMetrics], avg: f64) -> f64 {
        let variance: f64 = window.iter().map(|m| {
            let cpu = m.cpu_utilization_per_core.iter().sum::<f64>() / m.cpu_utilization_per_core.len() as f64;
            (cpu - avg).powi(2)
        }).sum();
        
        variance / window.len() as f64
    }

    fn calculate_average_memory(&self, window: &[WorkloadMetrics]) -> f64 {
        let total_memory: f64 = window.iter().map(|m| {
            (m.memory_utilization_breakdown.rust_processes_gb + 
             m.memory_utilization_breakdown.python_processes_gb + 
             m.memory_utilization_breakdown.shared_memory_gb) / 128.0 * 100.0
        }).sum();
        
        total_memory / window.len() as f64
    }

    fn calculate_throughput_trend(&self, window: &[WorkloadMetrics]) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }
        
        let first_half = &window[..window.len()/2];
        let second_half = &window[window.len()/2..];
        
        let first_avg = first_half.iter().map(|m| m.documents_per_minute).sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().map(|m| m.documents_per_minute).sum::<f64>() / second_half.len() as f64;
        
        (second_avg - first_avg) / first_avg.max(1.0)
    }

    fn calculate_queue_trend(&self, window: &[WorkloadMetrics]) -> f64 {
        if window.len() < 2 {
            return 0.0;
        }
        
        let first_half = &window[..window.len()/2];
        let second_half = &window[window.len()/2..];
        
        let first_avg = first_half.iter().map(|m| m.processing_queue_depth).sum::<usize>() as f64 / first_half.len() as f64;
        let second_avg = second_half.iter().map(|m| m.processing_queue_depth).sum::<usize>() as f64 / second_half.len() as f64;
        
        (second_avg - first_avg) / first_avg.max(1.0)
    }

    fn calculate_throughput_variance(&self, window: &[WorkloadMetrics]) -> f64 {
        let throughputs: Vec<f64> = window.iter().map(|m| m.documents_per_minute).collect();
        let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let variance = throughputs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / throughputs.len() as f64;
        variance.sqrt() / mean.max(1.0) // Coefficient of variation
    }

    fn calculate_latency_p95(&self, window: &[WorkloadMetrics]) -> f64 {
        let mut latencies: Vec<f64> = window.iter().map(|m| m.average_processing_time_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((latencies.len() as f64) * 0.95) as usize;
        latencies.get(index).copied().unwrap_or(0.0)
    }

    fn calculate_predictability_score(&self, window: &[WorkloadMetrics]) -> f64 {
        let throughput_variance = self.calculate_throughput_variance(window);
        let cpu_variance = self.calculate_cpu_variance(window, self.calculate_average_cpu(window));
        
        // Lower variance = higher predictability
        1.0 / (1.0 + throughput_variance + cpu_variance / 100.0)
    }

    fn calculate_pattern_duration(&self, window: &[WorkloadMetrics]) -> u64 {
        if window.len() < 2 {
            return 1;
        }
        
        let first_timestamp = window.first().unwrap().timestamp;
        let last_timestamp = window.last().unwrap().timestamp;
        
        (last_timestamp - first_timestamp) / (60 * 1000) // Convert to minutes
    }

    fn calculate_recommended_agents(&self, pattern: &PatternType, current: &WorkloadMetrics, avg_cpu: f64) -> usize {
        let base_agents = match pattern {
            PatternType::BurstTraffic => {
                // Scale up aggressively for burst traffic
                let queue_factor = (current.processing_queue_depth as f64 / 5.0).min(3.0);
                (2.0 + queue_factor) as usize
            },
            PatternType::BatchProcessing => {
                // Moderate scaling for batch processing
                if current.processing_queue_depth > 10 { 4 } else { 3 }
            },
            PatternType::SteadyState => {
                // Scale based on CPU utilization
                if avg_cpu > 80.0 { 4 } else if avg_cpu > 60.0 { 3 } else { 2 }
            },
            PatternType::GrowingLoad => {
                // Proactive scaling for growing load
                4
            },
            PatternType::DecliningLoad => {
                // Scale down for declining load
                2
            },
            PatternType::IdlePeriod => {
                // Minimum agents for idle periods
                2
            },
            PatternType::CyclicPattern => {
                // Medium scaling for predictable cycles
                3
            },
        };

        // Apply constraints
        base_agents.min(12).max(2) // Max 12 agents for M3 Max, min 2
    }

    fn calculate_memory_recommendation(&self, pattern: &PatternType, avg_memory: f64, current: &WorkloadMetrics) -> MemoryAllocation {
        let base_memory = match pattern {
            PatternType::BurstTraffic => MemoryAllocation {
                rust_memory_gb: 70.0,
                python_memory_gb: 50.0,
                shared_memory_gb: 20.0,
                cache_memory_gb: 8.0,
            },
            PatternType::BatchProcessing => MemoryAllocation {
                rust_memory_gb: 65.0,
                python_memory_gb: 48.0,
                shared_memory_gb: 18.0,
                cache_memory_gb: 7.0,
            },
            _ => MemoryAllocation {
                rust_memory_gb: 60.0,
                python_memory_gb: 45.0,
                shared_memory_gb: 15.0,
                cache_memory_gb: 5.0,
            },
        };

        // Adjust based on current memory pressure
        let memory_pressure = avg_memory / 100.0;
        let adjustment_factor = if memory_pressure > 0.9 {
            0.9 // Reduce allocation if under pressure
        } else if memory_pressure < 0.5 {
            1.1 // Increase allocation if underutilized
        } else {
            1.0
        };

        MemoryAllocation {
            rust_memory_gb: (base_memory.rust_memory_gb * adjustment_factor).min(80.0),
            python_memory_gb: (base_memory.python_memory_gb * adjustment_factor).min(60.0),
            shared_memory_gb: (base_memory.shared_memory_gb * adjustment_factor).min(25.0),
            cache_memory_gb: (base_memory.cache_memory_gb * adjustment_factor).min(15.0),
        }
    }

    fn default_pattern(&self, metrics: &WorkloadMetrics) -> WorkloadPattern {
        WorkloadPattern {
            pattern_type: PatternType::SteadyState,
            confidence_score: 0.5,
            duration_minutes: 5,
            characteristics: PatternCharacteristics {
                avg_cpu_utilization: metrics.cpu_utilization_per_core.iter().sum::<f64>() / metrics.cpu_utilization_per_core.len() as f64,
                avg_memory_utilization: (metrics.memory_utilization_breakdown.rust_processes_gb + 
                                       metrics.memory_utilization_breakdown.python_processes_gb) / 128.0 * 100.0,
                peak_queue_depth: metrics.processing_queue_depth,
                throughput_variance: 0.0,
                latency_p95_ms: metrics.average_processing_time_ms,
                predictability_score: 0.5,
            },
            recommended_agents: 2,
            memory_recommendation: MemoryAllocation {
                rust_memory_gb: 60.0,
                python_memory_gb: 45.0,
                shared_memory_gb: 15.0,
                cache_memory_gb: 5.0,
            },
        }
    }

    pub fn get_pattern_history(&self) -> HashMap<String, WorkloadPattern> {
        self.pattern_cache.lock().unwrap().clone()
    }

    pub fn predict_future_workload(&mut self, minutes_ahead: u64) -> Option<WorkloadMetrics> {
        let history = self.metrics_history.lock().unwrap();
        let historical_data: Vec<WorkloadMetrics> = history.iter().cloned().collect();
        drop(history);

        self.machine_learning_model.predict_workload(&historical_data)
    }

    // Get workload statistics for reporting
    pub fn get_workload_statistics(&self) -> WorkloadStatistics {
        let history = self.metrics_history.lock().unwrap();
        let data: Vec<WorkloadMetrics> = history.iter().cloned().collect();
        drop(history);

        if data.is_empty() {
            return WorkloadStatistics::default();
        }

        let total_docs = data.iter().map(|m| m.documents_per_minute).sum::<f64>();
        let avg_cpu = self.calculate_average_cpu(&data);
        let avg_memory = self.calculate_average_memory(&data);
        let max_queue = data.iter().map(|m| m.processing_queue_depth).max().unwrap_or(0);
        let avg_latency = data.iter().map(|m| m.average_processing_time_ms).sum::<f64>() / data.len() as f64;

        WorkloadStatistics {
            total_documents_processed: total_docs,
            average_cpu_utilization: avg_cpu,
            average_memory_utilization: avg_memory,
            peak_queue_depth: max_queue,
            average_processing_latency_ms: avg_latency,
            uptime_minutes: self.calculate_pattern_duration(&data),
        }
    }
}

impl PatternType {
    pub fn pattern_name(&self) -> &'static str {
        match self {
            PatternType::BurstTraffic => "burst_traffic",
            PatternType::SteadyState => "steady_state",
            PatternType::BatchProcessing => "batch_processing",
            PatternType::IdlePeriod => "idle_period",
            PatternType::GrowingLoad => "growing_load",
            PatternType::DecliningLoad => "declining_load",
            PatternType::CyclicPattern => "cyclic_pattern",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadStatistics {
    pub total_documents_processed: f64,
    pub average_cpu_utilization: f64,
    pub average_memory_utilization: f64,
    pub peak_queue_depth: usize,
    pub average_processing_latency_ms: f64,
    pub uptime_minutes: u64,
}

impl Default for WorkloadStatistics {
    fn default() -> Self {
        Self {
            total_documents_processed: 0.0,
            average_cpu_utilization: 0.0,
            average_memory_utilization: 0.0,
            peak_queue_depth: 0,
            average_processing_latency_ms: 0.0,
            uptime_minutes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_workload_analyzer_creation() {
        let config = AnalysisConfig::default();
        let analyzer = WorkloadAnalyzer::new(config);
        assert_eq!(analyzer.metrics_history.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_burst_pattern_detection() {
        let config = AnalysisConfig::default();
        let mut analyzer = WorkloadAnalyzer::new(config);

        // Create burst traffic scenario
        let burst_metrics = WorkloadMetrics {
            processing_queue_depth: 15,
            average_processing_time_ms: 200.0,
            documents_per_minute: 50.0,
            cpu_utilization_per_core: vec![90.0; 16],
            memory_utilization_breakdown: MemoryBreakdown {
                rust_processes_gb: 65.0,
                python_processes_gb: 48.0,
                shared_memory_gb: 18.0,
                cached_data_gb: 5.0,
                free_memory_gb: 12.0,
                memory_fragmentation_percent: 10.0,
            },
            ipc_throughput_mbps: 1500.0,
            error_rate_percent: 0.5,
            agent_utilization: HashMap::new(),
            timestamp: 1234567890,
            peak_hour_indicator: true,
        };

        let pattern = analyzer.analyze_workload(burst_metrics).await;
        assert!(matches!(pattern.pattern_type, PatternType::BurstTraffic));
        assert!(pattern.confidence_score > 0.8);
    }

    #[tokio::test]
    async fn test_memory_recommendation() {
        let config = AnalysisConfig::default();
        let analyzer = WorkloadAnalyzer::new(config);

        let metrics = WorkloadMetrics {
            processing_queue_depth: 5,
            average_processing_time_ms: 150.0,
            documents_per_minute: 25.0,
            cpu_utilization_per_core: vec![60.0; 16],
            memory_utilization_breakdown: MemoryBreakdown {
                rust_processes_gb: 60.0,
                python_processes_gb: 45.0,
                shared_memory_gb: 15.0,
                cached_data_gb: 5.0,
                free_memory_gb: 23.0,
                memory_fragmentation_percent: 5.0,
            },
            ipc_throughput_mbps: 1200.0,
            error_rate_percent: 0.2,
            agent_utilization: HashMap::new(),
            timestamp: 1234567890,
            peak_hour_indicator: false,
        };

        let memory_rec = analyzer.calculate_memory_recommendation(&PatternType::SteadyState, 75.0, &metrics);
        assert!(memory_rec.rust_memory_gb >= 50.0);
        assert!(memory_rec.python_memory_gb >= 40.0);
        assert!(memory_rec.shared_memory_gb >= 10.0);
    }
}