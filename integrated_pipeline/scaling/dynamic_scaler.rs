// Dynamic Scaling Engine for Hybrid Rust-Python Pipeline
// Phase 2 MCP Advanced Features - Agent 1 Implementation
// Automatically scales processing agents based on workload and resource utilization

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::time::sleep;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetrics {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub queue_depth: usize,
    pub throughput_docs_per_hour: f64,
    pub ipc_latency_ms: f64,
    pub active_agents: usize,
    pub timestamp: u64,
    pub rust_memory_gb: f64,
    pub python_memory_gb: f64,
    pub shared_memory_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub action: ScalingAction,
    pub target_agents: usize,
    pub reasoning: String,
    pub confidence: f64,
    pub estimated_impact: f64,
    pub memory_reallocation: Option<MemoryReallocation>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
    Maintain,
    Rebalance,
    OptimizeMemory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReallocation {
    pub rust_memory_gb: f64,
    pub python_memory_gb: f64,
    pub shared_memory_gb: f64,
    pub total_allocated: f64,
}

pub struct DynamicScaler {
    metrics_history: Arc<Mutex<VecDeque<ScalingMetrics>>>,
    scaling_decisions: Arc<Mutex<VecDeque<ScalingDecision>>>,
    config: ScalingConfig,
    last_scaling_action: Arc<Mutex<Instant>>,
    current_agents: Arc<Mutex<usize>>,
    performance_predictor: PerformancePredictor,
}

#[derive(Debug, Clone)]
pub struct ScalingConfig {
    pub max_agents: usize,
    pub min_agents: usize,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub cooldown_duration: Duration,
    pub prediction_window_size: usize,
    pub memory_constraint_gb: f64,
    pub min_throughput_docs_per_hour: f64,
    pub max_ipc_latency_ms: f64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            max_agents: 12,    // Conservative max for M3 Max
            min_agents: 2,
            target_cpu_utilization: 85.0,
            target_memory_utilization: 90.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 50.0,
            cooldown_duration: Duration::from_secs(30),
            prediction_window_size: 20,
            memory_constraint_gb: 128.0,
            min_throughput_docs_per_hour: 25.0,
            max_ipc_latency_ms: 5.0,
        }
    }
}

pub struct PerformancePredictor {
    cpu_predictions: VecDeque<f64>,
    memory_predictions: VecDeque<f64>,
    throughput_predictions: VecDeque<f64>,
    learning_rate: f64,
}

impl PerformancePredictor {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            cpu_predictions: VecDeque::with_capacity(10),
            memory_predictions: VecDeque::with_capacity(10),
            throughput_predictions: VecDeque::with_capacity(10),
            learning_rate,
        }
    }

    pub fn predict_performance(&mut self, historical_metrics: &[ScalingMetrics], agent_count: usize) -> ScalingMetrics {
        if historical_metrics.is_empty() {
            return ScalingMetrics {
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
                queue_depth: 0,
                throughput_docs_per_hour: 0.0,
                ipc_latency_ms: 0.0,
                active_agents: agent_count,
                timestamp: 0,
                rust_memory_gb: 60.0,
                python_memory_gb: 45.0,
                shared_memory_gb: 15.0,
            };
        }

        let latest = &historical_metrics[historical_metrics.len() - 1];
        
        // Linear scaling approximation with diminishing returns
        let scaling_factor = agent_count as f64 / latest.active_agents as f64;
        let efficiency_factor = 1.0 / (1.0 + (scaling_factor - 1.0) * 0.1); // Diminishing returns
        
        ScalingMetrics {
            cpu_utilization: (latest.cpu_utilization * scaling_factor).min(100.0),
            memory_utilization: (latest.memory_utilization + (scaling_factor - 1.0) * 10.0).min(100.0),
            queue_depth: (latest.queue_depth as f64 / scaling_factor).max(0.0) as usize,
            throughput_docs_per_hour: latest.throughput_docs_per_hour * scaling_factor * efficiency_factor,
            ipc_latency_ms: latest.ipc_latency_ms * (1.0 + (scaling_factor - 1.0) * 0.05),
            active_agents: agent_count,
            timestamp: latest.timestamp,
            rust_memory_gb: 60.0 + (agent_count - 2) as f64 * 5.0,
            python_memory_gb: 45.0 + (agent_count - 2) as f64 * 3.0,
            shared_memory_gb: 15.0 + (agent_count - 2) as f64 * 1.0,
        }
    }
}

impl DynamicScaler {
    pub fn new(config: ScalingConfig) -> Self {
        Self {
            metrics_history: Arc::new(Mutex::new(VecDeque::with_capacity(config.prediction_window_size))),
            scaling_decisions: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            config,
            last_scaling_action: Arc::new(Mutex::new(Instant::now())),
            current_agents: Arc::new(Mutex::new(2)), // Start with 2 agents
            performance_predictor: PerformancePredictor::new(0.1),
        }
    }

    pub async fn analyze_and_scale(&mut self, current_metrics: ScalingMetrics) -> Option<ScalingDecision> {
        // Add current metrics to history
        {
            let mut history = self.metrics_history.lock().unwrap();
            if history.len() >= self.config.prediction_window_size {
                history.pop_front();
            }
            history.push_back(current_metrics.clone());
        }

        // Check cooldown period
        let last_action = *self.last_scaling_action.lock().unwrap();
        if last_action.elapsed() < self.config.cooldown_duration {
            return None;
        }

        let current_agents = *self.current_agents.lock().unwrap();
        let decision = self.make_scaling_decision(&current_metrics, current_agents).await;

        if let Some(ref decision) = decision {
            if !matches!(decision.action, ScalingAction::Maintain) {
                *self.last_scaling_action.lock().unwrap() = Instant::now();
                *self.current_agents.lock().unwrap() = decision.target_agents;

                // Store decision in history
                let mut decisions = self.scaling_decisions.lock().unwrap();
                if decisions.len() >= 100 {
                    decisions.pop_front();
                }
                decisions.push_back(decision.clone());
            }
        }

        decision
    }

    async fn make_scaling_decision(&mut self, metrics: &ScalingMetrics, current_agents: usize) -> Option<ScalingDecision> {
        let history = self.metrics_history.lock().unwrap();
        let historical_metrics: Vec<ScalingMetrics> = history.iter().cloned().collect();
        drop(history);

        // Calculate load indicators
        let cpu_pressure = metrics.cpu_utilization / self.config.target_cpu_utilization;
        let memory_pressure = metrics.memory_utilization / self.config.target_memory_utilization;
        let queue_pressure = if metrics.queue_depth > 0 { metrics.queue_depth as f64 / 10.0 } else { 0.0 };
        let throughput_pressure = self.config.min_throughput_docs_per_hour / metrics.throughput_docs_per_hour.max(1.0);
        let latency_pressure = metrics.ipc_latency_ms / self.config.max_ipc_latency_ms;

        // Combined pressure score
        let overall_pressure = (cpu_pressure + memory_pressure + queue_pressure + throughput_pressure + latency_pressure) / 5.0;

        // Memory constraint check
        let total_memory_used = metrics.rust_memory_gb + metrics.python_memory_gb + metrics.shared_memory_gb;
        let memory_constraint_reached = total_memory_used > self.config.memory_constraint_gb * 0.95;

        let (action, target_agents, reasoning, confidence) = if overall_pressure > 1.2 && !memory_constraint_reached && current_agents < self.config.max_agents {
            // Scale up
            let new_agents = (current_agents + 1).min(self.config.max_agents);
            (ScalingAction::ScaleUp, new_agents, 
             format!("High pressure detected ({}): CPU {:.1}%, Memory {:.1}%, Queue depth {}, Throughput {:.1} docs/hr", 
                     overall_pressure, metrics.cpu_utilization, metrics.memory_utilization, metrics.queue_depth, metrics.throughput_docs_per_hour),
             0.8)
        } else if overall_pressure < 0.6 && current_agents > self.config.min_agents {
            // Scale down
            let new_agents = (current_agents - 1).max(self.config.min_agents);
            (ScalingAction::ScaleDown, new_agents,
             format!("Low pressure detected ({}): underutilized resources", overall_pressure),
             0.7)
        } else if memory_constraint_reached && metrics.memory_utilization > 85.0 {
            // Memory optimization
            (ScalingAction::OptimizeMemory, current_agents,
             "Memory constraint reached, optimizing allocation".to_string(),
             0.9)
        } else if metrics.cpu_utilization < 30.0 && metrics.memory_utilization > 80.0 {
            // Resource rebalancing
            (ScalingAction::Rebalance, current_agents,
             "CPU underutilized while memory is high, rebalancing workload".to_string(),
             0.6)
        } else {
            // Maintain current scale
            return None;
        };

        // Predict performance with target agent count
        let predicted_metrics = self.performance_predictor.predict_performance(&historical_metrics, target_agents);
        let estimated_impact = (predicted_metrics.throughput_docs_per_hour - metrics.throughput_docs_per_hour) / metrics.throughput_docs_per_hour.max(1.0);

        // Calculate memory reallocation if needed
        let memory_reallocation = if matches!(action, ScalingAction::OptimizeMemory | ScalingAction::Rebalance) {
            Some(self.calculate_optimal_memory_allocation(target_agents, metrics))
        } else if matches!(action, ScalingAction::ScaleUp | ScalingAction::ScaleDown) {
            Some(MemoryReallocation {
                rust_memory_gb: 60.0 + (target_agents - 2) as f64 * 4.0,
                python_memory_gb: 45.0 + (target_agents - 2) as f64 * 2.5,
                shared_memory_gb: 15.0 + (target_agents - 2) as f64 * 0.8,
                total_allocated: 120.0 + (target_agents - 2) as f64 * 7.3,
            })
        } else {
            None
        };

        Some(ScalingDecision {
            action,
            target_agents,
            reasoning,
            confidence,
            estimated_impact,
            memory_reallocation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    fn calculate_optimal_memory_allocation(&self, agents: usize, metrics: &ScalingMetrics) -> MemoryReallocation {
        let base_rust = 60.0;
        let base_python = 45.0;
        let base_shared = 15.0;
        let total_available = self.config.memory_constraint_gb - 8.0; // Reserve 8GB for system

        // Dynamic allocation based on current usage patterns
        let rust_factor = 1.0 + (metrics.cpu_utilization / 100.0) * 0.2;
        let python_factor = 1.0 + (metrics.throughput_docs_per_hour / 30.0) * 0.15;
        let shared_factor = 1.0 + (metrics.ipc_latency_ms / 10.0) * 0.1;

        let rust_memory = (base_rust * rust_factor).min(total_available * 0.5);
        let python_memory = (base_python * python_factor).min(total_available * 0.4);
        let shared_memory = (base_shared * shared_factor).min(total_available * 0.15);

        let total_requested = rust_memory + python_memory + shared_memory;

        // Scale down proportionally if over budget
        let scale_factor = if total_requested > total_available {
            total_available / total_requested
        } else {
            1.0
        };

        MemoryReallocation {
            rust_memory_gb: rust_memory * scale_factor,
            python_memory_gb: python_memory * scale_factor,
            shared_memory_gb: shared_memory * scale_factor,
            total_allocated: total_requested * scale_factor,
        }
    }

    pub fn get_current_agents(&self) -> usize {
        *self.current_agents.lock().unwrap()
    }

    pub fn get_scaling_history(&self) -> Vec<ScalingDecision> {
        self.scaling_decisions.lock().unwrap().iter().cloned().collect()
    }

    pub fn get_metrics_history(&self) -> Vec<ScalingMetrics> {
        self.metrics_history.lock().unwrap().iter().cloned().collect()
    }

    // Emergency scaling when system is under severe stress
    pub async fn emergency_scale(&mut self, metrics: &ScalingMetrics) -> Option<ScalingDecision> {
        if metrics.memory_utilization > 95.0 || metrics.cpu_utilization > 98.0 || metrics.ipc_latency_ms > 50.0 {
            let current_agents = *self.current_agents.lock().unwrap();
            
            if current_agents > self.config.min_agents {
                *self.current_agents.lock().unwrap() = self.config.min_agents;
                *self.last_scaling_action.lock().unwrap() = Instant::now();
                
                return Some(ScalingDecision {
                    action: ScalingAction::ScaleDown,
                    target_agents: self.config.min_agents,
                    reasoning: "EMERGENCY: System under severe stress, scaling to minimum".to_string(),
                    confidence: 1.0,
                    estimated_impact: -0.5, // Expect performance drop but system stability
                    memory_reallocation: Some(MemoryReallocation {
                        rust_memory_gb: 60.0,
                        python_memory_gb: 45.0,
                        shared_memory_gb: 15.0,
                        total_allocated: 120.0,
                    }),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                });
            }
        }
        None
    }

    // Calculate scaling effectiveness based on historical data
    pub fn calculate_scaling_effectiveness(&self) -> f64 {
        let decisions = self.scaling_decisions.lock().unwrap();
        if decisions.len() < 2 {
            return 0.0;
        }

        let mut total_effectiveness = 0.0;
        let mut count = 0;

        for decision in decisions.iter() {
            if decision.estimated_impact != 0.0 {
                let effectiveness = decision.confidence * decision.estimated_impact.abs();
                total_effectiveness += effectiveness;
                count += 1;
            }
        }

        if count > 0 {
            total_effectiveness / count as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dynamic_scaler_creation() {
        let config = ScalingConfig::default();
        let scaler = DynamicScaler::new(config);
        assert_eq!(scaler.get_current_agents(), 2);
    }

    #[tokio::test]
    async fn test_scaling_decision_scale_up() {
        let config = ScalingConfig::default();
        let mut scaler = DynamicScaler::new(config);
        
        let high_pressure_metrics = ScalingMetrics {
            cpu_utilization: 90.0,
            memory_utilization: 85.0,
            queue_depth: 15,
            throughput_docs_per_hour: 20.0,
            ipc_latency_ms: 8.0,
            active_agents: 2,
            timestamp: 1234567890,
            rust_memory_gb: 60.0,
            python_memory_gb: 45.0,
            shared_memory_gb: 15.0,
        };

        let decision = scaler.analyze_and_scale(high_pressure_metrics).await;
        assert!(decision.is_some());
        
        let decision = decision.unwrap();
        assert!(matches!(decision.action, ScalingAction::ScaleUp));
        assert_eq!(decision.target_agents, 3);
    }
}