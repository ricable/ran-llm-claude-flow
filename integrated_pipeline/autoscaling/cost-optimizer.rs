// Cost Optimization Engine for Auto-Scaling
// Minimizes cloud costs while maintaining performance SLAs

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::time::{interval, sleep};
use log::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceType {
    pub name: String,
    pub vcpus: u32,
    pub memory_gb: u32,
    pub cost_per_hour: f64,
    pub performance_score: f64, // Normalized performance metric
    pub spot_available: bool,
    pub spot_discount: f64, // Percentage discount for spot instances
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_vcpus: u32,
    pub min_memory_gb: u32,
    pub target_performance: f64,
    pub sla_response_time_ms: u32,
    pub sla_availability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimizationConfig {
    pub cost_weight: f64, // 0.0 to 1.0, higher means more cost-focused
    pub performance_weight: f64, // 0.0 to 1.0, higher means more performance-focused
    pub spot_instance_tolerance: f64, // Risk tolerance for spot instances
    pub max_cost_per_hour: f64,
    pub optimization_interval_seconds: u64,
    pub historical_window_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMetrics {
    pub timestamp: u64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub request_rate: f64,
    pub response_time_ms: f64,
    pub error_rate: f64,
    pub cost_per_hour: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub timestamp: u64,
    pub current_cost: f64,
    pub recommended_instances: Vec<InstanceRecommendation>,
    pub estimated_cost_savings: f64,
    pub estimated_performance_impact: f64,
    pub confidence_score: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceRecommendation {
    pub instance_type: String,
    pub count: u32,
    pub use_spot: bool,
    pub estimated_cost: f64,
    pub expected_performance: f64,
}

pub struct CostOptimizer {
    config: CostOptimizationConfig,
    instance_types: Vec<InstanceType>,
    workload_history: VecDeque<WorkloadMetrics>,
    optimization_history: VecDeque<OptimizationRecommendation>,
    current_instances: HashMap<String, u32>,
    sla_violations: VecDeque<(u64, String)>, // timestamp, violation description
}

impl Default for CostOptimizationConfig {
    fn default() -> Self {
        Self {
            cost_weight: 0.6,
            performance_weight: 0.4,
            spot_instance_tolerance: 0.3,
            max_cost_per_hour: 100.0,
            optimization_interval_seconds: 300, // 5 minutes
            historical_window_hours: 24,
        }
    }
}

impl CostOptimizer {
    pub fn new(config: CostOptimizationConfig) -> Self {
        let instance_types = Self::initialize_instance_types();
        
        Self {
            config,
            instance_types,
            workload_history: VecDeque::new(),
            optimization_history: VecDeque::new(),
            current_instances: HashMap::new(),
            sla_violations: VecDeque::new(),
        }
    }

    fn initialize_instance_types() -> Vec<InstanceType> {
        vec![
            InstanceType {
                name: "t3.micro".to_string(),
                vcpus: 2,
                memory_gb: 1,
                cost_per_hour: 0.0104,
                performance_score: 0.2,
                spot_available: true,
                spot_discount: 0.7,
            },
            InstanceType {
                name: "t3.small".to_string(),
                vcpus: 2,
                memory_gb: 2,
                cost_per_hour: 0.0208,
                performance_score: 0.3,
                spot_available: true,
                spot_discount: 0.68,
            },
            InstanceType {
                name: "t3.medium".to_string(),
                vcpus: 2,
                memory_gb: 4,
                cost_per_hour: 0.0416,
                performance_score: 0.4,
                spot_available: true,
                spot_discount: 0.65,
            },
            InstanceType {
                name: "c5.large".to_string(),
                vcpus: 2,
                memory_gb: 4,
                cost_per_hour: 0.085,
                performance_score: 0.6,
                spot_available: true,
                spot_discount: 0.6,
            },
            InstanceType {
                name: "c5.xlarge".to_string(),
                vcpus: 4,
                memory_gb: 8,
                cost_per_hour: 0.17,
                performance_score: 0.8,
                spot_available: true,
                spot_discount: 0.58,
            },
            InstanceType {
                name: "m5.large".to_string(),
                vcpus: 2,
                memory_gb: 8,
                cost_per_hour: 0.096,
                performance_score: 0.5,
                spot_available: true,
                spot_discount: 0.63,
            },
            InstanceType {
                name: "m5.xlarge".to_string(),
                vcpus: 4,
                memory_gb: 16,
                cost_per_hour: 0.192,
                performance_score: 0.7,
                spot_available: true,
                spot_discount: 0.6,
            },
            InstanceType {
                name: "r5.large".to_string(),
                vcpus: 2,
                memory_gb: 16,
                cost_per_hour: 0.126,
                performance_score: 0.6,
                spot_available: true,
                spot_discount: 0.65,
            },
        ]
    }

    pub async fn add_workload_metrics(&mut self, metrics: WorkloadMetrics) {
        self.workload_history.push_back(metrics);
        
        // Keep only data within the historical window
        let cutoff_time = self.current_timestamp() - (self.config.historical_window_hours as u64 * 3600);
        while let Some(front) = self.workload_history.front() {
            if front.timestamp < cutoff_time {
                self.workload_history.pop_front();
            } else {
                break;
            }
        }

        debug!("Added workload metrics, history size: {}", self.workload_history.len());
    }

    pub fn calculate_resource_requirements(&self) -> ResourceRequirements {
        if self.workload_history.is_empty() {
            return ResourceRequirements {
                min_vcpus: 2,
                min_memory_gb: 4,
                target_performance: 0.5,
                sla_response_time_ms: 500,
                sla_availability: 0.99,
            };
        }

        // Analyze recent workload patterns
        let recent_metrics: Vec<_> = self.workload_history
            .iter()
            .rev()
            .take(12) // Last hour (5-minute intervals)
            .collect();

        let avg_cpu = recent_metrics.iter().map(|m| m.cpu_utilization).sum::<f64>() / recent_metrics.len() as f64;
        let avg_memory = recent_metrics.iter().map(|m| m.memory_utilization).sum::<f64>() / recent_metrics.len() as f64;
        let max_response_time = recent_metrics.iter().map(|m| m.response_time_ms).fold(0.0, f64::max);
        let avg_request_rate = recent_metrics.iter().map(|m| m.request_rate).sum::<f64>() / recent_metrics.len() as f64;

        // Calculate requirements with safety margins
        let cpu_margin = 1.3; // 30% headroom
        let memory_margin = 1.25; // 25% headroom
        
        let min_vcpus = ((avg_cpu * cpu_margin / 80.0) * 2.0).ceil() as u32; // Assuming 80% target utilization
        let min_memory_gb = ((avg_memory * memory_margin / 80.0) * 4.0).ceil() as u32;
        
        // Performance target based on request rate and complexity
        let target_performance = (avg_request_rate / 1000.0).min(1.0);
        
        // SLA requirements based on current performance
        let sla_response_time_ms = (max_response_time * 1.2) as u32; // 20% buffer
        let sla_availability = if recent_metrics.iter().any(|m| m.error_rate > 1.0) { 0.995 } else { 0.99 };

        ResourceRequirements {
            min_vcpus: min_vcpus.max(1),
            min_memory_gb: min_memory_gb.max(1),
            target_performance,
            sla_response_time_ms: sla_response_time_ms.max(200),
            sla_availability,
        }
    }

    pub fn optimize_instance_mix(&self, requirements: &ResourceRequirements) -> Vec<InstanceRecommendation> {
        let mut recommendations = Vec::new();
        
        // Find suitable instance types
        let suitable_instances: Vec<_> = self.instance_types
            .iter()
            .filter(|instance| {
                instance.vcpus >= requirements.min_vcpus &&
                instance.memory_gb >= requirements.min_memory_gb
            })
            .collect();

        if suitable_instances.is_empty() {
            warn!("No suitable instance types found for requirements: {:?}", requirements);
            return recommendations;
        }

        // Multi-objective optimization: minimize cost while maintaining performance
        for instance in &suitable_instances {
            // Calculate how many instances we need
            let cpu_instances = (requirements.min_vcpus as f64 / instance.vcpus as f64).ceil() as u32;
            let memory_instances = (requirements.min_memory_gb as f64 / instance.memory_gb as f64).ceil() as u32;
            let required_instances = cpu_instances.max(memory_instances);

            // Consider performance requirements
            let performance_instances = if instance.performance_score < requirements.target_performance {
                ((requirements.target_performance / instance.performance_score) * required_instances as f64).ceil() as u32
            } else {
                required_instances
            };

            let final_instance_count = performance_instances.max(1);

            // Calculate costs for on-demand and spot instances
            let on_demand_cost = instance.cost_per_hour * final_instance_count as f64;
            let spot_cost = if instance.spot_available {
                instance.cost_per_hour * (1.0 - instance.spot_discount) * final_instance_count as f64
            } else {
                on_demand_cost
            };

            // Risk assessment for spot instances
            let spot_viable = instance.spot_available && 
                             self.config.spot_instance_tolerance > 0.2 &&
                             spot_cost < self.config.max_cost_per_hour;

            // Performance score calculation
            let total_performance = instance.performance_score * final_instance_count as f64;
            let performance_efficiency = total_performance / requirements.target_performance.max(0.1);

            // Add on-demand recommendation
            recommendations.push(InstanceRecommendation {
                instance_type: instance.name.clone(),
                count: final_instance_count,
                use_spot: false,
                estimated_cost: on_demand_cost,
                expected_performance: performance_efficiency,
            });

            // Add spot recommendation if viable
            if spot_viable {
                recommendations.push(InstanceRecommendation {
                    instance_type: instance.name.clone(),
                    count: final_instance_count,
                    use_spot: true,
                    estimated_cost: spot_cost,
                    expected_performance: performance_efficiency * 0.95, // Slight penalty for spot risk
                });
            }
        }

        // Sort by cost-performance ratio
        recommendations.sort_by(|a, b| {
            let ratio_a = a.estimated_cost / a.expected_performance.max(0.1);
            let ratio_b = b.estimated_cost / b.expected_performance.max(0.1);
            ratio_a.partial_cmp(&ratio_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep only top 5 recommendations
        recommendations.truncate(5);
        recommendations
    }

    pub async fn generate_optimization_recommendation(&mut self) -> Option<OptimizationRecommendation> {
        if self.workload_history.len() < 3 {
            debug!("Insufficient workload history for optimization");
            return None;
        }

        let requirements = self.calculate_resource_requirements();
        let recommendations = self.optimize_instance_mix(&requirements);
        
        if recommendations.is_empty() {
            return None;
        }

        // Calculate current cost
        let current_cost = self.calculate_current_cost();
        
        // Select best recommendation based on configuration weights
        let best_recommendation = recommendations
            .iter()
            .min_by(|a, b| {
                let score_a = self.calculate_optimization_score(a);
                let score_b = self.calculate_optimization_score(b);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })?
            .clone();

        let estimated_savings = current_cost - best_recommendation.estimated_cost;
        let performance_impact = best_recommendation.expected_performance - 1.0;

        // Calculate confidence based on data quality and historical accuracy
        let confidence_score = self.calculate_confidence_score();

        let reasoning = format!(
            "Recommended {} {} instances ({}). Current cost: ${:.2}/h, New cost: ${:.2}/h, Savings: ${:.2}/h, Performance impact: {:.1}%",
            best_recommendation.count,
            best_recommendation.instance_type,
            if best_recommendation.use_spot { "spot" } else { "on-demand" },
            current_cost,
            best_recommendation.estimated_cost,
            estimated_savings,
            performance_impact * 100.0
        );

        let optimization = OptimizationRecommendation {
            timestamp: self.current_timestamp(),
            current_cost,
            recommended_instances: vec![best_recommendation],
            estimated_cost_savings: estimated_savings,
            estimated_performance_impact: performance_impact,
            confidence_score,
            reasoning,
        };

        self.optimization_history.push_back(optimization.clone());
        
        // Keep optimization history within limits
        if self.optimization_history.len() > 100 {
            self.optimization_history.pop_front();
        }

        info!("Generated optimization recommendation: {}", optimization.reasoning);
        Some(optimization)
    }

    fn calculate_optimization_score(&self, recommendation: &InstanceRecommendation) -> f64 {
        // Multi-objective score combining cost and performance
        let cost_score = recommendation.estimated_cost / self.config.max_cost_per_hour;
        let performance_score = 1.0 / recommendation.expected_performance.max(0.1);
        
        // Risk penalty for spot instances
        let risk_penalty = if recommendation.use_spot { 0.1 } else { 0.0 };
        
        self.config.cost_weight * cost_score + 
        self.config.performance_weight * performance_score + 
        risk_penalty
    }

    fn calculate_confidence_score(&self) -> f64 {
        let history_quality = (self.workload_history.len() as f64 / 100.0).min(1.0);
        let data_freshness = if let Some(latest) = self.workload_history.back() {
            let age_minutes = (self.current_timestamp() - latest.timestamp) / 60;
            (1.0 / (1.0 + age_minutes as f64 / 60.0)).max(0.1)
        } else {
            0.1
        };
        
        let sla_stability = if self.sla_violations.len() < 5 { 0.9 } else { 0.6 };
        
        (history_quality * 0.4 + data_freshness * 0.4 + sla_stability * 0.2).min(0.95)
    }

    fn calculate_current_cost(&self) -> f64 {
        let mut total_cost = 0.0;
        
        for (instance_type, count) in &self.current_instances {
            if let Some(instance_info) = self.instance_types.iter().find(|i| i.name == *instance_type) {
                total_cost += instance_info.cost_per_hour * (*count as f64);
            }
        }
        
        total_cost
    }

    pub fn update_current_instances(&mut self, instances: HashMap<String, u32>) {
        self.current_instances = instances;
        info!("Updated current instances: {:?}", self.current_instances);
    }

    pub fn record_sla_violation(&mut self, violation: String) {
        self.sla_violations.push_back((self.current_timestamp(), violation));
        
        // Keep only recent violations (last 24 hours)
        let cutoff_time = self.current_timestamp() - 86400;
        while let Some((timestamp, _)) = self.sla_violations.front() {
            if *timestamp < cutoff_time {
                self.sla_violations.pop_front();
            } else {
                break;
            }
        }
        
        warn!("SLA violation recorded: {}", self.sla_violations.back().unwrap().1);
    }

    pub fn get_cost_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        metrics.insert("current_cost_per_hour".to_string(), self.calculate_current_cost());
        
        if let Some(latest) = self.optimization_history.back() {
            metrics.insert("potential_savings_per_hour".to_string(), latest.estimated_cost_savings);
            metrics.insert("optimization_confidence".to_string(), latest.confidence_score);
            
            let daily_savings = latest.estimated_cost_savings * 24.0;
            let monthly_savings = daily_savings * 30.0;
            
            metrics.insert("potential_daily_savings".to_string(), daily_savings);
            metrics.insert("potential_monthly_savings".to_string(), monthly_savings);
        }
        
        let total_instances: u32 = self.current_instances.values().sum();
        metrics.insert("total_instances".to_string(), total_instances as f64);
        
        let recent_violations = self.sla_violations.len() as f64;
        metrics.insert("recent_sla_violations".to_string(), recent_violations);
        
        metrics
    }

    pub async fn run_optimization_loop(&mut self) {
        let mut interval = interval(Duration::from_secs(self.config.optimization_interval_seconds));
        
        info!("Starting cost optimization loop with {}s intervals", 
               self.config.optimization_interval_seconds);
        
        loop {
            interval.tick().await;
            
            if let Some(recommendation) = self.generate_optimization_recommendation().await {
                if recommendation.estimated_cost_savings > 1.0 && recommendation.confidence_score > 0.7 {
                    info!("High-confidence optimization available: {}", recommendation.reasoning);
                    // Here you would integrate with your deployment system
                    // to apply the optimization
                }
            }
            
            // Periodic cleanup
            if self.optimization_history.len() % 20 == 0 {
                self.cleanup_old_data();
            }
        }
    }

    fn cleanup_old_data(&mut self) {
        let cutoff_time = self.current_timestamp() - (self.config.historical_window_hours as u64 * 3600);
        
        while let Some(front) = self.optimization_history.front() {
            if front.timestamp < cutoff_time {
                self.optimization_history.pop_front();
            } else {
                break;
            }
        }
        
        debug!("Cleaned up old optimization data");
    }

    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_cost_optimizer_initialization() {
        let config = CostOptimizationConfig::default();
        let optimizer = CostOptimizer::new(config);
        
        assert!(!optimizer.instance_types.is_empty());
        assert_eq!(optimizer.workload_history.len(), 0);
    }

    #[tokio::test]
    async fn test_workload_metrics_processing() {
        let config = CostOptimizationConfig::default();
        let mut optimizer = CostOptimizer::new(config);
        
        let metrics = WorkloadMetrics {
            timestamp: 1000,
            cpu_utilization: 75.0,
            memory_utilization: 80.0,
            request_rate: 100.0,
            response_time_ms: 200.0,
            error_rate: 1.0,
            cost_per_hour: 10.0,
        };
        
        optimizer.add_workload_metrics(metrics).await;
        assert_eq!(optimizer.workload_history.len(), 1);
    }

    #[tokio::test]
    async fn test_resource_requirements_calculation() {
        let config = CostOptimizationConfig::default();
        let optimizer = CostOptimizer::new(config);
        
        let requirements = optimizer.calculate_resource_requirements();
        
        assert!(requirements.min_vcpus >= 1);
        assert!(requirements.min_memory_gb >= 1);
        assert!(requirements.sla_availability > 0.9);
    }

    #[tokio::test]
    async fn test_instance_optimization() {
        let config = CostOptimizationConfig::default();
        let optimizer = CostOptimizer::new(config);
        
        let requirements = ResourceRequirements {
            min_vcpus: 4,
            min_memory_gb: 8,
            target_performance: 0.7,
            sla_response_time_ms: 300,
            sla_availability: 0.99,
        };
        
        let recommendations = optimizer.optimize_instance_mix(&requirements);
        assert!(!recommendations.is_empty());
        
        // Verify all recommendations meet requirements
        for rec in &recommendations {
            let instance = optimizer.instance_types
                .iter()
                .find(|i| i.name == rec.instance_type)
                .unwrap();
            
            assert!(instance.vcpus * rec.count >= requirements.min_vcpus);
            assert!(instance.memory_gb * rec.count >= requirements.min_memory_gb);
        }
    }
}

// Example usage
pub async fn example_cost_optimization() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let config = CostOptimizationConfig {
        cost_weight: 0.7,
        performance_weight: 0.3,
        spot_instance_tolerance: 0.4,
        max_cost_per_hour: 50.0,
        optimization_interval_seconds: 300,
        historical_window_hours: 24,
    };
    
    let mut optimizer = CostOptimizer::new(config);
    
    // Simulate adding current instance configuration
    let mut current_instances = HashMap::new();
    current_instances.insert("m5.large".to_string(), 3);
    current_instances.insert("c5.xlarge".to_string(), 2);
    optimizer.update_current_instances(current_instances);
    
    // Simulate workload metrics
    for i in 0..10 {
        let metrics = WorkloadMetrics {
            timestamp: 1000 + i * 300, // 5-minute intervals
            cpu_utilization: 60.0 + (i as f64 * 5.0),
            memory_utilization: 70.0 + (i as f64 * 3.0),
            request_rate: 150.0 + (i as f64 * 10.0),
            response_time_ms: 180.0 + (i as f64 * 20.0),
            error_rate: 0.5,
            cost_per_hour: 12.5,
        };
        
        optimizer.add_workload_metrics(metrics).await;
        sleep(Duration::from_millis(100)).await;
    }
    
    // Generate optimization recommendation
    if let Some(recommendation) = optimizer.generate_optimization_recommendation().await {
        info!("Optimization recommendation: {}", recommendation.reasoning);
        info!("Estimated savings: ${:.2}/hour", recommendation.estimated_cost_savings);
    }
    
    // Get cost metrics
    let metrics = optimizer.get_cost_metrics();
    for (key, value) in metrics {
        info!("{}: {:.2}", key, value);
    }
    
    Ok(())
}