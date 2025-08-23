/*! 
# Performance Analysis Module

Advanced performance analysis algorithms for the RAN-LLM pipeline,
including trend detection, bottleneck identification, and automated optimization suggestions.
*/

use crate::{Result, PipelineError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Performance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub timestamp: u64,
    pub memory_trend: TrendDirection,
    pub cpu_trend: TrendDirection, 
    pub bottlenecks: Vec<String>,
    pub recommendations: Vec<String>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Calculate trend direction from a series of values
pub fn calculate_trend(values: &[f64]) -> TrendDirection {
    if values.len() < 3 {
        return TrendDirection::Stable;
    }
    
    let first_half = &values[..values.len()/2];
    let second_half = &values[values.len()/2..];
    
    let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
    let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
    
    let change_percent = ((second_avg - first_avg) / first_avg) * 100.0;
    
    if change_percent > 5.0 {
        TrendDirection::Increasing
    } else if change_percent < -5.0 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

/// Get current timestamp in seconds
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Real-time bottleneck detector
pub struct BottleneckDetector {
    memory_samples: VecDeque<f64>,
    cpu_samples: VecDeque<f64>,
    throughput_samples: VecDeque<f64>,
    max_samples: usize,
}

impl BottleneckDetector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            memory_samples: VecDeque::with_capacity(max_samples),
            cpu_samples: VecDeque::with_capacity(max_samples),
            throughput_samples: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }
    
    pub fn add_sample(&mut self, memory_usage: f64, cpu_usage: f64, throughput: f64) {
        // Add new samples
        self.memory_samples.push_back(memory_usage);
        self.cpu_samples.push_back(cpu_usage);
        self.throughput_samples.push_back(throughput);
        
        // Maintain max sample size
        if self.memory_samples.len() > self.max_samples {
            self.memory_samples.pop_front();
            self.cpu_samples.pop_front();
            self.throughput_samples.pop_front();
        }
    }
    
    pub fn analyze(&self) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        
        if self.memory_samples.len() < 10 {
            return bottlenecks;
        }
        
        // Memory bottleneck detection
        let recent_memory = self.memory_samples.iter().rev().take(10).collect::<Vec<_>>();
        let avg_memory = recent_memory.iter().map(|&&x| x).sum::<f64>() / recent_memory.len() as f64;
        
        if avg_memory > 85.0 {
            bottlenecks.push("Critical memory pressure detected".to_string());
        } else if avg_memory > 70.0 {
            bottlenecks.push("High memory usage trend".to_string());
        }
        
        // CPU bottleneck detection  
        let recent_cpu = self.cpu_samples.iter().rev().take(10).collect::<Vec<_>>();
        let avg_cpu = recent_cpu.iter().map(|&&x| x).sum::<f64>() / recent_cpu.len() as f64;
        
        if avg_cpu > 90.0 {
            bottlenecks.push("CPU at maximum capacity".to_string());
        } else if avg_cpu > 75.0 {
            bottlenecks.push("High CPU utilization detected".to_string());
        }
        
        // Throughput bottleneck detection
        let recent_throughput = self.throughput_samples.iter().rev().take(10).collect::<Vec<_>>();
        let throughput_trend = calculate_trend(&recent_throughput.iter().map(|&&x| x).collect::<Vec<_>>());
        
        if matches!(throughput_trend, TrendDirection::Decreasing) {
            bottlenecks.push("Declining throughput performance".to_string());
        }
        
        // Correlation analysis
        if avg_memory > 70.0 && avg_cpu > 70.0 {
            bottlenecks.push("Resource contention detected - both memory and CPU under pressure".to_string());
        }
        
        bottlenecks
    }
    
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        let bottlenecks = self.analyze();
        let mut recommendations = Vec::new();
        
        for bottleneck in bottlenecks {
            match bottleneck.as_str() {
                s if s.contains("memory") => {
                    recommendations.push("Consider increasing memory allocation or implementing memory pooling".to_string());
                    recommendations.push("Review data structures for memory efficiency".to_string());
                },
                s if s.contains("CPU") => {
                    recommendations.push("Optimize CPU-intensive algorithms or distribute load".to_string());
                    recommendations.push("Consider horizontal scaling with additional workers".to_string());
                },
                s if s.contains("throughput") => {
                    recommendations.push("Analyze pipeline stages for processing bottlenecks".to_string());
                    recommendations.push("Consider batch size optimization".to_string());
                },
                s if s.contains("contention") => {
                    recommendations.push("Implement resource scheduling to reduce contention".to_string());
                    recommendations.push("Consider asynchronous processing patterns".to_string());
                },
                _ => {}
            }
        }
        
        // Add M3 Max specific optimizations
        if !recommendations.is_empty() {
            recommendations.push("Leverage M3 Max unified memory architecture for better performance".to_string());
            recommendations.push("Consider using Metal Performance Shaders for compute-intensive tasks".to_string());
        }
        
        recommendations
    }
}

/// Performance regression analyzer
pub struct RegressionAnalyzer {
    baseline_window: usize,
    comparison_window: usize,
    significance_threshold: f64,
}

impl RegressionAnalyzer {
    pub fn new() -> Self {
        Self {
            baseline_window: 100,
            comparison_window: 20,
            significance_threshold: 10.0, // 10% change threshold
        }
    }
    
    pub fn analyze_regression(&self, metrics: &[f64]) -> Option<RegressionResult> {
        if metrics.len() < self.baseline_window + self.comparison_window {
            return None;
        }
        
        // Take baseline from earlier data
        let baseline_start = metrics.len() - self.baseline_window - self.comparison_window;
        let baseline_end = baseline_start + self.baseline_window;
        let baseline = &metrics[baseline_start..baseline_end];
        
        // Take recent data for comparison
        let recent = &metrics[metrics.len() - self.comparison_window..];
        
        let baseline_avg = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        
        let percentage_change = ((recent_avg - baseline_avg) / baseline_avg) * 100.0;
        
        if percentage_change.abs() > self.significance_threshold {
            Some(RegressionResult {
                baseline_value: baseline_avg,
                current_value: recent_avg,
                percentage_change,
                is_regression: percentage_change < 0.0,
                confidence_score: self.calculate_confidence(baseline, recent),
            })
        } else {
            None
        }
    }
    
    fn calculate_confidence(&self, baseline: &[f64], recent: &[f64]) -> f64 {
        // Simple confidence calculation based on consistency
        let baseline_variance = self.calculate_variance(baseline);
        let recent_variance = self.calculate_variance(recent);
        
        // Higher confidence with lower variance
        let max_variance = baseline_variance.max(recent_variance);
        if max_variance < 0.1 {
            0.95
        } else if max_variance < 0.2 {
            0.85
        } else {
            0.70
        }
    }
    
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt() / mean // Coefficient of variation
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub baseline_value: f64,
    pub current_value: f64,
    pub percentage_change: f64,
    pub is_regression: bool,
    pub confidence_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trend_calculation() {
        let increasing = vec![10.0, 15.0, 20.0, 25.0, 30.0];
        assert!(matches!(calculate_trend(&increasing), TrendDirection::Increasing));
        
        let decreasing = vec![30.0, 25.0, 20.0, 15.0, 10.0];
        assert!(matches!(calculate_trend(&decreasing), TrendDirection::Decreasing));
        
        let stable = vec![20.0, 21.0, 19.0, 20.5, 19.5];
        assert!(matches!(calculate_trend(&stable), TrendDirection::Stable));
    }
    
    #[test]
    fn test_bottleneck_detector() {
        let mut detector = BottleneckDetector::new(50);
        
        // Add high memory usage samples
        for _ in 0..20 {
            detector.add_sample(85.0, 50.0, 100.0);
        }
        
        let bottlenecks = detector.analyze();
        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks.iter().any(|b| b.contains("memory")));
    }
    
    #[test]
    fn test_regression_analyzer() {
        let analyzer = RegressionAnalyzer::new();
        
        // Create data with regression
        let mut metrics = vec![];
        // Baseline performance
        for _ in 0..100 {
            metrics.push(100.0);
        }
        // Degraded performance
        for _ in 0..20 {
            metrics.push(80.0); // 20% regression
        }
        
        let result = analyzer.analyze_regression(&metrics);
        assert!(result.is_some());
        assert!(result.unwrap().is_regression);
    }
}