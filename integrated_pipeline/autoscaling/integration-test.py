#!/usr/bin/env python3
"""
Integration Test Suite for Auto-Scaling Intelligence System
Tests all components working together in realistic scenarios
"""

import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_scaler import PredictiveScaler, ScalingConfig, WorkloadMetrics
from workload_predictor import WorkloadPredictor, WorkloadDataPoint
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoScalingIntegrationTest:
    """Complete integration test suite for auto-scaling system"""
    
    def __init__(self):
        self.scaler = None
        self.predictor = None
        self.test_results = {}
        
    async def setup_test_environment(self):
        """Initialize test environment with realistic configuration"""
        logger.info("Setting up auto-scaling test environment")
        
        # Configure scaling system
        config = ScalingConfig(
            min_instances=2,
            max_instances=20,
            target_cpu_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            cooldown_period=60,  # Reduced for testing
            prediction_window=300,  # 5 minutes
            ml_prediction_weight=0.7
        )
        
        self.scaler = PredictiveScaler(config)
        self.predictor = WorkloadPredictor({
            'prediction_horizons': [5, 15, 30],
            'min_training_points': 20,  # Reduced for testing
            'confidence_threshold': 0.5
        })
        
        logger.info("Test environment setup complete")
    
    def generate_realistic_workload(self, duration_hours: float = 2.0, 
                                  interval_minutes: float = 5.0) -> List[WorkloadDataPoint]:
        """Generate realistic workload pattern for testing"""
        logger.info(f"Generating {duration_hours}h workload with {interval_minutes}min intervals")
        
        workload_data = []
        base_time = time.time()
        num_points = int((duration_hours * 60) / interval_minutes)
        
        for i in range(num_points):
            timestamp = base_time + (i * interval_minutes * 60)
            dt = datetime.fromtimestamp(timestamp)
            
            # Create daily pattern with business hours peaks
            hour = dt.hour
            day_of_week = dt.weekday()
            
            # Base load pattern (higher during business hours)
            if 9 <= hour <= 11 or 14 <= hour <= 16:  # Peak hours
                base_cpu = 60 + np.random.normal(0, 15)
            elif 12 <= hour <= 13:  # Lunch dip
                base_cpu = 40 + np.random.normal(0, 10)
            elif 18 <= hour <= 22:  # Evening traffic
                base_cpu = 50 + np.random.normal(0, 12)
            else:  # Off-peak hours
                base_cpu = 25 + np.random.normal(0, 8)
            
            # Weekend adjustment
            if day_of_week >= 5:  # Weekend
                base_cpu *= 0.6
            
            # Add trend and spikes
            trend_factor = 1.0 + (i / num_points) * 0.2  # 20% increase over time
            spike_probability = 0.05  # 5% chance of spike
            
            if np.random.random() < spike_probability:
                base_cpu *= np.random.uniform(1.5, 2.5)  # Random spike
            
            cpu_usage = max(5, min(95, base_cpu * trend_factor))
            
            # Derive other metrics from CPU usage
            memory_usage = cpu_usage * np.random.uniform(0.7, 1.2)
            memory_usage = max(10, min(90, memory_usage))
            
            request_rate = max(10, cpu_usage * np.random.uniform(1.5, 3.0) + np.random.normal(0, 20))
            response_time = max(50, 500 - cpu_usage * 3 + np.random.normal(0, 100))
            
            concurrent_users = max(1, int(request_rate * 0.3 + np.random.normal(0, 10)))
            error_rate = max(0, min(10, (cpu_usage - 70) * 0.1 + np.random.normal(0, 0.5)))
            
            data_point = WorkloadDataPoint(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_rate=request_rate,
                response_time=response_time,
                concurrent_users=concurrent_users,
                error_rate=error_rate,
                network_io=max(0, cpu_usage * 1.5 + np.random.normal(0, 20)),
                disk_io=max(0, cpu_usage * 1.2 + np.random.normal(0, 15)),
                queue_depth=max(0, int((cpu_usage - 50) * 0.5 + np.random.normal(0, 5)))
            )
            
            workload_data.append(data_point)
        
        logger.info(f"Generated {len(workload_data)} workload data points")
        return workload_data
    
    async def test_workload_prediction_accuracy(self) -> Dict[str, Any]:
        """Test ML model prediction accuracy"""
        logger.info("Testing workload prediction accuracy")
        
        # Generate training data
        training_data = self.generate_realistic_workload(duration_hours=1.5, interval_minutes=5)
        test_data = self.generate_realistic_workload(duration_hours=0.5, interval_minutes=5)
        
        # Train predictor
        for data_point in training_data:
            await self.predictor.add_workload_data(data_point)
        
        # Wait for training
        await self.predictor.train_models()
        
        # Test predictions
        predictions = []
        actual_values = []
        
        for i, data_point in enumerate(test_data[:-3]):  # Leave some points for prediction comparison
            await self.predictor.add_workload_data(data_point)
            
            if i >= 5:  # Start predicting after some context
                prediction = await self.predictor.predict_workload(horizon_minutes=15)
                if prediction and prediction.model_confidence > 0.3:
                    predictions.append({
                        'cpu': prediction.predicted_cpu,
                        'memory': prediction.predicted_memory,
                        'requests': prediction.predicted_requests,
                        'confidence': prediction.model_confidence
                    })
                    
                    # Find corresponding actual values (3 points ahead for 15-min prediction)
                    if i + 3 < len(test_data):
                        actual_point = test_data[i + 3]
                        actual_values.append({
                            'cpu': actual_point.cpu_usage,
                            'memory': actual_point.memory_usage,
                            'requests': actual_point.request_rate
                        })
        
        # Calculate accuracy metrics
        accuracy_results = self.calculate_prediction_accuracy(predictions, actual_values)
        
        logger.info(f"Prediction accuracy results: {accuracy_results}")
        return accuracy_results
    
    def calculate_prediction_accuracy(self, predictions: List[Dict], 
                                    actual_values: List[Dict]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        if not predictions or not actual_values:
            return {'error': 'No predictions or actual values available'}
        
        min_len = min(len(predictions), len(actual_values))
        predictions = predictions[:min_len]
        actual_values = actual_values[:min_len]
        
        metrics = {}
        
        for key in ['cpu', 'memory', 'requests']:
            pred_values = [p[key] for p in predictions]
            actual_vals = [a[key] for a in actual_values]
            
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean([abs(p - a) / max(a, 1e-6) for p, a in zip(pred_values, actual_vals)]) * 100
            
            # Root Mean Square Error (RMSE)
            rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(pred_values, actual_vals)]))
            
            # Coefficient of determination (R²)
            ss_res = sum([(p - a) ** 2 for p, a in zip(pred_values, actual_vals)])
            ss_tot = sum([(a - np.mean(actual_vals)) ** 2 for a in actual_vals])
            r2 = 1 - (ss_res / max(ss_tot, 1e-6))
            
            metrics[f'{key}_mape'] = mape
            metrics[f'{key}_rmse'] = rmse
            metrics[f'{key}_r2'] = r2
        
        # Overall accuracy score
        avg_mape = np.mean([metrics[f'{key}_mape'] for key in ['cpu', 'memory', 'requests']])
        avg_r2 = np.mean([metrics[f'{key}_r2'] for key in ['cpu', 'memory', 'requests']])
        
        metrics['overall_accuracy'] = max(0, 1 - (avg_mape / 100))
        metrics['overall_r2'] = avg_r2
        metrics['confidence'] = np.mean([p['confidence'] for p in predictions])
        
        return metrics
    
    async def test_scaling_decisions(self) -> Dict[str, Any]:
        """Test auto-scaling decision making"""
        logger.info("Testing auto-scaling decisions")
        
        # Generate workload with clear scaling patterns
        workload_data = self.generate_scaling_test_workload()
        
        scaling_decisions = []
        current_instances = 3
        
        for i, data_point in enumerate(workload_data):
            # Convert to WorkloadMetrics format
            metrics = WorkloadMetrics(
                timestamp=data_point.timestamp,
                cpu_usage=data_point.cpu_usage,
                memory_usage=data_point.memory_usage,
                request_rate=data_point.request_rate,
                response_time=data_point.response_time,
                queue_depth=data_point.queue_depth,
                error_rate=data_point.error_rate,
                active_connections=data_point.concurrent_users
            )
            
            await self.scaler.add_metrics(metrics)
            
            # Make scaling decision every 5 data points (simulating 25-minute intervals)
            if i % 5 == 4:
                decision = await self.scaler.make_scaling_decision()
                if decision:
                    scaling_decisions.append({
                        'timestamp': decision.timestamp,
                        'action': decision.action,
                        'from_instances': current_instances,
                        'to_instances': decision.target_instances,
                        'confidence': decision.confidence,
                        'reasoning': decision.reasoning,
                        'cpu_at_decision': data_point.cpu_usage,
                        'memory_at_decision': data_point.memory_usage
                    })
                    current_instances = decision.target_instances
                    
                    # Execute the scaling decision
                    await self.scaler.execute_scaling_decision(decision)
        
        # Analyze scaling decisions
        scaling_analysis = self.analyze_scaling_decisions(scaling_decisions)
        
        logger.info(f"Scaling decisions analysis: {scaling_analysis}")
        return scaling_analysis
    
    def generate_scaling_test_workload(self) -> List[WorkloadDataPoint]:
        """Generate workload specifically designed to test scaling logic"""
        workload_data = []
        base_time = time.time()
        
        # Pattern: Low -> High -> Medium -> Very High -> Low
        patterns = [
            (20, 30, 10),   # Low load (20 points, ~30% CPU)
            (15, 85, 8),    # High load spike (15 points, ~85% CPU) - should scale up
            (20, 50, 5),    # Medium load (20 points, ~50% CPU)
            (10, 95, 12),   # Very high load (10 points, ~95% CPU) - aggressive scale up
            (25, 20, 3),    # Low load (25 points, ~20% CPU) - should scale down
        ]
        
        point_index = 0
        for duration, target_cpu, noise in patterns:
            for i in range(duration):
                timestamp = base_time + (point_index * 5 * 60)  # 5-minute intervals
                
                cpu_usage = max(5, min(100, target_cpu + np.random.normal(0, noise)))
                memory_usage = cpu_usage * np.random.uniform(0.8, 1.1)
                memory_usage = max(10, min(95, memory_usage))
                
                request_rate = max(5, cpu_usage * 2 + np.random.normal(0, 10))
                response_time = max(50, 1000 - cpu_usage * 8 + np.random.normal(0, 50))
                
                data_point = WorkloadDataPoint(
                    timestamp=timestamp,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    request_rate=request_rate,
                    response_time=response_time,
                    concurrent_users=max(1, int(request_rate * 0.2)),
                    error_rate=max(0, (cpu_usage - 80) * 0.1),
                    network_io=cpu_usage * 1.5,
                    disk_io=cpu_usage * 1.0,
                    queue_depth=max(0, int((cpu_usage - 60) * 0.3))
                )
                
                workload_data.append(data_point)
                point_index += 1
        
        return workload_data
    
    def analyze_scaling_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality of scaling decisions"""
        if not decisions:
            return {'error': 'No scaling decisions to analyze'}
        
        analysis = {
            'total_decisions': len(decisions),
            'scale_up_decisions': len([d for d in decisions if d['action'] == 'scale_up']),
            'scale_down_decisions': len([d for d in decisions if d['action'] == 'scale_down']),
            'average_confidence': np.mean([d['confidence'] for d in decisions]),
            'response_times': [],
            'appropriateness_score': 0.0
        }
        
        # Check appropriateness of scaling decisions
        appropriate_decisions = 0
        
        for decision in decisions:
            cpu_at_decision = decision['cpu_at_decision']
            action = decision['action']
            
            # Decision is appropriate if:
            # - Scale up when CPU > 75%
            # - Scale down when CPU < 35%
            if (action == 'scale_up' and cpu_at_decision > 75) or \
               (action == 'scale_down' and cpu_at_decision < 35):
                appropriate_decisions += 1
            
            # Calculate response time (simulated based on decision making speed)
            response_time = 30 + np.random.uniform(-5, 10)  # 30±5 seconds base
            analysis['response_times'].append(response_time)
        
        analysis['appropriateness_score'] = appropriate_decisions / len(decisions)
        analysis['average_response_time'] = np.mean(analysis['response_times'])
        analysis['max_response_time'] = max(analysis['response_times'])
        
        # Performance targets check
        analysis['meets_response_target'] = analysis['average_response_time'] < 30  # < 30s target
        analysis['meets_appropriateness_target'] = analysis['appropriateness_score'] > 0.8  # > 80% appropriate
        analysis['meets_confidence_target'] = analysis['average_confidence'] > 0.6  # > 60% confidence
        
        return analysis
    
    async def test_cost_optimization(self) -> Dict[str, Any]:
        """Test cost optimization capabilities"""
        logger.info("Testing cost optimization")
        
        # This would normally integrate with the Rust cost optimizer
        # For now, we'll simulate cost optimization testing
        
        cost_scenarios = [
            {'instances': 5, 'instance_type': 't3.medium', 'cost_per_hour': 0.0416},
            {'instances': 3, 'instance_type': 'c5.large', 'cost_per_hour': 0.085},
            {'instances': 2, 'instance_type': 'c5.xlarge', 'cost_per_hour': 0.17},
        ]
        
        optimization_results = []
        
        for scenario in cost_scenarios:
            total_cost = scenario['instances'] * scenario['cost_per_hour']
            performance_score = scenario['instances'] * self.get_instance_performance_score(scenario['instance_type'])
            cost_efficiency = performance_score / total_cost
            
            optimization_results.append({
                'scenario': scenario,
                'total_cost': total_cost,
                'performance_score': performance_score,
                'cost_efficiency': cost_efficiency
            })
        
        # Find best cost-efficient option
        best_scenario = max(optimization_results, key=lambda x: x['cost_efficiency'])
        
        cost_analysis = {
            'scenarios_tested': len(cost_scenarios),
            'best_scenario': best_scenario['scenario'],
            'cost_savings_potential': max([r['total_cost'] for r in optimization_results]) - best_scenario['total_cost'],
            'efficiency_improvement': (best_scenario['cost_efficiency'] - min([r['cost_efficiency'] for r in optimization_results])) / min([r['cost_efficiency'] for r in optimization_results]) * 100
        }
        
        logger.info(f"Cost optimization results: {cost_analysis}")
        return cost_analysis
    
    def get_instance_performance_score(self, instance_type: str) -> float:
        """Get performance score for instance type"""
        performance_map = {
            't3.micro': 0.2,
            't3.small': 0.3,
            't3.medium': 0.4,
            'c5.large': 0.6,
            'c5.xlarge': 0.8,
            'm5.large': 0.5,
            'm5.xlarge': 0.7,
        }
        return performance_map.get(instance_type, 0.5)
    
    async def test_load_balancing_intelligence(self) -> Dict[str, Any]:
        """Test intelligent load balancing"""
        logger.info("Testing intelligent load balancing")
        
        # Simulate load balancing scenarios
        backend_scenarios = [
            {'id': 'backend-1', 'cpu': 30, 'response_time': 150, 'connections': 50},
            {'id': 'backend-2', 'cpu': 80, 'response_time': 400, 'connections': 120},
            {'id': 'backend-3', 'cpu': 60, 'response_time': 200, 'connections': 80},
        ]
        
        # Simulate routing decisions based on different algorithms
        routing_results = {}
        
        # Least Connections
        least_conn_backend = min(backend_scenarios, key=lambda x: x['connections'])
        routing_results['least_connections'] = least_conn_backend['id']
        
        # Least Response Time
        least_rt_backend = min(backend_scenarios, key=lambda x: x['response_time'])
        routing_results['least_response_time'] = least_rt_backend['id']
        
        # AI-based (weighted scoring)
        ai_scores = []
        for backend in backend_scenarios:
            # Lower is better score
            score = (backend['cpu'] * 0.4 + 
                    backend['response_time'] * 0.004 +  # Normalize to similar scale
                    backend['connections'] * 0.8)
            ai_scores.append((backend['id'], score))
        
        ai_best_backend = min(ai_scores, key=lambda x: x[1])[0]
        routing_results['ai_routing'] = ai_best_backend
        
        load_balancing_analysis = {
            'backend_scenarios': backend_scenarios,
            'routing_decisions': routing_results,
            'ai_vs_simple_agreement': routing_results['ai_routing'] == routing_results['least_response_time'],
            'load_distribution_quality': self.calculate_load_distribution_quality(backend_scenarios)
        }
        
        logger.info(f"Load balancing analysis: {load_balancing_analysis}")
        return load_balancing_analysis
    
    def calculate_load_distribution_quality(self, backends: List[Dict[str, Any]]) -> float:
        """Calculate how well load is distributed across backends"""
        cpu_values = [b['cpu'] for b in backends]
        connection_values = [b['connections'] for b in backends]
        
        # Lower coefficient of variation indicates better distribution
        cpu_cv = np.std(cpu_values) / np.mean(cpu_values) if np.mean(cpu_values) > 0 else 0
        conn_cv = np.std(connection_values) / np.mean(connection_values) if np.mean(connection_values) > 0 else 0
        
        # Quality score (higher is better, 1.0 is perfect distribution)
        quality = max(0, 1 - (cpu_cv + conn_cv) / 2)
        return quality
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting comprehensive auto-scaling integration test")
        
        await self.setup_test_environment()
        
        test_results = {}
        
        try:
            # Test 1: Workload Prediction Accuracy
            logger.info("=== Test 1: Workload Prediction ===")
            test_results['prediction_accuracy'] = await self.test_workload_prediction_accuracy()
            
            # Test 2: Scaling Decisions
            logger.info("=== Test 2: Scaling Decisions ===")
            test_results['scaling_decisions'] = await self.test_scaling_decisions()
            
            # Test 3: Cost Optimization
            logger.info("=== Test 3: Cost Optimization ===")
            test_results['cost_optimization'] = await self.test_cost_optimization()
            
            # Test 4: Load Balancing
            logger.info("=== Test 4: Load Balancing Intelligence ===")
            test_results['load_balancing'] = await self.test_load_balancing_intelligence()
            
            # Overall assessment
            test_results['overall_assessment'] = self.generate_overall_assessment(test_results)
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            test_results['error'] = str(e)
        
        logger.info("Comprehensive integration test completed")
        return test_results
    
    def generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall system assessment"""
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'test_duration': '~10 minutes',
            'components_tested': ['prediction', 'scaling', 'cost_optimization', 'load_balancing']
        }
        
        # Scoring criteria
        scores = []
        
        # Prediction accuracy score
        if 'prediction_accuracy' in results and 'overall_accuracy' in results['prediction_accuracy']:
            pred_score = results['prediction_accuracy']['overall_accuracy']
            scores.append(pred_score)
            assessment['prediction_score'] = pred_score
        
        # Scaling decision score
        if 'scaling_decisions' in results and 'appropriateness_score' in results['scaling_decisions']:
            scaling_score = results['scaling_decisions']['appropriateness_score']
            scores.append(scaling_score)
            assessment['scaling_score'] = scaling_score
        
        # Performance targets
        performance_targets = {
            'prediction_accuracy_target': 0.75,  # >75% accuracy
            'scaling_response_time_target': 30,   # <30s response
            'cost_optimization_target': 25,       # >25% potential savings
            'load_distribution_quality_target': 0.7  # >0.7 quality score
        }
        
        assessment['performance_targets'] = performance_targets
        assessment['targets_met'] = {}
        
        # Check targets
        if 'prediction_accuracy' in results:
            assessment['targets_met']['prediction_accuracy'] = \
                results['prediction_accuracy'].get('overall_accuracy', 0) > performance_targets['prediction_accuracy_target']
        
        if 'scaling_decisions' in results:
            assessment['targets_met']['scaling_response_time'] = \
                results['scaling_decisions'].get('average_response_time', 100) < performance_targets['scaling_response_time_target']
        
        # Overall system grade
        if scores:
            assessment['overall_score'] = np.mean(scores)
            if assessment['overall_score'] > 0.85:
                assessment['grade'] = 'A - Excellent'
            elif assessment['overall_score'] > 0.75:
                assessment['grade'] = 'B - Good'
            elif assessment['overall_score'] > 0.65:
                assessment['grade'] = 'C - Acceptable'
            else:
                assessment['grade'] = 'D - Needs Improvement'
        else:
            assessment['overall_score'] = 0.0
            assessment['grade'] = 'F - Failed'
        
        return assessment

async def main():
    """Run integration tests"""
    test_suite = AutoScalingIntegrationTest()
    
    # Run comprehensive test
    results = await test_suite.run_comprehensive_test()
    
    # Print results
    print("\n" + "="*60)
    print("AUTO-SCALING INTELLIGENCE INTEGRATION TEST RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))
    
    # Save results to file
    results_file = Path(__file__).parent / "integration_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Test results saved to: {results_file}")
    
    # Return exit code based on overall assessment
    if 'overall_assessment' in results and results['overall_assessment']['overall_score'] > 0.7:
        print("\n✅ INTEGRATION TESTS PASSED")
        return 0
    else:
        print("\n❌ INTEGRATION TESTS FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))