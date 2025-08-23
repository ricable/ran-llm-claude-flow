#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite Runner
Runs all integration tests and generates final report
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
import subprocess

# Import test modules
from test_framework import get_test_framework
from rust_python_ipc.test_ipc_communication import IPCCommunicationTests
from mcp_protocol.test_mcp_validation import MCPProtocolTests
from performance.test_performance_benchmarks import PerformanceBenchmarks
from model_switching.test_qwen3_variants import Qwen3ModelSwitchingTests

class ComprehensiveTestSuite:
    """Master test suite runner"""
    
    def __init__(self):
        self.framework = get_test_framework()
        self.start_time = time.time()
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration test suites"""
        self.framework.logger.info("🚀 Starting Comprehensive Integration Test Suite...")
        
        # Update GitHub issue
        await self._update_github_progress("RUNNING", "Starting comprehensive test execution")
        
        test_suites = {
            "ipc_communication": IPCCommunicationTests(),
            "mcp_protocol": MCPProtocolTests(), 
            "performance_benchmarks": PerformanceBenchmarks(),
            "model_switching": Qwen3ModelSwitchingTests()
        }
        
        suite_results = {}
        
        # Run each test suite
        for suite_name, suite_instance in test_suites.items():
            self.framework.logger.info(f"📋 Running {suite_name} tests...")
            
            suite_start_time = time.time()
            suite_results[suite_name] = await self._run_test_suite(suite_name, suite_instance)
            suite_duration = time.time() - suite_start_time
            
            suite_results[suite_name]["suite_duration_seconds"] = suite_duration
            
            # Update progress
            progress = len([s for s in suite_results if suite_results[s]["success"]]) / len(test_suites) * 100
            await self._update_github_progress("IN PROGRESS", f"{suite_name} completed - {progress:.0f}% total progress")
        
        # Generate comprehensive report
        final_report = await self._generate_final_report(suite_results)
        
        # Update GitHub with final results
        await self._update_github_final_results(final_report)
        
        return final_report
    
    async def _run_test_suite(self, suite_name: str, suite_instance: Any) -> Dict[str, Any]:
        """Run individual test suite"""
        suite_results = {
            "suite_name": suite_name,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "success": False,
            "test_results": {}
        }
        
        try:
            if suite_name == "ipc_communication":
                tests = [
                    ("shared_memory_communication", suite_instance.test_shared_memory_communication),
                    ("message_queue_communication", suite_instance.test_message_queue_communication),
                    ("process_coordination", suite_instance.test_process_coordination),
                    ("high_load_ipc", suite_instance.test_high_load_ipc)
                ]
            elif suite_name == "mcp_protocol":
                tests = [
                    ("mcp_server_startup", suite_instance.test_mcp_server_startup),
                    ("mcp_message_protocol", suite_instance.test_mcp_message_protocol),
                    ("rust_python_mcp_coordination", suite_instance.test_rust_python_mcp_coordination),
                    ("mcp_high_throughput", suite_instance.test_mcp_high_throughput),
                    ("mcp_error_handling", suite_instance.test_mcp_error_handling)
                ]
            elif suite_name == "performance_benchmarks":
                tests = [
                    ("baseline_performance", suite_instance.test_baseline_performance),
                    ("m3_max_optimized_performance", suite_instance.test_m3_max_optimized_performance),
                    ("parallel_processing_scaling", suite_instance.test_parallel_processing_scaling),
                    ("memory_intensive_workload", suite_instance.test_memory_intensive_workload),
                    ("sustained_performance", suite_instance.test_sustained_performance),
                    ("rust_python_performance_comparison", suite_instance.test_rust_python_performance_comparison)
                ]
            elif suite_name == "model_switching":
                tests = [
                    ("model_loading_performance", suite_instance.test_model_loading_performance),
                    ("intelligent_model_selection", suite_instance.test_intelligent_model_selection),
                    ("dynamic_switching_under_load", suite_instance.test_dynamic_switching_under_load),
                    ("memory_pressure_switching", suite_instance.test_memory_pressure_switching),
                    ("model_quality_vs_speed_tradeoff", suite_instance.test_model_quality_vs_speed_tradeoff)
                ]
            else:
                tests = []
            
            # Run all tests in the suite
            for test_name, test_func in tests:
                try:
                    result = await self.framework.run_test(test_func, f"{suite_name}_{test_name}")
                    suite_results["test_results"][test_name] = result.to_dict()
                    suite_results["tests_run"] += 1
                    
                    if result.status == "PASS":
                        suite_results["tests_passed"] += 1
                    else:
                        suite_results["tests_failed"] += 1
                        
                except Exception as e:
                    suite_results["test_results"][test_name] = {
                        "status": "FAIL",
                        "error": str(e)
                    }
                    suite_results["tests_run"] += 1
                    suite_results["tests_failed"] += 1
            
            # Suite succeeds if at least 80% of tests pass
            success_rate = suite_results["tests_passed"] / suite_results["tests_run"] if suite_results["tests_run"] > 0 else 0
            suite_results["success"] = success_rate >= 0.8
            suite_results["success_rate"] = success_rate
            
        except Exception as e:
            suite_results["error"] = str(e)
            suite_results["success"] = False
        
        return suite_results
    
    async def _generate_final_report(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        total_duration = time.time() - self.start_time
        
        # Aggregate statistics
        total_tests = sum(suite["tests_run"] for suite in suite_results.values())
        total_passed = sum(suite["tests_passed"] for suite in suite_results.values())
        total_failed = sum(suite["tests_failed"] for suite in suite_results.values())
        
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Performance analysis
        performance_metrics = self._extract_performance_metrics(suite_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(suite_results, performance_metrics)
        
        final_report = {
            "test_execution": {
                "start_time": self.start_time,
                "duration_seconds": total_duration,
                "total_test_suites": len(suite_results),
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "overall_success_rate": success_rate,
                "overall_status": "PASS" if success_rate >= 0.8 else "FAIL"
            },
            "suite_results": suite_results,
            "performance_analysis": performance_metrics,
            "system_info": self.framework.get_system_info(),
            "recommendations": recommendations,
            "validation_results": self._validate_requirements(suite_results)
        }
        
        # Save report to file
        report_path = Path("tests/integration/final_test_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2, default=str)
        
        return final_report
    
    def _extract_performance_metrics(self, suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from test results"""
        perf_suite = suite_results.get("performance_benchmarks", {})
        perf_tests = perf_suite.get("test_results", {})
        
        metrics = {
            "baseline_docs_per_hour": 0,
            "optimized_docs_per_hour": 0,
            "performance_improvement": 0,
            "memory_efficiency": 0,
            "meets_4x_target": False,
            "meets_5x_target": False
        }
        
        # Extract baseline performance
        baseline_test = perf_tests.get("baseline_performance", {})
        if baseline_test.get("status") == "PASS" and baseline_test.get("metrics"):
            metrics["baseline_docs_per_hour"] = baseline_test["metrics"].get("baseline_docs_per_hour", 0)
        
        # Extract optimized performance
        optimized_test = perf_tests.get("m3_max_optimized_performance", {})
        if optimized_test.get("status") == "PASS" and optimized_test.get("metrics"):
            metrics["optimized_docs_per_hour"] = optimized_test["metrics"].get("optimized_docs_per_hour", 0)
            metrics["performance_improvement"] = optimized_test["metrics"].get("performance_improvement_ratio", 0)
            metrics["meets_4x_target"] = optimized_test["metrics"].get("meets_4x_target", False)
            metrics["meets_5x_target"] = optimized_test["metrics"].get("meets_5x_target", False)
            metrics["memory_efficiency"] = optimized_test["metrics"].get("optimized_memory_efficiency", 0)
        
        return metrics
    
    def _generate_recommendations(self, suite_results: Dict[str, Any], performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check overall test success
        failed_suites = [name for name, suite in suite_results.items() if not suite.get("success", False)]
        if failed_suites:
            recommendations.append(f"Address failing test suites: {', '.join(failed_suites)}")
        
        # Performance recommendations
        if not performance_metrics.get("meets_4x_target", False):
            recommendations.append("Performance target of 4x improvement not met - review M3 Max optimization implementation")
        
        # IPC recommendations
        ipc_results = suite_results.get("ipc_communication", {})
        if not ipc_results.get("success", False):
            recommendations.append("IPC communication issues detected - verify Rust-Python process coordination")
        
        # MCP recommendations
        mcp_results = suite_results.get("mcp_protocol", {})
        if not mcp_results.get("success", False):
            recommendations.append("MCP protocol validation failed - check WebSocket communication and message handling")
        
        # Model switching recommendations
        model_results = suite_results.get("model_switching", {})
        if not model_results.get("success", False):
            recommendations.append("Model switching issues detected - verify Qwen3 variant coordination and memory management")
        
        return recommendations
    
    def _validate_requirements(self, suite_results: Dict[str, Any]) -> Dict[str, bool]:
        """Validate key project requirements"""
        return {
            "rust_python_ipc_working": suite_results.get("ipc_communication", {}).get("success", False),
            "mcp_protocol_compliant": suite_results.get("mcp_protocol", {}).get("success", False),
            "performance_targets_met": self._check_performance_targets(suite_results),
            "model_switching_functional": suite_results.get("model_switching", {}).get("success", False),
            "m3_max_optimization_working": self._check_m3_max_optimization(suite_results),
            "system_stable_under_load": self._check_system_stability(suite_results)
        }
    
    def _check_performance_targets(self, suite_results: Dict[str, Any]) -> bool:
        """Check if performance targets are met"""
        perf_suite = suite_results.get("performance_benchmarks", {})
        if not perf_suite.get("success", False):
            return False
        
        # Check for 4x improvement
        perf_tests = perf_suite.get("test_results", {})
        optimized_test = perf_tests.get("m3_max_optimized_performance", {})
        return optimized_test.get("metrics", {}).get("meets_4x_target", False)
    
    def _check_m3_max_optimization(self, suite_results: Dict[str, Any]) -> bool:
        """Check M3 Max optimization effectiveness"""
        perf_suite = suite_results.get("performance_benchmarks", {})
        return perf_suite.get("success", False)
    
    def _check_system_stability(self, suite_results: Dict[str, Any]) -> bool:
        """Check system stability under load"""
        # Check all high-load tests passed
        stability_tests = [
            suite_results.get("ipc_communication", {}).get("test_results", {}).get("high_load_ipc", {}),
            suite_results.get("mcp_protocol", {}).get("test_results", {}).get("mcp_high_throughput", {}),
            suite_results.get("model_switching", {}).get("test_results", {}).get("dynamic_switching_under_load", {})
        ]
        
        return all(test.get("status") == "PASS" for test in stability_tests)
    
    async def _update_github_progress(self, status: str, message: str):
        """Update GitHub issue with progress"""
        try:
            comment_body = f"🧪 Test Agent: {status} - {message}"
            subprocess.run([
                "gh", "issue", "comment", "1", "--body", comment_body
            ], check=True, capture_output=True)
        except Exception as e:
            self.framework.logger.error(f"Failed to update GitHub: {e}")
    
    async def _update_github_final_results(self, final_report: Dict[str, Any]):
        """Update GitHub with final test results"""
        try:
            execution = final_report["test_execution"]
            performance = final_report["performance_analysis"]
            validation = final_report["validation_results"]
            
            status_emoji = "✅" if execution["overall_status"] == "PASS" else "❌"
            
            comment_body = f"""{status_emoji} **Test Agent: COMPLETED** - Comprehensive Integration Testing Results

## 📊 Test Execution Summary
- **Total Tests**: {execution['total_tests']}
- **Tests Passed**: {execution['tests_passed']}
- **Tests Failed**: {execution['tests_failed']}
- **Success Rate**: {execution['overall_success_rate']:.1%}
- **Duration**: {execution['duration_seconds']:.1f} seconds

## 🚀 Performance Results
- **Baseline Performance**: {performance.get('baseline_docs_per_hour', 0):.1f} docs/hour
- **Optimized Performance**: {performance.get('optimized_docs_per_hour', 0):.1f} docs/hour
- **Improvement Ratio**: {performance.get('performance_improvement', 0):.1f}x
- **4x Target Met**: {"✅" if performance.get('meets_4x_target') else "❌"}
- **5x Target Met**: {"✅" if performance.get('meets_5x_target') else "❌"}

## 🔧 Component Validation
- **Rust-Python IPC**: {"✅" if validation.get('rust_python_ipc_working') else "❌"}
- **MCP Protocol**: {"✅" if validation.get('mcp_protocol_compliant') else "❌"}
- **Model Switching**: {"✅" if validation.get('model_switching_functional') else "❌"}
- **M3 Max Optimization**: {"✅" if validation.get('m3_max_optimization_working') else "❌"}
- **System Stability**: {"✅" if validation.get('system_stable_under_load') else "❌"}

## 📋 Recommendations
{chr(10).join(f"- {rec}" for rec in final_report.get('recommendations', []))}

**Full Report**: `tests/integration/final_test_report.json`"""
            
            subprocess.run([
                "gh", "issue", "comment", "1", "--body", comment_body
            ], check=True, capture_output=True)
            
        except Exception as e:
            self.framework.logger.error(f"Failed to update GitHub with final results: {e}")

async def main():
    """Main test execution"""
    test_suite = ComprehensiveTestSuite()
    
    try:
        final_report = await test_suite.run_all_tests()
        
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE INTEGRATION TEST RESULTS")
        print("="*80)
        
        execution = final_report["test_execution"]
        print(f"Overall Status: {execution['overall_status']}")
        print(f"Tests Passed: {execution['tests_passed']}/{execution['total_tests']}")
        print(f"Success Rate: {execution['overall_success_rate']:.1%}")
        print(f"Duration: {execution['duration_seconds']:.1f} seconds")
        
        performance = final_report["performance_analysis"]
        if performance.get("performance_improvement", 0) > 0:
            print(f"\nPerformance Improvement: {performance['performance_improvement']:.1f}x")
            print(f"4x Target Met: {performance.get('meets_4x_target', False)}")
            print(f"5x Target Met: {performance.get('meets_5x_target', False)}")
        
        validation = final_report["validation_results"]
        print(f"\nComponent Validation:")
        for component, status in validation.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        if final_report.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in final_report["recommendations"]:
                print(f"  - {rec}")
        
        print(f"\nDetailed report saved to: tests/integration/final_test_report.json")
        print("="*80)
        
        # Exit with appropriate code
        sys.exit(0 if execution["overall_status"] == "PASS" else 1)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())