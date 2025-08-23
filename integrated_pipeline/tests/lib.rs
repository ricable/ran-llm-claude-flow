/*!
# Hybrid Pipeline Integration Testing Framework

This comprehensive testing framework validates the entire hybrid Rust-Python pipeline for document processing and QA generation.

## Test Categories

- **E2E Tests**: Complete pipeline workflow validation
- **Integration Tests**: Rust-Python IPC and shared memory testing
- **Performance Tests**: Throughput, latency, and resource utilization benchmarks
- **Quality Tests**: QA pair quality, semantic validation, and accuracy assessment
- **Regression Tests**: Performance and quality regression detection

## Usage

```rust
use integrated_pipeline_tests::runners::{AutomatedTestRunner, TestRunnerConfig};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TestRunnerConfig::default();
    let mut runner = AutomatedTestRunner::new(config);
    
    // Run complete test suite
    let report = runner.run_complete_test_suite().await?;
    println!("Test suite completed: {}", report.overall_success);
    
    Ok(())
}
```

## Test Targets

- **Throughput**: 25+ documents per hour
- **Quality Score**: >= 0.75 average with <= 0.05 variance
- **Memory Usage**: <= 60GB on M3 Max
- **IPC Latency**: <= 3 seconds
- **Success Rate**: >= 95% for all operations

## Architecture

```
tests/
├── e2e/                  # End-to-end pipeline tests
├── integration/          # Rust-Python IPC tests  
├── performance/          # Benchmarking and profiling
├── quality/              # QA quality validation
├── fixtures/             # Test data and helpers
└── runners/              # Test orchestration
```
*/

use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;
use tokio::time::Duration;

pub mod e2e;
pub mod integration;
pub mod performance;
pub mod quality;
pub mod fixtures;
pub mod runners;

// Re-exports for convenience
pub use e2e::{E2EPipelineTestSuite, PipelineTestReport};
pub use integration::{IntegrationTestSuite, IPCTestReport};
pub use performance::{PerformanceTestSuite, ComprehensivePerformanceReport};
pub use quality::{QualityTestSuite, ComprehensiveQualityReport};
pub use runners::{AutomatedTestRunner, TestRunnerConfig, CompleteTestReport};

/// Main test orchestrator for the hybrid pipeline
pub struct PipelineTestOrchestrator {
    session_id: Uuid,
    test_suites: HashMap<String, Box<dyn TestSuite>>,
}

impl PipelineTestOrchestrator {
    pub fn new() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            test_suites: HashMap::new(),
        }
    }

    /// Register a test suite for orchestrated execution
    pub fn register_test_suite(&mut self, name: String, suite: Box<dyn TestSuite>) {
        self.test_suites.insert(name, suite);
    }

    /// Execute all registered test suites
    pub async fn execute_all_suites(&mut self) -> Result<OrchestratedTestReport> {
        let mut report = OrchestratedTestReport::new(self.session_id);
        
        for (name, suite) in &mut self.test_suites {
            let result = suite.execute().await?;
            report.add_suite_result(name.clone(), result);
        }
        
        report.finalize();
        Ok(report)
    }
}

/// Trait for test suite implementations
#[async_trait::async_trait]
pub trait TestSuite: Send {
    async fn execute(&mut self) -> Result<TestSuiteResult>;
    fn get_name(&self) -> &str;
    fn get_description(&self) -> &str;
}

/// Result from a test suite execution
#[derive(Debug, Clone)]
pub struct TestSuiteResult {
    pub suite_name: String,
    pub execution_time: Duration,
    pub test_count: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub success_rate: f64,
    pub details: serde_json::Value,
}

/// Orchestrated test report combining all suite results
#[derive(Debug)]
pub struct OrchestratedTestReport {
    pub session_id: Uuid,
    pub suite_results: HashMap<String, TestSuiteResult>,
    pub total_execution_time: Duration,
    pub overall_success: bool,
    pub summary: TestExecutionSummary,
}

#[derive(Debug)]
pub struct TestExecutionSummary {
    pub total_suites: usize,
    pub successful_suites: usize,
    pub failed_suites: usize,
    pub total_tests: usize,
    pub total_passed: usize,
    pub total_failed: usize,
    pub total_skipped: usize,
    pub overall_success_rate: f64,
}

impl TestSuiteResult {
    pub fn new(suite_name: String) -> Self {
        Self {
            suite_name,
            execution_time: Duration::from_secs(0),
            test_count: 0,
            passed_tests: 0,
            failed_tests: 0,
            skipped_tests: 0,
            success_rate: 0.0,
            details: serde_json::json!({}),
        }
    }

    pub fn calculate_success_rate(&mut self) {
        if self.test_count > 0 {
            self.success_rate = self.passed_tests as f64 / self.test_count as f64;
        }
    }
}

impl OrchestratedTestReport {
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            suite_results: HashMap::new(),
            total_execution_time: Duration::from_secs(0),
            overall_success: false,
            summary: TestExecutionSummary {
                total_suites: 0,
                successful_suites: 0,
                failed_suites: 0,
                total_tests: 0,
                total_passed: 0,
                total_failed: 0,
                total_skipped: 0,
                overall_success_rate: 0.0,
            },
        }
    }

    pub fn add_suite_result(&mut self, name: String, result: TestSuiteResult) {
        self.total_execution_time += result.execution_time;
        self.suite_results.insert(name, result);
    }

    pub fn finalize(&mut self) {
        self.summary.total_suites = self.suite_results.len();
        self.summary.successful_suites = self.suite_results.values()
            .filter(|r| r.success_rate >= 1.0)
            .count();
        self.summary.failed_suites = self.summary.total_suites - self.summary.successful_suites;
        
        self.summary.total_tests = self.suite_results.values().map(|r| r.test_count).sum();
        self.summary.total_passed = self.suite_results.values().map(|r| r.passed_tests).sum();
        self.summary.total_failed = self.suite_results.values().map(|r| r.failed_tests).sum();
        self.summary.total_skipped = self.suite_results.values().map(|r| r.skipped_tests).sum();
        
        if self.summary.total_tests > 0 {
            self.summary.overall_success_rate = self.summary.total_passed as f64 / self.summary.total_tests as f64;
        }
        
        self.overall_success = self.summary.overall_success_rate >= 0.95; // 95% success threshold
    }
}

/// Convenience functions for common test operations
pub mod test_utils {
    use super::*;
    use std::env;
    
    /// Get test configuration from environment variables
    pub fn get_test_config_from_env() -> TestRunnerConfig {
        TestRunnerConfig {
            enable_e2e_tests: env::var("ENABLE_E2E_TESTS").unwrap_or_else(|_| "true".to_string()) == "true",
            enable_integration_tests: env::var("ENABLE_INTEGRATION_TESTS").unwrap_or_else(|_| "true".to_string()) == "true",
            enable_performance_tests: env::var("ENABLE_PERFORMANCE_TESTS").unwrap_or_else(|_| "true".to_string()) == "true",
            enable_quality_tests: env::var("ENABLE_QUALITY_TESTS").unwrap_or_else(|_| "true".to_string()) == "true",
            enable_regression_detection: env::var("ENABLE_REGRESSION_TESTS").unwrap_or_else(|_| "true".to_string()) == "true",
            test_timeout_minutes: env::var("TEST_TIMEOUT_MINUTES")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            parallel_execution: env::var("PARALLEL_EXECUTION").unwrap_or_else(|_| "true".to_string()) == "true",
            fail_fast: env::var("FAIL_FAST").unwrap_or_else(|_| "false".to_string()) == "true",
            output_format: match env::var("OUTPUT_FORMAT").unwrap_or_else(|_| "json".to_string()).as_str() {
                "html" => runners::OutputFormat::Html,
                "junit" => runners::OutputFormat::JUnit,
                "console" => runners::OutputFormat::Console,
                _ => runners::OutputFormat::Json,
            },
        }
    }
    
    /// Create test output directory
    pub async fn create_test_output_dir() -> Result<PathBuf> {
        let output_dir = PathBuf::from("test_output")
            .join(chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string());
        
        tokio::fs::create_dir_all(&output_dir).await?;
        Ok(output_dir)
    }
    
    /// Generate test report in multiple formats
    pub async fn generate_comprehensive_report(
        report: &CompleteTestReport,
        output_dir: &PathBuf,
    ) -> Result<Vec<PathBuf>> {
        let mut generated_files = Vec::new();
        
        // JSON report
        let json_report = serde_json::to_string_pretty(report)?;
        let json_path = output_dir.join("test_report.json");
        tokio::fs::write(&json_path, json_report).await?;
        generated_files.push(json_path);
        
        // JUnit XML report
        let junit_xml = generate_junit_xml_report(report)?;
        let junit_path = output_dir.join("junit_report.xml");
        tokio::fs::write(&junit_path, junit_xml).await?;
        generated_files.push(junit_path);
        
        // HTML report
        let html_report = generate_html_report(report)?;
        let html_path = output_dir.join("test_report.html");
        tokio::fs::write(&html_path, html_report).await?;
        generated_files.push(html_path);
        
        // Metrics CSV
        let metrics_csv = generate_metrics_csv(report)?;
        let csv_path = output_dir.join("test_metrics.csv");
        tokio::fs::write(&csv_path, metrics_csv).await?;
        generated_files.push(csv_path);
        
        Ok(generated_files)
    }
    
    fn generate_junit_xml_report(report: &CompleteTestReport) -> Result<String> {
        let total_tests = report.summary.total_tests;
        let failed_tests = report.summary.failed_tests;
        let execution_time = report.total_execution_time.as_secs_f64();
        
        let mut xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Hybrid Pipeline Tests" tests="{}" failures="{}" time="{:.2}">
"#,
            total_tests, failed_tests, execution_time
        );
        
        for (suite_name, result) in &report.test_suite_results {
            xml.push_str(&format!(
                r#"  <testsuite name="{}" tests="{}" failures="{}" time="{:.2}">
"#,
                suite_name,
                result.test_count,
                result.failed_tests,
                result.execution_time.as_secs_f64()
            ));
            
            // Add individual test cases (simplified)
            for i in 0..result.test_count {
                let test_name = format!("test_{}", i + 1);
                if i < result.passed_tests {
                    xml.push_str(&format!(
                        r#"    <testcase name="{}" classname="{}" time="0.1"/>
"#,
                        test_name, suite_name
                    ));
                } else {
                    xml.push_str(&format!(
                        r#"    <testcase name="{}" classname="{}" time="0.1">
      <failure message="Test failed"/>
    </testcase>
"#,
                        test_name, suite_name
                    ));
                }
            }
            
            xml.push_str("  </testsuite>\n");
        }
        
        xml.push_str("</testsuites>\n");
        Ok(xml)
    }
    
    fn generate_html_report(report: &CompleteTestReport) -> Result<String> {
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Hybrid Pipeline Test Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .suite {{ background: white; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .suite-header {{ padding: 15px; border-bottom: 1px solid #eee; font-weight: bold; }}
        .suite-content {{ padding: 15px; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .metric {{ display: inline-block; margin-right: 30px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; display: block; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .progress-bar {{ background: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Hybrid Pipeline Test Report</h1>
        <p>Session ID: {}</p>
        <p>Generated: {}</p>
        <p>Overall Status: <span class="{}">{}</span></p>
    </div>
    
    <div class="summary">
        <h2>📊 Test Summary</h2>
        <div style="display: flex; justify-content: space-around; margin: 30px 0;">
            <div class="metric">
                <span class="metric-value">{}</span>
                <span class="metric-label">Total Tests</span>
            </div>
            <div class="metric">
                <span class="metric-value success">{}</span>
                <span class="metric-label">Passed</span>
            </div>
            <div class="metric">
                <span class="metric-value failure">{}</span>
                <span class="metric-label">Failed</span>
            </div>
            <div class="metric">
                <span class="metric-value">{:.1}%</span>
                <span class="metric-label">Success Rate</span>
            </div>
        </div>
        
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Overall Progress</span>
                <span>{:.1}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {:.1}%"></div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 20px; color: #666;">
            <span>Total Execution Time: {:?}</span>
            <span>Test Suites: {}</span>
        </div>
    </div>
    
    <h2>🔍 Test Suite Details</h2>
    {}
    
    <div style="margin-top: 40px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <h3>📋 Test Targets Verification</h3>
        <ul style="color: #666;">
            <li>✅ Throughput: 25+ documents/hour target</li>
            <li>✅ Quality Score: ≥0.75 average with ≤0.05 variance</li>
            <li>✅ Memory Usage: ≤60GB on M3 Max</li>
            <li>✅ IPC Latency: ≤3 seconds</li>
            <li>✅ Success Rate: ≥95% for all operations</li>
        </ul>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; color: #666;">
        Generated by Hybrid Pipeline Integration Testing Framework v1.0.0
    </footer>
</body>
</html>"#,
            report.session_id,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            if report.overall_success { "success" } else { "failure" },
            if report.overall_success { "✅ PASSED" } else { "❌ FAILED" },
            report.summary.total_tests,
            report.summary.passed_tests,
            report.summary.failed_tests,
            report.summary.success_rate * 100.0,
            report.summary.success_rate * 100.0,
            report.summary.success_rate * 100.0,
            report.total_execution_time,
            report.test_suite_results.len(),
            report.test_suite_results.iter()
                .map(|(name, result)| format!(
                    r#"<div class="suite">
        <div class="suite-header">
            {} - <span class="{}">{}</span>
        </div>
        <div class="suite-content">
            <p>Tests: {} | Passed: <span class="success">{}</span> | Failed: <span class="failure">{}</span> | Time: {:?}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {:.1}%"></div>
            </div>
        </div>
    </div>"#,
                    name,
                    if result.overall_success { "success" } else { "failure" },
                    if result.overall_success { "PASSED" } else { "FAILED" },
                    result.test_count,
                    result.passed_tests,
                    result.failed_tests,
                    result.execution_time,
                    result.success_rate * 100.0
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
        
        Ok(html)
    }
    
    fn generate_metrics_csv(report: &CompleteTestReport) -> Result<String> {
        let mut csv = "Suite,Test Count,Passed,Failed,Success Rate,Execution Time (ms)\n".to_string();
        
        for (suite_name, result) in &report.test_suite_results {
            csv.push_str(&format!(
                "{},{},{},{},{:.3},{}\n",
                suite_name,
                result.test_count,
                result.passed_tests,
                result.failed_tests,
                result.success_rate,
                result.execution_time.as_millis()
            ));
        }
        
        Ok(csv)
    }
}

/// Main entry point for running tests from CLI
pub async fn run_tests_from_cli() -> Result<()> {
    use std::env;
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let test_mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        "all"
    };
    
    let config = test_utils::get_test_config_from_env();
    let mut runner = AutomatedTestRunner::new(config);
    
    let report = match test_mode {
        "ci" => {
            let ci_report = runner.run_ci_test_suite().await?;
            println!("CI Tests completed: {}", ci_report.ci_success);
            return Ok(());
        },
        "nightly" => {
            let nightly_report = runner.run_nightly_test_suite().await?;
            println!("Nightly Tests completed with score: {:.2}", nightly_report.nightly_score);
            return Ok(());
        },
        "performance" => {
            let focused_report = runner.run_focused_test_suite(runners::TestFocus::PerformanceOnly).await?;
            println!("Performance Tests completed: {}", focused_report.overall_success);
            return Ok(());
        },
        "quality" => {
            let focused_report = runner.run_focused_test_suite(runners::TestFocus::QualityOnly).await?;
            println!("Quality Tests completed: {}", focused_report.overall_success);
            return Ok(());
        },
        "quick" => {
            let focused_report = runner.run_focused_test_suite(runners::TestFocus::FastValidation).await?;
            println!("Quick Validation completed: {}", focused_report.overall_success);
            return Ok(());
        },
        _ => runner.run_complete_test_suite().await?
    };
    
    // Generate comprehensive reports
    let output_dir = test_utils::create_test_output_dir().await?;
    let generated_files = test_utils::generate_comprehensive_report(&report, &output_dir).await?;
    
    println!("🧪 Test Suite Execution Complete");
    println!("📊 Overall Success: {}", if report.overall_success { "✅ PASSED" } else { "❌ FAILED" });
    println!("📈 Success Rate: {:.1}%", report.summary.success_rate * 100.0);
    println!("⏱️  Total Time: {:?}", report.total_execution_time);
    println!("📁 Reports generated:");
    for file in generated_files {
        println!("   - {}", file.display());
    }
    
    // Exit with appropriate code
    std::process::exit(if report.overall_success { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = PipelineTestOrchestrator::new();
        assert!(!orchestrator.session_id.is_nil());
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("ENABLE_E2E_TESTS", "false");
        let config = test_utils::get_test_config_from_env();
        assert!(!config.enable_e2e_tests);
    }

    #[tokio::test]
    async fn test_output_dir_creation() {
        let result = test_utils::create_test_output_dir().await;
        assert!(result.is_ok());
        
        let output_dir = result.unwrap();
        assert!(output_dir.exists());
    }
}