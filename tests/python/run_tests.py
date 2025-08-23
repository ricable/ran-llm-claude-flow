#!/usr/bin/env python3
"""
Comprehensive test runner for Python integration modules
Generates coverage reports, performance benchmarks, and validation results.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TestRunner:
    """Comprehensive test runner with coverage and reporting"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = project_root
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "coverage": {},
            "performance": {},
            "errors": []
        }
    
    def run_unit_tests(self):
        """Run unit tests with coverage"""
        print("🧪 Running unit tests with coverage...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "unit"),
            "--cov=docs.python.integration",
            "--cov-report=html:tests/python/htmlcov",
            "--cov-report=json:tests/python/coverage.json",
            "--cov-report=term-missing",
            "--cov-fail-under=95",
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=tests/python/unit_test_results.json"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            self.results["test_results"]["unit_tests"] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                print("✅ Unit tests passed!")
            else:
                print(f"❌ Unit tests failed with exit code {result.returncode}")
                print(f"Error output: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error running unit tests: {e}")
            self.results["errors"].append(f"Unit test execution error: {e}")
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=tests/python/integration_test_results.json"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            self.results["test_results"]["integration_tests"] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                print("✅ Integration tests passed!")
            else:
                print(f"❌ Integration tests failed with exit code {result.returncode}")
                print(f"Error output: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error running integration tests: {e}")
            self.results["errors"].append(f"Integration test execution error: {e}")
    
    def parse_coverage_report(self):
        """Parse coverage report"""
        print("📊 Parsing coverage report...")
        
        coverage_file = self.test_dir / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                # Extract key coverage metrics
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                self.results["coverage"] = {
                    "total_coverage": total_coverage,
                    "files": {}
                }
                
                # Per-file coverage
                for filename, file_data in coverage_data.get("files", {}).items():
                    if "docs/python/integration" in filename:
                        self.results["coverage"]["files"][filename] = {
                            "percent_covered": file_data.get("summary", {}).get("percent_covered", 0),
                            "num_statements": file_data.get("summary", {}).get("num_statements", 0),
                            "missing_lines": file_data.get("summary", {}).get("missing_lines", 0)
                        }
                
                print(f"✅ Total coverage: {total_coverage:.1f}%")
                
                if total_coverage >= 95:
                    print("🎯 Coverage target achieved!")
                else:
                    print(f"⚠️  Coverage below target (95%): {total_coverage:.1f}%")
                    
            except Exception as e:
                print(f"❌ Error parsing coverage report: {e}")
                self.results["errors"].append(f"Coverage parsing error: {e}")
        else:
            print("❌ Coverage report not found")
            self.results["errors"].append("Coverage report file not found")
    
    async def run_performance_benchmarks(self):
        """Run performance benchmarks"""
        print("⚡ Running performance benchmarks...")
        
        try:
            # Import test modules to run benchmarks
            from tests.python.benchmarks.performance_benchmarks import run_performance_tests
            
            benchmark_results = await run_performance_tests()
            self.results["performance"] = benchmark_results
            
            print("✅ Performance benchmarks completed")
            
        except ImportError:
            print("⚠️  Performance benchmarks not available (optional)")
        except Exception as e:
            print(f"❌ Error running performance benchmarks: {e}")
            self.results["errors"].append(f"Performance benchmark error: {e}")
    
    def validate_code_quality(self):
        """Validate code quality and errors"""
        print("🔍 Validating code quality...")
        
        error_patterns = [
            "SyntaxError",
            "ImportError", 
            "ModuleNotFoundError",
            "AttributeError",
            "NameError"
        ]
        
        python_files = list(Path("docs/python/integration").glob("*.py"))
        quality_issues = []
        
        for py_file in python_files:
            try:
                # Try to compile the file
                with open(py_file, 'r') as f:
                    code = f.read()
                
                compile(code, str(py_file), 'exec')
                
            except SyntaxError as e:
                quality_issues.append(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                quality_issues.append(f"Compilation error in {py_file}: {e}")
        
        self.results["code_quality"] = {
            "files_checked": len(python_files),
            "issues": quality_issues,
            "clean": len(quality_issues) == 0
        }
        
        if quality_issues:
            print(f"⚠️  Found {len(quality_issues)} code quality issues")
            for issue in quality_issues:
                print(f"  - {issue}")
        else:
            print("✅ No code quality issues found")
    
    def generate_report(self):
        """Generate final test report"""
        print("📝 Generating final report...")
        
        report_file = self.test_dir / "test_report.json"
        
        # Calculate overall success
        unit_success = self.results["test_results"].get("unit_tests", {}).get("success", False)
        integration_success = self.results["test_results"].get("integration_tests", {}).get("success", False)
        coverage_success = self.results["coverage"].get("total_coverage", 0) >= 95
        quality_success = self.results["code_quality"].get("clean", False)
        
        overall_success = unit_success and integration_success and coverage_success and quality_success
        
        self.results["summary"] = {
            "overall_success": overall_success,
            "unit_tests_passed": unit_success,
            "integration_tests_passed": integration_success,
            "coverage_target_met": coverage_success,
            "code_quality_clean": quality_success,
            "total_errors": len(self.results["errors"])
        }
        
        # Write report
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Overall Success: {'✅' if overall_success else '❌'}")
        print(f"Unit Tests: {'✅' if unit_success else '❌'}")
        print(f"Integration Tests: {'✅' if integration_success else '❌'}")
        print(f"Coverage Target (95%): {'✅' if coverage_success else '❌'} ({self.results['coverage'].get('total_coverage', 0):.1f}%)")
        print(f"Code Quality: {'✅' if quality_success else '❌'}")
        print(f"Total Errors: {len(self.results['errors'])}")
        print(f"Report saved to: {report_file}")
        
        if self.results["coverage"].get("total_coverage", 0) > 0:
            print(f"Coverage report: {self.test_dir / 'htmlcov' / 'index.html'}")
        
        return overall_success

async def main():
    """Main test execution function"""
    print("🚀 Starting comprehensive Python test suite")
    print(f"Project root: {project_root}")
    print(f"Test directory: {Path(__file__).parent}")
    
    runner = TestRunner()
    
    # Run all test phases
    runner.run_unit_tests()
    runner.run_integration_tests()
    runner.parse_coverage_report()
    await runner.run_performance_benchmarks()
    runner.validate_code_quality()
    
    # Generate final report
    success = runner.generate_report()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the report for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)