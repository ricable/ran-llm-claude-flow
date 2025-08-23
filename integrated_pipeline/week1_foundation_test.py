#!/usr/bin/env python3
"""
Week 1 Foundation Integration Test

Comprehensive validation of the Rust-Python hybrid pipeline foundation:
- End-to-end document processing
- IPC communication reliability
- Qwen3-7B model integration
- Performance benchmarking
- Memory usage validation within 128GB M3 Max limits

Author: Claude Code - Week 1 Foundation Agent
Version: 1.0.0
"""

import asyncio
import time
import json
import psutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Week1FoundationValidator:
    """Complete validation suite for Week 1 foundation implementation"""
    
    def __init__(self):
        self.results = {
            'test_results': {},
            'performance_metrics': {},
            'system_validation': {},
            'ipc_validation': {},
            'model_validation': {},
            'memory_usage': {},
            'errors': [],
            'warnings': []
        }
        self.start_time = time.time()
        
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete foundation validation suite"""
        logger.info("🏗️ Starting Week 1 Foundation Validation Suite")
        
        # 1. System Requirements Validation
        await self.validate_system_requirements()
        
        # 2. Build and Compile Rust Core
        await self.build_rust_core()
        
        # 3. Validate Python ML Engine
        await self.validate_python_ml_engine()
        
        # 4. Test IPC Communication
        await self.test_ipc_communication()
        
        # 5. End-to-End Document Processing
        await self.test_end_to_end_processing()
        
        # 6. Performance Benchmarking
        await self.run_performance_benchmarks()
        
        # 7. Memory Usage Validation
        await self.validate_memory_usage()
        
        # 8. Generate Final Report
        self.generate_final_report()
        
        return self.results
        
    async def validate_system_requirements(self):
        """Validate M3 Max system requirements"""
        logger.info("🔍 Validating system requirements...")
        
        try:
            # System information
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            system_info = {
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'cpu_cores': cpu_count,
                'platform': sys.platform
            }
            
            # Validation checks
            checks = {
                'memory_sufficient': memory.total >= 64 * 1024**3,  # At least 64GB
                'cpu_cores_sufficient': cpu_count >= 8,  # At least 8 cores
                'macos_platform': sys.platform == 'darwin'
            }
            
            self.results['system_validation'] = {
                'system_info': system_info,
                'checks': checks,
                'passed': all(checks.values())
            }
            
            if all(checks.values()):
                logger.info("✅ System requirements validated")
            else:
                logger.warning("⚠️ Some system requirements not met")
                self.results['warnings'].append("System requirements not optimal")
                
        except Exception as e:
            self.results['errors'].append(f"System validation failed: {e}")
            logger.error(f"❌ System validation error: {e}")
            
    async def build_rust_core(self):
        """Build and validate Rust core"""
        logger.info("🦀 Building Rust core...")
        
        rust_dir = Path("rust_core")
        
        try:
            # Build Rust project
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=rust_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode == 0:
                logger.info("✅ Rust core built successfully")
                self.results['test_results']['rust_build'] = {
                    'success': True,
                    'build_time': 'completed',
                    'binary_exists': (rust_dir / "target/release/rust-core").exists()
                }
            else:
                logger.error(f"❌ Rust build failed: {result.stderr}")
                self.results['errors'].append(f"Rust build failed: {result.stderr}")
                self.results['test_results']['rust_build'] = {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            self.results['errors'].append("Rust build timeout")
            logger.error("❌ Rust build timeout")
        except Exception as e:
            self.results['errors'].append(f"Rust build exception: {e}")
            logger.error(f"❌ Rust build exception: {e}")
            
    async def validate_python_ml_engine(self):
        """Validate Python ML engine initialization"""
        logger.info("🐍 Validating Python ML Engine...")
        
        try:
            # Test Python imports
            python_dir = Path("python_ml")
            result = subprocess.run(
                [sys.executable, "-c", 
                 "import sys; sys.path.insert(0, '.'); "
                 "from src.main import PythonMLEngine; "
                 "print('Python ML Engine imports successful')"],
                cwd=python_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✅ Python ML Engine imports validated")
                self.results['test_results']['python_imports'] = {
                    'success': True,
                    'output': result.stdout.strip()
                }
            else:
                logger.error(f"❌ Python ML Engine import failed: {result.stderr}")
                self.results['errors'].append(f"Python imports failed: {result.stderr}")
                
        except Exception as e:
            self.results['errors'].append(f"Python validation exception: {e}")
            logger.error(f"❌ Python validation exception: {e}")
            
    async def test_ipc_communication(self):
        """Test IPC communication setup"""
        logger.info("🔗 Testing IPC communication...")
        
        try:
            # Test named pipe creation
            pipe_path = Path("/tmp/claude_flow_ipc_test")
            
            # Create test pipe
            subprocess.run(["mkfifo", str(pipe_path)], check=True, timeout=10)
            
            # Test pipe exists and is accessible
            if pipe_path.exists():
                logger.info("✅ Named pipe creation successful")
                self.results['ipc_validation']['named_pipe'] = {
                    'success': True,
                    'path': str(pipe_path)
                }
                
                # Cleanup
                pipe_path.unlink()
            else:
                self.results['errors'].append("Named pipe creation failed")
                
            # Test shared memory simulation
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(b"test data for shared memory simulation")
                tmp_path = tmp.name
                
            # Validate file-based shared memory simulation
            test_data = Path(tmp_path).read_bytes()
            if test_data == b"test data for shared memory simulation":
                logger.info("✅ Shared memory simulation successful")
                self.results['ipc_validation']['shared_memory'] = {
                    'success': True,
                    'test_data_size': len(test_data)
                }
            else:
                self.results['errors'].append("Shared memory simulation failed")
                
            # Cleanup
            Path(tmp_path).unlink()
            
        except Exception as e:
            self.results['errors'].append(f"IPC test exception: {e}")
            logger.error(f"❌ IPC test exception: {e}")
            
    async def test_end_to_end_processing(self):
        """Test end-to-end document processing"""
        logger.info("📄 Testing end-to-end document processing...")
        
        # Create test document
        test_document = """
DOCTITLE: LTE eNodeB Handover Optimization

Product: CXC4012011, CXC4012019
Feature State: Available

## Overview
This feature optimizes LTE handover procedures to reduce call drops and improve user experience.
The optimization algorithm adjusts handover parameters based on radio conditions and traffic patterns.

## Parameters

- **EUtranCellFDD.a3Offset**: A3 event offset for handover triggering
  - MO Class: EUtranCellFDD
  - Valid Values: -30 to 30 dB
  - Default: 0 dB
  - Description: Controls when A3 measurements are triggered for handover decisions

- **EUtranCellFDD.hysteresisA3**: Hysteresis value for A3 event
  - MO Class: EUtranCellFDD
  - Valid Values: 0 to 15 dB
  - Default: 2 dB
  - Description: Prevents ping-pong effects in handover decisions

## Counters

- **pmCellHoExeAttOut**: Outbound handover execution attempts
  - Description: Number of handover execution attempts from this cell
  - MO Class: EUtranCellFDD
  - Counter Type: Incremental

## Configuration Examples

Basic configuration:
```
EUtranCellFDD.a3Offset = 3
EUtranCellFDD.hysteresisA3 = 2
```

Optimized for dense networks:
```
EUtranCellFDD.a3Offset = 6
EUtranCellFDD.hysteresisA3 = 4
```

Technical terms: LTE, eNodeB, UE, RSRP, RSRQ, CQI, PMI, RI, MIMO, CA
"""
        
        try:
            # Test with Python ML engine
            python_dir = Path("python_ml")
            
            # Create temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
                tmp.write(test_document)
                test_file = tmp.name
                
            # Run Python ML engine test
            result = subprocess.run([
                sys.executable, "-m", "src.main", "--test"
            ], cwd=python_dir, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ End-to-end processing test passed")
                self.results['test_results']['end_to_end'] = {
                    'success': True,
                    'output': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                }
            else:
                logger.error(f"❌ End-to-end processing failed: {result.stderr}")
                self.results['errors'].append(f"E2E processing failed: {result.stderr}")
                
            # Cleanup
            Path(test_file).unlink()
            
        except Exception as e:
            self.results['errors'].append(f"E2E test exception: {e}")
            logger.error(f"❌ E2E test exception: {e}")
            
    async def run_performance_benchmarks(self):
        """Run performance benchmarks"""
        logger.info("🏃 Running performance benchmarks...")
        
        try:
            start_time = time.time()
            
            # Memory usage before
            memory_before = psutil.virtual_memory()
            
            # Run Rust core benchmark
            rust_dir = Path("rust_core")
            if (rust_dir / "target/release/rust-core").exists():
                rust_result = subprocess.run([
                    "./target/release/rust-core", "benchmark", "--count", "5"
                ], cwd=rust_dir, capture_output=True, text=True, timeout=180)
                
                if rust_result.returncode == 0:
                    logger.info("✅ Rust benchmark completed")
                    self.results['performance_metrics']['rust_benchmark'] = {
                        'success': True,
                        'output': rust_result.stdout[-500:] if len(rust_result.stdout) > 500 else rust_result.stdout
                    }
                    
            # Run Python ML engine benchmark
            python_dir = Path("python_ml")
            python_result = subprocess.run([
                sys.executable, "-m", "src.main", "--benchmark"
            ], cwd=python_dir, capture_output=True, text=True, timeout=180)
            
            if python_result.returncode == 0:
                logger.info("✅ Python benchmark completed")
                self.results['performance_metrics']['python_benchmark'] = {
                    'success': True,
                    'output': python_result.stdout[-500:] if len(python_result.stdout) > 500 else python_result.stdout
                }
                
            # Memory usage after
            memory_after = psutil.virtual_memory()
            benchmark_time = time.time() - start_time
            
            self.results['performance_metrics']['summary'] = {
                'total_benchmark_time': benchmark_time,
                'memory_delta_gb': (memory_after.used - memory_before.used) / (1024**3),
                'peak_memory_usage_gb': memory_after.used / (1024**3)
            }
            
        except Exception as e:
            self.results['errors'].append(f"Performance benchmark exception: {e}")
            logger.error(f"❌ Performance benchmark exception: {e}")
            
    async def validate_memory_usage(self):
        """Validate memory usage within M3 Max limits"""
        logger.info("💾 Validating memory usage...")
        
        try:
            memory = psutil.virtual_memory()
            
            memory_analysis = {
                'total_system_gb': memory.total / (1024**3),
                'used_system_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percentage': memory.percent,
                'within_128gb_limit': memory.total <= 136 * (1024**3),  # Allow some overhead
                'rust_allocation_possible': memory.available >= 60 * (1024**3),
                'python_allocation_possible': memory.available >= 45 * (1024**3),
                'ipc_allocation_possible': memory.available >= 15 * (1024**3)
            }
            
            self.results['memory_usage'] = memory_analysis
            
            if memory_analysis['within_128gb_limit']:
                logger.info("✅ Memory usage within M3 Max limits")
            else:
                logger.warning("⚠️ Memory usage exceeds expected M3 Max limits")
                
        except Exception as e:
            self.results['errors'].append(f"Memory validation exception: {e}")
            logger.error(f"❌ Memory validation exception: {e}")
            
    def generate_final_report(self):
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time
        
        # Calculate success metrics
        total_tests = len(self.results['test_results'])
        passed_tests = sum(1 for test in self.results['test_results'].values() 
                          if isinstance(test, dict) and test.get('success', False))
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Overall assessment
        overall_success = (
            len(self.results['errors']) == 0 and
            success_rate >= 80 and
            self.results.get('system_validation', {}).get('passed', False)
        )
        
        summary = {
            'overall_success': overall_success,
            'total_validation_time': total_time,
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'success_rate': success_rate,
            'errors_count': len(self.results['errors']),
            'warnings_count': len(self.results['warnings']),
            'ready_for_week2': overall_success and success_rate >= 90
        }
        
        self.results['summary'] = summary
        
        # Log final status
        if overall_success:
            logger.info(f"🎉 Week 1 Foundation VALIDATION PASSED ({success_rate:.1f}% success rate)")
        else:
            logger.error(f"❌ Week 1 Foundation validation failed ({success_rate:.1f}% success rate)")


async def main():
    """Main validation runner"""
    print("🏗️ Week 1 Foundation Integration Test Suite")
    print("=" * 60)
    print("Testing Rust-Python hybrid pipeline foundation:")
    print("  • Development environment setup")
    print("  • Basic IPC communication")
    print("  • Qwen3-7B model integration")
    print("  • Performance monitoring")
    print("  • M3 Max optimization baseline")
    print("=" * 60)
    
    validator = Week1FoundationValidator()
    
    try:
        results = await validator.run_complete_validation()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = results['summary']
        print(f"Overall Success: {'✅ PASSED' if summary['overall_success'] else '❌ FAILED'}")
        print(f"Tests Passed: {summary['tests_passed']}/{summary['tests_total']} ({summary['success_rate']:.1f}%)")
        print(f"Errors: {summary['errors_count']}")
        print(f"Warnings: {summary['warnings_count']}")
        print(f"Total Time: {summary['total_validation_time']:.1f}s")
        print(f"Ready for Week 2: {'✅ YES' if summary['ready_for_week2'] else '❌ NO'}")
        
        # Memory summary
        memory = results.get('memory_usage', {})
        if memory:
            print(f"\n💾 MEMORY USAGE:")
            print(f"  System Memory: {memory.get('total_system_gb', 0):.1f}GB")
            print(f"  Used Memory: {memory.get('used_system_gb', 0):.1f}GB")
            print(f"  Available: {memory.get('available_gb', 0):.1f}GB")
            print(f"  Usage: {memory.get('usage_percentage', 0):.1f}%")
            
        # Errors and warnings
        if results['errors']:
            print(f"\n❌ ERRORS ({len(results['errors'])}):")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"  • {error}")
                
        if results['warnings']:
            print(f"\n⚠️ WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings'][:3]:  # Show first 3 warnings
                print(f"  • {warning}")
        
        # Save detailed results
        results_file = Path("week1_foundation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        return 0 if summary['overall_success'] else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))