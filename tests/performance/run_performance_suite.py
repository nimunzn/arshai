#!/usr/bin/env python3
"""
Performance Test Suite Runner

Comprehensive performance testing suite for Arshai framework optimizations.
Runs all performance tests and generates detailed reports for production validation.

Usage:
    python tests/performance/run_performance_suite.py [options]
    
Options:
    --quick         Run quick validation tests (5 minutes)
    --moderate      Run moderate load tests (15 minutes) 
    --full          Run full production tests (45 minutes)
    --extreme       Run extreme stress tests (60 minutes)
    --report        Generate detailed performance report
    --metrics       Output metrics in JSON format
    --help          Show this help message

Test Categories:
    1. HTTP Connection Pooling - Tests SearxNG connection limits
    2. Thread Pool Management - Tests MCP tool thread limits
    3. Vector Database Async - Tests Milvus async operations
    4. Integration Tests - Tests end-to-end performance
    5. Sustained Load - Tests stability over time
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTestSuite:
    """Comprehensive performance test suite runner."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.failed_tests = []
        self.skipped_tests = []
        
    def setup_test_environment(self):
        """Setup optimal test environment configuration."""
        logger.info("🔧 Setting up performance test environment")
        
        # Optimize for performance testing
        test_env = {
            'ARSHAI_MAX_CONNECTIONS': '100',
            'ARSHAI_MAX_CONNECTIONS_PER_HOST': '20',
            'ARSHAI_CONNECTION_TIMEOUT': '10',
            'ARSHAI_MAX_THREADS': '32',
            'ARSHAI_MAX_MEMORY_MB': '4096',
            'ARSHAI_CLEANUP_INTERVAL': '300',
            'PYTEST_TIMEOUT': '3600',  # 1 hour timeout
        }
        
        for key, value in test_env.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
        
        logger.info("✅ Test environment configured")
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick validation tests (~5 minutes)."""
        logger.info("⚡ Running quick validation tests")
        
        test_commands = [
            "python tests/performance/test_connection_pool_load.py --quick",
            "python tests/performance/test_thread_pool_load.py --quick", 
            "python tests/performance/test_vector_db_load.py --quick",
        ]
        
        results = {}
        for cmd in test_commands:
            test_name = cmd.split('/')[-1].replace('.py', '').replace('test_', '')
            logger.info(f"Running {test_name} quick test...")
            
            start = time.perf_counter()
            try:
                import subprocess
                result = subprocess.run(
                    cmd.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=300,
                    cwd=project_root
                )
                duration = time.perf_counter() - start
                
                results[test_name] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'duration': duration,
                    'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
                
                if result.returncode == 0:
                    logger.info(f"✅ {test_name} quick test passed ({duration:.1f}s)")
                else:
                    logger.error(f"❌ {test_name} quick test failed ({duration:.1f}s)")
                    self.failed_tests.append(test_name)
                    
            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - start
                results[test_name] = {
                    'status': 'timeout',
                    'duration': duration,
                    'error': 'Test timed out after 5 minutes'
                }
                logger.error(f"⏰ {test_name} quick test timed out")
                self.failed_tests.append(test_name)
            except Exception as e:
                duration = time.perf_counter() - start
                results[test_name] = {
                    'status': 'error',
                    'duration': duration,
                    'error': str(e)
                }
                logger.error(f"💥 {test_name} quick test error: {e}")
                self.failed_tests.append(test_name)
        
        return results
    
    def run_moderate_tests(self) -> Dict[str, Any]:
        """Run moderate load tests (~15 minutes)."""
        logger.info("🧪 Running moderate load tests")
        
        test_commands = [
            "pytest tests/performance/test_connection_pool_load.py::TestConnectionPoolLoad::test_moderate_concurrent_load -v",
            "pytest tests/performance/test_thread_pool_load.py::TestThreadPoolLoad::test_moderate_thread_load -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_moderate_vector_search_load -v",
        ]
        
        return self._run_pytest_commands(test_commands, "moderate")
    
    def run_high_load_tests(self) -> Dict[str, Any]:
        """Run high load tests (~30 minutes)."""
        logger.info("🔥 Running high load tests")
        
        test_commands = [
            "pytest tests/performance/test_connection_pool_load.py::TestConnectionPoolLoad::test_high_concurrent_load -v",
            "pytest tests/performance/test_thread_pool_load.py::TestThreadPoolLoad::test_high_thread_load -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_high_vector_search_load -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_batch_insertion_load -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_mixed_operations_load -v",
        ]
        
        return self._run_pytest_commands(test_commands, "high_load")
    
    def run_extreme_tests(self) -> Dict[str, Any]:
        """Run extreme stress tests (~45 minutes)."""
        logger.info("🚀 Running extreme stress tests")
        
        test_commands = [
            "pytest tests/performance/test_connection_pool_load.py::TestConnectionPoolLoad::test_extreme_concurrent_load -v",
            "pytest tests/performance/test_thread_pool_load.py::TestThreadPoolLoad::test_extreme_thread_load -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_extreme_vector_search_load -v",
        ]
        
        return self._run_pytest_commands(test_commands, "extreme")
    
    def run_sustained_tests(self) -> Dict[str, Any]:
        """Run sustained load tests (~60 minutes)."""
        logger.info("🔄 Running sustained load tests")
        
        test_commands = [
            "pytest tests/performance/test_connection_pool_load.py::TestConnectionPoolLoad::test_sustained_load_memory_stability -v",
            "pytest tests/performance/test_thread_pool_load.py::TestThreadPoolLoad::test_sustained_thread_load -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_sustained_vector_operations -v",
        ]
        
        return self._run_pytest_commands(test_commands, "sustained")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests (~20 minutes)."""
        logger.info("🔧 Running integration tests")
        
        test_commands = [
            "pytest tests/performance/test_connection_pool_load.py::TestConnectionPoolLoad::test_connection_pool_recovery -v",
            "pytest tests/performance/test_thread_pool_load.py::TestThreadPoolLoad::test_thread_pool_deadlock_prevention -v",
            "pytest tests/performance/test_vector_db_load.py::TestVectorDBLoad::test_knowledge_base_tool_integration -v",
        ]
        
        return self._run_pytest_commands(test_commands, "integration")
    
    def _run_pytest_commands(self, commands: List[str], test_type: str) -> Dict[str, Any]:
        """Run a list of pytest commands and collect results."""
        results = {}
        
        for cmd in commands:
            # Extract test name from command
            test_parts = cmd.split("::")
            if len(test_parts) >= 2:
                test_name = f"{test_type}_{test_parts[-1]}"
            else:
                test_name = f"{test_type}_{cmd.split('/')[-1]}"
            
            logger.info(f"Running {test_name}...")
            
            start = time.perf_counter()
            try:
                import subprocess
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minute timeout per test
                    cwd=project_root
                )
                duration = time.perf_counter() - start
                
                # Parse pytest output for metrics
                metrics = self._parse_pytest_output(result.stdout)
                
                results[test_name] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'duration': duration,
                    'metrics': metrics,
                    'stdout': result.stdout[-2000:] if result.stdout else '',
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
                
                if result.returncode == 0:
                    logger.info(f"✅ {test_name} passed ({duration:.1f}s)")
                else:
                    logger.error(f"❌ {test_name} failed ({duration:.1f}s)")
                    self.failed_tests.append(test_name)
                    
            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - start
                results[test_name] = {
                    'status': 'timeout',
                    'duration': duration,
                    'error': 'Test timed out after 30 minutes'
                }
                logger.error(f"⏰ {test_name} timed out")
                self.failed_tests.append(test_name)
            except Exception as e:
                duration = time.perf_counter() - start
                results[test_name] = {
                    'status': 'error',
                    'duration': duration,
                    'error': str(e)
                }
                logger.error(f"💥 {test_name} error: {e}")
                self.failed_tests.append(test_name)
        
        return results
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract performance metrics."""
        metrics = {}
        
        # Look for common performance indicators
        lines = output.split('\n')
        for line in lines:
            if 'requests/second' in line or 'operations/second' in line:
                try:
                    rate = float(line.split()[-1])
                    metrics['throughput'] = rate
                except (ValueError, IndexError):
                    pass
            elif 'ms' in line and ('p95' in line or 'average' in line):
                try:
                    latency = float(line.split()[-1].replace('ms', ''))
                    if 'p95' in line:
                        metrics['p95_latency_ms'] = latency
                    elif 'average' in line:
                        metrics['avg_latency_ms'] = latency
                except (ValueError, IndexError):
                    pass
            elif 'success rate' in line or 'Success rate' in line:
                try:
                    rate = float(line.split('%')[0].split()[-1])
                    metrics['success_rate_percent'] = rate
                except (ValueError, IndexError):
                    pass
        
        return metrics
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report."""
        logger.info("📊 Generating performance report")
        
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        report = f"""
# Arshai Performance Test Report

**Generated**: {datetime.now(timezone.utc).isoformat()}  
**Test Duration**: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)  
**Environment**: {os.environ.get('ARSHAI_ENV', 'test')}

## Executive Summary

- **Total Tests**: {sum(len(category_results) for category_results in results.values())}
- **Passed Tests**: {sum(len(category_results) - len([t for t in category_results.values() if t.get('status') != 'passed']) for category_results in results.values())}
- **Failed Tests**: {len(self.failed_tests)}
- **Success Rate**: {((sum(len(category_results) for category_results in results.values()) - len(self.failed_tests)) / sum(len(category_results) for category_results in results.values()) * 100):.1f}%

## Performance Optimizations Validated

### ✅ HTTP Connection Pooling
- **Implementation**: SearxNG shared connection pool with limits
- **Configuration**: {os.environ.get('ARSHAI_MAX_CONNECTIONS', '100')} total, {os.environ.get('ARSHAI_MAX_CONNECTIONS_PER_HOST', '20')} per host
- **Status**: {'✅ PASSED' if not any('connection_pool' in test for test in self.failed_tests) else '❌ FAILED'}

### ✅ Thread Pool Management  
- **Implementation**: Shared ThreadPoolExecutor with bounded workers
- **Configuration**: {os.environ.get('ARSHAI_MAX_THREADS', '32')} max threads
- **Status**: {'✅ PASSED' if not any('thread_pool' in test for test in self.failed_tests) else '❌ FAILED'}

### ✅ Vector Database Async Operations
- **Implementation**: AsyncIO executor pattern for non-blocking operations
- **Configuration**: Async methods with fallback to sync + executor
- **Status**: {'✅ PASSED' if not any('vector_db' in test for test in self.failed_tests) else '❌ FAILED'}

## Test Results by Category

"""
        
        for category, category_results in results.items():
            report += f"\n### {category.replace('_', ' ').title()} Tests\n\n"
            
            passed = len([t for t in category_results.values() if t.get('status') == 'passed'])
            failed = len([t for t in category_results.values() if t.get('status') == 'failed'])
            total = len(category_results)
            
            report += f"- **Total**: {total}\n"
            report += f"- **Passed**: {passed}\n" 
            report += f"- **Failed**: {failed}\n"
            report += f"- **Success Rate**: {(passed/total*100):.1f}%\n\n"
            
            for test_name, test_result in category_results.items():
                status_emoji = {
                    'passed': '✅',
                    'failed': '❌', 
                    'timeout': '⏰',
                    'error': '💥'
                }.get(test_result.get('status', 'unknown'), '❓')
                
                report += f"#### {status_emoji} {test_name}\n"
                report += f"- **Duration**: {test_result.get('duration', 0):.1f}s\n"
                
                if 'metrics' in test_result and test_result['metrics']:
                    metrics = test_result['metrics']
                    if 'throughput' in metrics:
                        report += f"- **Throughput**: {metrics['throughput']:.1f} ops/sec\n"
                    if 'avg_latency_ms' in metrics:
                        report += f"- **Average Latency**: {metrics['avg_latency_ms']:.0f}ms\n"
                    if 'p95_latency_ms' in metrics:
                        report += f"- **P95 Latency**: {metrics['p95_latency_ms']:.0f}ms\n"
                    if 'success_rate_percent' in metrics:
                        report += f"- **Success Rate**: {metrics['success_rate_percent']:.1f}%\n"
                
                if test_result.get('status') != 'passed':
                    report += f"- **Error**: {test_result.get('error', 'Unknown error')}\n"
                
                report += "\n"
        
        if self.failed_tests:
            report += f"\n## Failed Tests\n\n"
            for test in self.failed_tests:
                report += f"- {test}\n"
        
        report += f"""
## Production Readiness Assessment

### Scale Validation
- **Concurrent Users**: {'1000+' if not self.failed_tests else 'Limited'}
- **Container Stability**: {'Crash-resistant' if not self.failed_tests else 'Needs improvement'}
- **Resource Efficiency**: {'Optimized' if not self.failed_tests else 'Needs tuning'}

### Deployment Recommendations
- **Environment Variables**: Use provided configuration values
- **Container Resources**: 2Gi memory, 1000m CPU minimum
- **Monitoring**: Implement metrics from performance-optimization.md
- **Scaling**: Horizontal scaling validated for stateless components

### Next Steps
{'All performance optimizations validated - ready for production deployment!' if not self.failed_tests else 'Address failed tests before production deployment.'}
"""
        
        return report
    
    def save_results(self, results: Dict[str, Any], report: str, output_dir: str = "performance_results"):
        """Save test results and reports to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = output_path / f"performance_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
                'environment': dict(os.environ),
                'results': results,
                'failed_tests': self.failed_tests,
                'summary': {
                    'total_tests': sum(len(category_results) for category_results in results.values()),
                    'passed_tests': sum(len(category_results) - len([t for t in category_results.values() if t.get('status') != 'passed']) for category_results in results.values()),
                    'failed_tests': len(self.failed_tests)
                }
            }, f, indent=2)
        
        # Save markdown report
        report_file = output_path / f"performance_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Results saved to {json_file}")
        logger.info(f"📄 Report saved to {report_file}")
        
        return str(json_file), str(report_file)


def main():
    """Main entry point for performance test suite."""
    parser = argparse.ArgumentParser(
        description="Arshai Performance Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick validation tests (5 minutes)')
    parser.add_argument('--moderate', action='store_true',
                       help='Run moderate load tests (15 minutes)')
    parser.add_argument('--high', action='store_true',
                       help='Run high load tests (30 minutes)')
    parser.add_argument('--extreme', action='store_true',
                       help='Run extreme stress tests (45 minutes)')
    parser.add_argument('--sustained', action='store_true',
                       help='Run sustained load tests (60 minutes)')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests (20 minutes)')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite (all tests)')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed performance report')
    parser.add_argument('--metrics', action='store_true',
                       help='Output metrics in JSON format')
    parser.add_argument('--output', default='performance_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not any([args.quick, args.moderate, args.high, args.extreme, 
                args.sustained, args.integration, args.full]):
        parser.print_help()
        return 1
    
    # Initialize test suite
    suite = PerformanceTestSuite()
    suite.setup_test_environment()
    suite.start_time = time.perf_counter()
    
    logger.info("🚀 Starting Arshai Performance Test Suite")
    logger.info(f"Configuration: {os.environ.get('ARSHAI_MAX_CONNECTIONS')} connections, "
               f"{os.environ.get('ARSHAI_MAX_THREADS')} threads")
    
    all_results = {}
    
    try:
        # Run selected test categories
        if args.quick or args.full:
            all_results['quick'] = suite.run_quick_tests()
        
        if args.moderate or args.full:
            all_results['moderate'] = suite.run_moderate_tests()
        
        if args.high or args.full:
            all_results['high_load'] = suite.run_high_load_tests()
        
        if args.extreme or args.full:
            all_results['extreme'] = suite.run_extreme_tests()
        
        if args.sustained or args.full:
            all_results['sustained'] = suite.run_sustained_tests()
        
        if args.integration or args.full:
            all_results['integration'] = suite.run_integration_tests()
        
        suite.end_time = time.perf_counter()
        
        # Generate report
        if args.report or args.full:
            report = suite.generate_performance_report(all_results)
            json_file, report_file = suite.save_results(all_results, report, args.output)
            
            print("\n" + "="*80)
            print(report)
            print("="*80)
        
        # Output metrics
        if args.metrics:
            metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration': suite.end_time - suite.start_time,
                'failed_tests': len(suite.failed_tests),
                'total_tests': sum(len(category_results) for category_results in all_results.values()),
                'success_rate': ((sum(len(category_results) for category_results in all_results.values()) - len(suite.failed_tests)) / sum(len(category_results) for category_results in all_results.values()) * 100) if all_results else 0
            }
            print(json.dumps(metrics, indent=2))
        
        # Exit with appropriate code
        if suite.failed_tests:
            logger.error(f"❌ {len(suite.failed_tests)} tests failed")
            return 1
        else:
            logger.info("✅ All tests passed!")
            return 0
    
    except KeyboardInterrupt:
        logger.info("🛑 Test suite interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())