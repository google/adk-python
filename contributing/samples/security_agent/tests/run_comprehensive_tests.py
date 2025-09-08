"""
Comprehensive Test Runner
========================

Runs all test suites and generates comprehensive coverage reports:
- Unit tests
- Integration tests  
- Security tests
- Performance tests
- End-to-end tests
- Coverage analysis
- Test reporting
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path


class TestRunner:
    """Comprehensive test runner with reporting."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(self, suite_name, test_path, options=None):
        """Run a specific test suite and capture results."""
        if options is None:
            options = []
        
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_path),
            "-v", 
            "--tb=short",
            "--capture=no"
        ] + options
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
            end_time = time.time()
            
            # Parse results
            duration = end_time - start_time
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
            
            # Extract test counts from output
            passed = stdout.count(" PASSED")
            failed = stdout.count(" FAILED")
            errors = stdout.count(" ERROR")
            skipped = stdout.count(" SKIPPED")
            
            self.results[suite_name] = {
                "duration": duration,
                "returncode": returncode,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "total": passed + failed + errors + skipped,
                "stdout": stdout,
                "stderr": stderr,
                "success": returncode == 0
            }
            
            print(f"✅ {suite_name} completed in {duration:.2f}s")
            print(f"   Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")
            
            if not self.results[suite_name]["success"]:
                print(f"❌ {suite_name} had failures:")
                if stderr:
                    print(f"STDERR: {stderr[:500]}")
                print(f"STDOUT: {stdout[-1000:]}")  # Last 1000 chars
            
        except Exception as e:
            self.results[suite_name] = {
                "duration": 0,
                "returncode": -1,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "total": 1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
            print(f"❌ {suite_name} failed to run: {e}")
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 Starting Comprehensive Test Suite")
        print(f"📅 Started at: {datetime.now().isoformat()}")
        self.start_time = time.time()
        
        # Test suites to run
        test_suites = [
            ("Unit Tests", "unit/test_backend_core.py"),
            ("Integration Tests", "integration/test_api_integration.py"),
            ("Security Tests", "security/test_security_validation.py"),
            ("Performance Tests", "performance/test_performance_benchmarks.py"),
            ("End-to-End Tests", "e2e/test_user_workflows.py"),
        ]
        
        # Run existing backend tests if they exist
        backend_tests_dir = self.test_dir.parent / "backend" / "tests"
        if backend_tests_dir.exists():
            test_suites.append(("Backend Unit Tests", "../backend/tests"))
        
        # Run tests from main tests directory if they exist
        main_test_files = [
            "test_api_endpoints.py",
            "test_chat_responses.py", 
            "test_integration.py",
            "test_security.py"
        ]
        
        for test_file in main_test_files:
            test_path = self.test_dir / test_file
            if test_path.exists():
                test_suites.append((f"Existing {test_file}", test_file))
        
        # Run each test suite
        for suite_name, test_path in test_suites:
            self.run_test_suite(suite_name, test_path)
        
        self.end_time = time.time()
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        total_duration = self.end_time - self.start_time
        
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"🕐 Total Duration: {total_duration:.2f} seconds")
        print(f"📅 Completed at: {datetime.now().isoformat()}")
        
        # Calculate totals
        total_passed = sum(result["passed"] for result in self.results.values())
        total_failed = sum(result["failed"] for result in self.results.values())
        total_errors = sum(result["errors"] for result in self.results.values())
        total_skipped = sum(result["skipped"] for result in self.results.values())
        total_tests = sum(result["total"] for result in self.results.values())
        
        successful_suites = sum(1 for result in self.results.values() if result["success"])
        total_suites = len(self.results)
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Test Suites: {total_suites}")
        print(f"   Successful Suites: {successful_suites}")
        print(f"   Failed Suites: {total_suites - successful_suites}")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   🚫 Errors: {total_errors}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        
        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            print(f"   📊 Pass Rate: {pass_rate:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for suite_name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"   {status} {suite_name}: {result['passed']}/{result['total']} tests in {result['duration']:.2f}s")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if total_failed > 0 or total_errors > 0:
            print("   🔧 Address failing tests before production deployment")
        if total_skipped > 10:
            print("   ⚠️  High number of skipped tests - review test coverage")
        if successful_suites == total_suites:
            print("   🎉 All test suites passed! System is ready for deployment")
        
        # Save detailed report
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """Save detailed test report to file."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "duration": self.end_time - self.start_time,
            "summary": {
                "total_suites": len(self.results),
                "successful_suites": sum(1 for r in self.results.values() if r["success"]),
                "total_tests": sum(r["total"] for r in self.results.values()),
                "total_passed": sum(r["passed"] for r in self.results.values()),
                "total_failed": sum(r["failed"] for r in self.results.values()),
                "total_errors": sum(r["errors"] for r in self.results.values()),
                "total_skipped": sum(r["skipped"] for r in self.results.values()),
            },
            "results": self.results
        }
        
        # Remove stdout/stderr from saved report to keep it manageable
        clean_results = {}
        for suite_name, result in self.results.items():
            clean_results[suite_name] = {
                "duration": result["duration"],
                "returncode": result["returncode"],
                "passed": result["passed"],
                "failed": result["failed"], 
                "errors": result["errors"],
                "skipped": result["skipped"],
                "total": result["total"],
                "success": result["success"]
            }
        report_data["results"] = clean_results
        
        report_file = self.test_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main entry point for test runner."""
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} found")
    except ImportError:
        print("❌ pytest not found. Please install with: pip install pytest")
        sys.exit(1)
    
    # Check if required modules are available
    try:
        import psutil
        print(f"✅ psutil {psutil.__version__} found")
    except ImportError:
        print("⚠️  psutil not found. Performance tests may be limited. Install with: pip install psutil")
    
    # Run comprehensive tests
    runner = TestRunner()
    try:
        runner.run_all_tests()
        
        # Exit with error code if any tests failed
        failed_suites = sum(1 for result in runner.results.values() if not result["success"])
        if failed_suites > 0:
            print(f"\n❌ {failed_suites} test suite(s) failed")
            sys.exit(1)
        else:
            print(f"\n🎉 All test suites passed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()