#!/usr/bin/env python3
"""
Comprehensive test runner for Cloud Functions test suite.
Supports unit tests, integration tests, performance tests, and security scans.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import shutil

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TEST_ROOT = Path(__file__).parent
COVERAGE_MIN = 80
MAX_TEST_DURATION = 600  # 10 minutes


class TestRunner:
    """Comprehensive test runner for Cloud Functions"""

    def __init__(self, verbose=False, quick=False):
        self.verbose = verbose
        self.quick = quick
        self.results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "test_suites": [],
            "coverage": {},
            "security_scan": {},
            "performance": {},
            "summary": {}
        }

        # Ensure test environment
        self._setup_environment()

    def _setup_environment(self):
        """Set up test environment variables"""
        test_env = {
            "PROJECT_ID": "test-project-123",
            "BQ_DATASET_ID": "test_security_insights",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/test_credentials.json",
            "TESTING": "true",
            "PYTHONPATH": str(PROJECT_ROOT)
        }

        for key, value in test_env.items():
            os.environ[key] = value

    def _run_command(self, cmd, cwd=None, timeout=None):
        """Run shell command with error handling"""
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or TEST_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if self.verbose and result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(f"Error: {result.stderr}", file=sys.stderr)

            return result

        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout}s", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Command failed: {e}", file=sys.stderr)
            return None

    def install_dependencies(self):
        """Install test dependencies"""
        print("📦 Installing test dependencies...")

        # Install test requirements
        requirements_file = TEST_ROOT / "requirements.txt"
        if requirements_file.exists():
            result = self._run_command([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])

            if result and result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ Failed to install dependencies")
                return False
        else:
            print("⚠️  No requirements.txt found, skipping dependency installation")
            return True

    def run_unit_tests(self):
        """Run unit tests with coverage"""
        print("\n🧪 Running unit tests...")

        suite_start = time.time()

        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest",
            str(TEST_ROOT / "unit"),
            "-v",
            "--tb=short",
            f"--cov={PROJECT_ROOT}",
            "--cov-report=json",
            "--cov-report=term-missing",
            f"--cov-fail-under={COVERAGE_MIN}",
            "--junitxml=test_results_unit.xml"
        ]

        if not self.verbose:
            cmd.append("-q")

        result = self._run_command(cmd, timeout=300)

        suite_duration = time.time() - suite_start

        # Parse coverage results
        coverage_file = TEST_ROOT / "coverage.json"
        coverage_data = {}
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
            except Exception as e:
                print(f"Failed to parse coverage data: {e}")

        suite_result = {
            "name": "unit_tests",
            "status": "passed" if result and result.returncode == 0 else "failed",
            "duration": suite_duration,
            "coverage": coverage_data.get("totals", {}).get("percent_covered", 0) if coverage_data else 0,
            "test_count": self._count_tests("unit"),
            "output": result.stdout if result else "No output"
        }

        self.results["test_suites"].append(suite_result)
        self.results["coverage"] = coverage_data.get("totals", {}) if coverage_data else {}

        if suite_result["status"] == "passed":
            print(f"✅ Unit tests passed ({suite_result['test_count']} tests, {suite_result['coverage']:.1f}% coverage)")
        else:
            print(f"❌ Unit tests failed")

        return suite_result["status"] == "passed"

    def run_integration_tests(self):
        """Run integration tests"""
        if self.quick:
            print("⏭️  Skipping integration tests (quick mode)")
            return True

        print("\n🔗 Running integration tests...")

        suite_start = time.time()

        cmd = [
            sys.executable, "-m", "pytest",
            str(TEST_ROOT / "integration"),
            "-v",
            "--tb=short",
            "--junitxml=test_results_integration.xml",
            "-m", "not slow"  # Skip slow tests in normal mode
        ]

        if not self.verbose:
            cmd.append("-q")

        result = self._run_command(cmd, timeout=400)

        suite_duration = time.time() - suite_start

        suite_result = {
            "name": "integration_tests",
            "status": "passed" if result and result.returncode == 0 else "failed",
            "duration": suite_duration,
            "test_count": self._count_tests("integration"),
            "output": result.stdout if result else "No output"
        }

        self.results["test_suites"].append(suite_result)

        if suite_result["status"] == "passed":
            print(f"✅ Integration tests passed ({suite_result['test_count']} tests)")
        else:
            print(f"❌ Integration tests failed")

        return suite_result["status"] == "passed"

    def run_performance_tests(self):
        """Run performance tests"""
        if self.quick:
            print("⏭️  Skipping performance tests (quick mode)")
            return True

        print("\n⚡ Running performance tests...")

        suite_start = time.time()

        cmd = [
            sys.executable, "-m", "pytest",
            str(TEST_ROOT / "performance"),
            "-v",
            "--tb=short",
            "--junitxml=test_results_performance.xml"
        ]

        if not self.verbose:
            cmd.append("-q")

        result = self._run_command(cmd, timeout=300)

        suite_duration = time.time() - suite_start

        suite_result = {
            "name": "performance_tests",
            "status": "passed" if result and result.returncode == 0 else "failed",
            "duration": suite_duration,
            "test_count": self._count_tests("performance"),
            "output": result.stdout if result else "No output"
        }

        self.results["test_suites"].append(suite_result)
        self.results["performance"] = {
            "duration": suite_duration,
            "status": suite_result["status"]
        }

        if suite_result["status"] == "passed":
            print(f"✅ Performance tests passed ({suite_result['test_count']} tests)")
        else:
            print(f"❌ Performance tests failed")

        return suite_result["status"] == "passed"

    def run_security_scan(self):
        """Run security scans"""
        if self.quick:
            print("⏭️  Skipping security scan (quick mode)")
            return True

        print("\n🔒 Running security scans...")

        scan_results = {
            "bandit": self._run_bandit_scan(),
            "safety": self._run_safety_scan(),
            "status": "passed"
        }

        # Overall status
        if not scan_results["bandit"]["status"] or not scan_results["safety"]["status"]:
            scan_results["status"] = "failed"

        self.results["security_scan"] = scan_results

        if scan_results["status"] == "passed":
            print("✅ Security scans passed")
        else:
            print("❌ Security scans failed")

        return scan_results["status"] == "passed"

    def _run_bandit_scan(self):
        """Run Bandit security scan"""
        try:
            cmd = ["bandit", "-r", str(PROJECT_ROOT), "-f", "json", "-o", "bandit_results.json"]
            result = self._run_command(cmd, timeout=120)

            # Bandit returns non-zero for issues found, but that's not necessarily a failure
            bandit_data = {}
            bandit_file = TEST_ROOT / "bandit_results.json"

            if bandit_file.exists():
                try:
                    with open(bandit_file, 'r') as f:
                        bandit_data = json.load(f)
                except Exception:
                    pass

            high_severity = len([r for r in bandit_data.get("results", [])
                               if r.get("issue_severity") == "HIGH"])

            return {
                "status": high_severity == 0,  # Pass if no high-severity issues
                "high_severity_count": high_severity,
                "total_issues": len(bandit_data.get("results", [])),
                "output": result.stdout if result else "No output"
            }

        except Exception as e:
            print(f"Bandit scan failed: {e}")
            return {"status": False, "error": str(e)}

    def _run_safety_scan(self):
        """Run Safety vulnerability scan"""
        try:
            cmd = ["safety", "check", "--json"]
            result = self._run_command(cmd, timeout=60)

            if result and result.returncode == 0:
                return {
                    "status": True,
                    "vulnerabilities": 0,
                    "output": result.stdout
                }
            else:
                # Parse vulnerabilities if any
                vuln_count = 0
                if result and result.stdout:
                    try:
                        safety_data = json.loads(result.stdout)
                        vuln_count = len(safety_data)
                    except Exception:
                        pass

                return {
                    "status": vuln_count == 0,
                    "vulnerabilities": vuln_count,
                    "output": result.stdout if result else "No output"
                }

        except Exception as e:
            print(f"Safety scan failed: {e}")
            return {"status": False, "error": str(e)}

    def _count_tests(self, test_dir):
        """Count number of tests in directory"""
        try:
            test_path = TEST_ROOT / test_dir
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "--collect-only", "-q"
            ]

            result = self._run_command(cmd)

            if result and result.returncode == 0:
                # Parse pytest collection output
                lines = result.stdout.split('\n')
                for line in lines:
                    if ' tests collected' in line:
                        return int(line.split()[0])

            return 0

        except Exception:
            return 0

    def generate_report(self):
        """Generate comprehensive test report"""

        # Calculate summary
        total_duration = sum(suite["duration"] for suite in self.results["test_suites"])
        total_tests = sum(suite["test_count"] for suite in self.results["test_suites"])
        passed_suites = len([s for s in self.results["test_suites"] if s["status"] == "passed"])
        total_suites = len(self.results["test_suites"])

        self.results["end_time"] = datetime.now(timezone.utc).isoformat()
        self.results["summary"] = {
            "total_duration": total_duration,
            "total_tests": total_tests,
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "success_rate": (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            "coverage_percentage": self.results.get("coverage", {}).get("percent_covered", 0),
            "overall_status": "passed" if passed_suites == total_suites else "failed"
        }

        # Save detailed report
        report_file = TEST_ROOT / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate summary report
        self._print_summary_report()

        return self.results["summary"]["overall_status"] == "passed"

    def _print_summary_report(self):
        """Print summary report to console"""
        print("\n" + "="*60)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*60)

        summary = self.results["summary"]

        print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
        print(f"🧪 Total Tests: {summary['total_tests']}")
        print(f"📦 Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
        print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
        print(f"📈 Coverage: {summary['coverage_percentage']:.1f}%")

        # Suite breakdown
        print(f"\n📋 Suite Breakdown:")
        for suite in self.results["test_suites"]:
            status_icon = "✅" if suite["status"] == "passed" else "❌"
            print(f"  {status_icon} {suite['name']}: {suite['test_count']} tests ({suite['duration']:.1f}s)")

        # Security scan
        if self.results.get("security_scan"):
            scan = self.results["security_scan"]
            scan_icon = "✅" if scan["status"] == "passed" else "❌"
            print(f"  {scan_icon} security_scan: {scan.get('bandit', {}).get('total_issues', 0)} issues")

        # Overall result
        if summary["overall_status"] == "passed":
            print(f"\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n💥 SOME TESTS FAILED!")

        print("="*60)

    def run_all(self):
        """Run complete test suite"""
        print("🚀 Starting Cloud Functions test suite...")
        print(f"📁 Project root: {PROJECT_ROOT}")
        print(f"🧪 Test root: {TEST_ROOT}")
        print(f"⚡ Quick mode: {self.quick}")

        success = True

        # Install dependencies
        if not self.install_dependencies():
            return False

        # Run test suites
        if not self.run_unit_tests():
            success = False

        if not self.run_integration_tests():
            success = False

        if not self.run_performance_tests():
            success = False

        if not self.run_security_scan():
            success = False

        # Generate report
        return self.generate_report() and success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Cloud Functions Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quick", action="store_true", help="Quick mode (skip slow tests)")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--security-only", action="store_true", help="Run only security scans")

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, quick=args.quick)

    success = True

    if args.unit_only:
        success = runner.install_dependencies() and runner.run_unit_tests()
    elif args.integration_only:
        success = runner.install_dependencies() and runner.run_integration_tests()
    elif args.performance_only:
        success = runner.install_dependencies() and runner.run_performance_tests()
    elif args.security_only:
        success = runner.run_security_scan()
    else:
        success = runner.run_all()

    # Generate final report
    if not (args.unit_only or args.integration_only or args.performance_only or args.security_only):
        runner.generate_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()