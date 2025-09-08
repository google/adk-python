#!/usr/bin/env python3
"""
Playwright Test Runner for Security Agent
Run comprehensive E2E tests using Playwright
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class PlaywrightTestRunner:
    """Manages and runs Playwright tests for the Security Agent."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent
        self.test_results = []
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        # Check if npm/node is installed
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            print("✅ Node.js is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js is not installed. Please install Node.js first.")
            return False
            
        # Check if Playwright is installed
        playwright_installed = self.project_root / "node_modules" / "@playwright" / "test"
        if not playwright_installed.exists():
            print("📦 Installing Playwright...")
            try:
                subprocess.run(["npm", "install", "@playwright/test"], 
                             cwd=self.project_root, check=True)
                subprocess.run(["npx", "playwright", "install"], 
                             cwd=self.project_root, check=True)
                print("✅ Playwright installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install Playwright: {e}")
                return False
        else:
            print("✅ Playwright is installed")
            
        # Check if backend and frontend are accessible
        import httpx
        try:
            response = httpx.get("http://localhost:8000/health", timeout=2)
            print("✅ Backend is running")
        except:
            print("⚠️  Backend is not running. Tests will start it automatically.")
            
        try:
            response = httpx.get("http://localhost:8501", timeout=2)
            print("✅ Frontend is running")
        except:
            print("⚠️  Frontend is not running. Tests will start it automatically.")
            
        return True
        
    def setup_test_environment(self) -> None:
        """Set up the test environment."""
        print("\n🔧 Setting up test environment...")
        
        # Create test directories
        test_dirs = [
            self.project_root / "tests" / "e2e",
            self.project_root / "tests" / "fixtures",
            self.project_root / "playwright-report",
            self.project_root / "test-results"
        ]
        
        for test_dir in test_dirs:
            test_dir.mkdir(parents=True, exist_ok=True)
            
        # Set environment variables for testing
        os.environ["TESTING"] = "true"
        os.environ["FRONTEND_URL"] = "http://localhost:8501"
        os.environ["BACKEND_URL"] = "http://localhost:8000"
        
        print("✅ Test environment configured")
        
    def run_tests(self, test_pattern: Optional[str] = None, 
                  browser: str = "chromium",
                  headed: bool = False) -> Dict[str, Any]:
        """Run Playwright tests."""
        print(f"\n🎭 Running Playwright tests...")
        
        # Build test command
        cmd = ["npx", "playwright", "test"]
        
        if test_pattern:
            cmd.append(test_pattern)
            
        if not headed:
            cmd.append("--headed=false")
        else:
            cmd.append("--headed")
            
        if browser != "all":
            cmd.extend(["--project", browser])
            
        if self.verbose:
            cmd.append("--debug")
            
        # Add reporter
        cmd.extend(["--reporter", "json,html,list"])
        
        print(f"📝 Command: {' '.join(cmd)}")
        
        # Run tests
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse results
            test_results = {
                "success": result.returncode == 0,
                "duration": duration,
                "output": result.stdout,
                "errors": result.stderr,
                "exit_code": result.returncode
            }
            
            # Try to parse JSON results if available
            json_results_path = self.project_root / "test-results.json"
            if json_results_path.exists():
                with open(json_results_path) as f:
                    test_results["detailed_results"] = json.load(f)
                    
            return test_results
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }
            
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate test report."""
        print("\n📊 Generating test report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": results.get("duration", 0),
            "success": results.get("success", False),
            "summary": {}
        }
        
        if "detailed_results" in results:
            detailed = results["detailed_results"]
            report["summary"] = {
                "total_tests": detailed.get("stats", {}).get("expected", 0),
                "passed": detailed.get("stats", {}).get("expected", 0) - 
                         detailed.get("stats", {}).get("unexpected", 0),
                "failed": detailed.get("stats", {}).get("unexpected", 0),
                "skipped": detailed.get("stats", {}).get("skipped", 0)
            }
            
        # Save report
        report_path = self.project_root / "playwright-test-report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"✅ Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("TEST EXECUTION SUMMARY")
        print("="*50)
        print(f"Duration: {report['duration']:.2f} seconds")
        print(f"Status: {'✅ PASSED' if report['success'] else '❌ FAILED'}")
        
        if report["summary"]:
            print(f"Total Tests: {report['summary']['total_tests']}")
            print(f"Passed: {report['summary']['passed']}")
            print(f"Failed: {report['summary']['failed']}")
            print(f"Skipped: {report['summary']['skipped']}")
            
        print("\n📁 View detailed HTML report:")
        print(f"   npx playwright show-report")
        
    def run_specific_test_suites(self) -> None:
        """Run specific test suites in sequence."""
        test_suites = [
            ("Dashboard Tests", "dashboard"),
            ("Chat Interface Tests", "chat"),
            ("Security Analysis Tests", "security"),
            ("API Integration Tests", "api"),
            ("Performance Tests", "performance"),
            ("Accessibility Tests", "accessibility")
        ]
        
        print("\n🔄 Running test suites sequentially...")
        
        all_results = []
        for suite_name, pattern in test_suites:
            print(f"\n▶️  Running: {suite_name}")
            results = self.run_tests(test_pattern=pattern)
            all_results.append({
                "suite": suite_name,
                "results": results
            })
            
            if not results.get("success"):
                print(f"⚠️  {suite_name} had failures")
                
        return all_results
        
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Playwright tests for Security Agent")
    parser.add_argument("--browser", choices=["chromium", "firefox", "webkit", "all"],
                       default="chromium", help="Browser to test with")
    parser.add_argument("--headed", action="store_true", 
                       help="Run tests in headed mode (show browser)")
    parser.add_argument("--pattern", help="Test pattern to match")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--suite", action="store_true", 
                       help="Run all test suites sequentially")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick smoke tests only")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PlaywrightTestRunner(verbose=args.verbose)
    
    # Check prerequisites
    if not runner.check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        sys.exit(1)
        
    # Setup environment
    runner.setup_test_environment()
    
    # Run tests
    if args.quick:
        print("\n🚀 Running quick smoke tests...")
        results = runner.run_tests(test_pattern="smoke", browser=args.browser, 
                                 headed=args.headed)
    elif args.suite:
        results = runner.run_specific_test_suites()
    else:
        results = runner.run_tests(test_pattern=args.pattern, 
                                 browser=args.browser,
                                 headed=args.headed)
        
    # Generate report
    if isinstance(results, list):
        # Suite results
        for suite_result in results:
            print(f"\n📊 {suite_result['suite']}:")
            runner.generate_report(suite_result['results'])
    else:
        runner.generate_report(results)
        
    # Exit with appropriate code
    if isinstance(results, list):
        success = all(r['results'].get('success', False) for r in results)
    else:
        success = results.get('success', False)
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()