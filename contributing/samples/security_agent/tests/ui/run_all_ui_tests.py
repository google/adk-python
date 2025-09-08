#!/usr/bin/env python3

"""
Master UI Test Runner
====================

Executes all UI test suites and generates a comprehensive report.
Provides options for parallel execution, selective testing, and detailed reporting.

Usage:
    python tests/ui/run_all_ui_tests.py [options]

Options:
    --parallel      Run tests in parallel
    --headless      Run browser tests in headless mode (default: True)
    --browsers      Comma-separated list of browsers (chrome,firefox,safari)
    --quick         Run only critical tests
    --verbose       Detailed logging output
    --report-format Format for reports (text,json,html)
"""

import asyncio
import concurrent.futures
import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import test modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from test_navigation import NavigationTestSuite
    from test_pages import PagesTestSuite
    from test_components import ComponentsTestSuite
    from test_responsive import ResponsiveTestSuite
    from test_state import StateManagementTestSuite
    from playwright_automation import PlaywrightTestSuite
except ImportError as e:
    logger.error(f"Failed to import test modules: {e}")
    sys.exit(1)


class UITestOrchestrator:
    """Orchestrates the execution of all UI test suites."""
    
    def __init__(self, app_path: str = "frontend/unified_streaming_client.py"):
        self.app_path = app_path
        self.test_results = {}
        self.start_time = datetime.now()
        self.execution_summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "execution_time": 0,
            "test_suites": []
        }
    
    def validate_setup(self) -> bool:
        """Validate test environment and dependencies."""
        logger.info("🔍 Validating test environment...")
        
        # Check if Streamlit app exists
        if not Path(self.app_path).exists():
            logger.error(f"❌ Streamlit app not found at {self.app_path}")
            return False
        
        # Check required dependencies
        try:
            import selenium
            import playwright
            import httpx
            import requests
            logger.info("✅ All required dependencies found")
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            return False
        
        # Check browser drivers
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            
            driver = webdriver.Chrome(options=options)
            driver.quit()
            logger.info("✅ Chrome WebDriver available")
        except Exception as e:
            logger.warning(f"⚠️ Chrome WebDriver issue: {e}")
        
        return True
    
    def run_navigation_tests(self) -> Dict[str, Any]:
        """Run navigation test suite."""
        logger.info("🧭 Running Navigation Tests...")
        suite = NavigationTestSuite()
        try:
            results = suite.run_full_navigation_suite(self.app_path)
            report = suite.generate_report(results)
            
            # Save individual report
            report_file = Path("tests/ui/navigation_test_results.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "completed",
                "results": results,
                "report_file": str(report_file)
            }
        except Exception as e:
            logger.error(f"❌ Navigation tests failed: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "results": {}
            }
    
    def run_pages_tests(self) -> Dict[str, Any]:
        """Run pages functionality test suite."""
        logger.info("📄 Running Pages Tests...")
        suite = PagesTestSuite()
        try:
            results = suite.run_full_pages_suite(self.app_path)
            report = suite.generate_report(results)
            
            # Save individual report
            report_file = Path("tests/ui/pages_test_results.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "completed",
                "results": results,
                "report_file": str(report_file)
            }
        except Exception as e:
            logger.error(f"❌ Pages tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": {}
            }
    
    def run_components_tests(self) -> Dict[str, Any]:
        """Run components test suite."""
        logger.info("🔧 Running Components Tests...")
        suite = ComponentsTestSuite()
        try:
            results = suite.run_full_components_suite(self.app_path)
            report = suite.generate_report(results)
            
            # Save individual report
            report_file = Path("tests/ui/components_test_results.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "completed",
                "results": results,
                "report_file": str(report_file)
            }
        except Exception as e:
            logger.error(f"❌ Components tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": {}
            }
    
    def run_responsive_tests(self) -> Dict[str, Any]:
        """Run responsive design test suite."""
        logger.info("📱 Running Responsive Design Tests...")
        suite = ResponsiveTestSuite()
        try:
            results = suite.run_full_responsive_suite(self.app_path)
            report = suite.generate_report(results)
            
            # Save individual report
            report_file = Path("tests/ui/responsive_test_results.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "completed",
                "results": results,
                "report_file": str(report_file)
            }
        except Exception as e:
            logger.error(f"❌ Responsive tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": {}
            }
    
    def run_state_tests(self) -> Dict[str, Any]:
        """Run state management test suite."""
        logger.info("🔄 Running State Management Tests...")
        suite = StateManagementTestSuite()
        try:
            results = suite.run_full_state_suite(self.app_path)
            report = suite.generate_report(results)
            
            # Save individual report
            report_file = Path("tests/ui/state_test_results.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "completed",
                "results": results,
                "report_file": str(report_file)
            }
        except Exception as e:
            logger.error(f"❌ State management tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": {}
            }
    
    async def run_playwright_tests(self) -> Dict[str, Any]:
        """Run Playwright automation test suite."""
        logger.info("🎭 Running Playwright Automation Tests...")
        suite = PlaywrightTestSuite()
        try:
            results = await suite.run_full_playwright_suite(self.app_path)
            report = suite.generate_report()
            
            # Save individual report
            report_file = Path("tests/ui/playwright_test_results.txt")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "completed",
                "results": results,
                "report_file": str(report_file)
            }
        except Exception as e:
            logger.error(f"❌ Playwright tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "results": {}
            }
    
    def run_sequential_tests(self, test_suites: List[str] = None) -> Dict[str, Any]:
        """Run all test suites sequentially."""
        if test_suites is None:
            test_suites = ["navigation", "pages", "components", "responsive", "state", "playwright"]
        
        logger.info(f"🚀 Starting sequential UI test execution...")
        logger.info(f"📋 Test suites to run: {', '.join(test_suites)}")
        
        suite_runners = {
            "navigation": self.run_navigation_tests,
            "pages": self.run_pages_tests,
            "components": self.run_components_tests,
            "responsive": self.run_responsive_tests,
            "state": self.run_state_tests,
            "playwright": lambda: asyncio.run(self.run_playwright_tests())
        }
        
        results = {}
        
        for suite_name in test_suites:
            if suite_name in suite_runners:
                logger.info(f"▶️ Starting {suite_name} test suite...")
                start_time = time.time()
                
                try:
                    results[suite_name] = suite_runners[suite_name]()
                    execution_time = time.time() - start_time
                    results[suite_name]["execution_time"] = round(execution_time, 2)
                    
                    if results[suite_name]["status"] == "completed":
                        logger.info(f"✅ {suite_name} tests completed in {execution_time:.2f}s")
                    else:
                        logger.error(f"❌ {suite_name} tests failed")
                        
                except Exception as e:
                    logger.error(f"❌ {suite_name} tests crashed: {e}")
                    results[suite_name] = {
                        "status": "crashed",
                        "error": str(e),
                        "execution_time": round(time.time() - start_time, 2),
                        "results": {}
                    }
            else:
                logger.warning(f"⚠️ Unknown test suite: {suite_name}")
        
        return results
    
    def run_parallel_tests(self, test_suites: List[str] = None) -> Dict[str, Any]:
        """Run test suites in parallel (excluding Playwright)."""
        if test_suites is None:
            test_suites = ["navigation", "pages", "components", "responsive", "state"]
        
        logger.info(f"🚀 Starting parallel UI test execution...")
        logger.info(f"📋 Test suites to run in parallel: {', '.join(test_suites)}")
        
        suite_runners = {
            "navigation": self.run_navigation_tests,
            "pages": self.run_pages_tests,
            "components": self.run_components_tests,
            "responsive": self.run_responsive_tests,
            "state": self.run_state_tests
        }
        
        results = {}\n        \n        # Run parallel tests\n        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:\n            future_to_suite = {\n                executor.submit(suite_runners[suite_name]): suite_name \n                for suite_name in test_suites if suite_name in suite_runners\n            }\n            \n            for future in concurrent.futures.as_completed(future_to_suite):\n                suite_name = future_to_suite[future]\n                try:\n                    results[suite_name] = future.result()\n                    if results[suite_name]["status"] == "completed":\n                        logger.info(f"✅ {suite_name} tests completed")\n                    else:\n                        logger.error(f"❌ {suite_name} tests failed")\n                except Exception as e:\n                    logger.error(f"❌ {suite_name} tests crashed: {e}")\n                    results[suite_name] = {\n                        "status": "crashed",\n                        "error": str(e),\n                        "results": {}\n                    }\n        \n        # Run Playwright separately (requires async)\n        if "playwright" in test_suites:\n            logger.info("▶️ Running Playwright tests separately...")\n            results["playwright"] = asyncio.run(self.run_playwright_tests())\n        \n        return results\n    \n    def calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Calculate summary statistics from test results.\"\"\"\n        summary = {\n            "total_suites": len(results),\n            "completed_suites": 0,\n            "failed_suites": 0,\n            "crashed_suites": 0,\n            "total_tests": 0,\n            "passed_tests": 0,\n            "failed_tests": 0,\n            "overall_success_rate": 0.0,\n            "suite_details": {}\n        }\n        \n        for suite_name, suite_results in results.items():\n            suite_status = suite_results.get("status", "unknown")\n            \n            if suite_status == "completed":\n                summary["completed_suites"] += 1\n            elif suite_status == "failed":\n                summary["failed_suites"] += 1\n            elif suite_status == "crashed":\n                summary["crashed_suites"] += 1\n            \n            # Count individual tests\n            test_data = suite_results.get("results", {})\n            suite_passed = 0\n            suite_total = 0\n            \n            for section_name, section_results in test_data.items():\n                if isinstance(section_results, dict):\n                    for test_name, test_result in section_results.items():\n                        if isinstance(test_result, bool):\n                            suite_total += 1\n                            if test_result:\n                                suite_passed += 1\n                        elif isinstance(test_result, dict):\n                            # Handle nested test results\n                            for nested_test, nested_result in test_result.items():\n                                if isinstance(nested_result, bool):\n                                    suite_total += 1\n                                    if nested_result:\n                                        suite_passed += 1\n            \n            summary["total_tests"] += suite_total\n            summary["passed_tests"] += suite_passed\n            summary["failed_tests"] += (suite_total - suite_passed)\n            \n            suite_success_rate = (suite_passed / suite_total * 100) if suite_total > 0 else 0\n            summary["suite_details"][suite_name] = {\n                "status": suite_status,\n                "tests_passed": suite_passed,\n                "tests_total": suite_total,\n                "success_rate": round(suite_success_rate, 1),\n                "execution_time": suite_results.get("execution_time", 0)\n            }\n        \n        # Calculate overall success rate\n        if summary["total_tests"] > 0:\n            summary["overall_success_rate"] = round(\n                (summary["passed_tests"] / summary["total_tests"]) * 100, 1\n            )\n        \n        return summary\n    \n    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:\n        \"\"\"Generate a comprehensive test report.\"\"\"\n        summary = self.calculate_summary_statistics(results)\n        execution_time = (datetime.now() - self.start_time).total_seconds()\n        \n        report = [\"\\n\" + \"=\"*80]\n        report.append(\"                    COMPREHENSIVE UI TEST SUITE REPORT\")\n        report.append(\"=\"*80)\n        \n        # Executive Summary\n        report.append(f\"\\n📊 EXECUTIVE SUMMARY:\")\n        report.append(\"-\" * 50)\n        report.append(f\"  Test Execution Time: {execution_time:.2f} seconds\")\n        report.append(f\"  Test Suites Run: {summary['total_suites']}\")\n        report.append(f\"  Total Individual Tests: {summary['total_tests']}\")\n        report.append(f\"  Tests Passed: {summary['passed_tests']}\")\n        report.append(f\"  Tests Failed: {summary['failed_tests']}\")\n        report.append(f\"  Overall Success Rate: {summary['overall_success_rate']}%\")\n        \n        # Suite Status\n        report.append(f\"\\n✅ SUITE STATUS SUMMARY:\")\n        report.append(\"-\" * 50)\n        report.append(f\"  Completed Successfully: {summary['completed_suites']}\")\n        report.append(f\"  Failed: {summary['failed_suites']}\")\n        report.append(f\"  Crashed: {summary['crashed_suites']}\")\n        \n        # Detailed Results\n        report.append(f\"\\n🔍 DETAILED TEST SUITE RESULTS:\")\n        report.append(\"-\" * 50)\n        \n        for suite_name, details in summary[\"suite_details\"].items():\n            status_icon = {\n                \"completed\": \"✅\",\n                \"failed\": \"❌\", \n                \"crashed\": \"💥\"\n            }.get(details[\"status\"], \"❓\")\n            \n            report.append(f\"\\n  {status_icon} {suite_name.upper().replace('_', ' ')} TESTS:\")\n            report.append(f\"    Status: {details['status'].title()}\")\n            report.append(f\"    Tests Passed: {details['tests_passed']}/{details['tests_total']}\")\n            report.append(f\"    Success Rate: {details['success_rate']}%\")\n            report.append(f\"    Execution Time: {details['execution_time']:.2f}s\")\n            \n            if details[\"status\"] in [\"failed\", \"crashed\"]:\n                suite_results = results.get(suite_name, {})\n                error = suite_results.get(\"error\")\n                if error:\n                    report.append(f\"    Error: {error}\")\n        \n        # Recommendations\n        report.append(f\"\\n💡 RECOMMENDATIONS:\")\n        report.append(\"-\" * 50)\n        \n        if summary[\"overall_success_rate\"] >= 90:\n            report.append(\"  🎉 Excellent! UI is performing very well across all test scenarios.\")\n        elif summary[\"overall_success_rate\"] >= 80:\n            report.append(\"  👍 Good performance. Consider addressing failing tests for optimal UX.\")\n        elif summary[\"overall_success_rate\"] >= 70:\n            report.append(\"  ⚠️ Moderate performance. Several areas need attention for better UX.\")\n        else:\n            report.append(\"  🚨 Poor performance. Significant UI issues need immediate attention.\")\n        \n        if summary[\"failed_suites\"] > 0:\n            report.append(f\"  📋 Review {summary['failed_suites']} failed test suite(s) for specific issues.\")\n        \n        if summary[\"crashed_suites\"] > 0:\n            report.append(f\"  🔧 Fix {summary['crashed_suites']} crashed test suite(s) - may indicate setup issues.\")\n        \n        report.append(f\"\\n📁 INDIVIDUAL REPORTS:\")\n        report.append(\"-\" * 50)\n        for suite_name in results.keys():\n            report_file = f\"tests/ui/{suite_name}_test_results.txt\"\n            if Path(report_file).exists():\n                report.append(f\"  📄 {suite_name.title()}: {report_file}\")\n        \n        report.append(\"\\n\" + \"=\"*80)\n        report.append(f\"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")\n        report.append(\"GCP Security Agent UI Testing Suite v1.13.0\")\n        report.append(\"=\"*80)\n        \n        return \"\\n\".join(report)\n    \n    def save_results(self, results: Dict[str, Any], report: str):\n        \"\"\"Save test results and reports to files.\"\"\"\n        # Save comprehensive report\n        report_file = Path(\"tests/ui/COMPREHENSIVE_TEST_RESULTS.txt\")\n        with open(report_file, \"w\") as f:\n            f.write(report)\n        \n        # Save JSON results\n        json_file = Path(\"tests/ui/all_ui_test_results.json\")\n        with open(json_file, \"w\") as f:\n            json.dump({\n                \"execution_timestamp\": self.start_time.isoformat(),\n                \"execution_duration\": (datetime.now() - self.start_time).total_seconds(),\n                \"summary\": self.calculate_summary_statistics(results),\n                \"detailed_results\": results\n            }, f, indent=2, default=str)\n        \n        logger.info(f\"📄 Comprehensive report saved to: {report_file}\")\n        logger.info(f\"📊 JSON results saved to: {json_file}\")\n\n\ndef main():\n    \"\"\"Main execution function.\"\"\"\n    parser = argparse.ArgumentParser(description=\"Run comprehensive UI test suite\")\n    parser.add_argument(\"--parallel\", action=\"store_true\", help=\"Run tests in parallel\")\n    parser.add_argument(\"--quick\", action=\"store_true\", help=\"Run only critical tests\")\n    parser.add_argument(\"--suites\", type=str, help=\"Comma-separated list of test suites to run\")\n    parser.add_argument(\"--app-path\", type=str, default=\"frontend/unified_streaming_client.py\", \n                       help=\"Path to Streamlit application\")\n    parser.add_argument(\"--verbose\", action=\"store_true\", help=\"Verbose output\")\n    \n    args = parser.parse_args()\n    \n    if args.verbose:\n        logging.getLogger().setLevel(logging.DEBUG)\n    \n    # Determine which test suites to run\n    if args.suites:\n        test_suites = [suite.strip() for suite in args.suites.split(\",\")]\n    elif args.quick:\n        test_suites = [\"navigation\", \"pages\"]  # Quick critical tests only\n    else:\n        test_suites = [\"navigation\", \"pages\", \"components\", \"responsive\", \"state\", \"playwright\"]\n    \n    # Initialize orchestrator\n    orchestrator = UITestOrchestrator(args.app_path)\n    \n    # Validate setup\n    if not orchestrator.validate_setup():\n        logger.error(\"❌ Test environment validation failed\")\n        return 1\n    \n    # Run tests\n    logger.info(f\"🎯 Starting UI test execution with {len(test_suites)} suites...\")\n    \n    try:\n        if args.parallel and len(test_suites) > 1:\n            results = orchestrator.run_parallel_tests(test_suites)\n        else:\n            results = orchestrator.run_sequential_tests(test_suites)\n        \n        # Generate and save comprehensive report\n        report = orchestrator.generate_comprehensive_report(results)\n        print(report)\n        \n        orchestrator.save_results(results, report)\n        \n        # Determine exit code\n        summary = orchestrator.calculate_summary_statistics(results)\n        success_rate = summary[\"overall_success_rate\"]\n        \n        if success_rate >= 80:\n            logger.info(f\"🎉 UI tests completed successfully! Success rate: {success_rate}%\")\n            return 0\n        elif success_rate >= 70:\n            logger.warning(f\"⚠️ UI tests completed with warnings. Success rate: {success_rate}%\")\n            return 1\n        else:\n            logger.error(f\"❌ UI tests failed. Success rate: {success_rate}%\")\n            return 2\n            \n    except KeyboardInterrupt:\n        logger.info(\"🛑 Test execution interrupted by user\")\n        return 130\n    except Exception as e:\n        logger.error(f\"💥 Test execution crashed: {e}\")\n        return 1\n\n\nif __name__ == \"__main__\":\n    exit(main())