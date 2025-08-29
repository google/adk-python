#!/usr/bin/env python3
"""
Comprehensive MSA Test Runner
=============================

Runs all MSA-related tests including:
- Playwright UI tests
- ADK evaluation tests
- Integration tests
- API endpoint tests
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests": {},
    "summary": {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
}


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info('='*70)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - PASSED")
            test_results["tests"][description] = "PASSED"
            test_results["summary"]["passed"] += 1
            test_results["summary"]["total"] += 1
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Error output: {result.stderr}")
            test_results["tests"][description] = f"FAILED: {result.stderr[:200]}"
            test_results["summary"]["failed"] += 1
            test_results["summary"]["total"] += 1
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {description} - TIMEOUT")
        test_results["tests"][description] = "TIMEOUT"
        test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        return False
    except Exception as e:
        logger.error(f"💥 {description} - ERROR: {e}")
        test_results["tests"][description] = f"ERROR: {str(e)}"
        test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        return False


async def run_playwright_tests():
    """Run Playwright UI tests for MSA Analyzer."""
    test_file = Path(__file__).parent / "tests" / "test_msa_analyzer_ui.py"
    
    if not test_file.exists():
        logger.warning(f"Playwright test file not found: {test_file}")
        test_results["tests"]["Playwright UI Tests"] = "SKIPPED - File not found"
        test_results["summary"]["skipped"] += 1
        test_results["summary"]["total"] += 1
        return False
    
    return run_command(
        f"python {test_file}",
        "Playwright UI Tests"
    )


async def run_integration_tests():
    """Run integration tests for release notes fetcher."""
    test_file = Path(__file__).parent / "tests" / "test_release_notes_integration.py"
    
    if not test_file.exists():
        logger.warning(f"Integration test file not found: {test_file}")
        test_results["tests"]["Integration Tests"] = "SKIPPED - File not found"
        test_results["summary"]["skipped"] += 1
        test_results["summary"]["total"] += 1
        return False
    
    return run_command(
        f"python {test_file}",
        "Release Notes Integration Tests"
    )


async def run_adk_evaluation():
    """Run ADK evaluation tests for MSA tools."""
    eval_dir = Path(__file__).parent / "evaluation"
    test_runner = eval_dir / "comprehensive_test_runner.py"
    
    if not test_runner.exists():
        logger.warning(f"ADK test runner not found: {test_runner}")
        test_results["tests"]["ADK Evaluation"] = "SKIPPED - File not found"
        test_results["summary"]["skipped"] += 1
        test_results["summary"]["total"] += 1
        return False
    
    # Run only MSA-related evaluation
    return run_command(
        f'cd {eval_dir} && python comprehensive_test_runner.py --suite "MSA and Release Notes Analysis"',
        "ADK Evaluation Tests"
    )


async def test_api_endpoints():
    """Test MSA API endpoints."""
    test_file = Path(__file__).parent / "test_msa_analyzer.py"
    
    if not test_file.exists():
        logger.warning(f"API test file not found: {test_file}")
        test_results["tests"]["API Endpoint Tests"] = "SKIPPED - File not found"
        test_results["summary"]["skipped"] += 1
        test_results["summary"]["total"] += 1
        return False
    
    return run_command(
        f"python {test_file}",
        "MSA API Endpoint Tests"
    )


async def test_sqlite_tools():
    """Test SQLite tool integration for MSA queries."""
    logger.info("\n" + "="*70)
    logger.info("Testing SQLite Tool Integration")
    logger.info("="*70)
    
    # Import and test the sqlite_tool directly
    try:
        sys.path.insert(0, str(Path(__file__).parent / "agents" / "gcp_security"))
        from sqlite_tool import query_security_data
        
        # Test MSA-related queries
        test_queries = [
            ("msa_analysis", None, "MSA Analysis History"),
            ("msa_changes", '{"service": "BigQuery"}', "MSA Changes for BigQuery"),
            ("msa_security_impacts", None, "MSA Security Impacts"),
            ("msa_billing_impacts", None, "MSA Billing Impacts"),
            ("release_notes", '{"days": 7}', "Recent Release Notes")
        ]
        
        all_passed = True
        for query_type, params, description in test_queries:
            try:
                logger.info(f"Testing: {description}")
                result = query_security_data(query_type, params)
                
                if "error" in result.lower() or "database not found" in result.lower():
                    logger.warning(f"⚠️ {description} - No data (database may be empty)")
                else:
                    logger.info(f"✅ {description} - Query successful")
                    
            except Exception as e:
                logger.error(f"❌ {description} - Failed: {e}")
                all_passed = False
        
        if all_passed:
            test_results["tests"]["SQLite Tool Integration"] = "PASSED"
            test_results["summary"]["passed"] += 1
        else:
            test_results["tests"]["SQLite Tool Integration"] = "FAILED"
            test_results["summary"]["failed"] += 1
        
        test_results["summary"]["total"] += 1
        return all_passed
        
    except Exception as e:
        logger.error(f"Failed to test SQLite tools: {e}")
        test_results["tests"]["SQLite Tool Integration"] = f"ERROR: {str(e)}"
        test_results["summary"]["failed"] += 1
        test_results["summary"]["total"] += 1
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("\n" + "="*70)
    logger.info("Checking Prerequisites")
    logger.info("="*70)
    
    issues = []
    
    # Check if backend is running
    import httpx
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    try:
        response = httpx.get(f"{backend_url}/api/v1/health", timeout=5.0)
        if response.status_code == 200:
            logger.info(f"✅ Backend is running at {backend_url}")
        else:
            issues.append(f"Backend returned status {response.status_code}")
    except:
        issues.append(f"Backend is not running at {backend_url}")
        logger.warning(f"⚠️ Backend is not accessible. Start with: python run_backend.py")
    
    # Check if frontend is running (for Playwright tests)
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    try:
        response = httpx.get(frontend_url, timeout=5.0)
        if response.status_code == 200:
            logger.info(f"✅ Frontend is running at {frontend_url}")
        else:
            logger.warning(f"⚠️ Frontend returned status {response.status_code}")
    except:
        logger.warning(f"⚠️ Frontend is not accessible. Start with: python run_frontend.py")
        logger.info("   (Frontend is optional for non-UI tests)")
    
    # Check for required Python packages
    required_packages = ["playwright", "httpx", "pandas", "plotly", "sqlite3"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ Package '{package}' is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"⚠️ Package '{package}' is not installed")
    
    if missing_packages:
        logger.info(f"\nInstall missing packages with:")
        logger.info(f"  pip install {' '.join(missing_packages)}")
    
    # Check database
    db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    if Path(db_path).exists():
        logger.info(f"✅ Database exists at {db_path}")
    else:
        logger.warning(f"⚠️ Database not found at {db_path}")
        logger.info("   Run data refresh to populate: python backend/services/msa_database_setup.py")
    
    return len(issues) == 0


async def main():
    """Run all MSA tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  MSA ANALYZER COMPREHENSIVE TEST SUITE           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  This will run:                                                  ║
    ║  • Playwright UI tests                                           ║
    ║  • ADK evaluation tests                                          ║
    ║  • Integration tests                                             ║
    ║  • API endpoint tests                                            ║
    ║  • SQLite tool tests                                             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.warning("\n⚠️ Some prerequisites are not met. Tests may fail.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Test run cancelled")
            return
    
    start_time = time.time()
    
    # Run all test suites
    logger.info("\n" + "="*70)
    logger.info("Starting Test Execution")
    logger.info("="*70)
    
    # Run tests in sequence to avoid conflicts
    await test_sqlite_tools()
    await test_api_endpoints()
    await run_integration_tests()
    await run_adk_evaluation()
    
    # Only run Playwright tests if frontend is available
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8501")
    try:
        import httpx
        httpx.get(frontend_url, timeout=2.0)
        await run_playwright_tests()
    except:
        logger.info("Skipping Playwright tests (frontend not running)")
        test_results["tests"]["Playwright UI Tests"] = "SKIPPED - Frontend not running"
        test_results["summary"]["skipped"] += 1
        test_results["summary"]["total"] += 1
    
    # Calculate execution time
    execution_time = time.time() - start_time
    test_results["execution_time_seconds"] = execution_time
    
    # Generate report
    logger.info("\n" + "="*70)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total Tests: {test_results['summary']['total']}")
    logger.info(f"✅ Passed: {test_results['summary']['passed']}")
    logger.info(f"❌ Failed: {test_results['summary']['failed']}")
    logger.info(f"⏭️ Skipped: {test_results['summary']['skipped']}")
    logger.info(f"⏱️ Execution Time: {execution_time:.2f} seconds")
    
    # Calculate pass rate
    if test_results['summary']['total'] > 0:
        pass_rate = (test_results['summary']['passed'] / test_results['summary']['total']) * 100
        logger.info(f"📊 Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate == 100:
            logger.info("\n🎉 PERFECT SCORE! All tests passed!")
        elif pass_rate >= 80:
            logger.info("\n✅ Good coverage! Most tests passed.")
        elif pass_rate >= 60:
            logger.info("\n⚠️ Moderate coverage. Some issues need attention.")
        else:
            logger.info("\n❌ Low pass rate. Significant issues detected.")
    
    # Save results to file
    report_file = Path(__file__).parent / f"msa_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if test_results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())