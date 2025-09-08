#!/usr/bin/env python3
"""
Integration Test Runner for UI Components

Runs comprehensive integration tests and generates detailed reports.
"""

import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path

def run_test_suite(test_file):
    """Run a specific test suite and capture results."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, 
            '-v', 
            '--tb=short', 
            '--json-report',
            '--json-report-file=test_results.json'
        ], capture_output=True, text=True, timeout=120)
        
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'exit_code': -1,
            'stdout': '',
            'stderr': 'Test timed out after 120 seconds',
            'success': False
        }
    except Exception as e:
        return {
            'exit_code': -2,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }

def main():
    """Run all integration tests and generate report."""
    test_dir = Path(__file__).parent
    
    test_suites = [
        'test_backend_integration.py',
        'test_data_flow.py', 
        'test_api_calls.py',
        'test_session_management.py',
        'test_cache_integration.py'
    ]
    
    results = {}
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    print("🚀 Starting UI Integration Test Suite")
    print("="*50)
    
    for test_suite in test_suites:
        test_path = test_dir / test_suite
        if not test_path.exists():
            print(f"❌ Test suite not found: {test_suite}")
            continue
            
        print(f"\n📋 Running: {test_suite}")
        print("-" * 30)
        
        result = run_test_suite(str(test_path))
        results[test_suite] = result
        
        if result['success']:
            print(f"✅ {test_suite} - PASSED")
        else:
            print(f"❌ {test_suite} - FAILED")
            if result['stderr']:
                print(f"Error: {result['stderr'][:200]}...")
                
        # Parse test counts from output
        stdout = result['stdout']
        if 'failed' in stdout:
            # Extract test counts from pytest output
            lines = stdout.split('\n')
            for line in lines:
                if 'failed' in line and 'passed' in line:
                    # Parse line like "3 failed, 1 passed, 1 skipped"
                    parts = line.split(',')
                    for part in parts:
                        part = part.strip()
                        if 'failed' in part:
                            failed_tests += int(part.split()[0])
                        elif 'passed' in part:
                            passed_tests += int(part.split()[0])
                        elif 'skipped' in part:
                            skipped_tests += int(part.split()[0])
                    break
        elif 'passed' in stdout:
            # All tests passed
            lines = stdout.split('\n')
            for line in lines:
                if 'passed' in line and '==' in line:
                    try:
                        passed_tests += int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                    break
    
    total_tests = passed_tests + failed_tests + skipped_tests
    
    # Generate summary report
    print("\n" + "="*50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")  
    print(f"⏭️ Skipped: {skipped_tests}")
    
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    # Generate detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': success_rate if total_tests > 0 else 0
        },
        'test_suites': results
    }
    
    # Save results to file
    results_file = test_dir / 'integration_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Determine overall status
    if failed_tests == 0:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {failed_tests} TESTS FAILED - Review results for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())