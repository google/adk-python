#!/usr/bin/env python3
"""
Test script for the Day Two Operations SRE page functionality.
"""

import requests
import json
import sys
from typing import Dict, Any

def check_endpoint(url: str, description: str) -> bool:
    """Test a single endpoint and return success status."""
    try:
        print(f"Testing {description}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"  ✅ SUCCESS: {description}")
                assert True
            except json.JSONDecodeError:
                print(f"  ❌ FAILED: {description} - Invalid JSON response")
                assert False
        else:
            print(f"  ❌ FAILED: {description} - HTTP {response.status_code}")
            assert False
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ FAILED: {description} - {str(e)}")
        assert False

def test_log_analysis_endpoints():
    """Test all log analysis endpoints."""
    print("🔍 Testing Day Two SRE Log Analysis Endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        (f"{base_url}/api/v1/logs/health", "Log Analysis Health Check"),
        (f"{base_url}/api/v1/logs/list", "Log Files Listing"),
        (f"{base_url}/api/v1/logs/tail/backend.log?lines=5", "Log Tail (5 lines)"),
        (f"{base_url}/api/v1/logs/search/backend.log?pattern=INFO&max_results=3", "Log Search"),
    ]
    
    results = []
    for url, description in endpoints:
        success = check_endpoint(url, description)
        results.append(success)
    
    # Test log analysis with POST
    print("Testing Log Analysis (POST)...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/logs/analyze",
            json={
                "log_path": "/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/logs/backend.log",
                "lines": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("  ✅ SUCCESS: Log Analysis (POST)")
                results.append(True)
            else:
                print(f"  ❌ FAILED: Log Analysis (POST) - {data.get('error', 'Unknown error')}")
                results.append(False)
        else:
            print(f"  ❌ FAILED: Log Analysis (POST) - HTTP {response.status_code}")
            results.append(False)
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ FAILED: Log Analysis (POST) - {str(e)}")
        results.append(False)
    
    return results

def test_backend_health():
    """Test basic backend health."""
    print("\n🏥 Testing Backend Health")
    print("=" * 30)
    
    return check_endpoint("http://localhost:8000/health", "Backend Health Check")

def test_sre_page():
    """Run all tests."""
    print("🚀 Day Two Operations SRE Page Testing")
    print("=" * 60)
    
    # Test backend health first
    backend_healthy = test_backend_health()
    if not backend_healthy:
        print("\n❌ Backend is not healthy. Cannot proceed with tests.")
    
    # Test log analysis endpoints
    log_results = test_log_analysis_endpoints()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    
    total_tests = len(log_results) + 1  # +1 for backend health
    passed_tests = sum(log_results) + (1 if backend_healthy else 0)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The Day Two SRE page is ready for use.")
        print("\n📋 Next Steps:")
        print("1. Open the frontend at: http://localhost:8501")
        print("2. Navigate to '📊 Day Two SRE' in the sidebar")
        print("3. Test the log analysis features")
        print("4. Try the different tabs: Log Analysis, Error Detection, Performance Metrics, Alerting")
        assert True
    else:
        print(f"\n⚠️  Some tests failed. Please check the backend logs and fix any issues.")
        assert False

def main():
    test_sre_page()