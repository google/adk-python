#!/usr/bin/env python3
"""Test the async security functionality."""

import asyncio
import time
import requests
import json
from typing import Dict, Any

# Configuration
BACKEND_URL = "http://localhost:8000"
TEST_PROJECT_ID = "mgm-digitalconcierge"

def test_quick_analysis():
    """Test the quick analysis endpoint."""
    print("🧪 Testing quick security analysis...")
    
    payload = {
        "query": "What are the main security risks in my project?",
        "project_id": TEST_PROJECT_ID,
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/async-security/quick-analysis",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Quick analysis completed")
            print(f"   Response length: {len(result.get('response', ''))}")
            print(f"   Analysis type: {result.get('analysis_type', 'unknown')}")
            assert True
        else:
            print(f"❌ Quick analysis failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            assert False
            
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
        assert False

def test_async_security_scan():
    """Test the async security scan functionality."""
    print("\n🧪 Testing async security scan...")
    
    # Start scan
    scan_payload = {
        "project_id": TEST_PROJECT_ID,
        "scan_type": "standard",
        "user_id": "test_user",
        "include_vulnerability_scan": True,
        "include_compliance_check": True,
        "include_configuration_analysis": True,
        "include_dependency_analysis": True,
        "timeout_seconds": 300
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/async-security/scan",
            json=scan_payload,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start scan: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            assert False
        
        result = response.json()
        task_id = result.get("task_id")
        
        if not task_id:
            print("❌ No task ID returned")
            assert False
        
        print(f"✅ Scan started successfully")
        print(f"   Task ID: {task_id}")
        print(f"   Estimated duration: {result.get('estimated_duration')}")
        
        # Monitor progress
        return monitor_task_progress(task_id)
        
    except Exception as e:
        print(f"❌ Failed to start async scan: {e}")
        assert False

def monitor_task_progress(task_id: str, max_wait_time: int = 120) -> bool:
    """Monitor task progress until completion or timeout."""
    print(f"\n🔄 Monitoring task {task_id}...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(
                f"{BACKEND_URL}/api/v1/async-security/status/{task_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to get status: HTTP {response.status_code}")
                assert False
            
            status_data = response.json()
            status = status_data.get("status")
            progress = status_data.get("progress")
            
            # Print progress update
            if progress:
                percentage = progress.get("percentage", 0)
                current_step = progress.get("current_step", "Processing...")
                print(f"   Progress: {percentage:.1f}% - {current_step}")
            else:
                print(f"   Status: {status}")
            
            # Check if completed
            if status == "completed":
                print("✅ Task completed successfully!")
                result = status_data.get("result")
                if result:
                    print(f"   Scan duration: {result.get('scan_duration', 'unknown')} seconds")
                    print(f"   Results sections: {list(result.get('results', {}).keys())}")
                assert True
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                print(f"❌ Task failed: {error}")
                assert False
            elif status == "cancelled":
                print("⚠️ Task was cancelled")
                assert False
            
            # Wait before next check
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            assert False
    
    print(f"⏰ Task monitoring timed out after {max_wait_time} seconds")
    assert False

def test_task_cancellation():
    """Test task cancellation functionality."""
    print("\n🧪 Testing task cancellation...")
    
    # Start a deep scan that will take time
    scan_payload = {
        "project_id": TEST_PROJECT_ID,
        "scan_type": "deep",
        "user_id": "test_user",
        "timeout_seconds": 600
    }
    
    try:
        # Start scan
        response = requests.post(
            f"{BACKEND_URL}/api/v1/async-security/scan",
            json=scan_payload,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start scan for cancellation test")
            assert False
        
        task_id = response.json().get("task_id")
        print(f"   Started scan with task ID: {task_id}")
        
        # Wait a moment to let it start
        time.sleep(1)
        
        # Cancel the task
        cancel_response = requests.delete(
            f"{BACKEND_URL}/api/v1/async-security/cancel/{task_id}",
            timeout=10
        )
        
        if cancel_response.status_code == 200:
            print("✅ Task cancellation successful")
            assert True
        else:
            print(f"❌ Task cancellation failed: HTTP {cancel_response.status_code}")
            assert False
            
    except Exception as e:
        print(f"❌ Task cancellation test failed: {e}")
        assert False

def test_health_check():
    """Test the async security service health check."""
    print("\n🧪 Testing health check...")
    
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/v1/async-security/health",
            timeout=10
        )
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Running tasks: {health_data.get('running_tasks')}")
            print(f"   Total tasks: {health_data.get('total_tasks')}")
            assert True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            assert False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        assert False

def test_backend_connectivity():
    """Test basic backend connectivity."""
    print("🧪 Testing backend connectivity...")
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is accessible")
            assert True
        else:
            print(f"⚠️ Backend responded with status {response.status_code}")
            assert False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running (connection refused)")
        print("   Please start the backend with: python backend/main.py")
        assert False
    except Exception as e:
        print(f"❌ Backend connectivity test failed: {e}")
        assert False

def main():
    """Run all async security tests."""
    print("🚀 Starting Async Security Tests")
    print("=" * 50)
    
    tests = [
        ("Backend Connectivity", test_backend_connectivity),
        ("Health Check", test_health_check),
        ("Quick Analysis", test_quick_analysis),
        ("Async Security Scan", test_async_security_scan),
        ("Task Cancellation", test_task_cancellation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All async security tests passed!")
        print("\n💡 Usage Tips:")
        print("- Use quick analysis for simple questions")
        print("- Use async scans for comprehensive security analysis")
        print("- Monitor task progress with the status endpoint")
        print("- Cancel long-running tasks if needed")
        assert True
    else:
        print("❌ Some tests failed. Check the output above.")
        assert False

if __name__ == "__main__":
    main()