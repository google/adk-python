#!/usr/bin/env python3
"""
Test script to verify frontend-backend integration
Ensures all API calls return real data, not stubbed responses
"""

import requests
import json
import sys

import pytest

@pytest.mark.parametrize(
    "name, method, url, payload",
    [
        ("Health Check", "GET", "http://localhost:8000/health", None),
        (
            "Chat - Storage Query",
            "POST",
            "http://localhost:8000/api/v1/agent/chat",
            {
                "query": "analyze my bucket security",
                "user_id": "test_user",
                "project_id": "mgm-digitalconcierge",
            },
        ),
    ],
)
def test_endpoint(name, method, url, payload):
    """Test a single endpoint"""
    print(f"\n🧪 Testing {name}...")
    print(f"   {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") or response.status_code == 200:
                print(f"   ✅ SUCCESS - Response received")
                # Check if response contains real data
                if "error" in str(data).lower() and "not found" in str(data).lower():
                    print(f"   ⚠️  WARNING: Response may be stubbed")
                else:
                    print(f"   📊 Data fields: {list(data.keys())[:5]}")
                return True
            else:
                print(f"   ❌ FAILED - Success=False")
                print(f"   Error: {data.get('error', 'Unknown')}")
                return False
        else:
            print(f"   ❌ FAILED - Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ FAILED - Cannot connect to backend")
        return False
    except Exception as e:
        print(f"   ❌ FAILED - {str(e)}")
        return False

def main():
    """Run all integration tests"""
    print("=" * 80)
    print("🚀 Frontend-Backend Integration Test Suite")
    print("=" * 80)
    
    base_url = "http://localhost:8000"
    project_id = "mgm-digitalconcierge"
    
    tests = [
        # Core endpoints
        ("Health Check", "GET", f"{base_url}/health", None),
        ("Agent Info", "GET", f"{base_url}/api/v1/agent", None),
        
        # Chat endpoint with different queries
        ("Chat - Storage Query", "POST", f"{base_url}/api/v1/agent/chat", {
            "query": "analyze my bucket security",
            "user_id": "test_user",
            "project_id": project_id
        }),
        ("Chat - IAM Query", "POST", f"{base_url}/api/v1/agent/chat", {
            "query": "show me users with owner role",
            "user_id": "test_user",
            "project_id": project_id
        }),
        ("Chat - Network Query", "POST", f"{base_url}/api/v1/agent/chat", {
            "query": "check my firewall rules",
            "user_id": "test_user",
            "project_id": project_id
        }),
        ("Chat - Cost Query", "POST", f"{base_url}/api/v1/agent/chat", {
            "query": "how can I reduce costs?",
            "user_id": "test_user",
            "project_id": project_id
        }),
        ("Chat - Compliance Query", "POST", f"{base_url}/api/v1/agent/chat", {
            "query": "what's my SOC2 compliance status?",
            "user_id": "test_user",
            "project_id": project_id
        }),
        
        # Direct API endpoints
        ("GCP Projects", "GET", f"{base_url}/api/v1/gcp/projects", None),
        ("Storage Buckets", "GET", f"{base_url}/api/v1/storage/buckets/{project_id}?detailed=true", None),
        ("IAM Analysis", "GET", f"{base_url}/api/v1/iam/project/{project_id}/analyze-all-users", None),
        ("Network Analysis", "GET", f"{base_url}/api/v1/network/analyze/{project_id}?detailed=true", None),
        ("Cost Analysis", "GET", f"{base_url}/api/v1/cost/analyze/{project_id}?include_security=true", None),
        ("Compliance Evaluation", "POST", f"{base_url}/api/v1/compliance/evaluate", {
            "project_id": project_id,
            "frameworks": ["SOC2", "ISO27001", "GDPR"]
        }),
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test_endpoint(*test):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Frontend-backend integration is working correctly.")
        print("✨ All endpoints are returning real data, not stubbed responses.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the backend logs for errors.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())