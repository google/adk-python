#!/usr/bin/env python3
"""Comprehensive test script to verify ALL security agent endpoints are working."""

import requests
import json

BACKEND_URL = "http://localhost:8000"

def check_endpoint(endpoint, method="GET", data=None, timeout=10):
    """Test an endpoint and return result."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=timeout)
        elif method == "DELETE":
            response = requests.delete(url, timeout=timeout)
        
        print(f"{method} {endpoint}: {response.status_code}")
        if response.status_code in [200, 201]:
            print("✅ SUCCESS")
            if response.text and len(response.text) < 500:
                try:
                    result = response.json()
                    print(f"Response: {json.dumps(result, indent=2)[:200]}...")
                except:
                    print(f"Response: {response.text[:200]}...")
        else:
            print(f"❌ FAILED: {response.text[:200]}")
        print()
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"{method} {endpoint}: ERROR - {str(e)}")
        print("❌ FAILED")
        print()
        return False

if __name__ == "__main__":
    test_all_endpoints()
    """Test ALL endpoints."""
    print("🔍 COMPREHENSIVE Security Agent Endpoint Testing")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Core endpoints
    endpoints_to_test = [
        # Health and info
        ("/", "GET"),
        ("/health", "GET"),
        
        # Security endpoints
        ("/api/v1/security/evaluate", "POST", {
            "project_id": "mgm-digitalconcierge", 
            "user_email": "admin@stuartgano.altostrat.com"
        }),
        ("/api/v1/security/evaluate-vulnerability", "POST", {
            "text": "SQL injection vulnerability in user input"
        }),
        
        # Knowledge base endpoints
        ("/api/v1/knowledge/upload", "POST", {
            "content": "Test security knowledge",
            "source": "test"
        }),
        
        # Agent endpoints
        ("/api/v1/agent/chat", "POST", {
            "message": "What are the main security risks?",
            "project_id": "mgm-digitalconcierge",
            "context": "security_analysis"
        }),
        
        # Documentation endpoints
        ("/api/v1/documentation/scrape", "POST", {
            "url": "https://cloud.google.com/security"
        }),
        
        # API Hub endpoints
        ("/api/v1/apihub/toolsets", "GET"),
        
        # Compliance endpoints
        ("/api/v1/compliance/compliance", "GET"),
        ("/api/v1/compliance/evaluate", "POST", {
            "project_id": "mgm-digitalconcierge", 
            "framework": "SOC2"
        }),
        
        # Threat Intelligence endpoints
        ("/api/v1/threat-intelligence/threat_intelligence", "GET"),
        ("/api/v1/threat-intelligence/landscape", "POST", {
            "project_id": "mgm-digitalconcierge", 
            "scope": "global"
        }),
        
        # Configuration endpoints
        ("/api/v1/configuration/configuration", "GET"),
        ("/api/v1/configuration/analyze", "POST", {
            "project_id": "mgm-digitalconcierge", 
            "resource_type": "all"
        }),
        
        # Incident Response endpoints
        ("/api/v1/incidents/incidents", "GET"),
        
        # Evaluation endpoints
        ("/api/v1/evaluation/evaluation", "GET"),
        
        # MSA endpoints
        ("/api/v1/msa/msa", "GET"),
        
        # Tracing endpoints
        ("/api/v1/tracing/tracing", "GET"),
        
        # OpenAPI tools endpoints
        ("/api/v1/openapi-tools/openapi-tools", "GET"),
        
        # GCP endpoints
        ("/api/v1/gcp/projects", "GET"),
        ("/api/v1/gcp/project/mgm-digitalconcierge/info", "GET"),
        ("/api/v1/gcp/project/mgm-digitalconcierge/services", "GET"),
    ]
    
    for test_data in endpoints_to_test:
        total += 1
        endpoint = test_data[0]
        method = test_data[1]
        data = test_data[2] if len(test_data) > 2 else None
        
        if check_endpoint(endpoint, method, data):
            passed += 1
    
    print("=" * 60)
    print(f"🏁 Testing Complete!")
    print(f"✅ Passed: {passed}/{total} endpoints")
    print(f"❌ Failed: {total - passed}/{total} endpoints")
    
    if passed == total:
        print("🎉 ALL ENDPOINTS WORKING!")
    else:
        print("⚠️  Some endpoints need attention")

if __name__ == "__main__":
    test_all_endpoints()