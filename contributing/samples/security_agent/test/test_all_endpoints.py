#!/usr/bin/env python3
"""Test all endpoints to check for 404s."""

import requests

BACKEND_URL = "http://localhost:8000"

def test_all_endpoints():
    """Test all known endpoints."""
    endpoints = [
        # Core endpoints
        ("GET", "/health"),
        ("GET", "/"),
        
        # Security endpoints
        ("POST", "/api/v1/security/evaluate", {"project_id": "test", "user_email": "test@example.com"}),
        ("POST", "/api/v1/security/evaluate-vulnerability", {"text": "test vulnerability"}),
        
        # Compliance endpoints
        ("POST", "/api/v1/compliance/evaluate", {"project_id": "test", "framework": "SOC2"}),
        
        # Threat intelligence endpoints
        ("POST", "/api/v1/threat-intelligence/landscape", {"project_id": "test", "scope": "global"}),
        
        # Configuration endpoints
        ("POST", "/api/v1/configuration/analyze", {"project_id": "test", "resource_type": "all"}),
        
        # MSA endpoints
        ("GET", "/api/v1/msa/sample-msa", {"project_id": "test"}),
        ("POST", "/api/v1/msa/parse", {"content": "test", "name": "test", "msa_type": "agreement"}),
        ("POST", "/api/v1/msa/scan-gcp", {"project_id": "test"}),
        ("GET", "/api/v1/msa/records"),
        ("GET", "/api/v1/msa/impact-analyses"),
        ("GET", "/api/v1/msa/api-patterns"),
        ("GET", "/api/v1/msa/msa-patterns"),
        
        # Tracing endpoints
        ("GET", "/api/v1/tracing/statistics"),
        ("GET", "/api/v1/tracing/traces/recent"),
        ("GET", "/api/v1/tracing/errors/recent"),
        ("GET", "/api/v1/tracing/chat-performance"),
        
        # Recommendations endpoints
        ("POST", "/api/v1/recommendations/dashboard", {"project_id": "test", "user_email": "test@example.com"}),
        
        # GCP endpoints
        ("GET", "/api/v1/gcp/projects"),
        ("GET", "/api/v1/gcp/project/mgm-digitalconcierge/info"),
        ("GET", "/api/v1/gcp/project/mgm-digitalconcierge/services"),
    ]
    
    print("🔍 Testing all endpoints for 404 errors:")
    print("=" * 50)
    
    working = 0
    total = 0
    
    for test_data in endpoints:
        total += 1
        method = test_data[0]
        endpoint = test_data[1]
        data = test_data[2] if len(test_data) > 2 else {}
        
        try:
            if method == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint}", params=data, timeout=3)
            else:
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=3)
            
            if response.status_code == 200:
                print(f"✅ {method} {endpoint}")
                working += 1
            elif response.status_code == 404:
                print(f"❌ {method} {endpoint}: 404 NOT FOUND")
            else:
                print(f"⚠️  {method} {endpoint}: {response.status_code}")
                working += 1  # Not a 404, so endpoint exists
                
        except Exception as e:
            print(f"❌ {method} {endpoint}: ERROR ({str(e)})")
    
    print("=" * 50)
    print(f"✅ Working: {working}/{total}")
    print(f"❌ Issues: {total - working}/{total}")
    
    if working == total:
        print("🎉 All endpoints working!")
    else:
        print("⚠️  Some endpoints need attention")

if __name__ == "__main__":
    test_all_endpoints()