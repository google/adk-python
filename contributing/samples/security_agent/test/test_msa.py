#!/usr/bin/env python3
"""Test MSA endpoints."""

import requests

BACKEND_URL = "http://localhost:8000"

def test_msa_endpoints():
    """Test all MSA endpoints."""
    endpoints = [
        ("GET", "/api/v1/msa/sample-msa", {"project_id": "mgm-digitalconcierge"}),
        ("POST", "/api/v1/msa/parse", {"content": "Test MSA", "name": "Test", "msa_type": "agreement"}),
        ("POST", "/api/v1/msa/scan-gcp", {"project_id": "mgm-digitalconcierge"}),
        ("GET", "/api/v1/msa/records", {}),
        ("GET", "/api/v1/msa/impact-analyses", {}),
        ("GET", "/api/v1/msa/api-patterns", {}),
        ("GET", "/api/v1/msa/msa-patterns", {}),
    ]
    
    print("🔍 Testing MSA endpoints:")
    for method, endpoint, data in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint}", params=data, timeout=3)
            else:
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=3)
            
            if response.status_code == 200:
                print(f"✅ {method} {endpoint}")
            else:
                print(f"❌ {method} {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {method} {endpoint}: ERROR")

def main():
    test_msa_endpoints()