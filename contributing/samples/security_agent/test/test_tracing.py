#!/usr/bin/env python3
"""Test tracing endpoints."""

import requests

BACKEND_URL = "http://localhost:8000"

def test_tracing():
    """Test tracing endpoints."""
    endpoints = [
        "/api/v1/tracing/statistics",
        "/api/v1/tracing/traces/recent", 
        "/api/v1/tracing/errors/recent",
        "/api/v1/tracing/chat-performance"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=3)
            if response.status_code == 200:
                print(f"✅ {endpoint}")
            else:
                print(f"❌ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: ERROR")

def main():
    test_tracing()