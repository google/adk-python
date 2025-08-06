#!/usr/bin/env python3
"""Quick endpoint test with short timeouts."""

import requests

BACKEND_URL = "http://localhost:8000"

def quick_test(endpoint, method="GET", data=None):
    try:
        if method == "GET":
            r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=3)
        else:
            r = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=3)
        print(f"{method} {endpoint}: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"{method} {endpoint}: TIMEOUT/ERROR")
        return False

# Test key endpoints
endpoints = [
    ("/health", "GET"),
    ("/api/v1/security/evaluate", "POST", {"project_id": "mgm-digitalconcierge", "user_email": "admin@stuartgano.altostrat.com"}),
    ("/api/v1/compliance/evaluate", "POST", {"project_id": "mgm-digitalconcierge", "framework": "SOC2"}),
    ("/api/v1/threat-intelligence/landscape", "POST", {"project_id": "mgm-digitalconcierge", "scope": "global"}),
    ("/api/v1/configuration/analyze", "POST", {"project_id": "mgm-digitalconcierge", "resource_type": "all"}),
]

print("Quick endpoint test:")
for ep in endpoints:
    quick_test(ep[0], ep[1], ep[2] if len(ep) > 2 else None)