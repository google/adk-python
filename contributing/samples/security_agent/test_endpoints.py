#!/usr/bin/env python3
"""Test script to verify API endpoints are working correctly."""

import requests
import json

def test_endpoints():
    """Test the corrected API endpoints."""
    
    base_url = "http://localhost:8000"
    project_id = "mgm-digitalconcierge"
    
    print("=" * 60)
    print("TESTING API ENDPOINTS")
    print("=" * 60)
    
    # Test 1: Asset snapshot endpoint
    print("\n1. Testing Asset Snapshot Endpoint...")
    try:
        response = requests.get(
            f"{base_url}/api/v1/assets/snapshot/{project_id}",
            params={"force_refresh": False},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: Got snapshot data")
            if data.get("data"):
                summary = data["data"].get("summary", {})
                print(f"   Total assets: {summary.get('total_assets', 0)}")
                print(f"   Data source: {data['data'].get('api_metadata', {}).get('source', 'unknown')}")
        else:
            print(f"   ❌ ERROR: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 2: Cache status endpoint
    print("\n2. Testing Cache Status Endpoint...")
    try:
        response = requests.get(
            f"{base_url}/api/v1/assets/cache-status/{project_id}",
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: Cache status retrieved")
            if data.get("data"):
                cache_enabled = data["data"].get("cache_enabled", False)
                print(f"   Cache enabled: {cache_enabled}")
        else:
            print(f"   ❌ ERROR: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 3: Session creation
    print("\n3. Testing Session Creation...")
    try:
        response = requests.post(
            f"{base_url}/api/v1/sessions/create",
            json={"user_id": "test_user"},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            print(f"   ✅ SUCCESS: Session created")
            print(f"   Session ID: {session_id}")
            
            # Test session status
            if session_id:
                print("\n4. Testing Session Status...")
                response = requests.get(
                    f"{base_url}/api/v1/agent/sessions/{session_id}/status",
                    timeout=5
                )
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ SUCCESS: Session status retrieved")
                    print(f"   Active: {data.get('active', False)}")
                else:
                    print(f"   ❌ ERROR: {response.text[:200]}")
        else:
            print(f"   ❌ ERROR: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 5: Asset summary endpoint
    print("\n5. Testing Asset Summary Endpoint...")
    try:
        response = requests.get(
            f"{base_url}/api/v1/assets/summary",
            params={"project_id": project_id},
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: Summary retrieved")
        else:
            print(f"   ❌ ERROR: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    print("\n" + "=" * 60)
    print("ENDPOINT TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    print("Testing API endpoints (requires backend to be running)...")
    test_endpoints()