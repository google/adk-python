#!/usr/bin/env python3
"""
Test script for the updated ADK frontend service
"""

import sys
import os
from pathlib import Path

# Add the frontend directory to path
frontend_dir = Path(__file__).parent / "frontend"
sys.path.insert(0, str(frontend_dir))

# Import the updated service
from services.adk_service import send_message, check_backend_health, check_database_health

def test_adk_frontend():
    """Test the updated frontend service with ADK endpoints."""

    print("🧪 Testing Updated ADK Frontend Service")
    print("=" * 50)

    # Test 1: Backend health check
    print("\n1️⃣ Testing backend health check:")
    health = check_backend_health()
    print(f"   Status: {'✅ Success' if health['success'] else '❌ Failed'}")
    print(f"   Details: {health.get('status', 'N/A')}")

    # Test 2: Database health check
    print("\n2️⃣ Testing database health check:")
    db_health = check_database_health()
    print(f"   Status: {'✅ Success' if db_health['success'] else '❌ Failed'}")
    print(f"   Details: {db_health.get('status', 'N/A')}")

    # Test 3: Send a message to the agent
    print("\n3️⃣ Testing agent message (bucket query):")
    try:
        response = send_message("show me storage buckets")
        if response["success"]:
            print(f"   ✅ Success!")
            print(f"   📝 Response length: {len(response['response'])} characters")
            print(f"   ⏱️ Duration: {response['request_duration']:.2f}s")
            print(f"   🆔 Session ID: {response['metadata'].get('session_id', 'N/A')}")
            print(f"   📄 Response preview: {response['response'][:200]}...")
        else:
            print(f"   ❌ Failed: {response['error']}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

    print("\n🎉 ADK Frontend Service Test Complete!")

if __name__ == "__main__":
    test_adk_frontend()