#!/usr/bin/env python3
"""Test the recommendations API."""

import requests
import json

BACKEND_URL = "http://localhost:8000"

def test_recommendations():
    """Test the recommendations endpoint."""
    data = {
        "project_id": "mgm-digitalconcierge",
        "user_email": "admin@stuartgano.altostrat.com",
        "priority": "high"
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/recommendations/dashboard",
            json=data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Recommendations API working!")
            print(f"📊 Total recommendations: {result['data']['total_recommendations']}")
            print(f"🔴 High priority: {result['data']['high_priority']}")
            print(f"🟡 Medium priority: {result['data']['medium_priority']}")
            print(f"🟢 Low priority: {result['data']['low_priority']}")
            
            # Show first recommendation
            if result['data']['recommendations']:
                first_rec = result['data']['recommendations'][0]
                print(f"\n🎯 Top recommendation: {first_rec['title']}")
                print(f"📝 Description: {first_rec['description'][:100]}...")
        else:
            print(f"❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    test_recommendations()