#!/usr/bin/env python3
"""Test script to verify Resource Manager API authentication fixes."""

import requests
import json
import sys
import time

def check_api_endpoint(endpoint, description):
    """Test a specific API endpoint."""
    try:
        response = requests.get(f"http://localhost:8000{endpoint}", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {description}: SUCCESS")
            
            # Print relevant details based on endpoint
            if "/projects" in endpoint:
                projects = data.get("projects", [])
                print(f"   📋 Found {len(projects)} projects")
                if data.get("default_project"):
                    print(f"   🏗️  Default project: {data['default_project']}")
                    
            elif "/info" in endpoint:
                project = data.get("project", {})
                print(f"   📊 Project: {project.get('display_name', 'N/A')}")
                print(f"   🏷️  State: {project.get('state', 'N/A')}")
                
            elif "/services" in endpoint:
                total = data.get("total_services", 0)
                risk_summary = data.get("risk_summary", {})
                print(f"   🔧 Total services: {total}")
                print(f"   ⚠️  High/Critical risk: {risk_summary.get('high', 0) + risk_summary.get('critical', 0)}")
                
            assert True
        else:
            print(f"❌ {description}: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Error: {response.text[:200]}")
            assert False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {description}: Connection failed - {e}")
        assert False
    except Exception as e:
        print(f"❌ {description}: Unexpected error - {e}")
        assert False

def test_resource_manager_fix():
    """Run all Resource Manager API tests."""
    print("🔧 Testing Resource Manager API Authentication Fixes")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend server not running on localhost:8000")
            assert False
    except:
        print("❌ Backend server not accessible on localhost:8000") 
        print("💡 Please start the backend server first:")
        print("   cd backend && python main.py")
        assert False
        
    print("✅ Backend server is running")
    print()
    
    # Test endpoints that use Resource Manager API
    tests = [
        ("/api/v1/gcp/projects", "List projects (Resource Manager v1 API)"),
        ("/api/v1/gcp/project/mgm-digitalconcierge/info", "Get project info (Resource Manager v3 API)"),
        ("/api/v1/gcp/project/mgm-digitalconcierge/services", "Get project services (Service Usage API)")
    ]
    
    passed = 0
    total = len(tests)
    
    for endpoint, description in tests:
        print(f"🔍 Testing: {description}")
        if check_api_endpoint(endpoint, description):
            passed += 1
        print()
        
    # Summary
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Resource Manager API authentication tests passed!")
        print("✅ The authentication fixes are working correctly.")
        assert True
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        print("💡 Check the error messages above for details.")
        assert False

if __name__ == "__main__":
    test_resource_manager_fix()