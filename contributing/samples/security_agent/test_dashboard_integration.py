#!/usr/bin/env python3
"""
Dashboard Integration Test

This test validates the security dashboard integration with real GCP Asset Inventory API.
It tests all components and API endpoints to ensure proper functionality.
"""

import sys
import os
import requests
import json
from datetime import datetime

# Add frontend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

def test_backend_connectivity():
    """Test basic backend connectivity."""
    backend_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend connectivity: PASS")
            return True
        else:
            print(f"❌ Backend connectivity: FAIL (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Backend connectivity: FAIL ({e})")
        return False

def test_asset_inventory_api():
    """Test Asset Inventory API endpoints."""
    backend_url = "http://localhost:8000"
    base_url = f"{backend_url}/api/v1/asset-inventory"
    
    tests = [
        ("/health", "Asset Inventory health check"),
        ("/summary", "Asset inventory summary"),
    ]
    
    results = []
    
    for endpoint, description in tests:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {description}: PASS")
                print(f"   Response: {type(data)} with {len(str(data))} chars")
                results.append(True)
            else:
                print(f"❌ {description}: FAIL (HTTP {response.status_code})")
                results.append(False)
        except Exception as e:
            print(f"❌ {description}: FAIL ({e})")
            results.append(False)
    
    return all(results)

def test_dashboard_components():
    """Test dashboard component imports."""
    try:
        from components.dashboard.dashboard_view import render_dashboard_view
        from components.dashboard.asset_charts import (
            render_asset_breakdown_chart,
            render_security_analysis_chart,
            render_recommendations_chart,
            render_risk_assessment_chart
        )
        from components.dashboard.security_posture_widget import render_security_posture_widget
        from api.asset_inventory_client import AssetInventoryClient, get_asset_inventory_client
        
        print("✅ Dashboard component imports: PASS")
        return True
    except Exception as e:
        print(f"❌ Dashboard component imports: FAIL ({e})")
        return False

def test_asset_inventory_client():
    """Test Asset Inventory client functionality."""
    try:
        from api.asset_inventory_client import AssetInventoryClient
        
        client = AssetInventoryClient()
        
        # Test health check
        health_result = client.get_health_status()
        if health_result.get("status") in ["healthy", "degraded"]:
            print("✅ Asset Inventory client: PASS")
            print(f"   Status: {health_result.get('status')}")
            return True
        else:
            print(f"❌ Asset Inventory client: FAIL (Status: {health_result.get('status')})")
            return False
    except Exception as e:
        print(f"❌ Asset Inventory client: FAIL ({e})")
        return False

def test_mock_data_functionality():
    """Test dashboard with mock data to ensure rendering works."""
    try:
        from api.asset_inventory_client import AssetInventoryClient
        
        client = AssetInventoryClient()
        
        # Test summary endpoint (should work even with mock data)
        summary = client.get_asset_summary()
        
        print("✅ Mock data functionality: PASS")
        print(f"   Summary response: {type(summary)}")
        
        if summary.get("data"):
            data = summary["data"]
            print(f"   Assets found: {data.get('total_assets', 0)}")
            print(f"   Asset types: {len(data.get('asset_types', {}))}")
            print(f"   Security findings: {data.get('security_findings', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Mock data functionality: FAIL ({e})")
        return False

def test_full_integration():
    """Test full dashboard integration."""
    print("=== Dashboard Integration Test ===")
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    tests = [
        ("Backend Connectivity", test_backend_connectivity),
        ("Asset Inventory API", test_asset_inventory_api), 
        ("Dashboard Components", test_dashboard_components),
        ("Asset Inventory Client", test_asset_inventory_client),
        ("Mock Data Functionality", test_mock_data_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Test Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests PASSED! Dashboard is ready for use.")
        print("\n📋 Next Steps:")
        print("1. Start the backend: python backend/run_backend.py")
        print("2. Start the frontend: python frontend/run_frontend.py")
        print("3. Navigate to the Dashboard in the web interface")
        print("4. Select a GCP project to view real asset inventory data")
        return True
    else:
        print(f"⚠️  {total - passed} tests FAILED. Please check the issues above.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure backend is running on localhost:8000")
        print("2. Check GCP credentials and Asset Inventory API access")
        print("3. Verify all Python dependencies are installed")
        return False

if __name__ == "__main__":
    success = test_full_integration()
    sys.exit(0 if success else 1)