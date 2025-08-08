#!/usr/bin/env python3
"""Test script for GCP API Explorer integration with the security agent.

This script tests the integration of the GCP API Explorer as a modular service
within the existing security agent architecture.
"""

import sys
import os
import json

# Add backend to Python path
sys.path.append('backend')
sys.path.append('frontend')

def test_backend_integration():
    """Test backend service integration."""
    print("🧪 Testing Backend Integration...")
    
    try:
        # Test service import
        from gcp_api_explorer.service import GCPAPIExplorerService
        print("✅ Service class imported successfully")
        
        # Test service configuration
        from core.service_config import ServiceConfig
        config = ServiceConfig('backend/config/services.json')
        
        # Check if GCP API Explorer is in configuration
        service_config = config.get_service('gcp_api_explorer')
        print(f"✅ Service configuration found: {service_config.display_name}")
        
        # Verify service properties
        assert service_config.name == 'gcp_api_explorer'
        assert service_config.api_prefix == '/api/v1/gcp-api-explorer'
        assert service_config.requires_gcp_auth == True
        assert 'gcp' in [dep.service_name for dep in service_config.dependencies]
        print("✅ Service configuration validation passed")
        
        # Test service instantiation (without actual GCP credentials)
        service = GCPAPIExplorerService(
            name='gcp_api_explorer',
            config=service_config.config,
            credentials=None,
            project_id='test-project'
        )
        print("✅ Service instantiation successful")
        
        # Test router creation
        router = service.get_router()
        assert router is not None
        print("✅ FastAPI router creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frontend_integration():
    """Test frontend component integration."""
    print("\n🧪 Testing Frontend Integration...")
    
    try:
        # Add components to path
        sys.path.append('frontend/components')
        
        # Test component imports
        from gcp_api_explorer_view import (
            render_gcp_api_explorer_view,
            render_gcp_api_explorer_summary_card,
            discover_services,
            explore_service,
            test_endpoint
        )
        print("✅ Frontend component functions imported successfully")
        
        # Test that all required functions are callable
        functions = [
            render_gcp_api_explorer_view,
            render_gcp_api_explorer_summary_card,
            discover_services,
            explore_service,
            test_endpoint
        ]
        
        for func in functions:
            assert callable(func)
            print(f"✅ Function {func.__name__} is callable")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_navigation_integration():
    """Test navigation integration in main app."""
    print("\n🧪 Testing Navigation Integration...")
    
    try:
        # Add frontend to path
        sys.path.append('frontend')
        
        # Import main app functions
        from main_app import get_available_pages
        
        # Get available pages
        pages = get_available_pages()
        
        # Check if GCP API Explorer page is available
        assert 'gcp_api_explorer' in pages
        gcp_page = pages['gcp_api_explorer']
        
        assert gcp_page['name'] == '🚀 GCP API Explorer'
        assert gcp_page['service'] == 'gcp_api_explorer'
        print("✅ Navigation integration successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Navigation integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_integration():
    """Test dashboard summary card integration."""
    print("\n🧪 Testing Dashboard Integration...")
    
    try:
        sys.path.append('frontend/components')
        
        # Test dashboard view import
        from dashboard_view import render_dashboard_view
        from gcp_api_explorer_view import render_gcp_api_explorer_summary_card
        
        print("✅ Dashboard integration functions imported")
        
        # Verify functions are callable
        assert callable(render_dashboard_view)
        assert callable(render_gcp_api_explorer_summary_card)
        
        print("✅ Dashboard integration successful")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_configuration():
    """Test complete service configuration."""
    print("\n🧪 Testing Service Configuration...")
    
    try:
        with open('backend/config/services.json', 'r') as f:
            config = json.load(f)
        
        # Check GCP API Explorer service configuration
        assert 'gcp_api_explorer' in config['services']
        gcp_config = config['services']['gcp_api_explorer']
        
        # Verify all required fields
        required_fields = [
            'name', 'display_name', 'description', 'version',
            'enabled_by_default', 'dependencies', 'api_prefix',
            'service_module', 'requires_gcp_auth', 'tags'
        ]
        
        for field in required_fields:
            assert field in gcp_config, f"Missing field: {field}"
        
        # Verify runtime status
        assert 'gcp_api_explorer' in config['runtime_status']
        assert config['runtime_status']['gcp_api_explorer'] == 'running'
        
        print("✅ Service configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Service configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("🚀 GCP API Explorer Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_backend_integration,
        test_frontend_integration,
        test_navigation_integration,
        test_dashboard_integration,
        test_service_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! GCP API Explorer is successfully integrated.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())