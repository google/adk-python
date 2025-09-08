#!/usr/bin/env python3
"""
Test Custom Roles Analyzer
===========================

Comprehensive test script for the Custom Roles Analyzer functionality.
Tests backend API, SQLite tool integration, and basic UI functionality.
"""

import asyncio
import httpx
import json
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Add agents to path  
agents_dir = Path(__file__).parent / "agents" / "gcp_security"
sys.path.insert(0, str(agents_dir))

# Test configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project-123")


async def test_api_endpoints():
    """Test Custom Roles API endpoints."""
    print("\n" + "="*70)
    print("TESTING CUSTOM ROLES API ENDPOINTS")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        # Test 1: List custom roles
        print("\n1. Testing list custom roles...")
        try:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/custom-roles/roles",
                params={"project_id": PROJECT_ID},
                timeout=30.0
            )
            
            if response.status_code == 200:
                roles = response.json()
                print(f"✅ Successfully fetched {len(roles)} custom roles")
                
                # Store first role for further tests
                if roles:
                    test_role = roles[0]
                    print(f"   First role: {test_role['name']}")
                    
                    # Test 2: Analyze single role
                    print("\n2. Testing role analysis...")
                    analysis_response = await client.post(
                        f"{BACKEND_URL}/api/v1/custom-roles/analyze",
                        json=test_role,
                        timeout=30.0
                    )
                    
                    if analysis_response.status_code == 200:
                        analysis = analysis_response.json()
                        print(f"✅ Analysis successful:")
                        print(f"   - Risk Score: {analysis['risk_score']:.0f}/100")
                        print(f"   - Total Permissions: {analysis['total_permissions']}")
                        print(f"   - Matches Found: {len(analysis.get('matches', []))}")
                        
                        # Show best match
                        if analysis.get('matches'):
                            best_match = analysis['matches'][0]
                            print(f"   - Best Match: {best_match['role']} ({best_match['match_percentage']:.1f}%)")
                    else:
                        print(f"❌ Analysis failed: {analysis_response.status_code}")
                        
            else:
                print(f"❌ Failed to fetch roles: {response.status_code}")
                
        except Exception as e:
            print(f"❌ API test failed: {e}")
        
        # Test 3: Get statistics
        print("\n3. Testing statistics endpoint...")
        try:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/custom-roles/stats",
                params={"project_id": PROJECT_ID},
                timeout=30.0
            )
            
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Statistics retrieved:")
                print(f"   - Total Roles: {stats.get('total_roles', 0)}")
                print(f"   - Active Roles: {stats.get('active_roles', 0)}")
                print(f"   - Replaceable: {stats.get('optimization_potential', 0):.1f}%")
                
                # Show risk distribution
                if stats.get('risk_distribution'):
                    print(f"   - Risk Distribution:")
                    for level, count in stats['risk_distribution'].items():
                        print(f"     • {level.title()}: {count}")
            else:
                print(f"❌ Statistics failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Statistics test failed: {e}")
        
        # Test 4: Get recommendations
        print("\n4. Testing recommendations endpoint...")
        try:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/custom-roles/recommendations",
                params={
                    "project_id": PROJECT_ID,
                    "severity": "high"
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                print(f"✅ Found {len(recommendations)} high-severity recommendations")
                
                # Show first recommendation
                if recommendations:
                    rec = recommendations[0]
                    print(f"   - Role: {rec.get('role_name', 'Unknown').split('/')[-1]}")
                    print(f"   - Issue: {rec.get('message', 'N/A')}")
            else:
                print(f"❌ Recommendations failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Recommendations test failed: {e}")


def test_sqlite_tool():
    """Test SQLite tool integration for custom roles."""
    print("\n" + "="*70)
    print("TESTING SQLITE TOOL INTEGRATION")
    print("="*70)
    
    try:
        from sqlite_tool import query_security_data
        
        # Test 1: Query custom roles
        print("\n1. Testing custom_roles query...")
        result = query_security_data("custom_roles", None)
        
        if "❌" not in result:
            print("✅ Custom roles query successful")
            print(f"   Result preview: {result[:200]}...")
        else:
            print(f"⚠️ Custom roles query returned: {result[:100]}")
        
        # Test 2: Query specific role
        print("\n2. Testing custom_roles query with parameters...")
        params = json.dumps({"role_name": "customDeveloper"})
        result = query_security_data("custom_roles", params)
        
        if "❌" not in result:
            print("✅ Specific role query successful")
            if "customDeveloper" in result:
                print("   Found role: customDeveloper")
        else:
            print(f"⚠️ Specific role query returned: {result[:100]}")
        
        # Test 3: Query custom roles analysis
        print("\n3. Testing custom_roles_analysis query...")
        result = query_security_data("custom_roles_analysis", None)
        
        if "❌" not in result:
            print("✅ Custom roles analysis query successful")
            if "Risk" in result:
                print("   Analysis includes risk information")
        else:
            print(f"⚠️ Analysis query returned: {result[:100]}")
        
        # Test 4: Query role mappings
        print("\n4. Testing role_mappings query...")
        result = query_security_data("role_mappings", None)
        
        if "❌" not in result:
            print("✅ Role mappings query successful")
            if "Standard Role" in result or "match" in result:
                print("   Mappings include standard role suggestions")
        else:
            print(f"⚠️ Mappings query returned: {result[:100]}")
            
    except ImportError as e:
        print(f"❌ Could not import sqlite_tool: {e}")
    except Exception as e:
        print(f"❌ SQLite tool test failed: {e}")


def test_sample_analysis():
    """Test analysis with sample custom role data."""
    print("\n" + "="*70)
    print("TESTING SAMPLE CUSTOM ROLE ANALYSIS")
    print("="*70)
    
    # Sample custom role with mixed permissions
    sample_role = {
        "name": f"projects/{PROJECT_ID}/roles/testDeveloper",
        "title": "Test Developer Role",
        "description": "Developer role for testing",
        "stage": "BETA",
        "permissions": [
            # Compute permissions (medium risk)
            "compute.instances.create",
            "compute.instances.get",
            "compute.instances.list",
            "compute.instances.setMetadata",  # High risk
            
            # Storage permissions (mixed risk)
            "storage.buckets.create",
            "storage.buckets.get",
            "storage.buckets.list",
            "storage.buckets.delete",  # High risk
            "storage.objects.get",
            "storage.objects.list",
            
            # IAM permissions (high risk)
            "iam.serviceAccounts.actAs",  # High risk
            "iam.serviceAccounts.get",
            "iam.serviceAccounts.list"
        ],
        "deleted": False
    }
    
    print("\nSample Role Analysis:")
    print(f"  Name: {sample_role['title']}")
    print(f"  Total Permissions: {len(sample_role['permissions'])}")
    
    # Count risk levels
    high_risk = sum(1 for p in sample_role['permissions'] 
                   if any(r in p for r in ['delete', 'setMetadata', 'actAs', 'setIamPolicy']))
    medium_risk = sum(1 for p in sample_role['permissions']
                     if any(r in p for r in ['create', 'update', 'write']))
    low_risk = len(sample_role['permissions']) - high_risk - medium_risk
    
    print(f"  Risk Breakdown:")
    print(f"    • High Risk: {high_risk}")
    print(f"    • Medium Risk: {medium_risk}")
    print(f"    • Low Risk: {low_risk}")
    
    # Estimate risk score
    risk_score = (high_risk * 10 + medium_risk * 5) / len(sample_role['permissions']) * 100
    print(f"  Estimated Risk Score: {risk_score:.0f}/100")
    
    # Suggest similar standard roles
    print("\n  Potential Standard Role Matches:")
    
    # Check for developer-like permissions
    dev_perms = ['create', 'get', 'list']
    if all(any(dp in p for p in sample_role['permissions']) for dp in dev_perms):
        print("    • roles/editor - Partial match (missing some permissions)")
    
    # Check for viewer-like permissions
    if all('get' in p or 'list' in p for p in sample_role['permissions'] 
           if not any(r in p for r in ['create', 'delete', 'update'])):
        print("    • roles/viewer - Subset match")
    
    # Check for specific service roles
    compute_perms = [p for p in sample_role['permissions'] if 'compute' in p]
    if compute_perms:
        print(f"    • roles/compute.instanceAdmin - Partial match ({len(compute_perms)} compute permissions)")
    
    storage_perms = [p for p in sample_role['permissions'] if 'storage' in p]
    if storage_perms:
        print(f"    • roles/storage.admin - Partial match ({len(storage_perms)} storage permissions)")
    
    print("\n  Recommendations:")
    if high_risk > 3:
        print("    🔴 HIGH: Review and remove unnecessary high-risk permissions")
    if 'iam.serviceAccounts.actAs' in sample_role['permissions']:
        print("    🔴 CRITICAL: Remove 'actAs' permission unless absolutely necessary")
    if len(sample_role['permissions']) > 20:
        print("    🟡 MEDIUM: Consider splitting into multiple focused roles")
    print("    💡 TIP: Use principle of least privilege")


async def main():
    """Run all tests."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║               CUSTOM ROLES ANALYZER TEST SUITE                   ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  This will test:                                                 ║
    ║  • API endpoints for custom roles analysis                       ║
    ║  • SQLite tool integration                                       ║
    ║  • Sample role analysis logic                                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Project ID: {PROJECT_ID}")
    
    # Check if backend is running
    print("\nChecking backend connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ Backend is running")
            else:
                print(f"⚠️ Backend returned status {response.status_code}")
    except:
        print("⚠️ Backend is not accessible. Some tests may fail.")
        print("   Start backend with: python run_backend.py")
    
    # Run tests
    await test_api_endpoints()
    test_sqlite_tool()
    test_sample_analysis()
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70)
    
    print("""
    Next Steps:
    1. Run the frontend to see the UI: python run_frontend.py
    2. Test with real GCP project: Set GOOGLE_CLOUD_PROJECT environment variable
    3. Use ADK agent to query custom roles: "Show me custom roles analysis"
    """)


if __name__ == "__main__":
    asyncio.run(main())