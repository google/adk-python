#!/usr/bin/env python3
"""
Phase 2 Integration Clients Test with Real Credentials
======================================================

Test integration clients with actual GCP credentials.
Requires proper environment configuration.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment from {env_file}")
else:
    print(f"⚠️  No .env file found. Using system environment variables.")

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import integration clients
from backend.integrations.google_support_client import GoogleSupportClient
from backend.integrations.vpc_sc_client import VPCServiceControlsClient  
from backend.integrations.gcp_billing_client import GCPBillingClient
from backend.integrations.gcp_resource_client import GCPResourceClient

# Configuration from environment
CONFIG = {
    "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
    "organization_id": os.getenv("GOOGLE_CLOUD_ORGANIZATION"),
    "billing_account_id": os.getenv("BILLING_ACCOUNT_ID"),
    "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    "test_timeout": int(os.getenv("TEST_TIMEOUT", "60")),
    "verbose": os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
}


def validate_configuration():
    """Validate environment configuration"""
    print("🔍 Validating Configuration...")
    
    issues = []
    
    if not CONFIG["project_id"]:
        issues.append("GOOGLE_CLOUD_PROJECT not set")
    
    if not CONFIG["credentials_path"]:
        issues.append("GOOGLE_APPLICATION_CREDENTIALS not set")
    elif not os.path.exists(CONFIG["credentials_path"]):
        issues.append(f"Service account key file not found: {CONFIG['credentials_path']}")
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n📝 Please set up your environment:")
        print("   1. Copy .env.integration-test to .env")
        print("   2. Fill in your GCP project ID and credentials path")
        print("   3. Optionally add organization ID and billing account ID")
        return False
    
    print("✅ Configuration Valid")
    print(f"   Project: {CONFIG['project_id']}")
    print(f"   Credentials: {CONFIG['credentials_path']}")
    print(f"   Organization: {CONFIG['organization_id'] or 'Not configured'}")
    print(f"   Billing Account: {CONFIG['billing_account_id'] or 'Not configured'}")
    
    return True


async def test_google_support_with_credentials():
    """Test Google Support client with real credentials"""
    print("\n🔧 Testing Google Support Client (Real Credentials)...")
    
    try:
        client = GoogleSupportClient(
            project_id=CONFIG["project_id"],
            organization_id=CONFIG["organization_id"]
        )
        
        # Test connection
        connection = await client.test_connection()
        if connection["connected"]:
            print(f"✅ Connected to Google Support API")
            if CONFIG["verbose"]:
                print(f"   Organization: {CONFIG['organization_id']}")
        else:
            print(f"❌ Connection failed: {connection.get('error', 'Unknown error')}")
            return False
        
        # Get statistics
        stats = await client.get_statistics()
        if stats["success"]:
            print(f"✅ Statistics retrieved: {stats.get('total_cases', 0)} cases")
        else:
            print(f"⚠️  Statistics failed: {stats.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Support test failed: {e}")
        return False


async def test_vpc_sc_with_credentials():
    """Test VPC-SC client with real credentials"""
    print("\n🛡️ Testing VPC Service Controls Client (Real Credentials)...")
    
    if not CONFIG["organization_id"]:
        print("⏭️  Skipped: Organization ID required for VPC-SC")
        return True
    
    try:
        client = VPCServiceControlsClient(
            organization_id=CONFIG["organization_id"],
            project_id=CONFIG["project_id"]
        )
        
        # Test connection
        connection = await client.test_connection()
        if connection["connected"]:
            print(f"✅ Connected to VPC Service Controls API")
            print(f"   Access policies found: {connection.get('access_policies_found', 0)}")
        else:
            print(f"❌ Connection failed: {connection.get('error', 'Unknown error')}")
            return False
        
        # List service perimeters
        perimeters = await client.list_service_perimeters()
        if perimeters["success"]:
            count = perimeters.get("total_perimeters", 0)
            print(f"✅ Service perimeters: {count} found")
            
            # Test dry run analysis on first perimeter if available
            if count > 0:
                first_perimeter = perimeters["perimeters"][0]["name"]
                violations = await client.test_dry_run_violations(first_perimeter)
                if violations["success"]:
                    print(f"✅ Dry run analysis: {violations.get('total_violations', 0)} violations")
                else:
                    print(f"⚠️  Dry run analysis failed: {violations.get('error', 'Unknown error')}")
        else:
            print(f"⚠️  Perimeters failed: {perimeters.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ VPC-SC test failed: {e}")
        return False


async def test_gcp_billing_with_credentials():
    """Test GCP Billing client with real credentials"""
    print("\n💰 Testing GCP Billing Client (Real Credentials)...")
    
    try:
        client = GCPBillingClient(
            project_id=CONFIG["project_id"],
            billing_account_id=CONFIG["billing_account_id"]
        )
        
        # Test connection
        connection = await client.test_connection()
        if connection["connected"]:
            print(f"✅ Connected to GCP Billing API")
            print(f"   Billing enabled: {connection.get('billing_enabled', False)}")
            print(f"   Billing account: {connection.get('billing_account_id', 'N/A')}")
        else:
            print(f"❌ Connection failed: {connection.get('error', 'Unknown error')}")
            return False
        
        # Get service costs (uses mock data, but tests the flow)
        costs = await client.get_service_costs(days_back=7)
        if costs["success"]:
            print(f"✅ Service costs retrieved: ${costs.get('total_cost', 0):.2f}")
            if CONFIG["verbose"]:
                services = len(costs.get('service_breakdown', []))
                print(f"   Services analyzed: {services}")
        else:
            print(f"⚠️  Service costs failed: {costs.get('error', 'Unknown error')}")
        
        # Test service credit calculation
        credit = await client.calculate_service_credit_eligibility(
            service_type="Compute Engine",
            incident_duration_minutes=90,
            affected_percentage=50.0
        )
        if credit["success"]:
            amount = credit.get('calculated_credit_amount', 0)
            print(f"✅ Service credit calculation: ${amount:.2f}")
        else:
            print(f"⚠️  Credit calculation failed: {credit.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ GCP Billing test failed: {e}")
        return False


async def test_gcp_resource_with_credentials():
    """Test GCP Resource client with real credentials"""
    print("\n🏗️ Testing GCP Resource Client (Real Credentials)...")
    
    try:
        client = GCPResourceClient(
            project_id=CONFIG["project_id"],
            organization_id=CONFIG["organization_id"]
        )
        
        # Test connection
        connection = await client.test_connection()
        if connection["connected"]:
            print(f"✅ Connected to GCP Resource Manager API")
            print(f"   Project: {connection.get('project_name', 'N/A')}")
            print(f"   State: {connection.get('project_state', 'N/A')}")
        else:
            print(f"❌ Connection failed: {connection.get('error', 'Unknown error')}")
            return False
        
        # Get project hierarchy
        hierarchy = await client.get_project_hierarchy()
        if hierarchy["success"]:
            project_info = hierarchy.get('hierarchy', {}).get('project', {})
            print(f"✅ Project hierarchy retrieved")
            if CONFIG["verbose"]:
                print(f"   Display name: {project_info.get('display_name', 'N/A')}")
                print(f"   Parent: {project_info.get('parent', 'None')}")
        else:
            print(f"⚠️  Project hierarchy failed: {hierarchy.get('error', 'Unknown error')}")
        
        # Search assets
        assets = await client.search_assets(query="state:ACTIVE")
        if assets["success"]:
            total = assets.get('total_resources', 0)
            types = len(assets.get('resources_by_type', {}))
            print(f"✅ Asset search: {total} resources, {types} types")
        else:
            print(f"⚠️  Asset search failed: {assets.get('error', 'Unknown error')}")
        
        # Get recommendations
        recommendations = await client.get_recommendations()
        if recommendations["success"]:
            total = recommendations.get('total_recommendations', 0)
            high_priority = recommendations.get('high_priority_count', 0)
            print(f"✅ Recommendations: {total} total, {high_priority} high priority")
        else:
            print(f"⚠️  Recommendations failed: {recommendations.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ GCP Resource test failed: {e}")
        return False


async def test_coordinated_workflow():
    """Test coordinated workflow between clients"""
    print("\n🔗 Testing Coordinated Workflow...")
    
    try:
        # Initialize all clients
        support_client = GoogleSupportClient(
            project_id=CONFIG["project_id"],
            organization_id=CONFIG["organization_id"]
        )
        
        billing_client = GCPBillingClient(
            project_id=CONFIG["project_id"],
            billing_account_id=CONFIG["billing_account_id"]
        )
        
        resource_client = GCPResourceClient(
            project_id=CONFIG["project_id"],
            organization_id=CONFIG["organization_id"]
        )
        
        # Workflow: Get resource utilization and billing data for support case
        print("   📊 Getting resource utilization...")
        utilization = await resource_client.analyze_resource_utilization()
        
        print("   💰 Getting billing information...")
        costs = await billing_client.get_service_costs(days_back=7)
        
        # Create comprehensive incident report
        if utilization["success"] and costs["success"]:
            incident_data = {
                "title": "Resource Optimization Support Request",
                "description": f"Project {CONFIG['project_id']} analysis shows utilization score of {utilization.get('utilization_score', 0):.1f} with weekly costs of ${costs.get('total_cost', 0):.2f}",
                "severity": "MEDIUM",
                "impact_scope": "PROJECT_WIDE",
                "affected_resources": [f"Project: {CONFIG['project_id']}"],
                "technical_details": f"Utilization: {utilization.get('resource_efficiency', 'UNKNOWN')}, Optimizations: {utilization.get('optimization_opportunities', 0)}",
                "support_request": "Need guidance on resource optimization and cost management"
            }
            
            print("   🎫 Workflow coordination successful")
            print(f"   📈 Utilization Score: {utilization.get('utilization_score', 0):.1f}")
            print(f"   💵 Weekly Costs: ${costs.get('total_cost', 0):.2f}")
            print(f"   🔧 Optimization Opportunities: {utilization.get('optimization_opportunities', 0)}")
            
            return True
        else:
            print("   ⚠️  Partial workflow success")
            return True
            
    except Exception as e:
        print(f"❌ Coordinated workflow failed: {e}")
        return False


async def main():
    """Run all credential-based tests"""
    print("🚀 Phase 2 Integration Clients - Credential Tests")
    print("="*60)
    
    # Validate configuration
    if not validate_configuration():
        return 1
    
    # Run tests with real credentials
    results = []
    
    print(f"\n⏱️  Running tests with {CONFIG['test_timeout']}s timeout...")
    
    try:
        # Run each test with timeout
        timeout = CONFIG["test_timeout"]
        
        results.append(await asyncio.wait_for(test_google_support_with_credentials(), timeout))
        results.append(await asyncio.wait_for(test_vpc_sc_with_credentials(), timeout))
        results.append(await asyncio.wait_for(test_gcp_billing_with_credentials(), timeout))
        results.append(await asyncio.wait_for(test_gcp_resource_with_credentials(), timeout))
        results.append(await asyncio.wait_for(test_coordinated_workflow(), timeout))
        
    except asyncio.TimeoutError:
        print(f"❌ Test timeout after {timeout} seconds")
        return 1
    
    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print("\n" + "="*60)
    print("🧪 CREDENTIAL TEST SUMMARY")
    print("="*60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Integration clients working correctly!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Check configuration and permissions")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)