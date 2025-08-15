"""
Asset Inventory Usage Examples

This file demonstrates how to use the enhanced Asset Inventory integration
with natural language queries and ADK chat system.
"""

import asyncio
import json
from datetime import datetime

# Example 1: Natural Language Queries via Enhanced Service
async def example_natural_language_queries():
    """Demonstrate natural language processing capabilities."""
    print("="*60)
    print("EXAMPLE 1: Natural Language Queries")
    print("="*60)
    
    from backend.services.enhanced_asset_inventory_service import EnhancedGCPAssetInventoryService
    
    # Initialize service
    service = EnhancedGCPAssetInventoryService("your-project-id")
    
    # Example queries that users might ask
    example_queries = [
        # Compute resources
        "What compute instances do I have?",
        "Show me my VM instances",
        "List all my servers",
        
        # Storage resources  
        "What storage buckets exist?",
        "Show me my databases",
        "Do I have any Cloud SQL instances?",
        
        # Serverless resources
        "What cloud functions are deployed?",
        "Show me my App Engine applications",
        
        # Container resources
        "What Kubernetes clusters do I have?",
        "Show me my GKE clusters",
        
        # Security analysis
        "Analyze my security posture",
        "What security risks do I have?",
        "Check my firewall rules",
        
        # General overview
        "Give me an overview of my GCP resources",
        "What services am I using?",
        "Show me everything in my project"
    ]
    
    for query in example_queries:
        print(f"\n🗣️  User Query: '{query}'")
        
        # Parse intent and extract target resources
        intent = service._parse_query_intent(query)
        resources = service._extract_target_resources(query)
        
        print(f"   📊 Detected Intent: {intent}")
        print(f"   🎯 Target Resources: {len(resources)} types")
        
        if resources:
            print(f"   📋 Resource Types: {resources[:3]}...")  # Show first 3
        
        # In a real scenario, this would make API calls:
        # result = await service.process_natural_language_query(query)
        print(f"   📞 Would call: cloudasset.googleapis.com/ListAssets")

# Example 2: Direct Tool Usage
def example_direct_tool_usage():
    """Demonstrate direct tool function usage."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Direct Tool Usage")
    print("="*60)
    
    from tools.gcp_tools.asset_inventory_tools import (
        discover_gcp_resources,
        get_compute_instances,
        get_storage_buckets,
        analyze_security_assets
    )
    
    # Example 1: Natural language discovery
    print("\n🔍 Natural Language Discovery:")
    result = discover_gcp_resources("show me my compute instances")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Has Data: {'data' in result}")
    
    # Example 2: Specific resource types
    print("\n💻 Compute Instances:")
    instances = get_compute_instances()
    print(f"   Success: {instances.get('success', False)}")
    print(f"   Response Type: {type(instances.get('data', {}))}")
    
    # Example 3: Storage buckets
    print("\n🪣 Storage Buckets:")
    buckets = get_storage_buckets()
    print(f"   Success: {buckets.get('success', False)}")
    
    # Example 4: Security analysis
    print("\n🔒 Security Analysis:")
    security = analyze_security_assets()
    print(f"   Success: {security.get('success', False)}")

# Example 3: Chat Integration Scenarios
def example_chat_integration():
    """Demonstrate how queries flow through chat system."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Chat Integration Flow")
    print("="*60)
    
    chat_scenarios = [
        {
            "user_input": "What compute instances do I have?",
            "intent": "list_queries",
            "routing": "get_compute_instances()",
            "api_call": "cloudasset.googleapis.com/ListAssets",
            "response_type": "Structured list of VM instances with security analysis"
        },
        {
            "user_input": "Are there any security risks?",
            "intent": "security_queries", 
            "routing": "analyze_security_assets()",
            "api_call": "cloudasset.googleapis.com/ListAssets",
            "response_type": "Security findings with risk levels and recommendations"
        },
        {
            "user_input": "Show me my cloud functions",
            "intent": "list_queries",
            "routing": "get_cloud_functions()",
            "api_call": "cloudasset.googleapis.com/ListAssets",
            "response_type": "List of Cloud Functions with deployment details"
        }
    ]
    
    for i, scenario in enumerate(chat_scenarios, 1):
        print(f"\n💬 Scenario {i}:")
        print(f"   User: '{scenario['user_input']}'")
        print(f"   🧠 Intent Detection: {scenario['intent']}")
        print(f"   🎯 Tool Routing: {scenario['routing']}")
        print(f"   📞 API Call: {scenario['api_call']}")
        print(f"   📋 Response: {scenario['response_type']}")

# Example 4: API Endpoint Usage
def example_api_endpoints():
    """Demonstrate REST API endpoint usage."""
    print("\n" + "="*60)
    print("EXAMPLE 4: REST API Endpoints")
    print("="*60)
    
    api_examples = [
        {
            "method": "POST",
            "endpoint": "/api/v1/assets/discover",
            "body": {"query": "show me my compute instances"},
            "description": "Natural language discovery"
        },
        {
            "method": "GET", 
            "endpoint": "/api/v1/assets/compute/instances",
            "body": None,
            "description": "Get all compute instances"
        },
        {
            "method": "GET",
            "endpoint": "/api/v1/assets/storage/buckets",
            "body": None,
            "description": "Get all storage buckets"
        },
        {
            "method": "GET",
            "endpoint": "/api/v1/assets/security/analyze", 
            "body": None,
            "description": "Security analysis"
        },
        {
            "method": "GET",
            "endpoint": "/api/v1/assets/summary",
            "body": None,
            "description": "Complete inventory summary"
        },
        {
            "method": "GET",
            "endpoint": "/api/v1/assets/search?name_pattern=prod-*",
            "body": None,
            "description": "Search by name pattern"
        }
    ]
    
    for api in api_examples:
        print(f"\n🌐 {api['method']} {api['endpoint']}")
        if api['body']:
            print(f"   📦 Body: {json.dumps(api['body'])}")
        print(f"   📝 Description: {api['description']}")
        print(f"   📞 Makes call to: cloudasset.googleapis.com")

# Example 5: Security Analysis Output
def example_security_analysis():
    """Show example security analysis output structure."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Security Analysis Output")
    print("="*60)
    
    # Example of what security analysis returns
    example_security_result = {
        "analysis_type": "security",
        "security_analysis": {
            "total_assets_analyzed": 25,
            "security_findings": [
                {
                    "asset_name": "web-server-1",
                    "asset_type": "compute.googleapis.com/Instance",
                    "finding_type": "external_ip_exposure",
                    "description": "Instance has external IP address",
                    "risk_level": "medium",
                    "recommendation": "Consider using Cloud NAT or private instances"
                },
                {
                    "asset_name": "allow-all-firewall",
                    "asset_type": "compute.googleapis.com/Firewall",
                    "finding_type": "overly_permissive_firewall",
                    "description": "Firewall rule allows traffic from any IP (0.0.0.0/0)",
                    "risk_level": "high",
                    "recommendation": "Restrict source IP ranges to specific networks"
                }
            ],
            "risk_summary": {
                "high": 1,
                "medium": 3, 
                "low": 2
            },
            "recommendations": [
                "Review firewall rules for overly permissive access",
                "Enable public access prevention on storage buckets",
                "Implement service account key rotation"
            ]
        },
        "assets_by_category": {
            "compute": {"count": 5, "assets": ["..."]},
            "storage": {"count": 3, "assets": ["..."]},
            "networking": {"count": 10, "assets": ["..."]}
        },
        "api_calls_made": [
            {
                "api": "cloudasset.googleapis.com",
                "method": "ListAssets",
                "timestamp": "2024-01-01T12:00:00Z",
                "project": "your-project-id"
            }
        ]
    }
    
    print("📊 Example Security Analysis Result:")
    print(json.dumps(example_security_result, indent=2))

def main():
    """Run all examples."""
    print("🚀 ASSET INVENTORY USAGE EXAMPLES")
    print(f"📅 Generated: {datetime.now().isoformat()}")
    
    # Run all examples
    try:
        # Example 1: Natural language queries (async)
        asyncio.run(example_natural_language_queries())
        
        # Example 2: Direct tool usage
        example_direct_tool_usage()
        
        # Example 3: Chat integration
        example_chat_integration()
        
        # Example 4: API endpoints
        example_api_endpoints()
        
        # Example 5: Security analysis
        example_security_analysis()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED")
        print("="*60)
        print("\n📋 Key Takeaways:")
        print("   • Natural language queries are automatically parsed and routed")
        print("   • All Asset Inventory tools work in the ADK system")
        print("   • REST API endpoints provide programmatic access")
        print("   • Security analysis provides actionable insights")
        print("   • All API calls are logged for transparency")
        
        print("\n🚀 Next Steps:")
        print("   1. Enable Asset Inventory API: gcloud services enable cloudasset.googleapis.com")
        print("   2. Configure service account with Cloud Asset Viewer role")
        print("   3. Set GOOGLE_CLOUD_PROJECT environment variable")
        print("   4. Test with real project data")
        print("   5. Integrate with chat interface")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        print("   This is expected in environments without full dependencies")

if __name__ == "__main__":
    main()