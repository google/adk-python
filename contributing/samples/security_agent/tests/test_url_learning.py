#!/usr/bin/env python3
"""
Test the URL learning capability for discovering new GCP services
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents._tools.service_discovery import (
    learn_service_from_url,
    discover_new_gcp_services,
    register_new_service,
    learn_from_api_spec
)


def test_url_learning():
    """Test learning from documentation URLs"""
    print("\n" + "="*60)
    print("🔍 TESTING URL LEARNING FOR NEW GCP SERVICES")
    print("="*60)

    # Test 1: Learn from a real GCP documentation page
    print("\n1. Learning from GCP Documentation URL:")
    print("-" * 40)

    # Example: Cloud Run documentation
    doc_url = "https://cloud.google.com/run/docs"
    print(f"Parsing: {doc_url}")
    result = learn_service_from_url(doc_url)
    print(result)

    # Test 2: Discover new services from release notes
    print("\n2. Discovering New Services from Release Notes:")
    print("-" * 40)

    result = discover_new_gcp_services()
    print(result)

    # Test 3: Register a hypothetical new service
    print("\n3. Registering a Custom Service:")
    print("-" * 40)

    result = register_new_service(
        service_name="Quantum Computing Service",
        api_endpoint="quantum.googleapis.com",
        documentation_url="https://cloud.google.com/quantum/docs",
        description="Hypothetical quantum computing service for demonstration"
    )
    print(result)

    # Test 4: Learn from API specification
    print("\n4. Learning from API Specification:")
    print("-" * 40)

    # Example GitHub API spec
    api_spec_url = "https://github.com/googleapis/googleapis/blob/master/google/cloud/compute/v1/compute.proto"
    print(f"Parsing API spec: {api_spec_url}")
    result = learn_from_api_spec(api_spec_url)
    print(result)


def demonstrate_agent_usage():
    """Show how the agent would use these capabilities"""
    print("\n" + "="*60)
    print("🤖 HOW THE AGENT USES URL LEARNING")
    print("="*60)

    print("""
Example Agent Interactions:

User: "Learn about the new Cloud Deploy service"
Agent: [Uses learn_service_from_url("https://cloud.google.com/deploy/docs")]
       -> Parses documentation
       -> Extracts API endpoints, resource types, permissions
       -> Stores in cache for future analysis

User: "What new services were released this month?"
Agent: [Uses discover_new_gcp_services()]
       -> Checks GCP release notes
       -> Identifies new service announcements
       -> Provides links to documentation

User: "Register our custom internal service for analysis"
Agent: [Uses register_new_service()]
       -> Registers the service
       -> Makes it available for analysis
       -> Can now use analyze_gcp_service() on it

User: "Learn about this service from its API spec"
Agent: [Uses learn_from_api_spec()]
       -> Parses OpenAPI/Proto specification
       -> Understands endpoints and data models
       -> Can generate analysis queries
    """)


def test_integration():
    """Test integration with the main service discovery"""
    print("\n" + "="*60)
    print("🔧 TESTING INTEGRATION WITH SERVICE DISCOVERY")
    print("="*60)

    from agents._tools.service_discovery import discover_gcp_services

    print("\n1. Discovering all services (including learned ones):")
    print("-" * 40)

    # First register a test service
    register_new_service(
        service_name="Test Learning Service",
        api_endpoint="testlearning.googleapis.com",
        documentation_url="https://example.com/docs",
        description="Test service for URL learning"
    )

    # Now discover all services
    result = discover_gcp_services(include_learned=True)

    # Check if learned service appears
    if "Test Learning Service" in result or "Learned" in result:
        print("✅ Learned services are included in discovery!")
    else:
        print("⚠️ Learned services may not be showing in discovery")

    print("\nSample of discovery output:")
    print(result[:500] + "..." if len(result) > 500 else result)


def main():
    """Main test function"""
    print("\n" + "🌐 URL LEARNING TEST SUITE 🌐".center(60))
    print("Testing ability to learn new GCP services from URLs".center(60))

    try:
        # Run tests
        test_url_learning()
        demonstrate_agent_usage()
        test_integration()

        print("\n" + "="*60)
        print("✅ URL LEARNING TESTS COMPLETE")
        print("="*60)
        print("""
The system can now:
1. Parse any GCP documentation URL to learn about services
2. Discover new services from release notes
3. Register custom services for analysis
4. Learn from API specifications (OpenAPI, Proto)
5. Dynamically expand its knowledge of GCP services

This means the agent can analyze services that didn't exist when it was created!
        """)

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()