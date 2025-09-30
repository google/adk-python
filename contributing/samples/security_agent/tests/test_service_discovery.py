#!/usr/bin/env python3
"""
Test script for Service Discovery and On-Demand Analysis functionality
"""

import requests
import json
import time
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents._tools.service_discovery import (
    discover_gcp_services,
    analyze_gcp_service,
    get_service_resources,
    suggest_service_analysis
)

class ServiceDiscoveryTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []

    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(f"  {text}")
        print("=" * 60)

    def print_test(self, name, result, details=""):
        """Print test result"""
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if details:
            print(f"    Details: {details}")
        self.test_results.append((name, result))

    def test_direct_tools(self):
        """Test the service discovery tools directly"""
        self.print_header("Testing Direct Tool Functions")

        # Test 1: Discover services
        print("\n1. Testing discover_gcp_services()...")
        try:
            result = discover_gcp_services(include_all=False)
            if result['success']:
                services = result.get('services', [])
                self.print_test(
                    "Discover GCP Services",
                    True,
                    f"Found {len(services)} services"
                )
                if services:
                    print(f"    Sample services: {[s['name'] for s in services[:3]]}")
            else:
                self.print_test(
                    "Discover GCP Services",
                    False,
                    result.get('error', 'Unknown error')
                )
        except Exception as e:
            self.print_test("Discover GCP Services", False, str(e))

        # Test 2: Analyze a specific service
        print("\n2. Testing analyze_gcp_service()...")
        try:
            analysis_query = json.dumps({
                'service': 'Compute Engine',
                'types': ['security', 'compliance']
            })
            result = analyze_gcp_service(
                service_name="Compute Engine",
                analysis_query=analysis_query
            )
            self.print_test(
                "Analyze GCP Service",
                result['success'],
                "Analysis completed" if result['success'] else result.get('error')
            )
        except Exception as e:
            self.print_test("Analyze GCP Service", False, str(e))

        # Test 3: Get service resources
        print("\n3. Testing get_service_resources()...")
        try:
            result = get_service_resources(
                service_name="Cloud Storage",
                limit=10
            )
            if result['success']:
                resources = result.get('resources', [])
                self.print_test(
                    "Get Service Resources",
                    True,
                    f"Found {len(resources)} resources"
                )
            else:
                self.print_test(
                    "Get Service Resources",
                    False,
                    result.get('error')
                )
        except Exception as e:
            self.print_test("Get Service Resources", False, str(e))

        # Test 4: Suggest analysis
        print("\n4. Testing suggest_service_analysis()...")
        try:
            result = suggest_service_analysis(
                user_query="Find security issues in my Cloud Run services"
            )
            if result['success']:
                suggestions = result.get('suggestions', [])
                self.print_test(
                    "Suggest Service Analysis",
                    True,
                    f"Got {len(suggestions)} suggestions"
                )
                if suggestions:
                    print(f"    First suggestion: {suggestions[0].get('title', 'N/A')}")
            else:
                self.print_test(
                    "Suggest Service Analysis",
                    False,
                    result.get('error')
                )
        except Exception as e:
            self.print_test("Suggest Service Analysis", False, str(e))

    def test_api_endpoints(self):
        """Test the Flask API endpoints"""
        self.print_header("Testing Flask API Endpoints")

        # Check if Flask server is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print("❌ Flask server not responding at", self.base_url)
                print("   Please start the Flask app with: python flask_app.py")
                return
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Flask server at", self.base_url)
            print("   Please start the Flask app with: python flask_app.py")
            return

        # Test 1: Discover services endpoint
        print("\n1. Testing /api/services/discover...")
        try:
            response = requests.get(f"{self.base_url}/api/services/discover")
            if response.status_code == 200:
                data = response.json()
                self.print_test(
                    "API: Discover Services",
                    data.get('success', False),
                    f"Found {data.get('total_count', 0)} services"
                )
            else:
                self.print_test(
                    "API: Discover Services",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            self.print_test("API: Discover Services", False, str(e))

        # Test 2: Analyze service endpoint
        print("\n2. Testing /api/services/analyze...")
        try:
            payload = {
                "service_name": "Compute Engine",
                "analysis_types": ["security", "compliance"]
            }
            response = requests.post(
                f"{self.base_url}/api/services/analyze",
                json=payload
            )
            if response.status_code == 200:
                data = response.json()
                self.print_test(
                    "API: Analyze Service",
                    data.get('success', False),
                    data.get('message', 'No message')
                )
            else:
                self.print_test(
                    "API: Analyze Service",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            self.print_test("API: Analyze Service", False, str(e))

        # Test 3: Get resources endpoint
        print("\n3. Testing /api/services/resources/<service>...")
        try:
            response = requests.get(
                f"{self.base_url}/api/services/resources/Cloud%20Storage?limit=5"
            )
            if response.status_code == 200:
                data = response.json()
                self.print_test(
                    "API: Get Resources",
                    data.get('success', False),
                    f"Found {data.get('count', 0)} resources"
                )
            else:
                self.print_test(
                    "API: Get Resources",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            self.print_test("API: Get Resources", False, str(e))

        # Test 4: Suggest analysis endpoint
        print("\n4. Testing /api/services/suggest...")
        try:
            response = requests.get(
                f"{self.base_url}/api/services/suggest?query=Find%20unused%20resources"
            )
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                self.print_test(
                    "API: Suggest Analysis",
                    data.get('success', False),
                    f"Got {len(recommendations)} recommendations"
                )
            else:
                self.print_test(
                    "API: Suggest Analysis",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            self.print_test("API: Suggest Analysis", False, str(e))

        # Test 5: Get categories endpoint
        print("\n5. Testing /api/services/categories...")
        try:
            response = requests.get(f"{self.base_url}/api/services/categories")
            if response.status_code == 200:
                data = response.json()
                categories = data.get('categories', [])
                self.print_test(
                    "API: Get Categories",
                    data.get('success', False),
                    f"Got {len(categories)} categories"
                )
            else:
                self.print_test(
                    "API: Get Categories",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            self.print_test("API: Get Categories", False, str(e))

    def test_agent_integration(self):
        """Test integration with ADK agent"""
        self.print_header("Testing ADK Agent Integration")

        # Test if agent has the new tools
        try:
            from agents.agent import root_agent

            tool_names = []
            for tool in root_agent.tools:
                if hasattr(tool, 'function'):
                    tool_names.append(tool.function.__name__)

            # Check for service discovery tools
            service_tools = [
                'discover_gcp_services',
                'analyze_gcp_service',
                'get_service_resources',
                'suggest_service_analysis'
            ]

            for tool_name in service_tools:
                has_tool = tool_name in tool_names
                self.print_test(
                    f"Agent has tool: {tool_name}",
                    has_tool
                )

        except Exception as e:
            self.print_test("Agent Integration", False, str(e))

    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")

        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if pass_rate == 100:
            print("\n🎉 All tests passed! Service Discovery is working correctly.")
        elif pass_rate >= 80:
            print("\n✅ Most tests passed. Service Discovery is mostly functional.")
        elif pass_rate >= 50:
            print("\n⚠️ Some tests failed. Service Discovery needs attention.")
        else:
            print("\n❌ Many tests failed. Service Discovery needs debugging.")

def main():
    """Main test function"""
    tester = ServiceDiscoveryTester()

    print("\n" + "🔍 SERVICE DISCOVERY TEST SUITE 🔍".center(60))
    print("Testing On-Demand Analysis Functionality".center(60))

    # Run tests
    tester.test_direct_tools()
    tester.test_api_endpoints()
    tester.test_agent_integration()

    # Print summary
    tester.print_summary()

    print("\n" + "=" * 60)
    print("Test instructions:")
    print("1. To test API endpoints, start Flask app: python flask_app.py")
    print("2. To test ADK integration, start ADK: adk web")
    print("3. To test frontend, run: streamlit run frontend/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()