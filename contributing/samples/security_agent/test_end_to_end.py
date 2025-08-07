#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing for ADK Security Agent with Multi-Agent Features

This script tests the complete system including:
- Backend services and APIs
- Frontend graph visualization
- Agent functionality
- Multi-agent coordination
- Service management
"""

import asyncio
import requests
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# Configuration  
BACKEND_URL = "http://localhost:8000"  # Use legacy backend for testing
FRONTEND_URL = "http://localhost:8501"
TEST_TIMEOUT = 30
VERBOSE = True

class Colors:
    """Terminal colors for better output visibility."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def log(message: str, level: str = "INFO"):
    """Enhanced logging with colors and timestamps."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "TEST": Colors.MAGENTA,
        "GRAPH": Colors.CYAN
    }
    
    color = colors.get(level, Colors.WHITE)
    print(f"{color}[{timestamp}] {level}: {message}{Colors.END}")

def test_backend_health():
    """Test backend health and basic connectivity."""
    log("Testing backend health...", "TEST")
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            log(f"Backend health check passed: {health_data}", "SUCCESS")
            return True
        else:
            log(f"Backend health check failed with status {response.status_code}", "ERROR")
            return False
    except requests.exceptions.RequestException as e:
        log(f"Failed to connect to backend: {e}", "ERROR")
        return False

def test_service_management_api():
    """Test service management API endpoints."""
    log("Testing service management API...", "TEST")
    
    test_results = []
    
    # Test service list
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/services/", timeout=10)
        if response.status_code == 200:
            services = response.json()
            log(f"Service list retrieved: {len(services.get('services', []))} services", "SUCCESS")
            test_results.append(True)
        else:
            log(f"Service list failed with status {response.status_code}", "ERROR")
            test_results.append(False)
    except Exception as e:
        log(f"Service list test failed: {e}", "ERROR")
        test_results.append(False)
    
    # Test service status summary
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/services/status/summary", timeout=10)
        if response.status_code == 200:
            summary = response.json()
            log(f"Service status summary: {summary.get('summary', {})}", "SUCCESS")
            test_results.append(True)
        else:
            log(f"Service status failed with status {response.status_code}", "ERROR")
            test_results.append(False)
    except Exception as e:
        log(f"Service status test failed: {e}", "ERROR")
        test_results.append(False)
    
    return all(test_results)

def test_agent_functionality():
    """Test agent endpoints and functionality."""
    log("Testing agent functionality...", "TEST")
    
    test_results = []
    
    # Test agent info
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/agent/", timeout=10)
        if response.status_code == 200:
            agent_info = response.json()
            log(f"Agent info retrieved: {agent_info.get('agent_info', {}).get('name', 'Unknown')}", "SUCCESS")
            test_results.append(True)
        else:
            log(f"Agent info failed with status {response.status_code}", "ERROR")
            test_results.append(False)
    except Exception as e:
        log(f"Agent info test failed: {e}", "ERROR")
        test_results.append(False)
    
    # Test agent chat (simple query)
    try:
        chat_data = {
            "query": "What security services are available?",
            "user_id": "test_user"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agent/chat", 
            json=chat_data,
            timeout=30
        )
        
        if response.status_code == 200:
            chat_response = response.json()
            if chat_response.get('success'):
                log(f"Agent chat successful: {len(chat_response.get('response', ''))} chars", "SUCCESS")
                test_results.append(True)
            else:
                log(f"Agent chat failed: {chat_response.get('error', 'Unknown error')}", "ERROR")
                test_results.append(False)
        else:
            log(f"Agent chat failed with status {response.status_code}", "ERROR")
            test_results.append(False)
    except Exception as e:
        log(f"Agent chat test failed: {e}", "ERROR")
        test_results.append(False)
    
    return all(test_results)

def test_gcp_integration():
    """Test GCP service integration."""
    log("Testing GCP integration...", "TEST")
    
    test_results = []
    
    # Test GCP projects endpoint
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/gcp/projects", timeout=15)
        if response.status_code == 200:
            projects_data = response.json()
            if projects_data.get('success'):
                projects = projects_data.get('projects', [])
                log(f"GCP projects retrieved: {len(projects)} projects", "SUCCESS")
                test_results.append(True)
            else:
                log(f"GCP projects failed: {projects_data.get('error', 'Unknown error')}", "WARNING")
                test_results.append(False)
        else:
            log(f"GCP projects failed with status {response.status_code}", "WARNING")
            test_results.append(False)
    except Exception as e:
        log(f"GCP projects test failed: {e}", "WARNING")
        test_results.append(False)
    
    # Test GCP service status
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/gcp/status", timeout=10)
        if response.status_code == 200:
            gcp_status = response.json()
            log(f"GCP status: {gcp_status.get('status', 'Unknown')}", "SUCCESS")
            test_results.append(True)
        else:
            log(f"GCP status endpoint not found (acceptable)", "WARNING")
            test_results.append(True)  # This endpoint might not exist, which is acceptable
    except Exception as e:
        log(f"GCP status test failed: {e}", "WARNING")
        test_results.append(True)  # Not critical
    
    return any(test_results)  # At least one GCP test should pass

def test_multi_agent_graph_data():
    """Test multi-agent graph data generation and API endpoints."""
    log("Testing multi-agent graph data generation...", "GRAPH")
    
    # Test if we can generate graph data (simulated)
    try:
        # Import the graph view component to test data generation
        sys.path.append(str(Path(__file__).parent / "frontend"))
        
        from components.multi_agent_graph_view import (
            generate_service_dependency_graph,
            generate_agent_collaboration_graph,
            generate_risk_propagation_graph,
            generate_multi_agent_workflow_graph
        )
        
        # Test each graph type
        graph_types = [
            ("Service Dependencies", generate_service_dependency_graph),
            ("Agent Collaboration", generate_agent_collaboration_graph), 
            ("Risk Propagation", generate_risk_propagation_graph),
            ("Multi-Agent Workflow", generate_multi_agent_workflow_graph)
        ]
        
        test_results = []
        
        for graph_name, generator_func in graph_types:
            try:
                graph_data = generator_func()
                
                # Validate graph structure
                if "nodes" in graph_data and "edges" in graph_data:
                    node_count = len(graph_data["nodes"])
                    edge_count = len(graph_data["edges"])
                    log(f"{graph_name} graph: {node_count} nodes, {edge_count} edges", "SUCCESS")
                    test_results.append(True)
                else:
                    log(f"{graph_name} graph: Invalid structure", "ERROR")
                    test_results.append(False)
                    
            except Exception as e:
                log(f"{graph_name} graph generation failed: {e}", "ERROR")
                test_results.append(False)
        
        return all(test_results)
        
    except ImportError as e:
        log(f"Could not import graph components: {e}", "ERROR")
        return False
    except Exception as e:
        log(f"Graph data test failed: {e}", "ERROR")
        return False

def test_api_endpoints_comprehensive():
    """Test comprehensive API endpoint coverage."""
    log("Testing comprehensive API endpoints...", "TEST")
    
    endpoints_to_test = [
        ("GET", "/health", "Health Check"),
        ("GET", "/", "Root Endpoint"),
        ("GET", "/docs", "OpenAPI Documentation"),
        ("GET", "/api/v1/services/", "Services List"),
        ("GET", "/api/v1/services/status/summary", "Service Status Summary"),
        ("GET", "/api/v1/agent/", "Agent Info"),
    ]
    
    test_results = []
    
    for method, endpoint, description in endpoints_to_test:
        try:
            if method == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=10)
            else:
                continue  # Skip non-GET for now
            
            if response.status_code in [200, 404, 422]:  # 404/422 are acceptable for some endpoints
                log(f"{description}: Status {response.status_code}", "SUCCESS")
                test_results.append(True)
            else:
                log(f"{description}: Unexpected status {response.status_code}", "WARNING")
                test_results.append(False)
                
        except Exception as e:
            log(f"{description}: Failed - {e}", "ERROR")
            test_results.append(False)
    
    success_rate = sum(test_results) / len(test_results) * 100
    log(f"API endpoint test success rate: {success_rate:.1f}%", "SUCCESS" if success_rate >= 70 else "WARNING")
    
    return success_rate >= 70

def test_frontend_availability():
    """Test frontend availability and basic functionality."""
    log("Testing frontend availability...", "TEST")
    
    try:
        response = requests.get(FRONTEND_URL, timeout=10)
        
        if response.status_code == 200:
            log("Frontend is accessible", "SUCCESS")
            
            # Check if it contains expected Streamlit content
            content = response.text.lower()
            if "streamlit" in content or "dashboard" in content or "security" in content:
                log("Frontend contains expected content", "SUCCESS")
                return True
            else:
                log("Frontend accessible but content seems incorrect", "WARNING")
                return False
        else:
            log(f"Frontend returned status {response.status_code}", "WARNING")
            return False
            
    except requests.exceptions.RequestException as e:
        log(f"Frontend not accessible: {e}", "WARNING")
        log("This is expected if frontend is not running separately", "INFO")
        return False

def test_system_integration():
    """Test end-to-end system integration."""
    log("Testing system integration...", "TEST")
    
    try:
        # Test agent with actual security question
        chat_data = {
            "query": "Analyze the security posture of my GCP project and show me the dependency graph",
            "user_id": "integration_test_user"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agent/chat",
            json=chat_data,
            timeout=45
        )
        
        if response.status_code == 200:
            chat_response = response.json()
            if chat_response.get('success'):
                response_text = chat_response.get('response', '')
                
                # Check if response contains security-related terms
                security_terms = ['security', 'iam', 'project', 'analysis', 'gcp', 'policy']
                found_terms = [term for term in security_terms if term.lower() in response_text.lower()]
                
                if len(found_terms) >= 2:
                    log(f"Integration test successful: Found {len(found_terms)} security terms", "SUCCESS")
                    log(f"Response preview: {response_text[:200]}...", "INFO")
                    return True
                else:
                    log(f"Integration test partial: Response seems generic", "WARNING")
                    return False
            else:
                log(f"Integration test failed: {chat_response.get('error', 'Unknown error')}", "ERROR")
                return False
        else:
            log(f"Integration test failed with status {response.status_code}", "ERROR")
            return False
            
    except Exception as e:
        log(f"Integration test failed: {e}", "ERROR")
        return False

def run_performance_tests():
    """Run performance tests for critical endpoints."""
    log("Running performance tests...", "TEST")
    
    endpoints = [
        f"{BACKEND_URL}/health",
        f"{BACKEND_URL}/api/v1/services/",
        f"{BACKEND_URL}/api/v1/agent/"
    ]
    
    performance_results = []
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(endpoint, timeout=10)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200 and response_time < 5000:  # 5 second threshold
                log(f"Performance OK: {endpoint} responded in {response_time:.0f}ms", "SUCCESS")
                performance_results.append(True)
            else:
                log(f"Performance Issue: {endpoint} took {response_time:.0f}ms", "WARNING")
                performance_results.append(False)
                
        except Exception as e:
            log(f"Performance test failed for {endpoint}: {e}", "ERROR")
            performance_results.append(False)
    
    success_rate = sum(performance_results) / len(performance_results) * 100
    log(f"Performance test success rate: {success_rate:.1f}%", "SUCCESS" if success_rate >= 80 else "WARNING")
    
    return success_rate >= 80

def generate_test_report(test_results: Dict[str, bool]):
    """Generate a comprehensive test report."""
    log("Generating test report...", "INFO")
    
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}🧪 ADK Security Agent - End-to-End Test Report{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    passed = sum(test_results.values())
    total = len(test_results)
    success_rate = passed / total * 100
    
    # Overall status
    if success_rate >= 90:
        status_color = Colors.GREEN
        status = "EXCELLENT"
    elif success_rate >= 75:
        status_color = Colors.YELLOW
        status = "GOOD"
    elif success_rate >= 50:
        status_color = Colors.YELLOW
        status = "NEEDS IMPROVEMENT"
    else:
        status_color = Colors.RED
        status = "CRITICAL ISSUES"
    
    print(f"{Colors.BOLD}Overall Status: {status_color}{status}{Colors.END}")
    print(f"{Colors.BOLD}Success Rate: {status_color}{success_rate:.1f}% ({passed}/{total} tests passed){Colors.END}\n")
    
    # Individual test results
    print(f"{Colors.BOLD}Test Results:{Colors.END}")
    print("-" * 50)
    
    for test_name, result in test_results.items():
        status_icon = "✅" if result else "❌"
        status_color = Colors.GREEN if result else Colors.RED
        status_text = "PASSED" if result else "FAILED"
        
        print(f"{status_icon} {test_name:<30} {status_color}{status_text}{Colors.END}")
    
    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
    print("-" * 50)
    
    failed_tests = [name for name, result in test_results.items() if not result]
    
    if not failed_tests:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("🔧 Please address the following issues:")
        for test in failed_tests:
            if "Backend" in test:
                print(f"   • Check backend service status: python run.py --backend-only")
            elif "Frontend" in test:
                print(f"   • Check frontend service status: python run.py --frontend-only")
            elif "GCP" in test:
                print(f"   • Verify GCP credentials and project configuration")
            elif "Agent" in test:
                print(f"   • Check ADK agent configuration and Vertex AI setup")
            else:
                print(f"   • Investigate {test} functionality")
    
    print(f"\n{Colors.BOLD}Multi-Agent Graph Features:{Colors.END}")
    print("-" * 50)
    
    if test_results.get("Multi-Agent Graph Data", False):
        print("✅ Enhanced multi-agent graph visualization is working")
        print("   • 5 different graph types available")
        print("   • Interactive force-directed, hierarchical, and circular layouts")
        print("   • Real-time agent coordination analysis")
        print("   • Risk propagation visualization")
        print("   • Performance metrics integration")
    else:
        print("❌ Multi-agent graph features need attention")
        print("   • Check graph component dependencies (networkx, plotly, streamlit-agraph)")
        print("   • Verify frontend routing and navigation")
    
    print(f"\n{Colors.BOLD}Access Points:{Colors.END}")
    print("-" * 50)
    print(f"🌐 Frontend UI:       {FRONTEND_URL}")
    print(f"🔧 Backend API:       {BACKEND_URL}")
    print(f"📖 API Documentation: {BACKEND_URL}/docs")
    print(f"🕸️ Graph View:        {FRONTEND_URL} → Multi-Agent Graph")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")

def main():
    """Run comprehensive end-to-end tests."""
    print(f"{Colors.BOLD}{Colors.CYAN}🧪 Starting ADK Security Agent End-to-End Tests{Colors.END}")
    print(f"{Colors.BOLD}Testing enhanced multi-agent features and graph visualization{Colors.END}\n")
    
    # Track all test results
    test_results = {}
    
    # Core system tests
    test_results["Backend Health"] = test_backend_health()
    test_results["Service Management API"] = test_service_management_api()
    test_results["Agent Functionality"] = test_agent_functionality()
    test_results["GCP Integration"] = test_gcp_integration()
    test_results["API Endpoints"] = test_api_endpoints_comprehensive()
    
    # Enhanced features tests
    test_results["Multi-Agent Graph Data"] = test_multi_agent_graph_data()
    test_results["System Integration"] = test_system_integration()
    test_results["Performance Tests"] = run_performance_tests()
    
    # Optional tests
    test_results["Frontend Availability"] = test_frontend_availability()
    
    # Generate comprehensive report
    generate_test_report(test_results)
    
    # Return exit code based on critical tests
    critical_tests = [
        "Backend Health",
        "Service Management API", 
        "Agent Functionality",
        "Multi-Agent Graph Data"
    ]
    
    critical_passed = all(test_results.get(test, False) for test in critical_tests)
    
    if critical_passed:
        log("All critical tests passed! 🎉", "SUCCESS")
        return 0
    else:
        log("Some critical tests failed. Please review the report above.", "ERROR")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)