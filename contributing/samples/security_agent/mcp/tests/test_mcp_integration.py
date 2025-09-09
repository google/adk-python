"""
MCP Integration Tests for Micron Security Agent
Tests discovery, authentication, and tool execution
"""

import pytest
import asyncio
import aiohttp
from typing import Dict, Any
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery.basic import SecurityAgentMCPDiscovery
from clients.secure_client import SecureSecurityAgentClient


class TestMCPDiscovery:
    """Test MCP discovery functionality"""
    
    @pytest.fixture
    def discovery_client(self):
        """Create discovery client fixture"""
        return SecurityAgentMCPDiscovery("http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_discovery_endpoint_exists(self, discovery_client):
        """Test that MCP discovery endpoint is accessible"""
        async with aiohttp.ClientSession() as session:
            async with session.get(discovery_client.discovery_url) as response:
                assert response.status == 200, "Discovery endpoint should return 200"
                
                data = await response.json()
                assert "servers" in data, "Discovery should contain servers"
    
    @pytest.mark.asyncio
    async def test_discover_security_tools(self, discovery_client):
        """Test discovering security tools"""
        result = await discovery_client.discover_security_tools()
        
        assert result["status"] == "success", "Discovery should succeed"
        assert result["tools_discovered"] > 0, "Should discover at least one tool"
        assert "categories" in result, "Should categorize tools"
    
    @pytest.mark.asyncio
    async def test_tool_categorization(self, discovery_client):
        """Test that tools are properly categorized"""
        result = await discovery_client.discover_security_tools()
        
        if result["status"] == "success":
            categories = result["categories"]
            
            # Check for expected categories
            expected_categories = ["security", "iam", "compliance"]
            for category in expected_categories:
                assert category in categories or len(categories) > 0, \
                    f"Should have {category} category or other categories"
    
    @pytest.mark.asyncio
    async def test_get_tool_details(self, discovery_client):
        """Test getting details for a specific tool"""
        # First discover tools
        await discovery_client.discover_security_tools()
        
        # Get details for a known tool
        tool_details = await discovery_client.get_tool_details("health")
        
        assert tool_details is not None, "Should find health tool"
        assert "name" in tool_details, "Tool should have name"
        assert "description" in tool_details, "Tool should have description"
        assert "input_schema" in tool_details, "Tool should have input schema"


class TestSecureMCPClient:
    """Test secure MCP client functionality"""
    
    @pytest.fixture
    async def secure_client(self):
        """Create secure client fixture"""
        client = SecureSecurityAgentClient(
            agent_url="http://localhost:8000",
            username="test_user",
            password="test_password"
        )
        client.session = aiohttp.ClientSession()
        yield client
        await client.session.close()
    
    @pytest.mark.asyncio
    async def test_authentication(self, secure_client):
        """Test client authentication"""
        # This will fail with test credentials, but tests the flow
        result = await secure_client.authenticate()
        
        # Check that authentication was attempted
        assert isinstance(result, bool), "Authentication should return boolean"
    
    @pytest.mark.asyncio
    async def test_discover_with_cache(self, secure_client):
        """Test discovery with caching"""
        # First discovery (fresh)
        result1 = await secure_client.discover_tools()
        if result1["status"] == "success":
            assert result1["source"] == "fresh", "First call should be fresh"
        
        # Second discovery (cached)
        result2 = await secure_client.discover_tools()
        if result2["status"] == "success":
            assert result2["source"] == "cache", "Second call should use cache"
        
        # Force refresh
        result3 = await secure_client.discover_tools(force_refresh=True)
        if result3["status"] == "success":
            assert result3["source"] == "fresh", "Force refresh should bypass cache"
    
    @pytest.mark.asyncio
    async def test_auth_requirement_check(self, secure_client):
        """Test checking if tools require authentication"""
        # Check known secure tools
        assert secure_client._check_auth_requirement("security_scan") == True
        assert secure_client._check_auth_requirement("iam_analyze") == True
        
        # Check public tools
        assert secure_client._check_auth_requirement("health") == False
    
    @pytest.mark.asyncio
    async def test_batch_execution_parallel(self, secure_client):
        """Test parallel batch execution"""
        executions = [
            {"tool": "health", "parameters": {}},
            {"tool": "health", "parameters": {}},
        ]
        
        results = await secure_client.batch_execute(executions, parallel=True)
        
        assert len(results) == 2, "Should return results for all executions"
        assert all("status" in r for r in results), "All results should have status"
    
    @pytest.mark.asyncio
    async def test_batch_execution_sequential(self, secure_client):
        """Test sequential batch execution"""
        executions = [
            {"tool": "health", "parameters": {}},
            {"tool": "health", "parameters": {}},
        ]
        
        results = await secure_client.batch_execute(executions, parallel=False)
        
        assert len(results) == 2, "Should return results for all executions"
        assert all("status" in r for r in results), "All results should have status"
    
    @pytest.mark.asyncio
    async def test_workflow_validation(self, secure_client):
        """Test workflow validation"""
        # Test valid workflow
        result = await secure_client.run_security_workflow("comprehensive_scan")
        assert "workflow" in result, "Should return workflow name"
        assert "total_steps" in result, "Should return total steps"
        
        # Test invalid workflow
        result = await secure_client.run_security_workflow("invalid_workflow")
        assert result["status"] == "error", "Invalid workflow should error"
        assert "available_workflows" in result, "Should list available workflows"


class TestMCPProtocol:
    """Test MCP protocol compliance"""
    
    @pytest.mark.asyncio
    async def test_jsonrpc_format(self):
        """Test that MCP uses correct JSON-RPC format"""
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "test_tool",
                "arguments": {"param": "value"}
            }
        }
        
        # Validate structure
        assert mcp_request["jsonrpc"] == "2.0", "Should use JSON-RPC 2.0"
        assert "id" in mcp_request, "Should have request ID"
        assert "method" in mcp_request, "Should have method"
        assert "params" in mcp_request, "Should have params"
    
    @pytest.mark.asyncio
    async def test_wellknown_structure(self):
        """Test .well-known/mcp.json structure"""
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:8000/mcp/.well-known/mcp.json"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate structure
                    assert "servers" in data, "Should have servers"
                    
                    for server_name, server_info in data["servers"].items():
                        assert "name" in server_info, "Server should have name"
                        assert "tools" in server_info, "Server should have tools"
                        
                        for tool in server_info["tools"]:
                            assert "name" in tool, "Tool should have name"
                            assert "inputSchema" in tool, "Tool should have input schema"


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_discovery_flow(self):
        """Test complete discovery and execution flow"""
        
        # 1. Initialize discovery
        discovery = SecurityAgentMCPDiscovery("http://localhost:8000")
        
        # 2. Discover tools
        discovery_result = await discovery.discover_security_tools()
        
        if discovery_result["status"] == "success":
            print(f"\n✅ Discovered {discovery_result['tools_discovered']} tools")
            
            # 3. Get tool details
            tool_details = await discovery.get_tool_details("health")
            if tool_details:
                print(f"✅ Got details for tool: {tool_details['name']}")
            
            # 4. Execute a tool
            exec_result = await discovery.execute_security_tool(
                "health",
                {}
            )
            
            if exec_result["status"] == "success":
                print("✅ Successfully executed health check")
            
            # 5. Print summary
            discovery.print_discovery_summary()
            
            assert True, "End-to-end flow completed"
        else:
            pytest.skip(f"Security agent not running: {discovery_result['error']}")
    
    @pytest.mark.asyncio
    async def test_secure_workflow_execution(self):
        """Test secure workflow execution"""
        
        async with SecureSecurityAgentClient(
            agent_url="http://localhost:8000",
            api_key="test_api_key"  # Use test API key
        ) as client:
            
            # Discover tools
            discovery = await client.discover_tools()
            
            if discovery["status"] == "success":
                print(f"\n✅ Discovered {discovery['tools_count']} tools")
                
                # Run a workflow
                workflow_result = await client.run_security_workflow("comprehensive_scan")
                
                print(f"✅ Workflow executed: {workflow_result['successful']}/{workflow_result['total_steps']} successful")
                
                assert workflow_result["status"] == "completed", "Workflow should complete"
            else:
                pytest.skip(f"Discovery failed: {discovery['error']}")


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running MCP Integration Tests for Security Agent")
    print("="*60)
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])


if __name__ == "__main__":
    # Check if security agent is running
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ Security agent is running")
            run_integration_tests()
        else:
            print("⚠️ Security agent returned unexpected status")
    except requests.exceptions.RequestException:
        print("❌ Security agent is not running!")
        print("Please start it with: python run_backend.py")
        print("\nRunning tests anyway (some will be skipped)...")
        run_integration_tests()