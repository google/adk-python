"""
Why & How: ADK Agent Integration Guide
=====================================

Practical guide answering WHY you need service integration,
WHAT problems it solves, and HOW to implement it step-by-step.
"""

import streamlit as st
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from frontend.components.chat_widget import create_chat_widget
    from frontend.services.metrics_service import MetricsService
except ImportError:
    from components.chat_widget import create_chat_widget
    from services.metrics_service import MetricsService

# Page configuration
st.set_page_config(
    page_title="ADK Agent Integration Guide",
    page_icon="🔗",
    layout="wide",
)

def render_page():
    """Renders the practical ADK Agent Integration guide."""

    st.markdown("# 🔗 Why & How: ADK Agent Integration")
    st.markdown("*Real-world scenarios, business justification, and step-by-step implementation*")

    # Create main tabs focused on WHY and HOW
    tabs = st.tabs([
        "🤔 Why Integrate?",
        "🎯 When to Use MCP",
        "⚡ Quick Start (5 min)",
        "🔧 Full Implementation",
        "✅ Testing & Verification"
    ])

    with tabs[0]:
        render_why_integrate()

    with tabs[1]:
        render_when_to_use()

    with tabs[2]:
        render_quick_start()

    with tabs[3]:
        render_full_implementation()

    with tabs[4]:
        render_testing()

    # Add interactive chat widget for questions
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Integration")
    create_chat_widget(context="adk_integration", height=400)

def render_why_integrate():
    """Renders the WHY integrate section."""

    st.markdown("## 🤔 Why Integrate Your ADK Agent?")

    st.markdown("""
    You have a working ADK security agent. Great! But right now it probably works in isolation.

    **The problem**: Other teams keep asking "can you check this resource?" manually.

    **The solution**: Let other services discover and use your agent automatically.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 😤 Common Pain Points
        - Manual security scan requests
        - Data scattered across multiple dashboards
        - Teams can't self-serve security data
        - Repetitive work that could be automated
        """)

    with col2:
        st.markdown("""
        ### ✨ What Integration Gives You
        - **Automation**: Other systems trigger scans automatically
        - **Self-service**: Teams access security data directly
        - **Time savings**: 60-90% reduction in manual requests
        - **Better visibility**: Unified security + compliance view
        """)

    st.info("""
    **Bottom line**: Only integrate if you have multiple teams wanting your security data or repetitive manual requests.

    For simple cases, a direct API call might be enough.
    """)

def render_when_to_use():
    """Renders the WHEN to use MCP section."""

    st.markdown("## 🎯 When to Use MCP vs Alternatives")

    st.markdown("""
    **Not every integration needs MCP.** Here's when to use what approach:
    """)

    # Decision matrix
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✅ Use MCP When:

        **🏢 Enterprise Scenarios:**
        - Multiple internal services need your ADK agent
        - You're building a "marketplace" of AI tools
        - Teams want to discover available agents automatically
        - You have 3+ different agent types to integrate
        - Different teams maintain different agents

        **🔧 Technical Indicators:**
        - Service discovery is important
        - You want standardized tool calling
        - Multiple authentication methods needed
        - Tools need to be dynamically discovered
        - Cross-team agent sharing is required

        **📈 Scale Indicators:**
        - 5+ services in your ecosystem
        - 10+ different tools across services
        - Multiple teams requesting integrations
        """)

    with col2:
        st.markdown("""
        ### ❌ Don't Use MCP When:

        **🚫 Simple Scenarios:**
        - Just connecting 2 services
        - One-off integration project
        - Internal tool for single team
        - Static, unchanging requirements
        - Simple webhook triggers are enough

        **⚡ Use These Instead:**
        - **Direct API calls**: For 1-2 service integrations
        - **Webhooks**: For event-driven automation
        - **Shared database**: For simple data sharing
        - **Message queues**: For async processing
        - **GraphQL**: For unified data access

        **🎯 When in doubt**: Start simple, add MCP later
        """)

    # Decision tree
    st.markdown("""
    ### 🌳 Decision Tree

    ```
    Do you need service discovery?
    ├── No → Use direct API integration
    └── Yes → Do you have 3+ services?
        ├── No → Start with direct APIs, consider MCP later
        └── Yes → Do teams maintain separate agents?
            ├── No → Consider shared API gateway
            └── Yes → MCP is a good fit
    ```
    """)

    # Real examples
    with st.expander("📝 Real-World Examples"):
        st.markdown("""
        **✅ Good MCP Use Cases:**
        - Micron IT has security, compliance, and monitoring agents
        - Each team maintains their own ADK agent
        - Teams want to discover and use each other's tools
        - New agents are added regularly

        **❌ Poor MCP Use Cases:**
        - Just connecting security agent to Slack
        - Single team using their own tools
        - Static integration that never changes
        - Simple "send data to dashboard" requirement
        """)

def render_quick_start():
    """Renders the 5-minute quick start section."""

    st.markdown("## ⚡ Quick Start: Make Your ADK Agent Discoverable (5 minutes)")

    st.markdown("""
    **Goal**: Add a `.well-known/mcp` endpoint to your existing ADK agent so other services can discover it.
    """)

    # Step-by-step implementation
    with st.expander("📋 Prerequisites (30 seconds)", expanded=True):
        st.markdown("""
        - ✅ You have a working ADK agent (like this security agent)
        - ✅ It runs on FastAPI (ADK uses FastAPI by default)
        - ✅ You want other services to discover it

        **That's it!** No additional dependencies needed.
        """)

    with st.expander("🎯 Step 1: Add Discovery Endpoint (2 minutes)", expanded=True):
        st.markdown("""
        Add this code to your existing FastAPI app:

        ```python
        # Add to your main.py or wherever your FastAPI app is defined
        from fastapi import FastAPI

        app = FastAPI()  # Your existing app

        @app.get("/.well-known/mcp")
        async def agent_discovery():
            \"\"\"Make this agent discoverable via MCP\"\"\"
            return {
                "version": "1.0",
                "services": {
                    "gcp-security-agent": {
                        "name": "GCP Security Agent",
                        "description": "Analyze GCP security configurations and compliance",
                        "tools": [
                            "query_security_data",
                            "scan_storage_buckets",
                            "check_iam_policies"
                        ],
                        "endpoint": "/api/v1/agent",
                        "health_check": "/health"
                    }
                },
                "contact": {
                    "team": "Micron IT Security",
                    "docs": "/docs"
                }
            }
        ```

        **Copy-paste ready!** This works with your existing agent.
        """)

    with st.expander("🧪 Step 2: Test It (1 minute)", expanded=True):
        st.markdown("""
        1. Start your agent: `python run_backend.py`
        2. Open: `http://localhost:8000/.well-known/mcp`
        3. You should see JSON with your agent info

        **Test with curl:**
        ```bash
        curl http://localhost:8000/.well-known/mcp
        ```

        **Expected output:**
        ```json
        {
          "version": "1.0",
          "services": {
            "gcp-security-agent": {
              "name": "GCP Security Agent",
              ...
            }
          }
        }
        ```
        """)

    with st.expander("🔗 Step 3: Connect from Another Service (2 minutes)", expanded=True):
        st.markdown("""
        Now other services can discover your agent:

        ```python
        import httpx

        # Discover available agents
        async def discover_security_agent():
            async with httpx.AsyncClient() as client:
                response = await client.get("http://your-agent:8000/.well-known/mcp")
                discovery = response.json()

                # Find the security agent
                security_service = discovery["services"]["gcp-security-agent"]
                print(f"Found: {security_service['name']}")
                print(f"Tools: {security_service['tools']}")

                return security_service

        # Use it
        agent_info = await discover_security_agent()
        ```

        **That's it!** Your agent is now discoverable.
        """)

    st.success("""
    **🎉 Congratulations!** Your ADK agent is now MCP-discoverable.

    **What you've accomplished:**
    - ✅ Made your agent discoverable by other services
    - ✅ Standardized your tool listing
    - ✅ Added proper service metadata
    - ✅ Enabled future automation possibilities

    **Next step**: Scroll down to see how to add more advanced features.
    """)

def render_full_implementation():
    """Renders the full implementation guide."""

    st.markdown("## 🔧 Full Implementation: Advanced Integration")

    st.markdown("""
    **Beyond discovery**: Add authentication, tool calling, and cross-service communication.
    """)

    # Implementation tabs
    impl_tabs = st.tabs([
        "🔐 Add Authentication",
        "📞 Tool Calling API",
        "🔄 Cross-Service Usage",
        "📊 Add Monitoring"
    ])

    with impl_tabs[0]:
        st.markdown("""
        ### 🔐 Add Authentication to Your Agent

        **Why**: Secure access when other services call your tools.

        ```python
        from fastapi import FastAPI, Depends, HTTPException
        from fastapi.security import HTTPBearer
        import jwt

        app = FastAPI()
        security = HTTPBearer()

        # Authentication
        async def verify_token(token: str = Depends(security)):
            try:
                # Verify JWT token (use your company's auth)
                payload = jwt.decode(token.credentials, "your-secret", algorithms=["HS256"])
                return payload
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")

        # Protected tool endpoint
        @app.post("/api/v1/tools/query_security_data")
        async def call_security_tool(
            request: dict,
            user: dict = Depends(verify_token)
        ):
            # Your existing tool logic
            result = await query_security_data(
                query_type=request["query_type"],
                filters=request.get("filters", {})
            )
            return {"success": True, "data": result}

        # Update discovery to include auth info
        @app.get("/.well-known/mcp")
        async def agent_discovery():
            return {
                "version": "1.0",
                "services": {
                    "gcp-security-agent": {
                        "name": "GCP Security Agent",
                        "tools": ["query_security_data"],
                        "endpoint": "/api/v1/tools",
                        "authentication": {
                            "type": "bearer",
                            "required": True
                        }
                    }
                }
            }
        ```
        """)

    with impl_tabs[1]:
        st.markdown("""
        ### 📞 Standardized Tool Calling API

        **Why**: Let other services call your ADK tools programmatically.

        ```python
        from typing import Dict, Any
        from pydantic import BaseModel

        class ToolRequest(BaseModel):
            tool_name: str
            parameters: Dict[str, Any]

        class ToolResponse(BaseModel):
            success: bool
            data: Any = None
            error: str = None

        @app.post("/api/v1/tools/call", response_model=ToolResponse)
        async def call_tool(
            request: ToolRequest,
            user: dict = Depends(verify_token)
        ):
            \"\"\"Generic tool calling endpoint\"\"\"

            # Map of available tools
            available_tools = {
                "query_security_data": query_security_data,
                "scan_storage_buckets": scan_storage_buckets,
                "check_iam_policies": check_iam_policies
            }

            if request.tool_name not in available_tools:
                return ToolResponse(
                    success=False,
                    error=f"Tool '{request.tool_name}' not found"
                )

            try:
                tool_function = available_tools[request.tool_name]
                result = await tool_function(**request.parameters)

                return ToolResponse(success=True, data=result)

            except Exception as e:
                return ToolResponse(
                    success=False,
                    error=f"Tool execution failed: {str(e)}"
                )

        # List available tools
        @app.get("/api/v1/tools")
        async def list_tools(user: dict = Depends(verify_token)):
            return {
                "tools": [
                    {
                        "name": "query_security_data",
                        "description": "Query GCP security data",
                        "parameters": {
                            "query_type": {"type": "string", "required": True},
                            "filters": {"type": "object", "required": False}
                        }
                    },
                    # Add other tools...
                ]
            }
        ```
        """)

    with impl_tabs[2]:
        st.markdown("""
        ### 🔄 Using Your Agent from Other Services

        **Example**: How a compliance service would use your security agent.

        ```python
        # compliance_service.py
        import httpx
        from typing import Dict, Any

        class SecurityAgentClient:
            def __init__(self, base_url: str, auth_token: str):
                self.base_url = base_url
                self.auth_token = auth_token
                self.headers = {"Authorization": f"Bearer {auth_token}"}

            async def discover_capabilities(self):
                \"\"\"Discover what the security agent can do\"\"\"
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/.well-known/mcp")
                    return response.json()

            async def call_tool(self, tool_name: str, parameters: Dict[str, Any]):
                \"\"\"Call a security agent tool\"\"\"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/v1/tools/call",
                        json={
                            "tool_name": tool_name,
                            "parameters": parameters
                        },
                        headers=self.headers
                    )
                    return response.json()

            async def get_storage_security(self, project_id: str):
                \"\"\"Get security status of storage buckets\"\"\"
                return await self.call_tool(
                    "query_security_data",
                    {
                        "query_type": "storage_buckets",
                        "filters": {"project_id": project_id}
                    }
                )

        # Usage in compliance service
        class ComplianceChecker:
            def __init__(self):
                self.security_client = SecurityAgentClient(
                    base_url="http://security-agent:8000",
                    auth_token="your-service-token"
                )

            async def full_compliance_audit(self, project_id: str):
                \"\"\"Combine compliance checks with security data\"\"\"

                # Get security data from security agent
                security_data = await self.security_client.get_storage_security(project_id)

                # Run local compliance checks
                compliance_data = await self.run_compliance_checks(project_id)

                # Combine results
                return {
                    "project_id": project_id,
                    "security_status": security_data,
                    "compliance_status": compliance_data,
                    "overall_score": self.calculate_score(security_data, compliance_data)
                }
        ```
        """)

    with impl_tabs[3]:
        st.markdown("""
        ### 📊 Add Monitoring and Metrics

        **Why**: Track usage and performance of your integrated agent.

        ```python
        from prometheus_client import Counter, Histogram, make_asgi_app
        import time

        # Metrics
        tool_calls = Counter('agent_tool_calls_total', 'Total tool calls', ['tool_name', 'service'])
        tool_duration = Histogram('agent_tool_duration_seconds', 'Tool execution time', ['tool_name'])
        discovery_requests = Counter('agent_discovery_requests_total', 'Discovery endpoint requests')

        @app.get("/.well-known/mcp")
        async def agent_discovery():
            discovery_requests.inc()  # Track discovery requests
            # ... existing discovery code

        @app.post("/api/v1/tools/call")
        async def call_tool(request: ToolRequest, user: dict = Depends(verify_token)):
            # Track metrics
            start_time = time.time()
            tool_calls.labels(tool_name=request.tool_name, service=user.get('service', 'unknown')).inc()

            try:
                # ... existing tool calling code
                result = await tool_function(**request.parameters)

                # Record success duration
                tool_duration.labels(tool_name=request.tool_name).observe(time.time() - start_time)

                return ToolResponse(success=True, data=result)

            except Exception as e:
                # Record error duration
                tool_duration.labels(tool_name=request.tool_name).observe(time.time() - start_time)
                raise

        # Add metrics endpoint
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

        # Health check with integration status
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "version": "1.0.0",
                "integrations": {
                    "mcp_discovery": "enabled",
                    "tool_calling": "enabled",
                    "authentication": "enabled",
                    "metrics": "enabled"
                },
                "stats": {
                    "uptime_seconds": time.time() - start_time,
                    "total_tool_calls": sum(tool_calls._value.values())
                }
            }
        ```
        """)

def render_testing():
    """Renders the testing and verification section."""

    st.markdown("## ✅ Testing & Verification")

    st.markdown("""
    **Verify your integration works** before deploying to production.
    """)

    # Testing tabs
    test_tabs = st.tabs([
        "🧪 Manual Testing",
        "⚡ Automated Tests",
        "🔍 Integration Testing",
        "📈 Performance Testing"
    ])

    with test_tabs[0]:
        st.markdown("""
        ### 🧪 Manual Testing Checklist

        **Test your endpoints manually first:**

        ```bash
        # 1. Test discovery endpoint
        curl http://localhost:8000/.well-known/mcp

        # Expected: JSON with service info
        # ✅ Should return service metadata
        # ❌ Should not return 404 or errors

        # 2. Test authentication (if enabled)
        curl -H "Authorization: Bearer fake-token" \\
             http://localhost:8000/api/v1/tools

        # Expected: 401 Unauthorized
        # ✅ Should reject invalid tokens
        # ❌ Should not allow unauthenticated access

        # 3. Test tool calling
        curl -X POST http://localhost:8000/api/v1/tools/call \\
             -H "Content-Type: application/json" \\
             -H "Authorization: Bearer valid-token" \\
             -d '{
               "tool_name": "query_security_data",
               "parameters": {"query_type": "storage_buckets"}
             }'

        # Expected: Tool execution result
        # ✅ Should return success: true with data
        # ❌ Should not return errors for valid requests

        # 4. Test invalid tool name
        curl -X POST http://localhost:8000/api/v1/tools/call \\
             -H "Content-Type: application/json" \\
             -H "Authorization: Bearer valid-token" \\
             -d '{
               "tool_name": "nonexistent_tool",
               "parameters": {}
             }'

        # Expected: Error response
        # ✅ Should return success: false with error message
        # ❌ Should not crash or return 500 error
        ```

        **Visual Testing:**
        - 🌐 Visit `http://localhost:8000/.well-known/mcp` in browser
        - 📋 Visit `http://localhost:8000/docs` to see Swagger UI
        - 📊 Visit `http://localhost:8000/metrics` to see Prometheus metrics
        """)

    with test_tabs[1]:
        st.markdown("""
        ### ⚡ Automated Test Suite

        **Create tests to run in CI/CD:**

        ```python
        # test_mcp_integration.py
        import pytest
        import httpx
        from fastapi.testclient import TestClient
        from main import app  # Your FastAPI app

        client = TestClient(app)

        class TestMCPDiscovery:
            def test_discovery_endpoint_exists(self):
                \"\"\"Test that discovery endpoint is available\"\"\"
                response = client.get("/.well-known/mcp")
                assert response.status_code == 200

            def test_discovery_has_required_fields(self):
                \"\"\"Test discovery response structure\"\"\"
                response = client.get("/.well-known/mcp")
                data = response.json()

                assert "version" in data
                assert "services" in data
                assert len(data["services"]) > 0

            def test_service_metadata_complete(self):
                \"\"\"Test each service has required metadata\"\"\"
                response = client.get("/.well-known/mcp")
                services = response.json()["services"]

                for service_name, service_info in services.items():
                    assert "name" in service_info
                    assert "tools" in service_info
                    assert isinstance(service_info["tools"], list)
                    assert len(service_info["tools"]) > 0

        class TestToolCalling:
            def test_tool_list_endpoint(self):
                \"\"\"Test tool listing endpoint\"\"\"
                response = client.get("/api/v1/tools")
                assert response.status_code == 200

                tools = response.json()["tools"]
                assert len(tools) > 0

                # Check tool structure
                for tool in tools:
                    assert "name" in tool
                    assert "description" in tool
                    assert "parameters" in tool

            def test_valid_tool_call(self):
                \"\"\"Test calling a valid tool\"\"\"
                response = client.post("/api/v1/tools/call", json={
                    "tool_name": "query_security_data",
                    "parameters": {"query_type": "storage_buckets"}
                })

                assert response.status_code == 200
                result = response.json()
                assert result["success"] is True
                assert "data" in result

            def test_invalid_tool_call(self):
                \"\"\"Test calling non-existent tool\"\"\"
                response = client.post("/api/v1/tools/call", json={
                    "tool_name": "nonexistent_tool",
                    "parameters": {}
                })

                assert response.status_code == 200
                result = response.json()
                assert result["success"] is False
                assert "error" in result

        class TestAuthentication:
            def test_protected_endpoint_without_auth(self):
                \"\"\"Test that protected endpoints require auth\"\"\"
                response = client.post("/api/v1/tools/call", json={
                    "tool_name": "query_security_data",
                    "parameters": {}
                })
                assert response.status_code == 401

            def test_protected_endpoint_with_invalid_auth(self):
                \"\"\"Test invalid token is rejected\"\"\"
                response = client.post(
                    "/api/v1/tools/call",
                    json={"tool_name": "query_security_data", "parameters": {}},
                    headers={"Authorization": "Bearer invalid-token"}
                )
                assert response.status_code == 401

        # Run tests
        if __name__ == "__main__":
            pytest.main([__file__, "-v"])
        ```

        **Run the tests:**
        ```bash
        pip install pytest httpx
        pytest test_mcp_integration.py -v
        ```
        """)

    with test_tabs[2]:
        st.markdown("""
        ### 🔍 Integration Testing

        **Test the full integration flow:**

        ```python
        # test_integration_flow.py
        import asyncio
        import httpx

        async def test_full_integration_flow():
            \"\"\"Test complete flow from discovery to tool execution\"\"\"

            base_url = "http://localhost:8000"

            # Step 1: Discover agent capabilities
            async with httpx.AsyncClient() as client:
                discovery_response = await client.get(f"{base_url}/.well-known/mcp")
                assert discovery_response.status_code == 200

                discovery_data = discovery_response.json()
                services = discovery_data["services"]

                print(f"✅ Discovered {len(services)} services")

                # Step 2: Get available tools
                for service_name, service_info in services.items():
                    print(f"Service: {service_name}")
                    print(f"Tools: {service_info['tools']}")

                    # Step 3: Call each tool
                    for tool_name in service_info["tools"]:
                        try:
                            tool_response = await client.post(
                                f"{base_url}/api/v1/tools/call",
                                json={
                                    "tool_name": tool_name,
                                    "parameters": {"query_type": "storage_buckets"}
                                },
                                headers={"Authorization": "Bearer test-token"}
                            )

                            if tool_response.status_code == 200:
                                result = tool_response.json()
                                if result["success"]:
                                    print(f"✅ Tool {tool_name} works")
                                else:
                                    print(f"❌ Tool {tool_name} failed: {result.get('error')}")
                            else:
                                print(f"❌ Tool {tool_name} HTTP error: {tool_response.status_code}")

                        except Exception as e:
                            print(f"❌ Tool {tool_name} exception: {e}")

        # Run integration test
        if __name__ == "__main__":
            asyncio.run(test_full_integration_flow())
        ```
        """)

    with test_tabs[3]:
        st.markdown("""
        ### 📈 Performance Testing

        **Test performance under load:**

        ```python
        # test_performance.py
        import asyncio
        import time
        import httpx
        from concurrent.futures import ThreadPoolExecutor

        async def performance_test():
            \"\"\"Test agent performance under load\"\"\"

            base_url = "http://localhost:8000"
            num_requests = 100
            concurrent_requests = 10

            async def single_request():
                async with httpx.AsyncClient() as client:
                    start_time = time.time()
                    response = await client.post(
                        f"{base_url}/api/v1/tools/call",
                        json={
                            "tool_name": "query_security_data",
                            "parameters": {"query_type": "storage_buckets"}
                        },
                        headers={"Authorization": "Bearer test-token"}
                    )
                    duration = time.time() - start_time
                    return {
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.json().get("success", False) if response.status_code == 200 else False
                    }

            # Run concurrent requests
            start_time = time.time()

            tasks = []
            for _ in range(num_requests):
                task = asyncio.create_task(single_request())
                tasks.append(task)

                # Limit concurrency
                if len(tasks) >= concurrent_requests:
                    results = await asyncio.gather(*tasks)
                    tasks = []

            # Wait for remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks)

            total_time = time.time() - start_time

            # Analyze results
            successful_requests = sum(1 for r in results if r["success"])
            failed_requests = len(results) - successful_requests
            avg_duration = sum(r["duration"] for r in results) / len(results)
            max_duration = max(r["duration"] for r in results)
            min_duration = min(r["duration"] for r in results)

            print(f"Performance Test Results:")
            print(f"Total requests: {num_requests}")
            print(f"Successful: {successful_requests}")
            print(f"Failed: {failed_requests}")
            print(f"Success rate: {successful_requests/num_requests*100:.1f}%")
            print(f"Total time: {total_time:.2f}s")
            print(f"Requests per second: {num_requests/total_time:.1f}")
            print(f"Average response time: {avg_duration:.3f}s")
            print(f"Min response time: {min_duration:.3f}s")
            print(f"Max response time: {max_duration:.3f}s")

            # Performance assertions
            assert successful_requests >= num_requests * 0.95, "Success rate should be >= 95%"
            assert avg_duration < 2.0, "Average response time should be < 2 seconds"
            assert num_requests/total_time > 10, "Should handle > 10 requests per second"

        if __name__ == "__main__":
            asyncio.run(performance_test())
        ```

        **Load testing with external tools:**
        ```bash
        # Using Apache Bench
        ab -n 1000 -c 10 -H "Authorization: Bearer test-token" \\
           -p request.json -T application/json \\
           http://localhost:8000/api/v1/tools/call

        # Using wrk
        wrk -t12 -c400 -d30s \\
            -H "Authorization: Bearer test-token" \\
            --script=post.lua \\
            http://localhost:8000/api/v1/tools/call
        ```
        """)

    st.markdown("""
    ### 🎯 Testing Checklist

    **Before deploying to production:**

    ✅ **Functional Tests**
    - [ ] Discovery endpoint returns valid JSON
    - [ ] All tools are listed correctly
    - [ ] Tool calls work with valid parameters
    - [ ] Error handling works for invalid requests
    - [ ] Authentication rejects invalid tokens

    ✅ **Integration Tests**
    - [ ] End-to-end flow works (discovery → tool call)
    - [ ] Cross-service communication works
    - [ ] Error propagation works correctly
    - [ ] Timeouts are handled properly

    ✅ **Performance Tests**
    - [ ] Response time < 2 seconds under normal load
    - [ ] Success rate > 95% under load
    - [ ] Can handle expected concurrent requests
    - [ ] Memory usage stays within limits
    - [ ] No memory leaks during extended use

    ✅ **Security Tests**
    - [ ] Authentication is enforced
    - [ ] Invalid tokens are rejected
    - [ ] Rate limiting works (if implemented)
    - [ ] No sensitive data in error messages
    - [ ] HTTPS is used in production
    """)

# Render the page
render_page()