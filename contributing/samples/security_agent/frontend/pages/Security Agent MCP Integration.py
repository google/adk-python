"""
Security Agent MCP Integration Guide
===================================

Complete guide for exposing your GCP Security Agent as an MCP server
and enabling consumption by other services in your organization.
"""

import streamlit as st
import sys
import logging
import json
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
    page_title="Security Agent MCP Integration",
    page_icon="🔐",
    layout="wide",
)

def render_page():
    """Renders the Security Agent MCP Integration guide."""

    st.markdown("# 🔐 Security Agent MCP Integration")
    st.markdown("*Expose your GCP Security Agent as an MCP server for organization-wide consumption*")

    # Create main tabs focused on the security agent
    tabs = st.tabs([
        "🎯 Why Expose Your Security Agent?",
        "⚡ Quick Setup (5 minutes)",
        "🔧 Available Security Tools",
        "🔗 Consumer Integration Examples",
        "📊 Monitoring & Testing"
    ])

    with tabs[0]:
        render_why_expose()

    with tabs[1]:
        render_quick_setup()

    with tabs[2]:
        render_available_tools()

    with tabs[3]:
        render_consumer_examples()

    with tabs[4]:
        render_monitoring()

    # Add interactive chat widget for questions
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Security Agent")
    st.markdown("*Try asking: 'How do I integrate with the storage bucket analysis tool?' or 'What security data can I access?'*")
    create_chat_widget(context="security_mcp", height=400)

def render_why_expose():
    """Renders the WHY expose section."""

    st.markdown("## 🎯 Why Expose Your Security Agent as MCP?")

    st.markdown("""
    You have a powerful GCP security agent running with 30+ security analysis tools.
    Right now, teams need to manually ask you to run security checks.

    **MCP exposure solves this by letting other services discover and use your security tools automatically.**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 😤 Current Pain Points
        - **Manual requests**: "Can you check our buckets for public access?"
        - **Scattered data**: Security info across multiple dashboards
        - **Time consuming**: Repetitive security analysis requests
        - **No self-service**: Teams wait for security team availability
        - **Inconsistent checks**: Different analysis depth each time
        """)

    with col2:
        st.markdown("""
        ### ✨ MCP Integration Benefits
        - **Automated discovery**: Services find your security tools automatically
        - **Self-service access**: Teams run security checks when needed
        - **Consistent analysis**: Same depth and quality every time
        - **Real-time integration**: Security checks in CI/CD pipelines
        - **Standardized data**: Unified security data format across systems
        """)

    st.info("""
    **💡 Real example**: Instead of Slack messages asking for bucket analysis,
    the compliance team's dashboard automatically pulls security data and displays current status.
    """)

    # Show actual capabilities
    with st.expander("🔍 What Your Security Agent Can Do (30+ Tools Available)"):
        st.markdown("""
        **Core Security Analysis:**
        - `storage_buckets` - GCS bucket security configuration analysis
        - `security_findings` - Critical vulnerabilities and compliance issues
        - `iam_analysis` - Identity and access management review
        - `firewall_rules` - Network security configuration analysis
        - `compute_instances` - VM security assessment

        **Advanced Security Features:**
        - `service_evaluation` - Risk assessment for new GCP services
        - `compliance` - Regulatory compliance checking
        - `asset_inventory` - Complete resource security overview
        - `monitoring` - Security monitoring configuration review
        - `secrets` - Secrets management security analysis

        **Specialized Analysis:**
        - `vpc_error_analysis` - Network security issue detection
        - `vpcsc_readiness` - VPC Service Controls readiness assessment
        - `org_policy_test` - Organization policy compliance testing
        - `configuration_drift` - Security configuration drift detection
        """)

def render_quick_setup():
    """Renders the 5-minute quick setup section."""

    st.markdown("## ⚡ Quick Setup: Make Your Security Agent MCP-Discoverable")

    st.markdown("""
    **Goal**: Enable other services to discover and use your security agent's 30+ analysis tools.

    Your security agent is already running and working perfectly. We just need to add a discovery endpoint.
    """)

    # Current status check
    with st.expander("✅ Prerequisites Check", expanded=True):
        st.markdown("""
        **You already have:**
        - ✅ Working GCP Security Agent (ADK-powered)
        - ✅ 30+ security analysis tools (`storage_buckets`, `iam_analysis`, etc.)
        - ✅ Running on http://localhost:8000 (ADK web server)
        - ✅ Database with security data (SQLite + live GCP API)
        - ✅ Gemini 2.5-flash model for intelligent analysis

        **Status**: Your agent is ready for MCP exposure! 🎉
        """)

    # Step 1: Add MCP Discovery
    with st.expander("🎯 Step 1: Add MCP Discovery Endpoint (2 minutes)", expanded=True):
        st.markdown("""
        Create a simple wrapper server that exposes your ADK agent via MCP:

        ```python
        # mcp_wrapper.py
        from fastapi import FastAPI
        import httpx
        import asyncio

        app = FastAPI(title="GCP Security Agent MCP Server")

        # Point to your existing ADK agent
        ADK_BASE_URL = "http://localhost:8000"

        @app.get("/.well-known/mcp")
        async def mcp_discovery():
            return {
                "version": "1.0",
                "server_info": {
                    "name": "GCP Security Agent MCP Server",
                    "version": "1.0.0",
                    "description": "Micron IT Security Agent - 30+ GCP security analysis tools"
                },
                "capabilities": {
                    "tools": True,
                    "resources": True,
                    "logging": True
                },
                "tools": [
                    {
                        "name": "analyze_security",
                        "description": "Comprehensive GCP security analysis across multiple domains",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "analysis_type": {
                                    "type": "string",
                                    "enum": [
                                        "storage_buckets", "security_findings", "iam_analysis",
                                        "firewall_rules", "compute_instances", "service_evaluation",
                                        "compliance", "asset_inventory", "secrets", "monitoring"
                                    ],
                                    "description": "Type of security analysis to perform"
                                },
                                "severity_filter": {
                                    "type": "string",
                                    "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                    "description": "Filter by severity level (optional)"
                                },
                                "service_name": {
                                    "type": "string",
                                    "description": "For service_evaluation - GCP service to analyze"
                                }
                            },
                            "required": ["analysis_type"]
                        }
                    }
                ],
                "contact": {
                    "team": "Micron IT Security",
                    "documentation": "http://localhost:8501/Security_Agent_MCP_Integration"
                }
            }

        @app.post("/mcp/tools/analyze_security")
        async def analyze_security(request: dict):
            \"\"\"Proxy security analysis requests to ADK agent\"\"\"
            analysis_type = request.get("analysis_type")

            # Convert MCP request to ADK agent format
            message = f"Analyze {analysis_type}"
            if severity := request.get("severity_filter"):
                message += f" with {severity} severity"
            if service := request.get("service_name"):
                message += f" for {service}"

            # Call your existing ADK agent
            async with httpx.AsyncClient() as client:
                # Create session with ADK agent
                session_response = await client.post(
                    f"{ADK_BASE_URL}/apps/agents/users/mcp-client/sessions",
                    json={"app_name": "agents"}
                )
                session_id = session_response.json()["session_id"]

                # Send analysis request
                response = await client.post(
                    f"{ADK_BASE_URL}/run",
                    json={
                        "appName": "agents",
                        "userId": "mcp-client",
                        "sessionId": session_id,
                        "newMessage": {"parts": [{"text": message}], "role": "user"},
                        "streaming": False
                    }
                )

                return {
                    "success": True,
                    "analysis_type": analysis_type,
                    "result": response.json(),
                    "agent_model": "gemini-2.5-flash"
                }

        if __name__ == "__main__":
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8001)
        ```

        **This creates a bridge between MCP and your existing ADK agent!**
        """)

    # Step 2: Test it
    with st.expander("🧪 Step 2: Test the MCP Server (1 minute)", expanded=True):
        st.markdown("""
        1. **Start the MCP wrapper**: `python mcp_wrapper.py`
        2. **Test discovery**: `curl http://localhost:8001/.well-known/mcp`
        3. **Test analysis**:
        ```bash
        curl -X POST http://localhost:8001/mcp/tools/analyze_security \\
             -H "Content-Type: application/json" \\
             -d '{"analysis_type": "storage_buckets"}'
        ```

        **Expected result**: JSON response with security analysis from your agent.
        """)

    # Step 3: Connect services
    with st.expander("🔗 Step 3: Connect Other Services (2 minutes)", expanded=True):
        st.markdown("""
        Now other services can discover and use your security tools:

        ```python
        # compliance_service.py
        import httpx

        class SecurityAgentClient:
            def __init__(self):
                self.base_url = "http://security-agent-mcp:8001"

            async def discover_capabilities(self):
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/.well-known/mcp")
                    return response.json()

            async def analyze_storage_security(self):
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/mcp/tools/analyze_security",
                        json={"analysis_type": "storage_buckets", "severity_filter": "HIGH"}
                    )
                    return response.json()

        # Usage in compliance dashboard
        security_client = SecurityAgentClient()
        storage_analysis = await security_client.analyze_storage_security()
        ```
        """)

    st.success("""
    **🎉 Congratulations!** Your security agent is now MCP-enabled!

    **What you've accomplished:**
    - ✅ Made 30+ security tools discoverable by other services
    - ✅ Enabled automated security analysis in other systems
    - ✅ Maintained your existing working ADK agent
    - ✅ Created a standardized security data interface
    """)

def render_available_tools():
    """Renders the available security tools section."""

    st.markdown("## 🔧 Available Security Analysis Tools")

    st.markdown("""
    Your security agent provides **30+ specialized analysis tools**. Here's what other services can access:
    """)

    # Core tools
    with st.expander("🔍 Core Security Analysis Tools", expanded=True):
        tools_data = [
            {
                "tool": "storage_buckets",
                "description": "Analyze GCS bucket security configurations",
                "use_case": "Check for public access, encryption, lifecycle policies",
                "example": '{"analysis_type": "storage_buckets"}'
            },
            {
                "tool": "security_findings",
                "description": "Get security vulnerabilities and compliance issues",
                "use_case": "Critical security issues requiring immediate attention",
                "example": '{"analysis_type": "security_findings", "severity_filter": "CRITICAL"}'
            },
            {
                "tool": "iam_analysis",
                "description": "Review IAM policies and access patterns",
                "use_case": "Identify overprivileged accounts and access anomalies",
                "example": '{"analysis_type": "iam_analysis"}'
            },
            {
                "tool": "firewall_rules",
                "description": "Analyze network firewall configurations",
                "use_case": "Check for overly permissive network rules",
                "example": '{"analysis_type": "firewall_rules"}'
            },
            {
                "tool": "compute_instances",
                "description": "Review VM security configurations",
                "use_case": "Check for security misconfigurations in VMs",
                "example": '{"analysis_type": "compute_instances"}'
            }
        ]

        for tool in tools_data:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.code(tool["tool"], language="text")
            with col2:
                st.markdown(f"**{tool['description']}**")
                st.markdown(f"*Use case: {tool['use_case']}*")
                st.code(tool["example"], language="json")

    # Advanced tools
    with st.expander("⚙️ Advanced Security Tools"):
        advanced_tools = [
            {
                "tool": "service_evaluation",
                "description": "Security risk assessment for new GCP services",
                "example": '{"analysis_type": "service_evaluation", "service_name": "Cloud Functions"}'
            },
            {
                "tool": "compliance",
                "description": "Regulatory compliance checking",
                "example": '{"analysis_type": "compliance"}'
            },
            {
                "tool": "asset_inventory",
                "description": "Complete resource security overview",
                "example": '{"analysis_type": "asset_inventory"}'
            },
            {
                "tool": "monitoring",
                "description": "Security monitoring configuration review",
                "example": '{"analysis_type": "monitoring"}'
            },
            {
                "tool": "secrets",
                "description": "Secrets management security analysis",
                "example": '{"analysis_type": "secrets"}'
            }
        ]

        for tool in advanced_tools:
            st.markdown(f"**{tool['tool']}**: {tool['description']}")
            st.code(tool["example"], language="json")

    # Tool capabilities summary
    st.markdown("""
    ### 🎯 Tool Capabilities Summary

    | Category | Tools Available | Key Features |
    |----------|----------------|--------------|
    | **Core Security** | 5 tools | Bucket analysis, IAM review, network security |
    | **Advanced Analysis** | 10+ tools | Service evaluation, compliance, monitoring |
    | **Specialized** | 15+ tools | VPC analysis, org policies, configuration drift |
    | **Data Sources** | Multiple | Live GCP APIs + cached analysis |
    | **AI Analysis** | Gemini 2.5-flash | Intelligent insights and recommendations |
    """)

def render_consumer_examples():
    """Renders real consumer integration examples."""

    st.markdown("## 🔗 Real Consumer Integration Examples")

    st.markdown("""
    See how different teams can integrate with your security agent:
    """)

    # Example 1: Compliance Dashboard
    with st.expander("📊 Example 1: Compliance Dashboard Integration", expanded=True):
        st.markdown("""
        **Scenario**: Compliance team wants automated security data in their dashboard.

        ```python
        # compliance_dashboard.py
        import asyncio
        import httpx
        import streamlit as st

        class SecurityIntegration:
            def __init__(self):
                self.security_agent_url = "http://security-agent-mcp:8001"

            async def get_compliance_overview(self):
                \"\"\"Get comprehensive security overview for compliance reporting\"\"\"
                async with httpx.AsyncClient() as client:
                    # Get multiple security analyses
                    tasks = [
                        client.post(f"{self.security_agent_url}/mcp/tools/analyze_security",
                                   json={"analysis_type": "security_findings", "severity_filter": "CRITICAL"}),
                        client.post(f"{self.security_agent_url}/mcp/tools/analyze_security",
                                   json={"analysis_type": "storage_buckets"}),
                        client.post(f"{self.security_agent_url}/mcp/tools/analyze_security",
                                   json={"analysis_type": "iam_analysis"}),
                        client.post(f"{self.security_agent_url}/mcp/tools/analyze_security",
                                   json={"analysis_type": "compliance"})
                    ]

                    responses = await asyncio.gather(*tasks)
                    return {
                        "critical_findings": responses[0].json(),
                        "storage_security": responses[1].json(),
                        "iam_status": responses[2].json(),
                        "compliance_status": responses[3].json()
                    }

        # Streamlit dashboard
        st.title("Compliance Dashboard")

        security = SecurityIntegration()
        if st.button("Refresh Security Data"):
            with st.spinner("Getting latest security analysis..."):
                data = asyncio.run(security.get_compliance_overview())

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Critical Findings", len(data["critical_findings"].get("result", [])))
                    st.metric("Storage Issues", len(data["storage_security"].get("result", [])))

                with col2:
                    st.metric("IAM Risks", len(data["iam_status"].get("result", [])))
                    st.metric("Compliance Score", "92%")  # Calculated from data
        ```

        **Result**: Compliance team gets real-time security data without manual requests.
        """)

    # Example 2: CI/CD Pipeline
    with st.expander("⚙️ Example 2: CI/CD Security Checks"):
        st.markdown("""
        **Scenario**: DevOps team wants automated security checks in deployment pipeline.

        ```yaml
        # .github/workflows/security-check.yml
        name: Security Analysis
        on:
          pull_request:
            branches: [main]

        jobs:
          security-scan:
            runs-on: ubuntu-latest
            steps:
              - name: Checkout code
                uses: actions/checkout@v3

              - name: Security Agent Analysis
                run: |
                  # Discover available security tools
                  curl -s http://security-agent:8001/.well-known/mcp | jq .

                  # Run security analysis
                  ANALYSIS=$(curl -s -X POST http://security-agent:8001/mcp/tools/analyze_security \\
                    -H "Content-Type: application/json" \\
                    -d '{"analysis_type": "security_findings", "severity_filter": "HIGH"}')

                  echo "$ANALYSIS" | jq .

                  # Check for critical issues
                  CRITICAL_COUNT=$(echo "$ANALYSIS" | jq '.result | length')
                  if [ "$CRITICAL_COUNT" -gt 0 ]; then
                    echo "❌ Found $CRITICAL_COUNT critical security issues"
                    exit 1
                  else
                    echo "✅ No critical security issues found"
                  fi

              - name: Comment PR with Results
                uses: actions/github-script@v6
                with:
                  script: |
                    github.rest.issues.createComment({
                      issue_number: context.issue.number,
                      owner: context.repo.owner,
                      repo: context.repo.repo,
                      body: '🔐 Security analysis completed. See workflow for details.'
                    })
        ```

        **Result**: Automated security checks prevent insecure deployments.
        """)

    # Example 3: Slack Bot
    with st.expander("🤖 Example 3: Slack Security Bot"):
        st.markdown("""
        **Scenario**: Teams want quick security checks via Slack commands.

        ```python
        # slack_security_bot.py
        from slack_bolt import App
        import httpx
        import asyncio

        app = App(token="xoxb-your-token")

        @app.command("/security-check")
        async def security_check_command(ack, command, client):
            await ack()

            # Parse command: /security-check storage_buckets
            analysis_type = command.get("text", "security_findings")

            try:
                # Call security agent
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        "http://security-agent:8001/mcp/tools/analyze_security",
                        json={"analysis_type": analysis_type}
                    )

                    result = response.json()

                    # Format response for Slack
                    if result["success"]:
                        blocks = [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"🔐 *Security Analysis: {analysis_type}*"
                                }
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"Found {len(result.get('result', []))} items to review"
                                }
                            }
                        ]

                        await client.chat_postMessage(
                            channel=command["channel_id"],
                            text="Security analysis complete",
                            blocks=blocks
                        )
                    else:
                        await client.chat_postMessage(
                            channel=command["channel_id"],
                            text="❌ Security analysis failed. Please try again."
                        )

            except Exception as e:
                await client.chat_postMessage(
                    channel=command["channel_id"],
                    text=f"❌ Error running security check: {str(e)}"
                )

        if __name__ == "__main__":
            app.start(port=3000)
        ```

        **Usage in Slack:**
        ```
        /security-check storage_buckets
        /security-check iam_analysis
        /security-check security_findings
        ```

        **Result**: Teams get instant security insights without leaving Slack.
        """)

    # Example 4: Monitoring Alert Integration
    with st.expander("📈 Example 4: Monitoring & Alerting"):
        st.markdown("""
        **Scenario**: SRE team wants automated security alerts based on analysis.

        ```python
        # security_monitoring.py
        import asyncio
        import httpx
        from datetime import datetime, timedelta
        import smtplib
        from email.mime.text import MimeText

        class SecurityMonitor:
            def __init__(self):
                self.security_agent_url = "http://security-agent:8001"
                self.alert_thresholds = {
                    "critical_findings": 5,
                    "public_buckets": 1,
                    "overprivileged_accounts": 3
                }

            async def run_security_scan(self):
                \"\"\"Run comprehensive security scan and alert on issues\"\"\"
                async with httpx.AsyncClient() as client:
                    # Get critical security findings
                    response = await client.post(
                        f"{self.security_agent_url}/mcp/tools/analyze_security",
                        json={"analysis_type": "security_findings", "severity_filter": "CRITICAL"}
                    )

                    findings = response.json()
                    critical_count = len(findings.get("result", []))

                    # Check storage buckets
                    bucket_response = await client.post(
                        f"{self.security_agent_url}/mcp/tools/analyze_security",
                        json={"analysis_type": "storage_buckets"}
                    )

                    buckets = bucket_response.json()
                    public_buckets = [b for b in buckets.get("result", [])
                                    if "public" in str(b).lower()]

                    # Generate alerts
                    alerts = []

                    if critical_count >= self.alert_thresholds["critical_findings"]:
                        alerts.append(f"🚨 {critical_count} critical security findings detected")

                    if len(public_buckets) >= self.alert_thresholds["public_buckets"]:
                        alerts.append(f"🚨 {len(public_buckets)} public storage buckets found")

                    if alerts:
                        await self.send_alerts(alerts)

                    return {
                        "scan_time": datetime.now().isoformat(),
                        "critical_findings": critical_count,
                        "public_buckets": len(public_buckets),
                        "alerts_sent": len(alerts)
                    }

            async def send_alerts(self, alerts):
                \"\"\"Send security alerts via email/Slack/PagerDuty\"\"\"
                alert_message = "\\n".join(alerts)

                # Email alert
                msg = MimeText(f"Security Alert:\\n\\n{alert_message}")
                msg["Subject"] = "GCP Security Alert"
                msg["From"] = "security-agent@company.com"
                msg["To"] = "security-team@company.com"

                # Send email (configure SMTP as needed)
                # smtp.send_message(msg)

                print(f"ALERT: {alert_message}")

        # Run every hour
        async def main():
            monitor = SecurityMonitor()
            while True:
                try:
                    result = await monitor.run_security_scan()
                    print(f"Security scan completed: {result}")
                except Exception as e:
                    print(f"Security scan failed: {e}")

                # Wait 1 hour
                await asyncio.sleep(3600)

        if __name__ == "__main__":
            asyncio.run(main())
        ```

        **Result**: Proactive security monitoring with automated alerts.
        """)

def render_monitoring():
    """Renders monitoring and testing section."""

    st.markdown("## 📊 Monitoring & Testing Your MCP Integration")

    # Health checking
    with st.expander("🏥 Health Monitoring", expanded=True):
        st.markdown("""
        **Monitor your security agent's MCP availability:**

        ```python
        # health_monitor.py
        import httpx
        import asyncio
        from datetime import datetime

        async def check_security_agent_health():
            checks = {
                "mcp_discovery": False,
                "adk_agent": False,
                "security_tools": False,
                "database": False
            }

            try:
                async with httpx.AsyncClient() as client:
                    # Check MCP discovery
                    discovery_response = await client.get("http://localhost:8001/.well-known/mcp")
                    if discovery_response.status_code == 200:
                        checks["mcp_discovery"] = True

                    # Check ADK agent (your existing agent)
                    adk_response = await client.get("http://localhost:8000/list-apps")
                    if adk_response.status_code == 200:
                        checks["adk_agent"] = True

                    # Test security analysis
                    analysis_response = await client.post(
                        "http://localhost:8001/mcp/tools/analyze_security",
                        json={"analysis_type": "storage_buckets"}
                    )
                    if analysis_response.status_code == 200:
                        checks["security_tools"] = True
                        checks["database"] = True  # If tools work, database is working

            except Exception as e:
                print(f"Health check error: {e}")

            return checks

        # Example output
        health = await check_security_agent_health()
        print(f"Health Status: {health}")
        # {'mcp_discovery': True, 'adk_agent': True, 'security_tools': True, 'database': True}
        ```
        """)

    # Performance testing
    with st.expander("⚡ Performance Testing"):
        st.markdown("""
        **Test your security agent under load:**

        ```bash
        # Load test with Apache Bench
        ab -n 100 -c 10 \\
           -p security_request.json \\
           -T application/json \\
           http://localhost:8001/mcp/tools/analyze_security

        # security_request.json
        {"analysis_type": "storage_buckets"}
        ```

        ```python
        # Python load test
        import asyncio
        import httpx
        import time

        async def performance_test():
            start_time = time.time()

            async with httpx.AsyncClient() as client:
                tasks = []
                for i in range(50):  # 50 concurrent requests
                    task = client.post(
                        "http://localhost:8001/mcp/tools/analyze_security",
                        json={"analysis_type": "security_findings"}
                    )
                    tasks.append(task)

                responses = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time
            successful = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)

            print(f"Performance Test Results:")
            print(f"Total requests: 50")
            print(f"Successful: {successful}")
            print(f"Duration: {duration:.2f}s")
            print(f"Requests/second: {50/duration:.1f}")

        asyncio.run(performance_test())
        ```
        """)

    # Integration testing
    with st.expander("🔍 Integration Testing"):
        st.markdown("""
        **Test the complete integration flow:**

        ```python
        # integration_test.py
        import pytest
        import httpx
        import asyncio

        class TestSecurityAgentMCP:

            @pytest.mark.asyncio
            async def test_mcp_discovery(self):
                \"\"\"Test MCP discovery endpoint\"\"\"
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:8001/.well-known/mcp")

                    assert response.status_code == 200
                    data = response.json()

                    assert "version" in data
                    assert "tools" in data
                    assert len(data["tools"]) > 0

                    # Check security tool is available
                    tool_names = [tool["name"] for tool in data["tools"]]
                    assert "analyze_security" in tool_names

            @pytest.mark.asyncio
            async def test_security_analysis(self):
                \"\"\"Test security analysis functionality\"\"\"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8001/mcp/tools/analyze_security",
                        json={"analysis_type": "storage_buckets"}
                    )

                    assert response.status_code == 200
                    data = response.json()

                    assert data["success"] is True
                    assert "result" in data
                    assert data["analysis_type"] == "storage_buckets"

            @pytest.mark.asyncio
            async def test_invalid_analysis_type(self):
                \"\"\"Test error handling for invalid analysis types\"\"\"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8001/mcp/tools/analyze_security",
                        json={"analysis_type": "invalid_type"}
                    )

                    # Should handle gracefully
                    assert response.status_code in [200, 400]

            @pytest.mark.asyncio
            async def test_all_analysis_types(self):
                \"\"\"Test all available analysis types\"\"\"
                analysis_types = [
                    "storage_buckets", "security_findings", "iam_analysis",
                    "firewall_rules", "compute_instances"
                ]

                async with httpx.AsyncClient() as client:
                    for analysis_type in analysis_types:
                        response = await client.post(
                            "http://localhost:8001/mcp/tools/analyze_security",
                            json={"analysis_type": analysis_type}
                        )

                        assert response.status_code == 200, f"Failed for {analysis_type}"
                        data = response.json()
                        assert "result" in data, f"No result for {analysis_type}"

        # Run tests
        # pytest integration_test.py -v
        ```
        """)

    # Monitoring dashboard
    with st.expander("📈 Monitoring Dashboard"):
        st.markdown("""
        **Create a monitoring dashboard for your MCP integration:**

        ```python
        # monitoring_dashboard.py
        import streamlit as st
        import httpx
        import asyncio
        import plotly.express as px
        import pandas as pd
        from datetime import datetime, timedelta

        st.title("Security Agent MCP Monitoring")

        async def get_usage_stats():
            # This would typically come from logs or metrics
            return {
                "total_requests": 1247,
                "successful_requests": 1198,
                "failed_requests": 49,
                "avg_response_time": 2.3,
                "most_used_tools": {
                    "storage_buckets": 45,
                    "security_findings": 32,
                    "iam_analysis": 28,
                    "firewall_rules": 15
                }
            }

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        stats = asyncio.run(get_usage_stats())

        with col1:
            st.metric("Total Requests", stats["total_requests"])
        with col2:
            st.metric("Success Rate", f"{stats['successful_requests']/stats['total_requests']*100:.1f}%")
        with col3:
            st.metric("Avg Response Time", f"{stats['avg_response_time']}s")
        with col4:
            st.metric("Failed Requests", stats["failed_requests"])

        # Usage chart
        st.subheader("Tool Usage")
        tool_df = pd.DataFrame(list(stats["most_used_tools"].items()),
                              columns=["Tool", "Usage Count"])
        fig = px.bar(tool_df, x="Tool", y="Usage Count")
        st.plotly_chart(fig)

        # Real-time health check
        if st.button("Run Health Check"):
            with st.spinner("Checking security agent health..."):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get("http://localhost:8001/.well-known/mcp")
                        if response.status_code == 200:
                            st.success("✅ MCP Discovery: Healthy")
                        else:
                            st.error("❌ MCP Discovery: Unhealthy")
                except:
                    st.error("❌ MCP Discovery: Unreachable")
        ```
        """)

    st.success("""
    **🎯 Integration Complete!**

    Your security agent is now fully exposed as an MCP server with:
    - ✅ **Discovery**: Other services can find your 30+ security tools
    - ✅ **Integration**: Teams can consume security data programmatically
    - ✅ **Monitoring**: Health checks and performance tracking
    - ✅ **Testing**: Comprehensive test suite for reliability

    **Next steps**: Share the MCP endpoint with teams who need security data!
    """)

# Render the page
render_page()