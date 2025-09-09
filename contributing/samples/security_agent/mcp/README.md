# 🤖 AI-Native Security Operations via MCP

**Complete Model Control Protocol Integration for Enterprise Security**

## Overview

This directory contains the complete MCP (Model Control Protocol) discovery implementation for the Micron Security Agent, enabling AI assistants like Claude Code to automatically discover and interact with all security tools through natural language.

## 🏗️ Architecture

```
security_agent/mcp/
├── discovery/           # Service discovery implementations
│   ├── basic.py        # Single service discovery
│   ├── registry.py     # Multi-service registry
│   └── health.py       # Health-aware discovery
├── clients/            # MCP client implementations
│   ├── basic_client.py # Simple MCP client
│   ├── secure_client.py# Authenticated client
│   └── async_client.py # Async operations
├── patterns/           # Enterprise patterns
│   ├── authentication/ # JWT, OAuth, mTLS
│   ├── monitoring/     # Observability
│   └── deployment/     # Production guides
├── config/             # Configuration files
│   ├── services.yaml   # Service registry
│   └── mcp.json       # MCP settings
└── tests/              # Integration tests

```

## 🚀 Quick Start

### 1. Enable MCP in the Security Agent

The security agent already has MCP enabled via `fastapi_mcp`:

```python
# backend/main.py - Lines 240-242
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()
```

### 2. Discover Available Tools

```bash
# Check MCP discovery endpoint
curl http://localhost:8000/mcp/.well-known/mcp.json

# Use the discovery client
python mcp/discovery/basic.py
```

### 3. Connect Claude Code

```bash
# Connect Claude Code to the security agent
claude-code connect http://localhost:8000/mcp
```

## 📋 Available Security Tools via MCP

All 30+ security endpoints are now available as MCP tools:

### Security Operations
- `security_scan` - Comprehensive security scanning
- `vulnerability_check` - CVE and vulnerability detection
- `threat_assessment` - Real-time threat analysis
- `incident_response` - Automated incident handling

### IAM & Compliance
- `iam_analyze` - Identity and access management analysis
- `permission_audit` - Permission auditing
- `compliance_check` - Regulatory compliance validation
- `policy_enforce` - Security policy enforcement

### Cloud Security
- `cloud_security_posture` - CSPM capabilities
- `kubernetes_security` - K8s security scanning
- `container_scan` - Container vulnerability scanning
- `infrastructure_audit` - Infrastructure security audit

## 🔐 Security Patterns

### JWT Authentication
```python
from mcp.patterns.authentication import JWTAuth

auth = JWTAuth(secret_key="your-secret")
secure_client = SecureMCPClient(auth)
```

### OAuth 2.0
```python
from mcp.patterns.authentication import OAuth2Auth

oauth = OAuth2Auth(
    server_url="https://auth.company.com",
    client_id="security-agent"
)
```

### mTLS
```python
from mcp.patterns.authentication import mTLSAuth

mtls = mTLSAuth(
    ca_cert="/path/to/ca.pem",
    client_cert="/path/to/client.pem"
)
```

## 📊 Monitoring & Observability

### Prometheus Metrics
```python
from mcp.patterns.monitoring import MCPMetrics

metrics = MCPMetrics()
metrics.track_discovery()
metrics.track_tool_execution()
```

### Structured Logging
```python
from mcp.patterns.monitoring import MCPLogger

logger = MCPLogger("security-agent")
logger.log_discovery_event(service_name, tools_count)
```

## 🚢 Production Deployment

### Docker Deployment
```bash
# Build with MCP enabled
docker build -t security-agent-mcp .

# Run with MCP discovery
docker run -p 8000:8000 security-agent-mcp
```

### Kubernetes Deployment
```yaml
apiVersion: v1
kind: Service
metadata:
  name: security-agent-mcp
  annotations:
    mcp.ai/enabled: "true"
    mcp.ai/discovery-path: "/.well-known/mcp.json"
spec:
  ports:
  - port: 8000
    name: mcp
```

## 🧪 Testing

### Run MCP Tests
```bash
# Run all MCP tests
pytest mcp/tests/

# Test discovery
pytest mcp/tests/test_discovery.py

# Test authentication
pytest mcp/tests/test_auth.py
```

### Integration Testing
```python
# Test with real security agent
python mcp/tests/integration_test.py
```

## 📚 Documentation

- [Basic Integration Guide](./docs/basic-integration.md)
- [Enterprise Patterns](./docs/enterprise-patterns.md)
- [Security Best Practices](./docs/security.md)
- [Troubleshooting](./docs/troubleshooting.md)

## 🎯 Use Cases

### 1. Natural Language Security Operations
```
User: "Scan our production environment for vulnerabilities"
Claude: *Executes security_scan tool via MCP*
```

### 2. Automated Compliance Checks
```
User: "Check if we're compliant with SOC2"
Claude: *Executes compliance_check tool with SOC2 parameters*
```

### 3. Incident Response
```
User: "There's suspicious activity from IP 192.168.1.100"
Claude: *Executes threat_assessment and incident_response tools*
```

## 🔄 Integration with Other Services

The MCP-enabled security agent can be integrated with:
- **ServiceNow** - Automated ticket creation
- **Slack** - Security alerts and notifications
- **Splunk** - Log aggregation and analysis
- **PagerDuty** - Incident escalation
- **Jira** - Security task tracking

## 📈 Benefits

✅ **Instant AI Integration** - All security tools available to AI assistants
✅ **Natural Language Interface** - No need to remember commands
✅ **Automatic Documentation** - Tools self-document via schemas
✅ **Standards Compliant** - Follows MCP and .well-known specifications
✅ **Zero Code Changes** - Existing endpoints work automatically
✅ **Enterprise Ready** - Production authentication and monitoring

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/company/security-agent/issues)
- **Docs**: [MCP Documentation](./docs)
- **Slack**: #security-agent-mcp

---

**The Micron Security Agent with MCP Discovery - Making security operations accessible through natural language.**