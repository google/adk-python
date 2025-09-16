# MCP Integration for Security Agent

## Overview

The Security Agent uses the `fastapi-mcp` library to automatically expose all API endpoints as MCP (Model Control Protocol) tools, enabling AI assistants to discover and interact with the security features.

## Implementation

The MCP integration is incredibly simple - just 2 lines of code in `backend/main.py`:

```python
from fastapi_mcp import FastApiMCP

# Enable MCP integration
mcp = FastApiMCP(app)
mcp.mount()  # Creates MCP server at /mcp endpoint
```

This automatically:
- ✅ Exposes all FastAPI endpoints as MCP tools
- ✅ Generates tool descriptions from endpoint docstrings
- ✅ Creates the discovery endpoint at `/mcp/.well-known/mcp.json`
- ✅ Handles parameter validation and type conversion
- ✅ Provides error handling and response formatting

## Discovery

AI assistants can discover available tools at:
```
http://localhost:8000/mcp/.well-known/mcp.json
```

## Available Security Tools

Once MCP is enabled, all security agent endpoints become available as tools:

- **Security Scanning** - Comprehensive security assessments
- **IAM Analysis** - Identity and access management review
- **Vulnerability Assessment** - CVE and security vulnerability checks
- **Compliance Checking** - SOC2, ISO27001, HIPAA compliance
- **Cloud Security** - AWS, Azure, GCP security posture
- **Incident Response** - Automated threat response workflows
- **Storage Security** - Bucket and data security analysis

## Usage Example

AI assistants can use natural language to interact with the security agent:

```
"Run a comprehensive security scan of our cloud infrastructure"
"Check IAM policies for excessive permissions"
"Analyze storage buckets for public access"
"Generate a compliance report for SOC2"
```

## Installation

```bash
pip install fastapi-mcp
```

## Benefits

- **Zero Configuration** - Works out of the box with existing FastAPI apps
- **Automatic Discovery** - AI assistants find tools automatically
- **Natural Language** - No need to learn API syntax
- **Type Safety** - Parameter validation handled automatically
- **Error Handling** - Graceful error messages for AI consumption

## Learn More

- [fastapi-mcp on PyPI](https://pypi.org/project/fastapi-mcp/)
- [MCP Protocol Specification](https://github.com/modelcontextprotocol)