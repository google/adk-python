#!/usr/bin/env python3
"""
Security Agent MCP Wrapper
==========================

This wrapper exposes the existing GCP Security Agent as an MCP (Model Context Protocol) server,
enabling other services to discover and consume the security agent's 30+ analysis tools.

The wrapper acts as a bridge between MCP clients and the existing ADK-powered security agent
running on port 8000.

Usage:
    python mcp_wrapper.py

Then access:
    - MCP Discovery: http://localhost:8001/.well-known/mcp
    - Tool Calling: http://localhost:8001/mcp/tools/analyze_security
    - Documentation: http://localhost:8001/docs
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
ADK_BASE_URL = "http://localhost:8000"
MCP_SERVER_PORT = 8001

# FastAPI app
app = FastAPI(
    title="GCP Security Agent MCP Server",
    description="MCP wrapper for the GCP Security Agent - enables discovery and consumption of 30+ security analysis tools",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SecurityAnalysisRequest(BaseModel):
    analysis_type: str
    severity_filter: Optional[str] = None
    service_name: Optional[str] = None
    limit: Optional[int] = 10
    force_live_update: Optional[bool] = False

class SecurityAnalysisResponse(BaseModel):
    success: bool
    analysis_type: str
    result: Any
    agent_model: str
    execution_time: Optional[float] = None
    timestamp: str
    error: Optional[str] = None

# Available analysis types (from the actual security agent)
AVAILABLE_ANALYSIS_TYPES = [
    "security_summary", "assets", "security_findings", "iam_analysis",
    "storage_buckets", "firewall_rules", "networks", "compute_instances",
    "gke_clusters", "databases", "iam_accounts", "secrets", "monitoring",
    "logs", "service_evaluation", "recommendations", "org_policies",
    "service_usage", "cache_status", "statistics", "msa_analysis",
    "vpc_error_analysis", "support_tickets", "vpcsc_dry_run",
    "vpcsc_readiness", "asset_inventory", "configuration_drift",
    "asset_report", "msa_impact", "msa_permissions", "context_aware_analysis",
    "cross_impact_analysis", "coding_standards", "enterprise_policies",
    "best_practices", "compliance", "search_docs"
]

# Tool descriptions for common analysis types
TOOL_DESCRIPTIONS = {
    "storage_buckets": "Analyze GCS bucket security configurations including public access, encryption, and lifecycle policies",
    "security_findings": "Get security vulnerabilities and compliance issues with severity filtering",
    "iam_analysis": "Review IAM policies, access patterns, and identify overprivileged accounts",
    "firewall_rules": "Analyze network firewall configurations for overly permissive rules",
    "compute_instances": "Review VM security configurations and identify misconfigurations",
    "service_evaluation": "Security risk assessment for adopting new GCP services",
    "compliance": "Regulatory compliance checking across GCP resources",
    "asset_inventory": "Comprehensive resource security overview and inventory",
    "monitoring": "Security monitoring configuration review and recommendations",
    "secrets": "Secrets management security analysis and best practices review"
}

@app.get("/.well-known/mcp")
async def mcp_discovery():
    """
    MCP Discovery endpoint - exposes the security agent capabilities.

    This endpoint follows the MCP specification for service discovery,
    allowing other services to automatically discover available security tools.
    """
    logger.info("MCP discovery request received")

    return {
        "version": "1.0",
        "server_info": {
            "name": "GCP Security Agent MCP Server",
            "version": "1.0.0",
            "description": "Micron IT Security Agent - Comprehensive GCP security analysis with 30+ specialized tools"
        },
        "capabilities": {
            "tools": True,
            "resources": True,
            "logging": True,
            "prompts": False
        },
        "tools": [
            {
                "name": "analyze_security",
                "description": "Comprehensive GCP security analysis across multiple domains using ADK-powered agent",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "enum": AVAILABLE_ANALYSIS_TYPES,
                            "description": "Type of security analysis to perform"
                        },
                        "severity_filter": {
                            "type": "string",
                            "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                            "description": "Filter results by severity level (optional)"
                        },
                        "service_name": {
                            "type": "string",
                            "description": "For service_evaluation analysis - name of GCP service to evaluate"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum number of results to return"
                        },
                        "force_live_update": {
                            "type": "boolean",
                            "default": False,
                            "description": "Force live GCP API data instead of cached data"
                        }
                    },
                    "required": ["analysis_type"]
                }
            },
            {
                "name": "get_capabilities",
                "description": "Get detailed information about available security analysis tools and capabilities",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "health_check",
                "description": "Check the health and availability of the security agent and its dependencies",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ],
        "endpoints": {
            "analyze_security": "/mcp/tools/analyze_security",
            "get_capabilities": "/mcp/tools/get_capabilities",
            "health_check": "/mcp/tools/health_check",
            "documentation": "/docs"
        },
        "authentication": {
            "type": "none",
            "required": False,
            "description": "No authentication required for basic security analysis"
        },
        "contact": {
            "team": "Micron IT Security Team",
            "documentation": "http://localhost:8501/Security_Agent_MCP_Integration",
            "adk_agent": "http://localhost:8000",
            "support": "security-team@micron.com"
        },
        "usage_examples": [
            {
                "description": "Analyze storage bucket security",
                "request": {
                    "analysis_type": "storage_buckets"
                }
            },
            {
                "description": "Get critical security findings",
                "request": {
                    "analysis_type": "security_findings",
                    "severity_filter": "CRITICAL"
                }
            },
            {
                "description": "Evaluate Cloud Functions security risks",
                "request": {
                    "analysis_type": "service_evaluation",
                    "service_name": "Cloud Functions"
                }
            },
            {
                "description": "IAM security analysis",
                "request": {
                    "analysis_type": "iam_analysis"
                }
            }
        ]
    }

@app.post("/mcp/tools/analyze_security", response_model=SecurityAnalysisResponse)
async def analyze_security(request: SecurityAnalysisRequest):
    """
    Main security analysis endpoint - proxies requests to the ADK agent.

    This endpoint converts MCP tool calls into natural language queries
    for the ADK agent, maintaining the conversational interface.
    """
    logger.info(f"Security analysis request: {request.analysis_type}")

    start_time = datetime.now()

    try:
        # Validate analysis type
        if request.analysis_type not in AVAILABLE_ANALYSIS_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis_type. Must be one of: {', '.join(AVAILABLE_ANALYSIS_TYPES[:10])}..."
            )

        # Build natural language query for the ADK agent
        message = f"Analyze {request.analysis_type.replace('_', ' ')}"

        if request.severity_filter:
            message += f" with {request.severity_filter} severity"

        if request.service_name:
            message += f" for {request.service_name}"

        if request.limit and request.limit != 10:
            message += f" (limit {request.limit} results)"

        if request.force_live_update:
            message += " using live data"

        logger.info(f"Sending query to ADK agent: {message}")

        # Call the existing ADK agent
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create session with ADK agent
            session_response = await client.post(
                f"{ADK_BASE_URL}/apps/agents/users/mcp-client/sessions",
                json={"app_name": "agents"}
            )

            if session_response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to create ADK session: {session_response.status_code}"
                )

            session_data = session_response.json()
            session_id = session_data.get("id") or session_data.get("session_id")

            if not session_id:
                raise HTTPException(
                    status_code=503,
                    detail=f"No session ID returned from ADK agent. Response: {session_data}"
                )

            # Send analysis request to ADK agent
            analysis_response = await client.post(
                f"{ADK_BASE_URL}/run",
                json={
                    "appName": "agents",
                    "userId": "mcp-client",
                    "sessionId": session_id,
                    "newMessage": {
                        "parts": [{"text": message}],
                        "role": "user"
                    },
                    "streaming": False
                }
            )

            if analysis_response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail=f"ADK agent request failed: {analysis_response.status_code}"
                )

            # Parse ADK response
            adk_result = analysis_response.json()

            # Extract the actual response from ADK's event structure
            response_text = ""
            tool_used = False

            for event in adk_result:
                if isinstance(event, dict):
                    # Check for tool usage indicators
                    if "tool_calls" in event or "function_call" in str(event):
                        tool_used = True

                    # Extract text content
                    if "content" in event and isinstance(event["content"], dict):
                        content = event["content"]
                        if "parts" in content and isinstance(content["parts"], list):
                            for part in content["parts"]:
                                if isinstance(part, dict) and "text" in part:
                                    text = part["text"].strip()
                                    if len(text) > 20:  # Filter out short system messages
                                        response_text = text
                                        break
                    elif "text" in event:
                        response_text = event["text"]

            execution_time = (datetime.now() - start_time).total_seconds()

            return SecurityAnalysisResponse(
                success=True,
                analysis_type=request.analysis_type,
                result={
                    "analysis": response_text,
                    "tool_used": tool_used,
                    "query_sent": message,
                    "raw_response": adk_result if len(str(adk_result)) < 1000 else "Response too large to include"
                },
                agent_model="gemini-2.5-flash",
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )

    except httpx.TimeoutException:
        logger.error("ADK agent request timed out")
        raise HTTPException(status_code=504, detail="Security analysis request timed out")
    except httpx.RequestError as e:
        logger.error(f"ADK agent request error: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to security agent: {str(e)}")
    except Exception as e:
        logger.error(f"Security analysis error: {e}")
        execution_time = (datetime.now() - start_time).total_seconds()

        return SecurityAnalysisResponse(
            success=False,
            analysis_type=request.analysis_type,
            result={},
            agent_model="gemini-2.5-flash",
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.post("/mcp/tools/get_capabilities")
async def get_capabilities():
    """Get detailed information about security agent capabilities."""
    logger.info("Capabilities request received")

    return {
        "success": True,
        "agent_info": {
            "name": "GCP Security Agent",
            "model": "gemini-2.5-flash",
            "type": "ADK-powered security analysis agent"
        },
        "available_tools": len(AVAILABLE_ANALYSIS_TYPES),
        "tool_categories": {
            "core_security": [
                "storage_buckets", "security_findings", "iam_analysis",
                "firewall_rules", "compute_instances"
            ],
            "advanced_analysis": [
                "service_evaluation", "compliance", "asset_inventory",
                "monitoring", "secrets"
            ],
            "specialized": [
                "vpc_error_analysis", "vpcsc_readiness", "org_policies",
                "configuration_drift", "msa_analysis"
            ]
        },
        "tool_descriptions": TOOL_DESCRIPTIONS,
        "data_sources": [
            "Google Cloud Asset Inventory",
            "Google Cloud Security Command Center",
            "Cloud Storage API",
            "Compute Engine API",
            "IAM API",
            "Cached security analysis data"
        ],
        "features": {
            "real_time_analysis": True,
            "cached_results": True,
            "severity_filtering": True,
            "service_evaluation": True,
            "ai_powered_insights": True
        }
    }

@app.post("/mcp/tools/health_check")
async def health_check():
    """Check the health and availability of the security agent."""
    logger.info("Health check request received")

    health_status = {
        "timestamp": datetime.now().isoformat(),
        "mcp_server": "healthy",
        "adk_agent": "unknown",
        "database": "unknown",
        "overall_status": "unknown"
    }

    try:
        # Check if ADK agent is responding
        async with httpx.AsyncClient(timeout=5.0) as client:
            adk_response = await client.get(f"{ADK_BASE_URL}/list-apps")

            if adk_response.status_code == 200:
                health_status["adk_agent"] = "healthy"

                # Try a simple analysis to check database
                try:
                    session_response = await client.post(
                        f"{ADK_BASE_URL}/apps/agents/users/health-check/sessions",
                        json={"app_name": "agents"}
                    )

                    if session_response.status_code == 200:
                        session_data = session_response.json()
                        session_id = session_data.get("id") or session_data.get("session_id")

                        if session_id:
                            test_response = await client.post(
                                f"{ADK_BASE_URL}/run",
                                json={
                                    "appName": "agents",
                                    "userId": "health-check",
                                    "sessionId": session_id,
                                    "newMessage": {
                                        "parts": [{"text": "Get statistics"}],
                                        "role": "user"
                                    },
                                    "streaming": False
                                }
                            )

                            if test_response.status_code == 200:
                                health_status["database"] = "healthy"
                            else:
                                health_status["database"] = "degraded"
                        else:
                            health_status["database"] = "unavailable"
                    else:
                        health_status["database"] = "unavailable"

                except Exception:
                    health_status["database"] = "unavailable"
            else:
                health_status["adk_agent"] = "unhealthy"
                health_status["database"] = "unavailable"

    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_status["adk_agent"] = "unreachable"
        health_status["database"] = "unavailable"

    # Determine overall status
    if (health_status["adk_agent"] == "healthy" and
        health_status["database"] in ["healthy", "degraded"]):
        health_status["overall_status"] = "healthy"
    elif health_status["adk_agent"] == "healthy":
        health_status["overall_status"] = "degraded"
    else:
        health_status["overall_status"] = "unhealthy"

    status_code = 200 if health_status["overall_status"] in ["healthy", "degraded"] else 503

    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "name": "GCP Security Agent MCP Server",
        "version": "1.0.0",
        "description": "MCP wrapper for ADK-powered GCP security analysis",
        "endpoints": {
            "mcp_discovery": "/.well-known/mcp",
            "analyze_security": "/mcp/tools/analyze_security",
            "get_capabilities": "/mcp/tools/get_capabilities",
            "health_check": "/mcp/tools/health_check",
            "documentation": "/docs"
        },
        "adk_agent": {
            "url": ADK_BASE_URL,
            "model": "gemini-2.5-flash",
            "tools_available": len(AVAILABLE_ANALYSIS_TYPES)
        },
        "status": "ready"
    }

@app.get("/health")
async def simple_health():
    """Simple health endpoint for load balancers."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()

    logger.info(f"🔍 {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")

    response = await call_next(request)

    duration = (datetime.now() - start_time).total_seconds()
    status_emoji = "✅" if 200 <= response.status_code < 300 else ("⚠️" if 300 <= response.status_code < 400 else "❌")

    logger.info(f"{status_emoji} {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")

    response.headers["X-Process-Time"] = str(duration)

    return response

def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("GCP Security Agent MCP Server Starting")
    logger.info("=" * 60)
    logger.info(f"MCP Server Port: {MCP_SERVER_PORT}")
    logger.info(f"ADK Agent URL: {ADK_BASE_URL}")
    logger.info(f"Available Tools: {len(AVAILABLE_ANALYSIS_TYPES)}")
    logger.info("=" * 60)

    # Check if ADK agent is running
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{ADK_BASE_URL}/list-apps")
            if response.status_code == 200:
                logger.info("✅ ADK agent is running and accessible")
            else:
                logger.warning(f"⚠️ ADK agent returned status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to ADK agent at {ADK_BASE_URL}: {e}")
        logger.error("Please ensure the ADK agent is running with: python -m dotenv run -- adk web")

    # Start the MCP server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=MCP_SERVER_PORT,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()