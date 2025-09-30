#!/usr/bin/env python3
"""
MCP Server for ADK Security Agent
Exposes the security agent tools via Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import ADK agent and tools
from agents.agent import root_agent
from agents._tools import bigquery_tools
from agents._tools import security_tools
from agents._tools import exploration_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server setup
server = Server("adk-security-agent")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools from the ADK security agent."""
    return [
        types.Tool(
            name="query_security_data",
            description="Query security data from GCP including IAM, assets, and security findings",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about security data"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "iam_accounts", "users", "service_accounts", "custom_roles",
                            "standard_roles", "storage_buckets", "compute_instances",
                            "firewall_rules", "security_findings", "exploration"
                        ],
                        "description": "Type of security data to query"
                    },
                    "force_live_update": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force fetch from live GCP APIs instead of cache"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="analyze_security_posture",
            description="Analyze overall security posture and provide recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "focus_area": {
                        "type": "string",
                        "enum": ["iam", "network", "storage", "compute", "overall"],
                        "description": "Security area to focus analysis on"
                    }
                }
            }
        ),
        types.Tool(
            name="get_security_metrics",
            description="Get key security metrics and KPIs",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "enum": ["iam", "assets", "findings", "compliance"],
                        "description": "Type of metrics to retrieve"
                    }
                }
            }
        ),
        types.Tool(
            name="search_documentation",
            description="Search security documentation and policies",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for documentation"
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["policies", "procedures", "guidelines", "all"],
                        "default": "all",
                        "description": "Type of documentation to search"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Optional[Dict[str, Any]]
) -> List[types.TextContent]:
    """Handle tool calls from MCP clients."""
    try:
        if name == "query_security_data":
            return await handle_query_security_data(arguments)
        elif name == "analyze_security_posture":
            return await handle_analyze_security_posture(arguments)
        elif name == "get_security_metrics":
            return await handle_get_security_metrics(arguments)
        elif name == "search_documentation":
            return await handle_search_documentation(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

async def handle_query_security_data(arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
    """Handle security data queries."""
    if not arguments:
        arguments = {}

    query = arguments.get("query", "")
    query_type = arguments.get("query_type", "security_insights")

    logger.info(f"Querying security data: {query} (type: {query_type})")

    try:
        # Use the appropriate function based on query type
        if query_type == "security_insights" or query_type == "exploration":
            result = security_tools.get_security_insights_summary()
        elif query_type == "hello_world":
            result = bigquery_tools.hello_world()
        elif query_type == "datasets":
            result = bigquery_tools.list_datasets()
        else:
            # Try to get security insights by default
            result = security_tools.get_security_insights_summary()

        response = f"Security Data Query: {query}\n"
        response += f"Type: {query_type}\n\n"
        response += f"Results:\n{result}"

        return [types.TextContent(type="text", text=response)]

    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(
            type="text",
            text=error_msg
        )]

async def handle_analyze_security_posture(arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
    """Handle security posture analysis."""
    if not arguments:
        arguments = {}

    focus_area = arguments.get("focus_area", "overall")

    logger.info(f"Analyzing security posture for: {focus_area}")

    try:
        # Get security insights
        security_summary = security_tools.get_security_insights_summary()

        # Generate analysis report
        report = f"Security Posture Analysis - {focus_area.upper()}\n"
        report += "=" * 50 + "\n\n"
        report += f"Security Insights Summary:\n{security_summary}\n\n"

        # Add recommendations based on focus area
        recommendations = {
            "iam": [
                "Review service accounts with admin privileges",
                "Audit custom roles for over-privileged access",
                "Enable MFA for all user accounts"
            ],
            "network": [
                "Review firewall rules allowing 0.0.0.0/0 access",
                "Implement network segmentation",
                "Enable VPC Flow Logs"
            ],
            "storage": [
                "Check for publicly accessible buckets",
                "Enable bucket logging and monitoring",
                "Review bucket IAM policies"
            ],
            "compute": [
                "Audit compute instances for security misconfigurations",
                "Enable OS security features",
                "Review instance metadata access"
            ],
            "overall": [
                "Implement comprehensive logging and monitoring",
                "Regular security posture assessments",
                "Establish incident response procedures"
            ]
        }

        report += "RECOMMENDATIONS:\n"
        for rec in recommendations.get(focus_area, []):
            report += f"• {rec}\n"

        return [types.TextContent(type="text", text=report)]

    except Exception as e:
        error_msg = f"Error analyzing security posture: {str(e)}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]

async def handle_get_security_metrics(arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
    """Handle security metrics requests."""
    if not arguments:
        arguments = {}

    metric_type = arguments.get("metric_type", "overall")

    logger.info(f"Getting security metrics for: {metric_type}")

    # Query relevant data for metrics
    metric_queries = {
        "iam": ["iam_accounts", "users", "service_accounts"],
        "assets": ["storage_buckets", "compute_instances"],
        "findings": ["security_findings"],
        "compliance": ["firewall_rules", "storage_buckets"]
    }

    queries = metric_queries.get(metric_type, list(metric_queries.keys()))
    if metric_type == "overall":
        queries = ["iam_accounts", "storage_buckets", "compute_instances", "firewall_rules"]

    metrics = {}

    for query_type in queries:
        result = security_tool.query_security_data(
            query=f"count {query_type}",
            query_type=query_type,
            force_live_update=False
        )
        if result.get("success"):
            data = result.get("data", [])
            metrics[query_type] = {
                "count": len(data),
                "last_updated": result.get("last_updated", "unknown")
            }

    # Format metrics response
    response = f"Security Metrics - {metric_type.upper()}\n"
    response += "=" * 40 + "\n\n"

    for area, data in metrics.items():
        response += f"{area.replace('_', ' ').title()}: {data['count']} items\n"
        response += f"  Last updated: {data['last_updated']}\n\n"

    return [types.TextContent(type="text", text=response)]

async def handle_search_documentation(arguments: Optional[Dict[str, Any]]) -> List[types.TextContent]:
    """Handle documentation search requests."""
    if not arguments:
        arguments = {}

    query = arguments.get("query", "")
    doc_type = arguments.get("doc_type", "all")

    logger.info(f"Searching documentation: {query} (type: {doc_type})")

    # Use exploration tool for documentation search
    result = exploration_tool.search_knowledge_base(
        query=query,
        search_type="documentation"
    )

    if result.get("success"):
        docs = result.get("data", [])

        response = f"Documentation Search: {query}\n"
        response += f"Type: {doc_type}\n"
        response += f"Results: {len(docs)} documents found\n\n"

        for i, doc in enumerate(docs[:5]):  # Limit to first 5 results
            response += f"{i+1}. {doc.get('title', 'Untitled')}\n"
            response += f"   {doc.get('summary', 'No summary available')}\n\n"

        if len(docs) > 5:
            response += f"... and {len(docs) - 5} more documents\n"

        return [types.TextContent(type="text", text=response)]
    else:
        error_msg = result.get("error", "No documentation found")
        return [types.TextContent(
            type="text",
            text=f"Documentation search failed: {error_msg}"
        )]

async def main():
    """Main entry point for the MCP server."""
    # Set environment variables if not already set
    os.environ.setdefault("DATABASE_PATH", str(project_root / "backend" / "cache" / "gcp_data.db"))

    # Initialize the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="adk-security-agent",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    logger.info("Starting ADK Security Agent MCP Server")
    asyncio.run(main())