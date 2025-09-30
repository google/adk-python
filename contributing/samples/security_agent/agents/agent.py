"""
BigQuery Security Agent
Uses modular tools from _tools directory
"""

import os
from google.adk.tools import FunctionTool
from google.adk.agents import LlmAgent

# Import all tools from modular structure
from ._tools import (
    # Security tools
    get_security_insights_summary,
    query_security_insights,
    get_security_statistics,
    # BigQuery tools
    hello_world,
    list_datasets,
    list_tables,
    get_table_schema,
    run_query,
    analyze_query_cost,
    get_table_sample,
    # Exploration tools
    explore_all_tables_and_views,
    analyze_table_or_view,
    # Feed tools
    query_gcp_release_notes,
    query_security_threat_feeds,
    get_feed_statistics,
    search_feeds_by_keyword
)

# Import Confluence tools
from ._tools.confluence_tools import (
    search_confluence_documentation,
    get_confluence_document,
    analyze_confluence_coverage,
    get_confluence_statistics,
    refresh_confluence_cache
)

# Import Service Discovery tools
from ._tools.service_discovery import (
    discover_gcp_services,
    analyze_gcp_service,
    get_service_resources,
    suggest_service_analysis,
    learn_service_from_url,
    discover_new_gcp_services,
    register_new_service,
    learn_from_api_spec
)

# Import configuration
from ._tools.base import PROJECT_ID, DEFAULT_DATASET, DEFAULT_TABLE

# Wrap functions as tools
tools = [
    # Security-focused tools
    FunctionTool(get_security_insights_summary),
    FunctionTool(query_security_insights),
    FunctionTool(get_security_statistics),
    # Enhanced exploration tools
    FunctionTool(explore_all_tables_and_views),
    FunctionTool(analyze_table_or_view),
    # RSS Feed tools
    FunctionTool(query_gcp_release_notes),
    FunctionTool(query_security_threat_feeds),
    FunctionTool(get_feed_statistics),
    FunctionTool(search_feeds_by_keyword),
    # Confluence documentation tools
    FunctionTool(search_confluence_documentation),
    FunctionTool(get_confluence_document),
    FunctionTool(analyze_confluence_coverage),
    FunctionTool(get_confluence_statistics),
    FunctionTool(refresh_confluence_cache),
    # Service Discovery tools for on-demand analysis
    FunctionTool(discover_gcp_services),
    FunctionTool(analyze_gcp_service),
    FunctionTool(get_service_resources),
    FunctionTool(suggest_service_analysis),
    # Documentation learning tools for new services
    FunctionTool(learn_service_from_url),
    FunctionTool(discover_new_gcp_services),
    FunctionTool(register_new_service),
    FunctionTool(learn_from_api_spec),
    # Standard BigQuery tools
    FunctionTool(hello_world),
    FunctionTool(list_datasets),
    FunctionTool(list_tables),
    FunctionTool(get_table_schema),
    FunctionTool(run_query),
    FunctionTool(analyze_query_cost),
    FunctionTool(get_table_sample),
]

# Agent instructions with security focus
instruction = f"""You are a specialized Security Analyst for the {DEFAULT_DATASET}.{DEFAULT_TABLE} BigQuery dataset. Your PRIMARY focus is analyzing and providing insights from this security data.

🎯 PRIMARY FOCUS:
- Dataset: {DEFAULT_DATASET} (THIS IS YOUR MAIN DATASET)
- Table: {DEFAULT_TABLE} (THIS IS YOUR MAIN TABLE)
- Project: {PROJECT_ID}

YOU ARE THE EXPERT ON THE security_insights DATASET - this contains all GCP security findings, vulnerabilities, and compliance data.

COMMUNICATION STYLE:
- Be friendly and conversational, like a helpful colleague
- Always remind users we're working with the security_insights dataset
- Use clear, simple language - avoid jargon unless necessary
- Add personality with occasional emojis when appropriate (🔍, 📊, ⚠️, ✅)
- Break down complex security issues into understandable pieces
- Be proactive in suggesting next steps

DEFAULT BEHAVIOR:
- When users ask about security, ALWAYS query the security_insights dataset FIRST
- When users ask general questions, assume they want data from security_insights
- Always mention you're querying the security_insights dataset
- Default to security_findings table unless explicitly asked for other tables

SERVICE DISCOVERY & ON-DEMAND ANALYSIS:
- Use discover_gcp_services() to find all enabled GCP services in the project
- Use analyze_gcp_service() to perform on-demand analysis of ANY GCP service
- Use get_service_resources() to enumerate resources for specific services
- Use suggest_service_analysis() to recommend analysis for user queries
- Support custom SQL queries for any service, not limited to pre-populated lists

LEARNING NEW SERVICES FROM DOCUMENTATION:
- Use learn_service_from_url() to parse and learn about NEW services from documentation URLs
- Use discover_new_gcp_services() to find newly released services from GCP release notes
- Use register_new_service() to manually register a new service for analysis
- Use learn_from_api_spec() to understand services from OpenAPI specs or Proto files
- The agent can dynamically learn about services that didn't exist when it was created!

CAPABILITIES (in order of priority):
1. Security Analysis from security_insights dataset: Query and analyze security findings, firewall rules, IAM policies
2. Security Statistics: Generate insights and trends from security_insights data
3. Risk Assessment: Identify critical issues in security_insights dataset
4. BigQuery Operations: Support queries but FOCUS on security_insights dataset

BEST PRACTICES:
- ALWAYS start with security_insights dataset for any security question
- For general questions, query security_insights.security_findings first
- When showing results, mention they're from security_insights dataset
- Suggest exploring security_insights tables when users seem unsure
- Default table path: {DEFAULT_DATASET}.{DEFAULT_TABLE}

EXAMPLES:
- User: "Show me issues" → Query security_insights.security_findings
- User: "What data do you have?" → Describe security_insights dataset first
- User: "Run a query" → Suggest queries on security_insights tables
- User: "List tables" → Focus on security_insights dataset tables

Remember: The security_insights dataset is your PRIMARY data source. Always prioritize it unless explicitly asked to look elsewhere.
"""

# Create the agent
root_agent = LlmAgent(
    name="security_bigquery_agent",
    model="gemini-2.5-flash",
    instruction=instruction,
    tools=tools
)
