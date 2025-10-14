"""BigQuery Security Agent configuration and tool registration."""

from __future__ import annotations

import logging
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

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

# Import MSA Analyzer (Multi-Service Analyzer)
from ._tools.msa_analyzer import analyze_gcp_releases

# Import Service Evaluation Framework
from ._tools.service_evaluation import check_service_compliance
from ._tools.evaluation_tools import run_session_genai_evaluation

# Import configuration
from ._tools.base import PROJECT_ID, DEFAULT_DATASET, DEFAULT_TABLE
from ._tools.bigquery_tools import BigQueryTool

logger = logging.getLogger(__name__)

INSTRUCTION_PATH = Path(__file__).resolve().parent.parent / "docs" / "agent_instructions.md"


def _apply_instruction_tokens(text: str) -> str:
    replacements = {
        "{DEFAULT_DATASET}": DEFAULT_DATASET,
        "{DEFAULT_TABLE}": DEFAULT_TABLE,
        "{PROJECT_ID}": PROJECT_ID,
    }
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _load_instruction() -> str:
    """Load agent instructions from the markdown document."""

    try:
        markdown = INSTRUCTION_PATH.read_text(encoding="utf-8")
        return _apply_instruction_tokens(markdown)
    except FileNotFoundError:
        logger.warning("Instruction markdown not found at %s", INSTRUCTION_PATH)
        fallback = (
            "You are a specialized Security Analyst for the {DEFAULT_DATASET}.{DEFAULT_TABLE} "
            "BigQuery dataset in project {PROJECT_ID}. Focus on providing clear, actionable "
            "insights from the security_insights dataset and maintain a friendly, helpful tone."
        )
        return _apply_instruction_tokens(fallback)

# Shared tool instances
bigquery_toolset = BigQueryTool()

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
    # MSA (Multi-Service Analyzer) - Release notes impact analysis
    FunctionTool(analyze_gcp_releases),
    # Service Evaluation Framework - Comprehensive service security assessment
    FunctionTool(bigquery_toolset.evaluate_service),
    # Compliance Checker - Validate controls against BigQuery (actual environment state)
    FunctionTool(check_service_compliance),
    # GenAI Evaluation - Evaluate logged sessions with Vertex AI tooling
    FunctionTool(run_session_genai_evaluation),
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
instruction = _load_instruction()

# Create the agent
root_agent = LlmAgent(
    name="security_bigquery_agent",
    model="gemini-2.5-flash",
    instruction=instruction,
    tools=tools
)
