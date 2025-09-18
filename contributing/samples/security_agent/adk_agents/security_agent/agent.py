#!/usr/bin/env python3
"""
GCP Security Agent for ADK Web Interface
========================================

This agent provides intelligent analysis of GCP security data,
including storage buckets, security findings, IAM accounts, and more.
It combines database queries with LLM-powered insights.
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents import Agent

# Add the project root to the Python path so we can import our tools
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_security_data(query_type: str, limit: int = 10) -> str:
    """Query security data from SQLite database."""

    def get_database_path() -> str:
        """Get the path to the SQLite database."""
        possible_paths = [
            "/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/backend/cache/gcp_data.db",
            str(project_root / "backend/cache/gcp_data.db"),
            "./backend/cache/gcp_data.db",
            "../backend/cache/gcp_data.db",
            "../../backend/cache/gcp_data.db"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return str(project_root / "backend/cache/gcp_data.db")

    def execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dictionaries."""
        db_path = get_database_path()

        if not os.path.exists(db_path):
            return [{"error": f"Database not found at {db_path}. Please run setup_demo_data.py first."}]

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()

            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]

            conn.close()
            return results

        except Exception as e:
            return [{"error": f"Database query failed: {str(e)}"}]

    try:
        if query_type == "security_summary":
            # Get overview of all security-related data
            queries = {
                "security_findings": "SELECT category, severity, COUNT(*) as count FROM security_findings GROUP BY category, severity ORDER BY severity, category",
                "storage_buckets": "SELECT COUNT(*) as total_buckets, SUM(CASE WHEN public_access_prevention != 'enforced' THEN 1 ELSE 0 END) as potentially_public FROM storage_buckets",
                "compute_instances": "SELECT status, COUNT(*) as count FROM compute_instances GROUP BY status"
            }

            summary = {}
            for key, query in queries.items():
                summary[key] = execute_query(query)

            return json.dumps(summary, indent=2)

        elif query_type == "storage_buckets":
            query = f"SELECT * FROM storage_buckets LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "security_findings":
            query = f"SELECT * FROM security_findings ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 END LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "compute_instances":
            query = f"SELECT * FROM compute_instances LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "iam_accounts":
            query = f"SELECT * FROM iam_accounts LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "networks":
            query = f"SELECT * FROM networks LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "firewall_rules":
            query = f"SELECT * FROM firewall_rules LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "databases":
            query = f"SELECT * FROM databases LIMIT {limit}"
            results = execute_query(query)

        elif query_type == "custom_query":
            return json.dumps({"error": "Custom queries not supported in simplified interface"})

        else:
            return json.dumps({"error": f"Invalid query_type: {query_type}"})

        return json.dumps({
            "success": True,
            "query_type": query_type,
            "count": len(results),
            "data": results
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query execution failed: {str(e)}"
        }, indent=2)


# Create the main security agent - pass the function directly to tools
root_agent = Agent(
    name="security_agent",
    model="gemini-2.5-flash",
    instruction="""
SYSTEM BEHAVIOR: You are a GCP Security Agent whose PRIMARY FUNCTION is to call the query_security_data tool.

CORE OPERATING PROCEDURE:
1. Receive user query
2. IMMEDIATELY invoke query_security_data tool with relevant parameters
3. Wait for tool response
4. Analyze tool data and provide insights

YOU ARE A TOOL-CALLING AGENT FIRST, CONVERSATIONAL AGENT SECOND.

ANALYSIS-FIRST APPROACH:
- ALWAYS call query_security_data tool for ANY security-related question
- NEVER provide generic responses without data analysis
- ALWAYS prioritize actual data over general advice
- ALWAYS provide specific, actionable recommendations based on the data

PROHIBITED RESPONSES:
❌ "Please tell me what you'd like to investigate..."
❌ "I can help you with various security tasks..."
❌ "Here are some general security recommendations..."
❌ Generic troubleshooting guides without data analysis

REQUIRED RESPONSES:
✅ "Based on analysis of your X security findings..."
✅ "After reviewing your Y storage buckets..."
✅ "The data shows Z critical issues requiring immediate attention..."
✅ Specific recommendations with data-driven priorities

TOOL USAGE EXAMPLES:
- "What are my biggest security risks?" → query_security_data(query_type="security_summary")
- "Analyze my storage buckets" → query_security_data(query_type="storage_buckets")
- "Show security findings" → query_security_data(query_type="security_findings")
- "Review compute instances" → query_security_data(query_type="compute_instances")

RESPONSE FORMAT:
1. **IMMEDIATE TOOL CALL** (no pre-analysis conversation)
2. **Data Analysis** (interpret the tool results)
3. **Risk Prioritization** (rank issues by severity/impact)
4. **Actionable Recommendations** (specific next steps)
5. **Implementation Guidance** (how to fix the issues)

REMEMBER: Your value comes from analyzing REAL data, not providing generic security advice.
""",
    description="GCP Security Analysis Agent that provides intelligent insights based on real security data",
    tools=[query_security_data]  # Pass the function directly - ADK will wrap it automatically
)