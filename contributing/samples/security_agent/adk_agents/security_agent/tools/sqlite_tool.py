#!/usr/bin/env python3
"""
SQLite Database Tool for GCP Security Data Analysis
==================================================

This tool provides interface to query security-related data from SQLite database
containing GCP resources and security findings.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.adk.core import Tool

def get_database_path() -> str:
    """Get the path to the SQLite database."""
    # Look for database in multiple possible locations
    possible_paths = [
        "/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/backend/cache/gcp_data.db",
        "./backend/cache/gcp_data.db",
        "../backend/cache/gcp_data.db",
        "../../backend/cache/gcp_data.db",
        "../../../backend/cache/gcp_data.db"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If not found, return the primary path (might need creation)
    return "/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/backend/cache/gcp_data.db"

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

class SecurityDataTool(Tool):
    """
    Tool for querying GCP security data from SQLite database.

    This tool provides access to:
    - Storage buckets and their security configurations
    - Security findings from various GCP services
    - Compute instances and their status
    - IAM accounts and role assignments
    - Network configurations and firewall rules
    - Database instances and security settings
    """

    def __init__(self):
        super().__init__(
            name="query_security_data",
            description="Query GCP security data from SQLite database. Use this tool to get information about storage buckets, security findings, compute instances, IAM accounts, networks, and databases.",
            parameters={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "security_summary",
                            "storage_buckets",
                            "security_findings",
                            "compute_instances",
                            "iam_accounts",
                            "networks",
                            "firewall_rules",
                            "databases",
                            "custom_query"
                        ],
                        "description": "Type of security data to query"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string", "description": "Filter by severity (HIGH, MEDIUM, LOW, CRITICAL)"},
                            "resource_type": {"type": "string", "description": "Filter by resource type"},
                            "project_id": {"type": "string", "description": "Filter by project ID"}
                        },
                        "description": "Optional filters to apply to the query"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results to return"
                    },
                    "custom_sql": {
                        "type": "string",
                        "description": "Custom SQL query (only used when query_type is 'custom_query')"
                    }
                },
                "required": ["query_type"]
            }
        )

    def execute(self, query_type: str, filters: Optional[Dict] = None, limit: int = 10, custom_sql: Optional[str] = None) -> str:
        """Execute the security data query based on the specified type."""

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
                query = "SELECT * FROM storage_buckets"
                params = []

                if filters:
                    conditions = []
                    if filters.get("project_id"):
                        conditions.append("project_id = ?")
                        params.append(filters["project_id"])

                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)

                query += f" LIMIT {limit}"
                results = execute_query(query, tuple(params))

            elif query_type == "security_findings":
                query = "SELECT * FROM security_findings"
                params = []

                if filters:
                    conditions = []
                    if filters.get("severity"):
                        conditions.append("severity = ?")
                        params.append(filters["severity"])
                    if filters.get("project_id"):
                        conditions.append("project_id = ?")
                        params.append(filters["project_id"])

                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)

                query += f" ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 END LIMIT {limit}"
                results = execute_query(query, tuple(params))

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

            elif query_type == "custom_query" and custom_sql:
                # Only allow SELECT statements for security
                if not custom_sql.strip().upper().startswith("SELECT"):
                    return json.dumps({"error": "Only SELECT queries are allowed"})
                results = execute_query(custom_sql)

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

# Function-based interface for backward compatibility
def query_security_data(query_type: str, filters: Optional[Dict] = None, limit: int = 10, custom_sql: Optional[str] = None) -> str:
    """Function interface for querying security data."""
    tool = SecurityDataTool()
    return tool.execute(query_type, filters, limit, custom_sql)