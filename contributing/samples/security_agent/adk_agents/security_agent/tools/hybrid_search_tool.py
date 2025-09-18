#!/usr/bin/env python3
"""
Hybrid Security Search Tool - Database + Web Search
===================================================

This tool provides unified search capability for both:
1. Internal security data from SQLite database
2. External security intelligence via web search

Works around Gemini's limitation that multiple tools must all be search tools.
"""

import sqlite3
import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

def search_security_information(query: str, search_type: str = "auto") -> str:
    """
    Unified security search tool that handles both database queries and web searches.

    Args:
        query: The search query or question
        search_type: "database", "web", or "auto" to determine search approach

    Returns:
        JSON string with search results from database, web, or both
    """

    def get_database_path() -> str:
        """Get the path to the SQLite database."""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent

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

    def search_database(query_type: str) -> Dict[str, Any]:
        """Search internal security database."""
        db_path = get_database_path()

        if not os.path.exists(db_path):
            return {"error": f"Database not found at {db_path}"}

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if query_type == "security_summary":
                queries = {
                    "security_findings": "SELECT category, severity, COUNT(*) as count FROM security_findings GROUP BY category, severity ORDER BY severity, category",
                    "storage_buckets": "SELECT COUNT(*) as total_buckets, SUM(CASE WHEN public_access_prevention != 'enforced' THEN 1 ELSE 0 END) as potentially_public FROM storage_buckets",
                    "compute_instances": "SELECT status, COUNT(*) as count FROM compute_instances GROUP BY status"
                }

                summary = {}
                for key, sql in queries.items():
                    cursor.execute(sql)
                    summary[key] = [dict(row) for row in cursor.fetchall()]

                conn.close()
                return {"source": "database", "type": "security_summary", "data": summary}

            elif query_type == "storage_buckets":
                cursor.execute("SELECT * FROM storage_buckets LIMIT 10")
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return {"source": "database", "type": "storage_buckets", "count": len(results), "data": results}

            elif query_type == "security_findings":
                cursor.execute("SELECT * FROM security_findings ORDER BY CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 END LIMIT 10")
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return {"source": "database", "type": "security_findings", "count": len(results), "data": results}

            else:
                conn.close()
                return {"error": f"Unknown database query type: {query_type}"}

        except Exception as e:
            return {"error": f"Database query failed: {str(e)}"}

    def search_web(web_query: str) -> Dict[str, Any]:
        """Search web for security intelligence (simulated)."""
        # In a real implementation, this would use Google Search API
        # For now, return simulated web search results
        simulated_results = {
            "source": "web_search",
            "query": web_query,
            "results": [
                {
                    "title": "Latest GCP Security Best Practices 2024",
                    "url": "https://cloud.google.com/security/best-practices",
                    "snippet": "Updated security guidelines for Google Cloud Platform including storage bucket hardening, IAM least privilege, and network security configurations."
                },
                {
                    "title": "Recent GCP Security Vulnerabilities and Patches",
                    "url": "https://cloud.google.com/support/bulletins",
                    "snippet": "Security bulletins covering recent vulnerabilities in GCP services and recommended mitigation strategies."
                },
                {
                    "title": "Cloud Security Threat Intelligence 2024",
                    "url": "https://security.googleblog.com/",
                    "snippet": "Analysis of current threat landscape affecting cloud infrastructure, including storage misconfigurations and access control bypasses."
                }
            ]
        }
        return simulated_results

    # Determine search strategy
    query_lower = query.lower()

    try:
        if search_type == "database" or (search_type == "auto" and any(term in query_lower for term in [
            "my security", "my buckets", "my environment", "analyze my", "what are my", "show my", "security summary"
        ])):
            # Database search for internal data
            if "security risk" in query_lower or "security summary" in query_lower:
                results = search_database("security_summary")
            elif "bucket" in query_lower:
                results = search_database("storage_buckets")
            elif "finding" in query_lower:
                results = search_database("security_findings")
            else:
                results = search_database("security_summary")

            return json.dumps(results, indent=2)

        elif search_type == "web" or (search_type == "auto" and any(term in query_lower for term in [
            "latest", "current", "new", "recent", "threat", "vulnerability", "best practice", "advisory"
        ])):
            # Web search for external intelligence
            results = search_web(query)
            return json.dumps(results, indent=2)

        else:
            # Combined search - both database and web
            db_results = search_database("security_summary")
            web_results = search_web(query)

            combined = {
                "source": "combined",
                "query": query,
                "internal_data": db_results,
                "external_intelligence": web_results
            }

            return json.dumps(combined, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }, indent=2)