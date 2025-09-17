"""
ADK Security Agent - Google ADK Compliant Implementation
Uses proper LLM reasoning with custom function tools and built-in Google Search
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)

# Database path configuration
DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "backend/cache/gcp_data.db"
)

@FunctionTool
def query_security_data(
    query_type: str,
    severity: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Query the GCP security database for findings, statistics, buckets, and service accounts.

    Args:
        query_type: Type of query - 'security_findings', 'statistics', 'storage_buckets', 'service_accounts'
        severity: Filter by severity level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
        category: Filter by category
        limit: Maximum number of results (default: 10)

    Returns:
        Dictionary containing query results with security analysis
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if query_type == "security_findings":
            return _query_security_findings(cursor, severity, category, limit)
        elif query_type == "statistics":
            return _query_statistics(cursor)
        elif query_type == "storage_buckets":
            return _query_storage_buckets(cursor, limit)
        elif query_type == "service_accounts":
            return _query_service_accounts(cursor, limit)
        else:
            return {"error": f"Invalid query type: {query_type}"}

    except Exception as e:
        logger.error(f"Database error: {e}")
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


def _query_security_findings(cursor, severity: Optional[str], category: Optional[str], limit: int) -> Dict[str, Any]:
    """Query security findings with filters."""
    query = "SELECT * FROM security_findings WHERE 1=1"
    params = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    if category:
        query += " AND category = ?"
        params.append(category)

    query += f" LIMIT {limit}"

    cursor.execute(query, params)
    findings = [dict(row) for row in cursor.fetchall()]

    return {
        "findings": findings,
        "count": len(findings),
        "filters_applied": {"severity": severity, "category": category, "limit": limit}
    }


def _query_statistics(cursor) -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    stats = {}

    # Count findings by severity
    cursor.execute("""
        SELECT severity, COUNT(*) as count
        FROM security_findings
        GROUP BY severity
    """)
    stats["findings_by_severity"] = {row["severity"]: row["count"] for row in cursor.fetchall()}

    # Count findings by category
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM security_findings
        GROUP BY category
    """)
    stats["findings_by_category"] = {row["category"]: row["count"] for row in cursor.fetchall()}

    # Total counts
    cursor.execute("SELECT COUNT(*) as count FROM security_findings")
    stats["total_findings"] = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM service_accounts")
    stats["total_service_accounts"] = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM storage_buckets")
    stats["total_storage_buckets"] = cursor.fetchone()["count"]

    # Count buckets by public access
    cursor.execute("""
        SELECT public_access, COUNT(*) as count
        FROM storage_buckets
        GROUP BY public_access
    """)
    stats["buckets_by_public_access"] = {row["public_access"]: row["count"] for row in cursor.fetchall()}

    return stats


def _query_storage_buckets(cursor, limit: int) -> Dict[str, Any]:
    """Query storage buckets with comprehensive security analysis."""
    query = f"SELECT * FROM storage_buckets LIMIT {limit}"
    cursor.execute(query)
    buckets = [dict(row) for row in cursor.fetchall()]

    # Analyze security posture
    public_buckets = []
    private_buckets = []
    security_issues = []

    for bucket in buckets:
        is_public = bucket.get('public_access', '').lower() == 'public'
        if is_public:
            public_buckets.append(bucket)
            security_issues.append({
                "type": "PUBLIC_BUCKET",
                "severity": "HIGH",
                "resource": bucket.get('name'),
                "description": f"Bucket '{bucket.get('name')}' allows public access",
                "recommendation": "Remove public access and enable access prevention",
                "location": bucket.get('location'),
                "storage_class": bucket.get('storage_class')
            })
        else:
            private_buckets.append(bucket)

    return {
        "storage_buckets": buckets,
        "count": len(buckets),
        "security_analysis": {
            "public_buckets": len(public_buckets),
            "private_buckets": len(private_buckets),
            "security_issues": security_issues,
            "risk_level": "HIGH" if public_buckets else "LOW"
        }
    }


def _query_service_accounts(cursor, limit: int) -> Dict[str, Any]:
    """Query service accounts."""
    query = f"SELECT * FROM service_accounts LIMIT {limit}"
    cursor.execute(query)
    accounts = [dict(row) for row in cursor.fetchall()]

    return {
        "service_accounts": accounts,
        "count": len(accounts)
    }


# Create the ADK Security Agent with LLM reasoning
# NOTE: For ADK eval, temporarily using only database tool due to Gemini tool mixing restrictions
security_agent = Agent(
    name="GCP_Security_Agent",  # ADK requires valid identifier (no spaces)
    description="A security agent for GCP, capable of querying security data and performing Google searches.",
    tools=[query_security_data, google_search]
)