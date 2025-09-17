"""
Database Tool for ADK Agent
Provides structured database query capabilities following ADK tool patterns
"""

import sqlite3
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatabaseTool:
    """ADK-compliant database tool for security queries."""

    def __init__(self):
        """Initialize database tool with connection."""
        db_path = os.getenv(
            "DATABASE_PATH",
            "backend/cache/gcp_data.db"
        )
        self.db_path = Path(db_path)
        self.name = "database_query"
        self.description = "Query the GCP security database for findings, service accounts, and statistics"

    def get_schema(self) -> Dict[str, Any]:
        """Return ADK tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["security_findings", "service_accounts", "statistics", "storage_buckets", "custom_sql"],
                        "description": "Type of database query to perform"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "severity": {
                                "type": "string",
                                "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                "description": "Filter by severity level"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results"
                            },
                            "category": {
                                "type": "string",
                                "description": "Filter by category"
                            }
                        }
                    },
                    "sql": {
                        "type": "string",
                        "description": "Custom SQL query (only for custom_sql type)"
                    }
                },
                "required": ["query_type"]
            }
        }

    def execute(self, query_type: str, filters: Optional[Dict] = None, sql: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute database query.

        Args:
            query_type: Type of query to perform
            filters: Optional filters for the query
            sql: Custom SQL (only for custom_sql type)

        Returns:
            Query results
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if query_type == "security_findings":
                return self._query_security_findings(cursor, filters or {})
            elif query_type == "service_accounts":
                return self._query_service_accounts(cursor, filters or {})
            elif query_type == "statistics":
                return self._query_statistics(cursor)
            elif query_type == "storage_buckets":
                return self._query_storage_buckets(cursor, filters or {})
            elif query_type == "custom_sql" and sql:
                return self._execute_custom_sql(cursor, sql)
            else:
                return {"error": f"Invalid query type: {query_type}"}

        except Exception as e:
            logger.error(f"Database error: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()

    def _query_security_findings(self, cursor, filters: Dict) -> Dict[str, Any]:
        """Query security findings with filters."""
        query = "SELECT * FROM security_findings WHERE 1=1"
        params = []

        if filters.get("severity"):
            query += " AND severity = ?"
            params.append(filters["severity"])

        if filters.get("category"):
            query += " AND category = ?"
            params.append(filters["category"])

        query += f" LIMIT {filters.get('limit', 100)}"

        cursor.execute(query, params)
        findings = [dict(row) for row in cursor.fetchall()]

        return {
            "findings": findings,
            "count": len(findings),
            "filters_applied": filters
        }

    def _query_service_accounts(self, cursor, filters: Dict) -> Dict[str, Any]:
        """Query service accounts."""
        query = f"SELECT * FROM service_accounts LIMIT {filters.get('limit', 100)}"
        cursor.execute(query)
        accounts = [dict(row) for row in cursor.fetchall()]

        return {
            "service_accounts": accounts,
            "count": len(accounts)
        }

    def _query_statistics(self, cursor) -> Dict[str, Any]:
        """Get database statistics."""
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

        # Count storage buckets
        cursor.execute("SELECT COUNT(*) as count FROM storage_buckets")
        stats["total_storage_buckets"] = cursor.fetchone()["count"]

        # Count buckets by storage class
        cursor.execute("""
            SELECT storage_class, COUNT(*) as count
            FROM storage_buckets
            GROUP BY storage_class
        """)
        stats["buckets_by_storage_class"] = {row["storage_class"]: row["count"] for row in cursor.fetchall()}

        # Count buckets by public access
        cursor.execute("""
            SELECT public_access, COUNT(*) as count
            FROM storage_buckets
            GROUP BY public_access
        """)
        stats["buckets_by_public_access"] = {row["public_access"]: row["count"] for row in cursor.fetchall()}

        return stats

    def _query_storage_buckets(self, cursor, filters: Dict) -> Dict[str, Any]:
        """Query storage buckets with filters."""
        query = "SELECT * FROM storage_buckets WHERE 1=1"
        params = []

        if filters.get("public_access"):
            query += " AND public_access = ?"
            params.append(filters["public_access"])

        if filters.get("storage_class"):
            query += " AND storage_class = ?"
            params.append(filters["storage_class"])

        if filters.get("name"):
            query += " AND name LIKE ?"
            params.append(f"%{filters['name']}%")

        query += f" LIMIT {filters.get('limit', 100)}"

        cursor.execute(query, params)
        buckets = [dict(row) for row in cursor.fetchall()]

        return {
            "storage_buckets": buckets,
            "count": len(buckets),
            "filters_applied": filters
        }

    def _execute_custom_sql(self, cursor, sql: str) -> Dict[str, Any]:
        """Execute custom SQL query."""
        cursor.execute(sql)

        if sql.strip().upper().startswith("SELECT"):
            results = [dict(row) for row in cursor.fetchall()]
            return {"results": results, "count": len(results)}
        else:
            return {"affected_rows": cursor.rowcount}

# Create singleton instance
database_tool = DatabaseTool()