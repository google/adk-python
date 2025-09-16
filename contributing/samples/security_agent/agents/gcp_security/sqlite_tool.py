"""
SQLite Tool for GCP Security Agent
Provides database query capabilities for security analysis.
"""

import sqlite3
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import ADK tools for compatibility
try:
    from google.adk.tools import FunctionTool
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

logger = logging.getLogger(__name__)

class SQLiteTool:
    """SQLite database tool for security queries."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite tool with database path."""
        if db_path is None:
            db_path = os.getenv(
                "DATABASE_PATH",
                "/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/backend/cache/gcp_data.db"
            )
        # Remove any quotes that might be in the environment variable
        if db_path:
            db_path = db_path.strip('"').strip("'")
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            logger.warning(f"Database file not found: {self.db_path}")
            # Create empty database if it doesn't exist
            self._create_empty_database()

    def _create_empty_database(self):
        """Create an empty database with basic tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create basic security_findings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_findings (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    severity TEXT,
                    state TEXT,
                    resource_name TEXT,
                    description TEXT,
                    recommendation TEXT,
                    event_time TEXT,
                    data TEXT
                )
            """)

            conn.commit()
            conn.close()
            logger.info(f"Created empty database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Execute a SQL query on the database.

        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries

        Returns:
            Dictionary containing query results or error information
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Check if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                # Convert Row objects to dictionaries
                results = []
                for row in rows:
                    results.append(dict(row))

                conn.close()
                return {
                    "success": True,
                    "data": results,
                    "row_count": len(results)
                }
            else:
                # For INSERT, UPDATE, DELETE queries
                conn.commit()
                affected_rows = cursor.rowcount
                conn.close()
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"Query executed successfully. {affected_rows} rows affected."
                }

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        result = self.execute_query(query)

        if result["success"]:
            return [row["name"] for row in result["data"]]
        return []

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table."""
        query = f"PRAGMA table_info({table_name})"
        result = self.execute_query(query)

        if result["success"]:
            return {
                "table": table_name,
                "columns": result["data"]
            }
        return {"error": result.get("error", "Unknown error")}

    def get_security_findings(self, severity: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        Get security findings from the database.

        Args:
            severity: Optional severity filter (CRITICAL, HIGH, MEDIUM, LOW)
            limit: Maximum number of results to return

        Returns:
            Dictionary containing security findings
        """
        if severity:
            query = "SELECT * FROM security_findings WHERE severity = ? LIMIT ?"
            params = (severity, limit)
        else:
            query = "SELECT * FROM security_findings LIMIT ?"
            params = (limit,)

        return self.execute_query(query, params)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the database."""
        queries = {
            "total_findings": "SELECT COUNT(*) as count FROM security_findings",
            "severity_breakdown": """
                SELECT severity, COUNT(*) as count
                FROM security_findings
                GROUP BY severity
            """,
            "category_breakdown": """
                SELECT category, COUNT(*) as count
                FROM security_findings
                GROUP BY category
            """,
            "total_assets": "SELECT COUNT(*) as count FROM assets",
            "total_buckets": "SELECT COUNT(*) as count FROM storage_buckets",
            "total_firewall_rules": "SELECT COUNT(*) as count FROM firewall_rules"
        }

        stats = {}
        for key, query in queries.items():
            try:
                result = self.execute_query(query)
                if result["success"]:
                    stats[key] = result["data"]
            except:
                stats[key] = None

        return stats

# Create a singleton instance
sqlite_tool_instance = SQLiteTool()

# Create ADK-compatible tool if available
if ADK_AVAILABLE:
    # Wrap the execute_query method as an ADK FunctionTool
    sqlite_tool = FunctionTool(func=sqlite_tool_instance.execute_query)
    get_tables_tool = FunctionTool(func=sqlite_tool_instance.get_tables)
    get_summary_tool = FunctionTool(func=sqlite_tool_instance.get_summary_stats)
else:
    # Fallback to the instance itself
    sqlite_tool = sqlite_tool_instance
    get_tables_tool = None
    get_summary_tool = None

# Export for use
__all__ = ['sqlite_tool', 'SQLiteTool', 'sqlite_tool_instance', 'get_tables_tool', 'get_summary_tool']