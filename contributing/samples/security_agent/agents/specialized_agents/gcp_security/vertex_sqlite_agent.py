"""
Vertex AI SQLite Agent for GCP Security
This agent provides security analysis capabilities using SQLite database.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our SQLite tool
from sqlite_tool import sqlite_tool_instance

class SecurityAgent:
    """
    Security Agent that works with SQLite database.
    Compatible with ADK-like interface but works without Agent Engine.
    """

    def __init__(self):
        self.name = "gcp_security_analyst"
        self.db_tool = sqlite_tool_instance
        logger.info("✅ Initialized Security Agent with SQLite database access")

        # Test database connection
        tables = self.db_tool.get_tables()
        if tables:
            logger.info(f"   Database has {len(tables)} tables available")
        else:
            logger.warning("   Database appears empty or inaccessible")

    def execute_tool(self, query_type: str = "sql", **kwargs) -> Dict[str, Any]:
        """
        Execute database operations.

        Args:
            query_type: Type of operation - 'sql', 'list_tables', 'get_stats', 'get_schema', 'get_findings'
            **kwargs: Additional parameters based on query_type
        """
        try:
            if query_type == "sql":
                query = kwargs.get('query')
                if not query:
                    return {"success": False, "error": "SQL query required"}
                return self.db_tool.execute_query(query)

            elif query_type == "list_tables":
                tables = self.db_tool.get_tables()
                return {"success": True, "tables": tables, "count": len(tables)}

            elif query_type == "get_stats":
                stats = self.db_tool.get_summary_stats()
                return {"success": True, "stats": stats}

            elif query_type == "get_schema":
                table_name = kwargs.get('table_name')
                if not table_name:
                    return {"success": False, "error": "Table name required"}
                return self.db_tool.get_table_schema(table_name)

            elif query_type == "get_findings":
                severity = kwargs.get('severity')
                limit = kwargs.get('limit', 100)
                return self.db_tool.get_security_findings(severity, limit)

            else:
                return {"success": False, "error": f"Unknown query_type: {query_type}"}

        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return {"success": False, "error": str(e)}

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Process a user prompt and execute appropriate database operations.

        This is a simplified interface that can parse basic commands.
        """
        prompt_lower = prompt.lower()

        # Handle different types of requests
        if "list tables" in prompt_lower or "show tables" in prompt_lower:
            return self.execute_tool("list_tables")

        elif "statistics" in prompt_lower or "summary" in prompt_lower:
            return self.execute_tool("get_stats")

        elif "schema" in prompt_lower:
            # Extract table name if mentioned
            words = prompt.split()
            for i, word in enumerate(words):
                if word.lower() in ["schema", "describe"] and i + 1 < len(words):
                    table_name = words[i + 1].strip("'\"`,;")
                    return self.execute_tool("get_schema", table_name=table_name)
            return {"success": False, "error": "Please specify a table name"}

        elif "findings" in prompt_lower:
            # Check for severity
            severity = None
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if sev.lower() in prompt_lower:
                    severity = sev
                    break
            return self.execute_tool("get_findings", severity=severity)

        elif "select" in prompt_lower or "insert" in prompt_lower or "update" in prompt_lower:
            # It's likely a SQL query
            return self.execute_tool("sql", query=prompt)

        else:
            # Default to showing available options
            return {
                "success": True,
                "message": "Security Agent ready. Available operations:",
                "operations": [
                    "list tables - Show all database tables",
                    "get statistics - Show summary statistics",
                    "get schema [table] - Show table structure",
                    "get findings [severity] - Show security findings",
                    "SQL queries - Execute any SQL query directly"
                ],
                "example_queries": [
                    "SELECT * FROM security_findings WHERE severity = 'HIGH' LIMIT 10",
                    "SELECT COUNT(*) FROM storage_buckets WHERE public_access = true",
                    "SELECT * FROM iam_accounts WHERE has_service_account_keys = true"
                ]
            }

    def __call__(self, *args, **kwargs):
        """Make the agent callable."""
        if args:
            return self.run(args[0])
        elif 'prompt' in kwargs:
            return self.run(kwargs['prompt'])
        else:
            return self.run("")


# Try to import ADK for compatibility, but fall back to our custom agent
try:
    from google.adk.agents import LlmAgent
    from google.adk.agents.llm_agent_config import LlmAgentConfig
    from google.adk.tools import FunctionTool

    # If imports succeed but we're not on Agent Engine, use custom agent
    logger.info("ADK available but using custom agent for local execution")
    root_agent = SecurityAgent()

except ImportError:
    logger.info("ADK not available, using custom Security Agent")
    root_agent = SecurityAgent()

# Export the agent
__all__ = ['root_agent']

if __name__ == "__main__":
    # Test the agent
    agent = root_agent
    print(f"✅ Agent initialized: {agent.name}")
    print("   Type: SecurityAgent (SQLite-compatible)")

    # Test a simple operation
    result = agent.execute_tool("list_tables")
    if result.get("success"):
        print(f"   Database has {result.get('count', 0)} tables")

    print("\n📊 Ready for security analysis!")
    print("   - Database connected")
    print("   - Single unified tool approach (works without Agent Engine)")
    print("   - ADK-compatible interface")