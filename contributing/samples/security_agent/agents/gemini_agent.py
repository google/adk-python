"""
Hybrid Vertex AI Authenticated Security Agent
Uses service account credentials with Google Generative AI for true LLM reasoning
Single tool pattern for database queries with intelligent analysis
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional
from google.auth import default
from google.auth.transport.requests import Request
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Initialize with service account authentication
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")

# Use service account credentials to get API key
def get_authenticated_api_key():
    """Get API key using service account credentials"""
    try:
        # Get default credentials from service account
        credentials, project = default()
        if credentials.token is None:
            credentials.refresh(Request())

        # Use the access token as API key
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get authenticated API key: {e}")
        # Fallback to environment variable if available
        return os.getenv("GOOGLE_API_KEY")

# Initialize Gemini with authenticated credentials
api_key = get_authenticated_api_key()
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("No valid API key found - Gemini agent may not work")

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "backend/cache/gcp_data.db"
)

def query_security_database(
    query_type: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Comprehensive security database query tool for GCP security analysis.

    This is the SINGLE TOOL that provides access to all security data.
    The LLM will use this to gather information and provide analysis.

    Args:
        query_type: Type of data to query. Options:
            - "storage_buckets": Get storage bucket information
            - "security_findings": Get security vulnerabilities and findings
            - "service_accounts": Get service account information
            - "iam_policies": Get IAM policy information
            - "statistics": Get overall security statistics
            - "custom_sql": Execute custom SQL query (for complex analysis)
        filters: Optional filters to apply:
            - severity: Filter findings by severity (HIGH, MEDIUM, LOW)
            - category: Filter findings by category
            - project_id: Filter by specific project
            - public_access: Filter buckets by public access (true/false)
            - sql_query: For custom_sql type, the actual SQL query
        limit: Maximum number of records to return (default 20)

    Returns:
        Dictionary containing query results and metadata
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        filters = filters or {}
        result = {"query_type": query_type, "filters": filters, "limit": limit}

        if query_type == "storage_buckets":
            sql = "SELECT * FROM storage_buckets WHERE 1=1"
            params = []

            if filters.get("project_id"):
                sql += " AND project_id = ?"
                params.append(filters["project_id"])

            if filters.get("public_access") is not None:
                access_value = "public" if filters["public_access"] else "private"
                sql += " AND LOWER(public_access) = ?"
                params.append(access_value)

            sql += f" ORDER BY name LIMIT {limit}"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            buckets = [dict(row) for row in rows]
            result["storage_buckets"] = buckets
            result["count"] = len(buckets)

            # Security analysis
            public_count = sum(1 for b in buckets if b.get("public_access", "").lower() == "public")
            result["security_analysis"] = {
                "total_buckets": len(buckets),
                "public_buckets": public_count,
                "private_buckets": len(buckets) - public_count,
                "risk_level": "HIGH" if public_count > 0 else "LOW"
            }

        elif query_type == "security_findings":
            sql = "SELECT * FROM security_findings WHERE 1=1"
            params = []

            if filters.get("severity"):
                sql += " AND UPPER(severity) = ?"
                params.append(filters["severity"].upper())

            if filters.get("category"):
                sql += " AND category = ?"
                params.append(filters["category"])

            if filters.get("project_id"):
                sql += " AND project_id = ?"
                params.append(filters["project_id"])

            sql += f" ORDER BY CASE severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 END LIMIT {limit}"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            findings = [dict(row) for row in rows]
            result["security_findings"] = findings
            result["count"] = len(findings)

            # Security analysis by severity
            severity_counts = {}
            for finding in findings:
                sev = finding.get("severity", "UNKNOWN").upper()
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            result["security_analysis"] = {
                "severity_breakdown": severity_counts,
                "high_priority_count": severity_counts.get("HIGH", 0),
                "total_findings": len(findings),
                "risk_level": "HIGH" if severity_counts.get("HIGH", 0) > 0 else "MEDIUM" if len(findings) > 0 else "LOW"
            }

        elif query_type == "service_accounts":
            sql = "SELECT * FROM service_accounts WHERE 1=1"
            params = []

            if filters.get("project_id"):
                sql += " AND project_id = ?"
                params.append(filters["project_id"])

            sql += f" ORDER BY email LIMIT {limit}"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            accounts = [dict(row) for row in rows]
            result["service_accounts"] = accounts
            result["count"] = len(accounts)

            result["security_analysis"] = {
                "total_accounts": len(accounts),
                "requires_audit": len(accounts) > 10,  # Flag if many accounts
                "risk_level": "MEDIUM" if len(accounts) > 10 else "LOW"
            }

        elif query_type == "iam_policies":
            sql = "SELECT * FROM iam_policies WHERE 1=1"
            params = []

            if filters.get("project_id"):
                sql += " AND project_id = ?"
                params.append(filters["project_id"])

            if filters.get("resource_type"):
                sql += " AND resource_type = ?"
                params.append(filters["resource_type"])

            sql += f" ORDER BY resource_name LIMIT {limit}"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            policies = [dict(row) for row in rows]
            result["iam_policies"] = policies
            result["count"] = len(policies)

        elif query_type == "statistics":
            # Get comprehensive statistics
            stats = {}

            # Storage buckets stats
            cursor.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN LOWER(public_access) = 'public' THEN 1 END) as public FROM storage_buckets")
            bucket_stats = cursor.fetchone()
            stats["storage_buckets"] = {
                "total": bucket_stats[0],
                "public": bucket_stats[1],
                "private": bucket_stats[0] - bucket_stats[1]
            }

            # Security findings stats
            cursor.execute("SELECT severity, COUNT(*) as count FROM security_findings GROUP BY severity")
            findings_by_severity = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM security_findings")
            total_findings = cursor.fetchone()[0]

            stats["security_findings"] = {
                "total": total_findings,
                "by_severity": findings_by_severity
            }

            # Service accounts stats
            cursor.execute("SELECT COUNT(*) FROM service_accounts")
            stats["service_accounts"] = {"total": cursor.fetchone()[0]}

            # IAM policies stats
            cursor.execute("SELECT COUNT(*) FROM iam_policies")
            stats["iam_policies"] = {"total": cursor.fetchone()[0]}

            result["statistics"] = stats

            # Overall risk assessment
            high_findings = findings_by_severity.get("HIGH", 0)
            public_buckets = stats["storage_buckets"]["public"]

            if high_findings > 0 or public_buckets > 0:
                risk_level = "HIGH"
            elif total_findings > 0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            result["security_analysis"] = {
                "overall_risk_level": risk_level,
                "critical_issues": high_findings + public_buckets,
                "total_assets": stats["storage_buckets"]["total"] + stats["service_accounts"]["total"]
            }

        elif query_type == "custom_sql":
            sql_query = filters.get("sql_query", "")
            if not sql_query or not sql_query.strip().upper().startswith("SELECT"):
                raise ValueError("Custom SQL must be a SELECT statement")

            cursor.execute(sql_query)
            rows = cursor.fetchall()

            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []

            result["custom_query_results"] = [dict(zip(columns, row)) for row in rows]
            result["columns"] = columns
            result["count"] = len(rows)

        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        conn.close()
        return result

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return {
            "error": str(e),
            "query_type": query_type,
            "filters": filters
        }

# Define the function schema for Google Generative AI
FUNCTION_DECLARATION = {
    "name": "query_security_database",
    "description": "Query the GCP security database to retrieve information about storage buckets, security findings, service accounts, IAM policies, and statistics. This is your primary tool for gathering security data.",
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": ["storage_buckets", "security_findings", "service_accounts", "iam_policies", "statistics", "custom_sql"],
                "description": "Type of security data to query"
            },
            "filters": {
                "type": "object",
                "description": "Optional filters to apply to the query",
                "properties": {
                    "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "category": {"type": "string"},
                    "project_id": {"type": "string"},
                    "public_access": {"type": "boolean"},
                    "sql_query": {"type": "string"}
                }
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of records to return",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            }
        },
        "required": ["query_type"]
    }
}

class AuthenticatedGeminiSecurityAgent:
    """
    Service account authenticated Gemini security agent with function calling.
    Provides true LLM reasoning with a single database tool.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=[FUNCTION_DECLARATION]
        )

        # System prompt for security analysis
        self.system_prompt = """You are a GCP Security Analyst AI powered by Gemini. You have access to a comprehensive security database containing information about:

- Storage buckets and their access configurations
- Security findings and vulnerabilities
- Service accounts and their permissions
- IAM policies and access controls
- Overall security statistics

Your role is to:
1. Analyze security queries and determine what data to retrieve
2. Use the query_security_database tool to gather relevant information
3. Provide professional security analysis and recommendations
4. Identify risks, vulnerabilities, and compliance issues
5. Suggest actionable remediation steps

When responding:
- Always provide context about the GCP project (mgm-digitalconcierge)
- Highlight security risks with appropriate severity levels
- Give specific, actionable recommendations
- Use professional security terminology
- Format responses clearly with sections and bullet points

You have ONE TOOL: query_security_database - use it wisely to gather the data you need for thorough analysis."""

    def process_query(self, user_query: str) -> str:
        """
        Process a security query using Vertex AI with function calling.

        Args:
            user_query: The user's security question

        Returns:
            Comprehensive security analysis response
        """
        try:
            # Start a chat session with the system prompt
            chat = self.model.start_chat()

            # Send the user query with system context
            full_prompt = f"{self.system_prompt}\n\nUser Query: {user_query}"
            response = chat.send_message(full_prompt)

            # Handle function calls
            while response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Execute the function call
                        function_call = part.function_call
                        function_name = function_call.name
                        function_args = dict(function_call.args)

                        logger.info(f"Gemini calling function: {function_name} with args: {function_args}")

                        if function_name == "query_security_database":
                            # Execute the database query
                            function_result = query_security_database(**function_args)

                            # Send the result back to Gemini
                            function_response = genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=function_name,
                                    response={"result": function_result}
                                )
                            )
                            response = chat.send_message(function_response)
                        else:
                            # Unknown function
                            response = chat.send_message(f"Unknown function: {function_name}")
                            break
                    else:
                        # No more function calls, return the response
                        return response.text

            return response.text

        except Exception as e:
            logger.error(f"Gemini agent error: {e}")
            return f"Error processing security query: {str(e)}"

# Initialize the agent
authenticated_gemini_agent = AuthenticatedGeminiSecurityAgent()

def process_security_query(query: str) -> str:
    """
    Main entry point for processing security queries.

    Args:
        query: User's security question

    Returns:
        Authenticated Gemini's analysis and recommendations
    """
    return authenticated_gemini_agent.process_query(query)