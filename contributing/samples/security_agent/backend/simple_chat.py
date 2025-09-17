"""
Simple chat endpoint for the security agent.
Directly queries the SQLite database for security findings and service accounts.
"""

import sqlite3
import os
from typing import Dict, Any


def process_chat_query(query: str, context: str = "general") -> Dict[str, Any]:
    """Process a chat query and return results from the database."""

    # Database path
    db_path = os.path.join(
        os.path.dirname(__file__),
        "cache",
        "gcp_data.db"
    )

    if not os.path.exists(db_path):
        return {
            "response": "Database not found. Please run fetch_real_data.py first.",
            "success": False
        }

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query_lower = query.lower()

        # Handle different types of queries
        if "finding" in query_lower or "security" in query_lower:
            cursor.execute("""
                SELECT severity, category, name, resource_name, state, recommendation
                FROM security_findings
                LIMIT 10
            """)

            findings = cursor.fetchall()

            if findings:
                response = f"I found {len(findings)} security findings in your GCP project:\n\n"

                for severity, category, name, resource, state, recommendation in findings:
                    response += f"**{severity}** - {category}\n"
                    response += f"  • Finding: {name}\n"
                    response += f"  • Resource: {resource}\n"
                    response += f"  • State: {state}\n"
                    if recommendation:
                        response += f"  • Recommendation: {recommendation}\n"
                    response += "\n"

                return {"response": response, "success": True}
            else:
                return {"response": "No security findings found in the database.", "success": True}

        elif "service account" in query_lower or "iam" in query_lower:
            cursor.execute("""
                SELECT email, display_name, project_id
                FROM service_accounts
                LIMIT 10
            """)

            accounts = cursor.fetchall()

            if accounts:
                response = f"I found {len(accounts)} service accounts:\n\n"

                for email, display_name, project_id in accounts:
                    response += f"• **{email}**\n"
                    if display_name:
                        response += f"  Name: {display_name}\n"
                    response += f"  Project: {project_id}\n\n"

                return {"response": response, "success": True}
            else:
                return {"response": "No service accounts found in the database.", "success": True}

        elif "count" in query_lower or "how many" in query_lower:
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM security_findings")
            findings_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM service_accounts")
            accounts_count = cursor.fetchone()[0]

            response = f"Database contains:\n"
            response += f"• **{findings_count}** security findings\n"
            response += f"• **{accounts_count}** service accounts\n"

            return {"response": response, "success": True}

        else:
            # Default response with available queries
            response = "I can help you with:\n\n"
            response += "• **Security findings** - Ask about security issues in your GCP project\n"
            response += "• **Service accounts** - View IAM service accounts\n"
            response += "• **Counts** - Get statistics on your GCP resources\n\n"
            response += "Try asking: 'Show me security findings' or 'List service accounts'"

            return {"response": response, "success": True}

    except Exception as e:
        return {
            "response": f"Error querying database: {str(e)}",
            "success": False
        }
    finally:
        if conn:
            conn.close()