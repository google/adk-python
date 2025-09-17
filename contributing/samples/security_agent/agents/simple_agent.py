"""
Simple agent that uses the database query tool directly with LLM-like reasoning
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def process_query_with_reasoning(query: str) -> str:
    """
    Process a query with reasoning about what data to fetch.
    This simulates LLM reasoning while directly using the database tool.
    """
    try:
        # Import the underlying function (not the FunctionTool decorated version)
        from backend.adk_agent import _query_security_findings, _query_statistics, _query_storage_buckets, _query_service_accounts
        import sqlite3
        import os

        DATABASE_PATH = os.getenv(
            "DATABASE_PATH",
            "backend/cache/gcp_data.db"
        )

        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query_lower = query.lower()
        response = []

        # Reasoning: What is the user asking for?
        response.append("🤖 **ADK Agent Analysis**\n")
        response.append(f"Query: *{query}*\n\n")

        # Determine what data to fetch based on query
        if 'bucket' in query_lower or 'storage' in query_lower:
            response.append("**Reasoning:** Detected storage/bucket query. Fetching storage bucket data...\n\n")

            # Query storage buckets
            result = _query_storage_buckets(cursor, limit=20)

            if result.get('storage_buckets'):
                buckets = result['storage_buckets']
                analysis = result.get('security_analysis', {})

                response.append(f"📦 **Found {len(buckets)} storage buckets in project mgm-digitalconcierge**\n\n")

                for bucket in buckets[:10]:
                    response.append(f"• **{bucket.get('name', 'Unknown')}**\n")
                    response.append(f"  - Location: {bucket.get('location', 'N/A')}\n")
                    response.append(f"  - Storage Class: {bucket.get('storage_class', 'N/A')}\n")
                    response.append(f"  - Access: {bucket.get('public_access', 'N/A')}\n")
                    if bucket.get('public_access', '').lower() == 'public':
                        response.append(f"  🚨 **WARNING: Public access enabled**\n")
                    response.append("\n")

                # Add security analysis
                if analysis.get('security_issues'):
                    response.append("\n⚠️ **Security Issues Found:**\n")
                    for issue in analysis['security_issues']:
                        response.append(f"• {issue.get('description', 'Unknown issue')}\n")
                        response.append(f"  Recommendation: {issue.get('recommendation', 'N/A')}\n\n")
            else:
                response.append("No storage buckets found in the database.\n")

        elif 'finding' in query_lower or 'security' in query_lower:
            response.append("**Reasoning:** Detected security/findings query. Fetching security findings...\n\n")

            # Query security findings
            result = _query_security_findings(cursor, severity=None, category=None, limit=10)

            if result.get('findings'):
                findings = result['findings']
                response.append(f"🔍 **Found {len(findings)} security findings**\n\n")

                for finding in findings:
                    response.append(f"• **{finding.get('resource_type', 'Unknown')}**\n")
                    response.append(f"  - Severity: {finding.get('severity', 'N/A')}\n")
                    response.append(f"  - Category: {finding.get('category', 'N/A')}\n")
                    response.append(f"  - Description: {finding.get('description', 'N/A')}\n\n")
            else:
                response.append("No security findings in the database.\n")

        elif 'service account' in query_lower:
            response.append("**Reasoning:** Detected service account query. Fetching service account data...\n\n")

            # Query service accounts
            result = _query_service_accounts(cursor, limit=10)

            if result.get('service_accounts'):
                accounts = result['service_accounts']
                response.append(f"👤 **Found {len(accounts)} service accounts**\n\n")

                for account in accounts:
                    response.append(f"• **{account.get('email', 'Unknown')}**\n")
                    response.append(f"  - Display Name: {account.get('display_name', 'N/A')}\n")
                    response.append(f"  - Project: {account.get('project_id', 'N/A')}\n\n")
            else:
                response.append("No service accounts found in the database.\n")

        else:
            response.append("**Reasoning:** General query. Fetching overall security statistics...\n\n")

            # Default to statistics
            result = _query_statistics(cursor)

            if result:
                response.append("📊 **Security Statistics for mgm-digitalconcierge:**\n\n")
                response.append(f"• Total Storage Buckets: {result.get('total_storage_buckets', 0)}\n")
                response.append(f"• Total Security Findings: {result.get('total_findings', 0)}\n")
                response.append(f"• Total Service Accounts: {result.get('total_service_accounts', 0)}\n\n")

                if result.get('findings_by_severity'):
                    response.append("**Findings by Severity:**\n")
                    for sev, count in result['findings_by_severity'].items():
                        response.append(f"  • {sev}: {count}\n")
                    response.append("\n")

                if result.get('buckets_by_public_access'):
                    response.append("**Buckets by Access:**\n")
                    for access, count in result['buckets_by_public_access'].items():
                        response.append(f"  • {access}: {count}\n")

        conn.close()

        # Add security recommendations
        response.append("\n\n💡 **Recommendations:**\n")
        response.append("• Review and restrict public bucket access\n")
        response.append("• Investigate HIGH severity findings\n")
        response.append("• Audit service account permissions\n")

        return ''.join(response)

    except Exception as e:
        logger.error(f"Error in simple agent: {e}")
        return f"Error processing query: {str(e)}"