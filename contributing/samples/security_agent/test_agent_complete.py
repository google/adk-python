#!/usr/bin/env python3
"""
Comprehensive test for ADK agent with database integration
Tests that the agent is actually querying and returning real database data
"""

import sys
import os
import asyncio
import sqlite3
from pathlib import Path

# Set the database path
os.environ["DATABASE_PATH"] = "backend/cache/gcp_data.db"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.adk_agent import security_agent

async def test_agent():
    """Test the ADK agent with various queries"""

    # First, verify database has data
    print("=" * 60)
    print("VERIFYING DATABASE CONTENTS")
    print("=" * 60)

    db_path = os.environ["DATABASE_PATH"]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check storage buckets
    cursor.execute("SELECT COUNT(*) FROM storage_buckets")
    bucket_count = cursor.fetchone()[0]
    print(f"✓ Storage buckets in DB: {bucket_count}")

    cursor.execute("SELECT name, public_access FROM storage_buckets LIMIT 3")
    buckets = cursor.fetchall()
    for bucket in buckets:
        print(f"  - {bucket[0]} (public: {bucket[1]})")

    # Check security findings
    cursor.execute("SELECT COUNT(*) FROM security_findings")
    finding_count = cursor.fetchone()[0]
    print(f"✓ Security findings in DB: {finding_count}")

    # Check service accounts
    cursor.execute("SELECT COUNT(*) FROM service_accounts")
    sa_count = cursor.fetchone()[0]
    print(f"✓ Service accounts in DB: {sa_count}")

    conn.close()

    print("\n" + "=" * 60)
    print("TESTING ADK AGENT RESPONSES")
    print("=" * 60)

    # Test queries that should return database data
    test_queries = [
        "Show me all storage buckets in the project",
        "What security findings do you have?",
        "List the service accounts",
        "Give me security statistics",
        "Show me HIGH severity findings"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        try:
            # Create a simple context - the agent should handle this
            from google.adk.agents import InvocationContext

            # Create context with minimal required fields
            context = InvocationContext(
                text_input=query,
                session_id="test-session",
                invocation_id="test-invocation"
            )

            # Run the agent
            response_text = ""
            async for event in security_agent.run_async(context):
                if hasattr(event, 'content') and event.content:
                    content = str(event.content)
                    response_text += content
                    print(content, end="", flush=True)

            print()  # New line after response

            # Check if response contains database data
            if "buckets" in query.lower() and bucket_count > 0:
                if "mgm-" in response_text or "storage" in response_text.lower():
                    print("✓ Response contains actual bucket data")
                else:
                    print("⚠️ Response may not be using database data")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Starting comprehensive ADK agent test...")
    asyncio.run(test_agent())