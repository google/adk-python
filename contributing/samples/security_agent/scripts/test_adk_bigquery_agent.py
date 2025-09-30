#!/usr/bin/env python3
"""
Test script for ADK Agent BigQuery Integration
Tests that the ADK agent can query all BigQuery tables in the security_insights dataset
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup environment
os.environ["GOOGLE_CLOUD_PROJECT"] = "mgm-digitalconcierge"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / "config" / "mgm-digitalconcierge-8e6bb83a7e22.json")

from adk.models.llm import LlmAgent
from adk.models.llm.tools.function_tool import FunctionTool
from agents._tools.bigquery_tools import BigQueryTool
import time

def test_bigquery_agent():
    """Test the ADK agent's ability to query BigQuery tables"""

    print("🔧 Initializing BigQuery ADK Agent...")

    # Initialize BigQuery tool
    bq_tool = BigQueryTool()

    # Create agent with BigQuery tool
    agent = LlmAgent(
        name="bigquery_test_agent",
        model="gemini-1.5-flash",
        instruction="""
        You are a BigQuery analyst. When asked about data, query the appropriate BigQuery tables
        in the security_insights dataset. Provide clear, concise summaries of the data.

        Available tables in security_insights dataset:
        - custom_roles: Custom IAM roles and permissions
        - user_roles: Human user IAM bindings
        - service_account_roles: Service account IAM bindings
        - standard_roles: Predefined GCP roles catalog
        """,
        tools=[FunctionTool(bq_tool.execute_query)]
    )

    # Test queries for each table
    test_queries = [
        {
            "table": "custom_roles",
            "query": "How many custom roles are defined in the project? List them if any exist.",
            "sql": "SELECT COUNT(*) as total_custom_roles FROM `mgm-digitalconcierge.security_insights.custom_roles`"
        },
        {
            "table": "user_roles",
            "query": "How many users have IAM access to the project? How many are admins?",
            "sql": "SELECT COUNT(DISTINCT user_email) as unique_users, COUNT(DISTINCT CASE WHEN is_admin THEN user_email END) as admin_users FROM `mgm-digitalconcierge.security_insights.user_roles`"
        },
        {
            "table": "service_account_roles",
            "query": "How many service accounts exist and how many have user-managed keys?",
            "sql": "SELECT COUNT(DISTINCT service_account_email) as total_service_accounts, COUNT(DISTINCT CASE WHEN has_keys THEN service_account_email END) as sa_with_keys FROM `mgm-digitalconcierge.security_insights.service_account_roles`"
        },
        {
            "table": "standard_roles",
            "query": "What are the top 5 most common GCP services in predefined roles?",
            "sql": """
            WITH service_counts AS (
              SELECT service, COUNT(*) as role_count
              FROM `mgm-digitalconcierge.security_insights.standard_roles`
              GROUP BY service
            )
            SELECT service, role_count
            FROM service_counts
            ORDER BY role_count DESC
            LIMIT 5
            """
        }
    ]

    print("\n" + "="*60)
    print("Testing ADK Agent BigQuery Access")
    print("="*60)

    results = []

    for i, test in enumerate(test_queries, 1):
        print(f"\n📊 Test {i}: {test['table']} Table")
        print(f"Question: {test['query']}")
        print(f"SQL: {test['sql'][:100]}...")

        try:
            # Execute query directly through tool
            tool_result = bq_tool.execute_query(test['sql'])

            if tool_result.get("success"):
                print(f"✅ Direct query successful!")
                print(f"   Rows returned: {len(tool_result.get('data', []))}")
                if tool_result.get('data'):
                    print(f"   Sample data: {tool_result['data'][:2]}")
            else:
                print(f"❌ Direct query failed: {tool_result.get('error')}")

            # Test through agent
            print(f"\n🤖 Testing through ADK Agent...")
            agent_response = agent.invoke(test['query'])

            if agent_response:
                print(f"✅ Agent query successful!")
                print(f"   Response preview: {str(agent_response)[:200]}...")
                results.append({
                    "table": test['table'],
                    "status": "SUCCESS",
                    "direct_query": tool_result.get("success", False),
                    "agent_query": True
                })
            else:
                print(f"⚠️ Agent returned empty response")
                results.append({
                    "table": test['table'],
                    "status": "PARTIAL",
                    "direct_query": tool_result.get("success", False),
                    "agent_query": False
                })

        except Exception as e:
            print(f"❌ Error testing {test['table']}: {e}")
            results.append({
                "table": test['table'],
                "status": "FAILED",
                "error": str(e)
            })

        time.sleep(1)  # Rate limiting

    # Summary
    print("\n" + "="*60)
    print("📈 Test Summary")
    print("="*60)

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")

    print(f"✅ Successful: {success_count}/{len(test_queries)}")
    print(f"⚠️ Partial: {partial_count}/{len(test_queries)}")
    print(f"❌ Failed: {failed_count}/{len(test_queries)}")

    print("\nDetailed Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "SUCCESS" else "⚠️" if result["status"] == "PARTIAL" else "❌"
        print(f"{status_icon} {result['table']}: {result['status']}")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        else:
            print(f"   Direct Query: {'✅' if result.get('direct_query') else '❌'}")
            print(f"   Agent Query: {'✅' if result.get('agent_query') else '❌'}")

    return results

if __name__ == "__main__":
    try:
        results = test_bigquery_agent()

        # Exit code based on results
        if all(r["status"] == "SUCCESS" for r in results):
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        elif any(r["status"] == "SUCCESS" or r["status"] == "PARTIAL" for r in results):
            print("\n⚠️ Some tests passed, but there were issues.")
            sys.exit(1)
        else:
            print("\n❌ All tests failed.")
            sys.exit(2)

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(3)