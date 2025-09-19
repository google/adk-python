#!/usr/bin/env python3
"""
Test script to verify the ADK agent can handle service onboarding queries
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables if not already set
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
os.environ.setdefault("DATABASE_PATH", "backend/cache/gcp_data.db")

from agents.adk_agent import create_agent
from google.adk.common.types import query

def test_service_onboarding_queries():
    """Test various service onboarding related queries"""

    print("🚀 Testing Service Onboarding Agent Capabilities\n")
    print("=" * 60)

    # Create the agent
    print("Creating ADK agent...")
    agent = create_agent()

    # Test queries related to service onboarding
    test_queries = [
        "What APIs are currently enabled in my GCP project?",
        "What new GCP services should I evaluate for adoption?",
        "How can I check if Cloud Run Functions v2 is safe to enable?",
        "What are the security requirements for enabling a new GCP service?",
        "Show me the current IAM roles in the project",
        "What organization policies should I apply before testing new services?",
        "How long does it typically take to evaluate a new GCP service?",
        "What's the process for onboarding AlloyDB in my organization?",
        "List the APIs I need to enable for Vertex AI Vision",
        "What are the risk factors when adopting preview services?"
    ]

    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {test_query}")
        print("=" * 60)

        try:
            # Create query object
            user_query = query.Query(text=test_query)

            # Get response from agent
            response = agent.query(user_query)

            # Extract and display the response
            if hasattr(response, 'text'):
                print(f"\n✅ Agent Response:\n{response.text}")
            elif hasattr(response, 'content'):
                if hasattr(response.content, 'text'):
                    print(f"\n✅ Agent Response:\n{response.content.text}")
                else:
                    print(f"\n✅ Agent Response:\n{response.content}")
            else:
                print(f"\n✅ Agent Response:\n{response}")

            # Check if tools were used
            if hasattr(response, 'metadata') and response.metadata:
                if 'tools_used' in response.metadata:
                    print(f"\n🔧 Tools Used: {response.metadata['tools_used']}")

        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Service Onboarding Agent Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_service_onboarding_queries()