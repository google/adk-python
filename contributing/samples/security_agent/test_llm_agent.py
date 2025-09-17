#!/usr/bin/env python3
"""
Test the new LLM agent with Tool registration
"""

import os
import sys
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Set environment variables
os.environ["DATABASE_PATH"] = "backend/cache/gcp_data.db"
os.environ["GOOGLE_CLOUD_PROJECT"] = "mgm-digitalconcierge"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_agent():
    """Test the ADK agent with direct function calls"""
    print("=" * 60)
    print("TESTING ADK AGENT")
    print("=" * 60)

    print(f'GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")}')
    try:
        from agent.__main__ import root_agent

        session_service = InMemorySessionService()
        runner = Runner(session_service=session_service, agent=root_agent, app_name="security-agent")
        USER_ID = "test-user"
        SESSION_ID = "test-session"
        asyncio.run(session_service.create_session(app_name="security-agent", user_id=USER_ID, session_id=SESSION_ID))

        test_queries = [
            "Show me storage buckets in the project",
            "What security findings do you have?",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)

            try:
                # Test the ADK agent
                content = types.Content(
                    role="user",
                    parts=[types.Part(text=query)]
                )
                events = runner.run(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    new_message=content
                )
                for event in events:
                    if event.is_final_response():
                        response = event.content.parts[0].text
                        break

                if response and len(response) > 50:
                    print(f"✓ Response received ({len(response)} chars)")
                    # Show first 500 chars
                    preview = response[:500] + "..." if len(response) > 500 else response
                    print(f"Preview: {preview}")

                    # Check for LLM reasoning indicators
                    if "analysis" in response.lower() or "recommendation" in response.lower():
                        print("✓ Contains LLM reasoning/analysis")
                    if any(x in response for x in ["mgm-", "digitalconcierge", "bucket", "finding", "account"]):
                        print("✓ Contains project-specific data")
                else:
                    print(f"❌ Response too short or empty: {response}")

            except Exception as e:
                print(f"❌ Query failed: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"❌ ADK agent import failed: {e}")
        import traceback
        traceback.print_exc()

    print() 

if __name__ == "__main__":
    print("Testing new LLM agent implementation...")

    test_agent()

    print("Testing complete!")
