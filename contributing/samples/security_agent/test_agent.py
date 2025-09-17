#!/usr/bin/env python3
"""Test the ADK agent with database connection"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the database path
os.environ["DATABASE_PATH"] = "backend/cache/gcp_data.db"

from backend.adk_agent import security_agent
from google.adk.agents import InvocationContext

async def test_agent():
    # Test queries
    queries = [
        "Show me all storage buckets in the project",
        "What security findings do you have?",
        "List the service accounts"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        try:
            # Create an invocation context
            adk_context = InvocationContext(
                text_input=query,
                session_id="test-session"
            )

            # Run the agent asynchronously
            response_text = ""
            async for event in security_agent.run_async(adk_context):
                if hasattr(event, 'content') and event.content:
                    content = str(event.content)
                    response_text += content
                    print(content, end="", flush=True)

            print()  # New line after response

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())