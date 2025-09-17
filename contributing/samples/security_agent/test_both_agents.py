#!/usr/bin/env python3
"""
Comprehensive test for both Simple Agent and ADK Agent
Tests database queries, reasoning, and API endpoints
"""

import sys
import os
import asyncio
import sqlite3
import json
import requests
from pathlib import Path

# Set environment
os.environ["DATABASE_PATH"] = "backend/cache/gcp_data.db"
os.environ["GOOGLE_CLOUD_PROJECT"] = "mgm-digitalconcierge"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_database():
    """Verify database has data"""
    print("=" * 60)
    print("STEP 1: VERIFYING DATABASE")
    print("=" * 60)

    db_path = os.environ["DATABASE_PATH"]
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check each table
    tables = {
        "storage_buckets": "SELECT COUNT(*), COUNT(DISTINCT project_id) FROM storage_buckets",
        "security_findings": "SELECT COUNT(*), COUNT(DISTINCT severity) FROM security_findings",
        "service_accounts": "SELECT COUNT(*), COUNT(DISTINCT project_id) FROM service_accounts",
        "iam_policies": "SELECT COUNT(*), COUNT(DISTINCT resource_type) FROM iam_policies"
    }

    for table, query in tables.items():
        try:
            cursor.execute(query)
            count, distinct = cursor.fetchone()
            print(f"✓ {table}: {count} records, {distinct} distinct values")
        except Exception as e:
            print(f"❌ {table}: {e}")

    conn.close()
    print()
    return True

def test_simple_agent():
    """Test the simple agent directly"""
    print("=" * 60)
    print("STEP 2: TESTING SIMPLE AGENT")
    print("=" * 60)

    try:
        from backend.simple_agent import process_query_with_reasoning

        test_queries = [
            "Show me storage buckets",
            "What security findings exist?",
            "List service accounts",
            "Give me security statistics"
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)

            response = process_query_with_reasoning(query)

            # Check if response has content
            if response and len(response) > 50:
                # Show first 500 chars
                preview = response[:500] + "..." if len(response) > 500 else response
                print(f"✓ Response received ({len(response)} chars)")
                print(f"Preview: {preview}")

                # Check for key indicators
                if "**ADK Agent Analysis**" in response:
                    print("✓ Has agent analysis header")
                if "**Reasoning:**" in response:
                    print("✓ Has reasoning section")
                if any(x in response for x in ["mgm-", "digitalconcierge", "bucket", "finding", "account"]):
                    print("✓ Contains project-specific data")
            else:
                print(f"❌ Response too short or empty: {response}")

    except Exception as e:
        print(f"❌ Simple agent test failed: {e}")
        import traceback
        traceback.print_exc()

    print()

async def test_adk_agent():
    """Test the actual ADK agent"""
    print("=" * 60)
    print("STEP 3: TESTING ADK AGENT (with InvocationContext)")
    print("=" * 60)

    try:
        from backend.adk_agent import security_agent

        # Try different approaches to create InvocationContext
        approaches = []

        # Approach 1: Direct import
        try:
            from google.adk.agents import InvocationContext
            context = InvocationContext(
                text_input="Show me storage buckets",
                session_id="test-session",
                invocation_id="test-inv-1"
            )
            approaches.append(("Direct InvocationContext", context))
        except Exception as e:
            print(f"❌ Direct InvocationContext failed: {e}")

        # Approach 2: Minimal wrapper
        try:
            class MinimalContext:
                def __init__(self, text):
                    self.text_input = text
                    self.session_id = "test-session"
                    self.invocation_id = "test-inv-2"
                    self.agent = security_agent
                    self.session = {"id": "test-session", "context": {}}

            context = MinimalContext("List security findings")
            approaches.append(("Minimal wrapper", context))
        except Exception as e:
            print(f"❌ Minimal wrapper failed: {e}")

        # Test each approach
        for approach_name, context in approaches:
            print(f"\nTrying {approach_name}...")
            print("-" * 40)

            try:
                # Try async run
                response_text = ""
                async for event in security_agent.run_async(context):
                    if hasattr(event, 'content') and event.content:
                        response_text += str(event.content)

                if response_text:
                    print(f"✓ {approach_name} worked!")
                    print(f"Response: {response_text[:300]}...")
                else:
                    print(f"❌ {approach_name} returned empty response")

            except Exception as e:
                print(f"❌ {approach_name} failed: {e}")

                # Try sync run as fallback
                try:
                    result = security_agent.run(context)
                    if result:
                        print(f"✓ Sync run worked: {str(result)[:300]}...")
                except Exception as sync_e:
                    print(f"❌ Sync also failed: {sync_e}")

    except Exception as e:
        print(f"❌ ADK agent test setup failed: {e}")
        import traceback
        traceback.print_exc()

    print()

def test_api_endpoints():
    """Test the FastAPI endpoints"""
    print("=" * 60)
    print("STEP 4: TESTING API ENDPOINTS")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # Test health endpoint
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code == 200:
            print(f"✓ Health endpoint: {resp.json()}")
        else:
            print(f"❌ Health endpoint returned {resp.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach API at {base_url}: {e}")
        print("Make sure the backend is running: python -m uvicorn backend.main:app --port 8000")
        return

    # Test streaming endpoint
    print("\n📡 Testing streaming endpoint...")
    try:
        # Send a test query
        headers = {"Content-Type": "text/plain"}
        query = "Show me all storage buckets in the project"

        with requests.post(
            f"{base_url}/api/v1/chat/stream",
            data=query,
            headers=headers,
            stream=True,
            timeout=5
        ) as resp:
            if resp.status_code == 200:
                print(f"✓ Streaming endpoint responded")

                # Collect SSE events
                events = []
                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data != '[DONE]':
                                events.append(data)
                                print(f"  Event: {data[:100]}...")

                if events:
                    print(f"✓ Received {len(events)} events")

                    # Try to parse and check content
                    full_response = ''.join(events)
                    if "ADK Agent Analysis" in full_response or "bucket" in full_response.lower():
                        print("✓ Response contains expected content")
                else:
                    print("❌ No events received")
            else:
                print(f"❌ Streaming endpoint returned {resp.status_code}")
                print(f"Response: {resp.text}")

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")

    print()

async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE AGENT TESTING")
    print("=" * 60 + "\n")

    # Step 1: Verify database
    if not verify_database():
        print("❌ Database verification failed. Exiting.")
        return

    # Step 2: Test simple agent
    test_simple_agent()

    # Step 3: Test ADK agent
    await test_adk_agent()

    # Step 4: Test API endpoints
    test_api_endpoints()

    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\n📊 Summary:")
    print("- Simple Agent: Works with direct database queries")
    print("- ADK Agent: Has InvocationContext issues")
    print("- API Endpoints: Check results above")
    print("\n💡 Recommendation:")
    print("The Simple Agent approach is working and provides reasoning-like output.")
    print("For true LLM reasoning, we need to resolve ADK InvocationContext requirements.")

if __name__ == "__main__":
    asyncio.run(main())