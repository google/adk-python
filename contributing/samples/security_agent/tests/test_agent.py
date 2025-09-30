#!/usr/bin/env python3
"""
Test script to verify the BigQuery ADK Agent is working
"""

import requests
import json
import time

def test_adk_agent():
    """Test the complete ADK agent flow"""

    # Configuration
    ADK_BASE_URL = "http://127.0.0.1:8000"
    FLASK_URL = "http://127.0.0.1:5000"

    print("=" * 60)
    print("🔍 Testing BigQuery Security Agent")
    print("=" * 60)

    # Test 1: Check Flask health
    print("\n1. Testing Flask Frontend...")
    try:
        response = requests.get(f"{FLASK_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Flask is healthy: {data['status']}")
            print(f"   ✅ ADK connected: {data['adk_connected']}")
        else:
            print(f"   ❌ Flask health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Flask not accessible: {e}")

    # Test 2: Check ADK backend
    print("\n2. Testing ADK Backend...")
    try:
        response = requests.get(f"{ADK_BASE_URL}/list-apps")
        if response.status_code == 200:
            apps = response.json()
            if "agents" in apps:
                print(f"   ✅ ADK backend is running with 'agents' app")
            else:
                print(f"   ⚠️  ADK backend running but no 'agents' app found")
        else:
            print(f"   ❌ ADK backend check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ADK backend not accessible: {e}")

    # Test 3: Create session and send message
    print("\n3. Testing Agent Communication...")
    try:
        # Create session
        session_response = requests.post(
            f"{ADK_BASE_URL}/apps/agents/users/test-user/sessions",
            json={}
        )

        if session_response.status_code == 200:
            session_id = session_response.json().get("id")
            print(f"   ✅ Session created: {session_id}")

            # Send test message
            print("\n4. Sending test query to agent...")
            message_payload = {
                "appName": "agents",
                "userId": "test-user",
                "sessionId": session_id,
                "newMessage": {
                    "parts": [{"text": "What datasets are available in BigQuery?"}],
                    "role": "user"
                },
                "streaming": False
            }

            run_response = requests.post(
                f"{ADK_BASE_URL}/run",
                json=message_payload
            )

            if run_response.status_code == 200:
                response_data = run_response.json()

                # Extract agent response
                agent_message = ""
                if isinstance(response_data, list):
                    for event in response_data:
                        if isinstance(event, dict) and "content" in event:
                            content = event["content"]
                            if "parts" in content:
                                for part in content["parts"]:
                                    if "text" in part:
                                        text = part["text"].strip()
                                        if len(text) > 20:  # Skip short system messages
                                            agent_message = text
                                            break
                                if agent_message:
                                    break

                if agent_message:
                    print(f"   ✅ Agent responded successfully!")
                    print(f"\n   Agent's response (first 200 chars):")
                    print(f"   {agent_message[:200]}...")
                else:
                    print(f"   ⚠️  Agent responded but no meaningful message found")
                    print(f"   Raw response: {json.dumps(response_data)[:200]}...")

            else:
                print(f"   ❌ Failed to get agent response: {run_response.status_code}")
                print(f"   Error: {run_response.text[:200]}")
        else:
            print(f"   ❌ Failed to create session: {session_response.status_code}")

    except Exception as e:
        print(f"   ❌ Error testing agent: {e}")

    print("\n" + "=" * 60)
    print("✅ Test complete! You can access the web UI at:")
    print(f"   {FLASK_URL}")
    print("=" * 60)

if __name__ == "__main__":
    test_adk_agent()