#!/usr/bin/env python3
"""
Test firewall rules query directly with ADK
"""

import requests
import json

# Configuration
ADK_BASE_URL = "http://127.0.0.1:8000"

print("Testing firewall rules query...")

# Create session
session_response = requests.post(
    f"{ADK_BASE_URL}/apps/agents/users/test-user/sessions",
    json={}
)

if session_response.status_code == 200:
    session_id = session_response.json().get("id")
    print(f"✅ Session created: {session_id}")

    # Query about firewall rules
    message_payload = {
        "appName": "agents",
        "userId": "test-user",
        "sessionId": session_id,
        "newMessage": {
            "parts": [{"text": "Tell me about firewall rules in my GCP project"}],
            "role": "user"
        },
        "streaming": False
    }

    print("\nQuerying about firewall rules...")

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
                                if len(text) > 20:
                                    agent_message = text
                                    break
                        if agent_message:
                            break

        if agent_message:
            print("\n✅ Agent Response:")
            print("-" * 60)
            print(agent_message)
            print("-" * 60)
        else:
            print("⚠️ No response found")
            print(f"Raw response: {json.dumps(response_data, indent=2)[:500]}")
    else:
        print(f"❌ Error: {run_response.status_code}")
        print(run_response.text[:200])
else:
    print(f"❌ Failed to create session: {session_response.status_code}")