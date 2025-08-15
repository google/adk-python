#!/usr/bin/env python3
"""Test script to verify session persistence with SQLite storage."""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_session_persistence():
    """Test that sessions persist across backend restarts."""
    
    print("=" * 60)
    print("SESSION PERSISTENCE TEST")
    print("=" * 60)
    
    # Step 1: Create a session
    print("\n1. Creating new session...")
    response = requests.post(
        f"{BASE_URL}/api/v1/sessions/create",
        json={
            "user_id": "test_user_persistence",
            "project_id": "test_project",
            "metadata": {"test": "persistence"}
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.text}")
        return False
    
    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"✅ Created session: {session_id}")
    
    # Step 2: Send a chat message
    print("\n2. Sending chat message...")
    chat_response = requests.post(
        f"{BASE_URL}/api/v1/agent/chat",
        json={
            "query": "Test message for persistence",
            "user_id": "test_user_persistence",
            "project_id": "test_project",
            "session_id": session_id
        }
    )
    
    if chat_response.status_code != 200:
        print(f"❌ Failed to send message: {chat_response.text}")
        return False
    
    print("✅ Message sent successfully")
    
    # Step 3: Verify session exists
    print("\n3. Verifying session exists...")
    get_response = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}")
    
    if get_response.status_code != 200:
        print(f"❌ Failed to get session: {get_response.text}")
        return False
    
    print("✅ Session retrieved successfully")
    
    # Step 4: Get messages
    print("\n4. Getting session messages...")
    messages_response = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}/messages")
    
    if messages_response.status_code != 200:
        print(f"❌ Failed to get messages: {messages_response.text}")
        return False
    
    messages_data = messages_response.json()
    print(f"✅ Retrieved {messages_data['total_count']} messages")
    
    # Step 5: Session analytics
    print("\n5. Getting session analytics...")
    analytics_response = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}/analytics")
    
    if analytics_response.status_code != 200:
        print(f"❌ Failed to get analytics: {analytics_response.text}")
        return False
    
    analytics = analytics_response.json()["analytics"]
    print(f"✅ Session analytics:")
    print(f"   - Total messages: {analytics.get('total_messages', 0)}")
    print(f"   - User messages: {analytics.get('user_messages', 0)}")
    print(f"   - Assistant messages: {analytics.get('assistant_messages', 0)}")
    print(f"   - Status: {analytics.get('status', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ SESSION PERSISTENCE TEST PASSED")
    print("=" * 60)
    print("\nThe session has been created and stored in the SQLite database.")
    print("You can now restart the backend and the session will persist.")
    print(f"\nSession ID to test after restart: {session_id}")
    print("=" * 60)
    
    return True

def verify_persisted_session(session_id):
    """Verify a session exists after backend restart."""
    
    print("=" * 60)
    print("VERIFYING PERSISTED SESSION")
    print("=" * 60)
    
    print(f"\nChecking session: {session_id}")
    
    # Get session
    get_response = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}")
    
    if get_response.status_code != 200:
        print(f"❌ Session not found: {get_response.text}")
        return False
    
    session_data = get_response.json()
    print(f"✅ Session found!")
    print(f"   - User ID: {session_data['user_id']}")
    print(f"   - Status: {session_data['status']}")
    print(f"   - Created: {session_data['created_at']}")
    
    # Get messages
    messages_response = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}/messages")
    
    if messages_response.status_code != 200:
        print(f"❌ Failed to get messages: {messages_response.text}")
        return False
    
    messages_data = messages_response.json()
    print(f"✅ Messages preserved: {messages_data['total_count']} messages")
    
    for msg in messages_data['messages'][:3]:  # Show first 3 messages
        print(f"   - [{msg['sender_type']}]: {msg['content'][:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ SESSION SUCCESSFULLY PERSISTED ACROSS RESTART!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Check if we're verifying an existing session
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
        verify_persisted_session(session_id)
    else:
        # Run the full test
        test_session_persistence()