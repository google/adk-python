#!/usr/bin/env python3
"""
Manual Test Script for ADK Chat Session Flow
Run this to test the chat session functionality step by step
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_step(step_num: int, description: str):
    """Print a test step."""
    print(f"\n{BLUE}Step {step_num}: {description}{RESET}")
    print("-" * 60)

def print_result(success: bool, message: str):
    """Print test result."""
    if success:
        print(f"{GREEN}✅ {message}{RESET}")
    else:
        print(f"{RED}❌ {message}{RESET}")

def test_backend_health():
    """Test if backend is running."""
    print_step(1, "Check Backend Health")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print_result(True, "Backend is running")
            return True
    except requests.exceptions.ConnectionError:
        print_result(False, "Backend is not running")
        print(f"{YELLOW}Please start the backend with: python run_backend.py{RESET}")
        return False
    except Exception as e:
        print_result(False, f"Error checking backend: {e}")
        return False

def test_create_session():
    """Test session creation."""
    print_step(2, "Create ADK Session")
    
    try:
        # Create session
        payload = {
            "user_id": "test_user_123",
            "project_id": "test-project",
            "metadata": {
                "client_type": "test_script",
                "test_run": True
            }
        }
        
        response = requests.post(
            f"{API_BASE}/sessions/create",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            print_result(True, f"Session created: {session_id}")
            print(f"  User ID: {data.get('user_id')}")
            print(f"  Created: {data.get('created_at')}")
            return session_id
        else:
            print_result(False, f"Failed to create session: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print_result(False, f"Error creating session: {e}")
        return None

def test_chat_interaction(session_id: str):
    """Test chat interaction with session."""
    print_step(3, "Send Chat Message")
    
    try:
        # Send chat message
        payload = {
            "query": "Tell me about the storage buckets in my project",
            "user_id": "test_user_123",
            "session_id": session_id,
            "project_id": "test-project"
        }
        
        print(f"Sending: {payload['query']}")
        
        response = requests.post(
            f"{API_BASE}/agent/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result(True, "Chat message processed")
            print(f"\n{YELLOW}Response:{RESET}")
            print(f"  {data.get('response', 'No response')[:200]}...")
            print(f"\n  Agent Used: {data.get('agent_used', 'Unknown')}")
            print(f"  Session ID: {data.get('session_id', 'None')}")
            
            if data.get("suggestions"):
                print(f"\n{YELLOW}Suggestions:{RESET}")
                for suggestion in data["suggestions"][:3]:
                    print(f"  • {suggestion}")
            
            return True
        else:
            print_result(False, f"Chat failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_result(False, f"Error in chat: {e}")
        return False

def test_get_messages(session_id: str):
    """Test retrieving session messages."""
    print_step(4, "Retrieve Session Messages")
    
    try:
        response = requests.get(
            f"{API_BASE}/sessions/{session_id}/messages",
            params={"limit": 10},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", [])
            print_result(True, f"Retrieved {len(messages)} messages")
            
            for i, msg in enumerate(messages[:5], 1):
                sender = msg.get("sender_type", "unknown")
                content = msg.get("content", "")[:100]
                print(f"\n  Message {i} ({sender}):")
                print(f"    {content}...")
                if msg.get("agent_used"):
                    print(f"    Agent: {msg['agent_used']}")
            
            return True
        else:
            print_result(False, f"Failed to get messages: {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Error getting messages: {e}")
        return False

def test_session_analytics(session_id: str):
    """Test session analytics."""
    print_step(5, "Get Session Analytics")
    
    try:
        response = requests.get(
            f"{API_BASE}/sessions/{session_id}/analytics",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            analytics = data.get("analytics", {})
            print_result(True, "Retrieved session analytics")
            
            print(f"\n{YELLOW}Analytics:{RESET}")
            print(f"  Total Messages: {analytics.get('total_messages', 0)}")
            print(f"  Topics: {analytics.get('topics', [])}")
            print(f"  Status: {analytics.get('status', 'unknown')}")
            print(f"  Duration: {analytics.get('duration_minutes', 0):.1f} minutes")
            
            return True
        else:
            print_result(False, f"Failed to get analytics: {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Error getting analytics: {e}")
        return False

def test_session_continuity(session_id: str):
    """Test session continuity with follow-up message."""
    print_step(6, "Test Session Continuity")
    
    try:
        # Send follow-up message
        payload = {
            "query": "How do I fix the public access issues you mentioned?",
            "user_id": "test_user_123",
            "session_id": session_id,
            "project_id": "test-project"
        }
        
        print(f"Sending follow-up: {payload['query']}")
        
        response = requests.post(
            f"{API_BASE}/agent/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_result(True, "Session continuity maintained")
            print(f"\n{YELLOW}Follow-up Response:{RESET}")
            print(f"  {data.get('response', 'No response')[:200]}...")
            
            # Verify same session
            if data.get("session_id") == session_id:
                print_result(True, "Same session ID maintained")
            else:
                print_result(False, "Session ID changed unexpectedly")
            
            return True
        else:
            print_result(False, f"Follow-up failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Error in follow-up: {e}")
        return False

def run_full_test():
    """Run the complete test suite."""
    print(f"\n{BLUE}{'='*60}")
    print("ADK CHAT SESSION FLOW TEST")
    print(f"{'='*60}{RESET}")
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Backend URL: {BASE_URL}")
    
    # Track results
    results = []
    session_id = None
    
    # Run tests
    tests = [
        ("Backend Health", test_backend_health, None),
        ("Create Session", test_create_session, None),
    ]
    
    # First run basic tests
    for test_name, test_func, _ in tests:
        if test_func == test_create_session:
            session_id = test_func()
            results.append((test_name, session_id is not None))
        else:
            success = test_func()
            results.append((test_name, success))
            if not success and test_name == "Backend Health":
                print(f"\n{RED}Cannot continue without backend. Exiting.{RESET}")
                return False
    
    # If we have a session, run session-dependent tests
    if session_id:
        session_tests = [
            ("Chat Interaction", test_chat_interaction),
            ("Get Messages", test_get_messages),
            ("Session Analytics", test_session_analytics),
            ("Session Continuity", test_session_continuity),
        ]
        
        for test_name, test_func in session_tests:
            success = test_func(session_id)
            results.append((test_name, success))
    
    # Print summary
    print(f"\n{BLUE}{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}{RESET}\n")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        color = GREEN if success else RED
        print(f"{color}{status:6} {test_name}{RESET}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}✅ ALL TESTS PASSED ({passed}/{total}){RESET}")
        if session_id:
            print(f"\n{YELLOW}Session ID for manual testing: {session_id}{RESET}")
    else:
        print(f"{YELLOW}⚠️  PARTIAL SUCCESS ({passed}/{total} passed){RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)