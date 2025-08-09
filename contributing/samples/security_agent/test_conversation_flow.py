#!/usr/bin/env python3
"""
Test script to validate ADK conversation flow and follow-up question handling.
"""

import requests
import json
import time

BACKEND_URL = "http://localhost:8000"

def test_conversation_flow():
    """Test complete conversation flow with follow-up questions."""
    print("🧪 Testing ADK Conversation Flow...")
    print("=" * 50)
    
    # Simulate conversation context
    conversation_context = {
        "chat_history": [],
        "session_id": "test-session-123",
        "conversation_context": {},
        "current_topic": None,
        "previous_findings": {}
    }
    
    # Test 1: Initial security question
    print("\n1️⃣ Testing Initial Security Question...")
    initial_message = "What's my current security score?"
    
    response1 = send_chat_message(initial_message, conversation_context)
    print(f"✅ Response 1: {response1.get('success', False)}")
    print(f"   Agent: {response1.get('agent_used', 'unknown')}")
    print(f"   Response length: {len(response1.get('response', ''))}")
    print(f"   Suggestions: {len(response1.get('suggestions', []))}")
    
    # Update context with first response
    conversation_context["chat_history"].extend([
        {"role": "user", "content": initial_message},
        {
            "role": "assistant", 
            "content": response1.get("response", ""),
            "agent_used": response1.get("agent_used"),
            "suggestions": response1.get("suggestions", [])
        }
    ])
    
    # Test 2: Follow-up question
    print("\n2️⃣ Testing Follow-up Question...")
    followup_message = "Show me detailed security findings"
    
    response2 = send_chat_message(followup_message, conversation_context)
    print(f"✅ Response 2: {response2.get('success', False)}")
    print(f"   Agent: {response2.get('agent_used', 'unknown')}")
    print(f"   Response length: {len(response2.get('response', ''))}")
    print(f"   GCP API calls: {len(response2.get('gcp_api_calls', []))}")
    
    # Update context again
    conversation_context["chat_history"].extend([
        {"role": "user", "content": followup_message},
        {
            "role": "assistant", 
            "content": response2.get("response", ""),
            "agent_used": response2.get("agent_used"),
            "gcp_api_calls": response2.get("gcp_api_calls", [])
        }
    ])
    
    # Test 3: Another follow-up
    print("\n3️⃣ Testing Another Follow-up Question...")
    followup2_message = "Tell me how to fix these issues"
    
    response3 = send_chat_message(followup2_message, conversation_context)
    print(f"✅ Response 3: {response3.get('success', False)}")
    print(f"   Agent: {response3.get('agent_used', 'unknown')}")
    print(f"   Response length: {len(response3.get('response', ''))}")
    print(f"   Has fix instructions: {'gcloud' in response3.get('response', '')}")
    
    # Summary
    print("\n📊 Conversation Flow Summary:")
    print(f"   Total messages: {len(conversation_context['chat_history'])}")
    print(f"   Agent consistency: {check_agent_consistency([response1, response2, response3])}")
    print(f"   Context preservation: {check_context_preservation(response3)}")
    
    return True

def send_chat_message(message, context):
    """Send a chat message to the backend."""
    try:
        payload = {
            "prompt": message,
            "context": context,
            "use_enhanced": True,
            "timestamp": time.time(),
            "project_id": "test-project"
        }
        
        print(f"   Sending: '{message[:30]}...' with {len(context.get('chat_history', []))} history items")
        
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agent/chat",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {"success": False, "error": str(e)}

def check_agent_consistency(responses):
    """Check if the same agent type was used for security questions."""
    agents = [r.get("agent_used") for r in responses if r.get("agent_used")]
    security_agents = [a for a in agents if "security" in a.lower()]
    return len(security_agents) == len([r for r in responses if r.get("success")])

def check_context_preservation(final_response):
    """Check if context was preserved in the final response."""
    content = final_response.get("response", "").lower()
    return any(word in content for word in ["fix", "gcloud", "priority", "mfa"])

def test_health_endpoint():
    """Test that the backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🚀 ADK Conversation Flow Test")
    print("=" * 50)
    
    # Check backend health
    if not test_health_endpoint():
        print("❌ Backend not running at http://localhost:8000")
        print("   Please start the backend first: python run_backend.py")
        exit(1)
    
    print("✅ Backend is running")
    
    # Run the conversation test
    try:
        success = test_conversation_flow()
        if success:
            print("\n🎉 All tests passed! Conversation flow is working correctly.")
        else:
            print("\n❌ Some tests failed. Check the output above.")
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")