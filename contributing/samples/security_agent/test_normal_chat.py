#!/usr/bin/env python3
"""
Test the improved normal chat interface
"""

import requests
import time

def test_normal_chat_flow():
    """Test a natural conversation flow"""
    print("🗣️ Testing Normal Chat Interface")
    print("=" * 50)
    
    conversations = [
        "Tell me about my bucket security",
        "What about IAM permissions?", 
        "Are there any firewall issues?",
        "How much would it cost to fix these issues?",
        "Can you summarize the main security risks?"
    ]
    
    session_id = None
    
    for i, query in enumerate(conversations, 1):
        print(f"\n💬 Turn {i}: {query}")
        print("-" * 30)
        
        payload = {
            "query": query,
            "user_id": "test_user", 
            "project_id": "mgm-digitalconcierge"
        }
        
        if session_id:
            payload["session_id"] = session_id
            
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/agent/chat",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Update session ID for continuity
                if data.get("session_id"):
                    session_id = data["session_id"]
                
                agent = data.get("agent_used", "Unknown")
                response_text = data.get("response", "")
                suggestions = data.get("suggestions", [])
                
                print(f"🤖 Agent: {agent}")
                print(f"📝 Response: {response_text[:100]}...")
                
                if suggestions:
                    print(f"💡 Suggestions available: {len(suggestions)} (collapsed by default)")
                else:
                    print("💡 No suggestions (normal)")
                    
                print("✅ Session maintained" if session_id else "⚠️ No session")
                
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            
        time.sleep(0.5)  # Small delay between messages
    
    print(f"\n🎯 Final session ID: {session_id}")
    print("✅ Natural conversation flow completed!")

if __name__ == "__main__":
    test_normal_chat_flow()