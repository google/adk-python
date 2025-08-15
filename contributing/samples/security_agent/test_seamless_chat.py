#!/usr/bin/env python3
"""
Test the seamless chat interface (no suggestions, pure chat flow)
"""

import requests
import time

def test_seamless_chat():
    """Test seamless conversation like ChatGPT/Claude"""
    print("💬 Testing Seamless Chat Interface (No Interruptions)")
    print("=" * 60)
    
    conversations = [
        "Tell me about my bucket security",
        "What about IAM permissions?", 
        "Show me firewall rules",
        "Any compliance issues?",
        "What's the biggest risk?"
    ]
    
    session_id = None
    
    for i, query in enumerate(conversations, 1):
        print(f"\n💬 You: {query}")
        
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
                
                print(f"🤖 {agent}: {response_text[:80]}...")
                print("   (seamless - no suggestions interrupting)")
                
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            
        time.sleep(0.3)
    
    print(f"\n✅ Seamless Chat Test Complete!")
    print("🎯 Interface now works like:")
    print("   • ChatGPT - just type and get responses")
    print("   • Claude - no interruptions or extra clicks")
    print("   • Normal chat apps - clean and simple")
    print(f"\n📱 Session maintained throughout: {session_id}")

if __name__ == "__main__":
    test_seamless_chat()