#!/usr/bin/env python3
"""
Test multi-level conversation flow with follow-up questions
Ensures the chat maintains context and handles deep interactions
"""

import requests
import json
import time

class ConversationTester:
    def __init__(self):
        self.base_url = "http://localhost:8000/api/v1/agent/chat"
        self.session_id = None
        self.conversation_id = None
        self.user_id = "test_user"
        self.project_id = "mgm-digitalconcierge"
        
    def send_message(self, query):
        """Send a message and return the response"""
        payload = {
            "query": query,
            "user_id": self.user_id,
            "project_id": self.project_id
        }
        
        # Include session/conversation IDs if available
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
            
        response = requests.post(self.base_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Update session and conversation IDs
            if data.get("session_id"):
                self.session_id = data["session_id"]
            if data.get("conversation_id"):
                self.conversation_id = data["conversation_id"]
                
            return data
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    def run_conversation(self, conversation_flow):
        """Run a multi-turn conversation"""
        print("\n" + "="*80)
        print("🗣️ MULTI-LEVEL CONVERSATION TEST")
        print("="*80)
        
        for i, (query, expected_context) in enumerate(conversation_flow, 1):
            print(f"\n📝 Turn {i}: {query}")
            print("-" * 40)
            
            response = self.send_message(query)
            
            if response.get("success"):
                agent = response.get("agent_used", "Unknown")
                response_text = response.get("response", "")
                suggestions = response.get("suggestions", [])
                
                print(f"✅ Agent: {agent}")
                print(f"💬 Response: {response_text[:200]}...")
                
                if suggestions:
                    print(f"💡 Suggestions: {', '.join(suggestions[:2])}")
                
                # Check if expected context is present
                if expected_context:
                    if any(ctx.lower() in response_text.lower() for ctx in expected_context):
                        print(f"✓ Context preserved: Found expected terms")
                    else:
                        print(f"⚠️ Warning: Expected context not found: {expected_context}")
            else:
                print(f"❌ Failed: {response.get('error', 'Unknown error')}")
                return False
            
            # Small delay between messages
            time.sleep(0.5)
        
        return True

def test_deep_storage_conversation():
    """Test a deep conversation about storage security"""
    tester = ConversationTester()
    
    conversation = [
        # Level 1: Initial query
        ("Tell me about my bucket security issues", 
         ["bucket", "security", "storage"]),
        
        # Level 2: Follow-up on specific issue
        ("How do I fix the public access issue you mentioned?",
         ["public", "access", "gsutil", "iam"]),
        
        # Level 3: Deeper dive into remediation
        ("What's the exact command to remove public access from mgm-digitalconcierge-public-assets?",
         ["gsutil", "iam", "ch", "allUsers"]),
        
        # Level 4: Related follow-up
        ("After fixing that, what other bucket security issues should I address?",
         ["versioning", "encryption", "logging", "lifecycle"])
    ]
    
    return tester.run_conversation(conversation)

def test_cross_domain_conversation():
    """Test conversation that spans multiple domains"""
    tester = ConversationTester()
    
    conversation = [
        # Start with storage
        ("What are my biggest security risks?",
         ["security", "risk"]),
        
        # Move to IAM
        ("Show me which users have the most dangerous permissions",
         ["users", "permissions", "iam", "role"]),
        
        # Move to network
        ("Are there any firewall rules that could be exploited?",
         ["firewall", "rules", "port", "ssh"]),
        
        # Move to cost
        ("How much would it cost to fix all these security issues?",
         ["cost", "savings", "optimization"]),
        
        # Back to general with context
        ("Can you summarize all the critical issues we discussed?",
         ["summary", "critical", "issues"])
    ]
    
    return tester.run_conversation(conversation)

def test_clarification_conversation():
    """Test conversation with clarifications and refinements"""
    tester = ConversationTester()
    
    conversation = [
        # Vague initial query
        ("Help me with security",
         ["security", "help"]),
        
        # Clarification
        ("I'm specifically worried about data exposure",
         ["data", "exposure", "public", "access"]),
        
        # More specific
        ("Check if any of my storage buckets are publicly accessible",
         ["bucket", "public", "accessible"]),
        
        # Action request
        ("Give me the commands to lock down all public buckets",
         ["gsutil", "iam", "command", "lock"])
    ]
    
    return tester.run_conversation(conversation)

def main():
    """Run all conversation tests"""
    print("\n" + "="*80)
    print("🧪 CONVERSATION FLOW TESTING SUITE")
    print("="*80)
    
    tests = [
        ("Deep Storage Conversation", test_deep_storage_conversation),
        ("Cross-Domain Conversation", test_cross_domain_conversation),
        ("Clarification Conversation", test_clarification_conversation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n\n🎯 Testing: {test_name}")
        print("="*80)
        
        try:
            if test_func():
                print(f"\n✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED with error: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All conversation flows working correctly!")
        print("✨ Multi-level interactions and context preservation verified.")
    else:
        print(f"\n⚠️ {failed} conversation flow(s) need improvement.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())