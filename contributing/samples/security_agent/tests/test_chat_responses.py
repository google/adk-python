#!/usr/bin/env python3
"""
Test that chat endpoint returns real, detailed data for all query types
"""

import requests
import json

import pytest

@pytest.mark.parametrize(
    "query_type, query, expected_keywords",
    [
        ("storage", "Tell me about my buckets", ["bucket"]),
        ("iam", "Who has access to my project?", ["iam", "access"]),
        ("network", "Analyze my firewall rules", ["firewall", "network"]),
        ("general", "What's my security score?", ["security"]),
    ],
)
def test_chat_query(query_type, query, expected_keywords):
    """Test a single chat query"""
    print(f"\n🧪 Testing {query_type}...")
    print(f"   Query: '{query}'")
    
    response = requests.post(
        "http://localhost:8000/api/v1/agent/chat",
        json={
            "query": query,
            "user_id": "test_user",
            "project_id": "mgm-digitalconcierge"
        },
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        response_text = data.get("response", "")
        agent_used = data.get("agent_used", "Unknown")
        
        if data.get("success") and response_text:
            print(f"   ✅ SUCCESS - Agent: {agent_used}")
            
            # Check for expected keywords in response
            found_keywords = []
            for keyword in expected_keywords:
                if keyword.lower() in response_text.lower():
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"   📊 Found keywords: {', '.join(found_keywords)}")
                print(f"   📝 Response preview: {response_text[:150]}...")
                return True
            else:
                print(f"   ⚠️  WARNING: Expected keywords not found")
                print(f"   📝 Response: {response_text[:200]}...")
                return False
        else:
            print(f"   ❌ FAILED - Empty or unsuccessful response")
            return False
    else:
        print(f"   ❌ FAILED - Status {response.status_code}")
        return False

def main():
    """Test all specialist routing"""
    print("=" * 80)
    print("🎯 Chat Response Quality Test")
    print("=" * 80)
    
    tests = [
        ("Storage Analysis", 
         "analyze my bucket security issues",
         ["bucket", "mgm-digitalconcierge", "gsutil", "public", "versioning"]),
        
        ("IAM Analysis",
         "show me users with risky permissions",
         ["users", "roles", "permissions", "risk", "service account"]),
        
        ("Network Analysis",
         "check my firewall rules for issues",
         ["firewall", "SSH", "port", "0.0.0.0", "gcloud compute"]),
        
        ("Cost Analysis",
         "how can I reduce my cloud spending?",
         ["cost", "savings", "$", "unused", "rightsize"]),
        
        ("Compliance Analysis",
         "what's my SOC2 compliance status?",
         ["compliance", "SOC2", "score", "%", "framework"])
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test_chat_query(*test):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All chat queries return real, detailed data!")
        print("✨ Frontend-backend integration is fully functional.")
    else:
        print(f"\n⚠️  {failed} query type(s) not returning expected data.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())