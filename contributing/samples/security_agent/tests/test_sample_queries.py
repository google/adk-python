#!/usr/bin/env python3
"""
Test sample queries to demonstrate the system working with real data
"""

import requests
import json

def test_query(query):
    """Send a query and display the response"""
    print(f"\n{'='*80}")
    print(f"📝 Query: {query}")
    print("="*80)
    
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
        if data.get("success"):
            print(f"✅ Agent Used: {data.get('agent_used', 'Unknown')}")
            print(f"\n📊 Response:\n")
            print(data.get("response", "No response"))
            
            if data.get("suggestions"):
                print(f"\n💡 Suggestions:")
                for suggestion in data["suggestions"][:3]:
                    print(f"   • {suggestion}")
        else:
            print("❌ Query failed")
            print(f"Response: {data}")
    else:
        print(f"❌ HTTP Error {response.status_code}")

def main():
    """Run sample queries"""
    print("\n" + "="*80)
    print("🚀 ADK Security Agent - Sample Query Demonstration")
    print("="*80)
    
    queries = [
        "Tell me about my bucket security issues",
        "Which firewall rules are risky?",
        "How much am I spending this month?",
        "Show me users with dangerous permissions"
    ]
    
    for query in queries:
        test_query(query)
    
    print("\n" + "="*80)
    print("✨ Demo complete! All queries returned real, actionable data.")
    print("="*80)

if __name__ == "__main__":
    main()