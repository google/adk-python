#!/usr/bin/env python3
"""
Test script for Smart Query Routing
Demonstrates how different queries are routed to appropriate specialists
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.query_router import query_router, Specialist

def test_routing():
    """Test various queries to see how they're routed"""
    
    test_queries = [
        # Storage queries
        "tell me about the buckets in my project",
        "analyze my storage security",
        "check backup policies",
        
        # IAM queries
        "who has access to my resources?",
        "review user permissions",
        "check service account privileges",
        "show me users with owner role",
        
        # Network queries
        "analyze my firewall rules",
        "check network security",
        "are there any open ports?",
        "review VPC configuration",
        
        # Compliance queries
        "what's my compliance status?",
        "am I SOC2 compliant?",
        "check GDPR requirements",
        "audit my security controls",
        
        # Cost queries
        "how much am I spending?",
        "show me cost optimization opportunities",
        "what are my unused resources?",
        "help me reduce costs",
        
        # General queries
        "what's my security score?",
        "give me an overview",
        "help me improve security"
    ]
    
    print("🧪 Testing Smart Query Routing")
    print("=" * 80)
    
    for query in test_queries:
        routing_result = query_router.route_query(query)
        specialist = routing_result["specialist"]
        confidence = routing_result["confidence"]
        keywords = routing_result["context"]["matched_keywords"]
        
        # Emoji for specialist
        emoji_map = {
            Specialist.STORAGE: "🪣",
            Specialist.IAM: "🔐",
            Specialist.NETWORK: "🌐",
            Specialist.COMPLIANCE: "📋",
            Specialist.FINOPS: "💰",
            Specialist.COMPUTE: "💻",
            Specialist.DATABASE: "🗄️",
            Specialist.KUBERNETES: "☸️",
            Specialist.MONITORING: "📊",
            Specialist.GENERAL: "🛡️"
        }
        
        emoji = emoji_map.get(specialist, "❓")
        
        print(f"\nQuery: \"{query}\"")
        print(f"  {emoji} Specialist: {specialist.value}")
        print(f"  📊 Confidence: {confidence:.0%}")
        if keywords:
            print(f"  🔍 Keywords: {', '.join(keywords[:3])}")
        print(f"  🔗 Endpoint: {routing_result['endpoint']}")
    
    print("\n" + "=" * 80)
    print("✅ Routing test complete!")

def test_routing_accuracy():
    """Test routing accuracy with expected results"""
    
    test_cases = [
        {"query": "analyze my buckets", "expected": Specialist.STORAGE},
        {"query": "check IAM policies", "expected": Specialist.IAM},
        {"query": "firewall configuration", "expected": Specialist.NETWORK},
        {"query": "SOC2 compliance", "expected": Specialist.COMPLIANCE},
        {"query": "reduce cloud costs", "expected": Specialist.FINOPS},
    ]
    
    print("\n🎯 Testing Routing Accuracy")
    print("=" * 80)
    
    correct = 0
    total = len(test_cases)
    
    for test in test_cases:
        routing_result = query_router.route_query(test["query"])
        actual = routing_result["specialist"]
        expected = test["expected"]
        
        if actual == expected:
            print(f"✅ PASS: \"{test['query']}\" → {actual.value}")
            correct += 1
        else:
            print(f"❌ FAIL: \"{test['query']}\" → {actual.value} (expected {expected.value})")
    
    accuracy = (correct / total) * 100
    print(f"\n📊 Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    
    if accuracy == 100:
        print("🎉 Perfect routing accuracy!")
    elif accuracy >= 80:
        print("👍 Good routing accuracy!")
    else:
        print("⚠️ Routing needs improvement")

if __name__ == "__main__":
    test_routing()
    test_routing_accuracy()