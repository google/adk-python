#!/usr/bin/env python3
"""
Test script to verify if ADK agent does real LLM analysis or just templated responses.
"""

import requests
import json
import time

def test_query(query, expected_llm_behavior):
    """Test a query and analyze the response for LLM vs template behavior."""
    print(f"\n🧪 Testing: {query}")
    print(f"Expected LLM behavior: {expected_llm_behavior}")
    print("-" * 80)

    payload = {
        "message": query,
        "session_id": f"llm_test_{int(time.time())}",
        "user_id": "test_user"
    }

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/chat/message",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            response_text = data.get("response", "")

            print("✅ Response received:")
            print(response_text)

            # Check for signs of templated vs generated content
            template_indicators = [
                "Storage Security Analysis:",
                "Total Buckets:",
                "• **",  # Bullet points with bold
                "Location: ",
                "Access: 🔒 Private"
            ]

            llm_indicators = [
                "I recommend",
                "Based on the analysis",
                "The biggest risks",
                "You should prioritize",
                "In my assessment",
                "This suggests"
            ]

            template_count = sum(1 for indicator in template_indicators if indicator in response_text)
            llm_count = sum(1 for indicator in llm_indicators if indicator in response_text)

            print(f"\n📊 Analysis:")
            print(f"Template indicators found: {template_count}")
            print(f"LLM reasoning indicators found: {llm_count}")

            if template_count > llm_count:
                print("🤖 VERDICT: Likely templated response")
            elif llm_count > 0:
                print("🧠 VERDICT: Shows LLM reasoning")
            else:
                print("❓ VERDICT: Unclear - mixed signals")

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def main():
    """Run test queries to check for real LLM behavior."""
    print("🔍 Testing ADK Agent for Real LLM Generation vs Templates")
    print("=" * 80)

    test_cases = [
        {
            "query": "show me storage buckets",
            "expected": "Should be templated (baseline test)"
        },
        {
            "query": "What are my biggest security risks and how should I prioritize fixing them?",
            "expected": "Should show LLM analysis and reasoning"
        },
        {
            "query": "Compare the security of my terraform buckets vs regular buckets",
            "expected": "Should show LLM comparison and analysis"
        },
        {
            "query": "Which storage buckets should I prioritize for security improvements and why?",
            "expected": "Should show LLM prioritization reasoning"
        },
        {
            "query": "Analyze my security posture and give me 3 specific recommendations",
            "expected": "Should show LLM analysis and custom recommendations"
        }
    ]

    for test_case in test_cases:
        test_query(test_case["query"], test_case["expected"])
        time.sleep(2)  # Brief pause between tests

    print("\n" + "=" * 80)
    print("🎯 Test Summary:")
    print("If responses are mostly templated, the 'AI agent' is just a smart database interface.")
    print("If responses show reasoning/analysis, then real LLM generation is happening.")

if __name__ == "__main__":
    main()