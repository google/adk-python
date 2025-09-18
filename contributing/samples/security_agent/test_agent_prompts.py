#!/usr/bin/env python3
"""
Test script to verify ADK agent is following up on queries properly.
Sends multiple test prompts to check if the agent:
1. Invokes the query_security_data tool
2. Returns actual data from the database
3. Follows up on specific security queries
"""

import requests
import json
import time
from typing import Dict, Any

# Test prompts that should trigger tool invocation
TEST_PROMPTS = [
    # Direct data requests
    "Show me all storage buckets",
    "List storage buckets with their security status",
    "What storage buckets do we have?",
    "Display the GCP storage buckets",
    "Get me the bucket inventory",

    # Security-specific queries
    "Show me IAM policies for storage",
    "What are the security findings?",
    "List any security vulnerabilities",
    "Check bucket permissions",
    "Analyze storage security posture",

    # Analysis requests
    "Which buckets are publicly accessible?",
    "Find buckets with weak encryption",
    "Show me buckets without versioning",
    "Identify high-risk storage configurations",
    "What buckets need security remediation?",

    # Compliance queries
    "Are our buckets HIPAA compliant?",
    "Check PCI DSS compliance for storage",
    "Verify encryption at rest for all buckets",
    "Show compliance status for storage resources",
    "Which buckets violate our security policies?"
]

def test_agent_endpoint(prompt: str, endpoint: str = "http://localhost:8000/api/v1/chat/message") -> Dict[str, Any]:
    """Test a single prompt against the agent endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing prompt: '{prompt}'")
    print(f"{'='*60}")

    try:
        # Send the request
        response = requests.post(
            endpoint,
            json={"message": prompt},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: SUCCESS")

            # Check if tool was invoked
            if "tool_calls" in result or "tools_used" in result:
                print(f"🔧 Tool invoked: YES")
            else:
                print(f"⚠️  Tool invoked: NO (agent may not be following up)")

            # Check response content
            response_text = result.get("response", "")
            if len(response_text) > 200:
                print(f"📝 Response length: {len(response_text)} chars (substantial)")
            else:
                print(f"📝 Response length: {len(response_text)} chars (brief)")

            # Check for data indicators
            data_indicators = ["bucket", "storage", "security", "finding", "policy", "permission"]
            has_data = any(indicator in response_text.lower() for indicator in data_indicators)

            if has_data:
                print(f"📊 Contains data: YES")
            else:
                print(f"⚠️  Contains data: NO (might be generic response)")

            # Print first 300 chars of response
            print(f"\n📄 Response preview:")
            print(response_text[:300] + "..." if len(response_text) > 300 else response_text)

            return {
                "success": True,
                "tool_invoked": "tool_calls" in result or "tools_used" in result,
                "has_data": has_data,
                "response_length": len(response_text),
                "response": response_text
            }

        else:
            print(f"❌ Status: FAILED (HTTP {response.status_code})")
            print(f"Error: {response.text}")
            return {
                "success": False,
                "error": response.text,
                "status_code": response.status_code
            }

    except requests.exceptions.Timeout:
        print(f"⏱️  TIMEOUT: Request took longer than 30 seconds")
        return {"success": False, "error": "timeout"}

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {"success": False, "error": str(e)}

def run_all_tests():
    """Run all test prompts and generate a summary."""
    print("\n" + "="*60)
    print("ADK AGENT TEST SUITE")
    print("Testing agent's ability to follow up on queries")
    print("="*60)

    # Check if backend is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code == 200:
            print("✅ Backend is running")
        else:
            print("⚠️  Backend health check returned:", health.status_code)
    except:
        print("❌ Backend is not responding. Please ensure it's running.")
        return

    results = []

    # Test each prompt
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing...")
        result = test_agent_endpoint(prompt)
        results.append({
            "prompt": prompt,
            **result
        })

        # Small delay between requests
        if i < len(TEST_PROMPTS):
            time.sleep(2)

    # Generate summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    successful = sum(1 for r in results if r["success"])
    tool_invoked = sum(1 for r in results if r.get("tool_invoked", False))
    has_data = sum(1 for r in results if r.get("has_data", False))

    print(f"Total tests: {len(TEST_PROMPTS)}")
    print(f"Successful requests: {successful}/{len(TEST_PROMPTS)}")
    print(f"Tool invocations: {tool_invoked}/{len(TEST_PROMPTS)}")
    print(f"Responses with data: {has_data}/{len(TEST_PROMPTS)}")

    # Identify problematic prompts
    print("\n🔍 ANALYSIS:")

    no_tool_prompts = [r["prompt"] for r in results if r["success"] and not r.get("tool_invoked", False)]
    if no_tool_prompts:
        print("\n⚠️  Prompts where tool was NOT invoked:")
        for prompt in no_tool_prompts[:5]:  # Show first 5
            print(f"  - '{prompt}'")

    no_data_prompts = [r["prompt"] for r in results if r["success"] and not r.get("has_data", False)]
    if no_data_prompts:
        print("\n⚠️  Prompts with generic responses (no data):")
        for prompt in no_data_prompts[:5]:  # Show first 5
            print(f"  - '{prompt}'")

    # Overall assessment
    print("\n📊 AGENT ASSESSMENT:")
    if tool_invoked >= len(TEST_PROMPTS) * 0.8:  # 80% threshold
        print("✅ Agent is properly following up on most queries")
    elif tool_invoked >= len(TEST_PROMPTS) * 0.5:  # 50% threshold
        print("⚠️  Agent is inconsistent - sometimes follows up, sometimes doesn't")
    else:
        print("❌ Agent is NOT following up properly - needs instruction tuning")

    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📁 Detailed results saved to test_results.json")

if __name__ == "__main__":
    run_all_tests()