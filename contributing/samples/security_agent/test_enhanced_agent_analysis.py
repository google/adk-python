#!/usr/bin/env python3
"""
Enhanced Agent Analysis Test

Tests the ADK security agent with analytical queries to verify it provides
LLM-generated insights instead of raw JSON data.

Uses the ResponseQualityAssessor to validate analysis depth.
"""

import requests
import time
import json
from tests.test_response_quality import ResponseQualityAssessor


def test_agent_analysis_capability():
    """Test the agent with analytical queries and assess response quality"""

    # Initialize quality assessor
    assessor = ResponseQualityAssessor()

    # Backend URL
    backend_url = "http://localhost:8000"

    # Test queries that require analysis
    analytical_queries = [
        "What are my biggest security risks and how should I prioritize fixing them?",
        "Analyze my storage buckets and recommend security improvements",
        "Compare the security posture of my different resources",
        "Which security findings pose the highest risk to my organization?",
        "How can I improve my overall GCP security stance?"
    ]

    print("🧪 Testing Enhanced Agent Analysis Capability")
    print("=" * 60)

    passed_tests = 0
    total_tests = len(analytical_queries)

    for i, query in enumerate(analytical_queries, 1):
        print(f"\n📋 Test {i}/{total_tests}: {query}")
        print("-" * 50)

        try:
            # Send query to backend
            response = requests.post(
                f"{backend_url}/api/v1/chat/message",
                json={
                    "message": query,
                    "session_id": f"test_session_{int(time.time())}",
                    "user_id": "test_user"
                },
                timeout=30
            )

            if response.status_code == 200:
                response_data = response.json()
                agent_response = response_data.get("response", "")

                print(f"✅ Response received ({len(agent_response)} chars)")

                # Assess response quality
                metrics = assessor.assess_response_quality(agent_response)

                print(f"📊 Analysis Depth Score: {metrics.analysis_depth_score:.1f}/100")
                print(f"🔍 Response Type: {metrics.response_type.value}")
                print(f"💡 Reasoning Indicators: {metrics.reasoning_indicators}")
                print(f"📝 Recommendations: {metrics.recommendation_count}")
                print(f"⚖️ Raw Data Ratio: {metrics.raw_data_ratio:.2f}")

                # Check if it's LLM analysis
                is_analysis = assessor.is_llm_analysis(agent_response, threshold=50.0)

                if is_analysis:
                    print("✅ PASS: Response contains LLM analysis")
                    passed_tests += 1

                    # Show a snippet of the analysis
                    snippet = agent_response[:200] + "..." if len(agent_response) > 200 else agent_response
                    print(f"📄 Response snippet: {snippet}")
                else:
                    print("❌ FAIL: Response lacks sufficient LLM analysis")
                    print(f"📄 Raw response: {agent_response[:300]}...")

                    # Generate detailed quality report
                    quality_report = assessor.generate_quality_report(agent_response)
                    print(f"📊 Quality Report:\n{quality_report}")

            else:
                print(f"❌ Backend error: {response.status_code}")
                print(f"Error: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        except Exception as e:
            print(f"❌ Test error: {e}")

        # Small delay between tests
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 SUCCESS: All analytical queries produced LLM analysis!")
        return True
    elif passed_tests > 0:
        print(f"⚠️ PARTIAL: {passed_tests} out of {total_tests} queries produced good analysis")
        return False
    else:
        print("❌ FAILURE: No queries produced LLM analysis - still returning raw data")
        return False


def test_raw_data_detection():
    """Test that the assessor correctly identifies raw JSON responses"""

    assessor = ResponseQualityAssessor()

    # Simulate a raw JSON response like what we used to get
    raw_json_response = '{"success": true, "data": [{"id": 1, "name": "bucket1"}, {"id": 2, "name": "bucket2"}], "row_count": 2}'

    metrics = assessor.assess_response_quality(raw_json_response)

    print("\n🔍 Testing Raw Data Detection")
    print("-" * 30)
    print(f"Response Type: {metrics.response_type.value}")
    print(f"Analysis Score: {metrics.analysis_depth_score:.1f}")
    print(f"Is LLM Analysis: {assessor.is_llm_analysis(raw_json_response)}")

    assert metrics.response_type.value == "raw_data", "Should detect raw JSON"
    assert not assessor.is_llm_analysis(raw_json_response), "Should not classify raw JSON as analysis"
    print("✅ Raw data detection working correctly")


if __name__ == "__main__":
    print("🚀 Starting Enhanced Agent Analysis Tests")

    # Test raw data detection first
    test_raw_data_detection()

    # Test actual agent analysis capability
    success = test_agent_analysis_capability()

    if success:
        print("\n🎯 All tests passed! Agent is providing LLM analysis.")
        exit(0)
    else:
        print("\n⚠️ Some tests failed. Agent needs further improvement.")
        exit(1)