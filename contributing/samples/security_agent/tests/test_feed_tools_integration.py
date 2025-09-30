#!/usr/bin/env python3
"""
Test script to verify RSS feed tools integration
Tests imports and basic functionality without requiring BigQuery credentials
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all RSS feed tools can be imported correctly"""
    print("🔍 Testing RSS feed tools imports...")

    try:
        # Test import from _tools package
        from agents._tools import (
            query_gcp_release_notes,
            query_security_threat_feeds,
            get_feed_statistics,
            search_feeds_by_keyword
        )
        print("✅ Successfully imported all RSS feed tools from agents._tools")

        # Test direct import from feed_tools module
        from agents._tools.feed_tools import (
            query_gcp_release_notes as direct_query_gcp,
            query_security_threat_feeds as direct_query_security,
            get_feed_statistics as direct_get_stats,
            search_feeds_by_keyword as direct_search
        )
        print("✅ Successfully imported all RSS feed tools from feed_tools module")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_agent_integration():
    """Test that RSS feed tools are properly integrated into the agent"""
    print("\n🤖 Testing agent integration...")

    try:
        from agents.agent import root_agent, tools

        # Check that RSS feed tools are in the agent's tools list
        tool_names = []
        for tool in tools:
            # ADK FunctionTool has a 'name' attribute with the function name
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)

        print(f"   Found tool names: {tool_names}")

        expected_feed_tools = [
            'query_gcp_release_notes',
            'query_security_threat_feeds',
            'get_feed_statistics',
            'search_feeds_by_keyword'
        ]

        missing_tools = []
        found_tools = []

        for expected_tool in expected_feed_tools:
            if expected_tool in tool_names:
                found_tools.append(expected_tool)
            else:
                missing_tools.append(expected_tool)

        if missing_tools:
            print(f"❌ Missing RSS feed tools from agent: {missing_tools}")
            print(f"   Available tools: {tool_names}")
            return False
        else:
            print(f"✅ All RSS feed tools found in agent: {found_tools}")
            print(f"   Total tools in agent: {len(tools)}")
            return True

    except Exception as e:
        print(f"❌ Agent integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_signatures():
    """Test that RSS feed functions have expected signatures"""
    print("\n📝 Testing function signatures...")

    try:
        from agents._tools.feed_tools import (
            query_gcp_release_notes,
            query_security_threat_feeds,
            get_feed_statistics,
            search_feeds_by_keyword
        )

        # Test function signatures
        import inspect

        # Test query_gcp_release_notes signature
        sig = inspect.signature(query_gcp_release_notes)
        expected_params = ['days_back', 'security_only', 'service_category', 'min_security_score']
        actual_params = list(sig.parameters.keys())

        if all(param in actual_params for param in expected_params):
            print("✅ query_gcp_release_notes has correct parameters")
        else:
            print(f"❌ query_gcp_release_notes parameters mismatch. Expected: {expected_params}, Got: {actual_params}")
            return False

        # Test query_security_threat_feeds signature
        sig = inspect.signature(query_security_threat_feeds)
        expected_params = ['days_back', 'severity', 'threat_type', 'min_cvss_score', 'cloud_related_only', 'immediate_action_only']
        actual_params = list(sig.parameters.keys())

        if all(param in actual_params for param in expected_params):
            print("✅ query_security_threat_feeds has correct parameters")
        else:
            print(f"❌ query_security_threat_feeds parameters mismatch. Expected: {expected_params}, Got: {actual_params}")
            return False

        # Test get_feed_statistics signature
        sig = inspect.signature(get_feed_statistics)
        if len(sig.parameters) == 0:
            print("✅ get_feed_statistics has correct signature (no parameters)")
        else:
            print(f"❌ get_feed_statistics should have no parameters, got: {list(sig.parameters.keys())}")
            return False

        # Test search_feeds_by_keyword signature
        sig = inspect.signature(search_feeds_by_keyword)
        expected_params = ['keyword', 'days_back', 'include_release_notes', 'include_security_feeds']
        actual_params = list(sig.parameters.keys())

        if all(param in actual_params for param in expected_params):
            print("✅ search_feeds_by_keyword has correct parameters")
        else:
            print(f"❌ search_feeds_by_keyword parameters mismatch. Expected: {expected_params}, Got: {actual_params}")
            return False

        return True

    except Exception as e:
        print(f"❌ Function signature test error: {e}")
        return False

def test_error_handling():
    """Test that RSS feed functions handle errors gracefully"""
    print("\n🛡️ Testing error handling...")

    try:
        from agents._tools.feed_tools import (
            query_gcp_release_notes,
            query_security_threat_feeds,
            get_feed_statistics,
            search_feeds_by_keyword
        )

        # These should return string messages rather than throwing exceptions
        # They may return errors or "no results found" messages

        result1 = query_gcp_release_notes()
        if isinstance(result1, str):
            print(f"✅ query_gcp_release_notes returns string: {result1[:100]}...")
        else:
            print(f"❌ query_gcp_release_notes should return string, got: {type(result1)}")
            return False

        result2 = get_feed_statistics()
        if isinstance(result2, str):
            print(f"✅ get_feed_statistics returns string: {result2[:100]}...")
        else:
            print(f"❌ get_feed_statistics should return string, got: {type(result2)}")
            return False

        result3 = search_feeds_by_keyword("test")
        if isinstance(result3, str):
            print(f"✅ search_feeds_by_keyword returns string: {result3[:100]}...")
        else:
            print(f"❌ search_feeds_by_keyword should return string, got: {type(result3)}")
            return False

        # Test invalid query (should handle gracefully)
        result4 = query_security_threat_feeds(min_cvss_score=-1)  # Invalid score
        if isinstance(result4, str):
            print("✅ query_security_threat_feeds handles invalid parameters gracefully")
        else:
            print(f"❌ query_security_threat_feeds should handle invalid params, got: {type(result4)}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing RSS Feed Tools Integration")
    print("=" * 50)

    tests = [
        test_imports,
        test_agent_integration,
        test_function_signatures,
        test_error_handling
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test failed: {test.__name__}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All RSS feed tools integration tests PASSED!")
        return True
    else:
        print("💥 Some tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)