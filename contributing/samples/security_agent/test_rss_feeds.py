#!/usr/bin/env python3
"""
Test script to verify the RSS feed tools work with real deployed data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rss_agent_tools():
    """Test the RSS feed tools with real data"""
    print("🔍 Testing RSS Feed Tools with Real Data")
    print("=" * 50)

    try:
        from agents._tools.feed_tools import (
            query_gcp_release_notes,
            query_security_threat_feeds,
            get_feed_statistics,
            search_feeds_by_keyword
        )

        # Test 1: Query GCP Release Notes
        print("\n📰 Testing GCP Release Notes Query...")
        release_results = query_gcp_release_notes(days_back=7, security_only=False)
        print(f"Result length: {len(release_results)} characters")
        print(f"Preview: {release_results[:200]}...")

        # Test 2: Query Security Feeds
        print("\n🔒 Testing Security Threat Feeds Query...")
        security_results = query_security_threat_feeds(days_back=7)
        print(f"Result length: {len(security_results)} characters")
        print(f"Preview: {security_results[:200]}...")

        # Test 3: Get Feed Statistics
        print("\n📊 Testing Feed Statistics...")
        stats_results = get_feed_statistics()
        print(f"Result length: {len(stats_results)} characters")
        print(f"Preview: {stats_results[:200]}...")

        # Test 4: Search by Keyword
        print("\n🔍 Testing Keyword Search...")
        search_results = search_feeds_by_keyword("security", days_back=7)
        print(f"Result length: {len(search_results)} characters")
        print(f"Preview: {search_results[:200]}...")

        print("\n✅ All RSS feed tools tested successfully!")
        print("The agent now has access to:")
        print("  - Google Cloud release notes and service announcements")
        print("  - Security threat feeds with CVE and vulnerability data")
        print("  - Cross-feed search capabilities")
        print("  - Feed statistics and freshness monitoring")

        return True

    except Exception as e:
        print(f"❌ Error testing RSS feed tools: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rss_agent_tools()
    sys.exit(0 if success else 1)