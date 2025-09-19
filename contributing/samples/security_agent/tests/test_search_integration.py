#!/usr/bin/env python3
"""
Test Google Search Integration with Security Agent
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents', 'tools'))

from sqlite_tool import query_security_data

def test_search_integration():
    """Test the search functionality integration."""
    print("🔍 Testing Google Search Integration...")

    # Test 1: Direct search query
    print("\n1. Testing direct search_docs query:")
    result = query_security_data("search_docs", query="GCP bucket security", search_type="gcp_docs", num_results=3)
    print(f"   Result success: {result.get('success')}")
    print(f"   Result source: {result.get('source')}")
    if result.get('success'):
        print(f"   Found {result.get('count', 0)} results")
        for i, doc in enumerate(result.get('data', [])[:2]):  # Show first 2
            print(f"   - {doc.get('title', 'No title')}")
    else:
        print(f"   Error: {result.get('error')}")

    # Test 2: Storage buckets with search fallback
    print("\n2. Testing storage_buckets with search fallback:")
    result = query_security_data("storage_buckets")
    print(f"   Result success: {result.get('success')}")
    print(f"   Result source: {result.get('source')}")
    if result.get('success'):
        print(f"   Data entries: {len(result.get('data', []))}")
        if result.get('source') == 'google_search_fallback':
            print("   ✅ Google Search fallback activated!")
        elif result.get('source') == 'live_gcp_api':
            print("   ✅ Live GCP data retrieved!")
        elif result.get('source') == 'sqlite_cache':
            print("   ✅ SQLite cache used!")
    else:
        print(f"   Error: {result.get('error')}")

if __name__ == "__main__":
    test_search_integration()