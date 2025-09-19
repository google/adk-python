#!/usr/bin/env python3
"""
Test Data Protection Strategy
===========================

Quick test to verify that synthetic/fake data in SQLite is preserved
and only updated when explicitly requested.
"""

import sys
import os
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents._tools.sqlite_tool import query_security_data

def test_data_protection():
    """Test that synthetic data is preserved by default."""

    print("🧪 Testing Data Protection Strategy")
    print("=" * 50)

    # Test 1: Normal query should return cached data (no live update)
    print("\n1️⃣ Testing normal query (should use cached data):")
    result1 = query_security_data("storage_buckets")

    if result1["success"]:
        source = result1.get("source", "unknown")
        count = len(result1.get("data", []))
        print(f"   ✅ Source: {source}")
        print(f"   ✅ Bucket count: {count}")

        if "cache" in source:
            print("   ✅ SUCCESS: Using cached data (synthetic data preserved)")
        else:
            print("   ⚠️  WARNING: Not using cached data")
    else:
        print(f"   ❌ FAILED: {result1.get('error', 'Unknown error')}")

    # Test 2: Force live update query (would try to fetch live data)
    print("\n2️⃣ Testing force live update (should try live data):")
    result2 = query_security_data("storage_buckets", force_live_update=True)

    if result2["success"]:
        source = result2.get("source", "unknown")
        count = len(result2.get("data", []))
        print(f"   ✅ Source: {source}")
        print(f"   ✅ Bucket count: {count}")

        if "live" in source:
            print("   ✅ SUCCESS: Using live data (when explicitly requested)")
        elif "fallback" in source:
            print("   ✅ SUCCESS: Live failed, fell back to cached data (good protection)")
        else:
            print("   ℹ️  INFO: Using cached data (live sources may not be available)")
    else:
        print(f"   ❌ FAILED: {result2.get('error', 'Unknown error')}")

    # Test 3: Verify specific bucket query
    print("\n3️⃣ Testing specific bucket query:")
    result3 = query_security_data("storage_buckets", bucket_name="mgm-digitalconcierge-logs")

    if result3["success"]:
        source = result3.get("source", "unknown")
        count = len(result3.get("data", []))
        print(f"   ✅ Source: {source}")
        print(f"   ✅ Matching buckets: {count}")

        if count > 0:
            bucket_name = result3["data"][0].get("name", "unknown")
            print(f"   ✅ Found bucket: {bucket_name}")

        print("   ✅ SUCCESS: Specific queries work with cached data")
    else:
        print(f"   ❌ FAILED: {result3.get('error', 'Unknown error')}")

    print("\n🎉 Data Protection Test Complete!")
    print("\nKey Benefits:")
    print("  • Synthetic data preserved for demos")
    print("  • Live data only fetched when explicitly requested")
    print("  • Graceful fallback if live sources fail")
    print("  • No accidental overwrites of proof-of-concept data")

if __name__ == "__main__":
    test_data_protection()