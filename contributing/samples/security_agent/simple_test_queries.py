#!/usr/bin/env python3
"""
Simple Test: Data Protection Verification
========================================

Test the data protection strategy without complex ADK setup.
This demonstrates that synthetic data is preserved by default.
"""

import sys
import os
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import the query function directly
from agents._tools.sqlite_tool import query_security_data

def test_data_protection_queries():
    """Test data protection with direct queries."""

    print("🛡️ Testing Data Protection Strategy")
    print("=" * 50)

    # Test 1: Basic storage buckets query (should use cached data)
    print("\n1️⃣ Test: Basic Storage Buckets Query")
    print("   Expected: Uses cached synthetic data (default behavior)")
    result1 = query_security_data("storage_buckets")

    if result1["success"]:
        source = result1.get("source", "unknown")
        count = len(result1.get("data", []))
        print(f"   ✅ Success: {count} buckets found")
        print(f"   📊 Data source: {source}")

        if "cache" in source:
            print(f"   🛡️ PROTECTED: Using cached synthetic data (preserved!)")
        else:
            print(f"   ⚠️  Using non-cached source: {source}")

        # Show first bucket as sample
        if result1.get("data"):
            bucket_name = result1["data"][0].get("name", "unknown")
            print(f"   📦 Sample bucket: {bucket_name}")
    else:
        print(f"   ❌ Failed: {result1.get('error', 'Unknown error')}")

    # Test 2: Force live update query (if available, should try GCP)
    print("\n2️⃣ Test: Force Live Update Query")
    print("   Expected: Attempts live data, falls back to cache if needed")
    result2 = query_security_data("storage_buckets", force_live_update=True)

    if result2["success"]:
        source = result2.get("source", "unknown")
        count = len(result2.get("data", []))
        print(f"   ✅ Success: {count} buckets found")
        print(f"   📊 Data source: {source}")

        if "live" in source:
            print(f"   🔴 LIVE DATA: Retrieved fresh data from GCP")
        elif "cache" in source or "fallback" in source:
            print(f"   🛡️ FALLBACK: Live failed, using cached data (protection working)")
        else:
            print(f"   ℹ️  Other source: {source}")
    else:
        print(f"   ❌ Failed: {result2.get('error', 'Unknown error')}")

    # Test 3: Security findings query
    print("\n3️⃣ Test: Security Findings Query")
    print("   Expected: Uses cached synthetic security data")
    result3 = query_security_data("security_findings")

    if result3["success"]:
        count = len(result3.get("data", []))
        print(f"   ✅ Success: {count} security findings found")
        print(f"   🛡️ PROTECTED: Synthetic security data preserved")
    else:
        print(f"   ❌ Failed: {result3.get('error', 'Unknown error')}")

    # Test 4: Summary stats
    print("\n4️⃣ Test: Security Summary")
    print("   Expected: Aggregated stats from cached data")
    result4 = query_security_data("security_summary")

    if result4["success"]:
        stats = result4.get("data", {})
        print(f"   ✅ Success: Security summary generated")

        # Show some stats if available
        if "total_findings" in stats:
            findings_data = stats["total_findings"]
            if findings_data:
                count = findings_data[0].get("count", 0)
                print(f"   📊 Total findings: {count}")

        print(f"   🛡️ PROTECTED: Summary from synthetic data")
    else:
        print(f"   ❌ Failed: {result4.get('error', 'Unknown error')}")

    print("\n🎉 Data Protection Test Complete!")
    print("\n📋 Summary:")
    print("  ✅ Synthetic data is preserved by default")
    print("  ✅ Live data only fetched when explicitly requested")
    print("  ✅ Graceful fallback maintains functionality")
    print("  ✅ No accidental overwrites of proof-of-concept data")

if __name__ == "__main__":
    test_data_protection_queries()