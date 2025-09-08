#!/usr/bin/env python3
"""
Test Knowledge Base Integration with Chat Agent
================================================

This script tests that the knowledge base is fully integrated 
and queryable through the chat experience.
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents" / "gcp_security"))

# Import the SQLite tool
from sqlite_tool import query_security_data

def test_knowledge_base_queries():
    """Test all knowledge base query types"""
    
    print("=" * 60)
    print("Testing Knowledge Base Integration")
    print("=" * 60)
    
    tests = [
        {
            "name": "Knowledge Base Overview",
            "query": ("knowledge_base", None),
            "expected": "Knowledge Base Overview"
        },
        {
            "name": "Coding Standards",
            "query": ("coding_standards", None),
            "expected": "Coding Standards & Test Requirements"
        },
        {
            "name": "Test Standards Search",
            "query": ("coding_standards", '{"search": "test"}'),
            "expected": "Test"
        },
        {
            "name": "Enterprise Policies",
            "query": ("enterprise_policies", None),
            "expected": "Enterprise Security Policies"
        },
        {
            "name": "Best Practices",
            "query": ("best_practices", None),
            "expected": "GCP Best Practices"
        },
        {
            "name": "Compliance Requirements",
            "query": ("compliance", None),
            "expected": "Compliance Framework Requirements"
        },
        {
            "name": "Python Standards",
            "query": ("coding_standards", '{"language": "Python"}'),
            "expected": "Python"
        },
        {
            "name": "Critical Policies",
            "query": ("enterprise_policies", '{"severity": "CRITICAL"}'),
            "expected": "CRITICAL"
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n📝 Testing: {test['name']}")
        print("-" * 40)
        
        try:
            result = query_security_data(test['query'][0], test['query'][1])
            
            if test['expected'] in result:
                print(f"✅ PASSED - Found expected content: '{test['expected']}'")
                # Show first 200 chars of result
                print(f"   Result preview: {result[:200]}...")
                passed += 1
            else:
                print(f"❌ FAILED - Expected '{test['expected']}' not found")
                print(f"   Got: {result[:200]}...")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    # Test specific test standards
    print("\n" + "=" * 60)
    print("🧪 Verifying Test Standards")
    print("-" * 40)
    
    result = query_security_data("coding_standards", '{"search": "test"}')
    
    test_standards = [
        "Test Coverage Requirement",
        "Test Naming Convention",
        "Mock External Services",
        "Test Data Management",
        "Assert Meaningful Messages"
    ]
    
    found_standards = []
    for standard in test_standards:
        if standard in result:
            found_standards.append(standard)
            print(f"✅ Found: {standard}")
        else:
            print(f"❌ Missing: {standard}")
    
    if len(found_standards) == len(test_standards):
        print("\n✅ All test standards are available!")
        passed += 1
    else:
        print(f"\n⚠️ Only {len(found_standards)}/{len(test_standards)} test standards found")
        failed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"  • Tests Passed: {passed}")
    print(f"  • Tests Failed: {failed}")
    print(f"  • Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 SUCCESS! Knowledge base is fully integrated!")
        print("\nYou can now ask the chat agent questions like:")
        print("  • 'What are our coding standards?'")
        print("  • 'Show me test requirements'")
        print("  • 'What are our critical security policies?'")
        print("  • 'Show GCP best practices for Cloud Storage'")
        print("  • 'Check our compliance status'")
    else:
        print("\n⚠️ Some tests failed. Please check the integration.")
    
    return failed == 0


def main():
    """Run the test"""
    success = test_knowledge_base_queries()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())