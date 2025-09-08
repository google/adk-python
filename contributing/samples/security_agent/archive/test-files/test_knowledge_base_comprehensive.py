#!/usr/bin/env python3
"""
Comprehensive Knowledge Base Testing Suite
==========================================

Complete test coverage for knowledge base integration including:
- All query types and parameters
- Edge cases and error handling
- Performance and accuracy validation
- End-to-end chat simulation
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents" / "gcp_security"))

# Import the SQLite tool
from sqlite_tool import query_security_data

class KnowledgeBaseTestSuite:
    """Comprehensive test suite for knowledge base integration"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.error_details = []
        
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a single test with error handling"""
        print(f"\n📝 Testing: {test_name}")
        print("-" * 50)
        
        self.tests_run += 1
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if result:
                print(f"✅ PASSED ({elapsed:.2f}s)")
                self.tests_passed += 1
                return True
            else:
                print(f"❌ FAILED ({elapsed:.2f}s)")
                self.tests_failed += 1
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"💥 ERROR ({elapsed:.2f}s): {e}")
            self.error_details.append(f"{test_name}: {e}")
            self.tests_failed += 1
            return False
    
    def test_basic_queries(self) -> bool:
        """Test all basic knowledge base query types"""
        
        queries = [
            ("knowledge_base", None, "Knowledge Base Overview"),
            ("coding_standards", None, "Coding Standards & Test Requirements"),
            ("enterprise_policies", None, "Enterprise Security Policies"),
            ("best_practices", None, "GCP Best Practices"),
            ("compliance", None, "Compliance Framework Requirements")
        ]
        
        all_passed = True
        for query_type, params, expected in queries:
            result = query_security_data(query_type, params)
            if expected not in result:
                print(f"  ❌ {query_type}: Expected '{expected}' not found")
                all_passed = False
            else:
                print(f"  ✅ {query_type}: Found expected content")
        
        return all_passed
    
    def test_filtered_queries(self) -> bool:
        """Test queries with filters and parameters"""
        
        filter_tests = [
            ("coding_standards", '{"language": "Python"}', "Python"),
            ("coding_standards", '{"severity": "ERROR"}', "ERROR"),
            ("enterprise_policies", '{"severity": "CRITICAL"}', "CRITICAL"),
            ("enterprise_policies", '{"category": "Access Control"}', "Access Control"),
            ("best_practices", '{"service": "Cloud Storage"}', "Cloud Storage"),
            ("compliance", '{"framework": "SOC2"}', "SOC2")
        ]
        
        all_passed = True
        for query_type, params, expected in filter_tests:
            result = query_security_data(query_type, params)
            if expected not in result:
                print(f"  ❌ {query_type} filter: Expected '{expected}' not found")
                all_passed = False
            else:
                print(f"  ✅ {query_type} filter: Found expected content")
        
        return all_passed
    
    def test_search_functionality(self) -> bool:
        """Test search across all knowledge base content"""
        
        search_tests = [
            ("coding_standards", '{"search": "test"}', 5),  # Should find 5 test-related standards
            ("coding_standards", '{"search": "security"}', 1),  # Should find secrets standard
            ("enterprise_policies", '{"search": "encryption"}', 1),  # Should find encryption policy
            ("best_practices", '{"search": "versioning"}', 1),  # Should find versioning practice
            ("compliance", '{"search": "access"}', 1)  # Should find access control requirement
        ]
        
        all_passed = True
        for query_type, params, expected_count in search_tests:
            result = query_security_data(query_type, params)
            
            # Count entries by looking for bullet points or numbered items
            if query_type == "coding_standards":
                # Count by looking for "📌" markers
                actual_count = result.count("📌")
            else:
                # For other types, just check if we got results
                actual_count = 1 if len(result) > 100 else 0
            
            if query_type == "coding_standards" and expected_count > 1:
                if actual_count >= expected_count:
                    print(f"  ✅ {query_type} search: Found {actual_count} items (expected {expected_count})")
                else:
                    print(f"  ❌ {query_type} search: Found {actual_count} items (expected {expected_count})")
                    all_passed = False
            else:
                if actual_count > 0:
                    print(f"  ✅ {query_type} search: Found results")
                else:
                    print(f"  ❌ {query_type} search: No results found")
                    all_passed = False
        
        return all_passed
    
    def test_test_standards_specifically(self) -> bool:
        """Test that all test standards are accessible"""
        
        result = query_security_data("coding_standards", '{"search": "testing"}')
        
        test_standards = [
            "Test Coverage Requirement",
            "Test Naming Convention", 
            "Mock External Services",
            "Test Data Management",
            "Assert Meaningful Messages"
        ]
        
        found_count = 0
        for standard in test_standards:
            if standard in result:
                print(f"  ✅ Found: {standard}")
                found_count += 1
            else:
                print(f"  ❌ Missing: {standard}")
        
        # Also try searching for "test" to catch tags
        result2 = query_security_data("coding_standards", '{"search": "test"}')
        for standard in test_standards:
            if standard not in result and standard in result2:
                print(f"  ✅ Found in test search: {standard}")
                found_count += 1
        
        success = found_count == len(test_standards)
        if success:
            print(f"  🎉 All {len(test_standards)} test standards accessible!")
        else:
            print(f"  ⚠️ Only {found_count}/{len(test_standards)} test standards found")
        
        return success
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid queries"""
        
        error_tests = [
            ("invalid_type", None, "Unknown query type"),
            ("coding_standards", '{"invalid": "json"}', None),  # Should not crash
            ("coding_standards", 'invalid json', None),  # Should handle bad JSON
            ("coding_standards", '{"language": "NonExistent"}', "No coding standards found")
        ]
        
        all_passed = True
        for query_type, params, expected_error in error_tests:
            try:
                result = query_security_data(query_type, params)
                
                if expected_error and expected_error in result:
                    print(f"  ✅ Error handling: Got expected error message")
                elif expected_error is None and len(result) > 0:
                    print(f"  ✅ Error handling: Query handled gracefully")
                else:
                    print(f"  ❌ Error handling: Unexpected result for {query_type}")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ Error handling: Unexpected exception for {query_type}: {e}")
                all_passed = False
        
        return all_passed
    
    def test_performance(self) -> bool:
        """Test query performance"""
        
        performance_tests = [
            ("knowledge_base", None),
            ("coding_standards", None),
            ("enterprise_policies", None),
            ("best_practices", None),
            ("compliance", None)
        ]
        
        max_time = 2.0  # 2 seconds max
        all_passed = True
        
        for query_type, params in performance_tests:
            start_time = time.time()
            result = query_security_data(query_type, params)
            elapsed = time.time() - start_time
            
            if elapsed <= max_time and len(result) > 0:
                print(f"  ✅ {query_type}: {elapsed:.3f}s (under {max_time}s limit)")
            else:
                print(f"  ❌ {query_type}: {elapsed:.3f}s (over {max_time}s limit)")
                all_passed = False
        
        return all_passed
    
    def test_data_integrity(self) -> bool:
        """Test that data returned is complete and accurate"""
        
        # Test that we have the expected number of items
        kb_result = query_security_data("knowledge_base", None)
        
        expected_counts = {
            "Enterprise Policies: 3": "policies",
            "Coding Standards: 7": "standards", 
            "Compliance Requirements: 2": "compliance",
            "Best Practices: 2": "practices"
        }
        
        all_passed = True
        for expected_text, category in expected_counts.items():
            if expected_text in kb_result:
                print(f"  ✅ Data integrity: {category} count correct")
            else:
                print(f"  ❌ Data integrity: {category} count incorrect")
                all_passed = False
        
        return all_passed
    
    def simulate_chat_queries(self) -> bool:
        """Simulate realistic chat queries users might ask"""
        
        chat_scenarios = [
            ("What are our coding standards?", "coding_standards", None, "standards"),
            ("Show me test requirements", "coding_standards", '{"search": "test"}', "Test"),
            ("What are our critical policies?", "enterprise_policies", '{"severity": "CRITICAL"}', "CRITICAL"),
            ("Show Python standards", "coding_standards", '{"language": "Python"}', "Python"),
            ("What GCP best practices do we have?", "best_practices", None, "GCP Best Practices"),
            ("Check SOC2 compliance", "compliance", '{"framework": "SOC2"}', "SOC2"),
            ("Show all security policies", "enterprise_policies", None, "Enterprise Security"),
            ("What test coverage is required?", "coding_standards", '{"search": "coverage"}', "coverage")
        ]
        
        all_passed = True
        for question, query_type, params, expected in chat_scenarios:
            print(f"    💬 User: \"{question}\"")
            result = query_security_data(query_type, params)
            
            if expected.lower() in result.lower():
                print(f"    🤖 Agent: Found relevant information ({len(result)} chars)")
                print(f"    ✅ Response contains expected content")
            else:
                print(f"    ❌ Response missing expected content: {expected}")
                all_passed = False
            print()
        
        return all_passed
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        
        print("=" * 70)
        print("🧪 COMPREHENSIVE KNOWLEDGE BASE TEST SUITE")
        print("=" * 70)
        
        # Run all test categories
        test_results = {}
        
        test_results["basic_queries"] = self.run_test(
            "Basic Query Types", self.test_basic_queries
        )
        
        test_results["filtered_queries"] = self.run_test(
            "Filtered Queries", self.test_filtered_queries
        )
        
        test_results["search_functionality"] = self.run_test(
            "Search Functionality", self.test_search_functionality
        )
        
        test_results["test_standards"] = self.run_test(
            "Test Standards Access", self.test_test_standards_specifically
        )
        
        test_results["error_handling"] = self.run_test(
            "Error Handling", self.test_error_handling
        )
        
        test_results["performance"] = self.run_test(
            "Performance", self.test_performance
        )
        
        test_results["data_integrity"] = self.run_test(
            "Data Integrity", self.test_data_integrity
        )
        
        test_results["chat_simulation"] = self.run_test(
            "Chat Query Simulation", self.simulate_chat_queries
        )
        
        # Final summary
        print("\n" + "=" * 70)
        print("📊 FINAL TEST RESULTS")
        print("=" * 70)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"\n🎯 Overall Results:")
        print(f"  • Tests Run: {self.tests_run}")
        print(f"  • Tests Passed: {self.tests_passed}")
        print(f"  • Tests Failed: {self.tests_failed}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 Test Category Results:")
        for category, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  • {category.replace('_', ' ').title()}: {status}")
        
        if self.error_details:
            print(f"\n🔍 Error Details:")
            for error in self.error_details:
                print(f"  • {error}")
        
        if success_rate == 100:
            print(f"\n🎉 PERFECT SCORE! Knowledge base integration is 100% functional!")
            print(f"\n✨ Ready for production use!")
        elif success_rate >= 95:
            print(f"\n🌟 EXCELLENT! Knowledge base integration is highly functional!")
        elif success_rate >= 80:
            print(f"\n👍 GOOD! Knowledge base integration is mostly functional!")
        else:
            print(f"\n⚠️ Needs improvement. Several issues found.")
        
        return {
            "success_rate": success_rate,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "category_results": test_results,
            "errors": self.error_details
        }


def main():
    """Run the comprehensive test suite"""
    test_suite = KnowledgeBaseTestSuite()
    results = test_suite.run_all_tests()
    
    # Return appropriate exit code
    return 0 if results["success_rate"] == 100 else 1


if __name__ == "__main__":
    exit(main())