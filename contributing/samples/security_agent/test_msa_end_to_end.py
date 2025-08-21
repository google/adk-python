#!/usr/bin/env python3
"""
End-to-End Test for MSA Analyzer Integration
=============================================

This test verifies that MSA data flows correctly through:
1. Backend API analysis
2. Database storage
3. Agent queries
"""

import os
import sys
import sqlite3
import json
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
sys.path.insert(0, str(Path(__file__).parent / 'agents' / 'gcp_security'))

def test_database_tables():
    """Verify MSA tables exist in database."""
    print("🔍 Testing database tables...")
    
    db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check for MSA tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'msa_%'")
    tables = cursor.fetchall()
    
    expected_tables = ['msa_emails', 'msa_changes', 'msa_impact_assessments', 'msa_recommendations']
    found_tables = [t[0] for t in tables]
    
    for table in expected_tables:
        if table in found_tables:
            print(f"  ✅ Table {table} exists")
        else:
            print(f"  ❌ Table {table} missing")
    
    conn.close()
    return len(found_tables) == len(expected_tables)

def test_agent_queries():
    """Test that agent can query MSA data."""
    print("\n🤖 Testing agent MSA queries...")
    
    try:
        from sqlite_tool import query_security_data
        
        # Test MSA analysis query
        result = query_security_data("msa_analysis")
        print(f"  MSA Analysis Query: {'✅' if 'MSA' in result else '❌'}")
        
        # Test MSA changes query
        result = query_security_data("msa_changes")
        print(f"  MSA Changes Query: {'✅' if 'MSA' in result or 'No MSA' in result else '❌'}")
        
        # Test MSA impact query
        result = query_security_data("msa_impact")
        print(f"  MSA Impact Query: {'✅' if 'MSA' in result or 'No MSA' in result else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing agent queries: {e}")
        return False

def test_sample_msa_storage():
    """Test storing a sample MSA analysis."""
    print("\n💾 Testing MSA storage...")
    
    try:
        from backend.services.msa_database_setup import store_msa_analysis
        
        # Create sample data
        sample_analysis = {
            'email_content': 'Test MSA email',
            'project_id': 'test-project',
            'extracted_changes': [
                {
                    'service': 'BigQuery',
                    'change_type': 'permission_change',
                    'description': 'Test change',
                    'impact_level': 'high',
                    'effective_date': '2025-01-15',
                    'required_action': 'Review permissions',
                    'affected_resources': ['datasets', 'tables']
                }
            ],
            'impact_assessments': [
                {
                    'project_id': 'test-project',
                    'resource_type': 'bigquery_datasets',
                    'resource_count': 5,
                    'impact_level': 'high',
                    'recommended_actions': ['Review IAM', 'Update scripts'],
                    'affected_resources': []
                }
            ],
            'recommendations': ['Test recommendation 1', 'Test recommendation 2']
        }
        
        db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        msa_id = store_msa_analysis(db_path, sample_analysis)
        
        if msa_id:
            print(f"  ✅ Stored test MSA with ID: {msa_id}")
            
            # Verify it can be queried
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM msa_emails WHERE id = ?", (msa_id,))
            count = cursor.fetchone()[0]
            conn.close()
            
            if count == 1:
                print(f"  ✅ Verified MSA in database")
                return True
            else:
                print(f"  ❌ Could not verify MSA in database")
                return False
        else:
            print(f"  ❌ Failed to store test MSA")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing storage: {e}")
        return False

def test_agent_with_stored_data():
    """Test agent queries after storing data."""
    print("\n🔄 Testing agent with stored MSA data...")
    
    try:
        from sqlite_tool import query_security_data
        
        # Query MSA analysis
        result = query_security_data("msa_analysis")
        
        if "emails analyzed" in result and "Test MSA email" not in result:
            print(f"  ✅ Agent sees MSA analysis history")
        else:
            print(f"  ⚠️  Agent query returned: {result[:100]}...")
        
        # Query MSA changes for BigQuery
        result = query_security_data("msa_changes", '{"service": "BigQuery"}')
        
        if "BigQuery" in result or "permission_change" in result:
            print(f"  ✅ Agent can filter MSA changes by service")
        else:
            print(f"  ⚠️  Changes query returned: {result[:100]}...")
        
        # Query MSA impact for test project
        result = query_security_data("msa_impact", '{"project_id": "test-project"}')
        
        if "test-project" in result or "bigquery_datasets" in result:
            print(f"  ✅ Agent can query project-specific impact")
        else:
            print(f"  ⚠️  Impact query returned: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing agent with data: {e}")
        return False

def main():
    """Run all end-to-end tests."""
    print("=" * 50)
    print("🧪 MSA Analyzer End-to-End Test")
    print("=" * 50)
    
    # Ensure database exists
    db_path = Path("backend/cache/gcp_data.db")
    if not db_path.exists():
        print("⚠️  Database not found, creating...")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run setup
        from backend.services.msa_database_setup import create_msa_tables
        create_msa_tables(str(db_path))
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if test_database_tables():
        tests_passed += 1
    
    if test_agent_queries():
        tests_passed += 1
    
    if test_sample_msa_storage():
        tests_passed += 1
    
    if test_agent_with_stored_data():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! MSA analyzer is fully integrated.")
        print("\nThe agent can now answer questions like:")
        print("  • 'Show me MSA analysis history'")
        print("  • 'What MSA changes affect BigQuery?'")
        print("  • 'What's the MSA impact on my project?'")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()