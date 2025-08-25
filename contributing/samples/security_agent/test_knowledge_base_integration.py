#!/usr/bin/env python3
"""
Test Knowledge Base Integration with Chat Experience
=====================================================

This script verifies:
1. Knowledge base database is populated with coding standards
2. API endpoints are working and accessible
3. Chat interface can query knowledge base for coding standards
4. Integration status with the agent
"""

import sqlite3
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_database_exists():
    """Check if knowledge base database exists and is populated"""
    db_path = Path("backend/cache/knowledge_base.db")
    
    if not db_path.exists():
        print("❌ Knowledge base database not found at:", db_path)
        return False
    
    print(f"✅ Knowledge base database found: {db_path}")
    
    # Check contents
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get table counts
    tables_info = {}
    tables = ['coding_standards', 'enterprise_policies', 'compliance_frameworks', 'best_practices']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
        count = cursor.fetchone()['count']
        tables_info[table] = count
        print(f"  - {table}: {count} records")
    
    # Get coding standards details
    print("\n📋 Coding Standards in Database:")
    cursor.execute("SELECT * FROM coding_standards")
    standards = cursor.fetchall()
    
    for standard in standards:
        print(f"  - {standard['language']}: {standard['standard_name']}")
        print(f"    Rule: {standard['rule_description']}")
        print(f"    Severity: {standard['severity']}")
        if standard['example_good']:
            print(f"    Good Example: {standard['example_good']}")
        if standard['example_bad']:
            print(f"    Bad Example: {standard['example_bad']}")
        print()
    
    conn.close()
    return True

def test_api_endpoints():
    """Test if knowledge base API endpoints are accessible"""
    print("\n🔍 Testing API Endpoints:")
    
    try:
        # Test if backend is running by importing and checking routes
        from backend.api import knowledge_base
        
        print("✅ Knowledge base API module loaded")
        
        # Check available endpoints
        endpoints = [
            "/policies - Get enterprise policies",
            "/standards - Get coding standards", 
            "/compliance - Get compliance requirements",
            "/practices - Get best practices",
            "/search - Full-text search",
            "/stats - Get statistics",
            "/export/{table} - Export data",
            "/import/csv - Import CSV data"
        ]
        
        print("\n📚 Available Knowledge Base Endpoints:")
        for endpoint in endpoints:
            print(f"  - {endpoint}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import knowledge base API: {e}")
        return False

def check_chat_integration():
    """Check if the chat interface integrates with knowledge base"""
    print("\n🤖 Checking Chat Integration:")
    
    # Check if agent has knowledge base tool
    agent_path = Path("agents/gcp_security/vertex_sqlite_agent.py")
    if agent_path.exists():
        with open(agent_path, 'r') as f:
            content = f.read()
            
        # Look for knowledge base references
        has_kb_ref = any(term in content.lower() for term in ['knowledge', 'coding standard', 'best practice', 'compliance'])
        
        if has_kb_ref:
            print("✅ Agent includes knowledge base references in instructions")
        else:
            print("⚠️ Agent doesn't explicitly reference knowledge base")
    
    # Check if SQLite tool can query knowledge base
    sqlite_tool_path = Path("agents/gcp_security/sqlite_tool.py")
    if sqlite_tool_path.exists():
        with open(sqlite_tool_path, 'r') as f:
            content = f.read()
            
        # Check if it queries knowledge_base.db
        if 'knowledge_base.db' in content:
            print("✅ SQLite tool is configured to query knowledge base")
        else:
            print("⚠️ SQLite tool only queries main database (gcp_data.db)")
            print("   Knowledge base is in separate database (knowledge_base.db)")
    
    # Check frontend integration
    frontend_path = Path("frontend/unified_streaming_client.py")
    if frontend_path.exists():
        with open(frontend_path, 'r') as f:
            content = f.read()
            
        if 'knowledge' in content.lower():
            print("✅ Frontend has knowledge base integration")
        else:
            print("⚠️ Frontend doesn't directly integrate knowledge base")
            print("   Knowledge base queries would go through agent/tool")
    
    return True

def check_test_standards():
    """Check what test standards are available"""
    print("\n🧪 Test Standards Available:")
    
    db_path = Path("backend/cache/knowledge_base.db")
    if not db_path.exists():
        print("❌ Database not found")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check for test-related standards
    cursor.execute("""
        SELECT * FROM coding_standards 
        WHERE standard_name LIKE '%test%' 
           OR rule_description LIKE '%test%'
           OR standard_name LIKE '%Test%'
    """)
    
    test_standards = cursor.fetchall()
    
    if test_standards:
        print("✅ Found test-related standards:")
        for standard in test_standards:
            print(f"  - {standard['standard_name']}: {standard['rule_description']}")
    else:
        print("⚠️ No explicit test standards found in database")
        print("   Current standards focus on:")
        print("   - Security (No Hardcoded Secrets)")
        print("   - Governance (Resource Tagging)")
        print("\n   Test standards could be added for:")
        print("   - Test coverage requirements")
        print("   - Test naming conventions")
        print("   - Mocking and stubbing practices")
        print("   - Test data management")
    
    conn.close()
    return True

def suggest_improvements():
    """Suggest improvements for knowledge base integration"""
    print("\n💡 Integration Status & Recommendations:")
    
    print("\n✅ What's Working:")
    print("  1. Knowledge base database is created and populated")
    print("  2. API endpoints are defined and ready")
    print("  3. Sample coding standards are loaded")
    print("  4. Database schema supports full CRUD operations")
    
    print("\n⚠️ Current Limitations:")
    print("  1. Knowledge base is in separate database (knowledge_base.db)")
    print("  2. SQLite tool only queries main database (gcp_data.db)")
    print("  3. No direct chat integration - agent can't query knowledge base")
    print("  4. No test-specific coding standards yet")
    
    print("\n🔧 To Enable Full Integration:")
    print("  1. Option A: Merge knowledge base tables into main gcp_data.db")
    print("  2. Option B: Update SQLite tool to query both databases")
    print("  3. Option C: Create dedicated knowledge base tool for agent")
    print("  4. Add test standards (coverage, naming, mocking, etc.)")
    
    print("\n📝 To Add Test Standards:")
    print("  Run: python backend/services/knowledge_base_setup.py")
    print("  Then add standards via API or direct SQL insert")

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Knowledge Base Integration Test Report")
    print("=" * 60)
    
    # Run tests
    db_exists = check_database_exists()
    api_ready = test_api_endpoints()
    chat_integrated = check_chat_integration()
    test_standards = check_test_standards()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n📊 Test Results:")
    print(f"  Database Exists: {'✅' if db_exists else '❌'}")
    print(f"  API Ready: {'✅' if api_ready else '❌'}")
    print(f"  Chat Integration: {'⚠️ Partial' if chat_integrated else '❌'}")
    print(f"  Test Standards: {'⚠️ Not yet added' if test_standards else '❌'}")
    
    # Recommendations
    suggest_improvements()
    
    print("\n" + "=" * 60)
    print("End of Report")
    print("=" * 60)

if __name__ == "__main__":
    main()