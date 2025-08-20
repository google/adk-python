#!/usr/bin/env python3
"""
Test Dashboard Functionality
============================

Quick test to verify dashboard can load and process data correctly.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "frontend"))

def test_dashboard_import():
    """Test that dashboard module can be imported"""
    try:
        from dashboard import SecurityDashboard
        print("✅ Dashboard module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import dashboard module: {e}")
        return False

def test_database_connection():
    """Test database connection and basic queries"""
    try:
        from dashboard import SecurityDashboard
        
        database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        if not os.path.exists(database_path):
            print(f"⚠️  Database not found at {database_path}")
            return False
            
        dashboard = SecurityDashboard(database_path)
        
        # Test connection
        with dashboard.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
        print(f"✅ Database connection successful - Found {len(tables)} tables")
        for table in tables:
            print(f"   • {table[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_metrics_calculation():
    """Test metrics calculation functions"""
    try:
        from dashboard import SecurityDashboard
        
        database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        if not os.path.exists(database_path):
            print("⚠️  Skipping metrics test - no database")
            return True
            
        dashboard = SecurityDashboard(database_path)
        
        # Test overview metrics
        metrics = dashboard.get_overview_metrics()
        print("✅ Overview metrics calculated successfully")
        print(f"   • Total assets: {metrics.get('total_assets', 0)}")
        print(f"   • Security findings: {metrics.get('total_security_findings', 0)}")
        print(f"   • Storage buckets: {metrics.get('total_storage_buckets', 0)}")
        
        # Test security findings analysis
        findings_df = dashboard.get_security_findings_analysis()
        print(f"✅ Security findings analysis - {len(findings_df)} findings")
        
        return True
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        return False

def test_visualization_data():
    """Test data preparation for visualizations"""
    try:
        import pandas as pd
        import plotly.express as px
        
        # Test basic data structures
        test_data = pd.DataFrame({
            'category': ['Network', 'Storage', 'IAM', 'Compute'],
            'count': [10, 5, 8, 12]
        })
        
        # Test plotly chart creation
        fig = px.bar(test_data, x='category', y='count', title='Test Chart')
        
        print("✅ Visualization components working")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all dashboard tests"""
    print("🧪 Testing GCP Security Dashboard Components")
    print("=" * 50)
    
    tests = [
        ("Dashboard Import", test_dashboard_import),
        ("Database Connection", test_database_connection),
        ("Metrics Calculation", test_metrics_calculation),
        ("Visualization Data", test_visualization_data)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed: {test_name}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All dashboard tests passed!")
        print("\n✅ Dashboard is ready to use:")
        print("   • python run_dashboard.py              # Standalone dashboard")
        print("   • python run_frontend.py              # Full app with dashboard tab")
    else:
        print(f"⚠️  {total - passed} tests failed - check configuration")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())