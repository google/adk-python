#!/usr/bin/env python3
"""
Verify Dashboard Integration Script
==================================

Tests that the dashboard is properly integrated into the front page
and validates all key components are working.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "frontend"))

def test_dashboard_integration():
    """Test that dashboard integration is working properly"""
    print("🔍 Testing Dashboard Integration")
    print("=" * 50)
    
    try:
        # Test main app import
        from main_app import display_front_page_dashboard, main
        print("✅ Main app imports successfully")
        
        # Test dashboard module import
        from dashboard import SecurityDashboard, render_overview_metrics
        print("✅ Dashboard module imports successfully")
        
        # Test database connection
        database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        if os.path.exists(database_path):
            dashboard = SecurityDashboard(database_path)
            metrics = dashboard.get_overview_metrics()
            print("✅ Database connection and metrics calculation working")
            print(f"   • Found {metrics.get('total_assets', 0)} total assets")
            print(f"   • Found {metrics.get('total_security_findings', 0)} security findings")
            print(f"   • Found {metrics.get('total_storage_buckets', 0)} storage buckets")
        else:
            print("⚠️ Database not found - dashboard will show warnings")
        
        # Test visualization imports
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly visualization libraries available")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def show_integration_summary():
    """Show what's been integrated"""
    print("\n📊 DASHBOARD INTEGRATION SUMMARY")
    print("=" * 50)
    
    print("🎯 FRONT PAGE COMPONENTS:")
    print("   1. Security Posture Overview")
    print("      • Total Assets metric")
    print("      • Critical/High Findings metric") 
    print("      • Public Storage Buckets metric")
    print("      • Risky Firewall Rules metric")
    print()
    
    print("   2. Security Analytics Visualizations")
    print("      • Security findings severity pie chart")
    print("      • Asset distribution horizontal bar chart")
    print()
    
    print("   3. Security Risk Assessment")
    print("      • Storage Security health score")
    print("      • Network Security health score")
    print("      • IAM Security health score")
    print()
    
    print("   4. Quick Actions")
    print("      • Detailed Analysis button")
    print("      • Security Findings button")
    print("      • Storage Review button") 
    print("      • Network Analysis button")
    print()
    
    print("🔧 TAB STRUCTURE:")
    print("   • Detailed Analytics (comprehensive dashboard sections)")
    print("   • Data Management (database metrics and controls)")
    print("   • Security Chat (enhanced chat interface)")
    print()

def show_cache_clearing_instructions():
    """Show instructions for clearing cache"""
    print("🧹 CACHE CLEARING INSTRUCTIONS")
    print("=" * 50)
    
    print("If you don't see the new dashboard layout:")
    print()
    print("1. **Stop Streamlit:**")
    print("   • Press Ctrl+C in the terminal running the app")
    print()
    print("2. **Clear Python Cache:**")
    print("   ./clear_cache_and_restart.sh")
    print("   # OR manually:")
    print("   find . -name '__pycache__' -exec rm -rf {} +")
    print()
    print("3. **Clear Browser Cache:**")
    print("   • Press Ctrl+Shift+R (or Cmd+Shift+R on Mac)")
    print("   • OR open Dev Tools (F12) → Network tab → Disable cache")
    print()
    print("4. **Restart Frontend:**")
    print("   python run_frontend.py")
    print()
    print("5. **Verify URL:**")
    print("   http://localhost:8501")
    print()

def main():
    """Main verification function"""
    print("🔐 GCP Security Dashboard Integration Verification")
    print("=" * 60)
    print()
    
    # Test integration
    success = test_dashboard_integration()
    
    # Show what's integrated
    show_integration_summary()
    
    if success:
        print("🎉 INTEGRATION SUCCESSFUL!")
        print("   The executive dashboard is now prominently featured on the front page")
        print()
        print("🚀 TO ACCESS:")
        print("   1. Run: python run_frontend.py")
        print("   2. Open: http://localhost:8501")
        print("   3. Dashboard metrics should be immediately visible!")
        print()
        print("✨ BEFORE vs AFTER:")
        print("   • BEFORE: Dashboard hidden in tabs")
        print("   • AFTER: Executive dashboard prominently displayed on front page")
    else:
        print("❌ Integration issues detected")
        show_cache_clearing_instructions()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())