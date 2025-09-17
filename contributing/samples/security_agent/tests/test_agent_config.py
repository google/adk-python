#!/usr/bin/env python3
"""
Security Agent Configuration Tester
====================================

Run this script to validate your agent configuration before testing.
It will automatically detect and validate:
- Environment file location
- Backend connectivity
- Service account credentials
- API endpoint availability
- Log file access
"""

import sys
import os
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run configuration tests"""
    print("🔍 Security Agent Configuration Tester")
    print("=" * 60)
    print()
    
    try:
        # Import agent and run tests
        import agents.adk_agent as agent
        
        # The agent module will automatically load .env and log configuration
        # when imported, so we'll see that output first
        
        print("\n" + "=" * 60)
        print("Running Configuration Tests...")
        print("=" * 60 + "\n")
        
        # Run the configuration test
        results = agent.test_configuration()
        print(results)
        
        # Additional quick tests
        print("\n" + "=" * 60)
        print("Quick Functionality Test")
        print("=" * 60)
        
        # Test a simple function
        print("\n🧪 Testing IAM analysis function...")
        try:
            result = agent.analyze_iam()
            if "IAM SECURITY ANALYSIS" in result:
                print("✅ IAM analysis function working")
            else:
                print("⚠️ IAM analysis returned unexpected format")
        except Exception as e:
            print(f"❌ IAM analysis failed: {e}")
        
        print("\n" + "=" * 60)
        print("💡 Next Steps:")
        print("=" * 60)
        print()
        
        if "🎉 Agent is ready for use!" in results:
            print("✅ Your agent is fully configured and ready!")
            print("\nYou can now:")
            print("1. Test via web interface: http://localhost:8503")
            print("2. Monitor logs: ./monitor_logs.sh")
            print("3. Run agent functions directly in Python")
            print("\nTry these commands in the web interface:")
            print('  - "Test my configuration"')
            print('  - "Analyze IAM security"')
            print('  - "Check storage risks"')
        else:
            print("⚠️ Some configuration issues detected.")
            print("\nRecommended actions:")
            print("1. Check the warnings/errors above")
            print("2. Verify your .env file exists and has correct values")
            print("3. Ensure backend is running: python run_backend.py")
            print("4. Check service account credentials if needed")
        
        print("\n" + "=" * 60)
        
    except ImportError as e:
        print(f"❌ Failed to import agent module: {e}")
        print("\nMake sure you're in the correct directory and dependencies are installed:")
        print("  cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()