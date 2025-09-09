#!/usr/bin/env python3
"""
Quick MCP Testing Script for Micron Security Agent
Test MCP discovery and tool execution
"""

import asyncio
import sys
import os

# Add MCP directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discovery.basic import SecurityAgentMCPDiscovery, demo_security_agent_discovery
from clients.secure_client import SecureSecurityAgentClient, demo_secure_client


async def main():
    """Main test function"""
    print("🚀 Micron Security Agent - MCP Testing")
    print("="*60)
    
    # Test basic discovery
    print("\n1️⃣ Testing Basic MCP Discovery...")
    print("-"*40)
    await demo_security_agent_discovery()
    
    # Prompt to continue
    input("\n✅ Basic discovery complete. Press Enter to test secure client...")
    
    # Test secure client
    print("\n2️⃣ Testing Secure MCP Client...")
    print("-"*40)
    await demo_secure_client()
    
    print("\n" + "="*60)
    print("🎉 MCP Testing Complete!")
    print("\n📚 Next Steps:")
    print("1. Review the discovered tools above")
    print("2. Connect Claude Code: claude-code connect http://localhost:8000/mcp")
    print("3. Use natural language to interact with security tools")
    print("4. Check ./mcp/README.md for complete documentation")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure the security agent is running: python run_backend.py")
        print("2. Check that port 8000 is available")
        print("3. Verify requirements are installed: pip install -r requirements.txt")