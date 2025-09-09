#!/usr/bin/env python3
"""
Test MCP Integration

This script tests that your MCP integration is working
"""

import requests
import json
import sys
import time

def test_mcp_integration():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing MCP Integration for Your Security Agent")
    print("=" * 60)
    
    # Test health endpoint
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint returned {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        print("   Make sure your backend is running: python run_backend.py")
        return False
    
    # Test MCP discovery
    print("2️⃣ Testing MCP discovery endpoint...")
    try:
        response = requests.get(f"{base_url}/.well-known/mcp.json", timeout=5)
        if response.status_code == 200:
            mcp_data = response.json()
            print("✅ MCP discovery endpoint working")
            
            # Show available tools
            servers = mcp_data.get("servers", {})
            for server_name, server_info in servers.items():
                tools = server_info.get("tools", [])
                print(f"   Server: {server_name}")
                print(f"   Available tools: {len(tools)}")
                for tool in tools[:3]:  # Show first 3
                    print(f"     - {tool['name']}: {tool['description']}")
                if len(tools) > 3:
                    print(f"     ... and {len(tools) - 3} more")
                    
        else:
            print(f"❌ MCP discovery returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MCP discovery failed: {e}")
        return False
    
    # Test MCP health
    print("3️⃣ Testing MCP health endpoint...")
    try:
        response = requests.get(f"{base_url}/mcp/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ MCP health endpoint working")
            print(f"   MCP enabled: {health_data.get('mcp_enabled', False)}")
            print(f"   Backend status: {health_data.get('existing_backend', {}).get('status', 'unknown')}")
        else:
            print(f"❌ MCP health returned {response.status_code}")
    except Exception as e:
        print(f"❌ MCP health failed: {e}")
    
    print("\n🎉 MCP Integration Test Complete!")
    print("\n🚀 Next Steps:")
    print("1. Your existing agent now supports MCP protocol")
    print("2. Connect Claude Code or other MCP tools to: ws://localhost:8000/api/mcp") 
    print("3. Use Service Directory discovery for enterprise deployment")
    
    return True

if __name__ == "__main__":
    if not test_mcp_integration():
        sys.exit(1)
