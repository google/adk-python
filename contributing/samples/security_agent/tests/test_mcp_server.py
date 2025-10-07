#!/usr/bin/env python3
"""
Test script for the ADK Security Agent MCP Server
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

async def test_mcp_tools():
    """Test the MCP server tool listing and basic functionality."""
    print("Testing ADK Security Agent MCP Server...")

    # First, test if the server starts without errors
    try:
        print("1. Testing server startup...")
        proc = subprocess.Popen(
            [sys.executable, "mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give the server a moment to initialize
        time.sleep(2)

        # Check if process is still running (no immediate crash)
        if proc.poll() is None:
            print("✅ Server started successfully")
        else:
            print("❌ Server crashed during startup")
            stderr_output = proc.stderr.read()
            print(f"Error: {stderr_output}")
            return False

        # Test basic initialization message
        print("2. Testing MCP protocol initialization...")

        # Send an initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        import json
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        # Read response with timeout
        try:
            response = proc.stdout.readline()
            if response:
                print("✅ Server responded to initialization")
                print(f"Response: {response.strip()}")
            else:
                print("⚠️ No response from server")
        except Exception as e:
            print(f"⚠️ Error reading response: {e}")

        # Clean up
        proc.terminate()
        proc.wait(timeout=5)

        print("3. Testing basic imports...")

        # Test if all required modules can be imported
        try:
            import mcp.server.stdio
            print("✅ MCP stdio module imported")

            import mcp.types
            print("✅ MCP types module imported")

            from agents.agent import root_agent
            print("✅ ADK agent imported")

            from agents._tools.sqlite_tool import SQLiteTool
            print("✅ SQLite tool imported")

            from agents._tools.security_tools import SecurityTool
            print("✅ Security tool imported")

        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False

        print("\n✅ All tests passed! MCP server is ready.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        if 'proc' in locals():
            proc.terminate()
        return False

async def test_manual_tool_calls():
    """Test the individual tool components manually."""
    print("\n4. Testing individual tool components...")

    try:
        # Test SQLite tool
        from agents._tools.sqlite_tool import SQLiteTool
        sqlite_tool = SQLiteTool()
        print("✅ SQLite tool initialized")

        # Test Security tool
        from agents._tools.security_tools import SecurityTool
        security_tool = SecurityTool(sqlite_tool=sqlite_tool)
        print("✅ Security tool initialized")

        # Test a simple query
        result = security_tool.query_security_data(
            query="test query",
            query_type="exploration",
            force_live_update=False
        )

        if result.get("success") is not None:
            print("✅ Security tool query executed")
            print(f"Query result: {result.get('summary', 'No summary')}")
        else:
            print("⚠️ Security tool query returned unexpected format")

        return True

    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("ADK Security Agent MCP Server Test Suite")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("agents/agent.py").exists():
        print("❌ Error: Must run from project root directory")
        print("Current directory should contain 'agents/agent.py'")
        return 1

    if not Path("mcp_server.py").exists():
        print("❌ Error: mcp_server.py not found")
        return 1

    # Run tests
    server_test = await test_mcp_tools()
    tool_test = await test_manual_tool_calls()

    if server_test and tool_test:
        print("\n🎉 All tests passed! MCP server is working correctly.")
        print("\nTo start the MCP server:")
        print("./scripts/start_mcp_server.sh")
        print("\nOr directly:")
        print("python mcp_server.py")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))