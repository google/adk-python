#!/usr/bin/env python3
"""
Debug script to examine the structure of FunctionTool objects in the agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_agent_tools():
    """Debug the actual structure of agent tools"""
    print("🔍 Debugging Agent Tools Structure")
    print("=" * 50)

    try:
        from agents.agent import root_agent, tools
        from google.adk.tools import FunctionTool

        print(f"Number of tools: {len(tools)}")
        print(f"Agent type: {type(root_agent)}")
        print()

        for i, tool in enumerate(tools):
            print(f"Tool {i+1}:")
            print(f"  Type: {type(tool)}")
            print(f"  Dir: {[attr for attr in dir(tool) if not attr.startswith('_')]}")

            # Try different ways to access the function
            function_found = False

            if hasattr(tool, 'function'):
                func = tool.function
                print(f"  Function (via .function): {func}")
                if hasattr(func, '__name__'):
                    print(f"  Function name: {func.__name__}")
                    function_found = True

            if hasattr(tool, '_function'):
                func = tool._function
                print(f"  Function (via ._function): {func}")
                if hasattr(func, '__name__'):
                    print(f"  Function name: {func.__name__}")
                    function_found = True

            if hasattr(tool, '_func'):
                func = tool._func
                print(f"  Function (via ._func): {func}")
                if hasattr(func, '__name__'):
                    print(f"  Function name: {func.__name__}")
                    function_found = True

            if hasattr(tool, 'name'):
                print(f"  Tool name: {tool.name}")
                function_found = True

            if hasattr(tool, '_name'):
                print(f"  Tool _name: {tool._name}")
                function_found = True

            if not function_found:
                print("  ❌ Could not find function name")

            print()

            # Only show first 3 tools to avoid spam
            if i >= 2:
                print("... (showing first 3 tools only)")
                break

        # Let's also try creating a fresh FunctionTool to see its structure
        print("\n🔬 Creating test FunctionTool:")
        from agents._tools.feed_tools import query_gcp_release_notes
        test_tool = FunctionTool(query_gcp_release_notes)
        print(f"Test tool type: {type(test_tool)}")
        print(f"Test tool dir: {[attr for attr in dir(test_tool) if not attr.startswith('_')]}")

        if hasattr(test_tool, 'function'):
            print(f"Test tool function: {test_tool.function}")
            if hasattr(test_tool.function, '__name__'):
                print(f"Test tool function name: {test_tool.function.__name__}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_agent_tools()