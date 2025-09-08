#!/usr/bin/env python3
"""
MCP Server Discovery Tool
=========================

This script helps discover which MCP servers are available in your Claude instance.
Run this to get a complete list of available MCP tools and resources.
"""

import json
from typing import Dict, List, Any

# Common MCP server patterns to test
MCP_SERVERS_TO_TEST = {
    "exa": [
        "search",
        "find_similar", 
        "get_contents"
    ],
    "reference": [
        "search",
        "get_doc",
        "list_docs"
    ],
    "filesystem": [
        "read_file",
        "write_file",
        "list_directory"
    ],
    "git": [
        "status",
        "diff",
        "commit",
        "log"
    ],
    "fetch": [
        "get",
        "post",
        "download"
    ],
    "database": [
        "query",
        "insert",
        "update"
    ],
    "slack": [
        "send_message",
        "list_channels",
        "search"
    ],
    "github": [
        "create_issue",
        "create_pr",
        "list_repos"
    ],
    "anthropic": [
        "get_context",
        "set_context"
    ],
    "memory": [
        "store",
        "retrieve",
        "search"
    ],
    "browser": [
        "navigate",
        "click",
        "type"
    ],
    "puppeteer": [
        "launch",
        "goto",
        "screenshot"
    ],
    "search": [
        "web",
        "semantic",
        "similarity"
    ],
    "docs": [
        "search",
        "get",
        "list"
    ]
}

def generate_mcp_test_commands() -> List[str]:
    """Generate commands to test MCP server availability"""
    commands = []
    
    for server, tools in MCP_SERVERS_TO_TEST.items():
        for tool in tools:
            # Format: mcp__{server}__{tool}
            command = f"mcp__{server}__{tool}"
            commands.append(command)
    
    return commands

def create_mcp_discovery_report() -> Dict[str, Any]:
    """Create a report structure for MCP discovery"""
    return {
        "discovered_servers": [],
        "tested_commands": generate_mcp_test_commands(),
        "instructions": [
            "To discover available MCP servers in Claude:",
            "1. Try calling each command with minimal parameters",
            "2. Note which ones succeed vs fail",
            "3. Document the available servers for your project"
        ],
        "test_snippet": """
# Test in Claude by trying these commands:

# Test Exa (web search)
mcp__exa__search({"query": "test"})

# Test Reference (documentation)  
mcp__reference__list_docs()

# Test Filesystem
mcp__filesystem__list_directory({"path": "."})

# Test Git
mcp__git__status()

# Note: Tools that exist will execute or return proper errors
# Tools that don't exist will return "tool not found" errors
"""
    }

def main():
    """Main discovery function"""
    print("🔍 MCP Server Discovery Tool")
    print("=" * 50)
    
    report = create_mcp_discovery_report()
    
    print("\n📋 MCP Servers to Test:")
    print("-" * 30)
    for server in MCP_SERVERS_TO_TEST.keys():
        print(f"  • {server}")
    
    print("\n🧪 Generated Test Commands:")
    print("-" * 30)
    for cmd in report["tested_commands"][:10]:  # Show first 10
        print(f"  {cmd}()")
    print(f"  ... and {len(report['tested_commands']) - 10} more")
    
    print("\n📝 Instructions:")
    print("-" * 30)
    for instruction in report["instructions"]:
        print(f"  {instruction}")
    
    print("\n💾 Saving discovery report...")
    with open("mcp_discovery_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Report saved to mcp_discovery_report.json")
    
    print("\n🎯 Next Steps:")
    print("1. Copy the test commands into Claude")
    print("2. Try each command to see which servers are available")
    print("3. Update MCP_CONFIGURATION.md with discovered servers")
    
    # Create a test file for Claude
    with open("test_mcp_availability.md", "w") as f:
        f.write("# MCP Server Availability Test\n\n")
        f.write("Copy and run these commands in Claude to test availability:\n\n")
        f.write("```javascript\n")
        
        for server, tools in MCP_SERVERS_TO_TEST.items():
            f.write(f"// Test {server} server\n")
            for tool in tools[:2]:  # First 2 tools per server
                f.write(f"mcp__{server}__{tool}({{test: true}})\n")
            f.write("\n")
        
        f.write("```\n\n")
        f.write("## Expected Results\n\n")
        f.write("- ✅ Available: Tool executes or returns parameter errors\n")
        f.write("- ❌ Not Available: Returns 'tool not found' or similar\n")
    
    print("📄 Test file created: test_mcp_availability.md")

if __name__ == "__main__":
    main()