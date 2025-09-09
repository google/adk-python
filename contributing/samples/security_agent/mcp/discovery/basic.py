"""
Basic MCP Discovery for Micron Security Agent
Discovers and connects to the security agent's MCP endpoints
"""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
import json
from datetime import datetime


class SecurityAgentMCPDiscovery:
    """MCP Discovery client for the Micron Security Agent"""
    
    def __init__(self, agent_url: str = "http://localhost:8000"):
        self.agent_url = agent_url.rstrip('/')
        self.discovery_url = f"{self.agent_url}/mcp/.well-known/mcp.json"
        self.mcp_endpoint = f"{self.agent_url}/mcp"
        self.discovered_tools: List[Dict] = []
        self.discovery_timestamp: Optional[datetime] = None
    
    async def discover_security_tools(self) -> Dict[str, Any]:
        """Discover all security tools available via MCP"""
        async with aiohttp.ClientSession() as session:
            try:
                # Attempt discovery
                async with session.get(self.discovery_url, timeout=10) as response:
                    if response.status == 200:
                        discovery_data = await response.json()
                        self.discovery_timestamp = datetime.now()
                        
                        # Parse available tools
                        self._parse_tools(discovery_data)
                        
                        return {
                            "status": "success",
                            "agent_url": self.agent_url,
                            "timestamp": self.discovery_timestamp.isoformat(),
                            "tools_discovered": len(self.discovered_tools),
                            "categories": self._categorize_tools(),
                            "discovery_data": discovery_data
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"Discovery failed: HTTP {response.status}",
                            "agent_url": self.agent_url
                        }
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "error": "Discovery timeout - is the security agent running?",
                    "agent_url": self.agent_url,
                    "hint": "Start the agent with: python run_backend.py"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Discovery error: {str(e)}",
                    "agent_url": self.agent_url
                }
    
    def _parse_tools(self, discovery_data: Dict[str, Any]):
        """Parse tools from discovery data"""
        self.discovered_tools = []
        
        for server_name, server_info in discovery_data.get("servers", {}).items():
            for tool in server_info.get("tools", []):
                # Categorize security tools
                category = self._determine_tool_category(tool["name"])
                
                self.discovered_tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "category": category,
                    "server": server_name,
                    "input_schema": tool.get("inputSchema", {})
                })
    
    def _determine_tool_category(self, tool_name: str) -> str:
        """Categorize security tools by function"""
        categories = {
            "security": ["security_scan", "vulnerability", "threat", "incident"],
            "iam": ["iam", "permission", "access", "identity", "role"],
            "compliance": ["compliance", "audit", "policy", "regulation"],
            "cloud": ["cloud", "aws", "azure", "gcp", "kubernetes", "container"],
            "network": ["network", "firewall", "vpn", "traffic"],
            "data": ["data", "encryption", "privacy", "dlp"],
            "monitoring": ["monitor", "alert", "log", "metric", "trace"]
        }
        
        tool_lower = tool_name.lower()
        for category, keywords in categories.items():
            if any(keyword in tool_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _categorize_tools(self) -> Dict[str, List[str]]:
        """Group discovered tools by category"""
        categorized = {}
        
        for tool in self.discovered_tools:
            category = tool["category"]
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(tool["name"])
        
        return categorized
    
    async def get_tool_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool"""
        for tool in self.discovered_tools:
            if tool["name"] == tool_name:
                return tool
        return None
    
    async def execute_security_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a security tool via MCP"""
        
        # Build MCP request
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.mcp_endpoint, 
                    json=mcp_request,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "status": "success",
                            "tool": tool_name,
                            "result": result
                        }
                    else:
                        return {
                            "status": "error",
                            "tool": tool_name,
                            "error": f"Execution failed: HTTP {response.status}",
                            "response": await response.text()
                        }
            except Exception as e:
                return {
                    "status": "error",
                    "tool": tool_name,
                    "error": f"Execution error: {str(e)}"
                }
    
    def print_discovery_summary(self):
        """Print a formatted summary of discovered tools"""
        if not self.discovered_tools:
            print("❌ No tools discovered. Run discover_security_tools() first.")
            return
        
        print("\n" + "="*60)
        print("🔍 MICRON SECURITY AGENT - MCP DISCOVERY SUMMARY")
        print("="*60)
        print(f"📍 Agent URL: {self.agent_url}")
        print(f"⏰ Discovery Time: {self.discovery_timestamp}")
        print(f"🛠️  Total Tools: {len(self.discovered_tools)}")
        print("\n📊 Tools by Category:")
        
        categories = self._categorize_tools()
        for category, tools in sorted(categories.items()):
            emoji = {
                "security": "🔐",
                "iam": "👤",
                "compliance": "📋",
                "cloud": "☁️",
                "network": "🌐",
                "data": "💾",
                "monitoring": "📊",
                "general": "🔧"
            }.get(category, "📦")
            
            print(f"\n{emoji} {category.upper()} ({len(tools)} tools)")
            for tool in sorted(tools):
                tool_details = next((t for t in self.discovered_tools if t["name"] == tool), {})
                desc = tool_details.get("description", "")[:50]
                if desc and len(desc) == 50:
                    desc += "..."
                print(f"   • {tool}: {desc}")
        
        print("\n" + "="*60)
        print("✅ Security Agent is MCP-enabled and ready for AI integration!")
        print("="*60)


async def demo_security_agent_discovery():
    """Demonstrate MCP discovery for the security agent"""
    
    print("🚀 Starting Micron Security Agent MCP Discovery...")
    print("-" * 60)
    
    # Initialize discovery client
    discovery = SecurityAgentMCPDiscovery("http://localhost:8000")
    
    # Discover available tools
    print("🔍 Discovering security tools via MCP...")
    result = await discovery.discover_security_tools()
    
    if result["status"] == "success":
        print(f"✅ Discovery successful!")
        print(f"   Found {result['tools_discovered']} security tools")
        
        # Print detailed summary
        discovery.print_discovery_summary()
        
        # Example: Execute a security scan
        print("\n" + "="*60)
        print("🧪 EXAMPLE: Executing Security Scan")
        print("="*60)
        
        scan_result = await discovery.execute_security_tool(
            tool_name="security_scan",
            parameters={
                "target": "production",
                "scan_type": "comprehensive",
                "include_vulnerabilities": True
            }
        )
        
        if scan_result["status"] == "success":
            print("✅ Security scan completed successfully!")
            print(f"   Result: {json.dumps(scan_result['result'], indent=2)}")
        else:
            print(f"❌ Scan failed: {scan_result['error']}")
        
        # Show integration instructions
        print("\n" + "="*60)
        print("🤖 CLAUDE CODE INTEGRATION")
        print("="*60)
        print("To connect Claude Code to this security agent:")
        print(f"1. Ensure the agent is running at {discovery.agent_url}")
        print(f"2. Run: claude-code connect {discovery.mcp_endpoint}")
        print("3. Use natural language like:")
        print('   "Scan our production environment for vulnerabilities"')
        print('   "Check IAM permissions for user john.doe"')
        print('   "Analyze our cloud security posture"')
        
    else:
        print(f"❌ Discovery failed: {result['error']}")
        if "hint" in result:
            print(f"💡 Hint: {result['hint']}")
        print("\nTroubleshooting:")
        print("1. Ensure the security agent is running:")
        print("   cd ADK/contributing/samples/security_agent")
        print("   python run_backend.py")
        print("2. Check the agent is accessible:")
        print(f"   curl {discovery.agent_url}/health")
        print("3. Verify MCP is enabled:")
        print(f"   curl {discovery.discovery_url}")


if __name__ == "__main__":
    asyncio.run(demo_security_agent_discovery())