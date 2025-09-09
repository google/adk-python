"""
Secure MCP Client for Micron Security Agent
Handles authentication and secure communication with MCP endpoints
"""

import asyncio
import aiohttp
import jwt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class SecureSecurityAgentClient:
    """Secure MCP client with authentication for the Security Agent"""
    
    def __init__(
        self,
        agent_url: str = "http://localhost:8000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.agent_url = agent_url.rstrip('/')
        self.discovery_url = f"{self.agent_url}/mcp/.well-known/mcp.json"
        self.mcp_endpoint = f"{self.agent_url}/mcp"
        self.auth_endpoint = f"{self.agent_url}/auth/login"
        
        # Authentication
        self.username = username
        self.password = password
        self.api_key = api_key
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Discovered tools cache
        self.tools_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        if self.username and self.password:
            await self.authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def authenticate(self) -> bool:
        """Authenticate with the security agent"""
        if self.api_key:
            # API key authentication
            self.access_token = self.api_key
            self.token_expires = datetime.now() + timedelta(days=30)
            logger.info("Authenticated with API key")
            return True
        
        elif self.username and self.password:
            # Username/password authentication
            try:
                auth_payload = {
                    "username": self.username,
                    "password": self.password
                }
                
                async with self.session.post(
                    self.auth_endpoint,
                    json=auth_payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        auth_data = await response.json()
                        self.access_token = auth_data["access_token"]
                        
                        # Calculate token expiration
                        expires_in = auth_data.get("expires_in", 3600)
                        self.token_expires = datetime.now() + timedelta(seconds=expires_in)
                        
                        logger.info(f"Authenticated as {self.username}")
                        return True
                    else:
                        logger.error(f"Authentication failed: HTTP {response.status}")
                        return False
                        
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                return False
        
        else:
            logger.warning("No authentication credentials provided")
            return False
    
    async def ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if not self.access_token:
            return await self.authenticate()
        
        # Check if token is expired
        if self.token_expires and datetime.now() >= self.token_expires:
            logger.info("Token expired, re-authenticating...")
            return await self.authenticate()
        
        return True
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        headers = {"Content-Type": "application/json"}
        
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        return headers
    
    async def discover_tools(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Discover available security tools with caching"""
        
        # Use cache if available and not forcing refresh
        if not force_refresh and self.tools_cache and self.cache_timestamp:
            cache_age = (datetime.now() - self.cache_timestamp).seconds
            if cache_age < 300:  # 5 minutes cache
                return {
                    "status": "success",
                    "source": "cache",
                    "data": self.tools_cache,
                    "cache_age_seconds": cache_age
                }
        
        try:
            # Fetch fresh discovery data
            async with self.session.get(
                self.discovery_url,
                timeout=10
            ) as response:
                if response.status == 200:
                    discovery_data = await response.json()
                    
                    # Update cache
                    self.tools_cache = discovery_data
                    self.cache_timestamp = datetime.now()
                    
                    # Parse tools
                    tools = self._parse_tools(discovery_data)
                    
                    return {
                        "status": "success",
                        "source": "fresh",
                        "tools_count": len(tools),
                        "tools": tools,
                        "raw_data": discovery_data
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Discovery failed: HTTP {response.status}"
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "error": f"Discovery error: {str(e)}"
            }
    
    def _parse_tools(self, discovery_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tools from discovery data"""
        tools = []
        
        for server_name, server_info in discovery_data.get("servers", {}).items():
            for tool in server_info.get("tools", []):
                tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "server": server_name,
                    "requires_auth": self._check_auth_requirement(tool["name"]),
                    "input_schema": tool.get("inputSchema", {})
                })
        
        return tools
    
    def _check_auth_requirement(self, tool_name: str) -> bool:
        """Check if a tool requires authentication"""
        # Tools that require authentication
        auth_required_tools = [
            "security_scan",
            "vulnerability_check",
            "iam_analyze",
            "permission_audit",
            "compliance_check",
            "incident_response",
            "threat_assessment",
            "cloud_security_posture",
            "kubernetes_security",
            "container_scan"
        ]
        
        return tool_name in auth_required_tools
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute a security tool with authentication"""
        
        # Check if tool requires authentication
        if self._check_auth_requirement(tool_name):
            if not await self.ensure_authenticated():
                return {
                    "status": "error",
                    "error": "Authentication required but failed",
                    "tool": tool_name
                }
        
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
        
        try:
            headers = self._get_auth_headers()
            
            async with self.session.post(
                self.mcp_endpoint,
                json=mcp_request,
                headers=headers,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    return {
                        "status": "success",
                        "tool": tool_name,
                        "executed_at": datetime.now().isoformat(),
                        "result": result
                    }
                    
                elif response.status == 401:
                    # Try to re-authenticate once
                    if await self.authenticate():
                        # Retry with new token
                        headers = self._get_auth_headers()
                        async with self.session.post(
                            self.mcp_endpoint,
                            json=mcp_request,
                            headers=headers,
                            timeout=timeout
                        ) as retry_response:
                            if retry_response.status == 200:
                                result = await retry_response.json()
                                return {
                                    "status": "success",
                                    "tool": tool_name,
                                    "executed_at": datetime.now().isoformat(),
                                    "result": result,
                                    "note": "Re-authenticated successfully"
                                }
                    
                    return {
                        "status": "error",
                        "error": "Authentication failed",
                        "tool": tool_name
                    }
                    
                elif response.status == 403:
                    return {
                        "status": "error",
                        "error": "Insufficient permissions for this tool",
                        "tool": tool_name
                    }
                    
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "error": f"Tool execution failed: HTTP {response.status}",
                        "details": error_text,
                        "tool": tool_name
                    }
                    
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error": f"Tool execution timeout ({timeout}s)",
                "tool": tool_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Tool execution error: {str(e)}",
                "tool": tool_name
            }
    
    async def batch_execute(
        self,
        executions: List[Dict[str, Any]],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute multiple tools in batch"""
        
        if parallel:
            # Execute in parallel
            tasks = [
                self.execute_tool(exec_def["tool"], exec_def["parameters"])
                for exec_def in executions
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "status": "error",
                        "error": str(result),
                        "tool": executions[i]["tool"]
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        else:
            # Execute sequentially
            results = []
            for exec_def in executions:
                result = await self.execute_tool(
                    exec_def["tool"],
                    exec_def["parameters"]
                )
                results.append(result)
            
            return results
    
    async def run_security_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Run predefined security workflows"""
        
        workflows = {
            "comprehensive_scan": [
                {"tool": "security_scan", "parameters": {"scan_type": "comprehensive"}},
                {"tool": "vulnerability_check", "parameters": {"include_cves": True}},
                {"tool": "compliance_check", "parameters": {"frameworks": ["SOC2", "ISO27001"]}},
                {"tool": "iam_analyze", "parameters": {"check_permissions": True}}
            ],
            "incident_response": [
                {"tool": "threat_assessment", "parameters": {"priority": "high"}},
                {"tool": "incident_response", "parameters": {"auto_contain": True}},
                {"tool": "security_scan", "parameters": {"scan_type": "targeted"}}
            ],
            "cloud_security": [
                {"tool": "cloud_security_posture", "parameters": {"providers": ["aws", "azure", "gcp"]}},
                {"tool": "kubernetes_security", "parameters": {"namespaces": "all"}},
                {"tool": "container_scan", "parameters": {"registries": ["docker.io", "gcr.io"]}}
            ]
        }
        
        if workflow_name not in workflows:
            return {
                "status": "error",
                "error": f"Unknown workflow: {workflow_name}",
                "available_workflows": list(workflows.keys())
            }
        
        workflow_steps = workflows[workflow_name]
        
        logger.info(f"Running security workflow: {workflow_name}")
        results = await self.batch_execute(workflow_steps, parallel=True)
        
        # Analyze results
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        return {
            "status": "completed",
            "workflow": workflow_name,
            "total_steps": len(workflow_steps),
            "successful": successful,
            "failed": failed,
            "results": results,
            "executed_at": datetime.now().isoformat()
        }


async def demo_secure_client():
    """Demonstrate the secure MCP client"""
    
    print("🔐 Secure MCP Client Demo for Micron Security Agent")
    print("="*60)
    
    # Create secure client with authentication
    async with SecureSecurityAgentClient(
        agent_url="http://localhost:8000",
        username="admin",
        password="secure_password"  # In production, use environment variables
    ) as client:
        
        # Discover tools
        print("\n🔍 Discovering security tools...")
        discovery = await client.discover_tools()
        
        if discovery["status"] == "success":
            print(f"✅ Discovered {discovery['tools_count']} tools")
            
            # Show tools requiring authentication
            auth_tools = [
                t for t in discovery["tools"] 
                if t["requires_auth"]
            ]
            print(f"\n🔐 Tools requiring authentication: {len(auth_tools)}")
            for tool in auth_tools[:5]:
                print(f"   • {tool['name']}: {tool['description'][:50]}...")
            
            # Execute a security scan
            print("\n🛡️ Executing security scan...")
            scan_result = await client.execute_tool(
                "security_scan",
                {"target": "production", "scan_type": "quick"}
            )
            
            if scan_result["status"] == "success":
                print("✅ Security scan completed!")
                print(f"   Result: {json.dumps(scan_result['result'], indent=2)[:200]}...")
            else:
                print(f"❌ Scan failed: {scan_result['error']}")
            
            # Run a workflow
            print("\n🔄 Running comprehensive security workflow...")
            workflow_result = await client.run_security_workflow("comprehensive_scan")
            
            print(f"✅ Workflow completed:")
            print(f"   Successful steps: {workflow_result['successful']}/{workflow_result['total_steps']}")
            
            # Batch execution example
            print("\n📦 Batch executing multiple tools...")
            batch_results = await client.batch_execute([
                {"tool": "health", "parameters": {}},
                {"tool": "iam_analyze", "parameters": {"user": "john.doe"}},
                {"tool": "compliance_check", "parameters": {"framework": "SOC2"}}
            ], parallel=True)
            
            print(f"✅ Batch execution completed: {len(batch_results)} results")
            
        else:
            print(f"❌ Discovery failed: {discovery['error']}")
    
    print("\n" + "="*60)
    print("✨ Secure MCP client demo completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(demo_secure_client())