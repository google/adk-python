"""
MCP Integration Wrapper for Existing Micron Security Agent
Adds MCP protocol and Service Directory integration to your existing FastAPI backend
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import requests

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel

# Google Cloud Service Directory
from google.cloud import servicedirectory_v1
from google.api_core import exceptions as gcp_exceptions

logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'PROJECT_ID': os.getenv('GOOGLE_CLOUD_PROJECT'),
    'LOCATION': os.getenv('CLOUD_RUN_REGION', 'us-central1'),
    'NAMESPACE': 'mcp-security',
    'EXISTING_BACKEND_URL': os.getenv('SECURITY_AGENT_BACKEND_URL', 'http://localhost:8000')
}

class MCPIntegrationWrapper:
    """
    MCP wrapper that integrates with your existing security agent backend
    This is a THIN LAYER that doesn't replace your existing code
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.existing_backend_url = CONFIG['EXISTING_BACKEND_URL']
        self.sd_manager = ServiceDirectoryManager()
        
        # Add MCP routes to your existing FastAPI app
        self.setup_mcp_routes()
    
    def setup_mcp_routes(self):
        """Add MCP-specific routes to existing FastAPI app"""
        
        # MCP Discovery endpoint
        @self.app.get("/.well-known/mcp.json")
        async def mcp_discovery(request: Request):
            return await self.get_mcp_discovery_document(request)
        
        # MCP WebSocket protocol endpoint  
        @self.app.websocket("/api/mcp")
        async def mcp_websocket(websocket: WebSocket):
            await self.handle_mcp_protocol(websocket)
        
        # Enhanced health check with MCP info
        @self.app.get("/mcp/health")
        async def mcp_health():
            return await self.get_mcp_health()
    
    async def get_mcp_discovery_document(self, request: Request) -> Dict[str, Any]:
        """Generate MCP discovery document from existing API"""
        base_url = str(request.url).replace("/.well-known/mcp.json", "")
        
        # Map your existing API endpoints to MCP tools
        mcp_tools = [
            {
                "name": "discover_assets",
                "description": "Discover and inventory GCP resources using your existing asset discovery API",
                "category": "discovery",
                "endpoint": "/api/v1/assets/discover"
            },
            {
                "name": "security_scan",
                "description": "Run comprehensive security scan using your existing security API",
                "category": "security", 
                "endpoint": "/api/v1/security/scan"
            },
            {
                "name": "iam_analyze",
                "description": "Analyze IAM policies using your existing IAM analyzer",
                "category": "iam",
                "endpoint": "/api/v1/iam/analyze"
            },
            {
                "name": "compliance_check",
                "description": "Check compliance against frameworks using your existing compliance API",
                "category": "compliance",
                "endpoint": "/api/v1/compliance/check"
            },
            {
                "name": "monitor_resources",
                "description": "Monitor GCP resources using your existing monitoring API",
                "category": "monitoring",
                "endpoint": "/api/v1/monitoring/resources"
            },
            {
                "name": "analyze_logs",
                "description": "Analyze security logs using your existing log analysis",
                "category": "analysis",
                "endpoint": "/api/v1/logs/analyze"
            },
            {
                "name": "get_recommendations",
                "description": "Get security recommendations using your existing recommendation engine",
                "category": "recommendations",
                "endpoint": "/api/v1/recommendations"
            },
            {
                "name": "manage_sessions",
                "description": "Manage chat sessions with your existing session API",
                "category": "sessions",
                "endpoint": "/api/v1/sessions"
            }
        ]
        
        return {
            "version": "1.0.0",
            "servers": {
                "micron-security-agent": {
                    "name": "Micron Security Agent (MCP-Enabled)",
                    "description": "Your existing security agent now with MCP protocol support",
                    "version": "1.13.0-mcp",
                    "protocol": "https",
                    "endpoint": f"{base_url}/api/mcp",
                    "authentication": {
                        "type": "bearer",
                        "description": "Use existing authentication from your backend"
                    },
                    "capabilities": {
                        "tools": True,
                        "resources": True,
                        "streaming": True,
                        "batch": False
                    },
                    "tools": mcp_tools,
                    "metadata": {
                        "existing_backend": self.existing_backend_url,
                        "integration_type": "mcp_wrapper",
                        "service_directory": {
                            "project": CONFIG['PROJECT_ID'],
                            "location": CONFIG['LOCATION'], 
                            "namespace": CONFIG['NAMESPACE']
                        },
                        "organization": "Micron Technology",
                        "team": "IT Security",
                        "frontend_url": "http://localhost:8501"  # Your Streamlit app
                    },
                    "contact": {
                        "email": "it-security@micron.com",
                        "documentation": "Your existing docs",
                        "backend_health": f"{self.existing_backend_url}/health"
                    }
                }
            }
        }
    
    async def handle_mcp_protocol(self, websocket: WebSocket):
        """Handle MCP WebSocket protocol by proxying to your existing backend"""
        await websocket.accept()
        
        try:
            while True:
                # Receive MCP message
                data = await websocket.receive_text()
                mcp_request = json.loads(data)
                
                # Route MCP request to your existing API
                response = await self.route_mcp_to_existing_api(mcp_request)
                
                # Send response back via WebSocket
                await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            logger.info("MCP WebSocket client disconnected")
        except Exception as e:
            logger.error(f"MCP WebSocket error: {e}")
            await websocket.send_text(json.dumps({
                "error": {"code": -32603, "message": f"Internal error: {e}"}
            }))
    
    async def route_mcp_to_existing_api(self, mcp_request: Dict[str, Any]) -> Dict[str, Any]:
        """Route MCP tool calls to your existing API endpoints"""
        
        if mcp_request.get("method") == "tools/call":
            tool_name = mcp_request["params"]["name"]
            tool_args = mcp_request["params"].get("arguments", {})
            
            # Map MCP tool calls to your existing API endpoints
            endpoint_mapping = {
                "discover_assets": "/api/v1/assets/discover",
                "security_scan": "/api/v1/security/scan", 
                "iam_analyze": "/api/v1/iam/analyze",
                "compliance_check": "/api/v1/compliance/check",
                "monitor_resources": "/api/v1/monitoring/resources",
                "analyze_logs": "/api/v1/logs/analyze",
                "get_recommendations": "/api/v1/recommendations",
                "manage_sessions": "/api/v1/sessions"
            }
            
            if tool_name in endpoint_mapping:
                try:
                    # Call your existing API
                    endpoint = endpoint_mapping[tool_name]
                    url = f"{self.existing_backend_url}{endpoint}"
                    
                    # Make request to your existing backend
                    response = requests.post(url, json=tool_args, timeout=30)
                    response.raise_for_status()
                    
                    return {
                        "id": mcp_request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(response.json(), indent=2)
                                }
                            ]
                        }
                    }
                    
                except Exception as e:
                    return {
                        "id": mcp_request.get("id"),
                        "error": {
                            "code": -32603,
                            "message": f"Error calling {tool_name}: {e}"
                        }
                    }
            else:
                return {
                    "id": mcp_request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        elif mcp_request.get("method") == "tools/list":
            # Return list of available tools
            return {
                "id": mcp_request.get("id"),
                "result": {
                    "tools": [
                        {"name": "discover_assets", "description": "Discover GCP assets"},
                        {"name": "security_scan", "description": "Run security scan"},
                        {"name": "iam_analyze", "description": "Analyze IAM policies"},
                        {"name": "compliance_check", "description": "Check compliance"},
                        {"name": "monitor_resources", "description": "Monitor resources"},
                        {"name": "analyze_logs", "description": "Analyze security logs"},
                        {"name": "get_recommendations", "description": "Get recommendations"},
                        {"name": "manage_sessions", "description": "Manage sessions"}
                    ]
                }
            }
        
        else:
            return {
                "id": mcp_request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {mcp_request.get('method')}"
                }
            }
    
    async def get_mcp_health(self) -> Dict[str, Any]:
        """Health check that includes MCP and existing backend status"""
        health_status = {
            "status": "healthy",
            "mcp_enabled": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integration_type": "mcp_wrapper",
            "existing_backend": {
                "url": self.existing_backend_url,
                "status": "unknown"
            },
            "service_directory": {
                "registered": False,
                "namespace": CONFIG['NAMESPACE']
            }
        }
        
        # Check existing backend health
        try:
            response = requests.get(f"{self.existing_backend_url}/health", timeout=5)
            if response.status_code == 200:
                health_status["existing_backend"]["status"] = "healthy"
                health_status["existing_backend"]["details"] = response.json()
            else:
                health_status["existing_backend"]["status"] = "unhealthy"
        except Exception as e:
            health_status["existing_backend"]["status"] = "error"
            health_status["existing_backend"]["error"] = str(e)
        
        # Check Service Directory registration
        try:
            if self.sd_manager:
                health_status["service_directory"]["registered"] = True
        except Exception:
            pass
        
        return health_status

class ServiceDirectoryManager:
    """Manages Service Directory registration for the MCP-enabled agent"""
    
    def __init__(self):
        if CONFIG['PROJECT_ID']:
            self.registration_client = servicedirectory_v1.RegistrationServiceClient()
            self.service_url = self._get_service_url()
        else:
            self.registration_client = None
            self.service_url = None
    
    def _get_service_url(self) -> str:
        """Get the service URL for registration"""
        if os.getenv('ENVIRONMENT') == 'production':
            service_name = os.getenv('K_SERVICE', 'security-agent')
            return f"https://{service_name}-{os.getenv('CLOUD_RUN_REVISION', 'latest')}.a.run.app"
        else:
            return "http://localhost:8000"  # Your local development URL
    
    async def register_mcp_service(self):
        """Register the MCP-enabled security agent with Service Directory"""
        if not self.registration_client or not CONFIG['PROJECT_ID']:
            logger.warning("Service Directory registration skipped - no project ID or client")
            return
        
        try:
            await self._ensure_namespace_exists()
            await self._register_service()
            await self._register_endpoints()
            logger.info("Successfully registered MCP-enabled security agent with Service Directory")
        except Exception as e:
            logger.error(f"Failed to register with Service Directory: {e}")
    
    async def _ensure_namespace_exists(self):
        """Create namespace if it doesn't exist"""
        namespace_path = f"projects/{CONFIG['PROJECT_ID']}/locations/{CONFIG['LOCATION']}/namespaces/{CONFIG['NAMESPACE']}"
        
        try:
            self.registration_client.create_namespace(
                parent=f"projects/{CONFIG['PROJECT_ID']}/locations/{CONFIG['LOCATION']}",
                namespace_id=CONFIG['NAMESPACE'],
                namespace={
                    "annotations": {
                        "department": "it-security",
                        "purpose": "MCP-enabled security agent",
                        "integration_type": "mcp_wrapper",
                        "original_agent": "micron-security-agent-v1.13.0",
                        "contact": "it-security@micron.com"
                    }
                }
            )
        except gcp_exceptions.AlreadyExists:
            pass
    
    async def _register_service(self):
        """Register the service with comprehensive metadata"""
        namespace_path = f"projects/{CONFIG['PROJECT_ID']}/locations/{CONFIG['LOCATION']}/namespaces/{CONFIG['NAMESPACE']}"
        service_name = "micron-security-agent-mcp"
        
        service_config = {
            "annotations": {
                # MCP metadata
                "mcp.version": "1.0.0",
                "mcp.protocol": "https",
                "mcp.discovery-path": "/.well-known/mcp.json",
                "mcp.websocket-path": "/api/mcp",
                "mcp.integration_type": "wrapper",
                
                # Service metadata  
                "service.name": "Micron Security Agent (MCP-Enabled)",
                "service.version": "1.13.0-mcp",
                "service.description": "Your existing security agent with MCP protocol support",
                "service.owner": "it-security",
                "service.type": "mcp_wrapper",
                
                # Backend integration
                "backend.url": CONFIG['EXISTING_BACKEND_URL'],
                "backend.type": "fastapi",
                "frontend.url": "http://localhost:8501",
                "frontend.type": "streamlit",
                
                # Tool capabilities from your existing agent
                "tools.asset_discovery": "true",
                "tools.security_scanning": "true", 
                "tools.iam_analysis": "true",
                "tools.compliance_checking": "true",
                "tools.log_analysis": "true",
                "tools.monitoring": "true",
                "tools.recommendations": "true",
                "tools.session_management": "true"
            }
        }
        
        try:
            self.registration_client.create_service(
                parent=namespace_path,
                service_id=service_name,
                service=service_config
            )
        except gcp_exceptions.AlreadyExists:
            # Update existing
            service_path = f"{namespace_path}/services/{service_name}"
            self.registration_client.update_service(
                service={"name": service_path, "annotations": service_config["annotations"]}
            )
    
    async def _register_endpoints(self):
        """Register MCP endpoints"""
        service_path = f"projects/{CONFIG['PROJECT_ID']}/locations/{CONFIG['LOCATION']}/namespaces/{CONFIG['NAMESPACE']}/services/micron-security-agent-mcp"
        
        # MCP Discovery endpoint
        discovery_endpoint = {
            "address": self.service_url.replace("https://", "").replace("http://", ""),
            "port": 443 if self.service_url.startswith("https") else 8000,
            "annotations": {
                "protocol": "https" if self.service_url.startswith("https") else "http",
                "path": "/.well-known/mcp.json",
                "purpose": "mcp-discovery",
                "content-type": "application/json"
            }
        }
        
        # MCP WebSocket endpoint
        websocket_endpoint = {
            "address": self.service_url.replace("https://", "").replace("http://", ""),
            "port": 443 if self.service_url.startswith("https") else 8000,
            "annotations": {
                "protocol": "wss" if self.service_url.startswith("https") else "ws",
                "path": "/api/mcp",
                "purpose": "mcp-protocol",
                "connection-type": "websocket"
            }
        }
        
        endpoints = [
            ("mcp-discovery", discovery_endpoint),
            ("mcp-websocket", websocket_endpoint)
        ]
        
        for endpoint_id, endpoint_config in endpoints:
            try:
                self.registration_client.create_endpoint(
                    parent=service_path,
                    endpoint_id=endpoint_id,
                    endpoint=endpoint_config
                )
            except gcp_exceptions.AlreadyExists:
                endpoint_path = f"{service_path}/endpoints/{endpoint_id}"
                self.registration_client.update_endpoint(
                    endpoint={"name": endpoint_path, **endpoint_config}
                )

def add_mcp_to_existing_app(app: FastAPI) -> MCPIntegrationWrapper:
    """
    Add MCP capabilities to your existing FastAPI app
    
    Usage in your existing main.py:
    from mcp_wrapper import add_mcp_to_existing_app
    
    # Your existing FastAPI app
    app = FastAPI(title="Security Agent")
    
    # Add MCP integration 
    mcp_wrapper = add_mcp_to_existing_app(app)
    
    # Register with Service Directory on startup
    @app.on_event("startup")
    async def startup_event():
        await mcp_wrapper.sd_manager.register_mcp_service()
    """
    return MCPIntegrationWrapper(app)

# For standalone testing
if __name__ == "__main__":
    import uvicorn
    
    # Create a minimal FastAPI app for testing
    app = FastAPI(title="MCP Security Agent Wrapper")
    mcp_wrapper = add_mcp_to_existing_app(app)
    
    @app.on_event("startup")
    async def startup():
        await mcp_wrapper.sd_manager.register_mcp_service()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)