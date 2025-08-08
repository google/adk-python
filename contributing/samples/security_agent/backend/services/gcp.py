"""
Consolidated GCP Service
Combines: gcp/, gcp_api_explorer/ services
Provides: Generic GCP API calls, API discovery, and endpoint testing
"""

import asyncio
import logging
import requests
import httpx
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Set up logger first
logger = logging.getLogger(__name__)

# Google Cloud imports with graceful fallbacks
from google.auth import default
from google.auth.transport import requests as auth_requests

# Optional Google Cloud imports
try:
    from google.cloud import resourcemanager_v3
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    logger.warning("google.cloud.resourcemanager_v3 not available - using mock implementation")

try:
    from google.cloud import service_usage_v1
    SERVICE_USAGE_AVAILABLE = True
except ImportError:
    SERVICE_USAGE_AVAILABLE = False
    logger.warning("google.cloud.service_usage_v1 not available - using mock implementation")

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLEAPI_CLIENT_AVAILABLE = True
except ImportError:
    GOOGLEAPI_CLIENT_AVAILABLE = False
    logger.warning("googleapiclient not available - discovery features disabled")

# OpenTelemetry imports  
from opentelemetry import trace

from core.base_service import BaseService

logger = logging.getLogger(__name__)


class ConsolidatedGCPService(BaseService):
    """
    Unified GCP service providing:
    - Generic Google Cloud API calls with authentication
    - API discovery and exploration capabilities
    - Project and service management
    - API endpoint testing and validation
    """
    
    def __init__(self, service_name: str = 'consolidated_gcp', credentials=None, project_id=None):
        """Initialize the consolidated GCP service."""
        super().__init__(service_name, credentials, project_id)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Service configuration flags
        self.api_discovery_enabled = True
        self.project_management_enabled = True
        self.endpoint_testing_enabled = True
        
        # Initialize clients
        self.resource_manager_client = None
        self.service_usage_client = None
        self._discovery_service = None
        
        # Caches for performance
        self._cached_services = {}
        self._cached_endpoints = {}
        self._cache_timestamp = {}
        self.cache_ttl = 300  # 5 minutes
        
        # HTTP client for API calls
        self.http_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients."""
        try:
            # Get credentials if not provided
            if not self.credentials:
                self.credentials, self.project_id = self._get_credentials()
            
            # Initialize Resource Manager client
            if self.project_management_enabled:
                self.resource_manager_client = resourcemanager_v3.ProjectsClient(
                    credentials=self.credentials
                )
                
                self.service_usage_client = service_usage_v1.ServiceUsageClient(
                    credentials=self.credentials
                )
                
                logger.info(f"✅ GCP Resource Manager clients initialized")
            
            # Initialize discovery service
            if self.api_discovery_enabled:
                self._discovery_service = build(
                    'discovery', 'v1',
                    credentials=self.credentials,
                    cache_discovery=False
                )
                logger.info(f"✅ GCP API Discovery service initialized")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            logger.info(f"✅ Consolidated GCP Service initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCP clients: {e}")
    
    def _get_credentials(self):
        """Get Google Cloud credentials using standard approach."""
        try:
            # Use Google's standard default authentication flow
            credentials, project_id = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            
            # Use project from environment if available, otherwise use detected project
            import os
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or project_id
            
            logger.info("✅ GCP Service using Google Cloud default credentials")
            logger.info(f"✅ Project ID: {project_id}")
            return credentials, project_id
                    
        except Exception as e:
            logger.error(f"❌ GCP Service failed to get credentials: {e}")
            logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set for local development")
            return None, None

    # ==========================================
    # GENERIC GCP API METHODS
    # ==========================================
    
    async def call_google_api(
        self,
        service: str,
        version: str,
        resource_path: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a generic call to any Google Cloud API.
        
        Args:
            service: The GCP service name (e.g., 'compute', 'storage')
            version: API version (e.g., 'v1', 'v2')
            resource_path: The resource path (e.g., 'projects/my-project/zones')
            method: HTTP method (GET, POST, PUT, DELETE)
            body: Request body for POST/PUT requests
            query_params: Query parameters
            
        Returns:
            Dict containing the API response or error information
        """
        if not self.credentials:
            return {
                "success": False,
                "error": "No credentials available",
                "data": None
            }
        
        try:
            with self.tracer.start_as_current_span("google_api_call") as span:
                span.set_attributes({
                    "service": service,
                    "version": version,
                    "method": method,
                    "resource_path": resource_path
                })
                
                # Refresh credentials if needed
                self.credentials.refresh(auth_requests.Request())
                
                # Build the API URL
                base_url = f"https://{service}.googleapis.com/{version}"
                url = f"{base_url}/{resource_path.lstrip('/')}"
                
                # Prepare headers
                headers = {
                    "Authorization": f"Bearer {self.credentials.token}",
                    "Content-Type": "application/json"
                }
                
                # Make the request
                if method.upper() == "GET":
                    async with self.http_client as client:
                        response = await client.get(url, headers=headers, params=query_params)
                elif method.upper() == "POST":
                    async with self.http_client as client:
                        response = await client.post(url, headers=headers, json=body, params=query_params)
                elif method.upper() == "PUT":
                    async with self.http_client as client:
                        response = await client.put(url, headers=headers, json=body, params=query_params)
                elif method.upper() == "DELETE":
                    async with self.http_client as client:
                        response = await client.delete(url, headers=headers, params=query_params)
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported HTTP method: {method}",
                        "data": None
                    }
                
                # Parse response
                response_data = None
                if response.content:
                    try:
                        response_data = response.json()
                    except:
                        response_data = response.text
                
                if response.status_code >= 400:
                    return {
                        "success": False,
                        "error": f"API call failed with status {response.status_code}",
                        "data": response_data,
                        "status_code": response.status_code
                    }
                
                return {
                    "success": True,
                    "data": response_data,
                    "status_code": response.status_code
                }
        
        except Exception as e:
            logger.error(f"Google API call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

    # ==========================================
    # PROJECT MANAGEMENT METHODS
    # ==========================================
    
    async def get_projects(self, page_size: int = 50) -> Dict[str, Any]:
        """Get list of accessible GCP projects."""
        if not self.resource_manager_client:
            return {
                "success": False,
                "error": "Resource Manager client not initialized",
                "projects": []
            }
        
        try:
            with self.tracer.start_as_current_span("get_projects") as span:
                projects = []
                
                request = resourcemanager_v3.ListProjectsRequest(
                    page_size=page_size
                )
                
                page_result = self.resource_manager_client.list_projects(request=request)
                
                for project in page_result:
                    project_info = {
                        "project_id": project.project_id,
                        "name": project.display_name,
                        "state": project.state.name if project.state else "UNKNOWN",
                        "create_time": project.create_time.isoformat() if project.create_time else None,
                        "labels": dict(project.labels) if project.labels else {}
                    }
                    projects.append(project_info)
                
                return {
                    "success": True,
                    "projects": projects,
                    "count": len(projects)
                }
        
        except Exception as e:
            logger.error(f"Failed to get projects: {e}")
            return {
                "success": False,
                "error": str(e),
                "projects": []
            }
    
    async def get_project_services(self, project_id: str) -> Dict[str, Any]:
        """Get enabled services for a project."""
        if not self.service_usage_client:
            return {
                "success": False,
                "error": "Service Usage client not initialized",
                "services": []
            }
        
        try:
            with self.tracer.start_as_current_span("get_project_services") as span:
                span.set_attribute("project_id", project_id)
                
                services = []
                
                request = service_usage_v1.ListServicesRequest(
                    parent=f"projects/{project_id}",
                    filter="state:ENABLED"
                )
                
                page_result = self.service_usage_client.list_services(request=request)
                
                for service in page_result:
                    service_info = {
                        "name": service.name,
                        "display_name": service.config.display_name if service.config else None,
                        "documentation_url": service.config.documentation if service.config else None,
                        "state": service.state.name if service.state else "UNKNOWN"
                    }
                    services.append(service_info)
                
                return {
                    "success": True,
                    "project_id": project_id,
                    "services": services,
                    "count": len(services)
                }
        
        except Exception as e:
            logger.error(f"Failed to get project services: {e}")
            return {
                "success": False,
                "error": str(e),
                "services": []
            }

    # ==========================================
    # API DISCOVERY METHODS
    # ==========================================
    
    async def discover_apis(
        self, 
        service_name: str = None,
        preferred_only: bool = True,
        include_deprecated: bool = False
    ) -> Dict[str, Any]:
        """Discover available Google Cloud APIs."""
        if not self._discovery_service:
            return {
                "success": False,
                "error": "API Discovery service not initialized",
                "services": []
            }
        
        try:
            with self.tracer.start_as_current_span("discover_apis") as span:
                span.set_attributes({
                    "service_name": service_name or "all",
                    "preferred_only": preferred_only
                })
                
                # Check cache first
                cache_key = f"apis_{service_name}_{preferred_only}_{include_deprecated}"
                if self._is_cache_valid(cache_key):
                    return self._cached_services[cache_key]
                
                # Discover APIs
                apis = self._discovery_service.apis().list().execute()
                
                services = []
                for api in apis.get('items', []):
                    # Apply filters
                    if service_name and api.get('name') != service_name:
                        continue
                    
                    if preferred_only and not api.get('preferred', False):
                        continue
                    
                    if not include_deprecated and api.get('deprecated', False):
                        continue
                    
                    service_info = {
                        "name": api.get('name'),
                        "title": api.get('title'),
                        "version": api.get('version'),
                        "description": api.get('description'),
                        "documentation_link": api.get('documentationLink'),
                        "discovery_doc_url": api.get('discoveryRestUrl'),
                        "preferred": api.get('preferred', False),
                        "deprecated": api.get('deprecated', False)
                    }
                    services.append(service_info)
                
                result = {
                    "success": True,
                    "services": services,
                    "total_count": len(services),
                    "filtered_count": len(services)
                }
                
                # Cache result
                self._cached_services[cache_key] = result
                self._cache_timestamp[cache_key] = time.time()
                
                return result
        
        except Exception as e:
            logger.error(f"API discovery failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "services": []
            }
    
    async def explore_service(self, service_name: str, version: str) -> Dict[str, Any]:
        """Explore a specific Google Cloud API service."""
        try:
            with self.tracer.start_as_current_span("explore_service") as span:
                span.set_attributes({
                    "service_name": service_name,
                    "version": version
                })
                
                cache_key = f"service_{service_name}_{version}"
                if self._is_cache_valid(cache_key):
                    return self._cached_endpoints[cache_key]
                
                # Build the service client
                service_client = build(service_name, version, credentials=self.credentials, cache_discovery=False)
                
                # Get the discovery document URL
                discovery_url = service_client._discoveryUrl
                
                # Fetch the discovery document
                async with self.http_client as client:
                    response = await client.get(discovery_url)
                    response.raise_for_status()
                    discovery_doc = response.json()
                
                # Parse the discovery document
                service_info = {
                    "name": discovery_doc.get('name'),
                    "title": discovery_doc.get('title'),
                    "version": discovery_doc.get('version'),
                    "description": discovery_doc.get('description'),
                    "documentation_link": discovery_doc.get('documentationLink'),
                    "base_url": discovery_doc.get('baseUrl')
                }
                
                # Extract endpoints from resources
                endpoints = []
                resources = discovery_doc.get('resources', {})
                
                for resource_name, resource_data in resources.items():
                    methods = resource_data.get('methods', {})
                    
                    for method_name, method_data in methods.items():
                        endpoint_info = {
                            "id": f"{service_name}:{version}:{resource_name}:{method_name}",
                            "service": service_name,
                            "version": version,
                            "resource": resource_name,
                            "method_name": method_name,
                            "http_method": method_data.get('httpMethod', 'GET'),
                            "path": method_data.get('path', ''),
                            "description": method_data.get('description', ''),
                            "parameters": method_data.get('parameters', {}),
                            "scopes": method_data.get('scopes', [])
                        }
                        endpoints.append(endpoint_info)
                
                result = {
                    "success": True,
                    "service": service_info,
                    "endpoints": endpoints,
                    "endpoint_count": len(endpoints)
                }
                
                # Cache result
                self._cached_endpoints[cache_key] = result
                self._cache_timestamp[cache_key] = time.time()
                
                return result
        
        except Exception as e:
            logger.error(f"Service exploration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": {},
                "endpoints": []
            }
    
    async def search_endpoints(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search for API endpoints matching query."""
        try:
            matching_endpoints = []
            
            # Search through cached endpoints
            for cache_key, cached_data in self._cached_endpoints.items():
                if not self._is_cache_valid(cache_key):
                    continue
                
                if cached_data.get("success", False):
                    endpoints = cached_data.get("endpoints", [])
                    
                    for endpoint in endpoints:
                        # Simple text search in endpoint data
                        searchable_text = f"{endpoint.get('method_name', '')} {endpoint.get('description', '')} {endpoint.get('path', '')}".lower()
                        
                        if query.lower() in searchable_text:
                            matching_endpoints.append(endpoint)
                            
                            if len(matching_endpoints) >= max_results:
                                break
                
                if len(matching_endpoints) >= max_results:
                    break
            
            return matching_endpoints[:max_results]
        
        except Exception as e:
            logger.error(f"Endpoint search failed: {e}")
            return []

    # ==========================================
    # ENDPOINT TESTING METHODS
    # ==========================================
    
    async def test_endpoint(
        self,
        service: str,
        version: str,
        method_name: str,
        resource_path: str,
        http_method: str = "GET",
        path_parameters: Dict[str, Any] = None,
        query_parameters: Dict[str, Any] = None,
        body: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Test a specific API endpoint."""
        try:
            with self.tracer.start_as_current_span("test_endpoint") as span:
                span.set_attributes({
                    "service": service,
                    "version": version,
                    "method_name": method_name,
                    "http_method": http_method
                })
                
                start_time = time.time()
                
                # Use the generic API call method
                result = await self.call_google_api(
                    service=service,
                    version=version,
                    resource_path=resource_path,
                    method=http_method,
                    body=body,
                    query_params=query_parameters
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Format response for endpoint testing
                test_response = {
                    "success": result["success"],
                    "request_info": {
                        "service": service,
                        "version": version,
                        "method_name": method_name,
                        "http_method": http_method,
                        "resource_path": resource_path,
                        "url": f"https://{service}.googleapis.com/{version}/{resource_path.lstrip('/')}"
                    },
                    "response_data": result.get("data"),
                    "status_code": result.get("status_code"),
                    "execution_time_ms": execution_time_ms,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if not result["success"]:
                    test_response["error"] = result["error"]
                    test_response["error_details"] = result.get("data")
                
                return test_response
        
        except Exception as e:
            logger.error(f"Endpoint test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_info": {
                    "service": service,
                    "version": version,
                    "method_name": method_name
                }
            }

    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamp:
            return False
        
        age = time.time() - self._cache_timestamp[cache_key]
        return age < self.cache_ttl
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cached_services.clear()
        self._cached_endpoints.clear()
        self._cache_timestamp.clear()
        logger.info("API discovery cache cleared")
    
    async def get_service_quotas(self, project_id: str, service_name: str) -> Dict[str, Any]:
        """Get quota information for a service."""
        try:
            # Use the Service Usage API to get quota details
            result = await self.call_google_api(
                service="serviceusage",
                version="v1",
                resource_path=f"projects/{project_id}/services/{service_name}/consumerQuotaMetrics",
                method="GET"
            )
            
            return {
                "success": result["success"],
                "project_id": project_id,
                "service_name": service_name,
                "quotas": result.get("data", {}),
                "error": result.get("error")
            }
        
        except Exception as e:
            logger.error(f"Failed to get service quotas: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id,
                "service_name": service_name
            }

    # ==========================================
    # BASE SERVICE ABSTRACT METHODS
    # ==========================================
    
    async def initialize(self) -> bool:
        """Initialize the GCP service."""
        self._initialize_clients()
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the GCP service."""
        if self.http_client:
            await self.http_client.aclose()
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check implementation for BaseService."""
        return await self.check_health()

    # ==========================================
    # HEALTH CHECK
    # ==========================================
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        health_status = {
            "service": "consolidated_gcp",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "credentials": bool(self.credentials),
                "project_id": bool(self.project_id),
                "resource_manager": bool(self.resource_manager_client) if self.project_management_enabled else "disabled",
                "service_usage": bool(self.service_usage_client) if self.project_management_enabled else "disabled",
                "api_discovery": bool(self._discovery_service) if self.api_discovery_enabled else "disabled",
                "http_client": bool(self.http_client)
            },
            "cache_stats": {
                "cached_services": len(self._cached_services),
                "cached_endpoints": len(self._cached_endpoints),
                "cache_ttl_seconds": self.cache_ttl
            }
        }
        
        # Overall health check
        critical_components = [
            health_status["components"]["credentials"],
            health_status["components"]["project_id"],
            health_status["components"]["http_client"]
        ]
        
        if all(critical_components):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
        
        return health_status
    
    async def close(self):
        """Close HTTP client and cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
            logger.info("GCP service HTTP client closed")