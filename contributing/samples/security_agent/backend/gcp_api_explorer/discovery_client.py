"""Google Cloud API Discovery Client for the security agent."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx
from google.auth import default
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .models import (
    APIService, APIEndpoint, APIMethod,
    DiscoveryResponse, ExploreResponse
)

logger = logging.getLogger(__name__)


class DiscoveryClient:
    """Client for discovering and exploring Google Cloud APIs."""
    
    def __init__(self, credentials=None, project_id: Optional[str] = None):
        """Initialize the discovery client.
        
        Args:
            credentials: Google Cloud credentials
            project_id: GCP project ID
        """
        self.credentials = credentials or default()[0]
        self.project_id = project_id or default()[1]
        self._discovery_service = None
        self._cached_services = {}
        self._cached_endpoints = {}
        
    async def initialize(self):
        """Initialize the discovery service."""
        try:
            # Ensure credentials are refreshed
            if hasattr(self.credentials, 'refresh'):
                self.credentials.refresh(Request())
                
            # Build the discovery service
            self._discovery_service = build(
                'discovery', 'v1', 
                credentials=self.credentials,
                cache_discovery=False
            )
            logger.info("Discovery client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize discovery client: {e}")
            raise

    async def discover_services(
        self, 
        service_name: Optional[str] = None,
        preferred_only: bool = True,
        include_deprecated: bool = False
    ) -> DiscoveryResponse:
        """Discover available Google Cloud API services."""
        try:
            if not self._discovery_service:
                await self.initialize()
            
            # Get list of APIs
            request = self._discovery_service.apis().list()
            response = request.execute()
            
            services = []
            seen_services = set()
            
            for item in response.get('items', []):
                service = APIService(
                    name=item['name'],
                    title=item['title'],
                    version=item['version'],
                    description=item.get('description'),
                    documentation_link=item.get('documentationLink'),
                    discovery_doc_url=item['discoveryRestUrl'],
                    preferred=item.get('preferred', False)
                )
                
                # Apply filters
                if service_name and service.name != service_name:
                    continue
                    
                if preferred_only and not service.preferred:
                    if service.name in seen_services:
                        continue
                        
                if not include_deprecated and 'deprecated' in service.title.lower():
                    continue
                
                services.append(service)
                seen_services.add(service.name)
            
            # Cache the results
            self._cached_services = {s.name: s for s in services}
            
            return DiscoveryResponse(
                success=True,
                services=services,
                total_count=len(services)
            )
            
        except Exception as e:
            logger.error(f"Failed to discover services: {e}")
            return DiscoveryResponse(
                success=False,
                error=f"Discovery failed: {str(e)}"
            )

    async def explore_service(
        self, 
        service: str, 
        version: str,
        resource_filter: Optional[str] = None
    ) -> ExploreResponse:
        """Explore a specific Google Cloud API service."""
        try:
            cache_key = f"{service}:{version}"
            if cache_key in self._cached_endpoints:
                cached = self._cached_endpoints[cache_key]
                return ExploreResponse(
                    success=True,
                    service=cached['service'],
                    endpoints=cached['endpoints'],
                    total_endpoints=len(cached['endpoints'])
                )
            
            # Build service client to get discovery document
            service_client = build(
                service, version,
                credentials=self.credentials,
                cache_discovery=False
            )
            
            # Get the discovery document URL
            discovery_doc_url = f"https://{service}.googleapis.com/$discovery/rest?version={version}"
            
            # Fetch discovery document
            async with httpx.AsyncClient() as client:
                response = await client.get(discovery_doc_url)
                response.raise_for_status()
                doc = response.json()
            
            # Parse service information
            api_service = APIService(
                name=doc['name'],
                title=doc['title'],
                version=doc['version'],
                description=doc.get('description'),
                documentation_link=doc.get('documentationLink'),
                discovery_doc_url=discovery_doc_url
            )
            
            # Parse resources and methods
            endpoints = []
            
            def parse_resource(resource_name: str, resource_data: Dict[str, Any], parent_path: str = "") -> None:
                """Recursively parse API resources."""
                # Parse methods
                methods = resource_data.get('methods', {})
                for method_name, method_data in methods.items():
                    if resource_filter and resource_filter.lower() not in method_name.lower():
                        continue
                    
                    # Build endpoint path
                    path = method_data.get('path', '')
                    
                    # Determine HTTP method
                    http_method = APIMethod.GET
                    if method_data.get('httpMethod'):
                        try:
                            http_method = APIMethod(method_data['httpMethod'])
                        except ValueError:
                            http_method = APIMethod.GET
                    
                    endpoint = APIEndpoint(
                        id=f"{service}:{version}:{resource_name}:{method_name}",
                        service=service,
                        version=version,
                        resource=resource_name,
                        method_name=method_name,
                        http_method=http_method,
                        path=path,
                        description=method_data.get('description'),
                        parameters=method_data.get('parameters', {}),
                        request_schema=method_data.get('request'),
                        response_schema=method_data.get('response'),
                        scopes=method_data.get('scopes', [])
                    )
                    
                    endpoints.append(endpoint)
                
                # Parse nested resources
                nested_resources = resource_data.get('resources', {})
                for nested_name, nested_data in nested_resources.items():
                    parse_resource(nested_name, nested_data, f"{parent_path}/{resource_name}" if parent_path else resource_name)
            
            # Parse all top-level resources
            doc_resources = doc.get('resources', {})
            for resource_name, resource_data in doc_resources.items():
                parse_resource(resource_name, resource_data)
            
            # Cache results
            self._cached_endpoints[cache_key] = {
                'service': api_service,
                'endpoints': endpoints
            }
            
            return ExploreResponse(
                success=True,
                service=api_service,
                endpoints=endpoints,
                total_endpoints=len(endpoints)
            )
            
        except Exception as e:
            logger.error(f"Failed to explore service {service}:{version}: {e}")
            return ExploreResponse(
                success=False,
                error=f"Service exploration failed: {str(e)}"
            )

    async def get_endpoint_details(
        self, 
        service: str, 
        version: str, 
        method_name: str
    ) -> Optional[APIEndpoint]:
        """Get detailed information about a specific API endpoint."""
        try:
            cache_key = f"{service}:{version}"
            
            if cache_key not in self._cached_endpoints:
                await self.explore_service(service, version)
            
            if cache_key in self._cached_endpoints:
                endpoints = self._cached_endpoints[cache_key]['endpoints']
                for endpoint in endpoints:
                    if endpoint.method_name == method_name:
                        return endpoint
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get endpoint details: {e}")
            return None

    async def search_endpoints(
        self,
        query: str,
        services: Optional[List[str]] = None,
        max_results: int = 50
    ) -> List[APIEndpoint]:
        """Search for API endpoints across services."""
        results = []
        query_lower = query.lower()
        
        # If services not specified, search all cached services
        if not services:
            services = list(self._cached_services.keys())
        
        for service_name in services:
            if service_name in self._cached_services:
                service_obj = self._cached_services[service_name]
                
                # Explore service if not cached
                cache_key = f"{service_name}:{service_obj.version}"
                if cache_key not in self._cached_endpoints:
                    await self.explore_service(service_name, service_obj.version)
                
                # Search endpoints
                if cache_key in self._cached_endpoints:
                    endpoints = self._cached_endpoints[cache_key]['endpoints']
                    for endpoint in endpoints:
                        if (query_lower in endpoint.method_name.lower() or
                            query_lower in (endpoint.description or "").lower() or
                            query_lower in endpoint.path.lower()):
                            results.append(endpoint)
                            
                            if len(results) >= max_results:
                                return results
        
        return results

    def clear_cache(self):
        """Clear all cached discovery data."""
        self._cached_services.clear()
        self._cached_endpoints.clear()
        logger.info("Discovery cache cleared")