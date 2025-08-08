"""GCP API Explorer modular service."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
import httpx
import time

from core.base_service import BaseService
from .models import (
    DiscoveryRequest, DiscoveryResponse, ExploreRequest, ExploreResponse,
    TestRequest, TestResponse, APIService, APIEndpoint, APIMethod
)
from .discovery_client import DiscoveryClient
from .api_client import APIClient

logger = logging.getLogger(__name__)


class GCPAPIExplorerService(BaseService):
    """GCP API Explorer service for discovering and testing Google Cloud APIs."""
    
    def __init__(self, name: str, config: Dict[str, Any], credentials=None, project_id: str = None):
        """Initialize the GCP API Explorer service.
        
        Args:
            name: Service name
            config: Service configuration
            credentials: Google Cloud credentials
            project_id: GCP project ID
        """
        super().__init__(name, config)
        self.credentials = credentials
        self.project_id = project_id
        self.discovery_client: Optional[DiscoveryClient] = None
        self.api_client: Optional[APIClient] = None
        
    async def start(self) -> bool:
        """Start the GCP API Explorer service."""
        try:
            logger.info(f"Starting {self.name} service")
            
            # Initialize clients
            self.discovery_client = DiscoveryClient(
                credentials=self.credentials,
                project_id=self.project_id
            )
            await self.discovery_client.initialize()
            
            self.api_client = APIClient(
                credentials=self.credentials,
                project_id=self.project_id
            )
            await self.api_client.initialize()
            
            self.status = "running"
            logger.info(f"✅ {self.name} service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {self.name} service: {e}")
            self.status = "error"
            self.last_error = str(e)
            return False
    
    async def stop(self) -> bool:
        """Stop the GCP API Explorer service."""
        try:
            logger.info(f"Stopping {self.name} service")
            
            # Clean up clients
            if self.discovery_client:
                self.discovery_client.clear_cache()
                self.discovery_client = None
            
            self.api_client = None
            self.status = "stopped"
            
            logger.info(f"✅ {self.name} service stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop {self.name} service: {e}")
            return False
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        if self.status != "running":
            return {
                "healthy": False,
                "status": self.status,
                "error": self.last_error
            }
        
        try:
            # Test discovery client
            if not self.discovery_client:
                return {
                    "healthy": False,
                    "error": "Discovery client not initialized"
                }
            
            # Test API client
            if not self.api_client:
                return {
                    "healthy": False,
                    "error": "API client not initialized"
                }
            
            return {
                "healthy": True,
                "status": "running",
                "discovery_client": "ready",
                "api_client": "ready",
                "project_id": self.project_id
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def get_router(self) -> APIRouter:
        """Get FastAPI router for the service."""
        router = APIRouter()
        
        @router.post("/discover", response_model=DiscoveryResponse)
        async def discover_apis(request: DiscoveryRequest) -> DiscoveryResponse:
            """Discover available Google Cloud API services."""
            try:
                if not self.discovery_client:
                    raise HTTPException(status_code=503, detail="Discovery client not available")
                
                logger.info(f"Discovering APIs with filters: {request}")
                
                response = await self.discovery_client.discover_services(
                    service_name=request.service_name,
                    preferred_only=request.preferred_only,
                    include_deprecated=request.include_deprecated
                )
                
                logger.info(f"Discovered {response.total_count} API services")
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"API discovery failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to discover APIs: {str(e)}")
        
        @router.get("/services", response_model=DiscoveryResponse)
        async def list_services(
            service_name: Optional[str] = None,
            preferred_only: bool = True,
            include_deprecated: bool = False
        ) -> DiscoveryResponse:
            """List available Google Cloud API services."""
            try:
                request = DiscoveryRequest(
                    service_name=service_name,
                    preferred_only=preferred_only,
                    include_deprecated=include_deprecated
                )
                
                response = await self.discovery_client.discover_services(
                    service_name=request.service_name,
                    preferred_only=request.preferred_only,
                    include_deprecated=request.include_deprecated
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Service listing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list services: {str(e)}")
        
        @router.post("/explore", response_model=ExploreResponse)
        async def explore_service(request: ExploreRequest) -> ExploreResponse:
            """Explore a specific Google Cloud API service."""
            try:
                if not self.discovery_client:
                    raise HTTPException(status_code=503, detail="Discovery client not available")
                
                logger.info(f"Exploring service: {request.service} v{request.version}")
                
                response = await self.discovery_client.explore_service(
                    service=request.service,
                    version=request.version,
                    resource_filter=request.resource_filter
                )
                
                if response.success:
                    logger.info(f"Explored service with {response.total_endpoints} endpoints")
                else:
                    logger.error(f"Service exploration failed: {response.error}")
                
                return response
                
            except Exception as e:
                logger.error(f"Service exploration failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to explore service: {str(e)}")
        
        @router.get("/explore/{service}/{version}", response_model=ExploreResponse)
        async def explore_service_get(
            service: str,
            version: str,
            resource_filter: Optional[str] = None
        ) -> ExploreResponse:
            """Explore a specific Google Cloud API service (GET version)."""
            try:
                request = ExploreRequest(
                    service=service,
                    version=version,
                    resource_filter=resource_filter
                )
                
                response = await self.discovery_client.explore_service(
                    service=request.service,
                    version=request.version,
                    resource_filter=request.resource_filter
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Service exploration failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to explore service: {str(e)}")
        
        @router.get("/search")
        async def search_endpoints(
            query: str,
            services: Optional[List[str]] = None,
            max_results: int = 50
        ) -> List[APIEndpoint]:
            """Search for API endpoints across services."""
            try:
                if not self.discovery_client:
                    raise HTTPException(status_code=503, detail="Discovery client not available")
                
                logger.info(f"Searching endpoints with query: '{query}'")
                
                results = await self.discovery_client.search_endpoints(
                    query=query,
                    services=services,
                    max_results=max_results
                )
                
                logger.info(f"Found {len(results)} matching endpoints")
                return results
                
            except Exception as e:
                logger.error(f"Endpoint search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @router.post("/test", response_model=TestResponse)
        async def test_endpoint(request: TestRequest) -> TestResponse:
            """Test a Google Cloud API endpoint."""
            try:
                if not self.api_client:
                    raise HTTPException(status_code=503, detail="API client not available")
                
                logger.info(f"Testing endpoint: {request.service}.{request.version}.{request.method_name}")
                
                response = await self.api_client.test_endpoint(request)
                
                if response.success:
                    logger.info(f"Test successful: {response.status_code} in {response.execution_time_ms:.2f}ms")
                else:
                    logger.warning(f"Test failed: {response.error}")
                
                return response
                
            except Exception as e:
                logger.error(f"Endpoint test failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to test endpoint: {str(e)}")
        
        @router.delete("/cache")
        async def clear_discovery_cache() -> dict:
            """Clear the discovery cache."""
            try:
                if self.discovery_client:
                    self.discovery_client.clear_cache()
                    logger.info("Discovery cache cleared")
                
                return {
                    "success": True,
                    "message": "Discovery cache cleared successfully"
                }
                
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
        
        return router
    
    def get_dependencies(self) -> List[str]:
        """Get service dependencies."""
        # GCP API Explorer depends on GCP service for credentials
        return ["gcp"]
    
    def get_tags(self) -> List[str]:
        """Get service tags for API documentation."""
        return ["GCP API Explorer", "API Discovery", "API Testing"]