"""
Unified GCP Client Service
Single point of entry for all Google Cloud Platform operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from google.cloud import resourcemanager_v3
from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import iam
from google.oauth2 import service_account
from google.auth import default
from googleapiclient import discovery
import httpx

from ..models.api_models import (
    ProjectInfo, ServiceHealth, GCPCredentials, APIService, 
    APIEndpoint, APITestRequest, APITestResult
)

logger = logging.getLogger(__name__)

class GCPClientService:
    """Unified Google Cloud Platform client service."""
    
    def __init__(self, project_id: str = None, credentials_file: str = None):
        """Initialize GCP client service.
        
        Args:
            project_id: Default GCP project ID
            credentials_file: Path to service account key file
        """
        self.project_id = project_id
        self.credentials_file = credentials_file
        self.credentials = None
        
        # Client instances
        self._resource_manager = None
        self._compute_client = None
        self._storage_client = None
        self._iam_client = None
        self._discovery_cache = {}
        
        # Health tracking
        self._last_health_check = None
        self._health_status = "initializing"
        
    async def initialize(self) -> bool:
        """Initialize all GCP clients and authenticate."""
        try:
            logger.info("🔧 Initializing GCP client service...")
            
            # Setup credentials
            if not await self._setup_credentials():
                return False
            
            # Initialize core clients
            await self._initialize_clients()
            
            # Test connectivity
            if not await self._test_connectivity():
                return False
            
            self._health_status = "healthy"
            logger.info("✅ GCP client service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCP service: {e}")
            self._health_status = "unhealthy"
            return False
    
    async def _setup_credentials(self) -> bool:
        """Setup Google Cloud credentials."""
        try:
            if self.credentials_file and os.path.exists(self.credentials_file):
                # Use service account file
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                logger.info(f"Using service account from: {self.credentials_file}")
            else:
                # Use default credentials (ADC)
                self.credentials, _ = default()
                logger.info("Using Application Default Credentials")
            
            # Set project ID if not provided
            if not self.project_id and hasattr(self.credentials, 'project_id'):
                self.project_id = self.credentials.project_id
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup credentials: {e}")
            return False
    
    async def _initialize_clients(self):
        """Initialize GCP service clients."""
        try:
            # Resource Manager for project operations
            self._resource_manager = resourcemanager_v3.ProjectsClient(
                credentials=self.credentials
            )
            
            # Compute Engine client
            self._compute_client = compute_v1.InstancesClient(
                credentials=self.credentials
            )
            
            # Storage client
            self._storage_client = storage.Client(
                credentials=self.credentials,
                project=self.project_id
            )
            
            # IAM client
            self._iam_client = iam.Client(credentials=self.credentials)
            
            logger.info("All GCP clients initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise
    
    async def _test_connectivity(self) -> bool:
        """Test connectivity to GCP services."""
        try:
            # Test with a simple project list operation
            projects = await self.list_projects(limit=1)
            logger.info(f"Connectivity test passed - found {len(projects)} projects")
            return True
            
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False
    
    async def check_health(self) -> ServiceHealth:
        """Check service health."""
        start_time = datetime.utcnow()
        
        try:
            # Test basic operations
            await self._test_connectivity()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self._last_health_check = datetime.utcnow()
            
            return ServiceHealth(
                healthy=True,
                status="healthy",
                message="All GCP services operational",
                last_check=self._last_health_check,
                response_time_ms=response_time
            )
            
        except Exception as e:
            return ServiceHealth(
                healthy=False,
                status="unhealthy", 
                message=f"GCP service error: {str(e)}",
                last_check=datetime.utcnow()
            )
    
    async def list_projects(self, limit: int = 100) -> List[ProjectInfo]:
        """List accessible GCP projects."""
        try:
            projects = []
            
            # Use Resource Manager to list projects
            request = resourcemanager_v3.ListProjectsRequest()
            
            page_result = self._resource_manager.list_projects(request=request)
            
            for project in page_result:
                if len(projects) >= limit:
                    break
                    
                project_info = ProjectInfo(
                    project_id=project.project_id,
                    name=project.display_name or project.project_id,
                    project_number=project.name.split('/')[-1],
                    lifecycle_state=project.state.name,
                    parent={"type": project.parent.type, "id": project.parent.id} if project.parent else None,
                    labels=dict(project.labels) if project.labels else None,
                    create_time=project.create_time if project.create_time else None
                )
                projects.append(project_info)
            
            logger.info(f"Listed {len(projects)} GCP projects")
            return projects
            
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []
    
    async def get_project_info(self, project_id: str) -> Optional[ProjectInfo]:
        """Get detailed information about a specific project."""
        try:
            request = resourcemanager_v3.GetProjectRequest(
                name=f"projects/{project_id}"
            )
            
            project = self._resource_manager.get_project(request=request)
            
            return ProjectInfo(
                project_id=project.project_id,
                name=project.display_name or project.project_id,
                project_number=project.name.split('/')[-1],
                lifecycle_state=project.state.name,
                parent={"type": project.parent.type, "id": project.parent.id} if project.parent else None,
                labels=dict(project.labels) if project.labels else None,
                create_time=project.create_time if project.create_time else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get project info for {project_id}: {e}")
            return None
    
    async def discover_apis(self, 
                          service_filter: Optional[str] = None,
                          preferred_only: bool = True,
                          include_deprecated: bool = False) -> List[APIService]:
        """Discover available Google Cloud APIs."""
        try:
            cache_key = f"discover_{service_filter}_{preferred_only}_{include_deprecated}"
            
            if cache_key in self._discovery_cache:
                logger.info("Using cached API discovery results")
                return self._discovery_cache[cache_key]
            
            # Use Google API Discovery Service
            discovery_service = discovery.build(
                'discovery', 'v1', 
                credentials=self.credentials
            )
            
            # Get list of APIs
            apis_result = discovery_service.apis().list(
                preferred=preferred_only
            ).execute()
            
            services = []
            
            for item in apis_result.get('items', []):
                # Apply service filter
                if service_filter and service_filter.lower() not in item['name'].lower():
                    continue
                
                # Skip deprecated if requested
                if not include_deprecated and item.get('preferred') == False:
                    continue
                
                api_service = APIService(
                    name=item['name'],
                    version=item['version'],
                    title=item.get('title', item['name']),
                    description=item.get('description'),
                    preferred=item.get('preferred', False),
                    documentation_link=item.get('documentationLink'),
                    discovery_version=item.get('discoveryVersion'),
                    icons=item.get('icons'),
                    labels=item.get('labels')
                )
                services.append(api_service)
            
            # Cache results
            self._discovery_cache[cache_key] = services
            
            logger.info(f"Discovered {len(services)} API services")
            return services
            
        except Exception as e:
            logger.error(f"API discovery failed: {e}")
            return []
    
    async def explore_service(self, service: str, version: str) -> Optional[Dict[str, Any]]:
        """Explore a specific Google Cloud API service."""
        try:
            # Build discovery document
            discovery_service = discovery.build(
                service, version,
                credentials=self.credentials
            )
            
            # Get service schema
            schema = discovery_service._schema
            
            endpoints = []
            
            # Parse resources and methods
            if 'resources' in schema:
                await self._parse_resources(
                    schema['resources'], 
                    service, 
                    version, 
                    endpoints
                )
            
            result = {
                "service": APIService(
                    name=service,
                    version=version,
                    title=schema.get('title', service),
                    description=schema.get('description'),
                    documentation_link=schema.get('documentationLink')
                ),
                "endpoints": endpoints,
                "total_endpoints": len(endpoints),
                "resources": list(set(ep.resource for ep in endpoints))
            }
            
            logger.info(f"Explored {service} v{version}: {len(endpoints)} endpoints")
            return result
            
        except Exception as e:
            logger.error(f"Service exploration failed for {service}.{version}: {e}")
            return None
    
    async def _parse_resources(self, resources: Dict, service: str, version: str, 
                             endpoints: List[APIEndpoint], resource_path: str = ""):
        """Recursively parse API resources and methods."""
        for resource_name, resource_data in resources.items():
            current_path = f"{resource_path}.{resource_name}" if resource_path else resource_name
            
            # Parse methods
            if 'methods' in resource_data:
                for method_name, method_data in resource_data['methods'].items():
                    endpoint = APIEndpoint(
                        service=service,
                        version=version,
                        resource=current_path,
                        method_name=method_name,
                        http_method=method_data.get('httpMethod', 'GET'),
                        path=method_data.get('path', ''),
                        description=method_data.get('description'),
                        parameters=method_data.get('parameters'),
                        response_schema=method_data.get('response'),
                        scopes=method_data.get('scopes')
                    )
                    endpoints.append(endpoint)
            
            # Parse nested resources
            if 'resources' in resource_data:
                await self._parse_resources(
                    resource_data['resources'],
                    service,
                    version, 
                    endpoints,
                    current_path
                )
    
    async def test_endpoint(self, request: APITestRequest) -> APITestResult:
        """Test a Google Cloud API endpoint."""
        start_time = datetime.utcnow()
        
        try:
            # Build the discovery service
            service_client = discovery.build(
                request.service,
                request.version,
                credentials=self.credentials
            )
            
            # Navigate to the resource
            resource = service_client
            for part in request.resource_path.split('.'):
                resource = getattr(resource, part)()
            
            # Get the method
            method = getattr(resource, request.method_name)
            
            # Build request parameters
            params = {}
            
            # Add path parameters
            if request.path_parameters:
                params.update(request.path_parameters)
            
            # Add query parameters
            if request.query_parameters:
                params.update(request.query_parameters)
            
            # Add body
            if request.body:
                params['body'] = request.body
            
            # Execute the request
            result = method(**params).execute()
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return APITestResult(
                success=True,
                status_code=200,
                response_data=result,
                execution_time_ms=execution_time,
                request_info={
                    "service": request.service,
                    "version": request.version,
                    "method": request.method_name,
                    "parameters": params
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return APITestResult(
                success=False,
                error=str(e),
                error_details={"exception_type": type(e).__name__},
                execution_time_ms=execution_time,
                request_info={
                    "service": request.service,
                    "version": request.version,
                    "method": request.method_name
                },
                timestamp=datetime.utcnow()
            )
    
    async def cleanup(self):
        """Clean up resources and connections."""
        try:
            self._discovery_cache.clear()
            self._resource_manager = None
            self._compute_client = None
            self._storage_client = None
            self._iam_client = None
            
            logger.info("GCP client service cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_credentials_info(self) -> GCPCredentials:
        """Get sanitized credentials information."""
        if not self.credentials:
            return GCPCredentials(type="none")
        
        cred_info = GCPCredentials(
            type=type(self.credentials).__name__,
            project_id=getattr(self.credentials, 'project_id', self.project_id),
            has_private_key=hasattr(self.credentials, '_private_key')
        )
        
        if hasattr(self.credentials, 'service_account_email'):
            cred_info.client_email = self.credentials.service_account_email
        
        if hasattr(self.credentials, 'scopes'):
            cred_info.scopes = list(self.credentials.scopes)
        
        return cred_info