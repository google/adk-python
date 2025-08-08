"""Google Cloud API Client for making generic API calls within the security agent."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
import httpx
from google.auth import default
from google.auth.transport.requests import Request

from .models import TestRequest, TestResponse, APIMethod

logger = logging.getLogger(__name__)


class APIClient:
    """Generic client for Google Cloud API calls."""
    
    def __init__(self, credentials=None, project_id: Optional[str] = None):
        """Initialize the API client.
        
        Args:
            credentials: Google Cloud credentials
            project_id: GCP project ID
        """
        self.credentials = credentials or default()[0]
        self.project_id = project_id or default()[1]
        self._base_url = "https://www.googleapis.com"
        self._timeout = 30.0
        
    async def initialize(self):
        """Initialize the API client."""
        try:
            # Ensure credentials are refreshed
            if hasattr(self.credentials, 'refresh'):
                self.credentials.refresh(Request())
            logger.info("API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            raise

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if hasattr(self.credentials, 'token'):
            return {"Authorization": f"Bearer {self.credentials.token}"}
        return {}

    async def test_endpoint(self, request: TestRequest) -> TestResponse:
        """Test an API endpoint with the provided parameters."""
        start_time = time.time()
        
        try:
            # Build the full URL
            url = self._build_url(
                request.service,
                request.version,
                request.resource_path,
                request.path_parameters
            )
            
            # Prepare headers
            headers = self._get_auth_headers()
            headers.update(request.headers)
            headers.setdefault("Content-Type", "application/json")
            headers.setdefault("Accept", "application/json")
            
            # Prepare request data
            request_kwargs = {
                "method": request.http_method.value,
                "url": url,
                "headers": headers,
                "timeout": self._timeout,
                "params": request.query_parameters
            }
            
            if request.body and request.http_method in [APIMethod.POST, APIMethod.PUT, APIMethod.PATCH]:
                request_kwargs["json"] = request.body
            
            # Make the request
            async with httpx.AsyncClient() as client:
                response = await client.request(**request_kwargs)
                
                execution_time = (time.time() - start_time) * 1000
                
                # Parse response
                response_data = None
                try:
                    if response.content:
                        response_data = response.json()
                except Exception:
                    response_data = {"raw_content": response.text}
                
                return TestResponse(
                    success=response.status_code < 400,
                    request_info={
                        "url": url,
                        "method": request.http_method.value,
                        "headers": dict(headers),
                        "params": request.query_parameters,
                        "body": request.body
                    },
                    response_data=response_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    execution_time_ms=execution_time,
                    error=None if response.status_code < 400 else f"HTTP {response.status_code}",
                    error_details=None if response.status_code < 400 else {
                        "status_code": response.status_code,
                        "response_text": response.text
                    }
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Endpoint test failed: {e}")
            
            return TestResponse(
                success=False,
                request_info={
                    "service": request.service,
                    "version": request.version,
                    "method": request.method_name,
                    "resource_path": request.resource_path
                },
                error=f"Test failed: {str(e)}",
                execution_time_ms=execution_time,
                error_details={"exception": str(e)}
            )

    def _build_url(
        self, 
        service: str, 
        version: str, 
        path: str,
        path_parameters: Dict[str, Any] = None
    ) -> str:
        """Build the full API URL."""
        # Handle path parameters
        final_path = path
        if path_parameters:
            for key, value in path_parameters.items():
                placeholder = f"{{{key}}}"
                if placeholder in final_path:
                    final_path = final_path.replace(placeholder, str(value))
        
        # Build URL
        if final_path.startswith('/'):
            final_path = final_path[1:]
        
        # Handle special cases for different service URL patterns
        if service in ['storage']:
            base_url = f"https://storage.googleapis.com/storage/{version}"
        elif service in ['bigquery']:
            base_url = f"https://bigquery.googleapis.com/bigquery/{version}"
        else:
            base_url = f"{self._base_url}/{service}/{version}"
        
        return f"{base_url}/{final_path}"