"""Service for interacting with Google Cloud Platform APIs."""

import requests
from google.auth import default, transport
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

class GCPService:
    """A service for making generic calls to Google Cloud Platform APIs."""
    
    def __init__(self, credentials=None, project_id=None):
        """Initialize GCP service with optional credentials."""
        self.credentials = credentials
        self.project_id = project_id
        
        # If no credentials provided, try to get them
        if not self.credentials:
            self.credentials, self.project_id = self._get_credentials()

    def _get_credentials(self):
        """Get Google Cloud credentials using standard approach."""
        try:
            # Use Google's standard default authentication flow
            credentials, project_id = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            
            # Use project from environment if available, otherwise use detected project
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or project_id
            
            logger.info("✅ GCP Service using Google Cloud default credentials")
            logger.info(f"✅ Project ID: {project_id}")
            return credentials, project_id
                    
        except Exception as e:
            logger.error(f"❌ GCP Service failed to get credentials: {e}")
            logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set for local development")
            return None, None

    def call_google_api(
        self,
        service: str,
        version: str,
        resource_path: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Constructs and executes a REST call to a specified Google Cloud API endpoint.

        Args:
            service: The Google Cloud service name (e.g., 'storage', 'cloudresourcemanager').
            version: The API version (e.g., 'v1', 'v3').
            resource_path: The resource path for the API call (e.g., 'b/my-bucket/o').
            method: The HTTP method to use (GET, POST, PUT, DELETE).
            body: The JSON body for POST or PUT requests.

        Returns:
            The JSON response from the API as a dictionary.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            if not self.credentials:
                raise Exception("No credentials available for GCP API calls")
                
            auth_req = transport.requests.Request()
            self.credentials.refresh(auth_req)

            headers = {
                "Authorization": f"Bearer {self.credentials.token}",
                "Content-Type": "application/json",
            }

            # Handle service names that already contain .googleapis.com
            if service.endswith('.googleapis.com'):
                base_url = f"https://{service}/{version}"
            else:
                base_url = f"https://{service}.googleapis.com/{version}"
            full_url = f"{base_url}/{resource_path}"

            response = requests.request(method, full_url, headers=headers, json=body)
            response.raise_for_status()
            
            # Handle cases with no content in response
            if response.status_code == 204 or not response.content:
                return {}

            return response.json()

        except requests.exceptions.HTTPError as e:
            # Parse error response for better debugging
            error_details = ""
            try:
                if e.response.content:
                    error_json = e.response.json()
                    error_details = f" - {error_json.get('error', {}).get('message', str(error_json))}"
            except:
                error_details = f" - HTTP {e.response.status_code}"
            
            # Provide specific guidance for common Resource Manager API issues
            if service == "cloudresourcemanager" and "parent" in str(e).lower():
                error_details += " (Hint: Resource Manager v3 API requires parent parameter; consider using v1 API instead)"
            
            raise Exception(f"HTTP Error calling Google API '{service}/{version}/{resource_path}'{error_details}: {e}")
        except Exception as e:
            # Re-raise with a more informative message
            raise Exception(f"Error calling Google API '{service}/{version}/{resource_path}': {e}")

