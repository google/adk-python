"""Service for interacting with Google Cloud Platform APIs."""

import requests
from google.auth import default, transport
from google.oauth2 import service_account
from typing import Dict, Any, Optional
import os
import json
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
        """Get service account credentials."""
        try:
            # Check for service account key file (prefer clearer variable name)
            service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY_FILE') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            
            if service_account_path and os.path.exists(service_account_path):
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logger.info(f"✅ GCP Service using service account from: {service_account_path}")
                return credentials, project_id
            else:
                # Try service account JSON from environment variable
                service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
                if service_account_json:
                    service_account_info = json.loads(service_account_json)
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    project_id = service_account_info.get('project_id') or project_id
                    logger.info("✅ GCP Service using service account from environment JSON")
                    return credentials, project_id
                else:
                    # Fall back to default credentials
                    logger.warning("⚠️ GCP Service falling back to default credentials")
                    credentials, project_id = default()
                    return credentials, project_id
                    
        except Exception as e:
            logger.error(f"❌ GCP Service failed to get credentials: {e}")
            # Try default as last resort
            try:
                credentials, project_id = default()
                return credentials, project_id
            except Exception as fallback_error:
                logger.error(f"❌ GCP Service failed to get any credentials: {fallback_error}")
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

