"""Service for interacting with Google Cloud Platform APIs."""

import requests
from google.auth import default, transport
from typing import Dict, Any, Optional

class GCPService:
    """A service for making generic calls to Google Cloud Platform APIs."""

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
            credentials, project_id = default()
            auth_req = transport.requests.Request()
            credentials.refresh(auth_req)

            headers = {
                "Authorization": f"Bearer {credentials.token}",
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

        except Exception as e:
            # Re-raise with a more informative message
            raise Exception(f"Error calling Google API '{service}/{version}/{resource_path}': {e}")

