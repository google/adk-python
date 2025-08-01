import os
import requests
import asyncio
from typing import List, Dict, Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

import google.auth
import google.auth.transport.requests

# Get tracer
tracer = trace.get_tracer(__name__)

class APIHubService:
    def __init__(self, secret_manager_service):
        self.secret_manager_service = secret_manager_service
        self.apihub_endpoint = os.getenv("APIHUB_ENDPOINT", "https://apihub.googleapis.com/v1")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")

    async def _make_apihub_request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Helper to make authenticated requests to the API Hub API."""
        with tracer.start_as_current_span(f"APIHubService_{method.lower()}_request") as span:
            url = f"{self.apihub_endpoint}/{path}"
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", url)

            try:
                credentials, project = google.auth.default()
                auth_req = google.auth.transport.requests.Request()
                credentials.refresh(auth_req)

                headers = {
                    'Authorization': f'Bearer {credentials.token}',
                    'Content-Type': 'application/json'
                }

                if method == "GET":
                    response = requests.get(url, headers=headers, timeout=10)
                elif method == "POST":
                    response = requests.post(url, json=json_data, headers=headers, timeout=10)
                elif method == "PUT":
                    response = requests.put(url, json=json_data, headers=headers, timeout=10)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                span.set_status(Status(StatusCode.OK))
                return response.json()

            except requests.exceptions.Timeout as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Request timed out: {e}"))
                raise
            except requests.exceptions.RequestException as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"API Hub request failed: {e}"))
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"An unexpected error occurred: {e}"))
                raise

    async def fetch_api_deployments(self) -> List[Dict]:
        """Fetches a list of API deployments from API Hub."""
        with tracer.start_as_current_span("fetch_api_deployments") as span:
            try:
                # This is a placeholder for actual API Hub API call
                # In a real scenario, this would call a GCP API Hub endpoint
                # For example: self._make_apihub_request("GET", f"projects/{self.project_id}/locations/global/deployments")
                
                # Simulating a successful response
                deployments = [
                    {"name": "projects/test-project/locations/global/deployments/my-api-v1", "displayName": "My API v1"},
                    {"name": "projects/test-project/locations/global/deployments/another-api-v2", "displayName": "Another API v2"}
                ]
                span.set_attribute("apihub.deployment_count", len(deployments))
                span.set_status(Status(StatusCode.OK))
                return deployments
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to fetch API deployments: {e}"))
                raise

    async def register_api_deployment(self, api_deployment_data: Dict) -> Dict:
        """Registers a new API deployment in API Hub."""
        with tracer.start_as_current_span("register_api_deployment") as span:
            span.set_attribute("apihub.api_name", api_deployment_data.get("displayName"))
            try:
                # This is a placeholder for actual API Hub API call
                # For example: self._make_apihub_request("POST", f"projects/{self.project_id}/locations/global/deployments", api_deployment_data)

                # Simulating a successful registration response
                registered_deployment = {
                    "name": f"projects/test-project/locations/global/deployments/{api_deployment_data.get('id', 'new-api')}",
                    **api_deployment_data
                }
                span.set_attribute("apihub.registration_success", True)
                span.set_status(Status(StatusCode.OK))
                return registered_deployment
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to register API deployment: {e}"))
                raise

    async def update_api_documentation(self, deployment_id: str, documentation_content: str) -> Dict:
        """Updates documentation for an existing API deployment in API Hub."""
        with tracer.start_as_current_span("update_api_documentation") as span:
            span.set_attribute("apihub.deployment_id", deployment_id)
            try:
                # This is a placeholder for actual API Hub API call
                # For example: self._make_apihub_request("PUT", f"projects/{self.project_id}/locations/global/deployments/{deployment_id}/documentation", {"content": documentation_content})

                # Simulating a successful update response
                updated_documentation_status = {
                    "deploymentId": deployment_id,
                    "status": "Documentation updated successfully",
                    "content_length": len(documentation_content)
                }
                span.set_attribute("apihub.documentation_update_success", True)
                span.set_status(Status(StatusCode.OK))
                return updated_documentation_status
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to update API documentation: {e}"))
                raise