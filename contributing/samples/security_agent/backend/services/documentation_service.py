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

class DocumentationService:
    def __init__(self):
        self.doc_api_endpoint = os.getenv("DOCUMENTATION_API_ENDPOINT", "https://docs.googleapis.com/v1")

    async def _make_documentation_api_request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Helper to make authenticated requests to a documentation API."""
        with tracer.start_as_current_span(f"DocumentationService_{method.lower()}_request") as span:
            url = f"{self.doc_api_endpoint}/{path}"
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
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                span.set_status(Status(StatusCode.OK))
                return response.json()

            except requests.exceptions.Timeout as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Request timed out: {e}"))
                raise
            except requests.exceptions.RequestException as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Documentation API request failed: {e}"))
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"An unexpected error occurred: {e}"))
                raise

    async def fetch_api_documentation(self, api_name: str) -> str:
        """Fetches documentation for a given API."""
        with tracer.start_as_current_span("fetch_api_documentation") as span:
            span.set_attribute("api_name", api_name)
            try:
                # Placeholder for actual documentation API call
                # For example: self._make_documentation_api_request("GET", f"apis/{api_name}/docs")

                # Simulating documentation content
                documentation_content = f"## Documentation for {api_name}\n\nThis is a placeholder for the actual API documentation."
                span.set_attribute("doc_length", len(documentation_content))
                span.set_status(Status(StatusCode.OK))
                return documentation_content
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to fetch documentation for {api_name}: {e}"))
                raise

    async def search_documentation(self, query: str, api_name: Optional[str] = None) -> List[Dict]:
        """Searches documentation for a given query, optionally within a specific API."""
        with tracer.start_as_current_span("search_documentation") as span:
            span.set_attribute("search_query", query)
            if api_name: span.set_attribute("api_name", api_name)
            try:
                # Placeholder for actual documentation search API call
                # For example: self._make_documentation_api_request("GET", f"search?query={query}&api={api_name or ''}")

                # Simulating search results
                results = [
                    {"title": f"Result 1 for {query}", "url": "http://example.com/doc1"},
                    {"title": f"Result 2 for {query}", "url": "http://example.com/doc2"}
                ]
                span.set_attribute("search_results_count", len(results))
                span.set_status(Status(StatusCode.OK))
                return results
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to search documentation: {e}"))
                raise