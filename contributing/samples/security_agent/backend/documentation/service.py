import os
import requests
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

import google.auth
import google.auth.transport.requests
from core.base_service import BaseService

# Get tracer and logger
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

class DocumentationService(BaseService):
    def __init__(self, service_name: str = 'documentation', credentials=None, project_id=None):
        super().__init__(service_name, credentials, project_id)
        self.credentials = credentials
        self.project_id = project_id
        # Base URLs for Google Cloud documentation
    
    async def initialize(self) -> bool:
        """Initialize the Documentation service."""
        try:
            logger.info("Initializing Documentation service...")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Documentation service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Documentation service."""
        try:
            logger.info("Shutting down Documentation service...")
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown Documentation service: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Documentation service health."""
        try:
            return {
                "healthy": True,
                "status": "running",
                "message": "Documentation service is operational"
            }
        except Exception as e:
            logger.error(f"Documentation health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "message": "Documentation service health check failed"
            }
        self.gcp_docs_base = "https://cloud.google.com/docs"
        self.api_docs_base = "https://cloud.google.com"
        
        # Common API documentation URLs
        self.api_doc_urls = {
            "securitycenter": "https://cloud.google.com/security-center/docs/reference/rest",
            "monitoring": "https://cloud.google.com/monitoring/api/ref_v3/rest",
            "trace": "https://cloud.google.com/trace/docs/reference/v1/rest",
            "logging": "https://cloud.google.com/logging/docs/reference/v2/rest",
            "apihub": "https://cloud.google.com/api-hub/docs/reference/rest",
            "resourcemanager": "https://cloud.google.com/resource-manager/docs/reference/rest",
            "iam": "https://cloud.google.com/iam/docs/reference/rest",
            "serviceusage": "https://cloud.google.com/service-usage/docs/reference/rest"
        }

    async def _make_documentation_api_request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Helper to make authenticated requests to a documentation API."""
        with tracer.start_as_current_span(f"DocumentationService_{method.lower()}_request") as span:
            url = f"{self.doc_api_endpoint}/{path}"
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", url)

            try:
                if not self.credentials:
                    raise Exception("DocumentationService is not initialized with credentials.")
                
                auth_req = google.auth.transport.requests.Request()
                self.credentials.refresh(auth_req)

                headers = {
                    'Authorization': f'Bearer {self.credentials.token}',
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
        """Fetches real documentation for a given API by scraping Google Cloud docs."""
        with tracer.start_as_current_span("fetch_api_documentation") as span:
            span.set_attribute("api_name", api_name)
            try:
                # Get the documentation URL for the API
                doc_url = self.api_doc_urls.get(api_name.lower())
                
                if not doc_url:
                    # Try to construct a generic docs URL
                    doc_url = f"{self.gcp_docs_base}/{api_name}"
                
                logger.info(f"Fetching documentation from: {doc_url}")
                
                # Fetch the documentation page
                response = requests.get(doc_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; GCP-Security-Agent/1.0)'
                })
                response.raise_for_status()
                
                # Parse the HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract relevant documentation content
                documentation_content = self._extract_documentation_content(soup, api_name)
                
                span.set_attribute("doc_length", len(documentation_content))
                span.set_attribute("doc_url", doc_url)
                span.set_status(Status(StatusCode.OK))
                return documentation_content
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch docs from URL, using fallback: {e}")
                # Fallback to mock content with helpful information
                fallback_content = self._get_fallback_documentation(api_name)
                span.set_attribute("fallback_used", True)
                span.set_status(Status(StatusCode.OK))
                return fallback_content
            except Exception as e:
                logger.error(f"Failed to fetch documentation for {api_name}: {e}")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to fetch documentation for {api_name}: {e}"))
                # Return fallback content instead of raising
                return self._get_fallback_documentation(api_name)

    async def search_documentation(self, query: str, api_name: Optional[str] = None) -> List[Dict]:
        """Searches documentation for a given query, optionally within a specific API."""
        with tracer.start_as_current_span("search_documentation") as span:
            span.set_attribute("search_query", query)
            if api_name: span.set_attribute("api_name", api_name)
            try:
                results = []
                
                # If searching within a specific API, search its documentation
                if api_name:
                    api_docs = await self.fetch_api_documentation(api_name)
                    if self._content_matches_query(api_docs, query):
                        results.append({
                            "title": f"{api_name.title()} API Documentation",
                            "url": self.api_doc_urls.get(api_name.lower(), f"{self.gcp_docs_base}/{api_name}"),
                            "snippet": self._extract_snippet(api_docs, query)
                        })
                else:
                    # Search across all known APIs
                    for api, url in self.api_doc_urls.items():
                        try:
                            api_docs = await self.fetch_api_documentation(api)
                            if self._content_matches_query(api_docs, query):
                                results.append({
                                    "title": f"{api.title()} API Documentation",
                                    "url": url,
                                    "snippet": self._extract_snippet(api_docs, query)
                                })
                        except Exception as e:
                            logger.warning(f"Failed to search {api} documentation: {e}")
                            continue
                
                # Add generic search results if we didn't find much
                if len(results) < 3:
                    fallback_results = self._get_fallback_search_results(query, api_name)
                    results.extend(fallback_results[:3 - len(results)])
                
                span.set_attribute("search_results_count", len(results))
                span.set_status(Status(StatusCode.OK))
                return results
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to search documentation: {e}"))
                # Return fallback results instead of raising
                return self._get_fallback_search_results(query, api_name)
    
    def _extract_documentation_content(self, soup: BeautifulSoup, api_name: str) -> str:
        """Extract meaningful content from documentation HTML."""
        try:
            content_parts = []
            
            # Try to find main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                # Extract headings and text
                for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'code']):
                    if element.name.startswith('h'):
                        content_parts.append(f"\n## {element.get_text().strip()}\n")
                    elif element.name in ['pre', 'code']:
                        content_parts.append(f"```\n{element.get_text().strip()}\n```\n")
                    else:
                        text = element.get_text().strip()
                        if text and len(text) > 20:  # Only include substantial text
                            content_parts.append(text + "\n")
            
            # If we got content, format it nicely
            if content_parts:
                content = f"# {api_name.title()} API Documentation\n\n" + "".join(content_parts)
                # Limit content length
                if len(content) > 5000:
                    content = content[:5000] + "\n\n... (truncated for display)"
                return content
            else:
                return self._get_fallback_documentation(api_name)
                
        except Exception as e:
            logger.error(f"Error extracting documentation content: {e}")
            return self._get_fallback_documentation(api_name)
    
    def _get_fallback_documentation(self, api_name: str) -> str:
        """Get fallback documentation content when scraping fails."""
        doc_url = self.api_doc_urls.get(api_name.lower(), f"{self.gcp_docs_base}/{api_name}")
        
        return f"""# {api_name.title()} API Documentation

## Overview
This is the {api_name.title()} API for Google Cloud Platform.

## Official Documentation
For the most up-to-date and complete documentation, please visit:
{doc_url}

## Common Operations
- List resources
- Create resources
- Update resources
- Delete resources
- Get resource details

## Authentication
All requests require proper Google Cloud authentication using:
- Service account credentials
- Application Default Credentials (ADC)
- OAuth 2.0 tokens

## SDK Support
This API is supported by:
- Google Cloud Client Libraries
- REST API
- gRPC API

## Getting Started
1. Enable the {api_name.title()} API in your Google Cloud project
2. Set up authentication
3. Install the appropriate client library
4. Make your first API call

For detailed examples and API reference, visit the official documentation link above.
"""
    
    def _get_fallback_search_results(self, query: str, api_name: Optional[str] = None) -> List[Dict]:
        """Get fallback search results when real search fails."""
        base_results = []
        
        if api_name and api_name.lower() in self.api_doc_urls:
            base_results.append({
                "title": f"{api_name.title()} API Reference",
                "url": self.api_doc_urls[api_name.lower()],
                "snippet": f"Official {api_name.title()} API documentation and reference"
            })
        
        # Add some generic helpful results
        base_results.extend([
            {
                "title": f"Google Cloud Documentation - {query}",
                "url": f"https://cloud.google.com/docs/search?q={query.replace(' ', '+')}",
                "snippet": f"Search Google Cloud documentation for '{query}'"
            },
            {
                "title": "Google Cloud API References",
                "url": "https://cloud.google.com/apis/docs/overview",
                "snippet": "Overview of all Google Cloud APIs and their documentation"
            },
            {
                "title": "Google Cloud Client Libraries",
                "url": "https://cloud.google.com/apis/docs/client-libraries-explained",
                "snippet": "Information about client libraries for Google Cloud APIs"
            }
        ])
        
        return base_results
    
    def _content_matches_query(self, content: str, query: str) -> bool:
        """Check if content matches the search query."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Simple word matching
        query_words = query_lower.split()
        return any(word in content_lower for word in query_words if len(word) > 2)
    
    def _extract_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Extract a snippet from content around the query match."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find the first occurrence of any query word
        best_pos = -1
        for word in query_lower.split():
            if len(word) > 2:
                pos = content_lower.find(word)
                if pos != -1 and (best_pos == -1 or pos < best_pos):
                    best_pos = pos
        
        if best_pos == -1:
            # No match found, return the beginning
            snippet = content[:max_length]
        else:
            # Extract text around the match
            start = max(0, best_pos - 50)
            end = min(len(content), best_pos + max_length - 50)
            snippet = content[start:end]
            
            # Clean up the snippet
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
        
        # Remove excessive whitespace and markdown formatting
        snippet = re.sub(r'\s+', ' ', snippet)
        snippet = re.sub(r'[#`*]', '', snippet)
        
        return snippet.strip()