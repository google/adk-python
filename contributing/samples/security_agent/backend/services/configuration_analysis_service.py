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

class ConfigurationAnalysisService:
    def __init__(self):
        self.config_analysis_endpoint = os.getenv("CONFIG_ANALYSIS_ENDPOINT", "https://configanalysis.googleapis.com/v1")

    async def _make_config_analysis_request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Helper to make authenticated requests to a configuration analysis API."""
        with tracer.start_as_current_span(f"ConfigAnalysisService_{method.lower()}_request") as span:
            url = f"{self.config_analysis_endpoint}/{path}"
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

                if method == "POST":
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
                span.set_status(Status(StatusCode.ERROR, f"Configuration analysis API request failed: {e}"))
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"An unexpected error occurred: {e}"))
                raise

    async def analyze_configuration(self, config_data: Dict) -> Dict:
        """Sends configuration data for analysis and returns the results."""
        with tracer.start_as_current_span("analyze_configuration") as span:
            span.set_attribute("config_data_size", len(str(config_data)))
            try:
                # Placeholder for actual configuration analysis API call
                # For example: self._make_config_analysis_request("POST", "analyze", config_data)

                # Simulating an analysis result
                analysis_results = {
                    "status": "completed",
                    "findings": [
                        {"type": "security_issue", "description": "Open firewall port", "severity": "high"},
                        {"type": "compliance_violation", "description": "Non-compliant naming", "severity": "medium"}
                    ],
                    "recommendations": [
                        "Close port 8080",
                        "Review naming conventions"
                    ]
                }
                span.set_attribute("analysis.findings_count", len(analysis_results["findings"]))
                span.set_status(Status(StatusCode.OK))
                return analysis_results
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to analyze configuration: {e}"))
                raise