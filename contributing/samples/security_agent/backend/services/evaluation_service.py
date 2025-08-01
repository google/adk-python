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

class SecurityAgentEvaluationService:
    def __init__(self):
        self.evaluation_api_endpoint = os.getenv("EVALUATION_API_ENDPOINT", "https://evaluation.googleapis.com/v1")

    async def _make_evaluation_api_request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Helper to make authenticated requests to an evaluation API."""
        with tracer.start_as_current_span(f"EvaluationService_{method.lower()}_request") as span:
            url = f"{self.evaluation_api_endpoint}/{path}"
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
                span.set_status(Status(StatusCode.ERROR, f"Evaluation API request failed: {e}"))
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"An unexpected error occurred: {e}"))
                raise

    async def fetch_evaluation_criteria(self) -> List[Dict]:
        """Fetches available evaluation criteria for the security agent."""
        with tracer.start_as_current_span("fetch_evaluation_criteria") as span:
            try:
                # Placeholder for actual evaluation criteria API call
                # For example: self._make_evaluation_api_request("GET", "criteria")

                criteria = [
                    {"id": "vuln_scan", "name": "Vulnerability Scan", "description": "Checks for known vulnerabilities"},
                    {"id": "config_review", "name": "Configuration Review", "description": "Reviews cloud resource configurations"}
                ]
                span.set_attribute("evaluation.criteria_count", len(criteria))
                span.set_status(Status(StatusCode.OK))
                return criteria
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to fetch evaluation criteria: {e}"))
                raise

    async def submit_evaluation_request(self, target_data: Dict, evaluation_type: str) -> Dict:
        """Submits data for security evaluation."""
        with tracer.start_as_current_span("submit_evaluation_request") as span:
            span.set_attribute("evaluation.type", evaluation_type)
            span.set_attribute("target_data_size", len(str(target_data)))
            try:
                # Placeholder for actual evaluation submission API call
                # For example: self._make_evaluation_api_request("POST", "evaluate", {"data": target_data, "type": evaluation_type})

                # Simulating an evaluation result
                evaluation_result = {
                    "evaluation_id": "eval_123",
                    "status": "completed",
                    "score": 85,
                    "findings": [
                        {"severity": "high", "description": "Unencrypted storage bucket"},
                        {"severity": "medium", "description": "Excessive IAM permissions"}
                    ]
                }
                span.set_attribute("evaluation.status", evaluation_result["status"])
                span.set_attribute("evaluation.score", evaluation_result["score"])
                span.set_status(Status(StatusCode.OK))
                return evaluation_result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to submit evaluation request: {e}"))
                raise