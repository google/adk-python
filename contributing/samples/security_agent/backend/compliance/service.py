import os
import requests
import asyncio
import logging
from typing import List, Dict, Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

import google.auth
import google.auth.transport.requests
from core.base_service import BaseService

# Get tracer and logger
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

class ComplianceService(BaseService):
    def __init__(self, service_name: str = 'compliance', credentials=None, project_id=None):
        super().__init__(service_name, credentials, project_id)
        self.compliance_api_endpoint = os.getenv("COMPLIANCE_API_ENDPOINT", "https://compliance.googleapis.com/v1")
        self.credentials = credentials
        self.project_id = project_id
    
    async def initialize(self) -> bool:
        """Initialize the Compliance service."""
        try:
            logger.info("Initializing Compliance service...")
            # Test basic functionality
            if self.compliance_api_endpoint:
                logger.info("Compliance service initialized successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Compliance service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Compliance service."""
        try:
            logger.info("Shutting down Compliance service...")
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown Compliance service: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Compliance service health."""
        try:
            return {
                "healthy": True,
                "status": "running",
                "endpoint": self.compliance_api_endpoint,
                "message": "Compliance service is operational"
            }
        except Exception as e:
            logger.error(f"Compliance health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "message": "Compliance service health check failed"
            }

    async def _make_compliance_api_request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """Helper to make authenticated requests to a compliance API."""
        with tracer.start_as_current_span(f"ComplianceService_{method.lower()}_request") as span:
            url = f"{self.compliance_api_endpoint}/{path}"
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", url)

            try:
                if not self.credentials:
                    raise Exception("ComplianceService is not initialized with credentials.")

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
                span.set_status(Status(StatusCode.ERROR, f"Compliance API request failed: {e}"))
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"An unexpected error occurred: {e}"))
                raise

    async def fetch_compliance_standards(self) -> List[Dict]:
        """Fetches available compliance standards and their controls."""
        with tracer.start_as_current_span("fetch_compliance_standards") as span:
            try:
                # Placeholder for actual compliance API call
                # For example: self._make_compliance_api_request("GET", "standards")
                
                standards = [
                    {"id": "soc2", "name": "SOC 2", "controls": ["control_1", "control_2"]},
                    {"id": "iso27001", "name": "ISO 27001", "controls": ["control_a", "control_b"]}
                ]
                span.set_attribute("compliance.standards_count", len(standards))
                span.set_status(Status(StatusCode.OK))
                return standards
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to fetch compliance standards: {e}"))
                raise

    async def evaluate_compliance(self, config_data: Dict, standards: List[str]) -> Dict:
        """Evaluates a given configuration against specified compliance standards."""
        with tracer.start_as_current_span("evaluate_compliance") as span:
            span.set_attribute("compliance.standards_evaluated", str(standards))
            try:
                # Placeholder for actual compliance evaluation logic or API call
                # This would involve sending config_data to a compliance engine
                
                results = {"overall_status": "compliant", "details": {}}
                for standard_id in standards:
                    results["details"][standard_id] = {"status": "compliant", "findings": []}

                span.set_attribute("compliance.evaluation_status", results["overall_status"])
                span.set_status(Status(StatusCode.OK))
                return results
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to evaluate compliance: {e}"))
                raise