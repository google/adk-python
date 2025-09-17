"""
GCP Live Data Tool - Fetches real-time data from GCP APIs
ADK-compliant tool for live GCP resource analysis
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

# GCP SDK imports
try:
    from google.cloud import storage
    from google.cloud import resource_manager_v3
    from google.cloud import asset_v1
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class GCPLiveDataTool:
    """
    ADK-compliant tool for fetching live GCP data.
    Provides real-time bucket analysis and security findings.
    """

    def __init__(self):
        self.name = "gcp_live_data"
        self.description = "Fetch real-time GCP resource data including buckets, IAM, and security analysis"

        # Initialize GCP clients
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not self.project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT not set - using fallback data")

        if GCP_AVAILABLE and self.credentials_path and os.path.exists(self.credentials_path):
            try:
                self.storage_client = storage.Client(project=self.project_id)
                self.asset_client = asset_v1.AssetServiceClient()
                self.project_name = f"projects/{self.project_id}"
                logger.info("✅ GCP Live Data Tool initialized successfully")
            except Exception as e:
                logger.warning(f"GCP client initialization failed: {e}")
                self.storage_client = None
                self.asset_client = None
        else:
            logger.warning("GCP SDK not available or credentials missing - using simulation mode")
            self.storage_client = None
            self.asset_client = None

    def get_schema(self) -> Dict[str, Any]:
        """Return the tool schema for ADK registration."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["buckets", "bucket_security", "iam_analysis", "project_overview"],
                        "description": "Type of GCP data to fetch"
                    },
                    "bucket_name": {
                        "type": "string",
                        "description": "Specific bucket name to analyze (optional)"
                    },
                    "security_check": {
                        "type": "boolean",
                        "description": "Include security analysis in results",
                        "default": True
                    }
                },
                "required": ["query_type"]
            }
        }

    def execute(self, query_type: str, bucket_name: Optional[str] = None, security_check: bool = True) -> Dict[str, Any]:
        """Execute the GCP live data query."""
        try:
            logger.info(f"Executing GCP live query: {query_type}")

            if query_type == "buckets":
                return self._fetch_storage_buckets(security_check)
            elif query_type == "bucket_security":
                return self._analyze_bucket_security(bucket_name)
            elif query_type == "iam_analysis":
                return self._analyze_iam_security()
            elif query_type == "project_overview":
                return self._get_project_overview()
            else:
                return {"error": f"Invalid query type: {query_type}"}

        except Exception as e:
            logger.error(f"GCP Live Data Tool error: {e}")
            return {"error": str(e), "fallback_used": True}

    def _fetch_storage_buckets(self, security_check: bool = True) -> Dict[str, Any]:
        """Fetch real Cloud Storage buckets with security analysis."""
        if not self.storage_client:
            return self._fallback_bucket_data()

        try:
            buckets = list(self.storage_client.list_buckets())
            bucket_data = []
            security_issues = []

            for bucket in buckets:
                bucket_info = {
                    "name": bucket.name,
                    "location": bucket.location,
                    "storage_class": bucket.storage_class,
                    "created": bucket.time_created.isoformat() if bucket.time_created else None,
                    "versioning_enabled": bucket.versioning_enabled,
                    "public_access_prevention": getattr(bucket, 'iam_configuration', {}).get('public_access_prevention'),
                    "uniform_bucket_level_access": getattr(bucket, 'iam_configuration', {}).get('uniform_bucket_level_access_enabled'),
                }

                # Security analysis
                if security_check:
                    security_findings = self._analyze_bucket_security_detailed(bucket)
                    bucket_info["security_findings"] = security_findings
                    security_issues.extend(security_findings)

                bucket_data.append(bucket_info)

            return {
                "buckets": bucket_data,
                "count": len(bucket_data),
                "security_issues": security_issues,
                "project_id": self.project_id,
                "timestamp": datetime.now().isoformat(),
                "data_source": "live_gcp_api"
            }

        except Exception as e:
            logger.error(f"Error fetching buckets: {e}")
            return self._fallback_bucket_data()

    def _analyze_bucket_security_detailed(self, bucket) -> List[Dict[str, Any]]:
        """Analyze bucket for security issues."""
        issues = []

        try:
            # Check IAM policy for public access
            policy = bucket.get_iam_policy(requested_policy_version=3)

            for binding in policy.bindings:
                if "allUsers" in binding.members or "allAuthenticatedUsers" in binding.members:
                    issues.append({
                        "type": "PUBLIC_BUCKET",
                        "severity": "HIGH",
                        "description": f"Bucket {bucket.name} has public access via IAM",
                        "resource": f"//storage.googleapis.com/{bucket.name}",
                        "recommendation": "Remove allUsers and allAuthenticatedUsers from IAM policy",
                        "members": list(binding.members),
                        "role": binding.role
                    })

            # Check public access prevention
            if not getattr(bucket, 'iam_configuration', {}).get('public_access_prevention') == 'enforced':
                issues.append({
                    "type": "PUBLIC_ACCESS_NOT_PREVENTED",
                    "severity": "MEDIUM",
                    "description": f"Bucket {bucket.name} does not enforce public access prevention",
                    "resource": f"//storage.googleapis.com/{bucket.name}",
                    "recommendation": "Enable public access prevention",
                })

            # Check uniform bucket-level access
            if not getattr(bucket, 'iam_configuration', {}).get('uniform_bucket_level_access_enabled'):
                issues.append({
                    "type": "BUCKET_LEVEL_ACCESS_DISABLED",
                    "severity": "LOW",
                    "description": f"Bucket {bucket.name} does not use uniform bucket-level access",
                    "resource": f"//storage.googleapis.com/{bucket.name}",
                    "recommendation": "Enable uniform bucket-level access for better security",
                })

        except Exception as e:
            logger.warning(f"Could not analyze bucket {bucket.name} security: {e}")

        return issues

    def _analyze_bucket_security(self, bucket_name: Optional[str] = None) -> Dict[str, Any]:
        """Detailed security analysis for specific bucket or all buckets."""
        if not self.storage_client:
            return {"error": "GCP client not available", "fallback_used": True}

        try:
            if bucket_name:
                bucket = self.storage_client.bucket(bucket_name)
                issues = self._analyze_bucket_security_detailed(bucket)
                return {
                    "bucket": bucket_name,
                    "security_issues": issues,
                    "issue_count": len(issues),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Analyze all buckets
                return self._fetch_storage_buckets(security_check=True)

        except Exception as e:
            logger.error(f"Bucket security analysis error: {e}")
            return {"error": str(e)}

    def _analyze_iam_security(self) -> Dict[str, Any]:
        """Analyze IAM security across the project."""
        if not self.storage_client:
            return {"error": "GCP client not available", "fallback_used": True}

        try:
            # This would typically use Cloud Asset Inventory or IAM APIs
            # For now, focusing on storage bucket IAM
            buckets_result = self._fetch_storage_buckets(security_check=True)

            iam_issues = []
            for bucket in buckets_result.get("buckets", []):
                iam_issues.extend(bucket.get("security_findings", []))

            return {
                "iam_analysis": "bucket_focus",
                "total_issues": len(iam_issues),
                "issues_by_severity": self._group_by_severity(iam_issues),
                "issues": iam_issues,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"IAM analysis error: {e}")
            return {"error": str(e)}

    def _get_project_overview(self) -> Dict[str, Any]:
        """Get comprehensive project security overview."""
        try:
            overview = {
                "project_id": self.project_id,
                "timestamp": datetime.now().isoformat(),
                "data_source": "live_gcp_api"
            }

            # Get bucket analysis
            bucket_data = self._fetch_storage_buckets(security_check=True)
            overview["storage"] = {
                "bucket_count": bucket_data.get("count", 0),
                "security_issues": bucket_data.get("security_issues", [])
            }

            # Summary statistics
            all_issues = bucket_data.get("security_issues", [])
            overview["security_summary"] = {
                "total_findings": len(all_issues),
                "findings_by_severity": self._group_by_severity(all_issues),
                "findings_by_type": self._group_by_type(all_issues)
            }

            return overview

        except Exception as e:
            logger.error(f"Project overview error: {e}")
            return {"error": str(e)}

    def _group_by_severity(self, issues: List[Dict]) -> Dict[str, int]:
        """Group issues by severity level."""
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in issues:
            severity = issue.get("severity", "UNKNOWN")
            if severity in severity_counts:
                severity_counts[severity] += 1
        return severity_counts

    def _group_by_type(self, issues: List[Dict]) -> Dict[str, int]:
        """Group issues by type."""
        type_counts = {}
        for issue in issues:
            issue_type = issue.get("type", "UNKNOWN")
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        return type_counts

    def _fallback_bucket_data(self) -> Dict[str, Any]:
        """Fallback data when GCP API is not available."""
        logger.warning("Using fallback bucket data - GCP API not accessible")
        return {
            "buckets": [
                {
                    "name": "example-bucket-1",
                    "location": "US",
                    "storage_class": "STANDARD",
                    "created": "2024-01-01T00:00:00Z",
                    "versioning_enabled": False,
                    "public_access_prevention": None,
                    "uniform_bucket_level_access": False,
                    "security_findings": [
                        {
                            "type": "PUBLIC_BUCKET",
                            "severity": "HIGH",
                            "description": "Bucket has public access",
                            "resource": "//storage.googleapis.com/example-bucket-1",
                            "recommendation": "Remove public access"
                        }
                    ]
                }
            ],
            "count": 1,
            "security_issues": [
                {
                    "type": "PUBLIC_BUCKET",
                    "severity": "HIGH",
                    "description": "Bucket has public access",
                    "resource": "//storage.googleapis.com/example-bucket-1",
                    "recommendation": "Remove public access"
                }
            ],
            "project_id": self.project_id or "demo-project",
            "timestamp": datetime.now().isoformat(),
            "data_source": "fallback_simulation",
            "note": "This is simulated data. Configure GCP credentials for live data."
        }

# Create tool instance
gcp_live_tool = GCPLiveDataTool()