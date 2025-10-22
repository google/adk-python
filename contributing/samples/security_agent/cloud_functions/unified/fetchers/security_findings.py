"""
Security findings fetcher module
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from google.cloud import bigquery
import json

from .base import BaseFetcher
from shared import get_authenticated_client, Config


class SecurityFindingsFetcher(BaseFetcher):
    """Fetcher for Security Command Center findings"""

    @property
    def table_name(self) -> str:
        return 'security_findings'

    @property
    def schema(self) -> List[bigquery.SchemaField]:
        return [
            bigquery.SchemaField("finding_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("parent", "STRING"),
            bigquery.SchemaField("resource_name", "STRING"),
            bigquery.SchemaField("state", "STRING"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("external_uri", "STRING"),
            bigquery.SchemaField("severity", "STRING"),
            bigquery.SchemaField("cvss_score", "FLOAT"),
            bigquery.SchemaField("finding_class", "STRING"),
            bigquery.SchemaField("vulnerability_id", "STRING"),
            bigquery.SchemaField("indicator", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("recommendation", "STRING"),
            bigquery.SchemaField("create_time", "TIMESTAMP"),
            bigquery.SchemaField("event_time", "TIMESTAMP"),
            bigquery.SchemaField("update_time", "TIMESTAMP"),
            bigquery.SchemaField("ingestion_time", "TIMESTAMP"),
            bigquery.SchemaField("source_properties", "JSON"),
            bigquery.SchemaField("iam_bindings", "JSON"),
            bigquery.SchemaField("mute_state", "STRING"),
            bigquery.SchemaField("mute_update_time", "TIMESTAMP"),
            bigquery.SchemaField("canonical_name", "STRING"),
            bigquery.SchemaField("next_steps", "STRING"),
            bigquery.SchemaField("contacts", "JSON"),
            bigquery.SchemaField("compliances", "JSON"),
            bigquery.SchemaField("processes", "JSON"),
            bigquery.SchemaField("exfiltration", "JSON"),
            bigquery.SchemaField("mitre_attack", "JSON"),
            bigquery.SchemaField("access", "JSON"),
            bigquery.SchemaField("connections", "JSON"),
            bigquery.SchemaField("containers", "JSON"),
            bigquery.SchemaField("database", "JSON"),
            bigquery.SchemaField("files", "JSON"),
            bigquery.SchemaField("cloud_dlp_inspection", "JSON"),
            bigquery.SchemaField("cloud_dlp_data_profile", "JSON"),
            bigquery.SchemaField("kernel_rootkit", "JSON"),
            bigquery.SchemaField("kubernetes", "JSON"),
            bigquery.SchemaField("load_balancers", "JSON"),
            bigquery.SchemaField("log_entries", "JSON"),
            bigquery.SchemaField("org_policy", "JSON"),
            bigquery.SchemaField("security_posture", "JSON"),
            bigquery.SchemaField("security_marks", "JSON"),
            bigquery.SchemaField("project_id", "STRING")
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch security findings from Security Command Center"""
        findings = []

        try:
            scc_client = get_authenticated_client('securitycenter')

            # Query for recent findings
            parent = f"projects/{Config.PROJECT_ID}/sources/-/locations/global"
            cutoff_time = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
            filter_str = f'event_time >= "{cutoff_time}"'

            # List findings
            finding_result_iterator = scc_client.list_findings(
                request={
                    "parent": parent,
                    "filter": filter_str,
                }
            )

            for finding_result in finding_result_iterator:
                finding = finding_result.finding
                findings.append(self._convert_finding_to_dict(finding))

        except Exception as e:
            self.logger.warning(f"Failed to fetch real findings: {e}")
            # Return sample data if enabled
            if Config.ENABLE_SAMPLE_DATA:
                findings = self.get_sample_data()

        return findings

    def _convert_finding_to_dict(self, finding) -> Dict[str, Any]:
        """Convert a finding object to dictionary"""
        return {
            "finding_id": finding.name.split("/")[-1] if finding.name else "",
            "name": finding.name,
            "parent": finding.parent,
            "resource_name": finding.resource_name,
            "state": finding.state.name if finding.state else "UNKNOWN",
            "category": finding.category,
            "external_uri": finding.external_uri,
            "severity": finding.severity.name if finding.severity else "UNSPECIFIED",
            "cvss_score": finding.cvss_score if hasattr(finding, 'cvss_score') else None,
            "finding_class": finding.finding_class.name if finding.finding_class else "UNSPECIFIED",
            "vulnerability_id": finding.vulnerability.id if finding.vulnerability else None,
            "indicator": json.dumps([ind for ind in finding.indicator]) if finding.indicator else None,
            "description": finding.description if hasattr(finding, 'description') else "",
            "recommendation": finding.recommendation if hasattr(finding, 'recommendation') else "",
            "create_time": finding.create_time,
            "event_time": finding.event_time,
            "update_time": datetime.utcnow().isoformat(),
            "source_properties": json.dumps(dict(finding.source_properties)) if finding.source_properties else "{}",
            "iam_bindings": json.dumps([{
                "action": binding.action.name if binding.action else "",
                "role": binding.role,
                "member": binding.member
            } for binding in finding.iam_bindings]) if finding.iam_bindings else "[]",
            "mute_state": finding.mute.name if finding.mute else "UNMUTED",
            "mute_update_time": finding.mute_update_time if hasattr(finding, 'mute_update_time') else None,
            "canonical_name": finding.canonical_name,
            "next_steps": finding.next_steps if hasattr(finding, 'next_steps') else "",
            "contacts": json.dumps([{"email": c.email} for c in finding.contacts]) if finding.contacts else "[]",
            "compliances": json.dumps([{
                "standard": c.standard,
                "version": c.version,
                "ids": list(c.ids)
            } for c in finding.compliances]) if finding.compliances else "[]",
            "processes": "[]",
            "exfiltration": "{}",
            "mitre_attack": '{"tactics": [], "techniques": []}',
            "access": "{}",
            "connections": "[]",
            "containers": "[]",
            "database": "{}",
            "files": "[]",
            "cloud_dlp_inspection": "{}",
            "cloud_dlp_data_profile": "{}",
            "kernel_rootkit": "{}",
            "kubernetes": "{}",
            "load_balancers": "[]",
            "log_entries": "[]",
            "org_policy": "[]",
            "security_posture": "{}",
            "security_marks": json.dumps(dict(finding.security_marks.marks)) if finding.security_marks else "{}"
        }

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Get sample security findings data"""
        return [
            {
                "finding_id": "sample-finding-001",
                "name": f"organizations/{Config.PROJECT_ID}/sources/sample/findings/sample-finding-001",
                "parent": f"organizations/{Config.PROJECT_ID}/sources/sample",
                "resource_name": f"//compute.googleapis.com/projects/{Config.PROJECT_ID}/instances/web-server-1",
                "state": "ACTIVE",
                "category": "PUBLIC_IP_ADDRESS",
                "external_uri": "https://console.cloud.google.com/security",
                "severity": "HIGH",
                "cvss_score": 7.5,
                "finding_class": "VULNERABILITY",
                "vulnerability_id": "CVE-2024-1234",
                "indicator": json.dumps([]),
                "description": "Instance has public IP address exposed",
                "recommendation": "Consider using Cloud NAT or Private Google Access",
                "create_time": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "event_time": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "update_time": datetime.utcnow().isoformat(),
                "source_properties": "{}",
                "iam_bindings": "[]",
                "mute_state": "UNMUTED",
                "mute_update_time": None,
                "canonical_name": f"projects/{Config.PROJECT_ID}/sources/sample/findings/sample-finding-001",
                "next_steps": "Review instance network configuration",
                "contacts": "[]",
                "compliances": "[]",
                "processes": "[]",
                "exfiltration": "{}",
                "mitre_attack": '{"tactics": ["INITIAL_ACCESS"], "techniques": []}',
                "access": "{}",
                "connections": "[]",
                "containers": "[]",
                "database": "{}",
                "files": "[]",
                "cloud_dlp_inspection": "{}",
                "cloud_dlp_data_profile": "{}",
                "kernel_rootkit": "{}",
                "kubernetes": "{}",
                "load_balancers": "[]",
                "log_entries": "[]",
                "org_policy": "[]",
                "security_posture": "{}",
                "security_marks": "{}"
            }
        ]
