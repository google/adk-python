#!/usr/bin/env python3
"""
Type-safe security tools using Pydantic models from unified_data_api
Demonstrates improved tool quality with strong typing
"""

from typing import List, Optional
import sys
import os

# Add unified_data_api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unified_data_api.models import (
    IAMAccount, FirewallRule, StorageBucket, SecurityFinding,
    Severity, AccountType
)
from unified_data_api import BigQueryOperations

# Initialize BigQuery operations
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
bq_ops = BigQueryOperations(project_id=PROJECT_ID, dataset_id="security_insights")


def get_primitive_role_accounts() -> List[IAMAccount]:
    """
    Get all accounts with primitive roles (owner/editor/viewer)

    Returns:
        List of IAMAccount objects with primitive roles
    """
    query = """
    SELECT *
    FROM `security_insights.iam_accounts`
    WHERE is_primitive_role = true
    ORDER BY role DESC, email
    """

    accounts = bq_ops.query_to_models(query, IAMAccount)
    return accounts


def get_old_service_account_keys() -> List[IAMAccount]:
    """
    Get service accounts with keys older than 90 days

    Returns:
        List of IAMAccount objects with old keys
    """
    query = """
    SELECT *
    FROM `security_insights.iam_accounts`
    WHERE account_type = 'serviceAccount'
      AND key_age_days > 90
    ORDER BY key_age_days DESC
    """

    accounts = bq_ops.query_to_models(query, IAMAccount)
    return accounts


def get_open_firewall_rules() -> List[FirewallRule]:
    """
    Get firewall rules that allow traffic from anywhere (0.0.0.0/0)

    Returns:
        List of FirewallRule objects with open access
    """
    query = """
    SELECT *
    FROM `security_insights.firewall_rules`
    WHERE allows_all_ips = true
      AND action = 'ALLOW'
    ORDER BY priority
    """

    rules = bq_ops.query_to_models(query, FirewallRule)
    return rules


def get_ssh_accessible_resources() -> List[FirewallRule]:
    """
    Get firewall rules that allow SSH from anywhere

    Returns:
        List of FirewallRule objects allowing SSH
    """
    query = """
    SELECT *
    FROM `security_insights.firewall_rules`
    WHERE allows_ssh = true
      AND allows_all_ips = true
      AND action = 'ALLOW'
    """

    rules = bq_ops.query_to_models(query, FirewallRule)
    return rules


def get_public_storage_buckets() -> List[StorageBucket]:
    """
    Get Cloud Storage buckets that are publicly accessible

    Returns:
        List of StorageBucket objects with public access
    """
    query = """
    SELECT *
    FROM `security_insights.storage_buckets`
    WHERE is_public = true
    ORDER BY created_at DESC
    """

    buckets = bq_ops.query_to_models(query, StorageBucket)
    return buckets


def get_unencrypted_buckets() -> List[StorageBucket]:
    """
    Get buckets without customer-managed encryption keys (CMEK)

    Returns:
        List of StorageBucket objects without CMEK
    """
    query = """
    SELECT *
    FROM `security_insights.storage_buckets`
    WHERE encryption_type != 'CUSTOMER_MANAGED'
       OR encryption_type IS NULL
    ORDER BY size_bytes DESC
    """

    buckets = bq_ops.query_to_models(query, StorageBucket)
    return buckets


def get_critical_security_findings() -> List[SecurityFinding]:
    """
    Get CRITICAL severity security findings

    Returns:
        List of SecurityFinding objects with CRITICAL severity
    """
    query = """
    SELECT *
    FROM `security_insights.security_findings`
    WHERE severity = 'CRITICAL'
      AND state = 'ACTIVE'
    ORDER BY created_at DESC
    """

    findings = bq_ops.query_to_models(query, SecurityFinding)
    return findings


def get_high_severity_findings_by_resource(resource_type: str) -> List[SecurityFinding]:
    """
    Get high/critical severity findings for a specific resource type

    Args:
        resource_type: GCP resource type (e.g., "storage.buckets", "compute.instances")

    Returns:
        List of SecurityFinding objects for the resource type
    """
    query = f"""
    SELECT *
    FROM `security_insights.security_findings`
    WHERE resource_type = @resource_type
      AND severity IN ('CRITICAL', 'HIGH')
      AND state = 'ACTIVE'
    ORDER BY severity DESC, created_at DESC
    """

    from google.cloud import bigquery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("resource_type", "STRING", resource_type)
        ]
    )

    query_job = bq_ops.client.query(query, job_config=job_config)
    results = query_job.result()

    findings = []
    for row in results:
        finding = SecurityFinding(**dict(row.items()))
        findings.append(finding)

    return findings


def analyze_iam_security_posture() -> dict:
    """
    Comprehensive IAM security posture analysis with typed data

    Returns:
        Dict with analysis results and typed recommendations
    """
    # Get all IAM issues using typed queries
    primitive_roles = get_primitive_role_accounts()
    old_keys = get_old_service_account_keys()

    # Count by severity
    critical_issues = len([a for a in primitive_roles if a.role in ["roles/owner"]])
    high_issues = len([a for a in primitive_roles if a.role == "roles/editor"])
    medium_issues = len(old_keys)

    return {
        "total_issues": len(primitive_roles) + len(old_keys),
        "critical": {
            "count": critical_issues,
            "accounts": [
                {
                    "email": acc.email,
                    "role": acc.role,
                    "account_type": acc.account_type.value,
                    "created_at": acc.created_at.isoformat()
                }
                for acc in primitive_roles if acc.role == "roles/owner"
            ]
        },
        "high": {
            "count": high_issues,
            "accounts": [
                {
                    "email": acc.email,
                    "role": acc.role,
                    "account_type": acc.account_type.value
                }
                for acc in primitive_roles if acc.role == "roles/editor"
            ]
        },
        "medium": {
            "count": medium_issues,
            "old_keys": [
                {
                    "email": acc.email,
                    "key_age_days": acc.key_age_days,
                    "created_at": acc.created_at.isoformat()
                }
                for acc in old_keys[:5]  # Top 5 oldest
            ]
        },
        "recommendations": [
            "Replace primitive roles with predefined roles",
            "Rotate service account keys older than 90 days",
            "Enable key rotation policies",
            "Review and remove unnecessary IAM bindings"
        ]
    }


def analyze_network_security_posture() -> dict:
    """
    Comprehensive network security posture analysis with typed data

    Returns:
        Dict with analysis results and typed recommendations
    """
    # Get network security issues
    open_rules = get_open_firewall_rules()
    ssh_rules = get_ssh_accessible_resources()

    # Categorize by severity
    critical_rules = [r for r in ssh_rules if r.allows_ssh and r.allows_all_ips]
    high_rules = [r for r in open_rules if "22" in r.ports or "3389" in r.ports]

    return {
        "total_issues": len(open_rules),
        "critical": {
            "count": len(critical_rules),
            "ssh_from_internet": [
                {
                    "rule_name": rule.rule_name,
                    "network": rule.network,
                    "ports": rule.ports,
                    "source_ranges": rule.source_ranges,
                    "priority": rule.priority
                }
                for rule in critical_rules
            ]
        },
        "high": {
            "count": len(high_rules),
            "open_management_ports": [
                {
                    "rule_name": rule.rule_name,
                    "ports": rule.ports,
                    "protocols": rule.protocols
                }
                for rule in high_rules
            ]
        },
        "recommendations": [
            "Restrict SSH/RDP access to specific IP ranges",
            "Use Cloud IAP for instance access instead of public IPs",
            "Enable VPC Flow Logs for monitoring",
            "Review and tighten firewall rule source ranges"
        ]
    }


# Export typed tools for ADK agent
__all__ = [
    "get_primitive_role_accounts",
    "get_old_service_account_keys",
    "get_open_firewall_rules",
    "get_ssh_accessible_resources",
    "get_public_storage_buckets",
    "get_unencrypted_buckets",
    "get_critical_security_findings",
    "get_high_severity_findings_by_resource",
    "analyze_iam_security_posture",
    "analyze_network_security_posture",
]
