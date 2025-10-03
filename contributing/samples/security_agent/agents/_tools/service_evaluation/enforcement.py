#!/usr/bin/env python3
"""
Enforcement Methods Analyzer
Determines how security controls can be enforced (Org Policies, Cloud Functions, Terraform, etc.)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class EnforcementMethod(Enum):
    """Available enforcement mechanisms"""
    ORG_POLICY = "organization_policy"
    IAM_POLICY = "iam_policy"
    CLOUD_FUNCTION = "cloud_function"
    TERRAFORM = "terraform"
    CONFIG_CONNECTOR = "config_connector"
    CLOUD_BUILD = "cloud_build"
    SECURITY_COMMAND_CENTER = "security_command_center"
    FORSETI = "forseti"
    CUSTOM_SCRIPT = "custom_script"
    MANUAL = "manual_process"


@dataclass
class EnforcementOption:
    """Enforcement option for a security control"""
    method: EnforcementMethod
    name: str
    description: str
    complexity: str  # low, medium, high
    automation_level: str  # full, partial, manual
    implementation_template: Optional[str] = None
    cost_estimate: str = "low"
    maintenance_effort: str = "low"  # low, medium, high
    gcp_documentation: str = ""
    prerequisites: List[str] = None
    limitations: List[str] = None

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.limitations is None:
            self.limitations = []


class EnforcementAnalyzer:
    """Analyzes and recommends enforcement methods for security controls"""

    def __init__(self):
        self.enforcement_catalog = self._initialize_enforcement_catalog()

    def _initialize_enforcement_catalog(self) -> Dict[str, List[EnforcementOption]]:
        """Initialize catalog of enforcement methods per control"""
        catalog = {}

        # IAM Controls Enforcement
        catalog['IAM-001'] = [
            EnforcementOption(
                method=EnforcementMethod.ORG_POLICY,
                name='Disable Service Account Key Creation',
                description='Prevent creation of service account keys at organization level',
                complexity='low',
                automation_level='full',
                implementation_template='''
gcloud resource-manager org-policies set-policy \
  --organization=YOUR_ORG_ID \
  constraints/iam.disableServiceAccountKeyCreation.yaml
                ''',
                cost_estimate='free',
                prerequisites=['Organization admin access'],
                limitations=['Does not prevent use of existing keys']
            ),
            EnforcementOption(
                method=EnforcementMethod.CLOUD_FUNCTION,
                name='IAM Binding Scanner',
                description='Automated scanner to detect and alert on primitive role assignments',
                complexity='medium',
                automation_level='partial',
                implementation_template='''
# Cloud Function that runs daily
def scan_iam_bindings(event, context):
    # Query all IAM bindings
    # Alert on primitive roles (Owner, Editor, Viewer)
    # Send to Pub/Sub for remediation workflow
                ''',
                cost_estimate='low ($0.10/month)',
                prerequisites=['Cloud Functions API', 'Pub/Sub topic'],
                limitations=['Detection only, requires manual remediation']
            ),
            EnforcementOption(
                method=EnforcementMethod.TERRAFORM,
                name='IaC Policy Enforcement',
                description='Use Terraform sentinel policies to prevent primitive roles in code',
                complexity='medium',
                automation_level='full',
                implementation_template='''
policy "no-primitive-roles" {
  enforcement_level = "hard-mandatory"
  rules = [
    rule "deny_owner_role" {
      condition = google_project_iam_binding.role != "roles/owner"
    }
  ]
}
                ''',
                cost_estimate='free',
                prerequisites=['Terraform Enterprise/Cloud'],
                limitations=['Only prevents new deployments via Terraform']
            )
        ]

        # Network Security Enforcement
        catalog['NET-001'] = [
            EnforcementOption(
                method=EnforcementMethod.ORG_POLICY,
                name='Require Private Google Access',
                description='Org policy to enforce Private Google Access on all subnets',
                complexity='low',
                automation_level='full',
                implementation_template='''
# Organization Policy YAML
name: organizations/YOUR_ORG_ID/policies/compute.requirePrivateGoogleAccess
spec:
  rules:
  - enforce: true
                ''',
                cost_estimate='free',
                prerequisites=['Organization admin'],
                limitations=['Applies to new resources only']
            ),
            EnforcementOption(
                method=EnforcementMethod.TERRAFORM,
                name='VPC Module with Defaults',
                description='Terraform module that enforces Private Google Access by default',
                complexity='low',
                automation_level='full',
                implementation_template='''
module "vpc" {
  source = "./modules/secure-vpc"
  private_google_access = true  # Always enabled
  external_nat = true
}
                ''',
                cost_estimate='free',
                prerequisites=['Terraform infrastructure'],
                limitations=['Only for Terraform-managed resources']
            )
        ]

        catalog['NET-002'] = [
            EnforcementOption(
                method=EnforcementMethod.CLOUD_FUNCTION,
                name='Firewall Rule Auditor',
                description='Automated firewall rule review and alerting',
                complexity='medium',
                automation_level='partial',
                implementation_template='''
def audit_firewall_rules(event, context):
    # List all firewall rules
    # Check for 0.0.0.0/0 allow rules
    # Alert on overly permissive rules
    # Suggest alternatives with specific source ranges
                ''',
                cost_estimate='low ($0.15/month)',
                prerequisites=['Cloud Functions API', 'IAM permissions'],
                limitations=['Detection only, manual review required']
            ),
            EnforcementOption(
                method=EnforcementMethod.SECURITY_COMMAND_CENTER,
                name='SCC Firewall Findings',
                description='Use Security Command Center to detect overly permissive firewall rules',
                complexity='low',
                automation_level='full',
                implementation_template='''
# Enable SCC Premium tier
# Configure firewall rule detector
# Set up notification channels for findings
                ''',
                cost_estimate='medium (SCC Premium pricing)',
                prerequisites=['Security Command Center Premium'],
                limitations=['Detection only, requires response workflow']
            )
        ]

        # Data Protection Enforcement
        catalog['DATA-001'] = [
            EnforcementOption(
                method=EnforcementMethod.ORG_POLICY,
                name='Require CMEK for Storage',
                description='Enforce customer-managed encryption keys for Cloud Storage',
                complexity='medium',
                automation_level='full',
                implementation_template='''
# Org Policy: constraints/gcp.restrictNonCmekServices
# List of allowed services without CMEK
allowedValues:
  - compute.googleapis.com  # Boot disks can use Google-managed keys
deniedValues:
  - storage.googleapis.com  # Storage must use CMEK
                ''',
                cost_estimate='low (KMS key costs)',
                prerequisites=['Cloud KMS setup', 'Organization admin'],
                limitations=['Requires KMS key management process']
            ),
            EnforcementOption(
                method=EnforcementMethod.TERRAFORM,
                name='CMEK-Enabled Storage Module',
                description='Terraform module that requires CMEK for all storage buckets',
                complexity='medium',
                automation_level='full',
                implementation_template='''
resource "google_storage_bucket" "secure_bucket" {
  name     = var.bucket_name
  location = var.region

  encryption {
    default_kms_key_name = var.kms_key_id  # Required parameter
  }
}
                ''',
                cost_estimate='low (KMS costs)',
                prerequisites=['KMS keys provisioned'],
                limitations=['Terraform-managed buckets only']
            )
        ]

        catalog['DATA-002'] = [
            EnforcementOption(
                method=EnforcementMethod.ORG_POLICY,
                name='Block Public Access',
                description='Prevent public access to Cloud Storage buckets',
                complexity='low',
                automation_level='full',
                implementation_template='''
# constraints/storage.publicAccessPrevention
gcloud resource-manager org-policies set-policy \
  --organization=YOUR_ORG_ID \
  constraints/storage.publicAccessPrevention.yaml

# YAML content:
constraint: constraints/storage.publicAccessPrevention
listPolicy:
  deniedValues:
  - "allUsers"
  - "allAuthenticatedUsers"
                ''',
                cost_estimate='free',
                prerequisites=['Organization admin access'],
                limitations=['Organization-wide policy']
            ),
            EnforcementOption(
                method=EnforcementMethod.CLOUD_FUNCTION,
                name='Public Access Remediation',
                description='Automated detection and remediation of public buckets',
                complexity='high',
                automation_level='full',
                implementation_template='''
def remediate_public_buckets(event, context):
    storage_client = storage.Client()
    for bucket in storage_client.list_buckets():
        policy = bucket.get_iam_policy()
        if has_public_access(policy):
            remove_public_bindings(bucket, policy)
            alert_security_team(bucket.name)
                ''',
                cost_estimate='low ($0.20/month)',
                prerequisites=['Cloud Functions API', 'Storage Admin permissions'],
                limitations=['Continuous monitoring required']
            )
        ]

        # Logging & Monitoring Enforcement
        catalog['LOG-001'] = [
            EnforcementOption(
                method=EnforcementMethod.ORG_POLICY,
                name='Enforce Audit Logging',
                description='Organization policy to enable all audit log types',
                complexity='low',
                automation_level='full',
                implementation_template='''
# Enable via gcloud or console
gcloud logging settings update \
  --organization=YOUR_ORG_ID \
  --enable-audit-logs \
  --audit-log-types=ADMIN_READ,DATA_READ,DATA_WRITE
                ''',
                cost_estimate='medium (log storage costs)',
                prerequisites=['Organization admin', 'Logging admin'],
                limitations=['Storage costs can scale with usage']
            ),
            EnforcementOption(
                method=EnforcementMethod.TERRAFORM,
                name='Audit Config Module',
                description='Terraform module that configures audit logging for all services',
                complexity='medium',
                automation_level='full',
                implementation_template='''
resource "google_project_iam_audit_config" "all_services" {
  project = var.project_id
  service = "allServices"

  audit_log_config {
    log_type = "ADMIN_READ"
  }
  audit_log_config {
    log_type = "DATA_READ"
  }
  audit_log_config {
    log_type = "DATA_WRITE"
  }
}
                ''',
                cost_estimate='medium (log costs)',
                prerequisites=['Terraform setup'],
                limitations=['Per-project configuration']
            )
        ]

        # Configuration Management Enforcement
        catalog['CFG-001'] = [
            EnforcementOption(
                method=EnforcementMethod.ORG_POLICY,
                name='Organization Policy Bundle',
                description='Comprehensive org policies for security baseline',
                complexity='medium',
                automation_level='full',
                implementation_template='''
# Apply all recommended org policies
# - Disable SA key creation
# - Require OS Login
# - Enforce uniform bucket-level access
# - Restrict VM serial port access
# - Require Shielded VMs
# - Disable default network creation

gcloud resource-manager org-policies set-policy \
  --organization=YOUR_ORG_ID \
  security-baseline-policies/
                ''',
                cost_estimate='free',
                prerequisites=['Organization admin'],
                limitations=['May impact existing workflows']
            ),
            EnforcementOption(
                method=EnforcementMethod.CONFIG_CONNECTOR,
                name='Config Connector Policies',
                description='Kubernetes-based policy enforcement via Config Connector',
                complexity='high',
                automation_level='full',
                implementation_template='''
apiVersion: resourcemanager.cnrm.cloud.google.com/v1beta1
kind: OrganizationPolicy
metadata:
  name: disable-sa-key-creation
spec:
  organizationRef:
    external: "YOUR_ORG_ID"
  constraint: "iam.disableServiceAccountKeyCreation"
  listPolicy:
    allow:
      all: false
                ''',
                cost_estimate='low (GKE costs)',
                prerequisites=['GKE cluster', 'Config Connector installed'],
                limitations=['Requires Kubernetes knowledge']
            )
        ]

        return catalog

    def get_enforcement_options(self, control_id: str) -> List[EnforcementOption]:
        """Get all enforcement options for a control"""
        return self.enforcement_catalog.get(control_id, [])

    def recommend_enforcement_method(self, control_id: str,
                                    preferences: Dict[str, Any] = None) -> EnforcementOption:
        """
        Recommend best enforcement method based on preferences

        Preferences can include:
        - automation_preference: full, partial, manual
        - complexity_limit: low, medium, high
        - cost_limit: low, medium, high
        - existing_infrastructure: list of existing tools
        """
        options = self.get_enforcement_options(control_id)
        if not options:
            return None

        if not preferences:
            # Default: prefer full automation with low complexity
            return sorted(options, key=lambda x: (
                0 if x.automation_level == 'full' else 1,
                0 if x.complexity == 'low' else 1 if x.complexity == 'medium' else 2
            ))[0]

        # Score each option based on preferences
        scored_options = []
        for option in options:
            score = 0

            # Automation preference
            if preferences.get('automation_preference') == option.automation_level:
                score += 3

            # Complexity limit
            complexity_scores = {'low': 3, 'medium': 2, 'high': 1}
            complexity_limit = preferences.get('complexity_limit', 'high')
            if complexity_scores[option.complexity] >= complexity_scores[complexity_limit]:
                score += 2

            # Cost limit
            cost_scores = {'free': 3, 'low': 2, 'medium': 1, 'high': 0}
            cost_limit = preferences.get('cost_limit', 'medium')
            if cost_scores.get(option.cost_estimate, 1) >= cost_scores[cost_limit]:
                score += 2

            # Existing infrastructure match
            existing_infra = preferences.get('existing_infrastructure', [])
            if option.method.value in existing_infra:
                score += 3

            scored_options.append((score, option))

        # Return highest scoring option
        return max(scored_options, key=lambda x: x[0])[1]

    def generate_implementation_plan(self, control_ids: List[str],
                                    preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a complete implementation plan for multiple controls"""
        plan = {
            'controls': [],
            'by_method': {},
            'estimated_cost': 'TBD',
            'implementation_order': [],
            'prerequisites': set(),
            'total_complexity': 0
        }

        complexity_scores = {'low': 1, 'medium': 3, 'high': 5}

        for control_id in control_ids:
            recommended = self.recommend_enforcement_method(control_id, preferences)
            if not recommended:
                continue

            control_plan = {
                'control_id': control_id,
                'enforcement_method': recommended.method.value,
                'name': recommended.name,
                'complexity': recommended.complexity,
                'automation_level': recommended.automation_level,
                'implementation_template': recommended.implementation_template,
                'cost_estimate': recommended.cost_estimate,
                'prerequisites': recommended.prerequisites
            }

            plan['controls'].append(control_plan)

            # Group by method
            method = recommended.method.value
            if method not in plan['by_method']:
                plan['by_method'][method] = []
            plan['by_method'][method].append(control_id)

            # Collect prerequisites
            plan['prerequisites'].update(recommended.prerequisites)

            # Calculate total complexity
            plan['total_complexity'] += complexity_scores[recommended.complexity]

        # Determine implementation order (low complexity first)
        plan['implementation_order'] = sorted(
            plan['controls'],
            key=lambda x: complexity_scores[x['complexity']]
        )

        # Convert set to list for JSON serialization
        plan['prerequisites'] = list(plan['prerequisites'])

        return plan

    def get_method_summary(self) -> Dict[str, int]:
        """Get summary of available enforcement methods"""
        method_counts = {method.value: 0 for method in EnforcementMethod}

        for options in self.enforcement_catalog.values():
            for option in options:
                method_counts[option.method.value] += 1

        return method_counts
