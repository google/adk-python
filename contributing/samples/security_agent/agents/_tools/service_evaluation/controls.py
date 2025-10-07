#!/usr/bin/env python3
"""
Security Controls Inventory
Comprehensive catalog of security controls applicable to GCP services
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ControlCategory(Enum):
    """Security control categories"""
    IDENTITY = "identity_access"
    NETWORK = "network_security"
    DATA = "data_protection"
    LOGGING = "logging_monitoring"
    ENCRYPTION = "encryption"
    COMPLIANCE = "compliance"
    CONFIGURATION = "configuration"
    INCIDENT = "incident_response"


class ControlType(Enum):
    """Types of security controls"""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    DETERRENT = "deterrent"


@dataclass
class SecurityControl:
    """Individual security control"""
    id: str
    name: str
    category: ControlCategory
    control_type: ControlType
    description: str
    applies_to: List[str]  # Service types this applies to
    severity: str  # critical, high, medium, low
    cis_benchmark: str = None
    nist_framework: str = None
    pci_dss: str = None
    hipaa: str = None
    sox: str = None
    gdpr: str = None
    implementation_guidance: str = ""
    validation_query: str = None


class SecurityControlsInventory:
    """Comprehensive inventory of GCP security controls"""

    def __init__(self):
        self.controls = self._initialize_controls()

    def _initialize_controls(self) -> Dict[str, SecurityControl]:
        """Initialize the complete controls catalog"""
        controls = {}

        # Identity & Access Controls
        controls['IAM-001'] = SecurityControl(
            id='IAM-001',
            name='Least Privilege IAM Roles',
            category=ControlCategory.IDENTITY,
            control_type=ControlType.PREVENTIVE,
            description='Use least-privilege IAM roles, avoid primitive roles (Owner, Editor, Viewer)',
            applies_to=['all'],
            severity='critical',
            cis_benchmark='1.1, 1.2, 1.3',
            nist_framework='AC-6',
            pci_dss='7.1.2',
            hipaa='164.308(a)(4)',
            implementation_guidance='Replace primitive roles with predefined or custom roles with minimal permissions',
            validation_query='SELECT email, role, account_type FROM `security_insights.iam_accounts` WHERE role IN ("roles/owner", "roles/editor") OR role IN ("roles/viewer")'
        )

        controls['IAM-002'] = SecurityControl(
            id='IAM-002',
            name='Service Account Key Rotation',
            category=ControlCategory.IDENTITY,
            control_type=ControlType.PREVENTIVE,
            description='Rotate service account keys every 90 days maximum',
            applies_to=['compute', 'kubernetes', 'app_engine', 'cloud_functions'],
            severity='high',
            cis_benchmark='1.4',
            nist_framework='IA-5',
            pci_dss='8.2.4',
            implementation_guidance='Implement automated key rotation using Cloud Scheduler and Secret Manager',
            validation_query='SELECT email, created_at, DATE_DIFF(CURRENT_DATE(), CAST(created_at AS DATE), DAY) as days_old FROM `security_insights.iam_accounts` WHERE account_type = "serviceAccount" AND DATE_DIFF(CURRENT_DATE(), CAST(created_at AS DATE), DAY) > 90'
        )

        controls['IAM-003'] = SecurityControl(
            id='IAM-003',
            name='MFA for Admin Accounts',
            category=ControlCategory.IDENTITY,
            control_type=ControlType.PREVENTIVE,
            description='Require multi-factor authentication for all privileged accounts',
            applies_to=['all'],
            severity='critical',
            cis_benchmark='1.1',
            nist_framework='IA-2',
            pci_dss='8.3',
            hipaa='164.312(d)',
            implementation_guidance='Enable 2FA in Google Workspace and enforce via org policy',
            validation_query='SELECT email, role FROM `security_insights.iam_accounts` WHERE role IN ("roles/owner", "roles/editor") AND account_type = "user"'
        )

        # Network Security Controls
        controls['NET-001'] = SecurityControl(
            id='NET-001',
            name='Private Google Access',
            category=ControlCategory.NETWORK,
            control_type=ControlType.PREVENTIVE,
            description='Enable Private Google Access for VPC subnets to access Google APIs without external IPs',
            applies_to=['compute', 'kubernetes', 'cloud_functions'],
            severity='high',
            cis_benchmark='3.2',
            nist_framework='SC-7',
            implementation_guidance='Enable Private Google Access on all subnets, remove external IPs where possible',
            validation_query='SELECT name, network FROM `security_insights.networks` WHERE name LIKE "%subnet%"'
        )

        controls['NET-002'] = SecurityControl(
            id='NET-002',
            name='Firewall Rule Review',
            category=ControlCategory.NETWORK,
            control_type=ControlType.DETECTIVE,
            description='Regularly review and minimize firewall rules, especially 0.0.0.0/0 allow rules',
            applies_to=['compute', 'kubernetes'],
            severity='critical',
            cis_benchmark='3.6',
            nist_framework='SC-7',
            pci_dss='1.3',
            implementation_guidance='Audit firewall rules monthly, remove unused rules, restrict source ranges',
            validation_query='SELECT name, direction, source_ranges, allowed_ports FROM `security_insights.firewall_rules` WHERE source_ranges = "0.0.0.0/0" AND direction = "INGRESS"'
        )

        controls['NET-003'] = SecurityControl(
            id='NET-003',
            name='VPC Flow Logs',
            category=ControlCategory.NETWORK,
            control_type=ControlType.DETECTIVE,
            description='Enable VPC Flow Logs for network traffic analysis and threat detection',
            applies_to=['compute', 'kubernetes'],
            severity='medium',
            cis_benchmark='3.8',
            nist_framework='AU-2',
            pci_dss='10.1',
            hipaa='164.312(b)',
            implementation_guidance='Enable flow logs on all subnets with appropriate sampling rate',
            validation_query='SELECT name, network FROM `security_insights.networks`'
        )

        # Data Protection Controls
        controls['DATA-001'] = SecurityControl(
            id='DATA-001',
            name='Encryption at Rest (CMEK)',
            category=ControlCategory.DATA,
            control_type=ControlType.PREVENTIVE,
            description='Use customer-managed encryption keys (CMEK) for sensitive data',
            applies_to=['storage', 'bigquery', 'cloudsql', 'compute'],
            severity='high',
            cis_benchmark='5.1',
            nist_framework='SC-28',
            pci_dss='3.4',
            hipaa='164.312(a)(2)(iv)',
            gdpr='Article 32',
            implementation_guidance='Create Cloud KMS keys and configure services to use CMEK',
            validation_query='SELECT name, location, encryption_type FROM `security_insights.storage_buckets` WHERE encryption_type IS NULL OR encryption_type != "CMEK"'
        )

        controls['DATA-002'] = SecurityControl(
            id='DATA-002',
            name='Prevent Public Data Access',
            category=ControlCategory.DATA,
            control_type=ControlType.PREVENTIVE,
            description='Ensure storage buckets and datasets are not publicly accessible',
            applies_to=['storage', 'bigquery'],
            severity='critical',
            cis_benchmark='5.2',
            nist_framework='AC-3',
            pci_dss='7.2.1',
            hipaa='164.308(a)(3)',
            gdpr='Article 32',
            implementation_guidance='Use org policy to prevent public access, audit IAM bindings',
            validation_query='SELECT name, location, public_access_prevention FROM `security_insights.storage_buckets` WHERE public_access_prevention != "enforced"'
        )

        controls['DATA-003'] = SecurityControl(
            id='DATA-003',
            name='Data Loss Prevention (DLP)',
            category=ControlCategory.DATA,
            control_type=ControlType.DETECTIVE,
            description='Implement DLP scanning for PII, PHI, and sensitive data',
            applies_to=['storage', 'bigquery', 'cloudsql'],
            severity='high',
            nist_framework='SI-4',
            pci_dss='3.1',
            hipaa='164.308(a)(1)(ii)(A)',
            gdpr='Article 32',
            implementation_guidance='Configure Cloud DLP with appropriate detection templates',
            validation_query='SELECT name, category, severity FROM `security_insights.security_findings` WHERE category LIKE "%DLP%" AND state = "ACTIVE"'
        )

        # Logging & Monitoring Controls
        controls['LOG-001'] = SecurityControl(
            id='LOG-001',
            name='Cloud Audit Logging',
            category=ControlCategory.LOGGING,
            control_type=ControlType.DETECTIVE,
            description='Enable Admin Activity and Data Access audit logs for all services',
            applies_to=['all'],
            severity='critical',
            cis_benchmark='2.1',
            nist_framework='AU-2',
            pci_dss='10.1',
            hipaa='164.308(a)(1)(ii)(D)',
            sox='Sarbanes-Oxley 404',
            implementation_guidance='Enable all audit log types via org policy or project settings',
            validation_query='SELECT project_id FROM `security_insights.security_findings` WHERE category = "AUDIT_LOGGING" AND severity IN ("HIGH", "CRITICAL")'
        )

        controls['LOG-002'] = SecurityControl(
            id='LOG-002',
            name='Log Retention Policy',
            category=ControlCategory.LOGGING,
            control_type=ControlType.PREVENTIVE,
            description='Retain logs for minimum 1 year, preferably 7 years for compliance',
            applies_to=['all'],
            severity='high',
            cis_benchmark='2.2',
            nist_framework='AU-11',
            pci_dss='10.7',
            hipaa='164.308(a)(1)(ii)(D)',
            sox='Sarbanes-Oxley 404',
            implementation_guidance='Configure log sinks to BigQuery or Cloud Storage with lifecycle policies',
            validation_query='SELECT project_id FROM `security_insights.security_findings` WHERE category = "LOG_RETENTION"'
        )

        controls['LOG-003'] = SecurityControl(
            id='LOG-003',
            name='Security Monitoring Alerts',
            category=ControlCategory.LOGGING,
            control_type=ControlType.DETECTIVE,
            description='Configure alerts for security-critical events and anomalies',
            applies_to=['all'],
            severity='high',
            cis_benchmark='2.3, 2.4',
            nist_framework='SI-4',
            pci_dss='10.6',
            implementation_guidance='Create log-based metrics and alerting policies in Cloud Monitoring',
            validation_query='SELECT project_id FROM `security_insights.security_findings` WHERE category = "MONITORING"'
        )

        # Configuration Management Controls
        controls['CFG-001'] = SecurityControl(
            id='CFG-001',
            name='Organization Policies',
            category=ControlCategory.CONFIGURATION,
            control_type=ControlType.PREVENTIVE,
            description='Enforce security policies at the organization level',
            applies_to=['all'],
            severity='critical',
            cis_benchmark='1.10, 1.11',
            nist_framework='CM-7',
            implementation_guidance='Enable org policies: disable service account key creation, require OS Login, enforce uniform bucket-level access',
            validation_query='SELECT project_id FROM `security_insights.security_findings` WHERE category = "ORG_POLICY"'
        )

        controls['CFG-002'] = SecurityControl(
            id='CFG-002',
            name='Automatic OS Patching',
            category=ControlCategory.CONFIGURATION,
            control_type=ControlType.PREVENTIVE,
            description='Enable automatic OS patch management for compute instances',
            applies_to=['compute', 'kubernetes'],
            severity='high',
            cis_benchmark='4.1',
            nist_framework='SI-2',
            pci_dss='6.2',
            implementation_guidance='Configure OS Patch Management or use managed instance groups with auto-update',
            validation_query='SELECT name, status, machine_type FROM `security_insights.compute_instances`'
        )

        # Encryption Controls
        controls['ENC-001'] = SecurityControl(
            id='ENC-001',
            name='TLS 1.2+ for All Traffic',
            category=ControlCategory.ENCRYPTION,
            control_type=ControlType.PREVENTIVE,
            description='Enforce TLS 1.2 or higher for all network communication',
            applies_to=['load_balancer', 'app_engine', 'cloud_run'],
            severity='high',
            cis_benchmark='3.9',
            nist_framework='SC-8',
            pci_dss='4.1',
            hipaa='164.312(e)(1)',
            implementation_guidance='Configure minimum TLS version in load balancer and application settings',
            validation_query='SELECT name, category FROM `security_insights.security_findings` WHERE category = "TLS_VERSION"'
        )

        # Compliance Controls
        controls['CMP-001'] = SecurityControl(
            id='CMP-001',
            name='Security Command Center',
            category=ControlCategory.COMPLIANCE,
            control_type=ControlType.DETECTIVE,
            description='Enable Security Command Center for continuous compliance monitoring',
            applies_to=['all'],
            severity='high',
            nist_framework='CA-7',
            pci_dss='11.5',
            implementation_guidance='Enable SCC Premium tier for comprehensive security posture management',
            validation_query='SELECT name, category, severity FROM `security_insights.security_findings` WHERE state = "ACTIVE" AND severity IN ("HIGH", "CRITICAL")'
        )

        return controls

    def get_controls_for_service(self, service_type: str) -> List[SecurityControl]:
        """Get all applicable controls for a service type"""
        applicable_controls = []
        for control in self.controls.values():
            if 'all' in control.applies_to or service_type in control.applies_to:
                applicable_controls.append(control)
        return sorted(applicable_controls, key=lambda x: (
            0 if x.severity == 'critical' else 1 if x.severity == 'high' else 2
        ))

    def get_controls_by_category(self, category: ControlCategory) -> List[SecurityControl]:
        """Get all controls in a specific category"""
        return [c for c in self.controls.values() if c.category == category]

    def get_controls_by_framework(self, framework: str) -> List[SecurityControl]:
        """Get controls mapped to a compliance framework"""
        framework_map = {
            'cis': 'cis_benchmark',
            'nist': 'nist_framework',
            'pci': 'pci_dss',
            'hipaa': 'hipaa',
            'sox': 'sox',
            'gdpr': 'gdpr'
        }

        attr = framework_map.get(framework.lower())
        if not attr:
            return []

        return [c for c in self.controls.values() if getattr(c, attr, None) is not None]

    def get_control_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the controls inventory"""
        return {
            'total_controls': len(self.controls),
            'by_category': {
                cat.value: len(self.get_controls_by_category(cat))
                for cat in ControlCategory
            },
            'by_severity': {
                severity: len([c for c in self.controls.values() if c.severity == severity])
                for severity in ['critical', 'high', 'medium', 'low']
            },
            'by_type': {
                ctype.value: len([c for c in self.controls.values() if c.control_type == ctype])
                for ctype in ControlType
            },
            'coverage': {
                'cis_benchmark': len(self.get_controls_by_framework('cis')),
                'nist': len(self.get_controls_by_framework('nist')),
                'pci_dss': len(self.get_controls_by_framework('pci')),
                'hipaa': len(self.get_controls_by_framework('hipaa')),
                'sox': len(self.get_controls_by_framework('sox')),
                'gdpr': len(self.get_controls_by_framework('gdpr'))
            }
        }


# Initialize singleton instance
_inventory = None

def get_security_controls_inventory() -> str:
    """
    Get the security controls inventory as JSON

    Returns:
        JSON string with security controls inventory
    """
    import json
    from dataclasses import asdict

    global _inventory
    if _inventory is None:
        _inventory = SecurityControlsInventory()

    # Serialize to dict
    controls_dict = {
        'total_controls': len(_inventory.controls),
        'controls': [asdict(control) for control in _inventory.controls.values()],
        'categories': list(set(c.category.value for c in _inventory.controls.values()))
    }

    return json.dumps(controls_dict, indent=2, default=str)
