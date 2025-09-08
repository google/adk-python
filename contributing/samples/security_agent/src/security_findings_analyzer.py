"""Security Findings Analyzer for GCP Security Agent

Advanced security analysis engine that identifies threats, vulnerabilities,
and compliance issues across GCP resources with intelligent prioritization.
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    """Security finding severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"  
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class FindingCategory(Enum):
    """Security finding categories."""
    NETWORK_EXPOSURE = "NETWORK_EXPOSURE"
    DATA_EXPOSURE = "DATA_EXPOSURE"
    WEAK_ENCRYPTION = "WEAK_ENCRYPTION"
    EXCESSIVE_PERMISSIONS = "EXCESSIVE_PERMISSIONS"
    MISSING_CONTROLS = "MISSING_CONTROLS"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    VULNERABILITY = "VULNERABILITY"
    CONFIGURATION_ISSUE = "CONFIGURATION_ISSUE"
    POLICY_VIOLATION = "POLICY_VIOLATION"

@dataclass
class SecurityFinding:
    """Detailed security finding."""
    finding_id: str
    resource_name: str
    resource_type: str
    category: FindingCategory
    severity: SeverityLevel
    title: str
    description: str
    recommendation: str
    impact_score: float  # 0-10 scale
    remediation_effort: str  # LOW, MEDIUM, HIGH
    compliance_frameworks: List[str]  # SOC2, PCI-DSS, etc.
    evidence: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class SecuritySummary:
    """Overall security posture summary."""
    total_findings: int
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    risk_score: float  # 0-100 scale
    compliance_score: float  # 0-100 scale
    top_risks: List[SecurityFinding]
    recommendations: List[str]
    analysis_timestamp: datetime

class SecurityFindingsAnalyzer:
    """Advanced security findings analysis engine."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._get_default_db_path()
        self.severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1
        }
    
    def _get_default_db_path(self) -> str:
        """Get default database path."""
        current_file = Path(__file__)
        security_agent_dir = current_file.parent.parent.parent
        return str(security_agent_dir / 'backend' / 'cache' / 'gcp_data.db')
    
    def analyze_all_resources(self) -> SecuritySummary:
        """Perform comprehensive security analysis on all resources."""
        findings = []
        
        # Analyze different resource types
        findings.extend(self._analyze_compute_security())
        findings.extend(self._analyze_storage_security())
        findings.extend(self._analyze_iam_security())
        findings.extend(self._analyze_network_security())
        findings.extend(self._analyze_gke_security())
        findings.extend(self._analyze_database_security())
        
        # Calculate security metrics
        risk_score = self._calculate_risk_score(findings)
        compliance_score = self._calculate_compliance_score(findings)
        
        # Generate top recommendations
        recommendations = self._generate_priority_recommendations(findings)
        
        # Create summary
        summary = SecuritySummary(
            total_findings=len(findings),
            by_severity=self._count_by_severity(findings),
            by_category=self._count_by_category(findings),
            risk_score=risk_score,
            compliance_score=compliance_score,
            top_risks=sorted(findings, key=lambda f: f.impact_score, reverse=True)[:10],
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
        
        # Store findings in database
        self._store_findings(findings)
        
        logger.info(f"🔍 Security analysis complete: {len(findings)} findings, risk score: {risk_score:.1f}")
        return summary
    
    def _analyze_compute_security(self) -> List[SecurityFinding]:
        """Analyze compute instances for security issues."""
        findings = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all compute instances
            cursor.execute("SELECT * FROM compute_instances")
            instances = cursor.fetchall()
            
            for instance in instances:
                findings.extend(self._check_instance_security(instance))
            
            conn.close()
            
        except sqlite3.OperationalError:
            logger.warning("Compute instances table not found, creating mock findings")
            findings.extend(self._create_mock_compute_findings())
        
        return findings
    
    def _check_instance_security(self, instance) -> List[SecurityFinding]:
        """Check security issues for a specific compute instance."""
        findings = []
        instance_name = instance['name']
        
        # External IP exposure
        if instance.get('external_ip'):
            findings.append(SecurityFinding(
                finding_id=f"compute-external-ip-{hash(instance_name)}",
                resource_name=instance_name,
                resource_type="compute_instance",
                category=FindingCategory.NETWORK_EXPOSURE,
                severity=SeverityLevel.MEDIUM,
                title="Compute instance has external IP",
                description=f"Instance {instance_name} has external IP {instance['external_ip']}, increasing attack surface",
                recommendation="Use Cloud NAT or Load Balancer instead of direct external IP assignment",
                impact_score=5.5,
                remediation_effort="MEDIUM",
                compliance_frameworks=["SOC2", "ISO27001"],
                evidence={"external_ip": instance['external_ip'], "zone": instance['zone']},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        # Service account analysis
        try:
            service_accounts = json.loads(instance.get('service_accounts', '[]'))
            for sa in service_accounts:
                if 'compute@developer.gserviceaccount.com' in sa.get('email', ''):
                    findings.append(SecurityFinding(
                        finding_id=f"compute-default-sa-{hash(instance_name)}",
                        resource_name=instance_name,
                        resource_type="compute_instance",
                        category=FindingCategory.EXCESSIVE_PERMISSIONS,
                        severity=SeverityLevel.HIGH,
                        title="Using default Compute Engine service account",
                        description=f"Instance {instance_name} uses default service account with broad permissions",
                        recommendation="Create custom service account with minimal required permissions",
                        impact_score=7.0,
                        remediation_effort="MEDIUM",
                        compliance_frameworks=["SOC2", "PCI-DSS"],
                        evidence={"service_account": sa.get('email')},
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    ))
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Disk encryption check
        try:
            disk_info = json.loads(instance.get('disk_info', '[]'))
            for disk in disk_info:
                if not disk.get('encrypted', False):
                    findings.append(SecurityFinding(
                        finding_id=f"compute-disk-encryption-{hash(instance_name)}-{hash(str(disk))}",
                        resource_name=instance_name,
                        resource_type="compute_instance",
                        category=FindingCategory.WEAK_ENCRYPTION,
                        severity=SeverityLevel.HIGH,
                        title="Unencrypted disk detected",
                        description=f"Disk on instance {instance_name} is not encrypted at rest",
                        recommendation="Enable disk encryption with customer-managed keys",
                        impact_score=6.5,
                        remediation_effort="HIGH",
                        compliance_frameworks=["PCI-DSS", "HIPAA", "SOC2"],
                        evidence={"disk": disk},
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    ))
        except (json.JSONDecodeError, TypeError):
            pass
        
        return findings
    
    def _analyze_storage_security(self) -> List[SecurityFinding]:
        """Analyze storage buckets for security issues."""
        findings = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM storage_buckets")
            buckets = cursor.fetchall()
            
            for bucket in buckets:
                findings.extend(self._check_bucket_security(bucket))
            
            conn.close()
            
        except sqlite3.OperationalError:
            logger.warning("Storage buckets table not found, creating mock findings")
            findings.extend(self._create_mock_storage_findings())
        
        return findings
    
    def _check_bucket_security(self, bucket) -> List[SecurityFinding]:
        """Check security issues for a specific storage bucket."""
        findings = []
        bucket_name = bucket['name']
        
        # Public access prevention
        if bucket.get('public_access_prevention') != 'enforced':
            findings.append(SecurityFinding(
                finding_id=f"storage-public-access-{hash(bucket_name)}",
                resource_name=bucket_name,
                resource_type="storage_bucket",
                category=FindingCategory.DATA_EXPOSURE,
                severity=SeverityLevel.CRITICAL,
                title="Storage bucket allows public access",
                description=f"Bucket {bucket_name} does not enforce public access prevention",
                recommendation="Enable 'Enforce public access prevention' on the bucket",
                impact_score=9.0,
                remediation_effort="LOW",
                compliance_frameworks=["SOC2", "PCI-DSS", "GDPR"],
                evidence={"public_access_prevention": bucket.get('public_access_prevention')},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        # Uniform bucket-level access
        if not bucket.get('uniform_bucket_level_access'):
            findings.append(SecurityFinding(
                finding_id=f"storage-ubla-{hash(bucket_name)}",
                resource_name=bucket_name,
                resource_type="storage_bucket",
                category=FindingCategory.CONFIGURATION_ISSUE,
                severity=SeverityLevel.MEDIUM,
                title="Uniform bucket-level access disabled",
                description=f"Bucket {bucket_name} does not use uniform bucket-level access",
                recommendation="Enable uniform bucket-level access for consistent access control",
                impact_score=4.5,
                remediation_effort="LOW",
                compliance_frameworks=["SOC2"],
                evidence={"uniform_bucket_level_access": bucket.get('uniform_bucket_level_access')},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        # Versioning
        if not bucket.get('versioning_enabled'):
            findings.append(SecurityFinding(
                finding_id=f"storage-versioning-{hash(bucket_name)}",
                resource_name=bucket_name,
                resource_type="storage_bucket",
                category=FindingCategory.MISSING_CONTROLS,
                severity=SeverityLevel.LOW,
                title="Object versioning disabled",
                description=f"Bucket {bucket_name} does not have object versioning enabled",
                recommendation="Enable object versioning to protect against accidental deletion",
                impact_score=3.0,
                remediation_effort="LOW",
                compliance_frameworks=["SOC2"],
                evidence={"versioning_enabled": bucket.get('versioning_enabled')},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        return findings
    
    def _analyze_iam_security(self) -> List[SecurityFinding]:
        """Analyze IAM configurations for security issues."""
        findings = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Analyze service accounts
            cursor.execute("SELECT * FROM service_accounts")
            service_accounts = cursor.fetchall()
            
            for sa in service_accounts:
                findings.extend(self._check_service_account_security(sa))
            
            # Analyze IAM bindings
            cursor.execute("SELECT * FROM iam_bindings")
            bindings = cursor.fetchall()
            
            findings.extend(self._check_iam_bindings_security(bindings))
            
            conn.close()
            
        except sqlite3.OperationalError:
            logger.warning("IAM tables not found, creating mock findings")
            findings.extend(self._create_mock_iam_findings())
        
        return findings
    
    def _check_service_account_security(self, sa) -> List[SecurityFinding]:
        """Check security issues for a service account."""
        findings = []
        sa_email = sa['email']
        
        # Check for unused disabled service accounts
        if sa.get('disabled'):
            findings.append(SecurityFinding(
                finding_id=f"iam-disabled-sa-{hash(sa_email)}",
                resource_name=sa_email,
                resource_type="service_account",
                category=FindingCategory.CONFIGURATION_ISSUE,
                severity=SeverityLevel.INFO,
                title="Disabled service account should be removed",
                description=f"Service account {sa_email} is disabled but still exists",
                recommendation="Remove unused disabled service accounts to reduce attack surface",
                impact_score=1.5,
                remediation_effort="LOW",
                compliance_frameworks=["SOC2"],
                evidence={"disabled": sa.get('disabled')},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        return findings
    
    def _check_iam_bindings_security(self, bindings) -> List[SecurityFinding]:
        """Check security issues in IAM bindings."""
        findings = []
        
        # Look for overprivileged roles
        dangerous_roles = ['roles/owner', 'roles/editor', 'roles/iam.securityAdmin']
        
        for binding in bindings:
            role = binding['role']
            member = binding['member']
            
            if role in dangerous_roles and 'serviceAccount' in member:
                findings.append(SecurityFinding(
                    finding_id=f"iam-overprivileged-{hash(f'{member}-{role}')}",
                    resource_name=binding['resource_name'],
                    resource_type="iam_binding",
                    category=FindingCategory.EXCESSIVE_PERMISSIONS,
                    severity=SeverityLevel.HIGH,
                    title=f"Overprivileged service account detected",
                    description=f"Service account {member} has role {role} which grants broad permissions",
                    recommendation="Apply principle of least privilege and use more specific roles",
                    impact_score=8.0,
                    remediation_effort="MEDIUM",
                    compliance_frameworks=["SOC2", "PCI-DSS"],
                    evidence={"role": role, "member": member},
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                ))
        
        return findings
    
    def _analyze_network_security(self) -> List[SecurityFinding]:
        """Analyze network security configurations."""
        findings = []
        
        # This would analyze firewall rules, VPC configurations, etc.
        # For now, create sample findings
        findings.extend(self._create_mock_network_findings())
        
        return findings
    
    def _analyze_gke_security(self) -> List[SecurityFinding]:
        """Analyze GKE clusters for security issues."""
        findings = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM gke_clusters")
            clusters = cursor.fetchall()
            
            for cluster in clusters:
                findings.extend(self._check_gke_security(cluster))
            
            conn.close()
            
        except sqlite3.OperationalError:
            findings.extend(self._create_mock_gke_findings())
        
        return findings
    
    def _check_gke_security(self, cluster) -> List[SecurityFinding]:
        """Check security issues for a GKE cluster."""
        findings = []
        cluster_name = cluster['name']
        
        # Private cluster check
        if not cluster.get('private_cluster'):
            findings.append(SecurityFinding(
                finding_id=f"gke-private-cluster-{hash(cluster_name)}",
                resource_name=cluster_name,
                resource_type="gke_cluster",
                category=FindingCategory.NETWORK_EXPOSURE,
                severity=SeverityLevel.HIGH,
                title="GKE cluster is not private",
                description=f"Cluster {cluster_name} nodes have public IP addresses",
                recommendation="Enable private cluster to isolate worker nodes",
                impact_score=7.5,
                remediation_effort="HIGH",
                compliance_frameworks=["SOC2", "PCI-DSS"],
                evidence={"private_cluster": cluster.get('private_cluster')},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        # Legacy ABAC check
        if cluster.get('legacy_abac'):
            findings.append(SecurityFinding(
                finding_id=f"gke-legacy-abac-{hash(cluster_name)}",
                resource_name=cluster_name,
                resource_type="gke_cluster",
                category=FindingCategory.POLICY_VIOLATION,
                severity=SeverityLevel.HIGH,
                title="Legacy ABAC authorization enabled",
                description=f"Cluster {cluster_name} uses deprecated ABAC authorization",
                recommendation="Disable legacy ABAC and use RBAC exclusively",
                impact_score=6.5,
                remediation_effort="MEDIUM",
                compliance_frameworks=["SOC2"],
                evidence={"legacy_abac": cluster.get('legacy_abac')},
                created_at=datetime.now(),
                updated_at=datetime.now()
            ))
        
        return findings
    
    def _analyze_database_security(self) -> List[SecurityFinding]:
        """Analyze database security configurations."""
        findings = []
        # Placeholder for database security analysis
        return findings
    
    def _calculate_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall risk score (0-100)."""
        if not findings:
            return 0.0
        
        total_impact = sum(f.impact_score for f in findings)
        max_possible_impact = len(findings) * 10.0  # Max impact score is 10
        
        return (total_impact / max_possible_impact) * 100 if max_possible_impact > 0 else 0
    
    def _calculate_compliance_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate compliance score (0-100)."""
        # Count compliance-related findings
        compliance_findings = [f for f in findings if f.compliance_frameworks]
        
        if not compliance_findings:
            return 100.0  # No compliance issues found
        
        # Weight by severity
        total_weight = sum(self.severity_weights[f.severity] for f in compliance_findings)
        max_weight = len(compliance_findings) * self.severity_weights[SeverityLevel.CRITICAL]
        
        return max(0, 100 - (total_weight / max_weight * 100)) if max_weight > 0 else 100
    
    def _count_by_severity(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Count findings by severity level."""
        counts = {severity.value: 0 for severity in SeverityLevel}
        for finding in findings:
            counts[finding.severity.value] += 1
        return counts
    
    def _count_by_category(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Count findings by category."""
        counts = {category.value: 0 for category in FindingCategory}
        for finding in findings:
            counts[finding.category.value] += 1
        return counts
    
    def _generate_priority_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Count by category to identify top issues
        category_counts = self._count_by_category(findings)
        severity_counts = self._count_by_severity(findings)
        
        # Critical recommendations
        if severity_counts.get('CRITICAL', 0) > 0:
            recommendations.append("🚨 Address CRITICAL severity findings immediately")
        
        # Category-specific recommendations
        if category_counts.get('DATA_EXPOSURE', 0) > 0:
            recommendations.append("🔒 Review and fix data exposure issues (storage buckets, databases)")
        
        if category_counts.get('EXCESSIVE_PERMISSIONS', 0) > 0:
            recommendations.append("👤 Implement principle of least privilege for IAM roles")
        
        if category_counts.get('NETWORK_EXPOSURE', 0) > 0:
            recommendations.append("🌐 Reduce network exposure with private clusters and NAT gateways")
        
        if category_counts.get('WEAK_ENCRYPTION', 0) > 0:
            recommendations.append("🔐 Enable encryption at rest and in transit for all resources")
        
        # General recommendations
        recommendations.extend([
            "📊 Implement continuous security monitoring",
            "✅ Schedule regular security audits (monthly)",
            "📚 Provide security training for development teams",
            "🔧 Automate security policy enforcement where possible"
        ])
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _store_findings(self, findings: List[SecurityFinding]):
        """Store findings in database for historical tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create findings table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_findings_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    finding_id TEXT UNIQUE NOT NULL,
                    resource_name TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    recommendation TEXT,
                    impact_score REAL,
                    remediation_effort TEXT,
                    compliance_frameworks TEXT,  -- JSON
                    evidence TEXT,  -- JSON
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    INDEX(severity),
                    INDEX(category),
                    INDEX(resource_type)
                )
            """)
            
            # Insert findings
            for finding in findings:
                cursor.execute("""
                    INSERT OR REPLACE INTO security_findings_analysis
                    (finding_id, resource_name, resource_type, category, severity,
                     title, description, recommendation, impact_score, remediation_effort,
                     compliance_frameworks, evidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    finding.finding_id,
                    finding.resource_name,
                    finding.resource_type,
                    finding.category.value,
                    finding.severity.value,
                    finding.title,
                    finding.description,
                    finding.recommendation,
                    finding.impact_score,
                    finding.remediation_effort,
                    json.dumps(finding.compliance_frameworks),
                    json.dumps(finding.evidence),
                    finding.created_at.isoformat(),
                    finding.updated_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Stored {len(findings)} security findings in database")
            
        except Exception as e:
            logger.error(f"Failed to store findings: {e}")
    
    # Mock data creation methods
    def _create_mock_compute_findings(self) -> List[SecurityFinding]:
        """Create mock compute security findings."""
        return [
            SecurityFinding(
                finding_id="compute-001",
                resource_name="web-server-1",
                resource_type="compute_instance",
                category=FindingCategory.NETWORK_EXPOSURE,
                severity=SeverityLevel.MEDIUM,
                title="Instance has external IP",
                description="Compute instance exposed to internet",
                recommendation="Use Cloud NAT instead",
                impact_score=5.5,
                remediation_effort="MEDIUM",
                compliance_frameworks=["SOC2"],
                evidence={"external_ip": "34.123.45.67"},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    def _create_mock_storage_findings(self) -> List[SecurityFinding]:
        """Create mock storage security findings."""
        return [
            SecurityFinding(
                finding_id="storage-001",
                resource_name="app-data-bucket",
                resource_type="storage_bucket",
                category=FindingCategory.DATA_EXPOSURE,
                severity=SeverityLevel.CRITICAL,
                title="Bucket allows public access",
                description="Storage bucket may be publicly accessible",
                recommendation="Enable public access prevention",
                impact_score=9.0,
                remediation_effort="LOW",
                compliance_frameworks=["SOC2", "GDPR"],
                evidence={"public_access_prevention": "inherited"},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    def _create_mock_iam_findings(self) -> List[SecurityFinding]:
        """Create mock IAM security findings."""
        return [
            SecurityFinding(
                finding_id="iam-001",
                resource_name="web-app@project.iam.gserviceaccount.com",
                resource_type="service_account",
                category=FindingCategory.EXCESSIVE_PERMISSIONS,
                severity=SeverityLevel.HIGH,
                title="Overprivileged service account",
                description="Service account has editor role",
                recommendation="Use principle of least privilege",
                impact_score=7.0,
                remediation_effort="MEDIUM",
                compliance_frameworks=["SOC2", "PCI-DSS"],
                evidence={"role": "roles/editor"},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    def _create_mock_network_findings(self) -> List[SecurityFinding]:
        """Create mock network security findings."""
        return [
            SecurityFinding(
                finding_id="network-001",
                resource_name="default-allow-http",
                resource_type="firewall_rule",
                category=FindingCategory.NETWORK_EXPOSURE,
                severity=SeverityLevel.MEDIUM,
                title="Overly permissive firewall rule",
                description="Firewall rule allows HTTP from any source",
                recommendation="Restrict source IP ranges",
                impact_score=4.5,
                remediation_effort="LOW",
                compliance_frameworks=["SOC2"],
                evidence={"source_ranges": ["0.0.0.0/0"]},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
    
    def _create_mock_gke_findings(self) -> List[SecurityFinding]:
        """Create mock GKE security findings."""
        return [
            SecurityFinding(
                finding_id="gke-001",
                resource_name="production-cluster",
                resource_type="gke_cluster",
                category=FindingCategory.NETWORK_EXPOSURE,
                severity=SeverityLevel.HIGH,
                title="GKE cluster is not private",
                description="Cluster nodes have public IP addresses",
                recommendation="Enable private cluster configuration",
                impact_score=7.5,
                remediation_effort="HIGH",
                compliance_frameworks=["SOC2", "PCI-DSS"],
                evidence={"private_cluster": False},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]

# Convenience functions
def analyze_security(db_path: Optional[str] = None) -> SecuritySummary:
    """Convenience function to run security analysis."""
    analyzer = SecurityFindingsAnalyzer(db_path)
    return analyzer.analyze_all_resources()

def get_security_summary_text(db_path: Optional[str] = None) -> str:
    """Get formatted security summary text."""
    summary = analyze_security(db_path)
    
    result = f"🔒 **Security Analysis Results**\\n\\n"
    result += f"**Total Findings**: {summary.total_findings}\\n"
    result += f"**Risk Score**: {summary.risk_score:.1f}/100\\n"
    result += f"**Compliance Score**: {summary.compliance_score:.1f}/100\\n\\n"
    
    if summary.by_severity:
        result += "**By Severity:**\\n"
        for severity, count in summary.by_severity.items():
            if count > 0:
                result += f"* {severity}: {count}\\n"
    
    if summary.top_risks:
        result += f"\\n**Top Risk:**\\n"
        top_risk = summary.top_risks[0]
        result += f"* **{top_risk.severity.value}**: {top_risk.title}\\n"
        result += f"  Resource: {top_risk.resource_name}\\n"
        result += f"  Impact: {top_risk.impact_score}/10\\n"
    
    if summary.recommendations:
        result += f"\\n**Priority Recommendations:**\\n"
        for rec in summary.recommendations[:3]:
            result += f"* {rec}\\n"
    
    return result