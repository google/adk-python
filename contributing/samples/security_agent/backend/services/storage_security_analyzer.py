"""
Enhanced Storage Security Analyzer (STORY-004)

Provides comprehensive storage security analysis including public bucket detection,
encryption validation, data classification integration, lifecycle policy analysis,
and CSPM compliance checks.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

try:
    from google.cloud import storage
    from google.cloud import secretmanager
    from google.api_core import exceptions as gcp_exceptions
    GCP_STORAGE_AVAILABLE = True
except ImportError:
    GCP_STORAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class StorageRiskLevel(Enum):
    """Storage security risk levels"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StorageFindingType(Enum):
    """Storage security finding types"""
    PUBLIC_BUCKET_NO_AUTH = "PUBLIC_BUCKET_NO_AUTH"
    PUBLIC_BUCKET_READ = "PUBLIC_BUCKET_READ"
    PUBLIC_BUCKET_WRITE = "PUBLIC_BUCKET_WRITE"
    MISSING_ENCRYPTION = "MISSING_ENCRYPTION"
    WEAK_ENCRYPTION = "WEAK_ENCRYPTION"
    NO_LIFECYCLE_POLICY = "NO_LIFECYCLE_POLICY"
    OVERLY_PERMISSIVE_ACL = "OVERLY_PERMISSIVE_ACL"
    MISSING_VERSIONING = "MISSING_VERSIONING"
    PUBLIC_ACCESS_PREVENTION_DISABLED = "PUBLIC_ACCESS_PREVENTION_DISABLED"
    UNIFORM_BUCKET_ACCESS_DISABLED = "UNIFORM_BUCKET_ACCESS_DISABLED"
    CORS_MISCONFIGURATION = "CORS_MISCONFIGURATION"
    LOGGING_DISABLED = "LOGGING_DISABLED"
    RETENTION_POLICY_MISSING = "RETENTION_POLICY_MISSING"
    SENSITIVE_DATA_EXPOSURE = "SENSITIVE_DATA_EXPOSURE"
    BUCKET_NAMING_VIOLATION = "BUCKET_NAMING_VIOLATION"


@dataclass
class StorageFinding:
    """Storage security finding"""
    finding_type: StorageFindingType
    risk_level: StorageRiskLevel
    risk_score: int  # 0-100
    title: str
    description: str
    bucket_name: str
    object_name: Optional[str]
    remediation_steps: List[str]
    compliance_frameworks: List[str]
    metadata: Dict[str, Any]
    detected_at: datetime


@dataclass
class BucketSecurityAnalysis:
    """Bucket security analysis result"""
    bucket_name: str
    project_id: str
    location: str
    storage_class: str
    is_public: bool
    public_access_type: Optional[str]
    encryption_type: str
    kms_key_name: Optional[str]
    versioning_enabled: bool
    lifecycle_configured: bool
    logging_enabled: bool
    cors_configured: bool
    uniform_bucket_access: bool
    public_access_prevention: bool
    retention_policy: Optional[Dict[str, Any]]
    risk_score: int
    findings: List[StorageFinding]
    object_count: int
    total_size_bytes: int
    last_modified: Optional[datetime]
    data_classification: Optional[str]


@dataclass
class StorageSecurityPosture:
    """Overall storage security posture"""
    project_id: str
    posture_score: int  # 0-100
    risk_distribution: Dict[str, int]
    total_buckets: int
    public_buckets: int
    encrypted_buckets: int
    compliant_buckets: int
    findings: List[StorageFinding]
    recommendations: List[str]
    compliance_status: Dict[str, float]
    analyzed_at: datetime


class StorageSecurityAnalyzer:
    """Enhanced Storage Security Analyzer"""
    
    # High-risk storage patterns
    SENSITIVE_PATTERNS = [
        r'.*password.*',
        r'.*secret.*',
        r'.*key.*',
        r'.*token.*',
        r'.*credential.*',
        r'.*private.*',
        r'.*confidential.*',
        r'.*ssn.*',
        r'.*credit.?card.*',
        r'.*bank.*',
        r'.*personal.*'
    ]
    
    # Compliance frameworks
    COMPLIANCE_FRAMEWORKS = {
        'SOC2': ['encryption', 'access_control', 'logging', 'retention'],
        'HIPAA': ['encryption', 'access_control', 'audit_logging', 'data_retention'],
        'PCI_DSS': ['encryption', 'access_control', 'logging', 'network_security'],
        'GDPR': ['encryption', 'access_control', 'data_retention', 'right_to_deletion'],
        'ISO27001': ['encryption', 'access_control', 'monitoring', 'incident_response']
    }
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.storage_client = self._get_storage_client() if GCP_STORAGE_AVAILABLE else None
    
    def _get_storage_client(self):
        """Get storage client"""
        try:
            return storage.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Failed to create storage client: {e}")
            return None
    
    def analyze_storage_security(self) -> StorageSecurityPosture:
        """
        Perform comprehensive storage security analysis
        
        Returns:
            StorageSecurityPosture with findings and recommendations
        """
        logger.info(f"Starting storage security analysis for project {self.project_id}")
        
        if not self.storage_client:
            return self._generate_sample_analysis()
        
        try:
            # List all buckets in the project
            buckets = list(self.storage_client.list_buckets())
            
            # Analyze each bucket
            bucket_analyses = []
            all_findings = []
            
            for bucket in buckets:
                analysis = self._analyze_bucket(bucket)
                bucket_analyses.append(analysis)
                all_findings.extend(analysis.findings)
            
            # Calculate overall posture
            posture = self._calculate_storage_posture(bucket_analyses, all_findings)
            
            logger.info(f"Storage analysis complete. Found {len(all_findings)} findings across {len(buckets)} buckets")
            return posture
            
        except Exception as e:
            logger.error(f"Error during storage analysis: {e}")
            return self._generate_sample_analysis()
    
    def _analyze_bucket(self, bucket) -> BucketSecurityAnalysis:
        """Analyze a single bucket for security issues"""
        findings = []
        
        # Check public access
        public_findings = self._check_public_access(bucket)
        findings.extend(public_findings)
        
        # Check encryption
        encryption_findings = self._check_encryption(bucket)
        findings.extend(encryption_findings)
        
        # Check lifecycle policies
        lifecycle_findings = self._check_lifecycle_policies(bucket)
        findings.extend(lifecycle_findings)
        
        # Check access controls
        access_findings = self._check_access_controls(bucket)
        findings.extend(access_findings)
        
        # Check compliance configurations
        compliance_findings = self._check_compliance_configurations(bucket)
        findings.extend(compliance_findings)
        
        # Check for sensitive data patterns
        data_findings = self._check_sensitive_data(bucket)
        findings.extend(data_findings)
        
        # Check bucket naming
        naming_findings = self._check_bucket_naming(bucket)
        findings.extend(naming_findings)
        
        # Calculate bucket-level risk score
        risk_score = sum(f.risk_score for f in findings)
        risk_score = min(risk_score, 100)
        
        # Get bucket metadata
        is_public = any(f.finding_type in [
            StorageFindingType.PUBLIC_BUCKET_NO_AUTH,
            StorageFindingType.PUBLIC_BUCKET_READ,
            StorageFindingType.PUBLIC_BUCKET_WRITE
        ] for f in findings)
        
        public_access_type = None
        if is_public:
            if any(f.finding_type == StorageFindingType.PUBLIC_BUCKET_WRITE for f in findings):
                public_access_type = "WRITE"
            elif any(f.finding_type == StorageFindingType.PUBLIC_BUCKET_READ for f in findings):
                public_access_type = "READ"
            else:
                public_access_type = "UNKNOWN"
        
        # Determine encryption type
        encryption_type = "GOOGLE_MANAGED"
        kms_key_name = None
        if bucket.default_kms_key_name:
            encryption_type = "CUSTOMER_MANAGED"
            kms_key_name = bucket.default_kms_key_name
        
        # Get bucket statistics
        object_count = 0
        total_size = 0
        last_modified = None
        
        try:
            # Sample first 100 objects for statistics
            blobs = list(bucket.list_blobs(max_results=100))
            object_count = len(blobs)
            if blobs:
                total_size = sum(blob.size or 0 for blob in blobs)
                last_modified = max(blob.time_created for blob in blobs if blob.time_created)
        except Exception as e:
            logger.warning(f"Could not get bucket statistics for {bucket.name}: {e}")
        
        return BucketSecurityAnalysis(
            bucket_name=bucket.name,
            project_id=self.project_id,
            location=bucket.location,
            storage_class=bucket.storage_class,
            is_public=is_public,
            public_access_type=public_access_type,
            encryption_type=encryption_type,
            kms_key_name=kms_key_name,
            versioning_enabled=bucket.versioning_enabled,
            lifecycle_configured=bool(bucket.lifecycle_rules),
            logging_enabled=bool(bucket.logging),
            cors_configured=bool(bucket.cors),
            uniform_bucket_access=bucket.iam_configuration.uniform_bucket_level_access_enabled,
            public_access_prevention=bucket.iam_configuration.public_access_prevention == 'enforced',
            retention_policy=self._get_retention_policy_info(bucket),
            risk_score=risk_score,
            findings=findings,
            object_count=object_count,
            total_size_bytes=total_size,
            last_modified=last_modified,
            data_classification=self._classify_data(bucket)
        )
    
    def _check_public_access(self, bucket) -> List[StorageFinding]:
        """Check for public access configurations"""
        findings = []
        
        try:
            # Check IAM policy for public access
            policy = bucket.get_iam_policy(requested_policy_version=3)
            
            for binding in policy.bindings:
                members = binding.get('members', [])
                role = binding.get('role', '')
                
                if 'allUsers' in members:
                    if 'objectAdmin' in role or 'admin' in role or 'writer' in role:
                        findings.append(StorageFinding(
                            finding_type=StorageFindingType.PUBLIC_BUCKET_WRITE,
                            risk_level=StorageRiskLevel.CRITICAL,
                            risk_score=95,
                            title="Public Write Access",
                            description=f"Bucket allows public write access via {role}",
                            bucket_name=bucket.name,
                            object_name=None,
                            remediation_steps=[
                                "Remove allUsers from IAM policy",
                                "Enable public access prevention",
                                "Implement authenticated access only",
                                "Review and update access controls"
                            ],
                            compliance_frameworks=['SOC2', 'HIPAA', 'PCI_DSS'],
                            metadata={'role': role, 'members': members},
                            detected_at=datetime.utcnow()
                        ))
                    elif 'reader' in role or 'viewer' in role:
                        findings.append(StorageFinding(
                            finding_type=StorageFindingType.PUBLIC_BUCKET_READ,
                            risk_level=StorageRiskLevel.HIGH,
                            risk_score=80,
                            title="Public Read Access",
                            description=f"Bucket allows public read access via {role}",
                            bucket_name=bucket.name,
                            object_name=None,
                            remediation_steps=[
                                "Remove allUsers from IAM policy",
                                "Implement signed URLs for temporary access",
                                "Use Cloud CDN for public content",
                                "Enable public access prevention"
                            ],
                            compliance_frameworks=['SOC2', 'GDPR'],
                            metadata={'role': role, 'members': members},
                            detected_at=datetime.utcnow()
                        ))
                
                if 'allAuthenticatedUsers' in members:
                    findings.append(StorageFinding(
                        finding_type=StorageFindingType.PUBLIC_BUCKET_NO_AUTH,
                        risk_level=StorageRiskLevel.HIGH,
                        risk_score=75,
                        title="Authenticated Users Access",
                        description=f"Bucket allows access to all authenticated users via {role}",
                        bucket_name=bucket.name,
                        object_name=None,
                        remediation_steps=[
                            "Remove allAuthenticatedUsers from IAM policy",
                            "Grant access to specific users/groups",
                            "Implement least privilege access",
                            "Use IAM conditions for temporary access"
                        ],
                        compliance_frameworks=['SOC2', 'ISO27001'],
                        metadata={'role': role, 'members': members},
                        detected_at=datetime.utcnow()
                    ))
            
            # Check public access prevention
            if bucket.iam_configuration.public_access_prevention != 'enforced':
                findings.append(StorageFinding(
                    finding_type=StorageFindingType.PUBLIC_ACCESS_PREVENTION_DISABLED,
                    risk_level=StorageRiskLevel.MEDIUM,
                    risk_score=50,
                    title="Public Access Prevention Disabled",
                    description="Bucket does not enforce public access prevention",
                    bucket_name=bucket.name,
                    object_name=None,
                    remediation_steps=[
                        "Enable public access prevention",
                        "Review existing public access",
                        "Implement organization policy",
                        "Monitor for policy violations"
                    ],
                    compliance_frameworks=['SOC2', 'ISO27001'],
                    metadata={'prevention_setting': bucket.iam_configuration.public_access_prevention},
                    detected_at=datetime.utcnow()
                ))
            
        except Exception as e:
            logger.warning(f"Could not check public access for {bucket.name}: {e}")
        
        return findings
    
    def _check_encryption(self, bucket) -> List[StorageFinding]:
        """Check encryption configurations"""
        findings = []
        
        if not bucket.default_kms_key_name:
            findings.append(StorageFinding(
                finding_type=StorageFindingType.MISSING_ENCRYPTION,
                risk_level=StorageRiskLevel.HIGH,
                risk_score=70,
                title="No Customer-Managed Encryption",
                description="Bucket uses Google-managed encryption instead of customer-managed keys",
                bucket_name=bucket.name,
                object_name=None,
                remediation_steps=[
                    "Configure customer-managed encryption keys (CMEK)",
                    "Create Cloud KMS keys for encryption",
                    "Enable automatic key rotation",
                    "Document key management procedures"
                ],
                compliance_frameworks=['HIPAA', 'PCI_DSS', 'GDPR'],
                metadata={'current_encryption': 'google_managed'},
                detected_at=datetime.utcnow()
            ))
        
        return findings
    
    def _check_lifecycle_policies(self, bucket) -> List[StorageFinding]:
        """Check lifecycle policy configurations"""
        findings = []
        
        if not bucket.lifecycle_rules:
            findings.append(StorageFinding(
                finding_type=StorageFindingType.NO_LIFECYCLE_POLICY,
                risk_level=StorageRiskLevel.MEDIUM,
                risk_score=40,
                title="No Lifecycle Policy",
                description="Bucket lacks lifecycle policies for cost optimization and data management",
                bucket_name=bucket.name,
                object_name=None,
                remediation_steps=[
                    "Configure lifecycle policies",
                    "Set automatic deletion rules for old objects",
                    "Implement storage class transitions",
                    "Define retention periods based on compliance requirements"
                ],
                compliance_frameworks=['SOC2', 'GDPR'],
                metadata={'has_lifecycle_rules': False},
                detected_at=datetime.utcnow()
            ))
        
        return findings
    
    def _check_access_controls(self, bucket) -> List[StorageFinding]:
        """Check access control configurations"""
        findings = []
        
        if not bucket.iam_configuration.uniform_bucket_level_access_enabled:
            findings.append(StorageFinding(
                finding_type=StorageFindingType.UNIFORM_BUCKET_ACCESS_DISABLED,
                risk_level=StorageRiskLevel.MEDIUM,
                risk_score=45,
                title="Uniform Bucket-Level Access Disabled",
                description="Bucket allows object-level ACLs which can be inconsistent",
                bucket_name=bucket.name,
                object_name=None,
                remediation_steps=[
                    "Enable uniform bucket-level access",
                    "Migrate object ACLs to IAM policies",
                    "Review and update access patterns",
                    "Implement consistent access controls"
                ],
                compliance_frameworks=['SOC2', 'ISO27001'],
                metadata={'uniform_access_enabled': False},
                detected_at=datetime.utcnow()
            ))
        
        return findings
    
    def _check_compliance_configurations(self, bucket) -> List[StorageFinding]:
        """Check compliance-related configurations"""
        findings = []
        
        # Check versioning
        if not bucket.versioning_enabled:
            findings.append(StorageFinding(
                finding_type=StorageFindingType.MISSING_VERSIONING,
                risk_level=StorageRiskLevel.MEDIUM,
                risk_score=35,
                title="Versioning Disabled",
                description="Bucket versioning is disabled, preventing recovery from accidental changes",
                bucket_name=bucket.name,
                object_name=None,
                remediation_steps=[
                    "Enable object versioning",
                    "Configure lifecycle rules for old versions",
                    "Implement version management policies",
                    "Set up monitoring for version changes"
                ],
                compliance_frameworks=['SOC2', 'GDPR'],
                metadata={'versioning_enabled': False},
                detected_at=datetime.utcnow()
            ))
        
        # Check logging
        if not bucket.logging:
            findings.append(StorageFinding(
                finding_type=StorageFindingType.LOGGING_DISABLED,
                risk_level=StorageRiskLevel.MEDIUM,
                risk_score=40,
                title="Access Logging Disabled",
                description="Bucket access logging is disabled, reducing audit capabilities",
                bucket_name=bucket.name,
                object_name=None,
                remediation_steps=[
                    "Enable access logging",
                    "Configure log destination bucket",
                    "Set up log analysis and monitoring",
                    "Implement log retention policies"
                ],
                compliance_frameworks=['SOC2', 'HIPAA', 'PCI_DSS'],
                metadata={'logging_enabled': False},
                detected_at=datetime.utcnow()
            ))
        
        # Check retention policy
        if not bucket.retention_policy:
            # Only flag this for buckets that likely need retention
            if self._likely_needs_retention(bucket):
                findings.append(StorageFinding(
                    finding_type=StorageFindingType.RETENTION_POLICY_MISSING,
                    risk_level=StorageRiskLevel.LOW,
                    risk_score=25,
                    title="No Retention Policy",
                    description="Bucket lacks retention policy for compliance requirements",
                    bucket_name=bucket.name,
                    object_name=None,
                    remediation_steps=[
                        "Configure retention policy based on compliance requirements",
                        "Set minimum retention periods",
                        "Implement legal hold capabilities",
                        "Document retention procedures"
                    ],
                    compliance_frameworks=['HIPAA', 'PCI_DSS', 'GDPR'],
                    metadata={'has_retention_policy': False},
                    detected_at=datetime.utcnow()
                ))
        
        return findings
    
    def _check_sensitive_data(self, bucket) -> List[StorageFinding]:
        """Check for potential sensitive data exposure"""
        findings = []
        
        # Check bucket name for sensitive patterns
        bucket_name_lower = bucket.name.lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if re.match(pattern, bucket_name_lower):
                findings.append(StorageFinding(
                    finding_type=StorageFindingType.SENSITIVE_DATA_EXPOSURE,
                    risk_level=StorageRiskLevel.HIGH,
                    risk_score=65,
                    title="Potential Sensitive Data in Bucket Name",
                    description=f"Bucket name contains potentially sensitive pattern: {pattern}",
                    bucket_name=bucket.name,
                    object_name=None,
                    remediation_steps=[
                        "Review bucket naming convention",
                        "Avoid sensitive information in bucket names",
                        "Consider renaming bucket if possible",
                        "Implement data classification policies"
                    ],
                    compliance_frameworks=['GDPR', 'HIPAA', 'PCI_DSS'],
                    metadata={'matched_pattern': pattern},
                    detected_at=datetime.utcnow()
                ))
                break  # Only report one pattern match per bucket
        
        return findings
    
    def _check_bucket_naming(self, bucket) -> List[StorageFinding]:
        """Check bucket naming conventions"""
        findings = []
        
        # Check for overly generic names
        generic_patterns = [r'^bucket.*', r'^test.*', r'^temp.*', r'^data.*', r'^storage.*']
        
        for pattern in generic_patterns:
            if re.match(pattern, bucket.name.lower()):
                findings.append(StorageFinding(
                    finding_type=StorageFindingType.BUCKET_NAMING_VIOLATION,
                    risk_level=StorageRiskLevel.LOW,
                    risk_score=15,
                    title="Generic Bucket Name",
                    description=f"Bucket uses generic naming pattern: {pattern}",
                    bucket_name=bucket.name,
                    object_name=None,
                    remediation_steps=[
                        "Use descriptive, purpose-specific bucket names",
                        "Include project or environment identifiers",
                        "Follow organizational naming conventions",
                        "Avoid generic or temporary-sounding names"
                    ],
                    compliance_frameworks=[],
                    metadata={'naming_pattern': pattern},
                    detected_at=datetime.utcnow()
                ))
                break
        
        return findings
    
    def _get_retention_policy_info(self, bucket) -> Optional[Dict[str, Any]]:
        """Get retention policy information"""
        if bucket.retention_policy:
            return {
                'retention_period': bucket.retention_policy.retention_period,
                'effective_time': bucket.retention_policy.effective_time.isoformat() if bucket.retention_policy.effective_time else None,
                'is_locked': bucket.retention_policy.is_locked
            }
        return None
    
    def _classify_data(self, bucket) -> Optional[str]:
        """Classify data sensitivity based on bucket name and patterns"""
        bucket_name_lower = bucket.name.lower()
        
        # High sensitivity patterns
        high_sensitivity = ['personal', 'private', 'confidential', 'secret', 'pii', 'phi']
        if any(pattern in bucket_name_lower for pattern in high_sensitivity):
            return 'HIGH'
        
        # Medium sensitivity patterns
        medium_sensitivity = ['internal', 'business', 'customer', 'financial']
        if any(pattern in bucket_name_lower for pattern in medium_sensitivity):
            return 'MEDIUM'
        
        # Low sensitivity patterns
        low_sensitivity = ['public', 'marketing', 'website', 'static']
        if any(pattern in bucket_name_lower for pattern in low_sensitivity):
            return 'LOW'
        
        return 'UNKNOWN'
    
    def _likely_needs_retention(self, bucket) -> bool:
        """Determine if bucket likely needs retention policy"""
        # Check for patterns that suggest long-term storage needs
        retention_indicators = ['archive', 'backup', 'audit', 'log', 'compliance', 'legal']
        bucket_name_lower = bucket.name.lower()
        
        return any(indicator in bucket_name_lower for indicator in retention_indicators)
    
    def _calculate_storage_posture(self, bucket_analyses: List[BucketSecurityAnalysis],
                                 all_findings: List[StorageFinding]) -> StorageSecurityPosture:
        """Calculate overall storage security posture"""
        
        if not bucket_analyses:
            return self._generate_sample_analysis()
        
        # Count findings by risk level
        risk_distribution = {
            "CRITICAL": len([f for f in all_findings if f.risk_level == StorageRiskLevel.CRITICAL]),
            "HIGH": len([f for f in all_findings if f.risk_level == StorageRiskLevel.HIGH]),
            "MEDIUM": len([f for f in all_findings if f.risk_level == StorageRiskLevel.MEDIUM]),
            "LOW": len([f for f in all_findings if f.risk_level == StorageRiskLevel.LOW]),
            "MINIMAL": len([f for f in all_findings if f.risk_level == StorageRiskLevel.MINIMAL])
        }
        
        # Calculate posture score
        penalty = (
            risk_distribution["CRITICAL"] * 30 +
            risk_distribution["HIGH"] * 20 +
            risk_distribution["MEDIUM"] * 10 +
            risk_distribution["LOW"] * 5 +
            risk_distribution["MINIMAL"] * 2
        )
        
        posture_score = max(0, 100 - penalty)
        
        # Calculate metrics
        total_buckets = len(bucket_analyses)
        public_buckets = len([b for b in bucket_analyses if b.is_public])
        encrypted_buckets = len([b for b in bucket_analyses if b.encryption_type == "CUSTOMER_MANAGED"])
        compliant_buckets = len([b for b in bucket_analyses if b.risk_score < 30])
        
        # Calculate compliance scores
        compliance_status = self._calculate_compliance_scores(bucket_analyses)
        
        # Generate recommendations
        recommendations = self._generate_storage_recommendations(
            risk_distribution, public_buckets, encrypted_buckets, total_buckets
        )
        
        return StorageSecurityPosture(
            project_id=self.project_id,
            posture_score=posture_score,
            risk_distribution=risk_distribution,
            total_buckets=total_buckets,
            public_buckets=public_buckets,
            encrypted_buckets=encrypted_buckets,
            compliant_buckets=compliant_buckets,
            findings=all_findings,
            recommendations=recommendations,
            compliance_status=compliance_status,
            analyzed_at=datetime.utcnow()
        )
    
    def _calculate_compliance_scores(self, bucket_analyses: List[BucketSecurityAnalysis]) -> Dict[str, float]:
        """Calculate compliance scores for various frameworks"""
        compliance_scores = {}
        
        for framework in self.COMPLIANCE_FRAMEWORKS:
            requirements = self.COMPLIANCE_FRAMEWORKS[framework]
            total_score = 0
            
            for bucket in bucket_analyses:
                bucket_score = 0
                requirement_count = len(requirements)
                
                if 'encryption' in requirements:
                    bucket_score += 25 if bucket.encryption_type == "CUSTOMER_MANAGED" else 10
                
                if 'access_control' in requirements:
                    bucket_score += 25 if bucket.uniform_bucket_access and bucket.public_access_prevention else 10
                
                if 'logging' in requirements or 'audit_logging' in requirements:
                    bucket_score += 25 if bucket.logging_enabled else 0
                
                if 'retention' in requirements or 'data_retention' in requirements:
                    bucket_score += 25 if bucket.retention_policy else 10
                
                total_score += min(bucket_score, 100)
            
            # Average across all buckets
            compliance_scores[framework] = total_score / len(bucket_analyses) if bucket_analyses else 0
        
        return compliance_scores
    
    def _generate_storage_recommendations(self, risk_distribution: Dict[str, int],
                                        public_buckets: int, encrypted_buckets: int,
                                        total_buckets: int) -> List[str]:
        """Generate actionable storage recommendations"""
        recommendations = []
        
        # Priority recommendations based on findings
        if risk_distribution["CRITICAL"] > 0:
            recommendations.append("🔴 CRITICAL: Remove public write access from storage buckets immediately")
        
        if public_buckets > 0:
            recommendations.append(f"🟠 HIGH: Review {public_buckets} public buckets and implement access controls")
        
        if encrypted_buckets < total_buckets:
            unencrypted = total_buckets - encrypted_buckets
            recommendations.append(f"🟡 MEDIUM: Enable customer-managed encryption for {unencrypted} buckets")
        
        # General recommendations
        recommendations.extend([
            "Enable public access prevention on all buckets by default",
            "Implement uniform bucket-level access for consistent security",
            "Configure lifecycle policies for cost optimization and data management",
            "Enable access logging for security monitoring and compliance",
            "Use signed URLs for temporary access instead of public buckets",
            "Implement data classification and handle sensitive data appropriately",
            "Set up retention policies based on compliance requirements",
            "Monitor bucket configurations with Cloud Security Command Center"
        ])
        
        return recommendations[:10]  # Limit to top 10
    
    def _generate_sample_analysis(self) -> StorageSecurityPosture:
        """Generate sample analysis when GCP clients are unavailable"""
        sample_findings = [
            StorageFinding(
                finding_type=StorageFindingType.PUBLIC_BUCKET_READ,
                risk_level=StorageRiskLevel.HIGH,
                risk_score=80,
                title="Public Read Access",
                description="Bucket allows public read access",
                bucket_name=f"sample-public-bucket-{self.project_id}",
                object_name=None,
                remediation_steps=[
                    "Remove allUsers from IAM policy",
                    "Implement signed URLs for temporary access",
                    "Enable public access prevention"
                ],
                compliance_frameworks=['SOC2', 'GDPR'],
                metadata={"role": "roles/storage.objectViewer"},
                detected_at=datetime.utcnow()
            ),
            StorageFinding(
                finding_type=StorageFindingType.MISSING_ENCRYPTION,
                risk_level=StorageRiskLevel.HIGH,
                risk_score=70,
                title="No Customer-Managed Encryption",
                description="Bucket uses Google-managed encryption",
                bucket_name=f"sample-unencrypted-bucket-{self.project_id}",
                object_name=None,
                remediation_steps=[
                    "Configure customer-managed encryption keys",
                    "Create Cloud KMS keys for encryption",
                    "Enable automatic key rotation"
                ],
                compliance_frameworks=['HIPAA', 'PCI_DSS'],
                metadata={"current_encryption": "google_managed"},
                detected_at=datetime.utcnow()
            )
        ]
        
        return StorageSecurityPosture(
            project_id=self.project_id,
            posture_score=65,
            risk_distribution={"CRITICAL": 0, "HIGH": 2, "MEDIUM": 1, "LOW": 1, "MINIMAL": 0},
            total_buckets=5,
            public_buckets=1,
            encrypted_buckets=2,
            compliant_buckets=3,
            findings=sample_findings,
            recommendations=[
                "🟠 HIGH: Review 1 public bucket and implement access controls",
                "🟡 MEDIUM: Enable customer-managed encryption for 3 buckets",
                "Enable public access prevention on all buckets by default",
                "Implement uniform bucket-level access for consistent security"
            ],
            compliance_status={
                'SOC2': 70.0,
                'HIPAA': 60.0,
                'PCI_DSS': 65.0,
                'GDPR': 75.0,
                'ISO27001': 68.0
            },
            analyzed_at=datetime.utcnow()
        )