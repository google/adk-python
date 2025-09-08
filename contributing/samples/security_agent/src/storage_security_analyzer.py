"""Storage Security Analyzer for GCP Security Agent

Comprehensive storage security analysis engine that validates bucket policies,
identifies data exposure risks, and provides storage hardening recommendations.
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

class StorageRiskLevel(Enum):
    """Storage security risk levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class StorageViolationType(Enum):
    """Types of storage security violations."""
    PUBLIC_ACCESS = "PUBLIC_ACCESS"
    WEAK_ENCRYPTION = "WEAK_ENCRYPTION"
    MISSING_VERSIONING = "MISSING_VERSIONING"
    INADEQUATE_LIFECYCLE = "INADEQUATE_LIFECYCLE"
    IAM_MISCONFIGURATION = "IAM_MISCONFIGURATION"
    LOGGING_DISABLED = "LOGGING_DISABLED"
    CORS_MISCONFIGURATION = "CORS_MISCONFIGURATION"
    RETENTION_ISSUES = "RETENTION_ISSUES"

@dataclass
class StorageSecurityIssue:
    """Storage security issue."""
    issue_id: str
    bucket_name: str
    violation_type: StorageViolationType
    risk_level: StorageRiskLevel
    title: str
    description: str
    recommendation: str
    impact: str
    remediation_steps: List[str]
    compliance_impact: List[str]  # SOC2, PCI-DSS, etc.
    evidence: Dict[str, Any]
    detected_at: datetime

@dataclass
class BucketSecurityProfile:
    """Security profile for a storage bucket."""
    bucket_name: str
    location: str
    storage_class: str
    security_score: float  # 0-100
    issues: List[StorageSecurityIssue]
    security_features: Dict[str, bool]
    recommendations: List[str]
    compliance_status: Dict[str, str]
    last_analyzed: datetime

@dataclass
class StorageSecuritySummary:
    """Overall storage security summary."""
    total_buckets: int
    secure_buckets: int
    at_risk_buckets: int
    critical_issues: int
    total_issues: int
    average_security_score: float
    compliance_score: float
    top_risks: List[BucketSecurityProfile]
    priority_actions: List[str]
    analysis_timestamp: datetime

class StorageSecurityAnalyzer:
    """Advanced storage security analyzer."""
    
    def __init__(self, project_id: str, db_path: Optional[str] = None):
        self.project_id = project_id
        self.db_path = db_path or self._get_default_db_path()
        
        # Security feature weights for scoring
        self.security_weights = {
            'public_access_prevention': 25,
            'uniform_bucket_level_access': 20,
            'versioning_enabled': 15,
            'encryption_at_rest': 15,
            'logging_enabled': 10,
            'lifecycle_management': 10,
            'retention_policy': 5
        }
        
        # Compliance framework requirements
        self.compliance_requirements = {
            'SOC2': {
                'public_access_prevention': 'enforced',
                'uniform_bucket_level_access': True,
                'versioning_enabled': True,
                'encryption_at_rest': True,
                'logging_enabled': True
            },
            'PCI-DSS': {
                'public_access_prevention': 'enforced',
                'encryption_at_rest': True,
                'uniform_bucket_level_access': True,
                'retention_policy': True
            },
            'HIPAA': {
                'public_access_prevention': 'enforced',
                'encryption_at_rest': True,
                'uniform_bucket_level_access': True,
                'versioning_enabled': True,
                'logging_enabled': True
            },
            'GDPR': {
                'public_access_prevention': 'enforced',
                'encryption_at_rest': True,
                'retention_policy': True,
                'lifecycle_management': True
            }
        }
    
    def _get_default_db_path(self) -> str:
        """Get default database path."""
        current_file = Path(__file__)
        security_agent_dir = current_file.parent.parent.parent
        return str(security_agent_dir / 'backend' / 'cache' / 'gcp_data.db')
    
    def analyze_all_storage(self) -> StorageSecuritySummary:
        """Perform comprehensive storage security analysis."""
        start_time = datetime.now()
        
        # Get all storage buckets
        buckets_data = self._get_all_buckets()
        
        # Analyze each bucket
        bucket_profiles = []
        for bucket_data in buckets_data:
            profile = self._analyze_bucket_security(bucket_data)
            bucket_profiles.append(profile)
        
        # Calculate summary metrics
        total_buckets = len(bucket_profiles)
        secure_buckets = len([b for b in bucket_profiles if b.security_score >= 80])
        at_risk_buckets = len([b for b in bucket_profiles if b.security_score < 60])
        
        all_issues = []
        for profile in bucket_profiles:
            all_issues.extend(profile.issues)
        
        critical_issues = len([i for i in all_issues if i.risk_level == StorageRiskLevel.CRITICAL])
        average_score = sum(b.security_score for b in bucket_profiles) / total_buckets if total_buckets > 0 else 100
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(bucket_profiles)
        
        # Generate priority actions
        priority_actions = self._generate_priority_actions(all_issues, bucket_profiles)
        
        summary = StorageSecuritySummary(
            total_buckets=total_buckets,
            secure_buckets=secure_buckets,
            at_risk_buckets=at_risk_buckets,
            critical_issues=critical_issues,
            total_issues=len(all_issues),
            average_security_score=average_score,
            compliance_score=compliance_score,
            top_risks=sorted(bucket_profiles, key=lambda b: b.security_score)[:5],
            priority_actions=priority_actions,
            analysis_timestamp=datetime.now()
        )
        
        # Store analysis results
        self._store_storage_analysis(summary, bucket_profiles)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"🪣 Storage security analysis complete: {total_buckets} buckets analyzed in {duration:.2f}s")
        
        return summary
    
    def _get_all_buckets(self) -> List[Dict[str, Any]]:
        """Get all storage buckets from database."""
        buckets = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM storage_buckets")
            db_buckets = cursor.fetchall()
            
            for bucket in db_buckets:
                buckets.append(dict(bucket))
            
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not load buckets from database: {e}")
            buckets = self._create_mock_buckets()
        
        return buckets
    
    def _analyze_bucket_security(self, bucket_data: Dict[str, Any]) -> BucketSecurityProfile:
        """Analyze security for a specific bucket."""
        bucket_name = bucket_data['name']
        issues = []
        
        # Analyze each security aspect
        issues.extend(self._check_public_access(bucket_data))
        issues.extend(self._check_encryption(bucket_data))
        issues.extend(self._check_access_control(bucket_data))
        issues.extend(self._check_versioning(bucket_data))
        issues.extend(self._check_lifecycle_management(bucket_data))
        issues.extend(self._check_logging(bucket_data))
        issues.extend(self._check_cors_configuration(bucket_data))
        issues.extend(self._check_retention_policies(bucket_data))
        
        # Calculate security features
        security_features = self._evaluate_security_features(bucket_data)
        
        # Calculate security score
        security_score = self._calculate_security_score(security_features, issues)
        
        # Generate recommendations
        recommendations = self._generate_bucket_recommendations(bucket_data, issues, security_features)
        
        # Evaluate compliance status
        compliance_status = self._evaluate_compliance_status(security_features)
        
        return BucketSecurityProfile(
            bucket_name=bucket_name,
            location=bucket_data.get('location', 'unknown'),
            storage_class=bucket_data.get('storage_class', 'unknown'),
            security_score=security_score,
            issues=issues,
            security_features=security_features,
            recommendations=recommendations,
            compliance_status=compliance_status,
            last_analyzed=datetime.now()
        )
    
    def _check_public_access(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check for public access vulnerabilities."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Check public access prevention
        pap_setting = bucket_data.get('public_access_prevention', 'inherited')
        if pap_setting != 'enforced':
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-pap-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.PUBLIC_ACCESS,
                risk_level=StorageRiskLevel.CRITICAL,
                title="Public access prevention not enforced",
                description=f"Bucket {bucket_name} does not enforce public access prevention",
                recommendation="Enable 'Enforce public access prevention' on the bucket",
                impact="Bucket contents could be exposed to the internet",
                remediation_steps=[
                    "Navigate to Cloud Storage in GCP Console",
                    f"Select bucket {bucket_name}",
                    "Go to Permissions tab",
                    "Click 'Prevent public access'",
                    "Select 'Enforce public access prevention'"
                ],
                compliance_impact=["SOC2", "PCI-DSS", "HIPAA", "GDPR"],
                evidence={"public_access_prevention": pap_setting},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_encryption(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check encryption configuration."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Parse encryption configuration
        try:
            encryption_config = json.loads(bucket_data.get('encryption_config', '{}'))
        except (json.JSONDecodeError, TypeError):
            encryption_config = {}
        
        # Check for customer-managed encryption keys
        if not encryption_config.get('default_kms_key_name'):
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-encryption-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.WEAK_ENCRYPTION,
                risk_level=StorageRiskLevel.MEDIUM,
                title="Using Google-managed encryption keys",
                description=f"Bucket {bucket_name} uses default encryption instead of customer-managed keys",
                recommendation="Configure customer-managed encryption keys (CMEK) for better control",
                impact="Less control over encryption key lifecycle and access",
                remediation_steps=[
                    "Create or select a Cloud KMS key",
                    f"Navigate to bucket {bucket_name} settings",
                    "Update default encryption to use CMEK",
                    "Verify key permissions for Cloud Storage service"
                ],
                compliance_impact=["PCI-DSS", "HIPAA"],
                evidence={"encryption_config": encryption_config},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_access_control(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check access control configuration."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Check uniform bucket-level access
        ubla_enabled = bucket_data.get('uniform_bucket_level_access', False)
        if not ubla_enabled:
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-ubla-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.IAM_MISCONFIGURATION,
                risk_level=StorageRiskLevel.MEDIUM,
                title="Uniform bucket-level access disabled",
                description=f"Bucket {bucket_name} allows ACLs alongside IAM policies",
                recommendation="Enable uniform bucket-level access for consistent access control",
                impact="Mixed ACL and IAM policies can create security gaps",
                remediation_steps=[
                    f"Navigate to bucket {bucket_name} permissions",
                    "Click 'Switch to uniform'",
                    "Confirm uniform bucket-level access",
                    "Review and update IAM policies as needed"
                ],
                compliance_impact=["SOC2"],
                evidence={"uniform_bucket_level_access": ubla_enabled},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_versioning(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check versioning configuration."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Check if versioning is enabled
        versioning_enabled = bucket_data.get('versioning_enabled', False)
        if not versioning_enabled:
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-versioning-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.MISSING_VERSIONING,
                risk_level=StorageRiskLevel.LOW,
                title="Object versioning disabled",
                description=f"Bucket {bucket_name} does not have object versioning enabled",
                recommendation="Enable object versioning to protect against accidental deletion or modification",
                impact="No protection against accidental data loss or corruption",
                remediation_steps=[
                    f"Navigate to bucket {bucket_name}",
                    "Go to Configuration tab",
                    "Enable Object Versioning",
                    "Configure lifecycle rules to manage old versions"
                ],
                compliance_impact=["SOC2", "HIPAA"],
                evidence={"versioning_enabled": versioning_enabled},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_lifecycle_management(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check lifecycle management configuration."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Parse lifecycle rules
        try:
            lifecycle_rules = json.loads(bucket_data.get('lifecycle_rules', '[]'))
        except (json.JSONDecodeError, TypeError):
            lifecycle_rules = []
        
        # Check if lifecycle rules are configured
        if not lifecycle_rules:
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-lifecycle-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.INADEQUATE_LIFECYCLE,
                risk_level=StorageRiskLevel.LOW,
                title="No lifecycle management rules configured",
                description=f"Bucket {bucket_name} has no lifecycle rules for cost optimization and data management",
                recommendation="Configure lifecycle rules to automatically manage object lifecycle",
                impact="Potential cost inefficiency and lack of automated data management",
                remediation_steps=[
                    f"Navigate to bucket {bucket_name}",
                    "Go to Lifecycle tab",
                    "Add lifecycle rules for:",
                    "- Transitioning to cheaper storage classes",
                    "- Deleting old objects or versions",
                    "- Managing multipart uploads"
                ],
                compliance_impact=["GDPR"],
                evidence={"lifecycle_rules_count": len(lifecycle_rules)},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_logging(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check logging configuration."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Parse logging configuration
        try:
            logging_config = json.loads(bucket_data.get('logging_config', '{}'))
        except (json.JSONDecodeError, TypeError):
            logging_config = {}
        
        # Check if access logging is enabled
        if not logging_config.get('log_bucket'):
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-logging-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.LOGGING_DISABLED,
                risk_level=StorageRiskLevel.MEDIUM,
                title="Access logging disabled",
                description=f"Bucket {bucket_name} does not have access logging enabled",
                recommendation="Enable access logging to track bucket access patterns",
                impact="Limited visibility into access patterns and potential security incidents",
                remediation_steps=[
                    f"Navigate to bucket {bucket_name}",
                    "Go to Configuration tab",
                    "Enable Access logs",
                    "Specify log bucket and prefix",
                    "Ensure log bucket has appropriate permissions"
                ],
                compliance_impact=["SOC2", "HIPAA"],
                evidence={"logging_enabled": bool(logging_config.get('log_bucket'))},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _check_cors_configuration(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check CORS configuration for security issues."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Parse CORS rules
        try:
            cors_rules = json.loads(bucket_data.get('cors_rules', '[]'))
        except (json.JSONDecodeError, TypeError):
            cors_rules = []
        
        # Check for overly permissive CORS rules
        for i, rule in enumerate(cors_rules):
            origins = rule.get('origin', [])
            if '*' in origins:
                issues.append(StorageSecurityIssue(
                    issue_id=f"storage-cors-{hash(bucket_name)}-{i}",
                    bucket_name=bucket_name,
                    violation_type=StorageViolationType.CORS_MISCONFIGURATION,
                    risk_level=StorageRiskLevel.MEDIUM,
                    title="Overly permissive CORS policy",
                    description=f"Bucket {bucket_name} has CORS rule allowing all origins (*)",
                    recommendation="Restrict CORS origins to specific domains that need access",
                    impact="Potential for unauthorized cross-origin requests",
                    remediation_steps=[
                        f"Navigate to bucket {bucket_name}",
                        "Go to Permissions tab",
                        "Edit CORS configuration",
                        "Replace '*' with specific domain names",
                        "Test CORS functionality after changes"
                    ],
                    compliance_impact=["SOC2"],
                    evidence={"cors_rule": rule},
                    detected_at=datetime.now()
                ))
        
        return issues
    
    def _check_retention_policies(self, bucket_data: Dict[str, Any]) -> List[StorageSecurityIssue]:
        """Check retention policy configuration."""
        issues = []
        bucket_name = bucket_data['name']
        
        # Parse retention policy
        try:
            retention_policy = json.loads(bucket_data.get('retention_policy', '{}'))
        except (json.JSONDecodeError, TypeError):
            retention_policy = {}
        
        # Check if retention policy is needed but missing
        # This is a simplified check - in practice would depend on data classification
        if not retention_policy and bucket_name.find('backup') != -1:
            issues.append(StorageSecurityIssue(
                issue_id=f"storage-retention-{hash(bucket_name)}",
                bucket_name=bucket_name,
                violation_type=StorageViolationType.RETENTION_ISSUES,
                risk_level=StorageRiskLevel.LOW,
                title="No retention policy configured",
                description=f"Bucket {bucket_name} appears to store backup data but has no retention policy",
                recommendation="Configure retention policy for compliance and data governance",
                impact="May not meet regulatory retention requirements",
                remediation_steps=[
                    f"Navigate to bucket {bucket_name}",
                    "Go to Configuration tab",
                    "Set retention policy period",
                    "Consider bucket lock for compliance",
                    "Document retention policy justification"
                ],
                compliance_impact=["PCI-DSS", "HIPAA", "GDPR"],
                evidence={"retention_policy": retention_policy},
                detected_at=datetime.now()
            ))
        
        return issues
    
    def _evaluate_security_features(self, bucket_data: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate security features for a bucket."""
        try:
            encryption_config = json.loads(bucket_data.get('encryption_config', '{}'))
            logging_config = json.loads(bucket_data.get('logging_config', '{}'))
            lifecycle_rules = json.loads(bucket_data.get('lifecycle_rules', '[]'))
            retention_policy = json.loads(bucket_data.get('retention_policy', '{}'))
        except (json.JSONDecodeError, TypeError):
            encryption_config = {}
            logging_config = {}
            lifecycle_rules = []
            retention_policy = {}
        
        return {
            'public_access_prevention': bucket_data.get('public_access_prevention') == 'enforced',
            'uniform_bucket_level_access': bucket_data.get('uniform_bucket_level_access', False),
            'versioning_enabled': bucket_data.get('versioning_enabled', False),
            'encryption_at_rest': bool(encryption_config.get('default_kms_key_name')),
            'logging_enabled': bool(logging_config.get('log_bucket')),
            'lifecycle_management': len(lifecycle_rules) > 0,
            'retention_policy': bool(retention_policy)
        }
    
    def _calculate_security_score(self, security_features: Dict[str, bool], 
                                issues: List[StorageSecurityIssue]) -> float:
        """Calculate security score for a bucket (0-100)."""
        # Start with feature-based score
        feature_score = 0.0
        for feature, enabled in security_features.items():
            if enabled and feature in self.security_weights:
                feature_score += self.security_weights[feature]
        
        # Deduct points for issues
        issue_deductions = 0.0
        for issue in issues:
            if issue.risk_level == StorageRiskLevel.CRITICAL:
                issue_deductions += 20
            elif issue.risk_level == StorageRiskLevel.HIGH:
                issue_deductions += 15
            elif issue.risk_level == StorageRiskLevel.MEDIUM:
                issue_deductions += 10
            elif issue.risk_level == StorageRiskLevel.LOW:
                issue_deductions += 5
        
        return max(0, feature_score - issue_deductions)
    
    def _generate_bucket_recommendations(self, bucket_data: Dict[str, Any], 
                                       issues: List[StorageSecurityIssue],
                                       security_features: Dict[str, bool]) -> List[str]:
        """Generate recommendations for a specific bucket."""
        recommendations = []
        
        # Priority recommendations based on issues
        critical_issues = [i for i in issues if i.risk_level == StorageRiskLevel.CRITICAL]
        if critical_issues:
            recommendations.append("🚨 URGENT: Address critical security issues immediately")
        
        # Feature-specific recommendations
        if not security_features.get('public_access_prevention'):
            recommendations.append("🔒 Enable public access prevention")
        
        if not security_features.get('uniform_bucket_level_access'):
            recommendations.append("⚖️ Enable uniform bucket-level access")
        
        if not security_features.get('versioning_enabled'):
            recommendations.append("📚 Enable object versioning for data protection")
        
        if not security_features.get('encryption_at_rest'):
            recommendations.append("🔐 Configure customer-managed encryption keys")
        
        if not security_features.get('logging_enabled'):
            recommendations.append("📊 Enable access logging for audit trail")
        
        if not security_features.get('lifecycle_management'):
            recommendations.append("♻️ Configure lifecycle rules for cost optimization")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _evaluate_compliance_status(self, security_features: Dict[str, bool]) -> Dict[str, str]:
        """Evaluate compliance status against various frameworks."""
        compliance_status = {}
        
        for framework, requirements in self.compliance_requirements.items():
            compliant = True
            for requirement, expected_value in requirements.items():
                if requirement in security_features:
                    if security_features[requirement] != expected_value:
                        compliant = False
                        break
                else:
                    compliant = False
                    break
            
            compliance_status[framework] = "COMPLIANT" if compliant else "NON_COMPLIANT"
        
        return compliance_status
    
    def _calculate_compliance_score(self, bucket_profiles: List[BucketSecurityProfile]) -> float:
        """Calculate overall compliance score."""
        if not bucket_profiles:
            return 100.0
        
        total_score = 0.0
        for profile in bucket_profiles:
            compliant_frameworks = len([s for s in profile.compliance_status.values() if s == "COMPLIANT"])
            total_frameworks = len(profile.compliance_status)
            if total_frameworks > 0:
                total_score += (compliant_frameworks / total_frameworks) * 100
        
        return total_score / len(bucket_profiles)
    
    def _generate_priority_actions(self, all_issues: List[StorageSecurityIssue],
                                 bucket_profiles: List[BucketSecurityProfile]) -> List[str]:
        """Generate priority actions based on analysis."""
        actions = []
        
        # Count issues by type and severity
        issue_counts = {}
        for issue in all_issues:
            key = (issue.violation_type.value, issue.risk_level.value)
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        # Generate actions based on most common critical issues
        critical_public = issue_counts.get(('PUBLIC_ACCESS', 'CRITICAL'), 0)
        if critical_public > 0:
            actions.append(f"🚨 CRITICAL: Fix public access issues on {critical_public} buckets immediately")
        
        high_encryption = issue_counts.get(('WEAK_ENCRYPTION', 'MEDIUM'), 0)
        if high_encryption > 0:
            actions.append(f"🔐 Implement customer-managed encryption on {high_encryption} buckets")
        
        # Low-score buckets
        low_score_buckets = len([b for b in bucket_profiles if b.security_score < 50])
        if low_score_buckets > 0:
            actions.append(f"⚠️ Review {low_score_buckets} buckets with security scores below 50")
        
        # General recommendations
        actions.extend([
            "📊 Implement continuous storage security monitoring",
            "📋 Create storage security policy and procedures",
            "👥 Provide storage security training to teams",
            "🔄 Schedule quarterly storage security reviews"
        ])
        
        return actions[:10]  # Return top 10 actions
    
    def _store_storage_analysis(self, summary: StorageSecuritySummary, 
                              profiles: List[BucketSecurityProfile]):
        """Store storage analysis results in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create storage analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS storage_security_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_timestamp TIMESTAMP NOT NULL,
                    total_buckets INTEGER,
                    secure_buckets INTEGER,
                    at_risk_buckets INTEGER,
                    critical_issues INTEGER,
                    total_issues INTEGER,
                    average_security_score REAL,
                    compliance_score REAL,
                    priority_actions TEXT,  -- JSON
                    analysis_data TEXT      -- JSON
                )
            """)
            
            # Store summary
            cursor.execute("""
                INSERT INTO storage_security_analysis
                (analysis_timestamp, total_buckets, secure_buckets, at_risk_buckets,
                 critical_issues, total_issues, average_security_score, compliance_score,
                 priority_actions, analysis_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.analysis_timestamp.isoformat(),
                summary.total_buckets,
                summary.secure_buckets,
                summary.at_risk_buckets,
                summary.critical_issues,
                summary.total_issues,
                summary.average_security_score,
                summary.compliance_score,
                json.dumps(summary.priority_actions),
                json.dumps({
                    'buckets': [
                        {
                            'name': p.bucket_name,
                            'security_score': p.security_score,
                            'issues_count': len(p.issues),
                            'security_features': p.security_features,
                            'compliance_status': p.compliance_status
                        }
                        for p in profiles
                    ]
                })
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Storage security analysis results stored in database")
            
        except Exception as e:
            logger.error(f"Failed to store storage analysis: {e}")
    
    def _create_mock_buckets(self) -> List[Dict[str, Any]]:
        """Create mock bucket data for development."""
        return [
            {
                'name': 'app-data-bucket',
                'location': 'US',
                'storage_class': 'STANDARD',
                'versioning_enabled': False,
                'lifecycle_rules': '[]',
                'public_access_prevention': 'inherited',
                'uniform_bucket_level_access': False,
                'retention_policy': '{}',
                'encryption_config': '{}',
                'cors_rules': '[]',
                'logging_config': '{}'
            },
            {
                'name': 'secure-backup-bucket',
                'location': 'US',
                'storage_class': 'COLDLINE',
                'versioning_enabled': True,
                'lifecycle_rules': '[{"action": {"type": "Delete"}, "condition": {"age": 365}}]',
                'public_access_prevention': 'enforced',
                'uniform_bucket_level_access': True,
                'retention_policy': '{"retention_period": "31536000"}',
                'encryption_config': '{"default_kms_key_name": "projects/project/locations/us/keyRings/ring/cryptoKeys/key"}',
                'cors_rules': '[]',
                'logging_config': '{"log_bucket": "access-logs-bucket"}'
            }
        ]

# Convenience functions
def analyze_storage_security(project_id: str, db_path: Optional[str] = None) -> StorageSecuritySummary:
    """Convenience function to run storage security analysis."""
    analyzer = StorageSecurityAnalyzer(project_id, db_path)
    return analyzer.analyze_all_storage()

def get_storage_security_summary(project_id: str, db_path: Optional[str] = None) -> str:
    """Get formatted storage security summary."""
    summary = analyze_storage_security(project_id, db_path)
    
    result = f"🪣 **Storage Security Analysis**\\n\\n"
    result += f"**Total Buckets**: {summary.total_buckets}\\n"
    result += f"**Secure Buckets**: {summary.secure_buckets}\\n"
    result += f"**At-Risk Buckets**: {summary.at_risk_buckets}\\n"
    result += f"**Critical Issues**: {summary.critical_issues}\\n"
    result += f"**Average Security Score**: {summary.average_security_score:.1f}/100\\n"
    result += f"**Compliance Score**: {summary.compliance_score:.1f}/100\\n\\n"
    
    if summary.top_risks:
        result += "**Highest Risk Bucket:**\\n"
        top_risk = summary.top_risks[0]
        result += f"* {top_risk.bucket_name}\\n"
        result += f"  Security Score: {top_risk.security_score:.1f}/100\\n"
        result += f"  Issues: {len(top_risk.issues)}\\n"
        result += f"  Location: {top_risk.location}\\n\\n"
    
    if summary.priority_actions:
        result += "**Priority Actions:**\\n"
        for action in summary.priority_actions[:3]:
            result += f"* {action}\\n"
    
    return result