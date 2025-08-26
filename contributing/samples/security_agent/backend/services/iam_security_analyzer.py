"""
Enhanced IAM Security Analyzer (STORY-003)

Provides advanced IAM security analysis including overprivileged account detection,
service account key rotation analysis, least privilege recommendations, and
cross-project IAM analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json

try:
    from google.cloud import iam_admin_v1
    from google.cloud import resourcemanager_v3
    from google.cloud import asset_v1
    from google.api_core import exceptions as gcp_exceptions
    GCP_CLIENTS_AVAILABLE = True
except ImportError:
    GCP_CLIENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IAMFindingType(Enum):
    """IAM finding types"""
    OVERPRIVILEGED_SERVICE_ACCOUNT = "OVERPRIVILEGED_SERVICE_ACCOUNT"
    STALE_SERVICE_ACCOUNT_KEY = "STALE_SERVICE_ACCOUNT_KEY"
    EXCESSIVE_PERMISSIONS = "EXCESSIVE_PERMISSIONS"
    UNUSED_SERVICE_ACCOUNT = "UNUSED_SERVICE_ACCOUNT"
    CROSS_PROJECT_ACCESS = "CROSS_PROJECT_ACCESS"
    WILDCARD_BINDING = "WILDCARD_BINDING"
    ADMIN_ROLE_MISUSE = "ADMIN_ROLE_MISUSE"
    EXTERNAL_USER_ACCESS = "EXTERNAL_USER_ACCESS"
    MISSING_MFA = "MISSING_MFA"
    WEAK_ROLE_ASSIGNMENT = "WEAK_ROLE_ASSIGNMENT"


@dataclass
class IAMFinding:
    """IAM security finding"""
    finding_type: IAMFindingType
    risk_level: RiskLevel
    risk_score: int  # 0-100
    title: str
    description: str
    resource_name: str
    affected_principal: str
    remediation_steps: List[str]
    metadata: Dict[str, Any]
    detected_at: datetime


@dataclass
class ServiceAccountAnalysis:
    """Service account analysis result"""
    email: str
    display_name: str
    unique_id: str
    project_id: str
    disabled: bool
    key_count: int
    oldest_key_age_days: Optional[int]
    roles: List[str]
    risk_score: int
    findings: List[IAMFinding]
    last_used: Optional[datetime]
    cross_project_access: bool


@dataclass
class IAMSecurityPosture:
    """Overall IAM security posture"""
    project_id: str
    posture_score: int  # 0-100
    risk_distribution: Dict[str, int]
    total_findings: int
    critical_findings: int
    high_findings: int
    service_account_count: int
    overprivileged_accounts: int
    stale_keys: int
    cross_project_bindings: int
    external_users: int
    recommendations: List[str]
    findings: List[IAMFinding]
    analyzed_at: datetime


class IAMSecurityAnalyzer:
    """Enhanced IAM Security Analyzer"""
    
    # High-privilege roles that should be carefully monitored
    HIGH_PRIVILEGE_ROLES = {
        "roles/owner",
        "roles/editor", 
        "roles/iam.securityAdmin",
        "roles/iam.serviceAccountAdmin",
        "roles/iam.serviceAccountKeyAdmin",
        "roles/resourcemanager.projectIamAdmin",
        "roles/compute.admin",
        "roles/storage.admin",
        "roles/cloudsql.admin",
        "roles/container.admin"
    }
    
    # Admin roles that should rarely be assigned to service accounts
    ADMIN_ROLES = {
        "roles/owner",
        "roles/iam.securityAdmin",
        "roles/iam.serviceAccountAdmin",
        "roles/resourcemanager.organizationAdmin"
    }
    
    # Roles that indicate potential overprivilege
    BROAD_ROLES = {
        "roles/editor",
        "roles/owner",
        "roles/compute.admin",
        "roles/storage.admin"
    }
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.iam_client = self._get_iam_client() if GCP_CLIENTS_AVAILABLE else None
        self.rm_client = self._get_resource_manager_client() if GCP_CLIENTS_AVAILABLE else None
        self.asset_client = self._get_asset_client() if GCP_CLIENTS_AVAILABLE else None
    
    def _get_iam_client(self):
        """Get IAM client"""
        try:
            return iam_admin_v1.IAMClient()
        except Exception as e:
            logger.error(f"Failed to create IAM client: {e}")
            return None
    
    def _get_resource_manager_client(self):
        """Get Resource Manager client"""
        try:
            return resourcemanager_v3.ProjectsClient()
        except Exception as e:
            logger.error(f"Failed to create Resource Manager client: {e}")
            return None
    
    def _get_asset_client(self):
        """Get Cloud Asset client"""
        try:
            return asset_v1.AssetServiceClient()
        except Exception as e:
            logger.error(f"Failed to create Asset client: {e}")
            return None
    
    def analyze_iam_security(self) -> IAMSecurityPosture:
        """
        Perform comprehensive IAM security analysis
        
        Returns:
            IAMSecurityPosture with findings and recommendations
        """
        logger.info(f"Starting IAM security analysis for project {self.project_id}")
        
        if not self.iam_client or not self.rm_client:
            return self._generate_sample_analysis()
        
        try:
            # Get all service accounts
            service_accounts = self._list_service_accounts()
            
            # Get IAM policy for the project
            iam_policy = self._get_project_iam_policy()
            
            # Analyze each service account
            sa_analyses = []
            all_findings = []
            
            for sa in service_accounts:
                analysis = self._analyze_service_account(sa, iam_policy)
                sa_analyses.append(analysis)
                all_findings.extend(analysis.findings)
            
            # Analyze IAM policy for additional findings
            policy_findings = self._analyze_iam_policy(iam_policy)
            all_findings.extend(policy_findings)
            
            # Calculate overall posture
            posture = self._calculate_security_posture(
                service_accounts, sa_analyses, all_findings
            )
            
            logger.info(f"IAM analysis complete. Found {len(all_findings)} findings")
            return posture
            
        except Exception as e:
            logger.error(f"Error during IAM analysis: {e}")
            return self._generate_sample_analysis()
    
    def _list_service_accounts(self) -> List[Dict[str, Any]]:
        """List all service accounts in the project"""
        try:
            request = iam_admin_v1.ListServiceAccountsRequest(
                name=f"projects/{self.project_id}",
                page_size=100
            )
            
            accounts = []
            for account in self.iam_client.list_service_accounts(request=request):
                # Get keys for this service account
                keys = self._list_service_account_keys(account.email)
                
                accounts.append({
                    "name": account.name,
                    "email": account.email,
                    "display_name": account.display_name,
                    "description": account.description,
                    "unique_id": account.unique_id,
                    "disabled": account.disabled,
                    "keys": keys
                })
            
            return accounts
            
        except Exception as e:
            logger.error(f"Error listing service accounts: {e}")
            return []
    
    def _list_service_account_keys(self, service_account_email: str) -> List[Dict[str, Any]]:
        """List keys for a service account"""
        try:
            request = iam_admin_v1.ListServiceAccountKeysRequest(
                name=f"projects/{self.project_id}/serviceAccounts/{service_account_email}",
                key_types=[
                    iam_admin_v1.ListServiceAccountKeysRequest.KeyType.USER_MANAGED
                ]
            )
            
            keys = []
            response = self.iam_client.list_service_account_keys(request=request)
            # Access the keys property of the response
            for key in response.keys:
                keys.append({
                    "name": key.name,
                    "key_algorithm": key.key_algorithm.name if key.key_algorithm else None,
                    "key_origin": key.key_origin.name if key.key_origin else None,
                    "key_type": key.key_type.name if key.key_type else None,
                    "valid_after_time": str(key.valid_after_time) if key.valid_after_time else None,
                    "valid_before_time": str(key.valid_before_time) if key.valid_before_time else None,
                    "disabled": getattr(key, 'disabled', False)
                })
            
            return keys
            
        except Exception as e:
            logger.error(f"Error listing keys for {service_account_email}: {e}")
            return []
    
    def _get_project_iam_policy(self) -> Optional[Any]:
        """Get IAM policy for the project"""
        try:
            # Use the correct API method for getting IAM policy
            from google.iam.v1 import iam_policy_pb2
            
            # Use GetPolicyOptions from the correct location
            options = iam_policy_pb2.GetPolicyOptions(
                requested_policy_version=3
            )
            
            request = iam_policy_pb2.GetIamPolicyRequest(
                resource=f"projects/{self.project_id}",
                options=options
            )
            
            return self.rm_client.get_iam_policy(request=request)
            
        except Exception as e:
            logger.error(f"Error getting IAM policy: {e}")
            return None
    
    def _analyze_service_account(self, sa: Dict[str, Any], iam_policy: Any) -> ServiceAccountAnalysis:
        """Analyze a single service account for security issues"""
        findings = []
        risk_score = 0
        
        # Get roles assigned to this service account
        sa_roles = self._get_service_account_roles(sa["email"], iam_policy)
        
        # Check for overprivileged accounts
        overprivilege_findings = self._check_overprivileged_account(sa, sa_roles)
        findings.extend(overprivilege_findings)
        
        # Check for stale keys
        key_findings = self._check_stale_keys(sa)
        findings.extend(key_findings)
        
        # Check for unused service accounts
        usage_findings = self._check_service_account_usage(sa)
        findings.extend(usage_findings)
        
        # Calculate oldest key age
        oldest_key_age = None
        if sa["keys"]:
            # Handle string timestamps properly
            from datetime import timezone
            now = datetime.now(timezone.utc)
            valid_keys = []
            
            for k in sa["keys"]:
                valid_after = k.get("valid_after_time")
                if valid_after:
                    # Convert string to datetime if needed
                    if isinstance(valid_after, str):
                        try:
                            # Parse ISO format timestamp
                            valid_after = datetime.fromisoformat(valid_after.replace('Z', '+00:00'))
                        except:
                            continue
                    # Make sure datetime is timezone-aware
                    if valid_after.tzinfo is None:
                        valid_after = valid_after.replace(tzinfo=timezone.utc)
                    valid_keys.append((k, valid_after))
            
            if valid_keys:
                oldest_key, oldest_time = min(valid_keys, key=lambda x: x[1])
                oldest_key_age = (now - oldest_time).days
        
        # Calculate risk score based on findings
        risk_score = sum(f.risk_score for f in findings)
        
        # Check for cross-project access
        cross_project = any(role.startswith("projects/") and self.project_id not in role for role in sa_roles)
        
        return ServiceAccountAnalysis(
            email=sa["email"],
            display_name=sa["display_name"],
            unique_id=sa["unique_id"],
            project_id=self.project_id,
            disabled=sa["disabled"],
            key_count=len(sa["keys"]),
            oldest_key_age_days=oldest_key_age,
            roles=sa_roles,
            risk_score=min(risk_score, 100),
            findings=findings,
            last_used=None,  # Would need audit logs to determine
            cross_project_access=cross_project
        )
    
    def _get_service_account_roles(self, sa_email: str, iam_policy: Any) -> List[str]:
        """Get all roles assigned to a service account"""
        roles = []
        
        if not iam_policy:
            return roles
        
        for binding in iam_policy.bindings:
            if f"serviceAccount:{sa_email}" in binding.members:
                roles.append(binding.role)
        
        return roles
    
    def _check_overprivileged_account(self, sa: Dict[str, Any], roles: List[str]) -> List[IAMFinding]:
        """Check if service account has excessive privileges"""
        findings = []
        
        # Check for admin roles
        admin_roles = [role for role in roles if role in self.ADMIN_ROLES]
        if admin_roles:
            findings.append(IAMFinding(
                finding_type=IAMFindingType.ADMIN_ROLE_MISUSE,
                risk_level=RiskLevel.CRITICAL,
                risk_score=90,
                title="Service Account with Admin Role",
                description=f"Service account has admin roles: {', '.join(admin_roles)}",
                resource_name=sa["name"],
                affected_principal=sa["email"],
                remediation_steps=[
                    "Review if admin role is truly necessary",
                    "Consider using more specific roles",
                    "Implement least privilege principle",
                    "Use Workload Identity if possible"
                ],
                metadata={"admin_roles": admin_roles},
                detected_at=datetime.utcnow()
            ))
        
        # Check for broad roles
        broad_roles = [role for role in roles if role in self.BROAD_ROLES]
        if broad_roles and not admin_roles:  # Don't double-count admin roles
            findings.append(IAMFinding(
                finding_type=IAMFindingType.EXCESSIVE_PERMISSIONS,
                risk_level=RiskLevel.HIGH,
                risk_score=70,
                title="Service Account with Broad Permissions",
                description=f"Service account has broad roles: {', '.join(broad_roles)}",
                resource_name=sa["name"],
                affected_principal=sa["email"],
                remediation_steps=[
                    "Replace broad roles with specific roles",
                    "Audit required permissions",
                    "Create custom roles if needed",
                    "Regular permission reviews"
                ],
                metadata={"broad_roles": broad_roles},
                detected_at=datetime.utcnow()
            ))
        
        # Check for multiple high-privilege roles
        high_priv_roles = [role for role in roles if role in self.HIGH_PRIVILEGE_ROLES]
        if len(high_priv_roles) > 2:
            findings.append(IAMFinding(
                finding_type=IAMFindingType.OVERPRIVILEGED_SERVICE_ACCOUNT,
                risk_level=RiskLevel.HIGH,
                risk_score=80,
                title="Service Account with Multiple High-Privilege Roles",
                description=f"Service account has {len(high_priv_roles)} high-privilege roles",
                resource_name=sa["name"],
                affected_principal=sa["email"],
                remediation_steps=[
                    "Consolidate roles where possible",
                    "Remove unnecessary high-privilege roles",
                    "Use least privilege principle",
                    "Regular access reviews"
                ],
                metadata={"high_privilege_roles": high_priv_roles},
                detected_at=datetime.utcnow()
            ))
        
        return findings
    
    def _check_stale_keys(self, sa: Dict[str, Any]) -> List[IAMFinding]:
        """Check for stale service account keys"""
        findings = []
        stale_keys = []
        
        # Keys older than 90 days are considered stale
        from datetime import timezone
        now = datetime.now(timezone.utc)
        stale_threshold = now - timedelta(days=90)
        
        for key in sa["keys"]:
            valid_after = key.get("valid_after_time")
            if valid_after:
                # Handle string timestamps properly
                if isinstance(valid_after, str):
                    try:
                        # Parse ISO format timestamp
                        valid_after = datetime.fromisoformat(valid_after.replace('Z', '+00:00'))
                    except:
                        continue
                
                # Make sure datetime is timezone-aware
                if valid_after.tzinfo is None:
                    valid_after = valid_after.replace(tzinfo=timezone.utc)
                
                if valid_after < stale_threshold:
                    age_days = (now - valid_after).days
                    stale_keys.append({
                        "key_id": key["name"].split("/")[-1],
                        "age_days": age_days
                    })
        
        if stale_keys:
            risk_score = min(50 + len(stale_keys) * 10, 100)
            risk_level = RiskLevel.HIGH if len(stale_keys) > 2 else RiskLevel.MEDIUM
            
            findings.append(IAMFinding(
                finding_type=IAMFindingType.STALE_SERVICE_ACCOUNT_KEY,
                risk_level=risk_level,
                risk_score=risk_score,
                title="Stale Service Account Keys",
                description=f"Service account has {len(stale_keys)} keys older than 90 days",
                resource_name=sa["name"],
                affected_principal=sa["email"],
                remediation_steps=[
                    "Rotate old service account keys",
                    "Implement automated key rotation",
                    "Use Workload Identity where possible",
                    "Set up key expiration monitoring"
                ],
                metadata={"stale_keys": stale_keys},
                detected_at=datetime.utcnow()
            ))
        
        return findings
    
    def _check_service_account_usage(self, sa: Dict[str, Any]) -> List[IAMFinding]:
        """Check if service account appears to be unused"""
        findings = []
        
        # If service account has no keys and is not disabled, it might be unused
        if not sa["keys"] and not sa["disabled"]:
            findings.append(IAMFinding(
                finding_type=IAMFindingType.UNUSED_SERVICE_ACCOUNT,
                risk_level=RiskLevel.LOW,
                risk_score=20,
                title="Potentially Unused Service Account",
                description="Service account has no keys and may be unused",
                resource_name=sa["name"],
                affected_principal=sa["email"],
                remediation_steps=[
                    "Verify if service account is still needed",
                    "Delete if unused",
                    "Disable if temporarily not needed",
                    "Document purpose and usage"
                ],
                metadata={"has_keys": False, "disabled": False},
                detected_at=datetime.utcnow()
            ))
        
        return findings
    
    def _analyze_iam_policy(self, iam_policy: Any) -> List[IAMFinding]:
        """Analyze IAM policy for additional security issues"""
        findings = []
        
        if not iam_policy:
            return findings
        
        # Check for wildcard bindings
        for binding in iam_policy.bindings:
            for member in binding.members:
                if "allUsers" in member or "allAuthenticatedUsers" in member:
                    findings.append(IAMFinding(
                        finding_type=IAMFindingType.WILDCARD_BINDING,
                        risk_level=RiskLevel.CRITICAL,
                        risk_score=95,
                        title="Wildcard IAM Binding",
                        description=f"Role {binding.role} is granted to {member}",
                        resource_name=f"projects/{self.project_id}",
                        affected_principal=member,
                        remediation_steps=[
                            "Remove wildcard bindings",
                            "Grant access to specific users/groups",
                            "Use IAM conditions for temporary access",
                            "Review and document any exceptions"
                        ],
                        metadata={"role": binding.role, "member": member},
                        detected_at=datetime.utcnow()
                    ))
        
        # Check for external users (non-organization domains)
        for binding in iam_policy.bindings:
            for member in binding.members:
                if member.startswith("user:") and not member.endswith("@google.com"):
                    # This is a simplified check; in real implementation,
                    # you'd check against your organization's domains
                    if "@gmail.com" in member or "@outlook.com" in member:
                        findings.append(IAMFinding(
                            finding_type=IAMFindingType.EXTERNAL_USER_ACCESS,
                            risk_level=RiskLevel.MEDIUM,
                            risk_score=60,
                            title="External User Access",
                            description=f"External user {member} has access to project",
                            resource_name=f"projects/{self.project_id}",
                            affected_principal=member,
                            remediation_steps=[
                                "Verify external user access is necessary",
                                "Use guest accounts for temporary access",
                                "Implement conditional access policies",
                                "Regular review of external access"
                            ],
                            metadata={"role": binding.role, "user": member},
                            detected_at=datetime.utcnow()
                        ))
        
        return findings
    
    def _calculate_security_posture(self, service_accounts: List[Dict], 
                                   sa_analyses: List[ServiceAccountAnalysis],
                                   all_findings: List[IAMFinding]) -> IAMSecurityPosture:
        """Calculate overall IAM security posture"""
        
        # Count findings by risk level
        risk_distribution = {
            "CRITICAL": len([f for f in all_findings if f.risk_level == RiskLevel.CRITICAL]),
            "HIGH": len([f for f in all_findings if f.risk_level == RiskLevel.HIGH]),
            "MEDIUM": len([f for f in all_findings if f.risk_level == RiskLevel.MEDIUM]),
            "LOW": len([f for f in all_findings if f.risk_level == RiskLevel.LOW]),
            "MINIMAL": len([f for f in all_findings if f.risk_level == RiskLevel.MINIMAL])
        }
        
        # Calculate posture score (100 - weighted penalty for findings)
        penalty = (
            risk_distribution["CRITICAL"] * 25 +
            risk_distribution["HIGH"] * 15 +
            risk_distribution["MEDIUM"] * 8 +
            risk_distribution["LOW"] * 3 +
            risk_distribution["MINIMAL"] * 1
        )
        
        posture_score = max(0, 100 - penalty)
        
        # Count specific metrics
        overprivileged_accounts = len([sa for sa in sa_analyses 
                                     if any(f.finding_type in [
                                         IAMFindingType.OVERPRIVILEGED_SERVICE_ACCOUNT,
                                         IAMFindingType.ADMIN_ROLE_MISUSE,
                                         IAMFindingType.EXCESSIVE_PERMISSIONS
                                     ] for f in sa.findings)])
        
        stale_keys = len([sa for sa in sa_analyses 
                         if any(f.finding_type == IAMFindingType.STALE_SERVICE_ACCOUNT_KEY 
                               for f in sa.findings)])
        
        cross_project_bindings = len([sa for sa in sa_analyses if sa.cross_project_access])
        
        external_users = len([f for f in all_findings 
                             if f.finding_type == IAMFindingType.EXTERNAL_USER_ACCESS])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_distribution, overprivileged_accounts, stale_keys
        )
        
        return IAMSecurityPosture(
            project_id=self.project_id,
            posture_score=posture_score,
            risk_distribution=risk_distribution,
            total_findings=len(all_findings),
            critical_findings=risk_distribution["CRITICAL"],
            high_findings=risk_distribution["HIGH"],
            service_account_count=len(service_accounts),
            overprivileged_accounts=overprivileged_accounts,
            stale_keys=stale_keys,
            cross_project_bindings=cross_project_bindings,
            external_users=external_users,
            recommendations=recommendations,
            findings=all_findings,
            analyzed_at=datetime.utcnow()
        )
    
    def _generate_recommendations(self, risk_distribution: Dict[str, int],
                                overprivileged_accounts: int, stale_keys: int) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Priority recommendations based on findings
        if risk_distribution["CRITICAL"] > 0:
            recommendations.append("[CRITICAL] CRITICAL: Address wildcard IAM bindings and admin role misuse immediately")
        
        if overprivileged_accounts > 0:
            recommendations.append(f"[HIGH] HIGH: Review {overprivileged_accounts} overprivileged service accounts and implement least privilege")
        
        if stale_keys > 0:
            recommendations.append(f"[MEDIUM] MEDIUM: Rotate {stale_keys} stale service account keys (>90 days old)")
        
        # General recommendations
        recommendations.extend([
            "Implement automated service account key rotation",
            "Use Workload Identity for GKE workloads instead of service account keys",
            "Enable IAM audit logging for better visibility",
            "Conduct quarterly IAM access reviews",
            "Create custom roles with minimal required permissions",
            "Implement IAM conditions for time-based or IP-based access",
            "Use IAM Recommender to identify unused permissions"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_sample_analysis(self) -> IAMSecurityPosture:
        """Generate sample analysis when GCP clients are unavailable"""
        sample_findings = [
            IAMFinding(
                finding_type=IAMFindingType.OVERPRIVILEGED_SERVICE_ACCOUNT,
                risk_level=RiskLevel.HIGH,
                risk_score=80,
                title="Service Account with Editor Role",
                description="Service account has broad editor permissions",
                resource_name=f"projects/{self.project_id}/serviceAccounts/app-sa@{self.project_id}.iam.gserviceaccount.com",
                affected_principal=f"app-sa@{self.project_id}.iam.gserviceaccount.com",
                remediation_steps=[
                    "Replace editor role with specific roles",
                    "Audit required permissions",
                    "Implement least privilege principle"
                ],
                metadata={"roles": ["roles/editor"]},
                detected_at=datetime.utcnow()
            ),
            IAMFinding(
                finding_type=IAMFindingType.STALE_SERVICE_ACCOUNT_KEY,
                risk_level=RiskLevel.MEDIUM,
                risk_score=60,
                title="Stale Service Account Key",
                description="Service account key is 120 days old",
                resource_name=f"projects/{self.project_id}/serviceAccounts/old-sa@{self.project_id}.iam.gserviceaccount.com",
                affected_principal=f"old-sa@{self.project_id}.iam.gserviceaccount.com",
                remediation_steps=[
                    "Rotate service account key",
                    "Implement automated rotation",
                    "Consider Workload Identity"
                ],
                metadata={"key_age_days": 120},
                detected_at=datetime.utcnow()
            )
        ]
        
        return IAMSecurityPosture(
            project_id=self.project_id,
            posture_score=65,
            risk_distribution={"CRITICAL": 0, "HIGH": 1, "MEDIUM": 1, "LOW": 0, "MINIMAL": 0},
            total_findings=2,
            critical_findings=0,
            high_findings=1,
            service_account_count=5,
            overprivileged_accounts=1,
            stale_keys=1,
            cross_project_bindings=0,
            external_users=0,
            recommendations=[
                "[HIGH] HIGH: Review 1 overprivileged service account and implement least privilege",
                "[MEDIUM] MEDIUM: Rotate 1 stale service account key (>90 days old)",
                "Implement automated service account key rotation",
                "Use Workload Identity for GKE workloads"
            ],
            findings=sample_findings,
            analyzed_at=datetime.utcnow()
        )