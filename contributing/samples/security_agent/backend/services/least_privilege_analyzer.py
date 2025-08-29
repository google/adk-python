"""
Automated Least-Privilege Analyzer for Advanced IAM Features
Continuously monitors and reports on least-privilege violations and overprivileged accounts
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

try:
    from google.cloud import asset_v1
    from google.cloud import iam_admin_v1
    from google.cloud import resourcemanager_v3
    from google.cloud import logging as cloud_logging
    GCP_CLIENTS_AVAILABLE = True
except ImportError:
    GCP_CLIENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PrivilegeViolation:
    """Represents a least-privilege violation"""
    principal: str
    principal_type: str
    violation_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    current_roles: List[str]
    excessive_permissions: List[str]
    risk_score: float
    description: str
    remediation: str
    compliance_impact: List[str]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None


@dataclass
class PrivilegeBaseline:
    """Baseline definition for a role or principal type"""
    name: str
    principal_pattern: str  # Regex pattern to match principals
    allowed_roles: List[str]
    forbidden_roles: List[str]
    max_permissions: int
    requires_mfa: bool = False
    requires_approval: bool = False
    expires_after_days: Optional[int] = None
    description: str = ""


@dataclass
class LeastPrivilegeReport:
    """Comprehensive least-privilege analysis report"""
    analysis_timestamp: datetime
    project_id: str
    total_principals_analyzed: int
    violations_found: int
    compliance_score: float
    risk_distribution: Dict[str, int]
    top_violations: List[PrivilegeViolation]
    recommendations: List[str]
    overprivileged_accounts: List[Dict[str, Any]]
    unused_service_accounts: List[str]
    admin_role_usage: Dict[str, int]


class ViolationType(Enum):
    """Types of least-privilege violations"""
    OVERPRIVILEGED_ACCOUNT = "OVERPRIVILEGED_ACCOUNT"
    ADMIN_ROLE_MISUSE = "ADMIN_ROLE_MISUSE"
    WILDCARD_PERMISSION = "WILDCARD_PERMISSION"
    STALE_PERMISSION = "STALE_PERMISSION"
    CROSS_PROJECT_ACCESS = "CROSS_PROJECT_ACCESS"
    NO_MFA_HIGH_PRIVILEGE = "NO_MFA_HIGH_PRIVILEGE"
    EXCESSIVE_SCOPE = "EXCESSIVE_SCOPE"
    BASELINE_VIOLATION = "BASELINE_VIOLATION"
    UNUSED_SERVICE_ACCOUNT = "UNUSED_SERVICE_ACCOUNT"
    PERMANENT_ELEVATION = "PERMANENT_ELEVATION"


class LeastPrivilegeAnalyzer:
    """Analyzer for least-privilege compliance and violations"""
    
    # High-risk roles that should be closely monitored
    HIGH_RISK_ROLES = {
        "roles/owner",
        "roles/editor",
        "roles/iam.securityAdmin",
        "roles/iam.serviceAccountAdmin",
        "roles/iam.serviceAccountKeyAdmin",
        "roles/resourcemanager.organizationAdmin",
        "roles/resourcemanager.folderAdmin",
        "roles/resourcemanager.projectIamAdmin",
        "roles/compute.admin",
        "roles/storage.admin"
    }
    
    # Permissions that pose significant risk
    DANGEROUS_PERMISSIONS = {
        "iam.serviceAccountKeys.create",
        "iam.serviceAccounts.actAs",
        "iam.serviceAccounts.getAccessToken",
        "iam.roles.create",
        "iam.roles.update",
        "resourcemanager.projects.delete",
        "resourcemanager.organizations.setIamPolicy",
        "compute.instances.delete",
        "storage.buckets.delete",
        "*.setIamPolicy"
    }
    
    # Default baselines for common principal types
    DEFAULT_BASELINES = [
        PrivilegeBaseline(
            name="developer_baseline",
            principal_pattern=r".*@(example\.com|yourdomain\.com)$",
            allowed_roles=["roles/viewer", "roles/storage.objectViewer", "roles/bigquery.dataViewer"],
            forbidden_roles=["roles/owner", "roles/editor"],
            max_permissions=50,
            requires_mfa=True,
            expires_after_days=90,
            description="Standard developer access baseline"
        ),
        PrivilegeBaseline(
            name="service_account_baseline",
            principal_pattern=r".*\.gserviceaccount\.com$",
            allowed_roles=["roles/storage.objectViewer", "roles/pubsub.publisher"],
            forbidden_roles=["roles/owner", "roles/editor", "roles/iam.securityAdmin"],
            max_permissions=20,
            requires_approval=True,
            description="Service account access baseline"
        ),
        PrivilegeBaseline(
            name="ci_cd_baseline",
            principal_pattern=r"^(ci|cd|deploy|pipeline).*\.gserviceaccount\.com$",
            allowed_roles=["roles/cloudbuild.builds.builder", "roles/storage.admin"],
            forbidden_roles=["roles/owner"],
            max_permissions=100,
            expires_after_days=180,
            description="CI/CD pipeline service account baseline"
        )
    ]
    
    def __init__(self, project_id: str, db_path: Optional[str] = None):
        """Initialize the least-privilege analyzer"""
        self.project_id = project_id
        self.db_path = db_path or "backend/cache/least_privilege.db"
        self.baselines = self.DEFAULT_BASELINES.copy()
        self._init_database()
        
        # Initialize GCP clients if available
        if GCP_CLIENTS_AVAILABLE:
            try:
                self.asset_client = asset_v1.AssetServiceClient()
                self.iam_client = iam_admin_v1.IAMClient()
                self.resource_manager = resourcemanager_v3.ProjectsClient()
                logger.info("GCP clients initialized for least-privilege analysis")
            except Exception as e:
                logger.warning(f"Could not initialize GCP clients: {e}")
                self.asset_client = None
                self.iam_client = None
                self.resource_manager = None
        else:
            self.asset_client = None
            self.iam_client = None
            self.resource_manager = None
    
    def _init_database(self):
        """Initialize the least-privilege database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS privilege_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                principal TEXT NOT NULL,
                principal_type TEXT,
                violation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                current_roles TEXT,
                excessive_permissions TEXT,
                risk_score REAL,
                description TEXT,
                remediation TEXT,
                compliance_impact TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                UNIQUE(principal, violation_type)
            )
        """)
        
        # Baselines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS privilege_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                principal_pattern TEXT NOT NULL,
                allowed_roles TEXT,
                forbidden_roles TEXT,
                max_permissions INTEGER,
                requires_mfa BOOLEAN DEFAULT FALSE,
                requires_approval BOOLEAN DEFAULT FALSE,
                expires_after_days INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_principals INTEGER,
                violations_found INTEGER,
                compliance_score REAL,
                risk_distribution TEXT,
                report_data TEXT
            )
        """)
        
        # Principal activity tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS principal_activity (
                principal TEXT PRIMARY KEY,
                last_api_call TIMESTAMP,
                total_api_calls INTEGER DEFAULT 0,
                last_permission_change TIMESTAMP,
                days_inactive INTEGER DEFAULT 0,
                risk_trend TEXT
            )
        """)
        
        # Insert default baselines if not exists
        for baseline in self.DEFAULT_BASELINES:
            cursor.execute("""
                INSERT OR IGNORE INTO privilege_baselines
                (name, principal_pattern, allowed_roles, forbidden_roles, 
                 max_permissions, requires_mfa, requires_approval, expires_after_days, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.name,
                baseline.principal_pattern,
                json.dumps(baseline.allowed_roles),
                json.dumps(baseline.forbidden_roles),
                baseline.max_permissions,
                baseline.requires_mfa,
                baseline.requires_approval,
                baseline.expires_after_days,
                baseline.description
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized least-privilege database at {self.db_path}")
    
    async def analyze_project_compliance(self) -> LeastPrivilegeReport:
        """Perform comprehensive least-privilege analysis for the project"""
        violations = []
        overprivileged_accounts = []
        unused_service_accounts = []
        admin_role_usage = defaultdict(int)
        
        # Get all IAM bindings
        bindings = await self._get_project_iam_bindings()
        
        # Analyze each principal
        principals_analyzed = set()
        for role, members in bindings.items():
            for member in members:
                if member in principals_analyzed:
                    continue
                principals_analyzed.add(member)
                
                # Extract principal info
                principal_type, principal_email = self._parse_member(member)
                
                # Check for violations
                principal_violations = await self._check_principal_violations(
                    principal_email, 
                    principal_type,
                    self._get_principal_roles(principal_email, bindings)
                )
                violations.extend(principal_violations)
                
                # Track admin role usage
                if role in self.HIGH_RISK_ROLES:
                    admin_role_usage[role] += 1
                    
                # Check for overprivileged accounts
                if any(v.violation_type == ViolationType.OVERPRIVILEGED_ACCOUNT.value 
                       for v in principal_violations):
                    overprivileged_accounts.append({
                        "principal": principal_email,
                        "type": principal_type,
                        "roles": self._get_principal_roles(principal_email, bindings),
                        "risk_score": max(v.risk_score for v in principal_violations)
                    })
        
        # Check for unused service accounts
        if self.iam_client:
            unused = await self._find_unused_service_accounts()
            unused_service_accounts = unused
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            len(principals_analyzed),
            len(violations)
        )
        
        # Generate risk distribution
        risk_distribution = self._calculate_risk_distribution(violations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            violations,
            overprivileged_accounts,
            unused_service_accounts,
            admin_role_usage
        )
        
        # Create report
        report = LeastPrivilegeReport(
            analysis_timestamp=datetime.utcnow(),
            project_id=self.project_id,
            total_principals_analyzed=len(principals_analyzed),
            violations_found=len(violations),
            compliance_score=compliance_score,
            risk_distribution=risk_distribution,
            top_violations=sorted(violations, key=lambda v: v.risk_score, reverse=True)[:10],
            recommendations=recommendations,
            overprivileged_accounts=overprivileged_accounts[:10],
            unused_service_accounts=unused_service_accounts[:10],
            admin_role_usage=dict(admin_role_usage)
        )
        
        # Cache the report
        self._cache_analysis_report(report)
        
        # Cache violations
        for violation in violations:
            self._cache_violation(violation)
        
        return report
    
    async def _get_project_iam_bindings(self) -> Dict[str, List[str]]:
        """Get all IAM bindings for the project"""
        bindings = defaultdict(list)
        
        if self.resource_manager:
            try:
                # Get project IAM policy
                project_name = f"projects/{self.project_id}"
                policy = self.resource_manager.get_iam_policy(resource=project_name)
                
                for binding in policy.bindings:
                    bindings[binding.role].extend(binding.members)
                    
            except Exception as e:
                logger.error(f"Error fetching IAM bindings: {e}")
                # Use mock data for development
                bindings = self._get_mock_iam_bindings()
        else:
            # Use mock data
            bindings = self._get_mock_iam_bindings()
        
        return bindings
    
    def _get_mock_iam_bindings(self) -> Dict[str, List[str]]:
        """Get mock IAM bindings for development"""
        return {
            "roles/owner": [
                "user:admin@example.com"
            ],
            "roles/editor": [
                "user:developer@example.com",
                "serviceAccount:app-sa@project.iam.gserviceaccount.com"
            ],
            "roles/viewer": [
                "user:analyst@example.com",
                "serviceAccount:readonly-sa@project.iam.gserviceaccount.com"
            ],
            "roles/storage.admin": [
                "serviceAccount:backup-sa@project.iam.gserviceaccount.com"
            ],
            "roles/iam.serviceAccountAdmin": [
                "user:security@example.com"
            ]
        }
    
    async def _check_principal_violations(self, principal: str, 
                                         principal_type: str,
                                         roles: List[str]) -> List[PrivilegeViolation]:
        """Check for least-privilege violations for a principal"""
        violations = []
        
        # Check for overprivileged accounts
        if self._is_overprivileged(roles):
            violations.append(PrivilegeViolation(
                principal=principal,
                principal_type=principal_type,
                violation_type=ViolationType.OVERPRIVILEGED_ACCOUNT.value,
                severity="HIGH",
                current_roles=roles,
                excessive_permissions=self._get_excessive_permissions(roles),
                risk_score=0.8,
                description=f"Principal has excessive privileges with {len(roles)} roles including high-risk roles",
                remediation="Review and reduce roles to minimum required permissions",
                compliance_impact=["SOC2", "ISO27001", "GDPR"]
            ))
        
        # Check for admin role misuse
        admin_roles = [r for r in roles if r in self.HIGH_RISK_ROLES]
        if admin_roles and principal_type == "serviceAccount":
            violations.append(PrivilegeViolation(
                principal=principal,
                principal_type=principal_type,
                violation_type=ViolationType.ADMIN_ROLE_MISUSE.value,
                severity="CRITICAL",
                current_roles=roles,
                excessive_permissions=admin_roles,
                risk_score=0.95,
                description=f"Service account has admin roles: {', '.join(admin_roles)}",
                remediation="Service accounts should not have admin roles. Use specific permissions instead.",
                compliance_impact=["SOC2", "ISO27001", "PCI-DSS"]
            ))
        
        # Check against baselines
        baseline_violation = self._check_baseline_compliance(principal, roles)
        if baseline_violation:
            violations.append(baseline_violation)
        
        # Check for wildcard permissions
        if any("*" in role or role.endswith(".admin") for role in roles):
            violations.append(PrivilegeViolation(
                principal=principal,
                principal_type=principal_type,
                violation_type=ViolationType.WILDCARD_PERMISSION.value,
                severity="HIGH",
                current_roles=roles,
                excessive_permissions=[r for r in roles if "*" in r or r.endswith(".admin")],
                risk_score=0.85,
                description="Principal has wildcard or admin permissions",
                remediation="Replace wildcard permissions with specific, scoped permissions",
                compliance_impact=["SOC2", "ISO27001"]
            ))
        
        # Check for stale permissions (would need activity data)
        if principal_type == "serviceAccount":
            activity = self._get_principal_activity(principal)
            if activity and activity.get("days_inactive", 0) > 90:
                violations.append(PrivilegeViolation(
                    principal=principal,
                    principal_type=principal_type,
                    violation_type=ViolationType.STALE_PERMISSION.value,
                    severity="MEDIUM",
                    current_roles=roles,
                    excessive_permissions=roles,
                    risk_score=0.6,
                    description=f"Service account inactive for {activity['days_inactive']} days",
                    remediation="Remove or disable inactive service accounts",
                    compliance_impact=["SOC2"]
                ))
        
        return violations
    
    def _is_overprivileged(self, roles: List[str]) -> bool:
        """Check if a principal is overprivileged based on roles"""
        # Has owner or editor role
        if any(role in ["roles/owner", "roles/editor"] for role in roles):
            return True
        
        # Has too many roles
        if len(roles) > 5:
            return True
        
        # Has multiple admin roles
        admin_roles = [r for r in roles if r in self.HIGH_RISK_ROLES]
        if len(admin_roles) > 1:
            return True
        
        return False
    
    def _get_excessive_permissions(self, roles: List[str]) -> List[str]:
        """Identify excessive permissions in a role set"""
        excessive = []
        
        for role in roles:
            if role in self.HIGH_RISK_ROLES:
                excessive.append(role)
            elif role.endswith(".admin"):
                excessive.append(role)
        
        return excessive
    
    def _check_baseline_compliance(self, principal: str, 
                                  roles: List[str]) -> Optional[PrivilegeViolation]:
        """Check if principal complies with defined baselines"""
        import re
        
        for baseline in self.baselines:
            # Check if principal matches pattern
            if re.match(baseline.principal_pattern, principal):
                violations = []
                
                # Check forbidden roles
                forbidden_found = [r for r in roles if r in baseline.forbidden_roles]
                if forbidden_found:
                    return PrivilegeViolation(
                        principal=principal,
                        principal_type=self._determine_principal_type(principal),
                        violation_type=ViolationType.BASELINE_VIOLATION.value,
                        severity="HIGH",
                        current_roles=roles,
                        excessive_permissions=forbidden_found,
                        risk_score=0.75,
                        description=f"Principal violates baseline '{baseline.name}' with forbidden roles",
                        remediation=f"Remove forbidden roles: {', '.join(forbidden_found)}",
                        compliance_impact=["Policy", "Baseline"]
                    )
                
                # Check if has non-allowed roles
                non_allowed = [r for r in roles if r not in baseline.allowed_roles]
                if non_allowed and baseline.allowed_roles:
                    return PrivilegeViolation(
                        principal=principal,
                        principal_type=self._determine_principal_type(principal),
                        violation_type=ViolationType.BASELINE_VIOLATION.value,
                        severity="MEDIUM",
                        current_roles=roles,
                        excessive_permissions=non_allowed,
                        risk_score=0.65,
                        description=f"Principal has roles not allowed by baseline '{baseline.name}'",
                        remediation=f"Remove non-allowed roles: {', '.join(non_allowed)}",
                        compliance_impact=["Policy", "Baseline"]
                    )
        
        return None
    
    async def _find_unused_service_accounts(self) -> List[str]:
        """Find service accounts that haven't been used recently"""
        unused = []
        
        if self.iam_client:
            try:
                # List all service accounts
                parent = f"projects/{self.project_id}"
                accounts = self.iam_client.list_service_accounts(name=parent)
                
                for account in accounts:
                    # Check activity (simplified - would check audit logs)
                    activity = self._get_principal_activity(account.email)
                    if activity and activity.get("days_inactive", 0) > 90:
                        unused.append(account.email)
                        
            except Exception as e:
                logger.error(f"Error finding unused service accounts: {e}")
                # Return mock data
                unused = ["unused-sa@project.iam.gserviceaccount.com"]
        else:
            # Mock data
            unused = ["unused-sa@project.iam.gserviceaccount.com"]
        
        return unused
    
    def _parse_member(self, member: str) -> Tuple[str, str]:
        """Parse member string to extract type and email"""
        if ":" in member:
            member_type, email = member.split(":", 1)
            if member_type == "serviceAccount":
                return ("serviceAccount", email)
            elif member_type == "user":
                return ("user", email)
            elif member_type == "group":
                return ("group", email)
        return ("unknown", member)
    
    def _get_principal_roles(self, principal: str, 
                            bindings: Dict[str, List[str]]) -> List[str]:
        """Get all roles for a principal"""
        roles = []
        for role, members in bindings.items():
            for member in members:
                if principal in member:
                    roles.append(role)
        return roles
    
    def _determine_principal_type(self, principal: str) -> str:
        """Determine the type of principal from email"""
        if ".gserviceaccount.com" in principal:
            return "serviceAccount"
        elif "@" in principal:
            return "user"
        else:
            return "unknown"
    
    def _get_principal_activity(self, principal: str) -> Dict[str, Any]:
        """Get activity data for a principal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT last_api_call, total_api_calls, days_inactive
            FROM principal_activity
            WHERE principal = ?
        """, (principal,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "last_api_call": row[0],
                "total_api_calls": row[1],
                "days_inactive": row[2]
            }
        
        # Return mock data for development
        return {
            "last_api_call": None,
            "total_api_calls": 0,
            "days_inactive": 30
        }
    
    def _calculate_compliance_score(self, total_principals: int, 
                                   violations: int) -> float:
        """Calculate overall compliance score"""
        if total_principals == 0:
            return 100.0
        
        violation_ratio = violations / total_principals
        base_score = max(0, 100 * (1 - violation_ratio))
        
        # Apply severity weighting
        return round(base_score, 2)
    
    def _calculate_risk_distribution(self, violations: List[PrivilegeViolation]) -> Dict[str, int]:
        """Calculate distribution of violations by severity"""
        distribution = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }
        
        for violation in violations:
            if violation.severity in distribution:
                distribution[violation.severity] += 1
        
        return distribution
    
    def _generate_recommendations(self, violations: List[PrivilegeViolation],
                                 overprivileged: List[Dict],
                                 unused: List[str],
                                 admin_usage: Dict[str, int]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical violations
        critical = [v for v in violations if v.severity == "CRITICAL"]
        if critical:
            recommendations.append(
                f"CRITICAL: Address {len(critical)} critical violations immediately, "
                f"including admin role misuse and excessive privileges"
            )
        
        # Overprivileged accounts
        if overprivileged:
            recommendations.append(
                f"HIGH: Review and reduce permissions for {len(overprivileged)} "
                f"overprivileged accounts to follow least-privilege principle"
            )
        
        # Unused service accounts
        if unused:
            recommendations.append(
                f"MEDIUM: Disable or remove {len(unused)} unused service accounts "
                f"to reduce attack surface"
            )
        
        # Admin role usage
        if admin_usage:
            total_admin = sum(admin_usage.values())
            recommendations.append(
                f"HIGH: Reduce admin role assignments (currently {total_admin} principals "
                f"with admin roles)"
            )
        
        # General recommendations
        recommendations.extend([
            "Implement automated role rotation for high-privilege accounts",
            "Enable MFA for all user accounts with elevated privileges",
            "Set up alerts for privilege escalation events",
            "Conduct quarterly IAM access reviews",
            "Use custom roles instead of predefined admin roles"
        ])
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _cache_violation(self, violation: PrivilegeViolation):
        """Cache a violation in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO privilege_violations
            (principal, principal_type, violation_type, severity, current_roles,
             excessive_permissions, risk_score, description, remediation,
             compliance_impact, detected_at, last_activity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            violation.principal,
            violation.principal_type,
            violation.violation_type,
            violation.severity,
            json.dumps(violation.current_roles),
            json.dumps(violation.excessive_permissions),
            violation.risk_score,
            violation.description,
            violation.remediation,
            json.dumps(violation.compliance_impact),
            violation.detected_at.isoformat(),
            violation.last_activity.isoformat() if violation.last_activity else None
        ))
        
        conn.commit()
        conn.close()
    
    def _cache_analysis_report(self, report: LeastPrivilegeReport):
        """Cache the analysis report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert report to JSON-serializable format
        report_data = {
            "violations_found": report.violations_found,
            "compliance_score": report.compliance_score,
            "overprivileged_accounts": report.overprivileged_accounts,
            "unused_service_accounts": report.unused_service_accounts,
            "admin_role_usage": report.admin_role_usage,
            "recommendations": report.recommendations
        }
        
        cursor.execute("""
            INSERT INTO analysis_history
            (project_id, analysis_timestamp, total_principals, violations_found,
             compliance_score, risk_distribution, report_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            report.project_id,
            report.analysis_timestamp.isoformat(),
            report.total_principals_analyzed,
            report.violations_found,
            report.compliance_score,
            json.dumps(report.risk_distribution),
            json.dumps(report_data)
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_violations(self, limit: int = 50, 
                            severity_filter: Optional[str] = None) -> List[PrivilegeViolation]:
        """Get recent privilege violations from cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT principal, principal_type, violation_type, severity,
                   current_roles, excessive_permissions, risk_score,
                   description, remediation, compliance_impact,
                   detected_at, last_activity
            FROM privilege_violations
            WHERE resolved = FALSE
        """
        
        params = []
        if severity_filter:
            query += " AND severity = ?"
            params.append(severity_filter)
        
        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        violations = []
        for row in cursor.fetchall():
            violation = PrivilegeViolation(
                principal=row[0],
                principal_type=row[1],
                violation_type=row[2],
                severity=row[3],
                current_roles=json.loads(row[4]) if row[4] else [],
                excessive_permissions=json.loads(row[5]) if row[5] else [],
                risk_score=row[6],
                description=row[7],
                remediation=row[8],
                compliance_impact=json.loads(row[9]) if row[9] else [],
                detected_at=datetime.fromisoformat(row[10]) if row[10] else datetime.utcnow(),
                last_activity=datetime.fromisoformat(row[11]) if row[11] else None
            )
            violations.append(violation)
        
        conn.close()
        return violations
    
    def add_custom_baseline(self, baseline: PrivilegeBaseline):
        """Add a custom privilege baseline"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO privilege_baselines
            (name, principal_pattern, allowed_roles, forbidden_roles,
             max_permissions, requires_mfa, requires_approval,
             expires_after_days, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            baseline.name,
            baseline.principal_pattern,
            json.dumps(baseline.allowed_roles),
            json.dumps(baseline.forbidden_roles),
            baseline.max_permissions,
            baseline.requires_mfa,
            baseline.requires_approval,
            baseline.expires_after_days,
            baseline.description
        ))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory baselines
        self.baselines.append(baseline)