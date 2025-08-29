"""
Cross-Project Permission Analyzer for Advanced IAM Features
Analyzes permissions that span multiple projects, including inherited and delegated access
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
    from google.cloud import resourcemanager_v3
    from google.cloud import iam_admin_v1
    from google.cloud import logging as cloud_logging
    GCP_CLIENTS_AVAILABLE = True
except ImportError:
    GCP_CLIENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CrossProjectAccess:
    """Represents cross-project access for a principal"""
    principal: str
    principal_type: str
    source_project: str
    target_project: str
    access_type: str  # DIRECT, INHERITED, DELEGATED, IMPERSONATION
    roles: List[str]
    permissions: List[str]
    resource_path: str
    inheritance_chain: List[str]
    risk_level: str
    compliance_flags: List[str]
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None


@dataclass
class ProjectHierarchy:
    """Represents organizational hierarchy for a project"""
    project_id: str
    project_name: str
    organization_id: Optional[str]
    folder_ids: List[str]
    parent_path: str
    inherited_bindings: Dict[str, List[str]]
    effective_permissions: Dict[str, Set[str]]


@dataclass
class CrossProjectReport:
    """Comprehensive cross-project permission analysis report"""
    analysis_timestamp: datetime
    projects_analyzed: List[str]
    total_cross_project_accesses: int
    high_risk_accesses: int
    inheritance_depth_stats: Dict[str, int]
    delegation_chains: List[Dict[str, Any]]
    service_account_impersonations: List[Dict[str, Any]]
    compliance_violations: List[str]
    recommendations: List[str]
    access_matrix: Dict[str, Dict[str, List[str]]]  # principal -> project -> roles


class AccessType(Enum):
    """Types of cross-project access"""
    DIRECT = "DIRECT"
    INHERITED = "INHERITED"
    DELEGATED = "DELEGATED"
    IMPERSONATION = "IMPERSONATION"
    GROUP_MEMBERSHIP = "GROUP_MEMBERSHIP"
    SERVICE_ACCOUNT_KEY = "SERVICE_ACCOUNT_KEY"
    WORKLOAD_IDENTITY = "WORKLOAD_IDENTITY"


class CrossProjectAnalyzer:
    """Analyzer for cross-project permissions and access patterns"""
    
    # High-risk cross-project patterns
    HIGH_RISK_PATTERNS = {
        "external_owner": r".*@(?!your-domain\.com).*",  # Non-domain owners
        "wide_delegation": r".*\*.*",  # Wildcard delegations
        "service_account_chain": r".*serviceAccount.*serviceAccount.*",  # SA chains
        "cross_org_access": r".*organizations/\d+.*organizations/\d+.*"  # Cross-org
    }
    
    # Permissions that enable cross-project access
    CROSS_PROJECT_PERMISSIONS = {
        "iam.serviceAccounts.actAs",
        "iam.serviceAccounts.getAccessToken",
        "iam.serviceAccounts.signBlob",
        "iam.serviceAccounts.signJwt",
        "resourcemanager.projects.setIamPolicy",
        "resourcemanager.folders.setIamPolicy",
        "resourcemanager.organizations.setIamPolicy",
        "compute.instances.setServiceAccount",
        "cloudfunctions.functions.setIamPolicy",
        "run.services.setIamPolicy"
    }
    
    def __init__(self, organization_id: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize the cross-project analyzer"""
        self.organization_id = organization_id
        self.db_path = db_path or "backend/cache/cross_project.db"
        self.project_hierarchies = {}
        self._init_database()
        
        # Initialize GCP clients if available
        if GCP_CLIENTS_AVAILABLE:
            try:
                self.asset_client = asset_v1.AssetServiceClient()
                self.resource_manager = resourcemanager_v3.ProjectsClient()
                self.folder_client = resourcemanager_v3.FoldersClient()
                self.org_client = resourcemanager_v3.OrganizationsClient()
                self.iam_client = iam_admin_v1.IAMClient()
                logger.info("GCP clients initialized for cross-project analysis")
            except Exception as e:
                logger.warning(f"Could not initialize GCP clients: {e}")
                self._set_mock_clients()
        else:
            self._set_mock_clients()
    
    def _set_mock_clients(self):
        """Set all clients to None for mock mode"""
        self.asset_client = None
        self.resource_manager = None
        self.folder_client = None
        self.org_client = None
        self.iam_client = None
    
    def _init_database(self):
        """Initialize the cross-project database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cross-project access table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_project_access (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                principal TEXT NOT NULL,
                principal_type TEXT,
                source_project TEXT NOT NULL,
                target_project TEXT NOT NULL,
                access_type TEXT NOT NULL,
                roles TEXT,
                permissions TEXT,
                resource_path TEXT,
                inheritance_chain TEXT,
                risk_level TEXT,
                compliance_flags TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                UNIQUE(principal, source_project, target_project, access_type)
            )
        """)
        
        # Project hierarchy cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_hierarchy (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                organization_id TEXT,
                folder_ids TEXT,
                parent_path TEXT,
                inherited_bindings TEXT,
                effective_permissions TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Delegation chains table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS delegation_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_id TEXT UNIQUE NOT NULL,
                principal_chain TEXT,
                project_chain TEXT,
                delegation_type TEXT,
                risk_score REAL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Service account impersonation table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sa_impersonations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                impersonator TEXT NOT NULL,
                impersonated_sa TEXT NOT NULL,
                project_id TEXT,
                method TEXT,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMP,
                UNIQUE(impersonator, impersonated_sa)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized cross-project database at {self.db_path}")
    
    async def analyze_cross_project_permissions(self, 
                                               project_ids: List[str]) -> CrossProjectReport:
        """Analyze cross-project permissions across multiple projects"""
        all_accesses = []
        delegation_chains = []
        sa_impersonations = []
        access_matrix = defaultdict(lambda: defaultdict(list))
        
        # Build project hierarchies
        for project_id in project_ids:
            hierarchy = await self._get_project_hierarchy(project_id)
            self.project_hierarchies[project_id] = hierarchy
        
        # Analyze cross-project access for each project pair
        for source_project in project_ids:
            for target_project in project_ids:
                if source_project == target_project:
                    continue
                
                # Find cross-project accesses
                accesses = await self._find_cross_project_access(
                    source_project, target_project
                )
                all_accesses.extend(accesses)
                
                # Build access matrix
                for access in accesses:
                    access_matrix[access.principal][target_project].append(
                        access.roles[0] if access.roles else "unknown"
                    )
        
        # Find delegation chains
        delegation_chains = await self._find_delegation_chains(project_ids)
        
        # Find service account impersonations
        sa_impersonations = await self._find_sa_impersonations(project_ids)
        
        # Calculate statistics
        high_risk_count = len([a for a in all_accesses if a.risk_level in ["HIGH", "CRITICAL"]])
        inheritance_stats = self._calculate_inheritance_stats(all_accesses)
        compliance_violations = self._check_compliance_violations(all_accesses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_accesses,
            delegation_chains,
            sa_impersonations
        )
        
        # Create report
        report = CrossProjectReport(
            analysis_timestamp=datetime.utcnow(),
            projects_analyzed=project_ids,
            total_cross_project_accesses=len(all_accesses),
            high_risk_accesses=high_risk_count,
            inheritance_depth_stats=inheritance_stats,
            delegation_chains=[self._chain_to_dict(c) for c in delegation_chains],
            service_account_impersonations=[self._sa_to_dict(s) for s in sa_impersonations],
            compliance_violations=compliance_violations,
            recommendations=recommendations,
            access_matrix=dict(access_matrix)
        )
        
        # Cache results
        self._cache_report(report)
        for access in all_accesses:
            self._cache_cross_project_access(access)
        
        return report
    
    async def _get_project_hierarchy(self, project_id: str) -> ProjectHierarchy:
        """Get organizational hierarchy for a project"""
        if self.resource_manager:
            try:
                # Get project details
                project = self.resource_manager.get_project(name=f"projects/{project_id}")
                
                # Trace parent hierarchy
                parent_path = []
                folder_ids = []
                org_id = None
                
                current_parent = project.parent
                while current_parent:
                    parent_path.append(current_parent)
                    
                    if current_parent.startswith("folders/"):
                        folder_ids.append(current_parent)
                        # Get folder details
                        folder = self.folder_client.get_folder(name=current_parent)
                        current_parent = folder.parent
                    elif current_parent.startswith("organizations/"):
                        org_id = current_parent
                        break
                
                # Get inherited IAM bindings
                inherited_bindings = await self._get_inherited_bindings(
                    project_id, parent_path
                )
                
                # Calculate effective permissions
                effective_permissions = self._calculate_effective_permissions(
                    project_id, inherited_bindings
                )
                
                return ProjectHierarchy(
                    project_id=project_id,
                    project_name=project.display_name,
                    organization_id=org_id,
                    folder_ids=folder_ids,
                    parent_path="/".join(reversed(parent_path)),
                    inherited_bindings=inherited_bindings,
                    effective_permissions=effective_permissions
                )
                
            except Exception as e:
                logger.error(f"Error getting project hierarchy: {e}")
                return self._get_mock_hierarchy(project_id)
        else:
            return self._get_mock_hierarchy(project_id)
    
    def _get_mock_hierarchy(self, project_id: str) -> ProjectHierarchy:
        """Get mock project hierarchy for development"""
        return ProjectHierarchy(
            project_id=project_id,
            project_name=f"Project {project_id}",
            organization_id="organizations/123456789",
            folder_ids=["folders/987654321"],
            parent_path="organizations/123456789/folders/987654321",
            inherited_bindings={
                "roles/owner": ["user:admin@example.com"],
                "roles/viewer": ["group:all@example.com"]
            },
            effective_permissions={
                "user:admin@example.com": {"*"},
                "group:all@example.com": {"*.get", "*.list"}
            }
        )
    
    async def _find_cross_project_access(self, source_project: str, 
                                        target_project: str) -> List[CrossProjectAccess]:
        """Find all cross-project access between two projects"""
        accesses = []
        
        # Check direct IAM bindings
        direct_access = await self._check_direct_access(source_project, target_project)
        accesses.extend(direct_access)
        
        # Check inherited access
        inherited_access = await self._check_inherited_access(source_project, target_project)
        accesses.extend(inherited_access)
        
        # Check delegated access (service account impersonation)
        delegated_access = await self._check_delegated_access(source_project, target_project)
        accesses.extend(delegated_access)
        
        # Check group-based access
        group_access = await self._check_group_access(source_project, target_project)
        accesses.extend(group_access)
        
        return accesses
    
    async def _check_direct_access(self, source_project: str, 
                                  target_project: str) -> List[CrossProjectAccess]:
        """Check for direct cross-project IAM bindings"""
        accesses = []
        
        if self.resource_manager:
            try:
                # Get IAM policy for target project
                policy = self.resource_manager.get_iam_policy(
                    resource=f"projects/{target_project}"
                )
                
                for binding in policy.bindings:
                    for member in binding.members:
                        # Check if member is from source project
                        if source_project in member:
                            principal_type, principal = self._parse_member(member)
                            
                            access = CrossProjectAccess(
                                principal=principal,
                                principal_type=principal_type,
                                source_project=source_project,
                                target_project=target_project,
                                access_type=AccessType.DIRECT.value,
                                roles=[binding.role],
                                permissions=self._get_role_permissions(binding.role),
                                resource_path=f"projects/{target_project}",
                                inheritance_chain=[],
                                risk_level=self._assess_risk_level(binding.role, principal_type),
                                compliance_flags=self._check_compliance_flags(binding.role, member)
                            )
                            accesses.append(access)
                            
            except Exception as e:
                logger.error(f"Error checking direct access: {e}")
                # Use mock data
                accesses = self._get_mock_direct_access(source_project, target_project)
        else:
            accesses = self._get_mock_direct_access(source_project, target_project)
        
        return accesses
    
    def _get_mock_direct_access(self, source_project: str, 
                               target_project: str) -> List[CrossProjectAccess]:
        """Get mock cross-project access for development"""
        return [
            CrossProjectAccess(
                principal=f"sa-{source_project}@{source_project}.iam.gserviceaccount.com",
                principal_type="serviceAccount",
                source_project=source_project,
                target_project=target_project,
                access_type=AccessType.DIRECT.value,
                roles=["roles/storage.objectViewer"],
                permissions=["storage.objects.get", "storage.objects.list"],
                resource_path=f"projects/{target_project}",
                inheritance_chain=[],
                risk_level="MEDIUM",
                compliance_flags=[]
            )
        ]
    
    async def _check_inherited_access(self, source_project: str,
                                     target_project: str) -> List[CrossProjectAccess]:
        """Check for inherited access through organization/folder hierarchy"""
        accesses = []
        
        # Get hierarchies for both projects
        source_hierarchy = self.project_hierarchies.get(source_project)
        target_hierarchy = self.project_hierarchies.get(target_project)
        
        if not source_hierarchy or not target_hierarchy:
            return accesses
        
        # Check if principals from source have inherited access to target
        for principal, permissions in source_hierarchy.effective_permissions.items():
            if principal in target_hierarchy.inherited_bindings.values():
                for role, members in target_hierarchy.inherited_bindings.items():
                    if principal in members:
                        access = CrossProjectAccess(
                            principal=principal,
                            principal_type=self._determine_principal_type(principal),
                            source_project=source_project,
                            target_project=target_project,
                            access_type=AccessType.INHERITED.value,
                            roles=[role],
                            permissions=list(permissions),
                            resource_path=target_hierarchy.parent_path,
                            inheritance_chain=target_hierarchy.folder_ids,
                            risk_level=self._assess_risk_level(role, "inherited"),
                            compliance_flags=["INHERITED_ACCESS"]
                        )
                        accesses.append(access)
        
        return accesses
    
    async def _check_delegated_access(self, source_project: str,
                                     target_project: str) -> List[CrossProjectAccess]:
        """Check for delegated access through service account impersonation"""
        accesses = []
        
        # Query for service account delegations
        if self.iam_client:
            try:
                # List service accounts in target project
                parent = f"projects/{target_project}"
                sas = self.iam_client.list_service_accounts(name=parent)
                
                for sa in sas:
                    # Check if SA can be impersonated from source project
                    sa_policy = self.iam_client.get_iam_policy(resource=sa.name)
                    
                    for binding in sa_policy.bindings:
                        if "iam.serviceAccounts.actAs" in binding.role:
                            for member in binding.members:
                                if source_project in member:
                                    principal_type, principal = self._parse_member(member)
                                    
                                    access = CrossProjectAccess(
                                        principal=principal,
                                        principal_type=principal_type,
                                        source_project=source_project,
                                        target_project=target_project,
                                        access_type=AccessType.DELEGATED.value,
                                        roles=["serviceAccount.actAs"],
                                        permissions=["iam.serviceAccounts.actAs"],
                                        resource_path=sa.name,
                                        inheritance_chain=[],
                                        risk_level="HIGH",  # Impersonation is high risk
                                        compliance_flags=["SERVICE_ACCOUNT_IMPERSONATION"]
                                    )
                                    accesses.append(access)
                                    
            except Exception as e:
                logger.error(f"Error checking delegated access: {e}")
        
        return accesses
    
    async def _check_group_access(self, source_project: str,
                                 target_project: str) -> List[CrossProjectAccess]:
        """Check for cross-project access through group memberships"""
        # This would check group memberships that span projects
        # For now, return empty list (would need Google Workspace API)
        return []
    
    async def _find_delegation_chains(self, project_ids: List[str]) -> List[Dict[str, Any]]:
        """Find chains of delegated permissions across projects"""
        chains = []
        
        # Look for service account chains
        for project in project_ids:
            if self.iam_client:
                try:
                    parent = f"projects/{project}"
                    sas = self.iam_client.list_service_accounts(name=parent)
                    
                    for sa in sas:
                        # Check if this SA has permissions on other SAs
                        chain = await self._trace_delegation_chain(sa.email, project_ids)
                        if len(chain) > 1:
                            chains.append({
                                "chain_id": f"chain-{len(chains)}",
                                "principal_chain": chain,
                                "project_chain": project_ids,
                                "delegation_type": "SERVICE_ACCOUNT_CHAIN",
                                "risk_score": min(0.9, 0.3 * len(chain))  # Risk increases with chain length
                            })
                            
                except Exception as e:
                    logger.error(f"Error finding delegation chains: {e}")
        
        return chains
    
    async def _trace_delegation_chain(self, sa_email: str, 
                                     project_ids: List[str]) -> List[str]:
        """Trace delegation chain for a service account"""
        chain = [sa_email]
        
        # This would recursively check what other SAs this SA can impersonate
        # For now, return simple chain
        return chain
    
    async def _find_sa_impersonations(self, project_ids: List[str]) -> List[Dict[str, Any]]:
        """Find service account impersonations across projects"""
        impersonations = []
        
        # Query audit logs for impersonation events (simplified)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT impersonator, impersonated_sa, project_id, method, frequency, last_seen
            FROM sa_impersonations
            WHERE project_id IN ({})
            ORDER BY frequency DESC
        """.format(','.join(['?'] * len(project_ids))), project_ids)
        
        for row in cursor.fetchall():
            impersonations.append({
                "impersonator": row[0],
                "impersonated_sa": row[1],
                "project_id": row[2],
                "method": row[3],
                "frequency": row[4],
                "last_seen": row[5]
            })
        
        conn.close()
        
        # Add mock data if empty
        if not impersonations and project_ids:
            impersonations.append({
                "impersonator": "user@example.com",
                "impersonated_sa": f"sa@{project_ids[0]}.iam.gserviceaccount.com",
                "project_id": project_ids[0],
                "method": "generateAccessToken",
                "frequency": 10,
                "last_seen": datetime.utcnow().isoformat()
            })
        
        return impersonations
    
    async def _get_inherited_bindings(self, project_id: str, 
                                     parent_path: List[str]) -> Dict[str, List[str]]:
        """Get IAM bindings inherited from parent resources"""
        inherited = defaultdict(list)
        
        for parent in parent_path:
            if self.resource_manager and parent.startswith("folders/"):
                try:
                    policy = self.folder_client.get_iam_policy(resource=parent)
                    for binding in policy.bindings:
                        inherited[binding.role].extend(binding.members)
                except Exception as e:
                    logger.error(f"Error getting folder IAM policy: {e}")
            elif self.org_client and parent.startswith("organizations/"):
                try:
                    policy = self.org_client.get_iam_policy(resource=parent)
                    for binding in policy.bindings:
                        inherited[binding.role].extend(binding.members)
                except Exception as e:
                    logger.error(f"Error getting org IAM policy: {e}")
        
        return dict(inherited)
    
    def _calculate_effective_permissions(self, project_id: str,
                                        inherited_bindings: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Calculate effective permissions for principals"""
        effective = defaultdict(set)
        
        for role, members in inherited_bindings.items():
            permissions = self._get_role_permissions(role)
            for member in members:
                effective[member].update(permissions)
        
        return dict(effective)
    
    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role (simplified)"""
        # In production, would query IAM API for actual permissions
        role_permission_map = {
            "roles/owner": ["*"],
            "roles/editor": ["*.create", "*.update", "*.delete", "*.get", "*.list"],
            "roles/viewer": ["*.get", "*.list"],
            "roles/storage.objectViewer": ["storage.objects.get", "storage.objects.list"],
            "roles/iam.serviceAccountUser": ["iam.serviceAccounts.actAs"]
        }
        
        return role_permission_map.get(role, [role.replace("roles/", "") + ".*"])
    
    def _parse_member(self, member: str) -> Tuple[str, str]:
        """Parse IAM member string"""
        if ":" in member:
            member_type, principal = member.split(":", 1)
            return (member_type, principal)
        return ("unknown", member)
    
    def _determine_principal_type(self, principal: str) -> str:
        """Determine principal type from email/identifier"""
        if ".gserviceaccount.com" in principal:
            return "serviceAccount"
        elif principal.startswith("group:"):
            return "group"
        elif "@" in principal:
            return "user"
        else:
            return "unknown"
    
    def _assess_risk_level(self, role: str, context: str) -> str:
        """Assess risk level of cross-project access"""
        high_risk_roles = ["roles/owner", "roles/editor", "roles/iam.securityAdmin"]
        
        if role in high_risk_roles:
            return "CRITICAL"
        elif "admin" in role.lower():
            return "HIGH"
        elif context == "inherited":
            return "MEDIUM"
        else:
            return "LOW"
    
    def _check_compliance_flags(self, role: str, member: str) -> List[str]:
        """Check compliance violations"""
        flags = []
        
        if role == "roles/owner" and ".gserviceaccount.com" in member:
            flags.append("SERVICE_ACCOUNT_OWNER")
        
        if "@gmail.com" in member or "@yahoo.com" in member:
            flags.append("PERSONAL_EMAIL")
        
        if role in ["roles/owner", "roles/editor"] and "allUsers" in member:
            flags.append("PUBLIC_HIGH_PRIVILEGE")
        
        return flags
    
    def _calculate_inheritance_stats(self, accesses: List[CrossProjectAccess]) -> Dict[str, int]:
        """Calculate inheritance depth statistics"""
        stats = defaultdict(int)
        
        for access in accesses:
            depth = len(access.inheritance_chain)
            if depth == 0:
                stats["direct"] += 1
            elif depth == 1:
                stats["1_level"] += 1
            elif depth == 2:
                stats["2_levels"] += 1
            else:
                stats["3+_levels"] += 1
        
        return dict(stats)
    
    def _check_compliance_violations(self, accesses: List[CrossProjectAccess]) -> List[str]:
        """Check for compliance violations in cross-project access"""
        violations = []
        
        # Check for service accounts with owner role
        sa_owners = [a for a in accesses 
                    if a.principal_type == "serviceAccount" and "roles/owner" in a.roles]
        if sa_owners:
            violations.append(f"SERVICE_ACCOUNT_OWNERS: {len(sa_owners)} service accounts have owner role")
        
        # Check for external principals
        external = [a for a in accesses if "@your-domain.com" not in a.principal]
        if external:
            violations.append(f"EXTERNAL_ACCESS: {len(external)} external principals have access")
        
        # Check for excessive inheritance
        deep_inheritance = [a for a in accesses if len(a.inheritance_chain) > 2]
        if deep_inheritance:
            violations.append(f"DEEP_INHERITANCE: {len(deep_inheritance)} accesses through >2 levels")
        
        return violations
    
    def _generate_recommendations(self, accesses: List[CrossProjectAccess],
                                 delegation_chains: List[Dict],
                                 impersonations: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # High-risk access
        high_risk = [a for a in accesses if a.risk_level in ["HIGH", "CRITICAL"]]
        if high_risk:
            recommendations.append(
                f"CRITICAL: Review and reduce {len(high_risk)} high-risk cross-project accesses"
            )
        
        # Long delegation chains
        long_chains = [c for c in delegation_chains if len(c.get("principal_chain", [])) > 2]
        if long_chains:
            recommendations.append(
                f"HIGH: Simplify {len(long_chains)} delegation chains to reduce complexity"
            )
        
        # Frequent impersonations
        frequent_impersonations = [i for i in impersonations if i.get("frequency", 0) > 100]
        if frequent_impersonations:
            recommendations.append(
                f"MEDIUM: Review {len(frequent_impersonations)} frequently impersonated service accounts"
            )
        
        # General recommendations
        recommendations.extend([
            "Implement project-level VPC Service Controls for sensitive projects",
            "Use Workload Identity instead of service account keys for cross-project access",
            "Enable audit logging for all IAM changes and service account activities",
            "Implement time-bound access for temporary cross-project permissions",
            "Use custom roles to limit cross-project permissions to minimum required"
        ])
        
        return recommendations[:10]
    
    def _chain_to_dict(self, chain: Any) -> Dict[str, Any]:
        """Convert chain object to dictionary"""
        if isinstance(chain, dict):
            return chain
        return {"chain": str(chain)}
    
    def _sa_to_dict(self, sa: Any) -> Dict[str, Any]:
        """Convert service account impersonation to dictionary"""
        if isinstance(sa, dict):
            return sa
        return {"impersonation": str(sa)}
    
    def _cache_cross_project_access(self, access: CrossProjectAccess):
        """Cache cross-project access in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO cross_project_access
            (principal, principal_type, source_project, target_project, access_type,
             roles, permissions, resource_path, inheritance_chain, risk_level,
             compliance_flags, discovered_at, last_activity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            access.principal,
            access.principal_type,
            access.source_project,
            access.target_project,
            access.access_type,
            json.dumps(access.roles),
            json.dumps(access.permissions),
            access.resource_path,
            json.dumps(access.inheritance_chain),
            access.risk_level,
            json.dumps(access.compliance_flags),
            access.discovered_at.isoformat(),
            access.last_activity.isoformat() if access.last_activity else None
        ))
        
        conn.commit()
        conn.close()
    
    def _cache_report(self, report: CrossProjectReport):
        """Cache the analysis report"""
        # Would cache full report in database
        pass
    
    def get_cached_accesses(self, project_id: Optional[str] = None) -> List[CrossProjectAccess]:
        """Get cached cross-project accesses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if project_id:
            query = """
                SELECT * FROM cross_project_access
                WHERE source_project = ? OR target_project = ?
                ORDER BY discovered_at DESC
            """
            cursor.execute(query, (project_id, project_id))
        else:
            query = "SELECT * FROM cross_project_access ORDER BY discovered_at DESC"
            cursor.execute(query)
        
        accesses = []
        for row in cursor.fetchall():
            access = CrossProjectAccess(
                principal=row[1],
                principal_type=row[2],
                source_project=row[3],
                target_project=row[4],
                access_type=row[5],
                roles=json.loads(row[6]) if row[6] else [],
                permissions=json.loads(row[7]) if row[7] else [],
                resource_path=row[8],
                inheritance_chain=json.loads(row[9]) if row[9] else [],
                risk_level=row[10],
                compliance_flags=json.loads(row[11]) if row[11] else [],
                discovered_at=datetime.fromisoformat(row[12]) if row[12] else datetime.utcnow(),
                last_activity=datetime.fromisoformat(row[13]) if row[13] else None
            )
            accesses.append(access)
        
        conn.close()
        return accesses