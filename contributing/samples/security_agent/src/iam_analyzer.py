"""IAM Permission Analyzer for GCP Security Agent

Advanced IAM analysis engine that identifies permission issues, excessive privileges,
and provides least privilege recommendations with smart policy optimization.
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PermissionAnalysis:
    """Analysis result for a specific permission."""
    permission: str
    resource: str
    principal: str
    granted_by_role: str
    is_necessary: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    usage_frequency: str  # NEVER, RARELY, SOMETIMES, FREQUENTLY
    last_used: Optional[datetime]
    recommendation: str

@dataclass
class PrincipalAnalysis:
    """Analysis result for a principal (user/service account)."""
    principal_id: str
    principal_type: str  # user, serviceAccount, group
    total_permissions: int
    unnecessary_permissions: int
    high_risk_permissions: int
    roles: List[str]
    resources: List[str]
    risk_score: float  # 0-100
    recommendations: List[str]

@dataclass
class IAMSummary:
    """Overall IAM security summary."""
    total_principals: int
    total_permissions: int
    overprivileged_principals: int
    unused_service_accounts: int
    dangerous_permissions: int
    compliance_score: float  # 0-100
    risk_score: float  # 0-100
    top_risks: List[PrincipalAnalysis]
    recommendations: List[str]
    analysis_timestamp: datetime

class IAMAnalyzer:
    """Advanced IAM permission analyzer with least privilege recommendations."""
    
    def __init__(self, project_id: str, db_path: Optional[str] = None):
        self.project_id = project_id
        self.db_path = db_path or self._get_default_db_path()
        
        # Define dangerous permissions that require special attention
        self.dangerous_permissions = {
            'iam.serviceAccounts.actAs',
            'iam.serviceAccounts.getAccessToken',
            'iam.serviceAccountKeys.create',
            'resourcemanager.projects.setIamPolicy',
            'compute.instances.setMetadata',
            'storage.buckets.setIamPolicy',
            'cloudsql.instances.update',
            'container.clusters.update',
            'secretmanager.secrets.setIamPolicy'
        }
        
        # Define high-privilege roles that should be monitored
        self.high_privilege_roles = {
            'roles/owner',
            'roles/editor', 
            'roles/iam.securityAdmin',
            'roles/iam.serviceAccountAdmin',
            'roles/resourcemanager.projectIamAdmin',
            'roles/compute.admin',
            'roles/storage.admin'
        }
        
        # Mapping of roles to their permissions (simplified - in reality would come from IAM API)
        self.role_permissions = self._load_role_permission_mapping()
    
    def _get_default_db_path(self) -> str:
        """Get default database path."""
        current_file = Path(__file__)
        security_agent_dir = current_file.parent.parent.parent
        return str(security_agent_dir / 'backend' / 'cache' / 'gcp_data.db')
    
    def _load_role_permission_mapping(self) -> Dict[str, Set[str]]:
        """Load mapping of roles to permissions (simplified version)."""
        return {
            'roles/owner': {
                '*'  # All permissions
            },
            'roles/editor': {
                'compute.*', 'storage.*', 'bigquery.*', 'dataflow.*',
                'pubsub.*', 'logging.*', 'monitoring.*'
            },
            'roles/viewer': {
                '*.get', '*.list', 'resourcemanager.projects.get'
            },
            'roles/compute.instanceAdmin': {
                'compute.instances.*', 'compute.instanceGroups.*',
                'compute.instanceTemplates.*'
            },
            'roles/storage.objectAdmin': {
                'storage.objects.*'
            },
            'roles/iam.serviceAccountUser': {
                'iam.serviceAccounts.actAs',
                'iam.serviceAccounts.get'
            }
        }
    
    def analyze_all_iam(self) -> IAMSummary:
        """Perform comprehensive IAM analysis."""
        start_time = datetime.now()
        
        # Get all IAM data
        principals_data = self._get_all_principals()
        bindings_data = self._get_all_bindings()
        
        # Analyze each principal
        principal_analyses = []
        for principal_id, principal_info in principals_data.items():
            analysis = self._analyze_principal(principal_id, principal_info, bindings_data)
            principal_analyses.append(analysis)
        
        # Calculate overall metrics
        total_permissions = sum(p.total_permissions for p in principal_analyses)
        overprivileged = len([p for p in principal_analyses if p.risk_score > 70])
        unused_sas = len([p for p in principal_analyses if p.principal_type == 'serviceAccount' and p.total_permissions == 0])
        dangerous_perms = sum(p.high_risk_permissions for p in principal_analyses)
        
        # Calculate scores
        compliance_score = self._calculate_compliance_score(principal_analyses)
        risk_score = self._calculate_overall_risk_score(principal_analyses)
        
        # Generate recommendations
        recommendations = self._generate_iam_recommendations(principal_analyses)
        
        summary = IAMSummary(
            total_principals=len(principal_analyses),
            total_permissions=total_permissions,
            overprivileged_principals=overprivileged,
            unused_service_accounts=unused_sas,
            dangerous_permissions=dangerous_perms,
            compliance_score=compliance_score,
            risk_score=risk_score,
            top_risks=sorted(principal_analyses, key=lambda p: p.risk_score, reverse=True)[:10],
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
        
        # Store analysis results
        self._store_iam_analysis(summary, principal_analyses)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"🔐 IAM analysis complete: {len(principal_analyses)} principals analyzed in {duration:.2f}s")
        
        return summary
    
    def _get_all_principals(self) -> Dict[str, Dict[str, Any]]:
        """Get all principals (users, service accounts, groups) from database."""
        principals = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get service accounts
            cursor.execute("SELECT * FROM service_accounts")
            service_accounts = cursor.fetchall()
            
            for sa in service_accounts:
                principals[sa['email']] = {
                    'type': 'serviceAccount',
                    'name': sa['name'],
                    'display_name': sa['display_name'],
                    'disabled': sa['disabled'],
                    'project_id': sa['project_id'],
                    'created_time': sa.get('create_time'),
                    'data': json.loads(sa.get('account_data', '{}'))
                }
            
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not load principals from database: {e}")
            # Create mock data
            principals = self._create_mock_principals()
        
        return principals
    
    def _get_all_bindings(self) -> List[Dict[str, Any]]:
        """Get all IAM bindings from database."""
        bindings = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM iam_bindings")
            db_bindings = cursor.fetchall()
            
            for binding in db_bindings:
                bindings.append({
                    'resource_name': binding['resource_name'],
                    'role': binding['role'],
                    'member': binding['member'],
                    'member_type': binding['member_type'],
                    'condition': {
                        'title': binding.get('condition_title'),
                        'expression': binding.get('condition_expression')
                    } if binding.get('condition_expression') else None
                })
            
            conn.close()
            
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not load IAM bindings from database: {e}")
            bindings = self._create_mock_bindings()
        
        return bindings
    
    def _analyze_principal(self, principal_id: str, principal_info: Dict[str, Any], 
                          all_bindings: List[Dict[str, Any]]) -> PrincipalAnalysis:
        """Analyze a specific principal for permission issues."""
        
        # Get all bindings for this principal
        principal_bindings = [
            b for b in all_bindings 
            if b['member'].endswith(principal_id) or principal_id in b['member']
        ]
        
        # Extract roles and resources
        roles = list(set(b['role'] for b in principal_bindings))
        resources = list(set(b['resource_name'] for b in principal_bindings))
        
        # Calculate permissions
        all_permissions = self._get_permissions_for_roles(roles)
        dangerous_permissions = len([p for p in all_permissions if p in self.dangerous_permissions])
        
        # Analyze usage (mock data - in reality would come from Cloud Audit Logs)
        unnecessary_permissions = self._estimate_unnecessary_permissions(all_permissions, principal_info)
        
        # Calculate risk score
        risk_score = self._calculate_principal_risk_score(
            principal_info, roles, all_permissions, dangerous_permissions
        )
        
        # Generate recommendations
        recommendations = self._generate_principal_recommendations(
            principal_id, principal_info, roles, all_permissions, risk_score
        )
        
        return PrincipalAnalysis(
            principal_id=principal_id,
            principal_type=principal_info['type'],
            total_permissions=len(all_permissions),
            unnecessary_permissions=unnecessary_permissions,
            high_risk_permissions=dangerous_permissions,
            roles=roles,
            resources=resources,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def _get_permissions_for_roles(self, roles: List[str]) -> Set[str]:
        """Get all permissions granted by the given roles."""
        all_permissions = set()
        
        for role in roles:
            permissions = self.role_permissions.get(role, set())
            all_permissions.update(permissions)
        
        return all_permissions
    
    def _estimate_unnecessary_permissions(self, permissions: Set[str], principal_info: Dict[str, Any]) -> int:
        """Estimate unnecessary permissions (simplified heuristic)."""
        # In a real implementation, this would analyze Cloud Audit Logs
        # For now, use heuristics based on principal type and role patterns
        
        total_permissions = len(permissions)
        
        # Service accounts typically need fewer permissions than users
        if principal_info['type'] == 'serviceAccount':
            # Estimate that 30-50% of permissions might be unnecessary for SAs
            if '*' in permissions:  # Has owner/editor role
                return int(total_permissions * 0.7)  # 70% unnecessary for broad roles
            else:
                return int(total_permissions * 0.3)  # 30% unnecessary for specific roles
        else:
            # Users might have more legitimate broad access
            if '*' in permissions:
                return int(total_permissions * 0.4)  # 40% unnecessary
            else:
                return int(total_permissions * 0.2)  # 20% unnecessary
    
    def _calculate_principal_risk_score(self, principal_info: Dict[str, Any], roles: List[str], 
                                      permissions: Set[str], dangerous_permissions: int) -> float:
        """Calculate risk score for a principal (0-100)."""
        score = 0.0
        
        # High privilege roles add significant risk
        for role in roles:
            if role in self.high_privilege_roles:
                if role == 'roles/owner':
                    score += 40
                elif role == 'roles/editor':
                    score += 25
                else:
                    score += 15
        
        # Dangerous permissions
        score += dangerous_permissions * 5
        
        # Service account specific risks
        if principal_info['type'] == 'serviceAccount':
            # Unused service accounts are risky
            if principal_info.get('disabled') or len(permissions) == 0:
                score += 10
        
        # Broad permissions (wildcard)
        if '*' in permissions:
            score += 20
        
        # Many permissions indicate potential over-privilege
        if len(permissions) > 50:
            score += 10
        elif len(permissions) > 100:
            score += 20
        
        return min(score, 100.0)  # Cap at 100
    
    def _generate_principal_recommendations(self, principal_id: str, principal_info: Dict[str, Any],
                                          roles: List[str], permissions: Set[str], risk_score: float) -> List[str]:
        """Generate specific recommendations for a principal."""
        recommendations = []
        
        # High risk principals
        if risk_score > 80:
            recommendations.append("🚨 URGENT: Review and reduce permissions immediately")
        elif risk_score > 60:
            recommendations.append("⚠️ HIGH RISK: Consider reducing permissions")
        
        # Role-specific recommendations
        if 'roles/owner' in roles:
            recommendations.append("👑 Owner role detected - consider using more specific roles")
        
        if 'roles/editor' in roles:
            recommendations.append("✏️ Editor role detected - implement principle of least privilege")
        
        # Service account specific recommendations
        if principal_info['type'] == 'serviceAccount':
            if principal_info.get('disabled'):
                recommendations.append("🗑️ Remove unused disabled service account")
            
            if len(permissions) > 20:
                recommendations.append("🔧 Service account may be overprivileged - audit required")
        
        # Permission-specific recommendations
        dangerous_count = len([p for p in permissions if p in self.dangerous_permissions])
        if dangerous_count > 0:
            recommendations.append(f"⚠️ Has {dangerous_count} dangerous permissions - review necessity")
        
        # Cross-project access
        if any('projects/' in resource for resource in []):  # Would check actual resources
            recommendations.append("🌐 Cross-project access detected - validate necessity")
        
        return recommendations
    
    def _calculate_compliance_score(self, analyses: List[PrincipalAnalysis]) -> float:
        """Calculate overall IAM compliance score."""
        if not analyses:
            return 100.0
        
        total_score = 0.0
        for analysis in analyses:
            # Score based on risk level
            if analysis.risk_score < 30:
                total_score += 100
            elif analysis.risk_score < 50:
                total_score += 80
            elif analysis.risk_score < 70:
                total_score += 60
            else:
                total_score += 30
        
        return total_score / len(analyses)
    
    def _calculate_overall_risk_score(self, analyses: List[PrincipalAnalysis]) -> float:
        """Calculate overall IAM risk score."""
        if not analyses:
            return 0.0
        
        return sum(a.risk_score for a in analyses) / len(analyses)
    
    def _generate_iam_recommendations(self, analyses: List[PrincipalAnalysis]) -> List[str]:
        """Generate overall IAM recommendations."""
        recommendations = []
        
        # Count issues
        high_risk_count = len([a for a in analyses if a.risk_score > 70])
        owner_roles = len([a for a in analyses if 'roles/owner' in a.roles])
        editor_roles = len([a for a in analyses if 'roles/editor' in a.roles])
        unused_sas = len([a for a in analyses if a.principal_type == 'serviceAccount' and a.total_permissions == 0])
        
        # Priority recommendations
        if high_risk_count > 0:
            recommendations.append(f"🚨 Immediate action: {high_risk_count} high-risk principals need review")
        
        if owner_roles > 1:
            recommendations.append(f"👑 {owner_roles} principals have Owner role - minimize usage")
        
        if editor_roles > 3:
            recommendations.append(f"✏️ {editor_roles} principals have Editor role - consider more specific roles")
        
        if unused_sas > 0:
            recommendations.append(f"🗑️ Remove {unused_sas} unused service accounts")
        
        # General recommendations
        recommendations.extend([
            "📊 Implement regular IAM access reviews (quarterly)",
            "🔍 Enable IAM Recommender for automated suggestions",
            "📝 Document and justify high-privilege role assignments",
            "🎯 Implement just-in-time access where possible",
            "📚 Provide IAM security training to teams"
        ])
        
        return recommendations[:10]  # Return top 10
    
    def _store_iam_analysis(self, summary: IAMSummary, analyses: List[PrincipalAnalysis]):
        """Store IAM analysis results in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create IAM analysis table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iam_analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_timestamp TIMESTAMP NOT NULL,
                    total_principals INTEGER,
                    total_permissions INTEGER,
                    overprivileged_principals INTEGER,
                    unused_service_accounts INTEGER,
                    dangerous_permissions INTEGER,
                    compliance_score REAL,
                    risk_score REAL,
                    recommendations TEXT,  -- JSON
                    analysis_data TEXT     -- JSON with full analysis
                )
            """)
            
            # Store summary
            cursor.execute("""
                INSERT INTO iam_analysis_results
                (analysis_timestamp, total_principals, total_permissions, overprivileged_principals,
                 unused_service_accounts, dangerous_permissions, compliance_score, risk_score,
                 recommendations, analysis_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.analysis_timestamp.isoformat(),
                summary.total_principals,
                summary.total_permissions,
                summary.overprivileged_principals,
                summary.unused_service_accounts,
                summary.dangerous_permissions,
                summary.compliance_score,
                summary.risk_score,
                json.dumps(summary.recommendations),
                json.dumps({
                    'principals': [
                        {
                            'id': a.principal_id,
                            'type': a.principal_type,
                            'risk_score': a.risk_score,
                            'total_permissions': a.total_permissions,
                            'unnecessary_permissions': a.unnecessary_permissions,
                            'roles': a.roles,
                            'recommendations': a.recommendations
                        }
                        for a in analyses
                    ]
                })
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("✅ IAM analysis results stored in database")
            
        except Exception as e:
            logger.error(f"Failed to store IAM analysis: {e}")
    
    def get_least_privilege_recommendations(self, principal_id: str) -> Dict[str, Any]:
        """Get specific least privilege recommendations for a principal."""
        # Get current permissions for the principal
        principals_data = self._get_all_principals()
        bindings_data = self._get_all_bindings()
        
        if principal_id not in principals_data:
            return {"error": f"Principal {principal_id} not found"}
        
        principal_info = principals_data[principal_id]
        analysis = self._analyze_principal(principal_id, principal_info, bindings_data)
        
        # Generate detailed recommendations
        recommendations = {
            "current_risk_score": analysis.risk_score,
            "current_roles": analysis.roles,
            "total_permissions": analysis.total_permissions,
            "unnecessary_permissions": analysis.unnecessary_permissions,
            "recommended_actions": analysis.recommendations,
            "suggested_roles": self._suggest_minimal_roles(analysis),
            "permissions_to_remove": self._identify_removable_permissions(analysis)
        }
        
        return recommendations
    
    def _suggest_minimal_roles(self, analysis: PrincipalAnalysis) -> List[str]:
        """Suggest minimal roles based on usage patterns."""
        # This is a simplified version - in reality would analyze actual usage
        suggestions = []
        
        current_roles = set(analysis.roles)
        
        # If has owner/editor, suggest more specific alternatives
        if 'roles/owner' in current_roles:
            suggestions.extend([
                "Consider replacing with roles/resourcemanager.projectIamAdmin",
                "Or specific resource admin roles like roles/compute.admin"
            ])
        
        if 'roles/editor' in current_roles:
            suggestions.extend([
                "Replace with specific service roles:",
                "- roles/compute.instanceAdmin for VM management",
                "- roles/storage.objectAdmin for storage access",
                "- roles/cloudsql.editor for database management"
            ])
        
        return suggestions
    
    def _identify_removable_permissions(self, analysis: PrincipalAnalysis) -> List[str]:
        """Identify permissions that can likely be removed."""
        removable = []
        
        # This would typically analyze Cloud Audit Logs to find unused permissions
        # For now, provide heuristic-based suggestions
        
        if analysis.principal_type == 'serviceAccount':
            removable.extend([
                "Consider removing broad administrative permissions",
                "Remove unused API access permissions",
                "Evaluate cross-project permissions"
            ])
        
        return removable
    
    # Mock data methods
    def _create_mock_principals(self) -> Dict[str, Dict[str, Any]]:
        """Create mock principals for development."""
        return {
            f'web-app@{self.project_id}.iam.gserviceaccount.com': {
                'type': 'serviceAccount',
                'name': f'projects/{self.project_id}/serviceAccounts/web-app@{self.project_id}.iam.gserviceaccount.com',
                'display_name': 'Web Application Service Account',
                'disabled': False,
                'project_id': self.project_id,
                'data': {}
            },
            'admin@example.com': {
                'type': 'user',
                'name': 'admin@example.com',
                'display_name': 'Admin User',
                'disabled': False,
                'project_id': self.project_id,
                'data': {}
            }
        }
    
    def _create_mock_bindings(self) -> List[Dict[str, Any]]:
        """Create mock IAM bindings."""
        return [
            {
                'resource_name': f'projects/{self.project_id}',
                'role': 'roles/editor',
                'member': f'serviceAccount:web-app@{self.project_id}.iam.gserviceaccount.com',
                'member_type': 'serviceAccount',
                'condition': None
            },
            {
                'resource_name': f'projects/{self.project_id}',
                'role': 'roles/owner',
                'member': 'user:admin@example.com',
                'member_type': 'user',
                'condition': None
            }
        ]

# Convenience functions
def analyze_iam_permissions(project_id: str, db_path: Optional[str] = None) -> IAMSummary:
    """Convenience function to run IAM analysis."""
    analyzer = IAMAnalyzer(project_id, db_path)
    return analyzer.analyze_all_iam()

def get_iam_summary_text(project_id: str, db_path: Optional[str] = None) -> str:
    """Get formatted IAM analysis summary."""
    summary = analyze_iam_permissions(project_id, db_path)
    
    result = f"👤 **IAM Security Analysis**\\n\\n"
    result += f"**Total Principals**: {summary.total_principals}\\n"
    result += f"**Total Permissions**: {summary.total_permissions}\\n"
    result += f"**Overprivileged Principals**: {summary.overprivileged_principals}\\n"
    result += f"**Unused Service Accounts**: {summary.unused_service_accounts}\\n"
    result += f"**Risk Score**: {summary.risk_score:.1f}/100\\n"
    result += f"**Compliance Score**: {summary.compliance_score:.1f}/100\\n\\n"
    
    if summary.top_risks:
        result += "**Highest Risk Principal:**\\n"
        top_risk = summary.top_risks[0]
        result += f"* {top_risk.principal_id}\\n"
        result += f"  Risk Score: {top_risk.risk_score:.1f}/100\\n"
        result += f"  Permissions: {top_risk.total_permissions}\\n"
        result += f"  Roles: {', '.join(top_risk.roles[:3])}\\n\\n"
    
    if summary.recommendations:
        result += "**Priority Recommendations:**\\n"
        for rec in summary.recommendations[:3]:
            result += f"* {rec}\\n"
    
    return result