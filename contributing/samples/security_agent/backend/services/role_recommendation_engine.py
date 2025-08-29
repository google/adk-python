"""
Role Recommendation Engine for Advanced IAM Features
Analyzes actual API usage patterns to recommend optimal IAM roles
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter

try:
    from google.cloud import logging as cloud_logging
    from google.cloud import bigquery
    from google.cloud import iam_admin_v1
    from google.cloud import resourcemanager_v3
    GCP_CLIENTS_AVAILABLE = True
except ImportError:
    GCP_CLIENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RoleRecommendation:
    """Role recommendation for a principal"""
    principal: str
    principal_type: str  # user, serviceAccount, group
    current_roles: List[str]
    recommended_roles: List[str]
    custom_role_needed: bool
    unused_permissions: List[str]
    missing_permissions: List[str]
    confidence_score: float
    analysis_period_days: int
    api_calls_analyzed: int
    cost_impact: Optional[float] = None
    risk_reduction: str = "MEDIUM"
    compliance_impact: List[str] = field(default_factory=list)
    recommendation_reason: str = ""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class APIUsagePattern:
    """API usage pattern for a principal"""
    principal: str
    api_method: str
    service: str
    resource_type: str
    frequency: int
    last_used: datetime
    required_permissions: List[str]


class ConfidenceLevel(Enum):
    """Confidence levels for recommendations"""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


class RoleRecommendationEngine:
    """Engine for analyzing IAM usage and recommending roles"""
    
    # Mapping of common API methods to required permissions
    API_TO_PERMISSION_MAP = {
        # Compute
        "compute.instances.list": ["compute.instances.list"],
        "compute.instances.get": ["compute.instances.get"],
        "compute.instances.create": ["compute.instances.create"],
        "compute.instances.delete": ["compute.instances.delete"],
        "compute.instances.start": ["compute.instances.start"],
        "compute.instances.stop": ["compute.instances.stop"],
        
        # Storage
        "storage.buckets.list": ["storage.buckets.list"],
        "storage.buckets.get": ["storage.buckets.get"],
        "storage.buckets.create": ["storage.buckets.create"],
        "storage.objects.list": ["storage.objects.list"],
        "storage.objects.get": ["storage.objects.get"],
        "storage.objects.create": ["storage.objects.create"],
        "storage.objects.delete": ["storage.objects.delete"],
        
        # IAM
        "iam.serviceAccounts.list": ["iam.serviceAccounts.list"],
        "iam.serviceAccounts.create": ["iam.serviceAccounts.create"],
        "iam.serviceAccountKeys.create": ["iam.serviceAccountKeys.create"],
        "iam.roles.list": ["iam.roles.list"],
        "iam.roles.get": ["iam.roles.get"],
        
        # BigQuery
        "bigquery.tables.list": ["bigquery.tables.list"],
        "bigquery.tables.get": ["bigquery.tables.get"],
        "bigquery.tables.create": ["bigquery.tables.create"],
        "bigquery.tables.getData": ["bigquery.tables.getData"],
        "bigquery.jobs.create": ["bigquery.jobs.create"],
        
        # Cloud Functions
        "cloudfunctions.functions.list": ["cloudfunctions.functions.list"],
        "cloudfunctions.functions.get": ["cloudfunctions.functions.get"],
        "cloudfunctions.functions.create": ["cloudfunctions.functions.create"],
        "cloudfunctions.functions.invoke": ["cloudfunctions.functions.invoke"],
    }
    
    # Mapping of permissions to predefined roles
    PERMISSION_TO_ROLE_MAP = {
        # Storage roles
        frozenset(["storage.buckets.list", "storage.objects.list", "storage.objects.get"]): 
            "roles/storage.objectViewer",
        frozenset(["storage.buckets.list", "storage.objects.list", "storage.objects.get", 
                  "storage.objects.create", "storage.objects.delete"]): 
            "roles/storage.objectAdmin",
        
        # Compute roles
        frozenset(["compute.instances.list", "compute.instances.get"]): 
            "roles/compute.viewer",
        frozenset(["compute.instances.list", "compute.instances.get", 
                  "compute.instances.start", "compute.instances.stop"]): 
            "roles/compute.instanceAdmin",
        
        # BigQuery roles
        frozenset(["bigquery.tables.list", "bigquery.tables.get", "bigquery.tables.getData"]): 
            "roles/bigquery.dataViewer",
        frozenset(["bigquery.tables.list", "bigquery.tables.get", "bigquery.tables.create", 
                  "bigquery.jobs.create"]): 
            "roles/bigquery.dataEditor",
        
        # IAM roles
        frozenset(["iam.serviceAccounts.list", "iam.roles.list", "iam.roles.get"]): 
            "roles/iam.viewer",
        frozenset(["iam.serviceAccounts.list", "iam.serviceAccounts.create", 
                  "iam.serviceAccountKeys.create"]): 
            "roles/iam.serviceAccountAdmin",
    }
    
    def __init__(self, project_id: str, db_path: Optional[str] = None):
        """Initialize the role recommendation engine"""
        self.project_id = project_id
        self.db_path = db_path or "backend/cache/iam_recommendations.db"
        self._init_database()
        
        # Initialize GCP clients if available
        if GCP_CLIENTS_AVAILABLE:
            try:
                self.logging_client = cloud_logging.Client(project=project_id)
                self.bigquery_client = bigquery.Client(project=project_id)
                self.iam_client = iam_admin_v1.IAMClient()
                logger.info("GCP clients initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize GCP clients: {e}")
                self.logging_client = None
                self.bigquery_client = None
                self.iam_client = None
        else:
            self.logging_client = None
            self.bigquery_client = None
            self.iam_client = None
    
    def _init_database(self):
        """Initialize the recommendations database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # API usage patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                principal TEXT NOT NULL,
                api_method TEXT NOT NULL,
                service TEXT NOT NULL,
                resource_type TEXT,
                frequency INTEGER DEFAULT 1,
                last_used TIMESTAMP,
                required_permissions TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(principal, api_method)
            )
        """)
        
        # Role recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS role_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                principal TEXT NOT NULL,
                principal_type TEXT,
                current_roles TEXT,
                recommended_roles TEXT,
                custom_role_needed BOOLEAN,
                unused_permissions TEXT,
                missing_permissions TEXT,
                confidence_score REAL,
                analysis_period_days INTEGER,
                api_calls_analyzed INTEGER,
                cost_impact REAL,
                risk_reduction TEXT,
                compliance_impact TEXT,
                recommendation_reason TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(principal)
            )
        """)
        
        # Permission usage cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS permission_usage_cache (
                principal TEXT NOT NULL,
                permission TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(principal, permission)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized recommendations database at {self.db_path}")
    
    async def analyze_principal_usage(self, principal_email: str, 
                                     days_to_analyze: int = 30) -> List[APIUsagePattern]:
        """Analyze API usage patterns for a specific principal"""
        patterns = []
        
        if self.logging_client and self.bigquery_client:
            try:
                # Query audit logs for API usage
                patterns = await self._query_audit_logs(principal_email, days_to_analyze)
            except Exception as e:
                logger.error(f"Error querying audit logs: {e}")
                # Fall back to cached data
                patterns = self._get_cached_usage_patterns(principal_email)
        else:
            # Use mock data for development
            patterns = self._generate_mock_usage_patterns(principal_email)
        
        # Cache the patterns
        self._cache_usage_patterns(patterns)
        
        return patterns
    
    async def _query_audit_logs(self, principal_email: str, 
                               days: int) -> List[APIUsagePattern]:
        """Query BigQuery audit logs for API usage"""
        patterns = []
        
        query = f"""
        SELECT 
            protoPayload.authenticationInfo.principalEmail as principal,
            protoPayload.methodName as api_method,
            protoPayload.serviceName as service,
            protoPayload.resourceName as resource,
            COUNT(*) as frequency,
            MAX(timestamp) as last_used
        FROM `{self.project_id}.cloud_audit_logs.cloudaudit_googleapis_com_data_access`
        WHERE 
            protoPayload.authenticationInfo.principalEmail = @principal_email
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        GROUP BY 1,2,3,4
        ORDER BY frequency DESC
        LIMIT 1000
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("principal_email", "STRING", principal_email),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )
        
        try:
            query_job = self.bigquery_client.query(query, job_config=job_config)
            results = query_job.result()
            
            for row in results:
                # Map API method to required permissions
                permissions = self.API_TO_PERMISSION_MAP.get(row.api_method, [row.api_method])
                
                pattern = APIUsagePattern(
                    principal=row.principal,
                    api_method=row.api_method,
                    service=row.service,
                    resource_type=row.resource,
                    frequency=row.frequency,
                    last_used=row.last_used,
                    required_permissions=permissions
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"BigQuery query failed: {e}")
        
        return patterns
    
    def _generate_mock_usage_patterns(self, principal_email: str) -> List[APIUsagePattern]:
        """Generate mock usage patterns for development"""
        mock_patterns = []
        
        # Simulate common usage patterns
        if "service-account" in principal_email:
            # Service account patterns - typically focused on specific services
            mock_apis = [
                ("storage.objects.get", "storage.googleapis.com", "bucket", 150),
                ("storage.objects.list", "storage.googleapis.com", "bucket", 100),
                ("storage.objects.create", "storage.googleapis.com", "bucket", 50),
                ("bigquery.tables.getData", "bigquery.googleapis.com", "table", 75),
                ("logging.entries.create", "logging.googleapis.com", "log", 200),
            ]
        else:
            # User patterns - broader access
            mock_apis = [
                ("compute.instances.list", "compute.googleapis.com", "instance", 50),
                ("compute.instances.get", "compute.googleapis.com", "instance", 30),
                ("storage.buckets.list", "storage.googleapis.com", "bucket", 40),
                ("iam.serviceAccounts.list", "iam.googleapis.com", "serviceAccount", 10),
                ("bigquery.tables.list", "bigquery.googleapis.com", "dataset", 20),
            ]
        
        for api_method, service, resource_type, frequency in mock_apis:
            permissions = self.API_TO_PERMISSION_MAP.get(api_method, [api_method])
            pattern = APIUsagePattern(
                principal=principal_email,
                api_method=api_method,
                service=service,
                resource_type=resource_type,
                frequency=frequency,
                last_used=datetime.utcnow() - timedelta(days=frequency % 7),
                required_permissions=permissions
            )
            mock_patterns.append(pattern)
        
        return mock_patterns
    
    def generate_role_recommendations(self, principal_email: str,
                                     usage_patterns: List[APIUsagePattern],
                                     current_roles: List[str]) -> RoleRecommendation:
        """Generate role recommendations based on usage patterns"""
        
        # Collect all required permissions
        required_permissions = set()
        for pattern in usage_patterns:
            required_permissions.update(pattern.required_permissions)
        
        # Find matching predefined roles
        recommended_roles = self._find_matching_roles(required_permissions)
        
        # Calculate unused permissions
        current_permissions = self._get_role_permissions(current_roles)
        unused_permissions = list(current_permissions - required_permissions)
        missing_permissions = list(required_permissions - current_permissions)
        
        # Determine if custom role is needed
        custom_role_needed = len(recommended_roles) == 0 or len(recommended_roles) > 3
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            len(usage_patterns),
            len(required_permissions),
            len(unused_permissions)
        )
        
        # Determine risk reduction
        risk_reduction = self._calculate_risk_reduction(
            len(unused_permissions),
            len(current_permissions) if current_permissions else 1
        )
        
        # Build recommendation
        recommendation = RoleRecommendation(
            principal=principal_email,
            principal_type=self._determine_principal_type(principal_email),
            current_roles=current_roles,
            recommended_roles=recommended_roles if recommended_roles else ["Custom role required"],
            custom_role_needed=custom_role_needed,
            unused_permissions=unused_permissions[:20],  # Limit to top 20
            missing_permissions=missing_permissions,
            confidence_score=confidence_score,
            analysis_period_days=30,
            api_calls_analyzed=sum(p.frequency for p in usage_patterns),
            cost_impact=self._estimate_cost_impact(current_roles, recommended_roles),
            risk_reduction=risk_reduction,
            compliance_impact=self._assess_compliance_impact(unused_permissions),
            recommendation_reason=self._generate_recommendation_reason(
                unused_permissions, missing_permissions, recommended_roles
            )
        )
        
        # Cache the recommendation
        self._cache_recommendation(recommendation)
        
        return recommendation
    
    def _find_matching_roles(self, required_permissions: Set[str]) -> List[str]:
        """Find predefined roles that match required permissions"""
        matching_roles = []
        
        # Check each predefined role mapping
        for perm_set, role in self.PERMISSION_TO_ROLE_MAP.items():
            if required_permissions.issubset(perm_set):
                matching_roles.append(role)
        
        # If no exact matches, find roles that cover most permissions
        if not matching_roles:
            coverage_scores = {}
            for perm_set, role in self.PERMISSION_TO_ROLE_MAP.items():
                coverage = len(required_permissions.intersection(perm_set)) / len(required_permissions)
                if coverage > 0.7:  # At least 70% coverage
                    coverage_scores[role] = coverage
            
            # Sort by coverage and take top 3
            matching_roles = sorted(coverage_scores.keys(), 
                                   key=lambda r: coverage_scores[r], 
                                   reverse=True)[:3]
        
        return matching_roles
    
    def _get_role_permissions(self, roles: List[str]) -> Set[str]:
        """Get all permissions for a list of roles"""
        all_permissions = set()
        
        # Simplified permission mapping for common roles
        role_permission_map = {
            "roles/owner": set(["*"]),  # Owner has all permissions
            "roles/editor": set(["*.*.create", "*.*.update", "*.*.delete", "*.*.get", "*.*.list"]),
            "roles/viewer": set(["*.*.get", "*.*.list"]),
            "roles/storage.objectViewer": set(["storage.objects.get", "storage.objects.list"]),
            "roles/storage.objectAdmin": set([
                "storage.objects.get", "storage.objects.list",
                "storage.objects.create", "storage.objects.delete", "storage.objects.update"
            ]),
            "roles/compute.viewer": set(["compute.*.get", "compute.*.list"]),
            "roles/bigquery.dataViewer": set([
                "bigquery.tables.get", "bigquery.tables.list", "bigquery.tables.getData"
            ]),
        }
        
        for role in roles:
            if role in role_permission_map:
                all_permissions.update(role_permission_map[role])
            else:
                # For unknown roles, assume basic permissions
                all_permissions.add(f"{role}.get")
                all_permissions.add(f"{role}.list")
        
        return all_permissions
    
    def _calculate_confidence_score(self, pattern_count: int, 
                                   permission_count: int,
                                   unused_count: int) -> float:
        """Calculate confidence score for recommendation"""
        base_score = 0.5
        
        # More patterns = higher confidence
        if pattern_count > 100:
            base_score += 0.3
        elif pattern_count > 50:
            base_score += 0.2
        elif pattern_count > 20:
            base_score += 0.1
        
        # High unused permission ratio = higher confidence in recommendation
        if permission_count > 0:
            unused_ratio = unused_count / permission_count
            if unused_ratio > 0.7:
                base_score += 0.2
            elif unused_ratio > 0.5:
                base_score += 0.1
        
        return min(base_score, 0.95)
    
    def _calculate_risk_reduction(self, unused_permissions: int, 
                                 total_permissions: int) -> str:
        """Calculate risk reduction level"""
        if total_permissions == 0:
            return "LOW"
        
        reduction_ratio = unused_permissions / total_permissions
        
        if reduction_ratio > 0.7:
            return "CRITICAL"
        elif reduction_ratio > 0.5:
            return "HIGH"
        elif reduction_ratio > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_principal_type(self, principal_email: str) -> str:
        """Determine the type of principal"""
        if "@" not in principal_email:
            return "unknown"
        elif ".gserviceaccount.com" in principal_email:
            return "serviceAccount"
        elif principal_email.startswith("group:"):
            return "group"
        else:
            return "user"
    
    def _estimate_cost_impact(self, current_roles: List[str], 
                             recommended_roles: List[str]) -> float:
        """Estimate cost impact of role changes"""
        # Simplified cost estimation based on role complexity
        role_costs = {
            "roles/owner": 100,
            "roles/editor": 80,
            "roles/viewer": 20,
        }
        
        current_cost = sum(role_costs.get(r, 10) for r in current_roles)
        recommended_cost = sum(role_costs.get(r, 10) for r in recommended_roles)
        
        return current_cost - recommended_cost
    
    def _assess_compliance_impact(self, unused_permissions: List[str]) -> List[str]:
        """Assess compliance impact of permission reduction"""
        compliance_impacts = []
        
        # Check for high-risk permissions
        high_risk_permissions = [
            "iam.serviceAccountKeys.create",
            "compute.instances.delete", 
            "storage.buckets.delete",
            "resourcemanager.projects.delete"
        ]
        
        if any(perm in unused_permissions for perm in high_risk_permissions):
            compliance_impacts.extend(["SOC2", "ISO27001"])
        
        if len(unused_permissions) > 50:
            compliance_impacts.append("GDPR")
        
        return compliance_impacts
    
    def _generate_recommendation_reason(self, unused_permissions: List[str],
                                       missing_permissions: List[str],
                                       recommended_roles: List[str]) -> str:
        """Generate human-readable recommendation reason"""
        reasons = []
        
        if len(unused_permissions) > 0:
            reasons.append(f"Current roles grant {len(unused_permissions)} unused permissions")
        
        if len(missing_permissions) > 0:
            reasons.append(f"Missing {len(missing_permissions)} required permissions")
        
        if recommended_roles and recommended_roles[0] != "Custom role required":
            reasons.append(f"Can be replaced with: {', '.join(recommended_roles[:3])}")
        elif recommended_roles:
            reasons.append("Requires custom role for optimal permissions")
        
        return ". ".join(reasons) if reasons else "No significant changes recommended"
    
    def _cache_usage_patterns(self, patterns: List[APIUsagePattern]):
        """Cache usage patterns in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            cursor.execute("""
                INSERT OR REPLACE INTO api_usage_patterns
                (principal, api_method, service, resource_type, frequency, last_used, required_permissions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.principal,
                pattern.api_method,
                pattern.service,
                pattern.resource_type,
                pattern.frequency,
                pattern.last_used.isoformat() if pattern.last_used else None,
                json.dumps(pattern.required_permissions)
            ))
        
        conn.commit()
        conn.close()
    
    def _get_cached_usage_patterns(self, principal_email: str) -> List[APIUsagePattern]:
        """Get cached usage patterns from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT api_method, service, resource_type, frequency, last_used, required_permissions
            FROM api_usage_patterns
            WHERE principal = ?
            ORDER BY frequency DESC
        """, (principal_email,))
        
        patterns = []
        for row in cursor.fetchall():
            pattern = APIUsagePattern(
                principal=principal_email,
                api_method=row[0],
                service=row[1],
                resource_type=row[2],
                frequency=row[3],
                last_used=datetime.fromisoformat(row[4]) if row[4] else None,
                required_permissions=json.loads(row[5]) if row[5] else []
            )
            patterns.append(pattern)
        
        conn.close()
        return patterns
    
    def _cache_recommendation(self, recommendation: RoleRecommendation):
        """Cache recommendation in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO role_recommendations
            (principal, principal_type, current_roles, recommended_roles, custom_role_needed,
             unused_permissions, missing_permissions, confidence_score, analysis_period_days,
             api_calls_analyzed, cost_impact, risk_reduction, compliance_impact,
             recommendation_reason, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            recommendation.principal,
            recommendation.principal_type,
            json.dumps(recommendation.current_roles),
            json.dumps(recommendation.recommended_roles),
            recommendation.custom_role_needed,
            json.dumps(recommendation.unused_permissions),
            json.dumps(recommendation.missing_permissions),
            recommendation.confidence_score,
            recommendation.analysis_period_days,
            recommendation.api_calls_analyzed,
            recommendation.cost_impact,
            recommendation.risk_reduction,
            json.dumps(recommendation.compliance_impact),
            recommendation.recommendation_reason,
            recommendation.analyzed_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_all_recommendations(self) -> List[RoleRecommendation]:
        """Get all cached recommendations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT principal, principal_type, current_roles, recommended_roles,
                   custom_role_needed, unused_permissions, missing_permissions,
                   confidence_score, analysis_period_days, api_calls_analyzed,
                   cost_impact, risk_reduction, compliance_impact,
                   recommendation_reason, analyzed_at
            FROM role_recommendations
            ORDER BY confidence_score DESC
        """)
        
        recommendations = []
        for row in cursor.fetchall():
            rec = RoleRecommendation(
                principal=row[0],
                principal_type=row[1],
                current_roles=json.loads(row[2]) if row[2] else [],
                recommended_roles=json.loads(row[3]) if row[3] else [],
                custom_role_needed=bool(row[4]),
                unused_permissions=json.loads(row[5]) if row[5] else [],
                missing_permissions=json.loads(row[6]) if row[6] else [],
                confidence_score=row[7],
                analysis_period_days=row[8],
                api_calls_analyzed=row[9],
                cost_impact=row[10],
                risk_reduction=row[11],
                compliance_impact=json.loads(row[12]) if row[12] else [],
                recommendation_reason=row[13],
                analyzed_at=datetime.fromisoformat(row[14]) if row[14] else datetime.utcnow()
            )
            recommendations.append(rec)
        
        conn.close()
        return recommendations