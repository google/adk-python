"""
Enhanced Recommendation Engine (STORY-007)

Provides comprehensive security recommendations with CVSS-based prioritization,
business impact scoring, and actionable remediation steps.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class RecommendationCategory(Enum):
    """Recommendation categories"""
    SECURITY = "SECURITY"
    IAM = "IAM"
    STORAGE = "STORAGE"
    NETWORK = "NETWORK"
    COST = "COST"
    COMPLIANCE = "COMPLIANCE"
    PERFORMANCE = "PERFORMANCE"


class BusinessImpact(Enum):
    """Business impact levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class Priority(Enum):
    """Recommendation priority levels"""
    P0 = "P0"  # Critical - Fix immediately
    P1 = "P1"  # High - Fix within 24 hours
    P2 = "P2"  # Medium - Fix within 1 week
    P3 = "P3"  # Low - Fix within 1 month
    P4 = "P4"  # Minimal - Fix during next review


@dataclass
class Recommendation:
    """Enhanced recommendation with CVSS and business impact"""
    id: str
    title: str
    description: str
    category: RecommendationCategory
    priority: Priority
    cvss_score: float  # 0.0-10.0
    business_impact: BusinessImpact
    business_impact_score: int  # 0-100
    affected_resources: List[str]
    remediation_steps: List[str]
    automation_script: Optional[str]
    estimated_effort_hours: float
    cost_impact: Optional[str]
    compliance_frameworks: List[str]
    related_findings: List[str]
    created_at: datetime
    due_date: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class RecommendationSummary:
    """Summary of all recommendations"""
    total_recommendations: int
    by_priority: Dict[str, int]
    by_category: Dict[str, int]
    by_business_impact: Dict[str, int]
    total_estimated_effort: float
    critical_count: int
    overdue_count: int
    estimated_risk_reduction: float
    recommendations: List[Recommendation]


class RecommendationEngine:
    """Enhanced recommendation engine with CVSS and business impact scoring"""
    
    # Business impact scoring weights
    BUSINESS_IMPACT_WEIGHTS = {
        'financial_loss': 25,
        'reputation_damage': 20,
        'operational_disruption': 20,
        'compliance_violation': 15,
        'data_exposure': 20
    }
    
    # CVSS to Priority mapping
    CVSS_PRIORITY_MAP = {
        (9.0, 10.0): Priority.P0,
        (7.0, 8.9): Priority.P1,
        (4.0, 6.9): Priority.P2,
        (1.0, 3.9): Priority.P3,
        (0.0, 0.9): Priority.P4
    }
    
    def __init__(self, project_id: str, backend_base_url: str = "http://localhost:8000"):
        self.project_id = project_id
        self.backend_base_url = backend_base_url
    
    async def generate_comprehensive_recommendations(self) -> RecommendationSummary:
        """
        Generate comprehensive security recommendations from all analysis sources
        """
        logger.info(f"Generating comprehensive recommendations for project {self.project_id}")
        
        # Collect findings from all security analysis sources
        security_findings = await self._collect_security_findings()
        iam_findings = await self._collect_iam_findings()
        storage_findings = await self._collect_storage_findings()
        
        # Generate recommendations from findings
        recommendations = []
        recommendations.extend(self._generate_security_recommendations(security_findings))
        recommendations.extend(self._generate_iam_recommendations(iam_findings))
        recommendations.extend(self._generate_storage_recommendations(storage_findings))
        recommendations.extend(self._generate_general_recommendations())
        
        # Calculate priority and business impact for each recommendation
        for rec in recommendations:
            rec.priority = self._calculate_priority(rec.cvss_score, rec.business_impact_score)
            rec.due_date = self._calculate_due_date(rec.priority)
        
        # Sort by priority and CVSS score
        recommendations.sort(key=lambda r: (r.priority.value, -r.cvss_score))
        
        # Generate summary
        summary = self._generate_summary(recommendations)
        
        logger.info(f"Generated {len(recommendations)} recommendations with {summary.critical_count} critical items")
        return summary
    
    async def _collect_security_findings(self) -> List[Dict[str, Any]]:
        """Collect findings from security analysis API"""
        if not HTTPX_AVAILABLE:
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_base_url}/api/v1/security/analyze")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("vulnerabilities", [])
        except Exception as e:
            logger.warning(f"Could not collect security findings: {e}")
        
        return []
    
    async def _collect_iam_findings(self) -> List[Dict[str, Any]]:
        """Collect findings from IAM analysis API"""
        if not HTTPX_AVAILABLE:
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_base_url}/api/v1/iam/analyze")
                if response.status_code == 200:
                    data = response.json()
                    if "analysis" in data and "findings" in data["analysis"]:
                        return data["analysis"]["findings"]
        except Exception as e:
            logger.warning(f"Could not collect IAM findings: {e}")
        
        return []
    
    async def _collect_storage_findings(self) -> List[Dict[str, Any]]:
        """Collect findings from storage analysis API"""
        if not HTTPX_AVAILABLE:
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_base_url}/api/v1/storage/analyze/{self.project_id}")
                if response.status_code == 200:
                    data = response.json()
                    if "analysis" in data and "findings" in data["analysis"]:
                        return data["analysis"]["findings"]
        except Exception as e:
            logger.warning(f"Could not collect storage findings: {e}")
        
        return []
    
    def _generate_security_recommendations(self, findings: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations from security findings"""
        recommendations = []
        
        for finding in findings:
            cvss_score = finding.get("cvss_score", 0.0)
            severity = finding.get("severity", "LOW")
            
            # Convert severity to business impact
            business_impact = self._severity_to_business_impact(severity)
            business_impact_score = self._calculate_business_impact_score(finding)
            
            recommendation = Recommendation(
                id=f"sec-{finding.get('id', 'unknown')}",
                title=f"Address {severity} Security Vulnerability",
                description=finding.get("description", "Security vulnerability detected"),
                category=RecommendationCategory.SECURITY,
                priority=Priority.P2,  # Will be recalculated
                cvss_score=cvss_score,
                business_impact=business_impact,
                business_impact_score=business_impact_score,
                affected_resources=[finding.get("resource", "Unknown")],
                remediation_steps=finding.get("remediation", ["Review and remediate finding"]),
                automation_script=self._generate_security_automation(finding),
                estimated_effort_hours=self._estimate_security_effort(severity),
                cost_impact=self._estimate_security_cost_impact(severity),
                compliance_frameworks=finding.get("compliance_frameworks", []),
                related_findings=[finding.get("id", "")],
                created_at=datetime.utcnow(),
                due_date=None,  # Will be calculated
                metadata={"source": "security_analysis", "finding_data": finding}
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_iam_recommendations(self, findings: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations from IAM findings"""
        recommendations = []
        
        for finding in findings:
            risk_score = finding.get("risk_score", 0)
            risk_level = finding.get("risk_level", "LOW")
            finding_type = finding.get("type", "UNKNOWN")
            
            # Convert IAM risk to CVSS equivalent
            cvss_score = self._risk_score_to_cvss(risk_score)
            business_impact = self._risk_level_to_business_impact(risk_level)
            business_impact_score = min(risk_score, 100)
            
            recommendation = Recommendation(
                id=f"iam-{finding.get('resource_name', 'unknown').split('/')[-1]}",
                title=finding.get("title", f"Address {finding_type}"),
                description=finding.get("description", "IAM security issue detected"),
                category=RecommendationCategory.IAM,
                priority=Priority.P2,  # Will be recalculated
                cvss_score=cvss_score,
                business_impact=business_impact,
                business_impact_score=business_impact_score,
                affected_resources=[finding.get("resource_name", "Unknown")],
                remediation_steps=finding.get("remediation_steps", ["Review IAM configuration"]),
                automation_script=self._generate_iam_automation(finding),
                estimated_effort_hours=self._estimate_iam_effort(finding_type),
                cost_impact="$0 - Security improvement",
                compliance_frameworks=["SOC2", "ISO27001"],
                related_findings=[finding.get("resource_name", "")],
                created_at=datetime.utcnow(),
                due_date=None,  # Will be calculated
                metadata={"source": "iam_analysis", "finding_data": finding}
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_storage_recommendations(self, findings: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations from storage findings"""
        recommendations = []
        
        for finding in findings:
            risk_score = finding.get("risk_score", 0)
            risk_level = finding.get("risk_level", "LOW")
            finding_type = finding.get("type", "UNKNOWN")
            
            # Convert storage risk to CVSS equivalent
            cvss_score = self._risk_score_to_cvss(risk_score)
            business_impact = self._risk_level_to_business_impact(risk_level)
            business_impact_score = min(risk_score, 100)
            
            recommendation = Recommendation(
                id=f"storage-{finding.get('bucket_name', 'unknown')}",
                title=finding.get("title", f"Address {finding_type}"),
                description=finding.get("description", "Storage security issue detected"),
                category=RecommendationCategory.STORAGE,
                priority=Priority.P2,  # Will be recalculated
                cvss_score=cvss_score,
                business_impact=business_impact,
                business_impact_score=business_impact_score,
                affected_resources=[f"gs://{finding.get('bucket_name', 'unknown')}"],
                remediation_steps=finding.get("remediation_steps", ["Review storage configuration"]),
                automation_script=self._generate_storage_automation(finding),
                estimated_effort_hours=self._estimate_storage_effort(finding_type),
                cost_impact=self._estimate_storage_cost_impact(finding_type),
                compliance_frameworks=finding.get("compliance_frameworks", []),
                related_findings=[finding.get("bucket_name", "")],
                created_at=datetime.utcnow(),
                due_date=None,  # Will be calculated
                metadata={"source": "storage_analysis", "finding_data": finding}
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_general_recommendations(self) -> List[Recommendation]:
        """Generate general security best practice recommendations"""
        recommendations = []
        
        # General security recommendations
        general_recs = [
            {
                "id": "gen-001",
                "title": "Enable Security Command Center Standard Tier",
                "description": "Upgrade to Security Command Center Standard for advanced threat detection",
                "category": RecommendationCategory.SECURITY,
                "cvss_score": 6.5,
                "business_impact": BusinessImpact.MEDIUM,
                "business_impact_score": 65,
                "remediation_steps": [
                    "Navigate to Security Command Center in Cloud Console",
                    "Upgrade to Standard tier",
                    "Configure security findings notifications",
                    "Set up automated response workflows"
                ],
                "estimated_effort_hours": 2.0,
                "cost_impact": "$200-500/month depending on project size",
                "compliance_frameworks": ["SOC2", "ISO27001"]
            },
            {
                "id": "gen-002", 
                "title": "Implement Organization Policies",
                "description": "Configure organization-wide security policies for consistent governance",
                "category": RecommendationCategory.COMPLIANCE,
                "cvss_score": 5.5,
                "business_impact": BusinessImpact.MEDIUM,
                "business_impact_score": 55,
                "remediation_steps": [
                    "Review current organization policies",
                    "Implement public IP restriction policy",
                    "Configure service account key creation restrictions",
                    "Set up uniform bucket-level access enforcement"
                ],
                "estimated_effort_hours": 4.0,
                "cost_impact": "$0 - Policy enforcement only",
                "compliance_frameworks": ["SOC2", "HIPAA", "PCI_DSS"]
            },
            {
                "id": "gen-003",
                "title": "Enable VPC Flow Logs",
                "description": "Monitor network traffic for security analysis and compliance",
                "category": RecommendationCategory.NETWORK,
                "cvss_score": 4.5,
                "business_impact": BusinessImpact.LOW,
                "business_impact_score": 45,
                "remediation_steps": [
                    "Navigate to VPC networks in Cloud Console",
                    "Enable flow logs for all subnets",
                    "Configure log sampling and retention",
                    "Set up monitoring and alerting"
                ],
                "estimated_effort_hours": 1.5,
                "cost_impact": "$50-200/month depending on traffic",
                "compliance_frameworks": ["SOC2", "PCI_DSS"]
            }
        ]
        
        for rec_data in general_recs:
            recommendation = Recommendation(
                id=rec_data["id"],
                title=rec_data["title"],
                description=rec_data["description"],
                category=rec_data["category"],
                priority=Priority.P2,  # Will be recalculated
                cvss_score=rec_data["cvss_score"],
                business_impact=rec_data["business_impact"],
                business_impact_score=rec_data["business_impact_score"],
                affected_resources=[f"Project: {self.project_id}"],
                remediation_steps=rec_data["remediation_steps"],
                automation_script=None,
                estimated_effort_hours=rec_data["estimated_effort_hours"],
                cost_impact=rec_data["cost_impact"],
                compliance_frameworks=rec_data["compliance_frameworks"],
                related_findings=[],
                created_at=datetime.utcnow(),
                due_date=None,  # Will be calculated
                metadata={"source": "general_recommendations"}
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_priority(self, cvss_score: float, business_impact_score: int) -> Priority:
        """Calculate priority based on CVSS score and business impact"""
        # Weighted priority calculation
        weighted_score = (cvss_score * 0.7) + (business_impact_score / 10 * 0.3)
        
        for score_range, priority in self.CVSS_PRIORITY_MAP.items():
            if score_range[0] <= weighted_score <= score_range[1]:
                return priority
        
        return Priority.P4
    
    def _calculate_due_date(self, priority: Priority) -> datetime:
        """Calculate due date based on priority"""
        now = datetime.utcnow()
        
        if priority == Priority.P0:
            return now + timedelta(hours=4)  # 4 hours
        elif priority == Priority.P1:
            return now + timedelta(days=1)  # 24 hours
        elif priority == Priority.P2:
            return now + timedelta(days=7)  # 1 week
        elif priority == Priority.P3:
            return now + timedelta(days=30)  # 1 month
        else:
            return now + timedelta(days=90)  # 3 months
    
    def _calculate_business_impact_score(self, finding: Dict[str, Any]) -> int:
        """Calculate business impact score from finding details"""
        score = 0
        
        # Base score from severity/risk level
        if "severity" in finding:
            severity = finding["severity"].upper()
            if severity == "CRITICAL":
                score += 40
            elif severity == "HIGH":
                score += 30
            elif severity == "MEDIUM":
                score += 20
            elif severity == "LOW":
                score += 10
        
        # Additional factors
        if finding.get("public_exposure", False):
            score += 25
        
        if finding.get("data_exposure", False):
            score += 20
        
        if finding.get("privilege_escalation", False):
            score += 15
        
        return min(score, 100)
    
    def _severity_to_business_impact(self, severity: str) -> BusinessImpact:
        """Convert security severity to business impact"""
        severity = severity.upper()
        
        if severity == "CRITICAL":
            return BusinessImpact.CRITICAL
        elif severity == "HIGH":
            return BusinessImpact.HIGH
        elif severity == "MEDIUM":
            return BusinessImpact.MEDIUM
        elif severity == "LOW":
            return BusinessImpact.LOW
        else:
            return BusinessImpact.MINIMAL
    
    def _risk_level_to_business_impact(self, risk_level: str) -> BusinessImpact:
        """Convert risk level to business impact"""
        risk_level = risk_level.upper()
        
        if risk_level == "CRITICAL":
            return BusinessImpact.CRITICAL
        elif risk_level == "HIGH":
            return BusinessImpact.HIGH
        elif risk_level == "MEDIUM":
            return BusinessImpact.MEDIUM
        elif risk_level == "LOW":
            return BusinessImpact.LOW
        else:
            return BusinessImpact.MINIMAL
    
    def _risk_score_to_cvss(self, risk_score: int) -> float:
        """Convert risk score (0-100) to CVSS equivalent (0-10)"""
        return (risk_score / 100.0) * 10.0
    
    def _estimate_security_effort(self, severity: str) -> float:
        """Estimate effort hours for security remediation"""
        severity = severity.upper()
        
        if severity == "CRITICAL":
            return 8.0
        elif severity == "HIGH":
            return 4.0
        elif severity == "MEDIUM":
            return 2.0
        else:
            return 1.0
    
    def _estimate_iam_effort(self, finding_type: str) -> float:
        """Estimate effort hours for IAM remediation"""
        if "ADMIN_ROLE" in finding_type:
            return 3.0
        elif "OVERPRIVILEGED" in finding_type:
            return 2.0
        elif "STALE_KEY" in finding_type:
            return 0.5
        else:
            return 1.0
    
    def _estimate_storage_effort(self, finding_type: str) -> float:
        """Estimate effort hours for storage remediation"""
        if "PUBLIC_BUCKET" in finding_type:
            return 1.0
        elif "ENCRYPTION" in finding_type:
            return 2.0
        elif "LIFECYCLE" in finding_type:
            return 1.5
        else:
            return 1.0
    
    def _estimate_security_cost_impact(self, severity: str) -> str:
        """Estimate cost impact for security remediation"""
        severity = severity.upper()
        
        if severity == "CRITICAL":
            return "$0 - Security improvement (potential savings from breach prevention: $100K+)"
        elif severity == "HIGH":
            return "$0 - Security improvement (potential savings from breach prevention: $50K+)"
        else:
            return "$0 - Security improvement"
    
    def _estimate_storage_cost_impact(self, finding_type: str) -> str:
        """Estimate cost impact for storage remediation"""
        if "LIFECYCLE" in finding_type:
            return "Potential savings: $100-500/month through automated cleanup"
        elif "ENCRYPTION" in finding_type:
            return "Additional cost: $10-50/month for CMEK"
        else:
            return "$0 - Security improvement"
    
    def _generate_security_automation(self, finding: Dict[str, Any]) -> Optional[str]:
        """Generate automation script for security finding"""
        # Placeholder for security automation scripts
        return None
    
    def _generate_iam_automation(self, finding: Dict[str, Any]) -> Optional[str]:
        """Generate automation script for IAM finding"""
        finding_type = finding.get("type", "")
        
        if "STALE_KEY" in finding_type:
            return f"""
# Rotate stale service account key
gcloud iam service-accounts keys create new-key.json \\
    --iam-account={finding.get('affected_principal', 'SERVICE_ACCOUNT')}

# Delete old key after validation
gcloud iam service-accounts keys delete {finding.get('key_id', 'OLD_KEY_ID')} \\
    --iam-account={finding.get('affected_principal', 'SERVICE_ACCOUNT')}
"""
        
        return None
    
    def _generate_storage_automation(self, finding: Dict[str, Any]) -> Optional[str]:
        """Generate automation script for storage finding"""
        finding_type = finding.get("type", "")
        bucket_name = finding.get("bucket_name", "BUCKET_NAME")
        
        if "PUBLIC_BUCKET" in finding_type:
            return f"""
# Remove public access from bucket
gsutil iam ch -d allUsers gs://{bucket_name}
gsutil iam ch -d allAuthenticatedUsers gs://{bucket_name}

# Enable public access prevention
gcloud storage buckets update gs://{bucket_name} --public-access-prevention
"""
        elif "MISSING_ENCRYPTION" in finding_type:
            return f"""
# Enable customer-managed encryption
gcloud storage buckets update gs://{bucket_name} \\
    --default-encryption-key=projects/PROJECT_ID/locations/LOCATION/keyRings/RING/cryptoKeys/KEY
"""
        
        return None
    
    def _generate_summary(self, recommendations: List[Recommendation]) -> RecommendationSummary:
        """Generate summary of all recommendations"""
        # Count by priority
        by_priority = {p.value: 0 for p in Priority}
        for rec in recommendations:
            by_priority[rec.priority.value] += 1
        
        # Count by category
        by_category = {c.value: 0 for c in RecommendationCategory}
        for rec in recommendations:
            by_category[rec.category.value] += 1
        
        # Count by business impact
        by_business_impact = {b.value: 0 for b in BusinessImpact}
        for rec in recommendations:
            by_business_impact[rec.business_impact.value] += 1
        
        # Calculate totals
        total_effort = sum(rec.estimated_effort_hours for rec in recommendations)
        critical_count = len([r for r in recommendations if r.priority in [Priority.P0, Priority.P1]])
        overdue_count = len([r for r in recommendations if r.due_date and r.due_date < datetime.utcnow()])
        
        # Estimate risk reduction
        estimated_risk_reduction = sum(rec.cvss_score for rec in recommendations if rec.priority in [Priority.P0, Priority.P1])
        
        return RecommendationSummary(
            total_recommendations=len(recommendations),
            by_priority=by_priority,
            by_category=by_category,
            by_business_impact=by_business_impact,
            total_estimated_effort=total_effort,
            critical_count=critical_count,
            overdue_count=overdue_count,
            estimated_risk_reduction=estimated_risk_reduction,
            recommendations=recommendations
        )