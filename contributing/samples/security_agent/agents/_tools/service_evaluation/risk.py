#!/usr/bin/env python3
"""
Risk Assessment Engine
Calculate comprehensive risk scores for new GCP services
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class RiskFactor(Enum):
    """Risk assessment factors"""
    DATA_SENSITIVITY = "data_sensitivity"
    NETWORK_EXPOSURE = "network_exposure"
    COMPLIANCE_IMPACT = "compliance_impact"
    BLAST_RADIUS = "blast_radius"
    EXISTING_CONTROLS = "existing_controls"
    VULNERABILITY_HISTORY = "vulnerability_history"
    PRIVILEGE_LEVEL = "privilege_level"
    CRITICALITY = "business_criticality"


@dataclass
class RiskScore:
    """Individual risk score"""
    factor: RiskFactor
    score: int  # 0-100
    weight: float  # 0-1
    rationale: str  # Reasoning for the score
    mitigations: List[str]
    weighted_score: float = 0.0  # Calculated as score * weight

    def __post_init__(self):
        """Calculate weighted score after initialization"""
        if self.weighted_score == 0.0:
            self.weighted_score = self.score * self.weight


class RiskLevel(Enum):
    """Overall risk levels"""
    CRITICAL = "critical"  # 80-100
    HIGH = "high"  # 60-79
    MEDIUM = "medium"  # 40-59
    LOW = "low"  # 0-39


@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    service_name: str
    service_type: str = ""
    overall_score: int = 0  # 0-100
    risk_level: RiskLevel = RiskLevel.LOW
    factor_scores: List[RiskScore] = None
    risk_summary: str = ""
    mitigation_priorities: List[str] = None
    compliance_considerations: List[str] = None
    recommendations: List[str] = None
    required_mitigations: List[str] = None
    assessment_date: str = ""

    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.factor_scores is None:
            self.factor_scores = []
        if self.mitigation_priorities is None:
            self.mitigation_priorities = []
        if self.compliance_considerations is None:
            self.compliance_considerations = []
        if self.recommendations is None:
            self.recommendations = []
        if self.required_mitigations is None:
            self.required_mitigations = []


@dataclass
class ServiceProfile:
    """Detailed service configuration profile for risk assessment"""
    service_name: str
    service_type: str
    use_case: str
    data_classification: str  # public, internal, confidential, restricted
    network_exposure: str  # internal, vpc, public
    authentication_method: str  # iam, api_key, oauth, etc.
    encryption_at_rest: bool
    encryption_in_transit: bool
    compliance_requirements: List[str]  # e.g., ["HIPAA", "PCI-DSS", "GDPR"]
    third_party_integrations: List[str]
    expected_data_volume: str  # low, medium, high


class RiskAssessmentEngine:
    """Comprehensive risk assessment for GCP services"""

    def __init__(self):
        self.risk_weights = self._initialize_risk_weights()
        self.service_profiles = self._initialize_service_profiles()

    def _initialize_risk_weights(self) -> Dict[RiskFactor, float]:
        """Define weights for each risk factor"""
        return {
            RiskFactor.DATA_SENSITIVITY: 0.25,  # Highest weight
            RiskFactor.COMPLIANCE_IMPACT: 0.20,
            RiskFactor.NETWORK_EXPOSURE: 0.15,
            RiskFactor.PRIVILEGE_LEVEL: 0.15,
            RiskFactor.BLAST_RADIUS: 0.10,
            RiskFactor.CRITICALITY: 0.08,
            RiskFactor.EXISTING_CONTROLS: 0.04,
            RiskFactor.VULNERABILITY_HISTORY: 0.03
        }

    def _initialize_service_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Define baseline risk profiles for service types"""
        return {
            'storage': {
                'base_risk': 60,
                'high_risk_factors': ['DATA_SENSITIVITY', 'COMPLIANCE_IMPACT'],
                'common_issues': ['Public access', 'Unencrypted data', 'No retention policy']
            },
            'compute': {
                'base_risk': 50,
                'high_risk_factors': ['NETWORK_EXPOSURE', 'PRIVILEGE_LEVEL'],
                'common_issues': ['External IPs', 'SSH access', 'Unpatched OS']
            },
            'bigquery': {
                'base_risk': 65,
                'high_risk_factors': ['DATA_SENSITIVITY', 'COMPLIANCE_IMPACT'],
                'common_issues': ['Public datasets', 'No CMEK', 'Broad query access']
            },
            'kubernetes': {
                'base_risk': 70,
                'high_risk_factors': ['BLAST_RADIUS', 'NETWORK_EXPOSURE', 'PRIVILEGE_LEVEL'],
                'common_issues': ['Public clusters', 'RBAC misconfiguration', 'Container vulnerabilities']
            },
            'cloudsql': {
                'base_risk': 75,
                'high_risk_factors': ['DATA_SENSITIVITY', 'NETWORK_EXPOSURE', 'COMPLIANCE_IMPACT'],
                'common_issues': ['Public IP', 'Weak passwords', 'No backups', 'SSL not enforced']
            },
            'cloud_functions': {
                'base_risk': 45,
                'high_risk_factors': ['PRIVILEGE_LEVEL', 'DATA_SENSITIVITY'],
                'common_issues': ['Overly permissive IAM', 'Secrets in code', 'No VPC connector']
            },
            'pubsub': {
                'base_risk': 40,
                'high_risk_factors': ['DATA_SENSITIVITY', 'COMPLIANCE_IMPACT'],
                'common_issues': ['No encryption', 'Public topics', 'No DLQ']
            },
            'iam': {
                'base_risk': 85,
                'high_risk_factors': ['PRIVILEGE_LEVEL', 'BLAST_RADIUS'],
                'common_issues': ['Primitive roles', 'Service account keys', 'No MFA']
            }
        }

    def assess_service(self, service_name: str, service_type: str,
                      attributes: Dict[str, Any]) -> RiskAssessment:
        """
        Perform comprehensive risk assessment

        Args:
            service_name: Name of the service being evaluated
            service_type: Type (storage, compute, bigquery, etc.)
            attributes: Service-specific attributes for assessment
        """
        from datetime import datetime

        # Get service profile
        profile = self.service_profiles.get(service_type, {'base_risk': 50})

        # Calculate individual risk factors
        factor_scores = []

        # 1. Data Sensitivity
        data_sensitivity_score = self._assess_data_sensitivity(attributes)
        factor_scores.append(RiskScore(
            factor=RiskFactor.DATA_SENSITIVITY,
            score=data_sensitivity_score,
            weight=self.risk_weights[RiskFactor.DATA_SENSITIVITY],
            rationale=self._get_data_sensitivity_reasoning(data_sensitivity_score, attributes),
            mitigations=self._get_data_sensitivity_mitigations(data_sensitivity_score)
        ))

        # 2. Network Exposure
        network_score = self._assess_network_exposure(service_type, attributes)
        factor_scores.append(RiskScore(
            factor=RiskFactor.NETWORK_EXPOSURE,
            score=network_score,
            weight=self.risk_weights[RiskFactor.NETWORK_EXPOSURE],
            rationale=self._get_network_reasoning(network_score, attributes),
            mitigations=self._get_network_mitigations(network_score, service_type)
        ))

        # 3. Compliance Impact
        compliance_score = self._assess_compliance_impact(attributes)
        factor_scores.append(RiskScore(
            factor=RiskFactor.COMPLIANCE_IMPACT,
            score=compliance_score,
            weight=self.risk_weights[RiskFactor.COMPLIANCE_IMPACT],
            rationale=self._get_compliance_reasoning(compliance_score, attributes),
            mitigations=self._get_compliance_mitigations(compliance_score)
        ))

        # 4. Privilege Level
        privilege_score = self._assess_privilege_level(service_type, attributes)
        factor_scores.append(RiskScore(
            factor=RiskFactor.PRIVILEGE_LEVEL,
            score=privilege_score,
            weight=self.risk_weights[RiskFactor.PRIVILEGE_LEVEL],
            rationale=f"Service requires {attributes.get('required_permissions', 'standard')} permissions",
            mitigations=self._get_privilege_mitigations(privilege_score)
        ))

        # 5. Blast Radius
        blast_radius_score = self._assess_blast_radius(service_type, attributes)
        factor_scores.append(RiskScore(
            factor=RiskFactor.BLAST_RADIUS,
            score=blast_radius_score,
            weight=self.risk_weights[RiskFactor.BLAST_RADIUS],
            rationale=f"Potential impact scope: {attributes.get('scope', 'project-level')}",
            mitigations=["Implement resource isolation", "Use separate projects", "Apply org policies"]
        ))

        # 6. Business Criticality
        criticality_score = attributes.get('criticality_score', 50)
        factor_scores.append(RiskScore(
            factor=RiskFactor.CRITICALITY,
            score=criticality_score,
            weight=self.risk_weights[RiskFactor.CRITICALITY],
            rationale=f"Business criticality: {attributes.get('criticality', 'medium')}",
            mitigations=["Implement high availability", "Configure backups", "Set up monitoring"]
        ))

        # 7. Existing Controls
        controls_score = self._assess_existing_controls(attributes)
        factor_scores.append(RiskScore(
            factor=RiskFactor.EXISTING_CONTROLS,
            score=100 - controls_score,  # Inverse - more controls = lower risk
            weight=self.risk_weights[RiskFactor.EXISTING_CONTROLS],
            rationale=f"{attributes.get('existing_controls_count', 0)} existing controls in place",
            mitigations=[]
        ))

        # Calculate weighted overall score
        overall_score = sum(
            score.score * score.weight
            for score in factor_scores
        )
        overall_score = int(overall_score)

        # Determine risk level
        if overall_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif overall_score >= 60:
            risk_level = RiskLevel.HIGH
        elif overall_score >= 40:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Generate recommendations
        recommendations = self._generate_recommendations(
            service_type, overall_score, factor_scores, attributes
        )

        # Identify required mitigations
        required_mitigations = [
            mitigation
            for score in factor_scores
            if score.score >= 70  # High risk factors
            for mitigation in score.mitigations
        ]

        # Generate risk summary
        risk_summary = f"{service_name} has been assessed as {risk_level.value.upper()} risk with a score of {overall_score}/100. "
        high_risk_factors = [s.factor.value for s in factor_scores if s.score >= 70]
        if high_risk_factors:
            risk_summary += f"Primary concerns: {', '.join(high_risk_factors)}."

        # Identify mitigation priorities (top 3 highest scoring factors)
        sorted_factors = sorted(factor_scores, key=lambda x: x.score, reverse=True)
        mitigation_priorities = [
            f"{s.factor.value}: {s.rationale}"
            for s in sorted_factors[:3]
        ]

        # Extract compliance considerations
        compliance_requirements = attributes.get('compliance_requirements', [])
        compliance_considerations = []
        if compliance_requirements:
            for req in compliance_requirements:
                compliance_considerations.append(
                    f"{req} compliance requires specific controls and audit logging"
                )

        return RiskAssessment(
            service_name=service_name,
            service_type=service_type,
            overall_score=overall_score,
            risk_level=risk_level,
            factor_scores=factor_scores,
            risk_summary=risk_summary,
            mitigation_priorities=mitigation_priorities,
            compliance_considerations=compliance_considerations,
            recommendations=recommendations,
            required_mitigations=list(set(required_mitigations)),  # Deduplicate
            assessment_date=datetime.utcnow().isoformat()
        )

    def assess_service_risk(self, service_profile: ServiceProfile) -> RiskAssessment:
        """
        Perform risk assessment using ServiceProfile object

        Args:
            service_profile: ServiceProfile with detailed service configuration

        Returns:
            RiskAssessment with comprehensive risk analysis
        """
        # Convert ServiceProfile to attributes dict
        attributes = {
            'data_classification': service_profile.data_classification,
            'network_exposure': service_profile.network_exposure,
            'authentication_method': service_profile.authentication_method,
            'encryption_at_rest': service_profile.encryption_at_rest,
            'encryption_in_transit': service_profile.encryption_in_transit,
            'compliance_requirements': service_profile.compliance_requirements,
            'third_party_integrations': service_profile.third_party_integrations,
            'expected_data_volume': service_profile.expected_data_volume,
            'use_case': service_profile.use_case,
            # Map data classification to data types for sensitivity assessment
            'data_types': self._map_classification_to_types(service_profile.data_classification),
            # Map network exposure to risk attributes
            'public_access': service_profile.network_exposure == 'public',
            'external_ip': service_profile.network_exposure == 'public',
            'vpc_enabled': service_profile.network_exposure == 'internal',
            # Map compliance requirements to frameworks
            'compliance_frameworks': service_profile.compliance_requirements,
            # Infer other attributes
            'requires_cmek': service_profile.data_classification in ['confidential', 'restricted'],
            'has_public_access': service_profile.network_exposure == 'public',
            'has_external_integrations': len(service_profile.third_party_integrations) > 0,
            'existing_controls_count': 5 if service_profile.encryption_at_rest else 2
        }

        # Call the main assess_service method
        return self.assess_service(
            service_name=service_profile.service_name,
            service_type=service_profile.service_type,
            attributes=attributes
        )

    def _map_classification_to_types(self, classification: str) -> List[str]:
        """Map data classification to data types for risk assessment"""
        mapping = {
            'public': ['public'],
            'internal': ['internal'],
            'confidential': ['customer_data', 'financial'],
            'restricted': ['pii', 'phi', 'credentials', 'trade_secret']
        }
        return mapping.get(classification, ['internal'])

    def _assess_data_sensitivity(self, attributes: Dict[str, Any]) -> int:
        """Assess data sensitivity risk (0-100)"""
        data_types = attributes.get('data_types', [])
        sensitivity_scores = {
            'pii': 90,
            'phi': 95,
            'financial': 85,
            'credentials': 100,
            'trade_secret': 90,
            'customer_data': 70,
            'internal': 40,
            'public': 10
        }

        if not data_types:
            return 50  # Unknown = medium risk

        max_sensitivity = max(
            sensitivity_scores.get(dt.lower(), 50)
            for dt in data_types
        )
        return max_sensitivity

    def _assess_network_exposure(self, service_type: str, attributes: Dict[str, Any]) -> int:
        """Assess network exposure risk (0-100)"""
        score = 0

        if attributes.get('public_access', False):
            score += 50
        if attributes.get('external_ip', False):
            score += 30
        if not attributes.get('vpc_enabled', False):
            score += 20

        return min(score, 100)

    def _assess_compliance_impact(self, attributes: Dict[str, Any]) -> int:
        """Assess compliance impact (0-100)"""
        frameworks = attributes.get('compliance_frameworks', [])
        framework_scores = {
            'pci-dss': 90,
            'hipaa': 95,
            'sox': 85,
            'gdpr': 80,
            'fedramp': 90,
            'iso27001': 70
        }

        if not frameworks:
            return 30

        max_impact = max(
            framework_scores.get(fw.lower(), 50)
            for fw in frameworks
        )
        return max_impact

    def _assess_privilege_level(self, service_type: str, attributes: Dict[str, Any]) -> int:
        """Assess required privilege level risk (0-100)"""
        required_roles = attributes.get('required_roles', [])

        # Check for high-privilege roles
        high_privilege_keywords = ['owner', 'editor', 'admin', 'write']
        privilege_score = 0

        for role in required_roles:
            role_lower = role.lower()
            if any(keyword in role_lower for keyword in high_privilege_keywords):
                privilege_score = max(privilege_score, 80)
            elif 'viewer' in role_lower or 'reader' in role_lower:
                privilege_score = max(privilege_score, 20)
            else:
                privilege_score = max(privilege_score, 50)

        return privilege_score if required_roles else 40

    def _assess_blast_radius(self, service_type: str, attributes: Dict[str, Any]) -> int:
        """Assess potential blast radius (0-100)"""
        scope = attributes.get('scope', 'project')
        scope_scores = {
            'organization': 100,
            'folder': 80,
            'project': 50,
            'resource': 30
        }
        return scope_scores.get(scope, 50)

    def _assess_existing_controls(self, attributes: Dict[str, Any]) -> int:
        """Assess effectiveness of existing controls (0-100, higher = better)"""
        control_count = attributes.get('existing_controls_count', 0)
        control_effectiveness = attributes.get('control_effectiveness', 0.5)

        # Score based on quantity and quality
        quantity_score = min(control_count * 10, 50)  # Up to 50 points
        quality_score = control_effectiveness * 50    # Up to 50 points

        return int(quantity_score + quality_score)

    def _get_data_sensitivity_reasoning(self, score: int, attributes: Dict[str, Any]) -> str:
        """Generate reasoning for data sensitivity score"""
        data_types = attributes.get('data_types', [])
        if score >= 90:
            return f"Highly sensitive data types: {', '.join(data_types)}. Requires strict controls."
        elif score >= 60:
            return f"Contains sensitive data: {', '.join(data_types)}. Enhanced protection recommended."
        else:
            return f"Low sensitivity data. Standard protections adequate."

    def _get_data_sensitivity_mitigations(self, score: int) -> List[str]:
        """Get mitigations for data sensitivity"""
        if score >= 90:
            return [
                "Implement CMEK encryption",
                "Enable DLP scanning",
                "Configure VPC Service Controls",
                "Implement data classification labels",
                "Require data access justification"
            ]
        elif score >= 60:
            return [
                "Enable encryption at rest",
                "Implement access logging",
                "Configure retention policies"
            ]
        else:
            return ["Enable standard encryption", "Configure basic access controls"]

    def _get_network_reasoning(self, score: int, attributes: Dict[str, Any]) -> str:
        """Generate reasoning for network exposure"""
        if score >= 80:
            return "High network exposure. Public access with external IPs."
        elif score >= 50:
            return "Moderate exposure. Some public-facing components."
        else:
            return "Low exposure. Private network configuration."

    def _get_network_mitigations(self, score: int, service_type: str) -> List[str]:
        """Get network exposure mitigations"""
        if score >= 80:
            return [
                "Remove public access",
                "Implement VPC Service Controls",
                "Use Cloud Armor for DDoS protection",
                "Configure firewall rules with minimal allowed sources",
                "Enable Private Google Access"
            ]
        elif score >= 50:
            return [
                "Restrict source IP ranges",
                "Enable Cloud Armor",
                "Configure firewall rules"
            ]
        else:
            return ["Maintain current private configuration"]

    def _get_compliance_reasoning(self, score: int, attributes: Dict[str, Any]) -> str:
        """Generate compliance impact reasoning"""
        frameworks = attributes.get('compliance_frameworks', [])
        if score >= 90:
            return f"Critical compliance requirements: {', '.join(frameworks)}"
        elif score >= 60:
            return f"Moderate compliance impact: {', '.join(frameworks)}"
        else:
            return "Low compliance impact"

    def _get_compliance_mitigations(self, score: int) -> List[str]:
        """Get compliance mitigations"""
        if score >= 90:
            return [
                "Enable comprehensive audit logging",
                "Implement compliance-specific controls",
                "Configure automated compliance checking",
                "Set up security monitoring alerts",
                "Conduct regular compliance audits"
            ]
        elif score >= 60:
            return [
                "Enable audit logging",
                "Configure security monitoring",
                "Implement required controls"
            ]
        else:
            return ["Standard compliance controls"]

    def _get_privilege_mitigations(self, score: int) -> List[str]:
        """Get privilege level mitigations"""
        if score >= 80:
            return [
                "Use least-privilege custom roles",
                "Avoid primitive roles (Owner, Editor)",
                "Implement service account best practices",
                "Enable workload identity",
                "Require approval for elevated access"
            ]
        else:
            return ["Use predefined roles where possible", "Review permissions regularly"]

    def _generate_recommendations(self, service_type: str, overall_score: int,
                                 factor_scores: List[RiskScore],
                                 attributes: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []

        if overall_score >= 80:
            recommendations.append("⚠️ CRITICAL RISK: Implement all required mitigations before deployment")
            recommendations.append("Require CISO approval for this service")
            recommendations.append("Conduct security architecture review")

        elif overall_score >= 60:
            recommendations.append("⚠️ HIGH RISK: Address critical security controls before production use")
            recommendations.append("Require security team approval")

        # Service-specific recommendations
        service_recs = {
            'storage': [
                "Configure uniform bucket-level access",
                "Enable versioning for data protection",
                "Set up lifecycle policies"
            ],
            'cloudsql': [
                "Enable automated backups",
                "Use Cloud SQL Proxy for connections",
                "Rotate passwords regularly"
            ],
            'kubernetes': [
                "Enable binary authorization",
                "Configure pod security policies",
                "Use Workload Identity for service accounts"
            ]
        }

        recommendations.extend(service_recs.get(service_type, []))

        return recommendations
