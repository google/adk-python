#!/usr/bin/env python3
"""
Approval Workflow Manager
Determines approval requirements based on risk assessment
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from .risk import RiskLevel, RiskAssessment


class ApprovalLevel(Enum):
    """Approval authority levels"""
    SELF_SERVICE = "self_service"
    TEAM_LEAD = "team_lead"
    SECURITY_TEAM = "security_team"
    CISO = "ciso"
    SECURITY_BOARD = "security_board"


class ApprovalType(Enum):
    """Types of approvals required"""
    TECHNICAL = "technical_review"
    SECURITY = "security_review"
    COMPLIANCE = "compliance_review"
    ARCHITECTURE = "architecture_review"
    EXECUTIVE = "executive_review"


@dataclass
class ApprovalRequirement:
    """Individual approval requirement"""
    approval_type: ApprovalType
    approval_level: ApprovalLevel
    approver_role: str
    rationale: str
    required_artifacts: List[str]
    sla_hours: int  # Expected turnaround time
    can_parallel: bool = True  # Can be processed in parallel with others


@dataclass
class ApprovalWorkflowResult:
    """Complete approval workflow definition"""
    service_name: str
    risk_level: RiskLevel
    risk_score: int
    approval_requirements: List[ApprovalRequirement]
    estimated_timeline_days: int
    workflow_steps: List[str]
    escalation_criteria: List[str]
    automated_checks: List[str]


class ApprovalWorkflow:
    """Manages approval workflows based on risk levels"""

    def __init__(self):
        self.approval_matrix = self._initialize_approval_matrix()

    def _initialize_approval_matrix(self) -> Dict[RiskLevel, List[ApprovalRequirement]]:
        """Define approval requirements for each risk level"""
        matrix = {}

        # LOW RISK (0-39): Minimal approvals
        matrix[RiskLevel.LOW] = [
            ApprovalRequirement(
                approval_type=ApprovalType.TECHNICAL,
                approval_level=ApprovalLevel.TEAM_LEAD,
                approver_role="Engineering Manager or Tech Lead",
                rationale="Standard technical review for low-risk changes",
                required_artifacts=[
                    "Service configuration details",
                    "Cost estimate",
                    "Basic security checklist"
                ],
                sla_hours=24,
                can_parallel=True
            )
        ]

        # MEDIUM RISK (40-59): Team + Security review
        matrix[RiskLevel.MEDIUM] = [
            ApprovalRequirement(
                approval_type=ApprovalType.TECHNICAL,
                approval_level=ApprovalLevel.TEAM_LEAD,
                approver_role="Engineering Manager",
                rationale="Technical review for feasibility and resource allocation",
                required_artifacts=[
                    "Technical design document",
                    "Cost analysis",
                    "Integration plan"
                ],
                sla_hours=48,
                can_parallel=True
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.SECURITY,
                approval_level=ApprovalLevel.SECURITY_TEAM,
                approver_role="Security Engineer",
                rationale="Security review for medium-risk service deployment",
                required_artifacts=[
                    "Security controls checklist",
                    "IAM configuration",
                    "Network diagram",
                    "Data flow diagram"
                ],
                sla_hours=72,
                can_parallel=True
            )
        ]

        # HIGH RISK (60-79): Enhanced security + compliance
        matrix[RiskLevel.HIGH] = [
            ApprovalRequirement(
                approval_type=ApprovalType.TECHNICAL,
                approval_level=ApprovalLevel.TEAM_LEAD,
                approver_role="Senior Engineering Manager",
                rationale="Senior technical review for high-risk deployment",
                required_artifacts=[
                    "Comprehensive technical design",
                    "Disaster recovery plan",
                    "Cost/benefit analysis",
                    "Runbook and operational procedures"
                ],
                sla_hours=48,
                can_parallel=True
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.SECURITY,
                approval_level=ApprovalLevel.SECURITY_TEAM,
                approver_role="Senior Security Engineer",
                rationale="Enhanced security review with threat modeling",
                required_artifacts=[
                    "Threat model document",
                    "Security architecture diagram",
                    "Penetration test plan",
                    "Incident response plan",
                    "Complete security controls matrix"
                ],
                sla_hours=120,
                can_parallel=False  # Must complete before compliance
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.COMPLIANCE,
                approval_level=ApprovalLevel.SECURITY_TEAM,
                approver_role="Compliance Officer",
                rationale="Compliance review for regulatory requirements",
                required_artifacts=[
                    "Compliance checklist (PCI/HIPAA/SOX/GDPR as applicable)",
                    "Data classification document",
                    "Privacy impact assessment",
                    "Audit logging configuration"
                ],
                sla_hours=96,
                can_parallel=False  # Depends on security review
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.ARCHITECTURE,
                approval_level=ApprovalLevel.CISO,
                approver_role="Enterprise Architect or CISO delegate",
                rationale="Architecture review for enterprise standards alignment",
                required_artifacts=[
                    "Enterprise architecture alignment document",
                    "Integration points documentation",
                    "Future scalability plan"
                ],
                sla_hours=96,
                can_parallel=True
            )
        ]

        # CRITICAL RISK (80-100): Full governance
        matrix[RiskLevel.CRITICAL] = [
            ApprovalRequirement(
                approval_type=ApprovalType.TECHNICAL,
                approval_level=ApprovalLevel.TEAM_LEAD,
                approver_role="VP of Engineering or CTO",
                rationale="Executive technical approval for critical risk",
                required_artifacts=[
                    "Executive summary",
                    "Business justification",
                    "Complete technical design",
                    "Risk mitigation plan",
                    "Disaster recovery and business continuity plan"
                ],
                sla_hours=72,
                can_parallel=True
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.SECURITY,
                approval_level=ApprovalLevel.CISO,
                approver_role="CISO or Security Director",
                rationale="CISO-level security review for critical risk deployments",
                required_artifacts=[
                    "Comprehensive threat model",
                    "Red team assessment results",
                    "Complete security architecture",
                    "Incident response playbook",
                    "Security controls evidence",
                    "Third-party security assessment (if applicable)"
                ],
                sla_hours=168,  # 1 week
                can_parallel=False
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.COMPLIANCE,
                approval_level=ApprovalLevel.CISO,
                approver_role="Chief Compliance Officer",
                rationale="Executive compliance review for regulatory impact",
                required_artifacts=[
                    "Complete compliance assessment",
                    "Regulatory impact analysis",
                    "Legal review (if needed)",
                    "Data protection impact assessment",
                    "Privacy counsel sign-off (for PII/PHI)"
                ],
                sla_hours=168,
                can_parallel=False
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.ARCHITECTURE,
                approval_level=ApprovalLevel.CISO,
                approver_role="Chief Architect",
                rationale="Enterprise architecture board review",
                required_artifacts=[
                    "Enterprise architecture decision record",
                    "Technical standards compliance",
                    "Integration architecture",
                    "Long-term support plan"
                ],
                sla_hours=120,
                can_parallel=True
            ),
            ApprovalRequirement(
                approval_type=ApprovalType.EXECUTIVE,
                approval_level=ApprovalLevel.SECURITY_BOARD,
                approver_role="Security Review Board or Executive Committee",
                rationale="Board-level approval for highest risk deployments",
                required_artifacts=[
                    "Board presentation deck",
                    "Executive risk summary",
                    "Residual risk acceptance form",
                    "Insurance/liability considerations"
                ],
                sla_hours=240,  # 10 business days
                can_parallel=False  # Final approval
            )
        ]

        return matrix

    def determine_approval_workflow(self, risk_assessment: RiskAssessment) -> ApprovalWorkflowResult:
        """
        Determine complete approval workflow based on risk assessment

        Args:
            risk_assessment: RiskAssessment object from risk engine

        Returns:
            ApprovalWorkflowResult with complete workflow definition
        """
        # Get base approval requirements for risk level
        base_requirements = self.approval_matrix[risk_assessment.risk_level]

        # Additional requirements based on specific risk factors
        additional_requirements = []

        # Check for data sensitivity
        data_sensitivity_scores = [
            score for score in risk_assessment.factor_scores
            if score.factor.value == 'data_sensitivity'
        ]
        if data_sensitivity_scores and data_sensitivity_scores[0].score >= 90:
            additional_requirements.append(ApprovalRequirement(
                approval_type=ApprovalType.COMPLIANCE,
                approval_level=ApprovalLevel.CISO,
                approver_role="Data Protection Officer",
                rationale="High data sensitivity requires DPO review",
                required_artifacts=[
                    "Data protection impact assessment (DPIA)",
                    "Data classification labels",
                    "Data retention policy"
                ],
                sla_hours=96,
                can_parallel=True
            ))

        # Check for compliance impact
        compliance_scores = [
            score for score in risk_assessment.factor_scores
            if score.factor.value == 'compliance_impact'
        ]
        if compliance_scores and compliance_scores[0].score >= 90:
            additional_requirements.append(ApprovalRequirement(
                approval_type=ApprovalType.COMPLIANCE,
                approval_level=ApprovalLevel.CISO,
                approver_role="Regulatory Compliance Team",
                rationale="High compliance impact requires specialized review",
                required_artifacts=[
                    "Framework-specific compliance checklist",
                    "Audit evidence documentation",
                    "Control testing results"
                ],
                sla_hours=120,
                can_parallel=False
            ))

        # Combine all requirements
        all_requirements = base_requirements + additional_requirements

        # Generate workflow steps
        workflow_steps = self._generate_workflow_steps(all_requirements)

        # Calculate estimated timeline
        # Parallel tasks use max SLA, sequential tasks sum SLAs
        estimated_hours = self._calculate_timeline(all_requirements)
        estimated_days = (estimated_hours + 23) // 24  # Round up to days

        # Define escalation criteria
        escalation_criteria = self._generate_escalation_criteria(
            risk_assessment.risk_level,
            risk_assessment.overall_score
        )

        # Define automated pre-checks
        automated_checks = self._generate_automated_checks(risk_assessment)

        return ApprovalWorkflowResult(
            service_name=risk_assessment.service_name,
            risk_level=risk_assessment.risk_level,
            risk_score=risk_assessment.overall_score,
            approval_requirements=all_requirements,
            estimated_timeline_days=estimated_days,
            workflow_steps=workflow_steps,
            escalation_criteria=escalation_criteria,
            automated_checks=automated_checks
        )

    def _generate_workflow_steps(self, requirements: List[ApprovalRequirement]) -> List[str]:
        """Generate ordered workflow steps"""
        steps = [
            "1. Submit service evaluation request",
            "2. Automated security checks (pre-flight validation)"
        ]

        # Group parallel and sequential steps
        parallel_reqs = [r for r in requirements if r.can_parallel]
        sequential_reqs = [r for r in requirements if not r.can_parallel]

        step_num = 3

        # Add parallel steps first
        if parallel_reqs:
            parallel_steps = [
                f"   - {r.approval_type.value.replace('_', ' ').title()} ({r.approver_role})"
                for r in parallel_reqs
            ]
            steps.append(f"{step_num}. Parallel Reviews (can proceed concurrently):")
            steps.extend(parallel_steps)
            step_num += 1

        # Add sequential steps
        for req in sequential_reqs:
            steps.append(
                f"{step_num}. {req.approval_type.value.replace('_', ' ').title()} "
                f"by {req.approver_role} (SLA: {req.sla_hours}h)"
            )
            step_num += 1

        steps.append(f"{step_num}. Final approval and deployment authorization")
        steps.append(f"{step_num + 1}. Post-deployment security validation")

        return steps

    def _calculate_timeline(self, requirements: List[ApprovalRequirement]) -> int:
        """Calculate total estimated hours"""
        parallel_reqs = [r for r in requirements if r.can_parallel]
        sequential_reqs = [r for r in requirements if not r.can_parallel]

        # Parallel tasks: use maximum SLA
        parallel_hours = max([r.sla_hours for r in parallel_reqs]) if parallel_reqs else 0

        # Sequential tasks: sum SLAs
        sequential_hours = sum([r.sla_hours for r in sequential_reqs])

        # Add buffer for coordination (10%)
        total_hours = int((parallel_hours + sequential_hours) * 1.1)

        return total_hours

    def _generate_escalation_criteria(self, risk_level: RiskLevel, risk_score: int) -> List[str]:
        """Generate escalation criteria"""
        criteria = [
            "Escalate if approval SLA exceeded by 50%",
            "Escalate if multiple rejections occur",
            "Escalate if new risks are discovered during review"
        ]

        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            criteria.extend([
                "Escalate immediately if security vulnerabilities found",
                "Escalate if compliance gaps identified",
                "Escalate if business impact changes significantly"
            ])

        if risk_score >= 90:
            criteria.append("Automatic escalation to CISO for risk score >= 90")

        return criteria

    def _generate_automated_checks(self, risk_assessment: RiskAssessment) -> List[str]:
        """Generate list of automated pre-flight checks"""
        checks = [
            "Service API availability and quota limits",
            "IAM permissions validation",
            "Network configuration compliance",
            "Encryption at rest enabled",
            "Audit logging configured",
            "Required org policies in place",
            "Cost estimates within budget",
            "No known CVEs in service dependencies"
        ]

        # Add risk-specific checks
        for factor_score in risk_assessment.factor_scores:
            if factor_score.score >= 70:
                if factor_score.factor.value == 'data_sensitivity':
                    checks.extend([
                        "DLP policies configured",
                        "Data classification labels applied",
                        "CMEK encryption verified"
                    ])
                elif factor_score.factor.value == 'network_exposure':
                    checks.extend([
                        "No public IP addresses",
                        "VPC Service Controls enabled",
                        "Firewall rules reviewed"
                    ])

        return list(set(checks))  # Deduplicate

    def get_approval_summary(self, risk_level: RiskLevel) -> Dict[str, Any]:
        """Get summary of approval requirements for a risk level"""
        requirements = self.approval_matrix[risk_level]

        return {
            'risk_level': risk_level.value,
            'total_approvals_required': len(requirements),
            'approval_types': [r.approval_type.value for r in requirements],
            'approval_levels': list(set([r.approval_level.value for r in requirements])),
            'estimated_min_days': self._calculate_timeline(requirements) // 24,
            'parallel_reviews': sum(1 for r in requirements if r.can_parallel),
            'sequential_reviews': sum(1 for r in requirements if not r.can_parallel)
        }
