#!/usr/bin/env python3
"""
Service Evaluation Orchestrator
Main entry point for comprehensive GCP service security evaluation
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .controls import SecurityControlsInventory, SecurityControl
from .enforcement import EnforcementAnalyzer, EnforcementOption
from .risk import RiskAssessmentEngine, RiskAssessment, RiskLevel, ServiceProfile
from .approval import ApprovalWorkflow, ApprovalWorkflowResult


@dataclass
class ServiceEvaluationResult:
    """Complete service evaluation report"""
    service_name: str
    service_type: str
    evaluation_timestamp: str

    # Risk Assessment
    risk_assessment: Dict[str, Any]

    # Security Controls
    applicable_controls: List[Dict[str, Any]]
    controls_by_category: Dict[str, List[Dict[str, Any]]]
    controls_by_severity: Dict[str, int]

    # Enforcement Options
    enforcement_options: List[Dict[str, Any]]
    enforcement_by_method: Dict[str, List[Dict[str, Any]]]

    # Approval Workflow
    approval_workflow: Dict[str, Any]

    # Summary Metrics
    summary: Dict[str, Any]

    # Recommendations
    recommendations: List[str]
    next_steps: List[str]


class ServiceEvaluator:
    """Main orchestrator for service evaluation"""

    def __init__(self):
        self.controls_inventory = SecurityControlsInventory()
        self.enforcement_analyzer = EnforcementAnalyzer()
        self.risk_engine = RiskAssessmentEngine()
        self.approval_workflow = ApprovalWorkflow()

    def evaluate_service(
        self,
        service_name: str,
        service_type: str,
        service_profile: Optional[ServiceProfile] = None,
        use_case: Optional[str] = None,
        data_classification: Optional[str] = None
    ) -> ServiceEvaluationResult:
        """
        Perform comprehensive service evaluation

        Args:
            service_name: Name of the GCP service (e.g., "Cloud Storage", "BigQuery")
            service_type: Service type category (e.g., "storage", "compute", "database")
            service_profile: Optional detailed service profile for risk assessment
            use_case: Optional description of how the service will be used
            data_classification: Optional data classification (public, internal, confidential, restricted)

        Returns:
            ServiceEvaluationResult with complete evaluation
        """

        # Step 1: Risk Assessment
        if service_profile:
            risk_assessment = self.risk_engine.assess_service_risk(service_profile)
        else:
            # Create basic profile from provided info
            basic_profile = ServiceProfile(
                service_name=service_name,
                service_type=service_type,
                use_case=use_case or "General purpose usage",
                data_classification=data_classification or "internal",
                network_exposure="internal",  # Conservative default
                authentication_method="iam",
                encryption_at_rest=True,
                encryption_in_transit=True,
                compliance_requirements=[],
                third_party_integrations=[],
                expected_data_volume="medium"
            )
            risk_assessment = self.risk_engine.assess_service_risk(basic_profile)

        # Step 2: Get Applicable Security Controls
        applicable_controls = self.controls_inventory.get_controls_for_service(service_type)

        # Group controls by category and severity
        controls_by_category = {}
        controls_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for control in applicable_controls:
            # By category
            cat = control.category.value
            if cat not in controls_by_category:
                controls_by_category[cat] = []
            controls_by_category[cat].append(self._serialize_control(control))

            # By severity
            controls_by_severity[control.severity] += 1

        # Step 3: Get Enforcement Options
        enforcement_options = []
        enforcement_by_method = {}

        for control in applicable_controls:
            options = self.enforcement_analyzer.get_enforcement_options(control.id)
            for option in options:
                serialized_option = self._serialize_enforcement_option(option)
                enforcement_options.append(serialized_option)

                method = option.method.value
                if method not in enforcement_by_method:
                    enforcement_by_method[method] = []
                enforcement_by_method[method].append(serialized_option)

        # Step 4: Determine Approval Workflow
        approval_result = self.approval_workflow.determine_approval_workflow(risk_assessment)

        # Step 5: Generate Recommendations
        recommendations = self._generate_recommendations(
            risk_assessment,
            applicable_controls,
            enforcement_options
        )

        # Step 6: Generate Next Steps
        next_steps = self._generate_next_steps(
            risk_assessment,
            approval_result
        )

        # Step 7: Create Summary
        summary = {
            'risk_level': risk_assessment.risk_level.value,
            'risk_score': risk_assessment.overall_score,
            'total_controls': len(applicable_controls),
            'critical_controls': controls_by_severity['critical'],
            'high_controls': controls_by_severity['high'],
            'total_enforcement_options': len(enforcement_options),
            'automated_enforcement_available': len([
                e for e in enforcement_options
                if e.get('automation_level') == 'full'
            ]),
            'approvals_required': len(approval_result.approval_requirements),
            'estimated_timeline_days': approval_result.estimated_timeline_days,
            'evaluation_timestamp': datetime.utcnow().isoformat()
        }

        # Create final result
        return ServiceEvaluationResult(
            service_name=service_name,
            service_type=service_type,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            risk_assessment=self._serialize_risk_assessment(risk_assessment),
            applicable_controls=[self._serialize_control(c) for c in applicable_controls],
            controls_by_category=controls_by_category,
            controls_by_severity=controls_by_severity,
            enforcement_options=enforcement_options,
            enforcement_by_method=enforcement_by_method,
            approval_workflow=self._serialize_approval_workflow(approval_result),
            summary=summary,
            recommendations=recommendations,
            next_steps=next_steps
        )

    def _serialize_control(self, control: SecurityControl) -> Dict[str, Any]:
        """Convert SecurityControl to dict"""
        return {
            'id': control.id,
            'name': control.name,
            'category': control.category.value,
            'control_type': control.control_type.value,
            'description': control.description,
            'severity': control.severity,
            'compliance_mappings': {
                'cis_benchmark': control.cis_benchmark,
                'nist_framework': control.nist_framework,
                'pci_dss': control.pci_dss,
                'hipaa': control.hipaa,
                'sox': control.sox,
                'gdpr': control.gdpr
            },
            'implementation_guidance': control.implementation_guidance,
            'validation_query': control.validation_query
        }

    def _serialize_enforcement_option(self, option: EnforcementOption) -> Dict[str, Any]:
        """Convert EnforcementOption to dict"""
        return {
            'method': option.method.value,
            'name': option.name,
            'description': option.description,
            'complexity': option.complexity,
            'automation_level': option.automation_level,
            'implementation_template': option.implementation_template,
            'cost_estimate': option.cost_estimate,
            'maintenance_effort': option.maintenance_effort,
            'gcp_documentation': option.gcp_documentation
        }

    def _serialize_risk_assessment(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Convert RiskAssessment to dict"""
        return {
            'service_name': assessment.service_name,
            'overall_score': assessment.overall_score,
            'risk_level': assessment.risk_level.value,
            'factor_scores': [
                {
                    'factor': fs.factor.value,
                    'score': fs.score,
                    'weight': fs.weight,
                    'weighted_score': fs.weighted_score,
                    'rationale': fs.rationale
                }
                for fs in assessment.factor_scores
            ],
            'risk_summary': assessment.risk_summary,
            'mitigation_priorities': assessment.mitigation_priorities,
            'compliance_considerations': assessment.compliance_considerations
        }

    def _serialize_approval_workflow(self, workflow: ApprovalWorkflowResult) -> Dict[str, Any]:
        """Convert ApprovalWorkflowResult to dict"""
        return {
            'service_name': workflow.service_name,
            'risk_level': workflow.risk_level.value,
            'risk_score': workflow.risk_score,
            'approval_requirements': [
                {
                    'approval_type': req.approval_type.value,
                    'approval_level': req.approval_level.value,
                    'approver_role': req.approver_role,
                    'rationale': req.rationale,
                    'required_artifacts': req.required_artifacts,
                    'sla_hours': req.sla_hours,
                    'can_parallel': req.can_parallel
                }
                for req in workflow.approval_requirements
            ],
            'estimated_timeline_days': workflow.estimated_timeline_days,
            'workflow_steps': workflow.workflow_steps,
            'escalation_criteria': workflow.escalation_criteria,
            'automated_checks': workflow.automated_checks
        }

    def _generate_recommendations(
        self,
        risk_assessment: RiskAssessment,
        controls: List[SecurityControl],
        enforcement_options: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Risk-based recommendations
        if risk_assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append(
                f"⚠️ HIGH RISK SERVICE: This service has a risk score of {risk_assessment.overall_score}/100. "
                "Prioritize security controls implementation before deployment."
            )
            recommendations.append(
                "Engage security team early in the planning phase for threat modeling and architecture review."
            )

        # Control recommendations
        critical_controls = [c for c in controls if c.severity == 'critical']
        if critical_controls:
            recommendations.append(
                f"🔒 Implement {len(critical_controls)} critical security controls before deployment: "
                f"{', '.join([c.id for c in critical_controls[:3]])}{'...' if len(critical_controls) > 3 else ''}"
            )

        # Automation recommendations
        automated_options = [
            e for e in enforcement_options
            if e.get('automation_level') == 'full'
        ]
        if automated_options:
            recommendations.append(
                f"✅ {len(automated_options)} controls can be fully automated using Organization Policies and Terraform. "
                "Prioritize automation to reduce manual effort."
            )

        # Compliance recommendations
        if risk_assessment.compliance_considerations:
            recommendations.append(
                f"📋 Compliance considerations identified: {', '.join(risk_assessment.compliance_considerations[:3])}. "
                "Review compliance requirements before deployment."
            )

        # Data sensitivity recommendations
        data_sensitivity_scores = [
            fs for fs in risk_assessment.factor_scores
            if fs.factor.value == 'data_sensitivity'
        ]
        if data_sensitivity_scores and data_sensitivity_scores[0].score >= 70:
            recommendations.append(
                "🔐 High data sensitivity detected. Implement CMEK encryption, DLP scanning, and strict access controls."
            )

        return recommendations

    def _generate_next_steps(
        self,
        risk_assessment: RiskAssessment,
        approval_workflow: ApprovalWorkflowResult
    ) -> List[str]:
        """Generate ordered next steps"""
        steps = []

        # Step 1: Review evaluation
        steps.append(
            "1. Review this complete evaluation report with your team and security stakeholders"
        )

        # Step 2: Prioritize controls
        steps.append(
            "2. Prioritize implementation of critical and high-severity security controls"
        )

        # Step 3: Prepare artifacts
        if approval_workflow.approval_requirements:
            all_artifacts = []
            for req in approval_workflow.approval_requirements:
                all_artifacts.extend(req.required_artifacts)
            unique_artifacts = list(set(all_artifacts))

            steps.append(
                f"3. Prepare required approval artifacts ({len(unique_artifacts)} total): "
                f"{', '.join(unique_artifacts[:3])}{'...' if len(unique_artifacts) > 3 else ''}"
            )

        # Step 4: Automated checks
        steps.append(
            f"4. Run automated pre-flight checks ({len(approval_workflow.automated_checks)} checks available)"
        )

        # Step 5: Submit for approval
        if approval_workflow.approval_requirements:
            steps.append(
                f"5. Submit for approval (estimated timeline: {approval_workflow.estimated_timeline_days} days)"
            )
        else:
            steps.append(
                "5. Proceed with deployment following standard change management process"
            )

        # Step 6: Implement controls
        steps.append(
            "6. Implement security controls using recommended enforcement methods (Org Policies, Terraform, etc.)"
        )

        # Step 7: Validate
        steps.append(
            "7. Validate security controls are in place before production deployment"
        )

        # Step 8: Monitor
        steps.append(
            "8. Set up continuous monitoring and compliance validation (Security Command Center, Cloud Monitoring)"
        )

        return steps

    def get_quick_summary(self, evaluation_result: ServiceEvaluationResult) -> str:
        """Generate a quick text summary of evaluation results"""
        summary = f"""
═══════════════════════════════════════════════════════════════
  GCP SERVICE SECURITY EVALUATION REPORT
═══════════════════════════════════════════════════════════════

SERVICE: {evaluation_result.service_name} ({evaluation_result.service_type})
EVALUATED: {evaluation_result.evaluation_timestamp}

RISK ASSESSMENT
───────────────────────────────────────────────────────────────
Risk Level:  {evaluation_result.summary['risk_level'].upper()}
Risk Score:  {evaluation_result.summary['risk_score']}/100

SECURITY CONTROLS
───────────────────────────────────────────────────────────────
Total Controls:     {evaluation_result.summary['total_controls']}
  - Critical:       {evaluation_result.summary['critical_controls']}
  - High:           {evaluation_result.summary['high_controls']}

ENFORCEMENT
───────────────────────────────────────────────────────────────
Total Options:      {evaluation_result.summary['total_enforcement_options']}
Fully Automated:    {evaluation_result.summary['automated_enforcement_available']}

APPROVAL WORKFLOW
───────────────────────────────────────────────────────────────
Approvals Required: {evaluation_result.summary['approvals_required']}
Timeline:           {evaluation_result.summary['estimated_timeline_days']} days

KEY RECOMMENDATIONS
───────────────────────────────────────────────────────────────
"""
        for i, rec in enumerate(evaluation_result.recommendations[:5], 1):
            summary += f"{rec}\n\n"

        summary += """
NEXT STEPS
───────────────────────────────────────────────────────────────
"""
        for step in evaluation_result.next_steps[:5]:
            summary += f"{step}\n"

        summary += "\n═══════════════════════════════════════════════════════════════\n"

        return summary


# Convenience function for direct use
def evaluate_new_service(
    service_name: str,
    service_type: str,
    service_profile: Optional[ServiceProfile] = None,
    use_case: Optional[str] = None,
    data_classification: Optional[str] = None,
    return_format: str = 'object'
) -> Any:
    """
    Evaluate a new GCP service for security, compliance, and risk

    Args:
        service_name: Name of the GCP service
        service_type: Service type (storage, compute, database, etc.)
        service_profile: Optional detailed service profile
        use_case: Optional use case description
        data_classification: Optional data classification
        return_format: 'object', 'dict', or 'summary'

    Returns:
        ServiceEvaluationResult (object), dict, or text summary based on return_format
    """
    evaluator = ServiceEvaluator()
    result = evaluator.evaluate_service(
        service_name=service_name,
        service_type=service_type,
        service_profile=service_profile,
        use_case=use_case,
        data_classification=data_classification
    )

    if return_format == 'dict':
        return asdict(result)
    elif return_format == 'summary':
        return evaluator.get_quick_summary(result)
    else:
        return result
