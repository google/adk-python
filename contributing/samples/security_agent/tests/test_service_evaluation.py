#!/usr/bin/env python3
"""
Test Suite for Service Evaluation Framework
Tests security controls, enforcement, risk assessment, and approval workflows
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents._tools.service_evaluation import (
    evaluate_new_service,
    SecurityControlsInventory,
    EnforcementAnalyzer,
    RiskAssessmentEngine,
    ApprovalWorkflow
)
from agents._tools.service_evaluation.risk import ServiceProfile, RiskLevel
from agents._tools.service_evaluation.controls import ControlCategory


def test_security_controls_inventory():
    """Test Security Controls Inventory"""
    print("\n" + "="*80)
    print("TEST 1: Security Controls Inventory")
    print("="*80)

    inventory = SecurityControlsInventory()

    # Get summary
    summary = inventory.get_control_summary()
    print(f"\n📊 Controls Summary:")
    print(f"   Total Controls: {summary['total_controls']}")
    print(f"\n   By Category:")
    for cat, count in summary['by_category'].items():
        print(f"      {cat}: {count}")
    print(f"\n   By Severity:")
    for sev, count in summary['by_severity'].items():
        print(f"      {sev}: {count}")
    print(f"\n   Compliance Coverage:")
    for framework, count in summary['coverage'].items():
        print(f"      {framework.upper()}: {count} controls")

    # Test getting controls for specific service
    print(f"\n🔍 Controls for 'storage' services:")
    storage_controls = inventory.get_controls_for_service('storage')
    for control in storage_controls[:5]:  # Show first 5
        print(f"   [{control.id}] {control.name} ({control.severity})")
        print(f"      {control.description[:100]}...")

    assert summary['total_controls'] > 15, "Should have at least 15 controls"
    assert len(storage_controls) >= 3, "Storage should have multiple applicable controls"
    print("\n✅ Security Controls Inventory: PASSED")


def test_enforcement_analyzer():
    """Test Enforcement Analyzer"""
    print("\n" + "="*80)
    print("TEST 2: Enforcement Analyzer")
    print("="*80)

    analyzer = EnforcementAnalyzer()

    # Test getting enforcement options for a control
    print(f"\n🔧 Enforcement Options for IAM-001 (Least Privilege):")
    options = analyzer.get_enforcement_options('IAM-001')

    for i, option in enumerate(options[:3], 1):  # Show first 3
        print(f"\n   Option {i}: {option.name}")
        print(f"   Method: {option.method.value}")
        print(f"   Automation Level: {option.automation_level}")
        print(f"   Complexity: {option.complexity}")
        print(f"   Cost: {option.cost_estimate}")

        if option.implementation_template:
            print(f"   Template Preview:")
            template_lines = option.implementation_template.strip().split('\n')
            for line in template_lines[:3]:
                print(f"      {line}")

    # Count automated options
    automated_count = sum(1 for opt in options if opt.automation_level == 'full')
    print(f"\n📊 Enforcement Summary:")
    print(f"   Total Options shown: {len(options)}")
    print(f"   Fully Automated: {automated_count}")

    assert len(options) > 0, "Should have enforcement options for IAM-001"
    print("\n✅ Enforcement Analyzer: PASSED")


def test_risk_assessment_engine():
    """Test Risk Assessment Engine"""
    print("\n" + "="*80)
    print("TEST 3: Risk Assessment Engine")
    print("="*80)

    engine = RiskAssessmentEngine()

    # Test LOW risk service
    print(f"\n🟢 LOW RISK Service (Internal development tool):")
    low_risk_profile = ServiceProfile(
        service_name="Cloud Functions",
        service_type="cloud_functions",
        use_case="Internal development tools for CI/CD",
        data_classification="internal",
        network_exposure="internal",
        authentication_method="iam",
        encryption_at_rest=True,
        encryption_in_transit=True,
        compliance_requirements=[],
        third_party_integrations=[],
        expected_data_volume="low"
    )

    low_risk_assessment = engine.assess_service_risk(low_risk_profile)
    print(f"   Risk Level: {low_risk_assessment.risk_level.value}")
    print(f"   Risk Score: {low_risk_assessment.overall_score}/100")
    print(f"   Summary: {low_risk_assessment.risk_summary}")

    # Test HIGH/CRITICAL risk service
    print(f"\n🔴 HIGH/CRITICAL RISK Service (PHI data processing):")
    high_risk_profile = ServiceProfile(
        service_name="BigQuery",
        service_type="bigquery",
        use_case="Processing and storing patient health information (PHI)",
        data_classification="restricted",
        network_exposure="public",
        authentication_method="iam",
        encryption_at_rest=True,
        encryption_in_transit=True,
        compliance_requirements=["HIPAA", "GDPR", "SOX"],
        third_party_integrations=["external_analytics", "third_party_bi"],
        expected_data_volume="high"
    )

    high_risk_assessment = engine.assess_service_risk(high_risk_profile)
    print(f"   Risk Level: {high_risk_assessment.risk_level.value}")
    print(f"   Risk Score: {high_risk_assessment.overall_score}/100")
    print(f"   Summary: {high_risk_assessment.risk_summary}")

    print(f"\n   Top Risk Factors:")
    for factor_score in high_risk_assessment.factor_scores[:3]:
        print(f"      {factor_score.factor.value}: {factor_score.score}/100 (weight: {factor_score.weight})")
        print(f"         {factor_score.rationale[:100]}...")

    assert low_risk_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM], "Should be low/medium risk"
    assert high_risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL], "Should be high/critical risk"
    assert high_risk_assessment.overall_score > low_risk_assessment.overall_score, "High risk should score higher"
    print("\n✅ Risk Assessment Engine: PASSED")


def test_approval_workflow():
    """Test Approval Workflow"""
    print("\n" + "="*80)
    print("TEST 4: Approval Workflow")
    print("="*80)

    workflow = ApprovalWorkflow()
    engine = RiskAssessmentEngine()

    # Create a CRITICAL risk assessment
    critical_profile = ServiceProfile(
        service_name="Cloud SQL",
        service_type="cloudsql",
        use_case="Mission-critical customer database with PII and payment data",
        data_classification="restricted",
        network_exposure="public",
        authentication_method="iam",
        encryption_at_rest=True,
        encryption_in_transit=True,
        compliance_requirements=["PCI-DSS", "GDPR", "SOX", "HIPAA"],
        third_party_integrations=["payment_processor", "crm_system"],
        expected_data_volume="high"
    )

    risk_assessment = engine.assess_service_risk(critical_profile)
    approval_result = workflow.determine_approval_workflow(risk_assessment)

    print(f"\n📋 Approval Workflow for {approval_result.service_name}:")
    print(f"   Risk Level: {approval_result.risk_level.value}")
    print(f"   Risk Score: {approval_result.risk_score}/100")
    print(f"   Approvals Required: {len(approval_result.approval_requirements)}")
    print(f"   Estimated Timeline: {approval_result.estimated_timeline_days} days")

    print(f"\n   Approval Requirements:")
    for req in approval_result.approval_requirements:
        print(f"\n      {req.approval_type.value.upper()}")
        print(f"         Level: {req.approval_level.value}")
        print(f"         Approver: {req.approver_role}")
        print(f"         SLA: {req.sla_hours}h")
        print(f"         Artifacts: {len(req.required_artifacts)}")
        print(f"         Parallel: {req.can_parallel}")

    print(f"\n   Workflow Steps:")
    for step in approval_result.workflow_steps[:5]:
        print(f"      {step}")

    print(f"\n   Automated Checks ({len(approval_result.automated_checks)}):")
    for check in approval_result.automated_checks[:5]:
        print(f"      ✓ {check}")

    assert len(approval_result.approval_requirements) >= 3, "Critical risk should require multiple approvals"
    assert approval_result.estimated_timeline_days > 0, "Should have estimated timeline"
    print("\n✅ Approval Workflow: PASSED")


def test_complete_evaluation():
    """Test Complete Service Evaluation"""
    print("\n" + "="*80)
    print("TEST 5: Complete Service Evaluation (End-to-End)")
    print("="*80)

    # Test basic evaluation (without detailed profile)
    print(f"\n📝 Basic Evaluation: Cloud Storage for document storage")
    basic_result = evaluate_new_service(
        service_name="Cloud Storage",
        service_type="storage",
        use_case="Internal document storage for team collaboration",
        data_classification="internal",
        return_format='summary'
    )

    print(basic_result)

    # Test detailed evaluation with profile
    print(f"\n📝 Detailed Evaluation: BigQuery for data warehouse")
    detailed_profile = ServiceProfile(
        service_name="BigQuery",
        service_type="bigquery",
        use_case="Enterprise data warehouse with customer analytics",
        data_classification="confidential",
        network_exposure="internal",
        authentication_method="iam",
        encryption_at_rest=True,
        encryption_in_transit=True,
        compliance_requirements=["GDPR", "SOX"],
        third_party_integrations=["tableau", "looker"],
        expected_data_volume="high"
    )

    detailed_result = evaluate_new_service(
        service_name="BigQuery",
        service_type="bigquery",
        service_profile=detailed_profile,
        return_format='object'
    )

    print(f"\n   Service: {detailed_result.service_name}")
    print(f"   Risk Level: {detailed_result.summary['risk_level']}")
    print(f"   Risk Score: {detailed_result.summary['risk_score']}/100")
    print(f"   Total Controls: {detailed_result.summary['total_controls']}")
    print(f"   Critical Controls: {detailed_result.summary['critical_controls']}")
    print(f"   Enforcement Options: {detailed_result.summary['total_enforcement_options']}")
    print(f"   Automated Options: {detailed_result.summary['automated_enforcement_available']}")
    print(f"   Approvals Required: {detailed_result.summary['approvals_required']}")
    print(f"   Timeline: {detailed_result.summary['estimated_timeline_days']} days")

    print(f"\n   Key Recommendations:")
    for rec in detailed_result.recommendations[:3]:
        print(f"      {rec}")

    print(f"\n   Next Steps:")
    for step in detailed_result.next_steps[:3]:
        print(f"      {step}")

    # Test dict return format
    dict_result = evaluate_new_service(
        service_name="Cloud Run",
        service_type="cloud_run",
        return_format='dict'
    )

    assert isinstance(basic_result, str), "Summary should be string"
    assert detailed_result.service_name == "BigQuery", "Should match service name"
    assert isinstance(dict_result, dict), "Dict format should return dict"
    assert dict_result['service_name'] == "Cloud Run", "Dict should have service_name"
    print("\n✅ Complete Service Evaluation: PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 SERVICE EVALUATION FRAMEWORK - TEST SUITE")
    print("="*80)

    try:
        test_security_controls_inventory()
        test_enforcement_analyzer()
        test_risk_assessment_engine()
        test_approval_workflow()
        test_complete_evaluation()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nService Evaluation Framework is ready for deployment.")
        print("Agent now has 24 tools including comprehensive service evaluation.")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
