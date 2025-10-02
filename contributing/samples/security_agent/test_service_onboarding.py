#!/usr/bin/env python3
"""
Service Onboarding Demo - URL-based Service Discovery
Shows how users can onboard any GCP service by pasting documentation URLs
"""

from agents._tools.service_onboarding import ServiceOnboardingTool
import json

def demo_service_onboarding():
    """Demonstrate the service onboarding tool with various GCP services"""

    print("=" * 80)
    print("   🚀 SERVICE ONBOARDING TOOL - URL-BASED DISCOVERY")
    print("=" * 80)
    print()
    print("This tool allows freehand service input via documentation URLs.")
    print("Just paste any GCP service documentation link to get started!")
    print()

    # Test URLs covering different service categories
    test_cases = [
        {
            'url': "https://cloud.google.com/vertex-ai/docs",
            'description': "AI/ML Service"
        },
        {
            'url': "https://cloud.google.com/secret-manager/docs",
            'description': "Security Service"
        },
        {
            'url': "https://cloud.google.com/cloud-sql/docs",
            'description': "Database Service"
        },
        {
            'url': "https://cloud.google.com/run/docs",
            'description': "Compute Service"
        },
        {
            'url': "https://cloud.google.com/vpc/docs",
            'description': "Networking Service"
        }
    ]

    tool = ServiceOnboardingTool()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f" TEST {i}: {test_case['description']}")
        print(f" URL: {test_case['url']}")
        print("="*80)

        result = tool.onboard_service_from_url(test_case['url'])

        if result['success']:
            # Service Information
            print(f"\n✅ SERVICE IDENTIFIED")
            print(f"   Name: {result['service_info']['service_name']}")
            print(f"   Category: {result['service_info']['category']}")
            print(f"   ID: {result['service_info']['service_id']}")

            # Risk Assessment
            risk = result['risk_assessment']
            risk_emoji = "🟢" if risk['risk_level'] == 'low' else "🟡" if risk['risk_level'] == 'medium' else "🔴"
            print(f"\n{risk_emoji} RISK ASSESSMENT")
            print(f"   Risk Level: {risk['risk_level'].upper()}")
            print(f"   Risk Score: {risk['risk_score']}/100")
            if risk['factors']:
                print(f"   Factors:")
                for factor in risk['factors']:
                    print(f"     • {factor}")

            # Compliance Status
            compliance = result['compliance']
            comp_emoji = "✅" if compliance['overall_status'] == 'approved' else "⚠️" if compliance['overall_status'] == 'review_required' else "❌"
            print(f"\n{comp_emoji} COMPLIANCE CHECK")
            print(f"   Status: {compliance['overall_status'].upper()}")
            if compliance['checks_passed']:
                print(f"   ✅ Passed: {len(compliance['checks_passed'])} checks")
            if compliance['checks_failed']:
                print(f"   ❌ Failed: {len(compliance['checks_failed'])} checks")
                for check in compliance['checks_failed']:
                    print(f"      • {check}")

            # IAM Recommendations
            iam = result['iam_recommendations']
            print(f"\n🔐 IAM RECOMMENDATIONS (LEAST PRIVILEGE)")
            print(f"   Recommended Roles:")
            for role in iam['recommended_roles'][:3]:
                print(f"     ✅ {role}")
            print(f"   Avoid These Roles:")
            for role in iam['avoid_roles'][:2]:
                print(f"     ❌ {role}")

            if iam['custom_role_needed']:
                print(f"   💡 Custom role recommended to combine permissions")

            # Security Recommendations
            security = result['security_recommendations']
            if security['required_controls']:
                print(f"\n🛡️ REQUIRED SECURITY CONTROLS")
                for control in security['required_controls'][:2]:
                    print(f"   • {control['control']}")
                    print(f"     Reason: {control['reason']}")

            # Similar Services (for context)
            if result['similar_services']:
                print(f"\n📊 SIMILAR APPROVED SERVICES")
                for similar in result['similar_services'][:2]:
                    print(f"   • {similar['service_name']}")
                    if similar.get('security_score'):
                        print(f"     Security Score: {similar['security_score']}/100")

            # Next Steps
            if result['next_steps']:
                print(f"\n📋 NEXT STEPS")
                for step in result['next_steps']:
                    priority_emoji = "🔴" if step['priority'] == 'high' else "🟡" if step['priority'] == 'medium' else "🟢"
                    print(f"   {priority_emoji} [{step['priority'].upper()}] {step['action']}")

            # Approval Workflow
            workflow = result['approval_workflow']
            print(f"\n🔄 APPROVAL WORKFLOW")
            if workflow['auto_approved']:
                print(f"   ✅ AUTO-APPROVED (Low Risk)")
            else:
                print(f"   Status: {workflow['status']}")
                if workflow['required_approvals']:
                    print(f"   Required Approvals:")
                    for approval in workflow['required_approvals']:
                        print(f"     • {approval}")

        else:
            print(f"❌ Error: {result.get('error', 'Could not analyze service')}")

    print("\n" + "="*80)
    print("💡 KEY FEATURES DEMONSTRATED")
    print("="*80)
    print("""
    1. FREEHAND INPUT: Just paste any GCP documentation URL
    2. AUTOMATIC EXTRACTION: Service name and category identified from URL
    3. RISK ASSESSMENT: Calculates security risk based on documentation
    4. COMPLIANCE CHECKS: Validates against enterprise standards
    5. LEAST PRIVILEGE: Never recommends admin/owner/editor roles
    6. SIMILAR SERVICES: Learns from previously approved services
    7. GUIDED REMEDIATION: Provides specific steps to achieve compliance
    8. APPROVAL WORKFLOW: Routes based on risk level
    """)

    print("\n" + "="*80)
    print("🎯 BUSINESS VALUE")
    print("="*80)
    print("""
    • FASTER ONBOARDING: From weeks to minutes
    • NO GENERIC ADMIN ROLES: Enforces least privilege automatically
    • CONTEXT-AWARE: Learns from similar approved services
    • COMPLIANCE BUILT-IN: Pre-flight checks before deployment
    • AUDIT TRAIL: All analyses stored in BigQuery
    """)


if __name__ == "__main__":
    demo_service_onboarding()