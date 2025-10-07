#!/usr/bin/env python3
"""
Test script for IAM Custom Role Analyzer
Demonstrates the analysis capabilities without full swarm
"""

import json
from agents._tools.iam_custom_role_analyzer import CustomRoleAnalyzer

def demo_custom_role_analysis():
    """Demo the custom role analysis functionality"""

    print("=" * 60)
    print("   IAM CUSTOM ROLE ANALYZER - DEMO")
    print("=" * 60)
    print()

    # Create a mock custom role for demonstration
    mock_custom_role = {
        "name": "custom-data-analyst",
        "title": "Custom Data Analyst",
        "description": "Custom role for organizational data analysts",
        "permissions": [
            "bigquery.datasets.get",
            "bigquery.datasets.getIamPolicy",
            "bigquery.tables.get",
            "bigquery.tables.getData",
            "bigquery.tables.list",
            "bigquery.jobs.create",
            "storage.objects.get",
            "storage.objects.list",
            "storage.buckets.get",
            "pubsub.topics.publish",  # Extra permission
            "compute.instances.list"  # Extra permission
        ]
    }

    print("📋 Analyzing Custom Role: custom-data-analyst")
    print(f"   Permissions: {len(mock_custom_role['permissions'])}")
    print()

    # Simulate analysis results
    analysis_results = {
        "custom_role": mock_custom_role,
        "best_matches": [
            {
                "role": "roles/bigquery.dataViewer",
                "title": "BigQuery Data Viewer",
                "similarity_score": 72.5,
                "coverage_score": 85.0,
                "overlap_count": 7,
                "extra_permissions": [
                    "pubsub.topics.publish",
                    "compute.instances.list"
                ],
                "missing_permissions": [
                    "bigquery.models.list",
                    "bigquery.routines.list"
                ]
            },
            {
                "role": "roles/bigquery.user",
                "title": "BigQuery User",
                "similarity_score": 65.0,
                "coverage_score": 75.0,
                "overlap_count": 6,
                "extra_permissions": [
                    "storage.objects.get",
                    "storage.objects.list",
                    "pubsub.topics.publish",
                    "compute.instances.list"
                ],
                "missing_permissions": [
                    "bigquery.jobs.get",
                    "bigquery.jobs.list"
                ]
            },
            {
                "role": "roles/viewer",
                "title": "Viewer",
                "similarity_score": 45.0,
                "coverage_score": 50.0,
                "overlap_count": 5,
                "extra_permissions": [
                    "bigquery.jobs.create",
                    "pubsub.topics.publish"
                ],
                "missing_permissions": [
                    "resourcemanager.projects.get",
                    "resourcemanager.projects.list"
                ]
            }
        ],
        "recommendations": {
            "summary": "Consider using built-in role with modifications",
            "actions": [
                "Use roles/bigquery.dataViewer as base",
                "Add storage.objects permissions if needed",
                "Review if pubsub.topics.publish is necessary",
                "Remove compute.instances.list unless required"
            ],
            "risk_level": "medium"
        },
        "security_assessment": {
            "risk_score": 35,
            "risk_level": "medium",
            "findings": [
                "pubsub.topics.publish: Can publish messages to Pub/Sub topics",
                "Extra permissions beyond standard BigQuery viewer role"
            ],
            "dangerous_permissions": []
        }
    }

    # Display results
    print("🔍 ANALYSIS RESULTS")
    print("=" * 60)
    print()

    print("📊 Best Matching Built-in Roles:")
    for i, match in enumerate(analysis_results["best_matches"][:3], 1):
        print(f"\n{i}. {match['title']} ({match['role']})")
        print(f"   Similarity: {match['similarity_score']}%")
        print(f"   Coverage: {match['coverage_score']}%")
        print(f"   Overlapping: {match['overlap_count']} permissions")

        if match['extra_permissions']:
            print(f"   Extra in custom: {len(match['extra_permissions'])}")
            for perm in match['extra_permissions'][:2]:
                print(f"     • {perm}")

        if match['missing_permissions']:
            print(f"   Missing from custom: {len(match['missing_permissions'])}")
            for perm in match['missing_permissions'][:2]:
                print(f"     • {perm}")

    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    print()
    print(f"Summary: {analysis_results['recommendations']['summary']}")
    print("\nActions:")
    for action in analysis_results['recommendations']['actions']:
        print(f"  ✓ {action}")

    print("\n" + "=" * 60)
    print("🔒 SECURITY ASSESSMENT")
    print("=" * 60)
    print()
    print(f"Risk Level: {analysis_results['security_assessment']['risk_level'].upper()}")
    print(f"Risk Score: {analysis_results['security_assessment']['risk_score']}/100")

    if analysis_results['security_assessment']['findings']:
        print("\nFindings:")
        for finding in analysis_results['security_assessment']['findings']:
            print(f"  ⚠️ {finding}")

    print("\n" + "=" * 60)
    print("📈 WHAT THIS MEANS FOR YOUR ORGANIZATION")
    print("=" * 60)
    print()
    print("1. Custom Role Consolidation Opportunity:")
    print("   • This custom role is 72.5% similar to bigquery.dataViewer")
    print("   • Could use standard role + minimal additions")
    print()
    print("2. Security Improvements:")
    print("   • Remove unnecessary compute.instances.list permission")
    print("   • Review if pubsub.topics.publish is actually needed")
    print()
    print("3. Compliance Benefits:")
    print("   • Easier to audit standard roles")
    print("   • Reduces permission drift over time")
    print("   • Simplifies access reviews")

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print()
    print("This analyzer would run against ALL organizational custom roles:")
    print("• Automatically via Cloud Function (hourly)")
    print("• Store results in BigQuery for tracking")
    print("• Alert on high-risk permission combinations")
    print("• Generate weekly reports for security team")
    print()

if __name__ == "__main__":
    demo_custom_role_analysis()