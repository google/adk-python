#!/usr/bin/env python3
"""
Test Coverage Verification
==========================

Verifies comprehensive test coverage across all security agent features
and generates coverage reports for ADK evaluation completeness.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoverageArea:
    """Represents a functional area requiring test coverage"""
    name: str
    description: str
    required_test_cases: List[str]
    priority: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    implemented: bool = False
    test_files: List[str] = None


@dataclass
class CoverageReport:
    """Comprehensive coverage analysis report"""
    total_areas: int
    covered_areas: int
    coverage_percentage: float
    critical_gaps: List[str]
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]


class TestCoverageVerifier:
    """Verifies comprehensive test coverage across all features"""
    
    def __init__(self, evaluation_dir: Path = None):
        """Initialize coverage verifier"""
        self.eval_dir = evaluation_dir or Path(__file__).parent
        self.coverage_areas = self._define_coverage_areas()
        
    def _define_coverage_areas(self) -> List[CoverageArea]:
        """Define all functional areas requiring test coverage"""
        
        return [
            # Custom Roles Analyzer Coverage
            CoverageArea(
                name="Custom Roles Analysis",
                description="Complete custom IAM roles analysis functionality",
                priority="CRITICAL",
                required_test_cases=[
                    "basic_role_analysis",
                    "risk_scoring_validation",
                    "standard_role_mapping",
                    "recommendation_generation",
                    "terraform_export",
                    "permission_comparison",
                    "bulk_analysis",
                    "statistics_dashboard"
                ]
            ),
            
            # Knowledge Base Coverage
            CoverageArea(
                name="Enterprise Knowledge Base",
                description="Complete knowledge base functionality for policies and standards",
                priority="CRITICAL",
                required_test_cases=[
                    "enterprise_policies_crud",
                    "coding_standards_management",
                    "compliance_frameworks",
                    "best_practices_retrieval",
                    "knowledge_search",
                    "statistics_reporting",
                    "import_export_functionality"
                ]
            ),
            
            # API Integration Coverage
            CoverageArea(
                name="API Endpoint Integration",
                description="Complete API functionality across all endpoints",
                priority="HIGH",
                required_test_cases=[
                    "custom_roles_api_workflow",
                    "knowledge_base_api_workflow",
                    "core_security_apis_validation",
                    "error_response_handling",
                    "authentication_authorization",
                    "rate_limiting_compliance"
                ]
            ),
            
            # Security Analysis Coverage
            CoverageArea(
                name="IAM Security Analysis",
                description="Comprehensive IAM security assessment capabilities",
                priority="CRITICAL",
                required_test_cases=[
                    "iam_policy_analysis",
                    "overprivileged_user_detection",
                    "service_account_analysis",
                    "role_binding_assessment",
                    "principle_least_privilege"
                ]
            ),
            
            CoverageArea(
                name="Storage Security Analysis",
                description="Cloud Storage security assessment and recommendations",
                priority="HIGH",
                required_test_cases=[
                    "bucket_security_analysis",
                    "public_access_detection",
                    "encryption_compliance",
                    "lifecycle_policy_review",
                    "access_control_validation"
                ]
            ),
            
            CoverageArea(
                name="Network Security Analysis",
                description="Network and firewall security assessment",
                priority="HIGH",
                required_test_cases=[
                    "firewall_rule_analysis",
                    "vpc_security_assessment",
                    "network_exposure_detection",
                    "load_balancer_security",
                    "private_connectivity_validation"
                ]
            ),
            
            # Compliance Coverage
            CoverageArea(
                name="Compliance Assessment",
                description="Regulatory compliance checking and reporting",
                priority="HIGH",
                required_test_cases=[
                    "soc2_compliance_check",
                    "gdpr_assessment",
                    "hipaa_validation",
                    "pci_dss_compliance",
                    "compliance_reporting"
                ]
            ),
            
            CoverageArea(
                name="Vulnerability Assessment",
                description="Security vulnerability detection and prioritization",
                priority="CRITICAL",
                required_test_cases=[
                    "misconfiguration_detection",
                    "security_finding_analysis",
                    "risk_prioritization",
                    "remediation_guidance",
                    "vulnerability_reporting"
                ]
            ),
            
            # Error Handling Coverage
            CoverageArea(
                name="Error Handling & Edge Cases",
                description="Robust error handling and edge case management",
                priority="HIGH",
                required_test_cases=[
                    "malformed_input_handling",
                    "api_failure_scenarios",
                    "network_connectivity_issues",
                    "authentication_failures",
                    "data_boundary_conditions",
                    "timeout_handling",
                    "resource_exhaustion"
                ]
            ),
            
            # Performance Coverage
            CoverageArea(
                name="Performance & Scalability",
                description="System performance and scalability validation",
                priority="HIGH",
                required_test_cases=[
                    "response_time_benchmarks",
                    "concurrent_user_handling",
                    "large_dataset_processing",
                    "memory_usage_optimization",
                    "database_performance",
                    "scalability_limits",
                    "resource_utilization"
                ]
            ),
            
            # Integration Coverage
            CoverageArea(
                name="GCP Service Integration",
                description="Integration with Google Cloud Platform services",
                priority="CRITICAL",
                required_test_cases=[
                    "asset_inventory_integration",
                    "security_command_center",
                    "cloud_monitoring_integration",
                    "iam_api_integration",
                    "recommender_api_integration",
                    "logging_api_integration"
                ]
            ),
            
            # User Experience Coverage
            CoverageArea(
                name="User Interface & Experience",
                description="Frontend user interface and experience validation",
                priority="MEDIUM",
                required_test_cases=[
                    "streamlit_dashboard_functionality",
                    "interactive_analysis_workflow",
                    "data_visualization",
                    "export_functionality",
                    "real_time_updates",
                    "responsive_design"
                ]
            ),
            
            # Security Coverage
            CoverageArea(
                name="Application Security",
                description="Security of the application itself",
                priority="CRITICAL",
                required_test_cases=[
                    "input_sanitization",
                    "sql_injection_prevention",
                    "xss_protection",
                    "authentication_security",
                    "data_encryption",
                    "audit_logging"
                ]
            ),
            
            # Data Coverage
            CoverageArea(
                name="Data Management",
                description="Data persistence, caching, and management",
                priority="HIGH",
                required_test_cases=[
                    "database_operations",
                    "cache_management",
                    "data_consistency",
                    "backup_recovery",
                    "data_migration",
                    "retention_policies"
                ]
            ),
            
            # Deployment Coverage
            CoverageArea(
                name="Deployment & Operations",
                description="Deployment, monitoring, and operational concerns",
                priority="MEDIUM",
                required_test_cases=[
                    "cloud_run_deployment",
                    "environment_configuration",
                    "health_monitoring",
                    "log_management",
                    "metric_collection",
                    "alerting_setup"
                ]
            )
        ]
    
    def verify_coverage(self) -> CoverageReport:
        """Verify test coverage across all areas"""
        
        logger.info("🔍 Verifying comprehensive test coverage...")
        
        # Scan for test files
        test_files = self._scan_test_files()
        
        # Analyze coverage for each area
        covered_areas = 0
        critical_gaps = []
        recommendations = []
        detailed_analysis = {}
        
        for area in self.coverage_areas:
            coverage_result = self._analyze_area_coverage(area, test_files)
            detailed_analysis[area.name] = coverage_result
            
            if coverage_result["covered"]:
                covered_areas += 1
                area.implemented = True
                area.test_files = coverage_result["test_files"]
            else:
                if area.priority == "CRITICAL":
                    critical_gaps.append(area.name)
                
                # Generate recommendations
                missing_cases = coverage_result["missing_test_cases"]
                if missing_cases:
                    recommendations.append(
                        f"Implement missing test cases for {area.name}: {', '.join(missing_cases[:3])}"
                    )
        
        # Calculate coverage percentage
        coverage_percentage = (covered_areas / len(self.coverage_areas)) * 100
        
        # Generate additional recommendations
        recommendations.extend(self._generate_coverage_recommendations(coverage_percentage, critical_gaps))
        
        logger.info(f"📊 Coverage analysis complete: {coverage_percentage:.1f}% coverage")
        
        return CoverageReport(
            total_areas=len(self.coverage_areas),
            covered_areas=covered_areas,
            coverage_percentage=coverage_percentage,
            critical_gaps=critical_gaps,
            recommendations=recommendations,
            detailed_analysis=detailed_analysis
        )
    
    def _scan_test_files(self) -> Dict[str, Dict[str, Any]]:
        """Scan all test files and extract test case information"""
        
        test_files = {}
        
        # Scan evaluation datasets
        datasets_dir = self.eval_dir / "datasets"
        if datasets_dir.exists():
            for test_file in datasets_dir.glob("*.evalset.json"):
                try:
                    with open(test_file, 'r') as f:
                        test_data = json.load(f)
                    
                    test_files[test_file.name] = {
                        "path": str(test_file),
                        "eval_set_id": test_data.get("eval_set_id", ""),
                        "name": test_data.get("name", ""),
                        "description": test_data.get("description", ""),
                        "test_cases": [
                            case.get("eval_id", "") 
                            for case in test_data.get("eval_cases", [])
                        ]
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Error reading test file {test_file}: {e}")
        
        # Scan traditional test files
        tests_dir = self.eval_dir.parent / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("test_*.py"):
                test_files[test_file.name] = {
                    "path": str(test_file),
                    "type": "python_test",
                    "test_cases": self._extract_python_test_cases(test_file)
                }
        
        logger.info(f"📁 Found {len(test_files)} test files")
        return test_files
    
    def _extract_python_test_cases(self, test_file: Path) -> List[str]:
        """Extract test case names from Python test files"""
        
        test_cases = []
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Simple regex to find test function names
            import re
            test_functions = re.findall(r'def (test_\w+)', content)
            test_cases.extend(test_functions)
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting test cases from {test_file}: {e}")
        
        return test_cases
    
    def _analyze_area_coverage(
        self, 
        area: CoverageArea, 
        test_files: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze test coverage for a specific functional area"""
        
        covered_test_cases = set()
        relevant_test_files = []
        
        # Check each test file for relevant test cases
        for file_name, file_info in test_files.items():
            file_test_cases = file_info.get("test_cases", [])
            
            # Match test cases to required cases (fuzzy matching)
            for required_case in area.required_test_cases:
                for actual_case in file_test_cases:
                    if self._test_case_matches(required_case, actual_case):
                        covered_test_cases.add(required_case)
                        if file_name not in relevant_test_files:
                            relevant_test_files.append(file_name)
        
        # Determine if area is covered
        required_cases = set(area.required_test_cases)
        missing_cases = required_cases - covered_test_cases
        coverage_ratio = len(covered_test_cases) / len(required_cases) if required_cases else 0
        
        # Consider area covered if >= 70% of test cases are implemented
        is_covered = coverage_ratio >= 0.7
        
        return {
            "covered": is_covered,
            "coverage_ratio": coverage_ratio,
            "covered_test_cases": list(covered_test_cases),
            "missing_test_cases": list(missing_cases),
            "test_files": relevant_test_files,
            "priority": area.priority
        }
    
    def _test_case_matches(self, required_case: str, actual_case: str) -> bool:
        """Check if an actual test case matches a required test case"""
        
        # Convert to lowercase for comparison
        required = required_case.lower().replace("_", "").replace("-", "")
        actual = actual_case.lower().replace("_", "").replace("-", "")
        
        # Check for direct substring match
        if required in actual or actual in required:
            return True
        
        # Check for keyword matches
        required_keywords = required.split()
        actual_keywords = actual.split()
        
        # If 60% of keywords match, consider it a match
        if required_keywords and actual_keywords:
            matches = sum(1 for keyword in required_keywords if keyword in actual)
            match_ratio = matches / len(required_keywords)
            return match_ratio >= 0.6
        
        return False
    
    def _generate_coverage_recommendations(
        self, 
        coverage_percentage: float, 
        critical_gaps: List[str]
    ) -> List[str]:
        """Generate recommendations based on coverage analysis"""
        
        recommendations = []
        
        if coverage_percentage < 70:
            recommendations.append(
                "Coverage is below 70% - implement missing test cases as priority"
            )
        elif coverage_percentage < 85:
            recommendations.append(
                "Coverage is good but could be improved - focus on medium priority areas"
            )
        else:
            recommendations.append(
                "Excellent test coverage - maintain current quality and add edge cases"
            )
        
        if critical_gaps:
            recommendations.append(
                f"Address critical coverage gaps immediately: {', '.join(critical_gaps[:3])}"
            )
        
        # Specific recommendations based on common patterns
        recommendations.extend([
            "Ensure all new features have corresponding test cases",
            "Add performance regression tests for critical operations",
            "Implement end-to-end integration tests for major workflows",
            "Include security penetration testing scenarios",
            "Add chaos engineering tests for failure scenarios"
        ])
        
        return recommendations
    
    def generate_coverage_report(self, output_file: str = "coverage_report.json"):
        """Generate and save comprehensive coverage report"""
        
        logger.info("📋 Generating comprehensive coverage report...")
        
        # Verify coverage
        coverage_report = self.verify_coverage()
        
        # Create detailed report
        report_data = {
            "coverage_summary": {
                "total_functional_areas": coverage_report.total_areas,
                "covered_areas": coverage_report.covered_areas,
                "coverage_percentage": round(coverage_report.coverage_percentage, 2),
                "report_generated_at": datetime.now().isoformat()
            },
            "coverage_by_priority": self._analyze_coverage_by_priority(),
            "critical_gaps": coverage_report.critical_gaps,
            "recommendations": coverage_report.recommendations,
            "detailed_area_analysis": coverage_report.detailed_analysis,
            "test_file_inventory": self._generate_test_file_inventory(),
            "coverage_matrix": self._generate_coverage_matrix(),
            "quality_metrics": self._calculate_quality_metrics(coverage_report)
        }
        
        # Save to file
        output_path = self.eval_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📄 Coverage report saved to: {output_path}")
        
        # Generate human-readable summary
        self._generate_summary_report(coverage_report)
        
        return coverage_report
    
    def _analyze_coverage_by_priority(self) -> Dict[str, Dict[str, int]]:
        """Analyze coverage breakdown by priority level"""
        
        by_priority = {"CRITICAL": {}, "HIGH": {}, "MEDIUM": {}, "LOW": {}}
        
        for priority in by_priority.keys():
            priority_areas = [area for area in self.coverage_areas if area.priority == priority]
            covered = sum(1 for area in priority_areas if area.implemented)
            
            by_priority[priority] = {
                "total": len(priority_areas),
                "covered": covered,
                "percentage": round((covered / len(priority_areas)) * 100, 1) if priority_areas else 0
            }
        
        return by_priority
    
    def _generate_test_file_inventory(self) -> Dict[str, Any]:
        """Generate inventory of all test files"""
        
        test_files = self._scan_test_files()
        
        inventory = {
            "total_files": len(test_files),
            "evalset_files": len([f for f in test_files.values() if f.get("eval_set_id")]),
            "python_test_files": len([f for f in test_files.values() if f.get("type") == "python_test"]),
            "total_test_cases": sum(len(f.get("test_cases", [])) for f in test_files.values()),
            "files": test_files
        }
        
        return inventory
    
    def _generate_coverage_matrix(self) -> Dict[str, List[str]]:
        """Generate matrix showing which test files cover which areas"""
        
        matrix = {}
        test_files = self._scan_test_files()
        
        for area in self.coverage_areas:
            if area.test_files:
                matrix[area.name] = area.test_files
            else:
                matrix[area.name] = []
        
        return matrix
    
    def _calculate_quality_metrics(self, coverage_report: CoverageReport) -> Dict[str, Any]:
        """Calculate quality metrics for the test suite"""
        
        return {
            "coverage_score": round(coverage_report.coverage_percentage, 1),
            "critical_coverage": round(
                len([area for area in self.coverage_areas 
                     if area.priority == "CRITICAL" and area.implemented]) /
                len([area for area in self.coverage_areas if area.priority == "CRITICAL"]) * 100, 1
            ),
            "test_completeness": "EXCELLENT" if coverage_report.coverage_percentage >= 90 else
                               "GOOD" if coverage_report.coverage_percentage >= 80 else
                               "NEEDS_IMPROVEMENT" if coverage_report.coverage_percentage >= 70 else
                               "POOR",
            "areas_needing_attention": len(coverage_report.critical_gaps),
            "recommendation_count": len(coverage_report.recommendations)
        }
    
    def _generate_summary_report(self, coverage_report: CoverageReport):
        """Generate human-readable summary report"""
        
        summary_path = self.eval_dir / "coverage_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ADK Security Agent - Test Coverage Analysis\n")
            f.write("=" * 45 + "\n\n")
            
            f.write(f"Overall Coverage: {coverage_report.coverage_percentage:.1f}%\n")
            f.write(f"Covered Areas: {coverage_report.covered_areas}/{coverage_report.total_areas}\n\n")
            
            if coverage_report.critical_gaps:
                f.write("Critical Coverage Gaps:\n")
                for gap in coverage_report.critical_gaps:
                    f.write(f"- {gap}\n")
                f.write("\n")
            
            f.write("Recommendations:\n")
            for rec in coverage_report.recommendations[:10]:  # Top 10 recommendations
                f.write(f"- {rec}\n")
            f.write("\n")
            
            f.write("Coverage by Functional Area:\n")
            priority_coverage = self._analyze_coverage_by_priority()
            for priority, data in priority_coverage.items():
                f.write(f"{priority}: {data['covered']}/{data['total']} ({data['percentage']}%)\n")
        
        logger.info(f"📋 Coverage summary saved to: {summary_path}")


def main():
    """Main entry point for coverage verification"""
    
    verifier = TestCoverageVerifier()
    coverage_report = verifier.generate_coverage_report()
    
    print(f"\n🎯 Test Coverage Analysis Complete!")
    print(f"Overall Coverage: {coverage_report.coverage_percentage:.1f}%")
    print(f"Areas Covered: {coverage_report.covered_areas}/{coverage_report.total_areas}")
    
    if coverage_report.critical_gaps:
        print(f"❌ Critical Gaps: {len(coverage_report.critical_gaps)}")
        for gap in coverage_report.critical_gaps[:3]:
            print(f"  - {gap}")
    else:
        print("✅ No critical coverage gaps detected")
    
    if coverage_report.coverage_percentage >= 85:
        print("🎉 Excellent test coverage!")
    elif coverage_report.coverage_percentage >= 70:
        print("👍 Good test coverage")
    else:
        print("⚠️ Test coverage needs improvement")


if __name__ == "__main__":
    main()