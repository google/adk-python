#!/usr/bin/env python3
"""
Compliance Checker - Validates controls against actual BigQuery environment state
This is the "glue layer" that bridges policy (what SHOULD be) with reality (what IS)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from ..base import bq_client, check_client
from .controls import SecurityControlsInventory, SecurityControl

logger = logging.getLogger(__name__)


@dataclass
class ControlViolation:
    """Individual violation of a security control"""
    control_id: str
    control_name: str
    severity: str
    resource_name: str
    violation_details: str
    remediation: str


@dataclass
class ControlComplianceStatus:
    """Compliance status for a single control"""
    control_id: str
    control_name: str
    severity: str
    category: str
    status: str  # PASS, FAIL, UNKNOWN, NOT_APPLICABLE
    violations: List[ControlViolation]
    violation_count: int
    validation_query: Optional[str]
    checked_at: str


@dataclass
class ComplianceReport:
    """Complete compliance assessment report"""
    service_type: str
    total_controls_checked: int
    controls_passed: int
    controls_failed: int
    controls_unknown: int
    compliance_score: float  # 0-100
    control_statuses: List[ControlComplianceStatus]
    total_violations: int
    violations_by_severity: Dict[str, int]
    summary: str


class ComplianceChecker:
    """
    Validates security controls against actual GCP environment state in BigQuery

    This is the critical "glue layer" that:
    1. Takes security control policies (what SHOULD exist)
    2. Executes validation queries against BigQuery (what DOES exist)
    3. Returns compliance gaps and violations
    """

    def __init__(self):
        self.controls_inventory = SecurityControlsInventory()
        self.bq_client = check_client()

    def check_compliance(
        self,
        service_type: str,
        controls_to_check: Optional[List[str]] = None
    ) -> ComplianceReport:
        """
        Check compliance for a service type against actual BigQuery data

        Args:
            service_type: Service type to check (e.g., "storage", "compute", "all")
            controls_to_check: Optional list of control IDs to check. If None, checks all applicable controls.

        Returns:
            ComplianceReport with detailed status for each control
        """
        from datetime import datetime

        # Get applicable controls
        if controls_to_check:
            controls = [
                self.controls_inventory.controls[cid]
                for cid in controls_to_check
                if cid in self.controls_inventory.controls
            ]
        else:
            controls = self.controls_inventory.get_controls_for_service(service_type)

        control_statuses = []
        total_violations = 0
        violations_by_severity = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        # Check each control
        for control in controls:
            status = self._check_control(control)
            control_statuses.append(status)

            if status.status == 'FAIL':
                total_violations += status.violation_count
                violations_by_severity[status.severity] += status.violation_count

        # Calculate metrics
        controls_passed = sum(1 for s in control_statuses if s.status == 'PASS')
        controls_failed = sum(1 for s in control_statuses if s.status == 'FAIL')
        controls_unknown = sum(1 for s in control_statuses if s.status == 'UNKNOWN')

        compliance_score = (controls_passed / len(controls)) * 100 if controls else 0

        # Generate summary
        summary = self._generate_summary(
            service_type,
            len(controls),
            controls_passed,
            controls_failed,
            total_violations,
            compliance_score
        )

        return ComplianceReport(
            service_type=service_type,
            total_controls_checked=len(controls),
            controls_passed=controls_passed,
            controls_failed=controls_failed,
            controls_unknown=controls_unknown,
            compliance_score=round(compliance_score, 1),
            control_statuses=control_statuses,
            total_violations=total_violations,
            violations_by_severity=violations_by_severity,
            summary=summary
        )

    def _check_control(self, control: SecurityControl) -> ControlComplianceStatus:
        """Check a single control against BigQuery"""
        from datetime import datetime

        # If no validation query, mark as unknown
        if not control.validation_query:
            return ControlComplianceStatus(
                control_id=control.id,
                control_name=control.name,
                severity=control.severity,
                category=control.category.value,
                status='UNKNOWN',
                violations=[],
                violation_count=0,
                validation_query=None,
                checked_at=datetime.utcnow().isoformat()
            )

        try:
            # Execute validation query
            logger.info(f"Checking control {control.id}: {control.name}")
            logger.debug(f"Query: {control.validation_query}")

            query_job = self.bq_client.query(control.validation_query)
            results = list(query_job)

            # If query returns rows, there are violations
            violations = []
            if results:
                for row in results:
                    # Convert row to dict for easier processing
                    row_dict = dict(row.items())

                    # Extract resource identifier (first column typically)
                    resource_name = str(list(row_dict.values())[0]) if row_dict else "Unknown"

                    # Create violation details from all columns
                    violation_details = ", ".join([
                        f"{key}={value}"
                        for key, value in row_dict.items()
                    ])

                    violations.append(ControlViolation(
                        control_id=control.id,
                        control_name=control.name,
                        severity=control.severity,
                        resource_name=resource_name,
                        violation_details=violation_details,
                        remediation=control.implementation_guidance
                    ))

            status = 'FAIL' if violations else 'PASS'

            logger.info(f"Control {control.id}: {status} ({len(violations)} violations)")

            return ControlComplianceStatus(
                control_id=control.id,
                control_name=control.name,
                severity=control.severity,
                category=control.category.value,
                status=status,
                violations=violations,
                violation_count=len(violations),
                validation_query=control.validation_query,
                checked_at=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Error checking control {control.id}: {e}")
            return ControlComplianceStatus(
                control_id=control.id,
                control_name=control.name,
                severity=control.severity,
                category=control.category.value,
                status='UNKNOWN',
                violations=[],
                violation_count=0,
                validation_query=control.validation_query,
                checked_at=datetime.utcnow().isoformat()
            )

    def _generate_summary(
        self,
        service_type: str,
        total_controls: int,
        passed: int,
        failed: int,
        total_violations: int,
        score: float
    ) -> str:
        """Generate human-readable summary"""
        return (
            f"Compliance check for '{service_type}' services: "
            f"{passed}/{total_controls} controls passed ({score:.1f}% compliant). "
            f"{failed} controls failed with {total_violations} total violations."
        )

    def get_critical_violations(self, report: ComplianceReport) -> List[ControlViolation]:
        """Extract all critical severity violations from a report"""
        critical_violations = []
        for status in report.control_statuses:
            if status.severity == 'critical' and status.status == 'FAIL':
                critical_violations.extend(status.violations)
        return critical_violations

    def get_violations_by_control(
        self,
        report: ComplianceReport,
        control_id: str
    ) -> List[ControlViolation]:
        """Get all violations for a specific control"""
        for status in report.control_statuses:
            if status.control_id == control_id:
                return status.violations
        return []

    def format_report(self, report: ComplianceReport, detailed: bool = False) -> str:
        """Format compliance report as human-readable text"""
        output = []
        output.append("=" * 80)
        output.append(f"  COMPLIANCE REPORT: {report.service_type.upper()}")
        output.append("=" * 80)
        output.append("")
        output.append(f"Overall Compliance Score: {report.compliance_score}%")
        output.append(f"Controls Checked: {report.total_controls_checked}")
        output.append(f"  ✅ Passed: {report.controls_passed}")
        output.append(f"  ❌ Failed: {report.controls_failed}")
        output.append(f"  ❓ Unknown: {report.controls_unknown}")
        output.append("")
        output.append(f"Total Violations: {report.total_violations}")
        output.append(f"  🔴 Critical: {report.violations_by_severity['critical']}")
        output.append(f"  🟠 High: {report.violations_by_severity['high']}")
        output.append(f"  🟡 Medium: {report.violations_by_severity['medium']}")
        output.append(f"  🟢 Low: {report.violations_by_severity['low']}")
        output.append("")

        # Failed controls
        if report.controls_failed > 0:
            output.append("FAILED CONTROLS:")
            output.append("-" * 80)
            for status in report.control_statuses:
                if status.status == 'FAIL':
                    output.append(f"\n❌ [{status.control_id}] {status.control_name} ({status.severity})")
                    output.append(f"   Violations: {status.violation_count}")

                    if detailed:
                        for v in status.violations[:3]:  # Show first 3
                            output.append(f"   - {v.resource_name}: {v.violation_details}")
                        if status.violation_count > 3:
                            output.append(f"   ... and {status.violation_count - 3} more")
                        output.append(f"   Remediation: {v.remediation}")

        output.append("")
        output.append("=" * 80)

        return "\n".join(output)


# Convenience function for direct use
def check_service_compliance(
    service_type: str,
    detailed: bool = False
) -> str:
    """
    Check compliance for a service type

    Args:
        service_type: Service type (storage, compute, bigquery, etc.)
        detailed: Include detailed violation information

    Returns:
        JSON string with compliance report
    """
    checker = ComplianceChecker()
    report = checker.check_compliance(service_type)

    # Convert to dict for JSON serialization
    from dataclasses import asdict
    import json

    report_dict = asdict(report)

    if detailed:
        # Add formatted text for detailed view
        report_dict['formatted_report'] = checker.format_report(report, detailed=True)

    return json.dumps(report_dict, indent=2)
