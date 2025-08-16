"""
Assessment Agent - RADAR Phase 2

Evaluates security posture and compliance of discovered resources.
The "analyst" of the RADAR system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
# Direct API imports instead of base classes and tools

logger = logging.getLogger(__name__)


class AssessmentAgent:
    """
    RADAR Phase 2: Assess - Security and Compliance Evaluation
    
    This agent evaluates the security posture of discovered resources.
    It identifies vulnerabilities, misconfigurations, and compliance issues.
    
    Key responsibilities:
    - Security vulnerability scanning
    - IAM permission analysis
    - API key security evaluation
    - Compliance checking (CIS, PCI, HIPAA)
    - Risk classification
    """
    
    def __init__(self, project_id: str):
        """Initialize Assessment Agent for security evaluation."""
        self.project_id = project_id
        self.name = "AssessmentAgent"
        self.description = "Evaluates security posture and compliance"
        logger.info(f"🔒 Assessment Agent initialized for project {project_id}"
        )
    
    def get_instruction(self) -> str:
        """Get the instruction for this agent."""
        return """You are the Assessment Agent - the security analyst of RADAR.
        
        Your mission:
        1. Evaluate security posture of all discovered resources
        2. Identify vulnerabilities and misconfigurations
        3. Check compliance with security policies
        4. Assess risk levels for each finding
        5. Provide detailed technical analysis
        
        You have deep analysis tools for security evaluation.
        Be thorough and technical in your assessments.
        Classify all findings by severity and risk impact.
        """
    
    async def assess_security_posture(self, resources: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive security assessment.
        
        Args:
            resources: Optional resources from Recognition Agent
            
        Returns:
            Security assessment with findings, risks, and compliance status
        """
        logger.info(f"🔍 Starting security assessment for {self.project_id}")
        
        assessment_result = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "phase": "assessment"
        }
        
        # Run comprehensive scan (this aggregates multiple tools)
        scan_result = await comprehensive_security_scan(self.project_id, tool_context=self.context)
        
        if scan_result.get("success"):
            # Extract and enhance findings
            findings = scan_result.get("findings", [])
            
            # Classify findings by severity
            classified = self._classify_findings(findings)
            
            # Check compliance
            compliance = self._check_compliance(scan_result)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(classified, compliance)
            
            assessment_result.update({
                "success": True,
                "findings": findings,
                "classified_findings": classified,
                "compliance": compliance,
                "risk_metrics": risk_metrics,
                "risk_level": scan_result.get("risk_level", "UNKNOWN"),
                "risk_score": scan_result.get("risk_score", 0),
                "summary": scan_result.get("summary", {}),
                "requires_immediate_action": len(classified.get("critical", [])) > 0
            })
            
            # Add contextual analysis if resources provided
            if resources:
                assessment_result["resource_risks"] = self._analyze_resource_risks(
                    resources, findings
                )
        else:
            assessment_result.update({
                "success": False,
                "error": scan_result.get("error", "Assessment failed")
            })
        
        return assessment_result
    
    def _classify_findings(self, findings: List[Dict]) -> Dict[str, List]:
        """Classify findings by severity."""
        classified = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": []
        }
        
        for finding in findings:
            severity = finding.get("severity", "LOW").lower()
            if severity in classified:
                classified[severity].append(finding)
            else:
                classified["info"].append(finding)
        
        return classified
    
    def _check_compliance(self, scan_result: Dict) -> Dict[str, Dict]:
        """Check compliance with various standards."""
        compliance = {}
        
        # CIS Benchmark compliance
        compliance["cis"] = self._check_cis_compliance(scan_result)
        
        # PCI DSS compliance
        compliance["pci_dss"] = self._check_pci_compliance(scan_result)
        
        # HIPAA compliance
        compliance["hipaa"] = self._check_hipaa_compliance(scan_result)
        
        # SOC 2 compliance
        compliance["soc2"] = self._check_soc2_compliance(scan_result)
        
        # Overall compliance score
        compliant_count = sum(1 for c in compliance.values() if c.get("compliant"))
        compliance["overall"] = {
            "compliant_frameworks": compliant_count,
            "total_frameworks": len(compliance) - 1,  # Exclude 'overall'
            "compliance_percentage": (compliant_count / (len(compliance) - 1)) * 100
        }
        
        return compliance
    
    def _check_cis_compliance(self, scan_result: Dict) -> Dict[str, Any]:
        """Check CIS benchmark compliance."""
        issues = []
        summary = scan_result.get("summary", {})
        
        # CIS 1.4 - Service accounts with excessive privileges
        if summary.get("iam", {}).get("high_privilege_accounts", 0) > 0:
            issues.append({
                "control": "CIS 1.4",
                "description": "Service accounts with Owner/Editor roles",
                "severity": "HIGH"
            })
        
        # CIS 1.11 - Unrestricted API keys
        if summary.get("api_keys", {}).get("unrestricted", 0) > 0:
            issues.append({
                "control": "CIS 1.11",
                "description": "API keys without restrictions",
                "severity": "HIGH"
            })
        
        # CIS 2.1 - Security findings
        if summary.get("security_findings", {}).get("critical", 0) > 0:
            issues.append({
                "control": "CIS 2.1",
                "description": "Critical security vulnerabilities detected",
                "severity": "CRITICAL"
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - (len(issues) * 20))  # Deduct 20 points per issue
        }
    
    def _check_pci_compliance(self, scan_result: Dict) -> Dict[str, Any]:
        """Check PCI DSS compliance."""
        issues = []
        
        # PCI 2.2 - Security vulnerabilities
        if scan_result.get("risk_level") in ["CRITICAL", "HIGH"]:
            issues.append({
                "requirement": "PCI 2.2",
                "description": "Critical security vulnerabilities present",
                "severity": "HIGH"
            })
        
        # PCI 7.1 - Access control
        summary = scan_result.get("summary", {})
        if summary.get("iam", {}).get("risks", 0) > 5:
            issues.append({
                "requirement": "PCI 7.1",
                "description": "Excessive IAM permissions detected",
                "severity": "MEDIUM"
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - (len(issues) * 25))
        }
    
    def _check_hipaa_compliance(self, scan_result: Dict) -> Dict[str, Any]:
        """Check HIPAA compliance."""
        issues = []
        summary = scan_result.get("summary", {})
        
        # HIPAA 164.312(a) - Access control
        if summary.get("iam", {}).get("risks", 0) > 3:
            issues.append({
                "safeguard": "164.312(a)",
                "description": "Access control risks detected",
                "severity": "HIGH"
            })
        
        # HIPAA 164.312(b) - Audit controls
        if not summary.get("security_findings"):
            issues.append({
                "safeguard": "164.312(b)",
                "description": "Audit logging may be insufficient",
                "severity": "MEDIUM"
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - (len(issues) * 30))
        }
    
    def _check_soc2_compliance(self, scan_result: Dict) -> Dict[str, Any]:
        """Check SOC 2 compliance."""
        issues = []
        
        # Security principle
        if scan_result.get("risk_level") in ["CRITICAL", "HIGH"]:
            issues.append({
                "principle": "Security",
                "description": "High security risks present",
                "severity": "HIGH"
            })
        
        # Availability principle
        findings = scan_result.get("findings", [])
        availability_issues = [f for f in findings if "availability" in f.get("category", "").lower()]
        if availability_issues:
            issues.append({
                "principle": "Availability",
                "description": "Availability risks detected",
                "severity": "MEDIUM"
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "score": max(0, 100 - (len(issues) * 20))
        }
    
    def _calculate_risk_metrics(self, classified: Dict, compliance: Dict) -> Dict[str, Any]:
        """Calculate overall risk metrics."""
        # Count findings by severity
        critical_count = len(classified.get("critical", []))
        high_count = len(classified.get("high", []))
        medium_count = len(classified.get("medium", []))
        low_count = len(classified.get("low", []))
        
        # Calculate risk score (weighted)
        risk_score = (
            critical_count * 10 +
            high_count * 5 +
            medium_count * 2 +
            low_count * 1
        )
        
        # Determine risk level
        if risk_score >= 50 or critical_count > 0:
            risk_level = "CRITICAL"
        elif risk_score >= 30 or high_count > 5:
            risk_level = "HIGH"
        elif risk_score >= 15 or medium_count > 10:
            risk_level = "MEDIUM"
        elif risk_score > 0:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Calculate compliance impact
        compliance_score = compliance.get("overall", {}).get("compliance_percentage", 100)
        if compliance_score < 50:
            risk_score += 20  # Add penalty for poor compliance
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "finding_counts": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
                "total": critical_count + high_count + medium_count + low_count
            },
            "compliance_impact": 100 - compliance_score,
            "requires_immediate_action": critical_count > 0 or risk_score > 40
        }
    
    def _analyze_resource_risks(self, resources: Dict, findings: List[Dict]) -> Dict[str, List]:
        """Analyze risks for specific resources."""
        resource_risks = {}
        
        # Map findings to resources
        for finding in findings:
            resource_name = finding.get("resource_name", "unknown")
            if resource_name not in resource_risks:
                resource_risks[resource_name] = []
            
            resource_risks[resource_name].append({
                "severity": finding.get("severity"),
                "category": finding.get("category"),
                "description": finding.get("description"),
                "recommendation": finding.get("recommendation")
            })
        
        # Identify high-risk resources
        high_risk_resources = []
        for resource, risks in resource_risks.items():
            critical_risks = [r for r in risks if r.get("severity") == "CRITICAL"]
            if critical_risks:
                high_risk_resources.append({
                    "resource": resource,
                    "critical_risks": len(critical_risks),
                    "total_risks": len(risks)
                })
        
        return {
            "by_resource": resource_risks,
            "high_risk_resources": high_risk_resources,
            "total_affected_resources": len(resource_risks)
        }
    
    async def quick_security_check(self) -> Dict[str, Any]:
        """Perform a quick security check (stats only)."""
        logger.info(f"⚡ Quick security check for {self.project_id}")
        
        stats = await get_security_stats(tool_context=self.context)
        
        if stats.get("success"):
            return {
                "success": True,
                "phase": "assessment",
                "check_type": "quick",
                "stats": stats.get("stats", {}),
                "timestamp": datetime.now().isoformat()
            }
        
        return stats