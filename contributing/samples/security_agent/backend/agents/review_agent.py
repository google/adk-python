"""
Review Agent - RADAR Phase 5

Verifies changes and generates comprehensive reports.
The "auditor" of the RADAR system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
# Direct API imports instead of base classes and tools

logger = logging.getLogger(__name__)


class ReviewAgent:
    """
    RADAR Phase 5: Review - Verification and Reporting
    
    This agent verifies changes and generates reports.
    It uses the same tools as Assessment but for verification purposes.
    
    Key responsibilities:
    - Verify remediation effectiveness
    - Confirm security improvements
    - Track metrics and trends
    - Generate executive reports
    - Identify continuous improvement opportunities
    """
    
    def __init__(self, project_id: str):
        """Initialize Review Agent for verification and reporting."""
        self.project_id = project_id
        self.name = "ReviewAgent"
        self.description = "Verifies changes and generates reports"
        logger.info(f"📊 Review Agent initialized for project {project_id}")
        
        # Store historical data for trend analysis
        self.historical_data = []
    
    def get_instruction(self) -> str:
        """Get the instruction for this agent."""
        return """You are the Review Agent - the auditor of RADAR.
        
        Your mission:
        1. Verify all remediation actions were successful
        2. Confirm security improvements
        3. Track metrics and trends over time
        4. Generate clear executive reports
        5. Identify areas for continuous improvement
        
        You verify the work of other agents.
        Provide clear reports on what improved and what still needs work.
        Track progress over time and identify patterns.
        """
    
    async def review_and_report(
        self,
        actions_taken: Optional[Dict[str, Any]] = None,
        initial_assessment: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Review changes and generate comprehensive report.
        
        Args:
            actions_taken: Actions from Action Agent
            initial_assessment: Original assessment for comparison
            
        Returns:
            Comprehensive review report
        """
        logger.info(f"📊 Generating review report for {self.project_id}")
        
        review_result = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "phase": "review",
            "verification_status": {},
            "improvements": [],
            "remaining_issues": [],
            "metrics": {},
            "trends": [],
            "executive_summary": "",
            "next_steps": []
        }
        
        # Re-run security scan to verify current state
        current_scan = await comprehensive_security_scan(self.project_id, tool_context=self.context)
        
        if current_scan.get("success"):
            # Verify improvements if we have initial assessment
            if initial_assessment:
                verification = self._verify_improvements(initial_assessment, current_scan)
                review_result["verification_status"] = verification
                review_result["improvements"] = verification.get("improvements", [])
            
            # Analyze current state
            review_result["metrics"] = self._extract_metrics(current_scan)
            
            # Identify remaining issues
            review_result["remaining_issues"] = self._identify_remaining_issues(current_scan)
            
            # Track trends
            review_result["trends"] = self._analyze_trends(current_scan)
            
            # Verify actions if provided
            if actions_taken:
                action_verification = self._verify_actions(actions_taken, current_scan)
                review_result["action_verification"] = action_verification
            
            # Generate executive summary
            review_result["executive_summary"] = self._generate_executive_summary(
                review_result,
                actions_taken,
                current_scan
            )
            
            # Determine next steps
            review_result["next_steps"] = self._recommend_next_steps(current_scan, review_result)
            
            review_result["success"] = True
            
            # Store for historical tracking
            self.historical_data.append({
                "timestamp": review_result["timestamp"],
                "metrics": review_result["metrics"]
            })
            
        else:
            review_result["success"] = False
            review_result["error"] = current_scan.get("error", "Review scan failed")
        
        return review_result
    
    def _verify_improvements(self, initial: Dict, current: Dict) -> Dict[str, Any]:
        """Verify improvements between initial and current assessment."""
        verification = {
            "improved": [],
            "worsened": [],
            "unchanged": [],
            "improvements": []
        }
        
        # Compare risk scores
        initial_score = initial.get("risk_score", 0)
        current_score = current.get("risk_score", 0)
        
        if current_score < initial_score:
            improvement_pct = ((initial_score - current_score) / max(initial_score, 1)) * 100
            verification["improvements"].append(
                f"Risk score improved by {improvement_pct:.1f}% (from {initial_score} to {current_score})"
            )
            verification["improved"].append("overall_risk_score")
        elif current_score > initial_score:
            verification["worsened"].append("overall_risk_score")
        else:
            verification["unchanged"].append("overall_risk_score")
        
        # Compare finding counts
        initial_findings = initial.get("summary", {}).get("security_findings", {})
        current_findings = current.get("summary", {}).get("security_findings", {})
        
        for severity in ["critical", "high", "medium", "low"]:
            initial_count = initial_findings.get(severity, 0)
            current_count = current_findings.get(severity, 0)
            
            if current_count < initial_count:
                verification["improvements"].append(
                    f"Reduced {severity} findings from {initial_count} to {current_count}"
                )
                verification["improved"].append(f"{severity}_findings")
            elif current_count > initial_count:
                verification["worsened"].append(f"{severity}_findings")
        
        # Compare compliance
        initial_compliance = initial.get("compliance", {})
        current_compliance = current.get("compliance", {})
        
        for framework in ["cis", "pci_dss", "hipaa"]:
            if framework in initial_compliance and framework in current_compliance:
                was_compliant = initial_compliance[framework].get("compliant", False)
                is_compliant = current_compliance[framework].get("compliant", False)
                
                if not was_compliant and is_compliant:
                    verification["improvements"].append(
                        f"Achieved {framework.upper()} compliance"
                    )
                    verification["improved"].append(f"{framework}_compliance")
                elif was_compliant and not is_compliant:
                    verification["worsened"].append(f"{framework}_compliance")
        
        # Calculate overall verification status
        verification["overall_improved"] = len(verification["improved"]) > len(verification["worsened"])
        verification["improvement_rate"] = (
            len(verification["improved"]) / 
            max(len(verification["improved"]) + len(verification["worsened"]) + len(verification["unchanged"]), 1)
        ) * 100
        
        return verification
    
    def _extract_metrics(self, scan: Dict) -> Dict[str, Any]:
        """Extract key metrics from scan results."""
        summary = scan.get("summary", {})
        
        metrics = {
            "risk_level": scan.get("risk_level", "UNKNOWN"),
            "risk_score": scan.get("risk_score", 0),
            "total_findings": len(scan.get("findings", [])),
            "critical_findings": summary.get("security_findings", {}).get("critical", 0),
            "high_findings": summary.get("security_findings", {}).get("high", 0),
            "total_assets": summary.get("assets", {}).get("total", 0),
            "service_accounts": summary.get("iam", {}).get("service_accounts", 0),
            "unrestricted_api_keys": summary.get("api_keys", {}).get("unrestricted", 0),
            "compliance_score": self._calculate_compliance_score(scan),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
    
    def _calculate_compliance_score(self, scan: Dict) -> float:
        """Calculate overall compliance score."""
        compliance = scan.get("compliance", {})
        if not compliance:
            return 0.0
        
        scores = []
        for framework, status in compliance.items():
            if framework != "overall" and isinstance(status, dict):
                scores.append(status.get("score", 0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _identify_remaining_issues(self, scan: Dict) -> List[Dict]:
        """Identify top remaining issues that need attention."""
        remaining = []
        findings = scan.get("findings", [])
        
        # Get top critical and high findings
        for finding in findings[:10]:  # Top 10 issues
            severity = finding.get("severity", "UNKNOWN")
            if severity in ["CRITICAL", "HIGH"]:
                remaining.append({
                    "severity": severity,
                    "category": finding.get("category"),
                    "description": finding.get("description"),
                    "resource": finding.get("resource_name"),
                    "recommendation": finding.get("recommendation")
                })
        
        # Add compliance issues
        compliance = scan.get("compliance", {})
        for framework, status in compliance.items():
            if framework != "overall" and not status.get("compliant", True):
                for issue in status.get("issues", [])[:2]:  # Top 2 per framework
                    remaining.append({
                        "severity": "MEDIUM",
                        "category": "compliance",
                        "description": f"{framework.upper()}: {issue.get('description', 'Compliance issue')}",
                        "resource": "organization",
                        "recommendation": f"Address {framework.upper()} requirement"
                    })
        
        return remaining
    
    def _analyze_trends(self, current_scan: Dict) -> List[Dict]:
        """Analyze trends from historical data."""
        trends = []
        
        if len(self.historical_data) >= 2:
            # Compare with previous scan
            previous = self.historical_data[-2] if len(self.historical_data) > 1 else self.historical_data[-1]
            current_metrics = self._extract_metrics(current_scan)
            previous_metrics = previous.get("metrics", {})
            
            # Risk score trend
            risk_trend = current_metrics["risk_score"] - previous_metrics.get("risk_score", 0)
            if risk_trend < 0:
                trends.append({
                    "metric": "risk_score",
                    "direction": "improving",
                    "change": abs(risk_trend),
                    "description": f"Risk score decreased by {abs(risk_trend)} points"
                })
            elif risk_trend > 0:
                trends.append({
                    "metric": "risk_score",
                    "direction": "worsening",
                    "change": risk_trend,
                    "description": f"Risk score increased by {risk_trend} points"
                })
            
            # Finding trends
            finding_trend = current_metrics["total_findings"] - previous_metrics.get("total_findings", 0)
            if finding_trend != 0:
                trends.append({
                    "metric": "total_findings",
                    "direction": "improving" if finding_trend < 0 else "worsening",
                    "change": abs(finding_trend),
                    "description": f"Total findings {'decreased' if finding_trend < 0 else 'increased'} by {abs(finding_trend)}"
                })
        
        return trends
    
    def _verify_actions(self, actions_taken: Dict, current_scan: Dict) -> Dict[str, Any]:
        """Verify that actions were effective."""
        verification = {
            "actions_verified": 0,
            "actions_effective": 0,
            "actions_ineffective": 0,
            "effectiveness_rate": 0,
            "details": []
        }
        
        action_log = actions_taken.get("action_log", [])
        
        for action in action_log:
            if action.get("status") == "succeeded":
                verification["actions_verified"] += 1
                
                # Check if the action had the desired effect
                # This is simplified - in production would check specific metrics
                effective = self._check_action_effectiveness(action, current_scan)
                
                if effective:
                    verification["actions_effective"] += 1
                    verification["details"].append({
                        "action": action.get("description"),
                        "status": "effective",
                        "verification": "Confirmed improvement in security posture"
                    })
                else:
                    verification["actions_ineffective"] += 1
                    verification["details"].append({
                        "action": action.get("description"),
                        "status": "ineffective",
                        "verification": "Action completed but issue persists"
                    })
        
        if verification["actions_verified"] > 0:
            verification["effectiveness_rate"] = (
                verification["actions_effective"] / verification["actions_verified"]
            ) * 100
        
        return verification
    
    def _check_action_effectiveness(self, action: Dict, scan: Dict) -> bool:
        """Check if a specific action was effective."""
        # Simplified effectiveness check
        # In production, would check specific metrics related to the action
        
        description = action.get("description", "").lower()
        
        if "api key" in description:
            # Check if API key issues decreased
            return scan.get("summary", {}).get("api_keys", {}).get("unrestricted", 1) == 0
        elif "iam" in description:
            # Check if IAM issues decreased
            return scan.get("summary", {}).get("iam", {}).get("high_privilege_accounts", 1) == 0
        else:
            # Default: assume effective if risk score decreased
            return scan.get("risk_score", 100) < 50
    
    def _generate_executive_summary(
        self,
        review: Dict,
        actions: Optional[Dict],
        scan: Dict
    ) -> str:
        """Generate executive summary of the review."""
        metrics = review.get("metrics", {})
        verification = review.get("verification_status", {})
        trends = review.get("trends", [])
        
        summary = f"""
# RADAR Operations Review Report

**Date**: {review['timestamp']}
**Project**: {self.project_id}

## Current Security Status
- **Risk Level**: {metrics.get('risk_level', 'UNKNOWN')}
- **Risk Score**: {metrics.get('risk_score', 0)}
- **Compliance Score**: {metrics.get('compliance_score', 0):.1f}%

## Key Metrics
- **Total Security Findings**: {metrics.get('total_findings', 0)}
- **Critical Issues**: {metrics.get('critical_findings', 0)}
- **High Priority Issues**: {metrics.get('high_findings', 0)}
"""
        
        # Add action summary if available
        if actions:
            total_actions = actions.get('actions_attempted', 0)
            succeeded = actions.get('actions_succeeded', 0)
            summary += f"""
## Actions Taken
- **Total Actions**: {total_actions}
- **Successful**: {succeeded}
- **Success Rate**: {(succeeded/max(total_actions, 1))*100:.1f}%
"""
            
            # Add verification if available
            if "action_verification" in review:
                av = review["action_verification"]
                summary += f"""
## Action Effectiveness
- **Actions Verified**: {av.get('actions_verified', 0)}
- **Effective Actions**: {av.get('actions_effective', 0)}
- **Effectiveness Rate**: {av.get('effectiveness_rate', 0):.1f}%
"""
        
        # Add improvements if available
        if verification.get("improvements"):
            summary += "\n## Key Improvements\n"
            for improvement in verification["improvements"][:5]:
                summary += f"- ✅ {improvement}\n"
        
        # Add trends
        if trends:
            summary += "\n## Trends\n"
            for trend in trends[:3]:
                icon = "📈" if trend["direction"] == "worsening" else "📉"
                summary += f"- {icon} {trend['description']}\n"
        
        # Add remaining issues
        remaining = review.get("remaining_issues", [])
        if remaining:
            summary += f"\n## Top Remaining Issues ({len(remaining)} total)\n"
            for issue in remaining[:3]:
                summary += f"- [{issue['severity']}] {issue['description']}\n"
        
        # Add recommendation
        if metrics.get("risk_level") in ["CRITICAL", "HIGH"]:
            summary += """
## Recommendation
⚠️ **Immediate action required** - Critical security issues remain unresolved.
Continue with remediation efforts and schedule daily reviews until risk level improves.
"""
        elif metrics.get("risk_level") == "MEDIUM":
            summary += """
## Recommendation
📋 **Continued attention needed** - Moderate security risks present.
Schedule weekly reviews and address remaining high-priority issues.
"""
        else:
            summary += """
## Recommendation
✅ **Security posture acceptable** - Continue with regular monitoring.
Schedule monthly reviews to maintain security baseline.
"""
        
        return summary
    
    def _recommend_next_steps(self, scan: Dict, review: Dict) -> List[str]:
        """Recommend next steps based on review findings."""
        next_steps = []
        metrics = review.get("metrics", {})
        remaining = review.get("remaining_issues", [])
        
        # Priority 1: Address critical issues
        if metrics.get("critical_findings", 0) > 0:
            next_steps.append(f"URGENT: Address {metrics['critical_findings']} critical security findings immediately")
        
        # Priority 2: Compliance gaps
        compliance_score = metrics.get("compliance_score", 100)
        if compliance_score < 80:
            next_steps.append("Improve compliance posture - current score below 80%")
        
        # Priority 3: High findings
        if metrics.get("high_findings", 0) > 5:
            next_steps.append(f"Schedule remediation for {metrics['high_findings']} high-priority findings")
        
        # Priority 4: Trending issues
        for trend in review.get("trends", []):
            if trend.get("direction") == "worsening":
                next_steps.append(f"Investigate worsening trend: {trend.get('description')}")
        
        # Priority 5: Preventive measures
        if len(remaining) > 10:
            next_steps.append("Implement automated security scanning to prevent issue accumulation")
        
        # Always include continuous improvement
        next_steps.extend([
            "Schedule next RADAR cycle in 7 days",
            "Review and update security policies",
            "Conduct security training for development teams"
        ])
        
        return next_steps[:10]  # Top 10 next steps