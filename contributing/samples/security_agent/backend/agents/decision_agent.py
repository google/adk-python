"""
Decision Agent - RADAR Phase 3

Prioritizes issues and generates actionable recommendations.
The "strategist" of the RADAR system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
# Direct API imports instead of base classes and tools

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    RADAR Phase 3: Decide - Prioritization and Recommendations
    
    This agent prioritizes issues and generates actionable recommendations.
    It uses threat intelligence and business context to make decisions.
    
    Key responsibilities:
    - Issue prioritization based on risk and impact
    - Threat intelligence integration
    - Recommendation generation
    - Remediation planning
    - Cost-benefit analysis
    """
    
    def __init__(self, project_id: str):
        """Initialize Decision Agent for prioritization and recommendations."""
        self.project_id = project_id
        self.name = "DecisionAgent"
        self.description = "Prioritizes issues and generates recommendations"
        logger.info(f"🎯 Decision Agent initialized for project {project_id}")
    
    def get_instruction(self) -> str:
        """Get the instruction for this agent."""
        return """You are the Decision Agent - the strategist of RADAR.
        
        Your mission:
        1. Prioritize security issues based on risk and impact
        2. Consider threat intelligence from advisory notifications
        3. Generate actionable recommendations
        4. Create remediation plans with clear steps
        5. Balance security needs with operational requirements
        
        You analyze findings from Assessment Agent and external intelligence.
        Provide clear, prioritized action plans.
        Consider both immediate fixes and long-term improvements.
        """
    
    async def prioritize_and_recommend(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prioritize issues and generate recommendations based on assessment.
        
        Args:
            assessment: Security assessment from Assessment Agent
            
        Returns:
            Prioritized action plan with recommendations
        """
        logger.info(f"🎯 Generating prioritized recommendations for {self.project_id}")
        
        decision_result = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "phase": "decision",
            "priority_queue": [],
            "recommendations": {
                "immediate": [],     # Do within 24 hours
                "short_term": [],    # Do within 1 week
                "long_term": []      # Do within 1 month
            },
            "remediation_plan": [],
            "estimated_effort": {},
            "risk_reduction_impact": {}
        }
        
        # Check advisory notifications for context
        threat_context = await self._get_threat_context()
        
        # Build priority queue from assessment
        if assessment.get("success") and assessment.get("classified_findings"):
            priority_queue = self._build_priority_queue(
                assessment["classified_findings"],
                threat_context
            )
            decision_result["priority_queue"] = priority_queue
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                priority_queue,
                assessment,
                threat_context
            )
            decision_result["recommendations"] = recommendations
            
            # Create remediation plan
            remediation_plan = self._create_remediation_plan(priority_queue)
            decision_result["remediation_plan"] = remediation_plan
            
            # Estimate effort and impact
            decision_result["estimated_effort"] = self._estimate_effort(remediation_plan)
            decision_result["risk_reduction_impact"] = self._calculate_risk_reduction(
                assessment,
                remediation_plan
            )
            
            decision_result["success"] = True
        else:
            decision_result["success"] = False
            decision_result["error"] = "No assessment data to prioritize"
        
        return decision_result
    
    async def _get_threat_context(self) -> Dict[str, Any]:
        """Get current threat intelligence from advisory notifications."""
        logger.info("📢 Checking advisory notifications for threat context")
        
        threat_context = {
            "critical_advisories": [],
            "active_threats": [],
            "recommended_patches": []
        }
        
        # Check recent advisory notifications
        advisory_result = await analyze_advisory_notifications(
            days_back=7,
            tool_context=self.context
        )
        
        if advisory_result.get("success"):
            analysis = advisory_result.get("analysis", {})
            threat_context["critical_advisories"] = analysis.get("critical_notifications", [])
            
            # Extract active threats
            for notification in threat_context["critical_advisories"]:
                if "CVE" in notification.get("subject", "") or "vulnerability" in notification.get("subject", "").lower():
                    threat_context["active_threats"].append({
                        "type": "vulnerability",
                        "description": notification.get("subject"),
                        "date": notification.get("date"),
                        "action_required": True
                    })
        
        return threat_context
    
    def _build_priority_queue(self, classified_findings: Dict, threat_context: Dict) -> List[Dict]:
        """Build prioritized queue of issues to address."""
        priority_queue = []
        priority_counter = 0
        
        # Priority 1: Critical findings + active threats
        for finding in classified_findings.get("critical", []):
            priority_counter += 1
            priority_queue.append({
                "priority": 1,
                "priority_score": 100,
                "finding": finding,
                "reason": "Critical security vulnerability",
                "action": "Immediate remediation required",
                "estimated_time": "1-2 hours",
                "impact": "HIGH",
                "effort": "MEDIUM"
            })
        
        # Add critical advisories
        for advisory in threat_context.get("critical_advisories", []):
            priority_counter += 1
            priority_queue.append({
                "priority": 1,
                "priority_score": 95,
                "finding": {
                    "type": "advisory",
                    "description": advisory.get("subject", "Critical advisory"),
                    "severity": "CRITICAL"
                },
                "reason": "Critical security advisory",
                "action": "Review and apply security bulletin",
                "estimated_time": "2-4 hours",
                "impact": "HIGH",
                "effort": "LOW"
            })
        
        # Priority 2: High findings
        for finding in classified_findings.get("high", [])[:10]:  # Top 10 high findings
            priority_counter += 1
            priority_queue.append({
                "priority": 2,
                "priority_score": 75,
                "finding": finding,
                "reason": "High-risk security issue",
                "action": "Schedule remediation within 48 hours",
                "estimated_time": "2-4 hours",
                "impact": "MEDIUM",
                "effort": "MEDIUM"
            })
        
        # Priority 3: Medium findings with quick fixes
        quick_wins = self._identify_quick_wins(classified_findings.get("medium", []))
        for finding in quick_wins[:5]:  # Top 5 quick wins
            priority_counter += 1
            priority_queue.append({
                "priority": 3,
                "priority_score": 50,
                "finding": finding,
                "reason": "Quick security improvement",
                "action": "Implement during next maintenance window",
                "estimated_time": "30-60 minutes",
                "impact": "LOW",
                "effort": "LOW"
            })
        
        # Sort by priority score
        priority_queue.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return priority_queue
    
    def _identify_quick_wins(self, findings: List[Dict]) -> List[Dict]:
        """Identify findings that can be fixed quickly."""
        quick_wins = []
        
        for finding in findings:
            # Check for easy fixes
            if any(keyword in finding.get("category", "").lower() 
                   for keyword in ["configuration", "permission", "key", "setting"]):
                if "complex" not in finding.get("description", "").lower():
                    quick_wins.append(finding)
        
        return quick_wins
    
    def _generate_recommendations(
        self,
        priority_queue: List[Dict],
        assessment: Dict,
        threat_context: Dict
    ) -> Dict[str, List[str]]:
        """Generate categorized recommendations."""
        recommendations = {
            "immediate": [],
            "short_term": [],
            "long_term": []
        }
        
        # Immediate actions (Priority 1)
        for item in priority_queue:
            if item["priority"] == 1:
                finding = item["finding"]
                if finding.get("type") == "advisory":
                    recommendations["immediate"].append(
                        f"Review and apply: {finding.get('description')}"
                    )
                else:
                    recommendations["immediate"].append(
                        f"Fix {finding.get('severity', 'critical')} issue: {finding.get('description', 'Security vulnerability')}"
                    )
        
        # Add compliance-driven recommendations if needed
        compliance = assessment.get("compliance", {})
        if compliance:
            for framework, status in compliance.items():
                if framework != "overall" and not status.get("compliant"):
                    issues = status.get("issues", [])
                    if issues:
                        recommendations["short_term"].append(
                            f"Address {len(issues)} {framework.upper()} compliance issues"
                        )
        
        # Short-term actions (Priority 2)
        high_count = len([i for i in priority_queue if i["priority"] == 2])
        if high_count > 0:
            recommendations["short_term"].append(
                f"Remediate {high_count} high-priority security findings"
            )
        
        # Long-term improvements
        risk_metrics = assessment.get("risk_metrics", {})
        if risk_metrics.get("finding_counts", {}).get("medium", 0) > 10:
            recommendations["long_term"].append(
                "Implement automated security scanning and remediation"
            )
        
        if assessment.get("summary", {}).get("iam", {}).get("service_accounts", 0) > 20:
            recommendations["long_term"].append(
                "Audit and reduce service account proliferation"
            )
        
        # Always include best practices
        recommendations["long_term"].extend([
            "Implement continuous compliance monitoring",
            "Establish security baseline and drift detection",
            "Create incident response playbooks"
        ])
        
        return recommendations
    
    def _create_remediation_plan(self, priority_queue: List[Dict]) -> List[Dict]:
        """Create step-by-step remediation plan."""
        plan = []
        
        for i, item in enumerate(priority_queue[:20], 1):  # Top 20 items
            finding = item["finding"]
            
            step = {
                "step": i,
                "priority": item["priority"],
                "priority_score": item["priority_score"],
                "description": finding.get("description", "Address security issue"),
                "action": item["action"],
                "estimated_time": item["estimated_time"],
                "impact": item["impact"],
                "effort": item["effort"],
                "verification": "Run Review Agent after completion",
                "rollback_plan": "Revert changes if issues occur"
            }
            
            # Add specific remediation steps based on finding type
            if "iam" in finding.get("category", "").lower():
                step["specific_actions"] = [
                    "Review current permissions",
                    "Apply principle of least privilege",
                    "Remove unnecessary permissions",
                    "Document changes"
                ]
            elif "api" in finding.get("category", "").lower():
                step["specific_actions"] = [
                    "Identify API key usage",
                    "Add appropriate restrictions",
                    "Rotate if compromised",
                    "Monitor usage"
                ]
            else:
                step["specific_actions"] = [
                    "Identify affected resources",
                    "Apply security fix",
                    "Test functionality",
                    "Monitor for recurrence"
                ]
            
            plan.append(step)
        
        return plan
    
    def _estimate_effort(self, remediation_plan: List[Dict]) -> Dict[str, Any]:
        """Estimate total effort required."""
        total_hours = 0
        by_priority = {1: 0, 2: 0, 3: 0}
        
        for step in remediation_plan:
            # Parse estimated time
            time_str = step.get("estimated_time", "1 hour")
            if "minute" in time_str:
                hours = 0.5  # Approximate
            elif "-" in time_str:
                # Take average of range
                parts = time_str.split("-")
                try:
                    min_hours = float(parts[0].strip().split()[0])
                    max_hours = float(parts[1].strip().split()[0])
                    hours = (min_hours + max_hours) / 2
                except:
                    hours = 2  # Default
            else:
                hours = 2  # Default
            
            total_hours += hours
            by_priority[step["priority"]] = by_priority.get(step["priority"], 0) + hours
        
        return {
            "total_hours": round(total_hours, 1),
            "total_days": round(total_hours / 8, 1),  # Assuming 8-hour workday
            "by_priority": {
                "immediate": round(by_priority.get(1, 0), 1),
                "short_term": round(by_priority.get(2, 0), 1),
                "long_term": round(by_priority.get(3, 0), 1)
            },
            "team_size_recommendation": "2-3 engineers" if total_hours > 40 else "1-2 engineers"
        }
    
    def _calculate_risk_reduction(self, assessment: Dict, remediation_plan: List[Dict]) -> Dict[str, Any]:
        """Calculate expected risk reduction from remediation."""
        current_risk_score = assessment.get("risk_metrics", {}).get("risk_score", 0)
        current_risk_level = assessment.get("risk_level", "UNKNOWN")
        
        # Estimate risk reduction based on addressed findings
        addressed_critical = len([s for s in remediation_plan if s["priority"] == 1])
        addressed_high = len([s for s in remediation_plan if s["priority"] == 2])
        addressed_medium = len([s for s in remediation_plan if s["priority"] == 3])
        
        # Calculate reduction (simplified model)
        risk_reduction = (
            addressed_critical * 10 +
            addressed_high * 5 +
            addressed_medium * 2
        )
        
        new_risk_score = max(0, current_risk_score - risk_reduction)
        
        # Determine new risk level
        if new_risk_score >= 50:
            new_risk_level = "HIGH"
        elif new_risk_score >= 30:
            new_risk_level = "MEDIUM"
        elif new_risk_score >= 15:
            new_risk_level = "LOW"
        else:
            new_risk_level = "MINIMAL"
        
        improvement_percentage = min(100, (risk_reduction / max(current_risk_score, 1)) * 100)
        
        return {
            "current_risk_score": current_risk_score,
            "projected_risk_score": new_risk_score,
            "risk_reduction_points": risk_reduction,
            "current_risk_level": current_risk_level,
            "projected_risk_level": new_risk_level,
            "improvement_percentage": round(improvement_percentage, 1),
            "findings_addressed": addressed_critical + addressed_high + addressed_medium
        }