"""
Decision Phase Chat View - Prioritization and recommendation interface.

This module implements the Decision phase of RADAR, focusing on
prioritizing issues and generating actionable recommendations.
"""

import streamlit as st
import logging
from typing import Dict, Any, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from components.shared.chat_streaming_base import StreamingChatBase
from components.radar.radar_state_manager import radar_state_manager, RADARPhase
from unified_api_client import api_client

logger = logging.getLogger(__name__)


class DecisionChatView(StreamingChatBase):
    """
    Decision phase chat interface.
    
    This phase focuses on:
    - Risk-based prioritization
    - Actionable recommendation generation
    - Impact vs effort analysis
    - Resource allocation planning
    """
    
    def __init__(self):
        """Initialize Decision chat view."""
        super().__init__(
            phase_name="Decision",
            phase_icon="🎯",
            phase_description="Prioritize security issues and generate actionable recommendations"
        )
    
    def render_quick_actions(self):
        """Render Decision-specific quick actions."""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 Prioritize Issues", use_container_width=True):
                self.prioritize_issues()
        
        with col2:
            if st.button("💡 Generate Recommendations", use_container_width=True):
                self.generate_recommendations()
        
        with col3:
            if st.button("📊 Impact Analysis", use_container_width=True):
                self.analyze_impact()
        
        with col4:
            if st.button("📋 Create Action Plan", use_container_width=True):
                self.create_action_plan()
        
        # Standard actions
        super().render_quick_actions()
    
    def render_context_panel(self):
        """Render Decision-specific context panel."""
        context = radar_state_manager.get_context()
        
        if context:
            # Show decision statistics
            st.markdown("### 🎯 Decision Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_issues = self.get_phase_context().get("total_issues", 0)
                st.metric("Total Issues", total_issues)
            
            with col2:
                critical_items = self.get_phase_context().get("critical_priority_count", 0)
                st.metric("Critical Priority", critical_items)
            
            with col3:
                quick_wins = self.get_phase_context().get("quick_wins_count", 0)
                st.metric("Quick Wins", quick_wins)
            
            with col4:
                estimated_effort = self.get_phase_context().get("total_effort_days", 0)
                st.metric("Est. Effort", f"{estimated_effort}d")
            
            # Show assessment results if available
            assessment_result = context.get_phase_result(RADARPhase.ASSESSMENT)
            if assessment_result and assessment_result.status == "completed":
                st.markdown("### 🛡️ Assessment Summary")
                if assessment_result.results:
                    findings = assessment_result.results.get("total_findings", 0)
                    risk_level = assessment_result.results.get("overall_risk_level", "Unknown")
                    st.info(f"Prioritizing {findings} findings with {risk_level} overall risk")
            
            # Priority breakdown
            priority_breakdown = self.get_phase_context().get("priority_breakdown", {})
            if priority_breakdown:
                st.markdown("### 📊 Priority Distribution")
                for priority, count in priority_breakdown.items():
                    color = self._get_priority_color(priority)
                    st.write(f"{color} **{priority}:** {count} items")
            
            # Recommendation categories
            recommendations = self.get_phase_context().get("recommendation_categories", {})
            if recommendations:
                st.markdown("### 💡 Recommendation Categories")
                for category, count in recommendations.items():
                    st.write(f"- **{category}:** {count} recommendations")
        else:
            st.info("No decision data yet. Start by prioritizing issues from the assessment phase.")
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate Decision-specific response.
        
        Args:
            user_input: User's query
            
        Returns:
            Generated response
        """
        try:
            # Prepare context including Assessment results
            context = radar_state_manager.get_context()
            project_id = context.project_id if context else st.session_state.get('selected_project', 'default')
            
            # Get previous phase results
            phase_context = self.get_phase_context()
            if context:
                # Include Recognition results
                recognition_result = context.get_phase_result(RADARPhase.RECOGNITION)
                if recognition_result and recognition_result.status == "completed":
                    phase_context["recognition_results"] = recognition_result.results
                
                # Include Assessment results
                assessment_result = context.get_phase_result(RADARPhase.ASSESSMENT)
                if assessment_result and assessment_result.status == "completed":
                    phase_context["assessment_results"] = assessment_result.results
            
            # Call RADAR backend
            response = api_client.radar_chat({
                "query": user_input,
                "phase": "decision",
                "project_id": project_id,
                "context": phase_context
            })
            
            if response.get("success"):
                # Update phase context with results
                if "decision_results" in response:
                    self.update_phase_context("latest_decision", response["decision_results"])
                    self._parse_decision_results(response["decision_results"])
                
                # Update RADAR state
                if context and radar_state_manager.can_execute_phase(RADARPhase.DECISION):
                    if context.phases[RADARPhase.DECISION].status != "completed":
                        radar_state_manager.start_phase(RADARPhase.DECISION)
                        radar_state_manager.complete_phase(
                            RADARPhase.DECISION,
                            response.get("decision_results", {})
                        )
                
                return response.get("response", "Decision analysis completed.")
            else:
                return f"Error: {response.get('error', 'Failed to process decision query')}"
                
        except Exception as e:
            logger.error(f"Decision response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def prioritize_issues(self):
        """Prioritize security issues based on risk and impact."""
        with st.spinner("Prioritizing security issues..."):
            try:
                # Get assessment results
                context = radar_state_manager.get_context()
                if not context:
                    st.warning("No context available. Please complete Recognition and Assessment phases first.")
                    return
                
                assessment_result = context.get_phase_result(RADARPhase.ASSESSMENT)
                if not assessment_result or assessment_result.status != "completed":
                    st.warning("Assessment phase not completed. Please complete Assessment first.")
                    return
                
                # Extract findings from assessment
                findings = assessment_result.results.get("latest_scan", [])
                if not findings:
                    findings = self._generate_sample_findings(assessment_result.results)
                
                # Prioritize findings
                prioritized = self._prioritize_findings(findings)
                
                # Update context
                self.update_phase_context("prioritized_issues", prioritized)
                self.update_phase_context("total_issues", len(prioritized))
                
                # Count by priority
                priority_breakdown = {}
                for issue in prioritized:
                    priority = issue.get("priority", "Unknown")
                    priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
                
                self.update_phase_context("priority_breakdown", priority_breakdown)
                self.update_phase_context("critical_priority_count", priority_breakdown.get("Critical", 0))
                
                # Add message
                self.add_message(
                    "system",
                    f"🎯 Prioritization completed: {len(prioritized)} issues ranked"
                )
                
                st.success(f"Prioritized {len(prioritized)} issues")
                
                # Display top priorities
                st.markdown("### Top Priority Issues:")
                for issue in prioritized[:5]:
                    priority = issue.get("priority", "Unknown")
                    color = self._get_priority_color(priority)
                    st.write(f"{color} **{priority}:** {issue.get('title', 'Unknown issue')}")
                    st.caption(f"Impact: {issue.get('impact', 'Unknown')} | Effort: {issue.get('effort', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Issue prioritization failed: {e}")
                st.error(f"Failed to prioritize issues: {str(e)}")
        
        st.rerun()
    
    def generate_recommendations(self):
        """Generate actionable recommendations."""
        with st.spinner("Generating recommendations..."):
            try:
                # Get prioritized issues
                prioritized = self.get_phase_context().get("prioritized_issues", [])
                
                if not prioritized:
                    st.warning("No prioritized issues found. Please prioritize issues first.")
                    return
                
                # Generate recommendations
                recommendations = self._generate_recommendations_for_issues(prioritized)
                
                # Update context
                self.update_phase_context("recommendations", recommendations)
                self.update_phase_context("total_recommendations", len(recommendations))
                
                # Categorize recommendations
                categories = {}
                for rec in recommendations:
                    category = rec.get("category", "General")
                    categories[category] = categories.get(category, 0) + 1
                
                self.update_phase_context("recommendation_categories", categories)
                
                # Identify quick wins
                quick_wins = [r for r in recommendations if r.get("effort") == "Low" and r.get("impact") == "High"]
                self.update_phase_context("quick_wins", quick_wins)
                self.update_phase_context("quick_wins_count", len(quick_wins))
                
                # Add message
                self.add_message(
                    "system",
                    f"💡 Generated {len(recommendations)} recommendations ({len(quick_wins)} quick wins)"
                )
                
                st.success(f"Generated {len(recommendations)} recommendations")
                
                # Display recommendations by category
                st.markdown("### Recommendations by Category:")
                for category, recs in self._group_recommendations_by_category(recommendations).items():
                    st.markdown(f"**{category}** ({len(recs)} items)")
                    for rec in recs[:3]:  # Show top 3 per category
                        st.write(f"- {rec.get('title', 'Unknown')}")
                        if rec.get('effort') == "Low" and rec.get('impact') == "High":
                            st.caption("⚡ Quick Win")
                            
            except Exception as e:
                logger.error(f"Recommendation generation failed: {e}")
                st.error(f"Failed to generate recommendations: {str(e)}")
        
        st.rerun()
    
    def analyze_impact(self):
        """Analyze impact vs effort for prioritized issues."""
        with st.spinner("Analyzing impact vs effort..."):
            try:
                # Get recommendations
                recommendations = self.get_phase_context().get("recommendations", [])
                
                if not recommendations:
                    st.warning("No recommendations available. Please generate recommendations first.")
                    return
                
                # Analyze impact vs effort
                analysis = self._analyze_impact_effort(recommendations)
                
                # Update context
                self.update_phase_context("impact_analysis", analysis)
                self.update_phase_context("total_effort_days", analysis.get("total_effort_days", 0))
                self.update_phase_context("total_impact_score", analysis.get("total_impact_score", 0))
                
                # Add message
                self.add_message(
                    "system",
                    f"📊 Impact analysis completed: {analysis.get('total_effort_days', 0)} days total effort"
                )
                
                st.success("Impact analysis completed")
                
                # Display impact matrix
                st.markdown("### Impact vs Effort Matrix:")
                
                matrix = analysis.get("matrix", {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Quick Wins (Low Effort, High Impact):**")
                    for item in matrix.get("quick_wins", [])[:5]:
                        st.write(f"- {item.get('title', 'Unknown')}")
                    
                    st.markdown("**📈 Major Projects (High Effort, High Impact):**")
                    for item in matrix.get("major_projects", [])[:5]:
                        st.write(f"- {item.get('title', 'Unknown')}")
                
                with col2:
                    st.markdown("**🔄 Fill-ins (Low Effort, Low Impact):**")
                    for item in matrix.get("fill_ins", [])[:5]:
                        st.write(f"- {item.get('title', 'Unknown')}")
                    
                    st.markdown("**❓ Questionable (High Effort, Low Impact):**")
                    for item in matrix.get("questionable", [])[:5]:
                        st.write(f"- {item.get('title', 'Unknown')}")
                
                # Show summary metrics
                st.markdown("### Summary Metrics:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Effort", f"{analysis.get('total_effort_days', 0)} days")
                with col2:
                    st.metric("Avg Impact Score", f"{analysis.get('avg_impact_score', 0):.1f}/10")
                with col3:
                    st.metric("ROI Score", f"{analysis.get('roi_score', 0):.1f}")
                    
            except Exception as e:
                logger.error(f"Impact analysis failed: {e}")
                st.error(f"Failed to analyze impact: {str(e)}")
        
        st.rerun()
    
    def create_action_plan(self):
        """Create a phased action plan."""
        with st.spinner("Creating action plan..."):
            try:
                # Get recommendations and analysis
                recommendations = self.get_phase_context().get("recommendations", [])
                impact_analysis = self.get_phase_context().get("impact_analysis", {})
                
                if not recommendations:
                    st.warning("No recommendations available. Please generate recommendations first.")
                    return
                
                # Create phased action plan
                action_plan = self._create_phased_action_plan(recommendations, impact_analysis)
                
                # Update context
                self.update_phase_context("action_plan", action_plan)
                self.update_phase_context("total_phases", len(action_plan.get("phases", [])))
                
                # Add message
                self.add_message(
                    "system",
                    f"📋 Action plan created with {len(action_plan.get('phases', []))} phases"
                )
                
                st.success("Action plan created")
                
                # Display action plan
                st.markdown("### Phased Action Plan:")
                
                for phase in action_plan.get("phases", []):
                    with st.expander(f"**Phase {phase.get('number', 0)}: {phase.get('name', 'Unknown')}**"):
                        st.write(f"**Duration:** {phase.get('duration', 'Unknown')}")
                        st.write(f"**Focus:** {phase.get('focus', 'Unknown')}")
                        st.write(f"**Expected Impact:** {phase.get('expected_impact', 'Unknown')}")
                        
                        st.markdown("**Actions:**")
                        for action in phase.get("actions", []):
                            st.write(f"- {action.get('title', 'Unknown')}")
                            st.caption(f"  Priority: {action.get('priority', 'Unknown')} | Effort: {action.get('effort', 'Unknown')}")
                
                # Show timeline
                st.markdown("### Timeline Overview:")
                timeline = action_plan.get("timeline", {})
                st.write(f"- **Start Date:** Immediate")
                st.write(f"- **Total Duration:** {timeline.get('total_duration', 'Unknown')}")
                st.write(f"- **Expected Completion:** {timeline.get('completion', 'Unknown')}")
                st.write(f"- **Resource Requirements:** {timeline.get('resources', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Action plan creation failed: {e}")
                st.error(f"Failed to create action plan: {str(e)}")
        
        st.rerun()
    
    def _parse_decision_results(self, results: Any):
        """Parse and store decision results in context."""
        if isinstance(results, dict):
            # Extract recommendations
            if "recommendations" in results:
                self.update_phase_context("recommendations", results["recommendations"])
                self.update_phase_context("total_recommendations", len(results["recommendations"]))
            
            # Extract priorities
            if "priorities" in results:
                self.update_phase_context("prioritized_issues", results["priorities"])
                self.update_phase_context("total_issues", len(results["priorities"]))
    
    def _generate_sample_findings(self, assessment_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sample findings from assessment results."""
        findings = []
        
        # Add critical findings
        for i in range(assessment_results.get("critical_findings", 0)):
            findings.append({
                "severity": "CRITICAL",
                "title": f"Critical Security Issue {i+1}",
                "category": "Security",
                "description": "Critical vulnerability requiring immediate attention"
            })
        
        # Add high findings
        for i in range(assessment_results.get("high_findings", 0)):
            findings.append({
                "severity": "HIGH",
                "title": f"High Priority Issue {i+1}",
                "category": "Configuration",
                "description": "High severity issue that should be addressed soon"
            })
        
        return findings
    
    def _prioritize_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize findings based on severity, impact, and effort."""
        prioritized = []
        
        for finding in findings:
            # Calculate priority score
            severity = finding.get("severity", "LOW")
            severity_scores = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            score = severity_scores.get(severity, 1)
            
            # Determine priority level
            if score >= 4:
                priority = "Critical"
                impact = "High"
                effort = "Medium"
            elif score >= 3:
                priority = "High"
                impact = "High"
                effort = "Medium"
            elif score >= 2:
                priority = "Medium"
                impact = "Medium"
                effort = "Low"
            else:
                priority = "Low"
                impact = "Low"
                effort = "Low"
            
            prioritized.append({
                "title": finding.get("title", "Unknown"),
                "description": finding.get("description", ""),
                "category": finding.get("category", "General"),
                "severity": severity,
                "priority": priority,
                "impact": impact,
                "effort": effort,
                "score": score
            })
        
        # Sort by score (highest first)
        prioritized.sort(key=lambda x: x["score"], reverse=True)
        return prioritized
    
    def _generate_recommendations_for_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate specific recommendations for prioritized issues."""
        recommendations = []
        
        for issue in issues:
            # Generate recommendation based on issue type
            category = issue.get("category", "General")
            priority = issue.get("priority", "Medium")
            
            rec = {
                "title": f"Fix: {issue.get('title', 'Unknown')}",
                "description": f"Remediate {issue.get('description', 'security issue')}",
                "category": category,
                "priority": priority,
                "impact": issue.get("impact", "Medium"),
                "effort": issue.get("effort", "Medium"),
                "issue_ref": issue.get("title", ""),
                "steps": [
                    "Analyze the root cause",
                    "Implement the fix",
                    "Test the remediation",
                    "Deploy to production",
                    "Verify the fix"
                ]
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _group_recommendations_by_category(self, recommendations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group recommendations by category."""
        grouped = {}
        for rec in recommendations:
            category = rec.get("category", "General")
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(rec)
        return grouped
    
    def _analyze_impact_effort(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze impact vs effort for recommendations."""
        analysis = {
            "matrix": {
                "quick_wins": [],
                "major_projects": [],
                "fill_ins": [],
                "questionable": []
            },
            "total_effort_days": 0,
            "total_impact_score": 0,
            "avg_impact_score": 0,
            "roi_score": 0
        }
        
        effort_days = {"Low": 2, "Medium": 5, "High": 10}
        impact_scores = {"Low": 3, "Medium": 6, "High": 9}
        
        for rec in recommendations:
            effort = rec.get("effort", "Medium")
            impact = rec.get("impact", "Medium")
            
            # Calculate days and score
            days = effort_days.get(effort, 5)
            score = impact_scores.get(impact, 6)
            
            analysis["total_effort_days"] += days
            analysis["total_impact_score"] += score
            
            # Categorize into matrix
            if effort == "Low" and impact == "High":
                analysis["matrix"]["quick_wins"].append(rec)
            elif effort == "High" and impact == "High":
                analysis["matrix"]["major_projects"].append(rec)
            elif effort == "Low" and impact == "Low":
                analysis["matrix"]["fill_ins"].append(rec)
            elif effort == "High" and impact == "Low":
                analysis["matrix"]["questionable"].append(rec)
        
        # Calculate averages
        if recommendations:
            analysis["avg_impact_score"] = analysis["total_impact_score"] / len(recommendations)
            if analysis["total_effort_days"] > 0:
                analysis["roi_score"] = analysis["total_impact_score"] / analysis["total_effort_days"]
        
        return analysis
    
    def _create_phased_action_plan(self, recommendations: List[Dict[str, Any]], impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a phased action plan from recommendations."""
        action_plan = {
            "phases": [],
            "timeline": {}
        }
        
        # Get categorized recommendations
        matrix = impact_analysis.get("matrix", {})
        
        # Phase 1: Quick Wins (Week 1-2)
        phase1 = {
            "number": 1,
            "name": "Quick Wins",
            "duration": "2 weeks",
            "focus": "High-impact, low-effort fixes",
            "expected_impact": "Immediate risk reduction",
            "actions": matrix.get("quick_wins", [])[:10]
        }
        
        # Phase 2: Critical Fixes (Week 3-4)
        critical = [r for r in recommendations if r.get("priority") == "Critical"]
        phase2 = {
            "number": 2,
            "name": "Critical Remediations",
            "duration": "2 weeks",
            "focus": "Address critical vulnerabilities",
            "expected_impact": "Major risk mitigation",
            "actions": critical[:10]
        }
        
        # Phase 3: Major Projects (Month 2-3)
        phase3 = {
            "number": 3,
            "name": "Strategic Improvements",
            "duration": "8 weeks",
            "focus": "Long-term security enhancements",
            "expected_impact": "Comprehensive security posture improvement",
            "actions": matrix.get("major_projects", [])[:10]
        }
        
        action_plan["phases"] = [phase1, phase2, phase3]
        
        # Calculate timeline
        total_days = impact_analysis.get("total_effort_days", 0)
        action_plan["timeline"] = {
            "total_duration": f"{total_days} days",
            "completion": f"~{total_days // 20} months",
            "resources": "2-3 engineers recommended"
        }
        
        return action_plan
    
    def _get_priority_color(self, priority: str) -> str:
        """Get color emoji for priority level."""
        colors = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢",
            "Unknown": "⚪"
        }
        return colors.get(priority, "⚪")


def render_decision_chat_view():
    """Render the Decision phase chat view."""
    chat_view = DecisionChatView()
    
    # Start phase if needed
    context = radar_state_manager.get_context()
    if context and radar_state_manager.can_execute_phase(RADARPhase.DECISION):
        if context.phases[RADARPhase.DECISION].status == "pending":
            radar_state_manager.start_phase(RADARPhase.DECISION)
    
    # Render interface
    chat_view.render_chat_interface()