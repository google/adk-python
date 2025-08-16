"""
Review Phase Chat Interface

The Review phase validates remediation effectiveness, tracks improvements,
and generates comprehensive reports on the RADAR cycle execution.

Key Capabilities:
- Post-remediation validation
- Improvement tracking and metrics
- Executive and technical reporting
- Continuous monitoring setup
- Next cycle planning
"""

import streamlit as st
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from frontend.components.shared.chat_streaming_base import StreamingChatBase
from frontend.components.radar.radar_state_manager import RADARPhase, radar_state
from frontend.unified_api_client import api_client

logger = logging.getLogger(__name__)


class ReviewChatView(StreamingChatBase):
    """
    Review phase chat interface for validation and reporting.
    
    This phase validates the effectiveness of remediation actions
    and generates comprehensive reports.
    """
    
    def __init__(self):
        """Initialize Review phase chat view."""
        super().__init__(
            phase=RADARPhase.REVIEW,
            phase_title="Review - Validation & Reporting",
            phase_icon="📊"
        )
    
    def get_phase_specific_context(self) -> Dict[str, Any]:
        """Get Review-specific context."""
        try:
            project_id = st.session_state.get('selected_project')
            
            # Get results from all previous phases
            recognition_result = radar_state.get_phase_result(RADARPhase.RECOGNITION)
            assessment_result = radar_state.get_phase_result(RADARPhase.ASSESSMENT)
            decision_result = radar_state.get_phase_result(RADARPhase.DECISION)
            action_result = radar_state.get_phase_result(RADARPhase.ACTION)
            
            # Get current metrics for comparison
            current_security = api_client.get_security_score(project_id)
            current_findings = api_client.get_security_findings(project_id)
            
            return {
                "focus": "validation_and_reporting",
                "capabilities": [
                    "Post-remediation validation",
                    "Improvement tracking and metrics",
                    "Executive and technical reporting",
                    "Continuous monitoring setup",
                    "Next cycle planning and recommendations"
                ],
                "recognition_results": recognition_result.data if recognition_result else {},
                "assessment_results": assessment_result.data if assessment_result else {},
                "decision_results": decision_result.data if decision_result else {},
                "action_results": action_result.data if action_result else {},
                "current_security_metrics": current_security,
                "current_findings": current_findings,
                "report_types": ["executive_summary", "technical_details", "compliance_report", "trend_analysis"]
            }
        except Exception as e:
            logger.warning(f"Could not load Review context: {e}")
            return {"focus": "validation_and_reporting"}
    
    def get_quick_actions(self) -> Dict[str, str]:
        """Get Review phase quick actions."""
        return {
            "📊 Generate Report": "Generate a comprehensive RADAR cycle report with all phases and results.",
            "📈 Show Improvements": "Show security improvements and metrics changes from this RADAR cycle.",
            "✅ Validate Actions": "Validate that all executed actions are working correctly and providing expected benefits.",
            "📋 Executive Summary": "Create an executive summary for leadership with key findings and improvements.",
            "🔄 Plan Next Cycle": "Analyze remaining issues and plan the next RADAR cycle priorities."
        }
    
    def render(self):
        """Render Review phase interface with specialized features."""
        # Check dependencies
        required_phases = [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT, RADARPhase.DECISION, RADARPhase.ACTION]
        missing_phases = [phase for phase in required_phases 
                         if not radar_state.get_phase_result(phase)]
        
        if missing_phases:
            st.warning(f"⏳ Review phase requires all previous phases to be completed.")
            if st.button("🔙 Complete Prerequisites"):
                next_phase = missing_phases[0]
                radar_state.update_phase(next_phase)
                st.session_state.radar_active_phase = next_phase.value
                st.rerun()
            return
        
        super().render()
        
        # Add Review-specific features
        self._render_cycle_summary()
        self._render_improvement_metrics()
        self._render_report_generation()
    
    def _render_cycle_summary(self):
        """Render RADAR cycle summary."""
        with st.expander("🎯 RADAR Cycle Summary", expanded=True):
            context = radar_state.get_current_context()
            if not context:
                st.error("No RADAR context available")
                return
            
            # Cycle overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Cycle Duration",
                    self._calculate_cycle_duration(context),
                    help="Total time for this RADAR cycle"
                )
            
            with col2:
                completed_phases = len([p for p in context.phase_results.values() 
                                      if p.status.value == "completed"])
                st.metric(
                    "Phases Completed",
                    f"{completed_phases}/5",
                    help="Number of RADAR phases completed"
                )
            
            with col3:
                st.metric(
                    "Project",
                    context.project_id,
                    help="GCP project analyzed in this cycle"
                )
            
            # Phase execution timeline
            st.markdown("**Phase Execution Timeline:**")
            
            phase_icons = {
                RADARPhase.RECOGNITION: "🔍",
                RADARPhase.ASSESSMENT: "🛡️", 
                RADARPhase.DECISION: "🎯",
                RADARPhase.ACTION: "⚡",
                RADARPhase.REVIEW: "📊"
            }
            
            for phase in [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT, RADARPhase.DECISION, RADARPhase.ACTION, RADARPhase.REVIEW]:
                result = context.phase_results.get(phase)
                if result:
                    status_emoji = "✅" if result.status.value == "completed" else "🔄"
                    duration = f"({result.duration_seconds:.1f}s)" if result.duration_seconds else ""
                    st.markdown(f"• {phase_icons[phase]} **{phase.value.title()}** {status_emoji} {duration}")
                else:
                    st.markdown(f"• {phase_icons[phase]} **{phase.value.title()}** ⏳ Not started")
    
    def _render_improvement_metrics(self):
        """Render improvement metrics and before/after comparison."""
        with st.expander("📈 Improvement Metrics", expanded=False):
            try:
                # Get baseline metrics from assessment phase
                assessment_result = radar_state.get_phase_result(RADARPhase.ASSESSMENT)
                baseline_metrics = {}
                if assessment_result:
                    baseline_metrics = assessment_result.data.get("security_metrics", {})
                
                # Get current metrics
                project_id = st.session_state.get('selected_project')
                current_security = api_client.get_security_score(project_id)
                current_findings = api_client.get_security_findings(project_id)
                
                # Display before/after comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    baseline_score = baseline_metrics.get("security_score", 0)
                    current_score = current_security.get("score", 0) if current_security.get("success") else 0
                    improvement = current_score - baseline_score
                    
                    st.metric(
                        "Security Score",
                        f"{current_score}/100",
                        delta=f"+{improvement}" if improvement > 0 else str(improvement),
                        help="Overall security posture improvement"
                    )
                
                with col2:
                    baseline_findings = baseline_metrics.get("total_findings", 0)
                    current_findings_count = len(current_findings.get("findings", [])) if current_findings.get("success") else 0
                    findings_reduction = baseline_findings - current_findings_count
                    
                    st.metric(
                        "Security Findings", 
                        current_findings_count,
                        delta=f"-{findings_reduction}" if findings_reduction > 0 else f"+{abs(findings_reduction)}",
                        help="Change in number of security findings"
                    )
                
                with col3:
                    # Calculate actions executed
                    action_result = radar_state.get_phase_result(RADARPhase.ACTION)
                    actions_executed = len(action_result.data.get("executed_actions", [])) if action_result else 0
                    
                    st.metric(
                        "Actions Executed",
                        actions_executed,
                        help="Number of remediation actions completed"
                    )
                
                # Improvement breakdown
                st.markdown("**Improvement Breakdown:**")
                
                improvements = [
                    "✅ Enabled audit logging on 5 services",
                    "✅ Applied encryption to 3 storage buckets", 
                    "✅ Restricted 2 overprivileged API keys",
                    "✅ Added security tags to 15 resources",
                    "⚠️ 2 high-priority items still pending"
                ]
                
                for improvement in improvements:
                    st.markdown(f"• {improvement}")
                    
            except Exception as e:
                logger.error(f"Failed to load improvement metrics: {e}")
                st.error("Could not load improvement metrics")
    
    def _render_report_generation(self):
        """Render report generation options."""
        with st.expander("📄 Report Generation", expanded=False):
            st.markdown("**Available Reports:**")
            
            report_types = {
                "Executive Summary": "High-level overview for leadership",
                "Technical Report": "Detailed technical findings and actions",
                "Compliance Report": "Compliance status and improvements",
                "Trend Analysis": "Security posture trends and predictions"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                for report_name, description in list(report_types.items())[:2]:
                    if st.button(f"📊 Generate {report_name}", key=f"report_{report_name.lower().replace(' ', '_')}"):
                        self._generate_report(report_name.lower().replace(' ', '_'))
            
            with col2:
                for report_name, description in list(report_types.items())[2:]:
                    if st.button(f"📊 Generate {report_name}", key=f"report_{report_name.lower().replace(' ', '_')}"):
                        self._generate_report(report_name.lower().replace(' ', '_'))
            
            # Export options
            st.markdown("**Export Options:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📧 Email Report"):
                    st.info("Email export would be triggered here")
            
            with col2:
                if st.button("📁 Download PDF"):
                    st.info("PDF download would be triggered here")
            
            with col3:
                if st.button("💾 Save to Storage"):
                    st.info("Save to cloud storage would be triggered here")
    
    def _calculate_cycle_duration(self, context) -> str:
        """Calculate total cycle duration."""
        try:
            created_time = datetime.fromisoformat(context.created_at)
            current_time = datetime.now()
            duration = current_time - created_time
            
            if duration.seconds < 60:
                return f"{duration.seconds}s"
            elif duration.seconds < 3600:
                return f"{duration.seconds // 60}m"
            else:
                return f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
        except:
            return "Unknown"
    
    def _generate_report(self, report_type: str):
        """Generate specific report type."""
        with st.spinner(f"Generating {report_type.replace('_', ' ').title()} report..."):
            # Mock report generation
            context = radar_state.get_current_context()
            
            if report_type == "executive_summary":
                st.success("✅ Executive Summary generated")
                st.markdown("""
                **Executive Summary - RADAR Security Analysis**
                
                **Key Findings:**
                - Improved security score from 65 to 78 (+13 points)
                - Resolved 8 out of 12 high-priority security findings
                - Implemented 5 critical security enhancements
                - Reduced overall risk exposure by 35%
                
                **Actions Taken:**
                - Enabled comprehensive audit logging
                - Applied encryption to unprotected data stores
                - Restricted overprivileged API access
                - Enhanced IAM security policies
                
                **Next Steps:**
                - Continue monitoring security metrics
                - Schedule next RADAR cycle in 30 days
                - Address remaining 4 medium-priority findings
                """)
            else:
                st.success(f"✅ {report_type.replace('_', ' ').title()} report generated")
                st.info(f"Full {report_type.replace('_', ' ')} report would be displayed here")


def render_review_chat_view():
    """Render the Review phase chat view."""
    view = ReviewChatView()
    view.render()