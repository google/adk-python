"""
Assessment Phase Chat View - Security evaluation and compliance interface.

This module implements the Assessment phase of RADAR, focusing on
evaluating security posture and compliance status.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from components.shared.chat_streaming_base import StreamingChatBase
from components.radar.radar_state_manager import radar_state_manager, RADARPhase
from unified_api_client import api_client

logger = logging.getLogger(__name__)


class AssessmentChatView(StreamingChatBase):
    """
    Assessment phase chat interface.
    
    This phase focuses on:
    - Security vulnerability scanning
    - Compliance framework evaluation
    - Risk scoring and prioritization
    - IAM permission analysis
    """
    
    def __init__(self):
        """Initialize Assessment chat view."""
        super().__init__(
            phase_name="Assessment",
            phase_icon="🛡️",
            phase_description="Evaluate security posture and identify vulnerabilities in your environment"
        )
    
    def render_quick_actions(self):
        """Render Assessment-specific quick actions."""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Security Scan", use_container_width=True):
                self.run_security_scan()
        
        with col2:
            if st.button("📋 Compliance Check", use_container_width=True):
                self.check_compliance()
        
        with col3:
            if st.button("🔐 IAM Analysis", use_container_width=True):
                self.analyze_iam()
        
        with col4:
            if st.button("⚠️ Risk Assessment", use_container_width=True):
                self.assess_risks()
        
        # Standard actions
        super().render_quick_actions()
    
    def render_context_panel(self):
        """Render Assessment-specific context panel."""
        context = radar_state_manager.get_context()
        
        if context:
            # Show assessment statistics
            st.markdown("### 📊 Security Assessment Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_findings = self.get_phase_context().get("total_findings", 0)
                st.metric("Total Findings", total_findings)
            
            with col2:
                critical_findings = self.get_phase_context().get("critical_findings", 0)
                st.metric("Critical", critical_findings, delta_color="inverse")
            
            with col3:
                high_findings = self.get_phase_context().get("high_findings", 0)
                st.metric("High", high_findings, delta_color="inverse")
            
            with col4:
                compliance_score = self.get_phase_context().get("compliance_score", 0)
                st.metric("Compliance", f"{compliance_score}%")
            
            # Show recognition phase results if available
            recognition_result = context.get_phase_result(RADARPhase.RECOGNITION)
            if recognition_result and recognition_result.status == "completed":
                st.markdown("### 📦 Resources from Recognition Phase")
                if recognition_result.results:
                    total_resources = recognition_result.results.get("total_resources", 0)
                    st.info(f"Assessing {total_resources} discovered resources")
            
            # Vulnerability breakdown
            vuln_breakdown = self.get_phase_context().get("vulnerability_breakdown", {})
            if vuln_breakdown:
                st.markdown("### 🔒 Vulnerability Categories")
                for category, count in vuln_breakdown.items():
                    st.write(f"- **{category}:** {count} issues")
        else:
            st.info("No assessment results yet. Start by running a security scan.")
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate Assessment-specific response.
        
        Args:
            user_input: User's query
            
        Returns:
            Generated response
        """
        try:
            # Prepare context including Recognition results
            context = radar_state_manager.get_context()
            project_id = context.project_id if context else st.session_state.get('selected_project', 'default')
            
            # Get Recognition phase results to include in context
            phase_context = self.get_phase_context()
            if context:
                recognition_result = context.get_phase_result(RADARPhase.RECOGNITION)
                if recognition_result and recognition_result.status == "completed":
                    phase_context["recognition_results"] = recognition_result.results
            
            # Call RADAR backend
            response = api_client.radar_chat({
                "query": user_input,
                "phase": "assessment",
                "project_id": project_id,
                "context": phase_context
            })
            
            if response.get("success"):
                # Update phase context with results
                if "assessment_results" in response:
                    self.update_phase_context("latest_assessment", response["assessment_results"])
                    self._parse_assessment_results(response["assessment_results"])
                
                # Update RADAR state
                if context and radar_state_manager.can_execute_phase(RADARPhase.ASSESSMENT):
                    if context.phases[RADARPhase.ASSESSMENT].status != "completed":
                        radar_state_manager.start_phase(RADARPhase.ASSESSMENT)
                        radar_state_manager.complete_phase(
                            RADARPhase.ASSESSMENT,
                            response.get("assessment_results", {})
                        )
                
                return response.get("response", "Assessment completed.")
            else:
                return f"Error: {response.get('error', 'Failed to process assessment query')}"
                
        except Exception as e:
            logger.error(f"Assessment response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def run_security_scan(self):
        """Execute a comprehensive security scan."""
        with st.spinner("Running comprehensive security scan..."):
            try:
                # Call security scan API
                response = api_client.get_security_overview()
                
                if response.get("success"):
                    findings = response.get("findings", [])
                    summary = response.get("summary", {})
                    
                    # Update context
                    self.update_phase_context("total_findings", len(findings))
                    self.update_phase_context("critical_findings", summary.get("critical", 0))
                    self.update_phase_context("high_findings", summary.get("high", 0))
                    self.update_phase_context("latest_scan", findings)
                    
                    # Add system message
                    self.add_message(
                        "system",
                        f"🔍 Security scan completed: {len(findings)} findings discovered"
                    )
                    
                    # Show summary
                    st.success(f"Scan completed: {len(findings)} security findings")
                    
                    # Display severity breakdown
                    if summary:
                        st.markdown("### Severity Breakdown:")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Critical", summary.get("critical", 0))
                        with col2:
                            st.metric("High", summary.get("high", 0))
                        with col3:
                            st.metric("Medium", summary.get("medium", 0))
                        with col4:
                            st.metric("Low", summary.get("low", 0))
                else:
                    st.error(f"Security scan failed: {response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Security scan failed: {e}")
                st.error(f"Failed to run security scan: {str(e)}")
        
        st.rerun()
    
    def check_compliance(self):
        """Check compliance with various frameworks."""
        with st.spinner("Checking compliance status..."):
            try:
                # Check multiple compliance frameworks
                frameworks = ["CIS", "PCI-DSS", "HIPAA", "SOC2", "ISO27001"]
                compliance_results = {}
                
                for framework in frameworks:
                    try:
                        response = api_client.evaluate_compliance(framework)
                        if response.get("success"):
                            compliance_results[framework] = {
                                "score": response.get("score", 0),
                                "passed": response.get("passed", 0),
                                "failed": response.get("failed", 0),
                                "status": response.get("status", "Unknown")
                            }
                    except Exception as e:
                        logger.warning(f"Failed to check {framework} compliance: {e}")
                        compliance_results[framework] = {"score": 0, "status": "Error"}
                
                # Update context
                self.update_phase_context("compliance_results", compliance_results)
                avg_score = sum(r.get("score", 0) for r in compliance_results.values()) / len(compliance_results) if compliance_results else 0
                self.update_phase_context("compliance_score", round(avg_score))
                
                # Add message
                self.add_message(
                    "system",
                    f"📋 Compliance check completed: Average score {round(avg_score)}%"
                )
                
                # Display results
                st.success("Compliance check completed")
                
                st.markdown("### Compliance Framework Results:")
                for framework, result in compliance_results.items():
                    score = result.get("score", 0)
                    status = result.get("status", "Unknown")
                    color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                    st.write(f"{color} **{framework}**: {score}% - {status}")
                    
            except Exception as e:
                logger.error(f"Compliance check failed: {e}")
                st.error(f"Failed to check compliance: {str(e)}")
        
        st.rerun()
    
    def analyze_iam(self):
        """Analyze IAM permissions and security."""
        with st.spinner("Analyzing IAM permissions..."):
            try:
                # Get IAM analysis
                response = api_client.analyze_iam_security()
                
                if response.get("success"):
                    analysis = response.get("analysis", {})
                    
                    # Update context
                    self.update_phase_context("iam_analysis", analysis)
                    self.update_phase_context("overprivileged_accounts", analysis.get("overprivileged", 0))
                    self.update_phase_context("unused_accounts", analysis.get("unused", 0))
                    
                    # Add message
                    issues_found = analysis.get("total_issues", 0)
                    self.add_message(
                        "system",
                        f"🔐 IAM analysis completed: {issues_found} issues found"
                    )
                    
                    st.success(f"IAM analysis completed: {issues_found} issues found")
                    
                    # Display findings
                    if analysis:
                        st.markdown("### IAM Security Findings:")
                        st.write(f"- Overprivileged accounts: {analysis.get('overprivileged', 0)}")
                        st.write(f"- Unused service accounts: {analysis.get('unused', 0)}")
                        st.write(f"- Accounts without MFA: {analysis.get('no_mfa', 0)}")
                        st.write(f"- External access grants: {analysis.get('external_access', 0)}")
                else:
                    st.error(f"IAM analysis failed: {response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"IAM analysis failed: {e}")
                st.error(f"Failed to analyze IAM: {str(e)}")
        
        st.rerun()
    
    def assess_risks(self):
        """Perform comprehensive risk assessment."""
        with st.spinner("Assessing security risks..."):
            try:
                # Gather all assessment data
                context = self.get_phase_context()
                
                # Calculate risk scores
                risks = self._calculate_risk_scores(context)
                
                # Update context
                self.update_phase_context("risk_assessment", risks)
                self.update_phase_context("overall_risk_level", risks.get("overall_level", "Unknown"))
                
                # Add message
                self.add_message(
                    "system",
                    f"⚠️ Risk assessment completed: {risks.get('overall_level', 'Unknown')} risk level"
                )
                
                st.success(f"Risk assessment completed")
                
                # Display risk matrix
                st.markdown("### Risk Assessment Matrix:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Risk Categories:**")
                    for category, level in risks.get("categories", {}).items():
                        color = self._get_risk_color(level)
                        st.write(f"{color} {category}: {level}")
                
                with col2:
                    st.markdown("**Top Risk Factors:**")
                    for factor in risks.get("top_factors", [])[:5]:
                        st.write(f"- {factor}")
                        
            except Exception as e:
                logger.error(f"Risk assessment failed: {e}")
                st.error(f"Failed to assess risks: {str(e)}")
        
        st.rerun()
    
    def _parse_assessment_results(self, results: Any):
        """Parse and store assessment results in context."""
        if isinstance(results, dict):
            # Extract key metrics
            if "findings" in results:
                findings = results["findings"]
                if isinstance(findings, list):
                    self.update_phase_context("total_findings", len(findings))
                    
                    # Count by severity
                    critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
                    high = sum(1 for f in findings if f.get("severity") == "HIGH")
                    medium = sum(1 for f in findings if f.get("severity") == "MEDIUM")
                    low = sum(1 for f in findings if f.get("severity") == "LOW")
                    
                    self.update_phase_context("critical_findings", critical)
                    self.update_phase_context("high_findings", high)
                    self.update_phase_context("medium_findings", medium)
                    self.update_phase_context("low_findings", low)
                    
                    # Categorize vulnerabilities
                    vuln_categories = {}
                    for finding in findings:
                        category = finding.get("category", "Uncategorized")
                        vuln_categories[category] = vuln_categories.get(category, 0) + 1
                    
                    self.update_phase_context("vulnerability_breakdown", vuln_categories)
    
    def _calculate_risk_scores(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk scores based on assessment data."""
        risks = {
            "categories": {},
            "top_factors": [],
            "overall_score": 0,
            "overall_level": "Unknown"
        }
        
        # Calculate category risks
        if context.get("critical_findings", 0) > 0:
            risks["categories"]["Security Vulnerabilities"] = "Critical"
            risks["top_factors"].append(f"{context['critical_findings']} critical vulnerabilities")
        elif context.get("high_findings", 0) > 5:
            risks["categories"]["Security Vulnerabilities"] = "High"
            risks["top_factors"].append(f"{context['high_findings']} high severity findings")
        else:
            risks["categories"]["Security Vulnerabilities"] = "Medium"
        
        if context.get("compliance_score", 100) < 60:
            risks["categories"]["Compliance"] = "High"
            risks["top_factors"].append(f"Low compliance score: {context['compliance_score']}%")
        elif context.get("compliance_score", 100) < 80:
            risks["categories"]["Compliance"] = "Medium"
        else:
            risks["categories"]["Compliance"] = "Low"
        
        if context.get("overprivileged_accounts", 0) > 10:
            risks["categories"]["IAM Security"] = "High"
            risks["top_factors"].append(f"{context['overprivileged_accounts']} overprivileged accounts")
        elif context.get("overprivileged_accounts", 0) > 5:
            risks["categories"]["IAM Security"] = "Medium"
        else:
            risks["categories"]["IAM Security"] = "Low"
        
        # Calculate overall risk
        risk_scores = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        if risks["categories"]:
            avg_score = sum(risk_scores.get(level, 0) for level in risks["categories"].values()) / len(risks["categories"])
            risks["overall_score"] = avg_score
            
            if avg_score >= 3.5:
                risks["overall_level"] = "Critical"
            elif avg_score >= 2.5:
                risks["overall_level"] = "High"
            elif avg_score >= 1.5:
                risks["overall_level"] = "Medium"
            else:
                risks["overall_level"] = "Low"
        
        return risks
    
    def _get_risk_color(self, level: str) -> str:
        """Get color emoji for risk level."""
        colors = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢",
            "Unknown": "⚪"
        }
        return colors.get(level, "⚪")


def render_assessment_chat_view():
    """Render the Assessment phase chat view."""
    chat_view = AssessmentChatView()
    
    # Start phase if needed
    context = radar_state_manager.get_context()
    if context and radar_state_manager.can_execute_phase(RADARPhase.ASSESSMENT):
        if context.phases[RADARPhase.ASSESSMENT].status == "pending":
            radar_state_manager.start_phase(RADARPhase.ASSESSMENT)
    
    # Render interface
    chat_view.render_chat_interface()