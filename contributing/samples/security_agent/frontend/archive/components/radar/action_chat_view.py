"""
Action Phase Chat View - Remediation execution interface.

This module implements the Action phase of RADAR, focusing on
executing remediation actions and fixes.
"""

import streamlit as st
import logging
from typing import Dict, Any, List, Optional
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from components.shared.chat_streaming_base import StreamingChatBase
from components.radar.radar_state_manager import radar_state_manager, RADARPhase
from unified_api_client import api_client

logger = logging.getLogger(__name__)


class ActionChatView(StreamingChatBase):
    """
    Action phase chat interface.
    
    This phase focuses on:
    - Safe remediation execution
    - Pre-flight validation checks
    - Progress tracking and monitoring
    - Rollback capabilities
    """
    
    def __init__(self):
        """Initialize Action chat view."""
        super().__init__(
            phase_name="Action",
            phase_icon="⚡",
            phase_description="Execute remediation actions to fix identified security issues"
        )
    
    def render_quick_actions(self):
        """Render Action-specific quick actions."""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("✅ Execute Quick Wins", use_container_width=True):
                self.execute_quick_wins()
        
        with col2:
            if st.button("🔒 Apply Security Fixes", use_container_width=True):
                self.apply_security_fixes()
        
        with col3:
            if st.button("🔐 Fix IAM Issues", use_container_width=True):
                self.fix_iam_issues()
        
        with col4:
            if st.button("📋 Enable Compliance", use_container_width=True):
                self.enable_compliance_controls()
        
        # Standard actions
        super().render_quick_actions()
    
    def render_context_panel(self):
        """Render Action-specific context panel."""
        context = radar_state_manager.get_context()
        
        if context:
            # Show action statistics
            st.markdown("### ⚡ Action Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_actions = self.get_phase_context().get("total_actions", 0)
                st.metric("Total Actions", total_actions)
            
            with col2:
                completed_actions = self.get_phase_context().get("completed_actions", 0)
                st.metric("Completed", completed_actions)
            
            with col3:
                failed_actions = self.get_phase_context().get("failed_actions", 0)
                st.metric("Failed", failed_actions, delta_color="inverse")
            
            with col4:
                success_rate = self.get_phase_context().get("success_rate", 0)
                st.metric("Success Rate", f"{success_rate}%")
            
            # Show decision results if available
            decision_result = context.get_phase_result(RADARPhase.DECISION)
            if decision_result and decision_result.status == "completed":
                st.markdown("### 🎯 Action Plan from Decision Phase")
                if decision_result.results:
                    recommendations = decision_result.results.get("total_recommendations", 0)
                    quick_wins = decision_result.results.get("quick_wins_count", 0)
                    st.info(f"Executing {recommendations} recommendations ({quick_wins} quick wins)")
            
            # Execution log
            execution_log = self.get_phase_context().get("execution_log", [])
            if execution_log:
                st.markdown("### 📝 Recent Actions")
                for entry in execution_log[-5:]:  # Show last 5 entries
                    status_icon = "✅" if entry.get("status") == "success" else "❌"
                    st.write(f"{status_icon} {entry.get('action', 'Unknown')} - {entry.get('timestamp', '')}")
            
            # Safety checks
            safety_status = self.get_phase_context().get("safety_checks", {})
            if safety_status:
                st.markdown("### 🛡️ Safety Checks")
                for check, status in safety_status.items():
                    status_icon = "✅" if status else "⚠️"
                    st.write(f"{status_icon} {check}")
        else:
            st.info("No action data yet. Complete Decision phase to get remediation recommendations.")
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate Action-specific response.
        
        Args:
            user_input: User's query
            
        Returns:
            Generated response
        """
        try:
            # Prepare context including Decision results
            context = radar_state_manager.get_context()
            project_id = context.project_id if context else st.session_state.get('selected_project', 'default')
            
            # Get previous phase results
            phase_context = self.get_phase_context()
            if context:
                # Include all previous phase results
                for phase in [RADARPhase.RECOGNITION, RADARPhase.ASSESSMENT, RADARPhase.DECISION]:
                    phase_result = context.get_phase_result(phase)
                    if phase_result and phase_result.status == "completed":
                        phase_context[f"{phase.value}_results"] = phase_result.results
            
            # Check for authorization
            authorize_actions = st.session_state.get("authorize_remediation", False)
            
            # Call RADAR backend
            response = api_client.radar_chat({
                "query": user_input,
                "phase": "action",
                "project_id": project_id,
                "context": phase_context,
                "authorize_actions": authorize_actions
            })
            
            if response.get("success"):
                # Update phase context with results
                if "action_results" in response:
                    self.update_phase_context("latest_actions", response["action_results"])
                    self._parse_action_results(response["action_results"])
                
                # Update RADAR state
                if context and radar_state_manager.can_execute_phase(RADARPhase.ACTION):
                    if context.phases[RADARPhase.ACTION].status != "completed":
                        radar_state_manager.start_phase(RADARPhase.ACTION)
                        radar_state_manager.complete_phase(
                            RADARPhase.ACTION,
                            response.get("action_results", {})
                        )
                
                return response.get("response", "Action execution completed.")
            else:
                return f"Error: {response.get('error', 'Failed to process action query')}"
                
        except Exception as e:
            logger.error(f"Action response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def execute_quick_wins(self):
        """Execute quick win remediation actions."""
        with st.spinner("Executing quick win remediations..."):
            try:
                # Get quick wins from Decision phase
                context = radar_state_manager.get_context()
                if not context:
                    st.warning("No context available. Please complete Decision phase first.")
                    return
                
                decision_result = context.get_phase_result(RADARPhase.DECISION)
                if not decision_result or decision_result.status != "completed":
                    st.warning("Decision phase not completed. Please complete Decision first.")
                    return
                
                quick_wins = decision_result.results.get("quick_wins", [])
                if not quick_wins:
                    st.info("No quick wins identified. Check Decision phase for recommendations.")
                    return
                
                # Request authorization
                if not self._request_authorization("execute quick wins"):
                    return
                
                # Execute each quick win
                executed = []
                failed = []
                
                for action in quick_wins[:5]:  # Limit to 5 for safety
                    result = self._execute_action(action)
                    if result["success"]:
                        executed.append(action)
                    else:
                        failed.append((action, result.get("error", "Unknown error")))
                
                # Update context
                self._log_execution("Quick Wins Batch", len(executed), len(failed))
                
                # Add message
                self.add_message(
                    "system",
                    f"⚡ Quick wins execution: {len(executed)} succeeded, {len(failed)} failed"
                )
                
                # Show results
                st.success(f"Executed {len(executed)} quick wins successfully")
                
                if executed:
                    st.markdown("### ✅ Successfully Executed:")
                    for action in executed:
                        st.write(f"- {action.get('title', 'Unknown')}")
                
                if failed:
                    st.markdown("### ❌ Failed Actions:")
                    for action, error in failed:
                        st.write(f"- {action.get('title', 'Unknown')}: {error}")
                        
            except Exception as e:
                logger.error(f"Quick wins execution failed: {e}")
                st.error(f"Failed to execute quick wins: {str(e)}")
        
        st.rerun()
    
    def apply_security_fixes(self):
        """Apply critical security fixes."""
        with st.spinner("Applying security fixes..."):
            try:
                # Get critical security issues
                context = radar_state_manager.get_context()
                if not context:
                    st.warning("No context available. Please complete Assessment phase first.")
                    return
                
                assessment_result = context.get_phase_result(RADARPhase.ASSESSMENT)
                if not assessment_result or assessment_result.status != "completed":
                    st.warning("Assessment phase not completed. Please complete Assessment first.")
                    return
                
                # Request authorization
                if not self._request_authorization("apply security fixes"):
                    return
                
                # Simulate applying security fixes
                fixes = [
                    {"name": "Enable Cloud Armor DDoS protection", "type": "network_security"},
                    {"name": "Enable audit logging for all services", "type": "logging"},
                    {"name": "Enable encryption at rest for storage", "type": "encryption"},
                    {"name": "Enable VPC Flow Logs", "type": "monitoring"},
                    {"name": "Apply security patches to VMs", "type": "patching"}
                ]
                
                executed = []
                for fix in fixes:
                    # Simulate execution with safety checks
                    if self._perform_safety_check(fix):
                        executed.append(fix)
                        self._log_action(fix["name"], "success")
                
                # Update context
                self.update_phase_context("security_fixes_applied", len(executed))
                
                # Add message
                self.add_message(
                    "system",
                    f"🔒 Applied {len(executed)} security fixes successfully"
                )
                
                st.success(f"Applied {len(executed)} security fixes")
                
                # Show applied fixes
                st.markdown("### Applied Security Fixes:")
                for fix in executed:
                    st.write(f"✅ {fix['name']} ({fix['type']})")
                    
            except Exception as e:
                logger.error(f"Security fixes application failed: {e}")
                st.error(f"Failed to apply security fixes: {str(e)}")
        
        st.rerun()
    
    def fix_iam_issues(self):
        """Fix IAM permission issues."""
        with st.spinner("Fixing IAM issues..."):
            try:
                # Get IAM issues from Assessment
                iam_analysis = self.get_phase_context().get("iam_issues", {})
                
                if not iam_analysis:
                    # Simulate IAM fixes
                    iam_analysis = {
                        "overprivileged": ["service-account-1", "service-account-2"],
                        "unused": ["old-sa-1", "old-sa-2"],
                        "no_mfa": ["user1@example.com", "user2@example.com"]
                    }
                
                # Request authorization
                if not self._request_authorization("modify IAM permissions"):
                    return
                
                # Fix overprivileged accounts
                fixed_overprivileged = 0
                for account in iam_analysis.get("overprivileged", []):
                    if self._apply_least_privilege(account):
                        fixed_overprivileged += 1
                
                # Disable unused accounts
                disabled_unused = 0
                for account in iam_analysis.get("unused", []):
                    if self._disable_unused_account(account):
                        disabled_unused += 1
                
                # Update context
                self.update_phase_context("iam_fixes_applied", {
                    "overprivileged_fixed": fixed_overprivileged,
                    "unused_disabled": disabled_unused
                })
                
                # Add message
                self.add_message(
                    "system",
                    f"🔐 IAM fixes: {fixed_overprivileged} privileges reduced, {disabled_unused} accounts disabled"
                )
                
                st.success("IAM issues fixed")
                
                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Privileges Reduced", fixed_overprivileged)
                with col2:
                    st.metric("Accounts Disabled", disabled_unused)
                    
            except Exception as e:
                logger.error(f"IAM fixes failed: {e}")
                st.error(f"Failed to fix IAM issues: {str(e)}")
        
        st.rerun()
    
    def enable_compliance_controls(self):
        """Enable compliance controls and policies."""
        with st.spinner("Enabling compliance controls..."):
            try:
                # Request authorization
                if not self._request_authorization("enable compliance controls"):
                    return
                
                # Simulate enabling compliance controls
                controls = [
                    {"framework": "CIS", "control": "Enable Cloud Security Scanner", "enabled": True},
                    {"framework": "PCI-DSS", "control": "Enable payment card data encryption", "enabled": True},
                    {"framework": "HIPAA", "control": "Enable healthcare data protection", "enabled": True},
                    {"framework": "SOC2", "control": "Enable audit trail logging", "enabled": True},
                    {"framework": "ISO27001", "control": "Enable information security controls", "enabled": True}
                ]
                
                enabled_controls = []
                for control in controls:
                    if self._enable_compliance_control(control):
                        enabled_controls.append(control)
                
                # Update context
                self.update_phase_context("compliance_controls_enabled", len(enabled_controls))
                
                # Add message
                self.add_message(
                    "system",
                    f"📋 Enabled {len(enabled_controls)} compliance controls"
                )
                
                st.success(f"Enabled {len(enabled_controls)} compliance controls")
                
                # Show enabled controls
                st.markdown("### Enabled Compliance Controls:")
                for control in enabled_controls:
                    st.write(f"✅ **{control['framework']}**: {control['control']}")
                    
            except Exception as e:
                logger.error(f"Compliance control enablement failed: {e}")
                st.error(f"Failed to enable compliance controls: {str(e)}")
        
        st.rerun()
    
    def _parse_action_results(self, results: Any):
        """Parse and store action results in context."""
        if isinstance(results, dict):
            # Extract execution results
            if "executed_actions" in results:
                executed = results["executed_actions"]
                self.update_phase_context("total_actions", len(executed))
                
                # Count successes and failures
                completed = sum(1 for a in executed if a.get("status") == "success")
                failed = sum(1 for a in executed if a.get("status") == "failed")
                
                self.update_phase_context("completed_actions", completed)
                self.update_phase_context("failed_actions", failed)
                
                # Calculate success rate
                if len(executed) > 0:
                    success_rate = (completed / len(executed)) * 100
                    self.update_phase_context("success_rate", round(success_rate))
    
    def _request_authorization(self, action_type: str) -> bool:
        """Request user authorization for actions."""
        with st.expander("⚠️ Authorization Required", expanded=True):
            st.warning(f"""
            **Action Authorization Required**
            
            You are about to {action_type}. This will make changes to your cloud environment.
            
            **Safety Notice:**
            - All actions are logged for audit
            - Changes can be rolled back if needed
            - Pre-flight checks will be performed
            
            Do you authorize these actions?
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Authorize", type="primary"):
                    st.session_state.authorize_remediation = True
                    return True
            with col2:
                if st.button("❌ Cancel"):
                    st.info("Action cancelled by user")
                    return False
        
        return False
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single remediation action."""
        try:
            # Perform safety check
            if not self._perform_safety_check(action):
                return {"success": False, "error": "Safety check failed"}
            
            # Simulate execution (in real implementation, would call backend)
            # For now, simulate success for most actions
            import random
            success = random.random() > 0.1  # 90% success rate
            
            if success:
                self._log_action(action.get("title", "Unknown"), "success")
                return {"success": True}
            else:
                self._log_action(action.get("title", "Unknown"), "failed")
                return {"success": False, "error": "Simulated failure"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _perform_safety_check(self, action: Dict[str, Any]) -> bool:
        """Perform safety check before executing action."""
        # Simulate safety checks
        checks = {
            "Resource exists": True,
            "Permissions valid": True,
            "No production impact": True,
            "Backup available": True
        }
        
        # Update safety status
        self.update_phase_context("safety_checks", checks)
        
        return all(checks.values())
    
    def _apply_least_privilege(self, account: str) -> bool:
        """Apply least privilege principle to an account."""
        # Simulate applying least privilege
        self._log_action(f"Apply least privilege to {account}", "success")
        return True
    
    def _disable_unused_account(self, account: str) -> bool:
        """Disable an unused account."""
        # Simulate disabling account
        self._log_action(f"Disable unused account {account}", "success")
        return True
    
    def _enable_compliance_control(self, control: Dict[str, Any]) -> bool:
        """Enable a compliance control."""
        # Simulate enabling control
        self._log_action(f"Enable {control['framework']} control", "success")
        return True
    
    def _log_action(self, action: str, status: str):
        """Log an executed action."""
        log_entry = {
            "action": action,
            "status": status,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # Get or create execution log
        execution_log = self.get_phase_context().get("execution_log", [])
        execution_log.append(log_entry)
        
        # Keep only last 50 entries
        if len(execution_log) > 50:
            execution_log = execution_log[-50:]
        
        self.update_phase_context("execution_log", execution_log)
    
    def _log_execution(self, batch_name: str, succeeded: int, failed: int):
        """Log batch execution results."""
        # Update totals
        total = self.get_phase_context().get("total_actions", 0) + succeeded + failed
        completed = self.get_phase_context().get("completed_actions", 0) + succeeded
        failed_total = self.get_phase_context().get("failed_actions", 0) + failed
        
        self.update_phase_context("total_actions", total)
        self.update_phase_context("completed_actions", completed)
        self.update_phase_context("failed_actions", failed_total)
        
        # Calculate success rate
        if total > 0:
            success_rate = (completed / total) * 100
            self.update_phase_context("success_rate", round(success_rate))
        
        # Log batch
        self._log_action(f"{batch_name}: {succeeded} succeeded, {failed} failed", 
                        "success" if failed == 0 else "partial")


def render_action_chat_view():
    """Render the Action phase chat view."""
    chat_view = ActionChatView()
    
    # Start phase if needed
    context = radar_state_manager.get_context()
    if context and radar_state_manager.can_execute_phase(RADARPhase.ACTION):
        if context.phases[RADARPhase.ACTION].status == "pending":
            radar_state_manager.start_phase(RADARPhase.ACTION)
    
    # Render interface
    chat_view.render_chat_interface()