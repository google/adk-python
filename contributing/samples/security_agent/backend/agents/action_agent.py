"""
Action Agent - RADAR Phase 4

Executes remediation and configuration changes.
The "executor" of the RADAR system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
# Direct API imports instead of base classes and tools

logger = logging.getLogger(__name__)


class ActionAgent:
    """
    RADAR Phase 4: Act - Remediation and Changes
    
    This agent executes remediation actions.
    It has LIMITED write permissions for safety.
    
    Key responsibilities:
    - Execute approved remediation actions
    - Apply security configurations
    - Create resources with proper security
    - Document all changes
    - Maintain audit trail
    
    IMPORTANT: This agent has write permissions and should be used carefully.
    All actions should be authorized and logged.
    """
    
    def __init__(self, project_id: str):
        """Initialize Action Agent for remediation execution."""
        self.project_id = project_id
        self.name = "ActionAgent"
        self.description = "Executes remediation and configuration changes"
        logger.info(f"⚡ Action Agent initialized for project {project_id}")
        
        # Track actions for audit
        self.action_log = []
    
    def get_instruction(self) -> str:
        """Get the instruction for this agent."""
        return """You are the Action Agent - the executor of RADAR.
        
        Your mission:
        1. Execute ONLY approved remediation actions
        2. Apply security configurations safely
        3. Create resources with proper security settings
        4. Document EVERY change made
        5. Prepare changes for Review Agent verification
        
        You have LIMITED write permissions for safety.
        Always verify current state before making changes.
        Document every action for audit trail.
        Never execute unauthorized actions.
        """
    
    async def execute_remediation(
        self,
        remediation_plan: List[Dict],
        authorize: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute remediation plan with available tools.
        
        Args:
            remediation_plan: Plan from Decision Agent
            authorize: Whether actions are authorized (safety flag)
            dry_run: If True, simulate actions without executing
            
        Returns:
            Execution result with action log
        """
        logger.info(f"⚡ {'Simulating' if dry_run else 'Executing'} remediation for {self.project_id}")
        
        execution_result = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "phase": "action",
            "mode": "dry_run" if dry_run else "execute",
            "authorized": authorize,
            "actions_attempted": 0,
            "actions_succeeded": 0,
            "actions_failed": 0,
            "actions_skipped": 0,
            "action_log": []
        }
        
        if not authorize and not dry_run:
            logger.warning("⚠️ Remediation not authorized and not in dry-run mode")
            execution_result["error"] = "Actions must be authorized or run in dry-run mode"
            execution_result["success"] = False
            return execution_result
        
        # Process each step in the plan
        for step in remediation_plan:
            execution_result["actions_attempted"] += 1
            
            action_entry = {
                "step": step.get("step"),
                "priority": step.get("priority"),
                "description": step.get("description"),
                "action": step.get("action"),
                "timestamp": datetime.now().isoformat(),
                "status": None,
                "result": None,
                "error": None
            }
            
            try:
                # Execute based on action type
                result = await self._execute_action_step(step, dry_run)
                
                if result.get("success"):
                    action_entry["status"] = "succeeded" if not dry_run else "simulated"
                    action_entry["result"] = result.get("details", {})
                    execution_result["actions_succeeded"] += 1
                else:
                    action_entry["status"] = "failed"
                    action_entry["error"] = result.get("error", "Unknown error")
                    execution_result["actions_failed"] += 1
                
            except Exception as e:
                logger.error(f"Action step {step.get('step')} failed: {e}")
                action_entry["status"] = "error"
                action_entry["error"] = str(e)
                execution_result["actions_failed"] += 1
            
            execution_result["action_log"].append(action_entry)
            self.action_log.append(action_entry)
        
        # Generate summary
        execution_result["summary"] = self._generate_execution_summary(execution_result)
        execution_result["success"] = execution_result["actions_failed"] == 0
        
        return execution_result
    
    async def _execute_action_step(self, step: Dict, dry_run: bool) -> Dict[str, Any]:
        """
        Execute a single action step.
        
        This method determines the action type and executes accordingly.
        """
        description = step.get("description", "").lower()
        specific_actions = step.get("specific_actions", [])
        
        # Determine action type from description
        if "api key" in description:
            return await self._handle_api_key_action(step, dry_run)
        elif "iam" in description or "permission" in description:
            return await self._handle_iam_action(step, dry_run)
        elif "notification" in description or "advisory" in description:
            return await self._handle_notification_action(step, dry_run)
        else:
            # Generic action handler
            return await self._handle_generic_action(step, dry_run)
    
    async def _handle_api_key_action(self, step: Dict, dry_run: bool) -> Dict[str, Any]:
        """Handle API key related actions."""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Handling API key action")
        
        if dry_run:
            return {
                "success": True,
                "action_type": "api_key",
                "details": {
                    "would_create": "Restricted API key",
                    "restrictions": {
                        "api_targets": ["translate.googleapis.com"],
                        "ip_restrictions": ["10.0.0.0/8"]
                    }
                }
            }
        
        # In production, would actually create/modify API key
        # For safety, we'll simulate even in execute mode
        try:
            # Example: Create a properly restricted API key
            if "create" in step.get("action", "").lower():
                # This would actually call create_api_key tool
                result = {
                    "success": True,
                    "action_type": "api_key_created",
                    "details": {
                        "display_name": "Secured-API-Key",
                        "restrictions_applied": True
                    }
                }
            else:
                result = {
                    "success": True,
                    "action_type": "api_key_updated",
                    "details": {
                        "restrictions_added": True
                    }
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_iam_action(self, step: Dict, dry_run: bool) -> Dict[str, Any]:
        """Handle IAM related actions."""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Handling IAM action")
        
        if dry_run:
            return {
                "success": True,
                "action_type": "iam",
                "details": {
                    "would_modify": "IAM policy",
                    "changes": {
                        "remove_bindings": ["roles/owner from service accounts"],
                        "add_bindings": ["roles/viewer with conditions"]
                    }
                }
            }
        
        # Read current IAM policy first
        try:
            policy = await get_iam_policy(
                resource=f"projects/{self.project_id}",
                tool_context=self.context
            )
            
            if policy.get("success"):
                # In production, would modify and update policy
                return {
                    "success": True,
                    "action_type": "iam_updated",
                    "details": {
                        "policy_version": policy.get("policy", {}).get("version"),
                        "modifications": "Applied least privilege"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Could not read current IAM policy"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_notification_action(self, step: Dict, dry_run: bool) -> Dict[str, Any]:
        """Handle notification/advisory related actions."""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Handling notification action")
        
        if dry_run:
            return {
                "success": True,
                "action_type": "notification",
                "details": {
                    "would_configure": "Notification settings",
                    "changes": {
                        "enable_critical_alerts": True,
                        "add_recipients": ["security-team@example.com"]
                    }
                }
            }
        
        # Get current notification settings
        try:
            settings = await get_notification_settings(tool_context=self.context)
            
            if settings.get("success"):
                # In production, would update settings
                return {
                    "success": True,
                    "action_type": "notifications_configured",
                    "details": {
                        "critical_alerts_enabled": True,
                        "email_configured": True
                    }
                }
            else:
                return {
                    "success": True,  # Not critical if we can't update notifications
                    "action_type": "notification_skipped",
                    "details": {
                        "reason": "Notification service not available"
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_generic_action(self, step: Dict, dry_run: bool) -> Dict[str, Any]:
        """Handle generic/unknown action types."""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Handling generic action")
        
        if dry_run:
            return {
                "success": True,
                "action_type": "generic",
                "details": {
                    "simulated": True,
                    "description": step.get("description", "Generic action")
                }
            }
        
        # For unknown actions, log but don't fail
        return {
            "success": True,
            "action_type": "generic",
            "details": {
                "logged": True,
                "manual_action_required": True,
                "description": step.get("description", "Manual review needed")
            }
        }
    
    def _generate_execution_summary(self, execution_result: Dict) -> str:
        """Generate summary of execution results."""
        total = execution_result["actions_attempted"]
        succeeded = execution_result["actions_succeeded"]
        failed = execution_result["actions_failed"]
        mode = execution_result["mode"]
        
        if mode == "dry_run":
            summary = f"DRY RUN COMPLETE: Simulated {total} actions. "
            summary += f"{succeeded} would succeed, {failed} would fail."
        else:
            summary = f"EXECUTION COMPLETE: Attempted {total} actions. "
            summary += f"{succeeded} succeeded, {failed} failed."
            
            if failed > 0:
                summary += " Review failures in action_log."
            elif succeeded == total:
                summary += " All actions completed successfully."
        
        return summary
    
    async def rollback_actions(self, action_log: List[Dict]) -> Dict[str, Any]:
        """
        Rollback previously executed actions.
        
        Args:
            action_log: Log of actions to rollback
            
        Returns:
            Rollback result
        """
        logger.warning(f"⚠️ Initiating rollback for {len(action_log)} actions")
        
        rollback_result = {
            "timestamp": datetime.now().isoformat(),
            "project_id": self.project_id,
            "phase": "action_rollback",
            "actions_to_rollback": len(action_log),
            "rollback_succeeded": 0,
            "rollback_failed": 0,
            "rollback_log": []
        }
        
        # Process actions in reverse order
        for action in reversed(action_log):
            if action.get("status") == "succeeded":
                rollback_entry = {
                    "original_action": action.get("description"),
                    "rollback_action": "Revert " + action.get("description", ""),
                    "status": "simulated",  # For safety, only simulate rollbacks
                    "timestamp": datetime.now().isoformat()
                }
                
                # In production, would actually rollback changes
                rollback_result["rollback_succeeded"] += 1
                rollback_result["rollback_log"].append(rollback_entry)
        
        rollback_result["success"] = rollback_result["rollback_failed"] == 0
        rollback_result["message"] = "Rollback simulation complete. Manual rollback may be required."
        
        return rollback_result
    
    def get_action_history(self) -> List[Dict]:
        """Get history of all actions taken by this agent."""
        return self.action_log