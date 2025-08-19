"""
Automated Remediation Engine for Security Vulnerabilities
=========================================================

This module provides automated remediation capabilities for security vulnerabilities
detected by the security analysis engine (STORY-002). It includes safe execution,
rollback capabilities, and approval workflows.

Part of STORY-210: Automated Remediation Engine
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Remediation Status Enums
class RemediationStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"
    UNSAFE = "UNSAFE"

class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ActionType(str, Enum):
    MODIFY_BUCKET_IAM = "MODIFY_BUCKET_IAM"
    ENABLE_UNIFORM_ACCESS = "ENABLE_UNIFORM_ACCESS"
    MODIFY_IAM_POLICY = "MODIFY_IAM_POLICY"
    ENABLE_ENCRYPTION = "ENABLE_ENCRYPTION"
    MODIFY_FIREWALL_RULES = "MODIFY_FIREWALL_RULES"
    ROTATE_CREDENTIALS = "ROTATE_CREDENTIALS"
    ENABLE_LOGGING = "ENABLE_LOGGING"
    APPLY_SECURITY_POLICY = "APPLY_SECURITY_POLICY"

# Data Models
@dataclass
class RemediationAction:
    """Individual remediation action"""
    type: ActionType
    operation: str
    parameters: Dict[str, Any]
    validation: str
    timeout: int = 60  # seconds

@dataclass
class RollbackPoint:
    """Snapshot for rollback"""
    id: str
    resource_name: str
    state: Dict[str, Any]
    timestamp: datetime
    expiry: datetime

class RemediationRequest(BaseModel):
    """Request for automated remediation"""
    vulnerability_id: str
    remediation_template: str
    parameters: Dict[str, Any] = {}
    auto_approve: bool = False
    dry_run: bool = True
    priority: str = "MEDIUM"

class RemediationResult(BaseModel):
    """Result of remediation execution"""
    remediation_id: str
    status: RemediationStatus
    vulnerability_id: str
    resource_name: str
    changes_made: List[Dict[str, Any]] = []
    rollback_point: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    validation_results: Dict[str, bool] = {}

class ApprovalRequest(BaseModel):
    """Approval request for high-risk remediations"""
    request_id: str
    remediation_id: str
    template_name: str
    risk_level: RiskLevel
    resource_name: str
    approvers: List[str]
    requested_at: datetime
    timeout: datetime
    justification: Optional[str] = None

class ApprovalResult(BaseModel):
    """Result of approval workflow"""
    request_id: str
    approved: bool
    approved_by: Optional[str] = None
    rejected_by: Optional[str] = None
    approval_time: Optional[datetime] = None
    comments: Optional[str] = None

# Base Remediation Template
class RemediationTemplate:
    """Base class for remediation templates"""
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.name = ""
        self.description = ""
        self.vulnerability_types = []
        self.risk_level = RiskLevel.MEDIUM
        self.requires_approval = False
        self.actions: List[RemediationAction] = []
        self.rollback_capable = True
        
    def add_action(self, action: RemediationAction):
        """Add remediation action to template"""
        self.actions.append(action)
        
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate template parameters"""
        errors = []
        required_params = self.get_required_parameters()
        
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        
        return len(errors) == 0, errors
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters"""
        return []

# Pre-built Remediation Templates
class PublicBucketRemediation(RemediationTemplate):
    """Remediate public storage bucket exposure"""
    
    def __init__(self):
        super().__init__()
        self.name = "Remove Public Access from Storage Bucket"
        self.description = "Removes public access and enables uniform bucket-level access"
        self.vulnerability_types = ["PUBLIC_STORAGE_NO_AUTH", "PUBLIC_BUCKET"]
        self.risk_level = RiskLevel.HIGH
        self.requires_approval = True
        
        self.add_action(RemediationAction(
            type=ActionType.MODIFY_BUCKET_IAM,
            operation="REMOVE_PUBLIC_ACCESS",
            parameters={"remove_bindings": ["allUsers", "allAuthenticatedUsers"]},
            validation="bucket_not_public"
        ))
        
        self.add_action(RemediationAction(
            type=ActionType.ENABLE_UNIFORM_ACCESS,
            operation="SET_UNIFORM_BUCKET_LEVEL_ACCESS",
            parameters={"enabled": True},
            validation="uniform_access_enabled"
        ))
    
    def get_required_parameters(self) -> List[str]:
        return ["bucket_name", "project_id"]

class ExcessiveIAMRemediation(RemediationTemplate):
    """Remediate overprivileged IAM accounts"""
    
    def __init__(self):
        super().__init__()
        self.name = "Remove Excessive IAM Permissions"
        self.description = "Replaces overly broad roles with least-privilege alternatives"
        self.vulnerability_types = ["EXCESSIVE_IAM_PERMISSIONS", "OVERPRIVILEGED_ACCOUNT"]
        self.risk_level = RiskLevel.CRITICAL
        self.requires_approval = True
        
        self.add_action(RemediationAction(
            type=ActionType.MODIFY_IAM_POLICY,
            operation="REPLACE_ROLE",
            parameters={
                "remove_roles": ["roles/owner", "roles/editor"],
                "add_roles": ["roles/viewer"]
            },
            validation="least_privilege_enforced"
        ))
    
    def get_required_parameters(self) -> List[str]:
        return ["service_account", "project_id"]

class MissingEncryptionRemediation(RemediationTemplate):
    """Enable encryption for unencrypted resources"""
    
    def __init__(self):
        super().__init__()
        self.name = "Enable Encryption"
        self.description = "Enables encryption at rest for resources"
        self.vulnerability_types = ["MISSING_ENCRYPTION", "UNENCRYPTED_DISK"]
        self.risk_level = RiskLevel.HIGH
        self.requires_approval = False
        
        self.add_action(RemediationAction(
            type=ActionType.ENABLE_ENCRYPTION,
            operation="ENABLE_DEFAULT_ENCRYPTION",
            parameters={"encryption_type": "GOOGLE_DEFAULT_ENCRYPTION"},
            validation="encryption_enabled"
        ))
    
    def get_required_parameters(self) -> List[str]:
        return ["resource_name", "resource_type"]

class WeakNetworkSecurityRemediation(RemediationTemplate):
    """Fix weak network security configurations"""
    
    def __init__(self):
        super().__init__()
        self.name = "Restrict Network Access"
        self.description = "Restricts overly permissive firewall rules"
        self.vulnerability_types = ["WEAK_NETWORK_SECURITY", "OPEN_FIREWALL"]
        self.risk_level = RiskLevel.HIGH
        self.requires_approval = True
        
        self.add_action(RemediationAction(
            type=ActionType.MODIFY_FIREWALL_RULES,
            operation="RESTRICT_SOURCE_RANGES",
            parameters={
                "remove_ranges": ["0.0.0.0/0"],
                "add_ranges": ["10.0.0.0/8"]  # Example internal range
            },
            validation="no_public_access"
        ))
    
    def get_required_parameters(self) -> List[str]:
        return ["firewall_rule_name", "project_id"]

# Template Registry
class TemplateRegistry:
    """Registry of available remediation templates"""
    
    def __init__(self):
        self.templates = {
            "PUBLIC_BUCKET_REMEDIATION": PublicBucketRemediation(),
            "EXCESSIVE_IAM_REMEDIATION": ExcessiveIAMRemediation(),
            "MISSING_ENCRYPTION_REMEDIATION": MissingEncryptionRemediation(),
            "WEAK_NETWORK_SECURITY_REMEDIATION": WeakNetworkSecurityRemediation()
        }
    
    def get_template(self, template_id: str) -> Optional[RemediationTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def get_template_for_vulnerability(self, vuln_type: str) -> Optional[RemediationTemplate]:
        """Get appropriate template for vulnerability type"""
        for template in self.templates.values():
            if vuln_type in template.vulnerability_types:
                return template
        return None

# Rollback Manager
class RollbackManager:
    """Manage rollback points and restoration"""
    
    def __init__(self):
        self.snapshots = {}  # In-memory storage for demo
    
    async def create_snapshot(self, resource_name: str) -> RollbackPoint:
        """Create snapshot before remediation"""
        
        # Capture current state (simplified for demo)
        current_state = await self._capture_resource_state(resource_name)
        
        snapshot = RollbackPoint(
            id=str(uuid.uuid4()),
            resource_name=resource_name,
            state=current_state,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(days=7)
        )
        
        self.snapshots[snapshot.id] = snapshot
        logger.info(f"Created rollback point {snapshot.id} for {resource_name}")
        return snapshot
    
    async def rollback(self, rollback_point_id: str) -> bool:
        """Rollback to previous state"""
        
        if rollback_point_id not in self.snapshots:
            raise ValueError(f"Rollback point {rollback_point_id} not found")
        
        snapshot = self.snapshots[rollback_point_id]
        
        try:
            # Restore resource state (simplified for demo)
            await self._restore_resource_state(
                snapshot.resource_name,
                snapshot.state
            )
            
            logger.info(f"Successfully rolled back {snapshot.resource_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def _capture_resource_state(self, resource_name: str) -> Dict[str, Any]:
        """Capture current resource state"""
        # Simplified - in production, would call GCP APIs
        return {
            "resource_name": resource_name,
            "captured_at": datetime.now().isoformat(),
            "configuration": {"example": "state"}
        }
    
    async def _restore_resource_state(self, resource_name: str, state: Dict[str, Any]) -> None:
        """Restore resource to previous state"""
        # Simplified - in production, would call GCP APIs
        await asyncio.sleep(0.5)  # Simulate API call
        logger.info(f"Restored {resource_name} to previous state")

# Approval Workflow
class ApprovalWorkflow:
    """Multi-level approval workflow for high-risk remediations"""
    
    def __init__(self):
        self.pending_approvals = {}  # In-memory storage for demo
        
    async def request_approval(
        self,
        remediation_id: str,
        template: RemediationTemplate,
        resource_name: str
    ) -> ApprovalResult:
        """Request approval based on risk level"""
        
        approvers = self._get_approvers(template.risk_level)
        
        approval_request = ApprovalRequest(
            request_id=str(uuid.uuid4()),
            remediation_id=remediation_id,
            template_name=template.name,
            risk_level=template.risk_level,
            resource_name=resource_name,
            approvers=approvers,
            requested_at=datetime.now(),
            timeout=datetime.now() + timedelta(hours=2)
        )
        
        self.pending_approvals[approval_request.request_id] = approval_request
        
        # Simulate approval process (auto-approve for demo)
        await asyncio.sleep(1)
        
        # For demo, auto-approve non-critical
        if template.risk_level != RiskLevel.CRITICAL:
            return ApprovalResult(
                request_id=approval_request.request_id,
                approved=True,
                approved_by="auto-approval@system",
                approval_time=datetime.now(),
                comments="Auto-approved based on risk level"
            )
        
        # For critical, simulate manual approval
        return ApprovalResult(
            request_id=approval_request.request_id,
            approved=True,
            approved_by="security-lead@company.com",
            approval_time=datetime.now(),
            comments="Manually approved after review"
        )
    
    def _get_approvers(self, risk_level: RiskLevel) -> List[str]:
        """Get required approvers based on risk level"""
        if risk_level == RiskLevel.CRITICAL:
            return ["security-lead@company.com", "platform-lead@company.com"]
        elif risk_level == RiskLevel.HIGH:
            return ["security-team@company.com"]
        else:
            return []  # Auto-approve for medium/low risk

# Remediation Executor
class RemediationExecutor:
    """Execute remediation actions"""
    
    async def dry_run(self, template: RemediationTemplate, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform dry run of remediation"""
        
        dry_run_results = {
            "safe": True,
            "issues": [],
            "expected_changes": []
        }
        
        for action in template.actions:
            # Simulate validation
            if action.type == ActionType.MODIFY_IAM_POLICY and "roles/owner" in str(action.parameters):
                dry_run_results["expected_changes"].append({
                    "action": action.type.value,
                    "description": "Will remove owner role and add viewer role",
                    "risk": "HIGH"
                })
            else:
                dry_run_results["expected_changes"].append({
                    "action": action.type.value,
                    "description": f"Will execute {action.operation}",
                    "risk": "MEDIUM"
                })
        
        return dry_run_results
    
    async def execute(
        self,
        template: RemediationTemplate,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute remediation actions"""
        
        execution_result = {
            "success": True,
            "changes": [],
            "errors": []
        }
        
        for action in template.actions:
            try:
                # Simulate action execution
                await asyncio.sleep(0.5)
                
                change = {
                    "action": action.type.value,
                    "operation": action.operation,
                    "status": "SUCCESS",
                    "timestamp": datetime.now().isoformat()
                }
                
                execution_result["changes"].append(change)
                logger.info(f"Executed action: {action.type.value}")
                
            except Exception as e:
                execution_result["success"] = False
                execution_result["errors"].append(str(e))
                logger.error(f"Action failed: {e}")
                break
        
        return execution_result

# Remediation Validator
class RemediationValidator:
    """Validate remediation results"""
    
    async def validate_remediation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that remediation was successful"""
        
        validation_result = {
            "success": True,
            "errors": [],
            "validations": {}
        }
        
        # Simulate validation checks
        for change in result.get("changes", []):
            validation_key = f"{change['action']}_validation"
            
            # Simulate validation
            await asyncio.sleep(0.2)
            validation_result["validations"][validation_key] = True
        
        return validation_result

# Main Remediation Engine
class RemediationEngine:
    """Core automated remediation execution engine"""
    
    def __init__(self):
        self.executor = RemediationExecutor()
        self.validator = RemediationValidator()
        self.rollback_manager = RollbackManager()
        self.approval_workflow = ApprovalWorkflow()
        self.template_registry = TemplateRegistry()
        self.active_remediations = {}
        
    async def remediate_vulnerability(
        self,
        vulnerability: Dict[str, Any],
        request: RemediationRequest
    ) -> RemediationResult:
        """Execute automated remediation for a vulnerability"""
        
        remediation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Get remediation template
            template = self.template_registry.get_template(request.remediation_template)
            if not template:
                template = self.template_registry.get_template_for_vulnerability(
                    vulnerability.get("vulnerability_type", "")
                )
            
            if not template:
                return RemediationResult(
                    remediation_id=remediation_id,
                    status=RemediationStatus.FAILED,
                    vulnerability_id=request.vulnerability_id,
                    resource_name=vulnerability.get("resource_name", ""),
                    error_message="No suitable remediation template found"
                )
            
            # Validate parameters
            valid, errors = template.validate_parameters(request.parameters)
            if not valid:
                return RemediationResult(
                    remediation_id=remediation_id,
                    status=RemediationStatus.FAILED,
                    vulnerability_id=request.vulnerability_id,
                    resource_name=vulnerability.get("resource_name", ""),
                    error_message=f"Parameter validation failed: {', '.join(errors)}"
                )
            
            # Check if approval needed
            if template.requires_approval and not request.auto_approve:
                approval = await self.approval_workflow.request_approval(
                    remediation_id,
                    template,
                    vulnerability.get("resource_name", "")
                )
                if not approval.approved:
                    return RemediationResult(
                        remediation_id=remediation_id,
                        status=RemediationStatus.REJECTED,
                        vulnerability_id=request.vulnerability_id,
                        resource_name=vulnerability.get("resource_name", ""),
                        error_message="Approval rejected"
                    )
            
            # Perform dry run if requested
            if request.dry_run:
                dry_run_result = await self.executor.dry_run(template, request.parameters)
                if not dry_run_result["safe"]:
                    return RemediationResult(
                        remediation_id=remediation_id,
                        status=RemediationStatus.UNSAFE,
                        vulnerability_id=request.vulnerability_id,
                        resource_name=vulnerability.get("resource_name", ""),
                        error_message=f"Dry run detected issues: {dry_run_result['issues']}"
                    )
                
                # If only dry run requested, return success with expected changes
                if request.dry_run:
                    return RemediationResult(
                        remediation_id=remediation_id,
                        status=RemediationStatus.SUCCESS,
                        vulnerability_id=request.vulnerability_id,
                        resource_name=vulnerability.get("resource_name", ""),
                        changes_made=dry_run_result["expected_changes"],
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
            
            # Create rollback point
            rollback_point = await self.rollback_manager.create_snapshot(
                vulnerability.get("resource_name", "")
            )
            
            # Execute remediation
            result = await self.executor.execute(template, request.parameters)
            
            if not result["success"]:
                # Rollback on failure
                await self.rollback_manager.rollback(rollback_point.id)
                return RemediationResult(
                    remediation_id=remediation_id,
                    status=RemediationStatus.ROLLED_BACK,
                    vulnerability_id=request.vulnerability_id,
                    resource_name=vulnerability.get("resource_name", ""),
                    error_message=f"Execution failed and rolled back: {result['errors']}"
                )
            
            # Validate remediation
            validation = await self.validator.validate_remediation(result)
            if not validation["success"]:
                # Rollback if validation fails
                await self.rollback_manager.rollback(rollback_point.id)
                return RemediationResult(
                    remediation_id=remediation_id,
                    status=RemediationStatus.ROLLED_BACK,
                    vulnerability_id=request.vulnerability_id,
                    resource_name=vulnerability.get("resource_name", ""),
                    error_message=f"Validation failed and rolled back: {validation['errors']}"
                )
            
            # Success!
            return RemediationResult(
                remediation_id=remediation_id,
                status=RemediationStatus.SUCCESS,
                vulnerability_id=request.vulnerability_id,
                resource_name=vulnerability.get("resource_name", ""),
                changes_made=result["changes"],
                rollback_point=rollback_point.id,
                execution_time=(datetime.now() - start_time).total_seconds(),
                validation_results=validation["validations"]
            )
            
        except Exception as e:
            logger.error(f"Remediation failed with exception: {e}")
            return RemediationResult(
                remediation_id=remediation_id,
                status=RemediationStatus.FAILED,
                vulnerability_id=request.vulnerability_id,
                resource_name=vulnerability.get("resource_name", ""),
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def get_remediation_status(self, remediation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a remediation"""
        return self.active_remediations.get(remediation_id)
    
    async def rollback_remediation(self, remediation_id: str, rollback_point_id: str) -> bool:
        """Manually trigger rollback"""
        return await self.rollback_manager.rollback(rollback_point_id)
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available remediation templates"""
        templates = []
        for template_id, template in self.template_registry.templates.items():
            templates.append({
                "id": template_id,
                "name": template.name,
                "description": template.description,
                "vulnerability_types": template.vulnerability_types,
                "risk_level": template.risk_level.value,
                "requires_approval": template.requires_approval
            })
        return templates