"""
Comprehensive test suite for Organization Policy API endpoints.
Tests constraint management, policy CRUD operations, and compliance analysis.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
import json

# Import the org_policy module and related components
from backend.api import org_policy
from backend.api.org_policy import (
    list_constraints, get_policy, get_effective_policy, create_or_update_policy,
    delete_policy, create_custom_constraint, list_custom_constraints,
    analyze_org_policies, list_policies, _process_policy_rule, _categorize_constraints,
    PolicyConstraintRequest, PolicyRequest, CustomConstraintRequest, PolicyAnalysisRequest
)
from backend.main import app

client = TestClient(app)

# Test fixtures
@pytest.fixture
def mock_orgpolicy_client():
    """Mock Google Cloud Organization Policy client."""
    with patch('backend.api.org_policy.orgpolicy_v2.OrgPolicyClient') as mock_client:
        yield mock_client

@pytest.fixture
def mock_constraint():
    """Mock organization policy constraint."""
    constraint = Mock()
    constraint.name = "constraints/compute.disableSerialPortAccess"
    constraint.display_name = "Disable VM serial port access"
    constraint.description = "Disables serial port access to Compute Engine VMs"
    constraint.constraint_default = "ALLOW"
    constraint.supports_dry_run = True
    
    # Mock list constraint
    constraint.list_constraint = Mock()
    constraint.list_constraint.supports_in = True
    constraint.list_constraint.supports_under = False
    constraint.boolean_constraint = None
    
    return constraint

@pytest.fixture
def mock_policy():
    """Mock organization policy."""
    policy = Mock()
    policy.name = "organizations/123456789/policies/constraints/compute.disableSerialPortAccess"
    
    # Mock policy spec
    policy.spec = Mock()
    policy.spec.inherit_from_parent = False
    policy.spec.reset = False
    policy.spec.etag = "abc123"
    policy.spec.update_time = Mock()
    policy.spec.update_time.isoformat.return_value = "2024-01-15T10:30:00Z"
    
    # Mock rules
    rule = Mock()
    rule.enforce = True
    rule.values = None
    rule.allow_all = False
    rule.deny_all = False
    rule.condition = None
    policy.spec.rules = [rule]
    
    return policy

@pytest.fixture
def mock_custom_constraint():
    """Mock custom constraint."""
    constraint = Mock()
    constraint.name = "organizations/123456789/customConstraints/custom.requireSpecificLabel"
    constraint.display_name = "Require specific label"
    constraint.description = "Requires all resources to have a specific label"
    constraint.resource_types = ["compute.googleapis.com/Instance"]
    constraint.method_types = ["CREATE", "UPDATE"]
    constraint.condition = "resource.labels.has('environment')"
    constraint.action_type = "DENY"
    return constraint

@pytest.fixture
def sample_policy_request():
    """Sample policy request for testing."""
    return PolicyRequest(
        parent="organizations/123456789",
        constraint="compute.disableSerialPortAccess",
        rules=[{
            "enforce": True
        }],
        inherit_from_parent=False,
        reset=False
    )

class TestOrganizationPolicyEndpoints:
    """Test class for Organization Policy API endpoints."""

    @pytest.mark.asyncio
    async def test_list_constraints_success(self, mock_orgpolicy_client, mock_constraint):
        """Test successful listing of organization policy constraints."""
        # Setup mock
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.list_constraints.return_value = [mock_constraint]
        
        result = await list_constraints("organizations/123456789")
        
        # Assertions
        assert result["success"] is True
        assert result["parent"] == "organizations/123456789"
        assert result["count"] == 1
        assert len(result["constraints"]) == 1
        assert result["constraints"][0]["name"] == "constraints/compute.disableSerialPortAccess"
        assert result["constraints"][0]["type"] == "list"
        assert "categories" in result

    @pytest.mark.asyncio
    async def test_list_constraints_orgpolicy_unavailable(self):
        """Test list constraints when orgpolicy library is unavailable."""
        with patch('backend.api.org_policy.ORGPOLICY_AVAILABLE', False):
            result = await list_constraints("organizations/123456789")
            
            assert result["success"] is False
            assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_list_constraints_exception(self, mock_orgpolicy_client):
        """Test list constraints with exception."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.list_constraints.side_effect = Exception("API Error")
        
        result = await list_constraints("organizations/123456789")
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_policy_success(self, mock_orgpolicy_client, mock_policy):
        """Test successful policy retrieval."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.get_policy.return_value = mock_policy
        
        # Mock effective policy call
        with patch('backend.api.org_policy.get_effective_policy') as mock_effective:
            mock_effective.return_value = {"success": True, "policy": {"constraint": "test"}}
            
            request = PolicyConstraintRequest(
                parent="organizations/123456789",
                constraint="compute.disableSerialPortAccess"
            )
            
            result = await get_policy(request)
            
            assert result["success"] is True
            assert "policy" in result
            assert result["policy"]["name"] == mock_policy.name
            assert result["policy"]["constraint"] == "compute.disableSerialPortAccess"
            assert len(result["policy"]["rules"]) == 1

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, mock_orgpolicy_client):
        """Test policy retrieval when policy doesn't exist."""
        from google.api_core import exceptions
        
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.get_policy.side_effect = exceptions.NotFound("Policy not found")
        
        request = PolicyConstraintRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess"
        )
        
        result = await get_policy(request)
        
        assert result["success"] is True
        assert result["policy"] is None
        assert "No policy set" in result["message"]

    @pytest.mark.asyncio
    async def test_get_effective_policy_success(self, mock_orgpolicy_client, mock_policy):
        """Test successful effective policy retrieval."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.get_effective_policy.return_value = mock_policy
        
        request = PolicyConstraintRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess"
        )
        
        result = await get_effective_policy(request)
        
        assert result["success"] is True
        assert result["parent"] == "organizations/123456789"
        assert "policy" in result

    @pytest.mark.asyncio
    async def test_create_or_update_policy_success(self, mock_orgpolicy_client, mock_policy):
        """Test successful policy creation/update."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.update_policy.return_value = mock_policy
        
        request = PolicyRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess",
            rules=[{"enforce": True}],
            inherit_from_parent=False,
            reset=False
        )
        
        result = await create_or_update_policy(request)
        
        assert result["success"] is True
        assert "policy" in result
        assert result["policy"]["constraint"] == "compute.disableSerialPortAccess"
        assert "updated successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, mock_orgpolicy_client):
        """Test successful policy deletion."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.delete_policy.return_value = None
        
        request = PolicyConstraintRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess"
        )
        
        result = await delete_policy(request)
        
        assert result["success"] is True
        assert "deleted successfully" in result["message"]
        assert result["parent"] == "organizations/123456789"
        assert result["constraint"] == "compute.disableSerialPortAccess"

    @pytest.mark.asyncio
    async def test_create_custom_constraint_success(self, mock_orgpolicy_client, mock_custom_constraint):
        """Test successful custom constraint creation."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.create_custom_constraint.return_value = mock_custom_constraint
        
        request = CustomConstraintRequest(
            parent="organizations/123456789",
            constraint_id="custom.requireSpecificLabel",
            display_name="Require specific label",
            description="Requires all resources to have a specific label",
            resource_types=["compute.googleapis.com/Instance"],
            method_types=["CREATE", "UPDATE"],
            condition="resource.labels.has('environment')",
            action_type="DENY"
        )
        
        result = await create_custom_constraint(request)
        
        assert result["success"] is True
        assert "constraint" in result
        assert result["constraint"]["name"] == mock_custom_constraint.name
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_list_custom_constraints_success(self, mock_orgpolicy_client, mock_custom_constraint):
        """Test successful listing of custom constraints."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.list_custom_constraints.return_value = [mock_custom_constraint]
        
        result = await list_custom_constraints("organizations/123456789")
        
        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["custom_constraints"]) == 1
        assert result["custom_constraints"][0]["name"] == mock_custom_constraint.name

    @pytest.mark.asyncio
    async def test_list_policies_success(self, mock_orgpolicy_client, mock_policy):
        """Test successful listing of policies."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.list_policies.return_value = [mock_policy]
        
        result = await list_policies("organizations/123456789")
        
        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["policies"]) == 1

class TestPolicyAnalysis:
    """Test class for policy analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_org_policies_success(self):
        """Test successful organization policy analysis."""
        # Mock list_policies function
        with patch('backend.api.org_policy.list_policies') as mock_list:
            mock_list.return_value = {
                "success": True,
                "policies": [
                    {"constraint": "compute.disableSerialPortAccess"},
                    {"constraint": "compute.requireOsLogin"},
                    {"constraint": "storage.uniformBucketLevelAccess"},
                    {"constraint": "iam.disableServiceAccountKeyCreation"}
                ]
            }
            
            request = PolicyAnalysisRequest(scope="organizations/123456789")
            result = await analyze_org_policies(request)
            
            assert result["success"] is True
            assert "analysis" in result
            analysis = result["analysis"]
            
            assert "total_policies" in analysis
            assert analysis["total_policies"] == 4
            assert "policy_coverage" in analysis
            assert "compliance_status" in analysis
            assert "recommendations" in analysis
            assert "policy_gaps" in analysis

    @pytest.mark.asyncio
    async def test_analyze_org_policies_low_coverage(self):
        """Test policy analysis with low coverage."""
        with patch('backend.api.org_policy.list_policies') as mock_list:
            mock_list.return_value = {
                "success": True,
                "policies": [
                    {"constraint": "compute.disableSerialPortAccess"}
                ]
            }
            
            request = PolicyAnalysisRequest(scope="organizations/123456789")
            result = await analyze_org_policies(request)
            
            assert result["success"] is True
            analysis = result["analysis"]
            
            # Should have recommendations for low coverage
            assert len(analysis["recommendations"]) > 0
            assert any("security policy coverage" in rec for rec in analysis["recommendations"])
            
            # Should have policy gaps
            assert len(analysis["policy_gaps"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_org_policies_list_failure(self):
        """Test policy analysis when list_policies fails."""
        with patch('backend.api.org_policy.list_policies') as mock_list:
            mock_list.return_value = {"success": False, "error": "Permission denied"}
            
            request = PolicyAnalysisRequest(scope="organizations/123456789")
            result = await analyze_org_policies(request)
            
            assert result["success"] is False
            assert "error" in result

class TestHelperFunctions:
    """Test class for helper functions."""

    def test_process_policy_rule_boolean(self):
        """Test processing boolean policy rule."""
        rule = Mock()
        rule.enforce = True
        rule.values = None
        rule.allow_all = False
        rule.deny_all = False
        rule.condition = None
        
        result = _process_policy_rule(rule)
        
        assert result["type"] == "boolean"
        assert result["enforce"] is True

    def test_process_policy_rule_list(self):
        """Test processing list policy rule."""
        rule = Mock()
        rule.values = Mock()
        rule.values.allowed_values = ["value1", "value2"]
        rule.values.denied_values = ["value3"]
        rule.condition = None
        
        result = _process_policy_rule(rule)
        
        assert result["type"] == "list"
        assert result["allowed_values"] == ["value1", "value2"]
        assert result["denied_values"] == ["value3"]

    def test_process_policy_rule_allow_all(self):
        """Test processing allow-all policy rule."""
        rule = Mock()
        rule.allow_all = True
        rule.values = None
        rule.deny_all = False
        rule.condition = None
        
        # Mock hasattr to return True for allow_all
        with patch('builtins.hasattr', side_effect=lambda obj, attr: attr == 'allow_all'):
            result = _process_policy_rule(rule)
            
            assert result["type"] == "allow_all"
            assert result["allow_all"] is True

    def test_process_policy_rule_deny_all(self):
        """Test processing deny-all policy rule."""
        rule = Mock()
        rule.deny_all = True
        rule.values = None
        rule.allow_all = False
        rule.condition = None
        
        with patch('builtins.hasattr', side_effect=lambda obj, attr: attr == 'deny_all'):
            result = _process_policy_rule(rule)
            
            assert result["type"] == "deny_all"
            assert result["deny_all"] is True

    def test_process_policy_rule_with_condition(self):
        """Test processing policy rule with condition."""
        rule = Mock()
        rule.enforce = True
        rule.values = None
        rule.condition = Mock()
        rule.condition.expression = "resource.location == 'us-central1'"
        rule.condition.title = "Location restriction"
        rule.condition.description = "Restrict to US Central 1"
        rule.condition.location = "us-central1"
        
        result = _process_policy_rule(rule)
        
        assert result["type"] == "boolean"
        assert "condition" in result
        assert result["condition"]["expression"] == "resource.location == 'us-central1'"

    def test_categorize_constraints(self):
        """Test constraint categorization."""
        constraints = [
            {"name": "constraints/compute.disableSerialPortAccess"},
            {"name": "constraints/storage.uniformBucketLevelAccess"},
            {"name": "constraints/iam.disableServiceAccountKeyCreation"},
            {"name": "constraints/sql.restrictPublicIp"},
            {"name": "constraints/resourcemanager.restrictTgwAttachment"},
            {"name": "constraints/unknown.someConstraint"}
        ]
        
        result = _categorize_constraints(constraints)
        
        assert "compute" in result
        assert "storage" in result
        assert "iam" in result
        assert "sql" in result
        assert "resource_manager" in result
        assert "other" in result
        
        assert len(result["compute"]) == 1
        assert len(result["storage"]) == 1
        assert len(result["iam"]) == 1
        assert len(result["sql"]) == 1
        assert len(result["resource_manager"]) == 1
        assert len(result["other"]) == 1

class TestPydanticModels:
    """Test class for Pydantic model validation."""

    def test_policy_constraint_request_validation(self):
        """Test PolicyConstraintRequest validation."""
        request = PolicyConstraintRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess"
        )
        
        assert request.parent == "organizations/123456789"
        assert request.constraint == "compute.disableSerialPortAccess"

    def test_policy_request_validation(self):
        """Test PolicyRequest validation."""
        request = PolicyRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess",
            rules=[{"enforce": True}],
            inherit_from_parent=False,
            reset=False,
            etag="abc123"
        )
        
        assert request.parent == "organizations/123456789"
        assert request.constraint == "compute.disableSerialPortAccess"
        assert len(request.rules) == 1
        assert request.inherit_from_parent is False
        assert request.etag == "abc123"

    def test_custom_constraint_request_validation(self):
        """Test CustomConstraintRequest validation."""
        request = CustomConstraintRequest(
            parent="organizations/123456789",
            constraint_id="custom.requireSpecificLabel",
            display_name="Require specific label",
            description="Test description",
            resource_types=["compute.googleapis.com/Instance"],
            method_types=["CREATE", "UPDATE"],
            condition="resource.labels.has('environment')",
            action_type="DENY"
        )
        
        assert request.parent == "organizations/123456789"
        assert request.constraint_id == "custom.requireSpecificLabel"
        assert request.action_type == "DENY"
        assert len(request.resource_types) == 1
        assert len(request.method_types) == 2

    def test_policy_analysis_request_validation(self):
        """Test PolicyAnalysisRequest validation."""
        request = PolicyAnalysisRequest(
            scope="organizations/123456789",
            constraint_filter="compute.*",
            include_inherited=True,
            check_compliance=True
        )
        
        assert request.scope == "organizations/123456789"
        assert request.constraint_filter == "compute.*"
        assert request.include_inherited is True
        assert request.check_compliance is True

class TestErrorHandling:
    """Test class for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_orgpolicy_unavailable_error(self):
        """Test behavior when orgpolicy library is unavailable."""
        with patch('backend.api.org_policy.ORGPOLICY_AVAILABLE', False):
            request = PolicyConstraintRequest(
                parent="organizations/123456789",
                constraint="compute.disableSerialPortAccess"
            )
            
            result = await get_policy(request)
            assert result["success"] is False
            assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_permission_denied_error(self, mock_orgpolicy_client):
        """Test handling of permission denied errors."""
        from google.api_core import exceptions
        
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.get_policy.side_effect = exceptions.PermissionDenied("Permission denied")
        
        request = PolicyConstraintRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess"
        )
        
        result = await get_policy(request)
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_constraint_error(self, mock_orgpolicy_client):
        """Test handling of invalid constraint errors."""
        from google.api_core import exceptions
        
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        mock_client_instance.get_policy.side_effect = exceptions.InvalidArgument("Invalid constraint")
        
        request = PolicyConstraintRequest(
            parent="organizations/123456789",
            constraint="invalid.constraint"
        )
        
        result = await get_policy(request)
        assert result["success"] is False
        assert "error" in result

class TestIntegrationScenarios:
    """Test class for integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_policy_lifecycle(self, mock_orgpolicy_client, mock_policy):
        """Test complete policy lifecycle (create, read, update, delete)."""
        mock_client_instance = Mock()
        mock_orgpolicy_client.return_value = mock_client_instance
        
        # Test creation
        mock_client_instance.update_policy.return_value = mock_policy
        
        create_request = PolicyRequest(
            parent="organizations/123456789",
            constraint="compute.disableSerialPortAccess",
            rules=[{"enforce": True}]
        )
        
        create_result = await create_or_update_policy(create_request)
        assert create_result["success"] is True
        
        # Test reading
        mock_client_instance.get_policy.return_value = mock_policy
        
        with patch('backend.api.org_policy.get_effective_policy') as mock_effective:
            mock_effective.return_value = {"success": True, "policy": {"constraint": "test"}}
            
            read_request = PolicyConstraintRequest(
                parent="organizations/123456789",
                constraint="compute.disableSerialPortAccess"
            )
            
            read_result = await get_policy(read_request)
            assert read_result["success"] is True
        
        # Test deletion
        mock_client_instance.delete_policy.return_value = None
        
        delete_result = await delete_policy(read_request)
        assert delete_result["success"] is True

    @pytest.mark.asyncio
    async def test_compliance_analysis_workflow(self):
        """Test complete compliance analysis workflow."""
        # Mock policies with varying compliance levels
        with patch('backend.api.org_policy.list_policies') as mock_list:
            mock_list.return_value = {
                "success": True,
                "policies": [
                    {"constraint": "compute.disableSerialPortAccess"},
                    {"constraint": "compute.requireOsLogin"},
                    # Missing several essential policies
                ]
            }
            
            request = PolicyAnalysisRequest(
                scope="organizations/123456789",
                check_compliance=True
            )
            
            result = await analyze_org_policies(request)
            
            assert result["success"] is True
            analysis = result["analysis"]
            
            # Should identify gaps
            assert len(analysis["policy_gaps"]) > 0
            
            # Should have compliance score
            assert "compliance_status" in analysis
            assert "score" in analysis["compliance_status"]
            assert "rating" in analysis["compliance_status"]
            
            # Should provide recommendations
            assert len(analysis["recommendations"]) > 0

if __name__ == "__main__":
    pytest.main([__file__])