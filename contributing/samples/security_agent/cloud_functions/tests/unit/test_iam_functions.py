#!/usr/bin/env python3
"""
Unit tests for IAM-related Cloud Functions.
Tests fetch_custom_roles, fetch_user_roles, fetch_service_account_roles, fetch_standard_roles
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the Cloud Functions
from fetch_custom_roles import main as custom_roles_main
from fetch_user_roles import main as user_roles_main
from fetch_service_account_roles import main as service_account_roles_main
from fetch_standard_roles import main as standard_roles_main


class TestFetchCustomRoles:
    """Tests for fetch_custom_roles Cloud Function"""

    def test_fetch_custom_roles_success(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of custom roles"""
        # Arrange
        request = mock_http_request()

        # Mock custom role
        mock_role = MagicMock()
        mock_role.name = "projects/test-project-123/roles/customDataAnalyst"
        mock_role.title = "Custom Data Analyst"
        mock_role.description = "Analyzes data"
        mock_role.included_permissions = [
            "bigquery.datasets.create",
            "storage.buckets.delete",
            "iam.roles.update"
        ]
        mock_role.stage = MagicMock(name="GA")

        mock_iam_client.list_roles.return_value = [mock_role]

        # Act
        with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_custom_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = custom_roles_main.fetch_custom_roles(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_custom_roles"] == 1
        assert response["high_risk_roles"] == 1  # Has delete and update permissions
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_custom_roles_empty_results(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test when no custom roles exist"""
        # Arrange
        request = mock_http_request()
        mock_iam_client.list_roles.return_value = []

        # Act
        with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_custom_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = custom_roles_main.fetch_custom_roles(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_custom_roles"] == 0
        assert response["high_risk_roles"] == 0

    def test_fetch_custom_roles_api_error(self, mock_iam_client, mock_http_request):
        """Test handling of API errors"""
        # Arrange
        request = mock_http_request()
        mock_iam_client.list_roles.side_effect = Exception("API Error")

        # Act
        with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
            response, status_code = custom_roles_main.fetch_custom_roles(request)

        # Assert
        assert status_code == 500
        assert response["status"] == "error"
        assert "API Error" in response["message"]

    def test_fetch_custom_roles_bigquery_insert_error(self, mock_iam_client, mock_bigquery_with_errors, mock_http_request):
        """Test handling of BigQuery insert errors"""
        # Arrange
        request = mock_http_request()
        mock_role = MagicMock()
        mock_role.name = "projects/test-project-123/roles/testRole"
        mock_role.included_permissions = []
        mock_iam_client.list_roles.return_value = [mock_role]

        # Act
        with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_custom_roles.main.bigquery.Client', return_value=mock_bigquery_with_errors):
                response, status_code = custom_roles_main.fetch_custom_roles(request)

        # Assert
        assert status_code == 500
        assert response["status"] == "error"
        assert "Invalid row" in response["message"]

    def test_fetch_custom_roles_permission_analysis(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test permission risk analysis"""
        # Arrange
        request = mock_http_request()

        # Create role with various permission types
        mock_role = MagicMock()
        mock_role.name = "projects/test-project-123/roles/mixedPermissions"
        mock_role.included_permissions = [
            "storage.buckets.get",      # Low risk
            "storage.buckets.list",     # Low risk
            "storage.buckets.delete",   # High risk
            "iam.roles.create",         # High risk
            "iam.serviceAccounts.actAs" # High risk
        ]
        mock_iam_client.list_roles.return_value = [mock_role]

        # Act
        with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_custom_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = custom_roles_main.fetch_custom_roles(request)

        # Assert
        assert status_code == 200
        # Verify the data inserted to BigQuery
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["permission_count"] == 5
        assert insert_call["high_risk_permissions"] == 3


class TestFetchUserRoles:
    """Tests for fetch_user_roles Cloud Function"""

    def test_fetch_user_roles_success(self, mock_crm_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of user roles"""
        # Arrange
        request = mock_http_request()

        # Mock IAM policy with various user types
        mock_policy = MagicMock()
        mock_binding1 = MagicMock()
        mock_binding1.role = "roles/owner"
        mock_binding1.members = [
            "user:admin@example.com",
            "user:external@otherdomain.com"
        ]

        mock_binding2 = MagicMock()
        mock_binding2.role = "roles/viewer"
        mock_binding2.members = [
            "user:viewer@example.com",
            "group:developers@example.com"
        ]

        mock_policy.bindings = [mock_binding1, mock_binding2]
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_user_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_user_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = user_roles_main.fetch_user_roles(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_users"] == 3
        assert response["admin_users"] == 1
        assert response["external_users"] == 1
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_user_roles_identify_admins(self, mock_crm_client, mock_bigquery_client, mock_http_request):
        """Test correct identification of admin roles"""
        # Arrange
        request = mock_http_request()

        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/editor"  # Editor is considered admin
        mock_binding.members = ["user:editor@example.com"]
        mock_policy.bindings = [mock_binding]

        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_user_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_user_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = user_roles_main.fetch_user_roles(request)

        # Assert
        assert status_code == 200
        # Verify the user is marked as admin
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["is_admin"] == True
        assert insert_call["is_owner"] == False

    def test_fetch_user_roles_external_detection(self, mock_crm_client, mock_bigquery_client, mock_http_request):
        """Test detection of external users"""
        # Arrange
        request = mock_http_request()

        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/viewer"
        mock_binding.members = [
            "user:internal@test-project-123.iam.gserviceaccount.com",
            "user:external@gmail.com",
            "user:another@external-company.com"
        ]
        mock_policy.bindings = [mock_binding]

        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_user_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_user_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = user_roles_main.fetch_user_roles(request)

        # Assert
        assert status_code == 200
        assert response["external_users"] == 2

    def test_fetch_user_roles_no_users(self, mock_crm_client, mock_bigquery_client, mock_http_request):
        """Test when no user IAM bindings exist"""
        # Arrange
        request = mock_http_request()
        mock_policy = MagicMock()
        mock_policy.bindings = []
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_user_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_user_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = user_roles_main.fetch_user_roles(request)

        # Assert
        assert status_code == 200
        assert response["total_users"] == 0


class TestFetchServiceAccountRoles:
    """Tests for fetch_service_account_roles Cloud Function"""

    def test_fetch_service_account_roles_success(self, mock_crm_client, mock_iam_service_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of service account roles"""
        # Arrange
        request = mock_http_request()

        # Mock IAM policy with service accounts
        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/storage.admin"
        mock_binding.members = [
            "serviceAccount:test-sa@test-project-123.iam.gserviceaccount.com",
            "serviceAccount:123456789@cloudservices.gserviceaccount.com"  # Google-managed
        ]
        mock_policy.bindings = [mock_binding]
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_service_account_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_service_account_roles.main.IAMClient', return_value=mock_iam_service_client):
                with patch('fetch_service_account_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = service_account_roles_main.fetch_service_account_roles(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_service_accounts"] == 2
        assert response["google_managed_accounts"] == 1
        assert response["user_managed_accounts"] == 1

    def test_fetch_service_account_roles_with_keys(self, mock_crm_client, mock_iam_service_client, mock_bigquery_client, mock_http_request):
        """Test detection of service account keys"""
        # Arrange
        request = mock_http_request()

        # Mock service account with keys
        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/editor"
        mock_binding.members = ["serviceAccount:has-keys@test-project-123.iam.gserviceaccount.com"]
        mock_policy.bindings = [mock_binding]
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Mock service account details
        mock_sa = MagicMock()
        mock_sa.email = "has-keys@test-project-123.iam.gserviceaccount.com"
        mock_sa.disabled = False
        mock_iam_service_client.list_service_accounts.return_value = [mock_sa]

        # Mock keys
        mock_key1 = MagicMock()
        mock_key1.key_type = "USER_MANAGED"
        mock_key2 = MagicMock()
        mock_key2.key_type = "USER_MANAGED"
        mock_key3 = MagicMock()
        mock_key3.key_type = "SYSTEM_MANAGED"

        mock_iam_service_client.list_service_account_keys.return_value = [mock_key1, mock_key2, mock_key3]

        # Act
        with patch('fetch_service_account_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_service_account_roles.main.IAMClient', return_value=mock_iam_service_client):
                with patch('fetch_service_account_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = service_account_roles_main.fetch_service_account_roles(request)

        # Assert
        assert status_code == 200
        assert response["accounts_with_keys"] == 1
        # Verify key count in BigQuery data
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["key_count"] == 2  # Only USER_MANAGED keys
        assert insert_call["has_keys"] == True

    def test_fetch_service_account_roles_disabled_accounts(self, mock_crm_client, mock_iam_service_client, mock_bigquery_client, mock_http_request):
        """Test detection of disabled service accounts"""
        # Arrange
        request = mock_http_request()

        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/viewer"
        mock_binding.members = ["serviceAccount:disabled@test-project-123.iam.gserviceaccount.com"]
        mock_policy.bindings = [mock_binding]
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Mock disabled service account
        mock_sa = MagicMock()
        mock_sa.email = "disabled@test-project-123.iam.gserviceaccount.com"
        mock_sa.disabled = True
        mock_iam_service_client.list_service_accounts.return_value = [mock_sa]
        mock_iam_service_client.list_service_account_keys.return_value = []

        # Act
        with patch('fetch_service_account_roles.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_service_account_roles.main.IAMClient', return_value=mock_iam_service_client):
                with patch('fetch_service_account_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = service_account_roles_main.fetch_service_account_roles(request)

        # Assert
        assert status_code == 200
        # Verify disabled flag in BigQuery data
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["disabled"] == True


class TestFetchStandardRoles:
    """Tests for fetch_standard_roles Cloud Function"""

    def test_fetch_standard_roles_success(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of standard/predefined roles"""
        # Arrange
        request = mock_http_request()

        # Mock various types of standard roles
        roles = []

        # Primitive role
        primitive_role = MagicMock()
        primitive_role.name = "roles/owner"
        primitive_role.title = "Owner"
        primitive_role.description = "Full access to all resources"
        primitive_role.included_permissions = ["*"]
        primitive_role.stage = MagicMock(name="GA")
        roles.append(primitive_role)

        # Admin role
        admin_role = MagicMock()
        admin_role.name = "roles/storage.admin"
        admin_role.title = "Storage Admin"
        admin_role.description = "Full control of GCS resources"
        admin_role.included_permissions = [
            "storage.buckets.create",
            "storage.buckets.delete",
            "storage.buckets.update",
            "storage.objects.delete"
        ]
        admin_role.stage = MagicMock(name="GA")
        roles.append(admin_role)

        # Read-only role
        viewer_role = MagicMock()
        viewer_role.name = "roles/bigquery.dataViewer"
        viewer_role.title = "BigQuery Data Viewer"
        viewer_role.description = "View BigQuery datasets and tables"
        viewer_role.included_permissions = [
            "bigquery.datasets.get",
            "bigquery.tables.get",
            "bigquery.tables.list"
        ]
        viewer_role.stage = MagicMock(name="GA")
        roles.append(viewer_role)

        mock_iam_client.list_roles.return_value = roles

        # Act
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_roles"] == 3
        assert response["primitive_roles"] == 1
        assert response["admin_roles"] >= 1
        assert response["read_only_roles"] >= 1
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_standard_roles_categorization(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test role categorization logic"""
        # Arrange
        request = mock_http_request()

        # Create role with specific naming patterns
        roles = []

        # Writer role
        writer_role = MagicMock()
        writer_role.name = "roles/storage.objectCreator"
        writer_role.title = "Storage Object Creator"
        writer_role.included_permissions = ["storage.objects.create"]
        writer_role.stage = MagicMock(name="GA")
        roles.append(writer_role)

        # Editor role (should be categorized as WRITE)
        editor_role = MagicMock()
        editor_role.name = "roles/datastore.indexEditor"
        editor_role.title = "Datastore Index Editor"
        editor_role.included_permissions = ["datastore.indexes.update"]
        editor_role.stage = MagicMock(name="GA")
        roles.append(editor_role)

        mock_iam_client.list_roles.return_value = roles

        # Act
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)

        # Assert
        assert status_code == 200
        # Verify categorization in BigQuery data
        insert_calls = mock_bigquery_client.insert_rows_json.call_args[0][1]

        # Check that editor role is categorized as WRITE
        editor_data = next(r for r in insert_calls if "indexEditor" in r["role_name"])
        assert editor_data["category"] == "WRITE"

    def test_fetch_standard_roles_service_extraction(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test extraction of services from role names"""
        # Arrange
        request = mock_http_request()

        # Create roles from different services
        roles = []
        services = ["storage", "compute", "bigquery", "iam", "logging"]

        for service in services:
            role = MagicMock()
            role.name = f"roles/{service}.viewer"
            role.title = f"{service.title()} Viewer"
            role.included_permissions = [f"{service}.resources.get"]
            role.stage = MagicMock(name="GA")
            roles.append(role)

        mock_iam_client.list_roles.return_value = roles

        # Act
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)

        # Assert
        assert status_code == 200
        assert response["total_roles"] == 5
        assert "storage" in response["top_services"]
        assert "compute" in response["top_services"]

    def test_fetch_standard_roles_empty_permissions(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test handling of roles with no permissions"""
        # Arrange
        request = mock_http_request()

        # Role with no permissions
        role = MagicMock()
        role.name = "roles/empty.role"
        role.title = "Empty Role"
        role.description = "Role with no permissions"
        role.included_permissions = None  # No permissions
        role.stage = MagicMock(name="GA")

        mock_iam_client.list_roles.return_value = [role]

        # Act
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)

        # Assert
        assert status_code == 200
        # Verify empty permissions are handled gracefully
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["permission_count"] == 0
        assert insert_call["high_risk_permissions"] == 0
        assert json.loads(insert_call["included_permissions"]) == []


# Performance and edge case tests

class TestIAMFunctionsPerformance:
    """Performance tests for IAM Cloud Functions"""

    def test_large_role_dataset(self, mock_iam_client, mock_bigquery_client, mock_http_request, performance_timer):
        """Test handling of large number of roles"""
        # Arrange
        request = mock_http_request()

        # Create 1000 mock roles
        roles = []
        for i in range(1000):
            role = MagicMock()
            role.name = f"roles/custom{i}"
            role.included_permissions = [f"permission{j}" for j in range(10)]
            role.stage = MagicMock(name="GA")
            roles.append(role)

        mock_iam_client.list_roles.return_value = roles

        # Act
        performance_timer.start()
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)
        performance_timer.stop()

        # Assert
        assert status_code == 200
        assert response["total_roles"] == 1000
        assert performance_timer.elapsed < 10  # Should complete within 10 seconds

    def test_concurrent_function_execution(self, mock_iam_client, mock_crm_client, mock_bigquery_client, mock_http_request):
        """Test that functions can be executed concurrently without conflicts"""
        import threading
        import time

        # Arrange
        request = mock_http_request()
        results = []
        errors = []

        def run_function(func_name, func):
            try:
                with patch('google.cloud.iam_admin_v1.IAMClient', return_value=mock_iam_client):
                    with patch('google.cloud.resourcemanager_v3.ProjectsClient', return_value=mock_crm_client):
                        with patch('google.cloud.bigquery.Client', return_value=mock_bigquery_client):
                            response, status = func(request)
                            results.append((func_name, status))
            except Exception as e:
                errors.append((func_name, str(e)))

        # Act - Run all IAM functions concurrently
        threads = [
            threading.Thread(target=run_function, args=("custom_roles", custom_roles_main.fetch_custom_roles)),
            threading.Thread(target=run_function, args=("user_roles", user_roles_main.fetch_user_roles)),
            threading.Thread(target=run_function, args=("service_account_roles", service_account_roles_main.fetch_service_account_roles)),
            threading.Thread(target=run_function, args=("standard_roles", standard_roles_main.fetch_standard_roles))
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        # Assert
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 4
        for func_name, status in results:
            assert status == 200, f"{func_name} failed with status {status}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])