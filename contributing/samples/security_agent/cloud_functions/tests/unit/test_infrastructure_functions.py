#!/usr/bin/env python3
"""
Unit tests for infrastructure-related Cloud Functions.
Tests fetch_compute_instances, fetch_firewall_rules, fetch_storage_buckets,
fetch_security_findings, fetch_iam_accounts, and other functions.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock, PropertyMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the Cloud Functions
from fetch_compute_instances import main as compute_main
from fetch_firewall_rules import main as firewall_main
from fetch_storage_buckets import main as storage_main
from fetch_security_findings import main as security_findings_main
from fetch_iam_accounts import main as iam_accounts_main


class TestFetchComputeInstances:
    """Tests for fetch_compute_instances Cloud Function"""

    def test_fetch_compute_instances_success(self, mock_compute_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of compute instances"""
        # Arrange
        request = mock_http_request()

        # Act
        with patch('fetch_compute_instances.main.InstancesClient', return_value=mock_compute_client):
            with patch('fetch_compute_instances.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = compute_main.fetch_compute_instances(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_instances"] == 1
        assert response["running_instances"] == 1
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_compute_instances_multiple_zones(self, mock_compute_client, mock_bigquery_client, mock_http_request):
        """Test fetching instances from multiple zones"""
        # Arrange
        request = mock_http_request()

        # Mock instances in multiple zones
        mock_vm1 = MagicMock()
        mock_vm1.name = "instance-1"
        mock_vm1.status = "RUNNING"
        mock_vm1.machine_type = "zones/us-central1-a/machineTypes/n1-standard-1"

        mock_vm2 = MagicMock()
        mock_vm2.name = "instance-2"
        mock_vm2.status = "STOPPED"
        mock_vm2.machine_type = "zones/us-east1-b/machineTypes/n1-standard-2"

        mock_response = MagicMock()
        mock_response.items = {
            "zones/us-central1-a": MagicMock(instances=[mock_vm1]),
            "zones/us-east1-b": MagicMock(instances=[mock_vm2])
        }
        mock_compute_client.aggregated_list.return_value = mock_response

        # Act
        with patch('fetch_compute_instances.main.InstancesClient', return_value=mock_compute_client):
            with patch('fetch_compute_instances.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = compute_main.fetch_compute_instances(request)

        # Assert
        assert status_code == 200
        assert response["total_instances"] == 2
        assert response["running_instances"] == 1
        assert response["stopped_instances"] == 1

    def test_fetch_compute_instances_security_analysis(self, mock_compute_client, mock_bigquery_client, mock_http_request):
        """Test security analysis of compute instances"""
        # Arrange
        request = mock_http_request()

        # Create instance with external IP
        mock_vm = MagicMock()
        mock_vm.name = "exposed-instance"
        mock_vm.status = "RUNNING"

        # Mock network interface with external IP
        mock_network = MagicMock()
        mock_network.access_configs = [MagicMock(nat_i_p="35.123.45.67")]
        mock_vm.network_interfaces = [mock_network]

        # Mock service account with full scopes
        mock_sa = MagicMock()
        mock_sa.scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        mock_vm.service_accounts = [mock_sa]

        # Mock SSH keys in metadata
        mock_metadata_item = MagicMock()
        mock_metadata_item.key = "ssh-keys"
        mock_metadata_item.value = "user:ssh-rsa AAAAB3..."
        mock_vm.metadata = MagicMock(items=[mock_metadata_item])

        mock_response = MagicMock()
        mock_response.items = {"zones/us-central1-a": MagicMock(instances=[mock_vm])}
        mock_compute_client.aggregated_list.return_value = mock_response

        # Act
        with patch('fetch_compute_instances.main.InstancesClient', return_value=mock_compute_client):
            with patch('fetch_compute_instances.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = compute_main.fetch_compute_instances(request)

        # Assert
        assert status_code == 200
        # Verify security flags in BigQuery data
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["has_external_ip"] == True
        assert insert_call["has_full_api_access"] == True
        assert insert_call["ssh_keys_configured"] == True

    def test_fetch_compute_instances_no_instances(self, mock_compute_client, mock_bigquery_client, mock_http_request):
        """Test when no compute instances exist"""
        # Arrange
        request = mock_http_request()
        mock_response = MagicMock()
        mock_response.items = {}
        mock_compute_client.aggregated_list.return_value = mock_response

        # Act
        with patch('fetch_compute_instances.main.InstancesClient', return_value=mock_compute_client):
            with patch('fetch_compute_instances.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = compute_main.fetch_compute_instances(request)

        # Assert
        assert status_code == 200
        assert response["total_instances"] == 0

    def test_fetch_compute_instances_api_error(self, mock_compute_client, mock_http_request):
        """Test handling of Compute API errors"""
        # Arrange
        request = mock_http_request()
        mock_compute_client.aggregated_list.side_effect = Exception("Compute API Error")

        # Act
        with patch('fetch_compute_instances.main.InstancesClient', return_value=mock_compute_client):
            response, status_code = compute_main.fetch_compute_instances(request)

        # Assert
        assert status_code == 500
        assert response["status"] == "error"
        assert "Compute API Error" in response["message"]


class TestFetchFirewallRules:
    """Tests for fetch_firewall_rules Cloud Function"""

    def test_fetch_firewall_rules_success(self, mock_firewall_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of firewall rules"""
        # Arrange
        request = mock_http_request()

        # Act
        with patch('fetch_firewall_rules.main.FirewallsClient', return_value=mock_firewall_client):
            with patch('fetch_firewall_rules.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = firewall_main.fetch_firewall_rules(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_rules"] == 1
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_firewall_rules_security_analysis(self, mock_firewall_client, mock_bigquery_client, mock_http_request):
        """Test security analysis of firewall rules"""
        # Arrange
        request = mock_http_request()

        # Create overly permissive rule
        mock_rule = MagicMock()
        mock_rule.name = "allow-all"
        mock_rule.direction = "INGRESS"
        mock_rule.priority = 1000
        mock_rule.source_ranges = ["0.0.0.0/0"]  # Public access
        mock_rule.target_tags = []  # No target tags (applies to all)

        # Allow all protocols and ports
        mock_allowed = MagicMock()
        mock_allowed.i_p_protocol = "all"
        mock_allowed.ports = None  # All ports
        mock_rule.allowed = [mock_allowed]

        mock_firewall_client.list.return_value = [mock_rule]

        # Act
        with patch('fetch_firewall_rules.main.FirewallsClient', return_value=mock_firewall_client):
            with patch('fetch_firewall_rules.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = firewall_main.fetch_firewall_rules(request)

        # Assert
        assert status_code == 200
        assert response["overly_permissive_rules"] == 1
        # Verify risk assessment in BigQuery data
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["is_overly_permissive"] == True
        assert insert_call["allows_public_access"] == True
        assert insert_call["risk_level"] == "HIGH"

    def test_fetch_firewall_rules_common_ports_detection(self, mock_firewall_client, mock_bigquery_client, mock_http_request):
        """Test detection of common service ports"""
        # Arrange
        request = mock_http_request()

        rules = []
        # SSH rule
        ssh_rule = MagicMock()
        ssh_rule.name = "allow-ssh"
        ssh_rule.direction = "INGRESS"
        ssh_allowed = MagicMock()
        ssh_allowed.i_p_protocol = "tcp"
        ssh_allowed.ports = ["22"]
        ssh_rule.allowed = [ssh_allowed]
        ssh_rule.source_ranges = ["10.0.0.0/8"]
        rules.append(ssh_rule)

        # HTTP rule
        http_rule = MagicMock()
        http_rule.name = "allow-http"
        http_rule.direction = "INGRESS"
        http_allowed = MagicMock()
        http_allowed.i_p_protocol = "tcp"
        http_allowed.ports = ["80", "443"]
        http_rule.allowed = [http_allowed]
        http_rule.source_ranges = ["0.0.0.0/0"]
        rules.append(http_rule)

        # RDP rule
        rdp_rule = MagicMock()
        rdp_rule.name = "allow-rdp"
        rdp_rule.direction = "INGRESS"
        rdp_allowed = MagicMock()
        rdp_allowed.i_p_protocol = "tcp"
        rdp_allowed.ports = ["3389"]
        rdp_rule.allowed = [rdp_allowed]
        rdp_rule.source_ranges = ["0.0.0.0/0"]
        rules.append(rdp_rule)

        mock_firewall_client.list.return_value = rules

        # Act
        with patch('fetch_firewall_rules.main.FirewallsClient', return_value=mock_firewall_client):
            with patch('fetch_firewall_rules.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = firewall_main.fetch_firewall_rules(request)

        # Assert
        assert status_code == 200
        assert response["total_rules"] == 3
        assert response["ssh_exposed_rules"] == 1
        assert response["rdp_exposed_rules"] == 1
        assert response["http_exposed_rules"] == 1

    def test_fetch_firewall_rules_deny_rules(self, mock_firewall_client, mock_bigquery_client, mock_http_request):
        """Test handling of deny rules"""
        # Arrange
        request = mock_http_request()

        # Create deny rule
        mock_rule = MagicMock()
        mock_rule.name = "deny-all"
        mock_rule.direction = "INGRESS"
        mock_rule.priority = 65534  # Low priority deny rule
        mock_rule.source_ranges = ["0.0.0.0/0"]

        # Deny all traffic
        mock_denied = MagicMock()
        mock_denied.i_p_protocol = "all"
        mock_rule.denied = [mock_denied]
        mock_rule.allowed = []  # No allowed traffic

        mock_firewall_client.list.return_value = [mock_rule]

        # Act
        with patch('fetch_firewall_rules.main.FirewallsClient', return_value=mock_firewall_client):
            with patch('fetch_firewall_rules.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = firewall_main.fetch_firewall_rules(request)

        # Assert
        assert status_code == 200
        # Verify deny rule is processed correctly
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["action"] == "DENY"


class TestFetchStorageBuckets:
    """Tests for fetch_storage_buckets Cloud Function"""

    def test_fetch_storage_buckets_success(self, mock_storage_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of storage buckets"""
        # Arrange
        request = mock_http_request()

        # Act
        with patch('fetch_storage_buckets.main.storage.Client', return_value=mock_storage_client):
            with patch('fetch_storage_buckets.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = storage_main.fetch_storage_buckets(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_buckets"] == 1
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_storage_buckets_public_detection(self, mock_storage_client, mock_bigquery_client, mock_http_request):
        """Test detection of publicly accessible buckets"""
        # Arrange
        request = mock_http_request()

        # Create public bucket
        mock_bucket = MagicMock()
        mock_bucket.name = "public-bucket"

        # Mock public IAM policy
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {"role": "roles/storage.objectViewer", "members": ["allUsers"]},
            {"role": "roles/storage.legacyBucketReader", "members": ["allAuthenticatedUsers"]}
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy

        mock_storage_client.list_buckets.return_value = [mock_bucket]

        # Act
        with patch('fetch_storage_buckets.main.storage.Client', return_value=mock_storage_client):
            with patch('fetch_storage_buckets.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = storage_main.fetch_storage_buckets(request)

        # Assert
        assert status_code == 200
        assert response["public_buckets"] == 1
        # Verify public access flags in BigQuery data
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["is_public"] == True
        assert insert_call["public_access_type"] == "allUsers"
        assert insert_call["risk_level"] == "HIGH"

    def test_fetch_storage_buckets_security_features(self, mock_bigquery_client, mock_http_request):
        """Test evaluation of bucket security features"""
        # Arrange
        request = mock_http_request()

        # Create bucket with various security settings
        mock_bucket = MagicMock()
        mock_bucket.name = "secure-bucket"
        mock_bucket.versioning_enabled = True
        mock_bucket.default_kms_key_name = "projects/test/locations/global/keyRings/test/cryptoKeys/test-key"

        # Mock uniform bucket-level access
        mock_iam_config = MagicMock()
        mock_iam_config.uniform_bucket_level_access_enabled = True
        mock_bucket.iam_configuration = mock_iam_config

        # Mock logging configuration
        mock_logging = MagicMock()
        mock_logging.log_bucket = "audit-logs"
        mock_bucket.logging = mock_logging

        # Mock lifecycle rules
        mock_bucket.lifecycle_rules = [
            {"action": {"type": "Delete"}, "condition": {"age": 90}},
            {"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"}, "condition": {"age": 30}}
        ]

        # Private IAM policy
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {"role": "roles/storage.admin", "members": ["user:admin@example.com"]}
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy

        mock_storage_client = MagicMock()
        mock_storage_client.list_buckets.return_value = [mock_bucket]

        # Act
        with patch('fetch_storage_buckets.main.storage.Client', return_value=mock_storage_client):
            with patch('fetch_storage_buckets.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = storage_main.fetch_storage_buckets(request)

        # Assert
        assert status_code == 200
        assert response["encrypted_buckets"] == 1
        assert response["versioned_buckets"] == 1
        # Verify security features in BigQuery data
        insert_call = mock_bigquery_client.insert_rows_json.call_args[0][1][0]
        assert insert_call["versioning_enabled"] == True
        assert insert_call["encryption_type"] == "CMEK"
        assert insert_call["uniform_bucket_access"] == True
        assert insert_call["logging_enabled"] == True
        assert insert_call["lifecycle_rules_count"] == 2
        assert insert_call["risk_level"] == "LOW"

    def test_fetch_storage_buckets_no_buckets(self, mock_storage_client, mock_bigquery_client, mock_http_request):
        """Test when no storage buckets exist"""
        # Arrange
        request = mock_http_request()
        mock_storage_client.list_buckets.return_value = []

        # Act
        with patch('fetch_storage_buckets.main.storage.Client', return_value=mock_storage_client):
            with patch('fetch_storage_buckets.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = storage_main.fetch_storage_buckets(request)

        # Assert
        assert status_code == 200
        assert response["total_buckets"] == 0


class TestFetchSecurityFindings:
    """Tests for fetch_security_findings Cloud Function"""

    def test_fetch_security_findings_success(self, mock_scc_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of security findings"""
        # Arrange
        request = mock_http_request()

        # Act
        with patch('fetch_security_findings.main.SecurityCenterClient', return_value=mock_scc_client):
            with patch('fetch_security_findings.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = security_findings_main.fetch_security_findings(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_findings"] == 1
        assert response["active_findings"] == 1
        assert mock_bigquery_client.insert_rows_json.called

    def test_fetch_security_findings_severity_distribution(self, mock_bigquery_client, mock_http_request):
        """Test severity distribution of findings"""
        # Arrange
        request = mock_http_request()

        findings = []
        severities = ["CRITICAL", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "MEDIUM", "LOW"]
        categories = ["PUBLIC_BUCKET", "OPEN_FIREWALL", "WEAK_ENCRYPTION", "LOGGING_DISABLED"]

        for i, severity in enumerate(severities):
            mock_finding = MagicMock()
            mock_finding.name = f"finding-{i}"
            mock_finding.severity = severity
            mock_finding.state = "ACTIVE"
            mock_finding.category = categories[i % len(categories)]
            mock_finding.resource_name = f"//compute.googleapis.com/instance-{i}"
            mock_finding.finding_class = "VULNERABILITY"
            mock_finding.event_time = datetime.utcnow()
            findings.append(MagicMock(finding=mock_finding))

        mock_scc_client = MagicMock()
        mock_scc_client.list_findings.return_value = MagicMock(finding_result_list=findings)

        # Act
        with patch('fetch_security_findings.main.SecurityCenterClient', return_value=mock_scc_client):
            with patch('fetch_security_findings.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = security_findings_main.fetch_security_findings(request)

        # Assert
        assert status_code == 200
        assert response["total_findings"] == 7
        assert response["critical_findings"] == 1
        assert response["high_findings"] == 2
        assert response["medium_findings"] == 3
        assert response["low_findings"] == 1

    def test_fetch_security_findings_by_resource_type(self, mock_bigquery_client, mock_http_request):
        """Test grouping findings by resource type"""
        # Arrange
        request = mock_http_request()

        findings = []
        resources = [
            "//storage.googleapis.com/bucket-1",
            "//storage.googleapis.com/bucket-2",
            "//compute.googleapis.com/instance-1",
            "//compute.googleapis.com/instance-2",
            "//compute.googleapis.com/instance-3",
            "//iam.googleapis.com/serviceAccount-1"
        ]

        for i, resource in enumerate(resources):
            mock_finding = MagicMock()
            mock_finding.name = f"finding-{i}"
            mock_finding.resource_name = resource
            mock_finding.severity = "HIGH"
            mock_finding.state = "ACTIVE"
            mock_finding.category = "MISCONFIGURATION"
            mock_finding.finding_class = "VULNERABILITY"
            mock_finding.event_time = datetime.utcnow()
            findings.append(MagicMock(finding=mock_finding))

        mock_scc_client = MagicMock()
        mock_scc_client.list_findings.return_value = MagicMock(finding_result_list=findings)

        # Act
        with patch('fetch_security_findings.main.SecurityCenterClient', return_value=mock_scc_client):
            with patch('fetch_security_findings.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = security_findings_main.fetch_security_findings(request)

        # Assert
        assert status_code == 200
        assert "resource_distribution" in response
        assert response["resource_distribution"]["storage"] == 2
        assert response["resource_distribution"]["compute"] == 3
        assert response["resource_distribution"]["iam"] == 1

    def test_fetch_security_findings_inactive_findings(self, mock_bigquery_client, mock_http_request):
        """Test handling of inactive/resolved findings"""
        # Arrange
        request = mock_http_request()

        findings = []
        states = ["ACTIVE", "ACTIVE", "INACTIVE", "INACTIVE", "INACTIVE"]

        for i, state in enumerate(states):
            mock_finding = MagicMock()
            mock_finding.name = f"finding-{i}"
            mock_finding.state = state
            mock_finding.severity = "MEDIUM"
            mock_finding.category = "MISCONFIGURATION"
            mock_finding.resource_name = f"//compute.googleapis.com/instance-{i}"
            mock_finding.finding_class = "VULNERABILITY"
            mock_finding.event_time = datetime.utcnow()
            findings.append(MagicMock(finding=mock_finding))

        mock_scc_client = MagicMock()
        mock_scc_client.list_findings.return_value = MagicMock(finding_result_list=findings)

        # Act
        with patch('fetch_security_findings.main.SecurityCenterClient', return_value=mock_scc_client):
            with patch('fetch_security_findings.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = security_findings_main.fetch_security_findings(request)

        # Assert
        assert status_code == 200
        assert response["total_findings"] == 5
        assert response["active_findings"] == 2
        assert response["inactive_findings"] == 3


class TestFetchIAMAccounts:
    """Tests for fetch_iam_accounts Cloud Function (deprecated/special function)"""

    def test_fetch_iam_accounts_success(self, mock_crm_client, mock_iam_service_client, mock_bigquery_client, mock_http_request):
        """Test successful fetch of all IAM accounts"""
        # Arrange
        request = mock_http_request()

        # Mock IAM policy with mixed account types
        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/editor"
        mock_binding.members = [
            "user:user1@example.com",
            "user:user2@external.com",
            "serviceAccount:sa1@test-project-123.iam.gserviceaccount.com",
            "group:admins@example.com"
        ]
        mock_policy.bindings = [mock_binding]
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_iam_accounts.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_iam_accounts.main.IAMClient', return_value=mock_iam_service_client):
                with patch('fetch_iam_accounts.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = iam_accounts_main.fetch_iam_accounts(request)

        # Assert
        assert status_code == 200
        assert response["status"] == "success"
        assert response["total_accounts"] == 4
        assert response["user_accounts"] == 2
        assert response["service_accounts"] == 1
        assert response["group_accounts"] == 1

    def test_fetch_iam_accounts_account_types(self, mock_crm_client, mock_iam_service_client, mock_bigquery_client, mock_http_request):
        """Test identification of different account types"""
        # Arrange
        request = mock_http_request()

        # Mock various account types
        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/viewer"
        mock_binding.members = [
            "user:regular@example.com",
            "serviceAccount:app@test.iam.gserviceaccount.com",
            "serviceAccount:123456@cloudservices.gserviceaccount.com",  # Google-managed
            "group:team@example.com",
            "domain:example.com",
            "allUsers",
            "allAuthenticatedUsers"
        ]
        mock_policy.bindings = [mock_binding]
        mock_crm_client.get_iam_policy.return_value = mock_policy

        # Act
        with patch('fetch_iam_accounts.main.ProjectsClient', return_value=mock_crm_client):
            with patch('fetch_iam_accounts.main.IAMClient', return_value=mock_iam_service_client):
                with patch('fetch_iam_accounts.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = iam_accounts_main.fetch_iam_accounts(request)

        # Assert
        assert status_code == 200
        assert response["total_accounts"] == 7
        assert response["special_accounts"] == 2  # allUsers, allAuthenticatedUsers
        assert response["domain_accounts"] == 1
        # Verify account types in BigQuery data
        insert_calls = mock_bigquery_client.insert_rows_json.call_args[0][1]
        account_types = [call["account_type"] for call in insert_calls]
        assert "USER" in account_types
        assert "SERVICE_ACCOUNT" in account_types
        assert "GROUP" in account_types
        assert "DOMAIN" in account_types
        assert "SPECIAL" in account_types


class TestInfrastructureFunctionsPerformance:
    """Performance tests for infrastructure Cloud Functions"""

    def test_large_instance_dataset(self, mock_bigquery_client, mock_http_request, performance_timer, large_dataset):
        """Test handling of large number of compute instances"""
        # Arrange
        request = mock_http_request()

        # Create 500 mock instances across multiple zones
        zones = ["us-central1-a", "us-central1-b", "us-east1-a", "us-east1-b", "europe-west1-a"]
        mock_instances = {}

        for zone in zones:
            instances = []
            for i in range(100):  # 100 instances per zone
                mock_vm = MagicMock()
                mock_vm.name = f"instance-{zone}-{i}"
                mock_vm.status = "RUNNING" if i % 2 == 0 else "STOPPED"
                mock_vm.machine_type = f"zones/{zone}/machineTypes/n1-standard-1"
                mock_vm.network_interfaces = [MagicMock(access_configs=[])]
                mock_vm.service_accounts = []
                mock_vm.metadata = MagicMock(items=[])
                instances.append(mock_vm)
            mock_instances[f"zones/{zone}"] = MagicMock(instances=instances)

        mock_compute_client = MagicMock()
        mock_response = MagicMock()
        mock_response.items = mock_instances
        mock_compute_client.aggregated_list.return_value = mock_response

        # Act
        performance_timer.start()
        with patch('fetch_compute_instances.main.InstancesClient', return_value=mock_compute_client):
            with patch('fetch_compute_instances.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = compute_main.fetch_compute_instances(request)
        performance_timer.stop()

        # Assert
        assert status_code == 200
        assert response["total_instances"] == 500
        assert performance_timer.elapsed < 5  # Should complete within 5 seconds

    def test_concurrent_infrastructure_functions(self, mock_compute_client, mock_firewall_client,
                                                mock_storage_client, mock_scc_client,
                                                mock_bigquery_client, mock_http_request):
        """Test concurrent execution of all infrastructure functions"""
        import threading
        import time

        # Arrange
        request = mock_http_request()
        results = []
        errors = []

        def run_function(func_name, func, mock_client_name, mock_client):
            try:
                with patch(f'{func.__module__}.{mock_client_name}', return_value=mock_client):
                    with patch(f'{func.__module__}.bigquery.Client', return_value=mock_bigquery_client):
                        response, status = func(request)
                        results.append((func_name, status))
            except Exception as e:
                errors.append((func_name, str(e)))

        # Act - Run all infrastructure functions concurrently
        threads = [
            threading.Thread(target=run_function,
                           args=("compute", compute_main.fetch_compute_instances,
                                "InstancesClient", mock_compute_client)),
            threading.Thread(target=run_function,
                           args=("firewall", firewall_main.fetch_firewall_rules,
                                "FirewallsClient", mock_firewall_client)),
            threading.Thread(target=run_function,
                           args=("storage", storage_main.fetch_storage_buckets,
                                "storage.Client", mock_storage_client)),
            threading.Thread(target=run_function,
                           args=("findings", security_findings_main.fetch_security_findings,
                                "SecurityCenterClient", mock_scc_client))
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # Assert
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 4
        for func_name, status in results:
            assert status == 200, f"{func_name} failed with status {status}"

    def test_bigquery_batch_insert_performance(self, mock_bigquery_client, mock_http_request, performance_timer, large_dataset):
        """Test BigQuery batch insert performance"""
        # Arrange
        request = mock_http_request()

        # Generate large dataset
        large_data = large_dataset(size=10000)

        # Mock function that would insert large dataset
        def mock_large_insert(request):
            try:
                # Simulate batch insert
                batch_size = 500
                for i in range(0, len(large_data), batch_size):
                    batch = large_data[i:i+batch_size]
                    mock_bigquery_client.insert_rows_json(None, batch)
                return {"status": "success", "rows_inserted": len(large_data)}, 200
            except Exception as e:
                return {"status": "error", "message": str(e)}, 500

        # Act
        performance_timer.start()
        response, status_code = mock_large_insert(request)
        performance_timer.stop()

        # Assert
        assert status_code == 200
        assert response["rows_inserted"] == 10000
        assert performance_timer.elapsed < 10  # Should complete within 10 seconds
        # Verify batching occurred
        assert mock_bigquery_client.insert_rows_json.call_count == 20  # 10000/500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])