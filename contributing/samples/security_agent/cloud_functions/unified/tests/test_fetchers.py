"""
Tests for fetcher modules
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

from fetchers.security_findings import SecurityFindingsFetcher
from fetchers.custom_roles import CustomRolesFetcher
from fetchers.compute_instances import ComputeInstancesFetcher


class TestSecurityFindingsFetcher:
    """Tests for SecurityFindingsFetcher"""

    def test_table_name(self):
        """Test table name property"""
        fetcher = SecurityFindingsFetcher()
        assert fetcher.table_name == 'security_findings'

    def test_schema(self):
        """Test schema property"""
        fetcher = SecurityFindingsFetcher()
        schema = fetcher.schema
        assert len(schema) > 0
        assert any(field.name == 'finding_id' for field in schema)
        assert any(field.name == 'severity' for field in schema)

    @patch('fetchers.security_findings.get_authenticated_client')
    def test_fetch_data_with_sample(self, mock_get_client):
        """Test fetch_data returns sample data when enabled"""
        fetcher = SecurityFindingsFetcher()

        # Mock client to raise exception (triggers sample data)
        mock_get_client.side_effect = Exception("Test error")

        data = fetcher.fetch_data()
        assert len(data) > 0
        assert 'finding_id' in data[0]
        assert data[0]['severity'] in ['HIGH', 'CRITICAL', 'MEDIUM', 'LOW']

    def test_get_sample_data(self):
        """Test sample data generation"""
        fetcher = SecurityFindingsFetcher()
        sample_data = fetcher.get_sample_data()

        assert len(sample_data) > 0
        assert sample_data[0]['finding_id'] == 'sample-finding-001'
        assert sample_data[0]['state'] == 'ACTIVE'

    @patch('fetchers.base.get_bq_client')
    @patch('fetchers.base.ensure_dataset_exists')
    @patch('fetchers.base.ensure_table_exists')
    @patch('fetchers.base.insert_rows_batch')
    def test_run_with_sample_data(self, mock_insert, mock_ensure_table,
                                   mock_ensure_dataset, mock_get_client):
        """Test full run with sample data"""
        # Setup mocks
        mock_get_client.return_value = MagicMock()
        mock_insert.return_value = []  # No errors

        fetcher = SecurityFindingsFetcher()
        result = fetcher.run()

        assert result['status'] == 'success'
        assert result['records_processed'] > 0
        assert 'table' in result

        # Verify infrastructure setup was called
        mock_ensure_dataset.assert_called_once()
        mock_ensure_table.assert_called_once()
        mock_insert.assert_called_once()


class TestCustomRolesFetcher:
    """Tests for CustomRolesFetcher"""

    def test_table_name(self):
        """Test table name property"""
        fetcher = CustomRolesFetcher()
        assert fetcher.table_name == 'iam_custom_roles'

    def test_risk_calculation(self):
        """Test permission risk calculation"""
        fetcher = CustomRolesFetcher()

        # Test high risk permissions
        assert fetcher._is_high_risk_permission("iam.roles.delete")
        assert fetcher._is_high_risk_permission("compute.instances.setIamPolicy")
        assert fetcher._is_high_risk_permission("storage.buckets.admin")

        # Test non-high risk permissions
        assert not fetcher._is_high_risk_permission("compute.instances.list")
        assert not fetcher._is_high_risk_permission("storage.objects.get")

    def test_risk_level_calculation(self):
        """Test overall risk level calculation"""
        fetcher = CustomRolesFetcher()

        # Test different risk levels
        assert fetcher._calculate_risk_level([], 0) == "LOW"
        assert fetcher._calculate_risk_level(["perm1", "perm2"], 1) == "MEDIUM"
        assert fetcher._calculate_risk_level(["perm1", "perm2", "perm3"], 3) == "HIGH"
        assert fetcher._calculate_risk_level(["p1", "p2", "p3", "p4", "p5", "p6"], 6) == "CRITICAL"

    def test_similar_roles_finder(self):
        """Test finding similar predefined roles"""
        fetcher = CustomRolesFetcher()

        permissions = [
            "bigquery.datasets.get",
            "bigquery.tables.list",
            "bigquery.jobs.create"
        ]

        similar = fetcher._find_similar_predefined_roles(permissions)
        assert len(similar) <= 3
        if similar:
            assert 'role' in similar[0]
            assert 'similarity_percentage' in similar[0]


class TestComputeInstancesFetcher:
    """Tests for ComputeInstancesFetcher"""

    def test_table_name(self):
        """Test table name property"""
        fetcher = ComputeInstancesFetcher()
        assert fetcher.table_name == 'compute_instances'

    def test_schema(self):
        """Test schema property"""
        fetcher = ComputeInstancesFetcher()
        schema = fetcher.schema
        assert len(schema) > 0
        assert any(field.name == 'instance_id' for field in schema)
        assert any(field.name == 'machine_type' for field in schema)

    def test_get_sample_data(self):
        """Test sample data generation"""
        fetcher = ComputeInstancesFetcher()
        sample_data = fetcher.get_sample_data()

        assert len(sample_data) > 0
        instance = sample_data[0]
        assert instance['instance_id'] == 'sample-instance-001'
        assert instance['status'] == 'RUNNING'
        assert 'network_interfaces' in instance

        # Verify JSON fields are properly formatted
        network_interfaces = json.loads(instance['network_interfaces'])
        assert isinstance(network_interfaces, list)
        assert len(network_interfaces) > 0


class TestBaseFetcher:
    """Tests for BaseFetcher functionality"""

    @patch('fetchers.base.get_bq_client')
    def test_prepare_data(self, mock_get_client):
        """Test data preparation adds common fields"""
        from fetchers.security_findings import SecurityFindingsFetcher

        fetcher = SecurityFindingsFetcher()
        raw_data = [
            {"finding_id": "test-001"},
            {"finding_id": "test-002"}
        ]

        prepared = fetcher.prepare_data(raw_data)

        for record in prepared:
            assert 'ingestion_time' in record
            assert 'project_id' in record
            assert record['project_id'] == 'test-project'

    @patch('fetchers.base.get_bq_client')
    def test_validate_data(self, mock_get_client):
        """Test data validation"""
        from fetchers.security_findings import SecurityFindingsFetcher

        fetcher = SecurityFindingsFetcher()

        # Valid data
        valid_data = [
            {"finding_id": "test-001", "name": "test"}
        ]
        assert fetcher.validate_data(valid_data) is True

        # Invalid data - missing required field
        invalid_data = [
            {"name": "test"}  # Missing finding_id
        ]
        assert fetcher.validate_data(invalid_data) is False

        # Empty data
        assert fetcher.validate_data([]) is False