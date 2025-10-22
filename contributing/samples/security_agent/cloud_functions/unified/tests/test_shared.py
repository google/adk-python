"""
Tests for shared utilities
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

from shared.config import Config
from shared.response import create_response, create_error_response, create_success_response
from shared.bigquery_utils import ensure_dataset_exists, ensure_table_exists, insert_rows_batch


class TestConfig:
    """Tests for Config class"""

    def test_default_config(self):
        """Test default configuration values"""
        assert Config.PROJECT_ID == 'test-project'  # Set in conftest
        assert Config.BQ_DATASET_ID == 'test_dataset'
        assert Config.BQ_LOCATION == 'us-central1'
        assert Config.ENABLE_SAMPLE_DATA is True

    def test_get_table_id(self):
        """Test table ID generation"""
        table_id = Config.get_table_id('test_table')
        assert table_id == 'test-project.test_dataset.test_table'

        # Test with custom dataset
        table_id = Config.get_table_id('test_table', 'custom_dataset')
        assert table_id == 'test-project.custom_dataset.test_table'

    def test_validate(self):
        """Test configuration validation"""
        # Should not raise with PROJECT_ID set
        Config.validate()

        # Test missing required variable
        original = Config.PROJECT_ID
        Config.PROJECT_ID = None
        with pytest.raises(ValueError) as exc:
            Config.validate()
        assert "Missing required environment variables" in str(exc.value)
        Config.PROJECT_ID = original


class TestResponseUtils:
    """Tests for response utilities"""

    def test_create_response(self):
        """Test response creation"""
        data = {"test": "value"}
        response, status, headers = create_response(data)

        response_data = json.loads(response)
        assert response_data['test'] == 'value'
        assert 'timestamp' in response_data
        assert status == 200
        assert headers['Content-Type'] == 'application/json'

    def test_create_error_response(self):
        """Test error response creation"""
        error = ValueError("Test error")
        response, status, headers = create_error_response(error, 400)

        response_data = json.loads(response)
        assert response_data['error'] == 'Test error'
        assert response_data['error_type'] == 'ValueError'
        assert response_data['status'] == 'error'
        assert status == 400

    def test_create_success_response(self):
        """Test success response creation"""
        response, status, headers = create_success_response(
            "Operation completed",
            data={"count": 10},
            metadata={"duration": 1.5}
        )

        response_data = json.loads(response)
        assert response_data['status'] == 'success'
        assert response_data['message'] == 'Operation completed'
        assert response_data['data']['count'] == 10
        assert response_data['metadata']['duration'] == 1.5
        assert status == 200


class TestBigQueryUtils:
    """Tests for BigQuery utilities"""

    @patch('google.cloud.bigquery.Client')
    def test_ensure_dataset_exists_creates_new(self, mock_client_class):
        """Test dataset creation when it doesn't exist"""
        from google.cloud.exceptions import NotFound

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_dataset.side_effect = NotFound("Dataset not found")
        mock_client.create_dataset.return_value = MagicMock()

        dataset = ensure_dataset_exists(mock_client, "test_dataset")

        mock_client.create_dataset.assert_called_once()
        assert dataset is not None

    @patch('google.cloud.bigquery.Client')
    def test_ensure_dataset_exists_returns_existing(self, mock_client_class):
        """Test dataset returns existing when found"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_dataset = MagicMock()
        mock_client.get_dataset.return_value = mock_dataset

        dataset = ensure_dataset_exists(mock_client, "test_dataset")

        mock_client.get_dataset.assert_called_once()
        mock_client.create_dataset.assert_not_called()
        assert dataset == mock_dataset

    @patch('google.cloud.bigquery.Client')
    def test_ensure_table_exists_creates_new(self, mock_client_class):
        """Test table creation when it doesn't exist"""
        from google.cloud.exceptions import NotFound
        from google.cloud import bigquery

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_table.side_effect = NotFound("Table not found")
        mock_client.create_table.return_value = MagicMock()

        schema = [
            bigquery.SchemaField("id", "STRING"),
            bigquery.SchemaField("name", "STRING")
        ]

        table = ensure_table_exists(mock_client, "project.dataset.table", schema)

        mock_client.create_table.assert_called_once()
        assert table is not None

    @patch('google.cloud.bigquery.Client')
    def test_insert_rows_batch(self, mock_client_class):
        """Test batch row insertion"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.insert_rows_json.return_value = []  # No errors

        rows = [
            {"id": "1", "name": "test1"},
            {"id": "2", "name": "test2"}
        ]

        errors = insert_rows_batch(mock_client, "project.dataset.table", rows)

        assert errors == []
        mock_client.insert_rows_json.assert_called_once()

    @patch('google.cloud.bigquery.Client')
    def test_insert_rows_batch_with_large_data(self, mock_client_class):
        """Test batch insertion with large dataset"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.insert_rows_json.return_value = []

        # Create large dataset
        rows = [{"id": str(i), "name": f"test{i}"} for i in range(1500)]

        errors = insert_rows_batch(mock_client, "project.dataset.table", rows, max_batch_size=500)

        # Should be called 3 times (1500 / 500 = 3)
        assert mock_client.insert_rows_json.call_count == 3


class TestAuthUtils:
    """Tests for authentication utilities"""

    @patch('google.auth.default')
    def test_get_authenticated_client_bigquery(self, mock_default):
        """Test getting BigQuery client"""
        from shared.auth import get_authenticated_client

        mock_default.return_value = (MagicMock(), 'test-project')

        with patch('google.cloud.bigquery.Client') as mock_client:
            client = get_authenticated_client('bigquery')
            mock_client.assert_called_once()

    @patch('google.auth.default')
    def test_get_authenticated_client_unsupported(self, mock_default):
        """Test unsupported service raises error"""
        from shared.auth import get_authenticated_client

        mock_default.return_value = (MagicMock(), 'test-project')

        with pytest.raises(ValueError) as exc:
            get_authenticated_client('unsupported_service')
        assert "Unsupported service" in str(exc.value)