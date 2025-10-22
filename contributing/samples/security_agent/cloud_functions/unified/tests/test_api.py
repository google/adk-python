"""
Tests for FastAPI application endpoints
"""

import pytest
import json
from unittest.mock import patch, MagicMock


class TestAPIEndpoints:
    """Tests for API endpoints"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns service info"""
        response = test_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data['service'] == "Unified Security Data Fetchers"
        assert data['version'] == "2.0.0"
        assert data['status'] == "healthy"
        assert 'endpoints' in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == "healthy"
        assert 'checks' in data
        assert data['checks']['config'] == "valid"
        assert data['checks']['project_id'] == "test-project"

    def test_list_fetchers(self, test_client):
        """Test list fetchers endpoint"""
        response = test_client.get("/fetchers")
        assert response.status_code == 200

        data = response.json()
        assert 'fetchers' in data
        assert 'total' in data
        assert data['total'] > 0

        # Check fetcher structure
        for fetcher in data['fetchers']:
            assert 'name' in fetcher
            assert 'table' in fetcher
            assert 'dataset' in fetcher
            assert 'endpoint' in fetcher

    @patch('app.main.FETCHERS_REGISTRY')
    def test_fetch_specific_fetcher(self, mock_registry, test_client):
        """Test fetching data with specific fetcher"""
        # Mock fetcher
        mock_fetcher_class = MagicMock()
        mock_fetcher_instance = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher_instance
        mock_fetcher_instance.run.return_value = {
            'status': 'success',
            'message': 'Test successful',
            'records_processed': 10,
            'table': 'test_table'
        }

        mock_registry.__getitem__.return_value = mock_fetcher_class
        mock_registry.__contains__.return_value = True

        response = test_client.post("/fetch/test_fetcher")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert data['fetcher'] == 'test_fetcher'
        assert data['records_processed'] == 10

    def test_fetch_nonexistent_fetcher(self, test_client):
        """Test fetching with non-existent fetcher returns 404"""
        response = test_client.post("/fetch/nonexistent_fetcher")
        assert response.status_code == 404

        data = response.json()
        assert 'detail' in data
        assert 'nonexistent_fetcher' in data['detail']

    @patch('app.main.FETCHERS_REGISTRY')
    def test_fetch_all(self, mock_registry, test_client):
        """Test fetching all data"""
        # Create mock fetchers
        mock_fetcher1 = MagicMock()
        mock_fetcher1.run.return_value = {
            'status': 'success',
            'records_processed': 5
        }

        mock_fetcher2 = MagicMock()
        mock_fetcher2.run.return_value = {
            'status': 'success',
            'records_processed': 10
        }

        mock_registry.items.return_value = [
            ('fetcher1', lambda: mock_fetcher1),
            ('fetcher2', lambda: mock_fetcher2)
        ]

        response = test_client.post("/fetch/all")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'completed'
        assert data['summary']['total_fetchers'] == 2
        assert data['summary']['total_records'] == 15

    def test_trigger_endpoints_exist(self, test_client):
        """Test that Cloud Scheduler trigger endpoints exist"""
        # Test a few known fetchers have trigger endpoints
        fetchers_to_test = [
            'security_findings',
            'custom_roles',
            'compute_instances'
        ]

        for fetcher in fetchers_to_test:
            response = test_client.get(f"/trigger/{fetcher}")
            # Should return 200 or handle the request
            assert response.status_code in [200, 500]  # May error without full setup

    @patch('app.main.FETCHERS_REGISTRY')
    def test_error_handling(self, mock_registry, test_client):
        """Test error handling in fetch endpoint"""
        # Mock fetcher that raises an exception
        mock_fetcher_class = MagicMock()
        mock_fetcher_instance = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher_instance
        mock_fetcher_instance.run.side_effect = Exception("Test error")

        mock_registry.__getitem__.return_value = mock_fetcher_class
        mock_registry.__contains__.return_value = True

        response = test_client.post("/fetch/error_fetcher")
        assert response.status_code == 200  # Returns 200 with error status

        data = response.json()
        assert data['status'] == 'error'
        assert 'Test error' in data['message']
        assert data['records_processed'] == 0


class TestRequestModels:
    """Test request/response models"""

    def test_fetch_request_model(self):
        """Test FetchRequest model"""
        from app.main import FetchRequest

        # Test with defaults
        request = FetchRequest()
        assert request.fetcher is None
        assert request.async_mode is False
        assert request.force_refresh is False

        # Test with values
        request = FetchRequest(
            fetcher="test_fetcher",
            async_mode=True,
            force_refresh=True
        )
        assert request.fetcher == "test_fetcher"
        assert request.async_mode is True
        assert request.force_refresh is True

    def test_fetch_response_model(self):
        """Test FetchResponse model"""
        from app.main import FetchResponse
        from datetime import datetime

        response = FetchResponse(
            status="success",
            message="Test message",
            fetcher="test_fetcher",
            records_processed=100,
            table="test_table",
            timestamp=datetime.utcnow().isoformat()
        )

        assert response.status == "success"
        assert response.message == "Test message"
        assert response.fetcher == "test_fetcher"
        assert response.records_processed == 100
        assert response.table == "test_table"
