"""
Contract test for GET /health/database endpoint.
This test MUST FAIL until the implementation is added.
"""

import pytest
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
ENDPOINT = "/health/database"


class TestDatabaseHealthContract:
    """Test the database health check endpoint contract."""

    def test_endpoint_exists(self):
        """Test that the health endpoint exists."""
        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)
        # Should not return 404
        assert response.status_code != 404, "Health endpoint does not exist"

    def test_health_check_response_schema(self):
        """Test that health check returns expected schema."""
        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)

        # Should return 200 or 503 based on database status
        assert response.status_code in [200, 503], \
            f"Expected 200 or 503, got {response.status_code}"

        data = response.json()

        # Required fields
        assert "status" in data, "Response must contain 'status' field"
        assert "database_path" in data, "Response must contain 'database_path' field"

        # Status must be one of the valid values
        assert data["status"] in ["healthy", "degraded", "unavailable"], \
            f"Invalid status: {data['status']}"

        # Database path must be absolute
        assert Path(data["database_path"]).is_absolute(), \
            "Database path must be absolute"

    def test_healthy_database_response(self):
        """Test response when database is healthy."""
        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)

        if response.status_code == 200:
            data = response.json()

            assert data["status"] == "healthy", "Status should be 'healthy' for 200 response"
            assert "exists" in data, "Response must contain 'exists' field"
            assert "readable" in data, "Response must contain 'readable' field"

            if data["exists"]:
                assert data["readable"] is True, "Existing database should be readable"
                assert "table_count" in data, "Response must contain 'table_count'"
                assert isinstance(data["table_count"], int), "'table_count' must be integer"
                assert data["table_count"] > 0, "Database should have tables"

    def test_unhealthy_database_response(self):
        """Test response when database is unhealthy."""
        # Temporarily break the database path to test unhealthy state
        # This would be done by setting wrong DATABASE_PATH env var
        # For now, we just validate the schema if status is not healthy

        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)
        data = response.json()

        if data["status"] != "healthy":
            assert response.status_code == 503, \
                "Unhealthy database should return 503"
            assert "error" in data or "status_message" in data, \
                "Unhealthy response should contain error details"

    def test_database_info_completeness(self):
        """Test that database info is complete when healthy."""
        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Optional but recommended fields
            recommended_fields = [
                "exists",
                "readable",
                "table_count",
                "total_records",
                "tables"
            ]

            for field in recommended_fields:
                if field in data:
                    if field == "tables":
                        assert isinstance(data[field], list), f"'{field}' must be a list"
                    elif field in ["table_count", "total_records"]:
                        assert isinstance(data[field], int), f"'{field}' must be integer"
                    elif field in ["exists", "readable"]:
                        assert isinstance(data[field], bool), f"'{field}' must be boolean"

    def test_database_path_resolution(self):
        """Test that database path is correctly resolved."""
        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)
        data = response.json()

        db_path = data["database_path"]

        # Path should contain expected structure
        assert "backend/cache" in db_path or "gcp_data.db" in db_path, \
            f"Unexpected database path: {db_path}"

        # Path should be absolute
        assert db_path.startswith("/") or db_path[1:3] == ":\\", \
            f"Path should be absolute: {db_path}"

    def test_performance_requirement(self):
        """Test that health check responds quickly."""
        import time

        start = time.time()
        response = requests.get(f"{BASE_URL}{ENDPOINT}", timeout=5)
        duration = time.time() - start

        assert response.status_code in [200, 503]
        # Health check should be fast (under 1 second)
        assert duration < 1.0, f"Health check took {duration:.2f}s, should be < 1s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])