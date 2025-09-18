"""
Integration test for database path resolution.
This test MUST FAIL until the implementation is fixed.
"""

import pytest
import os
import sys
from pathlib import Path
import tempfile
import sqlite3

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.database import (
    get_database_path,
    validate_database,
    get_db_connection,
    create_database_if_missing,
    get_database_info
)


class TestDatabasePathResolution:
    """Test database path resolution across different contexts."""

    def test_absolute_path_resolution(self):
        """Test that database path is always resolved to absolute."""
        db_path = get_database_path()

        assert db_path.is_absolute(), f"Path should be absolute: {db_path}"
        assert str(db_path).startswith("/") or str(db_path)[1:3] == ":\\", \
            "Path should start with / (Unix) or drive letter (Windows)"

    def test_environment_variable_priority(self):
        """Test that DATABASE_PATH env var takes priority."""
        # Save original
        original = os.environ.get("DATABASE_PATH")

        try:
            # Set temporary path
            test_path = "/tmp/test_database.db"
            os.environ["DATABASE_PATH"] = test_path

            db_path = get_database_path()
            assert str(db_path) == test_path, \
                f"Should use env var path: {test_path}, got {db_path}"

        finally:
            # Restore original
            if original:
                os.environ["DATABASE_PATH"] = original
            else:
                os.environ.pop("DATABASE_PATH", None)

    def test_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        original = os.environ.get("DATABASE_PATH")

        try:
            # Set relative path
            os.environ["DATABASE_PATH"] = "backend/cache/test.db"

            db_path = get_database_path()

            # Should be absolute now
            assert db_path.is_absolute(), "Relative path should be made absolute"
            assert "backend/cache" in str(db_path), \
                "Should preserve relative structure"

        finally:
            if original:
                os.environ["DATABASE_PATH"] = original
            else:
                os.environ.pop("DATABASE_PATH", None)

    def test_database_validation(self):
        """Test database validation function."""
        # Test with actual database
        is_valid, message = validate_database()

        # If database exists, it should be valid
        db_path = get_database_path()
        if db_path.exists():
            assert is_valid is True, f"Existing database should be valid: {message}"
        else:
            assert is_valid is False, "Non-existent database should be invalid"
            assert "does not exist" in message, "Should indicate missing database"

    def test_database_connection_context_manager(self):
        """Test the database connection context manager."""
        # Create a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Use connection context manager
            with get_db_connection(tmp_path) as conn:
                assert conn is not None, "Connection should not be None"
                assert isinstance(conn, sqlite3.Connection), \
                    "Should return sqlite3 Connection"

                # Test that row factory is set
                cursor = conn.cursor()
                cursor.execute(
                    "CREATE TABLE test (id INTEGER, name TEXT)"
                )
                cursor.execute(
                    "INSERT INTO test VALUES (1, 'test')"
                )
                cursor.execute("SELECT * FROM test")
                row = cursor.fetchone()

                # Row factory should allow dict-like access
                assert row["id"] == 1, "Row factory should allow dict access"

        finally:
            # Clean up
            tmp_path.unlink()

    def test_create_database_if_missing(self):
        """Test automatic database creation."""
        # Use a temporary path
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_create.db"
            original = os.environ.get("DATABASE_PATH")

            try:
                os.environ["DATABASE_PATH"] = str(tmp_path)

                # Database should not exist yet
                assert not tmp_path.exists()

                # Create database
                created = create_database_if_missing()
                assert created is True, "Should return True when creating"
                assert tmp_path.exists(), "Database file should be created"

                # Call again - should not create
                created = create_database_if_missing()
                assert created is False, "Should return False when exists"

                # Verify structure
                with get_db_connection(tmp_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    tables = [row[0] for row in cursor.fetchall()]

                    expected_tables = [
                        "security_findings",
                        "assets",
                        "query_logs",
                        "session_state"
                    ]

                    for table in expected_tables:
                        assert table in tables, f"Table '{table}' should exist"

            finally:
                if original:
                    os.environ["DATABASE_PATH"] = original
                else:
                    os.environ.pop("DATABASE_PATH", None)

    def test_database_info_retrieval(self):
        """Test getting database information."""
        info = get_database_info()

        assert "database_path" in info, "Info should contain database path"
        assert "exists" in info, "Info should contain exists flag"
        assert "readable" in info, "Info should contain readable flag"
        assert "status_message" in info, "Info should contain status message"

        if info["exists"] and info["readable"]:
            assert "table_count" in info, "Should have table count"
            assert "total_records" in info, "Should have total records"
            assert "tables" in info, "Should have table list"
            assert isinstance(info["tables"], list), "Tables should be a list"

    def test_path_resolution_consistency(self):
        """Test that path resolution is consistent across calls."""
        path1 = get_database_path()
        path2 = get_database_path()

        assert path1 == path2, "Path resolution should be consistent"

    def test_working_directory_independence(self):
        """Test that path resolution is independent of working directory."""
        original_cwd = os.getcwd()
        path1 = get_database_path()

        try:
            # Change working directory
            os.chdir("/tmp")
            path2 = get_database_path()

            assert path1 == path2, \
                "Path should be same regardless of working directory"

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])