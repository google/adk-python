"""
Integration tests for ADK agent SQLite query functionality
Tests that the ADK agent can successfully query the SQLite database.
These tests should FAIL initially as part of TDD approach.
"""

import pytest
import sqlite3
import tempfile
import os
from typing import Dict, Any, List


class TestAgentQueries:
    """Test ADK agent SQLite query capabilities"""

    @pytest.fixture
    def sample_database(self):
        """Create a sample database with GCP security data"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        conn = sqlite3.connect(db_path)

        # Create tables that match expected GCP security schema
        conn.execute("""
            CREATE TABLE iam_policies (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                resource_name TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                bindings TEXT NOT NULL,
                created_date TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE security_findings (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                finding_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                remediation TEXT,
                status TEXT NOT NULL,
                discovered_date TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE audit_logs (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                method_name TEXT NOT NULL,
                principal_email TEXT,
                timestamp TEXT NOT NULL,
                resource_name TEXT
            )
        """)

        # Insert sample data
        sample_policies = [
            (1, "project-1", "//cloudresourcemanager.googleapis.com/projects/project-1",
             "PROJECT_IAM_POLICY", '{"bindings": [{"role": "roles/owner", "members": ["user:admin@example.com"]}]}',
             "2024-01-15"),
            (2, "project-2", "//storage.googleapis.com/bucket1",
             "BUCKET_IAM_POLICY", '{"bindings": [{"role": "roles/storage.admin", "members": ["user:dev@example.com"]}]}',
             "2024-01-16")
        ]

        sample_findings = [
            (1, "project-1", "OVERPRIVILEGED_ROLE", "HIGH",
             "User has excessive permissions", "Remove unnecessary roles", "ACTIVE", "2024-01-20"),
            (2, "project-2", "PUBLIC_BUCKET", "CRITICAL",
             "Storage bucket is publicly accessible", "Restrict bucket access", "ACTIVE", "2024-01-21")
        ]

        sample_logs = [
            (1, "project-1", "cloudresourcemanager.googleapis.com", "setIamPolicy",
             "admin@example.com", "2024-01-15T10:30:00Z", "projects/project-1"),
            (2, "project-2", "storage.googleapis.com", "objects.create",
             "dev@example.com", "2024-01-16T14:20:00Z", "bucket1/file.txt")
        ]

        conn.executemany("INSERT INTO iam_policies VALUES (?, ?, ?, ?, ?, ?)", sample_policies)
        conn.executemany("INSERT INTO security_findings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_findings)
        conn.executemany("INSERT INTO audit_logs VALUES (?, ?, ?, ?, ?, ?, ?)", sample_logs)

        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def adk_agent(self, sample_database):
        """Initialize ADK agent with database connection - this doesn't exist yet"""
        # This will fail initially - the actual agent class doesn't exist
        from agents.adk_agent import ADKAgent

        agent = ADKAgent()
        agent.configure_database(sample_database)
        return agent

    @pytest.mark.asyncio
    async def test_agent_query_iam_policies(self, adk_agent):
        """Test agent can query IAM policies"""
        query = "Show me all IAM policies for project-1"

        response = await adk_agent.process_query(query)

        assert response is not None
        assert "project-1" in response
        assert "iam_policies" in response.lower() or "iam" in response.lower()

        # Should contain the policy information
        assert "roles/owner" in response
        assert "admin@example.com" in response

    @pytest.mark.asyncio
    async def test_agent_query_security_findings(self, adk_agent):
        """Test agent can query security findings"""
        query = "What security findings do we have?"

        response = await adk_agent.process_query(query)

        assert response is not None
        assert "security" in response.lower() or "finding" in response.lower()

        # Should contain finding information
        assert "OVERPRIVILEGED_ROLE" in response or "overprivileged" in response.lower()
        assert "PUBLIC_BUCKET" in response or "public" in response.lower()

    @pytest.mark.asyncio
    async def test_agent_query_critical_findings(self, adk_agent):
        """Test agent can filter critical security findings"""
        query = "Show me critical security issues"

        response = await adk_agent.process_query(query)

        assert response is not None
        assert "critical" in response.lower()
        assert "PUBLIC_BUCKET" in response or "public" in response.lower()
        # Should not include HIGH severity items in critical filter
        assert "OVERPRIVILEGED_ROLE" not in response

    @pytest.mark.asyncio
    async def test_agent_query_audit_logs(self, adk_agent):
        """Test agent can query audit logs"""
        query = "Show me recent audit logs for project-1"

        response = await adk_agent.process_query(query)

        assert response is not None
        assert "project-1" in response
        assert "audit" in response.lower() or "log" in response.lower()

        # Should contain log information
        assert "setIamPolicy" in response
        assert "admin@example.com" in response

    @pytest.mark.asyncio
    async def test_agent_query_with_context(self, adk_agent):
        """Test agent maintains context across queries"""
        # First query
        first_query = "What projects do we have?"
        first_response = await adk_agent.process_query(first_query)

        assert "project-1" in first_response
        assert "project-2" in first_response

        # Follow-up query using context
        follow_up_query = "Show me security findings for the first project"
        follow_up_response = await adk_agent.process_query(follow_up_query)

        # Should understand "first project" refers to project-1
        assert "project-1" in follow_up_response
        assert "OVERPRIVILEGED_ROLE" in follow_up_response

    @pytest.mark.asyncio
    async def test_agent_query_complex_sql(self, adk_agent):
        """Test agent can handle complex SQL-like queries"""
        query = "How many security findings does each project have?"

        response = await adk_agent.process_query(query)

        assert response is not None
        assert "project-1" in response
        assert "project-2" in response
        # Should show counts
        assert "1" in response  # Each project has 1 finding

    @pytest.mark.asyncio
    async def test_agent_query_invalid_request(self, adk_agent):
        """Test agent handles invalid queries gracefully"""
        query = "DELETE FROM iam_policies"  # Potentially dangerous query

        response = await adk_agent.process_query(query)

        # Should refuse dangerous operations
        assert "cannot" in response.lower() or "not allowed" in response.lower() or "error" in response.lower()

    @pytest.mark.asyncio
    async def test_agent_query_natural_language(self, adk_agent):
        """Test agent understands natural language queries"""
        natural_queries = [
            "Which users have admin access?",
            "Are there any public buckets?",
            "Who made changes recently?",
            "What are our biggest security risks?"
        ]

        for query in natural_queries:
            response = await adk_agent.process_query(query)

            assert response is not None
            assert len(response) > 0
            assert "error" not in response.lower()

    @pytest.mark.asyncio
    async def test_agent_query_with_tool_usage(self, adk_agent):
        """Test agent uses SQLite tool properly"""
        query = "Run a database query to count IAM policies"

        response = await adk_agent.process_query(query)

        # Should use the SQLite tool
        tool_calls = adk_agent.get_last_tool_calls()
        assert len(tool_calls) > 0

        sqlite_tool_used = any("sqlite" in call.get("tool_name", "").lower() for call in tool_calls)
        assert sqlite_tool_used

        # Response should contain count
        assert "2" in response  # We have 2 IAM policies

    @pytest.mark.asyncio
    async def test_agent_query_performance(self, adk_agent):
        """Test agent query performance"""
        query = "Show me all security data"

        import time
        start_time = time.time()

        response = await adk_agent.process_query(query)

        end_time = time.time()
        query_time = end_time - start_time

        # Should complete within reasonable time
        assert query_time < 10.0, f"Query took too long: {query_time}s"
        assert response is not None

    @pytest.mark.asyncio
    async def test_agent_query_error_handling(self, adk_agent):
        """Test agent handles database errors gracefully"""
        # Simulate database connection issue
        adk_agent.configure_database("/nonexistent/path.db")

        query = "Show me IAM policies"
        response = await adk_agent.process_query(query)

        # Should handle error gracefully
        assert "error" in response.lower() or "unable" in response.lower()
        assert "database" in response.lower() or "connection" in response.lower()

    @pytest.mark.asyncio
    async def test_agent_query_session_isolation(self, adk_agent):
        """Test that different sessions are properly isolated"""
        session1_id = "session_1"
        session2_id = "session_2"

        # Query in session 1
        query1 = "Remember that I'm interested in project-1"
        response1 = await adk_agent.process_query(query1, session_id=session1_id)

        # Query in session 2
        query2 = "Remember that I'm interested in project-2"
        response2 = await adk_agent.process_query(query2, session_id=session2_id)

        # Follow-up queries should maintain separate context
        follow_up1 = "What am I interested in?"
        response_follow1 = await adk_agent.process_query(follow_up1, session_id=session1_id)
        assert "project-1" in response_follow1

        follow_up2 = "What am I interested in?"
        response_follow2 = await adk_agent.process_query(follow_up2, session_id=session2_id)
        assert "project-2" in response_follow2

    def test_agent_tool_registration(self, adk_agent):
        """Test that SQLite tool is properly registered with agent"""
        tools = adk_agent.get_available_tools()

        assert len(tools) > 0

        sqlite_tool_found = any("sqlite" in tool.get("name", "").lower() for tool in tools)
        assert sqlite_tool_found, "SQLite tool not found in available tools"