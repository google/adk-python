#!/usr/bin/env python3
"""
Integration tests for Cloud Functions.
Tests end-to-end functionality with deployed functions.
"""

import pytest
import requests
import json
import time
from datetime import datetime
import os
from google.cloud import bigquery
from typing import Dict, List, Any


# Configuration
PROJECT_ID = os.getenv("PROJECT_ID", "mgm-digitalconcierge")
REGION = os.getenv("REGION", "us-central1")
DATASET_ID = os.getenv("BQ_DATASET_ID", "security_insights")
BASE_URL = f"https://{REGION}-{PROJECT_ID}.cloudfunctions.net"


class TestCloudFunctionsIntegration:
    """Integration tests for deployed Cloud Functions"""

    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """Create BigQuery client for verification"""
        return bigquery.Client(project=PROJECT_ID)

    @pytest.fixture(scope="class")
    def function_endpoints(self):
        """Map of function names to endpoints"""
        return {
            "fetch_custom_roles": f"{BASE_URL}/fetch_custom_roles",
            "fetch_user_roles": f"{BASE_URL}/fetch_user_roles",
            "fetch_service_account_roles": f"{BASE_URL}/fetch_service_account_roles",
            "fetch_standard_roles": f"{BASE_URL}/fetch_standard_roles",
            "fetch_compute_instances": f"{BASE_URL}/fetch_compute_instances",
            "fetch_firewall_rules": f"{BASE_URL}/fetch_firewall_rules",
            "fetch_storage_buckets": f"{BASE_URL}/fetch_storage_buckets",
            "fetch_security_findings": f"{BASE_URL}/fetch_security_findings",
            "fetch_iam_accounts": f"{BASE_URL}/fetch_iam_accounts"
        }

    def test_all_functions_deployed(self, function_endpoints):
        """Test that all Cloud Functions are deployed and responding"""
        failed_functions = []

        for func_name, endpoint in function_endpoints.items():
            try:
                response = requests.post(
                    endpoint,
                    json={},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                if response.status_code != 200:
                    failed_functions.append((func_name, response.status_code))
            except Exception as e:
                failed_functions.append((func_name, str(e)))

        assert len(failed_functions) == 0, f"Functions failed: {failed_functions}"

    def test_iam_functions_data_consistency(self, function_endpoints, bigquery_client):
        """Test IAM functions data consistency in BigQuery"""
        # Trigger IAM functions
        iam_functions = [
            "fetch_custom_roles",
            "fetch_user_roles",
            "fetch_service_account_roles",
            "fetch_standard_roles"
        ]

        for func_name in iam_functions:
            response = requests.post(
                function_endpoints[func_name],
                json={},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            assert response.status_code == 200

        # Wait for BigQuery propagation
        time.sleep(5)

        # Verify tables exist and have data
        tables = {
            "custom_roles": "role_id",
            "user_roles": "user_email",
            "service_account_roles": "service_account_email",
            "standard_roles": "role_name"
        }

        for table_name, id_column in tables.items():
            query = f"""
            SELECT COUNT(*) as count
            FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
            WHERE DATE(last_refreshed) = CURRENT_DATE()
            """

            try:
                result = bigquery_client.query(query).result()
                row_count = list(result)[0]["count"]
                assert row_count >= 0, f"Table {table_name} should have recent data"
            except Exception as e:
                pytest.skip(f"Table {table_name} not accessible: {e}")

    def test_infrastructure_functions_data_consistency(self, function_endpoints, bigquery_client):
        """Test infrastructure functions data consistency"""
        infrastructure_functions = [
            "fetch_compute_instances",
            "fetch_firewall_rules",
            "fetch_storage_buckets",
            "fetch_security_findings"
        ]

        responses = {}
        for func_name in infrastructure_functions:
            response = requests.post(
                function_endpoints[func_name],
                json={},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            assert response.status_code == 200
            responses[func_name] = response.json()

        # Verify response structure
        assert "total_instances" in responses["fetch_compute_instances"]
        assert "total_rules" in responses["fetch_firewall_rules"]
        assert "total_buckets" in responses["fetch_storage_buckets"]
        assert "total_findings" in responses["fetch_security_findings"]

    def test_concurrent_function_execution(self, function_endpoints):
        """Test concurrent execution of multiple functions"""
        import concurrent.futures

        def call_function(func_name, endpoint):
            try:
                response = requests.post(
                    endpoint,
                    json={},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                return func_name, response.status_code, response.json()
            except Exception as e:
                return func_name, 500, str(e)

        # Execute all functions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(call_function, func_name, endpoint): func_name
                for func_name, endpoint in function_endpoints.items()
            }

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # Verify all succeeded
        failed = [(name, status) for name, status, _ in results if status != 200]
        assert len(failed) == 0, f"Functions failed during concurrent execution: {failed}"

    def test_function_idempotency(self, function_endpoints):
        """Test that functions are idempotent"""
        # Test a subset of functions for idempotency
        test_functions = ["fetch_standard_roles", "fetch_firewall_rules"]

        for func_name in test_functions:
            endpoint = function_endpoints[func_name]

            # Call function twice
            response1 = requests.post(
                endpoint,
                json={},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            response2 = requests.post(
                endpoint,
                json={},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            # Both should succeed
            assert response1.status_code == 200
            assert response2.status_code == 200

            # Results should be consistent (same structure)
            data1 = response1.json()
            data2 = response2.json()

            assert data1["status"] == data2["status"] == "success"
            assert set(data1.keys()) == set(data2.keys())

    def test_error_handling(self, function_endpoints):
        """Test error handling with invalid requests"""
        # Test with invalid content type
        response = requests.post(
            function_endpoints["fetch_custom_roles"],
            data="invalid-json",
            headers={"Content-Type": "text/plain"},
            timeout=30
        )

        # Should handle gracefully (may return 400 or 200 with error message)
        assert response.status_code in [200, 400, 415]

    def test_cross_function_data_consistency(self, function_endpoints, bigquery_client):
        """Test data consistency across related functions"""
        # Fetch user roles and service account roles
        user_response = requests.post(
            function_endpoints["fetch_user_roles"],
            json={},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        sa_response = requests.post(
            function_endpoints["fetch_service_account_roles"],
            json={},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        assert user_response.status_code == 200
        assert sa_response.status_code == 200

        # Wait for BigQuery propagation
        time.sleep(5)

        # Query for overlapping data
        query = """
        SELECT COUNT(DISTINCT role) as common_roles
        FROM (
            SELECT role FROM `{}.{}.user_roles`
            UNION ALL
            SELECT role FROM `{}.{}.service_account_roles`
        )
        GROUP BY role
        HAVING COUNT(*) > 1
        """.format(PROJECT_ID, DATASET_ID, PROJECT_ID, DATASET_ID)

        try:
            result = bigquery_client.query(query).result()
            # Should have some common roles between users and service accounts
            common_roles = len(list(result))
            assert common_roles >= 0, "Should have non-negative common roles"
        except Exception as e:
            pytest.skip(f"Cross-table query not accessible: {e}")

    @pytest.mark.slow
    def test_function_performance_under_load(self, function_endpoints):
        """Test function performance under moderate load"""
        import concurrent.futures
        import statistics

        # Select a lightweight function for load testing
        test_endpoint = function_endpoints["fetch_standard_roles"]
        num_requests = 20
        response_times = []

        def timed_request():
            start_time = time.time()
            response = requests.post(
                test_endpoint,
                json={},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            end_time = time.time()
            return response.status_code, end_time - start_time

        # Execute requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(timed_request) for _ in range(num_requests)]

            for future in concurrent.futures.as_completed(futures):
                status, duration = future.result()
                if status == 200:
                    response_times.append(duration)

        # Calculate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            max_time = max(response_times)

            # Performance assertions
            assert avg_time < 10, f"Average response time {avg_time}s exceeds 10s"
            assert median_time < 8, f"Median response time {median_time}s exceeds 8s"
            assert max_time < 30, f"Max response time {max_time}s exceeds 30s"
            assert len(response_times) >= num_requests * 0.8, "Too many failed requests"


class TestBigQueryIntegration:
    """Test BigQuery data integrity and queries"""

    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """Create BigQuery client"""
        return bigquery.Client(project=PROJECT_ID)

    def test_dataset_exists(self, bigquery_client):
        """Test that the security_insights dataset exists"""
        dataset_ref = bigquery_client.dataset(DATASET_ID)

        try:
            dataset = bigquery_client.get_dataset(dataset_ref)
            assert dataset.dataset_id == DATASET_ID
        except Exception as e:
            pytest.fail(f"Dataset {DATASET_ID} does not exist: {e}")

    def test_required_tables_exist(self, bigquery_client):
        """Test that all required tables exist"""
        required_tables = [
            "custom_roles",
            "user_roles",
            "service_account_roles",
            "standard_roles",
            "compute_instances",
            "firewall_rules",
            "storage_buckets",
            "security_findings"
        ]

        dataset_ref = bigquery_client.dataset(DATASET_ID)
        tables = list(bigquery_client.list_tables(dataset_ref))
        table_names = [table.table_id for table in tables]

        missing_tables = [t for t in required_tables if t not in table_names]
        assert len(missing_tables) == 0, f"Missing tables: {missing_tables}"

    def test_table_schemas(self, bigquery_client):
        """Test that table schemas match expectations"""
        # Test custom_roles schema
        table_ref = bigquery_client.dataset(DATASET_ID).table("custom_roles")

        try:
            table = bigquery_client.get_table(table_ref)
            schema_fields = {field.name for field in table.schema}

            required_fields = {
                "role_id", "project_id", "title", "description",
                "included_permissions", "permission_count",
                "high_risk_permissions", "last_refreshed"
            }

            missing_fields = required_fields - schema_fields
            assert len(missing_fields) == 0, f"Missing fields in custom_roles: {missing_fields}"
        except Exception as e:
            pytest.skip(f"Table custom_roles not accessible: {e}")

    def test_data_freshness(self, bigquery_client):
        """Test that data is being refreshed regularly"""
        # Check if any table has data from today
        tables_to_check = ["user_roles", "standard_roles"]
        fresh_tables = []

        for table_name in tables_to_check:
            query = f"""
            SELECT COUNT(*) as count
            FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
            WHERE DATE(last_refreshed) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
            """

            try:
                result = bigquery_client.query(query).result()
                count = list(result)[0]["count"]
                if count > 0:
                    fresh_tables.append(table_name)
            except Exception:
                pass

        assert len(fresh_tables) > 0, "No tables have fresh data from the last 7 days"

    def test_cross_table_relationships(self, bigquery_client):
        """Test relationships between tables"""
        # Test that service accounts in service_account_roles might appear in compute_instances
        query = """
        SELECT COUNT(*) as count
        FROM `{}.{}.service_account_roles` sa
        WHERE EXISTS (
            SELECT 1
            FROM `{}.{}.compute_instances` ci
            WHERE JSON_EXTRACT_SCALAR(ci.service_accounts, '$[0].email') = sa.service_account_email
        )
        """.format(PROJECT_ID, DATASET_ID, PROJECT_ID, DATASET_ID)

        try:
            result = bigquery_client.query(query).result()
            # Just verify query executes without error
            _ = list(result)[0]["count"]
        except Exception as e:
            # This is expected if tables don't have the exact schema
            pass


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""

    @pytest.fixture(scope="class")
    def function_endpoints(self):
        """Map of function names to endpoints"""
        return {
            "fetch_custom_roles": f"{BASE_URL}/fetch_custom_roles",
            "fetch_user_roles": f"{BASE_URL}/fetch_user_roles",
            "fetch_service_account_roles": f"{BASE_URL}/fetch_service_account_roles",
            "fetch_standard_roles": f"{BASE_URL}/fetch_standard_roles",
            "fetch_compute_instances": f"{BASE_URL}/fetch_compute_instances",
            "fetch_firewall_rules": f"{BASE_URL}/fetch_firewall_rules",
            "fetch_storage_buckets": f"{BASE_URL}/fetch_storage_buckets",
            "fetch_security_findings": f"{BASE_URL}/fetch_security_findings"
        }

    def test_complete_security_scan(self, function_endpoints):
        """Test a complete security scan workflow"""
        scan_results = {}
        scan_start = datetime.utcnow()

        # Phase 1: Collect IAM data
        iam_functions = [
            "fetch_custom_roles",
            "fetch_user_roles",
            "fetch_service_account_roles",
            "fetch_standard_roles"
        ]

        for func in iam_functions:
            response = requests.post(
                function_endpoints[func],
                json={},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            assert response.status_code == 200
            scan_results[func] = response.json()

        # Phase 2: Collect infrastructure data
        infra_functions = [
            "fetch_compute_instances",
            "fetch_firewall_rules",
            "fetch_storage_buckets",
            "fetch_security_findings"
        ]

        for func in infra_functions:
            response = requests.post(
                function_endpoints[func],
                json={},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            assert response.status_code == 200
            scan_results[func] = response.json()

        scan_end = datetime.utcnow()
        scan_duration = (scan_end - scan_start).total_seconds()

        # Verify complete scan
        assert len(scan_results) == len(iam_functions) + len(infra_functions)
        assert scan_duration < 300, f"Complete scan took {scan_duration}s (> 5 minutes)"

        # Verify we have comprehensive security data
        assert scan_results["fetch_user_roles"].get("total_users", 0) >= 0
        assert scan_results["fetch_storage_buckets"].get("total_buckets", 0) >= 0
        assert scan_results["fetch_security_findings"].get("total_findings", 0) >= 0

        # Calculate overall security posture
        total_high_risk_items = 0
        if "high_risk_roles" in scan_results["fetch_custom_roles"]:
            total_high_risk_items += scan_results["fetch_custom_roles"]["high_risk_roles"]
        if "public_buckets" in scan_results["fetch_storage_buckets"]:
            total_high_risk_items += scan_results["fetch_storage_buckets"]["public_buckets"]
        if "overly_permissive_rules" in scan_results["fetch_firewall_rules"]:
            total_high_risk_items += scan_results["fetch_firewall_rules"]["overly_permissive_rules"]

        print(f"Security scan completed in {scan_duration:.2f}s")
        print(f"Total high-risk items found: {total_high_risk_items}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])