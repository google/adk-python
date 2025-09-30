#!/usr/bin/env python3
"""
Performance tests for Cloud Functions.
Tests load handling, response times, and resource usage.
"""

import pytest
import time
import statistics
import concurrent.futures
from unittest.mock import MagicMock, patch
import psutil
import threading
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the Cloud Functions for local performance testing
from fetch_custom_roles import main as custom_roles_main
from fetch_standard_roles import main as standard_roles_main


class TestFunctionPerformance:
    """Performance tests for individual Cloud Functions"""

    @pytest.fixture
    def performance_monitor(self):
        """Monitor system resources during tests"""
        class PerformanceMonitor:
            def __init__(self):
                self.start_time = None
                self.end_time = None
                self.start_cpu = None
                self.end_cpu = None
                self.start_memory = None
                self.end_memory = None

            def start(self):
                self.start_time = time.time()
                self.start_cpu = psutil.cpu_percent()
                self.start_memory = psutil.virtual_memory().percent

            def stop(self):
                self.end_time = time.time()
                self.end_cpu = psutil.cpu_percent()
                self.end_memory = psutil.virtual_memory().percent

            @property
            def duration(self):
                if self.start_time and self.end_time:
                    return self.end_time - self.start_time
                return None

            @property
            def cpu_usage(self):
                if self.start_cpu is not None and self.end_cpu is not None:
                    return self.end_cpu - self.start_cpu
                return None

            @property
            def memory_usage(self):
                if self.start_memory is not None and self.end_memory is not None:
                    return self.end_memory - self.start_memory
                return None

        return PerformanceMonitor()

    def test_single_function_response_time(self, mock_iam_client, mock_bigquery_client,
                                         mock_http_request, performance_monitor):
        """Test response time for single function call"""
        # Arrange
        request = mock_http_request()

        # Act
        performance_monitor.start()
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)
        performance_monitor.stop()

        # Assert
        assert status_code == 200
        assert performance_monitor.duration < 5.0, f"Function took {performance_monitor.duration:.2f}s (>5s)"
        print(f"Function execution time: {performance_monitor.duration:.3f}s")

    def test_concurrent_function_calls(self, mock_iam_client, mock_bigquery_client,
                                     mock_http_request):
        """Test performance under concurrent load"""
        request = mock_http_request()
        num_concurrent = 10
        response_times = []
        errors = []

        def call_function():
            start_time = time.time()
            try:
                with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
                    with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                        response, status_code = standard_roles_main.fetch_standard_roles(request)

                        if status_code == 200:
                            return time.time() - start_time
                        else:
                            errors.append(f"Status {status_code}")
                            return None
            except Exception as e:
                errors.append(str(e))
                return None

        # Execute concurrent calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(call_function) for _ in range(num_concurrent)]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    response_times.append(result)

        # Analyze results
        success_rate = len(response_times) / num_concurrent
        avg_response_time = statistics.mean(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0

        print(f"Concurrent test results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Median response time: {median_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")

        # Assertions
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80%"
        assert avg_response_time < 10.0, f"Average response time {avg_response_time:.3f}s exceeds 10s"
        assert max_response_time < 30.0, f"Max response time {max_response_time:.3f}s exceeds 30s"
        assert len(errors) <= 2, f"Too many errors: {errors}"

    def test_large_dataset_processing(self, mock_bigquery_client, mock_http_request,
                                    performance_monitor):
        """Test performance with large datasets"""
        request = mock_http_request()

        # Create large mock dataset (1000 roles)
        mock_roles = []
        for i in range(1000):
            mock_role = MagicMock()
            mock_role.name = f"roles/test.role{i}"
            mock_role.title = f"Test Role {i}"
            mock_role.description = f"Test role {i} description"
            mock_role.included_permissions = [f"service{j}.resource.action" for j in range(10)]
            mock_role.stage = MagicMock(name="GA")
            mock_roles.append(mock_role)

        mock_iam_client = MagicMock()
        mock_iam_client.list_roles.return_value = mock_roles

        # Act
        performance_monitor.start()
        with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
            with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                response, status_code = standard_roles_main.fetch_standard_roles(request)
        performance_monitor.stop()

        # Assert
        assert status_code == 200
        assert response["total_roles"] == 1000
        assert performance_monitor.duration < 15.0, f"Large dataset processing took {performance_monitor.duration:.2f}s (>15s)"

        print(f"Large dataset processing:")
        print(f"  Dataset size: 1000 roles")
        print(f"  Processing time: {performance_monitor.duration:.3f}s")
        print(f"  Processing rate: {1000/performance_monitor.duration:.1f} roles/second")

    def test_memory_usage_pattern(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test memory usage patterns during execution"""
        request = mock_http_request()

        # Monitor memory usage
        memory_readings = []
        monitoring_active = threading.Event()
        monitoring_active.set()

        def memory_monitor():
            while monitoring_active.is_set():
                memory_readings.append(psutil.virtual_memory().percent)
                time.sleep(0.1)

        # Start memory monitoring
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()

        try:
            # Execute function
            with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
                with patch('fetch_custom_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = custom_roles_main.fetch_custom_roles(request)

            # Stop monitoring
            monitoring_active.clear()
            monitor_thread.join(timeout=1)

            # Analyze memory usage
            if memory_readings:
                initial_memory = memory_readings[0]
                peak_memory = max(memory_readings)
                final_memory = memory_readings[-1]
                memory_increase = peak_memory - initial_memory

                print(f"Memory usage pattern:")
                print(f"  Initial: {initial_memory:.1f}%")
                print(f"  Peak: {peak_memory:.1f}%")
                print(f"  Final: {final_memory:.1f}%")
                print(f"  Increase: {memory_increase:.1f}%")

                # Memory should not increase by more than 10%
                assert memory_increase < 10.0, f"Memory usage increased by {memory_increase:.1f}% (>10%)"

                # Memory should return to near initial levels
                memory_cleanup = abs(final_memory - initial_memory)
                assert memory_cleanup < 5.0, f"Memory not properly cleaned up: {memory_cleanup:.1f}% difference"

        finally:
            monitoring_active.clear()
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=1)

    def test_bigquery_batch_performance(self, mock_bigquery_client, mock_http_request,
                                      performance_monitor):
        """Test BigQuery batch insert performance"""
        request = mock_http_request()

        # Create large dataset for batch insert
        large_dataset = []
        for i in range(5000):
            record = {
                "role_name": f"roles/test.role{i}",
                "title": f"Test Role {i}",
                "description": f"Test role {i} description",
                "permission_count": 10,
                "high_risk_permissions": 2,
                "project_id": "test-project",
                "last_refreshed": "2024-01-01T00:00:00Z"
            }
            large_dataset.append(record)

        # Mock function that performs batch inserts
        def batch_insert_test():
            batch_size = 1000
            batches_processed = 0

            performance_monitor.start()

            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i+batch_size]
                # Simulate BigQuery insert
                mock_bigquery_client.insert_rows_json(None, batch)
                batches_processed += 1

            performance_monitor.stop()
            return batches_processed

        # Execute batch insert test
        batches = batch_insert_test()

        # Assert performance
        assert batches == 5  # 5000 records / 1000 batch size
        assert performance_monitor.duration < 5.0, f"Batch insert took {performance_monitor.duration:.2f}s (>5s)"

        insert_rate = len(large_dataset) / performance_monitor.duration
        assert insert_rate > 1000, f"Insert rate {insert_rate:.0f} records/second below 1000/s"

        print(f"Batch insert performance:")
        print(f"  Records: {len(large_dataset)}")
        print(f"  Batches: {batches}")
        print(f"  Duration: {performance_monitor.duration:.3f}s")
        print(f"  Rate: {insert_rate:.0f} records/second")


class TestLoadTesting:
    """Load testing scenarios"""

    def test_sustained_load(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test sustained load over time"""
        request = mock_http_request()
        duration_seconds = 30
        requests_per_second = 5
        total_requests = duration_seconds * requests_per_second

        start_time = time.time()
        completed_requests = []
        errors = []

        def make_request():
            request_start = time.time()
            try:
                with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
                    with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                        response, status_code = standard_roles_main.fetch_standard_roles(request)

                        request_duration = time.time() - request_start
                        completed_requests.append({
                            'duration': request_duration,
                            'status': status_code,
                            'timestamp': request_start
                        })

            except Exception as e:
                errors.append(str(e))

        # Execute sustained load
        with concurrent.futures.ThreadPoolExecutor(max_workers=requests_per_second * 2) as executor:
            futures = []

            for i in range(total_requests):
                future = executor.submit(make_request)
                futures.append(future)

                # Maintain steady rate
                time.sleep(1.0 / requests_per_second)

                # Stop if we've run for the target duration
                if time.time() - start_time >= duration_seconds:
                    break

            # Wait for all requests to complete
            for future in concurrent.futures.as_completed(futures, timeout=60):
                pass

        # Analyze results
        success_count = len([r for r in completed_requests if r['status'] == 200])
        error_count = len(errors)
        success_rate = success_count / len(completed_requests) if completed_requests else 0

        durations = [r['duration'] for r in completed_requests if r['status'] == 200]
        avg_response_time = statistics.mean(durations) if durations else 0
        percentile_95 = statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else 0

        print(f"Sustained load test results:")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Target rate: {requests_per_second} req/s")
        print(f"  Completed: {len(completed_requests)} requests")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Errors: {error_count}")
        print(f"  Avg response time: {avg_response_time:.3f}s")
        print(f"  95th percentile: {percentile_95:.3f}s")

        # Assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95%"
        assert avg_response_time < 5.0, f"Average response time {avg_response_time:.3f}s exceeds 5s"
        assert percentile_95 < 10.0, f"95th percentile {percentile_95:.3f}s exceeds 10s"
        assert error_count <= len(completed_requests) * 0.05, f"Too many errors: {error_count}"

    def test_spike_load(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test handling of sudden load spikes"""
        request = mock_http_request()
        spike_requests = 50  # Sudden spike of 50 concurrent requests

        response_times = []
        errors = []

        def spike_request():
            start_time = time.time()
            try:
                with patch('fetch_custom_roles.main.IAMClient', return_value=mock_iam_client):
                    with patch('fetch_custom_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                        response, status_code = custom_roles_main.fetch_custom_roles(request)

                        duration = time.time() - start_time
                        if status_code == 200:
                            response_times.append(duration)
                        else:
                            errors.append(f"Status {status_code}")

            except Exception as e:
                errors.append(str(e))

        # Execute spike load
        start_spike = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=spike_requests) as executor:
            futures = [executor.submit(spike_request) for _ in range(spike_requests)]

            for future in concurrent.futures.as_completed(futures, timeout=60):
                pass

        spike_duration = time.time() - start_spike

        # Analyze spike results
        success_count = len(response_times)
        error_count = len(errors)
        success_rate = success_count / spike_requests

        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            median_response_time = statistics.median(response_times)
        else:
            avg_response_time = max_response_time = median_response_time = 0

        print(f"Spike load test results:")
        print(f"  Spike size: {spike_requests} concurrent requests")
        print(f"  Spike duration: {spike_duration:.3f}s")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Errors: {error_count}")
        print(f"  Avg response: {avg_response_time:.3f}s")
        print(f"  Median response: {median_response_time:.3f}s")
        print(f"  Max response: {max_response_time:.3f}s")

        # Assertions for spike handling
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% during spike"
        assert avg_response_time < 15.0, f"Average response time {avg_response_time:.3f}s too high during spike"
        assert max_response_time < 30.0, f"Max response time {max_response_time:.3f}s too high during spike"

    def test_resource_cleanup(self, mock_iam_client, mock_bigquery_client, mock_http_request):
        """Test that resources are properly cleaned up after load"""
        request = mock_http_request()

        # Record initial resource usage
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent(interval=1)

        # Execute moderate load
        for i in range(20):
            with patch('fetch_standard_roles.main.IAMClient', return_value=mock_iam_client):
                with patch('fetch_standard_roles.main.bigquery.Client', return_value=mock_bigquery_client):
                    response, status_code = standard_roles_main.fetch_standard_roles(request)

            time.sleep(0.1)  # Small delay between requests

        # Wait for cleanup
        time.sleep(5)

        # Record final resource usage
        final_memory = psutil.virtual_memory().percent
        final_cpu = psutil.cpu_percent(interval=1)

        memory_diff = final_memory - initial_memory
        cpu_diff = final_cpu - initial_cpu

        print(f"Resource cleanup test:")
        print(f"  Initial memory: {initial_memory:.1f}%")
        print(f"  Final memory: {final_memory:.1f}%")
        print(f"  Memory difference: {memory_diff:.1f}%")
        print(f"  Initial CPU: {initial_cpu:.1f}%")
        print(f"  Final CPU: {final_cpu:.1f}%")
        print(f"  CPU difference: {cpu_diff:.1f}%")

        # Assert proper cleanup
        assert abs(memory_diff) < 5.0, f"Memory not cleaned up properly: {memory_diff:.1f}% difference"
        assert abs(cpu_diff) < 10.0, f"CPU usage not normalized: {cpu_diff:.1f}% difference"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])