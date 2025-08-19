"""
Test suite for optimized log analyzer (TASK-005).

Tests performance improvements including:
- Batch processing efficiency
- Cache hit rates
- Parallel processing
- Memory usage optimization
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.optimized_log_analyzer import (
    OptimizedLogAnalyzer,
    LogBatch,
    AnalysisResult,
    get_optimized_analyzer
)


class TestOptimizedLogAnalyzer:
    """Test suite for the optimized log analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance for testing."""
        return OptimizedLogAnalyzer()
    
    @pytest.fixture
    def sample_logs(self):
        """Generate sample log entries for testing."""
        logs = []
        severities = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        messages = [
            "Connection timeout after 30000ms",
            "Permission denied for user@example.com",
            "Database query completed in 150ms",
            "Rate limit exceeded for API calls",
            "Memory usage at 85%",
            "Successfully processed request",
            "Failed to connect to database",
            "Authentication failed: invalid token",
            "Request latency: 250ms",
            "Out of memory error"
        ]
        
        for i in range(100):
            logs.append({
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "severity": severities[i % len(severities)],
                "message": messages[i % len(messages)],
                "resource": {
                    "type": "gce_instance" if i % 2 == 0 else "k8s_container",
                    "labels": {"instance_id": f"instance-{i}"}
                }
            })
        
        return logs
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, analyzer, sample_logs):
        """Test that logs are processed in batches efficiently."""
        result = await analyzer.analyze_logs_batch(sample_logs, batch_size=20)
        
        assert result.entries_processed == 100
        assert result.processing_time_ms > 0
        assert len(result.patterns) > 0
        assert "timeout" in result.patterns
        assert "permission" in result.patterns
        assert "memory" in result.patterns
    
    @pytest.mark.asyncio
    async def test_pattern_caching(self, analyzer):
        """Test that pattern matching results are cached."""
        # First call should cache the result
        message = "Connection timeout after 30000ms"
        patterns1 = analyzer._match_patterns_cached(message)
        
        # Second call should hit the cache
        patterns2 = analyzer._match_patterns_cached(message)
        
        assert patterns1 == patterns2
        assert "timeout" in patterns1
        assert "connection" in patterns1
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self, analyzer, sample_logs):
        """Test that parallel processing improves performance."""
        # Large dataset to test parallel processing
        large_logs = sample_logs * 10  # 1000 logs
        
        start_time = datetime.now()
        result = await analyzer.analyze_logs_batch(large_logs, batch_size=100)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert result.entries_processed == 1000
        assert result.processing_time_ms > 0
        # Should process at least 100 logs per second with optimization
        assert result.entries_processed / processing_time > 100
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, analyzer):
        """Test anomaly detection in log patterns."""
        # Create logs with a spike pattern
        normal_logs = []
        spike_logs = []
        
        # Normal traffic
        for i in range(50):
            normal_logs.append({
                "timestamp": f"2024-01-01T10:{i:02d}:00Z",
                "severity": "INFO",
                "message": "Normal operation"
            })
        
        # Traffic spike
        for i in range(200):
            spike_logs.append({
                "timestamp": "2024-01-01T10:30:00Z",
                "severity": "ERROR",
                "message": f"Error {i}"
            })
        
        all_logs = normal_logs + spike_logs
        result = await analyzer.analyze_logs_batch(all_logs)
        
        assert len(result.anomalies) > 0
        spike_anomaly = next(
            (a for a in result.anomalies if a["type"] == "traffic_spike"),
            None
        )
        assert spike_anomaly is not None
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, analyzer):
        """Test that memory usage is optimized."""
        # Create a large number of logs
        huge_logs = []
        for i in range(10000):
            huge_logs.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "INFO",
                "message": f"Log entry {i}" * 10  # Larger messages
            })
        
        # Process in batches to avoid memory overflow
        result = await analyzer.analyze_logs_batch(huge_logs, batch_size=500)
        
        assert result.entries_processed == 10000
        # Check that deduplication is working (patterns should be limited)
        assert len(result.patterns) < 20  # Should have limited pattern types
    
    @pytest.mark.asyncio
    async def test_performance_metrics_extraction(self, analyzer):
        """Test extraction of performance metrics from logs."""
        perf_logs = [
            {"message": "Request latency: 100ms", "severity": "INFO"},
            {"message": "Query latency: 200ms", "severity": "INFO"},
            {"message": "API latency: 150ms", "severity": "INFO"},
            {"message": "Response latency: 300ms", "severity": "WARNING"},
        ]
        
        result = await analyzer.analyze_logs_batch(perf_logs)
        
        assert "avg_latency" in result.performance_metrics
        assert "max_latency" in result.performance_metrics
        assert "min_latency" in result.performance_metrics
        assert result.performance_metrics["avg_latency"] == 187.5  # (100+200+150+300)/4
        assert result.performance_metrics["max_latency"] == 300
        assert result.performance_metrics["min_latency"] == 100
    
    @pytest.mark.asyncio
    async def test_error_summary(self, analyzer, sample_logs):
        """Test error summary generation."""
        result = await analyzer.analyze_logs_batch(sample_logs)
        
        assert "error_count" in result.error_summary
        assert "warning_count" in result.error_summary
        assert "error_rate" in result.error_summary
        assert result.error_summary["error_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, analyzer):
        """Test streaming log analysis."""
        async def mock_log_stream():
            """Generate a mock stream of logs."""
            for i in range(200):
                yield {
                    "timestamp": datetime.now().isoformat(),
                    "severity": "ERROR" if i % 10 == 0 else "INFO",
                    "message": f"Stream log {i}"
                }
                await asyncio.sleep(0.001)
        
        # Process stream
        results = []
        async for analysis in analyzer.stream_analyze_logs(
            mock_log_stream(),
            window_size=50
        ):
            results.append(analysis)
            if len(results) >= 3:  # Get first 3 windows
                break
        
        assert len(results) == 3
        for result in results:
            assert result.entries_processed > 0
            assert len(result.patterns) >= 0
    
    def test_batch_creation(self, analyzer):
        """Test batch creation logic."""
        logs = [{"id": i} for i in range(250)]
        batches = analyzer._create_batches(logs, batch_size=100)
        
        assert len(batches) == 3
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50
    
    def test_cache_cleanup(self, analyzer):
        """Test cache cleanup mechanism."""
        # Add items to cache
        for i in range(10):
            result = AnalysisResult()
            result.patterns = {f"pattern_{i}": i}
            analyzer._cache_results(result)
        
        # Modify timestamps to simulate aging
        for key in list(analyzer.results_cache.keys())[:5]:
            analyzer.results_cache[key]["timestamp"] = (
                datetime.now() - timedelta(hours=2)
            )
            analyzer.results_cache[key]["ttl"] = 3600  # 1 hour TTL
        
        # Cleanup should remove expired entries
        analyzer._cleanup_cache()
        
        assert len(analyzer.results_cache) == 5  # Only non-expired remain
    
    def test_singleton_instance(self):
        """Test that get_optimized_analyzer returns singleton."""
        analyzer1 = get_optimized_analyzer()
        analyzer2 = get_optimized_analyzer()
        
        assert analyzer1 is analyzer2


class TestIntegrationWithLogsAPI:
    """Integration tests with the logs API."""
    
    @pytest.mark.asyncio
    @patch('api.logs.OPTIMIZED_ANALYZER_AVAILABLE', True)
    @patch('api.logs.get_optimized_analyzer')
    async def test_logs_api_uses_optimizer(self, mock_get_analyzer):
        """Test that logs API uses the optimized analyzer when available."""
        from api.logs import analyze_logs_optimized, LogQueryRequest
        
        # Mock the analyzer
        mock_analyzer = Mock(spec=OptimizedLogAnalyzer)
        mock_result = AnalysisResult()
        mock_result.patterns = {"test": 1}
        mock_result.processing_time_ms = 100
        mock_result.entries_processed = 50
        
        mock_analyzer.analyze_logs_batch = AsyncMock(return_value=mock_result)
        mock_analyzer.get_cached_analysis = Mock(return_value=None)
        mock_get_analyzer.return_value = mock_analyzer
        
        # Create request
        request = LogQueryRequest(
            project_id="test-project",
            time_range="1h",
            limit=100
        )
        
        # Mock list_logs to return sample data
        with patch('api.logs.list_logs') as mock_list_logs:
            mock_list_logs.return_value = {
                "success": True,
                "entries": [{"message": "test"} for _ in range(50)]
            }
            
            result = await analyze_logs_optimized(request)
        
        assert result["success"] is True
        assert result["analysis"]["optimization"] == "batch_processing"
        assert result["analysis"]["processing_time_ms"] == 100
        assert result["performance"]["entries_per_second"] == 500  # 50 / 0.1
        
        # Verify analyzer was called
        mock_analyzer.analyze_logs_batch.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])