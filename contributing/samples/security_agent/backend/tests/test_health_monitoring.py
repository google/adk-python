"""
Comprehensive test suite for Health Monitoring System - TASK-007

Tests health checks, component monitoring, performance metrics,
resource monitoring, and API endpoints.
"""

import pytest
import asyncio
import tempfile
import os
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

# Import health monitoring components
from backend.health import (
    HealthStatus, ComponentHealthCheck, DatabaseHealthCheck,
    GCPAPIHealthCheck, SystemResourcesHealthCheck, 
    ComponentAvailabilityHealthCheck, PerformanceHealthCheck,
    HealthMonitor
)
from backend.api.health import router
from backend.main import app

client = TestClient(app)

# ============================================================================
# HEALTH STATUS AND BASE CLASS TESTS
# ============================================================================

class TestHealthStatus:
    """Test HealthStatus constants"""
    
    def test_health_status_values(self):
        """Test health status constant values"""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"

class TestComponentHealthCheck:
    """Test base ComponentHealthCheck class"""
    
    def test_component_health_check_initialization(self):
        """Test component health check initialization"""
        check = ComponentHealthCheck("test_component", critical=True)
        
        assert check.name == "test_component"
        assert check.critical is True
        assert check.last_check is None
        assert check.last_result is None
    
    @pytest.mark.asyncio
    async def test_component_health_check_default_implementation(self):
        """Test default health check implementation"""
        check = ComponentHealthCheck("test")
        result = await check.check()
        
        assert result["status"] == HealthStatus.UNKNOWN
        assert "not implemented" in result["message"]
        assert "timestamp" in result

# ============================================================================
# DATABASE HEALTH CHECK TESTS
# ============================================================================

class TestDatabaseHealthCheck:
    """Test database connectivity health checks"""
    
    @pytest.mark.asyncio
    async def test_database_health_check_success(self):
        """Test successful database health check"""
        db_check = DatabaseHealthCheck()
        result = await db_check.check()
        
        assert result["status"] == HealthStatus.HEALTHY
        assert "successful" in result["message"]
        assert "response_time_ms" in result
        assert result["response_time_ms"] > 0
        assert "test_result" in result
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_database_health_check_performance(self):
        """Test database health check performance is reasonable"""
        db_check = DatabaseHealthCheck()
        start_time = time.time()
        result = await db_check.check()
        end_time = time.time()
        
        # Should complete within 1 second
        assert (end_time - start_time) < 1.0
        assert result["status"] == HealthStatus.HEALTHY
        assert result["response_time_ms"] < 1000
    
    @pytest.mark.asyncio
    async def test_database_health_check_with_sqlite_operations(self):
        """Test database health check performs actual SQLite operations"""
        db_check = DatabaseHealthCheck()
        result = await db_check.check()
        
        assert result["status"] == HealthStatus.HEALTHY
        # Verify it actually performed operations
        assert "count: 1" in result["test_result"]
    
    @pytest.mark.asyncio
    async def test_database_health_check_error_handling(self):
        """Test database health check error handling"""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            db_check = DatabaseHealthCheck()
            result = await db_check.check()
            
            assert result["status"] == HealthStatus.UNHEALTHY
            assert "Database connectivity failed" in result["message"]
            assert "Database error" in result["message"]

# ============================================================================
# GCP API HEALTH CHECK TESTS
# ============================================================================

class TestGCPAPIHealthCheck:
    """Test GCP API connectivity health checks"""
    
    @pytest.mark.asyncio
    async def test_gcp_health_check_no_credentials(self):
        """Test GCP health check without credentials"""
        with patch.dict(os.environ, {}, clear=True):
            gcp_check = GCPAPIHealthCheck()
            result = await gcp_check.check()
            
            assert result["status"] == HealthStatus.DEGRADED
            assert "not fully configured" in result["message"]
            assert "credentials_configured" in result["details"]
            assert result["details"]["credentials_configured"] is False
    
    @pytest.mark.asyncio
    async def test_gcp_health_check_with_credentials(self):
        """Test GCP health check with credentials configured"""
        test_env = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json",
            "GOOGLE_CLOUD_PROJECT": "test-project"
        }
        
        with patch.dict(os.environ, test_env):
            gcp_check = GCPAPIHealthCheck()
            result = await gcp_check.check()
            
            assert result["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            assert result["details"]["credentials_configured"] is True
            assert result["details"]["project_id"] == "test-project"
    
    @pytest.mark.asyncio
    async def test_gcp_health_check_api_success(self):
        """Test GCP health check with successful API call"""
        test_env = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json",
            "GOOGLE_CLOUD_PROJECT": "test-project"
        }
        
        with patch.dict(os.environ, test_env):
            with patch('google.auth.default') as mock_default:
                with patch('google.cloud.resource_manager.Client') as mock_client:
                    # Mock successful authentication and API call
                    mock_default.return_value = (Mock(), "test-project")
                    mock_project = Mock()
                    mock_project.name = "Test Project"
                    mock_client.return_value.fetch_project.return_value = mock_project
                    
                    gcp_check = GCPAPIHealthCheck()
                    result = await gcp_check.check()
                    
                    assert result["status"] == HealthStatus.HEALTHY
                    assert "accessible" in result["message"]
                    assert result["details"]["api_checks"]["resource_manager"]["status"] == "accessible"
    
    @pytest.mark.asyncio
    async def test_gcp_health_check_api_error(self):
        """Test GCP health check with API error"""
        test_env = {
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json", 
            "GOOGLE_CLOUD_PROJECT": "test-project"
        }
        
        with patch.dict(os.environ, test_env):
            with patch('google.auth.default', side_effect=Exception("Auth failed")):
                gcp_check = GCPAPIHealthCheck()
                result = await gcp_check.check()
                
                assert result["status"] == HealthStatus.DEGRADED
                assert "api_checks" in result["details"]
    
    @pytest.mark.asyncio  
    async def test_gcp_health_check_import_error(self):
        """Test GCP health check with missing Google Cloud libraries"""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test"}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                gcp_check = GCPAPIHealthCheck()
                result = await gcp_check.check()
                
                # Should still return a result, even if libraries are missing
                assert result["status"] in [HealthStatus.DEGRADED, HealthStatus.UNKNOWN]

# ============================================================================
# SYSTEM RESOURCES HEALTH CHECK TESTS
# ============================================================================

class TestSystemResourcesHealthCheck:
    """Test system resources monitoring"""
    
    @pytest.mark.asyncio
    async def test_system_resources_normal(self):
        """Test system resources health check with normal usage"""
        with patch('psutil.cpu_percent', return_value=25.0):
            with patch('psutil.virtual_memory') as mock_memory:
                with patch('psutil.disk_usage') as mock_disk:
                    # Mock normal resource usage
                    mock_memory.return_value = Mock(
                        total=8*1024**3, used=2*1024**3, percent=25.0
                    )
                    mock_disk.return_value = Mock(
                        total=100*1024**3, used=30*1024**3, percent=30.0
                    )
                    
                    resource_check = SystemResourcesHealthCheck()
                    result = await resource_check.check()
                    
                    assert result["status"] == HealthStatus.HEALTHY
                    assert "normal" in result["message"]
                    assert result["metrics"]["cpu"]["status"] == "normal"
                    assert result["metrics"]["memory"]["status"] == "normal"
                    assert result["metrics"]["disk"]["status"] == "normal"
    
    @pytest.mark.asyncio
    async def test_system_resources_warning_levels(self):
        """Test system resources with warning-level usage"""
        with patch('psutil.cpu_percent', return_value=75.0):
            with patch('psutil.virtual_memory') as mock_memory:
                with patch('psutil.disk_usage') as mock_disk:
                    # Mock warning-level usage
                    mock_memory.return_value = Mock(
                        total=8*1024**3, used=6*1024**3, percent=85.0
                    )
                    mock_disk.return_value = Mock(
                        total=100*1024**3, used=90*1024**3, percent=90.0
                    )
                    
                    resource_check = SystemResourcesHealthCheck()
                    result = await resource_check.check()
                    
                    assert result["status"] == HealthStatus.DEGRADED
                    assert "pressure" in result["message"]
                    assert result["metrics"]["cpu"]["status"] == "warning"
                    assert result["metrics"]["memory"]["status"] == "warning"
                    assert result["metrics"]["disk"]["status"] == "warning"
    
    @pytest.mark.asyncio
    async def test_system_resources_critical_levels(self):
        """Test system resources with critical usage"""
        with patch('psutil.cpu_percent', return_value=95.0):
            with patch('psutil.virtual_memory') as mock_memory:
                with patch('psutil.disk_usage') as mock_disk:
                    # Mock critical-level usage
                    mock_memory.return_value = Mock(
                        total=8*1024**3, used=7.5*1024**3, percent=95.0
                    )
                    mock_disk.return_value = Mock(
                        total=100*1024**3, used=97*1024**3, percent=97.0
                    )
                    
                    resource_check = SystemResourcesHealthCheck()
                    result = await resource_check.check()
                    
                    assert result["status"] == HealthStatus.UNHEALTHY
                    assert "constraints" in result["message"]
                    assert result["metrics"]["cpu"]["status"] == "critical"
                    assert result["metrics"]["memory"]["status"] == "critical"
                    assert result["metrics"]["disk"]["status"] == "critical"
    
    @pytest.mark.asyncio
    async def test_system_resources_error_handling(self):
        """Test system resources error handling"""
        with patch('psutil.cpu_percent', side_effect=Exception("psutil error")):
            resource_check = SystemResourcesHealthCheck()
            result = await resource_check.check()
            
            assert result["status"] == HealthStatus.UNKNOWN
            assert "Could not check system resources" in result["message"]

# ============================================================================
# COMPONENT AVAILABILITY HEALTH CHECK TESTS  
# ============================================================================

class TestComponentAvailabilityHealthCheck:
    """Test component availability monitoring"""
    
    @pytest.mark.asyncio
    async def test_component_availability_all_available(self):
        """Test component availability when all components are available"""
        # This test may need to be adjusted based on actual available components
        comp_check = ComponentAvailabilityHealthCheck()
        result = await comp_check.check()
        
        assert result["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "components" in result
        assert "summary" in result
        assert result["summary"]["total"] > 0
    
    @pytest.mark.asyncio
    async def test_component_availability_with_missing_components(self):
        """Test component availability with some missing components"""
        # Mock import failures for some modules
        original_import = __builtins__.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == "backend.middleware.validation":
                raise ImportError("Module not found")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            comp_check = ComponentAvailabilityHealthCheck()
            result = await comp_check.check()
            
            assert result["status"] in [HealthStatus.DEGRADED, HealthStatus.HEALTHY]
            assert result["summary"]["unavailable"] >= 0

# ============================================================================
# PERFORMANCE HEALTH CHECK TESTS
# ============================================================================

class TestPerformanceHealthCheck:
    """Test performance monitoring"""
    
    @pytest.mark.asyncio
    async def test_performance_health_check_normal(self):
        """Test performance health check with normal performance"""
        perf_check = PerformanceHealthCheck()
        result = await perf_check.check()
        
        assert result["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "metrics" in result
        assert "uptime_seconds" in result["metrics"]
        assert "performance_timings" in result["metrics"]
        assert result["metrics"]["uptime_seconds"] >= 0
    
    @pytest.mark.asyncio
    async def test_performance_timing_measurements(self):
        """Test that performance timing measurements are reasonable"""
        perf_check = PerformanceHealthCheck()
        result = await perf_check.check()
        
        timings = result["metrics"]["performance_timings"]
        
        # All timing measurements should be positive
        assert timings["basic_operation_ms"] >= 0
        assert timings["filesystem_operation_ms"] >= 0
        assert timings["total_check_ms"] >= 0
        
        # Total should be greater than or equal to sum of parts
        assert timings["total_check_ms"] >= timings["basic_operation_ms"]

# ============================================================================
# HEALTH MONITOR INTEGRATION TESTS
# ============================================================================

class TestHealthMonitor:
    """Test HealthMonitor integration"""
    
    @pytest.mark.asyncio
    async def test_health_monitor_initialization(self):
        """Test health monitor initialization"""
        monitor = HealthMonitor()
        
        assert len(monitor.checks) == 5  # All health check types
        assert monitor.last_full_check is None
        assert len(monitor.check_history) == 0
    
    @pytest.mark.asyncio
    async def test_health_monitor_run_all_checks(self):
        """Test running all health checks"""
        monitor = HealthMonitor()
        result = await monitor.run_all_checks()
        
        assert "overall_status" in result
        assert "overall_message" in result
        assert "checks" in result
        assert "summary" in result
        assert "timestamp" in result
        assert "check_duration_ms" in result
        
        # Should have results for all check types
        assert len(result["checks"]) == 5
        assert "database" in result["checks"]
        assert "gcp_apis" in result["checks"]
        assert "system_resources" in result["checks"]
        assert "components" in result["checks"]
        assert "performance" in result["checks"]
        
        # Summary should have counts
        summary = result["summary"]
        assert summary["total_checks"] == 5
        assert summary["healthy"] + summary["degraded"] + summary["unhealthy"] + summary["unknown"] == 5
    
    @pytest.mark.asyncio
    async def test_health_monitor_quick_status_no_history(self):
        """Test quick status when no checks have been run"""
        monitor = HealthMonitor()
        result = await monitor.get_quick_status()
        
        assert result["status"] == HealthStatus.UNKNOWN
        assert "No health checks run" in result["message"]
        assert result["last_check"] is None
    
    @pytest.mark.asyncio
    async def test_health_monitor_quick_status_with_history(self):
        """Test quick status after running checks"""
        monitor = HealthMonitor()
        
        # Run checks first
        await monitor.run_all_checks()
        
        # Then get quick status
        result = await monitor.get_quick_status()
        
        assert result["status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "last_check" in result
        assert "summary" in result
    
    @pytest.mark.asyncio 
    async def test_health_monitor_history_management(self):
        """Test health check history management"""
        monitor = HealthMonitor()
        
        # Run multiple checks
        for _ in range(3):
            await monitor.run_all_checks()
            await asyncio.sleep(0.01)  # Small delay between checks
        
        history = monitor.get_health_history()
        
        assert len(history) == 3
        # Should be sorted by timestamp (most recent last)
        for i in range(len(history) - 1):
            assert history[i]["timestamp"] <= history[i + 1]["timestamp"]
    
    @pytest.mark.asyncio
    async def test_health_monitor_history_limit(self):
        """Test health check history size limit"""
        monitor = HealthMonitor()
        monitor.max_history = 5
        
        # Run more checks than the limit
        for _ in range(7):
            await monitor.run_all_checks()
        
        history = monitor.get_health_history()
        assert len(history) == 5  # Should be limited to max_history

# ============================================================================
# HEALTH API ENDPOINT TESTS
# ============================================================================

class TestHealthAPIEndpoints:
    """Test health monitoring API endpoints"""
    
    def test_comprehensive_health_endpoint(self):
        """Test comprehensive health check endpoint"""
        response = client.get("/api/v1/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have comprehensive health check structure
        assert "overall_status" in data
        assert "overall_message" in data
        assert "checks" in data
        assert "summary" in data
        assert "timestamp" in data
    
    def test_quick_health_endpoint(self):
        """Test quick health check endpoint"""
        response = client.get("/api/v1/health/quick")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data
    
    def test_system_status_endpoint(self):
        """Test system status endpoint"""
        response = client.get("/api/v1/health/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system_status" in data
        assert "status_message" in data
        assert "last_updated" in data
        assert "components" in data
    
    def test_health_history_endpoint(self):
        """Test health history endpoint"""
        response = client.get("/api/v1/health/history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "history" in data
        assert "count" in data
        assert isinstance(data["history"], list)
    
    def test_health_history_with_limit(self):
        """Test health history endpoint with limit parameter"""
        response = client.get("/api/v1/health/history?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["limit"] == 5
        assert len(data["history"]) <= 5
    
    def test_component_status_endpoint(self):
        """Test component status endpoint"""
        response = client.get("/api/v1/health/components")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "component_status" in data
        assert "message" in data
        assert "components" in data or "mode" in data  # May fallback
    
    def test_system_resources_endpoint(self):
        """Test system resources endpoint"""
        response = client.get("/api/v1/health/resources")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "resource_status" in data
        assert "message" in data
    
    def test_performance_metrics_endpoint(self):
        """Test performance metrics endpoint"""
        response = client.get("/api/v1/health/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "performance_status" in data
        assert "message" in data
    
    def test_database_health_endpoint(self):
        """Test database health endpoint"""
        response = client.get("/api/v1/health/database")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "database_status" in data
        assert "message" in data
    
    def test_gcp_connectivity_endpoint(self):
        """Test GCP connectivity endpoint"""
        response = client.get("/api/v1/health/gcp")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "gcp_status" in data
        assert "message" in data

# ============================================================================
# MAIN HEALTH ENDPOINT INTEGRATION TESTS
# ============================================================================

class TestMainHealthEndpoint:
    """Test enhanced main health endpoint"""
    
    def test_main_health_endpoint_enhanced(self):
        """Test main health endpoint has been enhanced"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have enhanced features
        assert data["features"]["comprehensive_monitoring"] is True
        assert "health_summary" in data
        
        # Should have comprehensive health endpoints listed
        endpoints = data["endpoints"]
        assert "health_comprehensive" in endpoints
        assert "health_quick" in endpoints
        assert "health_status" in endpoints
        assert "health_history" in endpoints
        assert "health_components" in endpoints
        assert "health_resources" in endpoints
        assert "health_performance" in endpoints
        assert "health_database" in endpoints
        assert "health_gcp" in endpoints
        
        # Should mention TASK-007 in notes
        notes = " ".join(data["notes"])
        assert "TASK-007" in notes

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestHealthMonitoringErrorHandling:
    """Test error handling in health monitoring"""
    
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self):
        """Test health monitor handles check exceptions gracefully"""
        monitor = HealthMonitor()
        
        # Mock one check to raise an exception
        original_check = monitor.checks[0].check
        monitor.checks[0].check = AsyncMock(side_effect=Exception("Test error"))
        
        try:
            result = await monitor.run_all_checks()
            
            # Should still return a result
            assert "overall_status" in result
            assert "checks" in result
            
            # The failed check should be marked as unknown
            failed_check_name = monitor.checks[0].name
            assert result["checks"][failed_check_name]["status"] == HealthStatus.UNKNOWN
            
        finally:
            # Restore original check
            monitor.checks[0].check = original_check
    
    def test_health_api_fallback_behavior(self):
        """Test health API fallback when monitoring is unavailable"""
        # This tests the fallback behavior in the API endpoints
        with patch('backend.api.health.HEALTH_MONITORING_AVAILABLE', False):
            response = client.get("/api/v1/health/quick")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "mode" in data
            assert data["mode"] == "fallback"

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestHealthMonitoringPerformance:
    """Test health monitoring performance"""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test that health checks complete within reasonable time"""
        monitor = HealthMonitor()
        
        start_time = time.time()
        result = await monitor.run_all_checks()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Health checks should complete within 5 seconds
        assert duration < 5.0
        
        # Should also report its own duration
        assert result["check_duration_ms"] > 0
        assert result["check_duration_ms"] < 5000
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health checks don't interfere"""
        monitor = HealthMonitor()
        
        # Run multiple health checks concurrently
        tasks = [monitor.run_all_checks() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert "overall_status" in result
            assert result["overall_status"] in [
                HealthStatus.HEALTHY, HealthStatus.DEGRADED, 
                HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN
            ]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])