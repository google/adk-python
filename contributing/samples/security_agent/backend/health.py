"""
Comprehensive Health Check Monitoring System - TASK-007

Provides detailed health checks for all system components including:
- Database connectivity 
- GCP API access
- External service dependencies
- System resources
- Component availability
- Performance metrics
"""

import asyncio
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import sqlite3
import tempfile

# Configure logging
logger = logging.getLogger(__name__)

class HealthStatus:
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentHealthCheck:
    """Base class for component health checks"""
    
    def __init__(self, name: str, critical: bool = False):
        self.name = name
        self.critical = critical
        self.last_check = None
        self.last_result = None
        
    async def check(self) -> Dict[str, Any]:
        """Override this method to implement specific health check"""
        return {
            "status": HealthStatus.UNKNOWN,
            "message": "Health check not implemented",
            "timestamp": datetime.now().isoformat()
        }

class DatabaseHealthCheck(ComponentHealthCheck):
    """SQLite database connectivity check"""
    
    def __init__(self):
        super().__init__("database", critical=True)
        
    async def check(self) -> Dict[str, Any]:
        """Check SQLite database connectivity"""
        try:
            # Try to create a temporary database and perform basic operations
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
                temp_db_path = temp_file.name
                
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            
            # Test basic operations
            start_time = time.time()
            cursor.execute("CREATE TABLE test_health (id INTEGER PRIMARY KEY, timestamp TEXT)")
            cursor.execute("INSERT INTO test_health (timestamp) VALUES (?)", (datetime.now().isoformat(),))
            cursor.execute("SELECT COUNT(*) FROM test_health")
            result = cursor.fetchone()
            conn.commit()
            conn.close()
            
            # Cleanup
            os.unlink(temp_db_path)
            
            response_time = time.time() - start_time
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Database operations successful",
                "response_time_ms": round(response_time * 1000, 2),
                "test_result": f"Created test table, inserted 1 record, count: {result[0]}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database connectivity failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class GCPAPIHealthCheck(ComponentHealthCheck):
    """GCP API connectivity and credentials check"""
    
    def __init__(self):
        super().__init__("gcp_apis", critical=False)
        
    async def check(self) -> Dict[str, Any]:
        """Check GCP API connectivity and credentials"""
        try:
            # Check if credentials are configured
            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            
            status_info = {
                "credentials_configured": creds_path is not None,
                "credentials_path": creds_path,
                "project_id": project_id,
                "api_checks": {}
            }
            
            # Test basic Google Cloud client creation
            try:
                from google.auth import default
                from google.cloud import resource_manager
                
                start_time = time.time()
                credentials, detected_project = default()
                
                if project_id:
                    # Test Resource Manager API
                    client = resource_manager.Client(credentials=credentials)
                    # Simple API call to verify connectivity
                    project = client.fetch_project(project_id)
                    
                    status_info["api_checks"]["resource_manager"] = {
                        "status": "accessible",
                        "project_name": project.name if hasattr(project, 'name') else "Unknown",
                        "response_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                else:
                    status_info["api_checks"]["resource_manager"] = {
                        "status": "no_project_id",
                        "message": "GOOGLE_CLOUD_PROJECT not configured"
                    }
                    
            except ImportError:
                status_info["api_checks"]["google_cloud_libs"] = {
                    "status": "not_available",
                    "message": "Google Cloud libraries not installed"
                }
            except Exception as api_error:
                status_info["api_checks"]["resource_manager"] = {
                    "status": "error",
                    "message": str(api_error)
                }
            
            # Determine overall status
            if creds_path and project_id:
                if any(check.get("status") == "accessible" for check in status_info["api_checks"].values()):
                    status = HealthStatus.HEALTHY
                    message = "GCP APIs accessible"
                elif any(check.get("status") == "error" for check in status_info["api_checks"].values()):
                    status = HealthStatus.DEGRADED
                    message = "GCP APIs configured but some errors"
                else:
                    status = HealthStatus.DEGRADED
                    message = "GCP APIs configured but not tested"
            else:
                status = HealthStatus.DEGRADED
                message = "GCP credentials not fully configured"
            
            return {
                "status": status,
                "message": message,
                "details": status_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GCP API health check failed: {e}")
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"GCP API check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class SystemResourcesHealthCheck(ComponentHealthCheck):
    """System resources (CPU, memory, disk) health check"""
    
    def __init__(self):
        super().__init__("system_resources", critical=True)
        
    async def check(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate status based on thresholds
            status_factors = []
            
            # CPU check
            if cpu_percent > 90:
                cpu_status = "critical"
                status_factors.append("high_cpu")
            elif cpu_percent > 70:
                cpu_status = "warning"
            else:
                cpu_status = "normal"
            
            # Memory check  
            if memory.percent > 90:
                memory_status = "critical"
                status_factors.append("high_memory")
            elif memory.percent > 80:
                memory_status = "warning"
            else:
                memory_status = "normal"
            
            # Disk check
            if disk.percent > 95:
                disk_status = "critical"
                status_factors.append("high_disk")
            elif disk.percent > 85:
                disk_status = "warning"
            else:
                disk_status = "normal"
                
            # Determine overall status
            if status_factors:
                overall_status = HealthStatus.UNHEALTHY
                message = f"Resource constraints: {', '.join(status_factors)}"
            elif any(s == "warning" for s in [cpu_status, memory_status, disk_status]):
                overall_status = HealthStatus.DEGRADED
                message = "Some resources under pressure"
            else:
                overall_status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return {
                "status": overall_status,
                "message": message,
                "metrics": {
                    "cpu": {
                        "percent": cpu_percent,
                        "status": cpu_status
                    },
                    "memory": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "percent": memory.percent,
                        "status": memory_status
                    },
                    "disk": {
                        "total_gb": round(disk.total / (1024**3), 2),
                        "used_gb": round(disk.used / (1024**3), 2),
                        "percent": disk.percent,
                        "status": disk_status
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System resources health check failed: {e}")
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"Could not check system resources: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class ComponentAvailabilityHealthCheck(ComponentHealthCheck):
    """Check availability of critical application components"""
    
    def __init__(self):
        super().__init__("components", critical=False)
        
    async def check(self) -> Dict[str, Any]:
        """Check availability of application components"""
        try:
            components = {}
            
            # Check middleware
            try:
                from .middleware.validation import InputValidationMiddleware
                components["input_validation"] = {"status": "available", "critical": True}
            except ImportError:
                components["input_validation"] = {"status": "unavailable", "critical": True}
            
            try:
                from .middleware.rate_limiter import RateLimitMiddleware
                components["rate_limiting"] = {"status": "available", "critical": False}
            except ImportError:
                components["rate_limiting"] = {"status": "unavailable", "critical": False}
            
            # Check API routers
            routers_to_check = [
                ("sessions", ".api.sessions", False),
                ("security", ".api.security", True),
                ("iam", ".api.iam", True),
                ("gcp", ".api.gcp", False),
                                ("monitoring", ".api.monitoring", False),
                                ("storage", ".api.storage", False),
                ("asset_inventory", ".api.asset_inventory", True),
                ("keys", ".api.keys", False),
                ("recommendations", ".api.recommendations", False)
            ]
            
            for router_name, module_path, critical in routers_to_check:
                try:
                    __import__(module_path)
                    components[f"{router_name}_router"] = {"status": "available", "critical": critical}
                except ImportError:
                    components[f"{router_name}_router"] = {"status": "unavailable", "critical": critical}
            
            # Check external libraries
            libraries_to_check = [
                ("google_cloud", "google.cloud"),
                ("psutil", "psutil"),
                ("fastapi", "fastapi"),
                ("uvicorn", "uvicorn")
            ]
            
            for lib_name, import_path in libraries_to_check:
                try:
                    __import__(import_path)
                    components[f"{lib_name}_lib"] = {"status": "available", "critical": True}
                except ImportError:
                    components[f"{lib_name}_lib"] = {"status": "unavailable", "critical": True}
            
            # Calculate overall status
            critical_unavailable = [name for name, info in components.items() 
                                  if info["critical"] and info["status"] == "unavailable"]
            unavailable_count = len([info for info in components.values() if info["status"] == "unavailable"])
            
            if critical_unavailable:
                status = HealthStatus.DEGRADED
                message = f"Critical components unavailable: {', '.join(critical_unavailable)}"
            elif unavailable_count > 0:
                status = HealthStatus.DEGRADED  
                message = f"{unavailable_count} non-critical components unavailable"
            else:
                status = HealthStatus.HEALTHY
                message = "All components available"
            
            return {
                "status": status,
                "message": message,
                "components": components,
                "summary": {
                    "total": len(components),
                    "available": len([c for c in components.values() if c["status"] == "available"]),
                    "unavailable": unavailable_count,
                    "critical_unavailable": len(critical_unavailable)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Component availability health check failed: {e}")
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"Could not check component availability: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class PerformanceHealthCheck(ComponentHealthCheck):
    """Application performance metrics check"""
    
    def __init__(self):
        super().__init__("performance", critical=False)
        self.start_time = time.time()
        
    async def check(self) -> Dict[str, Any]:
        """Check application performance metrics"""
        try:
            # Basic performance tests
            uptime = time.time() - self.start_time
            
            # Test response times for key operations
            start_time = time.time()
            
            # Simulate a basic database operation timing
            test_start = time.time()
            await asyncio.sleep(0.001)  # Simulate minimal async operation
            basic_operation_time = time.time() - test_start
            
            # Test file system access
            fs_start = time.time()
            temp_file = tempfile.NamedTemporaryFile(delete=True)
            temp_file.write(b"health check test")
            temp_file.flush()
            temp_file.close()
            fs_operation_time = time.time() - fs_start
            
            total_check_time = time.time() - start_time
            
            # Performance thresholds
            performance_issues = []
            if basic_operation_time > 0.1:
                performance_issues.append("slow_async_operations")
            if fs_operation_time > 0.1:
                performance_issues.append("slow_filesystem")
            if total_check_time > 1.0:
                performance_issues.append("slow_overall_performance")
            
            if performance_issues:
                status = HealthStatus.DEGRADED
                message = f"Performance issues detected: {', '.join(performance_issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Performance metrics normal"
            
            return {
                "status": status,
                "message": message,
                "metrics": {
                    "uptime_seconds": round(uptime, 2),
                    "uptime_readable": str(timedelta(seconds=int(uptime))),
                    "performance_timings": {
                        "basic_operation_ms": round(basic_operation_time * 1000, 2),
                        "filesystem_operation_ms": round(fs_operation_time * 1000, 2),
                        "total_check_ms": round(total_check_time * 1000, 2)
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance health check failed: {e}")
            return {
                "status": HealthStatus.UNKNOWN,
                "message": f"Could not check performance: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

class HealthMonitor:
    """Main health monitoring service"""
    
    def __init__(self):
        self.checks = [
            DatabaseHealthCheck(),
            GCPAPIHealthCheck(), 
            SystemResourcesHealthCheck(),
            ComponentAvailabilityHealthCheck(),
            PerformanceHealthCheck()
        ]
        self.last_full_check = None
        self.check_history = []
        self.max_history = 100
        
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        start_time = time.time()
        results = {}
        
        # Run all checks concurrently
        tasks = [check.check() for check in self.checks]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for check, result in zip(self.checks, check_results):
            if isinstance(result, Exception):
                results[check.name] = {
                    "status": HealthStatus.UNKNOWN,
                    "message": f"Health check failed with exception: {str(result)}",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                results[check.name] = result
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(results)
        
        # Create comprehensive response
        response = {
            "overall_status": overall_status["status"],
            "overall_message": overall_status["message"],
            "check_duration_ms": round((time.time() - start_time) * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy": len([r for r in results.values() if r["status"] == HealthStatus.HEALTHY]),
                "degraded": len([r for r in results.values() if r["status"] == HealthStatus.DEGRADED]), 
                "unhealthy": len([r for r in results.values() if r["status"] == HealthStatus.UNHEALTHY]),
                "unknown": len([r for r in results.values() if r["status"] == HealthStatus.UNKNOWN])
            }
        }
        
        # Store in history
        self._store_check_result(response)
        self.last_full_check = datetime.now()
        
        return response
    
    def _calculate_overall_status(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Calculate overall system status from individual check results"""
        critical_checks = [name for name in results.keys() 
                         if any(check.name == name and check.critical for check in self.checks)]
        
        # Count statuses
        statuses = [result["status"] for result in results.values()]
        unhealthy_count = statuses.count(HealthStatus.UNHEALTHY)
        degraded_count = statuses.count(HealthStatus.DEGRADED)
        unknown_count = statuses.count(HealthStatus.UNKNOWN)
        
        # Check critical systems
        critical_issues = []
        for check_name in critical_checks:
            if check_name in results:
                if results[check_name]["status"] == HealthStatus.UNHEALTHY:
                    critical_issues.append(check_name)
        
        # Determine overall status
        if critical_issues:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Critical systems unhealthy: {', '.join(critical_issues)}"
            }
        elif unhealthy_count > 0:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"{unhealthy_count} systems unhealthy (non-critical)"
            }
        elif degraded_count > 0:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"{degraded_count} systems degraded"
            }
        elif unknown_count > 0:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"{unknown_count} systems status unknown"
            }
        else:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "All systems healthy"
            }
    
    def _store_check_result(self, result: Dict[str, Any]):
        """Store check result in history"""
        self.check_history.append({
            "timestamp": result["timestamp"],
            "overall_status": result["overall_status"],
            "summary": result["summary"]
        })
        
        # Maintain history size limit
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
    
    async def get_quick_status(self) -> Dict[str, Any]:
        """Get a quick status without running full checks"""
        if not self.last_full_check:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "No health checks run yet",
                "last_check": None
            }
        
        # Check if last check is recent (within 5 minutes)
        time_since_check = datetime.now() - self.last_full_check
        if time_since_check > timedelta(minutes=5):
            stale_message = f"Last check was {time_since_check} ago (may be stale)"
        else:
            stale_message = "Recent health check data"
        
        if self.check_history:
            latest = self.check_history[-1]
            return {
                "status": latest["overall_status"],
                "message": stale_message,
                "last_check": latest["timestamp"],
                "summary": latest["summary"]
            }
        else:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "No check history available",
                "last_check": self.last_full_check.isoformat() if self.last_full_check else None
            }
    
    def get_health_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get health check history"""
        return self.check_history[-limit:] if self.check_history else []

# Global health monitor instance
health_monitor = HealthMonitor()