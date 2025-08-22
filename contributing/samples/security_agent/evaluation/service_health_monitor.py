#!/usr/bin/env python3
"""
Service Health Monitor
======================

Real-time health monitoring for the ADK Security Agent services.
Provides continuous monitoring of service status, performance metrics,
and operational health indicators.
"""

import asyncio
import json
import logging
import time
import requests
import psutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Service health status"""
    timestamp: str
    service_name: str
    status: str  # UP, DOWN, DEGRADED
    response_time_ms: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    error_count: int
    uptime_seconds: float


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    url: str
    timeout: int = 5
    expected_status: int = 200


class ServiceHealthMonitor:
    """Real-time service health monitoring"""
    
    def __init__(self, config_file: str = None):
        """Initialize the health monitor"""
        self.config = self._load_config(config_file)
        self.endpoints = self._setup_endpoints()
        self.health_history = []
        self.is_running = False
        self.start_time = time.time()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "backend_url": "http://localhost:8000",
            "frontend_url": "http://localhost:8501", 
            "database_path": "../backend/cache/gcp_data.db",
            "check_interval": 30,  # seconds
            "alert_thresholds": {
                "response_time_ms": 2000,
                "cpu_percent": 80,
                "memory_percent": 85,
                "error_rate": 5
            },
            "retention_hours": 24
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_endpoints(self) -> List[ServiceEndpoint]:
        """Setup service endpoints to monitor"""
        base_url = self.config["backend_url"]
        
        return [
            ServiceEndpoint("health", f"{base_url}/health"),
            ServiceEndpoint("custom_roles_stats", f"{base_url}/api/v1/custom-roles/stats"),
            ServiceEndpoint("knowledge_stats", f"{base_url}/api/v1/knowledge/stats"),
            ServiceEndpoint("iam_analysis", f"{base_url}/api/v1/iam/policies"),
            ServiceEndpoint("storage_analysis", f"{base_url}/api/v1/storage/buckets")
        ]
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        logger.info("🔍 Starting Service Health Monitor")
        self.is_running = True
        
        while self.is_running:
            try:
                # Check all endpoints
                health_results = await self._check_all_endpoints()
                
                # Check system resources
                system_health = self._check_system_health()
                
                # Check database health
                db_health = self._check_database_health()
                
                # Combine results
                overall_status = self._calculate_overall_status(
                    health_results, system_health, db_health
                )
                
                # Store results
                self._store_health_data(overall_status)
                
                # Check for alerts
                self._check_alerts(overall_status)
                
                # Log status
                self._log_status(overall_status)
                
                # Wait for next check
                await asyncio.sleep(self.config["check_interval"])
                
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                await asyncio.sleep(5)  # Short retry interval
    
    async def _check_all_endpoints(self) -> List[Dict[str, Any]]:
        """Check all configured endpoints"""
        results = []
        
        for endpoint in self.endpoints:
            result = await self._check_endpoint(endpoint)
            results.append(result)
        
        return results
    
    async def _check_endpoint(self, endpoint: ServiceEndpoint) -> Dict[str, Any]:
        """Check a single endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(endpoint.url, timeout=endpoint.timeout)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            status = "UP" if response.status_code == endpoint.expected_status else "DEGRADED"
            
            return {
                "name": endpoint.name,
                "url": endpoint.url,
                "status": status,
                "response_time_ms": response_time,
                "status_code": response.status_code,
                "error": None
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return {
                "name": endpoint.name,
                "url": endpoint.url,
                "status": "DOWN",
                "response_time_ms": response_time,
                "status_code": 0,
                "error": str(e)
            }
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            "uptime_seconds": time.time() - psutil.boot_time()
        }
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        db_path = self.config.get("database_path", "../backend/cache/gcp_data.db")
        
        try:
            start_time = time.time()
            
            if not Path(db_path).exists():
                return {
                    "status": "DOWN",
                    "error": "Database file not found",
                    "response_time_ms": 0
                }
            
            # Test connection and query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Test a sample query
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            sample_table = cursor.fetchone()
            
            if sample_table:
                cursor.execute(f"SELECT COUNT(*) FROM {sample_table[0]} LIMIT 1")
                cursor.fetchone()
            
            conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "UP",
                "response_time_ms": response_time,
                "table_count": table_count,
                "error": None
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "DOWN",
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    def _calculate_overall_status(self, endpoint_results: List[Dict], 
                                 system_health: Dict, db_health: Dict) -> HealthStatus:
        """Calculate overall service status"""
        
        # Determine overall status
        endpoint_statuses = [r["status"] for r in endpoint_results]
        
        if db_health["status"] == "DOWN":
            overall_status = "DOWN"
        elif "DOWN" in endpoint_statuses:
            overall_status = "DEGRADED"
        elif "DEGRADED" in endpoint_statuses:
            overall_status = "DEGRADED"
        else:
            overall_status = "UP"
        
        # Calculate average response time
        response_times = [r["response_time_ms"] for r in endpoint_results if r["response_time_ms"] > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Count errors
        error_count = sum(1 for r in endpoint_results if r["status"] != "UP")
        
        return HealthStatus(
            timestamp=datetime.now().isoformat(),
            service_name="security_agent",
            status=overall_status,
            response_time_ms=avg_response_time,
            cpu_percent=system_health["cpu_percent"],
            memory_percent=system_health["memory_percent"],
            disk_percent=system_health["disk_percent"],
            error_count=error_count,
            uptime_seconds=time.time() - self.start_time
        )
    
    def _store_health_data(self, health_status: HealthStatus):
        """Store health data for trend analysis"""
        self.health_history.append(health_status)
        
        # Keep only recent data (retention policy)
        retention_seconds = self.config["retention_hours"] * 3600
        cutoff_time = datetime.now().timestamp() - retention_seconds
        
        self.health_history = [
            h for h in self.health_history 
            if datetime.fromisoformat(h.timestamp).timestamp() > cutoff_time
        ]
    
    def _check_alerts(self, health_status: HealthStatus):
        """Check for alert conditions"""
        thresholds = self.config["alert_thresholds"]
        alerts = []
        
        # Response time alert
        if health_status.response_time_ms > thresholds["response_time_ms"]:
            alerts.append(f"High response time: {health_status.response_time_ms:.1f}ms")
        
        # CPU alert
        if health_status.cpu_percent > thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {health_status.cpu_percent:.1f}%")
        
        # Memory alert
        if health_status.memory_percent > thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {health_status.memory_percent:.1f}%")
        
        # Service down alert
        if health_status.status == "DOWN":
            alerts.append("Service is DOWN")
        elif health_status.status == "DEGRADED":
            alerts.append("Service is DEGRADED")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"🚨 ALERT: {alert}")
    
    def _log_status(self, health_status: HealthStatus):
        """Log current health status"""
        status_icon = {
            "UP": "✅",
            "DEGRADED": "⚠️",
            "DOWN": "❌"
        }.get(health_status.status, "❓")
        
        logger.info(
            f"{status_icon} Status: {health_status.status} | "
            f"Response: {health_status.response_time_ms:.1f}ms | "
            f"CPU: {health_status.cpu_percent:.1f}% | "
            f"Memory: {health_status.memory_percent:.1f}% | "
            f"Errors: {health_status.error_count}"
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.health_history:
            return {"status": "NO_DATA", "message": "No health data available"}
        
        latest = self.health_history[-1]
        
        # Calculate trends (last 10 checks)
        recent_checks = self.health_history[-10:]
        avg_response_time = sum(h.response_time_ms for h in recent_checks) / len(recent_checks)
        avg_cpu = sum(h.cpu_percent for h in recent_checks) / len(recent_checks)
        avg_memory = sum(h.memory_percent for h in recent_checks) / len(recent_checks)
        
        return {
            "current_status": latest.status,
            "last_check": latest.timestamp,
            "uptime_hours": latest.uptime_seconds / 3600,
            "current_metrics": {
                "response_time_ms": latest.response_time_ms,
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_percent,
                "error_count": latest.error_count
            },
            "trends": {
                "avg_response_time_ms": avg_response_time,
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory
            },
            "total_checks": len(self.health_history)
        }
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        logger.info("⏹️ Stopping Service Health Monitor")
        self.is_running = False
    
    def save_health_report(self, filename: str = "health_report.json"):
        """Save health report to file"""
        report = {
            "report_generated": datetime.now().isoformat(),
            "monitoring_config": self.config,
            "health_summary": self.get_health_summary(),
            "recent_history": [asdict(h) for h in self.health_history[-100:]]  # Last 100 checks
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Health report saved to {filename}")


def main():
    """Run health monitor"""
    monitor = ServiceHealthMonitor()
    
    try:
        # Start monitoring in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run monitoring
        print("🔍 Starting Service Health Monitor...")
        print("Press Ctrl+C to stop")
        
        loop.run_until_complete(monitor.start_monitoring())
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitor...")
        monitor.stop_monitoring()
        
        # Save final report
        monitor.save_health_report()
        
        # Print summary
        summary = monitor.get_health_summary()
        print(f"\n📊 Final Status: {summary.get('current_status', 'UNKNOWN')}")
        print(f"Total Checks: {summary.get('total_checks', 0)}")
        print(f"Uptime: {summary.get('uptime_hours', 0):.1f} hours")
        
    except Exception as e:
        logger.error(f"❌ Monitor error: {e}")


if __name__ == "__main__":
    main()