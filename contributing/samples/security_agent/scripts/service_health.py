#!/usr/bin/env python3
"""
Service Health Monitor
Real-time monitoring of all services and their fallback status
"""

import asyncio
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ServiceHealthMonitor:
    """Monitor health of all services in the security agent"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.services = {}
        
    async def check_backend_health(self) -> Dict:
        """Check backend API health"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "data": response.json()
                }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e)
            }
    
    async def check_endpoint(self, endpoint: str, method: str = "GET", payload: Optional[Dict] = None) -> Dict:
        """Check a specific endpoint"""
        try:
            url = f"{self.backend_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=payload or {}, timeout=10)
            else:
                return {"status": "error", "error": f"Unsupported method: {method}"}
            
            return {
                "status": "healthy" if response.status_code in [200, 201] else "degraded",
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "using_fallback": self._check_if_fallback(response)
            }
        except requests.exceptions.Timeout:
            return {"status": "timeout", "error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return {"status": "offline", "error": "Cannot connect to backend"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_if_fallback(self, response) -> bool:
        """Check if response indicates fallback mode"""
        try:
            data = response.json()
            # Check for indicators of fallback mode
            if isinstance(data, dict):
                # Look for mock data indicators
                if data.get("source") == "mock":
                    return True
                if data.get("using_fallback"):
                    return True
                if "mock" in str(data).lower() and "data" in str(data).lower():
                    return True
                # Check if data seems too perfect (likely mock)
                if data.get("data") and isinstance(data["data"], dict):
                    if data["data"].get("total_assets") in [100, 150, 200]:  # Common mock values
                        return True
        except:
            pass
        return False
    
    async def check_all_services(self) -> Dict:
        """Check health of all services"""
        
        # Define all endpoints to check
        endpoints = [
            # Core APIs
            {"name": "Health Check", "endpoint": "/health", "critical": True},
            {"name": "Asset Inventory", "endpoint": "/api/v1/assets/summary?project_id=test", "critical": True},
            {"name": "Asset Snapshot", "endpoint": "/api/v1/assets/snapshot/test", "critical": True},
            
            # Agent APIs
            {"name": "Agent Chat", "endpoint": "/api/v1/agent/chat", "method": "POST", 
             "payload": {"query": "test", "user_id": "test", "project_id": "test"}},
            {"name": "Sessions", "endpoint": "/api/v1/sessions/create", "method": "POST",
             "payload": {"user_id": "test"}},
            
            # Service APIs
            {"name": "GCP Projects", "endpoint": "/api/v1/gcp/projects"},
            {"name": "Security Scan", "endpoint": "/api/v1/security/scan", "method": "POST",
             "payload": {"project_id": "test"}},
            {"name": "Recommendations", "endpoint": "/api/v1/recommendations/test"},
            {"name": "IAM Analysis", "endpoint": "/api/v1/iam/analyze-all-users"},
            {"name": "Storage Buckets", "endpoint": "/api/v1/storage/buckets"},
            {"name": "Network Assets", "endpoint": "/api/v1/network/vpc-networks"},
            {"name": "Compliance Check", "endpoint": "/api/v1/compliance/frameworks"},
            {"name": "Cost Analysis", "endpoint": "/api/v1/cost/optimization"},
            {"name": "Monitoring Metrics", "endpoint": "/api/v1/monitoring/dashboard"},
        ]
        
        results = {}
        
        # Check backend health first
        backend_health = await self.check_backend_health()
        results["backend"] = backend_health
        
        if backend_health["status"] != "healthy":
            print("❌ Backend is not healthy - skipping endpoint checks")
            return results
        
        # Check each endpoint
        for endpoint_config in endpoints:
            name = endpoint_config["name"]
            endpoint = endpoint_config["endpoint"]
            method = endpoint_config.get("method", "GET")
            payload = endpoint_config.get("payload")
            
            print(f"Checking {name}...", end=" ")
            result = await self.check_endpoint(endpoint, method, payload)
            results[name] = result
            
            # Print status
            if result["status"] == "healthy":
                if result.get("using_fallback"):
                    print("⚠️  OK (fallback)")
                else:
                    print("✅ OK")
            elif result["status"] == "degraded":
                print("⚠️  Degraded")
            elif result["status"] == "timeout":
                print("⏱️  Timeout")
            elif result["status"] == "offline":
                print("❌ Offline")
            else:
                print("❌ Error")
        
        return results
    
    def generate_health_report(self, results: Dict) -> str:
        """Generate a health report from results"""
        report = []
        report.append("=" * 60)
        report.append("SERVICE HEALTH REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 60)
        report.append("")
        
        # Overall status
        total_services = len(results) - 1  # Exclude backend from count
        healthy_services = sum(1 for k, v in results.items() 
                              if k != "backend" and v.get("status") == "healthy")
        degraded_services = sum(1 for k, v in results.items()
                               if k != "backend" and v.get("status") == "degraded")
        fallback_services = sum(1 for k, v in results.items()
                               if k != "backend" and v.get("using_fallback"))
        
        report.append("OVERALL STATUS:")
        report.append(f"  Total Services: {total_services}")
        report.append(f"  Healthy: {healthy_services} ({healthy_services/max(total_services,1)*100:.1f}%)")
        report.append(f"  Degraded: {degraded_services}")
        report.append(f"  Using Fallback: {fallback_services}")
        report.append("")
        
        # Backend status
        backend = results.get("backend", {})
        report.append("BACKEND STATUS:")
        if backend.get("status") == "healthy":
            report.append(f"  ✅ Healthy (Response time: {backend.get('response_time_ms', 0):.1f}ms)")
        else:
            report.append(f"  ❌ {backend.get('status', 'unknown').title()}")
            if backend.get("error"):
                report.append(f"     Error: {backend['error']}")
        report.append("")
        
        # Service details
        report.append("SERVICE DETAILS:")
        for service_name, status in results.items():
            if service_name == "backend":
                continue
            
            status_emoji = {
                "healthy": "✅",
                "degraded": "⚠️",
                "timeout": "⏱️",
                "offline": "❌",
                "error": "❌"
            }.get(status.get("status", "unknown"), "❓")
            
            fallback_text = " [FALLBACK]" if status.get("using_fallback") else ""
            response_time = f" ({status.get('response_time_ms', 0):.1f}ms)" if status.get("response_time_ms") else ""
            
            report.append(f"  {status_emoji} {service_name}: {status.get('status', 'unknown')}{fallback_text}{response_time}")
            
            if status.get("error"):
                report.append(f"     Error: {status['error']}")
        
        report.append("")
        report.append("=" * 60)
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if fallback_services > 0:
            report.append(f"  ⚠️  {fallback_services} services are using fallback/mock data")
            report.append("     Consider enabling the following GCP APIs:")
            report.append("     - Cloud Asset API")
            report.append("     - Recommender API")
            report.append("     - Security Command Center API")
            report.append("     - Cloud Resource Manager API")
        
        if degraded_services > 0:
            report.append(f"  ⚠️  {degraded_services} services are degraded")
            report.append("     Check logs for errors and ensure proper configuration")
        
        if healthy_services == total_services:
            report.append("  ✅ All services are healthy!")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    async def continuous_monitor(self, interval_seconds: int = 30):
        """Continuously monitor services"""
        print(f"Starting continuous monitoring (checking every {interval_seconds} seconds)")
        print("Press Ctrl+C to stop")
        print("")
        
        while True:
            try:
                results = await self.check_all_services()
                report = self.generate_health_report(results)
                
                # Clear screen (works on Unix/Linux/Mac)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print(report)
                
                # Save to file
                with open("health_report.txt", "w") as f:
                    f.write(report)
                
                # Wait before next check
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                await asyncio.sleep(interval_seconds)

async def main():
    """Main entry point"""
    monitor = ServiceHealthMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        await monitor.continuous_monitor(interval)
    else:
        print("🏥 Running health check...")
        print("")
        results = await monitor.check_all_services()
        report = monitor.generate_health_report(results)
        print(report)
        
        # Save report
        with open("health_report.txt", "w") as f:
            f.write(report)
        print("\n📄 Report saved to health_report.txt")
        
        # Usage help
        print("\n💡 Tip: Use 'python service_health.py continuous [interval]' for continuous monitoring")

if __name__ == "__main__":
    asyncio.run(main())