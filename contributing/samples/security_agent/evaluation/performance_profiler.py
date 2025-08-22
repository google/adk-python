#!/usr/bin/env python3
"""
Performance Profiler
====================

Advanced performance profiling tool for the ADK Security Agent.
Provides detailed performance analysis, bottleneck identification,
and optimization recommendations.
"""

import asyncio
import json
import logging
import time
import requests
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import concurrent.futures
import cProfile
import pstats
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    test_name: str
    timestamp: str
    duration_seconds: float
    requests_sent: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    error_rate_percent: float


@dataclass
class LoadTestConfig:
    """Load test configuration"""
    endpoint: str
    duration_seconds: int
    concurrent_users: int
    ramp_up_seconds: int
    think_time_ms: int


class PerformanceProfiler:
    """Advanced performance profiling and load testing"""
    
    def __init__(self, config_file: str = None):
        """Initialize the performance profiler"""
        self.config = self._load_config(config_file)
        self.results = []
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load profiler configuration"""
        default_config = {
            "backend_url": "http://localhost:8000",
            "test_endpoints": [
                "/api/v1/custom-roles/stats",
                "/api/v1/knowledge/stats",
                "/api/v1/iam/policies",
                "/health"
            ],
            "load_test_scenarios": [
                {"name": "light_load", "users": 5, "duration": 60},
                {"name": "moderate_load", "users": 25, "duration": 120},
                {"name": "heavy_load", "users": 50, "duration": 180}
            ],
            "performance_targets": {
                "response_time_p95_ms": 2000,
                "throughput_rps": 50,
                "error_rate_percent": 1.0,
                "cpu_threshold_percent": 80,
                "memory_threshold_mb": 4096
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        logger.info("⚡ Starting Performance Analysis")
        
        results = {
            "analysis_metadata": {
                "start_time": datetime.now().isoformat(),
                "profiler_version": "1.0.0",
                "target_url": self.config["backend_url"]
            },
            "baseline_performance": {},
            "load_test_results": [],
            "bottleneck_analysis": {},
            "optimization_recommendations": []
        }
        
        # 1. Baseline performance measurement
        logger.info("📊 Measuring baseline performance...")
        results["baseline_performance"] = await self._measure_baseline_performance()
        
        # 2. Load testing scenarios
        logger.info("🚀 Running load test scenarios...")
        for scenario in self.config["load_test_scenarios"]:
            scenario_results = await self._run_load_test_scenario(scenario)
            results["load_test_results"].append(scenario_results)
        
        # 3. Bottleneck analysis
        logger.info("🔍 Analyzing performance bottlenecks...")
        results["bottleneck_analysis"] = self._analyze_bottlenecks(results["load_test_results"])
        
        # 4. Generate recommendations
        logger.info("💡 Generating optimization recommendations...")
        results["optimization_recommendations"] = self._generate_recommendations(results)
        
        results["analysis_metadata"]["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def _measure_baseline_performance(self) -> Dict[str, Any]:
        """Measure baseline performance for each endpoint"""
        baseline_results = {}
        
        for endpoint in self.config["test_endpoints"]:
            logger.info(f"📏 Testing baseline for {endpoint}")
            
            url = f"{self.config['backend_url']}{endpoint}"
            response_times = []
            success_count = 0
            error_count = 0
            
            # Run 50 sequential requests to establish baseline
            for i in range(50):
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=10)
                    end_time = time.time()
                    
                    response_time_ms = (end_time - start_time) * 1000
                    response_times.append(response_time_ms)
                    
                    if response.status_code < 400:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Request failed: {e}")
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            # Calculate statistics
            if response_times:
                response_times.sort()
                baseline_results[endpoint] = {
                    "avg_response_time_ms": statistics.mean(response_times),
                    "min_response_time_ms": min(response_times),
                    "max_response_time_ms": max(response_times),
                    "p50_response_time_ms": response_times[len(response_times)//2],
                    "p95_response_time_ms": response_times[int(len(response_times)*0.95)],
                    "success_rate_percent": (success_count / (success_count + error_count)) * 100,
                    "total_requests": success_count + error_count
                }
            else:
                baseline_results[endpoint] = {
                    "error": "No successful requests",
                    "total_errors": error_count
                }
        
        return baseline_results
    
    async def _run_load_test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific load test scenario"""
        logger.info(f"🎯 Running load test: {scenario['name']}")
        
        # Use the first endpoint for load testing
        endpoint = self.config["test_endpoints"][0]
        url = f"{self.config['backend_url']}{endpoint}"
        
        # Test configuration
        concurrent_users = scenario["users"]
        duration_seconds = scenario["duration"]
        
        # Metrics collection
        response_times = []
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        # System metrics monitoring
        initial_cpu = psutil.cpu_percent()
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        peak_cpu = initial_cpu
        peak_memory = initial_memory
        
        # Create load test tasks
        async def load_test_worker():
            nonlocal response_times, success_count, error_count, peak_cpu, peak_memory
            
            worker_start = time.time()
            
            while time.time() - worker_start < duration_seconds:
                try:
                    request_start = time.time()
                    response = requests.get(url, timeout=5)
                    request_end = time.time()
                    
                    response_time_ms = (request_end - request_start) * 1000
                    response_times.append(response_time_ms)
                    
                    if response.status_code < 400:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
                
                # Monitor system resources
                current_cpu = psutil.cpu_percent()
                current_memory = psutil.virtual_memory().used / 1024 / 1024
                
                peak_cpu = max(peak_cpu, current_cpu)
                peak_memory = max(peak_memory, current_memory)
                
                # Think time between requests
                await asyncio.sleep(0.1)
        
        # Run concurrent workers
        tasks = [load_test_worker() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate results
        total_time = time.time() - start_time
        total_requests = success_count + error_count
        
        if response_times:
            response_times.sort()
            
            return {
                "scenario_name": scenario["name"],
                "concurrent_users": concurrent_users,
                "duration_seconds": total_time,
                "total_requests": total_requests,
                "successful_requests": success_count,
                "failed_requests": error_count,
                "requests_per_second": total_requests / total_time if total_time > 0 else 0,
                "error_rate_percent": (error_count / total_requests) * 100 if total_requests > 0 else 0,
                "response_times": {
                    "avg_ms": statistics.mean(response_times),
                    "min_ms": min(response_times),
                    "max_ms": max(response_times),
                    "p50_ms": response_times[len(response_times)//2],
                    "p95_ms": response_times[int(len(response_times)*0.95)],
                    "p99_ms": response_times[int(len(response_times)*0.99)]
                },
                "resource_usage": {
                    "peak_cpu_percent": peak_cpu,
                    "peak_memory_mb": peak_memory,
                    "cpu_increase": peak_cpu - initial_cpu,
                    "memory_increase_mb": peak_memory - initial_memory
                }
            }
        else:
            return {
                "scenario_name": scenario["name"],
                "error": "No successful requests during load test",
                "total_errors": error_count
            }
    
    def _analyze_bottlenecks(self, load_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        bottlenecks = {
            "identified_bottlenecks": [],
            "performance_degradation": {},
            "resource_constraints": {}
        }
        
        for result in load_test_results:
            if "error" in result:
                continue
                
            scenario_name = result["scenario_name"]
            
            # Check response time degradation
            p95_time = result["response_times"]["p95_ms"]
            target_p95 = self.config["performance_targets"]["response_time_p95_ms"]
            
            if p95_time > target_p95:
                bottlenecks["identified_bottlenecks"].append(
                    f"High response time in {scenario_name}: {p95_time:.1f}ms > {target_p95}ms"
                )
            
            # Check throughput
            rps = result["requests_per_second"]
            target_rps = self.config["performance_targets"]["throughput_rps"]
            
            if rps < target_rps:
                bottlenecks["identified_bottlenecks"].append(
                    f"Low throughput in {scenario_name}: {rps:.1f} RPS < {target_rps} RPS"
                )
            
            # Check error rate
            error_rate = result["error_rate_percent"]
            target_error_rate = self.config["performance_targets"]["error_rate_percent"]
            
            if error_rate > target_error_rate:
                bottlenecks["identified_bottlenecks"].append(
                    f"High error rate in {scenario_name}: {error_rate:.1f}% > {target_error_rate}%"
                )
            
            # Check resource usage
            cpu_usage = result["resource_usage"]["peak_cpu_percent"]
            memory_usage = result["resource_usage"]["peak_memory_mb"]
            
            cpu_threshold = self.config["performance_targets"]["cpu_threshold_percent"]
            memory_threshold = self.config["performance_targets"]["memory_threshold_mb"]
            
            if cpu_usage > cpu_threshold:
                bottlenecks["resource_constraints"]["cpu"] = f"CPU usage {cpu_usage:.1f}% exceeds {cpu_threshold}%"
            
            if memory_usage > memory_threshold:
                bottlenecks["resource_constraints"]["memory"] = f"Memory usage {memory_usage:.1f}MB exceeds {memory_threshold}MB"
            
            # Track performance degradation trends
            bottlenecks["performance_degradation"][scenario_name] = {
                "response_time_factor": p95_time / target_p95,
                "throughput_factor": rps / target_rps,
                "error_rate_factor": error_rate / target_error_rate if target_error_rate > 0 else 0
            }
        
        return bottlenecks
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        bottlenecks = analysis_results.get("bottleneck_analysis", {})
        load_results = analysis_results.get("load_test_results", [])
        
        # Response time recommendations
        high_response_times = any("High response time" in b for b in bottlenecks.get("identified_bottlenecks", []))
        if high_response_times:
            recommendations.extend([
                "Optimize database queries - add indexes for frequently accessed tables",
                "Implement query result caching for expensive operations",
                "Consider using async/await patterns for I/O operations",
                "Profile and optimize hot code paths"
            ])
        
        # Throughput recommendations
        low_throughput = any("Low throughput" in b for b in bottlenecks.get("identified_bottlenecks", []))
        if low_throughput:
            recommendations.extend([
                "Implement connection pooling for database connections",
                "Add horizontal scaling with load balancing",
                "Optimize serialization/deserialization of large objects",
                "Consider implementing request batching"
            ])
        
        # Error rate recommendations
        high_errors = any("High error rate" in b for b in bottlenecks.get("identified_bottlenecks", []))
        if high_errors:
            recommendations.extend([
                "Implement circuit breaker pattern for external dependencies",
                "Add retry logic with exponential backoff",
                "Improve error handling and graceful degradation",
                "Monitor and alert on error patterns"
            ])
        
        # Resource usage recommendations
        resource_constraints = bottlenecks.get("resource_constraints", {})
        if "cpu" in resource_constraints:
            recommendations.extend([
                "Profile CPU-intensive operations and optimize algorithms",
                "Consider caching expensive computations",
                "Implement background processing for non-critical tasks"
            ])
        
        if "memory" in resource_constraints:
            recommendations.extend([
                "Implement object pooling for frequently created objects",
                "Add streaming for large data processing",
                "Review memory leaks and optimize garbage collection"
            ])
        
        # General recommendations
        recommendations.extend([
            "Implement comprehensive monitoring and alerting",
            "Set up performance regression testing in CI/CD",
            "Consider implementing rate limiting to protect against overload",
            "Document performance SLAs and monitor compliance"
        ])
        
        return recommendations
    
    def save_performance_report(self, results: Dict[str, Any], filename: str = "performance_report.json"):
        """Save performance analysis report"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Performance report saved to {filename}")
        
        # Generate summary report
        self._generate_summary_report(results)
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report"""
        summary_path = "performance_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("ADK Security Agent - Performance Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Baseline performance
            baseline = results.get("baseline_performance", {})
            f.write("Baseline Performance:\n")
            for endpoint, metrics in baseline.items():
                if "error" not in metrics:
                    f.write(f"  {endpoint}:\n")
                    f.write(f"    Avg Response Time: {metrics.get('avg_response_time_ms', 0):.1f}ms\n")
                    f.write(f"    P95 Response Time: {metrics.get('p95_response_time_ms', 0):.1f}ms\n")
                    f.write(f"    Success Rate: {metrics.get('success_rate_percent', 0):.1f}%\n")
            f.write("\n")
            
            # Load test results
            f.write("Load Test Results:\n")
            for result in results.get("load_test_results", []):
                if "error" not in result:
                    f.write(f"  {result['scenario_name']}:\n")
                    f.write(f"    Throughput: {result.get('requests_per_second', 0):.1f} RPS\n")
                    f.write(f"    P95 Response Time: {result['response_times']['p95_ms']:.1f}ms\n")
                    f.write(f"    Error Rate: {result.get('error_rate_percent', 0):.1f}%\n")
                    f.write(f"    Peak CPU: {result['resource_usage']['peak_cpu_percent']:.1f}%\n")
            f.write("\n")
            
            # Bottlenecks
            bottlenecks = results.get("bottleneck_analysis", {}).get("identified_bottlenecks", [])
            if bottlenecks:
                f.write("Identified Bottlenecks:\n")
                for bottleneck in bottlenecks:
                    f.write(f"  - {bottleneck}\n")
                f.write("\n")
            
            # Recommendations
            recommendations = results.get("optimization_recommendations", [])
            if recommendations:
                f.write("Optimization Recommendations:\n")
                for i, rec in enumerate(recommendations[:10], 1):  # Top 10
                    f.write(f"  {i}. {rec}\n")
        
        logger.info(f"📋 Performance summary saved to {summary_path}")


async def main():
    """Run performance profiler"""
    profiler = PerformanceProfiler()
    
    try:
        results = await profiler.run_performance_analysis()
        
        # Save results
        profiler.save_performance_report(results)
        
        # Print summary
        print(f"\n⚡ Performance Analysis Complete!")
        
        # Show key metrics
        load_results = results.get("load_test_results", [])
        if load_results:
            for result in load_results:
                if "error" not in result:
                    print(f"{result['scenario_name']}: {result['requests_per_second']:.1f} RPS, "
                          f"{result['response_times']['p95_ms']:.1f}ms P95")
        
        # Show bottlenecks
        bottlenecks = results.get("bottleneck_analysis", {}).get("identified_bottlenecks", [])
        if bottlenecks:
            print(f"\n⚠️ Bottlenecks Identified: {len(bottlenecks)}")
            for bottleneck in bottlenecks[:3]:  # Show first 3
                print(f"  - {bottleneck}")
        else:
            print("\n✅ No significant bottlenecks identified")
            
    except Exception as e:
        logger.error(f"❌ Performance analysis failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())