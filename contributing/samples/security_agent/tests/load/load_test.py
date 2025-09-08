#!/usr/bin/env python3
"""
Load Testing Suite for Security Agent
=====================================

Simulates realistic user loads to test system scalability and breaking points:
- Gradual load ramp-up testing
- Sustained load testing  
- Spike load testing
- Breaking point detection
- Recovery testing

Load Scenarios:
- 100 concurrent users (typical enterprise usage)
- 500 concurrent users (peak usage)
- 1000 concurrent users (stress test)
- Variable load patterns (realistic usage)

Author: Performance Testing Engineer
Date: 2024-09-08
"""

import asyncio
import aiohttp
import time
import json
import statistics
import sys
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import threading
import queue
import random
import psutil
import argparse

class LoadTestResult:
    """Container for load test results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.error_details = []
        self.throughput_samples = []
        self.memory_samples = []
        self.cpu_samples = []
        
    def add_response(self, response_time: float, success: bool, error: str = None):
        """Add a response result."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.error_details.append(error)
        
        self.response_times.append(response_time)
    
    def add_system_metrics(self, memory_mb: float, cpu_percent: float):
        """Add system metrics sample."""
        self.memory_samples.append(memory_mb)
        self.cpu_samples.append(cpu_percent)
    
    def complete(self):
        """Mark test as complete."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        duration = (self.end_time or time.time()) - self.start_time
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        throughput = self.successful_requests / duration if duration > 0 else 0
        
        summary = {
            "test_name": self.test_name,
            "duration": duration,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "throughput": throughput
        }
        
        if self.response_times:
            import numpy as np
            summary["response_times"] = {
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p90": np.percentile(self.response_times, 90),
                "p95": np.percentile(self.response_times, 95),
                "p99": np.percentile(self.response_times, 99),
                "min": min(self.response_times),
                "max": max(self.response_times)
            }
        
        if self.memory_samples:
            summary["memory_usage"] = {
                "mean_mb": statistics.mean(self.memory_samples),
                "peak_mb": max(self.memory_samples),
                "min_mb": min(self.memory_samples)
            }
        
        if self.cpu_samples:
            summary["cpu_usage"] = {
                "mean_percent": statistics.mean(self.cpu_samples),
                "peak_percent": max(self.cpu_samples)
            }
        
        if self.error_details:
            summary["top_errors"] = list(set(self.error_details[:10]))  # Top 10 unique errors
        
        return summary

class LoadTestClient:
    """Async HTTP client for load testing."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=100)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Tuple[bool, float, str]:
        """Make HTTP request and return (success, response_time, error)."""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.request(method, url, **kwargs) as response:
                await response.text()  # Consume response body
                response_time = time.time() - start_time
                success = 200 <= response.status < 400
                error = None if success else f"HTTP {response.status}"
                return success, response_time, error
                
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return False, response_time, "Timeout"
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)

class UserScenario:
    """Defines a user behavior scenario."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.actions = []
    
    def add_action(self, method: str, endpoint: str, delay: float = 0, **kwargs):
        """Add an action to the scenario."""
        self.actions.append({
            "method": method,
            "endpoint": endpoint,
            "delay": delay,
            "kwargs": kwargs
        })
    
    async def execute(self, client: LoadTestClient, user_id: int) -> List[Tuple[bool, float, str]]:
        """Execute the scenario actions."""
        results = []
        
        for action in self.actions:
            if action["delay"] > 0:
                await asyncio.sleep(action["delay"])
            
            # Customize request data for this user
            kwargs = action["kwargs"].copy()
            if "json" in kwargs and isinstance(kwargs["json"], dict):
                kwargs["json"] = self._customize_json(kwargs["json"], user_id)
            
            result = await client.make_request(
                action["method"],
                action["endpoint"],
                **kwargs
            )
            results.append(result)
        
        return results
    
    def _customize_json(self, data: dict, user_id: int) -> dict:
        """Customize JSON data for specific user."""
        customized = data.copy()
        
        # Replace placeholders with user-specific values
        for key, value in customized.items():
            if isinstance(value, str):
                customized[key] = value.replace("{user_id}", str(user_id))
        
        return customized

class LoadTestRunner:
    """Main load test runner."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.scenarios = self._create_scenarios()
        self.results = []
        self.system_monitor_active = False
        self.system_metrics_queue = queue.Queue()
    
    def _create_scenarios(self) -> List[UserScenario]:
        """Create realistic user scenarios."""
        
        # Quick health check user
        health_check_user = UserScenario("health_check", weight=0.3)
        health_check_user.add_action("GET", "/health")
        health_check_user.add_action("GET", "/metrics", delay=1)
        
        # Interactive chat user
        chat_user = UserScenario("chat_user", weight=0.4)
        chat_user.add_action("POST", "/api/v1/chat/message", 
                            json={
                                "query": "What resources do I have?",
                                "session_id": "load-test-{user_id}",
                                "user_id": "user-{user_id}"
                            })
        chat_user.add_action("POST", "/api/v1/chat/message",
                            delay=2,
                            json={
                                "query": "Check my security posture",
                                "session_id": "load-test-{user_id}",
                                "user_id": "user-{user_id}"
                            })
        
        # Status monitoring user
        status_user = UserScenario("status_user", weight=0.2)
        status_user.add_action("GET", "/status")
        status_user.add_action("GET", "/health", delay=0.5)
        status_user.add_action("GET", "/status", delay=1)
        
        # Heavy user (multiple operations)
        heavy_user = UserScenario("heavy_user", weight=0.1)
        heavy_user.add_action("GET", "/health")
        heavy_user.add_action("GET", "/status", delay=0.5)
        heavy_user.add_action("GET", "/metrics", delay=0.5)
        heavy_user.add_action("POST", "/api/v1/chat/message",
                             delay=1,
                             json={
                                 "query": "Comprehensive security analysis",
                                 "session_id": "heavy-{user_id}",
                                 "user_id": "heavy-user-{user_id}"
                             })
        
        return [health_check_user, chat_user, status_user, heavy_user]
    
    def _select_scenario(self) -> UserScenario:
        """Select scenario based on weights."""
        total_weight = sum(scenario.weight for scenario in self.scenarios)
        r = random.random() * total_weight
        
        current_weight = 0
        for scenario in self.scenarios:
            current_weight += scenario.weight
            if r <= current_weight:
                return scenario
        
        return self.scenarios[0]  # Fallback
    
    def _start_system_monitoring(self):
        """Start background system monitoring."""
        self.system_monitor_active = True
        
        def monitor():
            while self.system_monitor_active:
                try:
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_metrics_queue.put((memory_mb, cpu_percent))
                except:
                    pass
                time.sleep(2)  # Sample every 2 seconds
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _stop_system_monitoring(self):
        """Stop system monitoring."""
        self.system_monitor_active = False
    
    async def run_gradual_load_test(self, max_users: int = 100, ramp_duration: int = 120):
        """Run gradual load ramp-up test."""
        print(f"\n🔄 Running Gradual Load Test (0 → {max_users} users over {ramp_duration}s)")
        
        result = LoadTestResult("Gradual Load Test")
        self._start_system_monitoring()
        
        try:
            async with LoadTestClient(self.base_url) as client:
                tasks = []
                
                # Gradually add users
                users_per_interval = max_users / (ramp_duration / 10)  # Add users every 10 seconds
                
                for interval in range(0, ramp_duration, 10):
                    current_users = min(int(interval * users_per_interval / 10), max_users)
                    
                    # Add new users
                    while len(tasks) < current_users:
                        user_id = len(tasks)
                        scenario = self._select_scenario()
                        
                        async def user_simulation(uid, scen):
                            try:
                                results = await scen.execute(client, uid)
                                return results
                            except Exception as e:
                                return [(False, 0, str(e))]
                        
                        task = asyncio.create_task(user_simulation(user_id, scenario))
                        tasks.append(task)
                    
                    print(f"  Active users: {current_users}/{max_users}")
                    
                    # Collect system metrics
                    try:
                        memory_mb, cpu_percent = self.system_metrics_queue.get_nowait()
                        result.add_system_metrics(memory_mb, cpu_percent)
                    except queue.Empty:
                        pass
                    
                    await asyncio.sleep(10)
                
                # Wait for all user sessions to complete
                print("  Waiting for user sessions to complete...")
                completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for task_result in completed_tasks:
                    if isinstance(task_result, list):
                        for success, response_time, error in task_result:
                            result.add_response(response_time, success, error)
                    elif isinstance(task_result, Exception):
                        result.add_response(0, False, str(task_result))
        
        finally:
            self._stop_system_monitoring()
            result.complete()
        
        # Collect remaining system metrics
        while not self.system_metrics_queue.empty():
            try:
                memory_mb, cpu_percent = self.system_metrics_queue.get_nowait()
                result.add_system_metrics(memory_mb, cpu_percent)
            except queue.Empty:
                break
        
        self.results.append(result)
        self._print_test_summary(result)
        return result
    
    async def run_sustained_load_test(self, concurrent_users: int = 200, duration: int = 300):
        """Run sustained load test."""
        print(f"\n⏱️  Running Sustained Load Test ({concurrent_users} users for {duration}s)")
        
        result = LoadTestResult("Sustained Load Test")
        self._start_system_monitoring()
        
        try:
            async with LoadTestClient(self.base_url) as client:
                end_time = time.time() + duration
                tasks = []
                
                # Start initial users
                for user_id in range(concurrent_users):
                    scenario = self._select_scenario()
                    
                    async def sustained_user(uid, scen):
                        user_results = []
                        while time.time() < end_time:
                            try:
                                session_results = await scen.execute(client, uid)
                                user_results.extend(session_results)
                                
                                # Wait before next session
                                await asyncio.sleep(random.uniform(5, 15))
                            except Exception as e:
                                user_results.append((False, 0, str(e)))
                                await asyncio.sleep(5)
                        
                        return user_results
                    
                    task = asyncio.create_task(sustained_user(user_id, scenario))
                    tasks.append(task)
                
                # Monitor progress
                start_time = time.time()
                while time.time() < end_time:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    print(f"  Progress: {elapsed:.0f}s/{duration}s ({remaining:.0f}s remaining)")
                    
                    # Collect system metrics
                    try:
                        memory_mb, cpu_percent = self.system_metrics_queue.get_nowait()
                        result.add_system_metrics(memory_mb, cpu_percent)
                    except queue.Empty:
                        pass
                    
                    await asyncio.sleep(30)  # Status update every 30s
                
                # Collect results
                print("  Collecting results...")
                completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
                
                for task_result in completed_tasks:
                    if isinstance(task_result, list):
                        for success, response_time, error in task_result:
                            result.add_response(response_time, success, error)
                    elif isinstance(task_result, Exception):
                        result.add_response(0, False, str(task_result))
        
        finally:
            self._stop_system_monitoring()
            result.complete()
        
        # Collect remaining system metrics
        while not self.system_metrics_queue.empty():
            try:
                memory_mb, cpu_percent = self.system_metrics_queue.get_nowait()
                result.add_system_metrics(memory_mb, cpu_percent)
            except queue.Empty:
                break
        
        self.results.append(result)
        self._print_test_summary(result)
        return result
    
    async def run_spike_load_test(self, spike_users: int = 500, spike_duration: int = 60):
        """Run spike load test."""
        print(f"\n⚡ Running Spike Load Test ({spike_users} users for {spike_duration}s)")
        
        result = LoadTestResult("Spike Load Test")
        self._start_system_monitoring()
        
        try:
            async with LoadTestClient(self.base_url, timeout=45) as client:  # Longer timeout for spike
                
                # Create all spike users simultaneously
                async def spike_user(user_id):
                    scenario = self._select_scenario()
                    try:
                        return await scenario.execute(client, user_id)
                    except Exception as e:
                        return [(False, 0, str(e))]
                
                print(f"  Launching {spike_users} concurrent users...")
                tasks = [asyncio.create_task(spike_user(i)) for i in range(spike_users)]
                
                # Monitor spike
                start_time = time.time()
                monitoring_task = asyncio.create_task(self._monitor_spike(result, spike_duration))
                
                # Wait for spike completion
                completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
                monitoring_task.cancel()
                
                # Process results
                for task_result in completed_tasks:
                    if isinstance(task_result, list):
                        for success, response_time, error in task_result:
                            result.add_response(response_time, success, error)
                    elif isinstance(task_result, Exception):
                        result.add_response(0, False, str(task_result))
        
        finally:
            self._stop_system_monitoring()
            result.complete()
        
        # Collect remaining system metrics
        while not self.system_metrics_queue.empty():
            try:
                memory_mb, cpu_percent = self.system_metrics_queue.get_nowait()
                result.add_system_metrics(memory_mb, cpu_percent)
            except queue.Empty:
                break
        
        self.results.append(result)
        self._print_test_summary(result)
        return result
    
    async def _monitor_spike(self, result: LoadTestResult, duration: int):
        """Monitor system during spike test."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                memory_mb, cpu_percent = self.system_metrics_queue.get_nowait()
                result.add_system_metrics(memory_mb, cpu_percent)
                
                elapsed = time.time() - (end_time - duration)
                print(f"  Spike progress: {elapsed:.0f}s/{duration}s (Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%)")
            except queue.Empty:
                pass
            
            await asyncio.sleep(5)
    
    def _print_test_summary(self, result: LoadTestResult):
        """Print test summary."""
        summary = result.get_summary()
        
        print(f"\n📊 {result.test_name} Results:")
        print(f"  Duration: {summary['duration']:.1f}s")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Throughput: {summary['throughput']:.2f} req/s")
        
        if "response_times" in summary:
            rt = summary["response_times"]
            print(f"  Response Times:")
            print(f"    Mean: {rt['mean']:.3f}s")
            print(f"    P95: {rt['p95']:.3f}s")
            print(f"    P99: {rt['p99']:.3f}s")
            print(f"    Max: {rt['max']:.3f}s")
        
        if "memory_usage" in summary:
            mem = summary["memory_usage"]
            print(f"  Memory Usage:")
            print(f"    Peak: {mem['peak_mb']:.1f}MB")
            print(f"    Average: {mem['mean_mb']:.1f}MB")
        
        if "cpu_usage" in summary:
            cpu = summary["cpu_usage"]
            print(f"  CPU Usage:")
            print(f"    Peak: {cpu['peak_percent']:.1f}%")
            print(f"    Average: {cpu['mean_percent']:.1f}%")
        
        if summary.get("top_errors"):
            print(f"  Top Errors: {summary['top_errors'][:3]}")
    
    async def run_breaking_point_test(self):
        """Find system breaking point."""
        print(f"\n🔍 Running Breaking Point Test")
        
        breaking_point = None
        test_levels = [50, 100, 200, 400, 800, 1200, 1600, 2000]
        
        for level in test_levels:
            print(f"\n  Testing {level} concurrent users...")
            
            try:
                # Run short spike test at this level
                result = await self.run_spike_load_test(level, 30)  # 30 second spikes
                summary = result.get_summary()
                
                # Check if system is breaking down
                success_rate = summary["success_rate"]
                avg_response_time = summary.get("response_times", {}).get("mean", 0)
                
                if success_rate < 50 or avg_response_time > 30:  # System breaking down
                    breaking_point = level
                    print(f"  ⚠️  Breaking point detected at {level} users")
                    print(f"     Success rate: {success_rate:.1f}%")
                    print(f"     Avg response time: {avg_response_time:.3f}s")
                    break
                else:
                    print(f"  ✅ {level} users handled successfully")
                    print(f"     Success rate: {success_rate:.1f}%")
                    print(f"     Avg response time: {avg_response_time:.3f}s")
                
                # Brief recovery period
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"  ❌ Test failed at {level} users: {e}")
                breaking_point = level
                break
        
        if breaking_point:
            print(f"\n🚨 System breaking point: {breaking_point} concurrent users")
        else:
            print(f"\n💪 System handled all test levels successfully (up to {test_levels[-1]} users)")
        
        return breaking_point
    
    async def run_all_load_tests(self):
        """Run complete load test suite."""
        print("🚀 Starting Comprehensive Load Testing Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Test server availability first
            async with LoadTestClient(self.base_url) as client:
                success, response_time, error = await client.make_request("GET", "/health")
                if not success:
                    print(f"❌ Server not available: {error}")
                    return
                print(f"✅ Server available (response time: {response_time:.3f}s)")
            
            # Run load tests
            await self.run_gradual_load_test(100, 60)  # Reduced for demo
            await asyncio.sleep(10)  # Recovery time
            
            await self.run_sustained_load_test(50, 120)  # Reduced for demo
            await asyncio.sleep(10)  # Recovery time
            
            await self.run_spike_load_test(200, 30)  # Reduced for demo
            await asyncio.sleep(10)  # Recovery time
            
            # Skip breaking point test in demo mode
            # breaking_point = await self.run_breaking_point_test()
            
        except KeyboardInterrupt:
            print("\n⚠️  Load testing interrupted by user")
        except Exception as e:
            print(f"\n❌ Load testing failed: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n" + "=" * 60)
        print(f"Load Testing Complete (Duration: {total_time:.1f}s)")
        print("=" * 60)
        
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate final load testing report."""
        if not self.results:
            print("No test results to report")
            return
        
        print("\n📈 Final Load Testing Report")
        print("-" * 40)
        
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        print(f"Overall Statistics:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Successful Requests: {total_successful}")
        print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
        
        print(f"\nTest Results Summary:")
        for result in self.results:
            summary = result.get_summary()
            print(f"  {result.test_name}:")
            print(f"    Success Rate: {summary['success_rate']:.1f}%")
            print(f"    Throughput: {summary['throughput']:.2f} req/s")
            if "response_times" in summary:
                print(f"    Avg Response Time: {summary['response_times']['mean']:.3f}s")
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "overall_success_rate": overall_success_rate
            },
            "test_results": [result.get_summary() for result in self.results]
        }
        
        report_path = os.path.join(os.path.dirname(__file__), "load_test_report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_path}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load Testing Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument("--users", type=int, default=100, help="Max concurrent users for gradual test")
    parser.add_argument("--duration", type=int, default=300, help="Sustained test duration (seconds)")
    parser.add_argument("--spike", type=int, default=500, help="Spike test user count")
    parser.add_argument("--test", choices=["gradual", "sustained", "spike", "breaking", "all"], 
                       default="all", help="Test type to run")
    
    args = parser.parse_args()
    
    runner = LoadTestRunner(args.url)
    
    if args.test == "gradual":
        await runner.run_gradual_load_test(args.users, 120)
    elif args.test == "sustained":
        await runner.run_sustained_load_test(args.users // 2, args.duration)
    elif args.test == "spike":
        await runner.run_spike_load_test(args.spike, 60)
    elif args.test == "breaking":
        await runner.run_breaking_point_test()
    else:
        await runner.run_all_load_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Load testing interrupted")
    except Exception as e:
        print(f"\n❌ Load testing failed: {e}")
        sys.exit(1)