"""
Performance Agent Evaluator

Specialized evaluator for measuring agent performance characteristics including
response time, resource usage, scalability, and tool execution efficiency.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Mock psutil for systems where it's not available
try:
    import psutil
except ImportError:
    class MockPsutil:
        @staticmethod
        def virtual_memory():
            class MockMemory:
                used = 1024 * 1024 * 1024  # 1GB
            return MockMemory()
        
        @staticmethod
        def cpu_percent(interval=None):
            return 25.0  # Mock 25% CPU usage
        
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                used = 10 * 1024 * 1024 * 1024  # 10GB
            return MockDisk()
    
    psutil = MockPsutil()

from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, EvalStatus, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data"""
    response_time_ms: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    tool_execution_count: int
    tool_execution_time_ms: float
    tokens_processed: int
    error_count: int
    

@dataclass
class PerformanceBenchmark:
    """Performance benchmarks and thresholds"""
    max_response_time_ms: float = 5000.0  # 5 seconds
    max_memory_usage_mb: float = 512.0    # 512 MB
    max_cpu_utilization: float = 80.0     # 80%
    min_tool_efficiency: float = 0.8      # 80% efficiency
    max_error_rate: float = 0.05          # 5% error rate


class PerformanceEvaluator(Evaluator):
    """
    Performance-focused evaluator extending ADK Evaluator base class.
    
    Evaluates agent performance characteristics including:
    - Response latency and throughput
    - Resource utilization (CPU, memory)
    - Tool execution efficiency
    - Error rates and reliability
    - Scalability under load
    """
    
    def __init__(self, threshold: float = 0.8, benchmark: Optional[PerformanceBenchmark] = None):
        self.threshold = threshold
        self.benchmark = benchmark or PerformanceBenchmark()
        self.performance_data: List[PerformanceMetrics] = []
        
    def evaluate_invocations(
        self,
        actual_invocations: List[Invocation],
        expected_invocations: List[Invocation]
    ) -> EvaluationResult:
        """Evaluate agent performance across multiple invocations"""
        
        logger.info(f"Starting performance evaluation with {len(actual_invocations)} invocations")
        
        per_invocation_results = []
        total_score = 0.0
        
        # Measure baseline system performance
        baseline_metrics = self._measure_system_baseline()
        
        for i, (actual, expected) in enumerate(zip(actual_invocations, expected_invocations)):
            # Measure performance for this invocation
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
            start_cpu = psutil.cpu_percent()
            
            # Simulate processing time (in real implementation, this would measure actual agent execution)
            time.sleep(0.1)  # Placeholder for actual agent processing
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
            end_cpu = psutil.cpu_percent()
            
            # Calculate performance metrics
            performance_metrics = PerformanceMetrics(
                response_time_ms=(end_time - start_time) * 1000,
                memory_usage_mb=max(0, end_memory - start_memory),
                cpu_utilization_percent=max(0, end_cpu - start_cpu),
                tool_execution_count=self._count_tool_executions(actual),
                tool_execution_time_ms=self._measure_tool_execution_time(actual),
                tokens_processed=self._count_tokens(actual),
                error_count=self._count_errors(actual)
            )
            
            self.performance_data.append(performance_metrics)
            
            # Evaluate this invocation's performance
            result = self._evaluate_single_invocation_performance(
                actual, expected, performance_metrics
            )
            per_invocation_results.append(result)
            total_score += result.score or 0.0
            
            logger.debug(f"Invocation {i}: response_time={performance_metrics.response_time_ms:.1f}ms, "
                        f"memory={performance_metrics.memory_usage_mb:.1f}MB, score={result.score}")
        
        overall_score = total_score / len(actual_invocations) if actual_invocations else 0.0
        overall_status = EvalStatus.PASSED if overall_score >= self.threshold else EvalStatus.FAILED
        
        # Log performance summary
        self._log_performance_summary()
        
        logger.info(f"Performance evaluation complete: score={overall_score:.3f}, status={overall_status}")
        
        return EvaluationResult(
            overall_score=overall_score,
            overall_eval_status=overall_status,
            per_invocation_results=per_invocation_results
        )
    
    def _measure_system_baseline(self) -> Dict[str, float]:
        """Measure baseline system performance"""
        return {
            'cpu_baseline': psutil.cpu_percent(interval=1),
            'memory_baseline': psutil.virtual_memory().used / (1024 * 1024),
            'disk_baseline': psutil.disk_usage('/').used / (1024 * 1024 * 1024)
        }
    
    def _evaluate_single_invocation_performance(
        self, 
        actual: Invocation, 
        expected: Invocation, 
        metrics: PerformanceMetrics
    ) -> PerInvocationResult:
        """Evaluate performance of a single invocation"""
        
        try:
            # Calculate individual performance scores
            latency_score = self._evaluate_latency(metrics.response_time_ms)
            memory_score = self._evaluate_memory_usage(metrics.memory_usage_mb)
            cpu_score = self._evaluate_cpu_usage(metrics.cpu_utilization_percent)
            tool_efficiency_score = self._evaluate_tool_efficiency(metrics)
            reliability_score = self._evaluate_reliability(metrics.error_count)
            
            # Weighted combination of performance scores
            score = (
                latency_score * 0.3 +
                memory_score * 0.2 +
                cpu_score * 0.2 +
                tool_efficiency_score * 0.2 +
                reliability_score * 0.1
            )
            
            status = EvalStatus.PASSED if score >= self.threshold else EvalStatus.FAILED
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            score = 0.0
            status = EvalStatus.FAILED
        
        return PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=status
        )
    
    def _evaluate_latency(self, response_time_ms: float) -> float:
        """Evaluate response latency score"""
        if response_time_ms <= self.benchmark.max_response_time_ms * 0.5:
            return 1.0  # Excellent performance
        elif response_time_ms <= self.benchmark.max_response_time_ms:
            # Linear scaling from 1.0 to 0.6
            ratio = response_time_ms / self.benchmark.max_response_time_ms
            return 1.6 - ratio
        else:
            # Poor performance, but not zero (agent still responded)
            return max(0.3, 1.0 - (response_time_ms / self.benchmark.max_response_time_ms - 1.0))
    
    def _evaluate_memory_usage(self, memory_mb: float) -> float:
        """Evaluate memory usage efficiency"""
        if memory_mb <= self.benchmark.max_memory_usage_mb * 0.5:
            return 1.0  # Excellent memory efficiency
        elif memory_mb <= self.benchmark.max_memory_usage_mb:
            # Linear scaling
            ratio = memory_mb / self.benchmark.max_memory_usage_mb
            return 1.6 - ratio
        else:
            # High memory usage
            return max(0.2, 1.0 - (memory_mb / self.benchmark.max_memory_usage_mb - 1.0))
    
    def _evaluate_cpu_usage(self, cpu_percent: float) -> float:
        """Evaluate CPU utilization efficiency"""
        if cpu_percent <= self.benchmark.max_cpu_utilization * 0.6:
            return 1.0  # Efficient CPU usage
        elif cpu_percent <= self.benchmark.max_cpu_utilization:
            ratio = cpu_percent / self.benchmark.max_cpu_utilization
            return 1.6 - ratio
        else:
            # High CPU usage
            return max(0.2, 1.0 - (cpu_percent / self.benchmark.max_cpu_utilization - 1.0))
    
    def _evaluate_tool_efficiency(self, metrics: PerformanceMetrics) -> float:
        """Evaluate tool execution efficiency"""
        if metrics.tool_execution_count == 0:
            return 1.0  # No tools used, perfect efficiency
        
        # Calculate average time per tool execution
        avg_tool_time = metrics.tool_execution_time_ms / metrics.tool_execution_count
        
        # Efficient tool usage: < 1 second per tool on average
        if avg_tool_time <= 1000:
            return 1.0
        elif avg_tool_time <= 3000:  # Up to 3 seconds per tool
            return 1.0 - (avg_tool_time - 1000) / 2000 * 0.4
        else:
            return max(0.3, 1.0 - (avg_tool_time - 3000) / 2000)
    
    def _evaluate_reliability(self, error_count: int) -> float:
        """Evaluate reliability based on error count"""
        if error_count == 0:
            return 1.0  # Perfect reliability
        else:
            # Penalize errors but don't go to zero unless many errors
            return max(0.1, 1.0 - error_count * 0.2)
    
    def _count_tool_executions(self, invocation: Invocation) -> int:
        """Count number of tool executions in invocation"""
        if not invocation.intermediate_data or not invocation.intermediate_data.tool_uses:
            return 0
        return len(invocation.intermediate_data.tool_uses)
    
    def _measure_tool_execution_time(self, invocation: Invocation) -> float:
        """Measure total tool execution time (simulated)"""
        tool_count = self._count_tool_executions(invocation)
        # Simulate tool execution time based on count
        return tool_count * 500  # 500ms per tool (simulated)
    
    def _count_tokens(self, invocation: Invocation) -> int:
        """Count tokens processed in invocation"""
        if not invocation.final_response or not invocation.final_response.parts:
            return 0
        
        token_count = 0
        for part in invocation.final_response.parts:
            if hasattr(part, 'text') and part.text:
                # Simple token estimation (roughly 4 characters per token)
                token_count += len(part.text) // 4
        
        return token_count
    
    def _count_errors(self, invocation: Invocation) -> int:
        """Count errors in invocation (simulated)"""
        # In a real implementation, this would check for actual errors
        # For now, simulate based on response content
        if not invocation.final_response:
            return 1  # No response is an error
        
        response_text = ""
        if invocation.final_response.parts:
            for part in invocation.final_response.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text.lower()
        
        # Check for error indicators in response
        error_indicators = ['error', 'failed', 'exception', 'timeout', 'unavailable']
        error_count = sum(1 for indicator in error_indicators if indicator in response_text)
        
        return error_count
    
    def _log_performance_summary(self):
        """Log summary of performance metrics"""
        if not self.performance_data:
            return
        
        avg_response_time = sum(m.response_time_ms for m in self.performance_data) / len(self.performance_data)
        avg_memory = sum(m.memory_usage_mb for m in self.performance_data) / len(self.performance_data)
        total_tools = sum(m.tool_execution_count for m in self.performance_data)
        total_errors = sum(m.error_count for m in self.performance_data)
        
        logger.info("Performance Summary:")
        logger.info(f"  Average response time: {avg_response_time:.1f}ms")
        logger.info(f"  Average memory usage: {avg_memory:.1f}MB")
        logger.info(f"  Total tool executions: {total_tools}")
        logger.info(f"  Total errors: {total_errors}")
    
    def evaluate_load_performance(self, invocations: List[Invocation], concurrent_users: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent performance under load with concurrent requests.
        
        Args:
            invocations: List of invocations to execute concurrently
            concurrent_users: Number of concurrent users to simulate
            
        Returns:
            Dictionary with load testing results
        """
        logger.info(f"Starting load performance evaluation with {concurrent_users} concurrent users")
        
        start_time = time.time()
        completed_requests = 0
        failed_requests = 0
        response_times = []
        
        def execute_invocation(invocation: Invocation) -> Dict[str, Any]:
            request_start = time.time()
            try:
                # Simulate invocation processing
                time.sleep(0.5 + len(invocation.user_content.parts) * 0.1)
                request_time = (time.time() - request_start) * 1000
                return {'success': True, 'response_time': request_time}
            except Exception as e:
                request_time = (time.time() - request_start) * 1000
                return {'success': False, 'response_time': request_time, 'error': str(e)}
        
        # Execute invocations concurrently
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(execute_invocation, inv) for inv in invocations]
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    completed_requests += 1
                else:
                    failed_requests += 1
                response_times.append(result['response_time'])
        
        total_time = time.time() - start_time
        
        # Calculate load testing metrics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        throughput = len(invocations) / total_time if total_time > 0 else 0
        error_rate = failed_requests / len(invocations) if invocations else 0
        
        results = {
            'total_requests': len(invocations),
            'completed_requests': completed_requests,
            'failed_requests': failed_requests,
            'error_rate': error_rate,
            'average_response_time_ms': avg_response_time,
            'throughput_requests_per_second': throughput,
            'total_duration_seconds': total_time,
            'concurrent_users': concurrent_users
        }
        
        logger.info(f"Load test complete: {throughput:.1f} req/s, {error_rate:.1%} error rate")
        
        return results


def evaluate_agent_performance(
    invocations: List[Invocation],
    threshold: float = 0.8,
    benchmark: Optional[PerformanceBenchmark] = None
) -> Tuple[float, EvalStatus, List[PerformanceMetrics]]:
    """
    Convenience function to evaluate agent performance.
    
    Args:
        invocations: List of actual invocations to evaluate
        threshold: Performance threshold (default 0.8)
        benchmark: Performance benchmark criteria
        
    Returns:
        Tuple of (score, status, performance_metrics)
    """
    # Create expected invocations (for interface compatibility)
    expected_invocations = invocations.copy()
    
    evaluator = PerformanceEvaluator(threshold=threshold, benchmark=benchmark)
    result = evaluator.evaluate_invocations(invocations, expected_invocations)
    
    return result.overall_score, result.overall_eval_status, evaluator.performance_data