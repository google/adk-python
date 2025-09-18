"""
Performance monitoring utilities for backend operations.
"""

import time
import logging
import sqlite3
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Container for query performance metrics."""
    query_id: str
    query_type: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    row_count: Optional[int] = None
    cache_hit: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""

    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.metrics_cache = []
        self.max_cache_size = 1000

    @contextmanager
    def monitor_query(self, query_type: str, query_id: Optional[str] = None):
        """
        Context manager to monitor query performance.

        Args:
            query_type: Type of query being executed
            query_id: Optional unique identifier for the query

        Yields:
            QueryMetrics: Metrics object that will be populated
        """
        if query_id is None:
            query_id = f"{query_type}_{int(time.time() * 1000)}"

        metrics = QueryMetrics(
            query_id=query_id,
            query_type=query_type,
            execution_time=0.0,
            success=False
        )

        start_time = time.time()

        try:
            yield metrics
            metrics.success = True

        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            if self.enable_logging:
                logger.error(f"Query {query_id} failed: {e}")
            raise

        finally:
            metrics.execution_time = time.time() - start_time

            if self.enable_logging:
                status = "✅" if metrics.success else "❌"
                logger.info(
                    f"{status} Query {query_id} ({query_type}) - "
                    f"{metrics.execution_time:.3f}s"
                )

            # Add to cache
            self._add_to_cache(metrics)

    def _add_to_cache(self, metrics: QueryMetrics):
        """Add metrics to internal cache."""
        self.metrics_cache.append(metrics)

        # Trim cache if too large
        if len(self.metrics_cache) > self.max_cache_size:
            self.metrics_cache = self.metrics_cache[-self.max_cache_size:]

    def get_recent_metrics(self, limit: int = 50) -> list[QueryMetrics]:
        """Get recent query metrics."""
        return self.metrics_cache[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_cache:
            return {"message": "No metrics available"}

        total_queries = len(self.metrics_cache)
        successful_queries = sum(1 for m in self.metrics_cache if m.success)
        failed_queries = total_queries - successful_queries

        execution_times = [m.execution_time for m in self.metrics_cache if m.success]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0

        # Group by query type
        query_types = {}
        for metric in self.metrics_cache:
            if metric.query_type not in query_types:
                query_types[metric.query_type] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0.0
                }

            query_types[metric.query_type]["count"] += 1
            if metric.success:
                query_types[metric.query_type]["success_count"] += 1
                query_types[metric.query_type]["total_time"] += metric.execution_time

        # Calculate averages per query type
        for qt in query_types.values():
            if qt["success_count"] > 0:
                qt["avg_time"] = qt["total_time"] / qt["success_count"]
            else:
                qt["avg_time"] = 0

        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
            "performance": {
                "avg_execution_time": avg_time,
                "max_execution_time": max_time,
                "min_execution_time": min_time
            },
            "query_types": query_types,
            "cache_size": len(self.metrics_cache)
        }

    def log_agent_performance(self, session_id: str, user_id: str,
                             query_text: str, metadata: Dict[str, Any]):
        """
        Log agent performance metrics.

        Args:
            session_id: Session identifier
            user_id: User identifier
            query_text: The query that was processed
            metadata: Response metadata from ADK agent
        """
        try:
            duration = metadata.get("total_duration", 0)
            success = metadata.get("event_count", 0) > 0
            tool_calls = len(metadata.get("tool_calls", []))

            if self.enable_logging:
                logger.info(
                    f"🤖 Agent Performance - "
                    f"Session: {session_id[:8]}... - "
                    f"Duration: {duration:.3f}s - "
                    f"Events: {metadata.get('event_count', 0)} - "
                    f"Tools: {tool_calls} - "
                    f"Success: {success}"
                )

            # Store in metrics cache as special agent query
            agent_metrics = QueryMetrics(
                query_id=f"agent_{session_id}_{int(time.time() * 1000)}",
                query_type="agent_query",
                execution_time=duration,
                success=success,
                row_count=metadata.get("response_length", 0)
            )

            self._add_to_cache(agent_metrics)

        except Exception as e:
            logger.error(f"Error logging agent performance: {e}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

@contextmanager
def monitor_database_query(query_type: str, query_id: Optional[str] = None):
    """
    Convenience function for monitoring database queries.

    Args:
        query_type: Type of database query
        query_id: Optional query identifier

    Yields:
        QueryMetrics: Metrics object
    """
    with performance_monitor.monitor_query(query_type, query_id) as metrics:
        yield metrics

def log_query_performance(func: Callable) -> Callable:
    """
    Decorator to automatically log performance of query functions.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with performance logging
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        query_id = f"{func_name}_{int(time.time() * 1000)}"

        with performance_monitor.monitor_query(func_name, query_id) as metrics:
            result = func(*args, **kwargs)

            # Try to extract row count from result
            if isinstance(result, dict):
                if "data" in result and isinstance(result["data"], list):
                    metrics.row_count = len(result["data"])
                elif "row_count" in result:
                    metrics.row_count = result["row_count"]

            return result

    return wrapper

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics."""
    return performance_monitor.get_performance_summary()

def log_agent_metrics(session_id: str, user_id: str, query_text: str,
                     metadata: Dict[str, Any]):
    """Log agent performance metrics."""
    performance_monitor.log_agent_performance(session_id, user_id, query_text, metadata)