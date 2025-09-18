"""
Request and database query monitoring utilities.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from functools import wraps
import sqlite3
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryMonitor:
    """Monitor and log database queries."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize query monitor with optional database path for logging."""
        self.db_path = db_path
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "slowest_query": 0.0,
            "fastest_query": float('inf')
        }

    def log_query(
        self,
        session_id: str,
        user_id: str,
        query_text: str,
        query_type: str,
        execution_time: float,
        success: bool,
        error_message: Optional[str] = None
    ):
        """
        Log a database query execution.

        Args:
            session_id: Session identifier
            user_id: User identifier
            query_text: The query executed
            query_type: Type of query (e.g., "security_findings")
            execution_time: Time taken to execute in seconds
            success: Whether query succeeded
            error_message: Error message if failed
        """
        # Update statistics
        self.query_stats["total_queries"] += 1
        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1

        self.query_stats["total_time"] += execution_time
        self.query_stats["average_time"] = (
            self.query_stats["total_time"] / self.query_stats["total_queries"]
        )

        if execution_time > self.query_stats["slowest_query"]:
            self.query_stats["slowest_query"] = execution_time

        if execution_time < self.query_stats["fastest_query"]:
            self.query_stats["fastest_query"] = execution_time

        # Log to standard logger
        log_level = logging.INFO if success else logging.ERROR
        logger.log(
            log_level,
            f"Query [{query_type}] - Session: {session_id}, "
            f"User: {user_id}, Time: {execution_time:.3f}s, "
            f"Success: {success}"
        )

        if error_message:
            logger.error(f"Query error: {error_message}")

        # Log to database if path provided
        if self.db_path:
            try:
                self._log_to_database(
                    session_id,
                    user_id,
                    query_text,
                    query_type,
                    execution_time,
                    success,
                    error_message
                )
            except Exception as e:
                logger.warning(f"Failed to log query to database: {e}")

    def _log_to_database(
        self,
        session_id: str,
        user_id: str,
        query_text: str,
        query_type: str,
        execution_time: float,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Log query to database for persistence."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs
                (session_id, user_id, query_text, query_type, execution_time, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                query_text,
                query_type,
                execution_time,
                success,
                error_message
            ))
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get current query statistics."""
        return self.query_stats.copy()

    def reset_stats(self):
        """Reset query statistics."""
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "slowest_query": 0.0,
            "fastest_query": float('inf')
        }


class RequestMonitor:
    """Monitor HTTP requests and responses."""

    def __init__(self):
        """Initialize request monitor."""
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "endpoints": {},
            "response_times": []
        }

    def log_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Log an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            status_code: HTTP response status code
            response_time: Time taken for request in seconds
            user_id: Optional user identifier
            session_id: Optional session identifier
            error: Optional error message
        """
        # Update statistics
        self.request_stats["total_requests"] += 1

        if 200 <= status_code < 300:
            self.request_stats["successful_requests"] += 1
        else:
            self.request_stats["failed_requests"] += 1

        # Track endpoint-specific stats
        endpoint_key = f"{method} {endpoint}"
        if endpoint_key not in self.request_stats["endpoints"]:
            self.request_stats["endpoints"][endpoint_key] = {
                "count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "errors": 0
            }

        endpoint_stats = self.request_stats["endpoints"][endpoint_key]
        endpoint_stats["count"] += 1
        endpoint_stats["total_time"] += response_time
        endpoint_stats["average_time"] = (
            endpoint_stats["total_time"] / endpoint_stats["count"]
        )

        if status_code >= 400:
            endpoint_stats["errors"] += 1

        # Keep last 100 response times for percentile calculations
        self.request_stats["response_times"].append(response_time)
        if len(self.request_stats["response_times"]) > 100:
            self.request_stats["response_times"].pop(0)

        # Log to standard logger
        log_level = logging.INFO if status_code < 400 else logging.WARNING
        logger.log(
            log_level,
            f"{method} {endpoint} - Status: {status_code}, "
            f"Time: {response_time:.3f}s, Session: {session_id}, User: {user_id}"
        )

        if error:
            logger.error(f"Request error: {error}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current request statistics."""
        stats = self.request_stats.copy()

        # Calculate percentiles if we have data
        if stats["response_times"]:
            sorted_times = sorted(stats["response_times"])
            n = len(sorted_times)
            stats["p50_response_time"] = sorted_times[n // 2]
            stats["p95_response_time"] = sorted_times[int(n * 0.95)]
            stats["p99_response_time"] = sorted_times[int(n * 0.99)]

        return stats


def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor function performance.

    Usage:
        @monitor_performance
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        error = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            execution_time = time.time() - start_time
            log_level = logging.INFO if error is None else logging.ERROR
            logger.log(
                log_level,
                f"Function {func.__name__} took {execution_time:.3f}s"
            )
            if error:
                logger.error(f"Function {func.__name__} failed: {error}")

    return wrapper


def monitor_async_performance(func: Callable) -> Callable:
    """
    Decorator to monitor async function performance.

    Usage:
        @monitor_async_performance
        async def my_async_function():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        error = None

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            execution_time = time.time() - start_time
            log_level = logging.INFO if error is None else logging.ERROR
            logger.log(
                log_level,
                f"Async function {func.__name__} took {execution_time:.3f}s"
            )
            if error:
                logger.error(f"Async function {func.__name__} failed: {error}")

    return wrapper


# Global instances
query_monitor = QueryMonitor()
request_monitor = RequestMonitor()


def setup_monitoring(db_path: Optional[str] = None):
    """
    Setup monitoring with optional database logging.

    Args:
        db_path: Optional path to database for persistent logging
    """
    global query_monitor
    if db_path:
        query_monitor = QueryMonitor(db_path)
        logger.info(f"Monitoring setup with database logging: {db_path}")
    else:
        logger.info("Monitoring setup without database logging")


def get_monitoring_summary() -> Dict[str, Any]:
    """
    Get a summary of all monitoring data.

    Returns:
        dict: Combined monitoring statistics
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "query_stats": query_monitor.get_stats(),
        "request_stats": request_monitor.get_stats()
    }


if __name__ == "__main__":
    # Test monitoring
    print("Testing monitoring utilities...")

    # Setup with database logging
    setup_monitoring("backend/cache/gcp_data.db")

    # Test query monitoring
    query_monitor.log_query(
        session_id="test-session",
        user_id="test-user",
        query_text="SELECT * FROM security_findings",
        query_type="security_findings",
        execution_time=0.123,
        success=True
    )

    # Test request monitoring
    request_monitor.log_request(
        method="POST",
        endpoint="/api/v1/chat/message",
        status_code=200,
        response_time=1.234,
        user_id="test-user",
        session_id="test-session"
    )

    # Get summary
    summary = get_monitoring_summary()
    print(f"Monitoring summary: {summary}")