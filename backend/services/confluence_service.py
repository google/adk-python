"""
Confluence API Service Layer

This service provides a robust wrapper around the Confluence API client with:
- Rate limiting (100 requests/min)
- Exponential backoff for retries
- Circuit breaker pattern for API failures
- Comprehensive error handling with fallback strategies
- Connection pooling and session management

Implements tasks T024-T026 from the ADK development plan.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import random
import threading
from contextlib import asynccontextmanager

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from atlassian import Confluence


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    max_requests: int = 100  # 100 requests per minute as required
    time_window: int = 60    # 60 seconds
    burst_limit: int = 10    # Allow burst of 10 requests


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Number of failures to open circuit
    recovery_timeout: int = 60      # Seconds to wait before trying recovery
    success_threshold: int = 3      # Successful calls to close circuit
    timeout: float = 30.0          # Request timeout in seconds


@dataclass
class RetryConfig:
    """Exponential backoff retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RateLimiter:
    """
    Token bucket rate limiter implementation.

    Supports 100 requests/min with burst capability as required by T025.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.max_requests
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            now = time.time()

            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            if elapsed > 0:
                refill_tokens = (elapsed / self.config.time_window) * self.config.max_requests
                self.tokens = min(self.config.max_requests, self.tokens + refill_tokens)
                self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
        """
        while not self.acquire(tokens):
            # Calculate wait time until next token is available
            wait_time = self.config.time_window / self.config.max_requests
            await asyncio.sleep(min(wait_time, 1.0))  # Cap at 1 second


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Implements the circuit breaker pattern as required by T025.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit state."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if we should try to recover
                if (self.last_failure_time and
                    time.time() - self.last_failure_time >= self.config.recovery_timeout):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failure count on success

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker back to OPEN state after failure during recovery")


class RetryHandler:
    """
    Exponential backoff retry handler.

    Implements exponential backoff with jitter as required by T025.
    """

    def __init__(self, config: RetryConfig):
        self.config = config

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )

        if self.config.jitter:
            # Add random jitter to prevent thundering herd
            jitter = delay * 0.1 * random.random()
            delay += jitter

        return delay

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries failed
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == self.config.max_retries:
                    logger.error(f"All {self.config.max_retries} retry attempts failed")
                    break

                delay = self.calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)

        raise last_exception


class ConnectionPool:
    """
    HTTP connection pool manager for efficient connection reuse.

    Implements connection pooling as required by T024.
    """

    def __init__(self, pool_size: int = 10, max_retries: int = 3):
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=retry_strategy
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set reasonable timeouts
        self.session.timeout = (10, 30)  # (connect, read) timeout

    def close(self):
        """Close the session and clean up connections."""
        self.session.close()


class ConfluenceServiceError(Exception):
    """Base exception for Confluence service errors."""
    pass


class ConfluenceRateLimitError(ConfluenceServiceError):
    """Raised when rate limit is exceeded."""
    pass


class ConfluenceCircuitOpenError(ConfluenceServiceError):
    """Raised when circuit breaker is open."""
    pass


class ConfluenceService:
    """
    Robust Confluence API service with enterprise-grade reliability features.

    This service wraps the Confluence API client and provides:
    - Rate limiting (100 requests/min)
    - Exponential backoff retries
    - Circuit breaker pattern
    - Connection pooling
    - Comprehensive error handling
    - Graceful degradation

    Implements T024-T026 requirements.
    """

    def __init__(
        self,
        confluence_url: str,
        username: str,
        api_token: str,
        rate_limit_config: Optional[RateLimitConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        pool_size: int = 10
    ):
        """
        Initialize Confluence service.

        Args:
            confluence_url: Confluence instance URL
            username: Username for authentication
            api_token: API token for authentication
            rate_limit_config: Rate limiting configuration
            circuit_breaker_config: Circuit breaker configuration
            retry_config: Retry configuration
            pool_size: Connection pool size
        """
        self.confluence_url = confluence_url
        self.username = username
        self.api_token = api_token

        # Initialize components
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig())
        self.retry_handler = RetryHandler(retry_config or RetryConfig())
        self.connection_pool = ConnectionPool(pool_size)

        # Initialize Confluence client
        self.confluence = None
        self._client_lock = threading.Lock()

        # Health check and fallback data
        self._last_health_check = None
        self._health_check_interval = 300  # 5 minutes
        self._fallback_cache = {}

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'circuit_breaker_open': 0,
            'cache_hits': 0
        }

        logger.info("ConfluenceService initialized with enterprise reliability features")

    async def _get_client(self) -> Confluence:
        """Get or create Confluence client with health checking."""
        if self.confluence is None:
            with self._client_lock:
                if self.confluence is None:
                    try:
                        self.confluence = Confluence(
                            url=self.confluence_url,
                            username=self.username,
                            password=self.api_token,
                            cloud=True,
                            session=self.connection_pool.session
                        )
                        logger.info("Confluence client initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize Confluence client: {e}")
                        raise ConfluenceServiceError(f"Client initialization failed: {e}")

        # Periodic health check
        await self._perform_health_check()

        return self.confluence

    async def _perform_health_check(self) -> bool:
        """
        Perform health check on Confluence service.

        Returns:
            True if service is healthy, False otherwise
        """
        now = time.time()

        # Skip if recent health check was successful
        if (self._last_health_check and
            now - self._last_health_check < self._health_check_interval):
            return True

        try:
            # Simple health check - get user info
            if self.confluence:
                self.confluence.get_current_user()
                self._last_health_check = now
                logger.debug("Confluence health check passed")
                return True
        except Exception as e:
            logger.warning(f"Confluence health check failed: {e}")
            return False

        return False

    async def _execute_with_protection(
        self,
        operation_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with full protection (rate limiting, circuit breaker, retries).

        Args:
            operation_name: Name of the operation for logging
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ConfluenceServiceError: Various service-related errors
        """
        self.metrics['total_requests'] += 1

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            self.metrics['circuit_breaker_open'] += 1
            raise ConfluenceCircuitOpenError(
                f"Circuit breaker is open for {operation_name}. "
                f"Service may be experiencing issues."
            )

        # Rate limiting
        await self.rate_limiter.wait_for_tokens()

        # Execute with retry and circuit breaker tracking
        try:
            result = await self.retry_handler.execute_with_retry(func, *args, **kwargs)

            # Record success
            self.circuit_breaker.record_success()
            self.metrics['successful_requests'] += 1

            logger.debug(f"Successfully executed {operation_name}")
            return result

        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            self.metrics['failed_requests'] += 1

            logger.error(f"Failed to execute {operation_name}: {e}")

            # Try fallback strategies
            fallback_result = await self._try_fallback(operation_name, *args, **kwargs)
            if fallback_result is not None:
                logger.info(f"Using fallback data for {operation_name}")
                return fallback_result

            raise ConfluenceServiceError(f"{operation_name} failed: {e}")

    async def _try_fallback(
        self,
        operation_name: str,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Attempt fallback strategies for failed operations.

        Implements graceful degradation as required by T026.

        Args:
            operation_name: Name of the failed operation
            *args: Original function arguments
            **kwargs: Original function keyword arguments

        Returns:
            Fallback data if available, None otherwise
        """
        cache_key = f"{operation_name}:{hash(str(args) + str(kwargs))}"

        # Try cached data
        if cache_key in self._fallback_cache:
            cached_data = self._fallback_cache[cache_key]
            cache_age = time.time() - cached_data['timestamp']

            # Use cache if it's not too old (1 hour for fallback)
            if cache_age < 3600:
                self.metrics['cache_hits'] += 1
                logger.info(f"Using cached fallback data for {operation_name}")
                return cached_data['data']

        # Could implement other fallback strategies here:
        # - Secondary Confluence instance
        # - Simplified responses
        # - Cached aggregate data

        return None

    def _cache_result(self, operation_name: str, result: Any, *args, **kwargs) -> None:
        """Cache successful results for fallback purposes."""
        cache_key = f"{operation_name}:{hash(str(args) + str(kwargs))}"

        self._fallback_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }

        # Limit cache size
        if len(self._fallback_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(
                self._fallback_cache.keys(),
                key=lambda k: self._fallback_cache[k]['timestamp']
            )
            del self._fallback_cache[oldest_key]

    async def search_content(
        self,
        query: str,
        spaces: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search Confluence content with full protection.

        Args:
            query: Search query
            spaces: List of space keys to search
            limit: Maximum number of results

        Returns:
            List of search results
        """

        async def _search():
            client = await self._get_client()

            # Build CQL query
            cql = query
            if not any(op in query.lower() for op in ['and', 'or', 'not', '~', '=']):
                cql = f"text ~ '{query}'"
                if spaces:
                    space_filter = " OR ".join([f"space = '{s}'" for s in spaces])
                    cql = f"({cql}) AND ({space_filter})"
                cql += " AND type = 'page'"

            # Execute search
            results = client.cql(cql, start=0, limit=limit, expand='body.view')

            # Process results
            processed_results = []
            for result in results.get('results', []):
                processed_result = {
                    'id': result.get('content', {}).get('id'),
                    'title': result.get('title'),
                    'space_key': result.get('space', {}).get('key'),
                    'url': result.get('url'),
                    'excerpt': result.get('excerpt', ''),
                    'type': result.get('content', {}).get('type'),
                    'status': result.get('content', {}).get('status')
                }
                processed_results.append(processed_result)

            return processed_results

        result = await self._execute_with_protection("search_content", _search)

        # Cache successful results
        self._cache_result("search_content", result, query, spaces, limit)

        return result

    async def get_page_content(
        self,
        page_id: str,
        expand: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get page content with full protection.

        Args:
            page_id: Confluence page ID
            expand: Fields to expand

        Returns:
            Page content
        """

        async def _get_page():
            client = await self._get_client()

            expand_fields = expand or 'body.storage,body.view,metadata.labels,version'
            page = client.get_page_by_id(page_id, expand=expand_fields)

            # Process page data
            processed_page = {
                'id': page.get('id'),
                'type': page.get('type'),
                'status': page.get('status'),
                'title': page.get('title'),
                'space': page.get('space', {}),
                'body': page.get('body', {}),
                'version': page.get('version', {}),
                'metadata': page.get('metadata', {}),
                'created_at': page.get('history', {}).get('createdDate'),
                'created_by': page.get('history', {}).get('createdBy', {}),
                '_links': page.get('_links', {})
            }

            return processed_page

        result = await self._execute_with_protection("get_page_content", _get_page)

        # Cache successful results
        self._cache_result("get_page_content", result, page_id, expand)

        return result

    async def get_space_content(
        self,
        space_key: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get space content with full protection.

        Args:
            space_key: Space key
            limit: Maximum number of pages

        Returns:
            List of pages in space
        """

        async def _get_space():
            client = await self._get_client()

            pages = client.get_all_pages_from_space(
                space=space_key,
                start=0,
                limit=limit,
                expand='body.view,version'
            )

            # Process pages
            processed_pages = []
            for page in pages:
                processed_page = {
                    'id': page.get('id'),
                    'title': page.get('title'),
                    'type': page.get('type'),
                    'status': page.get('status'),
                    'space_key': space_key,
                    'version': page.get('version', {}),
                    'created_at': page.get('history', {}).get('createdDate'),
                    '_links': page.get('_links', {})
                }
                processed_pages.append(processed_page)

            return processed_pages

        result = await self._execute_with_protection("get_space_content", _get_space)

        # Cache successful results
        self._cache_result("get_space_content", result, space_key, limit)

        return result

    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get service performance metrics.

        Returns:
            Dictionary with service metrics
        """
        circuit_state = self.circuit_breaker.state.value

        success_rate = 0.0
        if self.metrics['total_requests'] > 0:
            success_rate = (
                self.metrics['successful_requests'] /
                self.metrics['total_requests']
            ) * 100

        return {
            'total_requests': self.metrics['total_requests'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate_percent': round(success_rate, 2),
            'rate_limited_requests': self.metrics['rate_limited_requests'],
            'circuit_breaker_state': circuit_state,
            'circuit_breaker_open_count': self.metrics['circuit_breaker_open'],
            'cache_hits': self.metrics['cache_hits'],
            'cached_fallback_entries': len(self._fallback_cache),
            'last_health_check': self._last_health_check
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status information
        """
        is_healthy = await self._perform_health_check()
        metrics = self.get_service_metrics()

        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'rate_limiter_tokens': self.rate_limiter.tokens,
            'connection_pool_active': True,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def close(self):
        """Clean up resources."""
        logger.info("Closing ConfluenceService")

        if self.connection_pool:
            self.connection_pool.close()

        # Clear caches
        self._fallback_cache.clear()

        logger.info("ConfluenceService closed successfully")

    @asynccontextmanager
    async def session_context(self):
        """Context manager for proper resource cleanup."""
        try:
            yield self
        finally:
            await self.close()


# Factory function for easy service creation
def create_confluence_service(
    confluence_url: str,
    username: str,
    api_token: str,
    **kwargs
) -> ConfluenceService:
    """
    Factory function to create a configured ConfluenceService.

    Args:
        confluence_url: Confluence instance URL
        username: Username for authentication
        api_token: API token for authentication
        **kwargs: Additional configuration options

    Returns:
        Configured ConfluenceService instance
    """
    return ConfluenceService(
        confluence_url=confluence_url,
        username=username,
        api_token=api_token,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test_service():
        """Test the Confluence service."""
        service = create_confluence_service(
            confluence_url=os.getenv("CONFLUENCE_URL", ""),
            username=os.getenv("CONFLUENCE_USERNAME", ""),
            api_token=os.getenv("CONFLUENCE_API_TOKEN", "")
        )

        try:
            # Health check
            health = await service.health_check()
            print(f"Health check: {health}")

            # Search test
            results = await service.search_content("security", limit=5)
            print(f"Search results: {len(results)} found")

            # Metrics
            metrics = service.get_service_metrics()
            print(f"Service metrics: {metrics}")

        except Exception as e:
            print(f"Test failed: {e}")
        finally:
            await service.close()

    # Run test if executed directly
    asyncio.run(test_service())