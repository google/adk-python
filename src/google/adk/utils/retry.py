
"""Retry decorator with exponential backoff for handling transient failures."""

import asyncio
import functools
import logging
import random
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for the decorated function's return type
T = TypeVar('T')

# Default retryable exceptions
DEFAULT_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

# Try to import common HTTP exceptions
try:
  import requests.exceptions
  HTTP_RETRYABLE_EXCEPTIONS = DEFAULT_RETRYABLE_EXCEPTIONS + (
      requests.exceptions.ConnectionError,
      requests.exceptions.Timeout,
      requests.exceptions.HTTPError,  # For 5xx errors
  )
except ImportError:
  HTTP_RETRYABLE_EXCEPTIONS = DEFAULT_RETRYABLE_EXCEPTIONS


class RetryError(Exception):
  """Raised when all retry attempts have been exhausted."""
  
  def __init__(self, message: str, last_exception: Optional[Exception] = None):
    super().__init__(message)
    self.last_exception = last_exception


def retry(
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
  """Decorator that retries a function with exponential backoff.
  
  This decorator can be used on both synchronous and asynchronous functions.
  It will retry the function when specified exceptions are raised, using
  exponential backoff with optional jitter to prevent thundering herd issues.
  
  Args:
    max_attempts: Maximum number of attempts (including the initial attempt).
    initial_delay: Initial delay in seconds before the first retry.
    max_delay: Maximum delay in seconds between retries.
    exponential_base: Base for exponential backoff calculation.
    jitter: Whether to add random jitter to delays to prevent thundering herd.
    retryable_exceptions: Tuple of exception types that should trigger a retry.
        Defaults to connection and timeout errors.
    on_retry: Optional callback function called before each retry with the
        exception and attempt number.
  
  Returns:
    Decorated function that implements retry logic.
    
  Raises:
    RetryError: When all retry attempts have been exhausted.
    
  Example:
    ```python
    @retry(max_attempts=3, initial_delay=1.0)
    async def fetch_data(url: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
                
    # With custom configuration
    @retry(
        max_attempts=5,
        initial_delay=0.5,
        max_delay=30.0,
        retryable_exceptions=(ValueError, KeyError),
        on_retry=lambda exc, attempt: print(f"Retry {attempt}: {exc}")
    )
    def process_data(data: dict) -> str:
        # Processing that might fail
        return data['result']
    ```
  """
  if retryable_exceptions is None:
    retryable_exceptions = DEFAULT_RETRYABLE_EXCEPTIONS
    
  def decorator(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
      last_exception: Optional[Exception] = None
      
      for attempt in range(1, max_attempts + 1):
        try:
          return await func(*args, **kwargs)
        except retryable_exceptions as e:
          last_exception = e
          
          if attempt == max_attempts:
            logger.error(
                f"All {max_attempts} attempts failed for {func.__name__}. "
                f"Last error: {e}"
            )
            raise RetryError(
                f"Failed after {max_attempts} attempts",
                last_exception=e
            ) from e
            
          # Calculate delay with exponential backoff
          delay = min(
              initial_delay * (exponential_base ** (attempt - 1)),
              max_delay
          )
          
          # Add jitter if enabled
          if jitter:
            delay *= (0.5 + random.random())
            
          logger.warning(
              f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
              f"Retrying in {delay:.2f} seconds..."
          )
          
          # Call retry callback if provided
          if on_retry:
            on_retry(e, attempt)
            
          await asyncio.sleep(delay)
          
      # This should never be reached due to the raise in the loop
      raise RetryError(
          f"Failed after {max_attempts} attempts",
          last_exception=last_exception
      )
      
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
      last_exception: Optional[Exception] = None
      
      for attempt in range(1, max_attempts + 1):
        try:
          return func(*args, **kwargs)
        except retryable_exceptions as e:
          last_exception = e
          
          if attempt == max_attempts:
            logger.error(
                f"All {max_attempts} attempts failed for {func.__name__}. "
                f"Last error: {e}"
            )
            raise RetryError(
                f"Failed after {max_attempts} attempts",
                last_exception=e
            ) from e
            
          # Calculate delay with exponential backoff
          delay = min(
              initial_delay * (exponential_base ** (attempt - 1)),
              max_delay
          )
          
          # Add jitter if enabled
          if jitter:
            delay *= (0.5 + random.random())
            
          logger.warning(
              f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
              f"Retrying in {delay:.2f} seconds..."
          )
          
          # Call retry callback if provided
          if on_retry:
            on_retry(e, attempt)
            
          # Use asyncio.run if available, otherwise fall back to time.sleep
          try:
            asyncio.run(asyncio.sleep(delay))
          except RuntimeError:
            # Not in async context, use regular sleep
            import time
            time.sleep(delay)
          
      # This should never be reached due to the raise in the loop
      raise RetryError(
          f"Failed after {max_attempts} attempts",
          last_exception=last_exception
      )
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
      return async_wrapper  # type: ignore
    else:
      return sync_wrapper  # type: ignore
      
  return decorator


# Convenience decorators with common configurations
retry_on_network_error = functools.partial(
    retry,
    max_attempts=3,
    initial_delay=1.0,
    retryable_exceptions=HTTP_RETRYABLE_EXCEPTIONS + (OSError,)  # Include HTTP and OS errors
)
"""Retry decorator specifically for network operations."""

retry_on_rate_limit = functools.partial(
    retry,
    max_attempts=5,
    initial_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0,
)
"""Retry decorator for handling rate limit errors with longer delays."""

retry_aggressive = functools.partial(
    retry,
    max_attempts=10,
    initial_delay=0.1,
    max_delay=30.0,
    jitter=True,
)
"""Aggressive retry strategy for critical operations."""