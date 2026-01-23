"""
LLM utilities and rate limiting.

This module provides a global semaphore for limiting concurrent LLM calls
across all RLM components (agents, code executors, batched queries).
"""

import asyncio
from contextlib import contextmanager
import threading

# Global semaphore for limiting concurrent LLM calls.
# Uses threading.BoundedSemaphore because LLM calls happen across different
# contexts (sync code executor, async agents, different event loops).
LLM_CONCURRENCY_LIMIT = 30
_llm_semaphore = threading.BoundedSemaphore(LLM_CONCURRENCY_LIMIT)


@contextmanager
def llm_rate_limit():
  """Context manager for rate-limiting LLM calls (sync version).

  Usage:
      with llm_rate_limit():
          response = client.models.generate_content(...)
  """
  _llm_semaphore.acquire()
  try:
    yield
  finally:
    _llm_semaphore.release()


async def llm_rate_limit_async():
  """Async context manager for rate-limiting LLM calls.

  Usage:
      async with llm_rate_limit_async():
          response = await client.aio.models.generate_content(...)
  """
  # Acquire in a thread to avoid blocking the event loop
  await asyncio.to_thread(_llm_semaphore.acquire)
  return _AsyncSemaphoreReleaser()


class _AsyncSemaphoreReleaser:
  """Helper class for async context manager protocol."""

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    _llm_semaphore.release()
    return False


class AsyncLLMRateLimiter:
  """Async context manager for rate-limiting LLM calls.

  Usage:
      async with AsyncLLMRateLimiter():
          response = await client.aio.models.generate_content(...)
  """

  async def __aenter__(self):
    await asyncio.to_thread(_llm_semaphore.acquire)
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    _llm_semaphore.release()
    return False
