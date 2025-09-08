"""
Advanced Async Processing Utilities for GCP Security Agent
High-performance async patterns, batch processing, and concurrent operations
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import logging
from functools import wraps, partial
from contextlib import asynccontextmanager
import weakref
from collections import defaultdict
import heapq
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

@dataclass
class AsyncConfig:
    """Configuration for async operations"""
    # Concurrency limits
    max_concurrent_requests: int = 100
    max_concurrent_db_ops: int = 50
    max_concurrent_api_calls: int = 20
    
    # Batch processing
    batch_size: int = 100
    batch_timeout: float = 1.0  # seconds
    
    # Thread/process pools
    thread_pool_size: int = 20
    process_pool_size: int = 4
    
    # Rate limiting
    rate_limit: float = 100  # requests per second
    burst_size: int = 200
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Timeout settings
    default_timeout: float = 30.0
    long_operation_timeout: float = 300.0

class RateLimiter:
    """Token bucket rate limiter for async operations"""
    
    def __init__(self, rate: float, burst_size: int):
        self.rate = rate  # tokens per second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        async with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.01)

class AsyncBatchProcessor(Generic[T, R]):
    """Batch processor for high-throughput async operations"""
    
    def __init__(
        self,
        processor_func: Callable[[List[T]], asyncio.coroutine],
        batch_size: int = 100,
        batch_timeout: float = 1.0,
        max_concurrent_batches: int = 10
    ):
        self.processor_func = processor_func
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        
        self._queue = asyncio.Queue()
        self._results = {}
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._processor_task = None
        self._shutdown = False
    
    async def start(self):
        """Start the batch processor"""
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_batches())
    
    async def stop(self):
        """Stop the batch processor"""
        self._shutdown = True
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
    
    async def process(self, item: T) -> R:
        """Add item for batch processing and return result"""
        if self._shutdown:
            raise RuntimeError("Batch processor is shut down")
        
        # Create future for result
        result_future = asyncio.Future()
        await self._queue.put((item, result_future))
        
        return await result_future
    
    async def process_many(self, items: List[T]) -> List[R]:
        """Process multiple items and return results in order"""
        futures = []
        for item in items:
            result_future = asyncio.Future()
            await self._queue.put((item, result_future))
            futures.append(result_future)
        
        return await asyncio.gather(*futures)
    
    async def _process_batches(self):
        """Main batch processing loop"""
        while not self._shutdown:
            try:
                batch = []
                futures = []
                
                # Collect batch items
                deadline = time.time() + self.batch_timeout
                
                while len(batch) < self.batch_size and time.time() < deadline:
                    try:
                        remaining_time = max(0, deadline - time.time())
                        item, future = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining_time
                        )
                        batch.append(item)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                async with self._batch_semaphore:
                    try:
                        results = await self.processor_func(batch)
                        
                        # Return results to futures
                        for future, result in zip(futures, results):
                            if not future.done():
                                future.set_result(result)
                    
                    except Exception as e:
                        # Set exception for all futures in batch
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

class AsyncWorkQueue:
    """High-performance work queue for concurrent task execution"""
    
    def __init__(
        self,
        max_workers: int = 100,
        max_queue_size: int = 1000,
        priority_levels: int = 3
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.priority_levels = priority_levels
        
        # Priority queues (higher number = higher priority)
        self._queues = [asyncio.Queue(max_queue_size) for _ in range(priority_levels)]
        self._semaphore = asyncio.Semaphore(max_workers)
        self._workers = []
        self._shutdown = False
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_workers': 0
        }
    
    async def start(self):
        """Start worker tasks"""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
    
    async def stop(self):
        """Stop all workers"""
        self._shutdown = True
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
    
    async def submit(
        self,
        coro: asyncio.coroutine,
        priority: int = 0,
        timeout: float = None
    ) -> Any:
        """Submit a coroutine for execution"""
        if self._shutdown:
            raise RuntimeError("Work queue is shut down")
        
        priority = max(0, min(priority, self.priority_levels - 1))
        
        future = asyncio.Future()
        task_data = {
            'coro': coro,
            'future': future,
            'timeout': timeout,
            'submitted_at': time.time()
        }
        
        await self._queues[priority].put(task_data)
        self._stats['total_tasks'] += 1
        
        return await future
    
    async def submit_batch(
        self,
        coros: List[asyncio.coroutine],
        priority: int = 0,
        timeout: float = None
    ) -> List[Any]:
        """Submit multiple coroutines for execution"""
        futures = []
        for coro in coros:
            future = asyncio.Future()
            task_data = {
                'coro': coro,
                'future': future,
                'timeout': timeout,
                'submitted_at': time.time()
            }
            await self._queues[priority].put(task_data)
            futures.append(future)
        
        self._stats['total_tasks'] += len(coros)
        return await asyncio.gather(*futures, return_exceptions=True)
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks"""
        while not self._shutdown:
            try:
                # Try to get task from highest priority queue first
                task_data = None
                for priority in range(self.priority_levels - 1, -1, -1):
                    try:
                        task_data = self._queues[priority].get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue
                
                # If no high-priority tasks, wait for any task
                if task_data is None:
                    tasks = [q.get() for q in self._queues]
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                    
                    if done:
                        task_data = await done.pop()
                
                if task_data is None:
                    continue
                
                async with self._semaphore:
                    self._stats['active_workers'] += 1
                    
                    try:
                        # Execute task with timeout
                        timeout = task_data.get('timeout')
                        if timeout:
                            result = await asyncio.wait_for(task_data['coro'], timeout)
                        else:
                            result = await task_data['coro']
                        
                        # Set result
                        if not task_data['future'].done():
                            task_data['future'].set_result(result)
                        
                        self._stats['completed_tasks'] += 1
                    
                    except Exception as e:
                        # Set exception
                        if not task_data['future'].done():
                            task_data['future'].set_exception(e)
                        
                        self._stats['failed_tasks'] += 1
                        logger.error(f"Task failed in {worker_name}: {e}")
                    
                    finally:
                        self._stats['active_workers'] -= 1
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        queue_sizes = [q.qsize() for q in self._queues]
        return {
            **self._stats,
            'queue_sizes': queue_sizes,
            'total_queued': sum(queue_sizes)
        }

class AsyncRetryHandler:
    """Configurable retry handler for async operations"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        retriable_exceptions: Tuple[Exception, ...] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retriable_exceptions = retriable_exceptions or (Exception,)
    
    async def execute(self, coro_func: Callable[[], asyncio.coroutine]) -> Any:
        """Execute with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_func()
            
            except Exception as e:
                last_exception = e
                
                # Check if exception is retriable
                if not isinstance(e, self.retriable_exceptions):
                    raise
                
                # Don't retry on last attempt
                if attempt == self.max_retries:
                    raise
                
                # Calculate delay
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )
                
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        raise last_exception

# Decorators for async operations
def async_cached(ttl: int = 300, key_func: Callable = None):
    """Decorator for caching async function results"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

def async_rate_limited(rate: float, burst_size: int = None):
    """Decorator for rate limiting async functions"""
    burst_size = burst_size or int(rate * 2)
    limiter = RateLimiter(rate, burst_size)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.wait_for_tokens()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def async_timeout(timeout: float):
    """Decorator for adding timeout to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator

def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for adding retry logic to async functions"""
    retry_handler = AsyncRetryHandler(max_retries, delay, backoff)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute(lambda: func(*args, **kwargs))
        return wrapper
    return decorator

class AsyncMetrics:
    """Performance metrics collector for async operations"""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def record_execution_time(self, operation: str, duration: float):
        """Record execution time for an operation"""
        async with self._lock:
            self._metrics[f"{operation}_duration"].append({
                'value': duration,
                'timestamp': time.time()
            })
    
    async def record_counter(self, metric: str, value: int = 1):
        """Record a counter metric"""
        async with self._lock:
            self._metrics[f"{metric}_count"].append({
                'value': value,
                'timestamp': time.time()
            })
    
    async def get_metrics(self, operation: str = None, window: int = 3600) -> Dict[str, Any]:
        """Get metrics for the specified time window"""
        current_time = time.time()
        cutoff_time = current_time - window
        
        async with self._lock:
            result = {}
            
            for metric_name, values in self._metrics.items():
                if operation and not metric_name.startswith(operation):
                    continue
                
                # Filter by time window
                recent_values = [
                    v['value'] for v in values
                    if v['timestamp'] > cutoff_time
                ]
                
                if recent_values:
                    if 'duration' in metric_name:
                        result[metric_name] = {
                            'count': len(recent_values),
                            'avg': sum(recent_values) / len(recent_values),
                            'min': min(recent_values),
                            'max': max(recent_values),
                            'p95': sorted(recent_values)[int(len(recent_values) * 0.95)]
                        }
                    else:
                        result[metric_name] = {
                            'total': sum(recent_values),
                            'count': len(recent_values)
                        }
            
            return result
    
    async def cleanup_old_metrics(self, max_age: int = 86400):
        """Clean up metrics older than max_age seconds"""
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        async with self._lock:
            for metric_name in list(self._metrics.keys()):
                self._metrics[metric_name] = [
                    v for v in self._metrics[metric_name]
                    if v['timestamp'] > cutoff_time
                ]
                
                if not self._metrics[metric_name]:
                    del self._metrics[metric_name]

def async_monitored(metrics: AsyncMetrics, operation_name: str = None):
    """Decorator for monitoring async function performance"""
    def decorator(func):
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                await metrics.record_counter(f"{op_name}_success")
                return result
            except Exception as e:
                await metrics.record_counter(f"{op_name}_error")
                raise
            finally:
                duration = time.time() - start_time
                await metrics.record_execution_time(op_name, duration)
        
        return wrapper
    return decorator

# Utility functions
async def gather_with_concurrency(coros: List[asyncio.coroutine], max_concurrency: int) -> List[Any]:
    """Execute coroutines with limited concurrency"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[sem_coro(coro) for coro in coros])

async def gather_with_timeout(
    coros: List[asyncio.coroutine],
    timeout: float,
    return_exceptions: bool = True
) -> List[Any]:
    """Execute coroutines with overall timeout"""
    return await asyncio.wait_for(
        asyncio.gather(*coros, return_exceptions=return_exceptions),
        timeout=timeout
    )

# Global instances
config = AsyncConfig()
metrics = AsyncMetrics()