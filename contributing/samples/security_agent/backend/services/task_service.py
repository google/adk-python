"""Async task service for handling long-running security operations."""

import asyncio
import uuid
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskProgress:
    """Task progress information."""
    current_step: str
    completed_steps: int
    total_steps: int
    percentage: float
    details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    progress: Optional[TaskProgress]
    result: Optional[Any]
    error: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.progress:
            data['progress'] = self.progress.to_dict()
        return data

class TaskService:
    """Service for managing async security scan tasks."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize task service.
        
        Args:
            max_workers: Maximum number of concurrent background tasks.
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, TaskResult] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
    async def submit_task(
        self,
        task_func: Callable[..., Awaitable[Any]],
        task_type: str,
        user_id: str,
        *args,
        **kwargs
    ) -> str:
        """Submit a task for async execution.
        
        Args:
            task_func: Async function to execute.
            task_type: Type/name of the task for identification.
            user_id: User identifier.
            *args: Arguments to pass to task function.
            **kwargs: Keyword arguments to pass to task function.
            
        Returns:
            Task ID for tracking progress.
        """
        task_id = str(uuid.uuid4())
        
        # Create task result entry
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            progress=None,
            result=None,
            error=None
        )
        
        self.tasks[task_id] = task_result
        
        # Create and start async task
        async_task = asyncio.create_task(
            self._execute_task(task_id, task_func, task_type, user_id, *args, **kwargs)
        )
        self.running_tasks[task_id] = async_task
        
        logger.info(f"Submitted task {task_id} of type {task_type} for user {user_id}")
        return task_id
    
    async def _execute_task(
        self,
        task_id: str,
        task_func: Callable[..., Awaitable[Any]],
        task_type: str,
        user_id: str,
        *args,
        **kwargs
    ) -> None:
        """Execute a task with error handling and progress tracking.
        
        Args:
            task_id: Unique task identifier.
            task_func: Function to execute.
            task_type: Task type for logging.
            user_id: User identifier.
            *args: Function arguments.
            **kwargs: Function keyword arguments.
        """
        task_result = self.tasks[task_id]
        
        try:
            # Update status to running
            task_result.status = TaskStatus.RUNNING
            task_result.started_at = time.time()
            
            logger.info(f"Starting execution of task {task_id} ({task_type})")
            
            # Execute the task function
            # Pass progress callback to task function if it accepts it
            if 'progress_callback' in task_func.__code__.co_varnames:
                kwargs['progress_callback'] = lambda progress: self._update_progress(task_id, progress)
            
            result = await task_func(*args, **kwargs)
            
            # Task completed successfully
            task_result.status = TaskStatus.COMPLETED
            task_result.completed_at = time.time()
            task_result.result = result
            
            logger.info(f"Task {task_id} completed successfully")
            
        except asyncio.CancelledError:
            task_result.status = TaskStatus.CANCELLED
            task_result.completed_at = time.time()
            logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            task_result.status = TaskStatus.FAILED
            task_result.completed_at = time.time()
            task_result.error = str(e)
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            
        finally:
            # Clean up running task reference
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _update_progress(self, task_id: str, progress: TaskProgress) -> None:
        """Update task progress.
        
        Args:
            task_id: Task identifier.
            progress: Progress information.
        """
        if task_id in self.tasks:
            self.tasks[task_id].progress = progress
            logger.debug(f"Task {task_id} progress: {progress.percentage:.1f}% - {progress.current_step}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and progress.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Task status dictionary or None if task not found.
        """
        if task_id not in self.tasks:
            return None
            
        return self.tasks[task_id].to_dict()
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            True if cancelled, False if task not found or not running.
        """
        if task_id not in self.running_tasks:
            return False
            
        task = self.running_tasks[task_id]
        task.cancel()
        
        logger.info(f"Cancelled task {task_id}")
        return True
    
    def list_user_tasks(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent tasks for a user.
        
        Args:
            user_id: User identifier.
            limit: Maximum number of tasks to return.
            
        Returns:
            List of task status dictionaries.
        """
        # In a real implementation, you'd filter by user_id and use proper storage
        # For now, return all tasks sorted by creation time
        sorted_tasks = sorted(
            self.tasks.values(), 
            key=lambda t: t.created_at, 
            reverse=True
        )
        
        return [task.to_dict() for task in sorted_tasks[:limit]]
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed/failed tasks.
        
        Args:
            max_age_hours: Maximum age in hours for keeping completed tasks.
            
        Returns:
            Number of tasks cleaned up.
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task_result in self.tasks.items():
            if (task_result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                and task_result.completed_at 
                and task_result.completed_at < cutoff_time):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old tasks")
        return len(to_remove)


# Global task service instance
task_service = TaskService()