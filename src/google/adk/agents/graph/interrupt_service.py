"""Dynamic interrupt service for GraphAgent human-in-the-loop.

Enables real-time pause/resume control and message injection during graph execution.
Unlike static interrupt_before configuration, this provides:
- Dynamic runtime interrupts (not pre-configured)
- Actual pause/resume (escalate=True)
- Message queue for human input
- Per-session isolation for concurrent use
- Queue bounds to prevent OOM
- Timeout and cancellation support
- Queue status and message pagination APIs
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ...telemetry.checkpoint_tracing import record_interrupt_service_metrics
from ...telemetry.checkpoint_tracing import trace_interrupt_service_cancel
from ...telemetry.checkpoint_tracing import trace_interrupt_service_message
from ...telemetry.checkpoint_tracing import trace_interrupt_service_pause
from ...telemetry.checkpoint_tracing import trace_interrupt_service_resume
from ...telemetry.checkpoint_tracing import update_interrupt_queue_depth


@dataclass
class InterruptServiceConfig:
  """Configuration for InterruptService.

  Attributes:
      max_queue_size: Maximum messages per session queue (default: 100)
      default_timeout: Default timeout in seconds for wait operations (default: 300.0)
      enable_metrics: Whether to track session metrics (default: True)
      max_message_length: Maximum message text length in characters (default: 10000)
      max_metadata_size: Maximum metadata size in bytes (default: 100000)
      allowed_actions: Allowlist of action types, None for any (default: None)
      session_inactive_timeout: Seconds before inactive sessions are evicted (default: 3600, 0 = never)
      max_sessions: Maximum active sessions, oldest evicted first (default: 0 = unlimited)
  """

  max_queue_size: int = 100
  default_timeout: float = 300.0
  enable_metrics: bool = True
  max_message_length: int = 10000  # 10KB text
  max_metadata_size: int = 100000  # 100KB metadata
  allowed_actions: Optional[list[str]] = None  # None = allow any
  session_inactive_timeout: float = 3600.0  # 1 hour
  max_sessions: int = 0  # 0 = unlimited

  def __post_init__(self) -> None:
    """Validate configuration values."""
    if self.max_queue_size < 1:
      raise ValueError("max_queue_size must be positive")
    if self.default_timeout <= 0:
      raise ValueError("default_timeout must be positive")
    if self.max_message_length < 1:
      raise ValueError("max_message_length must be positive")
    if self.max_metadata_size < 1:
      raise ValueError("max_metadata_size must be positive")
    if self.session_inactive_timeout < 0:
      raise ValueError("session_inactive_timeout must be non-negative")
    if self.max_sessions < 0:
      raise ValueError("max_sessions must be non-negative")


@dataclass
class SessionMetrics:
  """Per-session interrupt metrics.

  Attributes:
      messages_queued: Total messages queued for this session
      pause_count: Number of times session was paused
      resume_count: Number of times session was resumed
      cancel_count: Number of times session was cancelled
      max_queue_depth: Maximum queue depth reached
      created_at: Timestamp when session was registered
      last_activity_at: Timestamp of last activity (for eviction)
  """

  messages_queued: int = 0
  pause_count: int = 0
  resume_count: int = 0
  cancel_count: int = 0
  max_queue_depth: int = 0
  created_at: float = 0.0
  last_activity_at: float = 0.0


@dataclass
class QueueStatus:
  """Queue status information for a session.

  Attributes:
      session_id: Session identifier
      is_paused: Whether session is currently paused
      is_cancelled: Whether session is cancelled
      queue_depth: Current number of queued messages
      max_queue_size: Maximum queue capacity
      metrics: Session metrics if enabled
  """

  session_id: str
  is_paused: bool
  is_cancelled: bool
  queue_depth: int
  max_queue_size: int
  metrics: Optional[SessionMetrics] = None


class InterruptMessage:
  """Message from human to paused agent.

  Attributes:
      text: Human message text
      metadata: Additional structured data
      action: Action type for message processing (e.g., "update_state")
  """

  def __init__(
      self,
      text: str,
      metadata: Optional[Dict[str, Any]] = None,
      action: Optional[str] = None,
  ):
    self.text = text
    self.metadata = metadata or {}
    self.action = action


class InterruptService:
  """Service for dynamic runtime interrupts in GraphAgent.

  Enables human-in-the-loop workflows with:
  - Real-time pause/resume during execution
  - Message injection from human to agent
  - State/condition updates mid-execution
  - Queue bounds to prevent OOM
  - Timeout and cancellation support
  - Metrics and observability

  Thread-safe for concurrent sessions (each session has isolated queue/event).

  Example:
      ```python
      # Create service with config
      config = InterruptServiceConfig(
          max_queue_size=100,
          default_timeout=300.0,
          enable_metrics=True
      )
      interrupt_service = InterruptService(config)

      # Create GraphAgent with interrupts enabled
      graph = GraphAgent(
          name="workflow",
          interrupt_service=interrupt_service
      )

      # In separate task/thread: pause and send message
      await interrupt_service.pause(session_id)
      await interrupt_service.send_message(
          session_id,
          "Change parameter X to Y",
          action="update_state",
          metadata={"param_x": "Y"}
      )

      # Check queue status
      status = interrupt_service.get_queue_status(session_id)
      print(f"Queue depth: {status.queue_depth}")

      # Resume or cancel
      await interrupt_service.resume(session_id)
      # or
      await interrupt_service.cancel(session_id)

      # Alternative: Use context manager for session lifecycle
      async with interrupt_service.session(session_id):
          await interrupt_service.pause(session_id)
          await interrupt_service.send_message(session_id, "Test")
          # Session automatically cleaned up on exit
      ```

  Note:
      This service is thread-safe per-session. Each session gets isolated
      asyncio.Queue and asyncio.Event primitives.
  """

  class SessionContext:
    """Async context manager for session lifecycle.

    Automatically registers session on enter and unregisters on exit.
    """

    def __init__(self, service: "InterruptService", session_id: str) -> None:
      self._service = service
      self._session_id = session_id

    async def __aenter__(self) -> "InterruptService.SessionContext":
      """Register session on context entry."""
      self._service.register_session(self._session_id)
      return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
      """Unregister session on context exit."""
      self._service.unregister_session(self._session_id)
      return False  # Don't suppress exceptions

  def __init__(self, config: Optional[InterruptServiceConfig] = None) -> None:
    """Initialize interrupt service with configuration.

    Args:
        config: Service configuration. Uses defaults if not provided.
    """
    self.config = config or InterruptServiceConfig()
    self._message_queues: Dict[str, "asyncio.Queue[InterruptMessage]"] = {}
    self._pause_events: Dict[str, asyncio.Event] = {}
    self._cancellation_events: Dict[str, asyncio.Event] = {}
    self._session_metrics: Dict[str, SessionMetrics] = {}
    self._active_sessions: set[str] = set()

  def session(self, session_id: str) -> "InterruptService.SessionContext":
    """Create context manager for session lifecycle.

    Provides cleaner API for managing session registration/unregistration.

    Args:
        session_id: Session identifier

    Returns:
        SessionContext async context manager

    Example:
        ```python
        async with interrupt_service.session("session-123"):
            # Session is automatically registered
            await interrupt_service.pause("session-123")
            # ... work ...
            # Session is automatically unregistered on exit
        ```
    """
    return InterruptService.SessionContext(self, session_id)

  def _validate_session_id(self, session_id: str) -> None:
    """Validate session ID format (alphanumeric + hyphens/underscores only).

    Args:
        session_id: Session identifier to validate

    Raises:
        ValueError: If session_id is invalid with helpful error message
    """
    if not session_id or not isinstance(session_id, str):
      raise ValueError(
          "session_id must be a non-empty string. "
          f"Received: {type(session_id).__name__} = {repr(session_id)}"
      )

    if len(session_id) > 256:
      raise ValueError(
          f"session_id exceeds maximum length: {len(session_id)} chars (max:"
          " 256). Consider using a shorter identifier or hash. Received:"
          f" '{session_id[:50]}...'"
      )

    if not re.match(r"^[a-zA-Z0-9\-_]+$", session_id):
      # Find invalid characters
      invalid_chars = set(
          c for c in session_id if not re.match(r"[a-zA-Z0-9\-_]", c)
      )
      raise ValueError(
          f"Invalid session_id format: '{session_id}'. Found invalid"
          f" characters: {sorted(invalid_chars)}. session_id must contain only"
          " alphanumeric characters, hyphens (-), and underscores (_)"
      )

  def _validate_message(
      self,
      text: str,
      metadata: Optional[Dict[str, Any]],
      action: Optional[str],
  ) -> None:
    """Validate message content before queuing.

    Args:
        text: Message text
        metadata: Message metadata
        action: Message action type

    Raises:
        ValueError: If message content is invalid with helpful error message
    """
    # Validate text length
    if not text or not isinstance(text, str):
      raise ValueError(
          "Message text must be a non-empty string. "
          f"Received: {type(text).__name__} = {repr(text)}"
      )

    if len(text) > self.config.max_message_length:
      raise ValueError(
          f"Message text exceeds maximum length: {len(text)} chars "
          f"(max: {self.config.max_message_length}). "
          "Consider splitting into multiple messages or adjusting config. "
          f"Preview: '{text[:100]}...'"
      )

    # Validate metadata size
    if metadata:
      # Use JSON serialization to get accurate size
      try:
        metadata_json = json.dumps(metadata)
        metadata_size = len(metadata_json.encode("utf-8"))
        if metadata_size > self.config.max_metadata_size:
          raise ValueError(
              f"Metadata exceeds maximum size: {metadata_size} bytes "
              f"(max: {self.config.max_metadata_size}). "
              "Consider reducing metadata complexity or adjusting config. "
              f"Metadata keys: {list(metadata.keys())}"
          )
      except (TypeError, ValueError) as e:
        raise ValueError(
            "Metadata must be JSON-serializable. "
            f"Failed to serialize: {e}. "
            "Check for circular references or non-serializable types. "
            f"Metadata type: {type(metadata).__name__}"
        )

    # Validate action against allowlist
    if action and self.config.allowed_actions:
      if action not in self.config.allowed_actions:
        raise ValueError(
            f"Action '{action}' is not allowed. "
            f"Allowed actions: {self.config.allowed_actions}. "
            "Check your InterruptServiceConfig.allowed_actions setting."
        )

  def register_session(self, session_id: str) -> None:
    """Register session for interrupt handling.

    Called automatically by GraphAgent at execution start.
    Creates isolated queue (with bounds) and events for this session.

    Args:
        session_id: Session identifier

    Raises:
        ValueError: If session_id format is invalid
    """
    # Validate session ID format
    self._validate_session_id(session_id)

    if session_id not in self._message_queues:
      # Create bounded queue to prevent OOM
      self._message_queues[session_id] = asyncio.Queue(
          maxsize=self.config.max_queue_size
      )
      self._pause_events[session_id] = asyncio.Event()
      self._pause_events[session_id].set()  # Start unpaused
      self._cancellation_events[session_id] = asyncio.Event()
      # Cancellation event starts clear (not cancelled)
      self._active_sessions.add(session_id)

      # Initialize metrics if enabled
      if self.config.enable_metrics:
        current_time = time.time()
        self._session_metrics[session_id] = SessionMetrics(
            created_at=current_time, last_activity_at=current_time
        )

      # Enforce max_sessions limit (LRU eviction)
      if (
          self.config.max_sessions > 0
          and len(self._active_sessions) > self.config.max_sessions
      ):
        self.evict_oldest_sessions(self.config.max_sessions)

  def unregister_session(self, session_id: str) -> None:
    """Clean up session resources.

    Called automatically by GraphAgent at execution end.
    Removes queue, events, metrics, and session tracking.

    Args:
        session_id: Session identifier
    """
    self._message_queues.pop(session_id, None)
    self._pause_events.pop(session_id, None)
    self._cancellation_events.pop(session_id, None)
    self._session_metrics.pop(session_id, None)
    self._active_sessions.discard(session_id)

  async def check_interrupt(
      self, session_id: str
  ) -> Optional[InterruptMessage]:
    """Non-blocking check for pending messages.

    Called by GraphAgent every iteration to check for human messages.
    Returns immediately if no messages (no performance impact).

    Args:
        session_id: Session identifier

    Returns:
        InterruptMessage if pending, None otherwise
    """
    if session_id not in self._message_queues:
      return None

    try:
      message = self._message_queues[session_id].get_nowait()

      # Update activity timestamp
      self._update_activity(session_id)

      # Update queue depth gauge after getting message
      if self.config.enable_metrics:
        queue_depth = self._message_queues[session_id].qsize()
        update_interrupt_queue_depth(session_id, queue_depth)

      return message
    except asyncio.QueueEmpty:
      return None

  async def wait_if_paused(
      self, session_id: str, timeout: Optional[float] = None
  ) -> bool:
    """Block if session is paused with timeout and cancellation support.

    Called by GraphAgent after processing interrupt message.
    Waits until human calls resume() or cancel().

    Args:
        session_id: Session identifier
        timeout: Maximum seconds to wait. Uses config.default_timeout if None.

    Returns:
        True if resumed normally, False if cancelled

    Raises:
        TimeoutError: If timeout exceeded while waiting
    """
    if session_id not in self._pause_events:
      return True  # Not registered, continue

    # Use configured default timeout if not specified
    if timeout is None:
      timeout = self.config.default_timeout

    # Create wait tasks for pause event and cancellation event
    pause_wait = asyncio.create_task(self._pause_events[session_id].wait())
    cancel_wait = asyncio.create_task(
        self._cancellation_events[session_id].wait()
    )

    try:
      # Wait for either resume or cancellation (or timeout)
      done, pending = await asyncio.wait(
          [pause_wait, cancel_wait],
          timeout=timeout,
          return_when=asyncio.FIRST_COMPLETED,
      )

      # Cancel pending tasks
      for task in pending:
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

      # Check if cancelled
      if self._cancellation_events[session_id].is_set():
        return False  # Cancelled

      # Check if timeout
      if not done:
        raise TimeoutError(
            f"Timed out waiting for resume on session '{session_id}' after"
            f" {timeout}s. "
            "Session is paused and waiting for human input. "
            "Call resume(session_id) or cancel(session_id) to continue. "
            f"Current status: paused={self.is_paused(session_id)}, "
            f"cancelled={self._cancellation_events[session_id].is_set()}, "
            f"queue_depth={self._message_queues.get(session_id, asyncio.Queue()).qsize()}"
        )

      return True  # Resumed normally

    except Exception:
      # Clean up tasks on exception
      for task in [pause_wait, cancel_wait]:
        if not task.done():
          task.cancel()
          try:
            await task
          except asyncio.CancelledError:
            pass
      raise

  def is_paused(self, session_id: str) -> bool:
    """Check if session is currently paused.

    Args:
        session_id: Session identifier

    Returns:
        True if paused, False otherwise
    """
    if session_id in self._pause_events:
      return not self._pause_events[session_id].is_set()
    return False

  async def pause(self, session_id: str) -> None:
    """Pause execution for this session.

    Called by human/client to pause graph execution.
    GraphAgent will block at next iteration until resume().

    Args:
        session_id: Session identifier
    """
    start_time = time.time()

    try:
      if session_id in self._pause_events:
        self._pause_events[session_id].clear()  # Block agent

        # Update activity timestamp
        self._update_activity(session_id)

        # Track metrics
        if self.config.enable_metrics and session_id in self._session_metrics:
          self._session_metrics[session_id].pause_count += 1

        # Trace operation
        trace_interrupt_service_pause(session_id)

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        record_interrupt_service_metrics("pause", duration_ms, "success")

    except Exception:
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics("pause", duration_ms, "error")
      raise

  async def resume(self, session_id: str) -> None:
    """Resume execution for this session.

    Called by human/client to resume graph execution.
    Unblocks GraphAgent to continue execution.

    Args:
        session_id: Session identifier
    """
    start_time = time.time()

    try:
      if session_id in self._pause_events:
        self._pause_events[session_id].set()  # Unblock agent

        # Update activity timestamp
        self._update_activity(session_id)

        # Track metrics
        if self.config.enable_metrics and session_id in self._session_metrics:
          self._session_metrics[session_id].resume_count += 1

        # Trace operation
        trace_interrupt_service_resume(session_id)

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        record_interrupt_service_metrics("resume", duration_ms, "success")

    except Exception:
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics("resume", duration_ms, "error")
      raise

  async def send_message(
      self,
      session_id: str,
      text: str,
      metadata: Optional[Dict[str, Any]] = None,
      action: Optional[str] = None,
      timeout: Optional[float] = None,
  ) -> None:
    """Send message to paused agent.

    Called by human/client to inject message into execution.
    Message will be processed by GraphAgent at next iteration.

    Args:
        session_id: Session identifier
        text: Human message text
        metadata: Additional structured data
        action: Action type (e.g., "update_state", "change_condition")
        timeout: Maximum seconds to wait if queue is full. Uses config.default_timeout if None.

    Raises:
        ValueError: If message content is invalid
        asyncio.TimeoutError: If queue is full and timeout exceeded
        asyncio.QueueFull: If queue is full and timeout is 0
    """
    # Validate message content
    self._validate_message(text, metadata, action)

    if session_id not in self._message_queues:
      return

    message = InterruptMessage(text, metadata, action)
    start_time = time.time()

    # Use configured default timeout if not specified
    if timeout is None:
      timeout = self.config.default_timeout

    # Put message with timeout
    try:
      await asyncio.wait_for(
          self._message_queues[session_id].put(message), timeout=timeout
      )

      # Update activity timestamp
      self._update_activity(session_id)

      # Track metrics
      current_depth = 0
      if self.config.enable_metrics and session_id in self._session_metrics:
        metrics = self._session_metrics[session_id]
        metrics.messages_queued += 1

        # Update max queue depth
        current_depth = self._message_queues[session_id].qsize()
        if current_depth > metrics.max_queue_depth:
          metrics.max_queue_depth = current_depth

        # Update queue depth gauge
        update_interrupt_queue_depth(session_id, current_depth)

      # Trace operation
      trace_interrupt_service_message(session_id, text, action, current_depth)

      # Record metrics
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics(
          "message", duration_ms, "success", current_depth
      )

    except asyncio.TimeoutError:
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics("message", duration_ms, "timeout")
      queue_size = self._message_queues[session_id].qsize()
      raise asyncio.TimeoutError(
          f"Failed to queue message for session '{session_id}' after"
          f" {timeout}s. Queue is full:"
          f" {queue_size}/{self.config.max_queue_size} messages. The agent may"
          " be paused or not processing messages fast enough. Try: (1)"
          " resume(session_id) if paused, (2) clear_queue(session_id) to"
          " discard old messages, (3) increase config.max_queue_size, or (4)"
          " wait longer with a higher timeout value."
      )
    except Exception:
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics("message", duration_ms, "error")
      raise

  async def cancel(self, session_id: str) -> None:
    """Cancel execution for this session.

    Called by human/client to abort graph execution (like Claude Code's Esc key).
    Clears the message queue and sets cancellation event.
    GraphAgent will detect cancellation and exit gracefully.

    Args:
        session_id: Session identifier
    """
    if session_id not in self._cancellation_events:
      return

    start_time = time.time()

    try:
      # Set cancellation event (unblocks wait_if_paused)
      self._cancellation_events[session_id].set()

      # Clear message queue and count messages
      messages_cleared = self._drain_queue(session_id)

      # Track metrics
      if self.config.enable_metrics and session_id in self._session_metrics:
        self._session_metrics[session_id].cancel_count += 1

      # Trace operation
      trace_interrupt_service_cancel(session_id, messages_cleared)

      # Record metrics
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics("cancel", duration_ms, "success")

    except Exception:
      duration_ms = (time.time() - start_time) * 1000
      record_interrupt_service_metrics("cancel", duration_ms, "error")
      raise

  def get_queue_status(self, session_id: str) -> Optional[QueueStatus]:
    """Get current queue status for session (like OpenCode's /queue status).

    Provides observability into interrupt state and queue depth.

    Args:
        session_id: Session identifier

    Returns:
        QueueStatus if session is registered, None otherwise
    """
    if session_id not in self._message_queues:
      return None

    queue_depth = self._message_queues[session_id].qsize()
    is_paused = self.is_paused(session_id)
    is_cancelled = self._cancellation_events[session_id].is_set()

    metrics = None
    if self.config.enable_metrics and session_id in self._session_metrics:
      metrics = self._session_metrics[session_id]

    return QueueStatus(
        session_id=session_id,
        is_paused=is_paused,
        is_cancelled=is_cancelled,
        queue_depth=queue_depth,
        max_queue_size=self.config.max_queue_size,
        metrics=metrics,
    )

  def list_queued_messages(
      self, session_id: str, page: int = 1, page_size: int = 50
  ) -> List[InterruptMessage]:
    """List queued messages with ADK-style pagination.

    Provides non-destructive peek at queue for observability and debugging.
    Messages remain in queue for processing by GraphAgent.

    Uses direct access to asyncio.Queue's internal deque for non-destructive
    reads. No drain/requeue needed — avoids QueueFull risk entirely.

    Args:
        session_id: Session identifier
        page: Page number (1-indexed, default: 1)
        page_size: Messages per page (default: 50, max: 1000)

    Returns:
        List of InterruptMessage objects (may be empty)
    """
    # Validate pagination bounds (ADK pattern)
    if page < 1:
      page = 1
    if page_size < 1 or page_size > 1000:
      page_size = 50

    if session_id not in self._message_queues:
      return []

    # Direct deque access: non-destructive read of the underlying deque.
    # asyncio.Queue stores items in _queue (a collections.deque).
    # This avoids drain/requeue and eliminates QueueFull risk.
    queue = self._message_queues[session_id]
    messages = list(queue._queue)  # type: ignore[attr-defined]
    offset = (page - 1) * page_size
    return messages[offset : offset + page_size]

  # ===========================
  # Convenience Methods
  # ===========================

  async def pause_and_send(
      self,
      session_id: str,
      text: str,
      metadata: Optional[Dict[str, Any]] = None,
      action: Optional[str] = None,
      timeout: Optional[float] = None,
  ) -> None:
    """Pause session and send message in one call.

    Convenience method for the common pattern of pausing then sending a message.

    Args:
        session_id: Session identifier
        text: Human message text
        metadata: Additional structured data
        action: Action type (e.g., "update_state")
        timeout: Maximum seconds to wait if queue is full

    Raises:
        ValueError: If message content is invalid
        asyncio.TimeoutError: If queue is full and timeout exceeded
    """
    await self.pause(session_id)
    await self.send_message(session_id, text, metadata, action, timeout)

  def is_active(self, session_id: str) -> bool:
    """Check if session is registered and active.

    Args:
        session_id: Session identifier

    Returns:
        True if session is registered and not cancelled, False otherwise
    """
    if session_id not in self._active_sessions:
      return False
    if session_id in self._cancellation_events:
      return not self._cancellation_events[session_id].is_set()
    return True

  def _drain_queue(self, session_id: str) -> int:
    """Drain all messages from the session queue and return the count.

    Handles the TOCTOU race between empty() check and get_nowait() by
    catching QueueEmpty. Safe to call whether or not the queue exists.

    Args:
        session_id: Session identifier

    Returns:
        Number of messages drained
    """
    if session_id not in self._message_queues:
      return 0
    queue = self._message_queues[session_id]
    drained = 0
    while not queue.empty():
      try:
        queue.get_nowait()
        drained += 1
      except asyncio.QueueEmpty:
        break
    return drained

  def clear_queue(self, session_id: str) -> int:
    """Clear all queued messages without cancelling session.

    Useful for discarding pending messages while keeping session active.

    Args:
        session_id: Session identifier

    Returns:
        Number of messages cleared
    """
    messages_cleared = self._drain_queue(session_id)

    # Update queue depth gauge
    if self.config.enable_metrics and messages_cleared > 0:
      update_interrupt_queue_depth(session_id, 0)

    return messages_cleared

  def has_queued_messages(self, session_id: str) -> bool:
    """Quick check if session has pending messages.

    Args:
        session_id: Session identifier

    Returns:
        True if session has queued messages, False otherwise
    """
    if session_id not in self._message_queues:
      return False
    return not self._message_queues[session_id].empty()

  def get_queue_depth(self, session_id: str) -> int:
    """Get current queue depth for session.

    Args:
        session_id: Session identifier

    Returns:
        Number of queued messages (0 if session not found)
    """
    if session_id not in self._message_queues:
      return 0
    return self._message_queues[session_id].qsize()

  # ===========================
  # Session Eviction (Scale)
  # ===========================

  def _update_activity(self, session_id: str) -> None:
    """Update last activity timestamp for session.

    Args:
        session_id: Session identifier
    """
    if self.config.enable_metrics and session_id in self._session_metrics:
      self._session_metrics[session_id].last_activity_at = time.time()

  def evict_inactive_sessions(
      self, max_age_seconds: Optional[float] = None
  ) -> int:
    """Evict sessions inactive for longer than max_age_seconds.

    Args:
        max_age_seconds: Maximum age in seconds. Uses config.session_inactive_timeout if None.

    Returns:
        Number of sessions evicted
    """
    if max_age_seconds is None:
      max_age_seconds = self.config.session_inactive_timeout

    # Don't evict if timeout is 0 (disabled)
    if max_age_seconds == 0:
      return 0

    current_time = time.time()
    evicted_count = 0

    # Find inactive sessions
    sessions_to_evict = []
    for session_id in list(self._active_sessions):
      if session_id in self._session_metrics:
        metrics = self._session_metrics[session_id]
        inactive_duration = current_time - metrics.last_activity_at
        if inactive_duration > max_age_seconds:
          sessions_to_evict.append(session_id)

    # Evict them
    for session_id in sessions_to_evict:
      self.unregister_session(session_id)
      evicted_count += 1

    return evicted_count

  def evict_oldest_sessions(self, keep_count: int) -> int:
    """Evict oldest sessions to maintain max_sessions limit.

    Uses LRU (least recently used) eviction based on last_activity_at.

    Args:
        keep_count: Maximum sessions to keep

    Returns:
        Number of sessions evicted
    """
    if keep_count <= 0:
      return 0

    session_count = len(self._active_sessions)
    if session_count <= keep_count:
      return 0

    evict_count = session_count - keep_count

    # Sort by last activity (oldest first)
    session_activity = []
    for session_id in self._active_sessions:
      if session_id in self._session_metrics:
        last_activity = self._session_metrics[session_id].last_activity_at
      else:
        last_activity = 0.0  # Sessions without metrics evicted first
      session_activity.append((last_activity, session_id))

    session_activity.sort()  # Oldest first

    # Evict oldest sessions
    evicted_count = 0
    for _, session_id in session_activity[:evict_count]:
      self.unregister_session(session_id)
      evicted_count += 1

    return evicted_count

  def get_active_session_count(self) -> int:
    """Get count of currently active sessions.

    Returns:
        Number of active sessions
    """
    return len(self._active_sessions)

  def list_active_sessions(self) -> List[str]:
    """List all active session IDs.

    Returns:
        List of session IDs sorted by last activity (most recent first)
    """
    # Sort by last activity (most recent first)
    session_activity = []
    for session_id in self._active_sessions:
      if session_id in self._session_metrics:
        last_activity = self._session_metrics[session_id].last_activity_at
      else:
        last_activity = 0.0
      session_activity.append((last_activity, session_id))

    session_activity.sort(reverse=True)  # Most recent first
    return [session_id for _, session_id in session_activity]
