"""Comprehensive tests for InterruptService (95%+ coverage target).

Tests cover:
- Configuration validation
- Session registration/cleanup
- Pause/resume operations
- Message queuing with bounds and timeout
- Cancellation support
- Queue status API
- Message pagination
- Metrics tracking
- Error handling and edge cases
"""

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.graph.interrupt_service import InterruptMessage
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.agents.graph.interrupt_service import InterruptServiceConfig
from google.adk.agents.graph.interrupt_service import QueueStatus
from google.adk.agents.graph.interrupt_service import SessionMetrics
import pytest


class TestInterruptServiceConfig:
  """Test InterruptServiceConfig validation."""

  def test_default_config(self):
    """Test default configuration values."""
    config = InterruptServiceConfig()
    assert config.max_queue_size == 100
    assert config.default_timeout == 300.0
    assert config.enable_metrics is True

  def test_custom_config(self):
    """Test custom configuration values."""
    config = InterruptServiceConfig(
        max_queue_size=50, default_timeout=60.0, enable_metrics=False
    )
    assert config.max_queue_size == 50
    assert config.default_timeout == 60.0
    assert config.enable_metrics is False

  def test_invalid_max_queue_size(self):
    """Test validation rejects invalid max_queue_size."""
    with pytest.raises(ValueError, match="max_queue_size must be positive"):
      InterruptServiceConfig(max_queue_size=0)

    with pytest.raises(ValueError, match="max_queue_size must be positive"):
      InterruptServiceConfig(max_queue_size=-1)

  def test_invalid_default_timeout(self):
    """Test validation rejects invalid default_timeout."""
    with pytest.raises(ValueError, match="default_timeout must be positive"):
      InterruptServiceConfig(default_timeout=0)

    with pytest.raises(ValueError, match="default_timeout must be positive"):
      InterruptServiceConfig(default_timeout=-10.0)


class TestInterruptMessage:
  """Test InterruptMessage data model."""

  def test_message_creation(self):
    """Test creating interrupt message."""
    msg = InterruptMessage(
        text="Update parameter",
        metadata={"key": "value"},
        action="update_state",
    )
    assert msg.text == "Update parameter"
    assert msg.metadata == {"key": "value"}
    assert msg.action == "update_state"

  def test_message_defaults(self):
    """Test message with default values."""
    msg = InterruptMessage(text="Simple message")
    assert msg.text == "Simple message"
    assert msg.metadata == {}
    assert msg.action is None


class TestSessionRegistration:
  """Test session registration and cleanup."""

  def test_register_session(self):
    """Test session registration creates resources."""
    service = InterruptService()
    service.register_session("session-1")

    assert "session-1" in service._message_queues
    assert "session-1" in service._pause_events
    assert "session-1" in service._cancellation_events
    assert "session-1" in service._active_sessions
    assert service._pause_events["session-1"].is_set()  # Not paused

  def test_register_session_with_metrics(self):
    """Test session registration creates metrics."""
    config = InterruptServiceConfig(enable_metrics=True)
    service = InterruptService(config)
    service.register_session("session-1")

    assert "session-1" in service._session_metrics
    metrics = service._session_metrics["session-1"]
    assert metrics.messages_queued == 0
    assert metrics.pause_count == 0
    assert metrics.resume_count == 0
    assert metrics.cancel_count == 0
    assert metrics.max_queue_depth == 0
    assert metrics.created_at > 0

  def test_register_session_without_metrics(self):
    """Test session registration without metrics."""
    config = InterruptServiceConfig(enable_metrics=False)
    service = InterruptService(config)
    service.register_session("session-1")

    assert "session-1" not in service._session_metrics

  def test_register_session_idempotent(self):
    """Test re-registering session doesn't create duplicates."""
    service = InterruptService()
    service.register_session("session-1")
    queue1 = service._message_queues["session-1"]

    service.register_session("session-1")  # Should be no-op
    queue2 = service._message_queues["session-1"]

    assert queue1 is queue2  # Same queue instance

  def test_unregister_session(self):
    """Test session cleanup removes all resources."""
    service = InterruptService()
    service.register_session("session-1")
    service.unregister_session("session-1")

    assert "session-1" not in service._message_queues
    assert "session-1" not in service._pause_events
    assert "session-1" not in service._cancellation_events
    assert "session-1" not in service._session_metrics
    assert "session-1" not in service._active_sessions

  def test_unregister_nonexistent_session(self):
    """Test unregistering non-existent session doesn't error."""
    service = InterruptService()
    service.unregister_session("nonexistent")  # Should not raise


class TestContextManager:
  """Test async context manager support."""

  @pytest.mark.asyncio
  async def test_context_manager_registers_and_unregisters(self):
    """Test context manager automatically registers/unregisters session."""
    service = InterruptService()

    async with service.session("session-1"):
      # Session should be registered inside context
      assert "session-1" in service._message_queues
      assert "session-1" in service._pause_events
      assert "session-1" in service._active_sessions

    # Session should be unregistered after context exit
    assert "session-1" not in service._message_queues
    assert "session-1" not in service._pause_events
    assert "session-1" not in service._active_sessions

  @pytest.mark.asyncio
  async def test_context_manager_returns_service(self):
    """Test context manager returns the SessionContext instance."""
    service = InterruptService()

    async with service.session("session-1") as ctx:
      assert isinstance(ctx, service.SessionContext)
      assert ctx._service is service
      assert ctx._session_id == "session-1"

  @pytest.mark.asyncio
  async def test_context_manager_with_operations(self):
    """Test context manager with actual interrupt operations."""
    service = InterruptService()

    async with service.session("session-1"):
      # Pause and send message
      await service.pause("session-1")
      await service.send_message("session-1", "Test message")

      # Check operations worked
      assert service.is_paused("session-1")
      status = service.get_queue_status("session-1")
      assert status.queue_depth == 1

    # Session cleaned up after exit
    assert "session-1" not in service._message_queues

  @pytest.mark.asyncio
  async def test_context_manager_exception_still_unregisters(self):
    """Test context manager unregisters session even on exception."""
    service = InterruptService()

    try:
      async with service.session("session-1"):
        assert "session-1" in service._message_queues
        raise ValueError("Test exception")
    except ValueError:
      pass

    # Session should be cleaned up despite exception
    assert "session-1" not in service._message_queues
    assert "session-1" not in service._pause_events

  @pytest.mark.asyncio
  async def test_context_manager_multiple_sessions(self):
    """Test multiple sessions with context managers."""
    service = InterruptService()

    async with service.session("session-1"):
      async with service.session("session-2"):
        # Both sessions should be registered
        assert "session-1" in service._message_queues
        assert "session-2" in service._message_queues

      # session-2 unregistered, session-1 still active
      assert "session-1" in service._message_queues
      assert "session-2" not in service._message_queues

    # Both unregistered
    assert "session-1" not in service._message_queues
    assert "session-2" not in service._message_queues

  @pytest.mark.asyncio
  async def test_context_manager_with_metrics(self):
    """Test context manager works with metrics enabled."""
    config = InterruptServiceConfig(enable_metrics=True)
    service = InterruptService(config)

    async with service.session("session-1"):
      # Metrics should be created
      assert "session-1" in service._session_metrics
      metrics = service._session_metrics["session-1"]
      assert metrics.created_at > 0

    # Metrics cleaned up after exit
    assert "session-1" not in service._session_metrics


class TestPauseResume:
  """Test pause and resume operations."""

  @pytest.mark.asyncio
  async def test_pause_session(self):
    """Test pausing session sets event."""
    service = InterruptService()
    service.register_session("session-1")

    await service.pause("session-1")
    assert not service._pause_events["session-1"].is_set()

  @pytest.mark.asyncio
  async def test_resume_session(self):
    """Test resuming session sets event."""
    service = InterruptService()
    service.register_session("session-1")
    await service.pause("session-1")

    await service.resume("session-1")
    assert service._pause_events["session-1"].is_set()

  @pytest.mark.asyncio
  async def test_pause_increments_metric(self):
    """Test pause increments pause counter."""
    service = InterruptService()
    service.register_session("session-1")

    await service.pause("session-1")
    assert service._session_metrics["session-1"].pause_count == 1

    await service.pause("session-1")
    assert service._session_metrics["session-1"].pause_count == 2

  @pytest.mark.asyncio
  async def test_resume_increments_metric(self):
    """Test resume increments resume counter."""
    service = InterruptService()
    service.register_session("session-1")

    await service.resume("session-1")
    assert service._session_metrics["session-1"].resume_count == 1

    await service.resume("session-1")
    assert service._session_metrics["session-1"].resume_count == 2

  def test_is_paused(self):
    """Test is_paused returns correct state."""
    service = InterruptService()
    service.register_session("session-1")

    assert not service.is_paused("session-1")

    service._pause_events["session-1"].clear()
    assert service.is_paused("session-1")

    service._pause_events["session-1"].set()
    assert not service.is_paused("session-1")

  def test_is_paused_nonexistent_session(self):
    """Test is_paused returns False for nonexistent session."""
    service = InterruptService()
    assert not service.is_paused("nonexistent")


class TestMessageQueuing:
  """Test message queuing with bounds and timeout."""

  @pytest.mark.asyncio
  async def test_send_message_success(self):
    """Test sending message to queue."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message(
        "session-1", "Test message", {"key": "value"}, "update_state"
    )

    msg = await service.check_interrupt("session-1")
    assert msg is not None
    assert msg.text == "Test message"
    assert msg.metadata == {"key": "value"}
    assert msg.action == "update_state"

  @pytest.mark.asyncio
  async def test_send_message_increments_metrics(self):
    """Test send_message updates metrics."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message("session-1", "Message 1")
    metrics = service._session_metrics["session-1"]
    assert metrics.messages_queued == 1
    assert metrics.max_queue_depth == 1

    await service.send_message("session-1", "Message 2")
    metrics = service._session_metrics["session-1"]
    assert metrics.messages_queued == 2
    assert metrics.max_queue_depth == 2

  @pytest.mark.asyncio
  async def test_send_message_to_nonexistent_session(self):
    """Test sending message to nonexistent session is no-op."""
    service = InterruptService()
    await service.send_message("nonexistent", "Test")  # Should not raise

  @pytest.mark.asyncio
  async def test_send_message_queue_full_timeout(self):
    """Test send_message times out when queue is full."""
    config = InterruptServiceConfig(max_queue_size=2, default_timeout=0.1)
    service = InterruptService(config)
    service.register_session("session-1")

    # Fill queue
    await service.send_message("session-1", "Msg 1")
    await service.send_message("session-1", "Msg 2")

    # Queue full, should timeout
    with pytest.raises(asyncio.TimeoutError, match="Queue is full"):
      await service.send_message("session-1", "Msg 3", timeout=0.1)

  @pytest.mark.asyncio
  async def test_check_interrupt_empty_queue(self):
    """Test check_interrupt returns None when queue is empty."""
    service = InterruptService()
    service.register_session("session-1")

    msg = await service.check_interrupt("session-1")
    assert msg is None

  @pytest.mark.asyncio
  async def test_check_interrupt_nonexistent_session(self):
    """Test check_interrupt returns None for nonexistent session."""
    service = InterruptService()
    msg = await service.check_interrupt("nonexistent")
    assert msg is None

  @pytest.mark.asyncio
  async def test_multiple_messages_fifo(self):
    """Test messages are processed in FIFO order."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message("session-1", "First")
    await service.send_message("session-1", "Second")
    await service.send_message("session-1", "Third")

    msg1 = await service.check_interrupt("session-1")
    msg2 = await service.check_interrupt("session-1")
    msg3 = await service.check_interrupt("session-1")

    assert msg1.text == "First"
    assert msg2.text == "Second"
    assert msg3.text == "Third"


class TestWaitIfPaused:
  """Test wait_if_paused with timeout and cancellation."""

  @pytest.mark.asyncio
  async def test_wait_if_paused_not_paused(self):
    """Test wait_if_paused returns immediately if not paused."""
    service = InterruptService()
    service.register_session("session-1")

    resumed = await service.wait_if_paused("session-1")
    assert resumed is True

  @pytest.mark.asyncio
  async def test_wait_if_paused_nonexistent_session(self):
    """Test wait_if_paused returns True for nonexistent session."""
    service = InterruptService()
    resumed = await service.wait_if_paused("nonexistent")
    assert resumed is True

  @pytest.mark.asyncio
  async def test_wait_if_paused_resume(self):
    """Test wait_if_paused blocks until resume."""
    service = InterruptService()
    service.register_session("session-1")
    await service.pause("session-1")

    async def resume_after_delay():
      await asyncio.sleep(0.1)
      await service.resume("session-1")

    resume_task = asyncio.create_task(resume_after_delay())

    resumed = await service.wait_if_paused("session-1")
    assert resumed is True

    await resume_task

  @pytest.mark.asyncio
  async def test_wait_if_paused_cancel(self):
    """Test wait_if_paused returns False on cancellation."""
    service = InterruptService()
    service.register_session("session-1")
    await service.pause("session-1")

    async def cancel_after_delay():
      await asyncio.sleep(0.1)
      await service.cancel("session-1")

    cancel_task = asyncio.create_task(cancel_after_delay())

    resumed = await service.wait_if_paused("session-1")
    assert resumed is False

    await cancel_task

  @pytest.mark.asyncio
  async def test_wait_if_paused_timeout(self):
    """Test wait_if_paused raises TimeoutError."""
    service = InterruptService()
    service.register_session("session-1")
    await service.pause("session-1")

    with pytest.raises(TimeoutError, match="Timed out waiting for resume"):
      await service.wait_if_paused("session-1", timeout=0.1)

  @pytest.mark.asyncio
  async def test_wait_if_paused_uses_default_timeout(self):
    """Test wait_if_paused uses config timeout if not specified."""
    config = InterruptServiceConfig(default_timeout=0.1)
    service = InterruptService(config)
    service.register_session("session-1")
    await service.pause("session-1")

    with pytest.raises(TimeoutError):
      await service.wait_if_paused("session-1")  # Should use 0.1s timeout


class TestCancellation:
  """Test cancellation support."""

  @pytest.mark.asyncio
  async def test_cancel_sets_event(self):
    """Test cancel sets cancellation event."""
    service = InterruptService()
    service.register_session("session-1")

    await service.cancel("session-1")
    assert service._cancellation_events["session-1"].is_set()

  @pytest.mark.asyncio
  async def test_cancel_clears_queue(self):
    """Test cancel clears message queue."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message("session-1", "Msg 1")
    await service.send_message("session-1", "Msg 2")
    await service.send_message("session-1", "Msg 3")

    await service.cancel("session-1")

    msg = await service.check_interrupt("session-1")
    assert msg is None  # Queue should be empty

  @pytest.mark.asyncio
  async def test_cancel_increments_metric(self):
    """Test cancel increments cancel counter."""
    service = InterruptService()
    service.register_session("session-1")

    await service.cancel("session-1")
    assert service._session_metrics["session-1"].cancel_count == 1

  @pytest.mark.asyncio
  async def test_cancel_nonexistent_session(self):
    """Test cancelling nonexistent session is no-op."""
    service = InterruptService()
    await service.cancel("nonexistent")  # Should not raise


class TestQueueStatus:
  """Test queue status API."""

  def test_get_queue_status_success(self):
    """Test getting queue status."""
    service = InterruptService()
    service.register_session("session-1")

    status = service.get_queue_status("session-1")
    assert status is not None
    assert status.session_id == "session-1"
    assert status.is_paused is False
    assert status.is_cancelled is False
    assert status.queue_depth == 0
    assert status.max_queue_size == 100
    assert status.metrics is not None

  def test_get_queue_status_nonexistent_session(self):
    """Test getting status for nonexistent session returns None."""
    service = InterruptService()
    status = service.get_queue_status("nonexistent")
    assert status is None

  @pytest.mark.asyncio
  async def test_get_queue_status_reflects_state(self):
    """Test queue status reflects current state."""
    service = InterruptService()
    service.register_session("session-1")

    # Add messages
    await service.send_message("session-1", "Msg 1")
    await service.send_message("session-1", "Msg 2")

    # Pause
    await service.pause("session-1")

    status = service.get_queue_status("session-1")
    assert status.is_paused is True
    assert status.queue_depth == 2

  @pytest.mark.asyncio
  async def test_get_queue_status_with_cancellation(self):
    """Test queue status shows cancellation."""
    service = InterruptService()
    service.register_session("session-1")
    await service.cancel("session-1")

    status = service.get_queue_status("session-1")
    assert status.is_cancelled is True

  def test_get_queue_status_without_metrics(self):
    """Test queue status without metrics enabled."""
    config = InterruptServiceConfig(enable_metrics=False)
    service = InterruptService(config)
    service.register_session("session-1")

    status = service.get_queue_status("session-1")
    assert status.metrics is None


class TestMessagePagination:
  """Test message pagination."""

  @pytest.mark.asyncio
  async def test_list_queued_messages_empty(self):
    """Test listing messages from empty queue."""
    service = InterruptService()
    service.register_session("session-1")

    messages = service.list_queued_messages("session-1")
    assert messages == []

  @pytest.mark.asyncio
  async def test_list_queued_messages_all(self):
    """Test listing all queued messages."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message("session-1", "Msg 1")
    await service.send_message("session-1", "Msg 2")
    await service.send_message("session-1", "Msg 3")

    messages = service.list_queued_messages("session-1")
    assert len(messages) == 3
    assert messages[0].text == "Msg 1"
    assert messages[1].text == "Msg 2"
    assert messages[2].text == "Msg 3"

  @pytest.mark.asyncio
  async def test_list_queued_messages_pagination(self):
    """Test message pagination."""
    service = InterruptService()
    service.register_session("session-1")

    for i in range(10):
      await service.send_message("session-1", f"Msg {i}")

    # Page 1
    page1 = service.list_queued_messages("session-1", page=1, page_size=3)
    assert len(page1) == 3
    assert page1[0].text == "Msg 0"
    assert page1[2].text == "Msg 2"

    # Page 2
    page2 = service.list_queued_messages("session-1", page=2, page_size=3)
    assert len(page2) == 3
    assert page2[0].text == "Msg 3"
    assert page2[2].text == "Msg 5"

  def test_list_queued_messages_invalid_page(self):
    """Test pagination bounds validation."""
    service = InterruptService()
    service.register_session("session-1")

    # Page < 1 defaults to 1
    messages = service.list_queued_messages("session-1", page=0)
    assert messages == []

    messages = service.list_queued_messages("session-1", page=-1)
    assert messages == []

  def test_list_queued_messages_invalid_page_size(self):
    """Test page_size bounds validation."""
    service = InterruptService()
    service.register_session("session-1")

    # page_size < 1 or > 1000 defaults to 50
    messages = service.list_queued_messages("session-1", page_size=0)
    assert messages == []

    messages = service.list_queued_messages("session-1", page_size=2000)
    assert messages == []

  def test_list_queued_messages_nonexistent_session(self):
    """Test listing messages for nonexistent session."""
    service = InterruptService()
    messages = service.list_queued_messages("nonexistent")
    assert messages == []


class TestTelemetryIntegration:
  """Test telemetry integration."""

  @pytest.mark.asyncio
  async def test_pause_calls_telemetry(self):
    """Test pause calls telemetry functions."""
    service = InterruptService()
    service.register_session("session-1")

    with patch(
        "google.adk.agents.graph.interrupt_service.trace_interrupt_service_pause"
    ) as mock_trace:
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        await service.pause("session-1")

        mock_trace.assert_called_once_with("session-1")
        mock_metrics.assert_called_once()
        assert mock_metrics.call_args[0][0] == "pause"
        assert mock_metrics.call_args[0][2] == "success"

  @pytest.mark.asyncio
  async def test_resume_calls_telemetry(self):
    """Test resume calls telemetry functions."""
    service = InterruptService()
    service.register_session("session-1")

    with patch(
        "google.adk.agents.graph.interrupt_service.trace_interrupt_service_resume"
    ) as mock_trace:
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        await service.resume("session-1")

        mock_trace.assert_called_once_with("session-1")
        mock_metrics.assert_called_once()
        assert mock_metrics.call_args[0][0] == "resume"

  @pytest.mark.asyncio
  async def test_send_message_calls_telemetry(self):
    """Test send_message calls telemetry functions."""
    service = InterruptService()
    service.register_session("session-1")

    with patch(
        "google.adk.agents.graph.interrupt_service.trace_interrupt_service_message"
    ) as mock_trace:
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        await service.send_message("session-1", "Test", action="update")

        mock_trace.assert_called_once()
        assert mock_trace.call_args[0][0] == "session-1"
        assert mock_trace.call_args[0][1] == "Test"
        assert mock_trace.call_args[0][2] == "update"

        mock_metrics.assert_called_once()
        assert mock_metrics.call_args[0][0] == "message"

  @pytest.mark.asyncio
  async def test_cancel_calls_telemetry(self):
    """Test cancel calls telemetry functions."""
    service = InterruptService()
    service.register_session("session-1")

    with patch(
        "google.adk.agents.graph.interrupt_service.trace_interrupt_service_cancel"
    ) as mock_trace:
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        await service.cancel("session-1")

        mock_trace.assert_called_once()
        assert mock_trace.call_args[0][0] == "session-1"

        mock_metrics.assert_called_once()
        assert mock_metrics.call_args[0][0] == "cancel"

  @pytest.mark.asyncio
  async def test_queue_depth_gauge_updated(self):
    """Test queue depth gauge is updated on send_message and check_interrupt."""
    service = InterruptService()
    service.register_session("session-1")

    with patch(
        "google.adk.agents.graph.interrupt_service.update_interrupt_queue_depth"
    ) as mock_gauge:
      # Send message - gauge should be updated
      await service.send_message("session-1", "Test message")
      assert mock_gauge.call_count == 1
      assert mock_gauge.call_args[0][0] == "session-1"
      assert mock_gauge.call_args[0][1] == 1  # Queue depth is 1

      # Send another message
      await service.send_message("session-1", "Test message 2")
      assert mock_gauge.call_count == 2
      assert mock_gauge.call_args[0][1] == 2  # Queue depth is 2

      # Check interrupt - gauge should be updated after dequeue
      await service.check_interrupt("session-1")
      assert mock_gauge.call_count == 3
      assert mock_gauge.call_args[0][1] == 1  # Queue depth is 1 after dequeue


class TestErrorHandling:
  """Test error handling and edge cases."""

  @pytest.mark.asyncio
  async def test_pause_records_error_metric(self):
    """Test pause records error metric on exception."""
    service = InterruptService()
    service.register_session("session-1")

    # Mock pause_events to raise exception
    with patch.object(
        service._pause_events["session-1"],
        "clear",
        side_effect=RuntimeError("Test error"),
    ):
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        with pytest.raises(RuntimeError, match="Test error"):
          await service.pause("session-1")

        # Should record error metric
        assert any(
            call[0][0] == "pause" and call[0][2] == "error"
            for call in mock_metrics.call_args_list
        )

  @pytest.mark.asyncio
  async def test_send_message_records_timeout_metric(self):
    """Test send_message records timeout metric."""
    config = InterruptServiceConfig(max_queue_size=1)
    service = InterruptService(config)
    service.register_session("session-1")

    await service.send_message("session-1", "Fill queue")

    with patch(
        "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
    ) as mock_metrics:
      with pytest.raises(asyncio.TimeoutError):
        await service.send_message("session-1", "Timeout", timeout=0.01)

      # Should record timeout metric
      assert any(
          call[0][0] == "message" and call[0][2] == "timeout"
          for call in mock_metrics.call_args_list
      )

  @pytest.mark.asyncio
  async def test_concurrent_operations(self):
    """Test concurrent operations on same session."""
    service = InterruptService()
    service.register_session("session-1")

    # Send messages concurrently
    tasks = [service.send_message("session-1", f"Msg {i}") for i in range(10)]
    await asyncio.gather(*tasks)

    # All messages should be queued
    assert service._session_metrics["session-1"].messages_queued == 10

  @pytest.mark.asyncio
  async def test_resume_records_error_metric(self):
    """Test resume records error metric on exception."""
    service = InterruptService()
    service.register_session("session-1")

    # Mock pause_events to raise exception
    with patch.object(
        service._pause_events["session-1"],
        "set",
        side_effect=RuntimeError("Test error"),
    ):
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        with pytest.raises(RuntimeError, match="Test error"):
          await service.resume("session-1")

        # Should record error metric
        assert any(
            call[0][0] == "resume" and call[0][2] == "error"
            for call in mock_metrics.call_args_list
        )

  @pytest.mark.asyncio
  async def test_send_message_generic_exception(self):
    """Test send_message records error metric on non-timeout exception."""
    service = InterruptService()
    service.register_session("session-1")

    # Mock queue.put to raise generic exception
    with patch.object(
        service._message_queues["session-1"],
        "put",
        side_effect=RuntimeError("Test error"),
    ):
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        with pytest.raises(RuntimeError, match="Test error"):
          await service.send_message("session-1", "Test message")

        # Should record error metric (not timeout)
        assert any(
            call[0][0] == "message" and call[0][2] == "error"
            for call in mock_metrics.call_args_list
        )

  @pytest.mark.asyncio
  async def test_cancel_records_error_metric(self):
    """Test cancel records error metric on exception."""
    service = InterruptService()
    service.register_session("session-1")

    # Mock cancellation_events to raise exception
    with patch.object(
        service._cancellation_events["session-1"],
        "set",
        side_effect=RuntimeError("Test error"),
    ):
      with patch(
          "google.adk.agents.graph.interrupt_service.record_interrupt_service_metrics"
      ) as mock_metrics:
        with pytest.raises(RuntimeError, match="Test error"):
          await service.cancel("session-1")

        # Should record error metric
        assert any(
            call[0][0] == "cancel" and call[0][2] == "error"
            for call in mock_metrics.call_args_list
        )

  @pytest.mark.asyncio
  async def test_wait_if_paused_exception_cleanup(self):
    """Test wait_if_paused cleans up tasks on exception."""
    service = InterruptService()
    service.register_session("session-1")
    await service.pause("session-1")

    # Mock asyncio.wait to raise exception
    with patch("asyncio.wait", side_effect=RuntimeError("Test error")):
      with pytest.raises(RuntimeError, match="Test error"):
        await service.wait_if_paused("session-1")


class TestInputValidation:
  """Test input validation for security hardening."""

  def test_session_id_validation_invalid_format(self):
    """Test session ID validation rejects invalid formats."""
    service = InterruptService()

    # Test invalid characters
    with pytest.raises(ValueError, match="Invalid session_id format"):
      service.register_session("session@123")

    with pytest.raises(ValueError, match="Invalid session_id format"):
      service.register_session("session 123")

    with pytest.raises(ValueError, match="Invalid session_id format"):
      service.register_session("session/123")

  def test_session_id_validation_too_long(self):
    """Test session ID validation rejects too long IDs."""
    service = InterruptService()

    # Test too long (>256 chars)
    long_session_id = "a" * 257
    with pytest.raises(ValueError, match="exceeds maximum length"):
      service.register_session(long_session_id)

  def test_session_id_validation_empty(self):
    """Test session ID validation rejects empty IDs."""
    service = InterruptService()

    with pytest.raises(ValueError, match="must be a non-empty string"):
      service.register_session("")

  def test_session_id_validation_valid_formats(self):
    """Test session ID validation accepts valid formats."""
    service = InterruptService()

    # These should all work
    valid_ids = [
        "session-123",
        "session_123",
        "SESSION-123-ABC",
        "sess_123",
        "a",
        "1",
        "a1b2c3",
    ]

    for session_id in valid_ids:
      service.register_session(session_id)
      assert session_id in service._message_queues

  @pytest.mark.asyncio
  async def test_message_validation_text_too_long(self):
    """Test message validation rejects text that's too long."""
    config = InterruptServiceConfig(max_message_length=100)
    service = InterruptService(config)
    service.register_session("session-1")

    long_text = "a" * 101
    with pytest.raises(ValueError, match="exceeds maximum length"):
      await service.send_message("session-1", long_text)

  @pytest.mark.asyncio
  async def test_message_validation_empty_text(self):
    """Test message validation rejects empty text."""
    service = InterruptService()
    service.register_session("session-1")

    with pytest.raises(ValueError, match="must be a non-empty string"):
      await service.send_message("session-1", "")

  @pytest.mark.asyncio
  async def test_message_validation_metadata_too_large(self):
    """Test message validation rejects metadata that's too large."""
    config = InterruptServiceConfig(max_metadata_size=1000)
    service = InterruptService(config)
    service.register_session("session-1")

    # Create large metadata
    large_metadata = {"data": "x" * 10000}
    with pytest.raises(ValueError, match="exceeds maximum size"):
      await service.send_message("session-1", "Test", metadata=large_metadata)

  @pytest.mark.asyncio
  async def test_message_validation_action_not_in_allowlist(self):
    """Test message validation rejects actions not in allowlist."""
    config = InterruptServiceConfig(allowed_actions=["update_state", "cancel"])
    service = InterruptService(config)
    service.register_session("session-1")

    with pytest.raises(ValueError, match="is not allowed"):
      await service.send_message("session-1", "Test", action="invalid_action")

  @pytest.mark.asyncio
  async def test_message_validation_action_in_allowlist(self):
    """Test message validation accepts actions in allowlist."""
    config = InterruptServiceConfig(allowed_actions=["update_state", "cancel"])
    service = InterruptService(config)
    service.register_session("session-1")

    # Should work
    await service.send_message("session-1", "Test", action="update_state")
    await service.send_message("session-1", "Test", action="cancel")

    # Check messages were queued
    assert service._message_queues["session-1"].qsize() == 2

  @pytest.mark.asyncio
  async def test_message_validation_no_allowlist_allows_any_action(self):
    """Test message validation allows any action when allowlist is None."""
    service = InterruptService()  # No allowlist (None)
    service.register_session("session-1")

    # Should all work
    await service.send_message("session-1", "Test", action="anything")
    await service.send_message("session-1", "Test", action="custom_action")

    assert service._message_queues["session-1"].qsize() == 2


class TestConvenienceMethods:
  """Test convenience methods for common operations."""

  @pytest.mark.asyncio
  async def test_pause_and_send(self):
    """Test pause_and_send combines pause and send operations."""
    service = InterruptService()
    service.register_session("session-1")

    await service.pause_and_send("session-1", "Test message")

    # Check session is paused
    assert service.is_paused("session-1")

    # Check message was sent
    assert service._message_queues["session-1"].qsize() == 1
    message = await service.check_interrupt("session-1")
    assert message.text == "Test message"

  @pytest.mark.asyncio
  async def test_pause_and_send_with_metadata(self):
    """Test pause_and_send with metadata and action."""
    service = InterruptService()
    service.register_session("session-1")

    await service.pause_and_send(
        "session-1",
        "Update state",
        metadata={"key": "value"},
        action="update_state",
    )

    message = await service.check_interrupt("session-1")
    assert message.text == "Update state"
    assert message.metadata == {"key": "value"}
    assert message.action == "update_state"

  def test_is_active_registered_session(self):
    """Test is_active returns True for registered session."""
    service = InterruptService()
    service.register_session("session-1")

    assert service.is_active("session-1") is True

  def test_is_active_unregistered_session(self):
    """Test is_active returns False for unregistered session."""
    service = InterruptService()

    assert service.is_active("nonexistent") is False

  @pytest.mark.asyncio
  async def test_is_active_cancelled_session(self):
    """Test is_active returns False for cancelled session."""
    service = InterruptService()
    service.register_session("session-1")

    await service.cancel("session-1")

    assert service.is_active("session-1") is False

  @pytest.mark.asyncio
  async def test_clear_queue(self):
    """Test clear_queue removes all messages."""
    service = InterruptService()
    service.register_session("session-1")

    # Send 3 messages
    await service.send_message("session-1", "Message 1")
    await service.send_message("session-1", "Message 2")
    await service.send_message("session-1", "Message 3")
    assert service._message_queues["session-1"].qsize() == 3

    # Clear queue
    cleared = service.clear_queue("session-1")

    assert cleared == 3
    assert service._message_queues["session-1"].qsize() == 0

  def test_clear_queue_nonexistent_session(self):
    """Test clear_queue returns 0 for nonexistent session."""
    service = InterruptService()

    cleared = service.clear_queue("nonexistent")

    assert cleared == 0

  @pytest.mark.asyncio
  async def test_clear_queue_updates_gauge(self):
    """Test clear_queue updates queue depth gauge."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message("session-1", "Message 1")
    await service.send_message("session-1", "Message 2")

    with patch(
        "google.adk.agents.graph.interrupt_service.update_interrupt_queue_depth"
    ) as mock_gauge:
      service.clear_queue("session-1")

      # Should update gauge to 0
      mock_gauge.assert_called_once_with("session-1", 0)

  @pytest.mark.asyncio
  async def test_has_queued_messages_true(self):
    """Test has_queued_messages returns True when messages exist."""
    service = InterruptService()
    service.register_session("session-1")

    await service.send_message("session-1", "Test")

    assert service.has_queued_messages("session-1") is True

  def test_has_queued_messages_false(self):
    """Test has_queued_messages returns False when queue is empty."""
    service = InterruptService()
    service.register_session("session-1")

    assert service.has_queued_messages("session-1") is False

  def test_has_queued_messages_nonexistent_session(self):
    """Test has_queued_messages returns False for nonexistent session."""
    service = InterruptService()

    assert service.has_queued_messages("nonexistent") is False

  @pytest.mark.asyncio
  async def test_get_queue_depth(self):
    """Test get_queue_depth returns correct depth."""
    service = InterruptService()
    service.register_session("session-1")

    assert service.get_queue_depth("session-1") == 0

    await service.send_message("session-1", "Message 1")
    assert service.get_queue_depth("session-1") == 1

    await service.send_message("session-1", "Message 2")
    assert service.get_queue_depth("session-1") == 2

    await service.check_interrupt("session-1")
    assert service.get_queue_depth("session-1") == 1

  def test_get_queue_depth_nonexistent_session(self):
    """Test get_queue_depth returns 0 for nonexistent session."""
    service = InterruptService()

    assert service.get_queue_depth("nonexistent") == 0


class TestSessionEviction:
  """Test session eviction for scale management."""

  @pytest.mark.asyncio
  async def test_evict_inactive_sessions(self):
    """Test evicting sessions based on inactivity timeout."""
    config = InterruptServiceConfig(session_inactive_timeout=1.0)  # 1 second
    service = InterruptService(config)

    # Register 3 sessions
    service.register_session("session-1")
    service.register_session("session-2")
    service.register_session("session-3")

    # Wait for session-1 and session-3 to become inactive
    await asyncio.sleep(1.1)

    # Update activity for session-2 (keep it active)
    await service.pause("session-2")

    # Evict inactive sessions
    evicted = service.evict_inactive_sessions()

    # session-2 was active recently, should not be evicted
    assert "session-2" in service._active_sessions
    # session-1 and session-3 were inactive
    assert "session-1" not in service._active_sessions
    assert "session-3" not in service._active_sessions
    assert evicted == 2

  def test_evict_inactive_sessions_disabled(self):
    """Test eviction disabled when timeout is 0."""
    config = InterruptServiceConfig(session_inactive_timeout=0.0)
    service = InterruptService(config)

    service.register_session("session-1")

    evicted = service.evict_inactive_sessions()

    # Should not evict when timeout is 0
    assert evicted == 0
    assert "session-1" in service._active_sessions

  def test_evict_oldest_sessions_lru(self):
    """Test LRU eviction based on last activity."""
    service = InterruptService()

    # Register 5 sessions
    for i in range(1, 6):
      service.register_session(f"session-{i}")

    # Evict to keep only 3 sessions (oldest 2 evicted)
    evicted = service.evict_oldest_sessions(keep_count=3)

    assert evicted == 2
    assert len(service._active_sessions) == 3

  @pytest.mark.asyncio
  async def test_evict_oldest_respects_activity(self):
    """Test LRU eviction respects last activity time."""
    service = InterruptService()

    # Register 3 sessions at different times
    service.register_session("session-1")
    await asyncio.sleep(0.01)
    service.register_session("session-2")
    await asyncio.sleep(0.01)
    service.register_session("session-3")

    # Update activity on session-1 (making it newest)
    await service.pause("session-1")

    # Evict to keep only 1 session
    evicted = service.evict_oldest_sessions(keep_count=1)

    # session-1 was most recently active, should be kept
    assert "session-1" in service._active_sessions
    # session-2 and session-3 should be evicted
    assert "session-2" not in service._active_sessions
    assert "session-3" not in service._active_sessions
    assert evicted == 2

  def test_max_sessions_auto_eviction(self):
    """Test automatic eviction when max_sessions is reached."""
    config = InterruptServiceConfig(max_sessions=3)
    service = InterruptService(config)

    # Register 3 sessions (at limit)
    service.register_session("session-1")
    service.register_session("session-2")
    service.register_session("session-3")
    assert len(service._active_sessions) == 3

    # Register 4th session - should trigger LRU eviction
    service.register_session("session-4")

    # Should still have 3 sessions
    assert len(service._active_sessions) == 3
    # Oldest (session-1) should be evicted
    assert "session-1" not in service._active_sessions
    # Newest should remain
    assert "session-4" in service._active_sessions

  def test_get_active_session_count(self):
    """Test getting active session count."""
    service = InterruptService()

    assert service.get_active_session_count() == 0

    service.register_session("session-1")
    assert service.get_active_session_count() == 1

    service.register_session("session-2")
    assert service.get_active_session_count() == 2

    service.unregister_session("session-1")
    assert service.get_active_session_count() == 1

  @pytest.mark.asyncio
  async def test_list_active_sessions_ordered(self):
    """Test listing active sessions ordered by activity."""
    service = InterruptService()

    # Register sessions
    service.register_session("session-1")
    await asyncio.sleep(0.01)
    service.register_session("session-2")
    await asyncio.sleep(0.01)
    service.register_session("session-3")

    # Update activity on session-1 (make it newest)
    await service.pause("session-1")

    # List should be ordered by last activity (most recent first)
    sessions = service.list_active_sessions()

    # session-1 should be first (most recent)
    assert sessions[0] == "session-1"
    # session-3 should be second
    assert sessions[1] == "session-3"
    # session-2 should be last
    assert sessions[2] == "session-2"

  @pytest.mark.asyncio
  async def test_activity_tracking_on_operations(self):
    """Test activity timestamp updated on all operations."""
    service = InterruptService()
    service.register_session("session-1")

    initial_activity = service._session_metrics["session-1"].last_activity_at

    # Wait a bit
    await asyncio.sleep(0.01)

    # Pause should update activity
    await service.pause("session-1")
    assert (
        service._session_metrics["session-1"].last_activity_at
        > initial_activity
    )

    pause_activity = service._session_metrics["session-1"].last_activity_at
    await asyncio.sleep(0.01)

    # Resume should update activity
    await service.resume("session-1")
    assert (
        service._session_metrics["session-1"].last_activity_at > pause_activity
    )

    resume_activity = service._session_metrics["session-1"].last_activity_at
    await asyncio.sleep(0.01)

    # Send message should update activity
    await service.send_message("session-1", "Test")
    assert (
        service._session_metrics["session-1"].last_activity_at > resume_activity
    )

    send_activity = service._session_metrics["session-1"].last_activity_at
    await asyncio.sleep(0.01)

    # Check interrupt should update activity
    await service.check_interrupt("session-1")
    assert (
        service._session_metrics["session-1"].last_activity_at > send_activity
    )

  def test_evict_oldest_sessions_no_eviction_needed(self):
    """Test evict_oldest_sessions when no eviction needed."""
    service = InterruptService()

    service.register_session("session-1")
    service.register_session("session-2")

    # Keep 5 sessions, but we only have 2
    evicted = service.evict_oldest_sessions(keep_count=5)

    assert evicted == 0
    assert len(service._active_sessions) == 2


class TestInterruptServiceConfigValidation:
  """__post_init__ raises on invalid config values."""

  def test_max_queue_size_must_be_positive(self):
    with pytest.raises(ValueError, match="max_queue_size must be positive"):
      InterruptServiceConfig(max_queue_size=0)

  def test_default_timeout_must_be_positive(self):
    with pytest.raises(ValueError, match="default_timeout must be positive"):
      InterruptServiceConfig(default_timeout=0.0)

  def test_max_message_length_must_be_positive(self):
    with pytest.raises(ValueError, match="max_message_length must be positive"):
      InterruptServiceConfig(max_message_length=0)

  def test_max_metadata_size_must_be_positive(self):
    with pytest.raises(ValueError, match="max_metadata_size must be positive"):
      InterruptServiceConfig(max_metadata_size=0)

  def test_session_inactive_timeout_must_be_non_negative(self):
    with pytest.raises(
        ValueError, match="session_inactive_timeout must be non-negative"
    ):
      InterruptServiceConfig(session_inactive_timeout=-1.0)

  def test_max_sessions_must_be_non_negative(self):
    with pytest.raises(ValueError, match="max_sessions must be non-negative"):
      InterruptServiceConfig(max_sessions=-1)


class TestIsActiveReturnTrue:
  """is_active returns True when session has no cancellation event."""

  def test_is_active_true_without_cancellation_event(self):
    """Defensive code path: session in _active_sessions but not in
    _cancellation_events (register_session always creates both, so this
    requires direct state manipulation)."""
    service = InterruptService()
    session_id = "active-only-session"

    service._active_sessions.add(session_id)

    assert session_id not in service._cancellation_events
    assert service.is_active(session_id) is True


class TestClearQueueQueueEmptyHandler:
  """QueueEmpty exception handler in clear_queue."""

  @pytest.mark.asyncio
  async def test_clear_queue_handles_queue_empty_race(self):
    """queue.empty() returns False but get_nowait() raises QueueEmpty —
    simulates a TOCTOU race condition; the except branch must swallow it."""
    from unittest.mock import MagicMock

    service = InterruptService()
    session_id = "clear-queue-test"
    service.register_session(session_id)

    mock_queue = MagicMock(spec=asyncio.Queue)
    mock_queue.empty.side_effect = [False, True]
    mock_queue.get_nowait.side_effect = asyncio.QueueEmpty()
    service._message_queues[session_id] = mock_queue

    cleared = service.clear_queue(session_id)
    assert cleared == 0


class TestEvictOldestSessionsEdgeCases:
  """evict_oldest_sessions edge cases: early-return and metrics-less sorting."""

  def test_evict_no_op_when_count_lte_keep_count(self):
    service = InterruptService()
    service.register_session("s1")
    service.register_session("s2")

    evicted = service.evict_oldest_sessions(keep_count=5)
    assert evicted == 0

  def test_evict_sessions_without_metrics_evicted_first(self):
    """Sessions without metrics get last_activity=0.0 (treated as oldest)."""
    config = InterruptServiceConfig(enable_metrics=False)
    service = InterruptService(config)
    service.register_session("no_metrics")
    service.register_session("also_no_metrics")

    evicted = service.evict_oldest_sessions(keep_count=1)
    assert evicted == 1


class TestGetSessionsByActivityWithoutMetrics:
  """list_active_sessions uses last_activity=0.0 when metrics disabled."""

  def test_get_sessions_by_activity_without_metrics(self):
    config = InterruptServiceConfig(enable_metrics=False)
    service = InterruptService(config)
    service.register_session("session_a")
    service.register_session("session_b")

    sessions = service.list_active_sessions()
    assert len(sessions) == 2
    assert "session_a" in sessions
    assert "session_b" in sessions


class TestCancelSessionQueueEmpty:
  """asyncio.QueueEmpty raised inside cancel queue-drain loop."""

  @pytest.mark.asyncio
  async def test_cancel_session_handles_queue_empty_race(self):
    """queue.empty() returns False but get_nowait raises QueueEmpty —
    the except branch must swallow it and break."""
    from unittest.mock import MagicMock

    service = InterruptService()
    session_id = "queue-empty-race"
    service.register_session(session_id)

    mock_queue = MagicMock(spec=asyncio.Queue)
    mock_queue.empty.return_value = False
    mock_queue.get_nowait.side_effect = asyncio.QueueEmpty()
    service._message_queues[session_id] = mock_queue

    await service.cancel(session_id)

    assert service._cancellation_events[session_id].is_set()
    mock_queue.get_nowait.assert_called_once()


class TestEvictOldestSessionsKeepCountNonPositive:
  """evict_oldest_sessions returns 0 immediately when keep_count <= 0."""

  def test_keep_count_zero_returns_zero_immediately(self):
    service = InterruptService()
    service.register_session("s1")
    service.register_session("s2")

    result = service.evict_oldest_sessions(keep_count=0)

    assert result == 0
    assert len(service._active_sessions) == 2

  def test_keep_count_negative_returns_zero_immediately(self):
    service = InterruptService()
    service.register_session("s1")

    result = service.evict_oldest_sessions(keep_count=-5)

    assert result == 0
    assert "s1" in service._active_sessions


class TestListQueuedMessagesDequeAccess:
  """Test direct deque access in list_queued_messages."""

  @pytest.mark.asyncio
  async def test_deque_access_non_destructive(self):
    """Messages remain in queue after listing (non-destructive read)."""
    service = InterruptService()
    session_id = "deque-non-destructive"
    service.register_session(session_id)

    await service.send_message(session_id, "Msg 1")
    await service.send_message(session_id, "Msg 2")

    # First listing
    messages1 = service.list_queued_messages(session_id)
    assert len(messages1) == 2
    assert messages1[0].text == "Msg 1"

    # Second listing — messages still there
    messages2 = service.list_queued_messages(session_id)
    assert len(messages2) == 2
    assert messages2[1].text == "Msg 2"

  @pytest.mark.asyncio
  async def test_deque_access_no_queue_full_risk(self):
    """Works at full capacity without QueueFull errors."""
    config = InterruptServiceConfig(max_queue_size=3)
    service = InterruptService(config=config)
    session_id = "deque-full-capacity"
    service.register_session(session_id)

    await service.send_message(session_id, "Msg 1")
    await service.send_message(session_id, "Msg 2")
    await service.send_message(session_id, "Msg 3")

    # Should work without any QueueFull issues
    messages = service.list_queued_messages(session_id)
    assert len(messages) == 3
    assert messages[0].text == "Msg 1"
    assert messages[2].text == "Msg 3"
