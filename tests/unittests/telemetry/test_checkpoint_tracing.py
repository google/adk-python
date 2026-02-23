"""Tests for checkpoint_tracing.py instrumentation functions."""

from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.telemetry import checkpoint_tracing
import pytest


@pytest.fixture
def mock_session():
  session = MagicMock()
  session.id = "session-abc"
  return session


class TestTraceInterruptCreate:
  """Tests for trace_interrupt_create."""

  def test_with_message_sets_span_message_attribute(self, mock_session):
    """trace_interrupt_create sets INTERRUPT_MESSAGE span attribute when message provided."""
    mock_span = MagicMock()
    with patch(
        "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
        return_value=mock_span,
    ):
      checkpoint_tracing.trace_interrupt_create(
          interrupt_id="int-1",
          interrupt_type="BEFORE",
          session=mock_session,
          agent_name="my_agent",
          message="Please review this",
      )

    calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
    assert calls[checkpoint_tracing.INTERRUPT_MESSAGE] == "Please review this"
    assert calls[checkpoint_tracing.INTERRUPT_OPERATION] == "create"
    assert calls[checkpoint_tracing.INTERRUPT_ID] == "int-1"

  def test_without_message_skips_span_message_attribute(self, mock_session):
    """trace_interrupt_create skips INTERRUPT_MESSAGE span attribute when message is None."""
    mock_span = MagicMock()
    with patch(
        "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
        return_value=mock_span,
    ):
      checkpoint_tracing.trace_interrupt_create(
          interrupt_id="int-2",
          interrupt_type="AFTER",
          session=mock_session,
          agent_name="my_agent",
          message=None,
      )

    attribute_keys = [c[0][0] for c in mock_span.set_attribute.call_args_list]
    assert checkpoint_tracing.INTERRUPT_MESSAGE not in attribute_keys

  def test_logs_interrupt_creation(self, mock_session):
    """trace_interrupt_create emits a structured log entry."""
    mock_span = MagicMock()
    with (
        patch(
            "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
            return_value=mock_span,
        ),
        patch.object(checkpoint_tracing.logger, "info") as mock_log,
    ):
      checkpoint_tracing.trace_interrupt_create(
          interrupt_id="int-3",
          interrupt_type="DYNAMIC",
          session=mock_session,
          agent_name="agent_x",
          message="Check this",
      )

    mock_log.assert_called_once()
    log_msg = mock_log.call_args[0][0]
    assert "Interrupt created" in log_msg


class TestTraceInterruptResolve:
  """Tests for trace_interrupt_resolve."""

  def test_sets_span_attributes(self, mock_session):
    """trace_interrupt_resolve sets required span attributes."""
    mock_span = MagicMock()
    with patch(
        "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
        return_value=mock_span,
    ):
      checkpoint_tracing.trace_interrupt_resolve(
          interrupt_id="int-1",
          session=mock_session,
          resolution="APPROVED",
          response_data={"decision": "yes"},
      )

    calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
    assert calls[checkpoint_tracing.INTERRUPT_OPERATION] == "resolve"
    assert calls[checkpoint_tracing.INTERRUPT_ID] == "int-1"
    assert calls[checkpoint_tracing.INTERRUPT_STATUS] == "approved"
    assert calls[checkpoint_tracing.INTERRUPT_SESSION_ID] == "session-abc"

  def test_logs_with_has_response_data_true(self, mock_session):
    """trace_interrupt_resolve logs has_response_data=True when data provided."""
    mock_span = MagicMock()
    with (
        patch(
            "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
            return_value=mock_span,
        ),
        patch.object(checkpoint_tracing.logger, "info") as mock_log,
    ):
      checkpoint_tracing.trace_interrupt_resolve(
          interrupt_id="int-1",
          session=mock_session,
          resolution="REJECTED",
          response_data={"reason": "not needed"},
      )

    mock_log.assert_called_once()
    extra = mock_log.call_args[1]["extra"]
    assert extra["has_response_data"] is True

  def test_logs_with_has_response_data_false(self, mock_session):
    """trace_interrupt_resolve logs has_response_data=False when no data."""
    mock_span = MagicMock()
    with (
        patch(
            "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
            return_value=mock_span,
        ),
        patch.object(checkpoint_tracing.logger, "info") as mock_log,
    ):
      checkpoint_tracing.trace_interrupt_resolve(
          interrupt_id="int-1",
          session=mock_session,
          resolution="APPROVED",
          response_data=None,
      )

    extra = mock_log.call_args[1]["extra"]
    assert extra["has_response_data"] is False


class TestTraceResumeWorkflow:
  """Tests for trace_resume_workflow."""

  def test_with_skipped_count_zero_skips_attribute(self, mock_session):
    """trace_resume_workflow does not set RESUME_SKIPPED_COUNT when count is 0."""
    mock_span = MagicMock()
    with patch(
        "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
        return_value=mock_span,
    ):
      checkpoint_tracing.trace_resume_workflow(
          checkpoint_id="ckpt-1",
          session=mock_session,
          skipped_count=0,
      )

    attribute_keys = [c[0][0] for c in mock_span.set_attribute.call_args_list]
    assert checkpoint_tracing.RESUME_SKIPPED_COUNT not in attribute_keys

  def test_with_skipped_count_nonzero_sets_attribute(self, mock_session):
    """trace_resume_workflow sets RESUME_SKIPPED_COUNT when count > 0."""
    mock_span = MagicMock()
    with patch(
        "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
        return_value=mock_span,
    ):
      checkpoint_tracing.trace_resume_workflow(
          checkpoint_id="ckpt-1",
          session=mock_session,
          skipped_count=3,
      )

    calls = {c[0][0]: c[0][1] for c in mock_span.set_attribute.call_args_list}
    assert calls[checkpoint_tracing.RESUME_SKIPPED_COUNT] == 3
    assert calls[checkpoint_tracing.RESUME_OPERATION] == "workflow_resume"
    assert calls[checkpoint_tracing.RESUME_CHECKPOINT_ID] == "ckpt-1"

  def test_logs_workflow_resume(self, mock_session):
    """trace_resume_workflow emits a structured log entry."""
    mock_span = MagicMock()
    with (
        patch(
            "google.adk.telemetry.checkpoint_tracing.trace.get_current_span",
            return_value=mock_span,
        ),
        patch.object(checkpoint_tracing.logger, "info") as mock_log,
    ):
      checkpoint_tracing.trace_resume_workflow(
          checkpoint_id="ckpt-2",
          session=mock_session,
          skipped_count=2,
      )

    mock_log.assert_called_once()
    extra = mock_log.call_args[1]["extra"]
    assert extra["skipped_count"] == 2
    assert extra["checkpoint_id"] == "ckpt-2"


class TestRecordInterruptMetrics:
  """Tests for record_interrupt_metrics."""

  def test_records_counter_and_latency(self):
    """record_interrupt_metrics calls counter.add and latency.record."""
    with (
        patch.object(checkpoint_tracing.interrupt_counter, "add") as mock_add,
        patch.object(
            checkpoint_tracing.interrupt_latency, "record"
        ) as mock_record,
    ):
      checkpoint_tracing.record_interrupt_metrics(
          operation="create", duration_ms=12.5, status="success"
      )

    mock_add.assert_called_once_with(
        1, attributes={"operation": "create", "status": "success"}
    )
    mock_record.assert_called_once_with(
        12.5, attributes={"operation": "create"}
    )

  def test_records_error_status(self):
    """record_interrupt_metrics correctly records error status."""
    with (
        patch.object(checkpoint_tracing.interrupt_counter, "add") as mock_add,
        patch.object(
            checkpoint_tracing.interrupt_latency, "record"
        ) as mock_record,
    ):
      checkpoint_tracing.record_interrupt_metrics(
          operation="approve", duration_ms=5.0, status="error"
      )

    mock_add.assert_called_once_with(
        1, attributes={"operation": "approve", "status": "error"}
    )
    mock_record.assert_called_once_with(
        5.0, attributes={"operation": "approve"}
    )


class TestRecordResumeMetrics:
  """Tests for record_resume_metrics."""

  def test_records_with_no_skipped(self):
    """record_resume_metrics sets has_skipped=False when skipped_count=0."""
    with (
        patch.object(checkpoint_tracing.resume_counter, "add") as mock_add,
        patch.object(
            checkpoint_tracing.resume_latency, "record"
        ) as mock_record,
    ):
      checkpoint_tracing.record_resume_metrics(
          operation="workflow_resume", duration_ms=20.0, skipped_count=0
      )

    mock_add.assert_called_once_with(
        1,
        attributes={
            "operation": "workflow_resume",
            "status": "success",
            "has_skipped": False,
        },
    )
    mock_record.assert_called_once_with(
        20.0, attributes={"operation": "workflow_resume"}
    )

  def test_records_with_skipped(self):
    """record_resume_metrics sets has_skipped=True when skipped_count > 0."""
    with (
        patch.object(checkpoint_tracing.resume_counter, "add") as mock_add,
        patch.object(
            checkpoint_tracing.resume_latency, "record"
        ) as mock_record,
    ):
      checkpoint_tracing.record_resume_metrics(
          operation="checkpoint_restore", duration_ms=8.0, skipped_count=5
      )

    mock_add.assert_called_once_with(
        1,
        attributes={
            "operation": "checkpoint_restore",
            "status": "success",
            "has_skipped": True,
        },
    )
    mock_record.assert_called_once_with(
        8.0, attributes={"operation": "checkpoint_restore"}
    )
