"""OpenTelemetry instrumentation for checkpoint, interrupt, and resume operations.

This module provides tracing, logging, and metrics for ADK's state management
features following OpenTelemetry semantic conventions.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from typing import TYPE_CHECKING

from opentelemetry import _logs
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.semconv.schemas import Schemas

from .. import version

if TYPE_CHECKING:
  from ..sessions.session import Session

# OpenTelemetry tracer for checkpoint/interrupt/resume operations
tracer = trace.get_tracer(
    instrumenting_module_name="gcp.vertex.agent.checkpoints",
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

# OpenTelemetry logger for structured logs
otel_logger = _logs.get_logger(
    instrumenting_module_name="gcp.vertex.agent.checkpoints",
    instrumenting_library_version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

# Python logger for standard logging
logger = logging.getLogger("google_adk." + __name__)

# OpenTelemetry meter for metrics
meter = metrics.get_meter(
    name="gcp.vertex.agent.checkpoints",
    version=version.__version__,
    schema_url=Schemas.V1_36_0.value,
)

# Metrics
checkpoint_counter = meter.create_counter(
    name="checkpoint.operations",
    description="Number of checkpoint operations",
    unit="1",
)

checkpoint_latency = meter.create_histogram(
    name="checkpoint.latency",
    description="Checkpoint operation latency",
    unit="ms",
)

interrupt_counter = meter.create_counter(
    name="interrupt.operations",
    description="Number of interrupt operations",
    unit="1",
)

interrupt_latency = meter.create_histogram(
    name="interrupt.latency",
    description="Interrupt operation latency",
    unit="ms",
)

resume_counter = meter.create_counter(
    name="resume.operations",
    description="Number of resume operations",
    unit="1",
)

resume_latency = meter.create_histogram(
    name="resume.latency",
    description="Resume operation latency",
    unit="ms",
)

# Queue depth gauge for monitoring
interrupt_queue_depth_gauge = meter.create_gauge(
    name="interrupt.queue_depth",
    description="Current queue depth per session",
    unit="1",
)

# Checkpoint state size histogram
checkpoint_state_size_histogram = meter.create_histogram(
    name="checkpoint.state_size",
    description="Size of checkpoint state in bytes",
    unit="bytes",
)

# Semantic Conventions - Checkpoint Attributes
CHECKPOINT_OPERATION = "checkpoint.operation"
CHECKPOINT_ID = "checkpoint.id"
CHECKPOINT_SESSION_ID = "checkpoint.session_id"
CHECKPOINT_AGENT_NAME = "checkpoint.agent_name"
CHECKPOINT_DESCRIPTION = "checkpoint.description"
CHECKPOINT_ARTIFACT_COUNT = "checkpoint.artifact_count"
CHECKPOINT_STATE_SIZE = "checkpoint.state_size"  # Size in bytes

# Semantic Conventions - Interrupt Attributes
INTERRUPT_OPERATION = "interrupt.operation"
INTERRUPT_ID = "interrupt.id"
INTERRUPT_TYPE = "interrupt.type"
INTERRUPT_STATUS = "interrupt.status"
INTERRUPT_SESSION_ID = "interrupt.session_id"
INTERRUPT_AGENT_NAME = "interrupt.agent_name"
INTERRUPT_MESSAGE = "interrupt.message"

# Semantic Conventions - Resume Attributes
RESUME_OPERATION = "resume.operation"
RESUME_CHECKPOINT_ID = "resume.checkpoint_id"
RESUME_SESSION_ID = "resume.session_id"
RESUME_SKIPPED_COUNT = "resume.skipped_count"


def trace_checkpoint_create(
    checkpoint_id: str,
    session: Session,
    agent_name: str | None = None,
    description: str | None = None,
    artifact_count: int = 0,
    state_size_bytes: int | None = None,
):
  """Trace checkpoint creation operation.

  Args:
      checkpoint_id: Unique checkpoint identifier
      session: Session being checkpointed
      agent_name: Name of agent creating checkpoint
      description: Human-readable checkpoint description
      artifact_count: Number of artifacts tracked in checkpoint
      state_size_bytes: Size of checkpoint state in bytes
  """
  span = trace.get_current_span()

  # Set span attributes
  span.set_attribute(CHECKPOINT_OPERATION, "create")
  span.set_attribute(CHECKPOINT_ID, checkpoint_id)
  span.set_attribute(CHECKPOINT_SESSION_ID, session.id)
  if agent_name:
    span.set_attribute(CHECKPOINT_AGENT_NAME, agent_name)
  if description:
    span.set_attribute(CHECKPOINT_DESCRIPTION, description)
  if artifact_count > 0:
    span.set_attribute(CHECKPOINT_ARTIFACT_COUNT, artifact_count)
  if state_size_bytes is not None:
    span.set_attribute(CHECKPOINT_STATE_SIZE, state_size_bytes)

  # Record state size histogram
  if state_size_bytes is not None:
    checkpoint_state_size_histogram.record(
        state_size_bytes,
        attributes={
            CHECKPOINT_SESSION_ID: session.id,
            CHECKPOINT_OPERATION: "create",
        },
    )

  # Structured logging
  logger.info(
      "Checkpoint created",
      extra={
          "checkpoint_id": checkpoint_id,
          "session_id": session.id,
          "agent_name": agent_name,
          "description": description,
          "artifact_count": artifact_count,
          "state_size_bytes": state_size_bytes,
      },
  )


def trace_checkpoint_restore(
    checkpoint_id: str,
    session: Session,
):
  """Trace checkpoint restore operation.

  Args:
      checkpoint_id: Checkpoint being restored
      session: Session being restored to checkpoint state
  """
  span = trace.get_current_span()

  span.set_attribute(CHECKPOINT_OPERATION, "restore")
  span.set_attribute(CHECKPOINT_ID, checkpoint_id)
  span.set_attribute(CHECKPOINT_SESSION_ID, session.id)

  logger.info(
      "Checkpoint restored",
      extra={
          "checkpoint_id": checkpoint_id,
          "session_id": session.id,
      },
  )


def trace_checkpoint_delete(
    checkpoint_id: str,
    session: Session,
):
  """Trace checkpoint deletion operation.

  Args:
      checkpoint_id: Checkpoint being deleted
      session: Session containing checkpoint
  """
  span = trace.get_current_span()

  span.set_attribute(CHECKPOINT_OPERATION, "delete")
  span.set_attribute(CHECKPOINT_ID, checkpoint_id)
  span.set_attribute(CHECKPOINT_SESSION_ID, session.id)

  logger.info(
      "Checkpoint deleted",
      extra={
          "checkpoint_id": checkpoint_id,
          "session_id": session.id,
      },
  )


def trace_checkpoint_list(
    session: Session,
    checkpoint_count: int,
):
  """Trace checkpoint list operation.

  Args:
      session: Session being queried
      checkpoint_count: Number of checkpoints found
  """
  span = trace.get_current_span()

  span.set_attribute(CHECKPOINT_OPERATION, "list")
  span.set_attribute(CHECKPOINT_SESSION_ID, session.id)
  span.set_attribute("checkpoint.count", checkpoint_count)

  logger.debug(
      "Checkpoints listed",
      extra={
          "session_id": session.id,
          "checkpoint_count": checkpoint_count,
      },
  )


def trace_interrupt_create(
    interrupt_id: str,
    interrupt_type: str,
    session: Session,
    agent_name: str,
    message: str | None = None,
):
  """Trace interrupt creation operation.

  Args:
      interrupt_id: Unique interrupt identifier
      interrupt_type: Type of interrupt (BEFORE, AFTER, DYNAMIC)
      session: Session where interrupt occurs
      agent_name: Agent requesting interrupt
      message: Human-readable interrupt message
  """
  span = trace.get_current_span()

  span.set_attribute(INTERRUPT_OPERATION, "create")
  span.set_attribute(INTERRUPT_ID, interrupt_id)
  span.set_attribute(INTERRUPT_TYPE, interrupt_type)
  span.set_attribute(INTERRUPT_STATUS, "pending")
  span.set_attribute(INTERRUPT_SESSION_ID, session.id)
  span.set_attribute(INTERRUPT_AGENT_NAME, agent_name)
  if message:
    span.set_attribute(INTERRUPT_MESSAGE, message)

  logger.info(
      "Interrupt created",
      extra={
          "interrupt_id": interrupt_id,
          "interrupt_type": interrupt_type,
          "session_id": session.id,
          "agent_name": agent_name,
          "interrupt_message": message,
      },
  )


def trace_interrupt_resolve(
    interrupt_id: str,
    session: Session,
    resolution: str,
    response_data: dict[str, Any] | None = None,
):
  """Trace interrupt resolution operation.

  Args:
      interrupt_id: Interrupt being resolved
      session: Session containing interrupt
      resolution: Resolution status (APPROVED, REJECTED)
      response_data: Optional response data from user
  """
  span = trace.get_current_span()

  span.set_attribute(INTERRUPT_OPERATION, "resolve")
  span.set_attribute(INTERRUPT_ID, interrupt_id)
  span.set_attribute(INTERRUPT_STATUS, resolution.lower())
  span.set_attribute(INTERRUPT_SESSION_ID, session.id)

  logger.info(
      "Interrupt resolved",
      extra={
          "interrupt_id": interrupt_id,
          "session_id": session.id,
          "resolution": resolution,
          "has_response_data": response_data is not None,
      },
  )


def trace_resume_workflow(
    checkpoint_id: str,
    session: Session,
    skipped_count: int = 0,
):
  """Trace workflow resume operation.

  Args:
      checkpoint_id: Checkpoint being resumed from
      session: Session being resumed
      skipped_count: Number of nodes/tasks skipped during resume
  """
  span = trace.get_current_span()

  span.set_attribute(RESUME_OPERATION, "workflow_resume")
  span.set_attribute(RESUME_CHECKPOINT_ID, checkpoint_id)
  span.set_attribute(RESUME_SESSION_ID, session.id)
  if skipped_count > 0:
    span.set_attribute(RESUME_SKIPPED_COUNT, skipped_count)

  logger.info(
      "Workflow resumed",
      extra={
          "checkpoint_id": checkpoint_id,
          "session_id": session.id,
          "skipped_count": skipped_count,
      },
  )


def record_checkpoint_metrics(
    operation: str,
    duration_ms: float,
    status: str = "success",
):
  """Record checkpoint operation metrics.

  Args:
      operation: Operation type (create, restore, delete, list)
      duration_ms: Operation duration in milliseconds
      status: Operation status (success, error)
  """
  checkpoint_counter.add(
      1,
      attributes={
          "operation": operation,
          "status": status,
      },
  )

  checkpoint_latency.record(
      duration_ms,
      attributes={
          "operation": operation,
      },
  )


def record_interrupt_metrics(
    operation: str,
    duration_ms: float,
    status: str = "success",
):
  """Record interrupt operation metrics.

  Args:
      operation: Operation type (create, approve, reject)
      duration_ms: Operation duration in milliseconds
      status: Operation status (success, error)
  """
  interrupt_counter.add(
      1,
      attributes={
          "operation": operation,
          "status": status,
      },
  )

  interrupt_latency.record(
      duration_ms,
      attributes={
          "operation": operation,
      },
  )


def record_resume_metrics(
    operation: str,
    duration_ms: float,
    skipped_count: int = 0,
    status: str = "success",
):
  """Record resume operation metrics.

  Args:
      operation: Operation type (workflow_resume, checkpoint_restore)
      duration_ms: Operation duration in milliseconds
      skipped_count: Number of nodes/tasks skipped
      status: Operation status (success, error)
  """
  resume_counter.add(
      1,
      attributes={
          "operation": operation,
          "status": status,
          "has_skipped": skipped_count > 0,
      },
  )

  resume_latency.record(
      duration_ms,
      attributes={
          "operation": operation,
      },
  )


# ===========================
# Dynamic InterruptService Tracing
# ===========================


def trace_interrupt_service_pause(
    session_id: str,
):
  """Trace dynamic interrupt pause operation.

  Args:
      session_id: Session being paused
  """
  span = trace.get_current_span()

  span.set_attribute(INTERRUPT_OPERATION, "pause")
  span.set_attribute(INTERRUPT_SESSION_ID, session_id)
  span.set_attribute(INTERRUPT_TYPE, "dynamic")

  logger.info(
      "InterruptService: Session paused",
      extra={
          "session_id": session_id,
          "operation": "pause",
      },
  )


def trace_interrupt_service_resume(
    session_id: str,
):
  """Trace dynamic interrupt resume operation.

  Args:
      session_id: Session being resumed
  """
  span = trace.get_current_span()

  span.set_attribute(INTERRUPT_OPERATION, "resume")
  span.set_attribute(INTERRUPT_SESSION_ID, session_id)
  span.set_attribute(INTERRUPT_TYPE, "dynamic")

  logger.info(
      "InterruptService: Session resumed",
      extra={
          "session_id": session_id,
          "operation": "resume",
      },
  )


def trace_interrupt_service_message(
    session_id: str,
    message_text: str,
    action: str | None = None,
    queue_depth: int = 0,
):
  """Trace dynamic interrupt message operation.

  Args:
      session_id: Session receiving message
      message_text: Message text
      action: Action type (e.g., 'update_state')
      queue_depth: Current queue depth after message
  """
  span = trace.get_current_span()

  span.set_attribute(INTERRUPT_OPERATION, "message")
  span.set_attribute(INTERRUPT_SESSION_ID, session_id)
  span.set_attribute(INTERRUPT_TYPE, "dynamic")
  span.set_attribute(
      INTERRUPT_MESSAGE, message_text[:200]
  )  # Truncate long messages
  if action:
    span.set_attribute("interrupt.action", action)
  span.set_attribute("interrupt.queue_depth", queue_depth)

  logger.info(
      "InterruptService: Message queued",
      extra={
          "session_id": session_id,
          "operation": "message",
          "action": action,
          "queue_depth": queue_depth,
          "message_length": len(message_text),
      },
  )


def trace_interrupt_service_cancel(
    session_id: str,
    messages_cleared: int = 0,
):
  """Trace dynamic interrupt cancel operation.

  Args:
      session_id: Session being cancelled
      messages_cleared: Number of messages cleared from queue
  """
  span = trace.get_current_span()

  span.set_attribute(INTERRUPT_OPERATION, "cancel")
  span.set_attribute(INTERRUPT_SESSION_ID, session_id)
  span.set_attribute(INTERRUPT_TYPE, "dynamic")
  span.set_attribute("interrupt.messages_cleared", messages_cleared)

  logger.info(
      "InterruptService: Session cancelled",
      extra={
          "session_id": session_id,
          "operation": "cancel",
          "messages_cleared": messages_cleared,
      },
  )


def record_interrupt_service_metrics(
    operation: str,
    duration_ms: float,
    status: str = "success",
    queue_depth: int = 0,
):
  """Record dynamic InterruptService operation metrics.

  Args:
      operation: Operation type (pause, resume, message, cancel, wait)
      duration_ms: Operation duration in milliseconds
      status: Operation status (success, error, timeout)
      queue_depth: Current queue depth (for message operations)
  """
  attributes = {
      "operation": operation,
      "status": status,
      "service": "interrupt_service",
  }

  if queue_depth > 0:
    attributes["queue_depth_bucket"] = (
        "0-10"
        if queue_depth <= 10
        else (
            "11-50"
            if queue_depth <= 50
            else "51-100"
            if queue_depth <= 100
            else "100+"
        )
    )

  interrupt_counter.add(1, attributes=attributes)
  interrupt_latency.record(duration_ms, attributes={"operation": operation})


def update_interrupt_queue_depth(
    session_id: str,
    queue_depth: int,
):
  """Update queue depth gauge for monitoring.

  Args:
      session_id: Session identifier
      queue_depth: Current queue depth
  """
  interrupt_queue_depth_gauge.set(
      queue_depth,
      attributes={
          "session_id": session_id,
          "service": "interrupt_service",
      },
  )
