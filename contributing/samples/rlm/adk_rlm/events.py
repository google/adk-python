"""
RLM Event types for streaming execution updates.

These events provide granular visibility into RLM execution, enabling
various interfaces (CLI, Web UI, API) to show real-time progress.
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


class RLMEventType(str, Enum):
  """Event types emitted during RLM execution."""

  # Lifecycle events
  RUN_START = "rlm.run.start"  # Agent starting execution
  RUN_END = "rlm.run.end"  # Agent finished (with final answer)
  RUN_ERROR = "rlm.run.error"  # Agent encountered error

  # Iteration events
  ITERATION_START = "rlm.iteration.start"  # Starting iteration N
  ITERATION_END = "rlm.iteration.end"  # Completed iteration N

  # LLM events
  LLM_CALL_START = "rlm.llm.start"  # Calling main LLM
  LLM_CALL_END = "rlm.llm.end"  # Main LLM response received
  LLM_RESPONSE = "rlm.llm.response"  # Streaming LLM response chunk

  # Code execution events
  CODE_FOUND = "rlm.code.found"  # Found code block in response
  CODE_EXEC_START = "rlm.code.start"  # Starting code execution
  CODE_EXEC_END = "rlm.code.end"  # Code execution completed
  CODE_OUTPUT = "rlm.code.output"  # Code produced output

  # Sub-LLM events (from llm_query calls)
  SUB_LLM_START = "rlm.sub_llm.start"  # Sub-LLM query started
  SUB_LLM_END = "rlm.sub_llm.end"  # Sub-LLM query completed
  SUB_LLM_BATCH = "rlm.sub_llm.batch"  # Batched sub-LLM queries

  # Final answer
  FINAL_DETECTED = "rlm.final.detected"  # FINAL() pattern found
  FINAL_ANSWER = "rlm.final.answer"  # Final answer content


@dataclass
class RLMEventData:
  """Structured data for RLM events."""

  event_type: RLMEventType
  iteration: int | None = None
  code: str | None = None  # Truncated code preview for sidebar
  code_full: str | None = None  # Full code for modal display
  output: str | None = None  # Truncated output preview for sidebar
  output_full: str | None = None  # Full output for modal display
  error: str | None = None  # Truncated error preview for sidebar
  error_full: str | None = None  # Full error for modal display
  model: str | None = None
  prompt_preview: str | None = None  # First N chars of prompt
  response_preview: str | None = None  # First N chars of response
  response_full: str | None = None  # Full LLM response for modal display
  answer: str | None = None
  token_count: int | None = None
  execution_time_ms: float | None = None
  total_iterations: int | None = None
  success: bool | None = None
  fallback: bool | None = None
  block_index: int | None = None
  has_error: bool | None = None
  source: str | None = None  # "text" or "variable" for FINAL detection
  # Parallel batch metadata (for llm_query_batched with recursive=True)
  parallel_batch_id: str | None = None  # UUID identifying the batch
  batch_index: int | None = None  # Position within the batch (0-indexed)
  batch_size: int | None = None  # Total number of items in the batch
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary, excluding None values."""
    result = {"event_type": self.event_type.value}
    for key, value in self.__dict__.items():
      if key != "event_type" and value is not None:
        if key == "metadata" and not value:
          continue
        result[key] = value
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "RLMEventData":
    """Create from dictionary."""
    event_type = RLMEventType(data.pop("event_type"))
    return cls(event_type=event_type, **data)
