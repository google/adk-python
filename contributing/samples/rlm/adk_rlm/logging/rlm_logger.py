"""
Logger for RLM iterations.

Writes RLMIteration data to JSON-lines files for analysis and debugging.
Compatible with the original RLM visualizer.
"""

from datetime import datetime
import json
import os
import threading
import uuid

from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata


class RLMLogger:
  """Logger that writes RLMIteration data to a JSON-lines file."""

  def __init__(self, log_dir: str, file_name: str = "rlm"):
    """
    Initialize the RLM logger.

    Args:
        log_dir: Directory to store log files.
        file_name: Base name for log files.
    """
    # Convert to absolute path to ensure it works from any working directory
    # (important for child agents that may run in different directories)
    self.log_dir = os.path.abspath(log_dir)
    os.makedirs(self.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = str(uuid.uuid4())[:8]
    self.log_file_path = os.path.join(
        self.log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl"
    )

    self._iteration_count = 0
    self._metadata_logged = False
    self._lock = threading.Lock()

  def log_metadata(self, metadata: RLMMetadata) -> None:
    """
    Log RLM metadata as the first entry in the file.

    Args:
        metadata: The RLM configuration metadata.
    """
    if self._metadata_logged:
      return

    entry = {
        "type": "metadata",
        "timestamp": datetime.now().isoformat(),
        **metadata.to_dict(),
    }

    # Serialize to string first, then write atomically under lock
    # to prevent interleaved writes from concurrent threads
    line = json.dumps(entry) + "\n"
    with self._lock:
      with open(self.log_file_path, "a") as f:
        f.write(line)

    self._metadata_logged = True

  def log(
      self,
      iteration: RLMIteration,
      depth: int = 0,
      agent_name: str | None = None,
      parent_agent: str | None = None,
      parent_iteration: int | None = None,
      parent_block_index: int | None = None,
      parallel_batch_id: str | None = None,
      batch_index: int | None = None,
      batch_size: int | None = None,
  ) -> None:
    """
    Log an RLMIteration to the file.

    Args:
        iteration: The iteration to log.
        depth: The recursion depth (0 = root agent).
        agent_name: Name of the agent logging this iteration.
        parent_agent: Name of the parent agent that spawned this one.
        parent_iteration: The iteration number of the parent that spawned this agent.
        parent_block_index: The code block index in the parent that spawned this agent.
        parallel_batch_id: UUID of the parallel batch (if part of a batch).
        batch_index: Position within the parallel batch (0-indexed).
        batch_size: Total number of items in the parallel batch.
    """
    # Build entry outside lock, but increment counter and write inside lock
    # to prevent race conditions from concurrent threads
    entry = {
        "type": "iteration",
        "iteration": 0,  # Placeholder, set under lock
        "timestamp": datetime.now().isoformat(),
        "depth": depth,
        "agent_name": agent_name,
        "parent_agent": parent_agent,
        **iteration.to_dict(),
    }

    # Add optional parent iteration context
    if parent_iteration is not None:
      entry["parent_iteration"] = parent_iteration
    if parent_block_index is not None:
      entry["parent_block_index"] = parent_block_index

    # Add optional parallel batch metadata
    if parallel_batch_id is not None:
      entry["parallel_batch_id"] = parallel_batch_id
    if batch_index is not None:
      entry["batch_index"] = batch_index
    if batch_size is not None:
      entry["batch_size"] = batch_size

    # Serialize to string first, then write atomically under lock
    # to prevent interleaved writes from concurrent threads
    with self._lock:
      self._iteration_count += 1
      entry["iteration"] = self._iteration_count
      line = json.dumps(entry) + "\n"
      with open(self.log_file_path, "a") as f:
        f.write(line)

  def log_simple_llm_call(
      self,
      prompt: str,
      response: str,
      model: str,
      execution_time_ms: float,
      depth: int = 0,
      agent_name: str | None = None,
      parent_iteration: int | None = None,
      parent_block_index: int | None = None,
      batch_index: int | None = None,
      batch_size: int | None = None,
      error: str | None = None,
  ) -> None:
    """
    Log a simple (non-recursive) LLM call.

    This is used when llm_query() or llm_query_batched() is called with
    recursive=False, so there's no full RLMIteration to log.

    Args:
        prompt: The prompt sent to the LLM.
        response: The response received (or error message if call failed).
        model: The model used.
        execution_time_ms: Execution time in milliseconds.
        depth: The recursion depth (0 = root agent).
        agent_name: Name of the agent that made the call.
        parent_iteration: The iteration number that spawned this call.
        parent_block_index: The code block index that spawned this call.
        batch_index: Position within a batch (0-indexed), if part of a batch.
        batch_size: Total number of items in the batch, if part of a batch.
        error: Error message if the call failed.
    """
    entry = {
        "type": "simple_llm_call",
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "prompt": prompt[:500] if len(prompt) > 500 else prompt,
        "prompt_full": prompt,
        "response": response[:500] if len(response) > 500 else response,
        "response_full": response,
        "execution_time_ms": execution_time_ms,
        "depth": depth,
        "agent_name": agent_name,
        "recursive": False,
        "success": error is None,
    }

    # Add error if present
    if error is not None:
      entry["error"] = error

    # Add optional context
    if parent_iteration is not None:
      entry["parent_iteration"] = parent_iteration
    if parent_block_index is not None:
      entry["parent_block_index"] = parent_block_index

    # Add batch metadata
    if batch_index is not None:
      entry["batch_index"] = batch_index
    if batch_size is not None:
      entry["batch_size"] = batch_size

    # Serialize and write atomically under lock
    line = json.dumps(entry) + "\n"
    with self._lock:
      with open(self.log_file_path, "a") as f:
        f.write(line)

  @property
  def iteration_count(self) -> int:
    """Return the number of iterations logged."""
    return self._iteration_count

  def get_log_path(self) -> str:
    """Return the path to the log file."""
    return self.log_file_path
