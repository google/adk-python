"""
RLM Code Executor using ADK's BaseCodeExecutor.

This module provides a custom code executor that wraps LocalREPL
and provides llm_query() and FINAL() functions for the RLM pattern.
"""

import asyncio
import concurrent.futures
import logging
from queue import Empty
from queue import Queue
import threading
import time
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING
import uuid

from google.genai import types

from google import genai

logger = logging.getLogger(__name__)
from adk_rlm.events import RLMEventData
from adk_rlm.events import RLMEventType
from adk_rlm.llm import AsyncLLMRateLimiter
from adk_rlm.llm import llm_rate_limit
from adk_rlm.repl.local_repl import LocalREPL
from adk_rlm.usage import UsageTracker
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.events.event import Event
from pydantic import PrivateAttr

if TYPE_CHECKING:
  from adk_rlm.logging.rlm_logger import RLMLogger


class RLMCodeExecutor(BaseCodeExecutor):
  """
  Code executor that provides llm_query() and FINAL() functions.

  This executor wraps the LocalREPL and provides the RLM-specific
  functions for recursive LLM calls and final answer detection.

  When current_depth < max_depth, llm_query() creates a nested RLM
  execution that can itself execute code and make further llm_query calls.
  When current_depth >= max_depth, llm_query() falls back to a simple
  LLM call without code execution capability.
  """

  stateful: bool = True  # Persist namespace across code blocks

  # Use ```repl delimiter instead of ```python
  code_block_delimiters: list[tuple[str, str]] = [
      ("```repl\n", "\n```"),
  ]

  # Private attributes (not part of the Pydantic schema)
  _sub_model: str = PrivateAttr(default="gemini-3-flash-preview")
  _current_depth: int = PrivateAttr(default=0)
  _max_depth: int = PrivateAttr(default=5)
  _max_iterations: int = PrivateAttr(default=30)
  _repl: LocalREPL | None = PrivateAttr(default=None)
  _final_answer: str | None = PrivateAttr(default=None)
  _usage_tracker: UsageTracker = PrivateAttr(default_factory=UsageTracker)
  _logger: "RLMLogger | None" = PrivateAttr(default=None)
  _parent_agent: str | None = PrivateAttr(default=None)
  _current_iteration: int = PrivateAttr(default=0)
  _current_block_index: int = PrivateAttr(default=0)

  # Real-time event streaming via thread-safe queue
  _event_queue: Queue = PrivateAttr(default_factory=Queue)
  _execution_complete: threading.Event = PrivateAttr(
      default_factory=threading.Event
  )

  # Ancestry tracking for nested agents
  _ancestry: list[dict] = PrivateAttr(default_factory=list)

  # Counter for unique child agent names
  _child_agent_counter: int = PrivateAttr(default=0)

  def __init__(
      self,
      sub_model: str = "gemini-3-flash-preview",
      current_depth: int = 0,
      max_depth: int = 5,
      max_iterations: int = 30,
      usage_tracker: UsageTracker | None = None,
      logger: "RLMLogger | None" = None,
      parent_agent: str | None = None,
      ancestry: list[dict] | None = None,
      **kwargs,
  ):
    """
    Initialize the RLM code executor.

    Args:
        sub_model: The model to use for sub-LLM queries.
        current_depth: Current recursion depth (0 = root level).
        max_depth: Maximum recursion depth for nested RLM calls.
        max_iterations: Maximum iterations for nested RLM calls.
        usage_tracker: Optional usage tracker to record token usage.
        logger: Optional logger for recording iterations.
        parent_agent: Name of the parent agent that created this executor.
        ancestry: List of ancestor agent context dicts for event tagging.
        **kwargs: Additional arguments for BaseCodeExecutor.
    """
    super().__init__(**kwargs)
    self._sub_model = sub_model
    self._current_depth = current_depth
    self._max_depth = max_depth
    self._max_iterations = max_iterations
    self._repl = None
    self._final_answer = None
    self._usage_tracker = usage_tracker or UsageTracker()
    self._logger = logger
    self._parent_agent = parent_agent
    self._ancestry = ancestry.copy() if ancestry else []

    # Initialize queue and threading event
    self._event_queue = Queue()
    self._execution_complete = threading.Event()

  def _create_llm_query_fn(self):
    """Create the llm_query function for the REPL environment.

    When current_depth < max_depth, this creates a nested RLM execution
    that can itself execute code and make further llm_query calls.
    When at max_depth, falls back to a simple LLM call.
    """

    def llm_query(
        prompt: str,
        context: Any = None,
        model: str | None = None,
        recursive: bool = True,
    ) -> str:
      """
      Query an LLM with the given prompt.

      Args:
          prompt: The prompt to send to the LLM.
          context: Optional context object(s) to pass to the child agent.
                   Can be a LazyFile, LazyFileCollection, dict, list, or string.
                   The child agent can access this via its `context` variable.
          model: Optional model override.
          recursive: If True and depth allows, use recursive RLM execution.
                    If False, always use simple LLM call.

      Returns:
          The LLM's response text.
      """
      target_model = model or self._sub_model

      # Check if we can do recursive execution
      can_recurse = recursive and (self._current_depth < self._max_depth)

      if can_recurse:
        # Create a nested RLM execution
        return self._run_recursive_rlm(
            prompt, target_model, context_obj=context
        )
      else:
        # Simple LLM call (no code execution)
        return self._simple_llm_call(prompt, target_model)

    return llm_query

  def _simple_llm_call(
      self,
      prompt: str,
      model: str,
      batch_index: int | None = None,
      batch_size: int | None = None,
  ) -> str:
    """Make a simple LLM call without code execution capability.

    Emits SUB_LLM_START and SUB_LLM_END events for UI visibility and logs
    the call to the JSONL logger.

    Args:
        prompt: The prompt to send to the LLM.
        model: The model to use.
        batch_index: Position within a batch (0-indexed), if part of a batch.
        batch_size: Total number of items in the batch, if part of a batch.

    Returns:
        The LLM's response text, or an error message if the call failed.
    """
    # Emit start event
    self._emit_sub_llm_event(
        RLMEventType.SUB_LLM_START,
        model=model,
        prompt=prompt,
        batch_index=batch_index,
        batch_size=batch_size,
    )

    start_time = time.perf_counter()
    error_msg = None
    response_text = None

    try:
      # Create a fresh client for each simple LLM call to avoid
      # "Event loop is closed" errors when called from thread pool.
      # A shared client may hold references to an event loop that
      # is no longer valid in this thread context.
      client = genai.Client(vertexai=True, location="global")
      # Disable function calling to prevent MALFORMED_FUNCTION_CALL errors
      config = types.GenerateContentConfig(
          tool_config=types.ToolConfig(
              function_calling_config=types.FunctionCallingConfig(mode="NONE")
          )
      )
      with llm_rate_limit():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
      self._usage_tracker.add_from_response(model, response.usage_metadata)

      # Handle None/empty responses with detailed logging
      if response.text is None or response.text == "":
        finish_reason = None
        block_reason = None
        if response.candidates:
          finish_reason = getattr(response.candidates[0], "finish_reason", None)
        if hasattr(response, "prompt_feedback"):
          block_reason = getattr(response.prompt_feedback, "block_reason", None)

        logger.warning(
            "Simple LLM call returned empty response: model=%s, "
            "finish_reason=%s, block_reason=%s, prompt_preview=%s",
            model,
            finish_reason,
            block_reason,
            prompt[:100] if prompt else None,
        )

        reason_parts = []
        if finish_reason:
          reason_parts.append(f"finish_reason={finish_reason}")
        if block_reason:
          reason_parts.append(f"block_reason={block_reason}")
        reason_str = (
            ", ".join(reason_parts) if reason_parts else "unknown reason"
        )
        response_text = f"[LLM returned empty response: {reason_str}]"
      else:
        response_text = response.text
    except Exception as e:
      error_msg = str(e)
      response_text = f"Error: LLM query failed - {e}"

    execution_time_ms = (time.perf_counter() - start_time) * 1000

    # Emit end event
    self._emit_sub_llm_event(
        RLMEventType.SUB_LLM_END,
        model=model,
        response=response_text if not error_msg else None,
        error=error_msg,
        execution_time_ms=execution_time_ms,
        batch_index=batch_index,
        batch_size=batch_size,
    )

    # Log to JSONL
    self._log_simple_llm_call(
        prompt=prompt,
        response=response_text,
        model=model,
        execution_time_ms=execution_time_ms,
        batch_index=batch_index,
        batch_size=batch_size,
        error=error_msg,
    )

    return response_text

  def _get_current_ancestry_entry(self) -> dict:
    """Get the current agent's context for ancestry chain."""
    return {
        "agent": self._parent_agent,
        "depth": self._current_depth,
        "iteration": self._current_iteration,
        "block_index": self._current_block_index,
    }

  def _emit_sub_llm_event(
      self,
      event_type: RLMEventType,
      model: str,
      prompt: str | None = None,
      response: str | None = None,
      error: str | None = None,
      execution_time_ms: float | None = None,
      batch_index: int | None = None,
      batch_size: int | None = None,
  ) -> None:
    """Emit a sub-LLM event for simple (non-recursive) LLM calls.

    Args:
        event_type: The type of event (SUB_LLM_START or SUB_LLM_END).
        model: The model being used.
        prompt: The prompt (for START events).
        response: The response (for END events).
        error: Error message if the call failed.
        execution_time_ms: Execution time in milliseconds (for END events).
        batch_index: Position within a batch (0-indexed).
        batch_size: Total number of items in the batch.
    """
    event_data = RLMEventData(
        event_type=event_type,
        model=model,
        prompt_preview=prompt[:200] if prompt else None,
        response_preview=response[:500] if response else None,
        response_full=response,
        error=error,
        execution_time_ms=execution_time_ms,
        iteration=self._current_iteration,
        block_index=self._current_block_index,
        batch_index=batch_index,
        batch_size=batch_size,
        metadata={"recursive": False},
    )

    metadata = event_data.to_dict()
    metadata["agent_name"] = self._parent_agent
    metadata["agent_depth"] = self._current_depth
    metadata["ancestry"] = self._ancestry + [self._get_current_ancestry_entry()]

    event = Event(
        invocation_id=str(uuid.uuid4()),
        author=self._parent_agent or "code_executor",
        custom_metadata=metadata,
    )

    self._event_queue.put(event)

  def _log_simple_llm_call(
      self,
      prompt: str,
      response: str,
      model: str,
      execution_time_ms: float,
      batch_index: int | None = None,
      batch_size: int | None = None,
      error: str | None = None,
  ) -> None:
    """Log a simple (non-recursive) LLM call to the JSONL logger.

    Args:
        prompt: The prompt sent to the LLM.
        response: The response received (or error message if failed).
        model: The model used.
        execution_time_ms: Execution time in milliseconds.
        batch_index: Position within a batch (0-indexed).
        batch_size: Total number of items in the batch.
        error: Error message if the call failed.
    """
    if self._logger is None:
      return

    self._logger.log_simple_llm_call(
        prompt=prompt,
        response=response,
        model=model,
        execution_time_ms=execution_time_ms,
        depth=self._current_depth,
        agent_name=self._parent_agent,
        parent_iteration=self._current_iteration,
        parent_block_index=self._current_block_index,
        batch_index=batch_index,
        batch_size=batch_size,
        error=error,
    )

  def _run_recursive_rlm(
      self,
      prompt: str,
      model: str,
      context_obj: Any = None,
      parallel_batch_id: str | None = None,
      batch_index: int | None = None,
      batch_size: int | None = None,
  ) -> str:
    """Run a nested RLM execution at depth + 1 with real-time event streaming.

    Args:
        prompt: The prompt to send to the child agent.
        model: The model to use for the child agent.
        context_obj: Optional context object to pass to the child agent.
                     This becomes the child's `context` variable directly.
        parallel_batch_id: Optional UUID identifying a parallel batch.
        batch_index: Optional position within the batch (0-indexed).
        batch_size: Optional total number of items in the batch.
    """
    next_depth = self._current_depth + 1

    # Generate unique child agent name using counter
    child_index = self._child_agent_counter
    self._child_agent_counter += 1
    nested_agent_name = f"rlm_agent_depth_{next_depth}_{child_index}"

    # Build child's ancestry = parent's ancestry + current context
    child_ancestry = self._ancestry + [self._get_current_ancestry_entry()]

    # Reference to the event queue for the nested function
    event_queue = self._event_queue

    # Capture context_obj for the nested async function
    child_context = context_obj

    async def run_nested_async():
      """Run the nested agent async and stream events to queue."""
      import uuid

      from adk_rlm.agents.rlm_agent import RLMAgent
      from google.adk.agents.invocation_context import InvocationContext
      from google.adk.sessions import InMemorySessionService
      from google.adk.sessions import Session

      # Create a nested RLM agent at the next depth level
      nested_agent = RLMAgent(
          name=nested_agent_name,
          model=model,
          sub_model=self._sub_model,
          max_iterations=self._max_iterations,
          max_depth=self._max_depth,
          current_depth=next_depth,
          logger=self._logger,
          parent_agent=self._parent_agent,
          ancestry=child_ancestry,  # Pass ancestry to child
          verbose=False,
      )

      # Create mock session with context
      # If context_obj is provided, it becomes the child's `context` variable directly
      # Otherwise, fall back to {"query": prompt} for backwards compatibility
      rlm_context = (
          child_context if child_context is not None else {"query": prompt}
      )

      mock_session = Session(
          id=str(uuid.uuid4()),
          app_name="adk_rlm",
          user_id="default_user",
          state={
              "rlm_context": rlm_context,
              "rlm_prompt": prompt,
          },
      )
      mock_session_service = InMemorySessionService()
      mock_ctx = InvocationContext(
          invocation_id=str(uuid.uuid4()),
          session=mock_session,
          session_service=mock_session_service,
          agent=nested_agent,
      )

      final_answer = None

      try:
        async for event in nested_agent._run_async_impl(mock_ctx):
          # Only add ancestry if not already present (preserve nested info)
          if event.custom_metadata and "ancestry" not in event.custom_metadata:
            event.custom_metadata["ancestry"] = child_ancestry
            event.custom_metadata["agent_name"] = nested_agent_name
            event.custom_metadata["agent_depth"] = next_depth
            # Add parent info for backwards compatibility
            event.custom_metadata["parent_agent"] = self._parent_agent
            event.custom_metadata["parent_iteration"] = self._current_iteration
            event.custom_metadata["parent_block_index"] = (
                self._current_block_index
            )
            # Add batch metadata if this is part of a parallel batch
            if parallel_batch_id is not None:
              event.custom_metadata["parallel_batch_id"] = parallel_batch_id
              event.custom_metadata["batch_index"] = batch_index
              event.custom_metadata["batch_size"] = batch_size

          # Push to queue immediately for real-time streaming
          event_queue.put(event)

          # Check for final answer
          if event.custom_metadata:
            from adk_rlm.events import RLMEventType

            event_type = event.custom_metadata.get("event_type")
            if event_type == RLMEventType.FINAL_ANSWER.value:
              final_answer = event.custom_metadata.get("answer")

        # Merge usage
        self._usage_tracker.merge(nested_agent._usage_tracker)
      finally:
        # Properly close the nested agent's genai client before event loop closes
        # This prevents "Event loop is closed" errors during cleanup
        if nested_agent._client is not None:
          try:
            await nested_agent._client.aio.aclose()
          except Exception:
            pass  # Ignore cleanup errors

      return final_answer

    try:
      # Run async in a thread pool to avoid event loop conflicts
      try:
        asyncio.get_running_loop()
        # Already in an event loop, use thread pool
        with concurrent.futures.ThreadPoolExecutor() as pool:
          future = pool.submit(asyncio.run, run_nested_async())
          final_answer = future.result()
      except RuntimeError:
        # No running loop, safe to use asyncio.run directly
        final_answer = asyncio.run(run_nested_async())

      if final_answer is None:
        return "[Recursive RLM returned no result]"
      return final_answer

    except Exception as e:
      # Fall back to simple call on error
      return (
          f"[Recursive RLM at depth {next_depth} failed: {e}]\n"
          + self._simple_llm_call(prompt, model)
      )

  def _create_llm_query_batched_fn(self):
    """Create the llm_query_batched function for the REPL environment.

    When recursive=True, runs child agents in parallel using ThreadPoolExecutor.
    When recursive=False, uses async gather for simple parallel LLM calls.
    """

    def llm_query_batched(
        prompts: list[str],
        contexts: list[Any] | None = None,
        model: str | None = None,
        recursive: bool = False,
    ) -> list[str]:
      """
      Query an LLM with multiple prompts concurrently.

      Args:
          prompts: List of prompts to send.
          contexts: Optional list of context objects (same length as prompts).
                    If provided, each prompt gets paired with its context.
          model: Optional model override.
          recursive: If True, use recursive RLM execution for each prompt.
                    Default is False for performance (simple LLM calls).

      Returns:
          List of LLM response texts in the same order as prompts.
      """
      if contexts is not None and len(contexts) != len(prompts):
        raise ValueError(
            f"contexts length ({len(contexts)}) must match prompts length"
            f" ({len(prompts)})"
        )

      target_model = model or self._sub_model

      if recursive and self._current_depth < self._max_depth:
        # Parallel recursive execution using ThreadPoolExecutor
        return self._run_parallel_recursive(prompts, contexts, target_model)

      # Simple async batched calls (no recursion) - emit events for each query
      batch_size = len(prompts)

      # Capture references needed in the async functions
      usage_tracker = self._usage_tracker
      emit_event = self._emit_sub_llm_event
      log_call = self._log_simple_llm_call

      async def query_single(
          client: genai.Client, prompt: str, batch_index: int
      ) -> str:
        # Emit start event
        emit_event(
            RLMEventType.SUB_LLM_START,
            model=target_model,
            prompt=prompt,
            batch_index=batch_index,
            batch_size=batch_size,
        )

        start_time = time.perf_counter()
        error_msg = None
        response_text = None

        try:
          # Disable function calling to prevent MALFORMED_FUNCTION_CALL errors
          config = types.GenerateContentConfig(
              tool_config=types.ToolConfig(
                  function_calling_config=types.FunctionCallingConfig(
                      mode="NONE"
                  )
              )
          )
          async with AsyncLLMRateLimiter():
            response = await client.aio.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )
          usage_tracker.add_from_response(target_model, response.usage_metadata)

          # Handle None/empty responses with detailed logging
          if response.text is None or response.text == "":
            finish_reason = None
            block_reason = None
            if response.candidates:
              finish_reason = getattr(
                  response.candidates[0], "finish_reason", None
              )
            if hasattr(response, "prompt_feedback"):
              block_reason = getattr(
                  response.prompt_feedback, "block_reason", None
              )

            logger.warning(
                "Batched LLM call returned empty response: model=%s, "
                "batch_index=%s/%s, finish_reason=%s, block_reason=%s",
                target_model,
                batch_index,
                batch_size,
                finish_reason,
                block_reason,
            )

            reason_parts = []
            if finish_reason:
              reason_parts.append(f"finish_reason={finish_reason}")
            if block_reason:
              reason_parts.append(f"block_reason={block_reason}")
            reason_str = (
                ", ".join(reason_parts) if reason_parts else "unknown reason"
            )
            response_text = f"[LLM returned empty response: {reason_str}]"
          else:
            response_text = response.text
        except Exception as e:
          error_msg = str(e)
          response_text = f"Error: LLM query failed - {e}"

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Emit end event
        emit_event(
            RLMEventType.SUB_LLM_END,
            model=target_model,
            response=response_text if not error_msg else None,
            error=error_msg,
            execution_time_ms=execution_time_ms,
            batch_index=batch_index,
            batch_size=batch_size,
        )

        # Log to JSONL
        log_call(
            prompt=prompt,
            response=response_text,
            model=target_model,
            execution_time_ms=execution_time_ms,
            batch_index=batch_index,
            batch_size=batch_size,
            error=error_msg,
        )

        return response_text

      async def run_all():
        # Create a fresh client in this event loop to avoid
        # "Event loop is closed" errors from cross-thread usage
        client = genai.Client(vertexai=True, location="global")
        tasks = [query_single(client, p, i) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)

      # Run in a new event loop if we're not already in one
      try:
        asyncio.get_running_loop()
        # If we're in a running loop, create a new one in a thread
        with concurrent.futures.ThreadPoolExecutor() as pool:
          future = pool.submit(asyncio.run, run_all())
          return future.result()
      except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(run_all())

    return llm_query_batched

  def _run_parallel_recursive(
      self,
      prompts: list[str],
      contexts: list[Any] | None,
      model: str,
  ) -> list[str]:
    """Run multiple recursive RLM calls in parallel.

    This spawns child agents concurrently using ThreadPoolExecutor.
    Rate limiting is handled by the global LLM semaphore.

    Args:
        prompts: List of prompts to send.
        contexts: Optional list of context objects (same length as prompts).
        model: The model to use for child agents.

    Returns:
        List of results in the same order as prompts.
    """
    contexts = contexts or [None] * len(prompts)
    batch_id = str(uuid.uuid4())
    batch_size = len(prompts)

    def run_one(idx: int) -> tuple[int, str]:
      """Run a single recursive RLM call and return (index, result)."""
      prompt = prompts[idx]
      context = contexts[idx]
      try:
        result = self._run_recursive_rlm(
            prompt,
            model,
            context_obj=context,
            parallel_batch_id=batch_id,
            batch_index=idx,
            batch_size=batch_size,
        )
        return (idx, result)
      except Exception as e:
        return (idx, f"[Error in batch item {idx}: {e}]")

    results = [None] * len(prompts)

    with concurrent.futures.ThreadPoolExecutor() as pool:
      futures = [pool.submit(run_one, i) for i in range(len(prompts))]

      for future in concurrent.futures.as_completed(futures):
        try:
          idx, result = future.result()
          results[idx] = result
        except Exception:
          # This shouldn't happen since run_one catches exceptions,
          # but handle it just in case
          pass

    # Replace any None results with error messages
    for i, result in enumerate(results):
      if result is None:
        results[i] = f"[Error: batch item {i} returned no result]"

    return results

  def _ensure_repl(self) -> LocalREPL:
    """Ensure the REPL is initialized."""
    if self._repl is None:
      self._repl = LocalREPL(
          llm_query_fn=self._create_llm_query_fn(),
          llm_query_batched_fn=self._create_llm_query_batched_fn(),
      )
    return self._repl

  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """
    Execute code in the RLM REPL environment.

    Args:
        invocation_context: The ADK invocation context.
        code_execution_input: The code to execute.

    Returns:
        CodeExecutionResult with stdout/stderr.
    """
    repl = self._ensure_repl()

    # Execute code
    result = repl.execute_code(code_execution_input.code)

    # Check for FINAL answer in namespace
    if "FINAL_ANSWER" in repl.locals:
      self._final_answer = str(repl.locals["FINAL_ANSWER"])

    return CodeExecutionResult(
        stdout=result.stdout,
        stderr=result.stderr,
        output_files=[],
    )

  def reset_event_state(self) -> None:
    """Reset the event queue and completion flag.

    This should be called BEFORE starting execute_code_async to avoid
    race conditions between the execution task and event polling.

    Note: We intentionally do NOT reset _child_agent_counter here.
    Keeping it monotonically increasing ensures unique agent names
    across all iterations and code blocks within a run.
    """
    self._event_queue = Queue()
    self._execution_complete.clear()

  async def execute_code_async(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """
    Execute code in the RLM REPL environment asynchronously.

    This runs the code execution in a thread pool to avoid blocking
    the event loop, which is important when the code calls llm_query()
    with recursive=True and spawns child agents.

    Note: Call reset_event_state() BEFORE creating the task to avoid
    race conditions with poll_child_events().

    Args:
        invocation_context: The ADK invocation context.
        code_execution_input: The code to execute.

    Returns:
        CodeExecutionResult with stdout/stderr.
    """
    # Run the synchronous execute_code in a thread pool
    # This allows the event loop to continue processing (e.g., sending websocket events)
    # while the code execution (which may spawn child agents) runs
    result = await asyncio.to_thread(
        self._execute_code_with_completion,
        invocation_context,
        code_execution_input,
    )
    return result

  def _execute_code_with_completion(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Execute code and signal completion when done."""
    try:
      return self.execute_code(invocation_context, code_execution_input)
    finally:
      self._execution_complete.set()

  async def poll_child_events(self) -> AsyncGenerator[Event, None]:
    """
    Poll for child agent events during code execution.

    This async generator yields events as they arrive from child agents
    running in the thread pool. It should be called in a loop while
    code execution is running.

    Yields:
        Event objects from child agents as they arrive.
    """
    while (
        not self._execution_complete.is_set() or not self._event_queue.empty()
    ):
      try:
        event = self._event_queue.get_nowait()
        yield event
      except Empty:
        # Small sleep to avoid busy-wait
        await asyncio.sleep(0.01)

  def load_context(self, context_payload: dict | list | str) -> None:
    """
    Load context into the REPL environment.

    Args:
        context_payload: The context data to load.
    """
    repl = self._ensure_repl()
    repl.load_context(context_payload)

  def add_context(self, context_payload: dict | list | str) -> int:
    """
    Add additional context to the REPL environment.

    Args:
        context_payload: The context data to add.

    Returns:
        The context index.
    """
    repl = self._ensure_repl()
    return repl.add_context(context_payload)

  def get_context_count(self) -> int:
    """Return the number of contexts loaded."""
    if self._repl is None:
      return 0
    return self._repl.get_context_count()

  def get_history_count(self) -> int:
    """Return the number of conversation histories stored."""
    if self._repl is None:
      return 0
    return self._repl.get_history_count()

  def add_history(self, message_history: list[dict[str, Any]]) -> int:
    """
    Store a conversation's message history.

    Args:
        message_history: The list of message dicts.

    Returns:
        The history index.
    """
    repl = self._ensure_repl()
    return repl.add_history(message_history)

  @property
  def final_answer(self) -> str | None:
    """Return the final answer if detected via FINAL_ANSWER variable."""
    return self._final_answer

  def reset_final_answer(self) -> None:
    """Reset the final answer state."""
    self._final_answer = None

  @property
  def locals(self) -> dict[str, Any]:
    """Return the REPL locals for variable inspection."""
    if self._repl is None:
      return {}
    return self._repl.locals

  @property
  def usage_tracker(self) -> UsageTracker:
    """Return the usage tracker."""
    return self._usage_tracker

  def set_iteration_context(self, iteration: int, block_index: int) -> None:
    """Set the current iteration context for child event tagging.

    Args:
        iteration: The current parent iteration number (1-indexed).
        block_index: The current code block index within the iteration.
    """
    self._current_iteration = iteration
    self._current_block_index = block_index

  def pop_child_events(self) -> list:
    """Get and clear any remaining child agent events from the queue.

    This is provided for backwards compatibility. With the new streaming
    architecture, events are yielded in real-time via poll_child_events().

    Returns:
        List of remaining events from the queue, cleared after retrieval.
    """
    events = []
    while not self._event_queue.empty():
      try:
        events.append(self._event_queue.get_nowait())
      except Empty:
        break
    return events

  def cleanup(self) -> None:
    """Clean up the REPL environment."""
    if self._repl:
      self._repl.cleanup()
      self._repl = None
    self._final_answer = None
    # Clear the event queue
    while not self._event_queue.empty():
      try:
        self._event_queue.get_nowait()
      except Empty:
        break
    self._execution_complete.clear()
