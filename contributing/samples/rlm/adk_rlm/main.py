"""
Main entry point and convenience wrapper for ADK-RLM.

This module provides the RLM class which is the primary interface
for using Recursive Language Models with ADK framework integration.
"""

from pathlib import Path
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from adk_rlm.agents.rlm_agent import RLMAgent
from adk_rlm.events import RLMEventType
from adk_rlm.files import FileLoader
from adk_rlm.files import FileParser
from adk_rlm.files import FileSource
from adk_rlm.logging.rlm_logger import RLMLogger
from adk_rlm.types import RLMChatCompletion
from google.adk import Runner
from google.adk.events.event import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

if TYPE_CHECKING:
  from adk_rlm.files import LazyFileCollection


class RLM:
  """
  Recursive Language Model - main user-facing class.

  This provides a simple interface to the RLM functionality, using
  Google ADK framework for agent execution and session management.

  The primary interface is `run_streaming()` which yields ADK Events
  for real-time UI updates. For simple synchronous usage, use the
  module-level `completion()` convenience function.

  Example:
      ```python
      from adk_rlm import RLM, completion

      # Streaming API for real-time UI updates
      rlm = RLM(model="gemini-3-pro-preview")

      async for event in rlm.run_streaming(context, prompt):
          event_type = event.custom_metadata.get("event_type")
          if event_type == "rlm.final.answer":
              print(event.custom_metadata["answer"])

      # Or use the convenience function for simple synchronous usage
      result = completion(
          context="Your long document here...",
          prompt="What are the key themes?",
      )
      print(result.response)
      ```
  """

  def __init__(
      self,
      model: str = "gemini-3-pro-preview",
      sub_model: str | None = None,
      max_iterations: int = 30,
      max_depth: int = 5,
      custom_system_prompt: str | None = None,
      log_dir: str | None = None,
      verbose: bool = False,
      persistent: bool = False,
      # File handling
      file_sources: dict[str, FileSource] | None = None,
      file_parsers: list[FileParser] | None = None,
      base_path: str | Path | None = None,
      # Legacy kwargs for compatibility
      backend: str | None = None,
      backend_kwargs: dict[str, Any] | None = None,
      **kwargs,
  ):
    """
    Initialize the RLM.

    Args:
        model: The main model to use (default: gemini-3-pro-preview).
        sub_model: The model for recursive sub-calls (defaults to model).
        max_iterations: Maximum number of RLM iterations (default: 30).
        max_depth: Maximum recursion depth (default: 5).
        custom_system_prompt: Custom system prompt (uses default if None).
        log_dir: Directory for JSONL logs (None disables logging).
        verbose: Whether to print Rich console output.
        persistent: Whether to persist REPL state across calls.
        file_sources: Dictionary of named file sources for file loading.
        file_parsers: List of file parsers for file loading.
        base_path: Base path for local file source.
        backend: Legacy parameter (ignored, always uses Gemini).
        backend_kwargs: Legacy parameter for backend configuration.
    """
    # Handle legacy backend_kwargs
    if (
        backend_kwargs
        and "model_name" in backend_kwargs
        and model == "gemini-3-pro-preview"
    ):
      model = backend_kwargs["model_name"]

    # Create logger if log_dir specified
    logger = RLMLogger(log_dir) if log_dir else None

    # Create the underlying agent
    self._agent = RLMAgent(
        name="rlm_agent",
        model=model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_depth=max_depth,
        custom_system_prompt=custom_system_prompt,
        logger=logger,
        verbose=verbose,
        persistent=persistent,
    )

    # Create session service for ADK Runner
    self._session_service = InMemorySessionService()

    # Create ADK Runner
    self._runner = Runner(
        app_name="adk_rlm",
        agent=self._agent,
        session_service=self._session_service,
    )

    # Create file loader for file handling
    self._file_loader = FileLoader(
        sources=file_sources,
        parsers=file_parsers,
        base_path=base_path,
    )

    # Store config for reference
    self.model = model
    self.sub_model = sub_model or model
    self.max_iterations = max_iterations
    self.max_depth = max_depth
    self.persistent = persistent
    self.verbose = verbose
    self._logger = logger

  async def run_streaming(
      self,
      context: str | dict | list,
      prompt: str | None = None,
      conversation_history: list[dict[str, str]] | None = None,
  ) -> AsyncGenerator[Event, None]:
    """
    Run RLM with streaming events.

    Yields ADK Event objects with custom_metadata containing:
    - event_type: The type of event (see RLMEventType)
    - Additional event-specific data

    This is the primary interface for building UIs on top of RLM.

    Args:
        context: The context to analyze.
        prompt: Optional user prompt/question about the context.
        conversation_history: Optional list of previous conversation messages.
            Each message should have 'role' ('user' or 'assistant') and 'content'.
            This enables multi-turn conversations where the agent remembers
            previous questions and answers.

    Yields:
        ADK Event objects with RLM-specific metadata.

    Example:
        ```python
        async for event in rlm.run_streaming(context, prompt):
            event_type = event.custom_metadata.get("event_type")

            if event_type == "rlm.iteration.start":
                print(f"Starting iteration {event.custom_metadata['iteration']}")

            elif event_type == "rlm.code.end":
                if event.custom_metadata.get("output"):
                    print(event.custom_metadata["output"])

            elif event_type == "rlm.final.answer":
                print(f"Final: {event.custom_metadata['answer']}")
        ```
    """
    # Create session with context in state
    session = await self._session_service.create_session(
        app_name="adk_rlm",
        user_id="default_user",
        state={
            "rlm_context": context,
            "rlm_prompt": prompt,
            "rlm_conversation_history": conversation_history,
        },
    )

    # Build user message (the agent reads from session state)
    message = types.Content(
        role="user", parts=[types.Part(text=prompt or "Analyze the context.")]
    )

    # Run agent and yield events
    async for event in self._runner.run_async(
        user_id="default_user",
        session_id=session.id,
        new_message=message,
    ):
      yield event

  def close(self) -> None:
    """Clean up resources (call when done with persistent mode)."""
    self._agent.close()

  def __enter__(self) -> "RLM":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    self.close()
    return False

  @property
  def log_path(self) -> str | None:
    """Return the path to the log file if logging is enabled."""
    return self._logger.get_log_path() if self._logger else None

  @property
  def file_loader(self) -> FileLoader:
    """Access the file loader for direct file operations."""
    return self._file_loader

  @property
  def agent(self) -> RLMAgent:
    """Access the underlying RLM agent."""
    return self._agent

  @property
  def runner(self) -> Runner:
    """Access the ADK Runner for advanced usage."""
    return self._runner

  def load_files(
      self, files: list[str], lazy: bool = True
  ) -> "LazyFileCollection | list":
    """
    Load files without running RLM.

    Convenience method for loading files directly.

    Args:
        files: List of file paths/URIs/globs
        lazy: If True, return LazyFileCollection. If False, return parsed content.

    Returns:
        LazyFileCollection if lazy=True, else list of ParsedContent
    """
    if lazy:
      return self._file_loader.create_lazy_files(files)
    else:
      return self._file_loader.load_files(files)


def completion(
    context: str | dict | list | None = None,
    prompt: str | None = None,
    *,
    files: list[str] | None = None,
    model: str = "gemini-3-pro-preview",
    sub_model: str | None = None,
    max_iterations: int = 30,
    max_depth: int = 5,
    log_dir: str | None = None,
    verbose: bool = False,
) -> RLMChatCompletion:
  """
  Convenience function for simple synchronous RLM completion.

  This creates a temporary RLM instance, runs `run_streaming()`, and
  collects the final answer. For more control, use the RLM class directly.

  Args:
      context: The context/data to analyze.
      prompt: Optional user prompt/question about the context.
      files: List of file paths/URIs/globs to load as context.
      model: The main model to use (default: gemini-3-pro-preview).
      sub_model: The model for recursive sub-calls (defaults to model).
      max_iterations: Maximum number of RLM iterations (default: 30).
      max_depth: Maximum recursion depth (default: 5).
      log_dir: Directory for JSONL logs (None disables logging).
      verbose: Whether to print Rich console output.

  Returns:
      RLMChatCompletion with response and metadata.

  Example:
      ```python
      from adk_rlm import completion

      # Simple usage
      result = completion(
          context="Your document here...",
          prompt="Summarize the key points",
      )
      print(result.response)

      # With files
      result = completion(
          files=["./docs/**/*.md"],
          prompt="What are the main themes?",
      )
      ```
  """
  import asyncio
  import time

  time_start = time.perf_counter()

  # Create RLM instance
  rlm = RLM(
      model=model,
      sub_model=sub_model,
      max_iterations=max_iterations,
      max_depth=max_depth,
      log_dir=log_dir,
      verbose=verbose,
  )

  # Build context from files if provided
  if files:
    file_context = rlm.file_loader.build_context(files, lazy=True)
    if context is not None:
      ctx = _merge_context(context, file_context)
    else:
      ctx = file_context
  else:
    if context is None:
      raise ValueError("Either 'context' or 'files' must be provided")
    ctx = context

  # Run streaming and collect final answer
  async def _run():
    final_answer = None
    async for event in rlm.run_streaming(ctx, prompt):
      if event.custom_metadata:
        event_type = event.custom_metadata.get("event_type")
        if event_type == RLMEventType.FINAL_ANSWER.value:
          final_answer = event.custom_metadata.get("answer")
    return final_answer

  try:
    final_answer = asyncio.run(_run())
  finally:
    rlm.close()

  time_end = time.perf_counter()

  return RLMChatCompletion(
      root_model=model,
      prompt=context or str(files),
      response=final_answer or "",
      usage_summary=None,
      execution_time=time_end - time_start,
  )


def _merge_context(
    context: str | dict | list,
    file_context: dict,
) -> dict:
  """Merge direct context with file context."""
  if isinstance(context, str):
    return {"user_context": context, **file_context}
  elif isinstance(context, dict):
    return {**context, **file_context}
  else:
    return {"user_context": context, **file_context}
