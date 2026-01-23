"""
Local REPL environment for ADK-RLM.

This module provides a sandboxed Python REPL environment that can
execute code with access to context data and LLM query functions.
"""

from collections.abc import Callable
from contextlib import contextmanager
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Any
import uuid

from adk_rlm.repl.safe_builtins import SAFE_BUILTINS
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMChatCompletion


class LocalREPL:
  """
  Local REPL environment with persistent Python namespace.
  Executes code in a sandboxed namespace with access to context data.
  """

  def __init__(
      self,
      llm_query_fn: Callable[[str, str | None], str] | None = None,
      llm_query_batched_fn: (
          Callable[[list[str], str | None], list[str]] | None
      ) = None,
      context_payload: dict | list | str | None = None,
      setup_code: str | None = None,
  ):
    """
    Initialize the LocalREPL environment.

    Args:
        llm_query_fn: Function to query an LLM with a prompt.
        llm_query_batched_fn: Function to query an LLM with multiple prompts.
        context_payload: Initial context to load into the environment.
        setup_code: Optional Python code to execute during setup.
    """
    self.llm_query_fn = llm_query_fn
    self.llm_query_batched_fn = llm_query_batched_fn

    self.original_cwd = os.getcwd()
    self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
    self._lock = threading.Lock()
    self._context_count: int = 0
    self._history_count: int = 0

    # Track LLM calls made during code execution
    self._pending_llm_calls: list[RLMChatCompletion] = []

    # Setup globals, locals
    self._setup()

    # Load context if provided
    if context_payload is not None:
      self.load_context(context_payload)

    # Run setup code if provided
    if setup_code:
      self.execute_code(setup_code)

  def _setup(self) -> None:
    """Setup the environment with sandboxed globals and helper functions."""
    self.globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS.copy(),
        "__name__": "__main__",
    }
    self.locals: dict[str, Any] = {}

    # Add helper functions
    self.globals["FINAL_VAR"] = self._final_var
    self.globals["llm_query"] = self._llm_query
    self.globals["llm_query_batched"] = self._llm_query_batched

  def _final_var(self, variable_name: str) -> str:
    """Return the value of a variable as a final answer."""
    variable_name = variable_name.strip().strip("\"'")
    if variable_name in self.locals:
      return str(self.locals[variable_name])
    return f"Error: Variable '{variable_name}' not found"

  def _llm_query(
      self,
      prompt: str,
      context: Any = None,
      model: str | None = None,
      recursive: bool = True,
  ) -> str:
    """Query the LLM with a single prompt.

    Args:
        prompt: The prompt to send to the LLM.
        context: Optional context object to pass to the child agent.
        model: Optional model override.
        recursive: If True and depth allows, use recursive RLM execution.
    """
    if self.llm_query_fn is None:
      return "Error: No LLM query function configured"

    try:
      result = self.llm_query_fn(
          prompt, context=context, model=model, recursive=recursive
      )

      # Track this LLM call if it returns an RLMChatCompletion
      if isinstance(result, tuple) and len(result) == 2:
        response, completion = result
        if isinstance(completion, RLMChatCompletion):
          self._pending_llm_calls.append(completion)
        return response

      return result
    except Exception as e:
      return f"Error: LLM query failed - {e}"

  def _llm_query_batched(
      self,
      prompts: list[str],
      contexts: list[Any] | None = None,
      model: str | None = None,
      recursive: bool = False,
  ) -> list[str]:
    """Query the LLM with multiple prompts concurrently.

    Args:
        prompts: List of prompts to send.
        contexts: Optional list of context objects (same length as prompts).
        model: Optional model override.
        recursive: If True, use recursive RLM execution for each prompt.
                  Default is False for performance.
    """
    if self.llm_query_batched_fn is None:
      if contexts is not None:
        return [
            self._llm_query(p, context=c, model=model, recursive=recursive)
            for p, c in zip(prompts, contexts)
        ]
      return [
          self._llm_query(p, model=model, recursive=recursive) for p in prompts
      ]

    try:
      results = self.llm_query_batched_fn(
          prompts, contexts=contexts, model=model, recursive=recursive
      )

      # Handle tuple results with completions
      if results and isinstance(results[0], tuple):
        responses = []
        for item in results:
          if len(item) == 2:
            response, completion = item
            if isinstance(completion, RLMChatCompletion):
              self._pending_llm_calls.append(completion)
            responses.append(response)
          else:
            responses.append(str(item))
        return responses

      return results
    except Exception as e:
      return [f"Error: LLM query failed - {e}"] * len(prompts)

  def load_context(self, context_payload: dict | list | str) -> None:
    """Load context into the environment as context_0 (and 'context' alias)."""
    self.add_context(context_payload, 0)

  def add_context(
      self, context_payload: dict | list | str, context_index: int | None = None
  ) -> int:
    """
    Add a context with versioned variable name.

    Args:
        context_payload: The context data to add.
        context_index: Optional explicit index. If None, auto-increments.

    Returns:
        The context index used.
    """
    if context_index is None:
      context_index = self._context_count

    var_name = f"context_{context_index}"

    if isinstance(context_payload, str):
      context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
      with open(context_path, "w") as f:
        f.write(context_payload)
      self.execute_code(
          f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()"
      )
    else:
      # Try JSON serialization first for simple data structures
      try:
        context_path = os.path.join(
            self.temp_dir, f"context_{context_index}.json"
        )
        with open(context_path, "w") as f:
          json.dump(context_payload, f)
        self.execute_code(
            f"import json\nwith open(r'{context_path}', 'r') as f:\n   "
            f" {var_name} = json.load(f)"
        )
      except (TypeError, ValueError):
        # For complex objects (e.g., LazyFileCollection), inject directly
        self.locals[var_name] = context_payload

    # Alias context_0 as 'context' for backward compatibility
    if context_index == 0:
      if var_name in self.locals:
        self.locals["context"] = self.locals[var_name]
      else:
        self.execute_code(f"context = {var_name}")

    self._context_count = max(self._context_count, context_index + 1)
    return context_index

  def get_context_count(self) -> int:
    """Return the number of contexts loaded."""
    return self._context_count

  def add_history(
      self,
      message_history: list[dict[str, Any]],
      history_index: int | None = None,
  ) -> int:
    """
    Store a conversation's message history as a versioned variable.

    Args:
        message_history: The list of message dicts from a completion call.
        history_index: Optional explicit index. If None, auto-increments.

    Returns:
        The history index used.
    """
    if history_index is None:
      history_index = self._history_count

    var_name = f"history_{history_index}"

    # Store deep copy to avoid reference issues
    self.locals[var_name] = copy.deepcopy(message_history)

    # Alias history_0 as 'history' for convenience
    if history_index == 0:
      self.locals["history"] = self.locals[var_name]

    self._history_count = max(self._history_count, history_index + 1)
    return history_index

  def get_history_count(self) -> int:
    """Return the number of conversation histories stored."""
    return self._history_count

  @contextmanager
  def _capture_output(self):
    """Thread-safe context manager to capture stdout/stderr."""
    with self._lock:
      old_stdout, old_stderr = sys.stdout, sys.stderr
      stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
      try:
        sys.stdout, sys.stderr = stdout_buf, stderr_buf
        yield stdout_buf, stderr_buf
      finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

  @contextmanager
  def _temp_cwd(self):
    """Temporarily change to temp directory for execution."""
    old_cwd = os.getcwd()
    try:
      os.chdir(self.temp_dir)
      yield
    finally:
      os.chdir(old_cwd)

  def execute_code(self, code: str) -> REPLResult:
    """Execute code in the persistent namespace and return result."""
    start_time = time.perf_counter()

    # Clear pending LLM calls from previous execution
    self._pending_llm_calls = []

    with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
      try:
        combined = {**self.globals, **self.locals}
        exec(code, combined, combined)

        # Update locals with new variables
        for key, value in combined.items():
          if key not in self.globals and not key.startswith("_"):
            self.locals[key] = value

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()
      except Exception as e:
        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

    return REPLResult(
        stdout=stdout,
        stderr=stderr,
        locals=self.locals.copy(),
        execution_time=time.perf_counter() - start_time,
        rlm_calls=self._pending_llm_calls.copy(),
    )

  def reset(self) -> None:
    """Reset the REPL environment to initial state."""
    self._setup()
    self._context_count = 0
    self._history_count = 0
    self._pending_llm_calls = []

  def cleanup(self) -> None:
    """Clean up temp directory and reset state."""
    try:
      shutil.rmtree(self.temp_dir)
    except Exception:
      pass
    self.globals.clear()
    self.locals.clear()

  def __enter__(self) -> "LocalREPL":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    self.cleanup()
    return False

  def __del__(self) -> None:
    self.cleanup()
