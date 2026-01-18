# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code executor for CodingAgent with tool injection support.

This module provides a code executor that wraps an underlying executor
(e.g., ContainerCodeExecutor) and adds tool injection via HTTP IPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import logging
import re
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..tools.base_tool import BaseTool
from .allowlist_validator import AllowlistValidator
from .allowlist_validator import DEFAULT_SAFE_IMPORTS
from .allowlist_validator import ImportValidationError
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult
from .tool_code_generator import generate_full_code_with_stubs
from .tool_execution_server import detect_docker_host_address
from .tool_execution_server import ToolExecutionServer
from .tool_execution_server import ToolTrace

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


# Markers for extracting data from execution output
TOOL_TRACE_MARKER = "__TOOL_TRACE__:"
FINAL_ANSWER_MARKER = "__FINAL_ANSWER__:"


@dataclass
class ExecutionStep:
  """Record of a single code execution step.

  Attributes:
    code: The code that was executed.
    code_hash: Hash of the code for comparison.
    result: The execution result.
    tool_traces: Tool call traces from this step.
    success: Whether the execution succeeded.
    final_answer: The final answer if one was provided.
  """

  code: str
  code_hash: str = ""
  result: Optional[CodeExecutionResult] = None
  tool_traces: List[Dict[str, Any]] = field(default_factory=list)
  success: bool = False
  final_answer: Optional[Any] = None

  def __post_init__(self):
    if not self.code_hash:
      self.code_hash = hashlib.sha256(self.code.encode()).hexdigest()[:16]


@dataclass
class CodingAgentExecutionResult:
  """Extended execution result with CodingAgent-specific fields.

  Attributes:
    code_result: The underlying code execution result.
    tool_traces: List of tool call traces.
    final_answer: The final answer if one was provided.
    has_final_answer: Whether a final answer was extracted.
    clean_stdout: Stdout with trace markers removed.
  """

  code_result: CodeExecutionResult
  tool_traces: List[Dict[str, Any]] = field(default_factory=list)
  final_answer: Optional[Any] = None
  has_final_answer: bool = False
  clean_stdout: str = ""


class CodingAgentCodeExecutor(BaseCodeExecutor):
  """Code executor with tool injection for CodingAgent.

  This executor wraps an underlying code executor and adds:
  - Tool stub prepending for HTTP-based tool calls
  - Import allowlist validation before execution
  - Tool execution server lifecycle management
  - Trace extraction from execution output
  - Final answer detection
  - History re-execution for stateful mode

  Attributes:
    underlying_executor: The actual code executor to use.
    tools: List of tools to make available.
    authorized_imports: Set of allowed import patterns.
    tool_server_host: Host for the tool server.
    tool_server_port: Port for the tool server.
    execution_history: List of execution steps for stateful mode.
  """

  underlying_executor: BaseCodeExecutor
  tools: List[BaseTool] = Field(default_factory=list)
  authorized_imports: FrozenSet[str] = DEFAULT_SAFE_IMPORTS
  tool_server_host: Optional[str] = None
  tool_server_port: int = 8765

  # Internal state - use PrivateAttr for Pydantic
  _tool_server: Optional[ToolExecutionServer] = PrivateAttr(default=None)
  _validator: Optional[AllowlistValidator] = PrivateAttr(default=None)
  _invocation_context: Optional[InvocationContext] = PrivateAttr(default=None)
  _tool_context: Optional[ToolContext] = PrivateAttr(default=None)
  _execution_history: List[ExecutionStep] = PrivateAttr(default_factory=list)

  class Config:
    """Pydantic config."""

    arbitrary_types_allowed = True

  def model_post_init(self, __context):
    """Initialize after model construction."""
    self._validator = AllowlistValidator(
        allowlist=self.authorized_imports,
    )
    self._execution_history = []

  def set_context(
      self,
      invocation_context: InvocationContext,
      tool_context: Optional[ToolContext] = None,
  ) -> None:
    """Set the execution context.

    Args:
      invocation_context: The invocation context.
      tool_context: The tool context.
    """
    self._invocation_context = invocation_context
    self._tool_context = tool_context
    if self._tool_server:
      self._tool_server.set_context(invocation_context, tool_context)

  def _start_tool_server(self) -> None:
    """Start the tool execution server if not already running."""
    if self._tool_server is not None:
      return

    host = self.tool_server_host or "0.0.0.0"
    self._tool_server = ToolExecutionServer(
        host=host,
        port=self.tool_server_port,
        tools=self.tools,
        invocation_context=self._invocation_context,
    )
    self._tool_server.start()

  def _stop_tool_server(self) -> None:
    """Stop the tool execution server."""
    if self._tool_server:
      self._tool_server.stop()
      self._tool_server = None

  def _get_tool_server_url(self) -> str:
    """Get the URL for the tool server.

    Returns:
      The tool server URL accessible from containers.
    """
    if self.tool_server_host:
      host = self.tool_server_host
    else:
      host = detect_docker_host_address()
    return f"http://{host}:{self.tool_server_port}"

  def _validate_imports(self, code: str) -> None:
    """Validate imports in the code against the allowlist.

    Args:
      code: The code to validate.

    Raises:
      ImportValidationError: If unauthorized imports are found.
    """
    if self._validator:
      self._validator.validate_strict(code)

  def _extract_traces_and_answer(
      self,
      result: CodeExecutionResult,
  ) -> CodingAgentExecutionResult:
    """Extract tool traces and final answer from execution output.

    Args:
      result: The raw execution result.

    Returns:
      Extended result with extracted data.
    """
    tool_traces = []
    final_answer = None
    has_final_answer = False
    clean_lines = []

    for line in result.stdout.split("\n"):
      if line.startswith(TOOL_TRACE_MARKER):
        try:
          trace_json = line[len(TOOL_TRACE_MARKER) :]
          traces = json.loads(trace_json)
          tool_traces.extend(traces)
        except json.JSONDecodeError as e:
          logger.warning("Failed to parse tool trace: %s", e)
      elif line.startswith(FINAL_ANSWER_MARKER):
        answer_str = line[len(FINAL_ANSWER_MARKER) :]
        try:
          final_answer = json.loads(answer_str)
        except json.JSONDecodeError:
          # Not JSON, treat as string
          final_answer = answer_str
        has_final_answer = True
      else:
        clean_lines.append(line)

    clean_stdout = "\n".join(clean_lines).strip()

    return CodingAgentExecutionResult(
        code_result=result,
        tool_traces=tool_traces,
        final_answer=final_answer,
        has_final_answer=has_final_answer,
        clean_stdout=clean_stdout,
    )

  def _should_skip_step(self, step: ExecutionStep, code_hash: str) -> bool:
    """Check if an execution step can be skipped.

    For stateful mode, we can skip re-executing code if:
    - The code hasn't changed (same hash)
    - The previous execution succeeded

    Args:
      step: The previous execution step.
      code_hash: Hash of the current code.

    Returns:
      True if the step can be skipped.
    """
    return step.success and step.code_hash == code_hash

  def _prepend_tool_stubs(self, code: str) -> str:
    """Prepend runtime header and tool stubs to user code.

    Args:
      code: The user code to wrap.

    Returns:
      Complete code with tool stubs.
    """
    return generate_full_code_with_stubs(
        user_code=code,
        tools=self.tools,
        tool_server_url=self._get_tool_server_url(),
    )

  def _replay_history(
      self,
      invocation_context: InvocationContext,
  ) -> Optional[CodeExecutionResult]:
    """Replay execution history for stateful mode.

    This re-executes previous successful steps to restore state
    before executing new code.

    Args:
      invocation_context: The invocation context.

    Returns:
      The result of the last replayed step, or None if no replay needed.
    """
    if not self.stateful or not self._execution_history:
      return None

    last_result = None
    for step in self._execution_history:
      if step.success:
        # Re-execute to restore state
        full_code = self._prepend_tool_stubs(step.code)
        input_data = CodeExecutionInput(code=full_code)
        last_result = self.underlying_executor.execute_code(
            invocation_context=invocation_context,
            code_execution_input=input_data,
        )
        logger.debug("Replayed history step: %s", step.code_hash)

    return last_result

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Execute code with tool injection.

    Args:
      invocation_context: The invocation context.
      code_execution_input: The code to execute.

    Returns:
      The execution result.
    """
    user_code = code_execution_input.code

    # Validate imports first (security check before execution)
    try:
      self._validate_imports(user_code)
    except ImportValidationError as e:
      return CodeExecutionResult(
          stdout="",
          stderr=str(e),
          output_files=[],
      )

    # Start tool server if needed
    self._start_tool_server()

    # Set context on tool server
    if self._tool_server:
      self._tool_server.set_context(
          invocation_context,
          self._tool_context,
      )
      self._tool_server.clear_traces()

    # Replay history for stateful mode
    if self.stateful:
      self._replay_history(invocation_context)

    # Prepend tool stubs to user code
    full_code = self._prepend_tool_stubs(user_code)

    # Execute the code
    input_with_stubs = CodeExecutionInput(
        code=full_code,
        input_files=code_execution_input.input_files,
        execution_id=code_execution_input.execution_id,
    )

    result = self.underlying_executor.execute_code(
        invocation_context=invocation_context,
        code_execution_input=input_with_stubs,
    )

    # Extract traces and final answer
    extended_result = self._extract_traces_and_answer(result)

    # Record execution step for stateful mode
    step = ExecutionStep(
        code=user_code,
        result=result,
        tool_traces=extended_result.tool_traces,
        success=not result.stderr,
        final_answer=extended_result.final_answer,
    )
    self._execution_history.append(step)

    # Return result with clean stdout (traces stripped)
    return CodeExecutionResult(
        stdout=extended_result.clean_stdout,
        stderr=result.stderr,
        output_files=result.output_files,
    )

  def execute_code_extended(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodingAgentExecutionResult:
    """Execute code and return extended result with traces.

    Args:
      invocation_context: The invocation context.
      code_execution_input: The code to execute.

    Returns:
      Extended execution result with tool traces and final answer.
    """
    user_code = code_execution_input.code

    # Validate imports first
    try:
      self._validate_imports(user_code)
    except ImportValidationError as e:
      return CodingAgentExecutionResult(
          code_result=CodeExecutionResult(
              stdout="",
              stderr=str(e),
              output_files=[],
          ),
          tool_traces=[],
          final_answer=None,
          has_final_answer=False,
          clean_stdout="",
      )

    # Start tool server if needed
    self._start_tool_server()

    # Set context on tool server
    if self._tool_server:
      self._tool_server.set_context(
          invocation_context,
          self._tool_context,
      )
      self._tool_server.clear_traces()

    # Replay history for stateful mode
    if self.stateful:
      self._replay_history(invocation_context)

    # Prepend tool stubs to user code
    full_code = self._prepend_tool_stubs(user_code)

    # Execute the code
    input_with_stubs = CodeExecutionInput(
        code=full_code,
        input_files=code_execution_input.input_files,
        execution_id=code_execution_input.execution_id,
    )

    result = self.underlying_executor.execute_code(
        invocation_context=invocation_context,
        code_execution_input=input_with_stubs,
    )

    # Extract traces and final answer
    extended_result = self._extract_traces_and_answer(result)

    # Record execution step for stateful mode
    step = ExecutionStep(
        code=user_code,
        result=result,
        tool_traces=extended_result.tool_traces,
        success=not result.stderr,
        final_answer=extended_result.final_answer,
    )
    self._execution_history.append(step)

    return extended_result

  def get_execution_history(self) -> List[ExecutionStep]:
    """Get the execution history.

    Returns:
      List of execution steps.
    """
    return self._execution_history.copy()

  def clear_execution_history(self) -> None:
    """Clear the execution history."""
    self._execution_history.clear()

  def get_tool_traces(self) -> List[ToolTrace]:
    """Get tool traces from the server.

    Returns:
      List of tool traces.
    """
    if self._tool_server:
      return self._tool_server.get_traces()
    return []

  def cleanup(self) -> None:
    """Clean up resources."""
    self._stop_tool_server()
    self._execution_history.clear()

  def __del__(self):
    """Destructor to clean up resources."""
    self.cleanup()
