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

"""CodingAgent - An agent that generates and executes Python code.

This module provides the CodingAgent class, which implements a ReAct-style
agent that generates Python code to accomplish tasks using available tools.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from google.genai import types
from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from ..code_executors.allowlist_validator import DEFAULT_SAFE_IMPORTS
from ..code_executors.base_code_executor import BaseCodeExecutor
from ..code_executors.code_execution_utils import CodeExecutionInput
from ..code_executors.code_execution_utils import CodeExecutionResult
from ..code_executors.code_execution_utils import CodeExecutionUtils
from ..code_executors.coding_agent_code_executor import CodingAgentCodeExecutor
from ..code_executors.coding_agent_code_executor import CodingAgentExecutionResult
from ..code_executors.tool_code_generator import generate_system_prompt
from ..events.event import Event
from ..events.event_actions import EventActions
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from ..tools.function_tool import FunctionTool
from ..tools.tool_context import ToolContext
from ..utils.feature_decorator import experimental
from .base_agent import BaseAgent
from .base_agent import BaseAgentState
from .base_agent_config import BaseAgentConfig
from .coding_agent_config import CodingAgentConfig
from .invocation_context import InvocationContext
from .readonly_context import ReadonlyContext

logger = logging.getLogger("google_adk." + __name__)


@experimental
class CodingAgentState(BaseAgentState):
    """State for CodingAgent tracking execution progress.

    Attributes:
      iteration_count: Number of ReAct loop iterations completed.
      error_count: Number of consecutive errors encountered.
      execution_history: List of execution steps with code, results, and traces.
    """

    iteration_count: int = 0
    error_count: int = 0
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)


ToolUnion = Union[Callable[..., Any], BaseTool, BaseToolset]


async def _convert_tool_union_to_tools(
    tool_union: ToolUnion,
    ctx: Optional[ReadonlyContext] = None,
) -> List[BaseTool]:
    """Convert a tool union to a list of BaseTool instances.

    Args:
      tool_union: A callable, BaseTool, or BaseToolset.
      ctx: Optional context for toolset resolution.

    Returns:
      List of BaseTool instances.
    """
    if isinstance(tool_union, BaseTool):
        return [tool_union]
    if callable(tool_union):
        return [FunctionTool(func=tool_union)]
    # BaseToolset
    if ctx:
        return await tool_union.get_tools_with_prefix(ctx)
    return await tool_union.get_tools_with_prefix(None)


@experimental
class CodingAgent(BaseAgent):
    """Agent that generates Python code to solve tasks using available tools.

    CodingAgent implements a ReAct-style loop where it:
    1. Receives a task from the user
    2. Generates Python code that calls available tools
    3. Executes the code in a sandboxed environment
    4. Processes the results and either provides an answer or continues

    Tools are made available as Python functions that the generated code
    can call. The code execution happens in a container for security,
    with tool calls routed via HTTP to the host.

    Attributes:
      model: The LLM model to use for code generation.
      instruction: Additional instructions for the agent.
      tools: List of tools available to the agent.
      code_executor: The underlying code executor (e.g., ContainerCodeExecutor).
      authorized_imports: Set of allowed Python imports.
      max_iterations: Maximum ReAct loop iterations.
      error_retry_attempts: Number of retries on execution errors.
      stateful: Whether to maintain state across iterations.
      tool_server_host: Host for the tool execution server.
      tool_server_port: Port for the tool execution server.
    """

    DEFAULT_MODEL: ClassVar[str] = "gemini-2.5-flash"

    config_type: ClassVar[Type[BaseAgentConfig]] = CodingAgentConfig

    model: Union[str, BaseLlm] = ""
    """The model to use for code generation."""

    instruction: str = ""
    """Additional instructions for the agent."""

    tools: List[ToolUnion] = Field(default_factory=list)
    """Tools available to the agent."""

    code_executor: Optional[BaseCodeExecutor] = None
    """The underlying code executor. If not set, uses ContainerCodeExecutor."""

    authorized_imports: FrozenSet[str] = DEFAULT_SAFE_IMPORTS
    """Set of allowed import patterns."""

    max_iterations: int = 10
    """Maximum number of ReAct loop iterations."""

    error_retry_attempts: int = 2
    """Number of retries on execution errors."""

    stateful: bool = False
    """Whether to maintain state across iterations."""

    tool_server_host: Optional[str] = None
    """Host for the tool execution server."""

    tool_server_port: int = 8765
    """Port for the tool execution server."""

    # Internal state
    _coding_executor: Optional[CodingAgentCodeExecutor] = None
    _resolved_tools: Optional[List[BaseTool]] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @property
    def canonical_model(self) -> BaseLlm:
        """Get the resolved model as BaseLlm."""
        if isinstance(self.model, BaseLlm):
            return self.model
        elif self.model:
            return LLMRegistry.new_llm(self.model)
        else:
            # Find model from ancestors
            ancestor_agent = self.parent_agent
            while ancestor_agent is not None:
                if hasattr(ancestor_agent, "canonical_model"):
                    return ancestor_agent.canonical_model
                ancestor_agent = ancestor_agent.parent_agent
            return LLMRegistry.new_llm(self.DEFAULT_MODEL)

    async def _resolve_tools(
        self,
        ctx: Optional[ReadonlyContext] = None,
    ) -> List[BaseTool]:
        """Resolve tool unions to BaseTool instances.

        Args:
          ctx: Optional context for toolset resolution.

        Returns:
          List of resolved BaseTool instances.
        """
        if self._resolved_tools is not None:
            return self._resolved_tools

        resolved = []
        for tool_union in self.tools:
            resolved.extend(await _convert_tool_union_to_tools(tool_union, ctx))

        self._resolved_tools = resolved
        return resolved

    async def _get_coding_executor(
        self,
        ctx: InvocationContext,
    ) -> CodingAgentCodeExecutor:
        """Get or create the CodingAgentCodeExecutor.

        Args:
          ctx: The invocation context.

        Returns:
          The configured code executor.
        """
        if self._coding_executor is not None:
            return self._coding_executor

        # Resolve tools
        tools = await self._resolve_tools(ReadonlyContext(ctx))

        # Get or create underlying executor
        if self.code_executor:
            underlying = self.code_executor
        else:
            # Default to ContainerCodeExecutor
            try:
                from ..code_executors.container_code_executor import (
                    ContainerCodeExecutor,
                )

                underlying = ContainerCodeExecutor(
                    image="python:3.11-slim",
                )
            except ImportError as e:
                raise ImportError(
                    "CodingAgent requires ContainerCodeExecutor. "
                    'Please install with: pip install "google-adk[extensions]" '
                    "or provide a custom code_executor."
                ) from e

        # Create the CodingAgentCodeExecutor wrapper
        self._coding_executor = CodingAgentCodeExecutor(
            underlying_executor=underlying,
            tools=tools,
            authorized_imports=self.authorized_imports,
            tool_server_host=self.tool_server_host,
            tool_server_port=self.tool_server_port,
            stateful=self.stateful,
            error_retry_attempts=self.error_retry_attempts,
        )

        return self._coding_executor

    def _build_system_prompt(self, tools: List[BaseTool]) -> str:
        """Build the system prompt with tool documentation.

        Args:
          tools: List of available tools.

        Returns:
          The complete system prompt.
        """
        return generate_system_prompt(
            tools=tools,
            custom_instruction=self.instruction,
        )

    def _extract_code_block(self, response_text: str) -> Optional[str]:
        """Extract code from the model response.

        Args:
          response_text: The model's response text.

        Returns:
          The extracted code, or None if no code block found.
        """
        # Try tool_code blocks first
        pattern = r"```tool_code\n(.*?)```"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fall back to python blocks
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _build_error_feedback(
        self,
        error: str,
        code: str,
    ) -> str:
        """Build feedback message for execution errors.

        Args:
          error: The error message.
          code: The code that caused the error.

        Returns:
          Formatted error feedback for the LLM.
        """
        return f"""The code execution failed with the following error:

```
{error}
```

The code that failed was:
```python
{code}
```

Please fix the error and try again. Common issues:
- Unauthorized imports (only use allowed imports)
- Tool call errors (check the tool documentation)
- Python syntax errors
"""

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Core implementation of the ReAct loop.

        Args:
          ctx: The invocation context.

        Yields:
          Events generated during execution.
        """
        # Load or initialize state
        state = self._load_agent_state(ctx, CodingAgentState)
        if state is None:
            state = CodingAgentState()

        # Resolve tools and get executor
        tools = await self._resolve_tools(ReadonlyContext(ctx))
        coding_executor = await self._get_coding_executor(ctx)

        # Create tool context for the executor
        tool_context = ToolContext(invocation_context=ctx)
        coding_executor.set_context(ctx, tool_context)

        # Build system prompt
        system_prompt = self._build_system_prompt(tools)

        # Get the model
        model = self.canonical_model

        # Build initial request with conversation history
        contents = []
        events = ctx._get_events(current_invocation=True, current_branch=True)
        for event in events:
            if event.content:
                contents.append(event.content)

        iteration = 0
        error_count = 0
        final_answer = None

        while iteration < self.max_iterations:
            iteration += 1
            state.iteration_count = iteration

            # Build LLM request
            llm_request = LlmRequest(
                model=model.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )

            # Call the model (generate_content_async returns an async generator)
            llm_response = None
            async for response in model.generate_content_async(
                llm_request, stream=False
            ):
                llm_response = response
                break

            # Extract response text
            response_text = ""
            if llm_response and llm_response.content and llm_response.content.parts:
                response_text = "".join(
                    part.text for part in llm_response.content.parts if part.text
                )

            # Check for code block
            code = self._extract_code_block(response_text)

            if not code:
                # No code generated - treat as final response
                # Check if the response looks like a final answer
                final_answer = response_text
                break

            # Execute the code
            code_input = CodeExecutionInput(code=code)
            exec_result = coding_executor.execute_code_extended(
                invocation_context=ctx,
                code_execution_input=code_input,
            )

            # Record execution in state
            state.execution_history.append(
                {
                    "iteration": iteration,
                    "code": code,
                    "stdout": exec_result.clean_stdout,
                    "stderr": exec_result.code_result.stderr,
                    "tool_traces": exec_result.tool_traces,
                    "has_final_answer": exec_result.has_final_answer,
                }
            )

            # Check for errors
            if exec_result.code_result.stderr:
                error_count += 1
                state.error_count = error_count

                if error_count > self.error_retry_attempts:
                    # Too many errors - give up
                    final_answer = (
                        f"I encountered too many errors while executing code. "
                        f"Last error: {exec_result.code_result.stderr}"
                    )
                    break

                # Build error feedback and add to conversation
                error_feedback = self._build_error_feedback(
                    exec_result.code_result.stderr,
                    code,
                )
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=response_text)],
                    )
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=error_feedback)],
                    )
                )
                continue

            # Reset error count on success
            error_count = 0
            state.error_count = 0

            # Check for final answer
            if exec_result.has_final_answer:
                final_answer = exec_result.final_answer
                break

            # Add execution result to conversation and continue
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=response_text)],
                )
            )

            # Add execution output as user message
            output_text = f"""Code execution result:
```
{exec_result.clean_stdout}
```
"""
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=output_text)],
                )
            )

        # Build final event
        if final_answer is None:
            final_answer = (
                "I was unable to complete the task within the allowed iterations."
            )

        # Convert final_answer to string if needed
        if not isinstance(final_answer, str):
            import json

            try:
                final_answer = json.dumps(final_answer)
            except (TypeError, ValueError):
                final_answer = str(final_answer)

        # Update state in context
        ctx.agent_states[self.name] = state.model_dump()

        # Yield final event
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=final_answer)],
            ),
            actions=EventActions(
                agent_state=state.model_dump(),
            ),
        )

    @model_validator(mode="after")
    def _validate_model(self) -> CodingAgent:
        """Validate the model after construction."""
        return self

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._coding_executor:
            self._coding_executor.cleanup()
            self._coding_executor = None
        self._resolved_tools = None

    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.cleanup()
        except Exception:
            pass
