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

from __future__ import annotations

import types
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types as genai_types
from pydantic import model_validator
from typing_extensions import override

from . import _automatic_function_calling_util
from ..agents.common_configs import AgentRefConfig
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..utils.context_utils import Aclosing
from ._forwarding_artifact_service import ForwardingArtifactService
from .base_tool import BaseTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent


class AgentTool(BaseTool):
  """A tool that wraps an agent.

  This tool allows an agent to be called as a tool within a larger application.
  The agent's input schema is used to define the tool's input parameters, and
  the agent's output is returned as the tool's result.

  Attributes:
    agent: The agent to wrap.
    skip_summarization: Whether to skip summarization of the agent output.
    persist_memory: Whether to persist the agent's memory across tool calls.
  """

  def __init__(self, agent: BaseAgent, skip_summarization: bool = False, persist_memory: bool = False):
    self.agent = agent
    self.skip_summarization: bool = skip_summarization
    self.persist_memory: bool = persist_memory

    super().__init__(name=agent.name, description=agent.description)

  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    data['name'] = data['agent'].name
    return data

  @override
  def _get_declaration(self) -> genai_types.FunctionDeclaration:
    from ..agents.llm_agent import LlmAgent
    from ..utils.variant_utils import GoogleLLMVariant

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      result = _automatic_function_calling_util.build_function_declaration(
          func=self.agent.input_schema, variant=self._api_variant
      )
    else:
      result = genai_types.FunctionDeclaration(
          parameters=genai_types.Schema(
              type=genai_types.Type.OBJECT,
              properties={
                  'request': genai_types.Schema(
                      type=genai_types.Type.STRING,
                  ),
              },
              required=['request'],
          ),
          description=self.agent.description,
          name=self.name,
      )

    # Set response schema for non-GEMINI_API variants
    if self._api_variant != GoogleLLMVariant.GEMINI_API:
      # Determine response type based on agent's output schema
      if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
        # Agent has structured output schema - response is an object
        result.response = genai_types.Schema(type=genai_types.Type.OBJECT)
      else:
        # Agent returns text - response is a string
        result.response = genai_types.Schema(type=genai_types.Type.STRING)

    result.name = self.name
    return result

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    from ..agents.llm_agent import LlmAgent
    from ..runners import Runner
    from ..sessions.in_memory_session_service import InMemorySessionService

    if self.skip_summarization:
      tool_context.actions.skip_summarization = True

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      input_value = self.agent.input_schema.model_validate(args)
      content = genai_types.Content(
          role='user',
          parts=[
              genai_types.Part.from_text(
                  text=input_value.model_dump_json(exclude_none=True)
              )
          ],
      )
    else:
      content = genai_types.Content(
          role='user',
          parts=[genai_types.Part.from_text(text=args['request'])],
      )

    memory_service = InMemoryMemoryService()
    if self.persist_memory:
      def get_state(self):
        with self._lock:
          return self._session_events
      
      def set_state(self, state):
        with self._lock:
          self._session_events = state
      
      memory_service.get_state = types.MethodType(get_state, memory_service)
      memory_service.set_state = types.MethodType(set_state, memory_service)

      memory_key = f"__memory_for_{self.agent.name}"
      if memory_key in tool_context.state:
        memory_service.set_state(tool_context.state[memory_key])

    runner = Runner(
        app_name=self.agent.name,
        agent=self.agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=memory_service,
        credential_service=tool_context._invocation_context.credential_service,
    )
    session = await runner.session_service.create_session(
        app_name=self.agent.name,
        user_id=tool_context._invocation_context.user_id,
        state=tool_context.state.to_dict(),
    )

    last_content = None
    async with Aclosing(
        runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=content
        )
    ) as agen:
      async for event in agen:
        # Forward state delta to parent session.
        if event.actions.state_delta:
          tool_context.state.update(event.actions.state_delta)
        if event.content:
          last_content = event.content

    if self.persist_memory:
      memory_key = f"__memory_for_{self.agent.name}"
      tool_context.state[memory_key] = memory_service.get_state()

    if not last_content:
      return ''
    merged_text = '\n'.join(p.text for p in last_content.parts if p.text)
    if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
      tool_result = self.agent.output_schema.model_validate_json(
          merged_text
      ).model_dump(exclude_none=True)
    else:
      tool_result = merged_text
    return tool_result

  @override
  @classmethod
  def from_config(
      cls,
      config: ToolArgsConfig,
      config_abs_path: str,
  ) -> AgentTool:
    from ..agents import config_agent_utils

    agent_tool_config = AgentToolConfig.model_validate(config.model_dump())

    agent = config_agent_utils.resolve_agent_reference(
        agent_tool_config.agent, config_abs_path
    )
    return cls(
        agent=agent,
        skip_summarization=agent_tool_config.skip_summarization,
        persist_memory=agent_tool_config.persist_memory
    )


class AgentToolConfig(BaseToolConfig):
  """The config for the AgentTool."""

  agent: AgentRefConfig
  """The reference to the agent instance."""

  skip_summarization: bool = False
  """Whether to skip summarization of the agent output."""

  persist_memory: bool = False
  """Whether to persist the agent's memory across tool calls."""
