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

from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import model_validator
from typing_extensions import override

from . import _automatic_function_calling_util
from ..agents.common_configs import AgentRefConfig
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ._forwarding_artifact_service import ForwardingArtifactService
from .base_tool import BaseTool
from .base_tool import BaseToolConfig
from .base_tool import ToolArgsConfig
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
  """

  def __init__(self, agent: BaseAgent, skip_summarization: bool = False):
    self.agent = agent
    self.skip_summarization: bool = skip_summarization

    super().__init__(name=agent.name, description=agent.description)

  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    data['name'] = data['agent'].name
    return data

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    from ..agents.llm_agent import LlmAgent
    from ..utils.variant_utils import GoogleLLMVariant

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      result = _automatic_function_calling_util.build_function_declaration(
          func=self.agent.input_schema, variant=self._api_variant
      )
    else:
      result = types.FunctionDeclaration(
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  'request': types.Schema(
                      type=types.Type.STRING,
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
        result.response = types.Schema(type=types.Type.OBJECT)
      else:
        # Agent returns text - response is a string
        result.response = types.Schema(type=types.Type.STRING)

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
      content = types.Content(
          role='user',
          parts=[
              types.Part.from_text(
                  text=input_value.model_dump_json(exclude_none=True)
              )
          ],
      )
    else:
      content = types.Content(
          role='user',
          parts=[types.Part.from_text(text=args['request'])],
      )
    runner = Runner(
        app_name=self.agent.name,
        agent=self.agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=tool_context._invocation_context.credential_service,
    )
    session = await runner.session_service.create_session(
        app_name=self.agent.name,
        user_id='tmp_user',
        state=tool_context.state.to_dict(),
    )

    last_event = None
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
      # Forward state delta to parent session.
      if event.actions.state_delta:
        tool_context.state.update(event.actions.state_delta)
      last_event = event

    if not last_event or not last_event.content or not last_event.content.parts:
      return ''
    merged_text = '\n'.join(p.text for p in last_event.content.parts if p.text)
    if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
      tool_result = self.agent.output_schema.model_validate_json(
          merged_text
      ).model_dump(exclude_none=True)
    else:
      tool_result = merged_text
    return tool_result

  @classmethod
  @override
  def from_config(
      cls, config: ToolArgsConfig, config_abs_path: str
  ) -> AgentTool:
    from ..agents import config_agent_utils

    agent_tool_config = AgentToolConfig.model_validate(config.model_dump())

    agent = config_agent_utils.resolve_agent_reference(
        agent_tool_config.agent, config_abs_path
    )
    return cls(
        agent=agent, skip_summarization=agent_tool_config.skip_summarization
    )


class AgentToolConfig(BaseToolConfig):
  """The config for the AgentTool."""

  agent: AgentRefConfig
  """The reference to the agent instance."""

  skip_summarization: bool = False
  """Whether to skip summarization of the agent output."""
