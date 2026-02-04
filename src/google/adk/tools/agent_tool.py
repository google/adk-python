# Copyright 2026 Google LLC
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
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import model_validator
from typing_extensions import override

from . import _automatic_function_calling_util
from ..agents.common_configs import AgentRefConfig
from ..features import FeatureName
from ..features import is_feature_enabled
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..utils.context_utils import Aclosing
from ._forwarding_artifact_service import ForwardingArtifactService
from .base_tool import BaseTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent


def _get_input_schema(agent: BaseAgent) -> Optional[type[BaseModel]]:
  """Extracts the input_schema from an agent.

  For LlmAgent, returns its input_schema directly.
  For agents with sub_agents, recursively searches the first sub-agent for an
  input_schema.

  Args:
    agent: The agent to extract input_schema from.

  Returns:
    The input_schema if found, None otherwise.
  """
  from ..agents.llm_agent import LlmAgent

  if isinstance(agent, LlmAgent):
    return agent.input_schema

  # For composite agents, check the first sub-agent
  if agent.sub_agents:
    return _get_input_schema(agent.sub_agents[0])

  return None


def _get_output_schema(agent: BaseAgent) -> Optional[type[BaseModel]]:
  """Extracts the output_schema from an agent.

  For LlmAgent, returns its output_schema directly.
  For agents with sub_agents, recursively searches the last sub-agent for an
  output_schema.

  Args:
    agent: The agent to extract output_schema from.

  Returns:
    The output_schema if found, None otherwise.
  """
  from ..agents.llm_agent import LlmAgent

  if isinstance(agent, LlmAgent):
    return agent.output_schema

  # For composite agents, check the last sub-agent
  if agent.sub_agents:
    return _get_output_schema(agent.sub_agents[-1])

  return None


class AgentTool(BaseTool):
  """A tool that wraps an agent.

  This tool allows an agent to be called as a tool within a larger application.
  The agent's input schema is used to define the tool's input parameters, and
  the agent's output is returned as the tool's result.

  Attributes:
    agent: The agent to wrap.
    skip_summarization: Whether to skip summarization of the agent output.
    include_plugins: Whether to propagate plugins from the parent runner context
      to the agent's runner. When True (default), the agent will inherit all
      plugins from its parent. Set to False to run the agent with an isolated
      plugin environment.
  """

  def __init__(
      self,
      agent: BaseAgent,
      skip_summarization: bool = False,
      *,
      include_plugins: bool = True,
  ):
    self.agent = agent
    self.skip_summarization: bool = skip_summarization
    self.include_plugins = include_plugins

    super().__init__(name=agent.name, description=agent.description)

  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    data['name'] = data['agent'].name
    return data

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    from ..utils.variant_utils import GoogleLLMVariant

    input_schema = _get_input_schema(self.agent)
    output_schema = _get_output_schema(self.agent)

    if input_schema:
      result = _automatic_function_calling_util.build_function_declaration(
          func=input_schema, variant=self._api_variant
      )
      # Override the description with the agent's description
      result.description = self.agent.description
    else:
      if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
        result = types.FunctionDeclaration(
            name=self.name,
            description=self.agent.description,
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'request': {'type': 'string'},
                },
                'required': ['request'],
            },
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
      if output_schema:
        # Agent has structured output schema - response is an object
        if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
          result.response_json_schema = {'type': 'object'}
        else:
          result.response = types.Schema(type=types.Type.OBJECT)
      else:
        # Agent returns text - response is a string
        if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
          result.response_json_schema = {'type': 'string'}
        else:
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
    from ..runners import Runner
    from ..sessions.in_memory_session_service import InMemorySessionService

    if self.skip_summarization:
      tool_context.actions.skip_summarization = True

    input_schema = _get_input_schema(self.agent)
    if input_schema:
      input_value = input_schema.model_validate(args)
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
    invocation_context = tool_context._invocation_context
    parent_app_name = (
        invocation_context.app_name if invocation_context else None
    )
    child_app_name = parent_app_name or self.agent.name
    plugins = (
        tool_context._invocation_context.plugin_manager.plugins
        if self.include_plugins
        else None
    )
    runner = Runner(
        app_name=child_app_name,
        agent=self.agent,
        artifact_service=ForwardingArtifactService(tool_context),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=tool_context._invocation_context.credential_service,
        plugins=plugins,
    )

    state_dict = {
        k: v
        for k, v in tool_context.state.to_dict().items()
        if not k.startswith('_adk')  # Filter out adk internal states
    }
    session = await runner.session_service.create_session(
        app_name=child_app_name,
        user_id=tool_context._invocation_context.user_id,
        state=state_dict,
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

    # Clean up runner resources (especially MCP sessions)
    # to avoid "Attempted to exit cancel scope in a different task" errors
    await runner.close()

    if last_content is None or last_content.parts is None:
      return ''
    merged_text = '\n'.join(
        p.text for p in last_content.parts if p.text and not p.thought
    )
    output_schema = _get_output_schema(self.agent)
    if output_schema:
      tool_result = output_schema.model_validate_json(merged_text).model_dump(
          exclude_none=True
      )
    else:
      tool_result = merged_text
    return tool_result

  @override
  @classmethod
  def from_config(
      cls, config: ToolArgsConfig, config_abs_path: str
  ) -> AgentTool:
    from ..agents import config_agent_utils

    agent_tool_config = AgentToolConfig.model_validate(config.model_dump())

    agent = config_agent_utils.resolve_agent_reference(
        agent_tool_config.agent, config_abs_path
    )
    return cls(
        agent=agent,
        skip_summarization=agent_tool_config.skip_summarization,
        include_plugins=agent_tool_config.include_plugins,
    )


class AgentToolConfig(BaseToolConfig):
  """The config for the AgentTool."""

  agent: AgentRefConfig
  """The reference to the agent instance."""

  skip_summarization: bool = False
  """Whether to skip summarization of the agent output."""

  include_plugins: bool = True
  """Whether to include plugins from parent runner context."""
